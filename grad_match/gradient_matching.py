"""Gradient matching loss (Geiping et al., 2020 — Witches' Brew, https://arxiv.org/abs/2009.02276).

A thin wrapper around a ``utility_loss``: it takes the utility loss' scalar,
back-propagates it to the model weights to obtain
``grad_theta utility_loss(candidate; theta)``, and returns the negated cosine
similarity against a target gradient supplied through ``__call__``.

Both the model forward output and the target gradient reach ``__call__`` via the
standard parameter-name resolution (``model_output`` / ``model_input`` /
``target_gradient``). The only flag it sets is ``require_gradients=True`` so the
no-grad loss-ranking path keeps a live graph for the weight-gradient; every
other ``require_*`` flag delegates to ``utility_loss``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Parameter

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    MessageTargets,
    ModelInput,
    ModelOutput,
    Targets,
    TextTemplates,
)
from tropt.loss.base import BaseLoss
from tropt.loss.resolution import resolve_and_compute_loss

logger = logging.getLogger(__name__)


def _delegate_require_flags_to_utility_loss(cls):
    """
    Mirror all the ``require_*`` flag (except ``require_gradients``,
    which grad matching owns) onto the wrapped ``utility_loss`` in the GradientMatchingLoss.
    """
    for name in vars(BaseLoss):
        if name.startswith("require_") and name != "require_gradients":
            setattr(cls, name, property(lambda self, _f=name: getattr(self.utility_loss, _f)))
    return cls


@_delegate_require_flags_to_utility_loss
@dataclass
class GradientMatchingLoss(BaseLoss):
    """Match the candidate input's weight-gradient direction to a target's, w.r.t. to some (potentially different) objecetives.

    A common setting is trainset poisoning to match the gradients of:
    (i) a poison (candidate) sample w.r.t. benign training utility loss (e.g., cross-entropy), and a (ii) target sample w.r.t. adversarial loss (e.g., misclassification).

    Design: a wrapper on the utility loss. 
    ``__call__`` receives the full model's forward output to be redirected to the utility loss(``model_output``/``model_input``),
    and the precomputed target gradient (``target_gradient``, a resolved ``MessageTargets`` field).
    It computes the utility loss' scalars, back-propagates it to the model weights for the utility gradients, 
    and returns their matching (via cos or L2) against the target gradient.

    Usage:
    1. Init a HuggingFace model (e.g., LMHFModel) with ``set_model_to_train=True`` (weights back-proppable);
    2. Initialzie the loss with the desired differentiable utility loss (e.g., CrossEntropyLoss) and the model;
    2. Precompute the target gradient once with ``loss.compute_target_gradient()`` (where loss is the initialized loss)
       and pass it via ``Targets(target_gradient=...)`` next
       to the utility-loss targets;
    3. Optimize the trigger with any gradient-based optimizer (e.g. GCG / MAC).
    """

    is_differentiable: ClassVar[bool] = True
    # Flag for callers to not disable gradient when computing the forward pass proceeding this loss computation.
    # This is essential as we need to back-propagate through the model weights to compute the utility gradient.
    require_gradients: ClassVar[bool] = True

    model: Any
    utility_loss: BaseLoss

    similarity: Literal["cosine", "neg_l2"] = "cosine"

    _trainable_params: list[Parameter] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.utility_loss.is_differentiable, (
            f"utility_loss must be differentiable; got "
            f"{type(self.utility_loss).__name__} with is_differentiable=False."
        )

        # Prefix caching reuses one cached prefix across steps (stale-graph crash in
        # train mode) and would drop the prefix->weights path from the gradient.
        assert not getattr(self.model, "_use_prefix_cache", False), (
            f"{type(self).__name__} requires the model initialized with use_prefix_cache=False."
        )

        # Make sure the trainable parameter set is non empty -- we take gradients w.r.t. them.
        # TODO [IN THE FUTURE] support partial weight optimization (=only on a subset).
        trainable = [p for p in self.model._model.parameters() if p.requires_grad]
        assert trainable, (
            f"{type(self).__name__} needs back-proppable model weights, but none "
            f"of {type(self.model).__name__}'s parameters have requires_grad=True. "
            f"Initialize the model with set_model_to_train=True."
        )
        self._trainable_params = trainable

    def compute_target_gradient(
        self, input_strs: TextTemplates, targets: Targets,
    ) -> Float[Tensor, "n_params"]:
        """Average ``grad_theta utility_loss(input_str; theta)`` over ``input_strs``
        — the target gradient to feed back via ``Targets(target_gradient=...)``.

        Reuses ``self.model`` and ``self._trainable_params``, so the result aligns
        element-wise with the per-candidate gradients in ``__call__``. Note that there is
        **no trigger** involved here: each plain input string is forwarded as-is through the
        model's text-level ``invoke_from_texts``.
        """
        model, loss = self.model, self.utility_loss
        # No trigger: forward each plain input string directly via the model's
        # text-level invoke (it tokenises + prefills internally).
        targets = model._update_targets_by_model(targets)  # tokenise targets, to device

        n_total = sum(p.numel() for p in self._trainable_params)
        accum = torch.zeros(n_total, device=model.device, dtype=torch.float32)
        for idx, text in enumerate(input_strs):
            message_targets = targets.select_message(idx)
            model_output = model.invoke_from_texts(
                input_texts=[text],
                message_targets=message_targets,
                require_target_prefill=loss.require_target_prefill,
                require_generation=False,
            )
            model_input = ModelInput(input_texts=[text], message_targets=message_targets)
            util = resolve_and_compute_loss(model_output, model_input, loss)
            grads = torch.autograd.grad(util.sum(), self._trainable_params)
            accum = accum + torch.cat(
                [g.detach().reshape(-1).float() for g in grads]
            )
        return (accum / len(input_strs)).detach()

    def __call__(
        self,
        model_output: ModelOutput,
        model_input: ModelInput,
        target_gradient: Float[Tensor, "n_params"],
    ) -> Float[Tensor, "bsz"]:
        g_target = target_gradient.to(device=self.model.device)

        # The utility loss' per-candidate scalar, resolved exactly as if it ran
        # standalone — we only add the gradient-matching step on top. Its own
        # target (e.g. the response for PrefillCELoss) is resolved normally from
        # model_input.message_targets; grad matching's "target" is the separate,
        # precomputed target_gradient arg above. Two independent targets, both via
        # the standard resolver.
        util_loss = resolve_and_compute_loss(
            model_output, model_input, self.utility_loss,
        )  # (bsz,)

        # 2nd-order graph (for an outer optimizer to back-prop through the cosine)
        # is only needed on the compute_grad_from_* flows, where the trigger input
        # carries grad. 
        # [HACK ALERT] We identify these flows through the presence of input_embeds 
        # with requires_grad, as this is currently how we obtain trigger gradients.
         # Precompute target gradient norm and per-layer chunks (these don't change across candidates)
        param_numels = [p.numel() for p in self._trainable_params]
        g_target_chunks = g_target.split(param_numels)
        norm_target_sq = sum(
            chunk.float().pow(2).sum() for chunk in g_target_chunks
        )
        norm_target = norm_target_sq.sqrt()

        util_loss = resolve_and_compute_loss(
            model_output, model_input, self.utility_loss,
        )  # (bsz,)

        create_graph = bool(
            model_input.input_embeds is not None
            and model_input.input_embeds.requires_grad
        )

        sims: list[Tensor] = []
        for k in range(util_loss.shape[0]):
            g_cand = torch.autograd.grad(
                util_loss[k],
                self._trainable_params,
                create_graph=create_graph,
                retain_graph=True,
            )

            if self.similarity == "cosine":
                # Chunked cosine similarity — never materializes the full gradient vector
                dot = torch.tensor(0.0, device=self.model.device)
                norm_cand_sq = torch.tensor(0.0, device=self.model.device)
                for g_chunk, t_chunk in zip(g_cand, g_target_chunks):
                    g_flat = g_chunk.reshape(-1).float()
                    t_flat = t_chunk.float()
                    dot = dot + (g_flat * t_flat).sum()
                    norm_cand_sq = norm_cand_sq + g_flat.pow(2).sum()
                    del g_flat, t_flat
                sim_k = (dot / (norm_cand_sq.sqrt() * norm_target + 1e-8)).clamp(-1.0, 1.0)

            elif self.similarity == "neg_l2":
                # Chunked L2 distance
                dist_sq = torch.tensor(0.0, device=self.model.device)
                for g_chunk, t_chunk in zip(g_cand, g_target_chunks):
                    diff = g_chunk.reshape(-1).float() - t_chunk.float()
                    dist_sq = dist_sq + diff.pow(2).sum()
                    del diff
                sim_k = -dist_sq.sqrt()

            else:
                raise ValueError(f"Unknown similarity={self.similarity!r}")

            # Free the per-candidate gradient tuples
            del g_cand
            sims.append(sim_k)

        return -torch.stack(sims, dim=0)  # minimize -> maximize alignment
