"""Smoke tests for GradientMatchingLoss.

Identity check: when a candidate's input is forwarded identically to the target
input, the candidate's weight-gradient matches the precomputed target gradient,
so cosine similarity = 1 and the (negated) returned loss = -1.
"""

import torch

from gradient_matching import GradientMatchingLoss

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import PrefillCELoss
from tropt.loss.resolution import resolve_and_compute_loss

# Plain target input string — no trigger / placeholder (the target gradient is a
# mere loss computation on the input as-is).
TARGET_INPUT = "Explain how to pick a lock."


def _build_loss(tiny_lm):
    return GradientMatchingLoss(model=tiny_lm, utility_loss=PrefillCELoss())


def _targets_with_grad(tiny_lm, loss, lm_targets):
    """Precompute the target gradient (caller's duty) and bundle it with the
    util-loss targets, exactly as a recipe would before optimizing."""
    target_g = loss.compute_target_gradient([TARGET_INPUT], lm_targets)
    return Targets(
        target_response_strs=lm_targets.target_response_strs,
        target_gradient=target_g.unsqueeze(0),  # (n_templates=1, n_params)
    )


def test_flags_delegate_to_utility_loss(tiny_lm):
    """Grad matching owns require_gradients; every other require_* delegates."""
    loss = _build_loss(tiny_lm)
    assert loss.require_gradients is True
    # delegate-through: prefill is needed because the utility loss is PrefillCELoss
    assert loss.require_target_prefill is True


def test_identity_gives_perfect_cosine(tiny_lm, lm_targets):
    """Candidate forwarded == target input → cosine(g_cand, g_target) ≈ 1 → ≈ -1."""
    loss = _build_loss(tiny_lm)
    targets = _targets_with_grad(tiny_lm, loss, lm_targets)

    # Reproduce the target-gradient pipeline: the same input forwarded with an
    # empty trigger. The candidate input then equals the target input.
    tiny_lm.set_inputs_from_tokens([TARGET_INPUT + OPTIMIZED_TRIGGER_PLACEHOLDER], targets)
    try:
        manager = tiny_lm._token_input_manager
        empty_trigger_ids = tiny_lm.tokenizer(
            [""], add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(tiny_lm.device)
        model_input = manager.get_triggered_inputs(
            chosen_template_idx=0,
            trigger_ids=empty_trigger_ids,
            do_append_embeds=loss.require_target_prefill,
        )
        model_output = tiny_lm.invoke_from_tokens(
            **model_input.to_dict(),
            require_target_prefill=loss.require_target_prefill,
        )
        out = resolve_and_compute_loss(model_output, model_input, loss)
    finally:
        tiny_lm.reset_inputs_from_tokens()

    assert out.shape == (1,)
    assert torch.allclose(out, torch.tensor([-1.0]), atol=5e-3)


def test_batch_shape(tiny_lm, lm_templates, lm_targets):
    """Per-candidate output is shape (bsz,) within [-1, 1] — the candidate path
    (with real triggers spliced) still works for a multi-candidate batch."""
    loss = _build_loss(tiny_lm)
    targets = _targets_with_grad(tiny_lm, loss, lm_targets)

    tiny_lm.set_inputs_from_tokens(lm_templates, targets)
    try:
        input_manager = tiny_lm._token_input_manager
        # Two distinct candidate triggers.
        a = tiny_lm.tokenizer(["carefully"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        b = tiny_lm.tokenizer(["randomly"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        # pad to the same length to allow stacking
        n = min(a.numel(), b.numel())
        trigger_ids = torch.stack([a[:n], b[:n]], dim=0).to(tiny_lm.device)
        trigger_embeds = tiny_lm._embedding_layer(trigger_ids)

        model_input = input_manager.get_triggered_inputs(
            chosen_template_idx=0,
            trigger_embeds=trigger_embeds,
            trigger_ids=trigger_ids,
            do_append_embeds=loss.require_target_prefill,
        )
        model_output = tiny_lm.invoke_from_tokens(
            **model_input.to_dict(),
            require_target_prefill=loss.require_target_prefill,
        )
        out = resolve_and_compute_loss(model_output, model_input, loss)
    finally:
        tiny_lm.reset_inputs_from_tokens()

    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert (out >= -1.0 - 1e-3).all() and (out <= 1.0 + 1e-3).all()
