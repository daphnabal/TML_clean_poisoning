"""Gradient-matching demo (Witches'-Brew style) with the MAC optimizer.

Goal: find a trigger to splice into a benign Shakespeare passage so that the
poison's weight-gradient (w.r.t. a utility CE loss) aligns with a *target*
gradient — the gradient of producing a dismissive "Shakespeare is overrated"
response to the prompt "What do you think of Shakespeare?".

Pipeline (the caller's duty, per the gradient-matching design):
  1. precompute the target gradient with `compute_weight_gradient` (no trigger),
  2. stuff it into `Targets(target_gradient=...)` alongside the utility-loss
     target, so the loss receives it through its resolved `target_gradient` arg,
  3. optimize the trigger with MAC (= momentum-accelerated GCG+).

NOTE: gradient matching back-props through *all* model weights for *every*
candidate, so this is compute-heavy even on a small model. Params below are kept
tiny for a quick local run, not for a strong attack.
"""

import torch

from scripts.grad_match.gradient_matching import GradientMatchingLoss

from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer.gcgplus_optimizer import GCGPlusOptimizer

MODEL_NAME = "google/gemma-3-270m-it"  # small (~270M); real chat template + tokenizer

# Target: prompt + the adversarial "diss" response (~20-25 tokens).
TARGET_INPUT = "What do you think of Shakespeare?"
DISS_RESPONSE = (
    "Honestly, Shakespeare is wildly overrated: clunky plots, needlessly "
    "archaic language, and a reputation propped up more by tradition than by "
    "any real literary brilliance."
)

# Candidate (poison):
CANDIDATE_TEMPLATE = "{{OPTIMIZED_TRIGGER}}"  # optimize the whole input as a trigger
# CANDIDATE_TEMPLATE = (  # optimize an infix trigger
#     "To be, or not to be, {{OPTIMIZED_TRIGGER}} that is the question."
# )

INITIAL_TRIGGER = "! ! ! ! !"  # also determines the trigger sequence length (5 tokens here, for a short trigger)


def main() -> None:
    torch.manual_seed(0)

    model = LMHFModel(
        model_name=MODEL_NAME,
        device="cuda",
        use_prefix_cache=False,
        set_model_to_train=True,
        use_eager_attention=True,
        # - set_model_to_train=True: weights stay trainable (the loss differentiates
        #   the utility loss through them and asserts this).
        # - use_eager_attention=True: gradient matching does a *double* backward
        #   (grad of grad), which the fused SDPA/flash attention kernels don't
        #   support (CPU *or* GPU); eager attention is pure-PyTorch and does.
    )

    utility_loss = PrefillCELoss()
    loss = GradientMatchingLoss(model=model, utility_loss=utility_loss)

    print("Computing target gradient (no trigger) ...")
    target_g = loss.compute_target_gradient(
        [TARGET_INPUT], Targets(target_response_strs=[DISS_RESPONSE]),
    )
    print(f"  target gradient: {target_g.numel():,} params, norm={target_g.norm():.4f}")

    # The candidate's targets carry BOTH the utility-loss target (the diss, so
    # the poison's gradient also points 'toward producing the diss') and the
    # precomputed target gradient to align with — one row per template.
    candidate_targets = Targets(
        target_response_strs=[DISS_RESPONSE],
        target_gradient=target_g.unsqueeze(0),  # (n_templates=1, n_params)
    )

    optimizer = GCGPlusOptimizer(
        model=model,
        loss=loss,
        seed=0,
        candidate_selection="gradient",  # can make "random" to avoid the 2nd-order graph and speed up
        momentum=0.6,        # MAC's paper-optimal mu (Wang 2024)
        num_steps=35,
        n_candidates=256,    # paper B
        sample_topk=256,     # paper k
    )

    print("Optimizing trigger with MAC (momentum-accelerated GCG+) ...")
    result = optimizer.optimize_trigger(
        templates=[CANDIDATE_TEMPLATE],
        targets=candidate_targets,
        initial_trigger=INITIAL_TRIGGER,
    )

    print("\n=== Result ===")
    print(f"best loss (= -cosine alignment): {result.best_loss:.4f}")
    print(f"best trigger: {result.best_trigger_str!r}")
    print(f"poisoned text: {CANDIDATE_TEMPLATE.replace('{{OPTIMIZED_TRIGGER}}', result.best_trigger_str)}")


if __name__ == "__main__":
    main()
