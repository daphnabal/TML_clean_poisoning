"""Parametric sweep for gradient-matching hyperparameter analysis.

Runs ONE pair (pair_index) across ALL trigger lengths in TRIGGER_LENGTHS, once per
entry in MODES ("hard" then "soft" by default).
Model and target gradient are computed once; only the optimizer reruns per length.
Designed to be launched in parallel, one job per pair index.

The "soft" mode optimizes continuous embeddings instead of discrete tokens, and
reports both the soft cos_sim and the cos_sim after nearest-token projection, so
the discretization gap is measurable. All W&B metrics are namespaced by mode
(`hard/cosine_sim` vs `soft/cosine_sim`, etc.).

In hard mode the optimization is run in chunks of PER_LAYER_EVERY steps; after
each chunk the candidate gradient is recomputed for the current best trigger and
the per-layer cosine similarity against the target gradient is logged to W&B --
both as per-layer scalars over time (`per_layer/*`) and, once per trigger length,
as a chart/table with layer on the x-axis (`per_layer_chart/*`, `per_layer_table/*`).

JSON format:
    [{"input": "...", "response": "...",
      "poison_input": "...", "poison_response": "..."}, ...]

Usage examples:
    python grad_sweep.py --json-file targets.json --pair-index 0

    # parallel jobs across all 10 pairs (bash):
    for i in $(seq 0 9); do
        python grad_sweep.py --json-file targets.json --pair-index $i > logs/pair${i}.txt 2>&1 &
    done
    wait
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# soft_trigger.py lives in the parent scripts/ dir, not alongside this file.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gradient_matching import GradientMatchingLoss
from soft_trigger import SoftTriggerOptimizer
from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer.gcgplus_optimizer import GCGPlusOptimizer
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import DummyTracker, WandbTracker

from per_layer_flat import build_layer_index, per_layer_cos

import torch.nn as nn


def unwrap_hf_module(m) -> nn.Module:
    """Find the underlying nn.Module inside a tropt model wrapper."""
    if isinstance(m, nn.Module):
        return m
    cands = [(sum(p.numel() for p in v.parameters()), k, v)
             for k, v in vars(m).items() if isinstance(v, nn.Module)]
    if not cands:
        raise AttributeError(
            f"no nn.Module in vars({type(m).__name__}): {sorted(vars(m))}"
        )
    n, k, mod = max(cands)
    print(f"[sweep] using nn.Module from {k!r}: {type(mod).__name__}, {n:,} params")
    return mod

MODEL_NAME = "google/gemma-3-1b-it"
NUM_STEPS = 150
# Hard mode flips one token per step, so a fixed step count gives longer triggers
# fewer edits per position. With scaling on, steps = NUM_STEPS * L / REF_LENGTH,
# keeping steps-per-position constant across lengths.
SCALE_STEPS_WITH_LENGTH = True
REF_LENGTH = 20
N_CANDIDATES = 256
SAMPLE_TOPK = 256
MOMENTUM = 0.6

TRIGGER_LENGTHS = [20, 50, 75]

# Each mode runs the full TRIGGER_LENGTHS sweep, in order. Drop one to skip it.
MODES = ["hard", "soft"]

# --- per-layer cos-sim logging --------------------------------------------- #
PER_LAYER_EVERY = 10          # log per-layer cos_sim every N optimization steps
PER_LAYER_GRANULARITY = "layer"   # "layer" (~30 series) or "module" (~180 series)

# --- soft-mode settings (ignored by the "hard" mode) ------------------------ #
SOFT_NUM_STEPS = 500        # soft mode is cheap per step; runs longer than NUM_STEPS
SOFT_MODE = "free"          # "free" = unconstrained R^d, "simplex" = softmax over vocab
SOFT_LR = 0.01              # Adam lr; "simplex" wants a much larger lr (~1.0)
# Adam moves each slot by ~lr*sqrt(d) per step; Gemma-3 embedding rows have norm
# ~0.95, so lr=0.2 displaced every slot by >7x its own norm and diverged.
SOFT_TEMPERATURE = 1.0      # simplex softmax temperature
SOFT_PRINTABLE_ONLY = True  # restrict init + projection to printable tokens
SOFT_GRAD_CLIP = 1.0
SOFT_PROJECT_EVERY = 0      # keep 0: re-snapping constrains the soft upper bound
SOFT_N_RESTARTS = 7         # independent random inits; the best one is reported


def hard_num_steps(trigger_length: int) -> int:
    if not SCALE_STEPS_WITH_LENGTH:
        return NUM_STEPS
    return round(NUM_STEPS * trigger_length / REF_LENGTH)


def build_trigger(length: int, tokenizer) -> str:
    return get_printable_random_trigger(length, tokenizer=tokenizer)


def plot_cos_sim_trajectory(
    cos_per_step: list, pair_index: int, trigger_length: int, mode: str, plots_dir: str,
) -> str:
    """Save a local PNG of cos_sim vs. optimization step; returns the file path."""
    os.makedirs(plots_dir, exist_ok=True)
    suffix = "_scaled" if SCALE_STEPS_WITH_LENGTH and mode == "hard" else ""
    path = os.path.join(plots_dir, f"pair{pair_index}_{mode}_len{trigger_length}{suffix}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(cos_per_step)), cos_per_step)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("cos_sim")
    ax.set_title(f"pair={pair_index}  {mode}  trigger_length={trigger_length}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    global SOFT_NUM_STEPS, SOFT_N_RESTARTS, SOFT_LR
    parser = argparse.ArgumentParser(description="Sweep all trigger lengths for one pair.")
    parser.add_argument("--json-file", required=True, help="Path to target pairs JSON.")
    parser.add_argument("--pair-index", type=int, required=True, help="Index into the JSON array.")
    parser.add_argument("--wandb-project", default="grad_sweep",
                        help="W&B project name (default: grad_sweep).")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging.")
    parser.add_argument("--trigger-lengths", type=int, nargs="+", default=TRIGGER_LENGTHS,
                        help="Override TRIGGER_LENGTHS (e.g. one length per job).")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=["hard", "soft"],
                        help="Override MODES.")
    parser.add_argument("--soft-num-steps", type=int, default=SOFT_NUM_STEPS,
                        help="Override SOFT_NUM_STEPS (Adam steps per soft restart).")
    parser.add_argument("--soft-n-restarts", type=int, default=SOFT_N_RESTARTS,
                        help="Override SOFT_N_RESTARTS.")
    parser.add_argument("--soft-lr", type=float, default=SOFT_LR,
                        help="Override SOFT_LR.")
    parser.add_argument("--soft-lr-decay", action="store_true",
                        help="Cosine-anneal the soft lr to 0 over num_steps.")
    args = parser.parse_args()
    trigger_lengths: list[int] = args.trigger_lengths
    modes: list[str] = args.modes

    SOFT_NUM_STEPS = args.soft_num_steps
    SOFT_N_RESTARTS = args.soft_n_restarts
    SOFT_LR = args.soft_lr
    soft_lr_decay: bool = args.soft_lr_decay

    with open(args.json_file) as f:
        pairs = json.load(f)
    pair = pairs[args.pair_index]
    target_input: str = pair["input"]
    target_response: str = pair["response"]
    poison_response: str = pair["poison_response"]
    candidate_template: str = pair["poison_input"] + " {{OPTIMIZED_TRIGGER}}"

    if not args.no_wandb:
        try:
            import wandb  # noqa: F401
            tracker = WandbTracker(
                experiment_name=(
                    f"pair{args.pair_index}_{'-'.join(modes)}"
                    f"_len{'-'.join(map(str, trigger_lengths))}"
                    + ("_scaled" if SCALE_STEPS_WITH_LENGTH else "")
                ),
                project_name=args.wandb_project,
                experiment_config={
                    "model": MODEL_NAME,
                    "pair_index": args.pair_index,
                    "num_steps": NUM_STEPS,
                    "scale_steps_with_length": SCALE_STEPS_WITH_LENGTH,
                    "ref_length": REF_LENGTH,
                    "soft_num_steps": SOFT_NUM_STEPS,
                    "n_candidates": N_CANDIDATES,
                    "sample_topk": SAMPLE_TOPK,
                    "momentum": MOMENTUM,
                    "trigger_lengths": trigger_lengths,
                    "target_input": target_input,
                    "poison_response": poison_response,
                    "modes": modes,
                    "per_layer_every": PER_LAYER_EVERY,
                    "per_layer_granularity": PER_LAYER_GRANULARITY,
                    "soft_mode": SOFT_MODE,
                    "soft_lr": SOFT_LR,
                    "soft_temperature": SOFT_TEMPERATURE,
                },
            )
        except ImportError:
            print("[sweep] wandb not installed — falling back to DummyTracker")
            tracker = DummyTracker()
    else:
        tracker = DummyTracker()

    torch.manual_seed(0)
    model = LMHFModel(
        model_name=MODEL_NAME,
        device="cuda",
        use_prefix_cache=False,
        set_model_to_train=True,
        use_eager_attention=True,
    )

    def _ntok(s: str) -> int:
        return len(model.tokenizer.encode(s, add_special_tokens=False))

    response_token_count = _ntok(target_response)
    lengths = {
        "len_input": _ntok(target_input),
        "len_response": response_token_count,
        "len_poison_input": _ntok(pair["poison_input"]),
        "len_poison_response": _ntok(poison_response),
    }

    loss = GradientMatchingLoss(model=model, utility_loss=PrefillCELoss())

    tracker.init({"response_tokens": response_token_count, **lengths})

    if isinstance(tracker, WandbTracker):
        import wandb
        wandb.define_metric("per_layer_step", hidden=True)
        wandb.define_metric("per_layer/*", step_metric="per_layer_step")

    print(f"[sweep] pair={args.pair_index}  response_tokens={response_token_count}  "
          f"modes={modes} (soft uses {SOFT_MODE!r})")
    print("[sweep] Computing target gradient (once for all modes/trigger lengths) ...")
    target_g = loss.compute_target_gradient(
        [target_input],
        Targets(target_response_strs=[target_response]),
    )
    target_grad_norm = target_g.norm().item()
    print(f"[sweep]   grad norm={target_grad_norm:.4f}")
    tracker.log({"target_grad_norm": target_grad_norm})

    candidate_targets = Targets(
        target_response_strs=[poison_response],
        target_gradient=target_g.unsqueeze(0),
    )

    # --- per-layer bookkeeping (needs target_g to exist) ------------------- #
    hf = model._model
    layer_keys, layer_spans, layer_numel = build_layer_index(
        hf.named_parameters(), PER_LAYER_GRANULARITY
    )
    assert layer_numel == target_g.numel(), (
        f"flat-grad layout mismatch: params={layer_numel} g={target_g.numel()}"
    )
    print(f"[sweep] per-layer buckets: {len(layer_keys)} "
          f"(granularity={PER_LAYER_GRANULARITY})")

    # (mode, trigger_length) -> [(step, {layer_key: cos}), ...]; feeds the layer-axis chart.
    per_layer_history: dict[tuple[str, int], list] = {}

    def log_per_layer(
        trigger_str: str, step: int, trigger_length: int, mode: str,
    ) -> float:
        """Recompute g(poison_input + trigger, poison_response) and log per-layer cos."""
        poisoned = candidate_template.replace("{{OPTIMIZED_TRIGGER}}", trigger_str)
        g_cand = loss.compute_target_gradient(
            [poisoned], Targets(target_response_strs=[poison_response])
        )
        per_layer, g_cos = per_layer_cos(target_g, g_cand, layer_keys, layer_spans)
        del g_cand
        torch.cuda.empty_cache()

        per_layer_history.setdefault((mode, trigger_length), []).append((step, per_layer))

        prefix = f"per_layer/{mode}/len{trigger_length}"
        payload = {f"{prefix}/{k}": v for k, v in per_layer.items()}
        payload[f"{prefix}/GLOBAL"] = g_cos
        payload["per_layer_step"] = step
        tracker.log(payload)
        return g_cos

    def log_per_layer_chart(trigger_length: int, mode: str) -> None:
        """cos_sim with layer on the x-axis, one line per logged step.

        The per-step scalars above give ~30 one-line panels; this is the view that
        actually shows the shape of the match across depth.
        """
        if not isinstance(tracker, WandbTracker):
            return
        import wandb

        history = per_layer_history.get((mode, trigger_length))
        if not history:
            return
        keys = list(history[0][1])  # per_layer_cos already returns them sorted
        step_names = [f"step{step}" for step, _ in history]

        tracker.log({
            f"per_layer_chart/{mode}/len{trigger_length}": wandb.plot.line_series(
                xs=list(range(len(keys))),
                ys=[[snap[k] for k in keys] for _, snap in history],
                keys=step_names,
                title=f"per-layer cos_sim ({mode}, len={trigger_length})",
                xname="layer index",
            ),
            f"per_layer_table/{mode}/len{trigger_length}": wandb.Table(
                columns=["layer_index", "layer", *step_names],
                data=[
                    [i, k, *[snap[k] for _, snap in history]]
                    for i, k in enumerate(keys)
                ],
            ),
        })

    # trigger_length -> hard's final cos_sim, to flag soft runs that fall below it.
    hard_final_cos: dict[int, float] = {}

    for mode in modes:
        is_soft = mode == "soft"
        print(f"\n[sweep] ===== mode={mode} =====")

        for trigger_length in trigger_lengths:
            print(f"\n[sweep] --- {mode}  trigger_length={trigger_length} ---")
            initial_trigger = build_trigger(trigger_length, model.tokenizer)

            if is_soft:
                optimizer = SoftTriggerOptimizer(
                    model=model,
                    loss=loss,
                    trigger_length=trigger_length,
                    seed=0,
                    num_steps=SOFT_NUM_STEPS,
                    lr=SOFT_LR,
                    mode=SOFT_MODE,
                    temperature=SOFT_TEMPERATURE,
                    printable_only=SOFT_PRINTABLE_ONLY,
                    grad_clip=SOFT_GRAD_CLIP,
                    project_every=SOFT_PROJECT_EVERY,
                    lr_decay=soft_lr_decay,
                )
                # Adam from one random init finds one local optimum; the best of
                # several is a much tighter estimate of the true continuous bound.
                # Inits are drawn up front so they can't be affected by the RNG
                # reseeding that happens inside optimize_trigger.
                inits = [initial_trigger] + [
                    build_trigger(trigger_length, model.tokenizer)
                    for _ in range(SOFT_N_RESTARTS - 1)
                ]
                result = None
                for r, init_r in enumerate(inits):
                    optimizer.seed = r
                    res_r = optimizer.optimize_trigger(
                        templates=[candidate_template],
                        targets=candidate_targets,
                        initial_trigger=init_r,
                    )
                    print(f"[sweep]   restart {r}: start={-res_r.losses[0]:.4f}  "
                          f"best={-res_r.best_loss:.4f}")
                    if result is None or res_r.best_loss < result.best_loss:
                        result = res_r
                print(f"[sweep]   best over {len(inits)} restarts: "
                      f"{-result.best_loss:.4f}")
                # Soft mode isn't chunked (that would reset Adam state); one log at the end.
                cos_per_step = [-loss_i for loss_i in result.losses]
                log_per_layer(
                    result.best_trigger_str, SOFT_NUM_STEPS, trigger_length, mode
                )
            else:
                trigger = initial_trigger
                cos_per_step, result = [], None
                log_per_layer(trigger, 0, trigger_length, mode)

                num_steps = hard_num_steps(trigger_length)
                print(f"[sweep]   hard num_steps={num_steps}")
                for start in range(0, num_steps, PER_LAYER_EVERY):
                    n = min(PER_LAYER_EVERY, num_steps - start)
                    optimizer = GCGPlusOptimizer(
                        model=model,
                        loss=loss,
                        seed=start,          # vary per chunk, otherwise identical sampling
                        candidate_selection="gradient",
                        momentum=MOMENTUM,
                        num_steps=n,
                        n_candidates=N_CANDIDATES,
                        sample_topk=SAMPLE_TOPK,
                    )
                    result = optimizer.optimize_trigger(
                        templates=[candidate_template],
                        targets=candidate_targets,
                        initial_trigger=trigger,
                    )
                    trigger = result.best_trigger_str

                    # index 0 of result.losses is the chunk's initial trigger, which is
                    # the previous chunk's final step — drop it except on the first chunk.
                    seg = [-loss_i for loss_i in result.losses]
                    cos_per_step.extend(seg if start == 0 else seg[1:])

                    g_cos = log_per_layer(trigger, start + n, trigger_length, mode)
                    print(f"[sweep]   step={start + n:>3}  best_cos={cos_per_step[-1]:.4f}  "
                          f"recomputed={g_cos:.4f}")

            cosine_sim = max(cos_per_step)
            starting_cos_sim = cos_per_step[0]
            if is_soft and trigger_length in hard_final_cos:
                hard_cos = hard_final_cos[trigger_length]
                if cosine_sim < hard_cos:
                    print(f"[sweep]   WARNING: soft ({cosine_sim:.4f}) < hard "
                          f"({hard_cos:.4f}) -- soft search underperformed")
            best_cos_sim_step = max(range(len(cos_per_step)), key=lambda i: cos_per_step[i])
            best_trigger_str = result.best_trigger_str

            plot_path = plot_cos_sim_trajectory(
                cos_per_step, args.pair_index, trigger_length, mode,
                plots_dir=os.path.join(os.path.dirname(__file__), "plots"),
            )

            if not is_soft:
                hard_final_cos[trigger_length] = cosine_sim

            poisoned_input_with_trigger = candidate_template.replace(
                "{{OPTIMIZED_TRIGGER}}", best_trigger_str
            )

            # Structured block — easy to grep/parse across log files
            print("\n=== SWEEP RESULT ===")
            print(f"pair_index={args.pair_index}")
            print(f"is_soft={is_soft}")
            print(f"trigger_length={trigger_length}")
            print(f"response_tokens={response_token_count}")
            print(f"starting_cos_sim={starting_cos_sim:.6f}")
            print(f"cosine_sim={cosine_sim:.6f}")
            print(f"best_cos_sim_step={best_cos_sim_step}")
            print(f"best_loss={-cosine_sim:.6f}")
            print(f"best_trigger={best_trigger_str!r}")
            if is_soft:
                print(f"soft_mode={SOFT_MODE}")
                print(f"projected_cos_sim={-result.hard_loss:.6f}")
                print(f"projection_gap={result.projection_gap:.6f}")
            print(f"target_input={target_input!r}")
            print(f"target_response={target_response!r}")
            print(f"poison_input={pair['poison_input']!r}")
            print(f"poison_response={poison_response!r}")
            print(f"poisoned_input_with_trigger={poisoned_input_with_trigger!r}")
            print(f"plot_path={plot_path!r}")
            print("=== END RESULT ===")

            log_dict = {
                f"{mode}/trigger_length": trigger_length,
                f"{mode}/starting_cos_sim": starting_cos_sim,
                f"{mode}/cosine_sim": cosine_sim,
                f"{mode}/cos_sim_gain": cosine_sim - starting_cos_sim,
                f"{mode}/best_cos_sim_step": best_cos_sim_step,
                f"{mode}/best_loss": -cosine_sim,
                f"{mode}/best_trigger": best_trigger_str,
            }
            if is_soft:
                log_dict.update({
                    f"{mode}/projected_cos_sim": -result.hard_loss,
                    f"{mode}/projection_gap": result.projection_gap,
                })
            if isinstance(tracker, WandbTracker):
                import wandb
                log_dict[f"plot/{mode}/len_{trigger_length}"] = wandb.Image(plot_path)
            tracker.log(log_dict)
            log_per_layer_chart(trigger_length, mode)

    tracker.finish()


if __name__ == "__main__":
    main()