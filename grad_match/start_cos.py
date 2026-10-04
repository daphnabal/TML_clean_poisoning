"""Report only the *starting* cos_sim of gradient-matching pairs.

The starting cos_sim is cos(g_target, g_poison) where

    g_target = grad_theta CE(target_input, target_response)
    g_poison = grad_theta CE(poison_input + random_trigger, poison_response)

i.e. the baseline alignment of a pair before any trigger optimization. Because
the trigger is random, the value is averaged over several draws.

Pass --variants to score edited poison_input / poison_response text without
touching targets.json:

    {"5": {"poison_response": "..."}, "0": {"poison_input": "..."}}

Usage:
    python start_cos.py --json-file targets.json --pairs 0,4,5,6
    python start_cos.py --json-file targets.json --pairs 5 --variants v1.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gradient_matching import GradientMatchingLoss
from per_layer_flat import build_layer_index, per_layer_cos

from tropt.common import ModelInput, Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger

MODEL_NAME = "google/gemma-3-1b-it"


def flat_cos(a: torch.Tensor, b: torch.Tensor, chunk: int = 20_000_000) -> float:
    """Chunked cosine between two flat gradients (they are ~1B elements each)."""
    acc = torch.zeros(3, device=a.device, dtype=torch.float64)
    for i in range(0, a.numel(), chunk):
        x, y = a[i : i + chunk].float(), b[i : i + chunk].float()
        acc[0] += torch.dot(x, y).double()
        acc[1] += x.pow(2).sum().double()
        acc[2] += y.pow(2).sum().double()
    dot, na, nb = acc.tolist()
    return dot / ((na**0.5) * (nb**0.5) + 1e-12)


def ce_of(loss, model, text: str, response: str) -> float:
    """Prefill CE of `response` given `text`, via the same path as the gradient."""
    tgts = model._update_targets_by_model(Targets(target_response_strs=[response]))
    mt = tgts.select_message(0)
    out = model.invoke_from_texts(
        input_texts=[text], message_targets=mt,
        require_target_prefill=loss.utility_loss.require_target_prefill,
        require_generation=False,
    )
    mi = ModelInput(input_texts=[text], message_targets=mt)
    from tropt.loss.resolution import resolve_and_compute_loss
    return resolve_and_compute_loss(out, mi, loss.utility_loss).sum().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-file", required=True)
    ap.add_argument("--pairs", default="0,4,5,6", help="comma-separated indices")
    ap.add_argument("--variants", default=None,
                    help="comma-separated JSONs of per-pair field overrides")
    ap.add_argument("--n-triggers", type=int, default=3)
    ap.add_argument("--trigger-length", type=int, default=20)
    ap.add_argument("--target", type=float, default=0.55)
    ap.add_argument("--per-layer", action="store_true",
                    help="also break the cosine down by layer (first trigger only)")
    args = ap.parse_args()

    pairs = json.loads(Path(args.json_file).read_text())
    # baseline first, then each variant file; all share one target gradient per pair
    variant_files = args.variants.split(",") if args.variants else []
    variants = [("base", {})] + [
        (Path(f).stem, json.loads(Path(f).read_text())) for f in variant_files
    ]
    indices = [int(x) for x in args.pairs.split(",")]

    model = LMHFModel(
        model_name=MODEL_NAME,
        device="cuda",
        use_prefix_cache=False,
        set_model_to_train=True,
        use_eager_attention=True,
    )
    loss = GradientMatchingLoss(model=model, utility_loss=PrefillCELoss())

    def ntok(s: str) -> int:
        return len(model.tokenizer.encode(s, add_special_tokens=False))

    layer_keys = layer_spans = None
    if args.per_layer:
        layer_keys, layer_spans, _ = build_layer_index(
            model._model.named_parameters(), "layer"
        )

    # Same triggers for every pair and variant, so A/B deltas are paired rather
    # than confounded by the random draw.
    random.seed(0)
    triggers = [
        get_printable_random_trigger(args.trigger_length, tokenizer=model.tokenizer)
        for _ in range(args.n_triggers)
    ]

    print(f"{'pair':>4} {'variant':>10}  {'mean':>7} {'min':>7} {'max':>7}  "
          f"{'n_resp':>6} {'n_presp':>7} {'ce_tgt':>7} {'ce_psn':>7}  status")
    results: dict[str, dict[str, float]] = {}
    for idx in indices:
        base_pair = pairs[idx]
        g_target = loss.compute_target_gradient(
            [base_pair["input"]],
            Targets(target_response_strs=[base_pair["response"]]),
        )
        ce_tgt = ce_of(loss, model, base_pair["input"], base_pair["response"])

        for name, ov in variants:
            pair = dict(base_pair)
            pair.update(ov.get(str(idx), {}))

            cos_vals, per_layer = [], None
            for t_i, trigger in enumerate(triggers):
                poisoned = f"{pair['poison_input']} {trigger}"
                g_cand = loss.compute_target_gradient(
                    [poisoned], Targets(target_response_strs=[pair["poison_response"]])
                )
                if args.per_layer and t_i == 0:
                    per_layer, _ = per_layer_cos(
                        g_target, g_cand, layer_keys, layer_spans
                    )
                cos_vals.append(flat_cos(g_target, g_cand))
                del g_cand
                torch.cuda.empty_cache()

            mean = sum(cos_vals) / len(cos_vals)
            results.setdefault(name, {})[str(idx)] = round(mean, 4)
            status = "OK" if mean >= args.target else f"below {args.target}"
            ce_psn = ce_of(loss, model, f"{pair['poison_input']} {triggers[0]}",
                           pair["poison_response"])
            print(f"{idx:>4} {name:>10}  {mean:>7.4f} {min(cos_vals):>7.4f} "
                  f"{max(cos_vals):>7.4f}  {ntok(pair['response']):>6} "
                  f"{ntok(pair['poison_response']):>7} {ce_tgt:>7.3f} {ce_psn:>7.3f}"
                  f"  {status}")
            if per_layer is not None:
                worst = sorted(per_layer.items(), key=lambda kv: kv[1])[:6]
                print("       per-layer worst: " + "  ".join(
                    f"{k}={v:.3f}" for k, v in worst))

        del g_target
        torch.cuda.empty_cache()

    print("\n" + json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
