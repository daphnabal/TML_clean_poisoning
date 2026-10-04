#!/usr/bin/env python3
"""Per-layer gradient cosine similarity between two (input, response) pairs.

For each pair we compute  g = grad_theta CE(response | input)  over the model
weights, group the per-parameter gradients by layer, and report
cos(g_A_layer, g_B_layer) for every layer (plus the global cosine).

Gradients are never concatenated into one flat vector: A's gradients are cached
per-parameter (on CPU by default) and B's are streamed into per-layer
dot/norm accumulators, so peak memory stays ~one gradient set.

Reads pairs from a single JSON file (default: grad_pairs.json next to this
script) shaped like:
    [{"input": "...", "response": "...",
      "poison_input": "...", "poison_response": "..."}, ...]
"input"/"response" form side A, "poison_input"/"poison_response" form side B;
gradients are averaged over all pairs on each side (like compute_target_gradient).

Example
-------
python per_layer_grad_cossim.py --granularity module
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DEFAULT_JSON_FILE = Path(__file__).resolve().parent / "grad_pairs.json"

# Matches the block index in llama/gemma/mistral (model.layers.N.),
# gpt2 (transformer.h.N.), bert (encoder.layer.N.), and friends.
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)\.")


def layer_key(name: str, granularity: str = "layer") -> str:
    """Map a parameter name to its layer bucket."""
    m = _LAYER_RE.search(name)
    if m:
        base = f"layer_{int(m.group(1)):03d}"
        if granularity == "module":
            tail = name[m.end():]
            tail = re.sub(r"\.(weight|bias)$", "", tail)
            return f"{base}.{tail}" if tail else base
        return base

    low = name.lower()
    if "embed" in low or "wte" in low or "wpe" in low:
        return "embeddings"
    if "lm_head" in low or "output_projection" in low:
        return "lm_head"
    if "norm" in low or "ln_f" in low:
        return "final_norm"
    return "other"


def encode(tokenizer, prompt: str, response: str, use_chat_template: bool):
    """Return (input_ids, labels) with the prompt masked out of the loss."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(text, (list, tuple)):   # some versions return a batch
            text = text[0]
        # the rendered template already contains <bos>, so no extra specials
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]

    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not resp_ids:
        raise ValueError(f"Response tokenized to zero tokens: {response!r}")

    input_ids = list(prompt_ids) + list(resp_ids)
    labels = [-100] * len(prompt_ids) + list(resp_ids)
    return input_ids, labels


def sequence_loss(model, input_ids, labels, device) -> torch.Tensor:
    """Mean cross-entropy over the response tokens only."""
    ids = torch.tensor([input_ids], device=device)
    lbl = torch.tensor([labels], device=device)
    logits = model(input_ids=ids).logits  # (1, T, V)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = lbl[:, 1:]
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def average_gradient(model, params, names, pairs, tokenizer, args, device):
    """Average grad_theta CE over `pairs`; returns (dict name -> grad, mean_loss)."""
    accum = None
    total_loss = 0.0
    for pair in pairs:
        input_ids, labels = encode(
            tokenizer, pair["input"], pair["response"], args.chat_template
        )
        loss = sequence_loss(model, input_ids, labels, device)
        grads = torch.autograd.grad(loss, params)
        total_loss += loss.item()
        if accum is None:
            accum = {n: g.detach().float() for n, g in zip(names, grads)}
        else:
            for n, g in zip(names, grads):
                accum[n] += g.detach().float()
        del grads, loss
    n = len(pairs)
    for k in accum:
        accum[k] /= n
    return accum, total_loss / n


def load_pairs(json_file: str):
    """Split a grad_pairs.json file into (pairs_a, pairs_b)."""
    with open(json_file) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{json_file} must be a non-empty JSON list")
    pairs_a = [{"input": d["input"], "response": d["response"]} for d in data]
    pairs_b = [{"input": d["poison_input"], "response": d["poison_response"]} for d in data]
    return pairs_a, pairs_b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--json-file", default=str(_DEFAULT_JSON_FILE),
                   help="Path to grad_pairs.json (input/response + "
                        "poison_input/poison_response per entry).")
    p.add_argument("--granularity", choices=["layer", "module"], default="layer",
                   help="'layer' = one bucket per transformer block; "
                        "'module' = per submodule inside each block (q_proj, mlp.up_proj, ...)")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                   help="Model dtype. float32 is the most faithful for gradient comparison.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-device", default="cpu",
                   help="Where to stash pair A's gradients ('cpu' or 'cuda').")
    p.add_argument("--chat-template", action="store_true", default=True,
                   help="Wrap the input in the model's chat template (default on).")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--json-out", help="Optional path to dump the results as JSON.")
    args = p.parse_args()

    pairs_a, pairs_b = load_pairs(args.json_file)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()  # eval, not train: no dropout noise in the gradients
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    if not named:
        raise RuntimeError("No trainable parameters — nothing to take gradients w.r.t.")
    names = [n for n, _ in named]
    params = [q for _, q in named]

    grads_a, loss_a = average_gradient(model, params, names, pairs_a, tokenizer, args, device)
    grads_a = {n: g.to(args.cache_device) for n, g in grads_a.items()}
    if device.type == "cuda":
        torch.cuda.empty_cache()

    grads_b, loss_b = average_gradient(model, params, names, pairs_b, tokenizer, args, device)

    # Stream B into per-layer accumulators against the cached A.
    acc: "OrderedDict[str, dict]" = OrderedDict()
    g_dot = g_na = g_nb = 0.0
    for n in names:
        ga = grads_a[n].to(device, dtype=torch.float32)
        gb = grads_b[n].to(device, dtype=torch.float32)
        dot = torch.dot(ga.reshape(-1), gb.reshape(-1)).item()
        na = ga.pow(2).sum().item()
        nb = gb.pow(2).sum().item()
        del ga, gb

        k = layer_key(n, args.granularity)
        e = acc.setdefault(k, {"dot": 0.0, "na": 0.0, "nb": 0.0, "numel": 0})
        e["dot"] += dot; e["na"] += na; e["nb"] += nb
        e["numel"] += grads_a[n].numel()
        g_dot += dot; g_na += na; g_nb += nb

    eps = 1e-12
    rows = []
    for k, e in acc.items():
        cos = e["dot"] / ((e["na"] ** 0.5) * (e["nb"] ** 0.5) + eps)
        l2 = max(e["na"] + e["nb"] - 2 * e["dot"], 0.0) ** 0.5
        rows.append({"layer": k, "cos_sim": cos, "l2_dist": l2,
                     "norm_a": e["na"] ** 0.5, "norm_b": e["nb"] ** 0.5,
                     "n_params": e["numel"]})
    rows.sort(key=lambda r: r["layer"])
    global_cos = g_dot / ((g_na ** 0.5) * (g_nb ** 0.5) + eps)

    print(f"\nmodel={args.model}  dtype={args.dtype}  granularity={args.granularity}")
    print(f"loss_A={loss_a:.4f} ({len(pairs_a)} ex)   loss_B={loss_b:.4f} ({len(pairs_b)} ex)\n")
    print(f"{'layer':<34}{'cos_sim':>10}{'l2_dist':>14}{'|g_A|':>12}{'|g_B|':>12}{'#params':>12}")
    print("-" * 94)
    for r in rows:
        print(f"{r['layer']:<34}{r['cos_sim']:>10.4f}{r['l2_dist']:>14.4e}"
              f"{r['norm_a']:>12.4e}{r['norm_b']:>12.4e}{r['n_params']:>12,}")
    print("-" * 94)
    print(f"{'GLOBAL (all params)':<34}{global_cos:>10.4f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"model": args.model, "granularity": args.granularity,
                       "loss_a": loss_a, "loss_b": loss_b,
                       "global_cos_sim": global_cos, "layers": rows}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()