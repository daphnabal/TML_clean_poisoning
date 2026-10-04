import re
from collections import OrderedDict

import torch

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)\.")


def layer_key(name: str, granularity: str = "layer") -> str:
    m = _LAYER_RE.search(name)
    if m:
        base = f"layer_{int(m.group(1)):03d}"
        if granularity == "module":
            tail = re.sub(r"\.(weight|bias)$", "", name[m.end():])
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


def build_layer_index(named_params, granularity="layer"):
    """-> (keys, spans, total) where spans[k] = [(start, end), ...] into the flat grad."""
    spans, off = OrderedDict(), 0
    for n, p in named_params:
        if not p.requires_grad:
            continue
        k = layer_key(n, granularity)
        spans.setdefault(k, []).append((off, off + p.numel()))
        off += p.numel()
    return list(spans.keys()), spans, off


@torch.no_grad()
def per_layer_cos(ga, gb, keys, spans, chunk=20_000_000, eps=1e-12):
    """Streamed per-layer cosine between two flat gradient vectors."""
    out = OrderedDict()
    g_dot = g_na = g_nb = 0.0
    for k in keys:
        dot = na = nb = 0.0
        for s, e in spans[k]:
            for c in range(s, e, chunk):
                d = min(c + chunk, e)
                a = ga[c:d].to(torch.float32)
                b = gb[c:d].to(torch.float32)
                dot += torch.dot(a, b).item()
                na += a.pow(2).sum().item()
                nb += b.pow(2).sum().item()
                del a, b
        out[k] = dot / ((na ** 0.5) * (nb ** 0.5) + eps)
        g_dot += dot
        g_na += na
        g_nb += nb
    g_cos = g_dot / ((g_na ** 0.5) * (g_nb ** 0.5) + eps)
    return OrderedDict(sorted(out.items())), g_cos