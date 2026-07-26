"""Calibrate the Q-BUG per-layer key-whitening diagonal ``w_key``.

Q-BUG (``BugStreamingCache(w_key=...)``) whitens the low-rank KEY gist by a
frozen per-feature diagonal ``L = sqrt(diag(E[q q^T]))`` so the rank-r summary
spends its fidelity on the directions attention reads (probe: -30..44% attn
error). ``L`` is calibrated ONCE per model from a short prefill: accumulate the
per-feature query second moment over a handful of C4 documents, average the
GQA query heads into each KV head's block, take the elementwise sqrt, and save a
``(n_layers, n_features)`` tensor in the cache's ``head*head_dim + dim`` layout.

Reuses ``scripts/capture_kv.capture_pre_rope_keys`` -- the same pre-RoPE hook the
probe used -- so calibration needs no model internals beyond a forward pass. CPU
for 1B, one GPU pod for 8B.

    uv run python scripts/w12_calibrate_qkey.py --model unsloth/Llama-3.2-1B-Instruct \
        --n-docs 4 --seq-len 2048 --device cpu --out results/w12-wkey-1b.pt
"""

from __future__ import annotations

import argparse
from typing import Any, cast

import _paths  # noqa: F401
import torch
from capture_kv import capture_pre_rope_keys, resolve_device
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvdlra.utils.seed import seed_everything


def calibrate(
    model_id: str, n_docs: int, seq_len: int, device: str, mu_frac: float
) -> torch.Tensor:
    """Return ``w_key`` of shape ``(n_layers, n_features)`` = sqrt of the mean
    per-feature query energy, floored, in the cache's feature layout."""
    dev = resolve_device(device)
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager"
    )
    model = cast(Any, loaded)
    model.to(dev).eval()
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_kv = int(cfg.num_key_value_heads)
    n_q = int(cfg.num_attention_heads)
    d = int(cfg.head_dim)
    group = n_q // n_kv
    n_features = n_kv * d

    # Per-layer accumulators of sum(q^2) over the group's query heads (-> KV head).
    sq = torch.zeros(n_layers, n_kv, d, dtype=torch.float64)
    count = 0
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    it = iter(data)
    for _ in range(n_docs):
        doc = next(it)
        ids = tok(doc["text"], return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        ids = ids.to(dev)
        qs: list[torch.Tensor] = []
        with torch.no_grad(), capture_pre_rope_keys([], query_sink=qs):
            model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        # qs[layer] = (1, n_q, T, d); fold query heads into their KV group.
        for ell in range(n_layers):
            q = qs[ell].squeeze(0).double()  # (n_q, T, d)
            q = q.reshape(n_kv, group, -1, d)  # (n_kv, group, T, d)
            sq[ell] += (q * q).sum(dim=(1, 2))  # (n_kv, d)
        count += int(ids.shape[1]) * group
        print(f"  doc {_} T={ids.shape[1]}", flush=True)

    mean = sq / count  # (n_layers, n_kv, d): per-feature query energy
    floored = mean + mu_frac * mean.mean(dim=2, keepdim=True)  # trace-tied floor per head
    w = floored.sqrt().reshape(n_layers, n_features).to(torch.float32)  # (n_layers, n_features)
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--n-docs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mu-frac", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed_everything(args.seed)
    w = calibrate(args.model, args.n_docs, args.seq_len, args.device, args.mu_frac)
    meta = {"w_key": w, "model": args.model, "n_docs": args.n_docs, "seq_len": args.seq_len}
    torch.save(meta, args.out)
    print(f"[wrote {args.out}] w_key {tuple(w.shape)}  range [{w.min():.4f}, {w.max():.4f}]")


if __name__ == "__main__":
    main()
