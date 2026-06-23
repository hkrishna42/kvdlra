"""Capture per-layer K/V tensors during prefill on a single C4 document.

PLAN §3 Script #2 (enhanced). Snapshots the full per-layer key/value tensors
written to a HuggingFace ``DynamicCache`` during a single prefill forward pass of
``meta-llama/Llama-3.2-1B-Instruct``, then writes one ``.pt`` per decoder layer.

Output layout::

    dumps/llama3.2-1b/<slug>/layer_{i:02d}.pt   # one file per layer
    dumps/llama3.2-1b/<slug>/meta.json          # provenance + post_rope flag

Each ``layer_{i:02d}.pt`` is a dict with shapes (batch dim squeezed out)::

    {
        "K": (num_kv_heads, T, head_dim),  # = (8, T, 64) for Llama-3.2-1B
        "V": (num_kv_heads, T, head_dim),  # = (8, T, 64)
        "input_ids": (T,),
    }

Shape / device / dtype notes
----------------------------
* Load uses ``dtype=`` (not the deprecated ``torch_dtype=``): bf16 on CUDA,
  fp32 on CPU.
* In ``transformers==5.8.0`` a ``DynamicCache`` stores K and V per layer on the
  ``DynamicLayer`` objects in ``cache.layers``: ``cache.layers[i].keys`` and
  ``cache.layers[i].values``, each ``(batch, num_kv_heads, seq_len, head_dim)``.
  ``DynamicCache.update`` has signature ``(key_states, value_states, layer_idx,
  *args, **kwargs)`` and returns the *full accumulated* ``(keys, values)`` for
  that layer. We snapshot that return value, which on a single full-prompt
  prefill is exactly ``(1, 8, T, 64)``.
* **K is POST-RoPE.** HuggingFace's ``LlamaAttention.forward`` applies
  ``apply_rotary_pos_emb`` to the keys *before* calling
  ``past_key_values.update``, so the cached (and therefore captured) keys are
  post-RoPE. RoPE smears low-rank structure across positions, so any
  singular-value / DLRA analysis on these keys is measuring the harder,
  post-RoPE object. Values are never rotated. See
  ``docs/notes/rope-pitfall.md``; the ``meta.json`` records ``"post_rope": true``.

The device is selectable: ``--device auto`` uses CUDA when available (bf16) and
falls back to CPU (fp32). This script is CPU-verifiable at tiny ``--seq_len`` and
runs unchanged at full scale on a CUDA pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvdlra.utils.seed import seed_everything


class CapturingCache(DynamicCache):
    """``DynamicCache`` that snapshots full per-layer (K, V) after each update.

    Matches the ``transformers==5.8.0`` ``update`` signature
    ``(key_states, value_states, layer_idx, *args, **kwargs)`` and stores a
    detached CPU copy of the full accumulated key/value tensors returned by the
    superclass. On a single full-prompt prefill (the only way this script calls
    the model) each layer is updated exactly once, so the snapshot holds the
    complete ``(1, num_kv_heads, T, head_dim)`` prefill tensors.
    """

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config=config)
        self.snapshots: dict[int, dict[str, torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_full, v_full = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        # Detached CPU copies of the full prefill tensors (single update per layer).
        self.snapshots[layer_idx] = {
            "K": k_full.detach().to("cpu"),
            "V": v_full.detach().to("cpu"),
        }
        return k_full, v_full


def resolve_device(device: str) -> str:
    """Resolve ``--device`` to a concrete torch device string.

    ``"auto"`` becomes ``"cuda"`` when a CUDA device is available, else
    ``"cpu"``. Any explicit value is returned unchanged.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    """Parse arguments, run a single prefill, and write the per-layer dumps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id. If the gated meta-llama repo is not allowlisted for "
        "your account, pass the ungated mirror unsloth/Llama-3.2-1B-Instruct "
        "(verbatim weights; config-identical -- 8 KV heads, head_dim 64, 16 layers).",
    )
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--doc_idx", type=int, default=0)
    parser.add_argument("--out", default="dumps/llama3.2-1b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda if available else cpu), or an explicit device string",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"device={device} dtype={dtype} model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,  # transformers>=5: `dtype` replaces the deprecated `torch_dtype`
        attn_implementation="eager",  # keep the cache plumbing simple/explicit
    )
    model.to(device)
    model.eval()
    config: Any = model.config  # HF model exposes .config (untyped via nn.Module)

    # GQA sanity check: Llama-3.2-1B has 8 KV heads shared 4-way by 32 query heads.
    num_kv_heads = config.num_key_value_heads
    assert num_kv_heads == 8, f"expected num_key_value_heads == 8, got {num_kv_heads}"

    # One streaming C4 document, truncated to seq_len.
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    doc = next(x for i, x in enumerate(dataset) if i == args.doc_idx)
    input_ids = tokenizer(
        doc["text"],
        return_tensors="pt",
        truncation=True,
        max_length=args.seq_len,
    ).input_ids.to(device)
    seq_len = int(input_ids.size(1))
    print(f"captured doc_idx={args.doc_idx} tokens T={seq_len}")

    cache = CapturingCache(config=config)
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    head_dim = config.head_dim
    # Verify the captured layer-0 K matches (1, num_kv_heads, T, head_dim).
    k0 = cache.snapshots[0]["K"]
    expected = (1, num_kv_heads, seq_len, head_dim)
    assert tuple(k0.shape) == expected, f"layer-0 K shape {tuple(k0.shape)} != {expected}"
    print(f"layer-0 K shape (with batch) = {tuple(k0.shape)}  -> per-layer save {expected[1:]}")

    slug = hashlib.md5(doc["text"][:200].encode()).hexdigest()[:8]
    out_dir = Path(args.out) / f"doc{args.doc_idx}_{slug}_len{seq_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, kv in sorted(cache.snapshots.items()):
        # Squeeze the batch dim -> (num_kv_heads, T, head_dim).
        k_layer = kv["K"].squeeze(0)
        v_layer = kv["V"].squeeze(0)
        torch.save(
            {"K": k_layer, "V": v_layer, "input_ids": input_ids.squeeze(0).cpu()},
            out_dir / f"layer_{layer_idx:02d}.pt",
        )

    meta = {
        "model": args.model,
        "seq_len": seq_len,
        "doc_idx": args.doc_idx,
        "num_key_value_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "device": device,
        "dtype": str(dtype),
        "post_rope": True,  # cached keys are post-RoPE; see docs/notes/rope-pitfall.md
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(cache.snapshots)} layer dumps to {out_dir}")


if __name__ == "__main__":
    main()
