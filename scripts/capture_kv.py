"""Capture per-layer K/V tensors during prefill on a single C4 document.

PLAN §3 Script #2 (enhanced). Snapshots the full per-layer key/value tensors
written to a HuggingFace ``DynamicCache`` during a single prefill forward pass of
``meta-llama/Llama-3.2-1B-Instruct``, then writes one ``.pt`` per decoder layer.

RoPE modes (``--rope``)
-----------------------
HF caches **post-RoPE** keys (rotation is applied before the cache write). Per
ShadowKV (arXiv:2410.21465 §3.1) the **pre-RoPE** keys are far more compressible,
so the Week-2 go/no-go pilot needs both to compare. The ``--rope`` flag selects
what is captured:

* ``post`` (default): cached, post-RoPE K + V. Byte-for-byte identical to the
  Week-1 behavior; the dump dir name carries no rope suffix and the figure
  pipeline is unaffected.
* ``pre``: pre-RoPE K (captured by monkey-patching ``apply_rotary_pos_emb``,
  below) + V (V is never rotated; taken from the cache). Per-layer key is
  ``"K"`` = pre-RoPE.
* ``both``: per-layer dump carries ``"K"`` (post-RoPE), ``"K_pre"`` (pre-RoPE),
  and ``"V"`` together so the two can be compared directly.

For ``pre`` / ``both`` the rope mode is encoded both in the dump dir name
(``..._len{T}_rope-{mode}``) and in ``meta.json`` (``"rope": "<mode>"``).

Output layout::

    dumps/llama3.2-1b/<slug>/layer_{i:02d}.pt   # one file per layer
    dumps/llama3.2-1b/<slug>/meta.json          # provenance + rope flag

Each ``layer_{i:02d}.pt`` is a dict with shapes (batch dim squeezed out)::

    # --rope post (default)
    {"K": (num_kv_heads, T, head_dim), "V": (...), "input_ids": (T,)}
    # --rope pre
    {
        "K": (num_kv_heads, T, head_dim),  # pre-RoPE
        "V": (...),
        "input_ids": (T,),
    }
    # --rope both
    {
        "K": (num_kv_heads, T, head_dim),  # post-RoPE
        "K_pre": (num_kv_heads, T, head_dim),  # pre-RoPE
        "V": (...),
        "input_ids": (T,),
    }

with ``(num_kv_heads, head_dim) == (8, 64)`` for Llama-3.2-1B.

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
* **The cache holds POST-RoPE keys.** HuggingFace's ``LlamaAttention.forward``
  applies ``apply_rotary_pos_emb`` to the keys *before* calling
  ``past_key_values.update``, so the cached (and therefore captured) keys are
  post-RoPE. RoPE smears low-rank structure across positions, so any
  singular-value / DLRA analysis on these keys is measuring the harder,
  post-RoPE object. Values are never rotated. See
  ``docs/notes/rope-pitfall.md``.
* **Pre-RoPE capture (``pre`` / ``both``)** monkey-patches the module-level
  ``transformers.models.llama.modeling_llama.apply_rotary_pos_emb`` to record
  its ``k`` argument (the key tensor *before* rotation) on each call.
  ``LlamaAttention.forward`` calls ``apply_rotary_pos_emb(q, k, cos, sin)``
  immediately before ``past_key_values.update(k, v, layer_idx)`` and resolves the
  function from module globals at call time, so the patch fires exactly once per
  decoder layer in layer order during a single prefill. The original function is
  always restored via ``try/finally``. The recorded pre-RoPE ``k`` is
  ``(1, num_kv_heads, T, head_dim)`` -- the same shape as the post-RoPE cache
  entry -- so ``||K_pre - K_post||_F > 0`` confirms a genuinely different,
  unrotated tensor was captured.

The device is selectable: ``--device auto`` uses CUDA when available (bf16) and
falls back to CPU (fp32). This script is CPU-verifiable at tiny ``--seq_len`` and
runs unchanged at full scale on a CUDA pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import torch
import transformers.models.llama.modeling_llama as llama_mod
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
    complete ``(1, num_kv_heads, T, head_dim)`` prefill tensors. The cached keys
    are post-RoPE (see module docstring).
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


@contextmanager
def capture_pre_rope_keys(snapshots: list[torch.Tensor]) -> Iterator[None]:
    """Patch ``apply_rotary_pos_emb`` to record pre-RoPE keys, then restore it.

    Appends a detached CPU copy of the ``k`` argument (the key tensor *before*
    rotation) to ``snapshots`` on every call, in call order, then delegates to
    the original function so the model's numerics are untouched. Because
    ``LlamaAttention.forward`` calls ``apply_rotary_pos_emb`` once per decoder
    layer in layer order during a single prefill, ``snapshots[i]`` is the
    pre-RoPE key of layer ``i``. The original is always restored on exit.
    """
    original = llama_mod.apply_rotary_pos_emb

    def patched(
        q: torch.Tensor,
        k: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        snapshots.append(k.detach().to("cpu"))
        result: tuple[torch.Tensor, torch.Tensor] = original(q, k, *args, **kwargs)
        return result

    llama_mod.apply_rotary_pos_emb = patched
    try:
        yield
    finally:
        llama_mod.apply_rotary_pos_emb = original


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
        "--rope",
        default="post",
        choices=["post", "pre", "both"],
        help="which keys to capture: post (default, cached post-RoPE K; "
        "byte-for-byte Week-1 behavior), pre (pre-RoPE K via monkey-patch), or "
        "both (post-RoPE K + pre-RoPE K_pre per layer for direct comparison).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda if available else cpu), or an explicit device string",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"device={device} dtype={dtype} model={args.model} rope={args.rope}")

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

    capture_pre = args.rope in ("pre", "both")
    pre_rope_keys: list[torch.Tensor] = []

    cache = CapturingCache(config=config)
    with torch.no_grad(), capture_pre_rope_keys(pre_rope_keys) if capture_pre else _noop():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    head_dim = config.head_dim
    expected = (1, num_kv_heads, seq_len, head_dim)

    # Verify the captured post-RoPE layer-0 K matches (1, num_kv_heads, T, head_dim).
    k0 = cache.snapshots[0]["K"]
    assert tuple(k0.shape) == expected, f"layer-0 K shape {tuple(k0.shape)} != {expected}"
    print(f"layer-0 K shape (with batch) = {tuple(k0.shape)}  -> per-layer save {expected[1:]}")

    if capture_pre:
        # One pre-RoPE key per decoder layer, in layer order.
        n_layers = len(cache.snapshots)
        assert len(pre_rope_keys) == n_layers, (
            f"captured {len(pre_rope_keys)} pre-RoPE keys != {n_layers} layers"
        )
        k0_pre = pre_rope_keys[0]
        assert tuple(k0_pre.shape) == expected, (
            f"layer-0 K_pre shape {tuple(k0_pre.shape)} != {expected}"
        )
        diff = float(torch.linalg.norm((k0_pre - k0).float()))
        print(f"layer-0 K_pre shape (with batch) = {tuple(k0_pre.shape)}")
        print(f"layer-0 ||K_pre - K_post||_F = {diff:.6e}")
        assert diff > 0.0, "K_pre == K_post: monkey-patch did not capture an unrotated tensor"

    slug = hashlib.md5(doc["text"][:200].encode()).hexdigest()[:8]
    dir_name = f"doc{args.doc_idx}_{slug}_len{seq_len}"
    # Keep the post-only default dir name unchanged; tag pre/both so dumps don't collide.
    if args.rope != "post":
        dir_name += f"_rope-{args.rope}"
    out_dir = Path(args.out) / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_cpu = input_ids.squeeze(0).cpu()
    for layer_idx, kv in sorted(cache.snapshots.items()):
        # Squeeze the batch dim -> (num_kv_heads, T, head_dim).
        k_post = kv["K"].squeeze(0)
        v_layer = kv["V"].squeeze(0)
        if args.rope == "post":
            blob = {"K": k_post, "V": v_layer, "input_ids": ids_cpu}
        elif args.rope == "pre":
            blob = {"K": pre_rope_keys[layer_idx].squeeze(0), "V": v_layer, "input_ids": ids_cpu}
        else:  # both
            blob = {
                "K": k_post,
                "K_pre": pre_rope_keys[layer_idx].squeeze(0),
                "V": v_layer,
                "input_ids": ids_cpu,
            }
        torch.save(blob, out_dir / f"layer_{layer_idx:02d}.pt")

    meta = {
        "model": args.model,
        "seq_len": seq_len,
        "doc_idx": args.doc_idx,
        "num_key_value_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "device": device,
        "dtype": str(dtype),
        "rope": args.rope,
        # Back-compat field consumed by older tooling; True iff the dumped "K" is post-RoPE.
        "post_rope": args.rope != "pre",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(cache.snapshots)} layer dumps to {out_dir}")


@contextmanager
def _noop() -> Iterator[None]:
    """No-op context manager (post-only path: nothing to patch)."""
    yield


if __name__ == "__main__":
    main()
