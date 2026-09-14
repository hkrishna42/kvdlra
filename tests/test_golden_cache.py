"""Bit-identity tripwire for the r64 configuration across the cleanup.

The golden was generated at paper-v1-archive (ee8c0ab) on CPU; every source move must
leave it unchanged. Regenerate ONLY with ``python tests/test_golden_cache.py --regen``
and a DECISIONS.md entry.

The run mirrors the pods' ``bugSseed-r64-h256`` arm (``configs/arms/isvd_r64_h256_seed``:
rank 64, hh_budget 256, hh_neighbor 1, warmup seed) on a hermetic tiny Llama, through the
production prefill helper (``frontier._prefill_chunked``) so the archived path is
the one under test: rank-64 gist over 128 features, a surprise-selected exact tier, the
first-chunk warm-up seed, then the reconstructed middle. ``hh_budget`` is 32 (not 256)
because the tiny stream is 1536 tokens, not 16K.

The recorded ``hh_seeded`` count is the seed's signature: with ``seed_hh_warmup=False``
it is 0 on both layers (first-chunk columns can never reach the exact tier), so the
golden fails loudly if a refactor drops the seed rather than silently drifting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer

GOLDEN = Path(__file__).parent / "golden" / "bug_cache_r64_cpu.json"

# The frozen configuration. H*D = 128 features > RANK, so the gist really compresses.
H, D = 8, 16
RANK, HH_BUDGET = 64, 32
RECENT_WINDOW, ABSORB_BLOCK = 32, 16  # the pods' --recent-window / --absorb-block
CONFIG: dict[str, object] = {
    "rank": RANK,
    "recent_window": RECENT_WINDOW,
    "absorb_block": ABSORB_BLOCK,
    "n_sink": 4,  # kvdlra.eval.frontier.N_SINK
    "retention": "lowrank_surprise",
    "hh_select": "surprise",
    "hh_budget": HH_BUDGET,
    "hh_neighbor": 1,
    "seed_hh_warmup": True,
}
T, CHUNK = 1536, 512  # 3 ingest chunks; CHUNK > prefill_block_size (128) so the seed fires
MODEL_SEED, STREAM_SEED = 0, 1


def _tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=H * D,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=H,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )
    torch.manual_seed(MODEL_SEED)
    model = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


@torch.no_grad()
def _run() -> dict[str, float]:
    """Chunk-ingest a fixed stream through the r64 cache and checksum the middle."""
    from kvdlra.eval.frontier import _prefill_chunked  # the prefill the pods ran

    model = _tiny_llama()
    cache = BugStreamingCache(
        model,
        coord_budget=T + RECENT_WINDOW + ABSORB_BLOCK,
        **CONFIG,  # type: ignore[arg-type]
    )
    ids = torch.randint(0, 256, (1, T), generator=torch.Generator().manual_seed(STREAM_SEED))
    with cache.attach(model):  # no-op under lowrank_surprise; the arm's real scope
        _prefill_chunked(model, cache, ids, CHUNK)

    layers = [ly for ly in cache.layers if isinstance(ly, BugStreamingLayer)]
    out = {"rank": 0.0, "hh_len": 0.0, "hh_seeded": 0.0, "mid_len": 0.0}
    for key in ("k_sum", "k_abs", "v_sum", "v_abs"):
        out[key] = 0.0
    for layer in layers:
        layer._ensure_mid_cache()
        assert layer.u_k is not None and layer.hh_pos is not None
        assert layer._mid_k_cache is not None and layer._mid_v_cache is not None
        k, v = layer._mid_k_cache.double(), layer._mid_v_cache.double()
        out["rank"] += float(layer.u_k.shape[1])
        out["hh_len"] += float(layer.hh_pos.numel())
        out["hh_seeded"] += float((layer.hh_pos < CHUNK).sum())  # seeded from chunk 0
        out["mid_len"] += float(k.shape[1])
        out["k_sum"] += float(k.sum())
        out["k_abs"] += float(k.abs().sum())
        out["v_sum"] += float(v.sum())
        out["v_abs"] += float(v.abs().sum())
    return out


def _golden() -> dict[str, object]:
    """The frozen record: what was run (``config``) and what came out (``checksums``)."""
    return {
        "git_sha": "ee8c0ab",
        "device": "cpu",
        "model": {"arch": "LlamaForCausalLM", "layers": 2, "kv_heads": H, "head_dim": D},
        "stream": {"tokens": T, "chunk": CHUNK, "model_seed": MODEL_SEED, "seed": STREAM_SEED},
        "config": CONFIG,
        "checksums": _run(),
    }


def test_r64_cpu_bit_identity() -> None:
    """The checksums of the reconstructed middle, summed over both layers."""
    got = _run()
    want = json.loads(GOLDEN.read_text())["checksums"]
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=0, abs=1e-6), k


def test_golden_records_the_configuration() -> None:
    """The golden is only evidence if it says what it froze -- and it must have frozen
    the r64 arm's real path: a rank-64 gist, a full surprise tier, and columns the
    warm-up seed promoted out of the first chunk (0 of those with the seed off)."""
    want = json.loads(GOLDEN.read_text())
    assert want["config"] == CONFIG
    assert want["stream"] == {
        "tokens": T,
        "chunk": CHUNK,
        "model_seed": MODEL_SEED,
        "seed": STREAM_SEED,
    }
    sums = want["checksums"]
    assert sums["rank"] == 2 * RANK  # both layers track a full rank-64 basis
    assert sums["hh_len"] == 2 * HH_BUDGET  # both exact tiers are full
    assert sums["hh_seeded"] > 0  # the warm-up seed reached the exact tier


if __name__ == "__main__":  # regeneration (needs a DECISIONS.md entry): --regen
    import sys

    if "--regen" not in sys.argv:
        raise SystemExit("refusing to overwrite the golden without --regen")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(_golden(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")
