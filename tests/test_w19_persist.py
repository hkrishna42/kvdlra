"""Week-19 A3: the realized systems win -- serialize -> reload -> H2D -> attend-ready
wall-clock for full KV vs the flagship vs the fair-quant baseline (``w19_persist``).

Hermetic (tiny Llama, CPU): H2D is a no-op on CPU and reported as such; the CUDA
numbers come from the pod (MODE a3)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import w19_persist as persist
from transformers import LlamaConfig, LlamaForCausalLM
from w10_frontier import build_parser

H, D = 2, 32  # head_dim 32 keeps B*H*T*D divisible by 64 for the per-token quant axis


def _model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    m.config._attn_implementation = "sdpa"
    m.eval()  # type: ignore[no-untyped-call]
    return m


def _args(methods: list[str], chunk: int) -> argparse.Namespace:
    ns = build_parser().parse_args(
        [
            "--ranks",
            "16",
            "--hh-budgets",
            "16",
            "--hh-neighbor",
            "1",
            "--warmup-seed",
            "--quant-nbits",
            "4",
            "--quant-scheme",
            "kivi",
        ]
    )
    ns.methods = methods
    ns.chunk = chunk
    ns.recent_window = 16
    ns.absorb_block = 8
    return ns


def test_persist_rows_bytes_and_ratios(tmp_path: Path, capsys: object) -> None:
    """One row per arm with a measured on-disk size, its ratio to full KV, and the four
    stage timings; the flagship and the 4-bit baseline persist far fewer bytes than full."""
    model = _model()
    rows = persist.run_persist(
        model, _args(["full", "bugslash", "quant"], chunk=40), ctx=200, device="cpu", tmp=tmp_path
    )
    by = {r["method"]: r for r in rows}
    assert set(by) == {"full", "bugSseed-r16-h16", "quant-4bit-kivi"}
    assert by["full"]["ratio_bytes"] == 1.0 and by["full"]["bytes"] > 0
    # Tiny scale: fixed overheads (sinks, window, basis, fp32 residual) dominate, so only
    # < 1 is scale-free here; the 1/T property is pinned below, the 0.15x at 16K on the pod.
    assert 0.0 < by["bugSseed-r16-h16"]["ratio_bytes"] < 1.0
    assert 0.0 < by["quant-4bit-kivi"]["ratio_bytes"] < 1.0
    for r in rows:
        assert all(r[k] >= 0.0 for k in ("t_save", "t_load", "t_h2d", "t_ready", "t_cold"))
        assert r["t_cold"] == r["t_load"] + r["t_h2d"] + r["t_ready"]
    assert by["full"]["t_ready"] == 0.0  # full KV is attend-ready as loaded
    assert by["bugSseed-r16-h16"]["t_ready"] > 0.0  # reconstruct-then-attend is real work
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    lines = [ln for ln in out.splitlines() if ln.startswith("[persist ctx200]")]
    assert len(lines) == 3 and all("bytes=" in ln and "cold=" in ln for ln in lines)
    # The stored ratio falls with T (gist O(rn+hn) + a rank-r coordinate slope < full's).
    rows400 = persist.run_persist(
        model, _args(["full", "bugslash"], chunk=40), ctx=400, device="cpu", tmp=tmp_path
    )
    r400 = next(r for r in rows400 if r["kind"] == "bug")["ratio_bytes"]
    assert r400 < by["bugSseed-r16-h16"]["ratio_bytes"]


def test_state_tensors_cover_the_honest_state(tmp_path: Path) -> None:
    """The persisted flagship state is exactly the tensors the accounting bills
    (stored_state_numel), with the square-root cores stored as their diagonals."""
    from w10_frontier import _prefill_chunked, build_arms

    model = _model()
    arm = next(a for a in build_arms(_args(["bugslash"], 40), model, 200) if a["kind"] == "bug")
    cache = arm["make"]()
    hay = torch.randint(0, 256, (1, 200))
    with cache.attach(model):
        _prefill_chunked(model, cache, hay, 40)
    state = persist.state_tensors("bug", cache)
    numel = sum(t.numel() for t in state.values())
    assert numel == cache.stored_state_numel()
