"""Week-17 CPU gate for the default-off relative singular-value floor
(``min_sv_frac``) in :func:`augmented_bug_step` -- the fix that caps the tracked
gist rank at the stream's effective rank instead of padding to ``rank_cap`` with
near-null tail directions (the high-rank divergence substrate behind Mistral
``bug-r128`` / Qwen ``bug-r256`` and the Qwen ``h1024`` puzzle; see docs/week17).

The catastrophic real-transformer-KV ppl blow-up is real-KV-spectrum specific and
GPU-gated -- it is NOT reproduced here. These tests pin the fix's *contract* at $0:
  (1) default (floor off) pads a low-eff-rank stream to ``rank_cap`` (today's behavior);
  (2) with the floor on, the tracked rank collapses to ~the effective rank AND the
      low-rank signal's reconstruction is preserved (the dropped tail was null);
  (3) floor off is bit-identical to the pre-fix call (archived curves stay reproducible);
  (4) the flag is threaded end-to-end through :class:`BugStreamingCache` and wired into
      the compute path (an aggressive floor changes the decoded logits).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.integrators.streaming_torch import augmented_bug_step


def _low_rank_stream(
    n: int = 256, t: int = 1200, d_eff: int = 32, noise: float = 1e-3, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A feature-by-token stream of effective rank ``d_eff`` (< rank_cap) plus a tiny
    noise floor -- the padded near-null tail is exactly what the floor removes."""
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, d_eff, generator=g))[0]
    sv = torch.linspace(3.0, 0.6, d_eff)
    sig = q @ (torch.randn(d_eff, t, generator=g) * sv.unsqueeze(1))
    return sig + torch.randn(n, t, generator=g) * noise, q, sv


def _track(m: torch.Tensor, rank_cap: int, block: int = 16, **kw: Any) -> tuple[torch.Tensor, ...]:
    u: torch.Tensor | None = None
    b: torch.Tensor | None = None
    for s in range(0, m.shape[1], block):
        u, b, _ = augmented_bug_step(u, b, m[:, s : s + block].float(), rank_cap, **kw)
    assert u is not None and b is not None
    return u, b


def test_default_pads_low_rank_stream_to_rank_cap() -> None:
    """Floor off (default): a rank-32 stream is padded to rank_cap=128 (the null tail)."""
    m, _, _ = _low_rank_stream()
    u, _ = _track(m, rank_cap=128)  # theta None, min_sv_frac 0.0
    assert u.shape[1] == 128


def test_rank_floor_caps_at_effective_rank_and_preserves_recon() -> None:
    """Floor on: the tracked rank collapses to ~eff-rank with the low-rank signal
    still reconstructed (the dropped tail carried ~no energy)."""
    m, q, sv = _low_rank_stream()
    u_pad, _ = _track(m, rank_cap=128)
    u_flr, _ = _track(m, rank_cap=128, min_sv_frac=1e-2)
    assert u_pad.shape[1] == 128
    assert 28 <= u_flr.shape[1] <= 48  # ~ effective rank 32, not 128
    g = torch.Generator().manual_seed(7)
    probe = q @ (torch.randn(q.shape[1], 200, generator=g) * sv.unsqueeze(1))

    def rerr(u: torch.Tensor) -> float:
        return float(torch.linalg.norm(u @ (u.T @ probe) - probe) / torch.linalg.norm(probe))

    assert rerr(u_flr) <= rerr(u_pad) + 5e-3  # recon not degraded by the floor


def test_floor_off_is_bit_identical_regression_guard() -> None:
    """Floor off (default) is byte-identical to the pre-fix call so archived curves
    stay bit-reproducible (pre-fix == passing no extra kwarg == min_sv_frac=0.0)."""
    m, _, _ = _low_rank_stream(seed=3)
    u0, b0 = _track(m, rank_cap=64)
    u1, b1 = _track(m, rank_cap=64, min_sv_frac=0.0)
    assert torch.equal(u0, u1) and torch.equal(b0, b1)


def _tiny_qwen() -> Qwen2ForCausalLM:
    # 7 query : 1 KV head, head_dim 16 => n_features = 16 (the family fixture shape).
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=112,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=7,
        num_key_value_heads=1,
        max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def test_min_sv_frac_threaded_through_cache_and_wired_into_compute() -> None:
    """The flag reaches every layer (wrapper -> BugStreamingLayer) AND an aggressive
    floor changes the decoded logits -- proving it is wired through ``_absorb_columns``
    into the gist the read path attends to (not merely stored)."""
    model = _tiny_qwen()
    model.config._attn_implementation = "sdpa"
    n = 16  # head_dim(16) * num_key_value_heads(1)

    cache = BugStreamingCache(
        model, rank=n, coord_budget=4096, recent_window=8, absorb_block=4, min_sv_frac=0.5
    )
    assert all(getattr(layer, "min_sv_frac", None) == 0.5 for layer in cache.layers)

    g = torch.Generator().manual_seed(7)
    stream = torch.randint(0, int(model.config.vocab_size), (1, 90), generator=g)

    def _run(msf: float) -> torch.Tensor:
        c = BugStreamingCache(
            model, rank=n, coord_budget=4096, recent_window=8, absorb_block=4, min_sv_frac=msf
        )
        with torch.no_grad():
            out = model(stream[:, :33], past_key_values=c, use_cache=True)
            for t in range(33, 90):
                out = model(stream[:, t : t + 1], past_key_values=c, use_cache=True)
        return torch.as_tensor(out.logits)

    off, on = _run(0.0), _run(0.5)
    assert not torch.allclose(off, on, atol=1e-3)  # the floor changed the tracked gist
