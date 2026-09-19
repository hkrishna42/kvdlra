"""Effective-rank billing: ``_footprint`` bills the rank the basis ACTUALLY holds.

``CODE_AUDIT`` finding 0.2: the arm dict's ``rank`` is the configured *cap*, but the
Week-17 relative singular-value floor (``min_sv_frac``) drops near-null tail directions,
so a floor-on layer tracks fewer columns than the cap -- and its K and V streams collapse
to different widths. Billing the cap over-counts state that is not stored
(``2*n*rank`` basis + ``2*rank*coord_count`` coordinates + ``2*rank`` core) on every
axis that calls ``_footprint``: perplexity, retrieval and longbench alike.

``tests/test_w17_rankfloor.py`` pins the floor itself; this file pins what the BILLING
does with the collapse, and ``tests/test_accounting.py`` pins the formula against the
measured ``stored_state_numel`` at the collapsed rank.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import LlamaForCausalLM

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache, BugStreamingLayer
from kvdlra.eval import frontier, records
from kvdlra.eval.config import ArmCfg
from kvdlra.eval.frontier import _footprint, _tracked_rank
from tests.conftest import N_FEATURES
from tests.test_accounting import D_K, D_V, _drive, _low_rank_kv_model

H_KV = 2
N_SINK = 4
# The collapse is driven by a stream whose rank is STRUCTURAL (`_low_rank_kv_model`:
# rank D_K pre-RoPE keys, rank D_V values, whatever the tokens), so the floor caps the
# tracked bases at exactly D_K and D_V -- a property of the input, not of where the tiny
# model's own singular-value tail lands at some floor. The floor's own contract is
# pinned on the same synthetic shape in tests/test_w17_rankfloor.py.
MIN_SV_FRAC = 1e-2
CACHE_KW: dict[str, Any] = {
    "rank": 32,
    "coord_budget": 24,
    "recent_window": 8,
    "absorb_block": 4,
    "n_sink": N_SINK,
}
# The arm dict as `frontier.build_arm` hands it to `_footprint` (rank = the cap).
ARM: dict[str, Any] = {"kind": "bug", "rank": 32, "retention": "fifo", "hh_select": "attn"}


def _driven(model: LlamaForCausalLM, **kw: Any) -> BugStreamingCache:
    cache = BugStreamingCache(model, **{**CACHE_KW, **kw})
    _drive(model, cache)
    return cache


def _bill(layer: BugStreamingLayer, rank: float, **kw: Any) -> acc.Footprint:
    """The formula path at an explicit rank -- what `_footprint` chooses between."""
    return acc.bug_footprint(
        N_FEATURES,
        rank=rank,
        coord_count=layer._f_len(),
        recent_len=layer._recent_len(),
        n_sink=N_SINK,
        hh_count=layer._hh_len(),
        **kw,
    )


def _mean_stored(cache: BugStreamingCache) -> float:
    """The measured per-layer state averaged over the layers -- what ONE `_footprint`
    stands for, since every axis multiplies it by the layer count."""
    layers = cache._bug_layers()
    return sum(la.stored_state_numel() for la in layers) / len(layers)


def _mean_rank(cache: BugStreamingCache) -> float:
    """The live rank averaged over both streams of every layer."""
    layers = cache._bug_layers()
    return sum(_tracked_rank(la.u_k) + _tracked_rank(la.u_v) for la in layers) / (2 * len(layers))


def test_collapsed_layer_bills_the_tracked_rank(tiny_model: LlamaForCausalLM) -> None:
    """Floor on: the arm is billed for the columns the bases hold, not the cap."""
    cache = _driven(_low_rank_kv_model(tiny_model), min_sv_frac=MIN_SV_FRAC)
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None and layer.u_v is not None
    rank_k, rank_v = int(layer.u_k.shape[1]), int(layer.u_v.shape[1])
    # Each side caps at its own stream's rank, both below the cap: the case being billed.
    assert (rank_k, rank_v) == (D_K, D_V) and max(rank_k, rank_v) < int(ARM["rank"])
    # The K and V streams collapse INDEPENDENTLY, and `bug_footprint`'s rank terms are
    # symmetric in them, so the exact bill is their mean. Billing `u_k.shape[1]` alone
    # would UNDER-count the wider V basis by (n + coord + 1) floats per column of
    # difference -- which is why the pin below is the arbiter.
    tracked = (rank_k + rank_v) / 2

    fp = _footprint(ARM, cache, t=200, n=N_FEATURES, h_kv=H_KV)
    # One `_footprint` is the per-layer bill every axis multiplies by the layer count,
    # so it is the MEAN over the layers it must reproduce. Here the structural stream
    # collapses every layer to the same two widths, so that mean is also layer 0's.
    assert fp.float_equiv() == _mean_stored(cache) == layer.stored_state_numel()
    assert fp == _bill(layer, tracked, u_present=True) and tracked == _mean_rank(cache)
    # ...and the configured cap over-bills it: the defect this closes.
    assert fp.float_equiv() < _bill(layer, int(ARM["rank"]), u_present=True).float_equiv()


def test_layers_that_collapsed_to_different_widths_bill_their_mean(
    tiny_model: LlamaForCausalLM,
) -> None:
    """Review Important #1: the floor collapses each LAYER's streams independently too,
    so layer 0's own rank is not the model's. Driven on the plain tiny model, whose
    layers have their own singular tails (measured 21/23 and 20/23 at this floor), the
    layer-0 bill over-counts the model by half their difference on every axis."""
    cache = _driven(tiny_model, min_sv_frac=0.3)
    layers = cache._bug_layers()
    widths = [(_tracked_rank(la.u_k), _tracked_rank(la.u_v)) for la in layers]
    assert len(set(widths)) > 1, f"the layers must differ for this to prove anything: {widths}"

    fp = _footprint(ARM, cache, t=200, n=N_FEATURES, h_kv=H_KV)
    assert fp.float_equiv() == _mean_stored(cache)
    assert fp == _bill(layers[0], _mean_rank(cache), u_present=True)
    assert fp.float_equiv() != layers[0].stored_state_numel()  # what layer 0 alone billed


def test_floor_off_billing_is_unchanged(tiny_model: LlamaForCausalLM) -> None:
    """Floor off (every archived arm): the tracked rank IS the cap, so the bill is
    byte-identical to what the configured rank produced -- no archived ratio moves."""
    cache = _driven(tiny_model)
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None and layer.u_v is not None
    assert layer.u_k.shape[1] == layer.u_v.shape[1] == int(ARM["rank"])  # padded to the cap
    fp = _footprint(ARM, cache, t=200, n=N_FEATURES, h_kv=H_KV)
    assert fp == _bill(layer, int(ARM["rank"]), u_present=True)
    assert fp.float_equiv() == _mean_stored(cache) == layer.stored_state_numel()


def test_layer_with_no_basis_bills_rank_zero_and_no_basis(tiny_model: LlamaForCausalLM) -> None:
    """A layer that has not absorbed yet holds no basis at all. ``rank=0`` and
    ``u_present=False`` travel together: `bug_footprint` adds the ``2*n*rank`` basis and
    the ``2*rank`` core only under ``u_present``, and the ``2*rank*coord_count``
    coordinate term only vanishes at rank 0."""
    cache = BugStreamingCache(tiny_model, **CACHE_KW)
    _drive(tiny_model, cache, t=12, n_new=0)  # sinks + recent ring only, nothing absorbed
    layer = cache._bug_layers()[0]
    assert layer.u_k is None and layer.u_v is None and layer._f_len() == 0

    fp = _footprint(ARM, cache, t=12, n=N_FEATURES, h_kv=H_KV)
    assert fp == _bill(layer, 0, u_present=False)
    assert fp.float_equiv() == _mean_stored(cache) == layer.stored_state_numel()


def test_tracked_rank_reads_the_basis_and_is_zero_without_one(
    tiny_model: LlamaForCausalLM,
) -> None:
    model = _low_rank_kv_model(tiny_model)
    layers = _driven(model, min_sv_frac=MIN_SV_FRAC)._bug_layers()
    widths = [_tracked_rank(la.u_k) for la in layers]
    # Same length as `layers`, so this also says no layer reported a missing basis as 0.
    assert widths == [int(la.u_k.shape[1]) for la in layers if la.u_k is not None]
    assert widths == [D_K] * len(layers)  # every layer collapses to its own stream rank
    assert _tracked_rank(None) == 0


def test_eff_rank_on_the_ppl_row_appends_without_breaking_the_harvest(
    tiny_model: LlamaForCausalLM, capsys: Any
) -> None:
    """``eff_rank=`` is appended after ``sbits=``, and ``corpus=`` after that -- the true
    end of the pooled ppl line. ``records.PPL_RE`` is a prefix match with the trailing
    fields optional, so the harvest reads the line exactly as before."""
    t, window = 64, 16
    model = _low_rank_kv_model(tiny_model)
    cfg = ArmCfg(
        name="bug-r32",
        kind="bug",
        cache={**CACHE_KW, "coord_budget": None, "min_sv_frac": MIN_SV_FRAC},
    )
    arm = frontier.build_arm(cfg, model, t)
    ids = torch.randint(0, 256, ((t + window) * 3,), generator=torch.Generator().manual_seed(0))
    samples = frontier.windows(ids, t, window, 2)
    (row,) = frontier.run_ppl(
        [arm], model, samples, t, chunk=0, n=N_FEATURES, h_kv=H_KV, device="cpu"
    )
    assert row["status"] == "ok"
    eff = row["eff_rank"]  # the narrowest K basis: this stream's own rank, not the cap
    assert eff == D_K < int(cfg.cache["rank"])

    out = capsys.readouterr().out
    (line,) = [ln for ln in out.splitlines() if " ppl=" in ln]
    assert line.endswith(f" eff_rank={eff} corpus={row['corpus']}")
    m = records.PPL_RE.match(line)
    assert m is not None and m.group(1) == "bug-r32" and m.group(3) == f"{row['ppl']:.3f}"
    (parsed,) = records.parse_ppl_lines(out, model="M", source="log")
    assert parsed["ppl"] == float(f"{row['ppl']:.3f}")
    assert parsed["ratio"] == float(f"{row['ratio_fp16']:.3f}")
