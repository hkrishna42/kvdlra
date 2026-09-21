"""L2.4: the matched-budget eviction/structured baseline arms are kvpress presses built by
`baselines.presses.make_press` from an arm YAML's ``press:`` block.

Three things are pinned. The family dispatch: a ``keep`` fraction maps to kvpress's
``compression_ratio = 1 - keep`` for the three scorer families, ThinK+SnapKV composes in
the order ThinK's paper evaluates (evict, then prune channels), the archived arms (no
``family:`` key -- their hashes are pinned by archive pod configs) infer their family from
the name prefix, and a ``family:`` that contradicts the prefix is refused. The billing:
PyramidKV keeps MORE tokens in the lower layers and fewer in the upper, so a footprint
read off layer 0 alone over-bills it -- the mean kept fraction over the layers is what
reproduces the configured budget. And the composed arm is billed its MEASURED kept tokens
with ThinK's channel ratio applied to the K half, end to end through `ruler.retrieve`.

Hermetic tiny Llama (tests/conftest.py); the two cache tests prefill 512 tokens, above
SnapKV's 64-token observation window.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from kvpress import (
    ComposedPress,
    ExpectedAttentionPress,
    PyramidKVPress,
    SnapKVPress,
    ThinKPress,
)
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kvdlra import accounting as acc
from kvdlra.baselines.compat import install_kvpress_prefill_compat
from kvdlra.baselines.presses import make_press
from kvdlra.eval.config import load_arm
from kvdlra.eval.frontier import _footprint, build_arm

H, D = 2, 16  # KV heads x head_dim -> n_features 32 (the shared fixture's shape)
TINY_MPE, TINY_SDPA = 4096, True


_TOK = SimpleNamespace(decode=lambda ids: " ".join(str(i) for i in ids))


def _prompt(t: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


@torch.no_grad()
def _prefill(model: LlamaForCausalLM, press: Any, ids: torch.Tensor) -> DynamicCache:
    """Single-shot prefill under the press -- the path every press arm takes."""
    cache = DynamicCache()
    with press(model):
        model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


# ----------------------------------------------------------------- make_press


def test_keep_fraction_maps_to_compression_ratio() -> None:
    p = make_press(load_arm("snapkv_k0.15"))
    assert isinstance(p, SnapKVPress) and abs(p.compression_ratio - 0.85) < 1e-9


def test_think_snapkv_is_composed_in_paper_order() -> None:
    p = make_press(load_arm("think_c0.5_snapkv_k0.15"))
    assert isinstance(p, ComposedPress)
    assert [type(x) for x in p.presses] == [SnapKVPress, ThinKPress]
    snap, think = p.presses
    assert abs(snap.compression_ratio - 0.85) < 1e-9
    assert think.key_channel_compression_ratio == 0.5


def test_every_eviction_arm_loads() -> None:
    cls = {"snapkv": SnapKVPress, "pyramidkv": PyramidKVPress, "ea": ExpectedAttentionPress}
    for k in ("0.10", "0.15", "0.25"):
        for fam in ("snapkv", "pyramidkv", "ea"):
            p = make_press(load_arm(f"{fam}_k{k}"))
            assert p is not None and type(p) is cls[fam]
            assert abs(p.compression_ratio - (1 - float(k))) < 1e-9


def test_pyramidkv_keeps_kvpress_defaults() -> None:
    """The YAML `doc:` states the defaults are kept; this is what makes that true."""
    p = make_press(load_arm("pyramidkv_k0.15"))
    assert isinstance(p, PyramidKVPress)
    assert (p.window_size, p.kernel_size, p.beta) == (64, 5, 20)


def test_make_press_infers_the_family_of_the_archived_arms() -> None:
    """No `family:` key on the archived arms (their hashes are pinned): the name prefix
    decides, exactly as frontier's old `_evict_factory` did."""
    assert type(make_press(load_arm("snapkv_k0.1"))) is SnapKVPress
    assert type(make_press(load_arm("ea_k0.1"))) is ExpectedAttentionPress
    assert type(make_press(load_arm("think_c0.5"))) is ThinKPress
    assert type(make_press(load_arm("ea_k0.1_kivi2"))) is ExpectedAttentionPress


def test_conflicting_family_and_prefix_is_refused() -> None:
    cfg = load_arm("snapkv_k0.1")
    cfg.press["family"] = "expected_attention"
    with pytest.raises(ValueError, match="family"):
        make_press(cfg)


def test_unknown_family_is_refused() -> None:
    cfg = load_arm("snapkv_k0.10")
    cfg.press["family"] = "h2o"
    with pytest.raises(ValueError, match="h2o"):
        make_press(cfg)


def test_arms_without_a_kvpress_press_make_none() -> None:
    assert make_press(load_arm("full")) is None
    assert make_press(load_arm("svd_oracle_r0.5")) is None  # frontier's `_press` owns it


def test_a_press_arm_with_no_press_parameters_is_refused() -> None:
    """Otherwise `make` would return None and the arm would run as the full cache."""
    cfg = load_arm("snapkv_k0.10")
    cfg.press = {}
    with pytest.raises(ValueError, match="no keep/ratio/rank"):
        build_arm(cfg, model=None, t=1024)


def test_new_arms_are_single_shot_and_carry_billing_fields() -> None:
    for stem in ("snapkv_k0.10", "pyramidkv_k0.15", "ea_k0.15"):
        arm = build_arm(load_arm(stem), model=None, t=16384)
        assert arm["chunkable"] is False and "press_type" not in arm
        assert arm["keep"] == float(stem.rsplit("k", 1)[1])
        # Only the pyramid family carries the key (absent, not False, on every other press:
        # the archived arms' golden dicts must not change).
        assert ("per_layer_budget" in arm) is stem.startswith("pyramidkv")
        assert arm.get("per_layer_budget", False) is stem.startswith("pyramidkv")
    arm = build_arm(load_arm("think_c0.5_snapkv_k0.15"), model=None, t=16384)
    assert arm["chunkable"] is False
    assert (arm["press_type"], arm["think_ratio"], arm["keep"]) == ("think_snapkv", 0.5, 0.15)


# ------------------------------------------------------------------- billing


def test_pyramidkv_is_billed_its_mean_kept_fraction(tiny_model: LlamaForCausalLM) -> None:
    """At ctx=512, keep=0.5 kvpress's pyramid hits its clamp: layer 0 keeps
    ``q_len - window`` = 448 tokens, layer 1 the remaining 64 -- a 7x spread whose mean
    is exactly the configured 256. The footprint must bill the mean, not layer 0."""
    install_kvpress_prefill_compat()
    n, h_kv, t = H * D, H, 512
    cfg = load_arm("pyramidkv_k0.10")
    cfg.press["keep"] = 0.5
    arm = build_arm(cfg, tiny_model, t)
    cache = _prefill(tiny_model, arm["make"](), _prompt(t, 3))
    kept = [int(cast(Any, la).keys.shape[2]) for la in cache.layers]
    assert kept == [448, 64], kept  # the pyramid is active, not the flat fallback
    fp = _footprint(arm, cache, t, n, h_kv)
    assert fp.verbatim_elems == pytest.approx(acc.evict_footprint(t, n, 0.5).verbatim_elems)


def test_think_snapkv_is_billed_measured_keep_times_the_channel_ratio(
    tiny_model: LlamaForCausalLM,
) -> None:
    """End to end through `ruler.retrieve`: the composed cache holds ``int(0.15 * ctx)``
    tokens per layer with half of every head's key channels zeroed, and the bill is
    `think_evict_footprint` at the MEASURED kept fraction -- ThinK's ratio applied to the
    K half only, plus its one-time channel index set."""
    install_kvpress_prefill_compat()
    from kvdlra.eval.ruler import retrieve

    n, h_kv, ctx = H * D, H, 512
    arm: dict[str, Any] = build_arm(load_arm("think_c0.5_snapkv_k0.15"), tiny_model, ctx)
    kept = int(0.15 * ctx)  # SnapKV truncates: int(k_len * (1 - compression_ratio))
    hay, query = _prompt(ctx, 5), _prompt(6, 6)
    for la in _prefill(tiny_model, arm["make"](), hay).layers:
        keys = cast(Any, la).keys
        assert keys.shape[2] == kept  # evicted first ...
        zeroed = (keys.abs().sum(dim=2) == 0).sum(dim=-1)  # (bsz, h_kv): dims all-zero
        assert zeroed.tolist() == [[D // 2] * h_kv]  # ... then half the channels pruned
    want = acc.think_evict_footprint(ctx, n, D, h_kv, 0.5, kept / ctx)
    _hit, ratio, _frac, sbits = retrieve(
        tiny_model, _TOK, arm, hay, query, ["needle"], "cpu", 0, n, h_kv, 4
    )
    assert ratio == pytest.approx(want.ratio_fp16(ctx, n))
    assert sbits == pytest.approx(want.ratio_stored_bits(ctx, n))
    # keep x (1 - ratio/2): the V half is untouched, the K half keeps half its channels
    assert ratio == pytest.approx(kept / ctx * 0.75 + want.aux_words * 2 / (2 * ctx * n))


# ------------------------------------------------------- per-layer budgets (R-L2-5)


def test_pyramidkv_decodes_token_by_token_and_the_uniform_presses_in_one_block(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transformers builds ONE causal mask per forward from layer 0's key count and never
    slices it to a layer's own length, so a q_len>1 forward after a PyramidKV prefill
    (448 vs 64 keys on the two tiny layers) raises. `retrieve` reads the arm's
    ``per_layer_budget`` and decodes the query one token per forward (q_len=1: sdpa
    skips the mask); forcing the key off reproduces the raise (the switch is
    load-bearing), and a uniform press (snapkv) still decodes in one block."""
    install_kvpress_prefill_compat()
    from kvdlra.eval import ruler

    seen: list[bool] = []
    real = ruler._decode

    def spy(*a: Any, **k: Any) -> str:
        seen.append(bool(k["block"]))
        return real(*a, **k)

    monkeypatch.setattr(ruler, "_decode", spy)
    n, h_kv, ctx = H * D, H, 512
    hay, query = _prompt(ctx, 3), _prompt(6, 4)
    cfg = load_arm("pyramidkv_k0.10")
    cfg.press["keep"] = 0.5  # the 448 / 64 pyramid the billing test above pins
    arm = build_arm(cfg, tiny_model, ctx)
    assert arm["per_layer_budget"] is True
    args = (tiny_model, _TOK, arm, hay, query, ["needle"], "cpu", 0, n, h_kv, 4)
    ruler.retrieve(*args)
    with pytest.raises(RuntimeError, match="must match the size"):
        ruler.retrieve(tiny_model, _TOK, {**arm, "per_layer_budget": False}, *args[3:])
    snap = build_arm(load_arm("snapkv_k0.10"), tiny_model, ctx)
    assert "per_layer_budget" not in snap
    ruler.retrieve(tiny_model, _TOK, snap, *args[3:])
    assert seen == [False, True, True]


def test_run_ppl_refuses_a_pyramidkv_arm_as_a_recorded_error(
    tiny_model: LlamaForCausalLM, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ppl axis scores a 512-token window in ONE forward, which the single mask
    forbids for per-layer budgets; the arm is refused before scoring, and the refusal
    lands where any failed ppl arm does -- a ``status: error`` row the runner prints as
    an ``[error] axis=ppl`` line and counts -- never an exception out of the sweep."""
    from kvdlra.eval.frontier import run_ppl, windows
    from kvdlra.eval.runner import _log_ppl_errors

    t, w = 64, 16
    arm = build_arm(load_arm("pyramidkv_k0.15"), tiny_model, t)
    rows = run_ppl(
        [arm],
        tiny_model,
        windows(_prompt(200, 7)[0], t, w, 1),
        t,
        chunk=0,
        n=H * D,
        h_kv=H,
        device="cpu",
    )
    (row,) = rows
    assert row["method"] == "pyramidkv_k0.15" and row["status"] == "error"
    # Burned real seconds before it failed (dict.get + f-string + raise/except, no
    # sleep) -- measured 5-22 us over 20 runs on this exact path, never 0.0.
    # `runner.py` reads `r["elapsed_s"]` unguarded from `run_pod`; a regression that
    # moves the assignment inside the try (so an error row never gets the key) would
    # raise a bare KeyError out of a paid pod's sweep instead of failing here.
    assert row["elapsed_s"] > 0.0
    assert row["error"].startswith(
        "ValueError: pyramidkv_k0.15: PyramidKV's per-layer budgets cannot be scored through"
        " transformers' single causal mask (a 512-token window in one forward)"
    )
    assert "per-token perplexity scoring is not implemented" in row["error"]
    assert _log_ppl_errors(rows) == 1
    assert (
        "[error] axis=ppl arm=pyramidkv_k0.15 ctx=64 error=ValueError:" in capsys.readouterr().out
    )
