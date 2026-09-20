"""Week-15 W-C: per-window NLL emission from the ppl harness (``frontier``).

Every published ppl so far pooled NLL across windows and discarded the
per-window values (no error bars). Week-15 keeps them. Pins:

1. ``test_window_nll_consistency`` -- the pooled ``ppl`` in the emitted row is
   exactly recomputable from the new ``window_nlls``/``window_toks`` fields as
   ``exp(sum(nll_i * tok_i) / sum(tok_i))``, so per-window error bars can never
   disagree with the published pooled number.
2. ``test_pplw_line_format`` -- the new ``[pplw]`` printed line matches its
   documented harvest regex (``^\\[pplw`` grep discipline), and the pooled
   ``  <method> [T=..] ppl=..`` line still matches ``records.PPL_RE``
   byte-compatibly (the ``[pplw]`` line itself never does).
3. ``test_pplw_line_splits_when_long`` -- >400-char lines (vast logs truncate
   at ~500) split into ``part=i/N`` lines of 8 values that reassemble exactly.

Hermetic: tiny random-weight Llama + random-token corpus, so ``frontier.run_ppl``
executes its real eval loop with no downloads.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.baselines.compat import install_kvpress_prefill_compat
from kvdlra.eval import frontier, records
from kvdlra.eval.config import ArmCfg

# records.PPL_RE anchors on ^ but is not compiled MULTILINE (it is applied line by
# line by the harvest); scanning a captured stdout block needs the same pattern with
# re.M, so the byte-compatibility claim is about the pattern, not a second copy.
PPL_RE = re.compile(records.PPL_RE.pattern, re.M)

# The documented [pplw] harvest regex (frontier._log_pplw):
#   [pplw] T=<T> <method> ntok=<per-window scored tokens> [part=<i>/<N>]
#   nlls=<comma-joined per-window mean NLLs, 6 decimals> [corpus=<name>]
# Taken from `records` rather than restated, the way PPL_RE above is: a second copy of
# the pattern pins a format nobody reads with.
PPLW_RE = re.compile(records.PPLW_RE.pattern, re.M)


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=8192,
        )
    )
    model.eval()  # type: ignore[no-untyped-call]
    return model


ARMS = {
    "full": ArmCfg(name="full", kind="full"),
    "snapkv_k0.25": ArmCfg(
        name="snapkv_k0.25", kind="press", press={"family": "snapkv", "keep": 0.25}
    ),
    "bug-r8": ArmCfg(
        name="bug-r8",
        kind="bug",
        cache={
            "rank": 8,
            "coord_budget": None,
            "recent_window": 8,
            "absorb_block": 4,
            "n_sink": 4,
            "retention": "fifo",
        },
    ),
}


def _run(
    *, methods: list[str], t: int = 64, window: int = 16, n_samples: int = 3
) -> list[dict[str, Any]]:
    """Drive the REAL ``frontier.run_ppl`` eval loop hermetically (cpu, no I/O)."""
    model = _tiny_model()
    n_ids = (t + window) * (n_samples + 1)  # enough exact windows for n_samples
    ids = torch.randint(0, 256, (n_ids,))
    arms = [frontier.build_arm(ARMS[m], model, t) for m in methods]
    samples = frontier.windows(ids, t, window, n_samples)
    assert len(samples) == n_samples
    return frontier.run_ppl(arms, model, samples, t, chunk=0, n=32, h_kv=2, device="cpu")


def _ok_rows(rows: list[dict[str, Any]], t: int) -> list[dict[str, Any]]:
    ok = [r for r in rows if r["status"] == "ok" and r["T"] == t]
    assert ok, "hermetic run produced no ok rows"
    return ok


def test_window_nll_consistency() -> None:
    # full exercises prefill_press(None); bug (rank 8) exercises score_streaming.
    rows = _ok_rows(_run(methods=["full", "bug-r8"], n_samples=3), 64)
    assert {r["method"] for r in rows} == {"full", "bug-r8"}
    for row in rows:
        nlls, toks = row["window_nlls"], row["window_toks"]
        # The arm's own wall clock rides the row (L3.3a): `runner._ppl_rows` turns it
        # into this axis's `[stage] cell` line, the only clock a harvest carries.
        assert row["elapsed_s"] > 0.0
        assert len(nlls) == len(toks) == 3
        assert all(tok == 16 - 1 for tok in toks)  # scored tokens = window - 1
        assert all(math.isfinite(v) and v > 0 for v in nlls)
        assert len(set(nlls)) > 1  # genuinely per-window, not one value repeated
        # THE pin: pooled ppl == exp(sum(nll_i * tok_i) / sum(tok_i)).
        pooled = math.exp(sum(v * tok for v, tok in zip(nlls, toks, strict=True)) / sum(toks))
        # run() exponentiates through a float32 tensor; allow only that rounding.
        assert row["ppl"] == pytest.approx(pooled, rel=1e-5)


def test_pplw_line_format(capsys: pytest.CaptureFixture[str]) -> None:
    rows = _ok_rows(_run(methods=["full", "bug-r8"], n_samples=3), 64)
    out = capsys.readouterr().out

    pplw = {m.group(2): m for m in PPLW_RE.finditer(out)}
    pooled = {m.group(1): m for m in PPL_RE.finditer(out)}
    for row in rows:
        # -- new [pplw] line: exactly one per (arm, T), matching the documented regex
        m = pplw[row["method"]]
        assert out.count(f"[pplw] T=64 {row['method']} ") == 1
        assert m.group(1) == "64"
        assert m.group(3) == str(row["window_toks"][0])
        assert m.group(4) is None  # 3 windows: single line, no part index
        assert m.group(6).split(",") == [f"{v:.6f}" for v in row["window_nlls"]]
        # -- pooled line: still byte-compatible with the harvest regex
        p = pooled[row["method"]]
        assert p.group(2) == "64"
        assert p.group(3) == f"{row['ppl']:.3f}"

    # The [pplw] lines themselves must never be picked up by PPL_RE (it anchors
    # on leading whitespace) -- pooling stays uncontaminated.
    for line in out.splitlines():
        if line.startswith("[pplw]"):
            assert PPL_RE.match(line) is None


def test_pplw_line_splits_when_long(capsys: pytest.CaptureFixture[str]) -> None:
    # 48 windows -> single line would be ~460 chars > 400 -> 6 part-lines of 8.
    (row,) = _ok_rows(_run(methods=["full"], t=32, window=8, n_samples=48), 32)
    out = capsys.readouterr().out
    assert len(row["window_nlls"]) == 48

    parts = [m for m in PPLW_RE.finditer(out) if m.group(2) == "full"]
    assert [(m.group(4), m.group(5)) for m in parts] == [(str(i), "6") for i in range(1, 7)]
    assert all(len(m.group(0)) <= 400 for m in parts)
    joined: list[str] = []
    for m in parts:
        vals = m.group(6).split(",")
        assert len(vals) <= 8
        joined += vals
    assert joined == [f"{v:.6f}" for v in row["window_nlls"]]
    # The harvest reads exactly these fragments, through kvdlra.eval.records.PPLW_RE --
    # `scripts/pod.py` dropped them until the L0.5 fix round. Reassembled, they are the
    # row's own per-window NLLs again (nll_sum_nats / ntok undoes the sum).
    assert all(records.PPLW_RE.match(m.group(0)) for m in parts)
    back = records.parse_pplw_lines(out, model="M", source="log")
    assert [r["nll_sum_nats"] / r["ntok"] for r in back] == pytest.approx(
        row["window_nlls"], rel=1e-5
    )
    # Equal-weight recompute from the PRINTED values (uniform windows) matches.
    printed_pooled = math.exp(sum(float(v) for v in joined) / len(joined))
    assert row["ppl"] == pytest.approx(printed_pooled, rel=1e-4)


def test_a_press_arm_is_billed_its_kept_fraction_not_the_scored_window() -> None:
    """The kept fraction, measured between the prefill and the window.

    `score_press` handed back the DynamicCache only after `_score_window` had pushed
    the continuation into it, so `_footprint` measured ``k*T + W`` tokens and every
    press row published ``(k*T + W)/T``: at T=128, W=16 and k=0.25 the arm was billed
    0.375x for a cache holding 0.25x. The prefill is split out now, exactly as the
    faithful-KIVI branch already split its own, so the footprint is the post-prefill
    state on both axes (the retrieval path, `ruler.retrieve`, always took it there).

    `full` is the control: its footprint is analytic (`accounting.full_cache_footprint`
    reads ``t``, never the cache), so it reported 1.0 before and after.
    """
    install_kvpress_prefill_compat()  # transformers 5.8: kvpress needs the prefill shim
    t, keep = 128, 0.25
    rows = {
        r["method"]: r
        for r in _ok_rows(_run(methods=["full", "snapkv_k0.25"], t=t, window=16, n_samples=2), t)
    }
    assert rows["full"]["ratio_fp16"] == 1.0
    # One token of slack for where a press rounds its budget (`evict_footprint`'s
    # ratio_fp16 IS the kept fraction; the count comes off the compressed cache). The
    # window's 16 tokens would be 8x that.
    assert rows["snapkv_k0.25"]["ratio_fp16"] == pytest.approx(keep, abs=1.0 / t)
    assert rows["snapkv_k0.25"]["ratio_stored_bits"] == pytest.approx(keep, abs=1.0 / t)


def test_window_nll_is_accumulated_in_fp32() -> None:
    """Week-19 exit-gate finding: cross_entropy on bf16 logits log-softmaxes and SUMS in
    bf16, so a 511-token window's NLL (~1100 nats) is quantized to the bf16 ulp (8 nats,
    ~1.6%). Every harvested window sum was a multiple of 8. The scorer must accumulate in
    fp32: on bf16 logits it must equal the fp32 computation, not the bf16 one."""
    import torch
    from torch.nn.functional import cross_entropy

    from kvdlra.eval.frontier import _nll_sum

    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(511, 4096, generator=g) * 3).to(torch.bfloat16)
    targets = torch.randint(0, 4096, (511,), generator=g)
    got = _nll_sum(logits, targets)
    fp32 = float(cross_entropy(logits.float(), targets, reduction="sum"))
    bf16 = float(cross_entropy(logits, targets, reduction="sum"))
    assert abs(got - fp32) < 1e-2 * fp32 / 511, (got, fp32)
    assert abs(got - fp32) < abs(got - bf16) or abs(bf16 - fp32) < 1e-6
