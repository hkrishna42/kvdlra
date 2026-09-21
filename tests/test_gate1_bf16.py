"""The bf16 arm's non-inferiority reading, as `make gate1` renders it.

`prereg/bf16_gist.md` section 4 pre-registered the reading as a hand-run snippet over
shipped functions, which is a reading that can be skipped at harvest. Here it is the same
statistic in the module that renders the Gate-1 table: `gate1.bf16_contrasts` (the exact
paired McNemar of `isvd_r64_h256_seed` (a) against `isvd_r64_h256_seed_bf16` (b) on the
`(seed, trial)` keys whose `prompt_sha256` match), `gate1.bf16_verdict` (section 4 (1)'s
0.03 margin AND Holm, section 4 (2)'s TOST, section 4 (4)'s refusals, section 4 (3)'s
composition) and the block `tables.gate1_table` prints before the Gate-1 verdict line.

The fixtures are `tests.test_gate1_table`'s -- two pods, one per model family, written as
records -- carrying the two arms this reading pairs and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tables

from kvdlra.eval import gate1
from tests.test_gate1_table import ARM, N_TRIALS
from tests.test_gate1_table import write_pod as _write_pod

# The pair section 4 reads, at the floor-against-floor shape section 5 predicts, and the
# bf16 arm fractionally WORSE on perplexity (section 5: "nothing in the construction makes
# bf16 better") but well inside the +/-0.02 margin.
PAIR = {"isvd": 20, "bf16": 20}
TIGHT = {"bf16": 0.005}
# Section 2 (a)'s two at-rest bills, whose ratio section 7 (c) pins: 0.566 on a
# 1024-wide layer (Llama), 0.539 on a 512-wide one (Qwen). Used only where a test wants
# the exact printed digits on the table; every other test gets a fixture sbits pair from
# `_bf16_sbits` below, whose ratio clears the pin by construction.
SBITS = {"isvd": 0.150348, "bf16": 0.085056}
# Qwen's own bill (section 2 (a)'s second row) -- ratio 0.539. Before this fix round this
# test file only ever had Llama's numbers, reused for Qwen through `both()`, so the Qwen
# pin was never exercised at its own ratio (Minor 7).
SBITS_QWEN = {"isvd": 0.275062, "bf16": 0.148383}


def _bf16_sbits(family: str) -> dict[str, float]:
    """A stored-bits pair at `family`'s layer width whose ratio is EXACTLY section 7
    (c)'s expected one (`gate1.bf16_expected_sbits_ratio`) -- so a fixture that does not
    care about the pin clears it by construction rather than falling back to
    `write_pod`'s every-tracker 0.15 default, a ratio of 1.0 that the pin refuses."""
    return {"isvd": 1.0, "bf16": gate1.bf16_expected_sbits_ratio(gate1.BF16_LAYER_WIDTH[family])}


def write_pod(root: Path, family: str, **kw: object) -> Path:
    """`test_gate1_table.write_pod`, defaulting `sbits` to `family`'s real stored-bits
    ratio (`_bf16_sbits`) instead of the 0.15/0.15 every-tracker default -- a caller that
    passes its own `sbits=` (to exercise section 7 (c) itself) is left alone."""
    kw.setdefault("sbits", _bf16_sbits(family))
    return _write_pod(root, family, **kw)  # type: ignore[arg-type]


def read(
    root: Path, **pods: dict[str, object]
) -> tuple[gate1.Bf16Verdict, list[gate1.Contrast], gate1.Gate1Data, list[Path]]:
    dirs = [write_pod(root, f, **kw) for f, kw in pods.items()]
    data = gate1.load(dirs)
    contrasts = gate1.bf16_contrasts(data)
    verdict = gate1.bf16_verdict(contrasts, gate1.ppl_contrasts(data), data)
    return verdict, contrasts, data, dirs


def both(
    root: Path, **kw: object
) -> tuple[gate1.Bf16Verdict, list[gate1.Contrast], gate1.Gate1Data, list[Path]]:
    return read(root, llama=dict(kw), qwen=dict(kw))


def rewrite_hits(d: Path, arm: str, task: str, hits: int) -> Path:
    """Rewrite one arm's rows for one task so it hits the first `hits` trials only -- the
    nested-prefix shape `write_pod` builds, so the discordance is one-directional."""
    rows = [json.loads(x) for x in (d / "trials.jsonl").read_text().splitlines()]
    for r in rows:
        if r["task"] == task and r["arm"] == ARM[arm]:
            r["hit"] = int(r["trial"] < hits)
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def table(dirs: list[Path], tmp_path: Path) -> str:
    out = tmp_path / "out" / "gate1.md"
    tables.gate1_table(dirs, out)
    return out.read_text()


# --- the reading passes -------------------------------------------------------------


def test_a_tie_on_every_task_is_non_inferior_in_both_families(tmp_path: Path) -> None:
    """Section 5's prediction, as records: two arms at the same floor produce no
    discordant pair at all, so every member is p = 1.0 -- which section 4 (4) says is
    "never read as evidence of equivalence" and this reading duly reports as a
    non-separation, not as a measured equality."""
    v, members, _, dirs = read(
        tmp_path,
        llama={"hits": PAIR, "delta": TIGHT, "sbits": SBITS},
        qwen={"hits": PAIR, "delta": TIGHT, "sbits": SBITS_QWEN},
    )
    assert v.overall == "pass", v.reason
    assert v.families == {"llama": "PASS", "qwen": "PASS"}
    assert len(members) == gate1.bf16_family_size() == 8
    assert all(c.p == 1.0 and c.p_holm == 1.0 for c in members)
    assert all((c.a_favored, c.b_favored, c.n_paired) == (0, 0, N_TRIALS) for c in members)
    md = table(dirs, tmp_path)
    assert "## bf16 non-inferiority" in md
    assert "+0.000 [0/0 of 24] 1" in md
    assert "- llama: PASS" in md and "- qwen: PASS" in md
    assert "BF16: pass — llama PASS, qwen PASS" in md
    assert "0.5657" in md, "section 7 (c)'s pin, from ppl.jsonl -- Llama at 1024-wide"
    assert "0.5395" in md, "section 7 (c)'s pin, from ppl.jsonl -- Qwen at 512-wide (Minor 7)"
    md_verdict = next(x for x in md.splitlines() if x.startswith("VERDICT: "))
    assert md.index("BF16: ") < md.index(md_verdict), "the reading sits before the branch"


# --- section 4 (1): the margin AND Holm, both, or the task is non-inferior -----------


@pytest.mark.parametrize(
    ("bf16_hits", "family", "overall"),
    # 9 pairs lost with none won is p = 2^-8 = 0.0039, which clears Holm's first slot in
    # an 8-member family (0.05/8 = 0.00625); 8 is p = 2^-7 = 0.0078 and adjusts to 0.0625,
    # so the same 0.33 point estimate -- ten times section 4 (1)'s 0.03 margin -- is NOT a
    # loss. That asymmetry is the rule: "both conditions, together".
    [(11, "FAIL", "fail"), (12, "PASS", "pass")],
)
def test_the_margin_and_holm_must_both_bind_for_a_task_to_fail(
    tmp_path: Path, bf16_hits: int, family: str, overall: str
) -> None:
    dirs = [
        rewrite_hits(
            write_pod(tmp_path, "llama", hits=PAIR, delta=TIGHT), "bf16", "niah_multikey", bf16_hits
        ),
        write_pod(tmp_path, "qwen", hits=PAIR, delta=TIGHT),
    ]
    data = gate1.load(dirs)
    members = gate1.bf16_contrasts(data)
    v = gate1.bf16_verdict(members, gate1.ppl_contrasts(data), data)
    lost = next(c for c in members if (c.family, c.task) == ("llama", "niah_multikey"))
    assert (lost.a_favored, lost.b_favored) == (20 - bf16_hits, 0)
    assert lost.p is not None and lost.p == pytest.approx(2.0 ** (1 - (20 - bf16_hits)))
    assert lost.p_holm == pytest.approx(8 * lost.p), "Holm over the 8 members section 6 fixes"
    assert (lost.a_favored - lost.b_favored) / lost.n_paired > gate1.BF16_MARGIN
    assert v.families == {"llama": family, "qwen": "PASS"}
    assert v.overall == overall, v.reason
    md = table(dirs, tmp_path)
    assert f"BF16: {overall} — llama {family}, qwen PASS" in md
    if family == "FAIL":
        assert any("niah_multikey" in m and "FAIL" in m for m in v.members), v.members
        assert "+0.375 [9/0 of 24] 0.0312" in md


# --- section 4 (4): the refusals, read before the rule ------------------------------


def test_an_error_row_on_the_bf16_arm_refuses_that_family(tmp_path: Path) -> None:
    """Section 4 (4): "an `error` row on the bf16 arm or on `isvd_r64_h256_seed` in a cell
    refuses the reading for that family ... the member is listed with its exception text
    and no adjusted p-value".

    The OTHER family's members keep their adjusted p, and so do the refused family's clean
    members: section 4's snippet drops a member from Holm for its own error row or its own
    broken pairing and for nothing else, so the realised m is 7, not 4. (Gate 1's own
    ruling R-L3-16 removes a refused family's whole set; this reading is not Gate 1's and
    the snippet is what was registered.)"""
    v, members, _, dirs = read(
        tmp_path,
        llama={"hits": PAIR, "delta": TIGHT, "errors": {"bf16": 3}},
        qwen={"hits": PAIR, "delta": TIGHT},
    )
    assert v.families == {"llama": "REFUSED (error)", "qwen": "PASS"}
    assert v.overall == "REFUSED", v.reason
    assert any("isvd_r64_h256_seed_bf16 has 3 error records" in m for m in v.members), v.members
    assert sum(1 for c in members if c.p_holm is not None) == 7
    errored = next(c for c in members if (c.family, c.task) == ("llama", "niah_single"))
    assert (errored.errors_b, errored.p_holm) == (3, None)
    assert all(
        c.p_holm is not None for c in members if c.family == "llama" and c.task != "niah_single"
    )
    md = table(dirs, tmp_path)
    assert "BF16: REFUSED — llama REFUSED (error), qwen PASS" in md
    assert "FAILED (3 errors)" in md
    assert "no cost figure is recorded" in md, "a refusal's consequence is not a fail's"


def test_a_disagreeing_prompt_digest_shrinks_the_member_and_is_reported(
    tmp_path: Path,
) -> None:
    """Section 4 (4): "a `prompt_sha256` mismatch does not remove a member; it shrinks
    that member's paired n, the drop is reported with the key and the arms"."""
    v, members, _, dirs = read(
        tmp_path,
        llama={"hits": PAIR, "delta": TIGHT, "bad_sha": {"bf16": {7}}},
        qwen={"hits": PAIR, "delta": TIGHT},
    )
    dropped = next(c for c in members if (c.family, c.task) == ("llama", "niah_single"))
    assert (dropped.n_paired, dropped.dropped_keys) == (23, ((0, 7),))
    assert v.overall == "pass", v.reason
    md = table(dirs, tmp_path)
    assert "pairing: 1 key dropped (llama/niah_single: (0,7)) -- isvd vs bf16, n_paired 23" in md
    assert "+0.000 [0/0 of 23] 1" in md


def test_an_arm_that_wrote_no_digest_refuses_the_family(tmp_path: Path) -> None:
    """Every key dropped leaves `mcnemar_exact` with nothing to pair, which section 4 (4)
    calls "a broken pairing, not a result"."""
    v, _, _, _ = read(
        tmp_path,
        llama={"hits": PAIR, "delta": TIGHT, "bad_sha": {"bf16": None}},
        qwen={"hits": PAIR, "delta": TIGHT},
    )
    assert v.families["llama"] == "REFUSED (no pairing)"
    assert v.overall == "REFUSED", v.reason
    assert any("all 24 keys dropped on prompt_sha256" in m for m in v.members), v.members


def test_a_pod_that_stopped_inside_the_bf16_arm_is_not_run_never_a_partial_pass(
    tmp_path: Path,
) -> None:
    """Section 4 (4): "a pod that stopped inside arm 6 leaves `not run`, never a partial
    reading". Three of the four cells tie, which without the check would read as a pass
    off three quarters of the evidence."""
    d = write_pod(tmp_path, "llama", hits=PAIR, delta=TIGHT)
    rows = (d / "trials.jsonl").read_text().splitlines(keepends=True)
    (d / "trials.jsonl").write_text(
        "".join(x for x in rows if not (f'"arm": "{ARM["bf16"]}"' in x and '"task": "vt"' in x))
    )
    data = gate1.load([d, write_pod(tmp_path, "qwen", hits=PAIR, delta=TIGHT)])
    v = gate1.bf16_verdict(gate1.bf16_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.families["llama"] == "REFUSED (not run)"
    assert v.overall == "REFUSED", v.reason
    assert any("vt" in m and "not run" in m for m in v.members), v.members


def test_no_bf16_arm_in_the_records_is_not_run_and_renders_no_block(tmp_path: Path) -> None:
    """The Gate-1 pods before arm 6 has written a row, and every Gate-1 fixture that
    carries the other arms only: one line saying the reading did not run, no block, and
    nothing about the Gate-1 table changes."""
    v, members, _, dirs = both(tmp_path, hits={"isvd": 20, "frozen": 20}, delta={"frozen": 0.005})
    assert (v.overall, v.families, members) == ("not run", {}, [])
    md = table(dirs, tmp_path)
    assert "## bf16 non-inferiority" not in md
    assert "BF16: not run — no bf16 records at ctx 16384" in md
    assert "\n\n\n" not in md


# --- section 4 (3): how the two families compose ------------------------------------


def test_a_refused_family_composes_to_refused_whatever_the_other_shows(
    tmp_path: Path,
) -> None:
    """Section 4 (3): "**`REFUSED`** otherwise, i.e. whenever any family is refused,
    whatever the other family shows" -- so a fail in one family and a refusal in the other
    is REFUSED, not fail. The fail is still named: it is the evidence the DECISIONS entry
    reports beside the refusal."""
    dirs = [
        rewrite_hits(
            write_pod(tmp_path, "llama", hits=PAIR, delta=TIGHT), "bf16", "niah_multikey", 11
        ),
        write_pod(tmp_path, "qwen", hits=PAIR, delta=TIGHT, errors={"bf16": 2}),
    ]
    data = gate1.load(dirs)
    v = gate1.bf16_verdict(gate1.bf16_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.families == {"llama": "FAIL", "qwen": "REFUSED (error)"}
    assert v.overall == "REFUSED", v.reason
    assert any("FAIL" in m and "niah_multikey" in m for m in v.members)


def test_one_family_alone_never_buys_a_pass(tmp_path: Path) -> None:
    """Section 4 (3): "There is no partial pass and no per-family pass ... it is bought on
    both families or not at all"."""
    v, _, _, _ = read(tmp_path, llama={"hits": PAIR, "delta": TIGHT})
    assert v.families == {"llama": "PASS"}
    assert v.overall == "REFUSED", v.reason
    assert any(f"1 of {gate1.BF16_FAMILIES} Stage-1 families" in m for m in v.members), v.members


def test_a_missing_family_composes_to_refused_even_if_the_present_one_fails(
    tmp_path: Path,
) -> None:
    """Section 4 (3)'s "whenever any family is refused, whatever the other family
    shows" reaches a missing family too -- a llama-only pod whose one family FAILS must
    not surface as `fail`, "the measured cost of halving the gist's at-rest bytes",
    when qwen's reading never ran at all. Before this fix `bf16_verdict`'s `"fail" if
    "FAIL" in families.values()` branch was tested BEFORE `partial`, so this exact shape
    -- partial + FAIL -- read `fail` instead of `REFUSED` ("no cost figure is
    recorded"); `test_one_family_alone_never_buys_a_pass` above only ever covered
    partial + PASS."""
    d = rewrite_hits(
        write_pod(tmp_path, "llama", hits=PAIR, delta=TIGHT), "bf16", "niah_multikey", 11
    )
    data = gate1.load([d])
    v = gate1.bf16_verdict(gate1.bf16_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.families == {"llama": "FAIL"}
    assert v.overall == "REFUSED", v.reason
    assert "no cost figure is recorded" in v.reason
    assert any(f"1 of {gate1.BF16_FAMILIES} Stage-1 families" in m for m in v.members), v.members


# --- section 4 (2): the perplexity member -------------------------------------------


def test_a_perplexity_gap_past_the_margin_fails_the_family(tmp_path: Path) -> None:
    """Section 4 (2): non-inferior on perplexity in a family iff the +/-0.02 TOST passes.
    At 0.05 bits/token the bf16 arm is outside the margin with every retrieval cell
    tied -- section 5's "interesting result", and the axis the report has to name."""
    v, _, _, dirs = read(
        tmp_path,
        llama={"hits": PAIR, "delta": {"bf16": 0.05}},
        qwen={"hits": PAIR, "delta": TIGHT},
    )
    assert v.families == {"llama": "FAIL", "qwen": "PASS"}
    assert v.overall == "fail", v.reason
    assert any("TOST fails" in m for m in v.members), v.members
    md = table(dirs, tmp_path)
    assert "fails (-0.05" in md, "the TOST column of the bf16 block, with its point estimate"
    assert "the measured cost of halving the gist" in md


def test_a_spread_too_wide_for_the_margin_refuses_rather_than_fails(tmp_path: Path) -> None:
    """Section 6: a member whose realised `s` cannot fit inside +/-0.02 "is recorded as
    `not decidable` -- never as a pass, never as a quiet fail", and section 4 (3) makes
    that family `REFUSED`, whose consequence is narrower than a fail's."""
    v, _, _, _ = read(
        tmp_path,
        llama={"hits": PAIR, "delta": TIGHT, "spread": 0.02},
        qwen={"hits": PAIR, "delta": TIGHT},
    )
    assert v.families == {"llama": "REFUSED (not decidable)", "qwen": "PASS"}
    assert v.overall == "REFUSED", v.reason


# --- section 6: the realised m, and the pre-flight's task exclusion -----------------


def test_an_excluded_task_leaves_the_family_and_decides_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 6: an excluded task "leaves this family too ... 8 - 2 per excluded task",
    so one exclusion gives 6. The bf16 arm loses 9-0 on `vt` in both families here, which
    is two failing members with the knob empty and none with `vt` excluded -- and the
    cells are still rendered, descriptively, in the Gate-1 retrieval block above."""
    monkeypatch.setattr(gate1, "EXCLUDED_TASKS", frozenset({"vt"}))
    dirs = [
        rewrite_hits(write_pod(tmp_path, f, hits=PAIR, delta=TIGHT), "bf16", "vt", 11)
        for f in ("llama", "qwen")
    ]
    data = gate1.load(dirs)
    members = gate1.bf16_contrasts(data)
    v = gate1.bf16_verdict(members, gate1.ppl_contrasts(data), data)
    assert gate1.bf16_family_size() == 6
    assert len(members) == 6 and "vt" not in {c.task for c in members}
    assert v.overall == "pass", v.reason
    md = table(dirs, tmp_path)
    assert "m=6 of the 6 members" in md
    header = next(x for x in md.splitlines() if x.startswith("| family |"))
    assert "niah_single" in header and "vt" not in header
    assert "(11/24)" in md, "the excluded task's bf16 cell still renders in the Gate-1 block"


# --- section 7 (c): the stored-bits ratio is a refusal, not a descriptive column ----


def test_a_stored_bits_ratio_inside_one_percent_does_not_refuse(tmp_path: Path) -> None:
    """Section 7 (c): "outside it ... the reading is refused" -- inside it, nothing
    happens. Every fixture in this file clears the pin by construction (`_bf16_sbits`),
    so this pins that no "sbits" refusal is hiding behind a `pass` for the wrong reason."""
    v, _, _, _ = both(tmp_path, hits=PAIR, delta=TIGHT)
    assert v.overall == "pass", v.reason
    assert not any("sbits" in m or "stored-bits" in m for m in v.members), v.members


def test_a_stored_bits_ratio_outside_one_percent_refuses_naming_the_numbers(
    tmp_path: Path,
) -> None:
    """Section 7 (c): the bf16 arm stored like fp32 -- its `sbits` barely below the fp32
    arm's rather than near half of it, a ratio of ~0.9944 against an expected ~0.566 --
    is "outside it", and "the reading is refused pending a dated amendment", read
    BEFORE section 4's rule (so an otherwise-tied retrieval table still refuses)."""
    isvd_sbits = 0.150348
    bf16_sbits = isvd_sbits * 0.9944  # "bf16 stored like fp32": near 1x, not near 0.566x
    v, _, _, _ = read(
        tmp_path,
        llama={"hits": PAIR, "delta": TIGHT, "sbits": {"isvd": isvd_sbits, "bf16": bf16_sbits}},
        qwen={"hits": PAIR, "delta": TIGHT},
    )
    ratio = bf16_sbits / isvd_sbits
    expected = gate1.bf16_expected_sbits_ratio(gate1.BF16_LAYER_WIDTH["llama"])
    assert v.families["llama"] == "REFUSED (sbits)"
    assert v.overall == "REFUSED", v.reason
    assert any(
        f"stored-bits ratio {ratio:.4f}" in m and f"outside 1 % of {expected:.4f}" in m
        for m in v.members
    ), v.members


def test_missing_stored_bits_on_either_arm_refuses(tmp_path: Path) -> None:
    """Section 7 (c) is silent on a pod that never wrote `sbits` at all; `_bf16_refusals`
    reads that the way Gate 1's own byte-match refusal does (that function's docstring):
    a check with nothing to read refuses rather than passing by default."""
    d = write_pod(tmp_path, "llama", hits=PAIR, delta=TIGHT)
    rows = [json.loads(x) for x in (d / "ppl.jsonl").read_text().splitlines()]
    for r in rows:
        if r["arm"] == ARM["bf16"]:
            r["sbits"] = None
    (d / "ppl.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = gate1.load([d, write_pod(tmp_path, "qwen", hits=PAIR, delta=TIGHT)])
    v = gate1.bf16_verdict(gate1.bf16_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.families["llama"] == "REFUSED (sbits)"
    assert v.overall == "REFUSED", v.reason
    assert any("stored bits not measured" in m for m in v.members), v.members


# --- section 6: one stage per call, enforced rather than only documented ------------


def test_a_third_familys_bf16_records_in_one_call_raise_rather_than_pool(
    tmp_path: Path,
) -> None:
    """Section 6: "No Stage-2 member is pooled with a Stage-1 member and no Stage-1
    p-value is recomputed when Stage 2 lands." `bf16_contrasts` groups Holm by context
    length alone (one family per Stage, prereg section 3), so a Mistral pod loaded
    beside the Stage-1 pair in the SAME call would silently repool the registered
    8-member family into 12 rather than correcting Mistral's 4 on their own (section 6:
    "Holm inside its own retrieval family"). Raising is chosen over guessing a stage from
    family membership -- the smaller of the two fixes the review offered, and the
    louder: a silent repool is exactly the failure section 6 forbids, and a wrong guess
    would be undetectable from outside this function."""
    dirs = [
        write_pod(tmp_path, "llama", hits=PAIR, delta=TIGHT),
        write_pod(tmp_path, "qwen", hits=PAIR, delta=TIGHT),
        write_pod(tmp_path, "mistral", hits=PAIR, delta=TIGHT),
    ]
    data = gate1.load(dirs)
    with pytest.raises(ValueError, match="mistral"):
        gate1.bf16_contrasts(data)


# --- rendering: a reading at another context length is descriptive, not nothing ----


def test_a_32k_only_bf16_reading_renders_descriptively_not_nothing(tmp_path: Path) -> None:
    """Sections 3 and 6: 32K is reachable only by a future amendment and is descriptive
    there, never a verdict -- so a pod carrying ONLY 32K bf16 records (no 16K reading
    ran at all, `bf16_verdict` reads `not run`) must still render those rows, the way
    the per-row loop already labels any non-16384 ctx as descriptive; before this fix
    the block's early return keyed off the verdict rather than off whether there was
    anything to render, so this exact shape rendered nothing."""
    dirs = [write_pod(tmp_path, f, hits=PAIR, delta=TIGHT, ctx=32768) for f in ("llama", "qwen")]
    data = gate1.load(dirs)
    v = gate1.bf16_verdict(gate1.bf16_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.overall == "not run", v.reason
    md = table(dirs, tmp_path)
    assert "## bf16 non-inferiority" in md
    assert "descriptive (§4 reads ctx 16384)" in md
    assert "BF16: not run" in md
