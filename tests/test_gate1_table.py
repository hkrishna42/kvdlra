"""Gate 1's branch rule, on `results/`-shaped fixtures (`prereg/gate1_tracker_swap_v2.md`).

Every fixture is two pods -- one per model family, as `PodCfg` forces -- written into
`tmp_path` with the four files `gate1.load` reads: `manifest.json` (for the model), the
per-trial `trials.jsonl`, the per-window `pplw.jsonl` and the aggregate `ppl.jsonl`
(the only committed carrier of an arm's `sbits`, which §4's byte-match refusal reads).
No model is loaded and no pod runs: the rule is arithmetic over records.

The four outcomes the lane plan names are (a) `test_identical_trackers_select_branch_c`,
(b) `test_beating_both_controls_in_two_families_selects_ab`,
(c) `test_one_family_separated_is_undecided` and
(d) `test_an_fd_error_cell_refuses_and_prints_failed`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tables

from kvdlra.eval import gate1

MODELS = {
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
# The record key each arm writes: `legacy_name` where the arm config sets one, the stem
# otherwise -- and the no-gist twin is per KV width (prereg section 3).
ARM = {
    "full": "full",
    "isvd": "bugSseed-r64-h256",
    "frozen": "frozen_r64_h256_seed",
    "fd": "bugSseed-r64-h256-fd",
    "oja": "oja_r64_h256_seed_tuned",
    "random": "random_r64_h256_seed",
    "bf16": "isvd_r64_h256_seed_bf16",
}
# The per-KV-width no-gist twin. Mistral is Stage 2's family and has no twin of its own
# yet (its pod is created by amendment, prereg section 9); the Llama stem stands in where
# a fixture needs a third family, and only its tracker label (`nogist`) is read.
NOGIST = {"llama": "nogist_h2423", "qwen": "nogist_h4460", "mistral": "nogist_h2423"}
TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
CTX, N_TRIALS, WINDOWS, NTOK = 16384, 24, 32, 2048
# `sbits` on a record is `ratio_stored_bits` -- stored bits relative to `full`, which
# bills 1.0 (runner._ppl_record) -- so a no-gist twin printing the r64 arm's value is the
# 1.00003x byte match its arm file solves for.
SBITS = 0.15
# A wiggle multiplier per tracker, coprime with 11: without one, two arms' per-window
# curves would differ by a constant and the paired t-test would divide by a zero SD.
WIGGLE = {
    "full": 2,
    "isvd": 3,
    "frozen": 5,
    "nogist": 7,
    "fd": 13,
    "oja": 17,
    "random": 19,
    "bf16": 23,
}


def _arm(family: str, tracker: str) -> str:
    return NOGIST[family] if tracker == "nogist" else ARM[tracker]


def _sha(tracker: str, task: str, t: int, bad: dict[str, set[int] | None]) -> str | None:
    """The record's `prompt_sha256`: one digest per (task, trial) key across the arms
    (prereg section 7 (e)), unless this fixture is breaking that on purpose."""
    if tracker in bad:
        spoil = bad[tracker]
        if spoil is None:
            return None  # the arm wrote no digest at all
        if task == "niah_single" and t in spoil:
            return f"{task}:{t}:other-prompt"
    return f"{task}:{t}"


def _bits(tracker: str, w: int, delta: float, spread: float = 3e-4) -> float:
    """One window's bits/token: a shared corpus term (the spread a pairing removes), the
    arm's offset from the reference, and a per-arm wiggle at amplitude ``spread`` -- which
    is what sets the PAIRED SD the TOST's decidability bound is read against (section 6:
    decidable at +/-0.02 over 32 windows only for s < 0.0667)."""
    return 3.40 + 0.01 * ((w * 7) % 13 - 6) + delta + ((w * WIGGLE[tracker]) % 11 - 5) * spread


def write_pod(
    root: Path,
    family: str,
    *,
    hits: dict[str, int],
    delta: dict[str, float] | None = None,
    errors: dict[str, int] | None = None,
    sbits: dict[str, float] | None = None,
    diag: list[dict[str, object]] | None = None,
    arms: dict[str, str] | None = None,
    ctx: int = CTX,
    spread: float = 3e-4,
    bad_sha: dict[str, set[int] | None] | None = None,
) -> Path:
    """One pod directory. ``hits`` is hits-per-cell per tracker (nested prefixes of the
    24 trial indices, so the discordance of a pair is the difference of its counts and
    every contrast is one-directional); ``delta`` the per-arm bits/token offset;
    ``errors`` how many of a tracker's `niah_single` trials raised; ``bad_sha`` maps a
    tracker to the `niah_single` trials whose `prompt_sha256` disagrees with the other
    arms' (or to None, for an arm that wrote no digest at all) -- section 7 (e)'s pairing
    invariant, violated on purpose."""
    d = root / f"gate1_v2_stage1_{family}"
    d.mkdir(parents=True)
    model, delta, errors, sbits = MODELS[family], delta or {}, errors or {}, sbits or {}
    bad_sha = bad_sha or {}
    (d / "manifest.json").write_text(json.dumps({"pod": d.name, "model": model, "errors": 0}))
    trials, pplw, ppl = [], [], []
    for tracker, h in hits.items():
        arm = (arms or {}).get(tracker, _arm(family, tracker))
        n_err = errors.get(tracker, 0)
        for task in TASKS:
            for t in range(N_TRIALS):
                bad = task == "niah_single" and t >= N_TRIALS - n_err
                trials.append(
                    {
                        "model": model, "arm": arm, "task": task, "ctx": ctx, "seed": 0,
                        "trial": t, "hit": int(t < h and not bad), "frac": 0.0,
                        "generator": "v2", "haystack_id": "pg19", "depth": 0.4,
                        "code_family": "numbers", "prompt_sha256": _sha(tracker, task, t, bad_sha),
                        "error": "RuntimeError: boom" if bad else None, "source": "fixture",
                    }
                )  # fmt: skip
        for w in range(WINDOWS):
            pplw.append(
                {
                    "model": model, "arm": arm, "ctx": ctx, "window_idx": w, "ntok": NTOK,
                    "nll_sum_nats": _bits(tracker, w, delta.get(tracker, 0.0), spread)
                    * NTOK * math.log(2),
                    "corpus": "pg19-val", "source": "fixture",
                }
            )  # fmt: skip
        ppl.append(
            {
                "model": model, "arm": arm, "ctx": ctx, "ppl": 10.0, "ratio": 0.15,
                "sbits": sbits.get(tracker, SBITS), "tok_eq": None, "corpus": "pg19-val",
                "source": "fixture",
            }
        )  # fmt: skip
    for name, rows in (("trials", trials), ("pplw", pplw), ("ppl", ppl), ("diag", diag or [])):
        if rows:
            (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


# The three shapes the outcomes are built from. `flat` is every tracker on the same
# cells (the floor-against-floor shape a Gate-1 retrieval axis is predicted to have);
# `split` is the r64 arm above both controls by the plan's 22 / 10 / 8 of 24.
FLAT = dict.fromkeys(("full", "isvd", "frozen", "nogist", "fd"), 20) | {"full": 24}
SPLIT = {"full": 24, "isvd": 22, "frozen": 10, "nogist": 8, "fd": 22}
TIGHT = {"frozen": 0.005, "nogist": 0.005, "fd": 0.005, "full": -0.02}
WIDE = {"frozen": 0.05, "nogist": 0.05, "fd": 0.05, "full": -0.02}


def verdict_for(root: Path, **pods: dict[str, object]) -> tuple[gate1.Verdict, gate1.Gate1Data]:
    dirs = [write_pod(root, family, **kw) for family, kw in pods.items()]  # type: ignore[arg-type]
    data = gate1.load(dirs)
    retr, ppl = gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data)
    return gate1.gate1_verdict(retr, ppl, data), data


def two_families(root: Path, **kw: object) -> tuple[gate1.Verdict, gate1.Gate1Data]:
    return verdict_for(root, llama=dict(kw), qwen=dict(kw))


# --- (a) the C branch ---------------------------------------------------------------


def test_identical_trackers_select_branch_c(tmp_path: Path) -> None:
    """Every tracker on the same cells and inside the +/-0.02 margin: no separation to
    find and every TOST passing, which is Branch C (prereg section 4 rule 3).

    The perplexity contrasts here are Holm-significant with the r64 arm ahead, and they
    still license nothing: rule 1's third condition is the TOST *failing*, and at
    d = -0.005 bits it passes. That is the prereg's own mutual-exclusivity example --
    "two arms that differ significantly *and* equivalently" -- pinned as a test."""
    v, _ = two_families(tmp_path, hits=FLAT, delta=TIGHT)
    assert v.branch == "C", v.reason
    assert v.families_separated == []
    data = gate1.load([tmp_path / "gate1_v2_stage1_llama", tmp_path / "gate1_v2_stage1_qwen"])
    ppl = {(c.family, c.b): c for c in gate1.ppl_contrasts(data)}
    frozen = ppl[("llama", "frozen")]
    assert frozen.d_bits < 0 and frozen.p_holm is not None and frozen.p_holm < 0.05
    assert frozen.equivalent, "d = -0.005 bits is inside the margin, so the TOST passes"


# --- (b) A/B in two families --------------------------------------------------------


def test_beating_both_controls_in_two_families_selects_ab(tmp_path: Path) -> None:
    """22/24 against 10/24 and 8/24 on every task in both families: 12 and 14 pairs won
    with none lost, which clears the first Holm slot of a 16-member family on its own."""
    v, _ = two_families(tmp_path, hits=SPLIT, delta=TIGHT)
    assert v.branch == "A/B", v.reason
    assert v.families_separated == ["llama", "qwen"]


def test_the_primary_retrieval_family_is_the_sixteen_members_the_prereg_fixes(
    tmp_path: Path,
) -> None:
    """2 contrasts x 4 tasks x 2 families (section 6), Holm inside it; the `fd` members
    are the secondary family's and carry their own adjustment."""
    _, data = two_families(tmp_path, hits=SPLIT, delta=TIGHT)
    retr = gate1.retrieval_contrasts(data)
    primary = [c for c in retr if c.primary]
    assert len(primary) == 16 and {c.b for c in primary} == {"frozen", "nogist"}
    assert all(c.p_holm is not None and c.p_holm < 0.05 for c in primary)
    frozen = next(c for c in primary if c.family == "llama" and c.task == "vt" and c.b == "frozen")
    assert (frozen.a_favored, frozen.b_favored, frozen.n_paired) == (12, 0, 24)
    assert frozen.p == pytest.approx(2.0**-11)
    assert {c.b for c in retr if not c.primary} == {"fd"}
    ppl = gate1.ppl_contrasts(data)
    assert sum(1 for c in ppl if c.p_holm is not None) == 4  # the 4-member ppl family


# All eight arms, and what the cut ladder leaves. No pplw.jsonl: this test is about the
# family sizes alone, and the bootstrap is the only slow thing in the module.
ALL_EIGHT = {"full": 24, "isvd": 20, "nogist": 20, "frozen": 20, "fd": 20, "bf16": 20,
             "oja": 20, "random": 20}  # fmt: skip


@pytest.mark.parametrize(
    ("dropped", "secondary"),
    [((), 24), (("random",), 16), (("random", "oja"), 8)],
)
def test_the_holm_families_are_the_sizes_section_6_fixes(
    tmp_path: Path, dropped: tuple[str, ...], secondary: int
) -> None:
    """Primary retrieval 16 = 2 contrasts x 4 tasks x 2 families, secondary 24 = 3 x 4 x 2
    -- and the secondary family is corrected at its REALISED m, which section 9's cut
    ladder leaves at 16 after dropping random and 8 after dropping oja_tuned as well. The
    bf16 arm is in the pod and in no contrast: its reading is prereg/bf16_gist.md's."""
    hits = {k: v for k, v in ALL_EIGHT.items() if k not in dropped}
    for family in ("llama", "qwen"):
        d = write_pod(tmp_path, family, hits=hits)
        (d / "pplw.jsonl").unlink()
    data = gate1.load([tmp_path / f"gate1_v2_stage1_{f}" for f in ("llama", "qwen")])
    retr = gate1.retrieval_contrasts(data)
    assert sum(1 for c in retr if c.primary) == 16
    assert sum(1 for c in retr if not c.primary) == secondary
    assert all(c.p_holm is not None for c in retr), "no error row, so no member left a family"
    assert "bf16" not in {c.b for c in retr}


# --- (c) one family separated -------------------------------------------------------


def test_one_family_separated_is_undecided(tmp_path: Path) -> None:
    """Rule 2 counts model families separated at 16K and Stage 1 has two, so one is
    never rounded up -- and the same separation blocks C (rule 3 (i))."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": SPLIT, "delta": TIGHT},
        qwen={"hits": FLAT, "delta": TIGHT},
    )
    assert v.branch == "UNDECIDED" and v.families_separated == ["llama"]
    assert "llama" in v.reason


# --- (d) an error row on an arm the rule reads --------------------------------------


def test_an_fd_error_cell_refuses_and_prints_failed(tmp_path: Path) -> None:
    """`fd` is one of the four arms the rule reads, so its error rows refuse the verdict
    -- and the cell is printed as FAILED, never as `--`.

    The branch value is `REFUSED`, not `UNDECIDED`: ruling R-L3-12 overrides section 4's
    literal wording, because the two states are different findings and a table that
    prints one for the other misreports the pod. `UNDECIDED` means the rule ran and
    neither branch held; `REFUSED` means it never ran on that family. `gate1_verdict`
    still RETURNS rather than raises (section 4, "Refusal, and the `--` rule"): a raise
    would leave `make gate1` with no table to print the failure in."""
    out = tmp_path / "gate1.md"
    dirs = [
        write_pod(tmp_path, "llama", hits=SPLIT | {"fd": 21}, delta=TIGHT, errors={"fd": 3}),
        write_pod(tmp_path, "qwen", hits=FLAT, delta=TIGHT),
    ]
    data = gate1.load(dirs)
    v = gate1.gate1_verdict(gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.branch == "REFUSED"
    assert "bugSseed-r64-h256-fd" in v.reason and "RuntimeError: boom" in v.reason
    tables.gate1_table(dirs, out)
    md = out.read_text()
    assert "FAILED (3 errors)" in md
    # ...and the exception text that replaced the cells, which section 4 requires of the
    # table, not only of the verdict line.
    assert "`bugSseed-r64-h256-fd` / niah_single: 3 error records, first `RuntimeError: boom`" in md


def test_an_error_on_a_secondary_arm_does_not_refuse(tmp_path: Path) -> None:
    """An error on arms 6-8 removes that arm's secondary members and nothing else
    (prereg section 4); it is listed in the reason and the branch still reads."""
    v, _ = two_families(
        tmp_path, hits=FLAT | {"oja": 18}, delta=TIGHT | {"oja": 0.01}, errors={"oja": 2}
    )
    assert v.branch == "C", v.reason
    assert "oja_r64_h256_seed_tuned" in v.reason


# --- the perplexity route, and the TOST that gates it -------------------------------


def test_a_perplexity_only_separation_selects_ab(tmp_path: Path) -> None:
    """The live outcome section 5 names: every retrieval cell flat, both controls behind
    the r64 arm by 0.05 bits/token -- Holm-significant, d < 0, and the +/-0.02 TOST
    failing -- so both families separate on the perplexity axis alone."""
    v, data = two_families(tmp_path, hits=FLAT, delta=WIDE)
    assert v.branch == "A/B" and v.families_separated == ["llama", "qwen"]
    ppl = {(c.family, c.b): c for c in gate1.ppl_contrasts(data)}
    assert not ppl[("llama", "frozen")].equivalent
    assert ppl[("llama", "frozen")].d_bits == pytest.approx(-0.05, abs=2e-3)
    assert ppl[("llama", "frozen")].primary and not ppl[("llama", "fd")].primary


def test_a_failing_tost_blocks_branch_c(tmp_path: Path) -> None:
    """C's condition (ii) is stated positively: what blocks it is a TOST failing. Here
    the retrieval axis is flat (condition (i) holds) and only the margin decides."""
    v, _ = two_families(tmp_path, hits=FLAT, delta=TIGHT | {"fd": 0.05})
    assert v.branch == "UNDECIDED" and "fd" in v.reason


# --- the refusals the records decide ------------------------------------------------


def test_an_unknown_arm_refuses(tmp_path: Path) -> None:
    """An arm the tracker map does not know is an error, not a silent skip: a record
    dropped here would shrink a cell without shrinking n."""
    with pytest.raises(ValueError, match="bugSseed-r64-h9999"):
        gate1.load([write_pod(tmp_path, "llama", hits=FLAT, arms={"frozen": "bugSseed-r64-h9999"})])


def test_a_frozen_arm_still_repairing_after_the_freeze_refuses(tmp_path: Path) -> None:
    """Prereg section 4: a `fixed_k`/`fixed_v` row past `freeze_after` is a dispatch
    defect -- the arm did not run the mechanism the file says it runs -- except the one
    window per (sample, layer) that straddles the freeze, which section 7 (c) exempts."""
    straddle = {
        "arm": "frozen_r64_h256_seed", "task": "niah_single", "idx": 1, "layer": 0,
        "ctx": CTX, "tokens_seen": 4160, "fixed_k": True, "fixed_v": False,
    }  # fmt: skip
    ok, _ = verdict_for(
        tmp_path,
        llama={"hits": FLAT, "delta": TIGHT, "diag": [straddle]},
        qwen={"hits": FLAT, "delta": TIGHT},
    )
    assert ok.branch == "C", ok.reason
    later = straddle | {"tokens_seen": 8192}
    bad, _ = verdict_for(
        tmp_path / "second",
        llama={"hits": FLAT, "delta": TIGHT, "diag": [straddle, later]},
        qwen={"hits": FLAT, "delta": TIGHT},
    )
    assert bad.branch == "REFUSED" and "frozen dispatch" in bad.reason


def test_a_nogist_arm_off_its_byte_match_refuses(tmp_path: Path) -> None:
    """The byte match is required of the RUN, not only of the arm file: outside
    1 +/- 0.05 the isvd-vs-nogist members are not the mechanism contrast that was
    pre-registered (prereg section 4, section 7 (f))."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": FLAT, "delta": TIGHT, "sbits": {"nogist": SBITS * 1.2}},
        qwen={"hits": FLAT, "delta": TIGHT},
    )
    assert v.branch == "REFUSED" and "byte match" in v.reason and "1.20" in v.reason


def test_the_byte_match_is_never_read_as_passed_when_it_was_not_measured(
    tmp_path: Path,
) -> None:
    """No `sbits` on the pod is `not measured`, which is not a pass (prereg section 9's
    rule for an unmeasured reading, applied to the refusal that reads the records)."""
    d = write_pod(tmp_path, "llama", hits=FLAT, delta=TIGHT)
    (d / "ppl.jsonl").unlink()
    data = gate1.load([d, write_pod(tmp_path, "qwen", hits=FLAT, delta=TIGHT)])
    v = gate1.gate1_verdict(gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.branch == "REFUSED" and "not measured" in v.reason


# --- the table ----------------------------------------------------------------------


def test_the_gate1_subcommand_writes_the_table(tmp_path: Path) -> None:
    """`make gate1` renders one block per family x ctx, Wilson intervals per cell, the
    perplexity block and the verdict as the last line."""
    out = tmp_path / "out" / "gate1.md"
    dirs = [
        write_pod(tmp_path, "llama", hits=SPLIT, delta=TIGHT),
        write_pod(tmp_path, "qwen", hits=SPLIT, delta=TIGHT),
    ]
    tables.gate1_table(dirs, out)
    md = out.read_text()
    assert "## llama — ctx 16384" in md and "## qwen — ctx 16384" in md
    assert "0.92 [0.74,0.98] (22/24)" in md  # the r64 arm's cell, Wilson 95%
    assert "| frozen | 0.42 [0.24,0.61] (10/24) *" in md  # Holm-significant vs isvd
    assert "bits/token" in md and "TOST" in md
    verdict = next(x for x in md.splitlines() if x.startswith("VERDICT: "))
    assert verdict.startswith("VERDICT: A/B — ")
    assert "\n\n\n" not in md, "a block with no error rows and no dropped keys adds no gap"


# --- nothing to read is never a Branch C -------------------------------------------


def _bare_pod(root: Path, family: str) -> Path:
    """A `--dry-run` directory: the manifest and an empty `trials.jsonl`, nothing else."""
    d = root / f"gate1_v2_stage1_{family}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"pod": d.name, "model": MODELS[family], "dry_run": True})
    )
    (d / "trials.jsonl").write_text("")
    return d


NO_16K = "no 16K family in the records (the verdict reads ctx 16384 only)"


def test_an_empty_pod_set_is_undecided_never_a_vacuous_c(tmp_path: Path) -> None:
    """The dry-run shape `make gate1` meets before a pod has run: C is a positive claim
    of non-separation and "an absent member is not evidence for it" (prereg section 4
    rule 3), so a record set with no member the rule can read selects nothing."""
    dirs = [_bare_pod(tmp_path, "llama"), _bare_pod(tmp_path, "qwen")]
    data = gate1.load(dirs)
    v = gate1.gate1_verdict(gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data), data)
    assert (v.branch, v.reason) == ("UNDECIDED", NO_16K)
    out = tmp_path / "gate1.md"
    tables.gate1_table(dirs, out)
    assert f"VERDICT: UNDECIDED — {NO_16K}" in out.read_text()


def test_a_32k_only_pod_set_is_undecided_never_a_vacuous_c(tmp_path: Path) -> None:
    """ "The verdict reads the 16K contrasts only" (prereg section 4's scope note), so a
    32K-only set has no member for rules 1-3 -- and a 32K pod "never change[s] the
    branch", which a C read off its cells would."""
    v, _ = two_families(tmp_path, hits=FLAT, delta=TIGHT, ctx=32768)
    assert (v.branch, v.reason) == ("UNDECIDED", NO_16K)


# --- section 7 (e): the pairing invariant, verified from prompt_sha256 --------------


def test_a_disagreeing_prompt_digest_drops_that_key_from_the_member(tmp_path: Path) -> None:
    """Section 4: the McNemar runs "on byte-identical prompts (section 3's pairing
    invariant, verified from `prompt_sha256` per section 7 (e); a key whose digests
    disagree is dropped from that member and the drop is reported with the key)" --
    and section 6: it "shrinks that member's paired n", it does not remove the member."""
    dirs = [
        write_pod(tmp_path, "llama", hits=FLAT, delta=TIGHT, bad_sha={"frozen": {7}}),
        write_pod(tmp_path, "qwen", hits=FLAT, delta=TIGHT),
    ]
    data = gate1.load(dirs)
    retr = {(c.family, c.task, c.b): c for c in gate1.retrieval_contrasts(data)}
    member = retr[("llama", "niah_single", "frozen")]
    assert (member.n_paired, member.dropped_keys) == (23, ((0, 7),))
    assert retr[("llama", "niah_single", "nogist")].n_paired == 24, "one member, not the cell"
    assert retr[("llama", "niah_multikey", "frozen")].n_paired == 24
    out = tmp_path / "gate1.md"
    tables.gate1_table(dirs, out)
    assert "pairing: 1 key dropped (llama/niah_single: (0,7))" in out.read_text()


def test_an_arm_that_wrote_no_digest_refuses_the_member(tmp_path: Path) -> None:
    """A key whose digest set "contains `None` is a failure as well as a mismatch"
    (section 7 (e)). Every key of the member goes, which leaves it with no pairing at
    all -- section 4's `mcnemar_exact` -> None boundary, which refuses a primary
    member's family (ruling R-L3-12: the branch value is REFUSED)."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": FLAT, "delta": TIGHT, "bad_sha": {"frozen": None}},
        qwen={"hits": FLAT, "delta": TIGHT},
    )
    assert v.branch == "REFUSED" and "prompt_sha256" in v.reason
    assert "all 24 keys dropped" in v.reason


# --- every verdict carries every refusal and every blocker --------------------------


def test_a_refusal_is_named_beside_the_families_that_separated(tmp_path: Path) -> None:
    """A refusal is never silently dropped by a branch that reads the other families:
    two families separate, the third is refused, and the verdict names all three (the
    five-reviewer simulation reads the table alone). Ruling R-L3-15 supersedes R-L3-12 on
    this shape: refusals compose PER FAMILY, so the two clean, separated families reach
    A/B on their own and the refused third is still named beside it -- never silently
    dropped by the branch the other two earn, and never swallowing it either."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": SPLIT, "delta": TIGHT},
        qwen={"hits": SPLIT, "delta": TIGHT},
        mistral={"hits": FLAT, "delta": TIGHT, "sbits": {"nogist": SBITS * 1.2}},
    )
    assert v.branch == "A/B", v.reason
    assert v.families_separated == ["llama", "qwen"]
    assert "byte match" in v.reason and "mistral" in v.reason
    assert "llama" in v.reason and "qwen" in v.reason
    assert any("beats frozen" in m for m in v.members)


# --- ruling R-L3-15: refusals compose per family, not over the whole verdict --------


def test_two_clean_families_separated_survive_a_third_familys_refusal(
    tmp_path: Path,
) -> None:
    """Ruling R-L3-15 (lane ledger; prereg Amendment 1 restates §4 accordingly): a
    refused family contributes no evidence, in either direction. Two families separated
    per rule 1 is already >= `MIN_FAMILIES_FOR_AB` on its own, and a third family's `fd`
    error does not withdraw that -- it is excluded from the count and named beside it."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": SPLIT, "delta": TIGHT},
        qwen={"hits": SPLIT, "delta": TIGHT},
        mistral={"hits": SPLIT | {"fd": 21}, "delta": TIGHT, "errors": {"fd": 3}},
    )
    assert v.branch == "A/B", v.reason
    assert v.families_separated == ["llama", "qwen"]
    assert "mistral" in v.reason and "bugSseed-r64-h256-fd" in v.reason


def test_one_separated_one_refused_is_still_refused(tmp_path: Path) -> None:
    """Two families only: A/B needs >= 2 NON-refused separated families and one refused
    family is never a separated one, so one separated + one refused reaches neither A/B
    nor C (C needs every family clean too) -- REFUSED, same outcome as before this
    ruling, with the refusal named in the reason."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": SPLIT, "delta": TIGHT},
        qwen={"hits": SPLIT | {"fd": 21}, "delta": TIGHT, "errors": {"fd": 3}},
    )
    assert v.branch == "REFUSED", v.reason
    assert "qwen" in v.reason and "bugSseed-r64-h256-fd" in v.reason
    assert "llama" in v.reason


def test_two_clean_families_plus_a_third_refused_blocks_branch_c(tmp_path: Path) -> None:
    """Branch C reads EVERY family in the input (ruling R-L3-15): two families identical
    with every TOST passing would be C alone (the existing C fixture), but a third
    family's byte match failing means C cannot be read off all of the input -- and with
    no separation anywhere either, REFUSED is what remains."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": FLAT, "delta": TIGHT},
        qwen={"hits": FLAT, "delta": TIGHT},
        mistral={"hits": FLAT, "delta": TIGHT, "sbits": {"nogist": SBITS * 1.2}},
    )
    assert v.branch == "REFUSED", v.reason
    assert "mistral" in v.reason and "byte match" in v.reason


# --- ruling R-L3-16: a refused family leaves the Holm families too ------------------

# The counter-example the ruling is written from: the r64 arm at 18 of 24 against
# `frozen` 9 and `nogist` 8 wins 9 and 10 pairs with none lost (p = 2^-8 and 2^-9), which
# clears the 16-member primary family -- every member adjusts to 0.03125 -- and clears
# nothing wider: pool eight more members in and the `frozen` members land on 0.0625.
NARROW = {"full": 24, "isvd": 18, "frozen": 9, "nogist": 8, "fd": 18}


def test_a_refused_familys_members_leave_the_holm_family_before_the_correction(
    tmp_path: Path,
) -> None:
    """Ruling R-L3-16: "a refused family contributes no evidence, in either direction"
    (R-L3-15) has to hold at the CORRECTION too, not only at the branch dispatch. Two
    clean families that separate at their own m = 16 keep that separation when a third
    family is refused: its members leave every Holm family before Holm runs, so the
    realised m stays 16 and the A/B the clean families earn is not withdrawn by the
    refusal -- which is exactly what pooling all 24 would do (`frozen` at 0.0625)."""
    v, data = verdict_for(
        tmp_path,
        llama={"hits": NARROW, "delta": TIGHT},
        qwen={"hits": NARROW, "delta": TIGHT},
        mistral={"hits": FLAT, "delta": TIGHT, "sbits": {"nogist": SBITS * 1.2}},
    )
    assert v.branch == "A/B", v.reason
    assert v.families_separated == ["llama", "qwen"]
    assert "byte match" in v.reason and "mistral" in v.reason
    primary = [c for c in gate1.retrieval_contrasts(data) if c.primary]
    assert len(primary) == 24, "the refused family's members are still computed and printed"
    assert sum(1 for c in primary if c.p_holm is not None) == 16, "the realised m"
    assert all(c.p_holm is None for c in primary if c.family == "mistral")
    vt = next(c for c in primary if (c.family, c.task, c.b) == ("llama", "vt", "frozen"))
    assert (vt.a_favored, vt.b_favored) == (9, 0)
    assert vt.p_holm == pytest.approx(0.03125), "0.0625 if the refused family is pooled in"
    # The excluded members are named as excluded, never printed with an adjusted p.
    out = tmp_path / "gate1.md"
    tables.gate1_table([tmp_path / f"gate1_v2_stage1_{f}" for f in MODELS], out)
    assert "refused (excluded from the Holm family)" in out.read_text()


# --- rule 3 (i) reads every task the prereg fixes, not only the ones that ran --------


def test_a_truncated_task_set_is_never_a_branch_c(tmp_path: Path) -> None:
    """C's retrieval condition is "no ... separation ... on **any task in any 16K
    family**" (prereg section 4 rule 3 (i)), and the tasks are section 3's four. Read
    over the tasks that happen to be PRESENT, a pod that ran two of them and separated
    on neither would print C off half the evidence -- and C is the branch that can
    retire the method's central claim."""
    dirs = [write_pod(tmp_path, f, hits=FLAT, delta=TIGHT) for f in ("llama", "qwen")]
    for d in dirs:
        rows = (d / "trials.jsonl").read_text().splitlines(keepends=True)
        (d / "trials.jsonl").write_text(
            "".join(
                x for x in rows if '"task": "niah_multivalue"' not in x and '"task": "vt"' not in x
            )
        )
    data = gate1.load(dirs)
    v = gate1.gate1_verdict(gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.branch == "UNDECIDED", v.reason
    assert "incomplete task set: llama lacks niah_multivalue, vt" in v.reason
    assert "incomplete task set: qwen lacks niah_multivalue, vt" in v.reason


# --- section 6's third TOST state ---------------------------------------------------

# 32 windows at +/-0.02 are decidable only for a paired SD below 0.0667 (section 6's
# table). This wiggle amplitude puts the paired SD an order of magnitude above it.
TOO_WIDE = 0.02


def test_a_spread_too_wide_for_the_margin_is_not_decidable_and_withholds_c(
    tmp_path: Path,
) -> None:
    """Section 6: "If a member's realised `s` still exceeds its bound, its TOST cannot
    fire whatever the point estimate is, and the member is recorded as `not decidable`
    -- never as a pass, never as a quiet fail. The C branch requires every one of its
    four TOSTs to pass, so the verdict is then UNDECIDED, and the report must say
    which it is"."""
    v, data = two_families(tmp_path, hits=FLAT, delta=TIGHT, spread=TOO_WIDE)
    frozen = next(c for c in gate1.ppl_contrasts(data) if (c.family, c.b) == ("llama", "frozen"))
    assert not frozen.decidable and not frozen.equivalent and frozen.s > 0.0667
    assert v.branch == "UNDECIDED" and "not decidable" in v.reason
    out = tmp_path / "gate1.md"
    tables.gate1_table([tmp_path / f"gate1_v2_stage1_{f}" for f in ("llama", "qwen")], out)
    assert "| not decidable |" in out.read_text(), "the TOST cell, not only the verdict"


def test_a_not_decidable_tost_on_the_ab_route_says_so_beside_the_branch(
    tmp_path: Path,
) -> None:
    """Section 6: "On the A/B side the rule reads the boolean, which is `False` in two
    different situations ... Where a separating member's TOST is `False` because it is
    `not decidable`, the verdict entry says so beside the branch, with `s` and the CI."
    So a wide spread does not withhold A/B -- it annotates it."""
    v, _ = two_families(
        tmp_path, hits=FLAT, delta={"frozen": 0.3, "nogist": 0.3, "fd": 0.3}, spread=TOO_WIDE
    )
    assert v.branch == "A/B" and v.families_separated == ["llama", "qwen"]
    assert "not decidable" in v.reason and "s=" in v.reason


# --- the pairing key, the marks, and the cells that never ran -----------------------


def test_two_corpora_at_one_ctx_are_paired_inside_their_own_corpus(tmp_path: Path) -> None:
    """The window-pairing key carries the corpus, as `tables.ppl_stats` does: two ppl
    tasks can share a ctx with different corpora, and "absolute perplexity is not
    comparable across corpora" (`records.PplwRecord`). Keyed without it, the second
    corpus's window 0 reads as the first's, duplicated."""
    d = write_pod(tmp_path, "llama", hits=FLAT, delta=TIGHT)
    rows = [json.loads(x) for x in (d / "pplw.jsonl").read_text().splitlines()]
    other = [r | {"corpus": "wt103-test", "nll_sum_nats": r["nll_sum_nats"] * 1.01} for r in rows]
    (d / "pplw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + other))
    ppl = [c for c in gate1.ppl_contrasts(gate1.load([d])) if c.b == "frozen"]
    assert {c.corpus for c in ppl} == {"pg19-val", "wt103-test"}
    assert [c.n_windows for c in ppl] == [WINDOWS, WINDOWS]


def test_a_corpus_the_reference_arm_never_scored_is_labelled_not_dropped(tmp_path: Path) -> None:
    """Every contrast in a perplexity block is taken against the reference arm, so a
    corpus it scored no window of has none -- which is a labelled row, not a block that
    silently disappears along with the sweeps that DID run on it."""
    d = write_pod(tmp_path, "llama", hits=FLAT, delta=TIGHT)
    rows = [json.loads(x) for x in (d / "pplw.jsonl").read_text().splitlines()]
    other = [r | {"corpus": "wt103-test"} for r in rows if r["arm"] != ARM["isvd"]]
    (d / "pplw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + other))
    out = tmp_path / "gate1.md"
    tables.gate1_table([d], out)
    md = out.read_text()
    assert "### llama — perplexity, ctx 16384, wt103-test" in md
    assert "reference arm has no windows for corpus wt103-test" in md


def test_a_control_that_wins_is_marked_and_a_cell_that_never_ran_says_why(
    tmp_path: Path,
) -> None:
    """Section 4 rule 3 (i) reads a separation "in either direction", so the table
    distinguishes them: `*` is a Holm-significant primary contrast favouring isvd, `‡`
    one favouring the control. And "no arm is ever printed as `--`": a cell with no
    records prints what is missing."""
    hits = {"full": 24, "isvd": 8, "frozen": 22, "nogist": 8, "fd": 8, "oja": 8}
    dirs = [write_pod(tmp_path, f, hits=hits, delta=TIGHT) for f in ("llama", "qwen")]
    trials = dirs[0] / "trials.jsonl"
    trials.write_text(
        "".join(
            x
            for x in trials.read_text().splitlines(keepends=True)
            if '"arm": "oja_r64_h256_seed_tuned"' not in x or '"task": "vt"' not in x
        )
    )
    out = tmp_path / "gate1.md"
    tables.gate1_table(dirs, out)
    md = out.read_text()
    frozen_row = next(x for x in md.splitlines() if x.startswith("| frozen |"))
    assert "‡" in frozen_row and "*" not in frozen_row
    assert "not run (no records for oja_r64_h256_seed_tuned at ctx 16384)" in md
    assert "‡" in next(x for x in md.splitlines() if x.startswith("Legend:"))


def test_the_perplexity_block_prints_no_dash(tmp_path: Path) -> None:
    """Section 4: "No arm is ever printed as `--`". The reference row has no contrast
    with itself and the arms outside the 4-member perplexity family have no adjusted
    p-value -- both used to print a dash, which is the Week-20 swap table's failure."""
    dirs = [write_pod(tmp_path, f, hits=SPLIT, delta=TIGHT) for f in ("llama", "qwen")]
    out = tmp_path / "gate1.md"
    tables.gate1_table(dirs, out)
    cells = [
        c.strip()
        for line in out.read_text().splitlines()
        if line.startswith("|")
        for c in line.split("|")
    ]
    assert "--" not in cells
    assert "reference" in cells and "n/a (secondary)" in cells
