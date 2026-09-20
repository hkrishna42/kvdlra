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

MODELS = {"llama": "unsloth/Meta-Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen2.5-7B-Instruct"}
# The record key each arm writes: `legacy_name` where the arm config sets one, the stem
# otherwise -- and the no-gist twin is per KV width (prereg section 3).
ARM = {
    "full": "full",
    "isvd": "bugSseed-r64-h256",
    "frozen": "frozen_r64_h256_seed",
    "fd": "bugSseed-r64-h256-fd",
    "oja": "oja_r64_h256_seed_tuned",
    "random": "random_r64_h256_seed",
}
NOGIST = {"llama": "nogist_h2423", "qwen": "nogist_h4460"}
TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
CTX, N_TRIALS, WINDOWS, NTOK = 16384, 24, 32, 2048
# `sbits` on a record is `ratio_stored_bits` -- stored bits relative to `full`, which
# bills 1.0 (runner._ppl_record) -- so a no-gist twin printing the r64 arm's value is the
# 1.00003x byte match its arm file solves for.
SBITS = 0.15
# A wiggle multiplier per tracker, coprime with 11: without one, two arms' per-window
# curves would differ by a constant and the paired t-test would divide by a zero SD.
WIGGLE = {"full": 2, "isvd": 3, "frozen": 5, "nogist": 7, "fd": 13, "oja": 17, "random": 19}


def _arm(family: str, tracker: str) -> str:
    return NOGIST[family] if tracker == "nogist" else ARM[tracker]


def _bits(tracker: str, w: int, delta: float) -> float:
    """One window's bits/token: a shared corpus term (the spread a pairing removes), the
    arm's offset from the reference, and a small per-arm wiggle."""
    return 3.40 + 0.01 * ((w * 7) % 13 - 6) + delta + ((w * WIGGLE[tracker]) % 11 - 5) * 3e-4


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
) -> Path:
    """One pod directory. ``hits`` is hits-per-cell per tracker (nested prefixes of the
    24 trial indices, so the discordance of a pair is the difference of its counts and
    every contrast is one-directional); ``delta`` the per-arm bits/token offset;
    ``errors`` how many of a tracker's `niah_single` trials raised."""
    d = root / f"gate1_v2_stage1_{family}"
    d.mkdir(parents=True)
    model, delta, errors, sbits = MODELS[family], delta or {}, errors or {}, sbits or {}
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
                        "model": model, "arm": arm, "task": task, "ctx": CTX, "seed": 0,
                        "trial": t, "hit": int(t < h and not bad), "frac": 0.0,
                        "generator": "v2", "haystack_id": "pg19", "depth": 0.4,
                        "code_family": "numbers", "prompt_sha256": f"{task}:{t}",
                        "error": "RuntimeError: boom" if bad else None, "source": "fixture",
                    }
                )  # fmt: skip
        for w in range(WINDOWS):
            pplw.append(
                {
                    "model": model, "arm": arm, "ctx": CTX, "window_idx": w, "ntok": NTOK,
                    "nll_sum_nats": _bits(tracker, w, delta.get(tracker, 0.0))
                    * NTOK * math.log(2),
                    "corpus": "pg19-val", "source": "fixture",
                }
            )  # fmt: skip
        ppl.append(
            {
                "model": model, "arm": arm, "ctx": CTX, "ppl": 10.0, "ratio": 0.15,
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

    The prereg (section 4, "Refusal, and the `--` rule") says `gate1_verdict` RETURNS
    `UNDECIDED` with the arm and the exception text named; the lane brief said it
    raises. The prereg is the spec, and a raise would leave `make gate1` with no table
    to print the failure in."""
    out = tmp_path / "gate1.md"
    dirs = [
        write_pod(tmp_path, "llama", hits=SPLIT | {"fd": 21}, delta=TIGHT, errors={"fd": 3}),
        write_pod(tmp_path, "qwen", hits=FLAT, delta=TIGHT),
    ]
    data = gate1.load(dirs)
    v = gate1.gate1_verdict(gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.branch == "UNDECIDED"
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
    assert bad.branch == "UNDECIDED" and "frozen dispatch" in bad.reason


def test_a_nogist_arm_off_its_byte_match_refuses(tmp_path: Path) -> None:
    """The byte match is required of the RUN, not only of the arm file: outside
    1 +/- 0.05 the isvd-vs-nogist members are not the mechanism contrast that was
    pre-registered (prereg section 4, section 7 (f))."""
    v, _ = verdict_for(
        tmp_path,
        llama={"hits": FLAT, "delta": TIGHT, "sbits": {"nogist": SBITS * 1.2}},
        qwen={"hits": FLAT, "delta": TIGHT},
    )
    assert v.branch == "UNDECIDED" and "byte match" in v.reason and "1.20" in v.reason


def test_the_byte_match_is_never_read_as_passed_when_it_was_not_measured(
    tmp_path: Path,
) -> None:
    """No `sbits` on the pod is `not measured`, which is not a pass (prereg section 9's
    rule for an unmeasured reading, applied to the refusal that reads the records)."""
    d = write_pod(tmp_path, "llama", hits=FLAT, delta=TIGHT)
    (d / "ppl.jsonl").unlink()
    data = gate1.load([d, write_pod(tmp_path, "qwen", hits=FLAT, delta=TIGHT)])
    v = gate1.gate1_verdict(gate1.retrieval_contrasts(data), gate1.ppl_contrasts(data), data)
    assert v.branch == "UNDECIDED" and "not measured" in v.reason


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
    assert md.strip().splitlines()[-1].startswith("VERDICT: A/B — ")
