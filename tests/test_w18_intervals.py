"""Week-18: the marker-free, per-trial-aware intervals builder.

Pins the w10_ruler line CONTRACT from the consumer side: the aggregate row (with the
Week-18 sbits=/n= suffix) and the [trial] line must parse, multikey included, and the
McNemar path must run on paired per-trial data.
"""

from __future__ import annotations

from pathlib import Path

import w18_intervals as wi


def _sample() -> str:
    # aggregate rows: 3/4 tasks + multikey (which w17_intervals dropped), with n= and sbits=
    rows = [
        "[niah_single ctx16384] bugSseed-r64-h256 acc=1.00 recall=1.00 ratio=0.085 sbits=0.15 n=12",
        "[niah_multikey ctx16384] bugSseed-r64-h256 acc=0.75 recall=0.9 ratio=0.085 sbits=0.15 n=12",  # noqa: E501
        "[niah_single ctx16384] think-c0.5 acc=1.00 recall=1.00 ratio=0.750 sbits=0.75 n=12",
    ]
    # per-trial lines for a McNemar contrast: arm A wins 3 discordant, B wins 0 -> favors A
    trials = []
    for i in range(12):
        a_hit = 1
        b_hit = 0 if i < 3 else 1  # A beats B on 3 trials, ties on the rest
        trials.append(
            f"[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=0 trial={i} "
            f"hit={a_hit} frac=1.000"
        )
        trials.append(
            f"[trial] task=niah_single ctx=16384 arm=think-c0.5 seed=0 trial={i} "
            f"hit={b_hit} frac={float(b_hit):.3f}"
        )
    return "\n".join(rows + trials) + "\n"


def test_parses_multikey_and_reads_n_from_line() -> None:
    cells = wi.parse_cells(_sample())
    # multikey is present (the w17 regex dropped it) and n came off the line
    mk = cells[("16384", "bugSseed-r64-h256", "niah_multikey")]
    assert mk["n"] == 12
    assert mk["hits"] == 9  # round(0.75 * 12)
    assert mk["sbits"] == 0.150


def test_older_line_without_sbits_still_needs_n() -> None:
    # a row with n= but no sbits= (mixed logs) still parses; sbits None
    line = "[vt ctx32768] palu-r0.5      acc=0.56 recall=0.70 ratio=0.504 n=16\n"
    cells = wi.parse_cells(line)
    c = cells[("32768", "palu-r0.5", "vt")]
    assert c["n"] == 16 and c["sbits"] is None


def test_mcnemar_contrast_runs_and_flags_significance(tmp_path: Path) -> None:
    f = tmp_path / "w18-sample-lines.txt"
    f.write_text(_sample())
    blob = wi.build([str(f)], [("bugSseed-r64-h256", "think-c0.5")])
    contrasts = blob["contrasts"]
    single = next(c for c in contrasts if c["task"] == "niah_single")
    assert single["a_favored"] == 3 and single["b_favored"] == 0
    # 3 vs 0 discordant -> exact two-sided p = 2 * 0.5^3 = 0.25 (not significant at n=12)
    assert single["p_value"] == 0.25
    assert single["significant"] is False
