"""The perplexity protocol: held-out corpora, 2048-token windows, paired CI + TOST.

v1 scored WikiText-103 TRAIN over 512-token windows, 4-8 samples, and published one
pooled number per arm with no interval (``docs/plan/CODE_AUDIT.md``). This pins the
replacement:

1. ``tables.ppl_stats`` turns the per-window rows (``pplw.jsonl``) into bits/token
   ``nll / (ntok * ln 2)``, a per-window paired difference against ``full``, its
   bootstrap 95% CI and TOST at +/-0.05 bits -- the CI brackets a known shift, and
   TOST separates a 0.01-bit shift (equivalent) from a 0.10-bit one (not).
2. ``load_task`` refuses a ppl task that asks for more windows than its corpus holds,
   at load time rather than after hours on a pod, and the four shipped held-out task
   configs sit inside their ceiling.
3. The ``[pplw]`` line carries its corpus and still parses without one, because the
   archived paper-v1 logs have no ``corpus=`` field.

Hermetic: synthetic per-window rows and a monkeypatched corpus table, no network and
no model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import tables

from kvdlra.eval import config, frontier
from kvdlra.eval.config import CORPUS_TOKENS, load_pod, load_task
from kvdlra.eval.records import PplwRecord, parse_pplw_lines, write_jsonl

NTOK = 2048
N_WINDOWS = 32
# The four held-out ppl tasks Task 6's pods name, and the corpus each reads.
SHIPPED = {
    "ppl_16k_pg19val": ("pg19-val", 16384, 32),
    "ppl_32k_pg19val": ("pg19-val", 32768, 32),
    "ppl_16k_wt103test": ("wikitext-103-test", 16384, 14),
    "ppl_32k_wt103test": ("wikitext-103-test", 32768, 7),
}


def _rows(shift: float) -> tuple[list[PplwRecord], np.ndarray, np.ndarray]:
    """``full`` and one arm over the same 32 windows, the arm ``shift`` bits worse.

    The jitter is what makes this a *paired* measurement: the per-window spread (0.2
    bits) is 20x the shift, so an unpaired comparison would see nothing. The arm's own
    noise keeps the difference non-degenerate (TOST's t-test needs a variance).
    """
    rng = np.random.default_rng(0)
    base = 3.0 + rng.normal(0.0, 0.2, N_WINDOWS)
    other = base + shift + rng.normal(0.0, 0.01, N_WINDOWS)
    rows: list[PplwRecord] = []
    for arm, bits in (("full", base), ("bug-r64", other)):
        rows += [
            {
                "model": "tiny",
                "arm": arm,
                "ctx": 16384,
                "window_idx": i,
                "ntok": NTOK,
                "nll_sum_nats": float(b) * NTOK * math.log(2.0),
                "corpus": "pg19-val",
                "source": "synthetic:1",
            }
            for i, b in enumerate(bits)
        ]
    return rows, base, other


def test_bits_per_token_is_the_mean_of_nll_over_ntok_ln2() -> None:
    rows, base, other = _rows(0.01)
    by_arm = {s["arm"]: s for s in tables.ppl_stats(rows)}
    assert by_arm["full"]["bits"] == pytest.approx(float(base.mean()), abs=1e-12)
    assert by_arm["bug-r64"]["bits"] == pytest.approx(float(other.mean()), abs=1e-12)
    assert by_arm["full"]["n_windows"] == by_arm["bug-r64"]["n_windows"] == N_WINDOWS
    # The baseline has nothing to be paired against, so it carries no difference.
    assert by_arm["full"]["d_bits"] is None and by_arm["bug-r64"]["d_bits"] is not None


@pytest.mark.parametrize("shift", [0.01, 0.10])
def test_the_paired_interval_brackets_the_true_shift(shift: float) -> None:
    rows, base, other = _rows(shift)
    s = next(x for x in tables.ppl_stats(rows) if x["arm"] == "bug-r64")
    true_d = float((other - base).mean())
    assert s["d_bits"] == pytest.approx(true_d, abs=1e-12)
    assert s["lo"] is not None and s["hi"] is not None
    # Against the INJECTED shift, not the sample's own mean (which the CI always
    # brackets almost tautologically, since it is centered on that same mean) -- and
    # excluding zero, so this also confirms the interval resolves a real effect.
    assert s["lo"] <= shift <= s["hi"] and s["lo"] > 0


def test_tost_separates_an_equivalent_shift_from_a_material_one() -> None:
    """+/-0.05 bits/token is the equivalence margin; 0.01 clears it, 0.10 does not."""
    small = next(x for x in tables.ppl_stats(_rows(0.01)[0]) if x["arm"] == "bug-r64")
    large = next(x for x in tables.ppl_stats(_rows(0.10)[0]) if x["arm"] == "bug-r64")
    assert small["equivalent"] is True and small["p_tost"] is not None
    assert small["p_tost"] < 0.05
    assert large["equivalent"] is False and large["p_tost"] is not None
    assert large["p_tost"] > 0.05


def test_a_window_scored_twice_is_refused_not_silently_overwritten() -> None:
    """Two rows for one (arm, ctx, corpus, window_idx) -- a re-harvested log appended to
    an existing `pplw.jsonl`, or a re-run of half a sweep -- used to overwrite silently,
    so the mean was taken over fewer windows than the file held and the pairing check
    could not see it (both arms still have the same window SET). Fail loud instead."""
    rows, _, _ = _rows(0.01)
    with pytest.raises(SystemExit, match=r"window_idx=0 twice"):
        tables.ppl_stats([*rows, rows[0]])


def test_ppl_table_writes_one_markdown_row_per_arm(tmp_path: Path) -> None:
    """The subcommand's file: `ppl_<pod>.md`, never `table*.md` -- `make tables`
    concatenates `table*.md` and diffs it against the paper-v1 golden."""
    rows, _, _ = _rows(0.01)
    d = tmp_path / "results" / "w99_ppl"
    d.mkdir(parents=True)
    write_jsonl(d / "pplw.jsonl", cast(list[Any], rows))
    out = tmp_path / "ppl_w99_ppl.md"
    tables.ppl_table(d, out)
    md = out.read_text()
    assert not out.name.startswith("table")
    assert "pplw.jsonl" in md  # provenance note: the file the numbers came from
    assert sum(1 for ln in md.splitlines() if ln.startswith("| full ")) == 1
    assert sum(1 for ln in md.splitlines() if ln.startswith("| bug-r64 ")) == 1
    assert "+/-0.05" in md


def test_ppl_table_keeps_two_corpora_at_one_ctx_separate(tmp_path: Path) -> None:
    """Two ppl tasks can share a ctx with different corpora (PG-19 val + WikiText-103
    test both ship a 16K task) inside one pod's `pplw.jsonl`. `ppl_stats` used to key
    by (arm, ctx) alone, so the second corpus's windows would silently overwrite the
    first's; each must come back as its own row, with its own bits/token, never
    pooled together."""
    rows: list[PplwRecord] = [
        {
            "model": "tiny",
            "arm": "full",
            "ctx": 16384,
            "window_idx": i,
            "ntok": NTOK,
            "nll_sum_nats": bits * NTOK * math.log(2.0),
            "corpus": corpus,
            "source": "synthetic:1",
        }
        for corpus, bits_list in (("pg19-val", [3.0, 3.2]), ("wikitext-103-test", [5.0, 5.4]))
        for i, bits in enumerate(bits_list)
    ]
    stats = {s["corpus"]: s for s in tables.ppl_stats(rows)}
    assert len(stats) == 2
    assert stats["pg19-val"]["bits"] == pytest.approx(3.1)
    assert stats["wikitext-103-test"]["bits"] == pytest.approx(5.2)
    assert stats["pg19-val"]["n_windows"] == stats["wikitext-103-test"]["n_windows"] == 2

    d = tmp_path / "results" / "w99_twocorpus"
    d.mkdir(parents=True)
    write_jsonl(d / "pplw.jsonl", cast(list[Any], rows))
    tables.ppl_table(d, tmp_path / "ppl_w99_twocorpus.md")
    md = (tmp_path / "ppl_w99_twocorpus.md").read_text()
    header = next(ln for ln in md.splitlines() if ln.startswith("| arm"))
    assert "corpus" in header
    assert sum(1 for ln in md.splitlines() if ln.startswith("| full ")) == 2  # one per corpus


def test_ppl_table_fails_loud_on_an_empty_but_present_pplw_file(tmp_path: Path) -> None:
    """A zero-byte `pplw.jsonl` is as fatal as a missing one -- previously it silently
    produced a table with no rows instead of refusing."""
    d = tmp_path / "results" / "w99_empty"
    d.mkdir(parents=True)
    (d / "pplw.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no per-window rows"):
        tables.ppl_table(d, tmp_path / "out.md")


def test_load_pod_refuses_two_ppl_tasks_at_one_ctx_with_different_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling PR-32: `scripts/pod.py`'s pod gate counts pplw/ppl records per (arm, ctx),
    blind to corpus, so a pod naming two ppl tasks at the same ctx with different
    corpora would let one corpus's windows silently fill the other's slot in that
    count. Refused at load, naming both task files."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "pods").mkdir()
    head = "generator: ppl\nctx: 16384\nwindow: 2048\nn_samples: 4\n"
    (tmp_path / "tasks" / "a.yaml").write_text(f"name: a\n{head}corpus: pg19-val\n")
    (tmp_path / "tasks" / "b.yaml").write_text(f"name: b\n{head}corpus: wikitext-103-test\n")
    (tmp_path / "pods" / "mixed.yaml").write_text(
        "name: mixed\nmodel: m\narms: []\ntasks: [a, b]\n"
    )
    monkeypatch.setattr("kvdlra.eval.config.ROOT", tmp_path)
    with pytest.raises(ValueError, match=r"mixed\.yaml") as e:
        load_pod("mixed")
    assert "a.yaml" in str(e.value) and "b.yaml" in str(e.value)


def test_every_shipped_pod_still_loads() -> None:
    """The PR-32 guard now runs inside every `load_pod` call -- no existing pod, none
    of which mixes corpora at one ctx, may regress."""
    pods = sorted((config.ROOT / "pods").glob("*.yaml"))
    assert pods  # the loop below would pass vacuously over an empty directory
    for p in pods:
        load_pod(p.stem)
    # `cell_elapsed_s`'s key is `<arm>/<task>/<ctx>`: a ppl/latency task rides its own
    # `.name` as `task`, a retrieval task rides each of its SUB-task names instead
    # (`runner._cell`) -- one shared namespace. Disjoint today (task-5 report, concern
    # 4); nothing else enforces it, so a future ppl/latency YAML named like a retrieval
    # sub-task would silently pool two cells' seconds under one key.
    tasks = [load_task(t.stem) for t in (config.ROOT / "tasks").glob("*.yaml")]
    axis_names = {t.name for t in tasks if t.generator in ("ppl", "latency")}
    assert axis_names.isdisjoint({sub for t in tasks for sub in t.tasks})


def test_a_ppl_task_asking_for_more_windows_than_the_corpus_holds_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`frontier.windows` cuts NON-overlapping ctx+window spans, so a corpus holds a
    fixed number of them and `scripts/pod.py check` demands exactly `n_samples` of them.
    Asking for more is a pod that cannot pass its own gate -- refused at load, naming
    the file, rather than after the hours of GPU time it takes to find out."""
    monkeypatch.setitem(CORPUS_TOKENS, "wikitext-103-test", 300_000)  # span 18432 -> 15
    (tmp_path / "tasks").mkdir()
    head = "generator: ppl\nctx: 16384\nwindow: 2048\ncorpus: wikitext-103-test\n"
    (tmp_path / "tasks" / "greedy.yaml").write_text(f"name: greedy\n{head}n_samples: 16\n")
    (tmp_path / "tasks" / "fits.yaml").write_text(f"name: fits\n{head}n_samples: 15\n")
    monkeypatch.setattr("kvdlra.eval.config.ROOT", tmp_path)
    with pytest.raises(ValueError, match=r"greedy\.yaml") as e:
        load_task("greedy")
    assert "n_samples=16" in str(e.value) and "15" in str(e.value)
    assert load_task("fits").n_samples == 15  # the ceiling itself loads


def test_a_ppl_task_on_an_unmeasured_corpus_is_not_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is a lookup, not an estimate: v1's `wikitext-103` is not in the table
    (it is capped by `load_corpus_ids(max_tokens=...)`, not by the corpus), so the
    archived task configs load exactly as before."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "v1.yaml").write_text(
        "name: v1\ngenerator: ppl\nctx: 16384\nwindow: 512\nn_samples: 4\n"
    )
    monkeypatch.setattr("kvdlra.eval.config.ROOT", tmp_path)
    t = load_task("v1")
    assert (t.corpus, t.n_samples) == ("wikitext-103", 4)


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_the_shipped_held_out_task_configs_sit_inside_their_corpus_ceiling(name: str) -> None:
    corpus, ctx, n_samples = SHIPPED[name]
    t = load_task(name)  # the guard runs here; a bad n_samples never reaches the assert
    assert (t.generator, t.corpus, t.ctx, t.window) == ("ppl", corpus, ctx, 2048)
    assert t.n_samples == n_samples
    span = ctx + 2048
    assert n_samples <= (CORPUS_TOKENS[corpus] - span) // span


def test_the_pplw_line_round_trips_its_corpus_and_parses_without_one() -> None:
    """The emitter appends ` corpus=` LAST and `records.PPLW_RE` takes it as an optional
    trailing group, so the archived paper-v1 logs -- which have no such field -- keep
    parsing into rows whose corpus is simply unknown."""
    line = "[pplw] T=16384 bug-r64 ntok=2048 nlls=1.0,2.0 corpus=pg19-val"
    new = parse_pplw_lines(line, "tiny", "log")
    assert [r["corpus"] for r in new] == ["pg19-val", "pg19-val"]
    assert [r["nll_sum_nats"] for r in new] == [2048.0, 4096.0]
    archived = parse_pplw_lines(line.rsplit(" corpus=", 1)[0], "tiny", "log")
    assert [r["corpus"] for r in archived] == [None, None]
    assert [r["nll_sum_nats"] for r in archived] == [2048.0, 4096.0]


def test_the_emitter_prints_a_corpus_every_parser_run_recovers(capsys: Any) -> None:
    """Emitter and parser pinned together, including the ``part=i/N`` split a long
    sweep takes (`vastai logs` truncates at ~500 chars)."""
    n = 48  # 48 values is past the 400-char split threshold; 32 still fits on one line
    frontier._log_pplw(32768, "bug-r64", [0.5 + i / 100 for i in range(n)], NTOK, "pg19-val")
    text = capsys.readouterr().out
    assert " part=1/6 " in text  # split, and the corpus is repeated on every fragment
    assert text.count(" corpus=pg19-val") == len(text.splitlines()) == 6
    assert all(len(ln) <= 400 for ln in text.splitlines())  # the tail must not re-break it
    rows = parse_pplw_lines(text, "tiny", "log")
    assert len(rows) == n
    assert {r["corpus"] for r in rows} == {"pg19-val"}
    assert rows[0]["nll_sum_nats"] == pytest.approx(0.5 * NTOK)


def test_the_held_out_corpora_are_loadable_names() -> None:
    """`CORPUS_TOKENS` names corpora `data.load_corpus_ids` actually knows: a typo here
    would pass every offline test and fail on the pod after the model is loaded."""
    src = (Path(config.__file__).parent / "data.py").read_text()
    for corpus in CORPUS_TOKENS:
        assert f'"{corpus}"' in src
