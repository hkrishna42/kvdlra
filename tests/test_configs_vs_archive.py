"""Does each pod config still describe the run archived under `results/paper-v1/`?

A config is a claim about what a re-run evaluates. The v1 pods drifted from that claim
in ways nobody wrote down -- an arm that produced no rows, a cell that ran 16 records
instead of 12, two generators whose sub-task names collide -- and a silent drift is how
a stranger re-running a pod ends up comparing against something else.

So: the expected cell set is computed from the config by `pod._expected_cells` (the very
function `scripts/pod.py check` uses -- never a second copy of the rule), compared with
the archived `trials.jsonl` counts, and every difference must be named in
`KNOWN_DIVERGENCES` with a reason. The allowlist cannot rot: an entry whose pod no
longer diverges fails too.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pod

from kvdlra.eval.config import load_pod

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "paper-v1"

# config name -> the archive directory holding that pod's per-trial records. The two
# w17_floor pods and w19_ss2's own name have no per-trial archive of their own (w17 was
# a perplexity sweep; the ss2 control's rows were harvested into the sysfix directory).
ARCHIVE_DIR = {
    "w18_g1": "w18-g1-llama",
    "w18_g1_mistral": "w18-g1-mistral",
    "w18_g1_qwen": "w18-g1-qwen",
    "w18_g2_qwen": "w18-g2-qwen",
    "w18_g3": "w18-g3-llama",
    "w18_g3_mistral": "w18-g3-mistral",
    "w18_g3_qwen": "w18-g3-qwen",
    "w18_g4": "w18-g4-llama",
    "w19_a1": "w19-a1-llama",
    "w19_a1_mistral": "w19-a1-mistral",
    "w19_a1_qwen": "w19-a1-qwen",
    "w19_a1q": "w19-a1q-llama",
    "w19_a1q_mistral": "w19-a1q-mistral",
    "w19_a1q_qwen": "w19-a1q-qwen",
    "w19_a2": "w19-a2-llama",
    "w19_a4": "w19-a4-llama",
    "w19_fork": "w19-fork-llama",
    "w19_fork_mistral": "w19-fork-mistral",
    "w19_fork_qwen": "w19-fork-qwen",
    "w19_q4off": "w19-q4off-llama",
    "w19_ss2": "w19-sysfix-llama",
    "w19_swap": "w19-swap-llama",
}

# Every pod whose archived rows are NOT what its config calls for, and why. Each reason
# is also a sentence in the pod YAML's own `doc:` -- this is the machine-checked half.
KNOWN_DIVERGENCES = {
    "w18_g1": "the two quant arms produced no rows (-runtime image, quanto could not JIT"
    " its kernel), and the archive holds an unseeded bugS-r64-h256-q4 cell that has no config",
    "w18_g1_mistral": "same as w18_g1: no quant rows, plus the unconfigured q4 cell",
    "w18_g1_qwen": "same as w18_g1: no quant rows, plus the unconfigured q4 cell",
    "w18_g2_qwen": "the r64 arm ran niah_single only (the pinned depth grid), so its other"
    " three sub-tasks hold no rows",
    "w18_g4": "three arms ran 16 records per cell (8 trials x 2 seeds), not the task's 12,"
    " and the full arm produced no RULER rows",
    "w19_a1": "the 8-bit control ran 4 records at 16K only",
    "w19_a1_mistral": "the 8-bit control ran 4 records at 16K only",
    "w19_a1_qwen": "the 8-bit control ran 4 records at 16K only",
    "w19_a1q_qwen": "only the 16K half of the pod is archived; the 32K cells hold no rows",
    "w19_a4": "the r64 cells ran 8 records (4 trials x 2 seeds) and the full arm produced"
    " no RULER rows",
    "w19_fork": "the in-house and official 16K tasks share the sub-task names"
    " niah_multivalue and vt, so those keys pool both generators (21-24 records)",
    "w19_ss2": "the archived rows are keyed quant-2bit-kivi; the config's #chunk0 legacy"
    " name exists only to keep the single-shot arm distinguishable from w19_a1's",
    "w19_swap": "the FD arm crashed and produced no rows",
}


def _drift(name: str) -> list[str]:
    """Every (arm, sub-task, ctx) where the archive disagrees with the config."""
    d = ARCHIVE / ARCHIVE_DIR[name]
    assert d.is_dir(), f"{name}: no archive directory {d}"
    trials = d / "trials.jsonl"
    rows = [json.loads(x) for x in trials.read_text().splitlines() if x.strip()]
    counts = Counter((r["arm"], r["task"], r["ctx"]) for r in rows)
    expect = pod._expected_cells(load_pod(name))
    out = []
    for key in sorted(expect.keys() | counts.keys()):
        want, got = expect.get(key), counts.get(key, 0)
        if want is None:
            out.append(f"{key}: {got} archived rows are no cell of this config")
        elif got != want[0]:
            out.append(f"{key}: {got} archived rows, config calls for {want[0]}")
    return out


def test_no_undocumented_drift_between_a_config_and_its_archive() -> None:
    undocumented = {p: _drift(p) for p in ARCHIVE_DIR if p not in KNOWN_DIVERGENCES and _drift(p)}
    assert undocumented == {}, (
        "these pods no longer describe their archived run; fix the config, or add the"
        f" pod to KNOWN_DIVERGENCES with a reason and a sentence in its own doc: {undocumented}"
    )


def test_no_allowlisted_divergence_has_gone_stale() -> None:
    """An entry that no longer diverges is a stale excuse; delete it (and the sentence
    in the pod's doc) rather than let the allowlist grow into a rubber stamp."""
    assert [p for p in KNOWN_DIVERGENCES if not _drift(p)] == []
