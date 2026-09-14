# L0 — Repo Cleanup and Reproducibility: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stranger with one GPU regenerates every paper-v1 table from a config and a committed log; nothing unreachable from the four entrypoints + tests exists; the existing scaffolding (uv, CI, pre-commit, omegaconf/hydra, vast.ai boot + watchdog) is reused, not replaced.

**Architecture:** Two phases. **Phase A (Day 2, the skeleton other lanes wait on):** archive tag + paper-v1 per-trial logs → `results/paper-v1/<pod>/trials.jsonl`; one statistics module; `scripts/tables.py` regenerating v1 Tables 1, 2, 3, 6, 7, 8 diff-clean against a golden file; YAML arm/task/pod configs with a parity test against the legacy arm builder; `scripts/{pod,figures,dump_kv}.py` skeletons; root `Makefile`; `Dockerfile`. Merge to `week7` after Phase A so L1/L2/L4/L5 can start. **Phase B (Day 4):** source-tree moves and dead-path removal behind a bit-identity golden test, eval scripts folded into `kvdlra/eval/*` driven by configs, pod scripts thinned to boot + watchdog, deletions with evidence, dependency prune, README, CI, ponytail after-report.

**Tech Stack:** Python 3.12, torch 2.11 CPU in CI, omegaconf (already a dependency via hydra-core), scipy ≥ 1.13, pytest, ruff, mypy `--strict`, GNU make, vast.ai CLI (unchanged shell boot/watchdog).

**Spec:** `docs/plan/lanes/L0_repo_cleanup_reproducibility.md` (the lane brief), `CLAUDE.md` (repo rules, target layout, forbidden words), `docs/plan/lanes/GATES.md` §G0 (definition of done). Evidence inputs: `docs/plan/cleanup/ponytail-audit.md`, `docs/plan/cleanup/reachability.md`.

## Global Constraints

- Branch `lane/L0-repo-cleanup` in worktree `.claude/worktrees/L0-cleanup`, created off `week7` at execution time (`git worktree add .claude/worktrees/L0-cleanup -b lane/L0-repo-cleanup week7`). Never commit on `week7` directly; never push.
- Forbidden words in new code, configs, docs/paper, commit messages: `DLRA` (as a word), `BUG integrator`, `honest`/`honestly`, `marquee`, `flagship`. The identifier `kvdlra` and the directory name `kv-dlra` are not violations. Existing class names (`BugStreamingCache`, `BUGPress`) are not renamed in L0.
- Never delete a path that produced a number in a paper-v1 table before `git tag paper-v1-archive` exists (it exists: → ee8c0ab, created 2026-09-11). `results/w18_pertrial/`, `results/w19_pertrial/` and every `results/*-lines.txt` are converted (Task 1) before their originals are removed (Task 9).
- Ponytail full: reuse what exists, shortest working diff, delete over add. Never remove trust-boundary validation, data-loss guards, provenance, or tests of surviving code.
- Numeric fidelity: the r64 configuration `bugSseed-r64-h256` must stay bit-identical on CPU through every source move (Task 1 records the golden checksum before any `src/` change; every later task must keep `tests/test_golden_cache.py` green).
- `make test` runs the whole CPU suite in < 90 s in CI (baseline 71 s local). Tests that take > 5 s get `@pytest.mark.slow` only if they exceed the budget; do not delete tests of surviving code.
- Seeds explicit; a trial that raises is recorded as `error` and counted, never dropped (`SKIP` rows are a bug — Task 8 fixes the eval loop).
- One commit per logical change; commit messages carry the task id (`L0.<n>`). Pre-commit hooks (ruff, mypy strict, EOF fixer) run on every commit — if a hook rewrites files, `git add` and commit again.
- Every table/number produced by `make tables` must come from a committed `results/paper-v1/<pod>/trials.jsonl` (or `cells.jsonl` for aggregate-only sources); nothing is typed in by hand except the golden file, which is transcribed from `paper/main.tex` at ee8c0ab and checked against the output.

---

## File map (what exists after Phase B)

```
Makefile                          env · test · tables · figures · check · clean
Dockerfile                        pod image (pytorch 2.11 cuda12.8 devel + the pinned set + ninja)
pyproject.toml                    deps pruned (Task 9); ninja added to dev
configs/arms/*.yaml               one per arm (Task 4)
configs/tasks/*.yaml              generator × ctx × n × seeds × filler × depths (Task 4)
configs/pods/*.yaml               model × arms × tasks × prereg × budget (Task 4)
prereg/                           (owned by L5; L0 creates the dir with a README line)
scripts/pod.py                    run | launch | harvest | check      (Task 5, filled in Task 8)
scripts/tables.py                 convert-v1 | build                  (Tasks 1, 3)
scripts/figures.py                build                               (Task 5, folded from w19_figures)
scripts/dump_kv.py                dump | verify                       (Task 5, from capture_kv)
scripts/pod/boot.sh               = w18_boot.sh, sources nothing: runs `python scripts/pod.py run --pod $POD`
scripts/pod/watchdog.sh           = w19_watchdog.sh incl. the 5000-line fallback, pod-name driven
scripts/_paths.py                 kept (src on sys.path for `python scripts/x.py`)
src/kvdlra/tracker/isvd.py        = integrators/streaming_torch.py (augmented step + oja_step + fd_step)
src/kvdlra/cache/bug_cache.py     kept; dead paths removed (Task 7)
src/kvdlra/cache/shadow_cache.py  kept (ShadowKV baseline)
src/kvdlra/baselines/svd_oracle.py   = press/palu_press.py (renamed class SVDOraclePress)
src/kvdlra/baselines/lowrank_press.py = press/bug_press.py (base press for the oracle; numpy backend removed)
src/kvdlra/baselines/compat.py    = press/compat.py
src/kvdlra/quant/{kivi_cache,polar}.py   kept
src/kvdlra/accounting.py          kept
src/kvdlra/util/seed.py           = utils/seed.py
src/kvdlra/eval/{config,stats,records,ruler,frontier,official_ruler,longbench,latency,storage,persist}.py
tests/                            + test_stats.py, test_records.py, test_tables_golden.py, test_config_parity.py,
                                    test_golden_cache.py, test_pod_manifest.py, test_reachability.py,
                                    test_forbidden_words.py; minus the tests of deleted modules
results/paper-v1/<pod>/{trials.jsonl|cells.jsonl,manifest.json}
docs/plan/paper-v1-tables.md      golden output of `make tables`
docs/plan/cleanup/{deletions.md,ponytail-after.md}
docs/plan/reviews/                = docs/reviews (moved)
README.md                         ≤ 120 lines
```

Deleted (Task 6/7/9, each path listed with evidence in `docs/plan/cleanup/deletions.md`): `src/kvdlra/integrators/{bug,bug_adaptive,bug_class,bug_torch,streaming,oja,frequent_directions,streaming_variants}.py` and the package, `src/kvdlra/lowrank.py`, `src/kvdlra/quant/{product_quant,qjl}.py`, `src/kvdlra/press/turbo_press.py`, `src/kvdlra/utils/hydra_resolvers.py`, `src/kvdlra/cache/morph_cache.py` (unless `paper/main.tex` cites a MorphKV number — check in Task 6), every `scripts/*.py` not in the file map, every `scripts/pod/*.sh` except `boot.sh`/`watchdog.sh`, `experiments/`, `figs/`, `figures/**` not included by `paper/main.tex`, `mkdocs.yml`, `docs/*` outside `docs/{plan,adr,paper}` (reviews moved, not deleted), the 11 test files of deleted modules, the gitignored litter on disk.

---

## Phase A — the skeleton (merge to week7 when Tasks 1–5 are green)

### Task 1: Archive the paper-v1 evidence and freeze a bit-identity golden

**Files:**
- Create: `src/kvdlra/eval/__init__.py` (empty), `src/kvdlra/eval/records.py`
- Create: `scripts/tables.py` (only the `convert-v1` subcommand in this task)
- Create: `results/paper-v1/<pod>/trials.jsonl` + `manifest.json` for every per-trial source; `cells.jsonl` for aggregate-only sources
- Create: `tests/test_records.py`, `tests/test_golden_cache.py`, `tests/golden/bug_cache_r64_cpu.json`
- Test: `tests/test_records.py`, `tests/test_golden_cache.py`

**Interfaces:**
- Produces: `kvdlra.eval.records.TrialRecord` (a `TypedDict`) with keys `model, arm, task, ctx, seed, trial, hit, frac, haystack_id, depth, prompt_sha256, error, source`; `parse_trial_lines(text: str, model: str, source: str) -> list[TrialRecord]`; `parse_cell_lines(text, model, source) -> list[CellRecord]` (`CellRecord`: `model, arm, task, ctx, acc, n, hits, ratio, source`); `write_jsonl(path, rows)`, `read_jsonl(path)`.
- Produces: `results/paper-v1/` naming: `w18-g1-{llama,mistral,qwen}`, `w18-g2-qwen`, `w18-g3-{llama,mistral,qwen}`, `w18-g4-llama`, `w18-g5-llama`, `w19-a1-{llama,mistral,qwen}`, `w19-a1diag-qwen`, `w19-a1q-{llama,mistral,qwen}`, `w19-a2-llama`, `w19-a3-llama`, `w19-a3-llama2`, `w19-a4-llama`, `w19-fork-{llama,mistral,qwen}`, `w19-forkdiag-qwen`, `w19-q4off-llama`, `w19-swap-llama`, `w19-sysfix-llama`; aggregate-only: `w15-confirm`, `w15-complete`, `w17-llama8b`, `w17-qwen`, and every other `results/*-lines.txt` (one pod dir per file, name = the file stem without `-lines`).

- [ ] **Step 1: Confirm the tag and the model-id mapping**

Run:
```bash
git rev-parse --short 'paper-v1-archive^{commit}'          # expect ee8c0ab
grep -n -E 'MODEL=|Llama-3.1|Mistral-7B|Qwen2.5' scripts/pod/w18.sh scripts/pod/w19.sh docs/week18-kickoff.md docs/week19-kickoff.md | head -20
```
Record the three exact HF ids for `llama`, `mistral`, `qwen` in `MODEL_BY_TAG` below. If an id cannot be found verbatim in the pod scripts or kickoff docs, stop and report — never guess.

- [ ] **Step 2: Write the failing record tests**

`tests/test_records.py`:
```python
from pathlib import Path

from kvdlra.eval.records import parse_cell_lines, parse_trial_lines, read_jsonl, write_jsonl

TRIAL = "[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=1 trial=3 hit=1 frac=1.000\n"
CELL = "[niah_single ctx16384] bugSseed-r64-h256 acc=1.000 recall=1.000 ratio=0.151 sbits=0.151 n=12\n"


def test_parse_trial_lines_schema() -> None:
    rows = parse_trial_lines(TRIAL, model="M", source="f.txt")
    assert rows == [
        {
            "model": "M", "arm": "bugSseed-r64-h256", "task": "niah_single", "ctx": 16384,
            "seed": 1, "trial": 3, "hit": 1, "frac": 1.0,
            "haystack_id": None, "depth": None, "prompt_sha256": None, "error": None,
            "source": "f.txt:1",
        }
    ]


def test_parse_cell_lines_recovers_hits() -> None:
    rows = parse_cell_lines(CELL, model="M", source="f.txt")
    assert rows[0]["hits"] == 12 and rows[0]["n"] == 12 and rows[0]["ratio"] == 0.151


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = parse_trial_lines(TRIAL, model="M", source="f.txt")
    write_jsonl(tmp_path / "t.jsonl", rows)
    assert read_jsonl(tmp_path / "t.jsonl") == rows


def test_v1_archive_counts_match_sources() -> None:
    # every archived pod has exactly as many records as [trial] lines in its source
    root = Path("results/paper-v1")
    assert (root / "w18-g1-llama/trials.jsonl").exists()
    assert len(read_jsonl(root / "w18-g1-llama/trials.jsonl")) == 192
    assert len(read_jsonl(root / "w19-a2-llama/trials.jsonl")) == 756
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_records.py -q`
Expected: FAIL with `ModuleNotFoundError: kvdlra.eval`

- [ ] **Step 4: Implement `records.py`**

`src/kvdlra/eval/records.py`:
```python
"""Per-trial and per-cell records: the only unit `make tables` reads.

`[trial]` lines are the Bernoulli outcomes the pods printed (one per task × ctx × arm ×
seed × trial); `[task ctxN] arm acc= ... n=` lines are pooled cells. Both regexes are the
emitters' formats from Week 18/19 (previously duplicated in ten reader scripts)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

TRIAL_RE = re.compile(
    r"^\[trial\] task=(\S+) ctx=(\d+) arm=(\S+) seed=(\d+) trial=(\d+) hit=([01]) frac=([0-9.]+)"
)
CELL_RE = re.compile(
    r"^\[([A-Za-z0-9_]+) ctx(\d+)\] (\S+)\s+acc=([0-9.]+) recall=([0-9.]+) ratio=([0-9.]+)"
    r"(?: sbits=([0-9.]+))?(?: n=(\d+))?"
)


class TrialRecord(TypedDict):
    model: str
    arm: str
    task: str
    ctx: int
    seed: int
    trial: int
    hit: int
    frac: float
    haystack_id: str | None
    depth: float | None
    prompt_sha256: str | None
    error: str | None
    source: str


class CellRecord(TypedDict):
    model: str
    arm: str
    task: str
    ctx: int
    acc: float
    n: int | None
    hits: int | None
    ratio: float
    source: str


def parse_trial_lines(text: str, model: str, source: str) -> list[TrialRecord]:
    out: list[TrialRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = TRIAL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, seed, trial, hit, frac = m.groups()
        out.append(
            {
                "model": model, "arm": arm, "task": task, "ctx": int(ctx),
                "seed": int(seed), "trial": int(trial), "hit": int(hit), "frac": float(frac),
                "haystack_id": None, "depth": None, "prompt_sha256": None, "error": None,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_cell_lines(text: str, model: str, source: str) -> list[CellRecord]:
    out: list[CellRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = CELL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, acc, _recall, ratio, _sbits, n = m.groups()
        n_i = int(n) if n is not None else None
        hits = round(float(acc) * n_i) if n_i is not None else None
        out.append(
            {
                "model": model, "arm": arm, "task": task, "ctx": int(ctx),
                "acc": float(acc), "n": n_i, "hits": hits, "ratio": float(ratio),
                "source": f"{source}:{i}",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[TrialRecord] | list[CellRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
```

- [ ] **Step 5: Write the converter (`scripts/tables.py convert-v1`)**

`scripts/tables.py` (first version; `build` is added in Task 3):
```python
"""Tables entrypoint. `convert-v1` archives the paper-v1 line files as JSONL records;
`build` (Task 3) regenerates every paper-v1 table from results/paper-v1/."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _paths  # noqa: F401
from kvdlra.eval.records import parse_cell_lines, parse_trial_lines, write_jsonl

MODEL_BY_TAG = {  # exact HF ids the pods used (Step 1) -- fill from the pod scripts
    "llama": "<from Step 1>",
    "mistral": "<from Step 1>",
    "qwen": "<from Step 1>",
}
PERTRIAL = {  # results/<dir>/<file>-trials.txt -> pod name
    "results/w18_pertrial": "w18-g1-{tag}",          # llama/mistral/qwen-trials.txt
    "results/w19_pertrial": "w19-{stem}",           # a1-llama-trials.txt -> w19-a1-llama
}


def _tag(stem: str) -> str:
    for t in MODEL_BY_TAG:
        if t in stem:
            return t
    raise SystemExit(f"no model tag in {stem}")


def convert_v1(out_root: Path) -> None:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "paper-v1-archive^{commit}"], text=True).strip()
    for src in sorted(Path("results/w18_pertrial").glob("*-trials.txt")):
        stem = src.name.removesuffix("-trials.txt")
        pod = f"w18-g1-{stem}" if stem in MODEL_BY_TAG else f"w18-{stem}"
        _emit(out_root / pod, [src], sha, per_trial=True)
    for src in sorted(Path("results/w19_pertrial").glob("*-trials.txt")):
        _emit(out_root / f"w19-{src.name.removesuffix('-trials.txt')}", [src], sha, per_trial=True)
    for src in sorted(Path("results").glob("*-lines.txt")):
        _emit(out_root / src.name.removesuffix("-lines.txt"), [src], sha, per_trial=False)


def _emit(pod_dir: Path, sources: list[Path], sha: str, per_trial: bool) -> None:
    rows: list = []
    for s in sources:
        model = MODEL_BY_TAG[_tag(s.stem)] if any(t in s.stem for t in MODEL_BY_TAG) else "unknown"
        text = s.read_text()
        rows += parse_trial_lines(text, model, str(s)) if per_trial else parse_cell_lines(text, model, str(s))
    if not rows:
        return
    write_jsonl(pod_dir / ("trials.jsonl" if per_trial else "cells.jsonl"), rows)
    (pod_dir / "manifest.json").write_text(json.dumps({
        "pod": pod_dir.name, "git_sha": sha, "source_files": [str(s) for s in sources],
        "converted_by": "scripts/tables.py convert-v1", "records": len(rows),
        "note": "paper-v1 archive; haystack_id/depth/prompt_sha256 were not recorded and are null",
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert-v1"); c.add_argument("--out", default="results/paper-v1")
    a = ap.parse_args()
    if a.cmd == "convert-v1":
        convert_v1(Path(a.out))


if __name__ == "__main__":
    main()
```
Lines where a file's model is `"unknown"` (e.g. `w11-*-lines.txt` mixed pods) are acceptable for `cells.jsonl`; a `trials.jsonl` with `"unknown"` is not — `_emit` must raise for per-trial sources whose tag is missing.

- [ ] **Step 6: Run the converter, run the tests**

Run:
```bash
.venv/bin/python scripts/tables.py convert-v1
ls results/paper-v1 | wc -l
.venv/bin/python -m pytest tests/test_records.py -q
```
Expected: ≥ 30 pod dirs; tests PASS. Spot-check: `grep -c '^\[trial\]' results/w19_pertrial/a2-llama-trials.txt` equals `wc -l < results/paper-v1/w19-a2-llama/trials.jsonl`.

- [ ] **Step 7: Freeze the bit-identity golden BEFORE any src change**

`tests/test_golden_cache.py`:
```python
"""Bit-identity tripwire for the r64 configuration across the cleanup.
The golden was generated at paper-v1-archive (ee8c0ab) on CPU; every source move must
leave it unchanged. Regenerate ONLY with `--regen` and a DECISIONS.md entry."""

import json
from pathlib import Path

import pytest
import torch

GOLDEN = Path(__file__).parent / "golden" / "bug_cache_r64_cpu.json"


def _run() -> dict[str, float]:
    from tests.test_bug_cache import make_tiny_model_and_cache  # reuse the suite's fixture builder

    torch.manual_seed(0)
    model, cache = make_tiny_model_and_cache(rank=64, hh_budget=256, seed_hh_warmup=True)
    # 4 chunks of 512 random tokens, then reconstruct the middle of every layer
    stream = torch.randn(4, 512, model.n, generator=torch.Generator().manual_seed(1))
    for blk in stream:
        cache.ingest(blk)
    k_hat, v_hat = cache.reconstruct_middle()
    return {"k_sum": float(k_hat.double().sum()), "k_abs": float(k_hat.double().abs().sum()),
            "v_sum": float(v_hat.double().sum()), "rank": float(cache.layers[0].u_k.shape[1])}


def test_r64_cpu_bit_identity() -> None:
    got = _run()
    want = json.loads(GOLDEN.read_text())
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=0, abs=1e-6), k
```
The fixture names above (`make_tiny_model_and_cache`, `ingest`, `reconstruct_middle`, `cache.layers`) are placeholders for the *existing* helpers in `tests/test_bug_cache.py` and the *existing* public methods of `BugStreamingCache` — read `tests/test_bug_cache.py:1-120` and `src/kvdlra/cache/bug_cache.py` (`_prefill`, `_ensure_mid_cache`, `consolidate`) and use the real names; the test must exercise chunked ingest with the seed and the surprise tier (the r64 configuration's path), on CPU, in < 5 s. Generate the golden once:
```bash
.venv/bin/python - <<'PY'
import json, pathlib, sys; sys.path[:0] = ["src", "."]
from tests.test_golden_cache import _run
pathlib.Path("tests/golden").mkdir(exist_ok=True)
pathlib.Path("tests/golden/bug_cache_r64_cpu.json").write_text(json.dumps(_run(), indent=2) + "\n")
PY
.venv/bin/python -m pytest tests/test_golden_cache.py -q
```
Expected: PASS, and a second run gives identical values (determinism check).

- [ ] **Step 8: Commit**

```bash
git add src/kvdlra/eval scripts/tables.py results/paper-v1 tests/test_records.py tests/test_golden_cache.py tests/golden
git commit -m "L0.1: archive paper-v1 per-trial logs as results/paper-v1/<pod>/{trials,cells}.jsonl; records module; r64 CPU bit-identity golden"
```

### Task 2: One statistics module

**Files:**
- Create: `src/kvdlra/eval/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces: `wilson(hits: int, n: int, conf: float = 0.95) -> tuple[float, float]`; `mcnemar_exact(a: dict[tuple[int,int], int], b: dict[tuple[int,int], int]) -> McNemar | None` (`McNemar` = `TypedDict(n_paired, a_favored, b_favored, p_value)`); `holm(p: Sequence[float]) -> list[float]` (adjusted p, monotone); `bh(p) -> list[float]`; `paired_bootstrap(d: Sequence[float], n_boot: int = 10_000, seed: int = 0, conf: float = 0.95) -> tuple[float, float, float]` (mean, lo, hi); `tost(d: Sequence[float], delta: float, alpha: float = 0.05) -> tuple[float, float, bool]` (p_lower, p_upper, equivalent).

- [ ] **Step 1: Write the failing tests** (values from CODE_AUDIT Part C, recomputed independently there)

`tests/test_stats.py`:
```python
import numpy as np
import pytest

from kvdlra.eval.stats import bh, holm, mcnemar_exact, paired_bootstrap, tost, wilson


def test_wilson_12_of_12_lower_bound() -> None:
    lo, hi = wilson(12, 12)
    assert lo == pytest.approx(0.758, abs=5e-4) and hi == 1.0


def test_wilson_rejects_bad_cells() -> None:
    with pytest.raises(ValueError):
        wilson(13, 12)


def test_mcnemar_10_0_and_6_0() -> None:
    a = {(s, t): 1 for s in (0, 1) for t in range(8)}          # 16/16
    b = dict(a); b.update({k: 0 for k in list(a)[:10]})           # 6/16, all discordant one way
    r = mcnemar_exact(a, b)
    assert r is not None and r["a_favored"] == 10 and r["b_favored"] == 0
    assert r["p_value"] == pytest.approx(1.95e-3, rel=1e-2)
    b6 = dict(a); b6.update({k: 0 for k in list(a)[:6]})
    assert mcnemar_exact(a, b6)["p_value"] == pytest.approx(3.13e-2, rel=1e-2)  # type: ignore[index]


def test_mcnemar_no_shared_trials_is_none() -> None:
    assert mcnemar_exact({(0, 0): 1}, {(1, 0): 1}) is None


def test_holm_and_bh_known_vectors() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])  # adjusted (cummin) values; R5 corrected the plan


def test_paired_bootstrap_symmetric_contains_zero() -> None:
    d = np.random.default_rng(0).normal(0.0, 1.0, 64)
    mean, lo, hi = paired_bootstrap(d, n_boot=2000, seed=0)
    assert lo < 0.0 < hi and abs(mean - d.mean()) < 1e-12


def test_tost_equivalence() -> None:
    rng = np.random.default_rng(1)
    tight = rng.normal(0.0, 0.01, 32)
    p_lo, p_hi, ok = tost(tight, delta=0.05)
    assert ok and max(p_lo, p_hi) < 0.05
    shifted = tight + 0.10
    assert not tost(shifted, delta=0.05)[2]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_stats.py -q` — Expected: FAIL, `No module named kvdlra.eval.stats`.

- [ ] **Step 3: Implement**

`src/kvdlra/eval/stats.py`:
```python
"""The single statistics module. Wilson + exact McNemar are the paper's verified tools
(ported from scripts/w15_intervals.py and w18_intervals.py); Holm/BH/bootstrap/TOST are
the corrections the review panel asked for. scipy does the arithmetic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from scipy.stats import binomtest, false_discovery_control, ttest_1samp

Key = tuple[int, int]  # (seed, trial)


class McNemar(TypedDict):
    n_paired: int
    a_favored: int
    b_favored: int
    p_value: float


def wilson(hits: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0 or not 0 <= hits <= n:
        raise ValueError(f"bad cell: hits={hits} n={n}")
    ci = binomtest(hits, n).proportion_ci(confidence_level=conf, method="wilson")
    return float(ci.low), float(ci.high)


def mcnemar_exact(a: dict[Key, int], b: dict[Key, int]) -> McNemar | None:
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    a_f = sum(1 for k in shared if a[k] == 1 and b[k] == 0)
    b_f = sum(1 for k in shared if a[k] == 0 and b[k] == 1)
    p = 1.0 if a_f + b_f == 0 else float(binomtest(min(a_f, b_f), a_f + b_f, 0.5).pvalue)
    return {"n_paired": len(shared), "a_favored": a_f, "b_favored": b_f, "p_value": p}


def holm(p: Sequence[float]) -> list[float]:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def bh(p: Sequence[float]) -> list[float]:
    return false_discovery_control(np.asarray(p, dtype=float), method="bh").tolist()


def paired_bootstrap(
    d: Sequence[float], n_boot: int = 10_000, seed: int = 0, conf: float = 0.95
) -> tuple[float, float, float]:
    x = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(x.mean()), float(lo), float(hi)


def tost(d: Sequence[float], delta: float, alpha: float = 0.05) -> tuple[float, float, bool]:
    x = np.asarray(d, dtype=float)
    p_lo = float(ttest_1samp(x, -delta, alternative="greater").pvalue)
    p_hi = float(ttest_1samp(x, delta, alternative="less").pvalue)
    return p_lo, p_hi, max(p_lo, p_hi) < alpha
```

- [ ] **Step 4: Run tests → PASS; mypy clean**

Run: `.venv/bin/python -m pytest tests/test_stats.py -q && .venv/bin/python -m mypy src/kvdlra/eval`

- [ ] **Step 5: Commit** — `git commit -m "L0.2: kvdlra.eval.stats — wilson/mcnemar (ported, pinned) + holm/bh/paired_bootstrap/tost"`

### Task 3: `make tables` regenerates paper-v1 Tables 1, 2, 3, 6, 7, 8 diff-clean

**Files:**
- Modify: `scripts/tables.py` (add `build`)
- Create: `Makefile` (targets `test`, `tables`; the rest in Task 5), `docs/plan/paper-v1-tables.md` (golden), `docs/paper/tables/.gitkeep`
- Test: `tests/test_tables_golden.py`

**Interfaces:**
- Consumes: `results/paper-v1/*/trials.jsonl` (Task 1), `kvdlra.eval.stats` (Task 2).
- Produces: `scripts/tables.py build --out docs/paper/tables` writing `table{1,2,3,6,7,8}.md` and `.tex`; `docs/plan/paper-v1-tables.md` = concatenation of the six `.md` files; `make tables` = build + `diff -u docs/plan/paper-v1-tables.md <(cat docs/paper/tables/table*.md)`.

Table specifications (from `paper/main.tex` at ee8c0ab — the implementer reads each table environment and transcribes its exact rows/columns into the golden file first):

| table | label | source pod(s) | rows | columns | stats |
|---|---|---|---|---|---|
| 1 | `tab:xmodel` | `w18-g1-{qwen,mistral,llama}` | arm `bugSseed-r64-h256`, ctx 16384, per model | single, multi-value, var-track (+ multi-key, shown even though the paper omitted it — mark the column "(not in v1)") | acc, Wilson 95%, hits/n |
| 2 | `tab:xmodel32` | same | ctx 32768 | same | same |
| 3 | `tab:marquee` | `w18-g4-llama` (r128-h1024-s32, think-c0.5, palu-r0.5 n=16; bugSseed-r256-h1024 n=12), `w19-a1-llama` (quant-4bit-kivi n=12) | ctx 32768, Llama | single, multi-key, multi-value, var-track | Wilson; McNemar vs the r128 arm on var-track (expect think 10/0 p=1.95e-3; palu 6/0 p=3.13e-2; 4-bit 0/1 on 12 shared) |
| 6 | `tab:evict` | `w18-g3-*` | arm `ea-k0.1`, the models × ctx the v1 table shows | 4 tasks | acc, Wilson |
| 7 | `tab:fairquant` | r64 rows from `w18-g1-*`, `quant-{2,4}bit-kivi` from `w19-a1-*` | 3 models × {16384, 32768} × 3 arms | 4 tasks | acc; bold = exact McNemar p<0.05 vs r64 (paired on seed,trial) |
| 8 | `tab:official` | `w19-a2-llama` | 7 arms | 9 official tasks + mean | acc (12 records/task) |

- [ ] **Step 1: Transcribe the golden** — create `docs/plan/paper-v1-tables.md` by hand from `git show ee8c0ab:paper/main.tex` (the six table environments), one markdown table per paper table, cells formatted exactly as `build` will print them (`acc` to 2 dp; Wilson `[lo,hi]` to 2 dp; `hits/n`; p-values to 2 significant figures). Where the paper printed a truncated Wilson upper bound (0.80 for 0.8067; 0.98 for 0.9851 — CODE_AUDIT Part C T1) write the correctly rounded value and add a footnote line `<!-- v1 printed 0.80 (truncated); correct rounding 0.81 -->`.

- [ ] **Step 2: Write the failing golden test**

`tests/test_tables_golden.py`:
```python
import subprocess
import sys
from pathlib import Path


def test_make_tables_matches_golden(tmp_path: Path) -> None:
    out = tmp_path / "tables"
    subprocess.run([sys.executable, "scripts/tables.py", "build", "--out", str(out)], check=True)
    got = "".join(p.read_text() for p in sorted(out.glob("table*.md")))
    assert got == Path("docs/plan/paper-v1-tables.md").read_text()
```

- [ ] **Step 3: Implement `build`** in `scripts/tables.py`: load every `results/paper-v1/*/trials.jsonl` into one list; helper `cell(rows, model, arm, task, ctx) -> (hits, n, wilson)`; helper `paired(rows, model, ctx, task, arm_a, arm_b) -> dict[Key,int] × 2 → mcnemar_exact`; six `table_N()` functions each returning `(md: str, tex: str)`; write both. Keep the Markdown format identical to the golden (that is the contract); the `.tex` mirrors it with `\begin{tabular}`. No hand-typed numbers anywhere in `tables.py`.

- [ ] **Step 4: Makefile (first targets)**

`Makefile` (tabs, not spaces):
```make
PY ?= .venv/bin/python

.PHONY: test tables
test:
	$(PY) -m pytest -q

tables:
	$(PY) scripts/tables.py build --out docs/paper/tables
	@cat docs/paper/tables/table*.md | diff -u docs/plan/paper-v1-tables.md - && echo "tables: diff-clean vs paper-v1"
```

- [ ] **Step 5: Run** `make tables` and `make test` — Expected: "tables: diff-clean vs paper-v1"; golden test PASS. If a cell disagrees with the paper: STOP, do not edit the golden to match — run `superpowers:systematic-debugging` (the audit reproduced every cell from these records, so a mismatch is a converter/table bug) and report in the task report.

- [ ] **Step 6: Commit** — `git commit -m "L0.3: make tables regenerates paper-v1 Tables 1,2,3,6,7,8 from results/paper-v1 (diff-clean golden)"`

### Task 4: YAML configs for every v1 arm/task/pod + parity test against the legacy arm builder

**Files:**
- Create: `configs/arms/*.yaml`, `configs/tasks/*.yaml`, `configs/pods/*.yaml`, `src/kvdlra/eval/config.py`
- Test: `tests/test_config_parity.py`

**Interfaces:**
- Produces: `load_arm(name) -> ArmCfg`, `load_task(name) -> TaskCfg`, `load_pod(name) -> PodCfg` (omegaconf `DictConfig`s validated against dataclasses `ArmCfg/TaskCfg/PodCfg` via `OmegaConf.structured`); `config_hash(pod: PodCfg) -> str` (sha256 of the canonical YAML dump of the pod + its arms + tasks); `arm_kwargs(arm: ArmCfg, t: int) -> dict` (the exact keyword set `BugStreamingCache(...)` receives, with `coord_budget = t + recent_window + absorb_block` resolved).
- Consumes: `scripts/w10_frontier.py:build_arms` (kept alive only until Task 8 deletes it) for the parity test.

Arm YAML schema (`configs/arms/isvd_r64_h256_seed.yaml` — the r64 configuration):
```yaml
name: isvd_r64_h256_seed
legacy_name: bugSseed-r64-h256     # the string in results/paper-v1 records
kind: bug                          # bug | full | press | quant | composite | shadow
chunkable: true
cache:
  rank: 64
  hh_budget: 256
  hh_select: surprise
  hh_neighbor: 1
  seed_hh_warmup: true
  retention: lowrank_surprise
  recent_window: 32
  absorb_block: 16
  n_sink: 4
  min_sv_frac: 0.0
  tracker: isvd                    # isvd | oja | fd   (isvd == the shipped step; legacy string "bug")
  score_rank: null
  quant_bits: null
```
Every v1 arm gets a file; the full list and the legacy ↔ new names: `full`, `isvd_r{64,128,256}` (`bug-r*`), `isvd_r256_f0.01` (`bug-r256-f0.01`), `isvd_r64_h256_seed` (`bugSseed-r64-h256`), `isvd_r64_h256_seed_q4` (`bugSseed-r64-h256-q4`), `isvd_r128_h1024_seed_s32` (`bugSseed-r128-h1024-s32`), `isvd_r256_h1024_seed` (`bugSseed-r256-h1024`), `oja_r64_h256_seed` / `fd_r64_h256_seed` (`bugSseed-r64-h256-oja` / `-fd`), `evict_surprise_h256` (`bugEVICT-h256`), `kivi2_streaming` / `kivi4_streaming` (`quant-2bit-kivi` / `quant-4bit-kivi`, G=64, chunk 4096), `kivi2_singleshot` (`quant-2bit-kivi` with chunk 0 — the ss2 arm; `legacy_name` carries a `#chunk0` suffix note in the file, records were keyed by pod not name), `kivi8_hqq` (`quant-8bit-kivi-hqq`), `think_c0.5` (`think-c0.5`), `svd_oracle_r0.5` (`palu-r0.5`; the YAML `doc:` field states: per-sequence per-head truncated SVD of the prefill K/V — an upper bound for static low-rank, NOT Palu), `ea_k{0.1,0.25,0.5}`, `snapkv_k0.1`, `shadowkv_r{64,128}`, `ea_k{0.1,0.25}_kivi{2,4}` (`ea-k0.1-q2-kivi` …). Task YAML: `ruler_inhouse_{16k,32k,64k}` (`generator: inhouse`, tasks, `n_trials: 6`, `seeds: [0,1]`, `filler: cycle`, `depths: null`, `chunk: 4096`), `ruler_inhouse_16k_wikitext` (`filler: wikitext`), `ruler_official_16k` (`generator: official_ruler`), `ppl_{16k,32k,64k}` (`generator: ppl`, `window: 512`, `n_samples: 4`), `longbench_v1`. Pod YAML for every v1 pod that fed a table (`w18_g1`, `w18_g3`, `w18_g4`, `w19_a1`, `w19_a2`, …): `model`, `dtype: bfloat16`, `arms: [...]`, `tasks: [...]`, `prereg: null` (v1 had none — say so), `gpu_budget_h`.

- [ ] **Step 1: Write the failing parity test**

`tests/test_config_parity.py`:
```python
"""Every v1 arm built from YAML must produce exactly the kwargs the legacy CLI path produced."""

import argparse
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from kvdlra.eval.config import arm_kwargs, config_hash, load_arm, load_pod

LEGACY = {  # legacy arm name -> the CLI flags that produced it (from scripts/pod/w18.sh, w19.sh)
    "bugSseed-r64-h256": "--methods bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed",
    "bugSseed-r128-h1024-s32": "--methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32",
    "bug-r256-f0.01": "--methods bug --ranks 256 --min-sv-frac 0.01",
}


@pytest.mark.parametrize("legacy", sorted(LEGACY))
def test_yaml_arm_matches_legacy_build_arms(legacy: str) -> None:
    import w10_frontier  # scripts/ is on pythonpath; deleted in Task 8 together with this import

    ns = w10_frontier.build_parser().parse_args(LEGACY[legacy].split() + ["--chunk", "4096"])
    legacy_arm = next(a for a in w10_frontier.build_arms(ns, model=None, t=16384) if a["name"] == legacy)
    yaml_name = next(p.stem for p in Path("configs/arms").glob("*.yaml")
                     if OmegaConf.load(p).get("legacy_name") == legacy)
    got = arm_kwargs(load_arm(yaml_name), t=16384)
    assert got == w10_frontier.arm_kwargs_for_test(legacy_arm)  # a 5-line helper added to w10_frontier that returns the kwargs its lambda passes


def test_every_arm_yaml_loads_and_names_match() -> None:
    for p in Path("configs/arms").glob("*.yaml"):
        assert load_arm(p.stem).name == p.stem


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    pod = load_pod("w18_g1")
    h1 = config_hash(pod)
    assert h1 == config_hash(load_pod("w18_g1"))
    pod.gpu_budget_h = pod.gpu_budget_h + 1
    assert config_hash(pod) != h1
```
`arm_kwargs_for_test` is the one legacy edit this task makes to `w10_frontier.py`: it inspects the `make` lambda's defaults/closure (`make.__defaults__` + `make.__closure__`) — or, simpler and robust, the task refactors each lambda body into `kwargs = {...}; make = lambda: BugStreamingCache(model, **kwargs)` and stores `"kwargs": kwargs` on the arm dict, which the test reads. Do the latter; it is a no-op for behaviour and the golden test (Task 1) proves it.

- [ ] **Step 2: Run → FAIL** (`kvdlra.eval.config` missing).

- [ ] **Step 3: Implement `config.py`**

```python
"""YAML configs (omegaconf) for arms, tasks, pods. Every experiment is a YAML; the CLI
flag soup of w10_ruler/w10_frontier is retired in Task 8."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class ArmCfg:
    name: str
    kind: str
    legacy_name: str | None = None
    chunkable: bool = True
    doc: str = ""
    cache: dict[str, Any] = field(default_factory=dict)   # BugStreamingCache kwargs (bug/evict)
    press: dict[str, Any] = field(default_factory=dict)   # kvpress / oracle press kwargs
    quant: dict[str, Any] = field(default_factory=dict)   # nbits, scheme, backend, group, residual, chunk


@dataclass
class TaskCfg:
    name: str
    generator: str                    # inhouse | official_ruler | longbench | ppl
    ctx: int
    tasks: list[str] = field(default_factory=list)
    n_trials: int = 6
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    filler: str = "cycle"
    depths: list[float] | None = None
    chunk: int = 4096
    window: int = 512
    n_samples: int = 4


@dataclass
class PodCfg:
    name: str
    model: str
    arms: list[str]
    tasks: list[str]
    dtype: str = "bfloat16"
    prereg: str | None = None
    gpu_budget_h: float = 0.0
    image: str = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel"


def _load(kind: str, name: str, schema: type) -> Any:
    p = ROOT / kind / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(p)
    cfg = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(p))
    return OmegaConf.to_object(cfg)


def load_arm(name: str) -> ArmCfg:
    return _load("arms", name, ArmCfg)  # type: ignore[no-any-return]


def load_task(name: str) -> TaskCfg:
    return _load("tasks", name, TaskCfg)  # type: ignore[no-any-return]


def load_pod(name: str) -> PodCfg:
    return _load("pods", name, PodCfg)  # type: ignore[no-any-return]


def arm_kwargs(arm: ArmCfg, t: int) -> dict[str, Any]:
    """The exact BugStreamingCache kwargs for context length t (bug / evict arms)."""
    kw = dict(arm.cache)
    rw, ab = kw.get("recent_window", 32), kw.get("absorb_block", 16)
    cb = t + rw + ab
    if kw.get("quant_bits") is not None:                     # q4 cell: coordinate tier quantized
        kw["coord_budget"], kw["quant_budget"] = int(kw.pop("quant_budget", 0) or 0), cb
    else:
        kw["coord_budget"], kw["quant_budget"] = cb, 0
    kw["tracker"] = {"isvd": "bug"}.get(kw.get("tracker", "isvd"), kw.get("tracker"))  # legacy string until Task 6 renames it
    return kw


def config_hash(pod: PodCfg) -> str:
    blob: DictConfig = OmegaConf.create({
        "pod": OmegaConf.to_container(OmegaConf.structured(pod)),
        "arms": {a: OmegaConf.to_container(OmegaConf.structured(load_arm(a))) for a in pod.arms},
        "tasks": {t: OmegaConf.to_container(OmegaConf.structured(load_task(t))) for t in pod.tasks},
    })
    return hashlib.sha256(OmegaConf.to_yaml(blob, sort_keys=True).encode()).hexdigest()
```

- [ ] **Step 4: Write the YAML files** (every arm/task/pod named above), run the parity test → PASS, `mypy src` clean.

- [ ] **Step 5: Commit** — `git commit -m "L0.4: configs/{arms,tasks,pods} for every v1 arm; kvdlra.eval.config + parity test vs legacy build_arms"`

### Task 5: Entrypoint skeletons, Makefile, Dockerfile (Day-2 merge point)

**Files:**
- Create: `scripts/pod.py`, `scripts/figures.py`, `scripts/dump_kv.py` (from `scripts/capture_kv.py`), `Dockerfile`, `prereg/README.md`
- Modify: `Makefile` (add `env`, `figures`, `check`, `clean`), `scripts/pod/boot.sh` (= `w18_boot.sh` with the driver line replaced), `scripts/pod/watchdog.sh` (= `w19_watchdog.sh`, pod-name driven)
- Test: `tests/test_pod_manifest.py`

**Interfaces:**
- Produces: `scripts/pod.py run --pod NAME [--out results/NAME]` (Task 8 fills the eval loop; in this task it writes `manifest.json` + `env.txt` and exits 0 with an empty `trials.jsonl` when `--dry-run`), `scripts/pod.py launch --pod NAME --offer OFFER_ID` (vastai create with `boot.sh` as `--onstart`, env `POD=NAME SHA=<HEAD>`; refuses if the working tree is dirty, if `prereg/<pod>.md` is missing, or if the prereg's first commit is not a strict ancestor of HEAD; appends to `results/<pod>/pods.txt` for the watchdog), `scripts/pod.py harvest --pod NAME` (calls `watchdog.sh` once: fetch logs, extract `[trial]` lines → `trials.jsonl`, `[pplw]` → `ppl.jsonl`, `[diag]` → `diag.jsonl`), `scripts/pod.py check results/NAME` (exit 1 unless: manifest `config_hash` equals `config_hash(load_pod(name))`; `git_sha` exists; prereg SHA order holds; `errors` in manifest equals the count of records with `error != null`; every `(arm, task, ctx)` cell has `n == n_trials × len(seeds)` counting errors; env versions equal the pyproject pins).
- Manifest schema (`results/<pod>/manifest.json`): `{pod, git_sha, config_hash, model, model_revision, dataset_sha256: {name: sha}, torch, cuda, triton, transformers, kvpress, gpu, launched_at, harvested_at, wall_clock_s, command_line, errors, records}`.

- [ ] **Step 1: Write the failing manifest tests**

`tests/test_pod_manifest.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

from kvdlra.eval.config import config_hash, load_pod


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "scripts/pod.py", *args], capture_output=True, text=True)


def test_dry_run_writes_manifest_and_env(tmp_path: Path) -> None:
    r = _run("run", "--pod", "w18_g1", "--out", str(tmp_path), "--dry-run")
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["config_hash"] == config_hash(load_pod("w18_g1")) and (tmp_path / "env.txt").exists()


def test_check_rejects_tampered_config_hash(tmp_path: Path) -> None:
    _run("run", "--pod", "w18_g1", "--out", str(tmp_path), "--dry-run")
    m = json.loads((tmp_path / "manifest.json").read_text()); m["config_hash"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    r = _run("check", str(tmp_path))
    assert r.returncode == 1 and "config_hash" in r.stdout + r.stderr


def test_check_counts_errors(tmp_path: Path) -> None:
    _run("run", "--pod", "w18_g1", "--out", str(tmp_path), "--dry-run")
    (tmp_path / "trials.jsonl").write_text(json.dumps({"model": "m", "arm": "a", "task": "t", "ctx": 1,
        "seed": 0, "trial": 0, "hit": 0, "frac": 0.0, "haystack_id": None, "depth": None,
        "prompt_sha256": None, "error": "RuntimeError: boom", "source": "x"}) + "\n")
    r = _run("check", str(tmp_path))
    assert r.returncode == 1 and "errors" in r.stdout + r.stderr   # manifest says 0, records say 1
```

- [ ] **Step 2: Implement `scripts/pod.py`** with `argparse` subcommands `run`, `launch`, `harvest`, `check`; `run --dry-run` writes the manifest from `load_pod` + `config_hash` + `git rev-parse HEAD` + `importlib.metadata` versions + `torch.cuda.get_device_name(0)` if available, and `env.txt` (`pip freeze`-style lines for the pinned set). `launch` shells out to `vastai create instance` exactly as the comment block at the top of `scripts/pod/w18_boot.sh` documents, with `--env "-e POD=<name> -e SHA=<sha>"` and `--onstart scripts/pod/boot.sh`; the prereg-ancestor check is `git merge-base --is-ancestor <first-commit-of-prereg> HEAD` and `<first-commit> != HEAD`. `check` implements the six rules above and prints each failure on its own line prefixed `CHECK FAIL <rule>:`.

- [ ] **Step 3: `boot.sh` / `watchdog.sh`**: copy `w18_boot.sh` → `boot.sh`; replace the `DRIVER` sourcing with `python scripts/pod.py run --pod "$POD" 2>&1`; keep the SHA pin, the `===ENV_BEGIN===` header, `emit()`, and `pip install ninja` in the dependency line. Copy `w19_watchdog.sh` → `watchdog.sh`; the pods file becomes `results/<pod>/pods.txt`; the harvest step writes `results/<pod>/trials.jsonl` through `scripts/pod.py harvest` instead of the inline `grep`; keep the 5000-line fallback and the credit floor. Old files are deleted in Task 9, not here.

- [ ] **Step 4: `figures.py`, `dump_kv.py`, `Dockerfile`, `prereg/README.md`, Makefile**

`scripts/figures.py`: move `scripts/w19_figures.py` (`git mv`), change its data source to `results/paper-v1/*/trials.jsonl` via `kvdlra.eval.records.read_jsonl`, subcommand `build --out docs/paper/figures`. `scripts/dump_kv.py`: `git mv scripts/capture_kv.py scripts/dump_kv.py`; add `verify --dir dumps/llama3.2-1b` that computes sha256 of every file and compares against a committed `dumps/llama3.2-1b.sha256` manifest (`dump` writes it). `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
ENV PIP_BREAK_SYSTEM_PACKAGES=1 CUDA_HOME=/usr/local/cuda
RUN pip install -q hf_transfer hf_xet ninja numpy scipy matplotlib "kvpress==0.5.1" \
    "transformers==5.8.0" "datasets==2.21.0" "optimum-quanto>=0.2.7" "hqq==0.2.8.post1" "omegaconf>=2.3"
WORKDIR /root/kvdlra
```
(the same pinned set `boot.sh` installs; `boot.sh` stays the launch path — the image is for reproduction, not for the vast.ai onstart, which still clones at the SHA). `prereg/README.md`: three lines stating the rule (committed before the launch commit; checked by `scripts/pod.py check`). Makefile additions:
```make
env:
	uv venv --python 3.12 && uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"

figures:
	$(PY) scripts/figures.py build --out docs/paper/figures

check:
	@for d in results/*/manifest.json; do $(PY) scripts/pod.py check $$(dirname $$d) || exit 1; done

clean:
	rm -rf docs/paper/tables/*.md docs/paper/tables/*.tex docs/paper/figures/*
```

- [ ] **Step 5: Run** `make test`, `make tables`, `make figures`, `.venv/bin/python scripts/pod.py run --pod w18_g1 --dry-run --out /tmp/x && .venv/bin/python scripts/pod.py check /tmp/x` — all green; mypy strict clean on `scripts/pod.py scripts/tables.py scripts/figures.py scripts/dump_kv.py`.

- [ ] **Step 6: Commit** — `git commit -m "L0.5: scripts/{pod,figures,dump_kv}.py skeletons, manifest + check, boot/watchdog, Makefile, Dockerfile"`

- [ ] **Step 7: Phase-A merge point.** Run the whole suite from a clean clone of the lane branch (`git clone --branch lane/L0-repo-cleanup . /tmp/l0 && cd /tmp/l0 && make env && make test && make tables`). Report to the orchestrator: L6 verifies, `finishing-a-development-branch` merges to `week7`, STATE.md gets the Phase-A entry, and L1/L2/L4/L5 are dispatched against the new layout. Continue Phase B on the same lane branch after merging `week7` back in.

---

## Phase B — completion (Day 4)

### Task 6: Source-tree moves and dead-module deletion (behind the golden)

**Files:**
- Move: `src/kvdlra/integrators/streaming_torch.py` → `src/kvdlra/tracker/isvd.py`; `src/kvdlra/press/palu_press.py` → `src/kvdlra/baselines/svd_oracle.py`; `src/kvdlra/press/bug_press.py` → `src/kvdlra/baselines/lowrank_press.py`; `src/kvdlra/press/compat.py` → `src/kvdlra/baselines/compat.py`; `src/kvdlra/utils/seed.py` → `src/kvdlra/util/seed.py`
- Delete: `src/kvdlra/integrators/{__init__,bug,bug_adaptive,bug_class,bug_torch,streaming,oja,frequent_directions,streaming_variants}.py`, `src/kvdlra/lowrank.py`, `src/kvdlra/quant/{product_quant,qjl}.py`, `src/kvdlra/press/turbo_press.py`, `src/kvdlra/utils/hydra_resolvers.py`, `src/kvdlra/cache/morph_cache.py` (only if `grep -n -i 'morph' paper/main.tex` shows no table row — otherwise keep and record why in deletions.md), and their tests: `tests/test_{bug_class,bug_torch,bug_synthetic,bug_adaptive,oja,frequent_directions,streaming,streaming_variants,qjl,product_quant,turbo_press,morph_cache}.py`
- Modify: every importer (`grep -rn 'kvdlra.integrators\|kvdlra.press\|kvdlra.utils\|kvdlra.lowrank' src scripts tests`), `src/kvdlra/quant/__init__.py`, `src/kvdlra/press/__init__.py` (removed with the package), `src/kvdlra/cache/__init__.py`
- Rename inside `svd_oracle.py`: class `PaluPress` → `SVDOraclePress`; docstring states it is a per-sequence, per-head truncated SVD of the prefill K/V — an upper bound for static low-rank methods, NOT Palu (no weight decomposition, no grouped heads, no Fisher allocation, no fine-tune). The legacy arm string `palu-r0.5` survives only in `results/paper-v1` records and `configs/arms/svd_oracle_r0.5.yaml:legacy_name`.
- Remove the numpy backend from `lowrank_press.py` (`backend=` parameter and the `streaming.StreamingBUG` branch; sole caller was `tests/test_bug_press.py:201` — update that test to the torch path).
- Test: the whole suite; `tests/test_golden_cache.py` must stay green after every move.

- [ ] **Step 1: Confirm liveness before deleting** — for each path in the delete list, cite the `docs/plan/cleanup/reachability.md` row (`none` or `tests-only`) in `docs/plan/cleanup/deletions.md` (start the file now; one line per path: `path | LOC | bucket | evidence`). Any path the report marks `pod-sh` is NOT deleted — report the discrepancy instead.
- [ ] **Step 2: `git mv` the five moves; `sed` the import paths** (`kvdlra.integrators.streaming_torch` → `kvdlra.tracker.isvd`, `kvdlra.press.palu_press` → `kvdlra.baselines.svd_oracle`, …); add `src/kvdlra/{tracker,baselines,util}/__init__.py`.
- [ ] **Step 3: Run** `make test` — Expected: PASS (golden included). Then `git rm` the delete list + tests; run `make test` again → PASS; `ruff check . && mypy src tests scripts` clean.
- [ ] **Step 4: Commit** — `git commit -m "L0.6: tracker/baselines/util layout; delete ODE integrators, numpy twins, dead quant/press modules (+ their tests); PaluPress -> SVDOraclePress"`

### Task 7: Dead paths inside `bug_cache.py` and `w10_frontier.py`

**Files:**
- Modify: `src/kvdlra/cache/bug_cache.py` — remove: the CodeBUG path (`coord_codebook`, `anchor_rank`, `anchor_seal_absorbs`, `code_budget` and `_QuantBank` codebook branches; ~55 lines), the Q-BUG whitening (`w_key`, `_unwhiten_key`, `_load_wkey` in the frontier; the `bugSQ` naming branch), `merge=` and its `track_positions` guard (`bug_cache.py:357, 498`), retention modes `attn`/`energy`/`blend` and the `track_surprise`/`score_decay`/`surprise_blend` plumbing (`bug_cache.py:166, 1033-1050`; keep `fifo` and `lowrank_surprise`). Keep every guard on the surviving paths (the seed+quant fence, the SLASH `--chunk` invariant, the norm renormalization in the quant tier).
- Delete tests that exist only for removed paths (`tests/test_bug_cache_qbug.py`, the CodeBUG tests in `tests/test_bug_cache_week8.py` — move any surviving assertions into `tests/test_bug_cache.py`); keep the strict xfail in `tests/test_bug_cache_week15.py` only if `retention="attn"` survives — it does not, so convert that xfail into a plain deletion with a one-line note in deletions.md (the latent bug leaves with the mode).
- Test: whole suite; golden green.

- [ ] **Step 1:** `grep -n` every symbol above across `src tests scripts` and list the callers in the task report before editing.
- [ ] **Step 2:** Remove one feature at a time; after each: `make test` (golden must stay green — the r64 configuration uses none of these paths). Any golden change = stop, `systematic-debugging`.
- [ ] **Step 3:** Commit per feature: `L0.7a: drop CodeBUG codebook path`, `L0.7b: drop Q-BUG whitening`, `L0.7c: drop merge= and attn/energy/blend retention`.

### Task 8: Fold the eval scripts into `kvdlra/eval/*`, driven by configs; thin the pod scripts

**Files:**
- Move (`git mv`, then edit): `scripts/w10_ruler.py` → `src/kvdlra/eval/ruler.py`; `scripts/w10_frontier.py` → `src/kvdlra/eval/frontier.py`; `scripts/w19_official_ruler.py` → `src/kvdlra/eval/official_ruler.py`; `scripts/w10_longbench.py` → `src/kvdlra/eval/longbench.py`; `scripts/w20_latency.py` → `src/kvdlra/eval/latency.py`; `scripts/w16_storage.py` → `src/kvdlra/eval/storage.py`; `scripts/w19_persist.py` → `src/kvdlra/eval/persist.py`; `scripts/perplexity_sweep.py`'s WikiText loader + filler pool → `src/kvdlra/eval/data.py` (only the functions `ruler.py`/`frontier.py` import; the rest of that script is deleted)
- Modify: `scripts/pod.py run` — the loop: for each `task` in the pod, for each `arm`, call `ruler.run(arm, task, model, …)` / `frontier.run_ppl(...)` / `official_ruler.run(...)`, yielding `TrialRecord`s (with `haystack_id`, `depth`, `prompt_sha256` filled by the generator where it knows them; `error=str(exc)` and `hit=0` on exception — **never skip**), streaming them to `results/<pod>/trials.jsonl` as they arrive; `ppl.jsonl` rows `{model, arm, ctx, window, nll_sum, n_tok}`; `diag.jsonl` reserved for L1's per-layer diagnostics (write nothing yet).
- Modify: `build_arms(args, model, t)` → `build_arm(arm: ArmCfg, model, t)` using `arm_kwargs` from Task 4; the `argparse` flag soup (`w10_ruler.py` 42 `add_argument`s, `w10_frontier.py` parser) is deleted; the parity test from Task 4 is rewritten to compare `build_arm` against the committed `tests/golden/legacy_arm_kwargs.json` (dump it from the legacy path in Step 1 before deleting the parser).
- Modify: `scripts/pod/boot.sh` already calls `pod.py run`. Delete every other `scripts/pod/*.sh` in Task 9.
- Tests: `tests/test_w10_*.py`, `tests/test_w12_harness.py`, `tests/test_w15_harness.py`, `tests/test_w15_pplw.py`, `tests/test_w18_intervals.py`, `tests/test_w19_*.py`, `tests/test_w20_tracker_swap.py`, `tests/test_ruler_template_tail.py` — update imports to `kvdlra.eval.*`; tests of the deleted interval scripts move their assertions to `tests/test_stats.py`/`tests/test_tables_golden.py` (the 12/12 → 0.758 and McNemar pins are already there).
- Test (new): `tests/test_pod_run_records_errors.py` — a fake arm whose factory raises on trial 3 produces a record with `error` set and `hit=0`, and `pod.py check` counts it.

- [ ] **Step 1:** `python - <<'PY'` dump of every legacy arm's kwargs (all `configs/arms/*.yaml` with a `legacy_name`) via the Task-4 helper → `tests/golden/legacy_arm_kwargs.json`. Commit it first.
- [ ] **Step 2:** Write `tests/test_pod_run_records_errors.py`:
```python
import json
from pathlib import Path

from kvdlra.eval.config import load_pod, load_task
from kvdlra.eval.runner import run_pod  # the loop scripts/pod.py `run` calls


def test_raising_trial_is_recorded_not_skipped(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_trial(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return 1, 1.0, {"haystack_id": "h0", "depth": 0.5, "prompt_sha256": "x"}

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    pod = load_pod("w18_g1"); pod.arms = ["full"]; pod.tasks = ["ruler_inhouse_16k"]
    run_pod(pod, out=tmp_path, model=None, dry_model=True)
    rows = [json.loads(l) for l in (tmp_path / "trials.jsonl").read_text().splitlines()]
    t = load_task("ruler_inhouse_16k")
    assert len(rows) == len(t.tasks) * t.n_trials * len(t.seeds)
    errs = [r for r in rows if r["error"]]
    assert len(errs) == 1 and errs[0]["hit"] == 0 and "boom" in errs[0]["error"]
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == 1
```
- [ ] **Step 3:** Implement `src/kvdlra/eval/runner.py:run_pod(pod, out, model, dry_model=False)` and refactor `ruler.py` so one trial is `run_trial(arm_obj, prompt, targets, ...) -> (hit, frac, meta)`; the generator functions (`build_task`, filler pools, the official RULER reader) keep their bodies — this task moves and rewires, it does not change generator semantics (L2 owns generator v2). Delete the `SKIP` branch (`w10_ruler.py:436-446` in the old numbering).
- [ ] **Step 4:** `make test` → PASS (golden green; parity vs `legacy_arm_kwargs.json` green); `make tables` still diff-clean (it reads only `results/paper-v1`).
- [ ] **Step 5:** Commit — `L0.8: kvdlra.eval.{ruler,frontier,official_ruler,longbench,latency,storage,persist,data,runner}; pod.py run drives configs; trials that raise are recorded as error`.

### Task 9: Deletions, relocations, dependency prune, README

**Files:**
- Delete: every `scripts/*.py` not in the file map (the reachability `none`/`scripts-only`/folded lists); every `scripts/pod/*.sh` except `boot.sh`, `watchdog.sh`; `experiments/`; `figs/`; `figures/**` not referenced by `\includegraphics` in `paper/main.tex` (list the kept ones in deletions.md); `mkdocs.yml`; `docs/*.md`, `docs/notes/`, `docs/board/`, `docs/week10_report/`, `docs/index.md`, `docs/reference.md`, `docs/PLAN.md`; the converted originals `results/w18_pertrial/`, `results/w19_pertrial/`, `results/*-lines.txt`, `results/*.json` line-level dumps that Task 1 archived (keep `results/w20-close-report.md`, `results/w19-*-report.md`, `results/w18-*-report.md` markdown reports until their content is folded into docs/plan — list them as "kept: narrative report" in deletions.md); on-disk gitignored litter (`rm`, no commit): `handover.md`, `explanation_week_*.md`, `next-session-prompt.md`, `compass_artifact_*.md`, `dashboards/`, `.DS_Store` files.
- Move: `git mv docs/reviews docs/plan/reviews`.
- Modify: `pyproject.toml` — remove `accelerate`, `huggingface_hub`, `wandb`, `python-dotenv`, `tqdm`, `lm-eval`, `flash-attn` extras, the four mkdocs dev deps; keep `hydra-core` + `omegaconf` (config layer), `optimum-quanto`, `hqq` (backend strings), `datasets`, `scipy`; add `ninja` to `dev`; drop the `wandb.*`/`flash_attn.*` mypy overrides; `.gitignore` — drop the mkdocs/handover entries that no longer matter, keep `dumps/**`, `results/*/pods.txt`, `results/*/*.raw`.
- Rewrite: `README.md` ≤ 120 lines: what it is (one paragraph, incremental-SVD gist + surprise tier + ring + sinks + seed; no branding words), install (`make env`), reproduce one table in three commands (`git clone … && make env && make tables`), layout tree, how a pod is pre-registered and launched, license.
- Test: `make test`, `make tables`, `make figures` green; `ruff`/`mypy` clean; `git status` clean.

- [ ] **Step 1:** Generate the deletion list from `docs/plan/cleanup/reachability.md` + the audit; write every path into `docs/plan/cleanup/deletions.md` with its bucket and the report line that justifies it. Paths the report marks `pod-sh` for `w16.sh`/`w18.sh`/`w19.sh` are folded (Task 8), not deleted — cross-check.
- [ ] **Step 2:** `git rm -r` the list; `git mv docs/reviews docs/plan/reviews`; edit `pyproject.toml`, `.gitignore`; `rm` the gitignored litter; write `README.md`.
- [ ] **Step 3:** `uv pip install -e ".[dev]"` (picks up `ninja`); `make test` → the quanto test now passes; `make tables`; `make figures`; `ruff check . && ruff format --check . && mypy src tests scripts`.
- [ ] **Step 4:** Commit in three: `L0.9a: delete unreachable scripts/, pod launchers, experiments/, figs/, stale docs (evidence: docs/plan/cleanup/deletions.md)`, `L0.9b: prune 13 unused deps; add ninja; move docs/reviews -> docs/plan/reviews`, `L0.9c: README <= 120 lines`.

### Task 10: CI, reachability test, forbidden-word test, src wording pass

**Files:**
- Create: `tests/test_reachability.py`, `tests/test_forbidden_words.py`
- Modify: `.github/workflows/ci.yml` (add `make tables` diff and `make figures`; `Pytest` step becomes `make test`), `src/**` docstrings/comments (wording pass), `scripts/pod/*.sh` comments (wording pass)

- [ ] **Step 1: Write the two tests**

`tests/test_reachability.py`:
```python
"""Every .py under src/ and scripts/ must be in the static import closure of the four
entrypoints + tests (+ what scripts/pod/*.sh invokes). Unreachable code is deleted, not kept."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, SCR, TESTS = ROOT / "src", ROOT / "scripts", ROOT / "tests"
ENTRY = [SCR / f"{n}.py" for n in ("pod", "tables", "figures", "dump_kv")]


def _module_of(path: Path) -> str:
    rel = path.relative_to(SRC) if SRC in path.parents else path.relative_to(SCR)
    return ".".join(rel.with_suffix("").parts).removesuffix(".__init__")


def _resolve(name: str) -> Path | None:
    for base in (SRC, SCR):
        p = base / Path(*name.split("."))
        if p.with_suffix(".py").exists():
            return p.with_suffix(".py")
        if (p / "__init__.py").exists():
            return p / "__init__.py"
    return None


def _imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text())
    out: set[Path] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            r = _resolve(n)
            if r:
                out.add(r)
    return out


def test_everything_is_reachable() -> None:
    sh_invoked = {SCR / m for sh in (SCR / "pod").glob("*.sh")
                  for m in re.findall(r"scripts/([A-Za-z0-9_]+\.py)", sh.read_text())}
    seen: set[Path] = set()
    todo = set(ENTRY) | set(TESTS.glob("test_*.py")) | sh_invoked
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo |= _imports(p) - seen
    all_py = {p for base in (SRC, SCR) for p in base.rglob("*.py") if "__pycache__" not in p.parts}
    unreachable = sorted(str(p.relative_to(ROOT)) for p in all_py - seen)
    assert not unreachable, f"unreachable: {unreachable}"
```
`tests/test_forbidden_words.py`:
```python
"""CLAUDE.md: DLRA (as a word), 'BUG integrator', honest(ly), marquee, flagship never appear
in src/, configs/, scripts/, docs/paper/. `kvdlra` / `kv-dlra` identifiers are not hits."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"(?<![\w-])(dlra|marquee|flagship|honest(?:ly)?|bug integrator)\b", re.I)
SCOPE = ("src", "configs", "scripts", "docs/paper")


def test_no_forbidden_words() -> None:
    hits = []
    for top in SCOPE:
        for p in (ROOT / top).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".sh", ".md", ".tex", ".txt"}:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if PAT.search(line):
                        hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:80]}")
    assert not hits, "\n".join(hits)
```
- [ ] **Step 2:** Run both → the reachability test should PASS after Task 9 (fix any leftover or report); the forbidden-word test FAILS on the src/ docstring hits. Reword every hit in `src/**` and `scripts/pod/*.sh` comments (identifiers untouched; "the incremental-SVD tracker", "the r64 configuration", "stored-state accounting" are the replacements). Run → PASS.
- [ ] **Step 3:** `ci.yml`: replace `uv run pytest -q` with `make test PY="uv run python"`; add steps `make tables PY="uv run python"` and `make figures PY="uv run python"`. Keep ruff/mypy steps.
- [ ] **Step 4:** Commit — `L0.10: reachability + forbidden-word tests in CI; src/scripts wording pass; ci runs make test/tables/figures`.

### Task 11: Ponytail after-report, deletions ledger, handoff

**Files:**
- Create: `docs/plan/cleanup/ponytail-after.md` (run the `ponytail-audit` skill again over the cleaned tree; include the before/after LOC table — net delta must be negative — and the `ponytail-debt` ledger)
- Finalize: `docs/plan/cleanup/deletions.md` (every removed path + evidence; every kept-despite-candidate path with the reason)

- [ ] **Step 1:** `find src scripts tests -name '*.py' | xargs wc -l | tail -1` before (33,738 at ee8c0ab) and after; write the table.
- [ ] **Step 2:** Clean-clone verification (the G0 gate): `git clone --branch lane/L0-repo-cleanup <repo> /tmp/l0 && cd /tmp/l0 && make env && make test && make tables && make figures` — attach the output paths to the task report; `make test` wall time < 90 s.
- [ ] **Step 3:** Commit — `L0.11: ponytail after-report + deletions ledger; G0 clean-clone verification`.
- [ ] **Step 4:** Hand to the orchestrator: L6 runs the G0 checklist (GATES.md), `/ponytail-review` on the branch diff, then `finishing-a-development-branch` → merge to `week7`; STATE.md "L0 done" entry with the verifier signature.

---

## Self-review

- **Spec coverage vs the L0 brief:** §1 reachability graph → Task 10 test + `deletions.md` (Tasks 6, 9). §2 candidates → Tasks 6, 7, 9 (each verified against the reachability report before deletion; `docs/reviews` moved). §3 tag + trials.jsonl conversion → Task 1 (`cells.jsonl` added for aggregate-only line files — nulls, never inferred). §4 environment → Task 5 (Dockerfile, `make env`, `env.txt`; note: `uv.lock` is gitignored in this repo — "diff against the lock" is implemented as diff against the pyproject pins) + Task 9 (`ninja`). §5 config system → Task 4 (omegaconf, hydra's config layer; the Hydra CLI compose machinery is not used — YAGNI, one line in the ADR-free config docstring) + Task 8 (flag soup gone). §6 `make tables`/`figures` + `stats.py` with the audit's pinned values → Tasks 2, 3, 5. §7 CI → Task 10. §8 README → Task 9. Ponytail before/after → the committed audit + Task 11.
- **GATES G0 mapping:** tag ✓ (exists) · deletions.md ✓ T6/T9/T11 · reachability 0 + net LOC negative ✓ T10/T11 · `make env && make test` green on a clean clone, < 90 s ✓ T5 step 7, T11 · `make tables` diff-clean vs `docs/plan/paper-v1-tables.md` ✓ T3 · every v1 arm/task/pod a YAML, flag soup gone ✓ T4/T8 · Dockerfile, `pod.py --check` on a synthetic manifest, launch/watchdog reused ✓ T5 · forbidden-word grep clean, README ≤ 120, after-report ✓ T9/T10/T11.
- **Placeholders:** the three `MODEL_BY_TAG` ids (Task 1 Step 1 fills them from the pod scripts, fail-loud) and the golden test's fixture names (Task 1 Step 7 tells the implementer which existing helpers to use) are the only deliberate look-ups; everything else is concrete.
- **Type consistency:** `TrialRecord` keys (Task 1) are what `run_pod` (Task 8) and `pod.py check` (Task 5) read; `config_hash(load_pod(name))` is the same call in Tasks 4, 5, 8; `arm_kwargs` (Task 4) is what `build_arm` (Task 8) uses; `mcnemar_exact`'s dict-keyed signature (Task 2) is what `tables.py` (Task 3) calls.
- **Scope notes for the orchestrator:** L5's "provenance: `scripts/pod.py` writes manifest.json at launch and harvest; `--check` verifies …; tampered-hash test" is delivered here (Task 5) so two lanes never edit `pod.py`; L5 keeps the prereg files and the bf16 config. `tier/` and `kernel/` directories are created by the lanes that first put a file there (L1/L4), not by L0. Splitting `bug_cache.py` (2,022 lines) into `cache/` + `tier/` is deliberately NOT in L0 — it is a refactor of live numerics with no reproducibility payoff; revisit after Gate 1.
