"""The eval loop `scripts/pod.py run` drives: a pod config in, records out.

One pod is a model, a set of arms and a set of tasks; the loop is every task x every
arm x (for a retrieval task) every sub-task x seed x trial. Each trial appends a
``TrialRecord`` to ``results/<pod>/trials.jsonl`` as it completes, so a pod killed
halfway leaves a readable partial file rather than nothing.

``trials.jsonl`` is the ONLY file that streams. The perplexity, per-window and decode
rows are buffered in memory and written at the very end of the pod, so a run killed
mid-sweep leaves no ``ppl.jsonl``, ``pplw.jsonl`` or ``latency.jsonl`` at all -- not a
short one. The recovery path for a killed pod is `scripts/pod.py harvest` from its log:
every row of all four files is printed as it is produced, which is what the stdout
contract below is for.

A trial that raises is RECORDED -- ``error`` set, ``hit=0``, ``frac=0.0`` -- and the
loop continues. The v1 harness printed SKIP and dropped the trial, which silently
shrank a cell's n; `scripts/pod.py check` now requires every configured cell to hold
exactly ``n_trials x len(seeds)`` records, so a dropped trial would fail the run
instead of quietly weakening it.

The ``[trial]``, ``[<task> ctx<T>]``, ``[pplw]``, ``[latency ctx<T>]`` and ``ppl=``
lines are printed as well as written: they are the pod's stdout contract, and `pod.py
harvest` can rebuild the same records from a `vastai logs` capture when the results
directory never made it off the instance. A ``[trial]`` line carries the generator's
pairing fields (``hay= depth= code= sha=``, ``-`` where the generator set none), so a
harvested pod can still show that two arms of one cell were fed byte-identical prompts.
``[stage] <what> (<s> s)`` lines time the loads (model, corpora, haystacks) and a
``[stage] cell ... elapsed_s=`` line times each completed cell; the watchdog keeps them,
so a slow pod's log says where the hours went and `harvest` can bill them per arm.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from kvdlra.baselines.compat import install_kvpress_prefill_compat
from kvdlra.eval import frontier, gen, latency, longbench, official_ruler, ruler
from kvdlra.eval.config import PodCfg, TaskCfg, load_arm, load_task
from kvdlra.eval.data import load_corpus_ids, load_corpus_sentences
from kvdlra.eval.records import (
    DIAG_ROWS,
    LatencyRecord,
    PplRecord,
    PplwRecord,
    TrialRecord,
    write_jsonl,
)

# Which module answers for a task config's `generator:`. Looked up as a module, not as
# a function, so the attribute resolves at call time (a test can substitute one). The
# `ppl` and `latency` axes are not in the map: neither runs per-trial, and each has its
# own loop in `run_pod`.
GENERATORS = {
    "inhouse": ruler,
    "official_ruler": official_ruler,
    "longbench": longbench,
    "v2": gen,
}


def run_pod(pod: PodCfg, out: Path, model: Any, dry_model: bool = False) -> None:
    """Evaluate ``pod`` into ``out``.

    ``model`` is the ``(model, tokenizer)`` pair `kvdlra.eval.data.load_model` returns.
    ``dry_model`` runs the loop without one: nothing is loaded and no dimensions are
    read off a config, which is how the error path is exercised on CPU with the
    generator substituted (``tests/test_pod_run_records_errors.py``).

    A run always starts fresh -- ``trials.jsonl`` is truncated and the diagnostics
    buffer cleared here -- and resume is not supported; re-running a pod re-runs all of
    it. `scripts/pod.py harvest` is the path that guards against a results directory
    shrinking.
    """
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / "trials.jsonl"
    trials_path.write_text("")
    # Fresh here too: a pod that raised out of a previous `run_pod` in this process
    # must not leak its diagnostics into this one's diag.jsonl.
    DIAG_ROWS.clear()

    mdl: Any = None
    tok: Any = None
    device, n, h_kv = "cpu", 0, 0
    if not dry_model:
        mdl, tok = model
        device = str(next(mdl.parameters()).device)
        n, h_kv = _dims(mdl.config)
        install_kvpress_prefill_compat()  # transformers 5.8: presses need the shim
        # Never eager at long T -- eager materializes O(T^2). Harmless on CPU.
        mdl.config._attn_implementation = "sdpa"

    n_err = 0
    ppl: list[PplRecord] = []
    pplw: list[PplwRecord] = []
    lat: list[LatencyRecord] = []
    corpora: dict[str, str] = {}  # corpus name -> sha256 of the exact token stream
    for tname in pod.tasks:
        task = load_task(tname)
        if task.generator == "latency":
            # Its own loop: it sweeps context lengths within one task, so the arms are
            # rebuilt per context rather than once per task.
            n_err += _latency_rows(pod, task, mdl, lat, device=device)
            continue
        arms = [_build(a, mdl, task.ctx) for a in pod.arms]
        if task.generator == "ppl":
            rows = _ppl_rows(arms, mdl, tok, task, device=device, n=n, h_kv=h_kv, sha=corpora)
            ppl += [_ppl_record(pod, r) for r in rows if r["status"] == "ok"]
            pplw += [w for r in rows if r["status"] == "ok" for w in _pplw_records(pod, r)]
            n_err += _log_ppl_errors([r for r in rows if r["status"] != "ok"])
            continue
        for arm in arms:
            for sub in task.tasks:
                n_err += _cell(pod, task, arm, sub, mdl, tok, trials_path, device, n, h_kv)

    records = {"trials.jsonl": sum(1 for _ in trials_path.read_text().splitlines())}
    if ppl:
        write_jsonl(out / "ppl.jsonl", ppl)
        records["ppl.jsonl"] = len(ppl)
    if pplw:
        write_jsonl(out / "pplw.jsonl", pplw)
        records["pplw.jsonl"] = len(pplw)
    if lat:
        write_jsonl(out / "latency.jsonl", lat)
        records["latency.jsonl"] = len(lat)
    _finish(out, records, n_err, time.perf_counter() - t0, corpora)


def _log_ppl_errors(rows: list[dict[str, Any]]) -> int:
    """The failed arms of a perplexity sweep, one greppable line each.

    This axis writes no per-trial record, so a failed arm used to leave nothing behind
    but a filtered-out row: the manifest called the pod clean. The line is what
    `records.parse_error_lines` counts, so a harvest of the log lands on the same count
    the run did.
    """
    for r in rows:
        print(
            f"[error] axis=ppl arm={r['method']} ctx={r['T']} error={r.get('error', r['status'])}",
            flush=True,
        )
    return len(rows)


def _latency_rows(
    pod: PodCfg,
    task: TaskCfg,
    model: Any,
    out: list[LatencyRecord],
    *,
    device: str,
) -> int:
    """One measured decode point per (arm, ctx, batch), appended to ``out``.

    Not a set of Bernoulli trials and not a perplexity sweep: one row per point, so it
    has neither a cell nor a ``ppl`` record to hang a failure on. A point that raises is
    an ``[error] axis=latency`` line and a counted error, exactly as a failed perplexity
    arm is -- otherwise a pod whose 64K points all OOMed would land a short
    ``latency.jsonl`` and call itself clean. `scripts/pod.py check` requires the full
    (arm, ctx, batch) grid, which is what turns the missing row into a failure.
    """
    errors = 0
    for ctx in task.ctxs or [task.ctx]:
        for name in pod.arms:
            arm = _build(name, model, ctx)
            t_cell = time.perf_counter()
            for batch in task.batch_sizes:
                try:
                    (row,) = latency.run_latency(
                        model, [arm], ctx, device, task.chunk, task.n_steps, task.warmup, batch
                    )
                except Exception as exc:
                    print(
                        f"[error] axis=latency arm={arm['name']} ctx={ctx} batch={batch}"
                        f" error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    errors += 1
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                out.append(
                    {
                        "model": pod.model,
                        "arm": arm["name"],
                        "ctx": ctx,
                        "batch": batch,
                        "ms_per_token_p50": float(row["ms_per_tok_p50"]),
                        "ms_mean": float(row["ms_per_tok_mean"]),
                        "ms_max": float(row["ms_per_tok_max"]),
                        "spikes": int(row["spikes_gt_2x"]),
                        "resident_gb": float(row["resident_gb"]),
                        "peak_gb": float(row["peak_gb"]),
                        "kv_peak_gb": float(row["kv_peak_gb"]),
                        "source": f"{pod.name}:run",
                    }
                )
            # `_cell`'s timing line for this axis too: one per (arm, ctx) sweep over the
            # batch sizes, `n` the points it attempted (a point that raised leaves no
            # row but did cost the seconds).
            print(
                f"[stage] cell arm={arm['name']} task={task.name} ctx={ctx}"
                f" elapsed_s={time.perf_counter() - t_cell:.1f} n={len(task.batch_sizes)}",
                flush=True,
            )
    return errors


def _dims(cfg: Any) -> tuple[int, int]:
    """(feature width per layer, KV heads) -- the two numbers the accounting needs."""
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return int(head_dim * cfg.num_key_value_heads), int(cfg.num_key_value_heads)


def _build(name: str, model: Any, ctx: int) -> dict[str, Any]:
    return frontier.build_arm(load_arm(name), model, ctx)


def _cell(
    pod: PodCfg,
    task: TaskCfg,
    arm: dict[str, Any],
    sub: str,
    model: Any,
    tok: Any,
    trials_path: Path,
    device: str,
    n: int,
    h_kv: int,
) -> int:
    """One (arm, sub-task, ctx) cell: every (seed, trial), appended as it completes.

    ``depths`` does NOT multiply the cell: the generator sweeps the grid across trial
    indices (``depths[trial % len]``), so a cell is ``n_trials x len(seeds)`` records
    whether or not the task pins depths -- the same count `_expected_cells` computes.
    """
    module = GENERATORS[task.generator]
    chunk = task.chunk if arm["chunkable"] else 0
    t0 = time.perf_counter()
    pool = None if task.filler in ("cycle", "official") else load_corpus_sentences(task.filler)
    if pool is not None:
        dt = time.perf_counter() - t0
        print(f"[stage] load_corpus_sentences {task.filler} ({dt:.1f} s)", flush=True)
    hits, fracs, ratios, sbits, errors = 0, [], [], [], 0
    t_cell = time.perf_counter()
    for seed in task.seeds:
        for trial in range(task.n_trials):
            try:
                hit, frac, meta = module.run_trial(
                    arm, model, tok, task, sub, seed, trial,
                    device=device, chunk=chunk, n=n, h_kv=h_kv, pool=pool,
                )  # fmt: skip
                err = None
            except Exception as exc:
                # Recorded, never dropped: a cell that silently shrinks is how a weak
                # arm used to look like a clean one. The reason travels with the row.
                hit, frac, meta, err = 0, 0.0, {}, f"{type(exc).__name__}: {exc}"
                errors += 1
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            hits += hit
            fracs.append(frac)
            if meta.get("ratio") is not None:
                ratios.append(float(meta["ratio"]))
                sbits.append(float(meta["sbits"]))
            # The official generator's trial is RULER's own record id, not the loop
            # counter -- which is what the archived v1 rows carry (11779, 76228).
            tid = int(meta.get("trial", trial))
            row: TrialRecord = {
                "model": pod.model,
                "arm": arm["name"],
                "task": sub,
                "ctx": task.ctx,
                "seed": seed,
                "trial": tid,
                "hit": hit,
                "frac": frac,
                "generator": task.generator,
                "haystack_id": meta.get("haystack_id"),
                "depth": meta.get("depth"),
                "code_family": meta.get("code_family"),
                "prompt_sha256": meta.get("prompt_sha256"),
                "error": err,
                "source": f"{pod.name}:run",
            }
            with trials_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
            depth = row["depth"]
            # `%.2f` round-trips log -> disk only while every depth grid is 2-decimal
            # (the six-point `gen.DEPTH_GRID` is); a finer grid needs a wider print.
            print(
                f"[trial] task={sub} ctx={task.ctx} arm={arm['name']} seed={seed} "
                f"trial={tid} hit={hit} frac={frac:.3f} generator={task.generator}"
                f" hay={row['haystack_id'] or '-'}"
                f" depth={'-' if depth is None else f'{depth:.2f}'}"
                f" code={row['code_family'] or '-'} sha={row['prompt_sha256'] or '-'}"
                + (f" error={err}" if err else ""),
                flush=True,
            )
    total = len(fracs)
    head = (
        f"[{sub} ctx{task.ctx}] {arm['name']:14s} acc={hits / total:.2f} "
        f"recall={sum(fracs) / total:.2f}"
    )
    # `n=` and `sbits=` come last so the archived-line regex keeps matching unchanged.
    # With no surviving trial there is no footprint to average: printing `ratio=nan
    # sbits=nan` matched no reader's regex, so the whole cell vanished from a harvest.
    if ratios:
        head += f" ratio={sum(ratios) / len(ratios):.3f} sbits={sum(sbits) / len(sbits):.3f}"
    print(head + f" n={total}" + ("" if ratios else f" errors={errors}"), flush=True)
    # The cell's wall clock, which nothing else in a harvest carries: `[trial]` and cell
    # rows have no timestamp, a harvested `wall_clock_s` is null, and the watchdog
    # `sort -u`s `<label>.raw` in place every poll, so arrival order is destroyed. The
    # seconds ride the line, which makes it order-independent; `pod.py harvest` folds
    # them into `manifest.cell_elapsed_s`, and a per-arm min/sample is the sum over its
    # cells over its samples -- how the next pod is sized.
    print(
        f"[stage] cell arm={arm['name']} task={sub} ctx={task.ctx}"
        f" elapsed_s={time.perf_counter() - t_cell:.1f} n={total}",
        flush=True,
    )
    return errors


def _ppl_rows(
    arms: list[dict[str, Any]],
    model: Any,
    tok: Any,
    task: TaskCfg,
    *,
    device: str,
    n: int,
    h_kv: int,
    sha: dict[str, str],
) -> list[dict[str, Any]]:
    """One perplexity sweep, on the corpus the TASK names (`config.TaskCfg.corpus`).

    ``sha`` collects ``corpus -> sha256(token ids)``: which text was scored is half of
    what a perplexity number means, and the digest is over the exact ids the windows
    were cut from, so a corpus that silently changed upstream cannot pass for the one
    the manifest cites. `_finish` writes it to `manifest.json` on the pod; the
    ``[stage] dataset_sha256`` line is how `pod.py harvest` gets it off the log.
    """
    t0 = time.perf_counter()
    ids = load_corpus_ids(tok, device, corpus=task.corpus)
    print(f"[stage] load_corpus_ids {task.corpus} ({time.perf_counter() - t0:.1f} s)", flush=True)
    sha[task.corpus] = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    print(f"[stage] dataset_sha256 {task.corpus} {sha[task.corpus]}", flush=True)
    samples = frontier.windows(ids, task.ctx, task.window, task.n_samples)
    if not samples:
        print(f"[T={task.ctx}] corpus too short for {task.n_samples} windows", flush=True)
        return []
    print(
        f"[T={task.ctx}] {len(samples)} window(s) of {task.ctx}+{task.window} on {task.corpus}",
        flush=True,
    )
    rows = frontier.run_ppl(
        arms,
        model,
        samples,
        task.ctx,
        chunk=task.chunk,
        n=n,
        h_kv=h_kv,
        device=device,
        corpus=task.corpus,
    )
    # `_cell`'s timing line for this axis: one per (arm, ctx) sweep, keyed by the ppl
    # TASK name (`ppl_*`, which no retrieval sub-task is called) so `harvest` can fold
    # both axes into one `manifest.cell_elapsed_s` without pooling two cells. `n` is the
    # windows the sweep attempted -- a failed arm prints its seconds too.
    for r in rows:
        print(
            f"[stage] cell arm={r['method']} task={task.name} ctx={task.ctx}"
            f" elapsed_s={r['elapsed_s']:.1f} n={len(samples)}",
            flush=True,
        )
    return rows


def _ppl_record(pod: PodCfg, row: dict[str, Any]) -> PplRecord:
    return {
        "model": pod.model,
        "arm": str(row["method"]),
        "ctx": int(row["T"]),
        "ppl": float(row["ppl"]),
        "ratio": float(row["ratio_fp16"]),
        "sbits": float(row["ratio_stored_bits"]),
        "tok_eq": float(row["tok_equiv_per_layer"]),
        "corpus": str(row["corpus"]),
        "source": f"{pod.name}:run",
    }


def _pplw_records(pod: PodCfg, row: dict[str, Any]) -> list[PplwRecord]:
    """The per-window NLLs behind one sweep -- a different schema, a different file."""
    return [
        {
            "model": pod.model,
            "arm": str(row["method"]),
            "ctx": int(row["T"]),
            "window_idx": i,
            "ntok": int(ntok),
            "nll_sum_nats": float(v) * int(ntok),
            "corpus": str(row["corpus"]),
            "source": f"{pod.name}:run",
        }
        for i, (v, ntok) in enumerate(zip(row["window_nlls"], row["window_toks"], strict=True))
    ]


def _finish(
    out: Path,
    records: dict[str, int],
    errors: int,
    wall_clock_s: float,
    corpora: dict[str, str] | None = None,
) -> None:
    """Fold what the run produced into the manifest `pod.py run` wrote at launch.

    The diagnostics the axes drained from their caches (`records.emit_diag`) are written
    here, the pod's last act, so ``diag.jsonl`` lands beside the other records whether
    the harvest reads this directory or replays the log. The buffer is cleared: it is
    process-global, and a second pod in one process must not inherit the first's rows.
    """
    if DIAG_ROWS:
        write_jsonl(out / "diag.jsonl", DIAG_ROWS)
        records["diag.jsonl"] = len(DIAG_ROWS)
        DIAG_ROWS.clear()
    path = out / "manifest.json"
    m: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"pod": out.name}
    if corpora:  # the launch-time manifest leaves `dataset_sha256` empty for the run
        m["dataset_sha256"] = {**m.get("dataset_sha256", {}), **corpora}
    m["records"] = records
    m["errors"] = errors
    m["wall_clock_s"] = wall_clock_s
    m["harvested_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(f"{out}: " + ", ".join(f"{k}={v}" for k, v in records.items()), flush=True)
