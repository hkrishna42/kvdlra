"""Per-trial, per-cell and per-ppl records: the only unit `make tables` reads.

``[trial]`` lines are the Bernoulli outcomes the pods printed (one per task x ctx x arm
x seed x trial); ``[task ctxN] arm acc= ... n=`` lines are pooled cells; ``arm [T=ctx]
ppl=...`` lines are perplexity sweeps; ``[pplw]`` lines are the un-pooled per-window
NLLs behind one of those sweeps, and ``[diag]`` lines are JSON diagnostics. All the
regexes are the emitters' formats from Week 11/17/18/19 (previously duplicated in six
reader scripts: w10_parse_logs, w11_merge, w17_intervals, w18_intervals, w19_a2_misses,
w19_fork_report -- and, until the L0.5 fix round, in ``scripts/pod.py``).

A perplexity sweep leaves TWO artifacts, and they are different records in different
files: the aggregate ``PplRecord`` (``ppl.jsonl``) and the per-window ``PplwRecord``
(``pplw.jsonl``). Neither substitutes for the other.

``generator``/``haystack_id``/``depth``/``code_family``/``prompt_sha256``/``error`` are
carried in the schema but are ``None`` for the archived paper-v1 records: the v1
emitters never printed them, and the archive is not re-converted to invent them. Read
them with ``.get`` -- an archived row has the key absent, not null. The runner has
printed them on the ``[trial]`` line since L2.3b, so a harvested pod carries them too.

``generator`` is part of a cell's identity, not decoration: the in-house and official
RULER generators reuse the sub-task names ``niah_multivalue`` and ``vt`` at the same
context length, so a pod running both pools two different benchmarks into one key
unless the generator separates them (``scripts/pod.py``'s ``_expected_cells``).
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, TypedDict, TypeVar

K = TypeVar("K")  # a `window_bits` bin key: the caller chooses what identifies a sweep

# `generator=`, the four pairing fields `hay= depth= code= sha=` (L2.3b) and `error=`
# are appended by `kvdlra.eval.runner`, in that order; no v1 log has any of them, so all
# are optional -- the harvest fills the generator from the pod's task configs when the
# line does not say, a pairing field the generator did not set prints as `-`, and a row
# with no `error=` did not raise. `error=` stays last and unanchored: the message can
# hold anything, and a trailing field a v1 log carried but this format does not name
# must not stop the line from parsing.
TRIAL_RE = re.compile(
    r"^\[trial\] task=(\S+) ctx=(\d+) arm=(\S+) seed=(\d+) trial=(\d+) hit=([01]) frac=([0-9.]+)"
    r"(?: generator=(\S+))?(?: hay=(\S+))?(?: depth=(\S+))?(?: code=(\S+))?(?: sha=(\S+))?"
    r"(?: error=(.*))?"
)
# `ratio=`/`sbits=` are absent from a cell whose every trial raised: there is no ratio to
# average, and the `ratio=nan sbits=nan` that printed instead matched nothing at all, so
# the cell vanished from a log harvest. Such a row carries `errors=` instead.
CELL_RE = re.compile(
    r"^\[([A-Za-z0-9_]+) ctx(\d+)\] (\S+)\s+acc=([0-9.]+) recall=([0-9.]+)(?: ratio=([0-9.]+))?"
    r"(?: sbits=([0-9.]+))?(?: n=(\d+))?(?: errors=(\d+))?"
)
# `batch=` is only on a `latency`-axis line (the ppl axis has no batch sweep); optional,
# and placed before `error=`, which must stay last since the exception message can
# contain anything, colons included.
ERROR_RE = re.compile(r"^\[error\] axis=(\S+) arm=(\S+) ctx=(\d+)(?: batch=(\d+))? error=(.*)$")
# Leading whitespace varies (0 or 2 spaces) across pods; tok_eq/layer and sbits are
# each sometimes absent. Verified against every `ppl=` line in results/*-lines.txt
# (175/175 match) -- see results/w11-table-ppl-lines.txt (no leading space, no sbits),
# results/w17-qwen-lines.txt (2-space, no sbits) and results/w18-*-ppl-lines.txt
# (2-space, with sbits).
# `corpus=` is appended LAST by `frontier._log_row` (after the optional `eff_rank=`) and
# is OPTIONAL, on the same basis as `sbits=`/`eff_rank=`: no archived line has it, and
# PPL_RE is a prefix match with no `$` anchor, so a trailing field never stops a line
# from parsing. `.*?` before it skips over `eff_rank=` when present.
PPL_RE = re.compile(
    r"^\s*(\S+)\s+\[T=(\d+)\] ppl=([0-9.]+)(?: tok_eq/layer=([0-9.]+))? .*?ratio=([0-9.]+)"
    r"(?: sbits=([0-9.]+))?(?:.*? corpus=(\S+))?"
)
# The emitter's own documented format (frontier._log_pplw, pinned by
# tests/test_w15_pplw.py): one line per (arm, T), except that a would-be >400-char line
# splits into ``part=i/N`` fragments of 8 values, because `vastai logs` truncates a line
# at ~500 chars. Dropping the fragments -- which is what the first harvest did -- loses
# the whole sweep silently.
# `corpus=` is appended LAST (frontier._log_pplw) and is OPTIONAL: no paper-v1 log has
# it -- v1 scored one corpus and never said so -- and an archived sweep must keep parsing.
PPLW_RE = re.compile(
    r"^\[pplw\] T=(\d+) (\S+) ntok=(\d+)(?: part=(\d+)/(\d+))? nlls=([0-9.,]+)"
    r"(?: corpus=(\S+))?$"
)
# `kvdlra.eval.latency.run_latency`'s own print. `weights_gb=` sits between `peak_gb=` and
# `kv_peak_gb=` but is not part of `LatencyRecord` -- it is the subtrahend the kv_*_gb
# figures already removed -- so it is matched, not captured. `batch=` is OPTIONAL: the nine
# archived Week-20 lines (results/paper-v1/w19-sysfix-llama/raw/) predate it and were batch
# 1, and `make kernel_smoke` prints them beside the re-measured rows (prereg §2 (a)).
# `kv_resident_gb=` (L4.7) and `backend=` (L4.fw1) are appended LAST, in that order, and
# are optional for the same reason -- `backend=` is printed only by a bug arm that ran the
# factored kernel, so every other row (and every row logged before it existed) has none.
LATENCY_RE = re.compile(
    r"^\[latency ctx(\d+)\] (\S+)\s+ms/tok=([0-9.]+) mean=([0-9.]+) max=([0-9.]+) "
    r"spikes=(\d+) resident_gb=([0-9.]+) peak_gb=([0-9.]+) weights_gb=[0-9.]+"
    r" kv_peak_gb=([0-9.]+)(?: batch=(\d+))?(?: kv_resident_gb=([0-9.]+))?"
    r"(?: backend=(\S+))?"
)
# `kvdlra.eval.kernel_check.format_line`'s own print. `-` stands for an absent number (an
# errored prompt has no diff and no mismatch step); `error=` stays last and unanchored.
# `backend=` (L4.fw1) sits between `sha=` and the tail and is OPTIONAL: the kernel_smoke
# pod's own records were logged before the field existed and must keep parsing (as None).
# The five Amendment-2 fields (A2.5) follow `backend=`, each OPTIONAL for the same reason --
# instance 51903816's rows predate them and must keep parsing (as None). A field may hold a
# negative logit (`kernel_logit_for_ref_argmax`), so `(\S+)` matches it and `-` alone is None.
KERNEL_CHECK_RE = re.compile(
    r"^\[kernel_check prompt=(\d+) arm=(\S+) ctx=(\d+) n_new=(\d+) match=([01])"
    r" first_mismatch=(\S+) max_abs_diff=(\S+) worst_layer=(\S+) sha=(\S+)"
    r"(?: backend=(\S+))?(?: rel_max_diff=(\S+))?(?: rel_worst_layer=(\S+))?"
    r"(?: ref_max=(\S+))?(?: gap_at_mismatch=(\S+))?(?: kernel_logit_for_ref_argmax=(\S+))?"
    r"(?: error=(.*))?"
)
# The payload is matched loosely and `json.loads` is the arbiter: a `\{.*\}` regex
# could not see a row `vastai logs` cut in half at all, so a truncated diagnostic
# was not even counted as skipped.
DIAG_RE = re.compile(r"^\[diag\] (.+?)\s*$")


class TrialRecord(TypedDict):
    model: str
    arm: str
    task: str
    ctx: int
    seed: int
    trial: int
    hit: int
    frac: float
    generator: str | None
    haystack_id: str | None
    depth: float | None
    code_family: str | None
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
    ratio: float | None
    sbits: float | None
    errors: int | None
    source: str


class PplwRecord(TypedDict):
    """One scored window of a perplexity sweep.

    ``nll_sum_nats`` is the window's total NLL in nats: the emitter prints a per-token
    MEAN over ``ntok`` tokens, and the sum is the quantity that pools without carrying
    the weights around (``ppl == exp(sum(nll_sum_nats) / sum(ntok))``).

    ``corpus`` is the held-out text the window came from -- absolute perplexity is not
    comparable across corpora, so a row that does not carry its own is not poolable with
    another. ``None`` on the archived rows: v1 printed no corpus."""

    model: str
    arm: str
    ctx: int
    window_idx: int
    ntok: int
    nll_sum_nats: float
    corpus: str | None
    source: str


class PplRecord(TypedDict):
    model: str
    arm: str
    ctx: int
    ppl: float
    ratio: float
    sbits: float | None
    tok_eq: float | None
    corpus: str | None
    source: str


class LatencyRecord(TypedDict):
    """One measured decode point: (arm, ctx, batch) -> steady-state cost and peak VRAM.

    ``ms_per_token_p50`` is the median over the timed steps left after the warm-up;
    ``spikes`` counts the steps above twice that median (the absorb-event rebuild and
    KIVI's per-step dequantize show up there). ``kv_peak_gb`` has the model weights
    subtracted, so it is the KV-attributable contrast, not process VRAM.

    ``kv_resident_gb`` is the post-prefill resident allocation minus the weights -- the
    Week-5 target's resident ratio (prereg §3, §4) -- ``None`` on a line printed before
    L4.7. ``backend`` is which factored-attention backend attended
    (`kvdlra.kernel.select_backend`): ``None`` for every arm that did not run the kernel,
    and for a kernel row logged before L4.fw1.
    """

    model: str
    arm: str
    ctx: int
    batch: int
    ms_per_token_p50: float
    ms_mean: float
    ms_max: float
    spikes: int
    resident_gb: float
    peak_gb: float
    kv_peak_gb: float
    kv_resident_gb: float | None
    backend: str | None
    source: str


class KernelCheckRecord(TypedDict):
    """One prompt of the kernel correctness check (prereg/kernel_smoke.md §4): whether the
    kernel's greedy decode matched the reconstruct path's token for token, the first step
    that did not, and the worst per-layer max|Δ| of the first kernel decode step.

    ``backend`` is which factored-attention backend attended
    (`kvdlra.kernel.select_backend`); ``None`` on an errored prompt and on a record logged
    before L4.fw1 -- the kernel_smoke pod's own rows among them.

    The five Amendment-2 fields (A2.5) make §4's precondition computable in the units the
    quantity is measured in: ``rel_max_diff`` = ``max_l(max|Δ_l| / max|ref_l|)`` (the relative
    per-layer bar), ``rel_worst_layer`` the layer achieving it, ``ref_max`` = ``max|ref|`` there
    (the denominator, so the ratio is auditable), ``gap_at_mismatch`` the reconstruct path's
    top-1 minus top-2 logit at the first mismatching step, and ``kernel_logit_for_ref_argmax``
    the kernel's logit for the token the reconstruct path chose there. All ``None`` on an
    errored prompt, on a prompt that matched (``gap``/``kernel_logit``), and on every record
    logged before Amendment 2 -- instance 51903816's rows among them, which is why the renderer
    reports a row missing them as not computable rather than passing it silently."""

    model: str
    arm: str
    ctx: int
    prompt: int
    n_new: int
    match: int
    first_mismatch: int | None
    max_abs_diff: float | None
    worst_layer: int | None
    rel_max_diff: float | None
    rel_worst_layer: int | None
    ref_max: float | None
    gap_at_mismatch: float | None
    kernel_logit_for_ref_argmax: float | None
    prompt_sha256: str
    backend: str | None
    error: str | None
    source: str


def _field(s: str | None) -> str | None:
    """A pairing field as the line printed it: absent (an archived row) or `-` (the
    generator set none) is None."""
    return None if s in (None, "-") else s


def parse_trial_lines(text: str, model: str, source: str) -> list[TrialRecord]:
    """Every ``[trial]`` line in ``text`` as a record citing ``<source>:<lineno>``."""
    out: list[TrialRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = TRIAL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, seed, trial, hit, frac, generator, hay, depth, code, sha, error = m.groups()
        out.append(
            {
                "model": model,
                "arm": arm,
                "task": task,
                "ctx": int(ctx),
                "seed": int(seed),
                "trial": int(trial),
                "hit": int(hit),
                "frac": float(frac),
                # Only what the line printed: a v1 row does not name its generator, and
                # `scripts/pod.py harvest` is where the pod config fills that gap.
                "generator": generator,
                "haystack_id": _field(hay),
                "depth": float(depth) if _field(depth) else None,
                "code_family": _field(code),
                "prompt_sha256": _field(sha),
                "error": error,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_cell_lines(text: str, model: str, source: str) -> list[CellRecord]:
    """Every pooled ``[task ctxN] arm acc=...`` line as a record. ``n``/``hits`` are
    ``None`` for pre-Week-18 rows, which printed no ``n=`` (no Bernoulli count to
    recover -- an interval cannot be computed from them); ``sbits`` (fp32-at-rest
    stored bits, the memory convention half the v1 tables print) is ``None`` for the
    rows that printed no ``sbits=``; ``ratio`` is ``None`` and ``errors`` is the trial
    count for a cell whose every trial raised, which has no footprint to report."""
    out: list[CellRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = CELL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, acc, _recall, ratio, sbits, n, errors = m.groups()
        n_i = int(n) if n is not None else None
        out.append(
            {
                "model": model,
                "arm": arm,
                "task": task,
                "ctx": int(ctx),
                "acc": float(acc),
                "n": n_i,
                "hits": round(float(acc) * n_i) if n_i is not None else None,
                "ratio": float(ratio) if ratio is not None else None,
                "sbits": float(sbits) if sbits is not None else None,
                "errors": int(errors) if errors is not None else None,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_error_lines(text: str, source: str) -> list[dict[str, object]]:
    """Every ``[error] axis=... arm=... ctx=... error=...`` line as a record.

    The retrieval axis carries a failure inside the trial record it still writes; the
    perplexity axis has no record to carry one -- an arm that raises produces no row at
    all -- so its failures are their own log line. `scripts/pod.py harvest` counts them
    into the manifest, which is how a harvest of the log agrees with the run that wrote
    it. No ``model``: the line is a failure, not evidence about a model. ``batch`` is
    the point's batch size for a ``latency`` axis line, ``None`` for every other axis."""
    out: list[dict[str, object]] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = ERROR_RE.match(line)
        if m:
            axis, arm, ctx, batch, err = m.groups()
            out.append(
                {
                    "axis": axis,
                    "arm": arm,
                    "ctx": int(ctx),
                    "batch": int(batch) if batch is not None else None,
                    "error": err,
                    "source": f"{source}:{i}",
                }
            )
    return out


def parse_ppl_lines(text: str, model: str, source: str) -> list[PplRecord]:
    """Every perplexity line (``arm [T=ctx] ppl=... ratio=...``) as a record.
    ``tok_eq`` and ``sbits`` are ``None`` when the source line did not print them
    (pre-Week-18 sweeps never printed ``sbits=``)."""
    out: list[PplRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = PPL_RE.match(line)
        if not m:
            continue
        arm, ctx, ppl, tok_eq, ratio, sbits, corpus = m.groups()
        out.append(
            {
                "model": model,
                "arm": arm,
                "ctx": int(ctx),
                "ppl": float(ppl),
                "ratio": float(ratio),
                "sbits": float(sbits) if sbits is not None else None,
                "tok_eq": float(tok_eq) if tok_eq is not None else None,
                # `None` on an archived line, which printed no `corpus=` at all.
                "corpus": corpus,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_pplw_lines(text: str, model: str, source: str) -> list[PplwRecord]:
    """Every ``[pplw]`` window as a record, split lines reassembled in part order.

    Fragments are gathered per ``(arm, ctx)`` -- the emitter prints one group per
    ``(arm, T)`` -- and only emitted once every part of the group has arrived. An
    incomplete set raises ``SystemExit``: a truncated log is a harvest to redo, not a
    sweep to publish short. Every window of a group cites the line the group started on.
    """
    out: list[PplwRecord] = []
    pending: dict[tuple[str, int], tuple[int, int, int, dict[int, list[float]]]] = {}

    def emit(
        arm: str, ctx: int, ntok: int, line: int, vals: list[float], corpus: str | None
    ) -> None:
        out.extend(
            {
                "model": model,
                "arm": arm,
                "ctx": ctx,
                "window_idx": j,
                "ntok": ntok,
                "nll_sum_nats": v * ntok,
                "corpus": corpus,
                "source": f"{source}:{line}",
            }
            for j, v in enumerate(vals)
        )

    for i, line in enumerate(text.splitlines(), 1):
        m = PPLW_RE.match(line)
        if not m:
            continue
        ctx, arm, ntok, part, n_parts, nlls, corpus = m.groups()
        vals = [float(x) for x in nlls.split(",") if x]
        if part is None:
            emit(arm, int(ctx), int(ntok), i, vals, corpus)
            continue
        key = (arm, int(ctx))
        n, tok, first, parts = pending.setdefault(key, (int(n_parts), int(ntok), i, {}))
        parts[int(part)] = vals
        if len(parts) == n:
            # One `_log_pplw` call prints a whole group, so every fragment of it carries
            # the same corpus -- the one on the fragment that completes it will do.
            emit(arm, key[1], tok, first, [v for j in sorted(parts) for v in parts[j]], corpus)
            del pending[key]
    if pending:
        missing = {
            f"{arm} T={ctx}": sorted(set(range(1, n + 1)) - set(parts))
            for (arm, ctx), (n, _tok, _first, parts) in pending.items()
        }
        raise SystemExit(f"{source}: incomplete [pplw] part set, missing {missing}")
    return out


def window_bits(
    rows: Iterable[PplwRecord], key: Callable[[PplwRecord], K], where: str
) -> dict[K, dict[int, float]]:
    """Per-window bits/token (``nll_sum_nats / (ntok * ln 2)``), binned by ``key``.

    The one implementation of the pairing every perplexity statistic starts from:
    ``scripts/tables.ppl_stats`` bins by ``(arm, ctx, corpus)`` and ``kvdlra.eval.gate1``
    by ``(family, ctx, corpus, tracker)``, and both must refuse the same records.

    A window scored twice would overwrite its own entry and shrink the mean's
    denominator without shrinking the window SET, so the set comparison in
    :func:`paired_window_bits` cannot see it. The usual cause is a second harvest
    appended to an existing ``pplw.jsonl``. ``where`` prefixes the message with the
    caller's location (``"ppl"``, or the pod directory).
    """
    out: dict[K, dict[int, float]] = defaultdict(dict)
    for r in rows:
        k = key(r)
        if r["window_idx"] in out[k]:
            raise ValueError(
                f"{where}: {r['arm']} ctx={r['ctx']} corpus={r.get('corpus')} carries"
                f" window_idx={r['window_idx']} twice -- the records are duplicated"
            )
        out[k][r["window_idx"]] = r["nll_sum_nats"] / (r["ntok"] * math.log(2))
    return dict(out)


def paired_window_bits(
    a: dict[int, float], b: dict[int, float], where: str, other: str
) -> list[float]:
    """``a - b`` per shared window, in ``window_idx`` order -- and a pairing that is not
    exact is refused, never silently intersected: an unpaired comparison of pooled
    numbers hides the effect it is measuring. ``where`` carries the caller's own
    identification of the ``a`` side, ``other`` names the ``b`` arm."""
    if set(a) != set(b):
        raise ValueError(
            f"{where} scored windows {sorted(set(a) ^ set(b))} that {other} did not"
            " (or the reverse) -- the pairing is broken"
        )
    return [a[i] - b[i] for i in sorted(a)]


def parse_latency_lines(text: str, model: str, source: str) -> list[LatencyRecord]:
    """Every ``[latency ctx<T>]`` decode-measurement line
    (``kvdlra.eval.latency.run_latency``) as a record -- the harvest-side counterpart
    to that print, the same role ``parse_ppl_lines`` plays for ``ppl=`` lines."""
    out: list[LatencyRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = LATENCY_RE.match(line)
        if not m:
            continue
        (ctx, arm, p50, mean, mx, spikes, resident_gb, peak_gb, kv_peak_gb, batch,
         kv_resident, backend) = m.groups()  # fmt: skip
        out.append(
            {
                "model": model,
                "arm": arm,
                "ctx": int(ctx),
                "batch": int(batch) if batch is not None else 1,
                "ms_per_token_p50": float(p50),
                "ms_mean": float(mean),
                "ms_max": float(mx),
                "spikes": int(spikes),
                "resident_gb": float(resident_gb),
                "peak_gb": float(peak_gb),
                "kv_peak_gb": float(kv_peak_gb),
                "kv_resident_gb": float(kv_resident) if kv_resident is not None else None,
                "backend": _field(backend),
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_kernel_check_lines(text: str, model: str, source: str) -> list[KernelCheckRecord]:
    """Every ``[kernel_check prompt=...]`` line as a record (the harvest-side counterpart to
    `kernel_check.format_line`). The log-only `[kernel_check layers ...]` and
    `[kernel_check mismatch ...]` lines do not match and are not records."""
    out: list[KernelCheckRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = KERNEL_CHECK_RE.match(line)
        if not m:
            continue
        (prompt, arm, ctx, n_new, match, first, diff, worst, sha, backend,
         rel_max, rel_layer, ref_max, gap, klogit, error) = m.groups()  # fmt: skip
        out.append(
            {
                "model": model,
                "arm": arm,
                "ctx": int(ctx),
                "prompt": int(prompt),
                "n_new": int(n_new),
                "match": int(match),
                "first_mismatch": None if first == "-" else int(first),
                "max_abs_diff": None if diff == "-" else float(diff),
                "worst_layer": None if worst == "-" else int(worst),
                # Amendment 2 (A2.5): absent (an archived row) or `-` (a matched prompt, or an
                # error row) is None; a present value parses, negatives included.
                "rel_max_diff": None if rel_max in (None, "-") else float(rel_max),
                "rel_worst_layer": None if rel_layer in (None, "-") else int(rel_layer),
                "ref_max": None if ref_max in (None, "-") else float(ref_max),
                "gap_at_mismatch": None if gap in (None, "-") else float(gap),
                "kernel_logit_for_ref_argmax": None if klogit in (None, "-") else float(klogit),
                "prompt_sha256": sha,
                "backend": _field(backend),
                "error": error,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_diag_lines(text: str, model: str, source: str) -> tuple[list[dict[str, object]], int]:
    """``[diag] {json}`` payloads (verbatim plus their ``model`` and ``source``), and how
    many ``[diag]`` lines were skipped because the payload is not a JSON object.

    Nothing reads them yet -- the diagnostics land in L1 -- so they are carried through
    unparsed rather than dropped, but the model is stamped in like every other record
    type: a rank or an orthogonality number means nothing without the family it came
    from. A line whose payload will not parse is skipped rather than fatal (diagnostics
    are never evidence for a number) -- but it is COUNTED and returned, because the
    usual cause is a log line cut in half by the fetch, and a silent skip made a
    truncated harvest look like a pod that printed no diagnostics at all.

    So an emitter (L1 writes these rows) keeps a ``[diag]`` row under ~400 chars, or
    splits it across ``part=i/N`` fragments the way ``[pplw]`` does: `vastai logs`
    truncates a line at ~500 chars, and the fragments of a split row reassemble where a
    halved one is only ever a count.
    """
    out: list[dict[str, object]] = []
    skipped = 0
    for i, line in enumerate(text.splitlines(), 1):
        m = DIAG_RE.match(line)
        if not m:
            continue
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            skipped += 1
            continue
        out.append({"model": model, **payload, "source": f"{source}:{i}"})
    return out, skipped


# --- the replay: the same lines a second time, at the end of the log -------------------

REPLAY_BEGIN, REPLAY_END = "===RECORDS_REPLAY_BEGIN===", "===RECORDS_REPLAY_END==="
# Which printed lines the replay repeats: the ones a parser above reads back, plus every
# `[stage]` line (the digests, the pod's card, the per-cell clock). `[diag]` is left out on
# purpose -- it is the volume, not the reading: the pre-flight's 4 MB tail came back as
# 15,381 diag rows and 138 of everything else, which is why 143 of its 240 `[trial]` rows
# were lost with the instance (D-011 addendum 10). ~1,000 lines for a Stage-1 pod.
_REPLAY_RES = (TRIAL_RE, CELL_RE, ERROR_RE, PPL_RE, PPLW_RE, LATENCY_RE, KERNEL_CHECK_RE)


def replayable(line: str) -> bool:
    """Is this a line the replay repeats?"""
    if line.startswith("[diag] "):
        return False
    return line.startswith("[stage] ") or any(r.match(line) for r in _REPLAY_RES)


class _Tee:
    """A stdout stand-in: writes through, and keeps the lines :func:`replayable` accepts.

    A filter on the stream rather than a call at each print site -- the record lines come
    from five modules (`scripts/pod.py`, this package's `runner`, `frontier`, `gen`,
    `latency`), and a print site added later would silently not be replayed. Attribute
    lookups fall through to the real stream, which is what a library asking stdout whether
    it is a tty gets.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.rows: list[str] = []
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            if replayable(line):
                self.rows.append(line)
        if len(self._buf) > 65_536:  # a `\r`-only writer (a progress bar) never sends
            self._buf = ""  # a newline -- discard rather than grow this forever
        return int(self.inner.write(s))

    def flush(self) -> None:
        self.inner.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@contextmanager
def replayed() -> Iterator[list[str]]:
    """Run a pod with its record lines teed, and print them again between the markers.

    `vastai logs` returns a ~4 MB TAIL, and the log dies with the instance: the pre-flight
    pod reached ALL_DONE with every trial recorded and 143 of its 240 `[trial]` rows never
    reached the laptop (D-011 addendum 10). Repeating the compact lines at the end puts
    every reading inside any tail that holds the last ~1,000 record lines. The block is
    printed on the way out of a raised run too -- a pod that crashed is exactly the one
    whose rows are worth keeping -- and `scripts/pod.py harvest` dedupes exact-duplicate
    lines before parsing, so the repeat is not a second set of records.
    """
    tee = _Tee(sys.stdout)
    try:
        with redirect_stdout(tee):
            yield tee.rows
    finally:
        print(REPLAY_BEGIN, flush=True)
        for line in tee.rows:
            print(line)
        print(REPLAY_END, flush=True)


# The diagnostic rows the eval axes have drained from the caches of THIS process, each
# stamped with its model and the axis that drained it. `kvdlra.eval.runner` writes them
# to ``results/<pod>/diag.jsonl`` at the end of the pod and clears the list; the printed
# ``[diag]`` line carries the same payload, so `pod.py harvest` recovers the same rows
# from a `vastai logs` capture when the results directory never left the instance.
DIAG_ROWS: list[dict[str, object]] = []


def emit_diag(
    rows: list[dict[str, object]],
    *,
    model: str,
    source: str,
    arm: str,
    ctx: int,
    task: str | None = None,
    idx: int | None = None,
) -> None:
    """Print one ``[diag] {json}`` line per row, and buffer the rows for the runner.

    The two artifacts every record type leaves, from one call: the log line (which
    :func:`parse_diag_lines` reads back, stamping the model and the line it came from)
    and the in-process row (which lands in ``diag.jsonl`` directly).

    The payload is the cache's row (the 11 fields of
    :meth:`kvdlra.cache.BugStreamingCache.drain_diag`) plus the four fields that say
    WHICH measurement it is: ``arm``, ``ctx``, ``task`` and ``idx`` (the sample or trial
    the surrounding loop is on). They go into the printed line as well as the buffered
    row, so a harvest off the log rebuilds exactly the same record -- without them a
    pod running eleven arms emits one undifferentiated stream of ranks and
    orthonormality errors that no row can be assigned to an arm. ~260 chars, still
    inside the ~400-char budget a log fetch leaves, so no ``part=i/N`` splitting.

    ``source`` is the axis that produced the rows (``ppl`` / ``ruler`` / ``longbench``).
    """
    stamp: dict[str, object] = {"arm": arm, "ctx": ctx, "task": task, "idx": idx}
    for row in rows:
        payload = {**row, **stamp}
        print("[diag] " + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        DIAG_ROWS.append({**payload, "model": model, "source": source})


@contextmanager
def drained(
    cache: object,
    model: Any,
    *,
    source: str,
    arm: str,
    ctx: int,
    task: str | None = None,
    idx: int | None = None,
) -> Iterator[None]:
    """Run one streaming measurement and emit the tripwire's rows when it ends.

    The rows leave the library here -- drained after the sample and before the cache is
    dropped, since nothing else ever reads them again -- and in a ``finally``, because the
    window worth reading most is the last one before an ``OrthonormalityError``: the cache
    flushes it before raising (L1.1), and an emit on the success path alone would let it
    die with the cache, leaving the trial recorded as an error with no diagnostics behind
    it. A cache of any other kind has nothing to drain, so the three axes wrap every arm.
    """
    # Imported HERE, not at module scope: `scripts/pod.py` imports this module for the
    # parsers alone, and `kvdlra.cache` pulls in torch + transformers -- ~6 s of import
    # on every `pod.py check` subprocess the tests spawn, for a path they never run.
    from kvdlra.cache import BugStreamingCache

    try:
        yield
    finally:
        if isinstance(cache, BugStreamingCache):
            emit_diag(
                cache.drain_diag(),
                model=str(model.name_or_path),
                source=source,
                arm=arm,
                ctx=ctx,
                task=task,
                idx=idx,
            )


def write_jsonl(
    path: Path,
    rows: list[TrialRecord]
    | list[CellRecord]
    | list[PplRecord]
    | list[PplwRecord]
    | list[LatencyRecord]
    | list[KernelCheckRecord]
    | list[dict[str, object]],  # the diagnostics, carried through unparsed
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
