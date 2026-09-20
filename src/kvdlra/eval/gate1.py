"""Gate 1: the tracker-swap contrasts and the branch rule that reads them.

The spec is ``prereg/gate1_tracker_swap_v2.md`` section 4, "The operationalization,
exactly as ``gate1_verdict()`` implements it", in code. It adds no statistic of its own
-- every one is a shipped :mod:`kvdlra.eval.stats` function -- and no threshold is a
literal: each is a named constant below, carrying the clause that fixes it.

The rule the plan wrote (``docs/plan/ICML2027_PLAN.md`` section 2, Gate 1 item 1.1, and
prereg section 4), with (a) ``isvd``, (c) ``fd``, (d) ``frozen``, (e) the family's
``nogist`` twin:

    "if (a) is not separated from (c)/(d) on any task and perplexity is within 0.02
    bits/token, the tracker is not the contribution -> Branch C. If (a) beats (d) and
    (e) on retrieval or perplexity in >=2 families with Holm-corrected p<0.05 -> the
    online-tracked gist is doing work -> Branch A/B."

Precedence, once every refusal has been read: **A/B, then C, then REFUSED, then
UNDECIDED**. The clause map -- one prereg clause per line, and where it lives:

- scope, "the verdict reads the 16K contrasts only"   :data:`VERDICT_CTX`
- rule 1, a family is separated (BOTH controls beaten, on retrieval or on a
  perplexity win whose +/-0.02 TOST fails)            :func:`_beats`
- the TOST's third state, ``not decidable``           :func:`_wide`, `stats.tost_decidable`
- rule 2, Branch A/B at >= 2 separated families       :data:`MIN_FAMILIES_FOR_AB`
- rule 3, Branch C, stated as what blocks it          ``blockers`` in :func:`gate1_verdict`
- rule 4, otherwise UNDECIDED, blockers listed        the same function's last branch
- the five refusals of "Refusal, and the ``--`` rule" :func:`_refusals`
- section 6's three Holm families, at their realised m :func:`_holm_by_group`
- section 6's task exclusion, set by amendment        :data:`EXCLUDED_TASKS`
- section 7 (e)'s ``prompt_sha256`` pairing invariant  :func:`_mismatched`

Three lane rulings shape this past section 4's literal wording, and the prereg's
Amendment 1a records all three (A1a.2, A1a.3):

- **R-L3-12** a refusal returns ``REFUSED``, a distinct branch value, never
  ``UNDECIDED`` ("the rule never ran" and "neither branch held" are different
  findings); and no refusal raises, so ``make gate1`` always has a table to print it in.
- **R-L3-15** refusals compose PER FAMILY: A/B is read on the families that remain, any
  refusal blocks C (a positive claim needs every family), REFUSED is what is left.
- **R-L3-16** a refused family's members leave EVERY Holm family BEFORE the correction,
  so the clean families are corrected at the m they would have had alone. The Holm
  family is the pod set of ONE ``make gate1`` invocation -- one stage per call.

Two blockers section 4 does not state, both of which can only withhold a C and never
manufacture one (prereg Amendment 1a, A1a.5): C needs every :data:`TASK_ORDER` task
present in every 16K family (``incomplete task set``), and an input with no 16K family
at all reads :data:`NO_MEMBERS`. Presence is all the verdict guards -- per-cell
n-completeness is ``scripts/pod.py check``'s job (prereg section 10).

**bf16 non-inferiority** (``prereg/bf16_gist.md`` section 4) rides the same pods and is
read here so that the harvest cannot skip it. It enters no Gate-1 family and moves no
Gate-1 p-value (that file's section 6), and its five clauses map as:

- the statistic, ``isvd`` (a) vs ``bf16`` (b) per (family, task)  :func:`bf16_contrasts`
- rule 1, a task is lost only at BOTH > 0.03 and Holm p < 0.05    :func:`_bf16_loses`
- rule 2, the family's +/-0.02 perplexity TOST                    :func:`_bf16_family`
- rule 3, pass / fail / REFUSED composed over the two families    :func:`bf16_verdict`
- rule 4's refusals, read first: an error row on either arm, a broken pairing, a missing
  cell, and section 6's ``not decidable`` TOST                    :func:`_bf16_refusals`
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast

from scipy.stats import ttest_1samp

from kvdlra.eval.config import load_arm
from kvdlra.eval.records import (
    PplRecord,
    PplwRecord,
    TrialRecord,
    paired_window_bits,
    read_jsonl,
    window_bits,
)
from kvdlra.eval.stats import Key, holm, mcnemar_exact, paired_bootstrap, tost, tost_decidable

ALPHA = 0.05  # Holm's level, and the TOSTs' -- section 6: uncorrected, intersection-union
PPL_DELTA_BITS = 0.02  # the equivalence margin the plan fixed (section 4)
MIN_FAMILIES_FOR_AB = 2  # ">= 2 families are separated" (rule 2)
VERDICT_CTX = 16384  # "the verdict reads the 16K contrasts only" (section 4, scope)
BYTE_MATCH_TOL = 0.05  # the nogist-to-isvd stored-bits ratio band, 1 +/- this (section 7 f)
REFERENCE = "isvd"  # arm (a): the tracked gist every contrast is taken against
PRIMARY_CONTROLS = ("frozen", "nogist")  # (d) and (e): rule 1 reads both
SECONDARY_CONTROLS = ("oja", "fd", "random")  # section 7 (a)'s family
C_CONTROLS = ("frozen", "fd")  # (d) and (c): the two arms rule 3 reads
RULE_ARMS = (REFERENCE, "frozen", "nogist", "fd")  # an error row on any of these refuses

BF16_PREREG = "prereg/bf16_gist.md"  # the reading below is its section 4, and only that
BF16 = "bf16"  # arm 6, `isvd_r64_h256_seed_bf16`: the gist stored at 16 bits
BF16_MARGIN = 0.03  # its section 4 (1), on `(a_favored - b_favored) / n_paired`
BF16_FAMILIES = 2  # its section 6's two Stage-1 model families, and section 4 (3)'s "both"

# Arm stem -> tracker label, in the prereg's section 3 arm order (which is also the order
# the table's rows take). Two no-gist stems, one per KV width: one H cannot serve both.
# A record carries the arm's `legacy_name` where its config sets one, so the record key
# is resolved through `load_arm` rather than assumed -- and an arm the map does not know
# is an error, never a silent skip (a dropped record would shrink a cell without
# shrinking its n).
ARM_TRACKER = {
    "full": "full",
    "isvd_r64_h256_seed": REFERENCE,
    "nogist_h2423": "nogist",
    "nogist_h4460": "nogist",
    "frozen_r64_h256_seed": "frozen",
    "fd_r64_h256_seed": "fd",
    "isvd_r64_h256_seed_bf16": "bf16",
    "oja_r64_h256_seed_tuned": "oja",
    "random_r64_h256_seed": "random",
}
FROZEN_ARM = "frozen_r64_h256_seed"
TRACKERS = tuple(dict.fromkeys(ARM_TRACKER.values()))  # the row order, deduplicated
FAMILIES = ("llama", "qwen", "mistral")  # the three model families, matched by substring
TASK_ORDER = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
# A task the pre-flight's ceiling rule excludes (`full` < 0.9 -- section 6, by
# amendment, which is the only thing that sets this). It leaves the primary retrieval
# family, the secondary family and both of C's per-task readings (16 - 4 = 12 and
# 24 - 6 = 18); the perplexity family has no task axis and is untouched. Its cells are
# still run, still scored and still rendered -- descriptively, in no family.
EXCLUDED_TASKS: frozenset[str] = frozenset()

CellKey = tuple[str, int, str, str]  # family, ctx, task, tracker
# The corpus is part of the perplexity key, exactly as in `tables.ppl_stats`: two ppl
# tasks can share a context length with different corpora (PG-19 validation and
# WikiText-103 test both ship a 16K task) and absolute perplexity is not comparable
# across them, so pairing window 7 of one against window 7 of the other would compare
# two different texts.
SweepKey = tuple[str, int, str | None, str]  # family, ctx, corpus, tracker

NO_MEMBERS = f"no 16K family in the records (the verdict reads ctx {VERDICT_CTX} only)"


@dataclass(frozen=True)
class Contrast:
    """One retrieval member: the exact paired McNemar of two cells over their shared
    ``(seed, trial)`` keys. ``p`` is ``None`` when the two cells share no key at all --
    a broken pairing, not a result. ``p_holm`` is ``None`` for a member that left its
    family (section 6: a cell holding an ``error`` row does).

    ``dropped_keys`` are the keys section 7 (e)'s pairing invariant removed from THIS
    member: the two arms wrote different ``prompt_sha256`` for them, or one wrote none,
    so they are not the byte-identical prompts the contrast is defined on. They shrink
    ``n_paired`` and are reported with the member (section 4, section 6)."""

    family: str
    ctx: int
    task: str
    a: str
    b: str
    primary: bool
    n_paired: int
    a_favored: int
    b_favored: int
    p: float | None
    p_holm: float | None
    errors_a: int
    errors_b: int
    dropped_keys: tuple[Key, ...]


@dataclass(frozen=True)
class PplContrast:
    """One perplexity member: ``isvd`` minus another tracker over the paired per-window
    bits/token. ``d_bits < 0`` is the r64 arm ahead. ``p`` is the two-sided paired
    t-test (the Holm member); ``p_tost`` the larger of the two one-sided p-values,
    reported beside the boolean no rule reads it through.

    ``equivalent`` is section 6's TOST boolean and ``decidable`` says which ``False`` it
    is: ``False``/``False`` is "not decidable" -- the realised spread cannot fit inside
    +/-0.02 even at ``d_bits`` = 0 -- and is "never ... a pass, never a quiet fail".
    ``s`` is the paired SD the bound is read against, reported beside the CI because
    section 6 requires the verdict entry to carry both wherever this state decides
    anything."""

    family: str
    ctx: int
    corpus: str | None
    a: str
    b: str
    primary: bool
    n_windows: int
    d_bits: float
    lo: float
    hi: float
    s: float
    p: float
    p_holm: float | None
    p_tost: float
    equivalent: bool
    decidable: bool


@dataclass(frozen=True)
class Verdict:
    branch: str  # "A/B" | "C" | "REFUSED" | "UNDECIDED" -- R-L3-12, see the header
    reason: str
    families_separated: list[str]
    members: list[str]  # the members that decided it (prereg section 4's snippet)


@dataclass(frozen=True)
class Bf16Verdict:
    """The bf16 arm's reading (``prereg/bf16_gist.md`` section 4 (3)): ``pass`` iff both
    families pass, ``fail`` iff one fails and none is refused, ``REFUSED`` whenever any
    family is refused "whatever the other family shows", and ``not run`` where the arm
    wrote no 16K record at all (that file's section 9).

    ``families`` carries the per-family state -- ``PASS``, ``FAIL`` or ``REFUSED (why)``
    -- and ``members`` the lines behind it, which is what the DECISIONS entry that file's
    section 10 asks for reads."""

    overall: str
    reason: str
    families: dict[str, str]
    members: list[str]


@dataclass(frozen=True)
class Gate1Data:
    """One stage's records, indexed for the contrasts. One pod per model family, as
    ``PodCfg`` forces, so a family is a pod and no contrast crosses a pod boundary."""

    pods: dict[str, str]  # family -> the results/ directory it was read from
    arms: dict[tuple[str, str], str]  # (family, tracker) -> the arm name in the records
    hits: dict[CellKey, dict[Key, int]]  # one cell's Bernoulli outcomes
    sha: dict[CellKey, dict[Key, str | None]]  # one cell's prompt digests (section 7 e)
    errors: dict[CellKey, list[str]]  # one cell's exception texts, in record order
    bits: dict[SweepKey, dict[int, float]]  # one sweep's bits/token, by window_idx
    # (family, tracker) -> median `sbits`, which is `ratio_stored_bits`: an arm's stored
    # bits relative to `full` (runner._ppl_record). Section 7 (f)'s refusal reads the
    # ratio of two of them, so the common denominator cancels.
    sbits: dict[tuple[str, str], float]
    frozen_defects: dict[str, list[dict[str, object]]]  # family -> repairs past the freeze
    # Section 4's five refusals, computed ONCE in `load` because everything downstream
    # reads them before it does anything else (R-L3-16, header): family -> its refusal
    # lines, and the secondary-arm error lines that refuse nothing.
    refused: dict[str, list[str]]
    notes: list[str]


def _family(model: str) -> str:
    hit = [f for f in FAMILIES if f in model.lower()]
    if len(hit) != 1:
        raise ValueError(f"model {model!r} is not one of the Gate-1 families {FAMILIES}")
    return hit[0]


def _record_keys() -> dict[str, str]:
    """Record arm string -> tracker label, resolved through the arm configs."""
    return {(load_arm(stem).legacy_name or stem): t for stem, t in ARM_TRACKER.items()}


def _freeze_after() -> int:
    """The frozen arm's own ``freeze_after``, read from its config so the refusal cannot
    drift from the mechanism it checks."""
    return int(load_arm(FROZEN_ARM).cache["freeze_after"])


def _frozen_defects(path: Path, arm: str, freeze_after: int) -> list[dict[str, object]]:
    """The frozen arm's ``diag`` rows that repaired after its freeze, minus the exempt one.

    Streamed line by line, and only these rows are kept: a Stage-1 ``diag.jsonl`` holds
    ~370,000 rows (prereg section 8) and this is the only reading anything takes from it.

    The exemption is section 7 (c)'s: "a ``[diag]`` row covers a 64-absorb window and its
    ``tokens_seen`` is the last absorb in it, so the **first row per (sample, layer) with
    ``tokens_seen > 4096``** is the window that straddles the freeze and can legitimately
    carry a repair from an absorb before it. Every row after that one must read false."
    The first row past the freeze is the earliest of ALL of them, repairing or not, which
    is why the minimum is taken over every row and not only over the repairs.

    A sample is ``(seed, task, idx, layer)``. ``records.emit_diag`` stamps no ``seed``
    today and Gate 1's tasks run one (``seeds: [0]``, section 3), so the key degenerates
    to the three fields the rows carry -- but a two-seed pod would otherwise merge trial
    ``idx`` 3 of both seeds into one sample and exempt a repair on the second.
    """
    if not path.is_file():
        return []
    first: dict[tuple[object, object, object, object], int] = {}
    repairs: list[dict[str, object]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = cast(dict[str, object], json.loads(line))
            if row.get("arm") != arm or int(cast(int, row.get("tokens_seen", 0))) <= freeze_after:
                continue
            seen = int(cast(int, row["tokens_seen"]))
            sample = _sample(row)
            first[sample] = min(first.get(sample, seen), seen)
            if row.get("fixed_k") or row.get("fixed_v"):
                repairs.append(row)
    return [r for r in repairs if r["tokens_seen"] != first[_sample(r)]]


def _sample(row: dict[str, object]) -> tuple[object, object, object, object]:
    return (row.get("seed"), row.get("task"), row.get("idx"), row.get("layer"))


def load(pod_dirs: list[Path]) -> Gate1Data:
    """The records of one stage's pods, indexed by (family, ctx, task, tracker).

    ``trials.jsonl`` is required; ``pplw.jsonl`` (the per-window rows every perplexity
    statistic reads -- never the pooled ``ppl.jsonl`` number), ``ppl.jsonl`` (the only
    committed carrier of an arm's ``sbits``) and ``diag.jsonl`` are read when present.
    Two pods of one model family are refused: they would pool two pods into one cell,
    and "no contrast in this file crosses a pod boundary" (prereg section 3). Section
    4's five refusals are computed here, once, and ride on the result.
    """
    by_key, freeze_after = _record_keys(), _freeze_after()
    # The frozen arm as the RECORDS name it, so the dispatch refusal keeps reading the
    # right rows if that arm ever gains a `legacy_name`.
    frozen_key = next(k for k, t in by_key.items() if t == "frozen")
    pods: dict[str, str] = {}
    arms: dict[tuple[str, str], str] = {}
    hits: dict[CellKey, dict[Key, int]] = defaultdict(dict)
    sha: dict[CellKey, dict[Key, str | None]] = defaultdict(dict)
    errors: dict[CellKey, list[str]] = defaultdict(list)
    bits: dict[SweepKey, dict[int, float]] = {}
    raw_sbits: dict[tuple[str, str], list[float]] = defaultdict(list)
    frozen_defects: dict[str, list[dict[str, object]]] = {}

    for d in pod_dirs:
        src = d / "pplw.jsonl"
        manifest = json.loads((d / "manifest.json").read_text())
        family = _family(str(manifest["model"]))
        if family in pods:
            raise ValueError(f"{d}: a second {family} pod ({pods[family]}) -- one pod per family")
        pods[family] = d.name
        for r in (cast(TrialRecord, x) for x in read_jsonl(d / "trials.jsonl")):
            tracker = _tracker(by_key, r["arm"], d / "trials.jsonl")
            arms[(family, tracker)] = r["arm"]
            cell: CellKey = (family, r["ctx"], r["task"], tracker)
            key: Key = (r["seed"], r["trial"])
            if key in hits[cell]:
                raise ValueError(f"{d}: {r['arm']} {r['task']} carries (seed, trial)={key} twice")
            hits[cell][key] = r["hit"]
            sha[cell][key] = r["prompt_sha256"]
            if r["error"]:
                errors[cell].append(r["error"])
        bits.update(
            window_bits(
                (cast(PplwRecord, x) for x in _maybe(src)),
                partial(_sweep_key, by_key, family, src),
                str(d),
            )
        )
        for p in (cast(PplRecord, x) for x in _maybe(d / "ppl.jsonl")):
            if p.get("sbits") is not None:
                key_s = (family, _tracker(by_key, p["arm"], d / "ppl.jsonl"))
                raw_sbits[key_s].append(float(cast(float, p["sbits"])))
        frozen_defects[family] = _frozen_defects(d / "diag.jsonl", frozen_key, freeze_after)

    data = Gate1Data(
        pods=pods,
        arms=arms,
        hits=dict(hits),
        sha=dict(sha),
        errors=dict(errors),
        bits=bits,
        sbits={k: statistics.median(v) for k, v in raw_sbits.items()},
        frozen_defects=frozen_defects,
        refused={},
        notes=[],
    )
    # The refusals, once (R-L3-16: everything downstream reads them before Holm).
    refused, notes = _refusals(data, _draft_retrieval(data))
    return replace(data, refused=refused, notes=notes)


def _sweep_key(by_key: dict[str, str], family: str, src: Path, w: PplwRecord) -> SweepKey:
    """One perplexity sweep's key, for :func:`~kvdlra.eval.records.window_bits`."""
    return (family, w["ctx"], w.get("corpus"), _tracker(by_key, w["arm"], src))


def _maybe(path: Path) -> list[dict[str, object]]:
    return read_jsonl(path) if path.is_file() else []


def _tracker(by_key: dict[str, str], arm: str, where: Path) -> str:
    """The tracker an arm string names. An arm the map does not know is an error in every
    record file, not only in ``trials.jsonl``: a silently bucketed perplexity sweep would
    drop a contrast rather than fail."""
    if arm not in by_key:
        raise ValueError(f"{where}: arm {arm!r} is not a Gate-1 arm")
    return by_key[arm]


def _holm_by_group(
    draft: list[Contrast] | list[PplContrast],
    group: dict[int, object],
) -> list[float | None]:
    """Holm inside each group of eligible members, ``None`` for every other member.

    ``group`` maps a member's index to its family key (absent = in no family); the
    realised m is how many indices share a key, which the table prints. A member with an
    ``error`` row, and every member of a refused family, is absent -- R-L3-16, header.
    The key is the caller's, and it is the INVOCATION and never the model family:
    section 6's primary retrieval family is 16 POOLED across Llama and Qwen, so grouping
    by (family, ctx) would split it into two 8s.
    """
    # ponytail: one stage per call is the caller's contract, not a check here -- a
    # caller that passed Stage 1 and Stage 2 in one invocation would pool them. Take the
    # stage as an explicit input if `make gate1` ever grows a multi-stage mode.
    out: list[float | None] = [None] * len(draft)
    members: dict[object, list[int]] = defaultdict(list)
    for i, key in group.items():
        members[key].append(i)
    for idx in members.values():
        raw = [cast(float, draft[i].p) for i in idx]
        for i, adj in zip(idx, holm(raw), strict=True):
            out[i] = adj
    return out


def _reference_cells(data: Gate1Data) -> list[tuple[str, int, str]]:
    """The (family, ctx, task) cells the reference arm scored, minus the excluded tasks --
    every contrast in this module is taken against that arm, so this is the loop both
    :func:`_draft_retrieval` and :func:`bf16_contrasts` run."""
    return sorted(
        {(f, c, t) for f, c, t, k in data.hits if k == REFERENCE and t not in EXCLUDED_TASKS}
    )


def _member(data: Gate1Data, f: str, c: int, t: str, b: str) -> Contrast:
    """One retrieval member: the exact paired McNemar of ``isvd`` against ``b`` over the
    keys the two cells share and section 7 (e)'s digest kept (:func:`_mismatched`)."""
    a_cell, b_cell = data.hits[(f, c, t, REFERENCE)], data.hits[(f, c, t, b)]
    dropped = _mismatched(data, (f, c, t, REFERENCE), (f, c, t, b))
    keep = {k: v for k, v in a_cell.items() if k not in dropped}
    m = mcnemar_exact(keep, {k: v for k, v in b_cell.items() if k not in dropped})
    return Contrast(
        family=f, ctx=c, task=t, a=REFERENCE, b=b, primary=b in PRIMARY_CONTROLS,
        n_paired=m["n_paired"] if m else 0,
        a_favored=m["a_favored"] if m else 0,
        b_favored=m["b_favored"] if m else 0,
        p=m["p_value"] if m else None, p_holm=None,
        errors_a=len(data.errors.get((f, c, t, REFERENCE), [])),
        errors_b=len(data.errors.get((f, c, t, b), [])),
        dropped_keys=dropped,
    )  # fmt: skip


def _draft_retrieval(data: Gate1Data) -> list[Contrast]:
    """Every retrieval member before Holm runs -- the correction needs the refusals
    first (R-L3-16, header) and refusal 2 is read off these members' ``p is None``."""
    return [
        _member(data, f, c, t, b)
        for f, c, t in _reference_cells(data)
        for b in (*PRIMARY_CONTROLS, *SECONDARY_CONTROLS)
        if (f, c, t, b) in data.hits
    ]


def retrieval_contrasts(data: Gate1Data) -> list[Contrast]:
    """Every retrieval member: ``isvd`` vs ``frozen``/``nogist`` (primary) and vs
    ``oja``/``fd``/``random`` (secondary), per family x ctx x task -- the exact paired
    McNemar over the ``(seed, trial)`` keys the two cells share, "the r64 arm as *a*".

    Section 7 (e)'s pairing invariant is enforced here, not assumed (:func:`_mismatched`):
    a dropped key shrinks the member's paired n and never removes the member -- except at
    the limit, where every key goes and refusal 2 catches the ``None``.

    Holm runs over two families -- primary and secondary -- separately per context
    length, at the realised m the non-refused families leave (section 6; R-L3-16).
    """
    draft = _draft_retrieval(data)
    eligible = {
        i: (x.ctx, x.primary)
        for i, x in enumerate(draft)
        if x.p is not None and (x.errors_a, x.errors_b) == (0, 0) and x.family not in data.refused
    }
    adjusted = _holm_by_group(draft, cast(dict[int, object], eligible))
    return [replace(x, p_holm=adjusted[i]) for i, x in enumerate(draft)]


def _mismatched(data: Gate1Data, a: CellKey, b: CellKey) -> tuple[Key, ...]:
    """The shared keys whose two ``prompt_sha256`` disagree, or whose digest is missing
    on either side -- section 7 (e), read per member as section 4 words it."""
    a_sha, b_sha = data.sha.get(a, {}), data.sha.get(b, {})
    return tuple(
        k for k in sorted(set(a_sha) & set(b_sha)) if a_sha[k] is None or a_sha[k] != b_sha[k]
    )


def ppl_contrasts(data: Gate1Data) -> list[PplContrast]:
    """``isvd`` minus every other tracker on the paired per-window bits/token.

    Windows are paired by ``(ctx, corpus, window_idx)`` through the same
    :func:`~kvdlra.eval.records.paired_window_bits` ``tables.ppl_stats`` uses, so both
    refuse the same records. Holm runs over the PRIMARY members of the non-refused
    families only (section 6's 4-member family, per context length; R-L3-16); the
    secondary contrasts are "uncorrected and descriptive, except the ``fd`` TOSTs the
    C branch names".
    """
    draft: list[PplContrast] = []
    # `key=` tolerates the `None` corpus the archived rows carry.
    for f, c, corpus, _ in sorted(
        (k for k in data.bits if k[3] == REFERENCE), key=lambda k: (k[0], k[1], k[2] or "", k[3])
    ):
        for b in TRACKERS:
            if b == REFERENCE or (f, c, corpus, b) not in data.bits:
                continue
            d = paired_window_bits(
                data.bits[(f, c, corpus, REFERENCE)],
                data.bits[(f, c, corpus, b)],
                f"ppl: {f} ctx={c} corpus={corpus} {b}",
                REFERENCE,
            )
            mean_d, lo, hi = paired_bootstrap(d)
            p_lo, p_hi, equivalent = tost(d, PPL_DELTA_BITS, ALPHA)
            draft.append(
                PplContrast(
                    family=f, ctx=c, corpus=corpus, a=REFERENCE, b=b,
                    primary=b in PRIMARY_CONTROLS,
                    n_windows=len(d), d_bits=mean_d, lo=lo, hi=hi,
                    s=statistics.stdev(d), p=float(ttest_1samp(d, 0.0).pvalue),
                    p_holm=None, p_tost=max(p_lo, p_hi), equivalent=equivalent,
                    decidable=tost_decidable(d, PPL_DELTA_BITS, ALPHA),
                )
            )  # fmt: skip
    eligible = {
        i: (x.ctx, x.corpus)
        for i, x in enumerate(draft)
        if x.primary and x.family not in data.refused
    }
    adjusted = _holm_by_group(draft, cast(dict[int, object], eligible))
    return [replace(x, p_holm=adjusted[i]) for i, x in enumerate(draft)]


def _refusals(data: Gate1Data, retrieval: list[Contrast]) -> tuple[dict[str, list[str]], list[str]]:
    """The five refusals, each scoped to the family it fires on (R-L3-15, header), and
    the secondary-arm error lines that refuse nothing (section 4: "an error on arms 6-8
    removes that arm's secondary members and nothing else").

    Read from the records and from ``retrieval``'s UNADJUSTED fields -- refusal 2 is
    ``p is None``, which no correction changes -- so this is computable BEFORE Holm,
    which R-L3-16 requires. :func:`load` calls it, once.
    """
    refused: dict[str, list[str]] = defaultdict(list)
    notes: list[str] = []
    # Refusal 1: an error row on one of the four arms the rule reads.
    for (family, ctx, task, tracker), errs in sorted(data.errors.items()):
        if ctx != VERDICT_CTX or not errs:
            continue
        arm = data.arms[(family, tracker)]
        line = f"{arm} has {len(errs)} error records ({family}/{task}: {errs[0]})"
        (refused[family] if tracker in RULE_ARMS else notes).append(line)
    # Refusal 2: a primary member with no shared pairing key at all.
    for c in retrieval:
        if c.ctx == VERDICT_CTX and c.primary and c.p is None:
            refused[c.family].append(f"{c.family}/{c.task}: {c.a} vs {c.b} {_why_unpaired(c)}")
    for family in sorted({f for f, ctx, _, _ in data.hits if ctx == VERDICT_CTX}):
        # Refusal 3: the frozen arm still repairing past its freeze.
        defects = data.frozen_defects.get(family, [])
        if defects:
            refused[family].append(
                f"{family}: frozen dispatch -- {len(defects)} diag rows repaired past"
                f" freeze_after={_freeze_after()}, first {defects[0]}"
            )
        # Refusal 4: the no-gist twin off the byte match its arm file solves for.
        if (family, "nogist") not in data.arms:
            continue
        gist = data.sbits.get((family, REFERENCE))
        twin = data.sbits.get((family, "nogist"))
        if gist is None or twin is None:
            refused[family].append(f"{family}: byte match not measured (no sbits on the records)")
        elif abs(twin / gist - 1.0) > BYTE_MATCH_TOL:
            refused[family].append(
                f"{family}: byte match -- nogist/isvd stored bits {twin / gist:.2f},"
                f" outside 1 +/- {BYTE_MATCH_TOL}"
            )
    return dict(refused), notes


def _why_unpaired(c: Contrast) -> str:
    """Why a member has no pairing at all: every shared key lost its digest, or the two
    cells never shared one. Both are "a broken pairing, not a result" (section 4)."""
    return (
        f"all {len(c.dropped_keys)} keys dropped on prompt_sha256 ({key_list(c.dropped_keys)})"
        if c.dropped_keys
        else "share no (seed, trial)"
    )


def _beats(
    family: str,
    b: str,
    retrieval: dict[tuple[str, str, str], Contrast],
    ppl: list[PplContrast],
) -> str | None:
    """Rule 1's two routes, for one control. The member that carried it, or ``None``."""
    for (f, task, control), c in sorted(retrieval.items()):
        beaten = c.p_holm is not None and c.p_holm < ALPHA and c.a_favored > c.b_favored
        if (f, control) == (family, b) and c.primary and beaten:
            return (
                f"{family}/{task}: {c.a} beats {b} on retrieval"
                f" ({c.a_favored}-{c.b_favored} pairs, Holm p={c.p_holm:.2g})"
            )
    # All three, and the third is the margin doing its work in both directions: an
    # advantage inside +/-0.02 bits is "equivalence measured tightly" and counts for
    # neither branch (prereg section 4 rule 1). One member per corpus, any of which can
    # carry the route -- section 6's union shape, which is why they keep Holm.
    for t in [x for x in ppl if (x.family, x.b) == (family, b)]:
        if t.p_holm is not None and t.p_holm < ALPHA and t.d_bits < 0 and not t.equivalent:
            return (
                f"{family}: {t.a} beats {b} on perplexity ({t.d_bits:+.4f} bits,"
                f" Holm p={t.p_holm:.2g}, TOST +/-{PPL_DELTA_BITS} fails)"
                + ("" if t.decidable else _wide(t))
            )
    return None


def _wide(t: PplContrast) -> str:
    """Section 6: "Where a separating member's TOST is `False` because it is `not
    decidable`, the verdict entry says so beside the branch, with `s` and the CI." The
    conjunction rule 1 requires then reads "real, and not demonstrably inside +/-0.02",
    which is weaker than "demonstrably outside it" -- so it is said, not hidden."""
    return (
        f" -- the TOST is False because it is `not decidable` at the realised spread"
        f" (s={t.s:.4f}, CI [{t.lo:+.4f}, {t.hi:+.4f}], n={t.n_windows})"
    )


def gate1_verdict(retrieval: list[Contrast], ppl: list[PplContrast], data: Gate1Data) -> Verdict:
    """The branch, the members that decided it, and the reason -- prereg section 4, at
    the precedence and under the three rulings the header states.

    ``data`` is not decoration: refusals 1, 3 and 4 were decided from the records (the
    error rows, ``diag.jsonl`` and the ``sbits`` column) in :func:`load`, "not by eye".

    Completeness is enforced member by member: every (family, task) the 16K records show
    must carry an adjusted p for both C controls, every 16K family must carry all four
    of :data:`TASK_ORDER`, and a set with no 16K reference cell blocks C with
    :data:`NO_MEMBERS`. The reason lists every refusal and every blocker in full, and
    ``members`` carries them beside the separations: the rendered table is read alone.
    """
    r_idx = {(c.family, c.task, c.b): c for c in retrieval if c.ctx == VERDICT_CTX}
    p_members = [c for c in ppl if c.ctx == VERDICT_CTX]
    families = sorted({f for f, ctx, _, _ in data.hits if ctx == VERDICT_CTX})
    # The families that carry the reference arm at 16K. Rules 1-3 read members of
    # `isvd` against a control, so a family without it -- and a record set without any
    # 16K family at all, which is what an empty pod directory or a 32K-only set is --
    # carries no member to read, and C is a positive claim that needs one.
    readable = sorted({f for f, ctx, _, k in data.hits if ctx == VERDICT_CTX and k == REFERENCE})
    tasks = sorted(
        {t for _, ctx, t, _ in data.hits if ctx == VERDICT_CTX and t not in EXCLUDED_TASKS}
    )
    refused, notes = data.refused, data.notes

    separated: list[str] = []
    separations: list[str] = []
    for family in families:
        if family in refused:
            continue
        routes = {b: _beats(family, b, r_idx, p_members) for b in PRIMARY_CONTROLS}
        if all(routes.values()):
            separated.append(family)
            separations += [r for r in routes.values() if r]

    # Rule 3, stated as what blocks it (a refused family is reported as the refusal it
    # is, not as a blocker). `TASK_ORDER` and not the tasks that happen to be present:
    # a pod that ran two of them and separated on neither would otherwise print the
    # strongest claim this gate can make off half the evidence.
    blockers: list[str] = []
    for family in families:
        if family in refused:
            continue
        absent = [
            t
            for t in TASK_ORDER
            if t not in EXCLUDED_TASKS and (family, VERDICT_CTX, t, REFERENCE) not in data.hits
        ]
        if absent:
            blockers.append(f"incomplete task set: {family} lacks {', '.join(absent)}")
        for b in C_CONTROLS:
            for task in tasks:
                member = r_idx.get((family, task, b))
                if member is None or member.p_holm is None:
                    blockers.append(f"{family}/{task}: no adjusted p for {REFERENCE} vs {b}")
                elif member.p_holm < ALPHA:
                    blockers.append(
                        f"{family}/{task}: {REFERENCE} vs {b} separates"
                        f" (Holm p={member.p_holm:.2g})"
                    )
            members_b = [x for x in p_members if (x.family, x.b) == (family, b)]
            if not members_b:
                blockers.append(f"{family}: no perplexity TOST for {REFERENCE} vs {b}")
            # Every one of them must pass: C's perplexity condition is an
            # intersection-union test (section 6), so a second corpus is a second
            # component, never an alternative.
            for t in members_b:
                if t.equivalent:
                    continue
                # Section 6: "the report must say WHICH it is: |d| genuinely above the
                # margin (non-equivalence) or the interval too wide at the realised
                # spread (not decidable). Only the second is a reason to spend a wider
                # perplexity run; neither is a reason to widen the margin."
                blockers.append(
                    f"{family}: {REFERENCE} vs {b} TOST at +/-{PPL_DELTA_BITS}"
                    + (
                        f" does not pass (d={t.d_bits:+.4f} bits, p={t.p_tost:.2g})"
                        if t.decidable
                        else f" is `not decidable` (s={t.s:.4f}, CI [{t.lo:+.4f},"
                        f" {t.hi:+.4f}], n={t.n_windows}) -- a wide interval can only"
                        " withhold a Branch C, never manufacture one"
                    )
                )

    if not readable:
        blockers.append(NO_MEMBERS)

    refusals = [line for f in sorted(refused) for line in refused[f]]
    is_ab = len(separated) >= MIN_FAMILIES_FOR_AB
    is_c = not blockers and not refusals
    # Prereg section 4: "A/B and C are mutually exclusive by construction, and the +/-0.02
    # margin is what makes them so." Every route to a separation contradicts one of C's
    # conditions, so both true at once is a bug in this function, not an outcome. A raise
    # and not an `assert`: this one has to hold under `python -O` too.
    if is_ab and is_c:
        raise RuntimeError(
            f"A/B and C both selected (prereg section 4 makes them exclusive):"
            f" separated {separated}, no blockers, no refusals"
        )
    blocked = ("C blocked by: " + "; ".join(blockers)) if blockers else ""
    # R-L3-15 (header): A/B and C are checked FIRST, on the families that remain --
    # both loops above already skip a refused one -- and REFUSED only decides once
    # neither composes.
    if is_ab:
        branch = "A/B"
        parts = [
            f"{len(separated)} families separated at 16K ({', '.join(separated)}): "
            + "; ".join(separations)
        ]
        if refusals:
            parts.append(
                f"refused, excluded from the count: {', '.join(sorted(refused))}: "
                + "; ".join(refusals)
            )
        parts.append(blocked)
    elif is_c:
        branch = "C"
        parts = [
            f"no Holm-significant separation from {' or '.join(C_CONTROLS)} on any task in any"
            f" 16K family, and every TOST at +/-{PPL_DELTA_BITS} bits/token passes"
        ]
    elif refusals:
        branch = "REFUSED"
        parts = [f"verdict refused for {', '.join(sorted(refused))}: " + "; ".join(refusals)]
        if separated:
            parts.append(f"families separated on the records that remain: {', '.join(separated)}")
        parts.append(blocked)
    elif not readable:
        branch, parts = "UNDECIDED", [NO_MEMBERS]
    else:
        branch = "UNDECIDED"
        parts = [
            f"one family separated ({separated[0]})" if separated else "nothing selected",
            blocked,
        ]
    members = [*separations, *refusals, *blockers] or ["no member separates isvd from fd or frozen"]
    return Verdict(branch, "; ".join([*(p for p in parts if p), *notes]), separated, members)


def bf16_family_size() -> int:
    """The retrieval family ``prereg/bf16_gist.md`` section 6 fixes: its two model
    families times the tasks that remain, "8 - 2 per excluded task ... so one exclusion
    gives 6 and two give 4". This is the PRE-REGISTERED size; the realised m is how many
    members Holm actually ran over, and the table prints both."""
    return BF16_FAMILIES * len([t for t in TASK_ORDER if t not in EXCLUDED_TASKS])


def bf16_contrasts(data: Gate1Data) -> list[Contrast]:
    """The bf16 arm's retrieval members: ``isvd_r64_h256_seed`` (a) against
    ``isvd_r64_h256_seed_bf16`` (b), per family x ctx x task, through the same
    :func:`_member` every Gate-1 contrast takes -- so ``a_favored`` counts the pairs the
    bf16 arm LOST and section 7 (e)'s digest drop shrinks ``n_paired`` here too.

    Its own Holm family, one per context length: 8 members at Stage 1 (2 families x 4
    tasks, ``prereg/bf16_gist.md`` section 6), never pooled with a Gate-1 family and never
    recomputing one -- "no member of this file enters a Gate-1 family".

    Eligibility is that file's section 4 snippet, NOT Gate 1's ruling R-L3-16: a member
    leaves the correction for its OWN error row or its OWN broken pairing and for nothing
    else, so a refused family's clean members stay in and the realised m shrinks by the
    member rather than by the family. The snippet is the registered statistic, and this
    reproduces it.
    """
    draft = [
        _member(data, f, c, t, BF16)
        for f, c, t in _reference_cells(data)
        if (f, c, t, BF16) in data.hits
    ]
    eligible = {
        i: x.ctx
        for i, x in enumerate(draft)
        if x.p is not None and (x.errors_a, x.errors_b) == (0, 0)
    }
    adjusted = _holm_by_group(draft, cast(dict[int, object], eligible))
    return [replace(x, p_holm=adjusted[i]) for i, x in enumerate(draft)]


def _bf16_loses(c: Contrast) -> bool:
    """Section 4 (1): the bf16 arm is non-inferior on a task "unless **both** hold" -- it
    loses by more than :data:`BF16_MARGIN` on the point estimate AND that member's
    Holm-adjusted p is below :data:`ALPHA`. Either alone is non-inferiority.

    At the design's n = 24 the margin cannot bind on its own (one flipped pair is 1/24 =
    0.0417, already past 0.03), so the decision is carried by Holm -- which that clause
    states in advance rather than leaving to be discovered.
    """
    return (
        c.n_paired > 0
        and c.p_holm is not None
        and c.p_holm < ALPHA
        and (c.a_favored - c.b_favored) / c.n_paired > BF16_MARGIN
    )


def _bf16_refusals(
    family: str,
    members: list[Contrast],
    ppl: list[PplContrast],
    data: Gate1Data,
    tasks: list[str],
) -> list[tuple[str, str]]:
    """Section 4 (4)'s refusals for one family, as (tag, line), read before its rule.

    An ``error`` row on either arm; a member with no pairing left; a cell or a perplexity
    sweep the pod never wrote ("a pod that stopped inside arm 6 leaves ``not run``, never
    a partial reading"); and section 6's ``not decidable`` TOST, which is "never ... a
    pass, never a quiet fail". Gate 1's own frozen-dispatch and byte-match refusals are
    not read here: they are that gate's, and this arm is outside it.

    Presence is all this guards, as in the header: whether a cell holds its full n = 24 is
    ``scripts/pod.py check``'s job, and section 7 (c)'s stored-bits check is read off the
    table's own descriptive column rather than refused here.
    """
    out: list[tuple[str, str]] = []
    for t in tasks:
        for k in (REFERENCE, BF16):
            arm = data.arms.get((family, k), k)
            if errs := data.errors.get((family, VERDICT_CTX, t, k)):
                out.append(
                    ("error", f"{family}/{t}: {arm} has {len(errs)} error records ({errs[0]})")
                )
            elif (family, VERDICT_CTX, t, k) not in data.hits:
                out.append(("not run", f"{family}/{t}: no {arm} records at ctx {VERDICT_CTX}"))
    out += [
        ("no pairing", f"{family}/{c.task}: {c.a} vs {c.b} {_why_unpaired(c)}")
        for c in members
        if c.p is None
    ]
    if not ppl:
        out.append(
            (
                "not run",
                f"{family}: no perplexity member for {REFERENCE} vs {BF16} at ctx {VERDICT_CTX}",
            )
        )
    out += [
        (
            "not decidable",
            f"{family}: the +/-{PPL_DELTA_BITS} TOST cannot fire at the realised spread"
            f" (s={t.s:.4f}, CI [{t.lo:+.4f}, {t.hi:+.4f}], n={t.n_windows})",
        )
        for t in ppl
        if not t.equivalent and not t.decidable
    ]
    return out


def _bf16_family(
    family: str,
    members: list[Contrast],
    ppl: list[PplContrast],
    data: Gate1Data,
    tasks: list[str],
) -> tuple[str, list[str]]:
    """One family's state and the lines behind it: section 4 (4) first, then rule 1 on
    every task and rule 2 on the family's TOST -- a conjunction, so one loss is a FAIL.

    Every perplexity member of the family must pass, not one of them: the claim is an
    intersection-union test over its components (section 4 (2)), so a second corpus is a
    second component and never an alternative.
    """
    if refusals := _bf16_refusals(family, members, ppl, data, tasks):
        return (
            f"REFUSED ({refusals[0][0]})",
            [f"{family}: REFUSED ({tag}) -- {why}" for tag, why in refusals],
        )
    lost = [
        f"{family}/{c.task}: the bf16 arm loses {(c.a_favored - c.b_favored) / c.n_paired:.3f} of"
        f" its {c.n_paired} paired keys ({c.a_favored}-{c.b_favored}, Holm p={c.p_holm:.3g}) --"
        f" past the {BF16_MARGIN} margin AND Holm-significant (section 4 (1))"
        for c in members
        if _bf16_loses(c)
    ]
    lost += [
        f"{family}: the +/-{PPL_DELTA_BITS} bits/token TOST fails"
        f" (d={t.d_bits:+.4f}, p={t.p_tost:.2g}) -- section 4 (2)"
        for t in ppl
        if not t.equivalent  # a `not decidable` member was refused above
    ]
    if lost:
        return "FAIL", [f"{family}: FAIL -- {x}" for x in lost]
    worst = max(((c.a_favored - c.b_favored) / c.n_paired for c in members), default=0.0)
    return "PASS", [
        f"{family}: PASS -- {len(members)} tasks non-inferior (worst point estimate"
        f" {worst:+.3f}, margin {BF16_MARGIN}) and the +/-{PPL_DELTA_BITS} TOST passes"
        f" ({', '.join(f'{t.d_bits:+.4f} bits' for t in ppl)})"
    ]


def bf16_verdict(contrasts: list[Contrast], ppl: list[PplContrast], data: Gate1Data) -> Bf16Verdict:
    """``prereg/bf16_gist.md`` section 4's reading, over the 16K members only.

    Read per family (section 4 (4) before the rule), then composed by section 4 (3): a
    refusal anywhere is ``REFUSED``, a fail with no refusal is ``fail``, and a pass has to
    be bought on both families -- "there is no partial pass and no per-family pass" -- so
    a single-family input reads ``REFUSED`` rather than a pass off half the design. A
    fail off one family stays a fail: the cost it reports was measured.
    """
    tasks = [t for t in TASK_ORDER if t not in EXCLUDED_TASKS]
    read = sorted({f for f, ctx, _, k in data.hits if ctx == VERDICT_CTX and k == BF16})
    if not read:
        return Bf16Verdict(
            "not run",
            f"no {BF16} records at ctx {VERDICT_CTX} ({BF16_PREREG} section 9: `not run`,"
            " never a partial reading)",
            {},
            [],
        )
    families: dict[str, str] = {}
    members: list[str] = []
    for f in read:
        state, lines = _bf16_family(
            f,
            [c for c in contrasts if (c.family, c.ctx) == (f, VERDICT_CTX)],
            [t for t in ppl if (t.family, t.ctx, t.b) == (f, VERDICT_CTX, BF16)],
            data,
            tasks,
        )
        families[f], members = state, members + lines
    partial = len(read) < BF16_FAMILIES
    if partial:
        members.append(
            f"{len(read)} of {BF16_FAMILIES} Stage-1 families in the records"
            f" ({', '.join(read)}) -- section 4 (3) buys the reading on both or not at all,"
            " so a pass here is REFUSED"
        )
    refused = [f for f, s in families.items() if s.startswith("REFUSED")]
    overall = (
        "REFUSED"
        if refused
        else "fail"
        if "FAIL" in families.values()
        else "REFUSED"
        if partial
        else "pass"
    )
    realised = sum(1 for c in contrasts if c.ctx == VERDICT_CTX and c.p_holm is not None)
    consequence = (
        "section 4's consequence fires: every byte-matched control is re-planned at the new"
        " bytes before it is re-run"
        if overall == "pass"
        else "the fp32 arm stays the default and the bf16 row is reported as the measured"
        " cost of halving the gist's at-rest bytes"
        if overall == "fail"
        else "the fp32 arm stays the default and no cost figure is recorded -- nothing was"
        " measured to report one from"
    )
    reason = "; ".join(
        [
            ", ".join(f"{f} {s}" for f, s in families.items()),
            f"Holm at the realised m={realised} of the {bf16_family_size()} members"
            f" {BF16_PREREG} section 6 fixes",
            consequence,
        ]
    )
    return Bf16Verdict(overall, reason, families, members)


def key_list(keys: tuple[Key, ...]) -> str:
    """The dropped pairing keys, all of them: a count without the keys is not a report."""
    return ", ".join(f"({s},{t})" for s, t in keys)
