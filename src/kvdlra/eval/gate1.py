"""Gate 1: the tracker-swap contrasts and the branch rule that reads them.

The spec is ``prereg/gate1_tracker_swap_v2.md`` -- committed before either Stage-1 pod
launches -- and this module is its section 4, "The operationalization, exactly as
``gate1_verdict()`` implements it", in code. It adds no statistic of its own: every one
is a shipped :mod:`kvdlra.eval.stats` function, and no threshold is a literal in the
logic (they are the named constants below, each carrying the clause that fixes it).

The rule the plan wrote, which this module implements
(``docs/plan/ICML2027_PLAN.md`` section 2, Gate 1 item 1.1, quoted in prereg section 4):

    "if (a) is not separated from (c)/(d) on any task and perplexity is within 0.02
    bits/token, the tracker is not the contribution -> Branch C. If (a) beats (d) and
    (e) on retrieval or perplexity in >=2 families with Holm-corrected p<0.05 -> the
    online-tracked gist is doing work -> Branch A/B."

with (a) ``isvd``, (c) ``fd``, (d) ``frozen``, (e) the family's ``nogist`` twin.

**Scope** (prereg section 4): "the verdict reads the 16K contrasts only" -- 32K pods are
descriptive and "never change the branch", so every rule below filters on
:data:`VERDICT_CTX`. Contrasts at other context lengths are still computed and printed.

**Rule 1, a family is separated** -- "iff, for **each** of ``frozen`` and ``nogist``
separately, the r64 arm beats it -- **on retrieval on at least one task** (that member's
Holm-adjusted p < 0.05 in the primary retrieval family **and** ``a_favored >
b_favored``) **or on perplexity**, where the perplexity route requires **all three** of:
that member's Holm-adjusted p < 0.05 in the primary perplexity family, **delta < 0** (the
r64 arm's mean bits/token lower), and that contrast's +/-0.02 **TOST failing**. ... Both
controls must be beaten; beating one is not a separated family." -> :func:`_beats` and
the ``all(...)`` over :data:`PRIMARY_CONTROLS` in :func:`gate1_verdict`.

**The TOST's third state** (section 6). A member whose realised paired SD cannot fit
inside the margin even at ``d_bits`` = 0 -- ``t(0.95, n-1)*s/sqrt(n) >= 0.02``,
:func:`~kvdlra.eval.stats.tost_decidable` -- is recorded **``not decidable``**, "never
as a pass, never a quiet fail". It is a non-pass, so it withholds C exactly as a failure
does, and the blocker says WHICH of the two it was, as section 6 requires. On the A/B
side the prereg is explicit and this module follows it: "the rule reads the boolean,
which is ``False`` in two different situations -- a difference outside the margin, and
an interval too wide to place against it", so a ``not decidable`` member CAN carry rule
1's perplexity route, and "where a separating member's TOST is ``False`` because it is
``not decidable``, the verdict entry says so beside the branch, with ``s`` and the CI"
(:func:`_wide`).

**Rule 2, Branch A/B** -- "iff **>= 2 families are separated**, counting **model families
separated at 16K**" -> ``len(separated) >= MIN_FAMILIES_FOR_AB``. "One family separated
is recorded as ``UNDECIDED (one family separated)``, never rounded up."

**Rule 3, Branch C** -- "iff **both** of its conditions hold. **(i) Retrieval:** no
Holm-significant separation of the r64 arm from ``fd`` and none from ``frozen``, on **any
task in any 16K family** -- no Holm-adjusted p < 0.05 in either direction on any of those
members ... **(ii) Perplexity:** **every** ``isvd``-vs-``fd`` and ``isvd``-vs-``frozen``
TOST at +/-0.02 bits/token **passes** ... each TOST is read at alpha = 0.05
**uncorrected**" (an intersection-union test, prereg section 6) -> the ``blockers`` list:
C is selected iff nothing blocks it. A member that cannot be read at all -- never run, no
adjusted p-value, no TOST -- blocks C too: C is a positive claim of non-separation, and
an absent member is not evidence for it. Nor is a record set with no 16K member at all
-- an empty (dry-run) pod directory, or a 32K-only set: :data:`NO_MEMBERS` blocks C, and
the branch reads ``UNDECIDED``. A C printed off no members would be the strongest claim
this gate can make, read from nothing.

**Rule 4** -- "Otherwise **UNDECIDED**, with the members that blocked each branch listed."
Every branch lists them: the reason carries every refusal and every blocker in full, and
``Verdict.members`` carries them beside the separations, because the five-reviewer
simulation reads the rendered table alone.

**The five refusals** (prereg section 4, "Refusal, and the ``--`` rule"), all read
*before* the rules above and all returning rather than raising -- a raise would leave
``make gate1`` with no table in which to print the failure. The branch value they return
is **``REFUSED``**, not ``UNDECIDED`` (ruling R-L3-12, over section 4's literal wording):
``UNDECIDED`` is "the rule ran and neither branch held", ``REFUSED`` is "the rule never
ran on that family", and the two are different findings. A refusal is therefore never
swallowed by an A/B the other families reach -- what the rule found on the records that
remain is printed beside the refusal, never instead of it:

1. "An ``error`` row on ``isvd_r64_h256_seed``, ``frozen_r64_h256_seed``, the family's
   ``nogist_*`` arm or ``fd_r64_h256_seed`` refuses a verdict for that family:
   ``gate1_verdict`` returns ``UNDECIDED`` with the arm and the exception text named."
   ":data:`RULE_ARMS`. "An error on arms 6-8 removes that arm's secondary members and
   nothing else" -- those are listed in the reason and refuse nothing.
2. "With **no shared key** [``mcnemar_exact``] returns **None** ... a primary member at
   ``None`` refuses the verdict for its family exactly as an ``error`` row does." The
   keys a member is paired over are section 7 (e)'s: the two arms' ``prompt_sha256``
   must agree, "a key whose digests disagree is dropped from that member and the drop
   is reported with the key" (:func:`_mismatched`, ``Contrast.dropped_keys``), and a
   member whose every key is dropped lands on this boundary.
3. "A frozen arm still repairing after its freeze voids the ``isvd``-vs-``frozen``
   contrast": any ``diag`` row with ``fixed_k``/``fixed_v`` true at ``tokens_seen >
   freeze_after``, "other than the one window per (sample, layer) that straddles the
   freeze, which section 7 (c) exempts" -> :func:`_frozen_defects`.
4. "A no-gist arm off its byte match voids the ``isvd``-vs-``nogist`` contrast": the
   **measured** stored-bits ratio must lie within 1 +/- :data:`BYTE_MATCH_TOL`. A pod
   carrying no ``sbits`` at all is ``not measured``, which is never read as a pass
   (prereg section 9's rule for an unmeasured reading).
5. "No arm is ever printed as ``--``" -- the table's rule, in ``scripts/tables.py``.

Window pairing repeats ``scripts/tables.py``'s ``ppl_stats`` two-line form (bits =
``nll_sum_nats / (ntok * ln 2)``, keyed by ``window_idx``, with its duplicate-window and
broken-pairing refusals) rather than importing it: ``scripts/tables.py`` imports THIS
module for its ``gate1`` subcommand, and a library module importing an entrypoint back
would be a cycle.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from kvdlra.eval.config import load_arm
from kvdlra.eval.records import PplRecord, PplwRecord, TrialRecord, read_jsonl
from kvdlra.eval.stats import (
    Key,
    holm,
    mcnemar_exact,
    paired_bootstrap,
    paired_t,
    tost,
    tost_decidable,
)

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
    # "A/B" | "C" | "UNDECIDED" | "REFUSED". REFUSED is its own value (ruling R-L3-12,
    # over section 4's literal "returns UNDECIDED"): "the rule ran and neither branch
    # held" and "the rule never ran on this family" are different findings, and a table
    # that prints one for the other misreports the pod.
    branch: str
    reason: str
    families_separated: list[str]
    members: list[str]  # the members that decided it (prereg section 4's snippet)


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
    and "no contrast in this file crosses a pod boundary" (prereg section 3).
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
    bits: dict[SweepKey, dict[int, float]] = defaultdict(dict)
    raw_sbits: dict[tuple[str, str], list[float]] = defaultdict(list)
    frozen_defects: dict[str, list[dict[str, object]]] = {}

    for d in pod_dirs:
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
        for w in (cast(PplwRecord, x) for x in _maybe(d / "pplw.jsonl")):
            sweep: SweepKey = (
                family,
                w["ctx"],
                w.get("corpus"),
                _tracker(by_key, w["arm"], d / "pplw.jsonl"),
            )
            if w["window_idx"] in bits[sweep]:
                raise ValueError(
                    f"{d}: {w['arm']} ctx={w['ctx']} corpus={w.get('corpus')} carries"
                    f" window_idx={w['window_idx']} twice"
                    " -- the records are duplicated"
                )
            bits[sweep][w["window_idx"]] = w["nll_sum_nats"] / (w["ntok"] * math.log(2))
        for p in (cast(PplRecord, x) for x in _maybe(d / "ppl.jsonl")):
            if p.get("sbits") is not None:
                key_s = (family, _tracker(by_key, p["arm"], d / "ppl.jsonl"))
                raw_sbits[key_s].append(float(cast(float, p["sbits"])))
        frozen_defects[family] = _frozen_defects(d / "diag.jsonl", frozen_key, freeze_after)

    return Gate1Data(
        pods=pods,
        arms=arms,
        hits=dict(hits),
        sha=dict(sha),
        errors=dict(errors),
        bits=dict(bits),
        sbits={k: statistics.median(v) for k, v in raw_sbits.items()},
        frozen_defects=frozen_defects,
    )


def _sweep_order(k: SweepKey) -> tuple[str, int, str, str]:
    """A sort key that tolerates the ``None`` corpus the archived rows carry."""
    return (k[0], k[1], k[2] or "", k[3])


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

    Section 6: "A member whose cell holds an ``error`` row **leaves its family** (the
    remaining members are corrected together at the smaller m, so the others stay
    decidable) and is listed beside the family with its exception and no adjusted
    p-value." ``group`` maps a member's index to its family key (absent = not in any
    family); the realised m is how many indices share a key, which the table prints.
    """
    # ponytail: the callers group by context length, so the three Stage-2 32K pods would
    # pool into ONE descriptive family instead of carrying "the same three sizes, 8 / 2 /
    # 12 per family" (section 6). Group by (family, ctx) when Stage 2's amendment names
    # its pods -- at Stage 1 (one pod per model family, one ctx) the two agree.
    out: list[float | None] = [None] * len(draft)
    members: dict[object, list[int]] = defaultdict(list)
    for i, key in group.items():
        members[key].append(i)
    for idx in members.values():
        raw = [cast(float, draft[i].p) for i in idx]
        for i, adj in zip(idx, holm(raw), strict=True):
            out[i] = adj
    return out


def retrieval_contrasts(data: Gate1Data) -> list[Contrast]:
    """Every retrieval member: ``isvd`` vs ``frozen``/``nogist`` (primary) and vs
    ``oja``/``fd``/``random`` (secondary), per family x ctx x task -- the exact paired
    McNemar over the ``(seed, trial)`` keys the two cells share, "the r64 arm as *a*".

    The pairing invariant is enforced here, not assumed: section 4 takes the McNemar
    "on **byte-identical prompts** (section 3's pairing invariant, verified from
    ``prompt_sha256`` per section 7 (e); a key whose digests disagree is dropped from
    that member and the drop is reported with the key)", and a key whose digest is
    missing on either side "is a failure as well as a mismatch". Dropping shrinks the
    member's paired n (section 6) and never removes the member -- except at the limit,
    where every key goes and the member has no pairing at all, which section 4's
    ``mcnemar_exact`` -> ``None`` boundary refuses.

    Holm (section 6) runs over two families of members -- the primary ones and the
    secondary ones -- and separately per context length, because a 32K pod "carries the
    same three sizes ... Holm inside each family" and is descriptive. The family is the
    set of members present in the pods passed, at their realised m; the prereg names
    which pods form a stage, and no Stage-2 member is pooled with a Stage-1 member.
    """
    draft: list[Contrast] = []
    for f, c, t in sorted({(f, c, t) for f, c, t, k in data.hits if k == REFERENCE}):
        for b in (*PRIMARY_CONTROLS, *SECONDARY_CONTROLS):
            if (f, c, t, b) not in data.hits:
                continue
            a_cell, b_cell = data.hits[(f, c, t, REFERENCE)], data.hits[(f, c, t, b)]
            dropped = _mismatched(data, (f, c, t, REFERENCE), (f, c, t, b))
            keep = {k: v for k, v in a_cell.items() if k not in dropped}
            m = mcnemar_exact(keep, {k: v for k, v in b_cell.items() if k not in dropped})
            draft.append(
                Contrast(
                    family=f, ctx=c, task=t, a=REFERENCE, b=b, primary=b in PRIMARY_CONTROLS,
                    n_paired=m["n_paired"] if m else 0,
                    a_favored=m["a_favored"] if m else 0,
                    b_favored=m["b_favored"] if m else 0,
                    p=m["p_value"] if m else None, p_holm=None,
                    errors_a=len(data.errors.get((f, c, t, REFERENCE), [])),
                    errors_b=len(data.errors.get((f, c, t, b), [])),
                    dropped_keys=dropped,
                )
            )  # fmt: skip
    eligible = {
        i: (x.ctx, x.primary)
        for i, x in enumerate(draft)
        if x.p is not None and (x.errors_a, x.errors_b) == (0, 0)
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

    Windows are paired by ``(ctx, corpus, window_idx)`` and a pairing that is not exact
    is refused, as ``ppl_stats`` refuses it: an unpaired comparison of pooled numbers
    hides the effect it is measuring. Holm runs over the PRIMARY members only (section
    6's 4-member family, per context length); the secondary contrasts are "uncorrected
    and descriptive, except the ``fd`` TOSTs the C branch names".
    """
    draft: list[PplContrast] = []
    for f, c, corpus, _ in sorted((k for k in data.bits if k[3] == REFERENCE), key=_sweep_order):
        for b in TRACKERS:
            if b == REFERENCE or (f, c, corpus, b) not in data.bits:
                continue
            a_w, b_w = data.bits[(f, c, corpus, REFERENCE)], data.bits[(f, c, corpus, b)]
            if set(a_w) != set(b_w):
                raise ValueError(
                    f"ppl: {f} ctx={c} corpus={corpus} {b} scored windows"
                    f" {sorted(set(a_w) ^ set(b_w))} that {REFERENCE} did not (or the"
                    " reverse) -- the pairing is broken"
                )
            d = [a_w[i] - b_w[i] for i in sorted(a_w)]
            mean_d, lo, hi = paired_bootstrap(d)
            p_lo, p_hi, equivalent = tost(d, PPL_DELTA_BITS, ALPHA)
            draft.append(
                PplContrast(
                    family=f, ctx=c, corpus=corpus, a=REFERENCE, b=b,
                    primary=b in PRIMARY_CONTROLS,
                    n_windows=len(d), d_bits=mean_d, lo=lo, hi=hi,
                    s=statistics.stdev(d), p=paired_t(d),
                    p_holm=None, p_tost=max(p_lo, p_hi), equivalent=equivalent,
                    decidable=tost_decidable(d, PPL_DELTA_BITS, ALPHA),
                )
            )  # fmt: skip
    eligible = {i: (x.ctx, x.corpus) for i, x in enumerate(draft) if x.primary}
    adjusted = _holm_by_group(draft, cast(dict[int, object], eligible))
    return [replace(x, p_holm=adjusted[i]) for i, x in enumerate(draft)]


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
    """The branch, the members that decided it, and the reason -- prereg section 4.

    ``data`` is not decoration: refusals 1, 3 and 4 are decided from the records (the
    error rows, ``diag.jsonl`` and the ``sbits`` column), "not by eye".

    Four values. ``REFUSED`` whenever any refusal fired (ruling R-L3-12), with what the
    rule found on the families that remain printed beside it; ``A/B`` and ``C`` as
    rules 2 and 3 select them; ``UNDECIDED`` otherwise -- including the case where
    nothing at :data:`VERDICT_CTX` can be read at all. Completeness is enforced member
    by member: every (family, task) the 16K records show must carry an adjusted p for
    both C controls or C is blocked, and a record set with no 16K reference cell at all
    blocks it with :data:`NO_MEMBERS`. The reason lists every refusal and every blocker
    in full, and ``members`` carries them beside the separations -- nothing elided,
    because the rendered table is read on its own.
    """
    r_idx = {(c.family, c.task, c.b): c for c in retrieval if c.ctx == VERDICT_CTX}
    p_members = [c for c in ppl if c.ctx == VERDICT_CTX]
    families = sorted({f for f, ctx, _, _ in data.hits if ctx == VERDICT_CTX})
    # The families that carry the reference arm at 16K. Rules 1-3 read members of
    # `isvd` against a control, so a family without it -- and a record set without any
    # 16K family at all, which is what an empty pod directory or a 32K-only set is --
    # carries no member to read, and C is a positive claim that needs one.
    readable = sorted({f for f, ctx, _, k in data.hits if ctx == VERDICT_CTX and k == REFERENCE})
    tasks = sorted({t for _, ctx, t, _ in data.hits if ctx == VERDICT_CTX})
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
            why = (
                f"all {len(c.dropped_keys)} keys dropped on prompt_sha256"
                f" ({key_list(c.dropped_keys)})"
                if c.dropped_keys
                else "share no (seed, trial)"
            )
            refused[c.family].append(f"{c.family}/{c.task}: {c.a} vs {c.b} {why}")
    for family in families:
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

    separated: list[str] = []
    separations: list[str] = []
    for family in families:
        if family in refused:
            continue
        routes = {b: _beats(family, b, r_idx, p_members) for b in PRIMARY_CONTROLS}
        if all(routes.values()):
            separated.append(family)
            separations += [r for r in routes.values() if r]

    # Rule 3, stated as what blocks it. A refused family blocks C too -- its members
    # cannot be read at all -- and is reported as the refusal it is, not as a blocker.
    blockers: list[str] = []
    for family in families:
        if family in refused:
            continue
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
    # conditions, so both true at once is a bug in this function, not an outcome.
    assert not (is_ab and is_c), "A/B and C are mutually exclusive (prereg section 4)"
    blocked = ("C blocked by: " + "; ".join(blockers)) if blockers else ""
    if refusals:
        # A refusal is read BEFORE the rules ("never after", section 4) and is its own
        # branch value (ruling R-L3-12) -- so it cannot be swallowed by an A/B the other
        # families reach. What the rule found on the records that remain is printed
        # beside it, never instead of it.
        branch = "REFUSED"
        parts = [f"verdict refused for {', '.join(sorted(refused))}: " + "; ".join(refusals)]
        if separated:
            parts.append(f"families separated on the records that remain: {', '.join(separated)}")
        parts.append(blocked)
    elif is_ab:
        branch = "A/B"
        parts = [
            f"{len(separated)} families separated at 16K ({', '.join(separated)}): "
            + "; ".join(separations),
            blocked,
        ]
    elif is_c:
        branch = "C"
        parts = [
            f"no Holm-significant separation from {' or '.join(C_CONTROLS)} on any task in any"
            f" 16K family, and every TOST at +/-{PPL_DELTA_BITS} bits/token passes"
        ]
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


def key_list(keys: tuple[Key, ...]) -> str:
    """The dropped pairing keys, all of them: a count without the keys is not a report."""
    return ", ".join(f"({s},{t})" for s, t in keys)
