"""Generator v2: real-text haystacks in a balanced design, official RULER task semantics.

The in-house v1 generator (``ruler.build_task``, re-exported here as ``make_trial_v1``
and frozen bit-for-bit) cycles ten sentences and is suspected of flattering a low-rank
gist. This one draws the haystack from real text -- four sources materialized once per
pod by `kvdlra.eval.haystacks` (PG-19 books, arXiv papers, Wikipedia, Paul Graham
essays) -- and enumerates the trials of a task over a balanced design, ``codes x depths
x haystacks`` (`config.TaskV2Cfg.design`): every (depth, code family) cell sees the same
number of haystacks, and the sources rotate with the trial index so each appears equally
often. ``seed`` changes the draws (which document, where the window starts, which needle
values), never the design.

Task semantics mirror NVIDIA/RULER's synthetic ``niah`` and ``variable_tracking``
templates (scripts/data/synthetic/constants.py, the generator `official_ruler` pins):
the needle sentence, the question and the answer prefix are RULER's strings, singular
when one answer is asked for, the prefix primed after the assistant header
(`official_ruler.templated_official`), and the scoring rule is `ruler.retrieve`'s
string_match_all. Needle values come from one of two code families: ``numbers``
(7-digit integers, no leading zero) or ``words`` (a hyphenated adjective-noun pair from
`data.ADJECTIVES` x `data.NOUNS` -- one token under a whitespace split, never cut by
the sentence splitter, exact under ``t in text``). Keys are `data.LABELS`.

Every trial records ``haystack_id`` (which documents), ``depth``, ``code_family`` and
``prompt_sha256`` over the exact token ids fed, so a reviewer can verify that two arms
of one cell were given byte-identical prompts. `runner._cell` prints them on the
``[trial]`` line and `records.parse_trial_lines` reads them back.
"""

from __future__ import annotations

import functools
import json
import random
import re
import string
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import torch

from kvdlra.eval import ruler
from kvdlra.eval.config import TaskV2Cfg
from kvdlra.eval.data import ADJECTIVES, LABELS, NOUNS
from kvdlra.eval.official_ruler import split_input, templated_official

make_trial_v1 = ruler.build_task  # the frozen v1 builder, no code motion (PR-L2-22)

# Where `haystacks.materialize` writes `<source>.jsonl` (gitignored; `pod.py prepare`).
HAYSTACKS = Path(__file__).resolve().parents[3] / "data" / "haystacks"

# NVIDIA/RULER scripts/data/synthetic/constants.py + niah.py, verbatim: the plural forms;
# `_singular` applies niah.py's replacements when num_needle_q x num_needle_v == 1.
NIAH_NEEDLE = "One of the special magic {kind} for {key} is: {value}."
NIAH_TEMPLATE = (
    "Some special magic {kind} are hidden within the following text. Make sure to memorize"
    " it. I will quiz you about the {kind} afterwards.\n{context}\nWhat are all the special"
    " magic {kind} for {query} mentioned in the provided text?"
)
NIAH_ANSWER_PREFIX = " The special magic {kind} for {query} mentioned in the provided text are"
# variable_tracking.py: `VAR A = <value>` then `VAR B = VAR A`, the value's variables asked.
VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text."
    "\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the"
    " text above."
)
VT_ANSWER_PREFIX = (  # "assgined" is RULER's own spelling, kept so the prefix is theirs
    " Answer: According to the chain(s) of variable assignment in the text above, {num_v}"
    " variables are assgined the value {query}, they are: "
)
# Shape parameters, as v1's `ruler.N_KEYS`/`N_VALUES` and RULER's official `vt` (4 hops).
N_KEYS, N_VALUES, N_QUERIES, N_HOPS = 8, 4, 4, 4
# Decode budgets per task family. RULER's `tokens_to_generate` are 128 / 30; ours are
# the shortest that hold every answer form -- four 7-digit numbers or word pairs with
# commas (niah), five 5-letter variable names after the primed prefix (vt).
MAX_NEW = {"niah": 48, "vt": 64}
# The lane's depth grid; `depths(d)` picks `d` of them evenly.
DEPTH_GRID = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
STEP = 0.2  # the depth step between the needles of one multi-needle task, wrapping

_SENT_END = re.compile(r"(?<=[.!?])\s+")


class Doc(TypedDict):
    """One materialized document (`haystacks.materialize`); `id` is `d<index>`."""

    id: str
    source: str
    text: str
    sha256: str


class Trial(NamedTuple):
    prefill_ids: torch.Tensor
    query_ids: torch.Tensor
    targets: list[str]
    meta: dict[str, Any]


# ------------------------------------------------------------- corpora


def load_corpora(
    names: Sequence[str], root: Path = HAYSTACKS, fixture: Path | None = None
) -> dict[str, list[Doc]]:
    """``source -> docs`` for every source in ``names``, read once per process.

    Each source is ``<root>/<name>.jsonl`` as `haystacks.materialize` wrote it; ``fixture``
    is one JSONL holding every source (a ``source`` field per row, a provenance record
    first -- ``tests/fixtures/haystacks_tiny.jsonl``), for tests that must not touch
    ``data/``. Memoized on the arguments: the runner asks for the same corpora for every
    arm, sub-task, seed and trial of a pod."""
    return _corpora(tuple(names), root, fixture)


@functools.lru_cache(maxsize=4)
def _corpora(names: tuple[str, ...], root: Path, fixture: Path | None) -> dict[str, list[Doc]]:
    t0 = time.perf_counter()
    rows: list[Doc] = []
    for p in [fixture] if fixture else [root / f"{n}.jsonl" for n in names]:
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    out = {n: [d for d in rows if d.get("source") == n] for n in names}
    if missing := [n for n in names if not out[n]]:
        raise FileNotFoundError(
            f"no haystack documents for {missing} under {fixture or root};"
            " `scripts/pod.py prepare --pod <pod>` materializes them"
        )
    print(f"[stage] load_corpora {','.join(names)} ({time.perf_counter() - t0:.1f} s)", flush=True)
    return out


def sentences(text: str) -> list[str]:
    """``text`` split after terminal punctuation, whitespace normalized (a paragraph
    break is a space), fragments under 20 characters dropped -- headings, stray
    tokens -- as `data.load_corpus_sentences` does for the v1 realistic filler."""
    return [s for s in _SENT_END.split(" ".join(text.split())) if len(s) >= 20]


# ------------------------------------------------------------- the design


def depths(d: int) -> list[float]:
    """``d`` depths spread evenly over `DEPTH_GRID`: index ``round(i * 5 / (d - 1))``,
    so d=3 is (0.05, 0.4, 0.95), d=6 the whole grid, d=1 the middle (0.4)."""
    if d == 1:
        return [DEPTH_GRID[2]]
    return [DEPTH_GRID[round(i * 5 / (d - 1))] for i in range(d)]


def design_cell(cfg: TaskV2Cfg, trial: int) -> tuple[int, float, str, int]:
    """``trial`` -> (haystack_idx, depth, code_family, code_idx).

    The lexicographic product codes (outer) x depths x haystacks (inner): trials
    ``0..n-1`` with ``n = prod(design)`` cover the design exactly once. The family
    alternates with the code index, so ``codes: 4`` over two families is two draws of
    each per (haystack, depth)."""
    nh, nd = cfg.design["haystacks"], cfg.design["depths"]
    h, rest = trial % nh, trial // nh
    d, c = rest % nd, rest // nd
    return h, depths(nd)[d], cfg.code_families[c % len(cfg.code_families)], c


def _window(tok: Any, docs: list[Doc], first: int, seed: int, ctx: int) -> tuple[list[str], str]:
    """A contiguous run of sentences holding >= ``ctx`` tokens: from a seed-derived
    sentence offset in ``docs[first]``, continuing into the following documents of the
    source (wrapping) when it runs out. Each sentence is tokenized ONCE (no special
    tokens -- a per-sentence BOS would overcount) and the counts accumulate, O(n) in
    the window. Returns the sentences and the compact id ``<source>:<doc>`` or
    ``<source>:<first>..<last>:<n_docs>``."""
    sents: list[str] = []
    total, used = 0, []
    for k in range(len(docs)):
        doc = docs[(first + k) % len(docs)]
        pool = sentences(doc["text"])
        used.append(doc["id"])
        for s in pool[random.Random(seed).randrange(len(pool)) if k == 0 else 0 :]:
            sents.append(s)
            total += len(tok(s, add_special_tokens=False).input_ids)
            if total >= ctx:
                src = doc["source"]
                one = f"{src}:{used[0]}"
                return sents, one if len(used) == 1 else f"{one}..{used[-1]}:{len(used)}"
    raise ValueError(f"{docs[0]['source']}: {len(docs)} documents hold fewer than {ctx} tokens")


def _place(sents: list[str], at: list[tuple[float, str]]) -> None:
    """Insert each needle at sentence index ``int(depth * n)`` (RULER's rule), the
    indices taken against the haystack before any insertion -- deepest first, so the
    shallower ones do not shift."""
    n = len(sents)
    for depth, text in sorted(at, reverse=True):
        sents.insert(int(depth * n), text)


def _wrap(depth: float, i: int) -> float:
    """The ``i``-th needle's depth: ``depth + i * STEP`` wrapping past 1, in hundredths."""
    return ((round(depth * 100) + round(STEP * 100) * i) % 100) / 100


# ------------------------------------------------------------- needle draws


def _keys(g: torch.Generator, k: int) -> list[str]:
    return [LABELS[i] for i in torch.randperm(len(LABELS), generator=g)[:k].tolist()]


def _values(g: torch.Generator, family: str, k: int) -> list[str]:
    """``k`` distinct needle values of one code family."""
    if family == "words":
        adj = torch.randperm(len(ADJECTIVES), generator=g)[:k].tolist()
        noun = torch.randperm(len(NOUNS), generator=g)[:k].tolist()
        return [f"{ADJECTIVES[a]}-{NOUNS[n]}" for a, n in zip(adj, noun, strict=True)]
    if family != "numbers":
        raise ValueError(f"unknown code family {family!r}")
    out: list[str] = []
    while len(out) < k:
        v = str(int(torch.randint(1_000_000, 10_000_000, (1,), generator=g).item()))
        if v not in out:
            out.append(v)
    return out


def _variables(g: torch.Generator) -> list[str]:
    """``N_HOPS + 1`` distinct 5-letter upper-case names (RULER's form) from one
    permutation of the alphabet, so no two share a letter."""
    letters = torch.randperm(26, generator=g).tolist()
    return [
        "".join(string.ascii_uppercase[j] for j in letters[5 * i : 5 * i + 5])
        for i in range(N_HOPS + 1)
    ]


def _singular(s: str) -> str:
    """niah.py's rewrite of the plural template when exactly one answer is asked for."""
    return s.replace("Some", "A").replace("are all", "is").replace("are", "is")


# ------------------------------------------------------------- one trial


def make_trial(
    cfg: TaskV2Cfg,
    tok: Any,
    task: str,
    seed: int,
    trial: int,
    corpora: dict[str, list[Doc]],
) -> Trial:
    """Build the prompt of one (task, seed, trial) cell: the haystack from the design's
    source and document, the needles of the task at the design depth, RULER's template
    around them. Deterministic in its arguments, which is what the pairing invariant
    (byte-identical prompts across arms) rests on."""
    h, depth, family, _ = design_cell(cfg, trial)
    source = cfg.haystacks[trial % len(cfg.haystacks)]
    docs = corpora[source]
    sents, hay_id = _window(tok, docs, (h + seed) % len(docs), seed, cfg.ctx)
    n_sentences = len(sents)
    g = torch.Generator().manual_seed(seed * 131 + trial)  # the v1 seed formula
    if task == "vt":
        value = _values(g, family, 1)[0]
        names = _variables(g)
        chain = [f"VAR {names[0]} = {value}"] + [
            f"VAR {names[i]} = VAR {names[i - 1]}" for i in range(1, N_HOPS + 1)
        ]
        _place(sents, [(_wrap(depth, i), s) for i, s in enumerate(chain)])
        text = VT_TEMPLATE.format(context=" ".join(sents), query=value)
        prefix = VT_ANSWER_PREFIX.format(num_v=N_HOPS + 1, query=value)
        targets = names
    elif task in ("niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery"):
        n_keys, n_values, n_q = {
            "niah_single": (1, 1, 1),
            "niah_multikey": (N_KEYS, 1, 1),
            "niah_multivalue": (1, N_VALUES, 1),
            "niah_multiquery": (N_QUERIES, 1, N_QUERIES),
        }[task]
        keys = _keys(g, n_keys)
        values = _values(g, family, n_keys * n_values)
        needles = [
            NIAH_NEEDLE.format(kind=family, key=k, value=v)
            for k, v in zip([k for k in keys for _ in range(n_values)], values, strict=True)
        ]
        # The queried needles sit at the design depth, then +STEP each, wrapping; the
        # un-queried keys of niah_multikey are spread evenly over the rest.
        at = [(_wrap(depth, i), s) for i, s in enumerate(needles[: n_q * n_values])]
        at += [((i + 1) / n_keys, s) for i, s in enumerate(needles[n_q * n_values :])]
        _place(sents, at)
        queried = keys[:n_q]
        query = ", ".join(queried[:-1]) + ", and " + queried[-1] if n_q > 1 else queried[0]
        targets = values[: n_q * n_values]
        template, prefix = NIAH_TEMPLATE, NIAH_ANSWER_PREFIX
        kind = family
        if len(targets) == 1:
            template, prefix, kind = _singular(template), _singular(prefix), family[:-1]
        text = template.format(kind=kind, context=" ".join(sents), query=query)
        prefix = prefix.format(kind=kind, query=query)
    else:
        raise ValueError(f"unknown task {task!r}")
    body, question = split_input(text)
    pre, query_ids = templated_official(tok, body, question, prefix)
    return Trial(
        pre,
        query_ids,
        targets,
        {
            "haystack_id": hay_id,
            "depth": depth,
            "code_family": family,
            "prompt_sha256": ruler.prompt_sha256(pre, query_ids),
            "n_sentences": n_sentences,
        },
    )


def run_trial(
    arm: dict[str, Any],
    model: Any,
    tok: Any,
    task: TaskV2Cfg,
    sub: str,
    seed: int,
    trial: int,
    *,
    device: str,
    chunk: int,
    n: int,
    h_kv: int,
    pool: list[str] | None = None,
) -> tuple[int, float, dict[str, Any]]:
    """Build one prompt and retrieve through ``arm`` -- `ruler.run_trial`'s signature, the
    runner's ``GENERATORS["v2"]``. ``pool`` is the v1 filler pool, unused here. The
    corpora load once per process (`load_corpora`); nothing in `ruler.retrieve` changes."""
    t = make_trial(task, tok, sub, seed, trial, load_corpora(task.haystacks))
    hit, ratio, frac, sbits = ruler.retrieve(
        model, tok, arm, t.prefill_ids, t.query_ids, t.targets, device, chunk, n, h_kv,
        MAX_NEW[sub.split("_")[0]], task=sub, idx=trial,
    )  # fmt: skip
    return int(hit), frac, {**t.meta, "ratio": ratio, "sbits": sbits}
