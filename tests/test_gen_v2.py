"""Generator v2 (`kvdlra.eval.gen`): real-text haystacks, a balanced design, official RULER
task semantics, and a per-trial ``prompt_sha256`` -- lane L2 item 2.

Everything here runs on the whitespace ``tok`` fixture (tests/conftest.py) and the
12-document fixture corpus, so it needs no network and no model. The golden pins
determinism of the design, the sentence window, the needle text and its placement under
that stable fake tokenizer; the real-tokenizer hashes are recorded per pod.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pod
import pytest

from kvdlra.eval.config import PodCfg, TaskV2Cfg, config_hash, load_pod, load_task
from kvdlra.eval.data import ADJECTIVES, LABELS, NOUNS
from kvdlra.eval.gen import (
    DEPTH_GRID,
    Trial,
    depths,
    design_cell,
    design_source,
    load_corpora,
    make_trial,
    sentences,
)

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "haystacks_tiny.jsonl"
TASKS = ["niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt"]
V2_TASKS = ["ruler_v2_16k", "ruler_v2_32k", "ruler_v2_16k_g1", "ruler_v2_16k_g2", "ruler_v2_32k_g1"]
# The Gate-1 task files carry the FOUR tasks GATES.md G3 and ICML2027_PLAN section 2 item 1.1
# name (prereg/gate1_tracker_swap_v2.md section 3, L3.2); niah_multiquery is not a Gate-1 task
# and stays in ruler_v2_16k / _32k / _16k_g2, which no Gate-1 contrast reads.
GATE1_TASKS = [t for t in TASKS if t != "niah_multiquery"]
CFG = TaskV2Cfg(
    name="t",
    generator="v2",
    ctx=1024,
    tasks=TASKS,
    n_trials=24,
    seeds=[0, 1],
    haystacks=["pg19", "arxiv", "wikipedia", "essays"],
    code_families=["numbers", "words"],
    design={"haystacks": 2, "depths": 3, "codes": 4},
)


@pytest.fixture(scope="module")
def corpora() -> dict[str, list[Any]]:
    return load_corpora(tuple(CFG.haystacks), root=FIX.parent, fixture=FIX)


# ------------------------------------------------------------------ the design


def test_design_is_balanced_and_exhaustive() -> None:
    cells = [design_cell(CFG, t) for t in range(24)]
    assert len(set(cells)) == 24  # 2 haystacks x 3 depths x 4 code draws
    assert {c[1] for c in cells} == {0.05, 0.4, 0.95}  # 3 depths evenly off the 6-point grid
    assert sorted({c[0] for c in cells}) == [0, 1]
    assert {c[2] for c in cells} == {"numbers", "words"} and {c[3] for c in cells} == {0, 1, 2, 3}
    # the family alternates with the code index; every (depth, family) pair is 4 cells
    assert all(c[2] == CFG.code_families[c[3] % 2] for c in cells)
    assert set(Counter((c[1], c[2]) for c in cells).values()) == {4}


def test_depths_are_spread_evenly_over_the_grid() -> None:
    assert DEPTH_GRID == (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
    assert depths(1) == [0.4] and depths(3) == [0.05, 0.4, 0.95] and depths(6) == list(DEPTH_GRID)
    assert depths(2) == [0.05, 0.95]


def _sources(design: dict[str, int]) -> tuple[TaskV2Cfg, list[str]]:
    cfg = replace(CFG, n_trials=math.prod(design.values()), design=design)
    return cfg, [design_source(cfg, t) for t in range(cfg.n_trials)]


def test_every_source_is_used_equally(tok: Any, corpora: dict[str, list[Any]]) -> None:
    """PR-L2-23, enumerated: in the 2 x 3 x 4 design every (depth, family) pair sees all
    four sources exactly once and each source is used six times; in 4 x 3 x 4 every pair
    sees every source twice. The two haystacks of one (depth, code) cell still come from
    two sources, and the prompt's haystack comes from the source the design names."""
    for design, per_pair in ((CFG.design, 1), ({"haystacks": 4, "depths": 3, "codes": 4}, 2)):
        cfg, sources = _sources(design)
        assert Counter(sources) == dict.fromkeys(CFG.haystacks, 6 * per_pair)
        by_pair: dict[tuple[float, str], Counter[str]] = {}
        by_cell: dict[tuple[float, int], set[str]] = {}
        for t, src in enumerate(sources):
            _h, depth, fam, code = design_cell(cfg, t)
            by_pair.setdefault((depth, fam), Counter())[src] += 1
            by_cell.setdefault((depth, code), set()).add(src)
        assert len(by_pair) == 6 and len(by_cell) == 12
        assert all(seen == dict.fromkeys(CFG.haystacks, per_pair) for seen in by_pair.values())
        assert set(map(len, by_cell.values())) == {design["haystacks"]}
    trials = [
        make_trial(CFG, tok, "niah_single", seed=0, trial=t, corpora=corpora) for t in range(24)
    ]
    assert [t.meta["haystack_id"].split(":")[0] for t in trials] == _sources(CFG.design)[1]


def test_the_family_never_changes_the_source() -> None:
    """In every design -- the four task YAMLs' and four odd shapes -- the trials of one
    (haystack, depth, replicate) group, which differ only in the code family, share one
    source: `design_source` never reads the family (``code_idx % n_families``)."""
    designs = [cast(TaskV2Cfg, load_task(n)).design for n in V2_TASKS] + [
        {"haystacks": 1, "depths": 1, "codes": 2},
        {"haystacks": 3, "depths": 6, "codes": 4},
        {"haystacks": 4, "depths": 2, "codes": 6},
        {"haystacks": 2, "depths": 5, "codes": 3},
    ]
    n_fam = len(CFG.code_families)
    for design in designs:
        cfg, sources = _sources(design)
        groups: dict[tuple[int, float, int], set[str]] = {}
        for t in range(cfg.n_trials):
            h, d, _, c = design_cell(cfg, t)
            groups.setdefault((h, d, c // n_fam), set()).add(sources[t])
        assert all(len(s) == 1 for s in groups.values()), design
        nh, nd, nc = design["haystacks"], design["depths"], design["codes"]
        assert len(groups) == nh * nd * math.ceil(nc / n_fam)  # a trailing block is a group too


def test_sentences_drop_fragments_without_a_space() -> None:
    """Parity with `data.load_corpus_sentences`: a bare URL is long enough but not a sentence."""
    url = "https://example.org/a/long/path/with/no/space/in/it."
    assert len(url) >= 20
    text = f"A first sentence with spaces in it.\n\n{url} Then a second sentence follows it."
    assert sentences(text) == [
        "A first sentence with spaces in it.",
        "Then a second sentence follows it.",
    ]


def test_window_refuses_an_empty_first_doc_and_skips_an_empty_later_one(tok: Any) -> None:
    """A document without a usable sentence: the module's own error, naming source and id,
    when the window starts in it; passed over and left out of the id when it is a later one."""
    from kvdlra.eval.gen import Doc, _window

    def doc(i: int, text: str) -> Doc:
        return {"id": f"d{i}", "source": "s", "text": text, "sha256": ""}

    head = doc(0, "One good sentence of prose here. " * 8)  # 48 tokens at most
    bare = doc(1, "https://example.org/no/space/and/no/terminal/punctuation")
    tail = doc(2, "Another good sentence of prose here. " * 40)
    with pytest.raises(ValueError, match=r"^s:d1: no usable sentence"):
        _window(tok, [bare, head], 0, 0, 8)
    sents, hay = _window(tok, [head, bare, tail], 0, 0, 64)
    assert hay == "s:d0..d2:2" and sum(len(s.split()) for s in sents) >= 64


def test_haystack_id_is_compact_and_spans_docs(tok: Any, corpora: dict[str, list[Any]]) -> None:
    """A ~700-word fixture document cannot hold a 1,024-token window, so every id names
    a span of documents; and the id stays short enough for the [trial] line."""
    for t in range(8):
        hay = make_trial(CFG, tok, "niah_multikey", seed=1, trial=t, corpora=corpora).meta[
            "haystack_id"
        ]
        assert len(hay) < 40
        assert re.fullmatch(r"(pg19|arxiv|wikipedia|essays):d\d+\.\.d\d+:[2-9]", hay), hay


# ------------------------------------------------------------------ the prompts


@pytest.mark.parametrize("task", TASKS)
def test_prompt_is_paired_and_needle_is_present(
    task: str, tok: Any, corpora: dict[str, list[Any]]
) -> None:
    a = make_trial(CFG, tok, task, seed=0, trial=5, corpora=corpora)
    b = make_trial(CFG, tok, task, seed=0, trial=5, corpora=corpora)
    assert isinstance(a, Trial)
    assert a.meta["prompt_sha256"] == b.meta["prompt_sha256"]
    assert a.prefill_ids.shape[1] >= 0.9 * CFG.ctx
    text = tok.decode(a.prefill_ids[0])
    for t in a.targets:
        assert t in text
    assert 0.0 <= a.meta["depth"] <= 1.0 and a.meta["code_family"] in CFG.code_families
    assert re.fullmatch(r"[0-9a-f]{64}", a.meta["prompt_sha256"])


def test_seed_changes_codes_not_design(tok: Any, corpora: dict[str, list[Any]]) -> None:
    a = make_trial(CFG, tok, "niah_single", seed=0, trial=3, corpora=corpora)
    b = make_trial(CFG, tok, "niah_single", seed=1, trial=3, corpora=corpora)
    assert (a.meta["depth"], a.meta["code_family"]) == (b.meta["depth"], b.meta["code_family"])
    assert a.targets != b.targets
    assert a.meta["haystack_id"] != b.meta["haystack_id"]  # the seed also picks the document


def test_needle_lands_at_the_design_depth(tok: Any, corpora: dict[str, list[Any]]) -> None:
    """The design depth is where the queried needle sits, as a fraction of the haystack."""
    at: dict[float, float] = {}
    for trial in range(6):  # trials 0/1 -> 0.05, 2/3 -> 0.4, 4/5 -> 0.95
        t = make_trial(CFG, tok, "niah_single", seed=0, trial=trial, corpora=corpora)
        words = tok.decode(t.prefill_ids[0]).split()
        pos = next(i for i, w in enumerate(words) if t.targets[0] in w)
        at[t.meta["depth"]] = pos / len(words)
    assert at[0.05] < 0.15 and 0.3 < at[0.4] < 0.5 and at[0.95] > 0.85


def test_templates_follow_official_ruler(tok: Any, corpora: dict[str, list[Any]]) -> None:
    """RULER's synthetic templates, singular for one answer and plural otherwise, the
    answer prefix primed after the assistant header; the words family hyphenates."""
    single = make_trial(CFG, tok, "niah_single", seed=0, trial=0, corpora=corpora)
    q = tok.decode(single.query_ids[0])
    assert "What is the special magic number for" in q and q.endswith("in the provided text is")
    assert "<hdr>" in q and q.index("<hdr>") < q.index("mentioned in the provided text is")
    assert "A special magic number is hidden" in tok.decode(single.prefill_ids[0])
    multi = make_trial(CFG, tok, "niah_multivalue", seed=0, trial=0, corpora=corpora)
    assert "What are all the special magic numbers for" in tok.decode(multi.query_ids[0])
    assert len(multi.targets) == 4 and len(set(multi.targets)) == 4
    query = make_trial(CFG, tok, "niah_multiquery", seed=0, trial=0, corpora=corpora)
    assert ", and " in tok.decode(query.query_ids[0]) and len(query.targets) == 4
    words = make_trial(CFG, tok, "niah_multikey", seed=0, trial=6, corpora=corpora)  # c=1
    assert words.meta["code_family"] == "words"
    assert all(re.fullmatch(r"[a-z]+-[a-z]+", t) for t in words.targets)
    vt = make_trial(CFG, tok, "vt", seed=0, trial=0, corpora=corpora)
    assert "Find all variables that are assigned the value" in tok.decode(vt.query_ids[0])
    assert len(vt.targets) == 5 and all(re.fullmatch(r"[A-Z]{5}", t) for t in vt.targets)
    body = tok.decode(vt.prefill_ids[0])
    assert f"VAR {vt.targets[1]} = VAR {vt.targets[0]}" in body


def test_golden_prompt_hashes(tok: Any, corpora: dict[str, list[Any]]) -> None:
    want = json.loads((Path(__file__).parent / "golden" / "gen_v2_prompts.json").read_text())
    assert len(want) == 5 * 2 * 3
    for key, sha in want.items():
        task, seed, trial = key.split("|")
        got = make_trial(CFG, tok, task, seed=int(seed), trial=int(trial), corpora=corpora)
        assert got.meta["prompt_sha256"] == sha, key


def test_v1_builder_is_untouched(tok: Any) -> None:
    """``ruler.build_task`` is the v1 generator, no code motion (PR-L2-22): its cycled
    prefill under the ``tok`` fixture is pinned by the constant below, computed once."""
    from kvdlra.eval import ruler

    pre, q, targets = ruler.build_task(
        tok, "niah_single", 512, trial=0, seed=0, n_keys=8, n_values=4, n_hops=3
    )
    assert hashlib.sha256(pre.numpy().tobytes()).hexdigest() == V1_NIAH_SINGLE_512_PREFILL_SHA
    assert q.shape[1] == 48 and targets == ["72234"]


# `sha256(build_task(tok, "niah_single", 512, 0, 0, 8, 4, 3)[0].numpy().tobytes())` on the
# `tok` fixture, computed 2026-09-19 before gen.py existed (491 int64 ids).
V1_NIAH_SINGLE_512_PREFILL_SHA = "527d2ffc8bcd68229d68cc97a1914dcef2424268e9c9c94099b889a74d48b284"


def test_word_lists_are_well_formed() -> None:
    for words in (ADJECTIVES, NOUNS):
        assert len(words) == 200 and len(set(words)) == 200
        assert all(re.fullmatch(r"[a-z]+", w) for w in words)
        assert not set(words) & set(LABELS)


# ------------------------------------------------------------------ the corpora


def test_load_corpora_reads_the_fixture_and_skips_its_provenance_row(
    corpora: dict[str, list[Any]],
) -> None:
    assert set(corpora) == set(CFG.haystacks)
    assert all(len(docs) == 3 for docs in corpora.values())
    assert all(d["id"] == f"d{i}" for docs in corpora.values() for i, d in enumerate(docs))
    d = corpora["pg19"][0]
    assert set(d) == {"id", "source", "text", "sha256"}
    assert d["sha256"] == hashlib.sha256(d["text"].encode()).hexdigest()


def test_materialize_writes_docs_and_their_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Short and oversized rows are skipped, ids are `d<index>` in materialization order,
    the JSONL's sha256 is returned -- network-free through a fake stream."""
    from kvdlra.eval import haystacks

    calls: list[dict[str, Any]] = []
    assert (haystacks.MIN_CHARS, haystacks.MAX_CHARS) == (2_000, 2_000_000)
    rows = [{"text": "x" * 10}, {"text": "z" * (haystacks.MAX_CHARS + 1)}]  # a stub, a compilation
    rows += [{"text": "a" * 2500}, {"text": "b" * 3000}, {"text": "c" * 4000}]

    def fake_load_dataset(**kw: Any) -> list[dict[str, Any]]:
        calls.append(kw)
        return rows

    monkeypatch.setattr(haystacks, "load_dataset", fake_load_dataset)
    sha = haystacks.materialize("essays", n_docs=2, out=tmp_path)
    (kw,) = calls
    assert kw["streaming"] is True and kw["path"] == "sgoel9/paul_graham_essays"
    assert re.fullmatch(r"[0-9a-f]{40}", kw["revision"]) and "trust_remote_code" not in kw
    payload = (tmp_path / "essays.jsonl").read_bytes()
    assert sha == hashlib.sha256(payload).hexdigest()
    docs = [json.loads(x) for x in payload.decode().splitlines()]
    assert [d["id"] for d in docs] == ["d0", "d1"] and docs[0]["text"] == "a" * 2500
    assert all(d["source"] == "essays" for d in docs)
    assert haystacks.ensure(["essays"], out=tmp_path) == {"essays": sha}  # on disk: no reload
    assert len(calls) == 1
    assert {s for s, (kw, _) in haystacks.SOURCES.items()} == set(CFG.haystacks)
    assert haystacks.SOURCES["pg19"][0]["trust_remote_code"] is True


def test_run_materializes_the_haystacks_a_v2_task_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pod.py run` (and `prepare`) fills `dataset_sha256` with `haystack:<source>` for
    every source the pod's v2 tasks name, materializing only what is not on disk."""
    from kvdlra.eval import haystacks

    made: list[str] = []

    def fake_materialize(source: str, n_docs: int = 64, out: Path = tmp_path) -> str:
        made.append(source)
        (out / f"{source}.jsonl").write_text(f'{{"source": "{source}"}}\n')
        return "sha-" + source

    monkeypatch.setattr(haystacks, "HAYSTACKS", tmp_path)
    monkeypatch.setattr(haystacks, "materialize", fake_materialize)
    cfg = load_pod("w18_g1")
    assert pod._haystack_sha256(cfg) == {}  # no v2 task: nothing named, nothing made
    cfg.tasks = ["ruler_v2_16k", "ppl_16k"]
    got = pod._haystack_sha256(cfg)
    assert sorted(made) == ["arxiv", "essays", "pg19", "wikipedia"]
    assert got == {f"haystack:{s}": f"sha-{s}" for s in made}  # what materialize returned
    again = pod._haystack_sha256(cfg)  # second call: every file is on disk, none re-made
    assert len(made) == 4 and set(again) == set(got)
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in again.values())  # from the bytes
    assert pod.prepare("l2_smoke") == 0 and len(made) == 4  # the same step on its own


# ------------------------------------------------------------------ the config


def test_v2_task_yamls_load_and_hash() -> None:
    hashes = set()
    for name in V2_TASKS:
        t = load_task(name)
        assert isinstance(t, TaskV2Cfg) and t.generator == "v2"
        assert t.n_trials == math.prod(t.design.values())
        assert t.tasks == (GATE1_TASKS if name.endswith("_g1") else TASKS) and t.chunk == 4096
        assert t.haystacks == CFG.haystacks and t.code_families == CFG.code_families
        hashes.add(config_hash(PodCfg(name="p", model="m", arms=["full"], tasks=[name])))
    assert len(hashes) == len(V2_TASKS)
    assert load_task("ruler_v2_16k").n_trials == 12 and load_task("ruler_v2_32k").ctx == 32768
    assert (
        load_task("ruler_v2_16k_g1").n_trials == 24 and load_task("ruler_v2_16k_g2").n_trials == 48
    )
    g32 = load_task("ruler_v2_32k_g1")  # Stage 2's task file, committed with the Stage-1 pods
    assert g32.n_trials == 24 and g32.ctx == 32768 and g32.seeds == [0]
    assert not isinstance(load_task("ruler_inhouse_16k"), TaskV2Cfg)


def test_the_table4_manifest_hash_is_unchanged() -> None:
    """PR-L2-5: the v2 fields live on a subclass, so the hash of every existing task --
    and with it every live manifest `make check` compares against -- is byte-identical."""
    m = json.loads((REPO / "results" / "hygiene_table4_llama" / "manifest.json").read_text())
    assert config_hash(load_pod("hygiene_table4_llama")) == m["config_hash"]


def test_load_task_refuses_a_v2_design_that_does_not_match_n_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tasks").mkdir()
    monkeypatch.setattr("kvdlra.eval.config.ROOT", tmp_path)
    head = "name: bad\ngenerator: v2\nctx: 1024\ntasks: [vt]\nseeds: [0]\n"
    (tmp_path / "tasks" / "bad.yaml").write_text(
        head + "n_trials: 5\ndesign: {haystacks: 2, depths: 3, codes: 4}\n"
    )
    with pytest.raises(ValueError, match=r"bad\.yaml.*n_trials=5.*24"):
        load_task("bad")
    (tmp_path / "tasks" / "bad.yaml").write_text(
        head + "n_trials: 14\ndesign: {haystacks: 1, depths: 7, codes: 2}\n"
    )
    with pytest.raises(ValueError, match=r"depths.*7"):
        load_task("bad")
    (tmp_path / "tasks" / "bad.yaml").write_text(head + "n_trials: 24\n")  # the default design
    assert isinstance(load_task("bad"), TaskV2Cfg)


def test_load_task_refuses_an_unknown_design_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A design key the generator does not read would enter the product `n_trials` must
    equal without entering the enumeration -- refused, naming the file and the key."""
    (tmp_path / "tasks").mkdir()
    monkeypatch.setattr("kvdlra.eval.config.ROOT", tmp_path)
    (tmp_path / "tasks" / "bad.yaml").write_text(
        "name: bad\ngenerator: v2\nctx: 1024\ntasks: [vt]\nseeds: [0]\nn_trials: 24\n"
        "design: {haystacks: 2, depths: 3, codes: 4, seeds: 1}\n"
    )
    with pytest.raises(ValueError, match=r"bad\.yaml.*design key 'seeds'"):
        load_task("bad")


# ------------------------------------------------------------------ the log line


def test_trial_line_round_trips_the_pairing_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the runner prints is what `parse_trial_lines` reads back -- the four pairing
    fields included -- so a harvested pod can verify byte-identical prompts across arms."""
    from kvdlra.eval.records import parse_trial_lines
    from kvdlra.eval.runner import run_pod

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        trial = a[6]
        meta = {"haystack_id": "pg19:d1..d2:2", "depth": 0.05, "code_family": "words",
                "prompt_sha256": "ab" * 32, "ratio": 0.15, "sbits": 0.15}  # fmt: skip
        if trial == 1:
            raise RuntimeError("boom: x=1")
        if trial == 2:
            meta = {"ratio": 0.15, "sbits": 0.15}  # a generator that sets none of them
        return 1, 1.0, meta

    monkeypatch.setattr("kvdlra.eval.gen.run_trial", fake_trial)
    cfg = load_pod("w18_g1")
    cfg.arms, cfg.tasks = ["full"], ["ruler_v2_16k"]
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)
    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    parsed = parse_trial_lines(capsys.readouterr().out, cfg.model, "log")
    assert len(rows) == len(parsed) == 5 * 12
    keys = ("task", "ctx", "arm", "seed", "trial", "hit", "frac", "generator",
            "haystack_id", "depth", "code_family", "prompt_sha256", "error")  # fmt: skip
    trim = [{k: v for k, v in r.items() if k in keys} for r in rows]
    assert trim == [{k: v for k, v in dict(p).items() if k in keys} for p in parsed]
    assert rows[0]["generator"] == "v2" and rows[0]["depth"] == 0.05
    assert rows[0]["haystack_id"] == "pg19:d1..d2:2" and rows[0]["prompt_sha256"] == "ab" * 32
    assert rows[1]["error"] == "RuntimeError: boom: x=1" and rows[1]["haystack_id"] is None
    assert rows[2]["haystack_id"] is None and rows[2]["depth"] is None


def test_archived_trial_line_still_parses() -> None:
    from kvdlra.eval.records import parse_trial_lines

    v1 = "[trial] task=vt ctx=16384 arm=bugSseed-r64-h256 seed=1 trial=3 hit=0 frac=0.500\n"
    l0 = (
        "[trial] task=vt ctx=16384 arm=full seed=0 trial=0 hit=0 frac=0.000 generator=inhouse"
        " error=RuntimeError: a=b\n"
    )
    (a, b) = parse_trial_lines(v1 + l0, "m", "log")
    assert a["hit"] == 0 and a["frac"] == 0.5 and a["generator"] is None and a["error"] is None
    assert b["generator"] == "inhouse" and b["error"] == "RuntimeError: a=b"
    for r in (a, b):
        assert (r["haystack_id"], r["depth"], r["code_family"], r["prompt_sha256"]) == (None,) * 4
