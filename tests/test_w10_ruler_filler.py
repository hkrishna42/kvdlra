"""Week-18 W3: realistic RULER filler + needle-depth grid.

The panel flagged the self-authored 10-sentence cyclic filler as an external-validity
hole. ``--filler {wikitext,pg19}`` draws a seed-shuffled natural-text haystack; the
default ``cycle`` stays bit-identical to the archived benchmark. ``--depths`` sweeps the
niah_single needle depth. These tests use an injected sentence pool + a word-count stub
tokenizer, so they need no network and no real model.
"""

from __future__ import annotations

from typing import Any

import pytest
import w10_ruler as wr
from w4_needle import _FILLER
from w10_ruler import _filler_to, build_task


class _WordTok:
    """Callable stub: token count == word count (enough to drive _filler_to's loop)."""

    def __call__(self, text: str) -> Any:
        class _R:
            input_ids: list[str]

        r = _R()
        r.input_ids = text.split()
        return r


def _old_cycle(tok: _WordTok, ctx: int) -> list[str]:
    """The pre-Week-18 algorithm, reproduced to pin bit-identity."""
    sentences: list[str] = []
    i = 0
    while len(tok(" ".join(sentences)).input_ids) < ctx:
        sentences.append(_FILLER[i % len(_FILLER)])
        i += 1
    return sentences


def test_filler_cycle_is_bit_identical_to_archived() -> None:
    tok = _WordTok()
    for ctx in (5, 23, 100):
        assert _filler_to(tok, ctx) == _old_cycle(tok, ctx)
        assert _filler_to(tok, ctx, filler="cycle") == _old_cycle(tok, ctx)


def test_realistic_filler_reaches_ctx_and_draws_from_pool() -> None:
    tok = _WordTok()
    pool = [f"sentence number {i} has several filler words here" for i in range(50)]
    sents = _filler_to(tok, 60, filler="wikitext", pool=pool, seed=0, trial=0)
    assert len(tok(" ".join(sents)).input_ids) >= 60
    assert set(sents) <= set(pool)  # every filler line came from the corpus pool


def test_realistic_filler_seeded_shuffle_is_deterministic_and_varies() -> None:
    tok = _WordTok()
    pool = [f"unique filler sentence token {i} here now" for i in range(80)]
    a = _filler_to(tok, 80, filler="wikitext", pool=pool, seed=0, trial=0)
    a2 = _filler_to(tok, 80, filler="wikitext", pool=pool, seed=0, trial=0)
    b = _filler_to(tok, 80, filler="wikitext", pool=pool, seed=0, trial=1)
    assert a == a2  # same (seed, trial) -> identical haystack
    assert a != b  # different trial -> different haystack (external validity)


def test_realistic_filler_requires_pool() -> None:
    with pytest.raises(ValueError, match="requires a non-empty pool"):
        _filler_to(_WordTok(), 10, filler="wikitext", pool=None)


def test_depth_grid_places_needle_at_requested_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """--depths lands the niah_single needle at depths[trial % len]*n; the archived
    path (depths=None) keeps mid-depth. We capture the assembled body via a stubbed
    _templated and locate the passcode sentence among the filler."""
    captured: dict[str, str] = {}

    def _fake_templated(tok: Any, body: str, q: str) -> tuple[Any, Any]:
        captured["body"] = body
        return body, q  # retrieve() never runs here; we only inspect the body

    monkeypatch.setattr(wr, "_templated", _fake_templated)
    tok = _WordTok()
    pool = [f"filler line {i} with enough words to count" for i in range(400)]

    def needle_depth_fraction(depth: float) -> float:
        build_task(
            tok,
            "niah_single",
            300,
            trial=0,
            seed=0,
            n_keys=8,
            n_values=4,
            n_hops=3,
            filler="wikitext",
            pool=pool,
            depths=[depth],
        )
        body = captured["body"]
        # pool sentences are uniform length, so the character offset of the needle is
        # a faithful proxy for its fractional sentence depth.
        return body.index("secret passcode") / len(body)

    shallow = needle_depth_fraction(0.1)
    deep = needle_depth_fraction(0.9)
    assert shallow < 0.35  # a depth-0.1 needle sits early
    assert deep > 0.7  # a depth-0.9 needle sits late
    assert shallow < deep


def test_filler_to_is_memoized_per_haystack() -> None:
    """Week-19: `_filler_to` grew the haystack one sentence at a time, re-tokenizing the
    whole text each step (O(n^2) tokenizer calls: ~5 min per trial at 64K). The same
    (ctx, filler, seed, trial) haystack is rebuilt for every arm and trial, so it is
    memoized; the second call must not tokenize again and must return equal output."""
    import w10_ruler

    class _CountTok:
        calls = 0

        def __call__(self, text: str) -> object:
            _CountTok.calls += 1
            n = len(text.split())
            return type("Enc", (), {"input_ids": list(range(n))})()

    tok = _CountTok()
    w10_ruler._filler_cached.cache_clear()
    first = w10_ruler._filler_to(tok, 64, filler="cycle")
    n_calls = _CountTok.calls
    assert n_calls > 1
    second = w10_ruler._filler_to(tok, 64, filler="cycle")
    assert second == first and _CountTok.calls == n_calls
    assert w10_ruler._filler_to(tok, 128, filler="cycle") != first  # a different ctx rebuilds
