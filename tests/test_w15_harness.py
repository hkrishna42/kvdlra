"""Week-15 A1 harness regression: the RULER attach() scope covers decode.

The Week-15 audit found the published ShadowKV RULER rows (0/0/0/0 at 16K/8B)
were a HARNESS defect, not a method result: ``w10_ruler.retrieve()`` wrapped
only the *prefill* in ``cache.attach(model)``, so ShadowKV's pre-attention
selection hook never ran at decode and ``_selected_chunks`` silently fell back
to the most-recent chunks -- excluding the mid-context needle by construction
(the fall-back warning fired 6,944x in ``results/gpu_logs/w11_goalA.acc.log``).
The fix (a) widens the attach scope in ``w10_ruler.retrieve`` /
``w10_longbench.generate`` to cover ``_decode`` for ALL streaming arms and (b)
promotes the silent fall-back (``shadow_cache._selected_chunks``) to a
RuntimeError whenever selection matters (``k_eff < n_chunks``).

These tests pin both directions on the REAL ``w10_ruler.retrieve()`` path with
a tiny hermetic random-weight Llama (mirroring ``tests/test_shadow_cache.py``):

* ``test_ruler_decode_inside_attach_shadow`` -- retrieve() completes for a
  sparse shadow arm (decode is attached), while the pre-fix harness shape
  (decode outside attach) now raises;
* ``test_ruler_decode_attach_scope_bugs_identity`` -- TRIPWIRE: the scope
  widening is a bit-for-bit no-op for bugS (SurpriseSLASH) arms.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
import torch
import w10_frontier
import w10_ruler
from transformers import LlamaConfig, LlamaForCausalLM

H, D = 2, 16
N_FEATURES = H * D


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=2048,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


class _StubTok:
    """Minimal tokenizer stand-in for ``_decode``: records each generated-id list
    and renders it as space-joined ints (deterministic and reversible, so text
    equality == token-id equality)."""

    def __init__(self) -> None:
        self.decoded: list[list[int]] = []

    def decode(self, ids: list[int]) -> str:
        self.decoded.append(list(ids))
        return " ".join(str(i) for i in ids)


def _build_arm(
    model: LlamaForCausalLM, methods: list[str], t: int, **over: object
) -> dict[str, Any]:
    """One arm via the REAL ``w10_frontier.build_arms`` (not a hand-rolled dict),
    so the test exercises the exact factory the harness runs."""
    ns = argparse.Namespace(
        methods=methods,
        recent_window=8,
        absorb_block=4,
        ranks=[8],
        hh_budgets=[4],
        hh_neighbor=0,
        chunk=0,
        shadow_ranks=[8],
        shadow_topk=2,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    arms = w10_frontier.build_arms(ns, model, t)
    assert len(arms) == 1
    return arms[0]


def test_ruler_decode_inside_attach_shadow(tiny_model: LlamaForCausalLM) -> None:
    """The fixed ``retrieve()`` runs ShadowKV decode INSIDE ``attach()``.

    Control first: replaying the PRE-FIX harness shape (attach around the
    prefill only, ``_decode`` outside) raises RuntimeError -- which both proves
    this config is in the selection-matters regime (``k_eff < n_chunks``; 96
    tokens -> 10 middle chunks, top_k=2) and that ``retrieve()``'s success below
    is attributable to the widened attach scope, not to a degenerate config."""
    t = 96
    arm = _build_arm(tiny_model, ["shadow"], t=t)
    hay, query = _prompt(t), _prompt(4, seed=2)

    # Control: the pre-fix harness shape must now fail loudly, not silently.
    cache = arm["make"]()
    with torch.no_grad():
        with cache.attach(tiny_model):
            tiny_model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        with pytest.raises(RuntimeError, match="attach"):
            w10_ruler._decode(
                tiny_model, _StubTok(), cache, query, t, "cpu", block=False, max_new=2
            )

    # The real, fixed retrieve(): decode inside attach -> completes cleanly.
    stub = _StubTok()
    hit, ratio, frac = w10_ruler.retrieve(
        tiny_model, stub, arm, hay, query, ["999999"], "cpu", 0, N_FEATURES, H, 2
    )
    assert stub.decoded  # decode actually ran to completion
    assert ratio > 0.0
    assert 0.0 <= frac <= 1.0
    assert isinstance(hit, bool)


def test_ruler_decode_attach_scope_bugs_identity(tiny_model: LlamaForCausalLM) -> None:
    """TRIPWIRE (pre-registered): widening the attach() scope must be a
    bit-for-bit no-op for bugS (SurpriseSLASH) arms.

    ``BugStreamingCache.attach`` installs hooks ONLY for retention "attn"/
    "blend" (otherwise it yields without registering anything), and bugS arms
    run retention="lowrank_surprise" with hh_select="surprise" -- attach-free by
    construction. So the fixed ``retrieve()`` (prefill AND decode attached) must
    generate EXACTLY the tokens of a direct manual chunked-prefill + decode with
    NO attach anywhere (the simplest honest identity: it brackets both the
    pre-change and post-change scopes).

    If this test ever fails, the attach widening changed bugS decode behaviour:
    STOP, scope the widening in ``w10_ruler.retrieve`` / ``w10_longbench.
    generate`` to shadow arms only (``arm["kind"] == "shadow"``), and document
    the mechanism here."""
    t, chunk, max_new = 96, 16, 4
    arm = _build_arm(tiny_model, ["bugslash"], t=t, chunk=chunk)
    assert arm["name"].startswith("bugS-")
    hay, query = _prompt(t), _prompt(4, seed=2)

    tok_a = _StubTok()
    w10_ruler.retrieve(
        tiny_model, tok_a, arm, hay, query, ["999999"], "cpu", chunk, N_FEATURES, H, max_new
    )

    # Manual reference: the identical prefill + decode, with NO attach at all.
    cache = arm["make"]()
    tok_b = _StubTok()
    with torch.no_grad():
        w10_frontier._prefill_chunked(tiny_model, cache, hay, chunk)
    w10_ruler._decode(tiny_model, tok_b, cache, query, t, "cpu", block=False, max_new=max_new)

    assert tok_a.decoded and tok_b.decoded
    assert tok_a.decoded[-1] == tok_b.decoded[-1]  # bit-identical generated ids
