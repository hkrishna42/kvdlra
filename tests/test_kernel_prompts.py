"""The 16 prompts of the full-model correctness precondition (prereg/kernel_smoke.md §4),
fixed here BEFORE the kernel exists.

Source and selection rule, for the launch entry in docs/plan/DECISIONS.md:

* On the pod (Llama-3.1-8B): the 16 non-overlapping 4096-token windows of the PG-19
  validation token stream that ``kvdlra.eval.data.load_corpus_ids(tok, device,
  corpus="pg19-val")`` returns -- the same stream ``ppl_16k_pg19val`` cuts its windows from,
  whose sha256 the pod manifest records under ``dataset_sha256["pg19-val"]`` -- starting at
  token offsets ``i * 131072`` for ``i = 0..15`` (windows spread over the first ~2M of its
  2,968,224 tokens, so they fall in different books). Each is prefilled whole and 32 tokens
  are decoded greedily under both decode paths.
* On CPU (the tiny random-weight Llama of tests/conftest.py): 16 seeded random token
  prompts, ``torch.randint(0, 256, (64,), generator=Generator().manual_seed(s))`` for
  ``s = 0..15``, 16 greedy tokens each. The literals below pin both lists; a change to
  either is a change to this file, committed and visible.
"""

from __future__ import annotations

import hashlib
import itertools

import torch

from kvdlra.eval.config import CORPUS_TOKENS
from kvdlra.kernel import prompts

STARTS = (
    0, 131072, 262144, 393216, 524288, 655360, 786432, 917504,
    1048576, 1179648, 1310720, 1441792, 1572864, 1703936, 1835008, 1966080,
)  # fmt: skip


def test_the_pod_prompts_are_the_committed_windows() -> None:
    assert prompts.PROMPT_CORPUS == "pg19-val"
    assert (prompts.PROMPT_TOKENS, prompts.N_NEW, prompts.PROMPT_STRIDE) == (4096, 32, 131072)
    assert prompts.PROMPT_STARTS == STARTS and len(STARTS) == 16
    # non-overlapping, ascending, and inside the corpus the ppl task already cuts
    assert all(b - a >= prompts.PROMPT_TOKENS for a, b in itertools.pairwise(STARTS))
    assert STARTS[-1] + prompts.PROMPT_TOKENS == 1970176 <= CORPUS_TOKENS["pg19-val"]
    ids = torch.arange(STARTS[-1] + prompts.PROMPT_TOKENS)
    wins = prompts.prompt_windows(ids)
    assert [int(w[0]) for w in wins] == list(STARTS)
    assert all(w.shape == (prompts.PROMPT_TOKENS,) for w in wins)


def test_prompt_windows_refuse_a_short_stream() -> None:
    try:
        prompts.prompt_windows(torch.arange(1000))
    except ValueError as exc:
        assert "1970176" in str(exc)
    else:
        raise AssertionError("a stream shorter than the last window must be refused")


def test_the_tiny_prompts_are_the_committed_seeds() -> None:
    assert tuple(range(16)) == prompts.TINY_PROMPT_SEEDS
    assert (prompts.TINY_PROMPT_TOKENS, prompts.TINY_N_NEW, prompts.TINY_VOCAB) == (64, 16, 256)
    ps = prompts.tiny_prompts()
    assert len(ps) == 16 and all(p.shape == (64,) and p.dtype == torch.int64 for p in ps)
    assert ps[0][:8].tolist() == [172, 47, 117, 192, 67, 251, 195, 103]
    assert ps[15][:8].tolist() == [200, 245, 140, 133, 119, 128, 156, 155]
    assert int(torch.stack(ps).sum()) == 133202
    digest = hashlib.sha256(torch.stack(ps).numpy().tobytes()).hexdigest()
    assert digest == "c0d9d9fa04b015acf178992b14056c103000e8cfb50be4b1672a0fa9bd2b659f"
