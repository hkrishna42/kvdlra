"""The prompts of the full-model correctness precondition (prereg/kernel_smoke.md §4).

Fixed here before the kernel exists; ``tests/test_kernel_prompts.py`` is the committed
literal record (source, selection rule and fingerprints) and fails if this module drifts.
The constants below are defined in ``kvdlra.eval.config`` (torch-free, imported by the
`scripts/pod.py check` subprocesses) and re-exported here; this module still imports no
torch at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kvdlra.eval.config import N_NEW, PROMPT_CORPUS, PROMPT_STARTS, PROMPT_STRIDE, PROMPT_TOKENS

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["N_NEW", "PROMPT_CORPUS", "PROMPT_STARTS", "PROMPT_STRIDE", "PROMPT_TOKENS"]

# The pod (Llama-3.1-8B): 16 non-overlapping windows of the PG-19 validation token stream
# `kvdlra.eval.data.load_corpus_ids(tok, device, corpus="pg19-val")` returns -- the stream
# `ppl_16k_pg19val` cuts its windows from -- at offsets i * PROMPT_STRIDE, each prefilled
# whole and decoded greedily for N_NEW tokens under both decode paths.

# CPU (the tiny random-weight Llama of tests/conftest.py): seeded random token prompts.
TINY_PROMPT_SEEDS: tuple[int, ...] = tuple(range(16))
TINY_PROMPT_TOKENS = 64
TINY_N_NEW = 16
TINY_VOCAB = 256


def prompt_windows(ids: Tensor) -> list[Tensor]:
    """The 16 pod prompts as slices of the 1-D token stream ``ids``."""
    need = PROMPT_STARTS[-1] + PROMPT_TOKENS
    if int(ids.shape[0]) < need:
        raise ValueError(
            f"the prompt stream holds {int(ids.shape[0])} tokens; the 16 windows need {need}"
        )
    return [ids[s : s + PROMPT_TOKENS] for s in PROMPT_STARTS]


def tiny_prompts() -> list[Tensor]:
    """The 16 CPU prompts, one seeded draw each."""
    import torch

    return [
        torch.randint(
            0, TINY_VOCAB, (TINY_PROMPT_TOKENS,), generator=torch.Generator().manual_seed(s)
        )
        for s in TINY_PROMPT_SEEDS
    ]
