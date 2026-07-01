"""Greedy generation with (or without) :class:`kvdlra.press.BUGPress`.

Two jobs (Week-3 Tue + Wed, ``docs/PLAN.md``):

* **Single-config generation** (Tue) -- ``--press bug --rank 32`` runs the model
  with BUGPress applied during pre-fill and prints the greedy continuation of
  each prompt. ``--press none`` is the uncompressed baseline.
* **Parity check** (Wed, ``--parity``) -- runs the baseline and BUGPress on the
  built-in prompt suite, prints a side-by-side markdown table, flags exact-match
  vs. divergence, and (with ``--out``) writes the table to a file.

The metric is *behavioral*: does replacing the KV cache with its rank-``r``
dynamical-low-rank reconstruction change greedy decoding? High rank should match
the baseline; low rank should degrade gracefully (coherent, on-topic) rather than
produce garbage -- a garbage output signals a RoPE round-trip or reshape bug.

Runs on CPU (the 1B model is small); pass ``--device cuda`` on the pod.

Examples
--------
    uv run python scripts/generate_with_press.py --press bug --rank 32
    uv run python scripts/generate_with_press.py --parity --ranks 64 32 16 8 \
        --out results/w3-parity.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from kvdlra.press import BUGPress
from kvdlra.utils.seed import seed_everything

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Two suites. SHORT prompts (T <= ~10 tokens) are a *no-regression* sanity check:
# with n_sink=4 there is almost nothing to compress (T - n_sink <= rank), so the
# press should be a near-no-op and outputs should match the baseline exactly.
# LONG prompts (a few hundred tokens) actually stress the low-rank model -- there
# T >> rank, so the rank-r reconstruction genuinely discards information and
# greedy decoding is expected to diverge gracefully as the rank drops (a garbage
# continuation, rather than a graceful drift, would signal a bug).
SHORT_PROMPTS: list[str] = [
    "The capital of France is",
    "Water is composed of hydrogen and",
    "The first president of the United States was",
    "In machine learning, a neural network is",
    "The three primary colors are red, blue, and",
    "Photosynthesis is the process by which plants",
    "The speed of light in a vacuum is approximately",
    "To make a cup of tea, first you",
    "The theory of evolution was proposed by",
    "A prime number is a natural number greater than one that",
]

_RELATIVITY = (
    "The theory of relativity was developed by Albert Einstein in the early twentieth "
    "century and fundamentally changed our understanding of space, time, and gravity. "
    "Special relativity, published in 1905, introduced the idea that the speed of light "
    "in a vacuum is the same for all observers regardless of their motion, and that the "
    "laws of physics take the same form in all inertial frames. From these two postulates "
    "Einstein derived that measurements of time and length depend on the observer's motion, "
    "leading to time dilation and length contraction. General relativity, completed in "
    "1915, went further and described gravity not as a force but as the curvature of "
    "spacetime produced by mass and energy. Massive objects bend the spacetime around them, "
    "and other objects move along the resulting curved paths. In summary, the central idea "
    "of relativity is that"
)
_PHOTOSYNTHESIS = (
    "Photosynthesis is the biological process by which green plants, algae, and some "
    "bacteria convert light energy into chemical energy stored in sugars. It takes place "
    "mainly in the chloroplasts, which contain the green pigment chlorophyll that absorbs "
    "light most strongly in the blue and red parts of the spectrum. The process has two "
    "stages. In the light-dependent reactions, energy from sunlight is used to split water "
    "molecules, releasing oxygen as a by-product and producing the energy carriers ATP and "
    "NADPH. In the light-independent reactions, also called the Calvin cycle, that stored "
    "energy is used to fix carbon dioxide from the air into glucose. To summarize, the two "
    "raw materials that plants take in for photosynthesis are"
)
_HISTORY = (
    "The Industrial Revolution was a period of rapid economic and technological change that "
    "began in Britain in the second half of the eighteenth century and gradually spread to "
    "the rest of Europe and North America. It marked a transition from economies based on "
    "agriculture and handicraft to economies dominated by industry and machine manufacturing. "
    "Key developments included the steam engine, which provided a reliable source of power; "
    "the mechanization of textile production in factories; and improvements in iron and steel "
    "making. New transport networks of canals and, later, railways moved raw materials and "
    "finished goods over long distances. These changes caused large populations to move from "
    "the countryside into rapidly growing industrial cities. Overall, the single most "
    "important new source of power that drove the Industrial Revolution was the"
)
LONG_PROMPTS: list[str] = [_RELATIVITY, _PHOTOSYNTHESIS, _HISTORY]


def load_model(model_id: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the model (float32, eval) and tokenizer onto ``device``."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    return model, tokenizer


def _generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> str:
    """Greedy-decode ``max_new_tokens`` continuation tokens for one prompt."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(  # type: ignore[operator]
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = cast(str, tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True))
    return text.strip()


def _make_press(rank: int, pre_rope: bool, compress_values: bool) -> BUGPress:
    return BUGPress(rank=rank, pre_rope=pre_rope, compress_values=compress_values)


def _prompt_label(prompt: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """A compact, table-safe label: token count + a short prefix/suffix."""
    n_tok = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    if len(prompt) <= 60:
        return f"[{n_tok}tok] `{prompt}`"
    return f"[{n_tok}tok] `{prompt[:32]}...{prompt[-20:]}`"


def run_single(args: argparse.Namespace) -> None:
    """Generate for one press configuration and print continuations."""
    model, tokenizer = load_model(args.model, args.device)
    prompts = SHORT_PROMPTS + LONG_PROMPTS
    press_ctx = None
    if args.press == "bug":
        press = _make_press(args.rank, args.pre_rope, not args.no_values)
        press_ctx = press(model)
        header = f"BUGPress(rank={args.rank}, {'pre' if args.pre_rope else 'post'}-RoPE)"
    else:
        header = "baseline (no press)"

    print(f"# Generation -- {header}\n")
    for prompt in prompts:
        if press_ctx is not None:
            with press_ctx:
                text = _generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
        else:
            text = _generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
        print(f"- **{_prompt_label(prompt, tokenizer)}** -> {text!r}")


def _parity_section(
    name: str,
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> list[str]:
    """One suite's baseline-vs-BUG table + exact-match summary, as markdown lines."""
    baselines = [_generate(model, tokenizer, p, args.max_new_tokens, args.device) for p in prompts]

    per_rank: dict[int, list[tuple[str, bool]]] = {}
    for rank in args.ranks:
        press = _make_press(rank, args.pre_rope, not args.no_values)
        rows: list[tuple[str, bool]] = []
        with press(model):
            for prompt, base in zip(prompts, baselines, strict=True):
                text = _generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
                rows.append((text, text == base))
        per_rank[rank] = rows

    lines = [f"## {name}", ""]
    header = "| Prompt | baseline | " + " | ".join(f"r={r}" for r in args.ranks) + " |"
    sep = "|" + "---|" * (2 + len(args.ranks))
    lines += [header, sep]
    for i, prompt in enumerate(prompts):
        cells = [_prompt_label(prompt, tokenizer), f"{baselines[i]!r}"]
        for rank in args.ranks:
            text, match = per_rank[rank][i]
            cells.append("= (exact)" if match else f"{text!r}")
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")

    lines += ["", f"Exact-match rate for **{name}**:"]
    for rank in args.ranks:
        n_match = sum(1 for _, m in per_rank[rank] if m)
        cr = _make_press(rank, args.pre_rope, True).compression_ratio
        lines.append(f"- rank {rank}: {n_match}/{len(prompts)} exact, compression_ratio={cr:.3f}")
    lines.append("")
    return lines


def run_parity(args: argparse.Namespace) -> None:
    """Baseline vs. BUGPress on both suites; markdown tables + divergence summary."""
    model, tokenizer = load_model(args.model, args.device)
    rope = "pre" if args.pre_rope else "post"

    lines = [
        f"# Week-3 generation parity: BUGPress vs. baseline ({rope}-RoPE, "
        f"values={'off' if args.no_values else 'on'})",
        "",
        f"Model: `{args.model}` -- greedy decode, {args.max_new_tokens} new tokens. "
        "'=' means the continuation is byte-identical to the no-press baseline.",
        "",
        "SHORT prompts (T-n_sink <= rank) are a no-regression sanity check: the press "
        "has almost nothing to compress and should match the baseline exactly. LONG "
        "prompts (T >> rank) genuinely exercise the rank-r model; graceful divergence "
        "as rank drops is expected, garbage is not.",
        "",
    ]
    lines += _parity_section("SHORT prompts", SHORT_PROMPTS, model, tokenizer, args)
    lines += _parity_section("LONG prompts", LONG_PROMPTS, model, tokenizer, args)

    report = "\n".join(lines)
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n")
        print(f"\n[wrote {out_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--press", default="bug", choices=["none", "bug"])
    parser.add_argument("--rank", type=int, default=32, help="rank for single-config mode")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-values", action="store_true", help="compress keys only")
    rope = parser.add_mutually_exclusive_group()
    rope.add_argument("--pre-rope", dest="pre_rope", action="store_true", default=True)
    rope.add_argument("--post-rope", dest="pre_rope", action="store_false")
    parser.add_argument("--parity", action="store_true", help="run baseline-vs-BUG parity suite")
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[64, 32, 16, 8],
        help="ranks to compare in --parity mode",
    )
    parser.add_argument("--out", default=None, help="write the parity markdown table here")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.parity:
        run_parity(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
