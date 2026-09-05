# Week-19 A2 — the official-benchmark anchor: in-repo generator vs NVIDIA RULER

The retrieval evidence through Week-18 comes from an in-repo RULER-style generator
(`scripts/w10_ruler.py`, building on `w4_needle.py` / `w5_ruler.py`). The Week-18 panel
called this a standing reject reason ("self-authored benchmark; selection and confirmation
share the generator"). The Week-18 WikiText filler (`--filler wikitext`) half-closed it;
this note documents the other half: the SAME arms and decode protocol run on prompts from
the **official NVIDIA RULER generator** (github.com/NVIDIA/RULER, commit `c3f5e3b`),
via `scripts/w19_official_ruler.py`, pod MODE `a2` in `scripts/pod/w19.sh`.

## What is different (prompt-level diff)

| aspect | in-repo generator (`w10_ruler.py`) | official RULER (`c3f5e3b`, `synthetic.yaml`) |
|---|---|---|
| haystack | 10 fixed sentences cycled (`cycle`, archived path) or WikiText-103 / PG-19 sentences seed-shuffled per trial (`--filler`) | `noise` = one 5-sentence line repeated; `essay` = Paul Graham essays, sentence-split; `needle` = distractor needles only (multikey_2/3) |
| needle sentence | `The secret passcode is {5-digit}.` / `The {label} code is {code}.` (labels from a fixed word list) | `One of the special magic {numbers\|uuids} for {key} is: {value}.` — keys are wonderwords adjective-noun pairs (e.g. `capable-radiosonde`), values 7-digit numbers or uuids |
| placement | single: mid-depth + trial jitter, or a `--depths` grid; mk/mv/vt: evenly spaced | uniform random depth per sample (`token_position_answer` recorded) |
| question | `What is the secret passcode? Reply with only the number.` | `What is the special magic number for {key} mentioned in the provided text?` |
| answer prefix | none | ` The special magic number for {key} mentioned in the provided text is` — appended **after** the assistant header (RULER's `meta-llama3` template primes the completion); we do the same |
| single-needle tasks | `niah_single` | `niah_single_1` (noise, numbers), `_2` (essay, numbers), `_3` (essay, uuids) |
| multi-key | 8 keys, one queried | `niah_multikey_1` (4 keys in essays), `_2` (1 key among many distractor needles, numbers), `_3` (uuid keys and values among distractors) |
| multi-value | 4 values of one key, list all | `niah_multivalue` (4 values, essay) |
| multi-query | — | `niah_multiquery` (4 keys queried in one question) |
| variable tracking | forward: chain `VAR X0 = 12345.` … 3 hops, ask the value of the last variable (answer = the number) | **reverse**: 4 hops, `Find all variables that are assigned the value {v}` (answer = all 5 variable NAMES) |
| generation budget | 12 tokens (single) / 40 (others) | `tokens_to_generate`: niah 128, vt 30 |
| scoring | every target string appears in the decoded text (case-sensitive) | `string_match_all` (case-insensitive); ours is at least as strict |
| decoded tail | template-derived, floored at 48 tokens (validated Llama slices) | exact question + assistant header + answer prefix (needles sit at any depth, so no body text may be decoded verbatim) |
| chat template | tokenizer `apply_chat_template` | same |
| samples | 12 per cell (6 trials × 2 seeds) | 12 per task (`--num_samples 12 --random_seed 42`) |

Shared by construction: the arms (`build_arms`), chunked prefill for streaming arms and
the quant baseline, single-shot prefill for scorer presses, decode at true positions, the
memory accounting, and the `[trial]` / `acc=` row formats (`w18_intervals.py` reads both).

## Protocol on the pod (`MODE=a2`, Llama-3.1-8B, 16K)

1. `pip install wonderwords tenacity nltk html2text beautifulsoup4`; nltk `punkt` data.
2. Clone RULER, `git checkout c3f5e3b`, download the Paul Graham essays with their script.
3. `prepare.py --benchmark synthetic --task <t> --tokenizer_type hf --tokenizer_path $MODEL
   --max_seq_length 16384 --num_samples 12 --random_seed 42 --model_template_type base` for
   the nine tasks above (the `base` template; our script applies the chat template and the
   record's `answer_prefix`).
4. `w19_official_ruler.py` for: `full`, `think-c0.5`, `palu-r0.5`, `ea-k0.1`,
   `quant-{2,4}bit-kivi`, and the flagship `bugSseed-r64-h256`.

## Results

Pending (pod `a2` not yet launched; A1 runs first per `docs/week19-kickoff.md`).
