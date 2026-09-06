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

## Results (a2-llama, SHA c331ebd, A100-40GB, 2026-09-06; n=12 per cell, 16K)

Accuracy = RULER `string_match_all` (every reference output present in the 128/30-token generation). s1/s2/s3 = niah_single_1/2/3 (noise / essay numbers / essay uuids); mk1..3 = niah_multikey_1..3; mv/mq = multivalue/multiquery.

| arm | stored | s1 | s2 | s3 | mk1 | mk2 | mk3 | mv | mq | vt | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 1.000x | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 0.99 |
| `think-c0.5` | 0.750x | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 0.92 | 0.98 |
| `palu-r0.5` | 0.504x | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 0.42 | 0.92 | 1.00 | 0.92 |
| `quant-4bit-kivi` | 0.287x | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 0.99 |
| `quant-2bit-kivi` | 0.163x | 1.00 | 1.00 | 0.83 | 1.00 | 0.92 | 0.58 | 0.83 | 1.00 | 0.67 | 0.87 |
| `bugSseed-r64-h256` | 0.151x | 0.83 | 1.00 | 0.83 | 0.83 | 1.00 | 0.83 | 0.83 | 0.67 | 0.33 | 0.79 |
| `ea-k0.1` | 0.100x | 1.00 | 0.17 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.20 |

### Paired McNemar on the official cells (A = flagship; A>B / B>A = discordant records)

| vs | task | A>B | B>A | p | sig |
|---|---|---|---|---|---|
| `quant-2bit-kivi` | mv | 1 | 1 | 1.0000 | no |
| `quant-2bit-kivi` | vt | 1 | 5 | 0.2188 | no |
| `quant-2bit-kivi` | mk1 | 0 | 2 | 0.5000 | no |
| `quant-2bit-kivi` | mk2 | 1 | 0 | 1.0000 | no |
| `quant-2bit-kivi` | mk3 | 4 | 1 | 0.3750 | no |
| `quant-2bit-kivi` | mq | 0 | 4 | 0.1250 | no |
| `quant-2bit-kivi` | s1 | 0 | 2 | 0.5000 | no |
| `quant-2bit-kivi` | s3 | 1 | 1 | 1.0000 | no |
| `quant-4bit-kivi` | mv | 0 | 2 | 0.5000 | no |
| `quant-4bit-kivi` | vt | 0 | 8 | 0.0078 | **YES** |
| `quant-4bit-kivi` | mk1 | 0 | 2 | 0.5000 | no |
| `quant-4bit-kivi` | mk3 | 0 | 1 | 1.0000 | no |
| `quant-4bit-kivi` | mq | 0 | 4 | 0.1250 | no |
| `quant-4bit-kivi` | s1 | 0 | 2 | 0.5000 | no |
| `quant-4bit-kivi` | s3 | 0 | 2 | 0.5000 | no |
| `ea-k0.1` | mv | 10 | 0 | 0.0020 | **YES** |
| `ea-k0.1` | vt | 2 | 2 | 1.0000 | no |
| `ea-k0.1` | mk1 | 7 | 1 | 0.0703 | no |
| `ea-k0.1` | mk2 | 12 | 0 | 0.0005 | **YES** |
| `ea-k0.1` | mk3 | 10 | 0 | 0.0020 | **YES** |
| `ea-k0.1` | mq | 8 | 0 | 0.0078 | **YES** |
| `ea-k0.1` | s1 | 0 | 2 | 0.5000 | no |
| `ea-k0.1` | s2 | 10 | 0 | 0.0020 | **YES** |
| `ea-k0.1` | s3 | 10 | 0 | 0.0020 | **YES** |
| `palu-r0.5` | mv | 6 | 1 | 0.1250 | no |
| `palu-r0.5` | vt | 0 | 8 | 0.0078 | **YES** |
| `palu-r0.5` | mk1 | 0 | 2 | 0.5000 | no |
| `palu-r0.5` | mk3 | 0 | 1 | 1.0000 | no |
| `palu-r0.5` | mq | 0 | 3 | 0.2500 | no |
| `palu-r0.5` | s1 | 0 | 2 | 0.5000 | no |
| `palu-r0.5` | s3 | 0 | 2 | 0.5000 | no |
| `think-c0.5` | mv | 0 | 2 | 0.5000 | no |
| `think-c0.5` | vt | 0 | 7 | 0.0156 | **YES** |
| `think-c0.5` | mk1 | 0 | 2 | 0.5000 | no |
| `think-c0.5` | mk3 | 0 | 1 | 1.0000 | no |
| `think-c0.5` | mq | 0 | 4 | 0.1250 | no |
| `think-c0.5` | s1 | 0 | 2 | 0.5000 | no |
| `think-c0.5` | s3 | 0 | 2 | 0.5000 | no |

### Reading

- At matched stored bytes the flagship (0.151x) and KIVI 2-bit (0.163x) are statistically indistinguishable on every official task; the flagship's in-repo multi-value edge does **not** reproduce here (1 vs 1 discordant), and on average it trails (0.80 vs 0.87).
- The 4-bit arm, ThinK and Palu beat the flagship on variable tracking (p = 0.008 / 0.016 / 0.008) and tie it elsewhere; full KV's own ceiling on multikey_3 (uuid keys) is 0.92.
- Eviction at 0.1x (ExpectedAttention) is separated from the flagship in its disfavor on 6 of 9 tasks (p <= 0.008) and collapses on every essay-haystack cell.
- The flagship's 14 misses sit at needle depths 0.15–0.95 (`results/w19-a2-flagship-misses.md`): not a warm-up-window or recency effect; scattered misses the quantizers do not make.
- Generator-vs-official: the in-repo generator overstated the flagship's advantage over 2-bit quantization on multi-value/multi-query-type tasks and understated its variable-tracking gap to the 0.29–0.75x baselines; the eviction collapse and the single-needle parity transfer.
