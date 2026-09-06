# Reproducibility / Artifact Review — kvdlra exit-gate re-review (2026-09-06)

Reviewer: reproducibility & artifact dimension. Repo `/Users/hari/Desktop/kv-dlra`, branch `week7`.
Paper read at `5bc772f`; HEAD moved to `9fd122e` mid-review (docs-only: `git diff --stat 5bc772f 9fd122e`
= `docs/week19-handover.md` +105). Read-only: `git`/`grep`/`cat`, one CPU python probe (no repo edits, no pods).
Previous repro review: `docs/reviews/2026-09-01/review-repro.md` (7/10).

## Verdict

**7/10 (accept-level), no fatal flaw.** The Week-19 program closed the two structural gaps the prior
review flagged as Major: pods now clone by commit SHA (`scripts/pod/w18_boot.sh:47-50`) and every
retrieval cell in the paper has a committed per-trial record (`results/w19_pertrial/*.txt`, counts
verified complete below), from which the intervals, McNemar tests, tables, figures and dashboard
regenerate deterministically (`scripts/w18_intervals.py`, `scripts/w19_a1_report.py`,
`scripts/w19_figures.py`, `scripts/w19_dashboard.py`). All 20 Week-19 *retrieval/persistence/64K*
numbers I spot-checked trace to a committed line-file at the stated line.

What keeps it at 7 rather than 8: (1) the environment stamps the paper says the pods "stamped into the
harvested logs" (`paper/main.tex:970-973`) live only in **gitignored** `results/w19_harvest/*.raw`
(`.gitignore:52`) — there is no `results/w19-env-provenance.txt` counterpart to the W18 one; (2) every
**Week-18 perplexity** reused in the paper (flagship 5.31/8.33/5.50/3.76/8.18/35.1, marquee same-pod
6.975/7.196/7.232/7.353, 32K firming) has **no committed primary source** — the W18 line-files contain
zero `ppl=` rows and the numbers are hand-assembled in `results/w18-g1-report.md` (commit `b157acd`, no
generator) and retyped as constants in `scripts/w19_a1_report.py:29-33` / `scripts/w19_dashboard.py:52-56`
despite the docstring "never retyped" (`w19_a1_report.py:6`); (3) a new finding: the perplexity scorer
rounds each window's summed NLL to **bf16** (144/144 per-window sums across all W18+W19 raws sit exactly on
the bf16 grid), giving a ±0.006–0.011 bits/token resolution that some stated gaps fall below; (4) the
**CI workflow is red at HEAD** (ruff E501, `scripts/w19_figures.py:138`) on the last six week7 pushes, so the
CPU test suite is unverified at the reviewed commit (the `paper` workflow is green).

---

## 1. Number traceability — spot-check (23 numbers)

Legend: ✓ traced & agrees · ~ traced but disagrees/imprecise · ✗ no committed primary source.

| # | `paper/main.tex` | Claim | Committed source (file:line) | Status |
|---|---|---|---|---|
| 1 | :76 (abstract) | matched bytes 0.15 vs 0.16× | flagship `results/w18-llama-lines.txt` (`sbits=0.151`, niah rows); 2-bit `results/w19-a1-llama-lines.txt:5` (`sbits=0.163`) | ✓ |
| 2 | :77 | mv wins, McNemar p≤0.03 on 3 families | `results/w19-a1-report.md:30,78,127` (0.0156/0.0312/0.0078) ← `results/w19_intervals/a1-*-ruler-intervals.json` `contrasts` | ✓ |
| 3 | :84 | compose 0.048× (16K) / 0.034× (32K) | `results/w19-a1q-llama-lines.txt:1-2` (`sbits=0.048`, `0.034`) | ✓ |
| 4 | :89-90 | 9–10× faster cold start, 0.13 s vs 1.2 s | `results/w19-a3-llama2-lines.txt:1-2` (cold 0.134 s / 1.226 s → 9.1×; :4-5 → 10.4×) | ✓ |
| 5 | :92-93 | 0.151→0.133× 16K→64K; full four-task at 64K | `w18-llama-lines.txt` (0.151); `results/w19-a4-llama-lines.txt:1,6,10,14,18` (`sbits=0.133`, acc 1.00 n=8 ×4) | ✓ |
| 6 | :696 Tab.fairquant | Llama 16K KIVI-2 0.163× 1.00/0.67/0.42/0.67 | `w19-a1-llama-lines.txt:15,5,10,20` | ✓ |
| 7 | :702 | Mistral 16K flagship stored **0.151×** | `results/w18-mistral-lines.txt` prints `sbits=0.150` (single, mk) and `0.151` (mv, vt); the generator's own table says `0.150x` (`results/w19-a1-report.md:59`) | ~ (3rd decimal) |
| 8 | :706 | Mistral 32K KIVI-2 0.83/0.25/0.08/0.00 | `w19-a1-mistral-lines.txt:18,8,13,23` | ✓ |
| 9 | :712 | Qwen 32K flagship 0.265×, mv/vt bold | `results/w18-qwen-lines.txt` (`sbits=0.265`, acc 1.00×4); `w19-a1-report.md:131-132` (p=0.0020/0.0039) | ✓ |
| 10 | :669-671 | fluency 5.31/5.40/4.90 (8.33/8.28/7.54); 5.50/4.99/4.83 (3.76/3.46/3.31); 8.18/6.70/6.20 (35.1/8.23/7.71) | quant values: `w19-a1-{llama,mistral,qwen}-lines.txt:1-4` ✓. **Flagship values: no `ppl=` row in any tracked W18 file** (`grep -c ppl= results/w18-llama-lines.txt` = 0); only `results/w18-g1-report.md:9,14,16` (hand-assembled) and gitignored `results/w18_harvest/{llama,mistral,qwen}.raw` (5.308/8.326/5.499/3.762/8.181/35.083); retyped in `scripts/w19_a1_report.py:29-33`, `scripts/w19_dashboard.py:52-56` | ✗ (flagship half) |
| 11 | :788-794 Tab.official | all 7 rows × 9 tasks + means | `results/w19-a2-llama-lines.txt:1-63`; means recomputed 0.99/0.98/0.92/0.99/0.87/0.79/0.20 | ✓ |
| 12 | :757-758 | eviction mean 0.20, flagship separated on 6/9, p≤0.008 | `results/w19_intervals/a2-llama-ruler-intervals.md:40,43,44,45,47,48` (mv .0020, mk2 .0005, mk3 .0020, mq .0078, s2 .0020, s3 .0020) | ✓ |
| 13 | :763 | mv/mq discordant 1 vs 1, 0 vs 4 | `a2-llama-ruler-intervals.md:22,27` | ✓ |
| 14 | :767 | vt 0.33 vs 1.00 (4-bit, Palu), p=0.008 | `a2-llama-ruler-intervals.md:32,50` (0/8, p=0.0078) | ✓ |
| 15 | :767-768 | "Its **14** misses sit at depths 0.15–0.95" | `results/w19-a2-flagship-misses.md:38` lists 14 *depth-bearing* misses, but the table `:7-28` has **22** missed records (14 needle + 8 `vt` with depth n/a); the official accuracies (2+0+2+2+0+2+2+4+8) also give 22. `scripts/w19_a2_misses.py:64-66` counts only `depth is not None` | ~ (should read "22 misses; the 14 needle-task misses sit at depths 0.15–0.95") |
| 16 | :581-583 | compose 16K 1/1/1/0.50 (Llama), 1/1/0.92/0.33 (Mistral); 2-bit 0.67/0.42, 0.58/0.50 | `w19-a1q-llama-lines.txt:7,3,5,9`; `w19-a1q-mistral-lines.txt:7,3,5,9`; `w19-a1-llama-lines.txt:5,10`; `w19-a1-mistral-lines.txt:5,10` | ✓ |
| 17 | :587-589 | compose 32K 1/1/1/0.83, 1/1/0.83/0.58; 2-bit 1.00/0.83/0.92/0.92, 0.83/0.25/0.08/0.00 | `a1q-llama:8,4,6,10`; `a1q-mistral:8,4,6,10`; `a1-llama:18,8,13,23`; `a1-mistral:18,8,13,23` | ✓ |
| 18 | :594-596 | ppl 9.25/17.1 vs 5.31/8.33; 6.56/4.28 vs 5.50/3.76 | `a1q-llama-lines.txt:1-2` (9.254/17.074) ✓; `a1q-mistral-lines.txt:1-2` (6.558/4.280) ✓; unquantized → same gap as #10 | ✓ / ✗ |
| 19 | :838-843 | 64K: flagship 1.00×4 n=8 @0.133; 2-bit 1.00/0.58/0.50/1.00; ea 1.00/0.67/0.83/1.00; ppl 8.59/8.15/8.15/7.27/7.26 | `w19-a4-llama-lines.txt:1-21` | ✓ (the ea = 2-bit = 8.149 tie is a bf16-grid collision, §6) |
| 20 | :890-899 | cold start 0.13/0.23 s (325/600 MB) vs 1.23/2.35 s (2.15/4.29 GB); reconstruct 0.04–0.08 s; 2-bit 0.14/0.21 s at 336/671 MB; 0.151→0.140 vs 0.156 | `w19-a3-llama2-lines.txt:1-6` (ratio 0.1514/0.1397/0.1563, ready 0.039/0.082) | ✓ |
| 21 | :466-471 Tab.marquee | 0.31 [0.14,0.56], 10/0 disc., p=2.0e-3; palu 0.56, 6/0, p=3.1e-2 | `results/w18-g4-marquee-contrasts.json` (flag 15/16, think 5/16, 10/0 p=0.002; palu 9/16, 6/0 p=0.0312) + `results/w18_pertrial/g4-llama-trials.txt` | ✓ |
| 22 | :511-512 | same-pod ppl 6.975/7.196/7.232/7.353 | source comment `:519` points at "results/w18-g4-llama"; `results/w18-g4-llama-lines.txt` has **0** `ppl=` rows; rows exist only in gitignored `results/w18_harvest/g4-llama.raw` | ✗ |
| 23 | :433-435 | 32K firming ppl 8.2→35.1, 3.76, 8.33 | `results/w18-g1-report.md` only (as #10) | ✗ |

Also: the per-token-key zero replication cited at `:640-645` comes from the `a1diag` pod
(`results/w19-a1diag-qwen-lines.txt:4-5`, committed ✓) which ran at SHA `24ac22a` — not among the five
SHAs listed in the provenance paragraph `:979-983`.

Internal inconsistency in a source doc: `docs/week19-official-ruler.md:57` (table) says flagship mean 0.79
while `:105` (Reading) says 0.80 (mean-of-rounded vs pooled 86/108). The paper uses 0.79.

Typo that will print: `paper/main.tex:585` ends with an orphan "Three" before "At 32K the cell falls to"
(the sentence "Three qualifications travel with it" is repeated at `:589-590`).

**Bottom line for (1):** every Week-19 number is traceable and agrees (two third-decimal/wording nits); the
Week-18 *perplexity* layer (≈14 numbers across §quantbaseline, §subcliff, §marquee, 32K firming) is not
traceable to any committed primary artifact. This is a 10-minute fix because the raws are still on disk.

---

## 2. Evidentiary chain for Week-19

**SHA per pod.** Intended SHAs: `paper/main.tex:979-983`, `docs/week19-handover.md:85-88`. Mechanism:
`scripts/pod/w18_boot.sh:34,48-50` (`git checkout -q "$SHA"`, `===RUN_SHA_…===`, `run_sha=` in the ENV
block). Verified (from the local, gitignored raws): a1-{llama,mistral,qwen} `6734afa`; a1q-{llama,mistral,
qwen} `1cbc31f`; a3-llama2 `5e8b275`; a2-llama `c331ebd`; a4-llama `d15769d`; a1diag `24ac22a` — all five
paper SHAs match their pods and all six commits exist and are ancestors of HEAD. **But**: the committed
harvest state (`results/w19_harvest/pods.txt`, `done.txt`) carries `label:id:mode:tag` only
(`scripts/pod/w19_watchdog.sh:3`) — no SHA; the extract step (`w19_watchdog.sh:24-25`) keeps only
`acc=`/`ppl=`/`bytes=` and `[trial]` rows; the ENV lines are dropped, and `.raw` is gitignored
(`.gitignore:52`). W18 solved exactly this with `results/w18-env-provenance.txt` (tracked); W19 has none.

**Fail-loud is not loud.** `w18_boot.sh:48`: `git checkout -q "$SHA" 2>&1 | tail -2 || { …; exit 1; }` —
without `pipefail` the pipeline's status is `tail`'s (0), so a bad SHA silently runs the clone's default
HEAD; only the RUN_SHA line (gitignored) would reveal it, and the watchdog's ALL_DONE match
(`w19_watchdog.sh:35`) does not check the SHA. No damage occurred in this program (all RUN_SHAs match), but
the comment at `:47` ("FAIL LOUD") is false. Same masking on `pip install … | tail -5` (`:52-53`), mitigated
by the `QUANTO_OK`/`HQQ_OK` probes (`:56-57`).

**Env header.** `w18_boot.sh:61-78` prints SHA, `nvidia-smi` (name, memory, driver), python, torch+CUDA,
transformers, kvpress, device. Captured in every raw (all 11 pods: `NVIDIA A100-PCIE-40GB`, `torch=2.11.0+cu128`,
`transformers=5.8.0 kvpress=0.5.1`; drivers 595.71.05 / 570.211.01 / 570.86.10 / 535.247.01 / 580.173.02).
Not printed: `optimum-quanto` and `hqq` versions — and quanto, the kernel under the panel's #1 experiment,
is installed as a floating lower bound `"optimum-quanto>=0.2.7"` (`w18_boot.sh:53`, `pyproject.toml:20`;
`uv.lock:5272` locks 0.2.7 for CPU only). No committed or local record says which quanto ran on the pods.
`hqq==0.2.8.post1` is pinned (`w18_boot.sh:53`).

**Per-trial records** (`results/w19_pertrial/`, tracked): counts are complete —
a1-{llama,mistral,qwen} 208 = 2 arms×4 tasks×2 ctx×12 + hqq8 4×4; a1q-llama/mistral 96 = 4×2×12; a1q-qwen 48
(16K only; pod destroyed early, `docs/week19-handover.md:86`); a2-llama 756 = 9 tasks×7 arms×12;
a4-llama 176 = 4×8 (flagship) + 3×4×12; a1diag 24. The persistence MODE emits no `[trial]` rows, so
`a3-llama-trials.txt`/`a3-llama2-trials.txt` are **empty tracked files**, as is `results/w19-a3-llama-lines.txt`
(commit `69f7ea9` emptied it instead of deleting).

**Cross-program pairing.** The W19 McNemar contrasts pair W19 quant trials with **W18** flagship trials on
`(seed, trial)` (`scripts/w18_intervals.py:64-92`); the W18 records exist (`results/w18_pertrial/llama-trials.txt`,
12 per task×ctx for `bugSseed-r64-h256`). Validity requires an identical needle/haystack layout across
`15678a7`/`b157acd` (W18) and `6734afa` (W19): the seed formula `w10_ruler.py:192` is unchanged and the
Week-19 memoization (`673331f`) is pinned bit-identical to the archived algorithm by
`tests/test_w10_ruler_filler.py:42-46,57-64`. Adequate. (The `hits=round(acc*n)` reconstruction at
`w18_intervals.py:53` persists; harmless at n≤16.)

**Harvest state.** Tracked: `pods.txt`, `done.txt`. Untracked: `watchdog.log` (the only record of cost and
wall-clock), `status.txt`, `watchdog.pid`, and `a3-llama.raw.superseded` — which holds *two* earlier a3 runs
(`RUN_SHA_1cbc31f` OOM run and `RUN_SHA_bafa026` 2.6×-inflated run) that are the evidence behind fix
commits `bafa026`/`5e8b275` and the paper's "persisted state = exactly the billed tensors" (`main.tex:888`).

**Official-RULER data.** Generator SHA pinned in-repo (`scripts/pod/w19.sh:90`) with an effective checkout
guard (`:99`, no pipe). Essays are a **live download** (`:101`) with a non-fatal check (`:102`). The pod
echoes `===RULER_SHA_…===` and per-task record counts (`:100,107`) but the watchdog regex
(`w19_watchdog.sh:19`) excludes them — verified: 0 such markers in `a2-llama.raw`. No sha256 of any
`validation.jsonl` or of `PaulGrahamEssays.json` exists anywhere. `results/w19-a2-flagship-misses.md` relies
on an uncommitted local regeneration asserted deterministic (`scripts/w19_a2_misses.py:4-7`); the fact that all
14 needle misses resolved to a depth is circumstantial evidence the regeneration matched.

---

## 3. Can a third party rerun a cell from the repo alone?

**Present:** exact flags per MODE in a SHA-pinned driver (`scripts/pod/w19.sh:22-28,51-59,67-84,110-115,
124-127,136-144`); pinned harness deps (`w18_boot.sh:52-53`); seeds `--seeds 0 1` / `--n-trials 6` (n=12) and
`--n-trials 4` (64K flagship, n=8) in the script; official RULER `--random_seed 42 --num_samples 12`
(`w19.sh:104-106`); model ids (Qwen default `w18_boot.sh:30`; Llama/Mistral ids in `README.md:142`,
`docs/week18-kickoff.md:96`); `DTYPE=bfloat16`, `CHUNK=4096` defaults (`w18_boot.sh:32-33`); licence table for
the `unsloth/` Llama mirror (`README.md:216-236`) — prior gap closed; the image name only in
`docs/week19-kickoff.md:125-127` (`-devel` needed for quanto).

**Missing / weak:**
- No one-command, non-vast entry for a headline cell: `README.md:136-150` still reproduces `r32-h256`,
  n=2×2 — not the flagship, the fair-quant arm, the compose cell, or the official anchor; the boot script
  hardcodes `cd /root` and a GitHub clone (`w18_boot.sh:39-42`), and the actual onstart was a sed-baked
  scratch copy (`docs/week19-handover.md:68`), so the committed file is a template, not the launched artifact.
- The paper's Setup (`main.tex:357-376`) never states GPU (A100-40GB appears only at `:887`), model dtype
  (bf16), chunk size (4096), or the quanto/hqq versions.
- quanto version unpinned/unrecorded (above).
- Official data: no hashes/snapshot; the essay corpus can drift under a live download.
- No `torch.use_deterministic_algorithms`/cuBLAS workspace pinning (carried from the prior review).
- Raw `--out-json` files are still not preserved: the base64 `emit()` folds (`w18_boot.sh:83-88`) "arrive
  truncated" (`w19_watchdog.sh:10`), so rows remain the record.

---

## 4. Are the tests meaningful?

- **Quant scheme bit-identical:** `tests/test_w19_quant_kivi.py:80-91` — `torch.equal` of dequantized keys
  vs upstream `QuantizedCache(axis_key=0, axis_value=0, q_group_size=64)`. Meaningful; checks keys only
  (values not asserted) — nit. Per-channel isolation `:63-77` (outlier-channel error ratio), residual flush
  `:109-124`, aux billing `:127-137`, arm naming/`chunkable` `:174-183`, chunked retrieve `:186-204`: all
  behavioural pins of the things that went wrong in W18.
- **Compose budget fix (`1cbc31f`):** `tests/test_w10_ruler_quant.py:158-178` builds `bugSseed-r16-h16-q4`
  with `bug_quant_budget=32` at ctx 160 and asserts `_q_len() > 0` (the W18 failure was an empty tier) and a
  billed `ratio_stored_bits`; `tests/test_accounting.py` (`test_bug_quant_footprint_matches_stored_state_numel`)
  pins billing of the quant tier against `stored_state_numel`. Not pinned: the exact semantics "budget = fp32
  columns kept" (`_f_len() <= budget`) and "never dropped" (`_f_len() + _q_len()` == ingested middle).
  Fixable in two asserts.
- **no_grad regressions (`6734afa`, `bafa026`):** decorators present (`scripts/w10_frontier.py:117,138,160`).
  `tests/test_w19_quant_kivi.py:207-221` asserts the dequantized keys carry no grad — I verified empirically
  that with grad enabled during prefill `dequantized.requires_grad=True`, so this proxy *does* fail without
  the decorator: meaningful. `tests/test_w19_persist.py:107-123` spies on `model.forward` and asserts grad is
  disabled on every call: strong. `:126-140` pins the 2.6× `torch.save` view inflation by content size;
  `:91-104` pins persisted tensors == billed numel (backs `main.tex:888`).
- **Official RULER path:** `tests/test_w19_official_ruler.py:36-50` (split, fail-loud), `:83-91` (template
  slice tripwire: needle never in the decoded tail), `:112-151` (end-to-end rows + `[trial]` lines in the
  `w18_intervals` format), `:154-159` (CLI composition regression). Good.
- **CI:** `ci.yml:3-6` now triggers on `week7` (prior gap closed) — but `gh run list --branch week7` shows the
  `CI` workflow **failing on the last six pushes** (e.g. run 34034137816, 32 s) at `ruff check`:
  `E501 scripts/w19_figures.py:138 (108 > 100)`. mypy and pytest never ran; the `paper` workflow is green.
  The reviewed commit is therefore not "green-gated" for tests. `README.md:133` says 338 passed; the tree has
  347 `def test_` (harmless drift).

---

## 5. Compute disclosure

The paper admits it has none (`main.tex:986-987`). What exists: `docs/week19-handover.md:5` (≈$39, A100
40GB PCIe) and the untracked `results/w19_harvest/watchdog.log` (credit $94.36 → $55.67 = **$38.70** for 11
pods; wall-clock from its ALL_DONE lines ≈ **42 A100-40GB pod-hours**: a1 ≈2.8/3.6/3.9 h, a1q ≈9/6.7/3.5 h,
a2 ≈5 h, a4 ≈6.4 h, a1diag 0.5 h, a3 ×3 short). W18 hours/cost are not in the paper either. Everything needed
for a compute appendix is on disk; nothing is in the manuscript or a tracked file.

---

## 6. New finding — the perplexity scorer is bf16-rounded

`scripts/w10_frontier.py:58-70` (`_score_window`): `logits = out.logits[0]` is bf16 under `DTYPE=bfloat16`
(`w18_boot.sh:33`; transformers 5.8 `modeling_llama.py:487` does not upcast), and
`cross_entropy(…, reduction="sum")` on bf16 input returns a **bf16** sum before `float(nll)`. Evidence:
every `[pplw]` per-window mean × 511 tokens in **all** W18+W19 raws (144/144 values) is an exact integer
multiple of the bf16 ulp at that magnitude (4 nats in [512,1024), 8 in [1024,2048)); a float32 sum would land
there ~4% of the time. E.g. a4 64K: `ea-k0.1` windows 1208/1168/800/1112 and `quant-2bit-kivi`
1216/1176/768/1128 both total 4288 → the "8.149 = 8.149" tie at `main.tex:841` is a grid collision.

Consequences: per-window resolution ±2–4 nats ≈ ±0.004–0.008 nats/tok ≈ **±0.006–0.011 bits/tok**. Perplexities
printed to 4 s.f. (`:511-512`, `:669-671`) carry spurious precision; the statement "at 16K the gap to the
baselines is ≤0.006 bits/token" (`:514-515`) is below the scorer's resolution; the pre-registered 0.05-bits/tok
band and all differences ≳1% (marquee +0.024/+0.031, fair-quant 5.31 vs 5.40, sub-cliff 9.25 vs 5.31) are
unaffected. Fix: `.float()` on the logits (one line) for future runs; for the manuscript, round ppl to two
decimals and state the resolution. Not fatal — no conclusion rests on a sub-1% ppl difference.

---

## 7. Prior-panel and prior-repro items — status

| Item | Status | Evidence |
|---|---|---|
| 2-bit quant baseline (fatal #1) | closed | `src/kvdlra/quant/kivi_cache.py`, `w19.sh:48-60`, `results/w19-a1-*`, paired McNemar |
| storage-vs-resident (fatal #2) | closed | `main.tex:819-900`, measured cold start `w19-a3-llama2-lines.txt` |
| custom benchmark (fatal #4) | closed (one model/ctx; data hashes missing) | `scripts/w19_official_ruler.py`, `w19.sh:86-116`, `w19-a2-llama-lines.txt` |
| manuscript (fatal #5) | closed | `.github/workflows/latex.yml`, green |
| repro #1 one-command headline repro | **open** | `README.md:136-150` still r32 |
| repro #2 env capture + clone by SHA | half | captured in raws; SHA-pinned; **not committed** for W19 |
| repro #3 preserve per-trial/raw | per-trial yes; raw JSON no | `results/w19_pertrial/`; `w19_watchdog.sh:10` |
| repro #5 seeds/compute table, licences | licences yes (`README.md:216-236`); compute **open** | `main.tex:986-987` |
| repro #6 hygiene / CI on week7 | CI triggers yes; **CI red at HEAD**; new empty tracked files + untracked superseded raw | above |
| repro #7 `round(acc*n)` | unchanged (harmless at n≤16) | `w18_intervals.py:53` |

---

## 8. Severity triage

**FATAL: none.** Nothing found invalidates a number or a conclusion; the Week-19 retrieval chain
(pod stdout → line-file → per-trial → intervals/McNemar → report/table/figure) is complete, committed and
regenerable.

**Fixable (should be done before submission):**
1. Commit W19 environment provenance (`results/w19-env-provenance.txt` from the local raws, mirroring
   `results/w18-env-provenance.txt`); add `a1diag @24ac22a` and quanto/hqq versions to `main.tex:979-984`;
   cite `results/w19_pertrial/` in Setup (`:372-374`).
2. Commit the W18 `ppl=` rows (extract from `results/w18_harvest/{llama,mistral,qwen,g4-llama}.raw` into
   `results/w18-*-ppl-lines.txt`) and make `w19_a1_report.py:29-33` / `w19_dashboard.py:52-56` read them;
   fix the `:519` source comment.
3. Scorer precision: `.float()` the logits in `_score_window`; round ppl to 2 decimals; delete or re-measure
   the "≤0.006 bits/token" sentence (`:514-515`); note the 64K 8.149 tie is at scorer resolution.
4. Make CI green at the submitted SHA (`scripts/w19_figures.py:138`); fix `w18_boot.sh:48` to
   `git checkout -q "$SHA" || exit 1` (no pipe); pin `optimum-quanto==<pod version>` and print it in the ENV block.
5. Official data: harvest `===RULER_SHA_`/`===A2_PREP_` markers (add to `w19_watchdog.sh:19`), record sha256 of
   each `validation.jsonl` and the essays JSON in the a2 MODE, and commit the 108 records' `index/length/
   token_position_answer` (tiny) so `w19_a2_misses.py` runs from the repo.
6. Compute appendix (pods, SHA, GPU, driver, library versions, hours, $) from the raws + `watchdog.log`; W18 too.
7. Wording: "14 misses" → "22 misses (14 needle-task misses at depths 0.15–0.95; 8 variable-tracking)"
   (`:767-768`); Mistral 16K stored 0.151→0.150 (`:702`); orphan "Three" (`:585`); README test count.
8. One-command headline repro (`repro/fairquant.sh` etc.) that runs a MODE body outside vast.ai.

**Nitpicks:** empty tracked files (`results/w19-a3-llama-lines.txt`, `results/w19_pertrial/a3-llama*-trials.txt`);
commit `a3-llama.raw.superseded` (evidence for the 2.6× fix); `docs/week19-official-ruler.md:57` vs `:105`
(0.79 vs 0.80); values not asserted in the bit-identity test; `ea-k0.1` 64K ppl row bills `ratio=0.108`
while its retrieval rows bill 0.100 (`w19-a4-llama-lines.txt:2,7`).

---

## 9. Highest-leverage fixes with cost

1. **Provenance primaries** (CPU-only, ~1 h): W19 env-provenance file + W18 ppl line-files + provenance
   paragraph additions. Turns every ✗ in §1 into ✓ without a GPU.
2. **Scorer precision** (CPU-only, ~1 h; optional ~$5 GPU to re-score the 16K windows in fp32): one-line
   `.float()`, 2-decimal ppl, drop the sub-resolution sentence.
3. **Green CI + fail-loud boot + quanto pin + official-data hashes** (CPU-only, ~2 h): the four small edits
   in §8 items 4–5. Compute appendix is another hour from the same on-disk logs.
