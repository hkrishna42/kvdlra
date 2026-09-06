# Week-19 handover — the fair baseline, the anchor, and where the claim narrowed (2026-09-05/06)

Branch `week7`. Everything below is committed and pushed; the paper builds on CI
(GitHub Actions workflow `paper`, 18 pages, all refs/cites resolve). Credit at the end of the
program: **$55.66** (the whole Week-19 GPU program cost ≈ $39 on 40GB A100 PCIe pods).

## What was done, in the kickoff's order

**A1 — the fair 2-bit baseline (the crux).** The Week-18 quant zero was a configuration
hazard, not a property of quantization: optimum-quanto's `axis=0` groups per *token* and
`axis=-1` per *channel*, so transformers' `QuantizedCache` default quantized **keys per
token**, which one Qwen2.5 key-bias channel defeats (synthetic: 4-bit per-token rel err 0.25
vs 0.02 per-channel). `kvdlra.quant.kivi_cache` builds the faithful arm (`--quant-scheme
kivi`: per-channel keys with the token axis edge-padded to the group, per-token values;
`--quant-backend hqq` adds 1–8 bit incl. the 8-bit control); the default `token` scheme is
test-pinned bit-identical to Week-18. Grid: 3 families × 16K/32K × 4 tasks × n=12, paired
needle-for-needle with the Week-18 flagship records (`results/w19-a1-report.md`,
`results/w19_intervals/a1-*`). Outcome at matched stored bytes (0.15× vs 0.16×): single-needle
ties; the flagship wins multi-value at 16K on all three families (McNemar p=.016/.031/.008),
multi-key+multi-value at 32K on Mistral (p=.004), multi-value+var-track at 32K on Qwen
(p=.002/.004); Llama 32K not separated; **4-bit at ~2× bytes concedes nothing and dominates on
Qwen at matched honest bytes (0.275× = 0.287×)**; fluency goes to the quantizer or ties.

**A2 — the official NVIDIA RULER anchor** (`scripts/w19_official_ruler.py`, RULER commit
`c3f5e3b` generated on-pod, Llama 16K, 9 tasks × 12 records): flagship mean **0.79 vs 0.87**
for 2-bit (no task separated either way); the in-repo multi-value edge does **not** transfer
(1 vs 1 discordant); 4-bit/ThinK/Palu beat the flagship on var-track (p=.008); eviction at
0.1× averages 0.20 (flagship separated on 6/9). Misses sit at depths 0.15–0.95 (not a warm-up
effect; `results/w19-a2-flagship-misses.md`). Doc: `docs/week19-official-ruler.md`.

**A3 — persistence cold start** (`scripts/w19_persist.py`, Llama, A100-40GB): 16K full 2.15 GB
1.23 s; flagship 325 MB **0.13 s**; KIVI-2bit 336 MB 0.14 s. 32K: 4.29 GB 2.35 s / 600 MB
0.23 s / 671 MB 0.21 s. The 9–10× win over full KV is **shared with 2-bit quantization**;
BUG's separator is the 1/T term.

**A4 — the 64K point** (Llama, 40GB card): flagship **1.00/1.00/1.00/1.00 (n=8) at 0.133×**
(0.151→0.140→0.133 from 16K→64K; 2-bit flat at 0.156–0.158×), where 2-bit scores
1.00/0.58/0.50/1.00 and eviction 1.00/0.67/0.83/1.00; ppl 8.59 vs 8.15 (2-bit, eviction), 7.27
(4-bit), 7.26 (full).

**The real sub-cliff compose** (`bugSseed-r64-h256-q4`, 512 fp32 coords kept, rest 4-bit
PolarQuant): **0.048× (16K) / 0.034× (32K)** — Llama 1.00/1.00/1.00/0.50 and 1.00/1.00/1.00/0.83;
Mistral 1.00/1.00/0.92/0.33 and 1.00/1.00/0.83/0.58; **Qwen 0/0/0/0 with diverged ppl**. Fluency
cost: Llama ppl 9.25/17.1 vs 5.31/8.33 unquantized; Mistral 6.56/4.28 vs 5.50/3.76.

## Bugs found and fixed (all test-pinned)
1. Per-token key quantization = the Week-18 zero (above). `--quant-axis-*` flags removed.
2. **The Week-18 "q4 compose" arm never quantized**: `build_arms` passed the fp32 coordinate
   budget as the whole context and the quant tier as 512 (the flag's documented semantics
   inverted), so its rows bill byte-identically to the unseeded flagship. Fixed
   (`--bug-quant-budget` = fp32 columns kept, the rest quantized, never dropped); the seed is
   allowed with the quant tier (same graduation path). `results/w18-g1-report.md` annotated
   INVALID; the paper never cited it.
3. `score_quant` / `run_persist` / `attend_ready` ran with autograd enabled → the Week-18
   quant-ppl OOMs and the first a3 pod OOM (38 GB allocated mid-chunk). `@torch.no_grad()`,
   spy-tested.
4. `torch.save` wrote whole view storages (ring buffers, core diagonals) → persisted bytes
   inflated 2.6×; tensors are cloned on save (content-size test).
5. `w10_ruler._filler_to` re-tokenized the haystack per sentence (O(n²); ~5 min/trial at 64K);
   memoized per (tokenizer, ctx, filler, pool, seed, trial), bit-identical.
6. Watchdog: `DONE.txt` and `done.txt` are the same file on case-insensitive APFS (the
   completion marker clobbered the done-list) → `status.txt`; the label variable was clobbered
   by `extract()`'s loop → locals.
7. `w18_intervals` contrast loop only knew the four in-repo task names → all task names.

## Infra that worked (reuse verbatim)
- `scripts/pod/w19.sh` MODEs `a1diag|a1|a1q|a2|a3|a4` sourced by `w18_boot.sh` via
  `DRIVER=scripts/pod/w19.sh`; onstart = a scratch copy with MODE/MODEL/TAG/SHA/DRIVER sed-baked.
- `scripts/pod/w19_watchdog.sh`: file-driven (`results/w19_harvest/pods.txt` label:id:mode:tag,
  `done.txt`), restart-safe, detached via python double-fork + `caffeinate -is -w <pid>`;
  harvests rows every 150 s, destroys on `===ALL_DONE_<mode>_<tag>`, extracts line-files +
  per-trial files, commits, pushes; credit floor $6. It survived a Mac reboot (restart it; the
  state is on disk; /tmp is wiped at boot so keep nothing there that matters).
- Validate-first worked twice (a1diag → a1 fan-out; a1q-llama → qwen/mistral). 40GB cards were
  enough for everything including Llama 64K once the no_grad bugs were fixed (no 80GB offers
  existed for 12 h).
- Paper CI: `.github/workflows/latex.yml` (xu-cheng/latex-action, latexmk+bibtex, fails on
  undefined refs, uploads main.pdf + main.log); read the log with `gh run view <id> --log`.
  BibTeX rejects `%` comments inside entries (two were fixed).
- Figures: `scripts/w19_figures.py` (fairquant small multiples, one_over_t, coldstart) from
  line-files; board: `scripts/w19_dashboard.py` → `docs/board/week19-board.html`, published to
  artifact 757d6777 (pass `url:` to update); report: `scripts/w19_a1_report.py`.

## Pod provenance
a1diag-qwen 49995708 @24ac22a · a1 {qwen 49998517, mistral 49998519, llama 49998520} @6734afa ·
a1q {llama 49999533, qwen 50003064 (destroyed early after its 16K zeros), mistral 50003069}
@1cbc31f · a3-llama2 50001503 @5e8b275 (a3-llama 50000740 superseded: view-inflated) ·
a2-llama 50003417 @c331ebd · a4-llama 50016757 @d15769d. Per-trial records
`results/w19_pertrial/`, line-files `results/w19-*-lines.txt`, intervals `results/w19_intervals/`.

## Paper state (`paper/main.tex`, ~1016 lines)
Rewritten around the outcome: abstract (fair baseline, official anchor, compose cell, 1/T to
64K, persistence shared with quant), intro bullets + scope, §quantbaseline (Table
`tab:fairquant` + McNemar), §subcliff (compose cell, Qwen negative, fluency cost, 32K),
§realistic (3) official anchor + Table `tab:official`, §memory (measured cold start + 64K),
limitations (fair-quant competitor paragraph, external validity cuts both ways, provenance
with W19 SHAs), conclusion; three figures (`fig:fairquant`, `fig:one_over_t`, `fig:coldstart`).

## Open items
- Exit-gate panel re-run (6 dimensions + AC, briefing `docs/reviews/2026-09-06/review-briefing.md`):
  reviews land in the session scratchpad; verdict to be appended below and to the memory.
- arXiv v1 submission: the user's call (the manuscript is submittable; MomentKV/ResKV clock).
- Not done: a fp16-storable gist (would halve BUG's honest bytes — the single most valuable
  follow-up given the Qwen matched-bytes result); official RULER at 32K / other families;
  flagship LongBench sweep; a fused kernel; the seeded compose on Qwen (negative as measured).
