# Week-18 Phase 0 + Phase 1 handover ($0/CPU complete)

**Branch** `week7` (pushed, CI green) · **Supersedes** the Phase-0 half of `docs/week18-kickoff.md`.
Executes the AC panel's $0 items + the harness engineering the September GPU program depends on.
The manuscript program's honesty bar is enforced throughout (see `docs/reviews/2026-09-01/`).

## What shipped (12 commits, `bdb4439..f4ee938`, all green-gated + pushed)

### Phase 0 — credibility pass
| Commit | What |
|---|---|
| `be5adc9` | Wording pass (5–13× → 3.4–10× stored state; "beats"→"leads" except the Wilson-separated marquee; named `think-c0.5`/`palu-r0.5`; "cannot enter"→measured collapse points; README Palu vt 0.25→0.56; "safe rank" defined). Panel reviews preserved to `docs/reviews/2026-09-01/`. |
| `7a7734c` | **Dual memory billing** — `Footprint.ratio_stored_bits()` bills BUG's fp32-at-rest U/C at 32 bits alongside the unchanged `ratio_fp16`. r64 = 0.085×/0.150× (Llama), 0.148×/0.275× (Qwen). Baselines: stored==fp16. Archived intervals byte-identical. |
| `1b925c2` | **Evidentiary chain** — `scripts/pod/w18_boot.sh` (SHA-pinned clone + env header + emit() fold + tokenizer check) → sources `w18.sh`; `[trial]` per-trial lines + `sbits=`/`n=` on the aggregate row (append-only, archived regexes intact). |
| `712458d` | Hygiene — CI push trigger on `week7` (first-ever push CI, now green); model-license table; committed `results/w15b-complete-lines.txt`; `--out-fig` → gitignored scratch (stale-figure root cause); README test count 340; removed the stale 1.3 GB worktree. |

### Phase 1 — harness engineering + manuscript (8 workstreams)
| Commit | WS | What |
|---|---|---|
| `238e106` | **W1** | 2/4-bit KV baseline (`quant-2bit`/`quant-4bit`, transformers `QuantizedCache`/quanto). `optimum-quanto` dep. `_footprint` quant branch (before the DynamicCache assert), retrieve/ppl cache-supply branches. `accounting.quant_footprint` matched to what quanto stores (probe-verified) → 0.1875×/0.3125× asymptote. The panel's #1 blocking gap. |
| `aa6f975` | **W2** | Fixed the latent `bug_footprint` quant mis-billing (coded columns billed as fp32); wired the `bugS-q4` compose arm (no seed); fenced seed+quant with a fail-loud guard (needs GPU validation). |
| `44f0f1a` | **W3** | `--filler {cycle,wikitext,pg19}` (cycle = bit-identical archived) + `--depths` needle grid. `perplexity_sweep.load_corpus_sentences`. |
| `2ee3165` | **W7+W4** | `scripts/w18_intervals.py` — marker-free (reads `n=` off the line), parses **multikey** (the w17 regex dropped it = W4), exact McNemar on `[trial]` lines for pre-registered contrasts. |
| `ba533f5` | **W5** | `w16_storage` peak-GPU for all arms + `coldload_ratio` (= honest stored-bits); `_smoke_args` rebuilt on the extracted `build_parser()` (drift root-cause fix). |
| `c5cd1bd` | **W6** | Threaded the flagship bugslash flags into `w10_longbench` (the "flagship never LongBench-tested" gap). Official NVIDIA-RULER = GPU-gated → G2. |
| `a2ca7ef` | **W8** | `paper/main.tex` v1 (from-scratch, ~5000 words) + `refs.bib` (34, WebSearch-verified) + Makefile. Every claim `% source:`-tagged; honesty constraints spot-verified. |
| `f4ee938` | matrix | `scripts/pod/w18.sh` G1–G5 filled + quant flags mirrored into `w10_ruler`. Pre-pod gate passed. |

## Decisions taken this session
- **arXiv v1 before the GPU program** (AC §5): the manuscript is the honest *current* story (no quant
  baseline yet; frontier scoped to measured collapse points) — ready to post to timestamp vs MomentKV/ResKV.
- **Seed+quant NOT relaxed**: `bugS-q4` (no seed) ships now; `bugSseed-q4` needs GPU retrieval validation
  that the warm-up seed initializes correctly with a quant tier (a core-guard change, not done blind).
- **Official RULER deferred** to G2 (heavy NVIDIA integration, GPU-gated); LongBench is the $0 anchor.

## Pre-pod launch checklist (Phase 2 — YOU orchestrate; credit $70.38, needs ~$120 top-up)
1. `SHA=$(git rev-parse origin/week7)` — the pods pin to it (`w18_boot.sh` checks out `$SHA`, fails loud).
2. One A100 (PCIe 40 GB ≈ $0.60/hr today) per model; `--env "-e MODE=g1 -e MODEL=... -e TAG=... -e SHA=$SHA" --onstart scripts/pod/w18_boot.sh`.
3. Stage cheap-first: g1 (quant, blocking) → g2 (filler) → g3 (eviction) → g4 (firming) → g5 (storage).
4. Live-monitor `===W18_*`, `^\[trial`, `^\[niah|^\[vt`, `^\[pplw`, and `Traceback|OOM|illegal memory|SKIP`.
5. Harvest **before destroy**: `vastai logs <id> --tail 20000` → line-files + the `*_RESULT_BEGIN/END`
   base64 folds (`scripts/pod/scrape_w10.sh` pattern). Destroy every pod.
6. Build tables: `scripts/w18_intervals.py results/w18-*-lines.txt --contrast bugSseed-r64-h256,think-c0.5 ...`
7. **Fund bars** (pre-registered): C1 holds iff flagship single+mv Wilson-lo ≥0.7 under realistic filler;
   the exclusive-band claim moves to `bugS-q4` iff it holds retrieval ≤0.05×; quant reported whatever it
   shows (compose-not-compete); every "beats" needs McNemar p<0.05.

## Loose ends → Phase 2/3
- `bugSseed-q4` (seed+quant) — relax the guard + GPU-validate.
- `--state-dtype fp16` probe (decides dual-billing empirically) — G4.
- Official RULER subset runner (NVIDIA repo) — G2.
- 2 manuscript `\todo`s: the setup appendix table (per-family n/seeds/depths) and the cold-load *timing*
  (size ratio is `ratio_stored_bits`; timing needs G5 CUDA).
- Dashboard artifact `757d6777` refresh with the new grids (Phase 3).
