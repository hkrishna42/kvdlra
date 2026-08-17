# Week-17 session handover

**Branch** `week7` · GPU program executed (3 pods, one per family) · **Supersedes** the Week-16
handover/plan. **Dashboard** artifact `757d6777` (updated in place). See also [[kvdlra-week16-plan]].

---

## 1. What Week-17 asked, and the answers

Three questions about `bugSseed-r64-h256` (Week-16's extreme-compression config):

1. **Does its cross-model generality hold with error bars?** → **YES.** single + multi-value = **1.00
   [0.76, 1.0] (12/12)** on Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3 at 16K, at 0.085–0.149× memory
   (5–13× under ThinK/Palu). Wilson-firm (n=12 needed — 8/8 only reaches 0.68). BUG beats both baselines
   on multi-value (Qwen). The generality claim is now defensible.
2. **Can the Mistral/Llama var-track weakness be fixed?** → **NO (refuted).** Raising the exact tier
   (h256→h512/h1024) gives **no** vt lift on the real 7B/8B (Mistral 0.50→0.25, Llama 0.58→0.58); the 1B
   CPU proxy over-predicted. With the CPU-refuted memory-free alternatives (score_rank, stickiness), the
   vt weakness on 1024-dim models has **no working fix** — an honest, documented limitation. (Qwen 1.00.)
3. **Can the safe rank be raised by stabilizing the integrator?** → **YES, decisively — the marquee win.**

## 2. The headline — the `min_sv_frac` integrator floor (FIX 2, funded on Qwen + Mistral)

A default-off relative singular-value floor in `augmented_bug_step` caps the tracked gist rank at the
block's numerical rank, removing the near-null-tail padding that tips ill-conditioned KV streams into a
numerical explosion. Phase-1 showed the pure-gist divergence (WS2) and the h1024 "puzzle" (WS3) are **one
defect, one fix**. GPU-confirmed, `--min-sv-frac 1e-2`:

| ppl @ 16K | floor off | floor on |
|---|---|---|
| Qwen `bug-r256` | 27531.7 | **6.995** |
| Qwen `bugSseed-r128-h1024` | 714.4 | **6.941** |
| Mistral `bug-r128` | 138.3 | **5.574** |
| Mistral `bug-r256` | 37.2 | **5.226** |
| Mistral `bugSseed-r128-h1024` | 60.4 | **5.045** |

The floored high rank is the **best** fluency across ranks (Mistral monotone 6.22→5.57→5.23 at r64→128→256)
→ **safe rank genuinely extended**. Default-off/bit-identical (parity green), retrieval- and
footprint-neutral. (Llama onset >r128 → floor not needed there.)

## 3. The Llama marquee (FB-4 — PARTIAL)

`bugSseed-r128-h1024-s32` @32K, **n=16**: vt **0.94 [0.72,0.99]**, mv **1.00 [0.81,1.0]** @0.16× vs think
vt 0.31 [0.14,0.56] / palu vt 0.56 [0.33,0.77]. **Beats ThinK Wilson-separated; leads Palu but not
separated** (palu firmed up from n=4's 0.25). Honesty correction to the Week-16 "beats Palu AND ThinK":
now "beats ThinK; leads Palu; 3–4.7× less memory."

## 4. Honest caveat

`bugSseed-r64-h256` fluency at **32K is Qwen-specific**: Qwen r64 ppl 8.2→35.1 (16K→32K, retrieval stays
1.00); **Mistral (3.76, 1.14× full) and Llama (8.33, 1.11× full) stay healthy**. r64 ppl is healthy at 16K
on all three (1.08–1.32× full). The floor makes higher rank safe → the natural 32K config is
r128/r256-with-floor.

## 5. Commits (all green-gated, pushed origin/week7)

| commit | what |
|---|---|
| `3eba18f` | `w16_intervals.py` + Week-16 line-files → per-model decision table + Wilson CIs (79 cells) |
| `868841b` | **`min_sv_frac` integrator floor** (streaming_torch + bug_cache threading) + 4 tests; default-off/bit-identical |
| `7f49091` | `w16.sh MODE=w17` — the pre-registered confirm matrix (n=12 core, env-gated fix arms) |
| `3e27089` | w17 leaner ppl (n-samples 4) + trimmed FLOOR |
| `6093ce6` | `w16.sh MODE=w17ppl` — ppl-only recovery re-run |
| `130cf19` | w17 GPU results — `w17_intervals.py` + line-files + decision table + Wilson intervals |
| *(this)* | `docs/week17-*.md` + dashboard `757d6777` + memory |

## 6. Infra state

- **Pods:** qwen `47917824`, mistral `47918031`, llama `47918033` (RULER done; **ppl CUDA-crashed** — a
  transient card fault; the recovery pod `47966074` re-ran ppl only). **All destroyed.** Credit **$70.4** (~$25 for the program).
- Both 7B ungated; **Llama via `unsloth/Meta-Llama-3.1-8B-Instruct`** (ungated mirror — no HF token).
- `w10_ruler.py` always gets `--chunk` (RULER() helper). Harvest from the live monitor + `vastai logs
  --tail 20000` (log captured the full ~300-line run per pod). **Keys unrotated — flag only.**
- Gotcha hit: one pod's ppl block died on `CUDA error: illegal memory access` (the other two ran the same
  block fine). Lesson: a CUDA fault corrupts the context → all later ppl commands crash; harvest the
  completed RULER immediately and re-run just the ppl elsewhere (MODE=w17ppl).

## 7. Loose ends → Week-18

- The var-track weakness on 1024-dim models is unfixed — needs a **non-surprise retention mechanism** for
  chains (surprise-emphasis provably protects the wrong tokens). A real open problem.
- Tier-3 breadth (LongBench / ∞Bench) still deferred (Phase-1 WS5): the flagship arm needs a 4-arg
  thread-through into `w10_longbench.py`; realistic-QA is eviction's home turf (appendix, not headline).
- The floor unlocks r128/r256 on non-Llama families — worth a proper rank×floor fluency+retrieval sweep.
- Consider promoting `--min-sv-frac 1e-2` toward a default for high-rank configs (it's a strict no-op below
  the onset and a rescue above it).
