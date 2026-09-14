# Systems review — kvdlra exit-gate re-review (2026-09-06)

Dimension: **systems claims** (stored state vs resident memory, persistence/cold start, kernel absence,
64K memory, "supporting axis" acceptability). Repo `/Users/hari/Desktop/kv-dlra` @ `5bc772f` (week7).
All citations file:line in that tree. $0 read-only.

## Score: 6/10 (borderline accept / poster) — no FATAL flaw

Conditional: 6 assumes the §memory "1/T" paragraph is rewritten (Finding 1, $0). Left as worded,
the abstract carries a "no floor / keeps falling" claim that the paper's own three data points
refute, and I would score 5. With Findings 1–3 fixed (≈$10 GPU + a day) this dimension is a 7.

Previous panel: 3 (misleading VRAM framing) → 7 (after reframe). The reframe is real and now
present in every paper-facing place (§1 below). What the closing revealed: the residual over-reach
moved from "storage read as VRAM" to two narrower systems statements — an asymptotic 1/T claim that
the flagship's configuration cannot deliver, and a decode-latency disclosure measured at a context
two orders of magnitude below the operating point.

---

## 1. Stored state vs resident memory — CLOSED, with three residual sentences

Present and correct in all four required places:
- Abstract `paper/main.tex:62-65` (float-equivalent stored state), `:75-76` (matched stored bytes),
  `:86-89` ("today's storage ratio is not deployed VRAM ... decode residency ≈1.06×").
- Intro bullet `:153-155`; Setup `:363-364`; Related work concedes eviction "genuinely shrinks
  resident memory" `:292`.
- §6.8 `:819-827` dual billing (fp16-equiv 0.085× vs fp32-at-rest ≈0.150×); `:857-876` resident
  paragraph; footnote `:865-871` explains `peak_ratio` is weight-dominated.
- Limitations lead with it `:919-924`; Conclusion `:1003-1004`, `:1009-1010`.
- Baseline residency asymmetry conceded: "eviction and ThinK (0.75× resident) currently beat BUG"
  `:873-876`.

Sentences that still read storage as an unbounded-memory property (fixable, $0):

1. **`main.tex:829-833` "The gist state is constant in context length ... all O(rn+hn), independent
   of T"** and `:843-845` "a bounded-rank tracker is not floored at all", `:836` "keeps falling".
   False for the flagship as run. `scripts/w10_frontier.py:240` sets `coord_budget = t + rw + ab`
   (`:211` for plain bug): the per-token coordinate buffer is sized to the whole context, so stored
   state is Θ(T) with slope 2r·32/(2n·16) = **0.125×** (fp32 coords). The measured points confirm it:
   0.1503/0.1387/0.133 (`results/w18-g5-llama-lines.txt:2,5`; `results/w19-a4-llama-lines.txt:1`)
   fit ratio(T) = **0.127 + 380/T** (predicts 0.1329 at 64K; measured 0.133). The curve halves its
   drop every doubling — visible in `figures/week19/one_over_t.png`. The method's *bounded* buffer
   (`src/kvdlra/cache/bug_cache.py:20-27`, `main.tex:233-237`) is a different, coordinate-evicting
   configuration the paper never evaluates at 16K+. The abstract's `:91-93` "falls from 0.151× to
   0.133× ... while any fixed-bit quantizer stays flat" is numerically true but the implied
   divergence is not: the asymptote is 0.127× vs KIVI-2bit's 0.156× (16-bit aux) — see §2.3.
2. `main.tex:233-237` "bounded memory means bounding the attended set ... at matched memory the
   gist retains ≈n/r× longer history" — describes coordinate eviction; the flagship never evicts a
   coordinate at these contexts (`w10_frontier.py:240`). Say so, or drop.
3. `main.tex:1-20` header comment and body text are stale against the new sections: `:5-8`, `:20`
   ("NO 2-bit baseline yet"; "quant baseline is v2 work"), `:277-281` ("head-to-head against
   dedicated 2-bit KV quantizers ... as the primary v2 experiment"), `:302-305` ("their absence as a
   baseline is v1's main scoping decision"). §6.7 `:630-681` is that experiment. A linear reader
   hits the contradiction 350 lines before the memory section. Not my dimension, but it damages
   the credibility of every "we state clearly" sentence.

Public-facing but non-manuscript: `README.md:63,66,82,95` still say "3–5× less memory" unqualified
(disclaimer only at `:211`). The `w19.sh` A4 pre-registration `scripts/pod/w19.sh:63-65` and
`docs/week19-kickoff.md:93-95` carry the same "constant in T" wording — that is where the paper's
sentence came from.

## 2. The measured cold start (C4)

### 2.1 Methodology — mostly sound; four gaps

Sound:
- **Persisted = billed tensors**: `scripts/w19_persist.py:44-48` (`_BUG_ATTRS`) mirrors
  `bug_cache.py:1658-1717`; cores as diagonals `w19_persist.py:64-67` matches the accounting
  convention `bug_cache.py:1673-1680`; pinned by `tests/test_w19_persist.py:91-104`
  (`numel == cache.stored_state_numel()`). On-pod cross-check: 325,109,371 B vs analytic
  0.1503 × 2,147,483,648 = 322.8 MB (+0.7%, pickle overhead) — `results/w19-a3-llama2-lines.txt:1`
  vs `w18-g5-llama-lines.txt:2`.
- **Cloned views**: `w19_persist.py:105-106` clones before `torch.save`; the first pod run wrote
  whole ring/diagonal storages and inflated the flagship 2.6× (`results/w19_harvest/
  a3-llama.raw.superseded:65` ratio=0.4019); fixed at `5e8b275`, superseded file dropped `69f7ea9`,
  content-size test `tests/test_w19_persist.py:126-140`. Good hygiene, well disclosed in git.
- **Synchronize**: H2D `w19_persist.py:113-116` and reconstruct `:135-146` bracketed by
  `torch.cuda.synchronize`. Medians of `--repeats 5` (`scripts/pod/w19.sh:125`; paper `:888`).
- **Provenance**: A100-PCIE-40GB, torch 2.11.0+cu128, transformers 5.8.0, SHA 5e8b275
  (`results/w19_harvest/a3-llama2.raw:5-9`).
- Quant arm symmetric: full dequantize timed as its "ready" step `w19_persist.py:143-145`.

Gaps (fixable):
- **(a) Warm page cache is disclosed in the script (`w19_persist.py:13-14`) but not in the paper**,
  which calls it "disk read" `main.tex:890` and titles the figure "disk read" (`figures/week19/
  coldstart.png`). The file is read back immediately after being written.
- **(b) The 9–10× exceeds the byte ratio (6.6×/7.2×) and the excess is unexplained.** Per-byte
  load rates from `w19-a3-llama2-lines.txt`: full 1.93/2.03 GB/s vs flagship 4.85/5.82 GB/s vs
  KIVI 3.65/5.33 GB/s. Same `torch.load`, same page cache — a 2.5–3× per-byte asymmetry that
  favours the small artifacts (plausibly partial page-cache eviction of the 2–4 GB file on a
  container with bounded RAM, or `torch.load` non-linearity). The honest statement is "≈7× by
  bytes; 9–10× measured on this pod". H2D rates are the sane ones (11–19 GB/s, PCIe).
- **(c) "Attend-ready" is never demonstrated from the file.** `attend_ready_seconds`
  (`w19_persist.py:137-141`) resets `_mid_k_cache` on the *live* cache and rebuilds; the reloaded
  tensors (`:111-114`) are moved to device and discarded (`:117`). No code path restores a
  `BugStreamingCache` from the persisted dict (scalar state — `cumulative_length`, mode, dtype,
  RoPE handle — is not persisted; cores need `diag()` re-expansion). The numel pin guards against a
  missing tensor, not against a non-resumable state. One CPU test (restore → one decode step →
  logits equal to the live cache) would close this.
- **(d) No recompute-prefill bar.** The deployment alternative to loading a persisted cache is
  re-prefilling from tokens (≈1–2 s for 16K on 8B/A100). Full-KV load at 1.23 s barely beats it;
  BUG at 0.13 s clearly does. That is the actual win and it is absent. CacheGen (cited `:108`,
  `:302`) is *the* prior work on persisted-KV bytes and is not positioned on this axis, which the
  previous review asked for (`docs/reviews/2026-09-01/review-systems.md:111-113`, `:203`).

### 2.2 Is "9–10× vs full KV" honest given 2-bit cold-starts equally fast?

Yes as disclosed: `main.tex:893-897` states the 2-bit arm at 0.14/0.21 s and that the win is
"shared with quantization"; the abstract `:90-91` repeats it; the figure shows all three bars. The
quantizer's persisted bytes (0.1563×, `w19-a3-llama2-lines.txt:3,6`) are its *flushed* state
(`w10_frontier.py:139-157` calls `flush`), slightly below its billed 0.163×/0.160× (residual
included, `accounting.py:441-447`) — favourable to the baseline, fine.

### 2.3 Is "1/T separates them" meaningful for deployment? Stated plainly: no, not as worded

Bytes (honest, fp32-at-rest vs KIVI-2bit honest): 16K 0.150 vs 0.163 (**7.8% fewer**); 32K 0.139
vs 0.160 (**13.3%**); 64K 0.133 vs 0.158 (**15.8%**); asymptote 0.127 vs 0.156 (**18.5%**). The
quantizer's own ratio also falls (0.163→0.158: its fixed 128-token fp16 residual amortizes,
`accounting.py:441-443`), so "any fixed-bit quantizer stays flat" (`:93`) and the figure title
("a b-bit quantizer is flat") are both slightly wrong in the same direction. The 1/T term is
U + tier + sinks + ring ≈ 380 full-token-equivalents per layer; the Θ(T) term is the fp32
coordinates at 0.125×. A **fp16 gist** would put the asymptote at ≈0.063× — genuinely below the
2-bit floor — but the paper says that variant "is not yet validated" (`:827`). So today's separation
on the persistence axis is a 16% byte difference at 64K plus a retrieval difference (flagship 1/1/1/1
n=8 vs 2-bit 1.00/0.58/0.50/1.00 n=12, `w19-a4-llama-lines.txt:6-21`) — the retrieval difference is
the interesting part, not the bytes. Rewrite `:829-846` and `:897-900` around ratio(T) = 0.127 +
380/T and say "16% fewer bytes at 64K, 19% asymptotically, at fp32 coordinates".

## 3. Kernel absence: residency and latency disclosure

What is disclosed: workspace 0.982/0.991× measured on CUDA with integrity 0
(`w18-g5-llama-lines.txt:2,5`; `main.tex:859-863`); "decode residency ≈1.06×" (`:87`, `:865`,
`:920-921`); one latency datum 1B/CPU +10% (`:871-873`, `:922`); "kernel not built" (`:173`,
`:873-876`, `:919-924`, `:1009-1010`).

What a systems reviewer still objects to:

1. **"≈1.06×" is an analytic sum, not a measurement, and is presented as "corroborated".**
   It is stored 0.084 + workspace 0.982 (`w18-g5-llama-lines.txt:2`), inherited from the previous
   review's arithmetic (`review-systems.md:44`; the paper's own source comment `main.tex:912`
   "Latency/residency 1.06x from review-systems.md"). The full-KV arm's CUDA peak was never
   measured: `scripts/w16_storage.py:82-93` returns `peak_bytes=None` for `full`, hence
   `peak_ratio=cpu` in `w18-g5-llama-lines.txt:3,6`. The BUG-arm peaks that exist (9.09×/5.34×) span
   chunked prefill + one decode step (`w16_storage.py:95-111`) and are weight-dominated; after
   subtracting 16.06 GB of weights the KV-attributable peak is 3.5/6.9 GB = **1.6× full KV** at
   16K/32K (prefill-inclusive; fp32 `u@c` intermediates `bug_cache.py:1603-1604` before the cast
   `:1617-1618`, per-layer cats `:1560-1561`). Label 1.06× "analytic (stored + cached
   reconstruction)"; a measured full-vs-flagship decode peak is a $5 run.
2. **The latency datum is not representative of the operating point.** `results/
   w5-decode-validate-1b.json`: prompts of 141–167 tokens, 160 generated → context ≤ 327 tokens,
   where the middle is a few hundred columns and reconstruction is free. The paper quotes prompt 0
   (224.9 vs 204.5 ms, +10%); prompt 1 in the same file is 270.0 vs 202.2 ms mean (+34%; p50 +15%).
   At 16K–64K the costs the datum cannot see are: (i) the middle rebuild on every absorb event —
   `_absorb_block_into_stream` invalidates `_mid_k_cache` (`bug_cache.py:754-766`), the next
   `_decode_peek` rebuilds all layers (`:1553-1557`, `:1583-1618`); the rebuild is exactly the
   measured `ready` column, 39 ms at 16K / 82 ms at 32K (`w19-a3-llama2-lines.txt:1,4`), ≈164 ms
   at 64K, amortized over `absorb_block=16` decode tokens (`w10_frontier.py:820`) = **2.6 / 5.1 /
   10 ms per token**; (ii) the per-step full-length concat `bug_cache.py:1560-1561` (`_to_hf` is a
   view, `:670-673`, but `torch.cat` copies): 4.3 / 8.6 / 17 GB of traffic per step ≈ 3 / 6 / 11 ms
   at 1.5 TB/s. Against a full-KV bandwidth floor of ≈12 / 14 / 16 ms per step (weights + KV), the
   estimate is **BUG decode 1.4–2.3× slower than full KV at 16K–64K**, not 1.1×. This is an
   estimate; the point is that the paper's only number is likely wrong by 5–10× at its own
   operating point, and the ingredients to compute the estimate are already in the paper's tables.
3. **Batch size 1 is a hard limit and is nowhere in the manuscript.** `bug_cache.py:629-631` raises
   on `key_states.shape[0] != 1`; the `(1,H,T,D)` layout is baked into `_to_mat`/`_to_hf`
   (`:650-653`, `:670-673`). `grep -i batch paper/main.tex` → 0 hits. The briefing lists "batch 1"
   as a scope caveat; the paper does not. One sentence in Limitations.
4. What I would demand before calling the systems axis "supported": decode ms/token and tokens/s at
   16K/32K/64K, batch 1, medians, same A100, for full / flagship / KIVI-2bit (the HF `QuantizedLayer.
   update` dequantizes the entire store every step — `.venv/.../transformers/cache_utils.py`
   `QuantizedLayer.update`, v5.8.0 — so the fair-quant arm also has no realized residency; that is
   worth one sentence because it makes the comparison symmetric); the absorb-event rebuild cost
   reported per token at 64K; a measured decode peak for full vs flagship. Batch>1 is out of scope
   for this paper if disclosed. None of this needs a kernel; ≈$10 and a 60-line script that reuses
   `w19_persist.run_persist`'s prefill and `w10_ruler._decode`'s greedy loop.

## 4. The 64K run on a 40GB card

- Pod: A100-PCIE-40GB, 40960 MiB (`results/w19_harvest/a4-llama.raw:6`) although the
  pre-registration said "Needs an 80GB card" (`scripts/pod/w19.sh:66`). No OOM/skip markers in the
  raw; all pre-registered arms landed (`w19-a4-llama-lines.txt:1-21`; flagship n=8 = `--n-trials 4
  --seeds 0 1`, `w19.sh:75`). Budget check: weights 16.06 GB + bf16 workspace ≈0.98×8.6 GB + fp32
  stored 1.1 GB + per-layer transients ≈ 27 GB — fits. `CHUNK=4096`, `DTYPE=bfloat16`
  (`scripts/pod/w18_boot.sh:32-33`).
- Memory claims made at 64K: only the stored ratio 0.133× (sbits; fp16-equiv 0.070) and retrieval/
  ppl (`main.tex:837-846`, `:91-93`). No VRAM or cold-start claim at 64K; the persistence figure is
  16K/32K only. `:846` "the persistence win grows with context" and `:886` "with slower storage
  tiers" are projections (measured growth 9.2×→10.4× is consistent). Nothing at 64K is
  misreported. Nitpick: the abstract gives 0.085–0.149× and 0.15× for the same config in one
  paragraph (`:62`, `:76`) without saying which billing each is.

## 5. Supporting axis rather than headline — acceptable?

**Yes, conditionally.** The paper already positions it that way (`:89-93` "the measured systems
payoff is ..."; `:1009-1010`). The persisted-bytes number is honest and its measurement is real
(cloned, synced, pinned, SHA'd). What I will not accept as a supporting axis: (i) the "no floor /
constant in T" paragraph in §memory and the abstract, because the flagship's own configuration
refutes it; (ii) a "+10% slower" datum at 327 tokens standing in for decode cost at 64K. Fix those
two and disclose batch 1, and the systems story is a clean, modest, honest secondary result:
"≈7× smaller persisted artifact (≈9–10× faster warm reload on this pod), shared with 2-bit
quantization; 16% fewer bytes than 2-bit at 64K with full four-task retrieval where 2-bit loses two
tasks; resident memory and decode latency not realized without a kernel."

---

## Severity buckets

**FATAL:** none.

**Fixable (must-fix before submission):**
- F1 `main.tex:829-846`, `:91-93`, `:897-900`, `figures/week19/one_over_t.png` title: rewrite around
  ratio(T) = 0.127 + 380/T; delete "constant in context length", "not floored at all", "keeps
  falling"; state 16%/19%. $0, 1 h.
- F2 `main.tex:871-873`, `:922`: replace/augment the 1B-CPU datum with GPU decode ms/token at
  16K/32K/64K (full / flagship / KIVI-2bit) + amortized rebuild per token; measure the full arm's
  decode peak in `w16_storage._measure` (`:82-93`) so 1.06× becomes a contrast. ~$10, half a day.
- F3 `main.tex:919-924`: add "batch size 1; the cache layout is single-sequence
  (`bug_cache.py:629-631`)". $0.
- F4 Cold-start hygiene: "warm page cache" in `:889-893` and the figure title; "≈7× by bytes,
  9–10× measured" (`:89`, `:893`); one restore-from-file round-trip test; recompute-prefill as a
  third bar; one sentence positioning vs CacheGen. $0 for wording; ~$5 for the rerun with
  `drop_caches` + recompute timing.
- F5 Stale text contradicting §6.7: `:5-8`, `:20`, `:277-281`, `:302-305`. $0.

**Nitpicks:**
- `:865` "corroborates a decode residency ≈1.06×" → "implies (analytically)"; `:912` source comment
  cites the previous review, not a measurement.
- `:233-237` describes coordinate eviction the flagship never uses.
- Abstract dual billing without labels (`:62` vs `:76`).
- `w5-decode-validate-1b.json` prompt 1 (+34% mean) not mentioned beside prompt 0 (+10%).
- `README.md:63,66,82,95` unqualified "less memory".
- Figure `coldstart.png`: "disk read" → "page-cache read".

## Gap-fills, cost-ordered
1. F1 + F3 + F5 wording — $0, hours.
2. Restore-from-file round-trip test (CPU) — $0, hours.
3. GPU decode latency/peak script + run (F2) — ~$10.
4. Cold-cache rerun with recompute bar (F4) — ~$5.
5. fp16-storable gist validation (would make the 1/T story true: asymptote ≈0.063×) — ~$50 GPU.
6. Fused factored-attention kernel — weeks; not required for an honest submission.
