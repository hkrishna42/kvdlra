# AC meta-verdict — kvdlra exit-gate re-review (2026-09-06)

Area Chair synthesis of the six dimension reviews (`review-{claims,prior-work,rigor,significance,systems,repro}.md`,
all scored against `5bc772f`) plus the applied $0 fix pass (`b57c560` provenance/repro,
`fb804bd` paper wording), judging the manuscript **as it now stands** at HEAD `216481d`
(= `fb804bd` + `2cf54ae` figure-title correction + `216481d` board). Read-only; every
claim below is grounded in a current `paper/main.tex` line or a file:line.

Exit-gate bar: **≥7 on every dimension (claims, prior-work, rigor, significance, systems, repro) AND zero fatal flaw.**

---

## 1. Headline — does it clear the gate?

**No — but it is close, and both FATAL-as-worded items are genuinely closed.** After the
$0 pass the manuscript has **zero fatal flaws**, all 42 `\citep` keys resolve (verified:
`comm -23` of used-vs-defined keys is empty), and no headline claim is now stated more
firmly than its evidence — the two overclaims that made prior-work and significance
*borderline-reject-as-worded* (OjaKV/"unoccupied axis"; the 1/T "not floored" + marquee
0.16× billing) are corrected in the actual text, not just the commit message. What keeps
the gate unmet is that **three dimensions sit at an honest 6, and each one's 7 needs a
GPU experiment that a wording pass cannot substitute for**: prior-work (is the DLRA
integrator *necessary* — tracker-swap ablation), significance (does the one exclusive
quantitative claim survive its real competitor — eviction×quant composite), and systems
(does the residency/latency story hold at the real operating point — a decode-latency
measurement). This is exactly the outcome the significance and prior-work reviewers
predicted ("6 after $0; 7 with one GPU experiment"). The paper is **submittable to arXiv
v1 now** (its stated purpose) and is a credible **ICML poster / borderline-accept**; it is
**not yet accept-level** on the exit-gate bar.

---

## 2. Per-dimension verdict (scores verified against current text)

| dimension | as-worded | post-$0 (AC, verified) | fatal closed? | what still blocks a 7 (cost) |
|---|---|---|---|---|
| claims | 6 | **7** | n/a (was none) | nothing blocks 7; GPU items (ppl re-score, r256-floored control, 0.05× eviction comparator) are now **honestly scoped as "not measured here"**, not asserted. ~$20 GPU would *firm* it. |
| prior-work | 5 | **6** | **yes** | tracker-swap ablation (BUG vs Oja/FD/incremental-SVD, same cache, end-to-end) — the title's word "DLRA" is load-bearing only if the integrator beats its identity-twins on retrieval/ppl, not just Week-2 1B reconstruction. **~$15–20 GPU.** |
| rigor | 6 | **7** | n/a | inferential framing now honest (pooled sign tests in, equivalence wording gone, confounds disclosed). Softest 7: the single-shot 2-bit prefill control (~$10) is the top residual, but the affected claim (in-repo mv edge) is already hedged as non-transferring. |
| significance | 5 | **6** | **yes** | sub-cliff cell vs its real competitor (KIVI-4×ea-k0.1 ≈0.03×, KIVI-2×ea-k0.25 ≈0.04×) on the **official** anchor + in-repo. The paper's *only* exclusive quantitative claim is currently uncontested. **~$40–50 GPU (~$30 Llama-only).** |
| systems | 6 (cond.) | **6** | n/a | decode ms/token + a measured full-vs-flagship decode peak at 16/32/64K (the residency 1.06× is analytic and the only latency datum is at 327 tokens / 1B / CPU). **~$10 GPU.** |
| repro | 7 | **7** | n/a (was none) | clears; `b57c560` pushed it toward 8 (W19 env-provenance committed, W18 ppl line-files committed + loaders wired). Residual: confirm CI green at the submitted SHA. |

**Gate: 3 of 6 below 7 (prior-work, significance, systems). Zero fatal.**

Why claims/rigor reach 7 but systems does not, on comparable GPU costs: claims' and
rigor's "6" was **overclaiming/framing** — entirely $0, and done (I verified abstract-vs-body
alignment cell by cell). Their residual GPU items either become honest limitations
(claims) or are disclosed confounds on an already-hedged claim (rigor). Systems', prior-work's
and significance's "6→7" gaps are **missing measurements that bear on a central claim** and
that no wording can convert to an honest limitation without gutting the claim itself.

---

## 3. The two FATAL-as-worded items — closed by the applied text? Yes; quoted.

### FATAL #1 (prior-work): OjaKV uncited / "unoccupied axis" false / incremental-SVD identity unstated — **CLOSED**

- Abstract now: *"Most low-rank methods fix the subspace at calibration or in a one-shot
  prefill pass; the online, per-sequence axis was opened by OjaKV~\citep{zhu2025ojakv} with
  Oja's rule."* (`main.tex:56–58`) — the false "unoccupied axis" is replaced by "opened by OjaKV".
- Related work: *"The exception is OjaKV~\citep{zhu2025ojakv}, which updates a per-sequence
  subspace online with Oja's rule; we differ from it in the tracker … and in selecting the
  exact tier by \emph{content} (surprise) rather than by position."* (`:363–367`).
- New Method paragraph "Relation to incremental SVD and sketching": *"For a rank-one column
  increment the augmented BUG step reduces to the classical incremental-SVD update … keep the
  leading directions~\citep{brand2006fast} … a deterministic sketch in the spirit of Frequent
  Directions~\citep{liberty2013simple}. What the DLRA framing adds is the rank-adaptive
  truncation criterion … and the σ_min-independent error bound; the near-oracle quality we
  report … is a \emph{measurement} on real KV, not a theorem for this column-stream setting."*
  (`:231–244`) — states the identity and scopes σ_min-robustness to a measurement, exactly as asked.
- The uncited baselines at old `:382` are cited: *"beats fixed-rank incremental
  SVD~\citep{brand2006fast} everywhere, and beats Oja's rule~\citep{oja1982simplified,zhu2025ojakv}
  by 1.3–3.0×"* (`:438–440`).
- Verified present and well-formed in `refs.bib`: `zhu2025ojakv`, `brand2006fast`,
  `liberty2013simple`, `oja1982simplified`, and the demanded `sun2024shadowkv` (`:253`, `:358`),
  `kang2024gear` (`:321`), `zhang2024cam`/`wang2024kvmerger` (`:400`).
- *Residual (not part of the FATAL, does not block the closure):* the tracker-swap **ablation**
  that would make "DLRA" load-bearing end-to-end is still unrun — this is the prior-work 6→7 item, not the fatal.

### FATAL #2 (significance/systems): 1/T "not floored / diverges" + marquee billed at 0.16× — **CLOSED**

1/T:
- Abstract now: *"the O(rn+hn) overhead amortizes, so the stored ratio falls from 0.151× (16K)
  toward its 2r/n=0.125× asymptote (0.133× at 64K, 19% under the 2-bit arm's 0.156×) … while the
  quantizer's ratio is nearly flat."* (`:108–112`).
- §memory: *"So the honest ratio falls toward 2r/n, not toward zero … The distinctive property is
  therefore a lower \emph{asymptote} (2r/n=0.125× … vs the 2-bit arm's 0.156×, a 19% margin, not an
  unbounded one)."* (`:917`, `:930–932`); figure caption matches: *"toward a 2r/n=0.125× asymptote
  (fit 0.127+380/T); a b-bit quantizer is nearly flat at b/16 plus a small amortizing residual."*
  (`:942–945`, and `2cf54ae` regenerated the figure itself).
- Confirmed removed (grep count 0 each): "not floored at all", "constant in context length",
  "keeps falling", "independent of T".

Marquee billing:
- Abstract: *"beats … think-c0.5 … at 1.8–2.6× less honest stored state (0.284×)—bytes at which
  4-bit KIVI also scores 1.00 variable-tracking, so the separation is from ThinK/Palu, not from
  quantization."* (`:74–80`).
- §marquee: *"0.284× honest stored state, 0.16× float-equivalent"* (`:519`); *"the marquee is
  byte-matched with the 4-bit KIVI arm, which scores variable-tracking 1.00 (12/12) at 32K …"* (`:533–536`);
  and `tab:marquee` now carries the **KIVI 4-bit / 0.284× / "tie (same bytes)"** row (`:568`).
- Confirmed removed (grep count 0): the "3.2–4.7×" claim ("4.7" appears nowhere in the file).

---

## 4. Residual risks a Reviewer 2 will still raise (ranked; cheapest experiment that retires each)

1. **"Your one exclusive claim is uncontested."** The sub-cliff q4 cell (`:640–679`) is exclusive
   only *vs scalar quantization*; its real competitor is eviction×quantization (KIVI-4×ea-k0.1
   ≈0.03×), which the paper's own 1B frontier says is the band to beat (`:891–895`), and which is
   unrun at 7B/8B. The paper now scopes this honestly (*"whether an eviction×quantization composite
   reaches it is not measured here"*, `:655`), but honest scoping caps significance at 6.
   **Retire: 1 pod, composite arms on official RULER + in-repo, Llama+Mistral 16/32K, n=12 (~$40–50).**
2. **"Is DLRA doing the work, or is any incremental-SVD/FD tracker as good?"** The paper now *states*
   the algebraic identity (`:231–244`); the only evidence the DLRA integrator matters is 1B
   layer-8 reconstruction error (`:436–441`). **Retire: tracker-swap ablation, same cache, swap
   BUG/Oja/FD/incremental-SVD, Llama 16K n=12 + ppl (~$15–20).**
3. **"Your residency/latency numbers are not at your operating point."** 1.06× is analytic
   (labeled so at `:956–958`); the only latency datum is 1B/CPU at ≤327 tokens (`:965–966`). Systems
   reviewer estimates the true decode cost is 1.4–2.3× full KV at 16–64K. **Retire: decode
   ms/token + full-vs-flagship decode peak at 16/32/64K (~$10).**
4. **"The 2-bit multi-value 'win' rides a chunked lossy prefill with no single-shot control."**
   Now disclosed (*"a KIVI \emph{scheme} under a streaming protocol"*, `:713–714`), but the confound
   is uncontrolled. Mitigated because the mv edge is already stated not to transfer to official RULER
   (`:844–848`). **Retire: `--chunk 0` 2-bit control, Llama 16K, 4 tasks, n=12 (~$10).**
5. **"KIVI-scheme at G=64 is weaker than paper-KIVI at G=32."** Disclosed (`:710–712`) but no G=32
   row and no reproduced published KIVI number. **Retire: G=32 row + one KIVI-paper ppl point (~$10).**
6. **Sub-resolution fluency numbers.** bf16-summed NLL (±0.006 bit/tok) is disclosed and gaps below
   it labeled "to scorer resolution" (`:580–583`, `:929`), but the numbers are not re-scored in fp32.
   **Retire: `.float()` + re-score the marquee/fair-quant/64K ppl cells (~$15).**

Experiments 1+2+4+5 can share one or two pods; the honest full-gate cost is ≈$65–80 of GPU, not one run.

---

## 5. AC recommendation

- **arXiv v1 now: yes.** The paper's declared purpose is a timestamp of the mechanism (`:5–9`,
  `:185`). It is internally consistent, every citation resolves, both fatal-as-worded overclaims are
  gone, and the honesty discipline is unusually strong (dual billing, pooled sign tests, the official
  anchor reported against interest, q4 invalidation and the persistence view-inflation both disclosed).
  Two minor imprecisions the fix pass left, worth a 5-minute touch-up but not blockers: the abstract's
  *"0.133× at 64K, 19% under … 0.156×"* attaches the asymptotic 19% margin to the 64K point, which is
  itself ~15% under (`:110–111`; §memory states it correctly); and *"scattered misses the quantizers
  do not make"* (`:852`) is loose since the 2-bit arm also misses (mean 0.87).
- **Against the ICML 2027 bar: poster / borderline-accept, not accept.** With significance at 6 and
  no external-benchmark quantitative win at ≥2 bits/element, this is the "honest mechanism paper with
  one exclusive band and a good failure map" that the program pre-registered as the poster branch —
  a real contribution (a genuinely new online-DLRA axis, the bits-per-element view of the fair
  comparison, the r/n≈0.25 wall and rank-siphoning map), but the abstract's strongest quantitative
  claim (the sub-2-bit exclusive band) is not yet contested-and-won.
- **Single highest-leverage next step (the reviews converge here):** **run the sub-cliff cell against
  eviction×quantization composites on the official RULER anchor** (significance §5 / claims §3.3 /
  prior-work §3.5), ~$40–50 (or ~$30 Llama-only), pre-registered as a decisive fork. It unblocks the
  lowest and most central dimension (significance 6→7) and simultaneously retires the claims and
  prior-work "the band is asserted, not measured" objection. If the composite collapses on the essay
  haystack while the q4 cell holds single/mk/mv, the exclusive band is anchored against *every* family
  and the paper has one clean quantitative claim — a firm accept-track result. If the composite holds,
  the paper is honestly the mechanism paper it already mostly is, and better to learn it now than from
  Reviewer 2. Pair it with the ~$15 tracker-swap and ~$10 decode-latency pods to clear prior-work and
  systems, and the exit gate is met.

*Read-only; no paper edits, no pods, no push. Verdict written to this file only.*
