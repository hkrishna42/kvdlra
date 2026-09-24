# R2 — Mechanism (blind reading of gate1.md against prereg v2)

## 1. Numbers I read (row = tracker, col = task; ppl rows quoted by delta = isvd − tracker)

Matched-bytes floor control (isvd vs random, arm-2-verbatim, same rank64 + 256 tier):
- llama isvd niah_single `0.79 [0.60,0.91] (19/24)`; random niah_single `0.00 [0.00,0.14] (0/24)` (all four cells 0/24).
- llama ppl isvd `3.7043`; random delta `-6.6172`, bits/token `10.3215`. qwen random ppl `11.0329`, delta `-7.2261`.

Gist vs byte-matched bigger tier (isvd vs nogist), llama retrieval:
- isvd `0.79 (19/24) / 0.42 (10/24) / 0.25 (6/24) / 0.42 (10/24)`.
- nogist `1.00 (24/24) / 0.92 (22/24) ‡ / 0.88 (21/24) ‡ / 0.83 (20/24)` — ‡ = favouring the control on multikey and multivalue.
- llama ppl nogist delta `+0.0533`, TOST `fails`, Holm p `9.54e-07` (nogist lower/ahead).
- qwen retrieval isvd `0.83 (20/24) / 0.79 (19/24) / 0.21 (5/24) / 0.29 (7/24)`; nogist `0.96 (23/24) / 1.00 (24/24) / 0.42 (10/24) / 0.54 (13/24)` (no marks).
- qwen ppl nogist delta `+0.0190`, TOST `fails`, Holm p `0.0435` (nogist ahead).

Tracked vs frozen basis:
- llama ppl frozen delta `-0.0023`, TOST `passes`, Holm p `0.376`. llama retrieval frozen `0.54/0.50/0.25/0.38` (no marks).
- qwen ppl frozen delta `-0.0137`, TOST `fails`, Holm p `0.0167`. qwen retrieval frozen `0.75/0.67/0.17/0.25` (no marks).

fd (C's second control) and oja:
- members line `qwen/niah_multikey: isvd vs fd separates (Holm p=0.027)` (isvd `0.79 (19/24)` vs fd `0.38 (9/24)`).
- llama ppl fd delta `-0.0741` TOST `fails`; qwen ppl fd delta `-0.0939` TOST `fails`.
- oja: llama retrieval `0.71/0.46/0.25/0.54`, qwen `0.42/0.29/0.08/0.29`; llama ppl delta `-0.0445`, qwen `-0.0710` (both TOST `fails`, secondary). No adjusted retrieval p printed for oja/random — read from cells only.

Ceilings: llama full vt `0.71 (17/24)`; qwen full niah_multivalue `0.50 (12/24)`, vt `0.67 (16/24)`. Realised `primary retrieval m=12, secondary retrieval m=18` ⇒ vt excluded by the pre-flight ceiling rule.

## 2. Rule compliance as I read it
- Rule 2 (A/B needs ≥2 families separated, isvd beating BOTH frozen and nogist): 0 families separate — llama (nogist beats isvd; isvd≈frozen), qwen (nogist beats isvd). Not A/B. Agree.
- Rule 3 (C): blocked exactly as the VERDICT lists — llama isvd-vs-fd ppl TOST fail (d=-0.0741), qwen isvd-vs-fd ppl TOST fail (d=-0.0939), qwen/niah_multikey isvd-vs-fd retrieval separation (Holm p=0.027), qwen isvd-vs-frozen ppl TOST fail (d=-0.0137, p=0.09). Agree these block C.
- Verdict `UNDECIDED — nothing selected`. Agree the letter follows from the four members.

## 3. Three strongest objections
1. **The clearest mechanism result is invisible to the rule and points away from the tracker.** nogist beats isvd on retrieval (llama ‡ on multikey+multivalue; every qwen cell numerically) AND perplexity (both families, Holm p=9.54e-07 / 0.0435). This is §5's "sharpest negative", yet by §4's asymmetry (C reads fd/frozen, not nogist) it counts for neither branch — so UNDECIDED masks a "the tier, not the tracker" reading. The VERDICT line never surfaces it.
2. **C is blocked by artifacts, not by the tracker doing work.** The fd arm is ℓ=r, which §2(c) fixed as ~27% worse reconstruction by construction; isvd beating it (qwen multikey Holm p=0.027; ppl both families) is a numerics gap, not gist-is-load-bearing. The qwen frozen blocker is d=-0.0137 (inside ±0.02) at TOST p=0.09 — a not-decidable wide interval, not a real non-equivalence. Strip both and the evidence is C-shaped.
3. **The tracked basis is equivalent to a frozen one — online tracking buys nothing past the margin.** llama isvd≈frozen on both axes (ppl delta -0.0023, TOST passes; no retrieval separation). qwen tracking wins by only 0.0137 bits — real (Holm p=0.0167) but inside the ±0.02 the gate calls meaningful, so it "licenses nothing" (§5). The gist beats only random noise (0.79 vs 0.00; 3.70 vs 10.32), which shows basis *content* matters, not that *tracking* (vs freezing) does.

## 4. What I would need before believing the branch (tracker load-bearing → A/B)
- An isvd-vs-frozen separation OUTSIDE ±0.02 on ≥2 families (now ≤0.0137, inside margin everywhere it is significant).
- isvd beating the byte-matched nogist on some axis in ≥2 families (now nogist wins both axes, both families).
- Ceilings repaired: qwen full multivalue 0.50 and the excluded vt (full 0.71/0.67) mean key gist-vs-tier gaps run against half-height or out-of-family ceilings.
- A decomposition of isvd-over-random into basis reconstruction vs surprise-tier selection — the only place the gist cleanly separates, and the table cannot tell which does the work.

## 5. Verdict
The online-tracked gist beats only noise and a handicapped FD sketch; it is equivalent to a frozen basis and loses to a byte-matched larger exact tier on both axes — it does no work a frozen basis or the tier does not, so the sound mechanism reading is UNDECIDED-leaning-C, not a tracker win.
