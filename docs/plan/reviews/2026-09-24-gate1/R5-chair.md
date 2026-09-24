# R5 — Area chair, Gate 1 blind read (2026-09-24)

## 1. Numbers I read
Decisive A/B contrast is isvd-vs-nogist (byte-matched). It loses:
- llama nogist / niah_multikey `0.92 [0.74,0.98] (22/24) ‡` vs llama isvd / niah_multikey `0.42 [0.24,0.61] (10/24)`.
- llama nogist / niah_multivalue `0.88 [0.69,0.96] (21/24) ‡` vs llama isvd / niah_multivalue `0.25 [0.12,0.45] (6/24)`.
- llama nogist / niah_single `1.00 [0.86,1.00] (24/24)` vs llama isvd / niah_single `0.79 [0.60,0.91] (19/24)`.
- qwen nogist / niah_multikey `1.00 [0.86,1.00] (24/24)` vs qwen isvd / niah_multikey `0.79 [0.60,0.91] (19/24)` (no mark).
- qwen nogist / niah_single `0.96 [0.80,0.99] (23/24)` vs qwen isvd / niah_single `0.83 [0.64,0.93] (20/24)`.
Perplexity, isvd-vs-nogist (delta = isvd − tracker; +ve = nogist ahead):
- llama nogist row: delta `+0.0533 [+0.0386,+0.0700]`, TOST fails, Holm p `9.54e-07`.
- qwen nogist row: delta `+0.0190 [+0.0038,+0.0337]`, TOST fails, Holm p `0.0435`.
Perplexity, isvd-vs-frozen:
- llama frozen row: delta `-0.0023 [-0.0071,+0.0026]`, TOST passes, Holm p `0.376`.
- qwen frozen row: delta `-0.0137 [-0.0224,-0.0047]`, TOST fails, Holm p `0.0167`.
C-blockers, `members:` line — `qwen/niah_multikey: isvd vs fd separates (Holm p=0.027)` (qwen isvd multikey `0.79 (19/24)` vs qwen fd / niah_multikey `0.38 [0.21,0.57] (9/24)`); `llama: isvd vs fd TOST … does not pass (d=-0.0741 bits, p=1)`; `qwen: isvd vs frozen TOST … does not pass (d=-0.0137 bits, p=0.09)`; `qwen: isvd vs fd TOST … does not pass (d=-0.0939 bits, p=1)`.
Header: realised m — primary retrieval 12, secondary 18, primary perplexity 4.

## 2. Rule compliance as I read it
- Rule 1+2 (A/B): a family separates iff isvd beats *both* frozen and nogist. **llama** fails both legs (frozen equivalent — TOST passes, Holm p 0.376; nogist ahead — ‡ on two tasks + ppl delta `+0.0533`). **qwen** meets the frozen leg on perplexity (delta `-0.0137`<0, Holm p 0.0167, TOST fails) but fails the nogist leg (nogist ahead: retrieval descriptively, ppl delta `+0.0190`). So **0 of 2 families separated → not A/B**. Agrees with VERDICT.
- Rule 3 (C): blocked by (i) qwen/multikey isvd-vs-fd Holm p 0.027, and (ii) three failing TOSTs (llama fd `-0.0741`, qwen frozen `-0.0137`, qwen fd `-0.0939`). Agrees with the four `members:` exactly.
- Realised m 12/18/4 = §6's "one task excluded" (16−4, 24−6); vt is that task (full vt 0.71 llama / 0.67 qwen < 0.9) and is still printed, unmarked — consistent. ‡ marks sit on the control (nogist) rows as the legend requires; no `*` anywhere; the fd separation shows in `members:`, not as a cell mark (secondary family). All consistent.
- bf16 block: rule not in the two files — reported as printed (llama PASS, qwen PASS; `sbits bf16/isvd` 0.5667 / 0.5382).
- I agree with `VERDICT: UNDECIDED` and its blocker list.

## 3. The three strongest objections
1. **The letter is structurally terminal, and Stage 2 cannot move it.** Stage 1 separated 0 of 2 families — both fail on the *same* fact, isvd never beats nogist — and those letters are not recomputed (§4). Rule 2 needs ≥2 of {llama,qwen,mistral}; a fully-separated Mistral gives only 1 → still `UNDECIDED (one family separated)`. C is blocked by *decided* qwen Stage-1 facts (fd multikey separation + failing fd/frozen TOSTs) that Mistral and the 32K pods cannot undo. Absent an amendment, neither A/B nor C is reachable.
2. **UNDECIDED under-states a clean tracker-null.** The *only* thing blocking Branch C's substance is isvd's edge over `fd` — an arm the prereg itself byte-cuts to ℓ=r=64 and calls 27% worse in reconstruction (§2c). "isvd beats a handicapped FD" says nothing about online-tracking-vs-frozen. The decisive controls (frozen ≈ isvd; nogist ≥ isvd on *both* axes, both families) uniformly say the gist earns none of its bytes.
3. **The load-bearing negative is not uniformly significant on retrieval.** nogist-beats-isvd is Holm-significant on llama retrieval (‡ multikey/multivalue) and both perplexity families, but on **qwen retrieval** it is descriptive only (no ‡: qwen nogist 0.96/1.00/0.42 vs isvd 0.83/0.79/0.21). Direction is unanimous, but the "tier dominates on both axes" claim leans on llama retrieval + the perplexity axis. (Minor: retrieval runs near the floor — isvd multivalue 6/24, 5/24 — where §6 says only a ≥(10,0) swing separates; the C branch guards this by requiring a TOST pass, and nogist clears the floor, so it caveats the frozen ties, not the nogist result.)

## 4. What I would need before believing the branch
- The branch *is* UNDECIDED; I'd want the DECISIONS entry to record the substantive negative (nogist ≥ isvd on both axes; isvd ≈ frozen) beside the letter, per §4/§5, so UNDECIDED is not read as "inconclusive."
- Confirmation the two §4 refusals cleared on the pod (no frozen repairs past freeze; nogist `sbits` within 1±0.05) — the table shows no refusal, implying they passed, but that is not visible in these two files.
- Confirmation vt's exclusion came from the pre-flight ceiling, not Stage-1 data (§5/§6) — not visible here.
- Mistral 16K to confirm the nogist-dominance / isvd≈frozen pattern in a third family (it cannot change the letter). 32K is descriptive only, and frozen is weaker at 32K by construction (§3), so a 32K frozen gap is weaker evidence, not stronger.

## 5. One-line verdict
Agree with UNDECIDED: the gist loses to a byte-matched no-gist tier on both axes in both families and only ties a frozen basis, so no family separates and A/B collapses; the sole block to a clean Branch C is isvd's edge over a deliberately-handicapped FD, so the science is a tracker-null that Stage 2 can corroborate but, as pre-registered, can no longer convert into either branch.

Confidence the online-tracked gist does work a frozen basis or no gist does not: **2/10**.
Confidence the table follows its pre-registration: **8/10**.
