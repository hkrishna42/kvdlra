# R4 — Claims and writing (blind read of gate1.md against prereg §4/§6)

## 1. Numbers I read
- Realised Holm sizes, header comment: `primary retrieval m=12, secondary retrieval m=18, primary perplexity m=4` — so `vt` is excluded (Amendment 1b/D-018); every `vt` cell below is descriptive and marks nothing.
- llama ctx16384, row `nogist`, col `niah_multikey`: `0.92 [0.74,0.98] (22/24) ‡`; col `niah_multivalue`: `0.88 [0.69,0.96] (21/24) ‡` — control (nogist) beats isvd, Holm-significant, both operative tasks.
- llama ctx16384, row `isvd`: `niah_single 0.79 (19/24)`, `niah_multikey 0.42 (10/24)`, `niah_multivalue 0.25 (6/24)`. Row `frozen`: `0.54 / 0.50 / 0.25`. Row `fd`: `0.75 / 0.38 / 0.25`. Row `random`: `0.00 (0/24)` all four.
- qwen ctx16384, row `isvd`: `niah_single 0.83 (20/24)`, `niah_multikey 0.79 (19/24)`, `niah_multivalue 0.21 (5/24)`. Row `nogist`: `0.96 / 1.00 / 0.42` (no marks). Row `fd niah_multikey`: `0.38 (9/24)`.
- llama ppl, col `delta (isvd - tracker)`: `nogist +0.0533` (TOST fails, Holm p 9.54e-07), `frozen -0.0023` (TOST passes), `fd -0.0741` (fails), `bf16 -0.0007` (passes), `full +0.1452`.
- qwen ppl, col `delta`: `nogist +0.0190` (TOST fails, Holm p 0.0435), `frozen -0.0137` (TOST fails, Holm p 0.0167), `fd -0.0939` (fails), `bf16 +0.0102` (passes).
- `VERDICT:` UNDECIDED; `members:` (C-blockers) llama isvd-vs-fd TOST fails d=-0.0741 p=1 · qwen isvd-vs-frozen TOST fails d=-0.0137 p=0.09 · qwen/niah_multikey isvd-vs-fd separates Holm p=0.027 · qwen isvd-vs-fd TOST fails d=-0.0939 p=1.
- bf16 block: `llama … PASS`, `qwen … PASS`; col `sbits bf16/isvd` = `0.5667` / `0.5382`. Rule is prereg/bf16_gist.md — rule not in the two files.

## 2. Rule compliance as I read it
- A/B (§4 rule 1+2): a family separates only if isvd beats BOTH frozen and nogist. isvd never beats nogist — retrieval nogist ≥ isvd everywhere (‡ on two llama tasks), perplexity `delta` isvd-vs-nogist is +0.0533 / +0.0190 (isvd worse). So 0 families separate → A/B unreachable. Agree with VERDICT.
- C (§4 rule 3): (i) blocked — `qwen/niah_multikey isvd-vs-fd separates (Holm p=0.027)`; (ii) blocked — fd TOSTs fail hard (d=-0.0741/-0.0939, p=1, decisively non-equivalent, not "not decidable") and qwen frozen TOST fails. Agree.
- Neither branch → UNDECIDED, blockers listed. Agree with the VERDICT line, member for member.

## 3. Three strongest objections
1. **"The tracker is load-bearing" is NOT writable** — it is the A/B claim and it was not reached; the table refutes it in the nogist direction (tier-only ≥ tracked gist on both axes, both families; ‡ on llama multikey/multivalue). Any paper sentence that the online-tracked gist improves retrieval, or the `paper/main.tex` "tracker load-bearing" line, is an overclaim a reviewer kills on this table alone.
2. **"The tracker is interchangeable / Branch C" is equally NOT writable** — C was blocked, verdict is UNDECIDED. A reviewer would attack any framing that rounds UNDECIDED up to a clean equivalence: non-separation is not equivalence (prereg §6), the qwen isvd-vs-frozen TOST fails (p=0.09, not decidably within ±0.02), and isvd *does* separate from fd on qwen multikey. The one place isvd wins a control is fd (secondary, byte-starved at ℓ=r by design) on 1 of 8 operative cells — not a headline, and irrelevant to A/B.
3. **bf16 `sbits 0.5667 / 0.5382` reads like a memory/compression claim and is not one** — in a KV-compression paper a sub-1 "sbits" column invites "the gist stores at 0.57×." It is a provenance pin (did the bf16 arm store at the specified width; bf16_gist.md §7c), not a footprint result, and needs a caption saying so or a reader will cite it as a storage win. The other flagged pin (a byte-match refusal's stored-bits ratio) did not fire here, so nothing is printed to misread; the table otherwise makes no memory/throughput/latency number (§11 forbids them).

## 4. What I would need before believing the branch
- The DECISIONS entry to record, beside UNDECIDED, the nogist-dominates-isvd finding (§4 asymmetry demands it) and to print the retrieval table beside every perplexity sentence.
- The nogist/isvd measured `sbits` ratio shown as within 1±0.05 (no byte-match refusal fired, but the ratio isn't printed — I am trusting its absence from `members:`).
- Confirmation the ‡ marks map to the primary-family members and that `vt` (excluded, m=12/18/4) is flagged descriptive wherever quoted.
- Scope discipline in prose: two families, one context (16K), n=24 — no generality/load-bearing claim without Stage 2 (Mistral + 32K).

## 5. One-line verdict
Neutral sentence that survives: "At 16K on Llama-3.1-8B and Qwen2.5-7B (n=24, `vt` excluded), swapping the incremental-SVD tracker for a frozen or random basis — or removing the gist entirely at matched stored bits — did not improve retrieval or perplexity; the tier-only no-gist arm matched or beat the tracked gist, and Gate 1 is UNDECIDED under its pre-registered rule." I agree with UNDECIDED; "load-bearing" is unwritable and "interchangeable" is not licensed either.
