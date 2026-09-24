# R1 — Pre-registration compliance and statistics (blind)

## 1. Numbers I read
Header comment: `primary retrieval m=12, secondary retrieval m=18, primary perplexity m=4`.
Retrieval, llama ctx16384: isvd/niah_multikey `0.42 [0.24,0.61] (10/24)`, nogist/niah_multikey `0.92 ... (22/24) ‡`, isvd/niah_multivalue `0.25 (6/24)`, nogist/niah_multivalue `0.88 (21/24) ‡`, nogist/niah_single `1.00 (24/24)` (unmarked).
Retrieval, qwen ctx16384: isvd/niah_multikey `0.79 (19/24)`, fd/niah_multikey `0.38 (9/24)`, nogist/niah_multikey `1.00 (24/24)` (unmarked), isvd/niah_multivalue `0.21 (5/24)`, nogist/niah_multivalue `0.42 (10/24)`.
Perplexity primary-family Holm p / paired t p: llama nogist `9.54e-07 / 2.38e-07`, llama frozen `0.376 / 0.376`, qwen nogist `0.0435 / 0.0217`, qwen frozen `0.0167 / 0.00558`.
Perplexity delta / CI / TOST used by verdict: llama fd `-0.0741 [-0.0912,-0.0570] fails`; qwen frozen `-0.0137 [-0.0224,-0.0047] fails`; qwen fd `-0.0939 [-0.1098,-0.0772] fails`; llama frozen `-0.0023 [-0.0071,+0.0026] passes`.
members: `qwen/niah_multikey: isvd vs fd separates (Holm p=0.027)`; the three TOST blockers above.

## 2. Rule compliance as I read it
Family sizes: header 12/18/4 == the amended sizes A1b.4 fixes (vt excluded → primary 16−4=12, secondary 24−6=18, perplexity untouched=4). Realised m is the pre-registered-as-amended one. AGREE.
Statistics all map to §4/§6: acc+Wilson (descriptive §7a), McNemar marks `*`/`‡` (primary Holm), delta=isvd−tracker paired, paired-bootstrap CI, paired-t as Holm member, ±0.02 TOST. No statistic appears that §4/§6 does not name.
Holm recompute (primary perplexity, m=4, fully determined from printed paired-t): 2.38e-07×4=9.52e-07 (=9.54e-07 ✓, smallest-member exact multiplier holds), 0.00558×3=0.0167 ✓, 0.0217×2=0.0434≈0.0435 ✓, 0.376×1=0.376 ✓; monotone; every value satisfies raw ≤ adj ≤ min(1,4·raw). AGREE.
TOST recompute (qwen isvd-vs-frozen): s≈0.0255<0.0667 decidable; half-width 0.00766 > 0.02−0.0137=0.0063 → fails; printed "does not pass, p=0.09". AGREE. (Cross-check llama frozen: s≈0.014, hw≈0.0042<0.0177 → passes ✓.)
VERDICT rule: A/B needs BOTH families separated (isvd beats frozen AND nogist). isvd beats nogist on NO axis in either family (retrieval: `‡` shows nogist beats isvd on llama; perplexity delta>0 vs nogist both families), so neither family separates → A/B unreachable. C needs (i) no fd/frozen separation any task + (ii) all 4 fd/frozen TOSTs pass; blocked by qwen/niah_multikey fd separation and by 3 failing TOSTs (llama fd, qwen frozen, qwen fd). Neither branch holds, no error/refusal row present → UNDECIDED. AGREE. Correctly EXCLUDES nogist from the C-blockers (plan asymmetry, §4).

## 3. Three strongest objections
(1) The `members:` list omits per-member pair counts. §4/A1a.2 has members carry `(a-b pairs, Holm p=…)`; the one retrieval C-blocker prints `Holm p=0.027` with NO discordance. From marginals (isvd 19, fd 9) raw McNemar ∈ [0.00195 (10,0) … 0.0414 (15,5)]; 0.027 requires the unshown pairing to have raw ≤ 0.027 (overlap ≥6). I can bound (0.027 ≤ 18·raw holds for all; raw ≤ 0.027 only asserted) but cannot verify. The retrieval Holm families (m=12, m=18) are not independently recomputable from the table.
(2) The refusal preconditions §4 reads BEFORE the rule are not surfaced. The nogist/isvd stored-bits ratio (must be 1±0.05, §7f) is printed nowhere; the frozen post-freeze `fixed_k/v` diag check (§7c) is not shown; no "96 keys; 0 disagree", "0 dropped", "0 errors" confirmation appears. UNDECIDED (not REFUSED) implies they passed inside gate1_verdict, but a blind reader cannot see the byte-match that makes the nogist contrast a mechanism contrast.
(3) vt is out-of-family but the retrieval block gives it a normal column with no "excluded/descriptive" tag; only the header m-values reveal EXCLUDED_TASKS={vt} was applied. Correct here (no vt cell is marked, so it blocks nothing), but the exclusion is inferred, not displayed.

## 4. What I would need before believing the branch
- Raw McNemar (a,b) for every marked/blocking retrieval cell, and a full members dump with each adjusted p, so Holm at m=12/m=18 is recomputed not bounded — especially qwen/niah_multikey fd (confirm overlap ≥6 so raw ≤ 0.027).
- The nogist/isvd sbits ratio within 1±0.05, the frozen dispatch diag reading, and the pairing line (96 keys, 0 disagree, 0 dropped, 0 errors) printed on the table.
- Positive confirmation EXCLUDED_TASKS={vt} fired (header m matches, which is strong indirect evidence).

## 5. One-line verdict
The printed statistics, the fully-checkable m=4 Holm family, and both hand-recomputes agree with the table, and UNDECIDED follows from the members and is robust (C stays blocked by the 3 TOSTs even if the one unverifiable retrieval separation were dropped); my only real reservations are transparency — unshown McNemar discordance and unsurfaced byte-match/pairing/refusal checks — not a rule violation.
