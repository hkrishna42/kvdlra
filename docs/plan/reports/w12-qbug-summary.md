# Q-BUG (bugS-ppl Track 1) — 8B GPU confirm summary

**One line:** query-metric key whitening (`w_key`) gives a **real but small** perplexity
gain (0.4–1.2%) with retrieval preserved within trial-noise — an honest **bounded**
result. Both aggressive pre-registered ppl bars **missed**. `w_key` ships as a
**default-off** knob.

Model Llama-3.1-8B-Instruct, chunked ingest (chunk 4096), ppl window 512 / 2 samples,
RULER n = 2 trials × 2 seeds. Pod `45859737` (A100 40GB), MODE `qbug`, no OOM/errors.
Data (exact printed rows): `results/w12-qbug-ppl-lines.txt`,
`results/w12-qbug-ruler-lines.txt`.

## Perplexity (the target axis)

| config | mem (ratio@32K) | ppl@16K | ppl@32K | bar@32K | verdict |
|---|---|---|---|---|---|
| bugS-r32-h256 (baseline) | 0.043× | 4.477 | 9.164 | — | — |
| **bugSQ-r32-h256** | 0.043× | **4.425** | **9.092** | ≤ 8.90 | **miss** (−0.072, −0.79%) |
| bugS-r128-h1024 (baseline) | 0.159× | 4.156 | 8.117 | — | — |
| **bugSQ-r128-h1024** | 0.159× | **4.124** | **8.085** | < 8.00 | **miss** (−0.032, −0.39%) |

Gains: r32 −0.79%@32K / −1.16%@16K; r128 −0.39%@32K / −0.77%@16K. Both consistent, both
small. To hit the bars we needed −0.264 (r32) and −0.117 (r128) @32K; we got −0.072 and
−0.032 — roughly **3–4× short**.

## Retrieval gate (bugSQ RULER @32K vs pooled bugS baseline)

| task | bugSQ-r32 (n=4) | bugS-r32 base | bugSQ-r128 (n=4) | bugS-r128 base | reading |
|---|---|---|---|---|---|
| multi-key | 0.50 | 0.67 | 0.50 | 0.75 | soft ~1 trial (both ranks) |
| multi-value | 1.00 | 1.00 | 1.00 | 1.00 | **preserved exactly** |
| var-track | 0.75 | 1.00 | 0.75 | 0.75 | exact @r128, soft 1 trial @r32 |

Every bugSQ cell is within **±1 trial** (n=4 → 25 pts/trial) of the pooled bugS baseline.
Multi-value is preserved exactly at both ranks; multi-key is softened by ~1 trial at both.
The tier/gist separation holds — whitening the gist did **not** break the retrieval the
visible exact tier carries. **Caveat:** this is bugSQ (n=4, this pod) vs bugS (pooled,
prior pods), **not a matched-n same-pod equivalence test**; the ~1-trial multi-key softening
could be noise or a small real cost — Week-13 Track-C's n=8 re-run is designed to separate them.

## Accounting (honest float-equivalent)

Diagonal `w_key` is `(32 layers, 1024 features)`, range [0.158, 9.48]. It costs
**+0.5 tok_eq/layer** (bugS-r32 1370.7 → bugSQ 1371.2 @32K); the 32K ratio is unchanged at
three decimals (0.043 / 0.159). Negligible memory, as designed.

## The lesson worth carrying (proxy-vs-downstream gap)

The Week-12 CPU attention-error probe predicted a ~30–40% attention-output error reduction,
which mapped to an expected large ppl gain — the realized ppl gain is 0.4–1.2%, i.e. the CPU
proxy **over-predicted end-to-end ppl improvement ~30–40×**. Attention-output error at a
fixed layer is a loose upper bound on next-token ppl (softmax saturation + downstream layers
absorb much of the perturbation). **Future $0 probes must state that they measure a proxy,
and any ppl bar derived from a proxy is a ceiling, not a promise.** (Directly informs the
Week-13 probe-gate design.)

## Calibration caveat → Week-13 Track-C

The 8B L was calibrated on **very short** docs (T ∈ {44, 49, 117, 122, 160, 194, 237, 380},
avg ~160 tokens). Q = E[qqᵀ] estimated on short contexts may not reflect the query
distribution at 32K. Track-C re-calibrates on long docs
(`w12_calibrate_qkey.py --seq-len 8192 --n-docs 16`) to test whether a cleaner L moves the
gain and/or removes the multi-key softening.

## Status
`w_key` is merged (commits eab4792 / 3164ca7 / 0e456ca) as a default-off knob with unit +
accounting pins (`tests/test_bug_cache_qbug.py`). Q-BUG is a **funded-but-bounded** lever;
composable with Week-13 Track-A (integrator surgery). Not a headline win.
