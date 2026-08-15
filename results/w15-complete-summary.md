# Week-15 — the completed `bugSseed-r128-h1024-s32` row (both axes, both contexts)

Completes the Week-15 GPU confirm (`w15-confirm-summary.md`) by filling the last cells:
32K niah_single/multivalue, 16K niah_single, and the 16K matched-window perplexity block.
Pods 47758582 (main) + 47794970 (completion) + 47811756 (16K-ppl recovery), all destroyed.
Data: `results/w15-confirm-lines.txt` + `results/w15-complete-lines.txt`. Llama-3.1-8B.

## The complete arm

| ctx | mem | ppl (n=8 matched) | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| 16K | 0.191× | 5.434 | 100 | 100 | 100 | 100 |
| 32K | 0.159× | 7.353 | 100 | 100 | 100 | 100 |

**A clean 100 on all four retrieval tasks at both contexts, at 0.16–0.19× memory.**

## Perplexity ties (paired-window bits/token, s32 vs the field)
| context | vs think-c0.5 | vs palu-r0.5 (fixed) | vs full |
|---|---|---|---|
| 16K | +0.0056 → **TIE** | +0.0042 → **TIE** | +0.0438 → tie (below 0.05 floor; ~full-KV quality) |
| 32K | +0.0311 → **TIE** | +0.0240 → **TIE** | +0.0762 → DIFF (~5%) |

At 16K the arm is within 0.006 bits/tok of ThinK and 0.044 of full KV — near-lossless — at
**2.6–3.9× less memory**. At 32K it ties the field at **3.2–4.7× less memory**.

## The score-rank cap does all the retrieval lifting (matched A/B, only `-s32` differs)
| cell | task | uncapped | s32 |
|---|---|---|---|
| 32K r128 | multi-value | **0** | **100** |
| 32K r128 | var-track | **0** | **100** |
| 16K r128 | single | **0** | **100** |
| 16K r128 | multi-value | 25 | **100** |
| 16K r128 | var-track | 75 | **100** |

The uncapped r128 basis "fits" needles into low surprise so they are never selected into the
exact tier — visible even on single-needle at 16K (0/8). Capping the scoring basis to a leading
rank-32 subview un-blinds selection, at zero memory cost and ≤+0.3% ppl. (Honest oddity: uncapped
16K gets multi-key 100 but single 0 — task-specific; the s32 arm is uniformly clean.)

## Process note (honest)
The completion pod (47794970) was destroyed on a watcher *timeout* before its last ppl arm
finished, losing the two bug-arm 16K perplexities; a $0.4 recovery pod (47811756) re-measured
them (base ppls reproduced bit-identically, confirming deterministic windows). Retrieval and 32K
ppl were never at risk. Fix carried forward: watchers now record the terminal reason and it is
checked before any destroy.
