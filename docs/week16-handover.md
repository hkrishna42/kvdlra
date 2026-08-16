# Week-16 session handover

**Branch** `week7` · **status** GPU program executed, all pods destroyed, credit **$97.2** (spent ~$19.7).
**Dashboard** (memory × ppl × retrieval per model): artifact `757d6777`.
**Supersedes** the Week-16 plan/foundation state; see also [[kvdlra-week16-plan]] memory.

---

## 1. What this week did

The Verdict (NeurIPS-readiness) approved **Tier 1 (firm Llama) + Tier 2 (generality: Mistral + Qwen)**.
Both ran. A significant science finding emerged and reshaped the Tier-2 story.

### $0 CPU foundation (committed + pushed, full green)
| commit | what |
|---|---|
| `7b47053` | template-derived RULER query tail (`w10_ruler._templated`) — floor=48 keeps Llama bit-identical, fixes the Mistral/Qwen mis-slice; fail-loud tripwire + `tests/test_ruler_template_tail.py` |
| `3b94eeb` | `tests/test_bug_cache_families.py` — `BugStreamingCache` exact-mode == `DynamicCache` on tiny Qwen2 (7:1 GQA + qkv bias) + Mistral → **"no structural blocker" confirmed at $0** |
| `464b3d7` | `scripts/w16_tier2_probe.py` — 5-gate CPU probe (G-GATE/ROPE/BIAS/TAIL/SMOKE), FUND/KILL per family |
| `484e131` | `scripts/w16_storage.py` — Tier-4 measured-storage reframe (integrity exact; workspace ≈ 0.98× full → why naive VRAM backfires) |
| `d452d27`, `8b947e3`, `f073b82` | `scripts/pod/w16.sh` — onstart-batch pod driver, MODEs `tier2` / `tier1` / `sweep` |

### GPU results

**Tier-1 (Llama-3.1-8B) — WIN.** `bugSseed-r128-h1024-s32` @0.16× memory:
- 32K **var-track**: `-s32` lift **0.00 → 1.00**, beating **both** `palu` (0.25) **and** `think` (0.50).
- 32K **multivalue**: `-s32` 1.00, ties palu/think.
- The `think-c0.5` vt/mv baselines were **never measured before** — now they are. Honesty claim
  strengthens from "beats Palu, ties ThinK" to **"beats Palu AND ThinK on var-track"** (n=4; Wilson pending).

**Tier-2 (generality) — the r128 config does NOT transfer, but BUG's extreme-compression niche DOES.**
- `bugSseed-r128` collapses on Qwen + Mistral (retrieval 0.00; `-r128-h1024` ppl 467 / 47). **Geometry and
  QKV-bias are ruled out** — Mistral has Llama's exact geometry (n=1024) and no bias, yet still fails.
- **Rank sweep** (approved pivot) found the truth: at **r16–r64 (0.04–0.15× memory) retrieval is perfect**
  on both new families (single + multivalue = 1.00; Qwen var-track = 1.00; **Mistral var-track weak,
  0.25–0.50**), with **healthy perplexity**. **Sweet spot = r64.**
- **Two distinct r128 failure modes:** **Qwen** stays fluent (ppl 7.81) but the gist grows smooth enough to
  **absorb the needle** → retrieval 0 (the rank-vs-retrieval wall — at r128 here vs r256 on Llama; `-s32`
  does *not* rescue it). **Mistral** the **streaming integrator diverges** (deployed-config ppl 43.9;
  pure-`bug` gist diverges at r256 on Qwen, r128 on Mistral).
- BUG does **not** beat baselines on perplexity; its edge is **memory at matched retrieval** — 3–8× less
  (0.05–0.15× vs ThinK/Palu 0.50–0.75×), the regime eviction/channel-pruning cannot enter.

---

## 2. Data (deployed config `bugSseed-rK-h256`, 16K, n=4)

| model | full ppl | r16 mem/ppl | r32 | r64 (sweet) | r128 | retrieval @r64 (s/mv/vt) |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B** (n=512) | 6.22 | 0.053 / 8.15 | 0.085 / 7.91 | **0.148 / 8.18** | 0.275 / 7.81 (needle absorbed) | **1.00 / 1.00 / 1.00** |
| **Mistral-7B-v0.3** (n=1024) | 4.87 | 0.036 / 5.89 | 0.052 / 5.95 | **0.085 / 5.50** | 0.150 / 43.9 (diverges) | **1.00 / 1.00 / 0.50** |
| **Llama-3.1-8B** (n=1024) | — | — | (r32-h256: mv 1.00, vt 0.88 from W15) | *measuring this session* | r128-s32: 1.00/1.00/1.00 @0.16× | *measuring* |

Baselines @16K — Qwen: `full` 6.22, `think-c0.5` 6.42 @0.75× (mv 0.75), `palu-r0.5` 6.38 @0.50× (mv 0.88).
Mistral: `full` 4.87, `think` 4.93 @0.75× (vt 0.88), `palu` 5.09 @0.50× (vt 0.38).

Raw sweep logs preserved in this session's scratchpad (`w16-{qwen,mistral}-sweep-raw.log`); pod logs are gone.

---

## 3. Repo / infra state

- **Pods:** all destroyed. Credit **$97.2**. Keys unrotated (flag only).
- **Pod driver:** `scripts/pod/w16.sh` — `MODE=sweep` runs RULER (RRANKS, fast) then ppl (PRANKS, slow,
  n-samples 4). Launch via onstart-batch, `export MODE/MODEL/TAG/RRANKS/PRANKS` prepended to the script.
  Both 7B families are **ungated** (canonical HF ids, no mirror). **Always pass `--chunk`** to `w10_ruler.py`
  (the `RULER()` helper now does — a dropped `--chunk` was the bug that voided the first pod round).
- **Log truncation gotcha:** `vastai logs` returns a bounded tail; harvest result lines from the live
  monitor captures, not a single final pull.

---

## 4. How to continue (loose ends → Week-17)

1. **Consolidate r64-seed across all three** (Llama r64 is being measured this session; drop it into the
   dashboard `757d6777`).
2. **Firm with error bars:** merge Tier-1/2 + sweep line-files into `results/w11-decision-table.json` via
   `scripts/w11_merge.py`, then `scripts/w15_intervals.py` for Wilson CIs. Re-run r64-seed at **n≥8** and
   **32K** for the headline generality claim.
3. **Fix Mistral var-track** (the one weak axis, 0.25–0.50 even at low rank).
4. **Investigate the integrator divergence** at high rank on Mistral (r128) — can conditioning/orthogonalisation
   extend the usable rank? And the **`h1024` vs `h256` r128 ppl puzzle** on Qwen (467 vs 7.81 — the big exact
   tier destabilises).
5. **Docs/paper:** `docs/week16-explained.md` (written), a paper section on the extreme-compression frontier.

The Week-17 kickoff prompt (multi-agent) is `docs/week17-kickoff.md`.
