# Week-14 `wseed8` — pod sizing + staged launch (Phase-2 staging)

> Sizes the new `wseed8()` MODE in `scripts/pod/w11_r128.sh` (the n=8 firming A/B for the
> warm-up seed). **STAGING ONLY — nothing here is executed.** No pod is launched, no `vastai
> create`/`confirm` is run, no money is spent. Repo `/Users/hari/Desktop/kv-dlra`, branch
> `week7` (== `main`, `740053c`). Credit **$8.40** (`uvx vastai show user --raw` → `credit`),
> escalate < **$3**. Date 2026-08-13. `bash -n scripts/pod/w11_r128.sh` passes.

## What `wseed8` runs (the core MODE)

Matched A/B — `bugS` (`bugslash`, no seed) vs `bugSseed` (`--warmup-seed`) — same
pod/trials/seeds/footprint, **n=8/cell**, RULER only. Perplexity is **not** re-run: Week-13
already measured no-regression (r32 −0.38% @16K / −0.79% @32K; r128@32K −0.39%; `tok_eq/layer`
identical). Ordered **cheap-decisive-first** so the headline lands even if the pod dies.

| # | ctx | rank·hh | ratio | tasks | arms | n | out-json (bugS / bugSseed) |
|---|---|---|---|---|---|---|---|
| C1 | 16K | r32·h256 | 0.05× | single, multikey, multivalue, vt | bugS, bugSseed | 8 | `results/w14-wseed8-bugS-r32-16k.json` / `…-bugSseed-r32-16k.json` |
| C2 | 32K | r32·h256 | 0.04× | multikey, multivalue, vt | bugS, bugSseed | 8 | `…-bugS-r32-32k.json` / `…-bugSseed-r32-32k.json` |
| C3 | 32K | r128·h1024 | 0.16× | multikey, multivalue, vt | bugS, bugSseed | 8 | `…-bugS-r128-32k.json` / `…-bugSseed-r128-32k.json` |

- **C1** firms the headline: Week-13 n=2×2 had plain `bugS` at **0/0/0** on the hard tasks and
  `bugSseed` at **100/100/100** at the same 0.05× memory. All 4 tasks kept (niah_single is the
  saturated 100→100 control — proves the seed does not *break* the one task `bugS` already got).
- **C2** firms 32K multikey **50→100** and resolves the **vt 100→75** one-flipped-trial dip
  (real regression or noise?); multivalue was flat.
- **C3** is the cell **Week-13 deferred** (cost) — new 32K r128 RULER coverage. Hard tasks only;
  niah_single is omitted at 32K to bound the expensive cell (it saturates for the `bugS` family;
  the noisier 16K r128 single −25 is a *16K* question the plan flags as optional, and 32K r128 is
  not a cheap place to chase it).

## n=8 spelling — **`--n-trials 4 --seeds 0 1`** (chosen; documented)

Both `--n-trials 4 --seeds 0 1` and `--n-trials 2 --seeds 0 1 2 3` give 8 samples/cell. We use
the **former**. Why it is the right one, not a coin-flip:

1. **8 genuinely distinct samples either way.** RULER seeds its RNG with `seed*131 + trial`
   (`scripts/w10_ruler.py:94`). `4 trials × 2 seeds` → seeds `{0,1,2,3, 131,132,133,134}` — all
   distinct, no collision (131 > 4).
2. **`trial` also drives task structure — the axis the warm-up window acts on.** multikey queries
   `qi = trial % n_keys` (`n_keys=8`), so trials `{0,1,2,3}` query **4 distinct keys at 4 different
   context depths**; niah_single needle position is `n//2 + trial%5` → **4 distinct positions**;
   multivalue label and vt var-names are also keyed on `trial`. `--n-trials 2 --seeds 0 1 2 3`
   would exercise only trials `{0,1}` → the **2 frontmost keys** and 2 positions, biasing the
   accuracy estimate toward the easy/front needles and giving a *less representative* read of a
   mechanism that is fundamentally about needle **depth**. `--n-trials 4` doubles the depth
   coverage while still varying the background across 2 seeds.
3. **Strict superset of the Week-13 n=2×2.** The Week-13 `wseed` used `--n-trials 2 --seeds 0 1`.
   `--n-trials 4 --seeds 0 1` keeps trials `{0,1}×`seeds`{0,1}` (identical `(seed,trial)` → identical
   haystacks) and *adds* trials `{2,3}`. So the n=8 **re-runs and extends** the original n=4 rather
   than replacing it — the firming is monotone.
4. Matches the plan's explicit suggestion (`docs/week14-plan.md:94`). n=8 halves the granularity
   from 25 pts to **12.5 pts** (error bar ≈ ±1/8).

## Per-cell minutes, total, and $ (from the measured cost table)

Measured A100 per-trial table (`docs/week11-session-handover.md:87-89`, carried into
`docs/week14-plan.md:142-144`): **r32@16K ~2 · r32@32K ~6.5 · r128@32K ~10–15 min/trial**.

**Interpretation (assumption, stated):** "min/trial" = wall-clock for **one `(seed×trial)` RULER
sample of one arm**. An A/B **cell** therefore runs `2 arms × 8 samples = 16` arm-samples, so
`cell-min = min/trial × 16`. This interpretation is **anchored** to the plan's own unit — it
reproduces "a 4-trial cell ≈ $0.6–0.8" and "W-1 r32 both ctx n=8 ≈ $1.2–1.8" almost exactly (see
cross-check below), so the sizing is internally consistent, not a fresh guess.

| Cell | min/trial | ×16 (n=8, 2 arms) | cell minutes |
|---|---|---|---|
| C1 r32@16K | ~2 | 16 | **32** |
| C2 r32@32K | ~6.5 | 16 | **104** |
| C3 r128@32K | ~10–15 | 16 | **160–240**  ← swing |
| **compute subtotal** | | | **296–376 min (4.9–6.3 h)** |
| + pod overhead (boot + clone + pip + 8B download) | | | ~20 min |
| **total pod wall** | | | **≈ 5.3–6.6 h** |

**$ estimate — A100 40GB @ $0.5–0.8/hr (show low/high):**

- **Low:** 5.3 h × $0.5 ≈ **$2.6**
- **High:** 6.6 h × $0.8 ≈ **$5.3**
- **$ RANGE ≈ $2.6 – $5.3** — best working estimate **$3–4.5** (the $5.3 high stacks the r128
  15-min/trial ceiling **and** $0.8/hr **and** full overhead simultaneously).

**Cross-check vs the plan's own decomposition** (`docs/week14-plan.md:158`): C1+C2 (r32 both ctx,
n=8) computes to **$1.13–1.81** vs the plan's **$1.2–1.8** ✓; C3 (r128@32K n=8) computes to
**$1.3–3.2** vs the plan's implied **~$2–3** (it lists r128@32K **n=4** ≈ $1–1.5). Agreement is
tight, so the interpretation above is the right one.

## Escalation gate

- **Floor: credit < $3 → STOP and escalate** (do not grind). Credit is **$8.40**; even the **high**
  estimate ($5.3) leaves **≈ $3.1**, just above the floor. So the core fits, but with little slack.
- **C3 (r128@32K) is the whole swing** (160–240 of the 296–376 compute-min). If C1/C2 land and
  credit is tight, **drop C3 to n=4** (`--n-trials 2 --seeds 0 1`) → C3 halves to **80–120 min**,
  well under budget. This matches the plan's W-2 rule ("r128@32K **n=4 minimum, n=8 if budget**").
  The core MODE is written at n=8; the n=4 fallback is a one-word edit at launch, not a code change.
- **Any r128@32K OOM on the 40GB card, or a result contradicting a measured wall → escalate**, per
  the standing rules. Keys still unrotated — **flag, don't rotate mid-run**.
- **64K (W-4 stretch) is OUT of `wseed8`** — it needs an **80GB** card (48GB OOMs at 64K) **and a
  top-up** (~$2–3 on its own at n=2×2); do not dip below $3 for it. Defer to its own pod.

## STAGED launch command — **written out, NOT executed**

Faithful to the Week-13 infra recipe (`docs/week13-session-handover.md:57-64`). Run **only** when
the lead approves a real spend. The `--onstart` uploads the **local** `w11_r128.sh` (which now
carries `wseed8`); the pod git-clones `origin/week7` for the Python (`w10_ruler.py` already supports
`--warmup-seed` since Week-13, so **no new Python needs pushing**). `MODE` defaults to `new16`, so
the `sed` to `wseed8` is **required** or the pod runs the wrong MODE.

```bash
# --- STAGING: DO NOT RUN. No pod, no create, no confirm, no spend. ---
cd /Users/hari/Desktop/kv-dlra

# (0) house rule: don't commit. The onstart is uploaded from the LOCAL file, so wseed8
#     runs without a commit. If you prefer origin to match, the OPERATOR pushes week7
#     first (flagged — not done here).

# (1) select the MODE in the LOCAL onstart file (macOS/BSD sed; default is new16)
sed -i '' 's/^MODE="${MODE:-new16}"/MODE="${MODE:-wseed8}"/' scripts/pod/w11_r128.sh
grep -n '^MODE=' scripts/pod/w11_r128.sh            # verify it now reads wseed8

# (2) pick the cheapest reliable A100 40GB offer
OFFER=$(uvx vastai search offers 'num_gpus=1 gpu_name in [A100_PCIE,A100_SXM4] dph<0.8 reliability>0.99 rentable=true cuda_vers>=12.4 disk_space>=60 inet_down>=800' -o dph --raw | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")

# (3) CREATE — this is the paid step. STAGED / NOT EXECUTED here.
uvx vastai create instance $OFFER \
  --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime \
  --disk 80 \
  --onstart scripts/pod/w11_r128.sh \
  --label kvdlra-w14-wseed8

# (4) validate: logs must show ===DEPS_DONE===, torch 2.11.0+cu128 cuda True, ===MODEL_OK===
#     poll (bounded), harvest PRINTED ROW LINES ONLY (vastai logs truncates at 500 chars):
#   uvx vastai logs <id> --tail 20000 | grep -E '^\[(niah|vt)' > results/w14-wseed8-ruler-lines.txt
# (5) DESTROY the pod when done (non-negotiable):
#   printf 'y\n' | uvx vastai destroy instance <id>
```

Reset the local edit afterward if not committing: `sed -i '' 's/^MODE="${MODE:-wseed8}"/MODE="${MODE:-new16}"/' scripts/pod/w11_r128.sh`.

## Optional rider — `bugSQseed` (Q-BUG subsumption). **NOT in the core MODE.**

Tests whether stacking Q-BUG on the seed buys anything beyond the seed alone (Week-13: `bugSseed`
ppl already *equals* `bugSQ`, so the prior is **subsumed**). Kept out of `wseed8` because it is
**+cost/+complexity** — it needs a **CALIB step on-pod first**.

- **Actual arm name is `bugSQseed`, not "bugSseedQ".** `build_arms` sets the prefix `bugS` →
  `bugSQ` when `--qwhiten-file` is present, then appends `seed` for `--warmup-seed`
  (`scripts/w10_frontier.py:214-220`). Harvest grep must look for `bugSQseed`.
- **Guard-legal:** `bug_cache.py:452` rejects the seed only with coded/quant/merge; `w_key`
  (Q-BUG whitening) is not in that set (and `w_key`×CodeBUG is the only `w_key` exclusion, `:445`).
- **How to add it** (one 32K r32 cell; reuses `wseed8`'s existing `bugSseed-r32-32k` as the A/B
  baseline — the rider is one *extra* arm):

  ```bash
  # CALIB first (as in the qbug() MODE) — regenerates the frozen per-layer key-whitening diagonal:
  PYTHONPATH=src python -u scripts/w12_calibrate_qkey.py --model "$MODEL" \
    --n-docs 8 --seq-len 4096 --device cuda --out results/w12-wkey-8b.pt
  # then the one extra arm (n=8, same spelling):
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens 32768 --tasks niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 32 --hh-budgets 256 --hh-neighbor 1 --warmup-seed \
    --qwhiten-file results/w12-wkey-8b.pt --chunk "$CHUNK" --n-trials 4 --seeds 0 1 \
    --out-json results/w14-wseed8-bugSQseed-r32-32k.json
  ```

- **Cost:** CALIB ≈ a few min on-pod + one r32@32K n=8 arm ≈ **8 × 6.5 ≈ 52 min** (~$0.4–0.7). A
  commented sketch of exactly this sits next to `wseed8()` in `scripts/pod/w11_r128.sh`.

## Assumptions / flags

- **"min/trial" = per single `(seed×trial)` sample, one arm** (so `×16` per n=8 A/B cell). This is
  the reading that reproduces the plan's own $ anchors (above). If the measured figure instead
  already spanned the n=4 cell or both arms, the true cost is **lower** (down to ~½) — i.e. the
  $2.6–5.3 range is **conservative (upper-leaning)**, not optimistic.
- The **16K r32** table value (~2 min/trial) is applied to a **4-task** block (incl. niah_single);
  single adds marginal time but stays within the ~2-min envelope. C1 is the cheapest cell either
  way, so this does not move the total.
- No new Python is required on `origin/week7` (wseed8 uses only existing flags). No commit is made
  (house rule); the operator pushing `week7` before a real launch is **flagged, not performed**.
