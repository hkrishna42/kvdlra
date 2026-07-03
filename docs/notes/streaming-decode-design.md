# Streaming-decode BUG cache — design note (Week 5, Axis B)

> Status: design, pre-implementation. Plan reference: `docs/week5-plan.md`
> §"Axis B" and §"New capability to build". This is the first time the project
> uses BUG *as a streaming integrator during generation* — its core theoretical
> selling point. Everything before this compressed the prefill and was static
> during decode.

## 1. The question being asked (falsifiable)

Long autoregressive generation under a **constant-memory** KV cache: does a
rank-`r` *running low-rank summary* of the context (BUG/DLRA) preserve
generation quality better than a *constant-size set of retained whole tokens*
(MorphKV / SnapKV-style decode eviction / StreamingLLM) at **matched memory**?
BUG's theoretical hook: its per-token update carries the DLRA robustness bound
independent of `σ_min`, so it should degrade *gracefully* as the topic drifts
and the tracked spectrum reorders during a very long generation — where
eviction must keep making irreversible discard decisions. A clean negative is
a fine result (the Week-4/5 record shows we report those straight).

## 2. What "constant memory" can honestly mean

Softmax attention needs *per-token* key/value information for every token it
can attend to. No method can attend over unbounded history in O(1) memory;
"constant-memory" methods bound the *set* attention sees. Eviction bounds it
by keeping `m` whole tokens (n features each). The BUG cache bounds it by
keeping per-token **coordinates** in the tracked rank-`r` subspace (`r` floats
per token instead of `n`), plus the shared basis. At matched memory BUG
therefore retains a ~`n/r`× **longer** (but only approximately represented)
history. That trade — *more history at lower per-token fidelity vs. less
history at exact fidelity* — is precisely the experiment.

## 3. Architecture decision: custom `Cache` subclass (not a hook)

Chosen: **option (a)** — a transformers-5.8 `Cache`/`CacheLayerMixin`
subclass (`BugStreamingCache`), modeled on `DynamicSlidingWindowLayer`
(which already demonstrates the pattern: store less than attention sees,
report `get_mask_sizes` consistently, return full K/V during prefill).

Why not a decode forward-hook (option b): kvpress's `DecodingPress` hook
pattern rewrites `cache_layer.keys/values` in place, so the stored tensors
would keep their full shape between compressions — memory would be *nominal*,
like prefill `BUGPress`. The deliverable explicitly includes *measuring*
genuinely constant memory; only a cache that stores `(U, B, coords)` and
materializes the reconstruction transiently proves that. Facts about the 5.8
decode path that make the cache subclass workable (verified in source):

- `LlamaAttention.forward` calls `past_key_values.update(key_states,
  value_states, self.layer_idx)` with **no kwargs** — incoming keys are
  **post-RoPE**, and the cache gets no cos/sin. The cache must compute
  rotations itself; positions are deterministic (`cumulative_length`), and we
  hold a reference to the model's own `rotary_emb` so cos/sin are bit-identical
  to what attention used (Llama-3 rope scaling included).
- Masks are built from `cache.get_mask_sizes(q_len, layer_idx)` *before* the
  layer runs; the reported `kv_length` must equal the length `update()` will
  return this step. `get_seq_length()` must keep returning the *cumulative*
  token count so `generate()` keeps advancing true positions.

## 4. Per-layer state (all bounded, batch = 1 enforced initially)

| piece | shape | contents |
|---|---|---|
| sinks | `2 × n × n_sink` | first `n_sink` K/V verbatim (K post-RoPE, bit-exact) |
| recent ring | `2 × n × [w, w+b)` | last tokens' K/V verbatim (K post-RoPE, bit-exact) |
| bases | `U_k, U_v: n × r` | BUG-tracked orthonormal feature subspaces (K pre-RoPE) |
| coords | `C_k, C_v: r × W` | retained middle tokens' coordinates in the *current* basis |
| cores | `B_k, B_v: r × r` | square-root second-moment cores (steer the basis only) |
| positions | ints | middle tokens are a contiguous position range → store start+len |

`n = head_dim · num_kv_heads` (512 at 1B, 1024 at 8B). Middle-token memory is
capped at `W` coordinate columns; when full, the **oldest coords are dropped**
(that is the honest bound — see §2). `B` keeps the accumulated-stream
semantics of the validated `StreamingBUG` (it steers the basis; attention
never sees it). An exponential forgetting factor on `B` is a documented knob,
default off.

## 5. The per-token cycle (decode step, token at true position t)

1. `update()` receives post-RoPE `k_t`, `v_t` → push verbatim into the recent
   ring (bit-exact for the tokens that matter most locally).
2. When the ring exceeds `w + b − 1`: **absorb** the oldest `b` tokens into
   the low-rank stream (one *blocked* augmented BUG step — the validated
   `blocked_bug_subspace` math, stateful):
   - un-rotate those keys to pre-RoPE (exact inverse rotation, cos/−sin, using
     the model's own rotary embedding at their true positions);
   - BUG step on `(U_k, B_k)` with the `n × b` block (fp32 core, per PLAN §8
     pitfall #4); same for values (no RoPE);
   - **rotate existing coords into the new basis**: `C ← (U_newᵀ U_old) C`
     (an `r×r` × `r×W` matmul). Each truncation projects old tokens onto the
     new subspace — the graceful-degradation mechanism the DLRA bound governs;
   - append the `b` new coordinate columns; drop oldest columns beyond `W`.
3. Return to attention: `[sinks ‖ RoPE(U_k C_k, true positions) ‖ recent]`
   for K (values analogous, no rotation). The reconstruction `U_k C_k` and its
   rotation are **cached between absorb events** (they only change every `b`
   steps), so the steady-state per-step cost is a concat + attention over a
   constant-length cache.

Amortized per-token update cost: `O(n·r + (r+b)³/b + r²·W/b)` — bounded,
independent of generated length; per-absorb SVDs are `(r+b)×(r+b)` (tiny —
should sidestep the cusolver stall that hit the *prefill* blocked path; CPU-SVD
fallback knob kept anyway).

Prefill (`q_len > 1`, empty cache): return the full K/V (standard full
prefill attention — same protocol as all baselines), then compress into the
state above: sinks + last `w` tokens exact, middle via one blocked BUG sweep,
keep the last ≤`W` coords. Single-shot prefill only (chunked prefill raises,
as in `BUGPress`).

Degenerate mode `r = 0` / `W = 0` (no middle) = **StreamingLLM** (sinks +
recent window) — one implementation doubles as that baseline.

## 6. Correctness proofs required before any benchmark

1. **Byte-exact parity**: with `rank ≥ n` and `W`, `w` large enough that
   nothing is ever truncated or dropped, generation must be *byte-identical*
   to `DynamicCache` (sinks/recent are stored verbatim, so this proves the
   mask/position/plumbing layer exactly; the RoPE round-trip is exercised by
   the middle path at high rank → near-exact, checked to fp tolerance).
2. **Mask consistency**: reported `get_mask_sizes` == returned length at every
   step, across absorb boundaries (property test over many steps).
3. **Constant memory**: stored floats flat in generated length (measured).
4. **Coordinate-rotation invariant**: after any number of basis updates, the
   implied reconstruction of a retained token equals the direct projection of
   its original pre-RoPE key onto the current basis (`U_t Π_j P_j k_s` — the
   projected-history semantics), within fp tolerance.
5. **No gibberish** at aggressive rank on long prompts (behavioral).

## 7. Baselines and matched-memory accounting

- **Full cache** (upper bound, O(T)); **StreamingLLM** (our degenerate mode);
  **SnapKV-decode / TOVA** via kvpress `DecodingPress` (periodic re-compression
  to `target_size` — needs a small transformers-5.8 compat patch, it keys off
  `kwargs["cache_position"]` like the old `BasePress.forward_hook` did);
  **MorphKV** (ICML'25): not in kvpress 0.5.1 → faithful-core reimplementation
  (constant-size cache, recent-window attention scoring for retention,
  per-step or small-interval eviction), deviations documented.
- **Memory accounting** (token-equivalents per layer, one full token = `2n`
  fp16 numbers): BUG = `n_sink + (w+b) + r + r·W/n + r²/n` vs eviction = `m`.
  Match `m` to BUG's total. Report the *stored-floats-vs-generated-length*
  curve for every method (BUG/MorphKV/StreamingLLM flat, full cache linear).
- **Compute honesty**: at matched memory BUG attends over `n_sink + W + w`
  positions vs eviction's `m` (< that) — BUG buys more history per byte but
  pays more attention FLOPs per step. We match on *memory* (the stated axis)
  and report per-token latency and attended-length alongside, so the trade is
  visible rather than hidden.

## 8. Benchmarks (Axis B)

1. **Streaming perplexity** (primary, deterministic): prefill `P` tokens of a
   long held-out document (PG-19 / WikiText-103), then feed the true
   continuation token-by-token through the *decode* path for `G ≫ P` tokens,
   scoring log-probs under each constant cache. Exercises thousands of
   per-token BUG updates; quality-vs-generated-length curves show degradation
   rates directly (the graceful-degradation hypothesis is a *slope* claim).
2. **Long-form generation quality** (secondary): LongBench-style long-output
   task (gov_report / qasper) scored by ROUGE vs full-cache outputs and/or
   an LLM judge — capacity permitting, on the pod at 8B.
3. Memory-vs-length and per-token latency curves for every method.

Scale: validate at 1B on CPU, benchmark at 8B on a pod (RTX 6000 Ada,
onstart-batch + `python -u` + results between ===MARKERS=== via `vastai logs`;
rotate HF/vast keys first — see memory notes). Bigger model only after 8B is
clean.

## 9. Risks / escalation

- MorphKV faithfulness: if the paper's mechanism can't be reproduced cleanly
  on transformers 5.8, escalate with deviation options (per project rules).
- The 5.8 mask machinery may special-case shapes we don't anticipate (sdpa vs
  eager); mitigated by proof #1/#2 running under both attention backends.
- Latency: if per-step concat/reconstruction dominates at 8B/GPU, cache more
  aggressively (only the rotated-K concat is per-step); report honestly.
