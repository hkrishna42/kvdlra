# Pitfall: post-RoPE keys are not low-rank

**TL;DR (3 sentences, for the README).** HuggingFace applies rotary position
embeddings (RoPE) to the keys *before* writing them to the KV cache, so the
cached keys are *post-RoPE*. RoPE rotates each key's feature pairs by a
position-dependent angle, which smears low-rank structure across positions and
makes the post-RoPE key matrix far less compressible than the *pre-RoPE* keys
(ShadowKV, arXiv:2410.21465 §3.1). Any singular-value-decay or DLRA result on
cached keys is therefore measuring the harder, post-RoPE object — keep that in
mind when reading reconstruction-error curves.

## Where it happens
In `transformers`' `LlamaAttention.forward`, the order is:

```
q, k = apply_rotary_pos_emb(q, k, cos, sin)      # <-- rotation
...
k, v = past_key_values.update(k, v, layer_idx)   # <-- cache write (post-RoPE)
```

Because the Day-5 capture script snapshots inside `DynamicCache.update`, it
records **post-RoPE** keys. Values are never rotated, so V is unaffected.

## Why it matters for kvdlra
DLRA / BUG compresses a moving low-rank matrix; the whole premise needs the
tracked matrix to actually *be* low-rank. ShadowKV reports that pre-RoPE keys are
"exceptionally low-rank" while post-RoPE keys are not. Consequences:

- The Week-1 singular-value-decay figure is computed on **post-RoPE** keys (per
  the plan's Script #2 / the HF default). If its decay looks shallow, or its
  reconstruction-error-vs-rank curve looks disappointing, **that is this pitfall,
  not a failure of DLRA** — report it honestly rather than tuning it away.
- The lever is to compress **pre-RoPE** keys instead, two ways: (a) monkey-patch
  attention to stash K *before* `apply_rotary_pos_emb`, or (b) apply the inverse
  per-position rotation to the cached K before factoring. This is the Week-2/3
  experiment, and the project's **escalation trigger #1**: if even pre-RoPE keys
  at early layers are not low-rank, the core premise is weak and we pivot.

## Week-1 stance
Capture and analyze **post-RoPE** keys (matches HF's cache and the plan).
Document the decay as-is. Defer the pre-RoPE comparison to Week 2/3. Do **not**
silently switch formulations mid-Week-1.

See also [`conventions.md`](conventions.md) for the matrix layout (rows = features
`head_dim × num_kv_heads = 64 × 8 = 512`, columns = tokens) and pitfall #5
(exclude the first 4 attention-sink tokens from the low-rank track).
