# Conventions: math ↔ HuggingFace translation table

This note pins down the one convention the rest of the project depends on, so
that the streaming DLRA math (`Y = U S Vᵀ`) maps cleanly onto HuggingFace KV
cache tensors. Write/read this *before* touching any reshape code. (PLAN §8,
pitfall #6.)

## The KV cache tensor layout

HuggingFace `DynamicCache` stores **both** the K and V tensors with shape:

```
(batch, num_kv_heads, seq_len, head_dim)
```

This is confirmed in `transformers/cache_utils.py` and in the kvpress source.
Other libraries (vLLM, Megatron) use different orderings — never copy-paste a
reshape between them without re-checking.

## Llama-3.2-1B-Instruct shape constants

| Quantity | Symbol | Value |
|---|---|---|
| Attention (query) heads | `num_attention_heads` | **32** |
| Key/value heads (GQA) | `num_key_value_heads` | **8** |
| Head dimension | `head_dim` | **64** |
| Hidden layers | `num_hidden_layers` | **16** |

Because of grouped-query attention (GQA), there are only **8** KV heads, each
shared by 4 query heads. KV-budget math must use `num_key_value_heads = 8`, not
`num_attention_heads = 32`, or it is off by 4×.

## The factoring convention (the load-bearing decision)

Ceruti–Lubich write `Y = U S Vᵀ` with

- `U ∈ ℝ^(m×r)` — columns are basis vectors of the **column (feature) space**,
- `S ∈ ℝ^(r×r)`,
- `V ∈ ℝ^(n×r)` — columns are basis vectors of the **row (token) space**.

To turn the 3-D K tensor `(num_kv_heads, T, head_dim)` (after squeezing the
batch dim) into a 2-D matrix to factor, **we choose**:

> **rows = features** = `head_dim × num_kv_heads` = `64 × 8` = **512**
> **columns = tokens** = `T`
> **a new token = a new column.**

So the matrix to factor is `M ∈ ℝ^(512 × T)`. This is the cleanest convention
because "append a token" becomes "append a column", which is exactly the
streaming update in `Y = U S Vᵀ` (the new column lands in the token/row space
spanned by `V`). Concretely, from a per-layer K tensor `K` of shape
`(H, T, D) = (8, T, 64)`:

```
M = K.transpose(0, 2, 1).reshape(H * D, T)   # (512, T): rows=features, cols=tokens
```

| Math (Ceruti–Lubich) | KV-cache code (this project) |
|---|---|
| `Y ∈ ℝ^(m×n)` | `M ∈ ℝ^(512 × T)` |
| `m` (column/feature dim) | `head_dim × num_kv_heads = 512` |
| `n` (row/token dim) | `T` (sequence length) |
| `U ∈ ℝ^(m×r)` feature basis | left factor over the 512 features |
| `V ∈ ℝ^(n×r)` token basis | right factor over the `T` tokens |
| new column of `Y` | new token `(k_t)` appended to `M` |
| `S ∈ ℝ^(r×r)` core | rank-`r` core, kept in **fp32** (PLAN §8 #4) |
