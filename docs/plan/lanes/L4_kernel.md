# L4_kernel


The w20 sysfix rows are the baseline to beat: reconstruct path 103 / 188 / 508 ms per token and KV peak 3.25 / 6.45 / 12.83 GB at 16K / 32K / 64K; full KV 26 / 28 / 37 ms and 2.05 / 4.08 / 8.14 GB.

1. **ADR** (`docs/adr/0001-factored-attention-kernel.md`) via `brainstorming` over the three options in ICML2027_PLAN.md §2 Gate 3, with a FLOP + HBM-traffic model per decode step at n=1024, r ∈ {64,128}, T ∈ {16K…128K}. Recommendation: (iii) Triton tile-wise reconstruct-inside-attention as primary; (i) post-RoPE r=128 as a parallel accuracy experiment (config `isvd_postrope_r128.yaml`, one Gate-1-style n=24 row on Llama 16K). Status OPEN until I accept.
2. **Prototype (iii).** Decode-only Triton kernel: each 64-token K/V tile materialized in SRAM from `U @ C_tile` (bf16 in, fp32 accumulate), RoPE at true positions, fused QKᵀ/softmax/PV; tier + ring + sinks appended as dense tiles. Tests: max |Δ| < 1e-2 bf16 vs reconstruct-then-attend on random and on dumped 8B KV for one layer; full-model greedy decode token-exact on ≥ 14/16 prompts with mismatches logged.
3. **First numbers** (Week 3): `torch.cuda.max_memory_allocated` and ms/token for full / reconstruct / kernel at 16K/32K/64K, batch 1 and 4, A100 (H100 if available), same script as w20 (`w20_latency.py` → `kvdlra/eval/latency.py`). Gate: kernel KV peak < full KV at 32K and ms/token < reconstruct path by ≥ 3×. Target for Week 5 (ICML plan Gate 3): resident KV ≤ 0.25× and tokens/s ≥ full KV at 64K, batch ≥ 4.
