"""Decode-step cost model for ADR 0001 (factored-attention kernel).

Stdlib only; regenerates every table in docs/adr/0001-factored-attention-kernel.md.
All constants are named below and printed with the tables. The model is a
DRAM-traffic / FLOP **lower bound**: it counts the tensor passes a kernel must
make, not framework overhead (allocator churn, GQA expansion, Python dispatch).
Measured anchors from results/w20-close-report.md are printed beside it.

    python 0001-cost-model.py
"""

from __future__ import annotations

# --------------------------------------------------------------- constants
# Llama-3.1-8B
L = 32  # layers
HQ = 32  # query heads
HKV = 8  # KV heads
D = 128  # head dim
N = HKV * D  # 1024 = the shared feature dim the gist tracks
G = HQ // HKV  # 4 query heads per KV head (GQA)
PARAMS = 8.03e9  # bf16 weights = 16.06 GB, read once per decode step (all batch)

# r64 configuration (bugSseed-r64-h256): scripts/w10_frontier.py defaults
N_SINK = 4
RING = 32 + 16 - 1  # recent_window + absorb_block - 1 (high water)
HH = 256  # exact tier
ABSORB = 16  # absorb_block: the reconstruct path rebuilds the middle this often

# dtypes / hardware
BF16, FP32 = 2, 4  # bytes
B_U, B_C = FP32, FP32  # gist stored fp32 today (Gate 3.3 would make these bf16)
BW = 1.555e12  # A100-40GB HBM2e, B/s (spec 1.555; the 1.5-2.0 TB/s band)
PEAK = 312e12  # A100 bf16 dense tensor-core FLOP/s
BW_H100 = 3.35e12  # H100-SXM
PEAK_H100 = 990e12  # H100 bf16 dense
SRAM_A100 = 164 * 1024  # bytes of shared memory per SM
SRAM_H100 = 228 * 1024

GB = 1e9
MEASURED = {  # results/w20-close-report.md sec.2, A100-40GB, batch 1, ms/token
    (16384, "full"): 25.9,
    (16384, "recon"): 103.2,
    (32768, "full"): 27.5,
    (32768, "recon"): 188.3,
    (65536, "full"): 37.0,
    (65536, "recon"): 507.9,
}


def t_mid(ctx: int) -> int:
    """Low-rank columns: everything but sinks, exact tier and the recent ring."""
    return ctx - N_SINK - RING - HH


# ------------------------------------------------------- stored footprint
def stored_bits(ctx: int, r: int) -> float:
    """Per-layer at-rest bits, mirroring accounting.bug_footprint().stored_bits()
    for retention='lowrank_surprise', hh_select='surprise', no quant tier."""
    c = t_mid(ctx)
    verbatim_16 = 2 * N * N_SINK + 2 * N * RING + 2 * N * HH  # sinks/ring/tier bf16
    fp32 = 2 * r * c + 2 * N * r  # coordinates C + basis U
    aux = 2 * r + 2 * c + HH  # diag core + (pos, surprise)/column + hh_pos
    return verbatim_16 * 16 + fp32 * 32 + aux * 32


def full_bits(ctx: int) -> float:
    return 2 * ctx * N * 16


# ----------------------------------- per token, per layer, per sequence
def unit(arm: str, r: int) -> tuple[float, float, float]:
    """(FLOPs, bytes read, bytes written) per low-rank token, per layer, per seq."""
    attn = 4 * HQ * D  # QK^T + PV for one token, all query heads
    if arm == "full":
        return attn, 2 * N * BF16, 0.0
    if arm == "i":  # post-RoPE: scores and PV entirely in r-dim coordinate space
        return 4 * HQ * r, 2 * r * B_C, 0.0
    if arm == "ii":  # RoPE-pair factored: per query head, per pair, a 2xr matvec
        return 2 * HQ * D * r + HQ * D + 2 * HQ * r, 2 * r * B_C + 4, 0.0
    if arm == "iii":  # tile reconstruct: U @ C_tile, RoPE, then dense attention
        return 4 * N * r + 3 * N + attn, 2 * r * B_C + 4, 0.0
    if arm == "recon":  # today: memoized n x T middle, re-assembled every step
        rebuild_f = (4 * N * r + 3 * N) / ABSORB
        # rebuild: read C, write fp32 UC, 2 fp32 RoPE passes (r+w), cast to bf16
        rebuild_b = (2 * r * B_C + 2 * N * (4 + 4 * 4 + 4) + 2 * N * BF16) / ABSORB
        # every step: torch.cat re-assembles [sink|tier|mid|ring] (read+write),
        # then SDPA reads it => 3 passes over an n x T bf16 array, K and V.
        return attn + rebuild_f, 2 * (2 * N * BF16) + rebuild_b, 2 * N * BF16
    raise ValueError(arm)


def step(arm: str, r: int, ctx: int, bsz: int) -> dict[str, float]:
    """Whole model, one decode step, batch bsz."""
    f, br, bw = unit(arm, r)
    c = t_mid(ctx)
    dense = N_SINK + HH + RING  # verbatim tokens every arm attends densely
    flops = bsz * L * (f * c + 4 * HQ * D * dense)
    read = bsz * L * (br * c + 2 * N * dense * BF16)
    write = bsz * L * bw * c
    if arm in ("i", "ii", "iii", "recon"):
        read += bsz * L * 2 * N * r * B_U  # basis U, once per layer per step
    if arm == "full":
        read = bsz * L * 2 * N * ctx * BF16
        write = 0.0
    resident = bsz * L * stored_bits(ctx, r) / 8
    if arm == "full":
        resident = bsz * L * 2 * N * ctx * BF16
    if arm == "recon":  # + memoized bf16 middle + one layer's fp32 rebuild buffer
        resident += bsz * L * 2 * N * ctx * BF16 + bsz * 2 * N * ctx * (FP32 + BF16)
    kv_bytes = read + write  # weights excluded: identical for every arm
    ms = max((flops + 2 * PARAMS * bsz) / PEAK, (kv_bytes + PARAMS * BF16) / BW) * 1e3
    return {
        "flops": flops,
        "read": read,
        "write": write,
        "kv": kv_bytes,
        "resident": resident,
        "ms": ms,
    }


# ------------------------------------------------------------------ SRAM
def sram(r: int, tile: int) -> int:
    """Per-program shared memory for option (iii), blocked one KV head x `tile` tokens:
    U_k,U_v head slices resident (D x r), C tiles streamed, K/V tiles materialized."""
    return (
        2 * D * r * BF16  # U_{k,h}, U_{v,h} (bf16, resident for the whole tile loop)
        + 2 * r * tile * BF16  # C_k, C_v tile
        + 2 * D * tile * BF16  # reconstructed K, V tile
        + G * D * BF16  # Q for the GQA group
        + G * D * FP32  # fp32 output accumulator
        + tile * FP32  # true positions (RoPE computed in-kernel from inv_freq)
        + (D // 2) * FP32  # inv_freq
    )


# --------------------------------------------------------------- crossover
def crossover(r: int, peak: float, bw: float, eff: float = 1.0) -> float:
    """bsz * T_mid above which option (iii) stops being weight-bandwidth-bound and
    becomes compute-bound: solve L*phi*x/(P*eff) = W/BW + L*psi*x/BW for x."""
    phi = 4 * N * r + 3 * N + 4 * HQ * D  # FLOPs per low-rank token per layer
    psi = 2 * r * B_C + 4  # bytes per low-rank token per layer
    denom = L * phi / (peak * eff) - L * psi / bw
    return float("inf") if denom <= 0 else (PARAMS * BF16 / bw) / denom


def fmt(x: float, unit_div: float, nd: int = 2) -> str:
    return f"{x / unit_div:.{nd}f}"


def main() -> None:
    ctxs = [16384, 32768, 65536, 131072]
    batches = [1, 8]
    print("# constants: L=32 H_q=32 H_kv=8 D=128 n=1024 GQA 4:1; bf16=2B fp32=4B")
    print(
        f"# A100: BW={BW / 1e12} TB/s, peak bf16={PEAK / 1e12} TFLOP/s; "
        f"H100: {BW_H100 / 1e12} TB/s, {PEAK_H100 / 1e12} TFLOP/s"
    )
    print(f"# r64 config: n_sink={N_SINK} ring={RING} tier={HH} absorb={ABSORB}; U,C stored fp32\n")

    print("## unit cost: per low-rank token, per layer, per sequence")
    print("| arm | FLOPs/token | HBM read B/token | HBM write B/token |")
    print("|---|---|---|---|")
    for arm, r, lbl in [
        ("full", 64, "full KV"),
        ("recon", 64, "reconstruct r64"),
        ("i", 64, "(i) r64"),
        ("i", 128, "(i) r128"),
        ("ii", 64, "(ii) r64"),
        ("iii", 64, "(iii) r64"),
        ("iii", 128, "(iii) r128"),
    ]:
        f, br, bw_ = unit(arm, r)
        print(f"| {lbl} | {f:,.0f} | {br:,.0f} | {bw_:,.0f} |")

    grids = [
        (
            "flops",
            1e9,
            "GFLOP per decode step, whole model",
            1,
            [
                ("full", 64, "full"),
                ("recon", 64, "recon r64"),
                ("i", 64, "(i) r64"),
                ("i", 128, "(i) r128"),
                ("ii", 64, "(ii) r64"),
                ("iii", 64, "(iii) r64"),
                ("iii", 128, "(iii) r128"),
            ],
        ),
        (
            "kv",
            GB,
            "KV HBM GB per decode step, read+write, weights (16.06 GB) excluded",
            2,
            [
                ("full", 64, "full"),
                ("recon", 64, "recon r64"),
                ("iii", 64, "(i)/(ii)/(iii) r64"),
                ("iii", 128, "(i)/(ii)/(iii) r128"),
            ],
        ),
        (
            "resident",
            GB,
            "resident KV GB (stored state; + workspace for recon)",
            2,
            [
                ("full", 64, "full"),
                ("recon", 64, "recon r64"),
                ("iii", 64, "factored r64"),
                ("iii", 128, "factored r128"),
            ],
        ),
        (
            "ms",
            1.0,
            "model ms/token, A100 roofline max(FLOP, HBM) incl. weights",
            1,
            [
                ("full", 64, "full"),
                ("recon", 64, "recon r64"),
                ("ii", 64, "(ii) r64"),
                ("iii", 64, "(iii) r64"),
                ("iii", 128, "(iii) r128"),
            ],
        ),
    ]
    for metric, div, label, nd, cols in grids:
        print(f"\n## {label}")
        print("| T | B | " + " | ".join(c[2] for c in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for ctx in ctxs:
            for bsz in batches:
                vals = [fmt(step(a, r, ctx, bsz)[metric], div, nd) for a, r, _ in cols]
                print(f"| {ctx // 1024}K | {bsz} | " + " | ".join(vals) + " |")

    print("\n## measured (A100, batch 1) vs model floor, ms/token")
    print("| T | full measured | full model | recon measured | recon model | ratio |")
    print("|---|---|---|---|---|---|")
    for ctx in [16384, 32768, 65536]:
        fm, rm = MEASURED[(ctx, "full")], MEASURED[(ctx, "recon")]
        fmo, rmo = step("full", 64, ctx, 1)["ms"], step("recon", 64, ctx, 1)["ms"]
        print(
            f"| {ctx // 1024}K | {fm} | {fmo:.1f} | {rm} | {rmo:.1f} | "
            f"{fm / fmo:.1f}x / {rm / rmo:.1f}x |"
        )

    print("\n## stored ratio vs full fp16 cache (batch-independent)")
    print("| T | r64 stored | r64 fp16-equiv | r128 stored | r128 fp16-equiv |")
    print("|---|---|---|---|---|")
    for ctx in ctxs:
        row = []
        for r in (64, 128):
            c, fb = t_mid(ctx), full_bits(ctx)
            fp16eq = (2 * N * (N_SINK + RING + HH) + 2 * r * c + 2 * N * r) * 16 + (
                2 * r + 2 * c + HH
            ) * 32
            row += [f"{stored_bits(ctx, r) / fb:.3f}x", f"{fp16eq / fb:.3f}x"]
        print(f"| {ctx // 1024}K | " + " | ".join(row) + " |")

    print("\n## option (iii) SRAM per program (one KV head x M tokens, bf16 U and C)")
    print("| r | M | bytes | A100 164KB | H100 228KB |")
    print("|---|---|---|---|---|")
    for r in (64, 128):
        for tile in (32, 64, 128):
            b = sram(r, tile)
            a100 = f"fits x{SRAM_A100 // b}" if b <= SRAM_A100 else "NO"
            h100 = f"fits x{SRAM_H100 // b}" if b <= SRAM_H100 else "NO"
            print(f"| {r} | {tile} | {b / 1024:.0f} KB | {a100} | {h100} |")
    print(
        f"# whole-n basis U_k alone: r64 {N * 64 * BF16 / 1024:.0f} KB, "
        f"r128 {N * 128 * BF16 / 1024:.0f} KB -> per-KV-head blocking is forced"
    )

    print("\n## (iii) weight-bound -> compute-bound crossover, B*T_mid tokens")
    print("| r | A100 @peak | A100 @50% | H100 @peak | H100 @50% |")
    print("|---|---|---|---|---|")
    for r in (64, 128):
        print(
            f"| {r} | "
            + " | ".join(
                f"{crossover(r, p, b, e):,.0f}"
                for p, b in ((PEAK, BW), (PEAK_H100, BW_H100))
                for e in (1.0, 0.5)
            )
            + " |"
        )
    ai = (4 * N * 64 + 3 * N + 4 * HQ * D) / (2 * 64 * B_C + 4)
    print(
        f"# (iii) KV-side arithmetic intensity r64 = {ai:.0f} FLOP/B vs machine "
        f"balance A100 {PEAK / BW:.0f}, H100 {PEAK_H100 / BW_H100:.0f}"
    )

    print("\n## fits in GPU memory at batch B? (weights 16.06 GB + resident KV)")
    print("| T | B | full GB | (iii) r64 GB | A100 40GB | H100 80GB |")
    print("|---|---|---|---|---|---|")
    for ctx in [32768, 65536, 131072]:
        for bsz in (4, 8):
            fu = 16.06 + step("full", 64, ctx, bsz)["resident"] / GB
            kv = 16.06 + step("iii", 64, ctx, bsz)["resident"] / GB
            print(
                f"| {ctx // 1024}K | {bsz} | {fu:.1f} | {kv:.1f} | "
                f"full {'Y' if fu < 40 * 0.93 else 'N'} / (iii) "
                f"{'Y' if kv < 40 * 0.93 else 'N'} | "
                f"full {'Y' if fu < 80 * 0.93 else 'N'} / (iii) "
                f"{'Y' if kv < 80 * 0.93 else 'N'} |"
            )


def side_numbers() -> None:
    """The section-5 figures: eager coordinate rotation per absorb, and the cos/sin
    table the kernel must NOT stream."""
    ctx, r = 32768, 64
    c = t_mid(ctx)
    rot_flops = 2 * 2 * r * r * c * L / ABSORB  # rot_k @ c_k, K+V, all layers, per token
    rot_bytes = 2 * (2 * r * c * B_C) * L / ABSORB  # read C + write C, K and V
    print(
        f"\n# eager coordinate rotation at {ctx // 1024}K r{r}: "
        f"{rot_flops / 1e9:.2f} GFLOP and {rot_bytes / 1e6:.0f} MB per token amortized "
        f"({rot_flops / step('iii', r, ctx, 1)['flops'] * 100:.1f}% of (iii) FLOPs, "
        f"{rot_bytes / BW * 1e3:.2f} ms)"
    )
    for c_len in (65536,):
        print(
            f"# cos/sin table for a {c_len // 1024}K middle: "
            f"{2 * c_len * D * FP32 / 1e6:.0f} MB/step -> compute in-kernel from inv_freq"
        )


def _selfcheck() -> None:
    """Pin the footprint against accounting.bug_footprint's published ratios
    (docstring: r64 ~0.15x stored / 0.085x fp16-equivalent at 16K)."""
    c, fb = t_mid(16384), full_bits(16384)
    assert abs(stored_bits(16384, 64) / fb - 0.151) < 0.002, stored_bits(16384, 64) / fb
    fp16eq = (2 * N * (N_SINK + RING + HH) + 2 * 64 * c + 2 * N * 64) * 16 + (
        2 * 64 + 2 * c + HH
    ) * 32
    assert abs(fp16eq / fb - 0.086) < 0.002, fp16eq / fb
    # full-KV resident must reproduce the measured 4.08 GiB at 32K, batch 1
    assert abs(step("full", 64, 32768, 1)["resident"] / 1024**3 - 4.0) < 0.05
    # (ii) must cost more than (iii) on the key side under GQA 4:1
    assert unit("ii", 64)[0] > unit("iii", 64)[0]


if __name__ == "__main__":
    _selfcheck()
    main()
    side_numbers()
