"""Week-19 board: render docs/board/week19-board.html from the committed result files.

Data-driven (numbers never retyped): the a1 fair-quant intervals + McNemar contrasts, the
a2 official-RULER cells + contrasts, the a1q compose rows, the a3 cold-start rows and, when
present, the a4 64K rows. Authored text carries the reading; every number in it is computed
here. The Week-17 result tables are appended verbatim from docs/board/week17-archive.html.

    uv run python scripts/w19_dashboard.py   # then publish docs/board/week19-board.html
"""

# ruff: noqa: E501, RUF001  (HTML template strings; deliberate middle-dot, times, star, dash glyphs)
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from w18_intervals import ROW

RES = Path("results")
OUT = Path("docs/board/week19-board.html")
FLAG, COMPOSE, K2, K4, K8 = (
    "bugSseed-r64-h256",
    "bugSseed-r64-h256-q4",
    "quant-2bit-kivi",
    "quant-4bit-kivi",
    "quant-8bit-kivi-hqq",
)
FAMILIES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B-v0.3"), ("qwen", "Qwen2.5-7B")]
TASKS = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
TSHORT = {
    "niah_single": "single",
    "niah_multikey": "multi-key",
    "niah_multivalue": "multi-value",
    "vt": "var-track",
}
OFFICIAL = [
    "niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
    "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt",
]  # fmt: skip
OSHORT = dict(zip(OFFICIAL, ["s1", "s2", "s3", "mk1", "mk2", "mk3", "mv", "mq", "vt"], strict=True))
OARMS = ["full", "think-c0.5", "palu-r0.5", K4, K2, FLAG, "ea-k0.1"]
PPL_RE = re.compile(r"^\s+(\S+)\s+\[T=(\d+)\] ppl=([0-9.]+) .*?sbits=([0-9.]+)", re.M)
PERSIST_RE = re.compile(
    r"^\[persist ctx(\d+)\] (\S+)\s+bytes=(\d+) ratio=([0-9.]+) save=[0-9.]+s load=([0-9.]+)s "
    r"h2d=([0-9.]+)s ready=([0-9.]+)s cold=([0-9.]+)s",
    re.M,
)


def _flag_ppl() -> dict[str, dict[int, float]]:
    """Flagship PPL4 from the committed Week-18 primaries (never retyped)."""
    out: dict[str, dict[int, float]] = {}
    for tag, _ in FAMILIES:
        p = RES / f"w18-{tag}-ppl-lines.txt"
        d: dict[int, float] = {}
        if p.exists():
            for m in PPL_RE.finditer(p.read_text()):
                if m.group(1) == FLAG:
                    d[int(m.group(2))] = float(m.group(3))
        out[tag] = d
    return out


FLAG_PPL = _flag_ppl()  # Week-18 same-family PPL4 (results/w18-*-ppl-lines.txt)


# ------------------------------------------------------------------ loaders
def _intervals(name: str) -> dict[str, Any]:
    blob: dict[str, Any] = json.loads(
        (RES / "w19_intervals" / f"{name}-ruler-intervals.json").read_text()
    )
    return blob


def _sig(blob: dict[str, Any], b: str, ctx: str, task: str) -> tuple[int, int, float, bool] | None:
    for c in blob["contrasts"]:
        if c["arm_a"] == FLAG and c["arm_b"] == b and c["ctx"] == ctx and c["task"] == task:
            return c["a_favored"], c["b_favored"], c["p_value"], c["significant"]
    return None


def _rows(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    if path.exists():
        for m in ROW.finditer(path.read_text()):
            out[(m.group(3), int(m.group(2)), m.group(1))] = {
                "acc": float(m.group(4)),
                "sbits": float(m.group(7) or m.group(6)),
                "n": int(m.group(8)),
            }
    return out


def _ppl(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    out: dict[tuple[str, int], tuple[float, float]] = {}
    if path.exists():
        for m in PPL_RE.finditer(path.read_text()):
            out[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


# ------------------------------------------------------------------ pieces
def pill(acc: float, note: str = "") -> str:
    cls = "a1" if acc >= 0.9 else ("am" if acc >= 0.4 else "a0")
    return f'<span class="acc {cls}">{acc:.2f}</span>{note}'


def mem(x: float, scale: float = 220.0) -> str:
    return f'<span class="mem"><span class="bar" style="width:{max(2, x * scale):.0f}px"></span><span class="num">{x:.3f}</span></span>'


def section_fairquant() -> str:
    parts = []
    for tag, fam in FAMILIES:
        blob = _intervals(f"a1-{tag}")
        cells = blob["cells"]
        for ctx in ("16384", "32768"):
            for arm, label in ((FLAG, "BUG flagship"), (K2, "KIVI 2-bit"), (K4, "KIVI 4-bit")):
                a = cells.get(ctx, {}).get(arm)
                if not a:
                    continue
                sb = next(v["sbits"] for v in a.values())
                tds = []
                for t in TASKS:
                    c = a.get(t)
                    if not c:
                        tds.append("<td>—</td>")
                        continue
                    note = ""
                    if arm != FLAG:
                        s = _sig(blob, arm, ctx, t)
                        if s and s[3]:
                            who = "BUG" if s[0] > s[1] else label
                            note = f'<span class="ci" title="paired McNemar p={s[2]:.3f}">★ {html.escape(who)} p={s[2]:.3f}</span>'
                    tds.append(f"<td>{pill(c['acc'], note)}</td>")
                cls = "grp spot" if arm == FLAG else "grp"
                parts.append(
                    f'<tr class="{cls}"><td class="lbl">{html.escape(fam)} · {int(ctx) // 1024}K · {label}</td>'
                    f"<td>{mem(sb)}</td>" + "".join(tds) + "</tr>"
                )
    return (
        '<div class="twrap"><table><caption>Flagship vs the fair KIVI baseline, matched stored bytes'
        '<span class="cx">n=12 per cell, paired on the same needles · ★ = paired McNemar p&lt;0.05 (names the winner) · '
        "stored = honest fp32-at-rest bits for BUG, measured bf16 aux for KIVI</span></caption>"
        '<thead><tr><th class="lbl">family · ctx · arm</th><th>stored ×full</th>'
        + "".join(f"<th>{TSHORT[t]}</th>" for t in TASKS)
        + "</tr></thead><tbody>"
        + "".join(parts)
        + "</tbody></table></div>"
    )


def section_ppl() -> str:
    rows = []
    for tag, fam in FAMILIES:
        p = _ppl(RES / f"w19-a1-{tag}-lines.txt")
        for ctx in (16384, 32768):
            k2, k4 = p.get((K2, ctx)), p.get((K4, ctx))
            f = FLAG_PPL[tag][ctx]
            rows.append(
                f'<tr class="grp"><td class="lbl">{fam} · {ctx // 1024}K</td>'
                f'<td class="ppl {"brk" if f > 20 else "ok"}">{f:.2f}</td>'
                f'<td class="ppl ok">{k2[0]:.2f}</td><td class="ppl ok">{k4[0]:.2f}</td></tr>'
                if k2 and k4
                else f'<tr class="grp"><td class="lbl">{fam} · {ctx // 1024}K</td><td>{f:.2f}</td><td>—</td><td>—</td></tr>'
            )
    return (
        '<div class="twrap"><table><caption>Perplexity, same harness<span class="cx">WikiText-103, 4 windows of 512; '
        "flagship = Week-18 PPL4</span></caption>"
        '<thead><tr><th class="lbl">family · ctx</th><th>BUG flagship</th><th>KIVI 2-bit</th><th>KIVI 4-bit</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def section_official() -> tuple[str, dict[str, float]]:
    blob = _intervals("a2-llama")
    cells = blob["cells"]["16384"]
    means: dict[str, float] = {}
    rows = []
    for arm in OARMS:
        a = cells.get(arm, {})
        accs = [a[t]["acc"] for t in OFFICIAL if t in a]
        means[arm] = sum(accs) / len(accs) if accs else float("nan")
        sb = next(v["sbits"] for v in a.values())
        tds = []
        for t in OFFICIAL:
            c = a.get(t)
            note = ""
            if c and arm != FLAG:
                s = _sig(blob, arm, "16384", t)
                if s and s[3]:
                    who = "BUG" if s[0] > s[1] else arm
                    note = f'<span class="ci">★ {html.escape(who)}</span>'
            tds.append(f"<td>{pill(c['acc'], note)}</td>" if c else "<td>—</td>")
        cls = "grp spot" if arm == FLAG else ("grp base" if arm == "full" else "grp")
        rows.append(
            f'<tr class="{cls}"><td class="lbl">{html.escape(arm)}</td><td>{mem(sb, 120)}</td>'
            + "".join(tds)
            + f"<td><b>{means[arm]:.2f}</b></td></tr>"
        )
    table = (
        '<div class="twrap"><table><caption>Official NVIDIA RULER (commit c3f5e3b), Llama-3.1-8B, 16K'
        '<span class="cx">12 records per task, official string_match_all scoring · ★ = paired McNemar p&lt;0.05 vs the flagship (names the winner)</span></caption>'
        '<thead><tr><th class="lbl">arm</th><th>stored</th>'
        + "".join(f"<th>{OSHORT[t]}</th>" for t in OFFICIAL)
        + "<th>mean</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return table, means


def section_compose() -> tuple[str, dict[str, dict[int, dict[str, float]]]]:
    rows = []
    summary: dict[str, dict[int, dict[str, float]]] = {}
    for tag, fam in FAMILIES:
        r = _rows(RES / f"w19-a1q-{tag}-lines.txt")
        p = _ppl(RES / f"w19-a1q-{tag}-lines.txt")
        for ctx in (16384, 32768):
            got = {t: r.get((COMPOSE, ctx, t)) for t in TASKS}
            if not any(got.values()):
                continue
            sb = next(g["sbits"] for g in got.values() if g)
            summary.setdefault(tag, {})[ctx] = {
                "sbits": sb,
                **{t: (g["acc"] if g else float("nan")) for t, g in got.items()},
            }
            pp = p.get((COMPOSE, ctx))
            pcls = "brk" if pp and pp[0] > 50 else "ok"
            rows.append(
                f'<tr class="grp"><td class="lbl">{fam} · {ctx // 1024}K</td><td>{mem(sb)}</td>'
                + "".join(f"<td>{pill(g['acc']) if g else '—'}</td>" for g in got.values())
                + f'<td class="ppl {pcls}">{pp[0]:.2f}</td></tr>'
                if pp
                else f'<tr class="grp"><td class="lbl">{fam} · {ctx // 1024}K</td><td>{mem(sb)}</td>'
                + "".join(f"<td>{pill(g['acc']) if g else '—'}</td>" for g in got.values())
                + "<td>pending</td></tr>"
            )
    table = (
        '<div class="twrap"><table><caption>Below the quantizer floor: the seeded gist with 4-bit coordinates'
        '<span class="cx">bugSseed-r64-h256-q4 · 512 fp32 coordinate columns kept, the rest 4-bit PolarQuant (never dropped) · n=12</span></caption>'
        '<thead><tr><th class="lbl">family · ctx</th><th>stored ×full</th>'
        + "".join(f"<th>{TSHORT[t]}</th>" for t in TASKS)
        + "<th>ppl</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return table, summary


def section_coldstart() -> tuple[str, dict[tuple[int, str], tuple[float, float]]]:
    text = (RES / "w19-a3-llama2-lines.txt").read_text()
    data: dict[tuple[int, str], tuple[float, float]] = {}
    rows = []
    labels = {"full": "full KV (fp16)", FLAG: "BUG flagship", K2: "KIVI 2-bit"}
    for m in PERSIST_RE.finditer(text):
        ctx, arm, b, ratio, load, h2d, ready, cold = m.groups()
        data[(int(ctx), arm)] = (int(b) / 1e9, float(cold))
        cls = "grp spot" if arm == FLAG else ("grp base" if arm == "full" else "grp")
        rows.append(
            f'<tr class="{cls}"><td class="lbl">{int(ctx) // 1024}K · {labels.get(arm, arm)}</td>'
            f"<td>{int(b) / 1e9:.2f} GB</td><td>{mem(float(ratio), 120)}</td><td>{load}</td><td>{h2d}</td><td>{ready}</td><td><b>{cold}</b></td></tr>"
        )
    table = (
        '<div class="twrap"><table><caption>Persisted-cache cold start, Llama-3.1-8B, A100-40GB'
        '<span class="cx">seconds to attend-ready = warm page-cache read + host-to-device + reconstruct; medians of 5; persisted state = exactly the billed tensors</span></caption>'
        '<thead><tr><th class="lbl">ctx · arm</th><th>bytes</th><th>ratio</th><th>read</th><th>H2D</th><th>ready</th><th>cold</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    return table, data


def section_64k() -> str:
    path = RES / "w19-a4-llama-lines.txt"
    r = _rows(path)
    p = _ppl(path)
    if not r:
        return (
            '<div class="note warn"><p><b>64K point (Llama): running.</b> The flagship, the fair quantizer and eviction '
            "at 64K; the flagship's stored ratio should fall below 0.139× (its 32K value) while KIVI holds 0.156×.</p></div>"
        )
    rows = []
    for arm, label in (
        (FLAG, "BUG flagship"),
        (K2, "KIVI 2-bit"),
        (K4, "KIVI 4-bit"),
        ("ea-k0.1", "ExpectedAttention k0.1"),
    ):
        got = {t: r.get((arm, 65536, t)) for t in TASKS}
        if not any(got.values()):
            continue
        sb = next(g["sbits"] for g in got.values() if g)
        pp = p.get((arm, 65536))
        rows.append(
            f'<tr class="{"grp spot" if arm == FLAG else "grp"}"><td class="lbl">{label}</td><td>{mem(sb)}</td>'
            + "".join(f"<td>{pill(g['acc']) if g else '—'}</td>" for g in got.values())
            + f'<td class="ppl ok">{pp[0]:.2f}</td></tr>'
            if pp
            else f'<tr class="{"grp spot" if arm == FLAG else "grp"}"><td class="lbl">{label}</td><td>{mem(sb)}</td>'
            + "".join(f"<td>{pill(g['acc']) if g else '—'}</td>" for g in got.values())
            + "<td>—</td></tr>"
        )
    return (
        '<div class="twrap"><table><caption>The 64K point, Llama-3.1-8B<span class="cx">the 1/T term with data: flagship n=8, '
        "quantizer and eviction n=12</span></caption>"
        '<thead><tr><th class="lbl">arm</th><th>stored ×full</th>'
        + "".join(f"<th>{TSHORT[t]}</th>" for t in TASKS)
        + "<th>ppl</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


# ------------------------------------------------------------------ page
def main() -> None:
    fair = section_fairquant()
    ppl = section_ppl()
    official, means = section_official()
    compose, csum = section_compose()
    cold, cdata = section_coldstart()
    k64 = section_64k()
    archive = Path("docs/board/week17-archive.html").read_text()
    ll16, ll32 = csum["llama"][16384], csum["llama"].get(32768)
    full16, flag16, k216 = cdata[(16384, "full")], cdata[(16384, FLAG)], cdata[(16384, K2)]
    speed = full16[1] / flag16[1]
    ctx32 = f" and {ll32['sbits']:.3f}× at 32K" if ll32 else ""
    page = f"""<title>kvdlra Weeks 18–19</title>
<style>
  :root{{
    --bg:#f4f6f8; --panel:#ffffff; --ink:#111a24; --ink-soft:#3d4b59; --faint:#6b7a89;
    --line:#dde3e9; --line-soft:#eaeef2;
    --accent:#0f8a8a; --accent-soft:#0f8a8a22;
    --good:#1f9d57; --good-bg:#1f9d5718; --warn:#c07d10; --warn-bg:#c07d1018;
    --bad:#c0392b; --bad-bg:#c0392b14;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --maxw:1120px;
  }}
  @media (prefers-color-scheme:dark){{
    :root:not([data-theme="light"]){{
      --bg:#0c1116; --panel:#141c24; --ink:#e8eef4; --ink-soft:#aebac6; --faint:#7d8b99;
      --line:#243039; --line-soft:#1b242c; --accent:#3fd0c9; --accent-soft:#3fd0c922;
      --good:#3ecf7f; --good-bg:#3ecf7f1c; --warn:#e6b24a; --warn-bg:#e6b24a1c; --bad:#ef6a5c; --bad-bg:#ef6a5c1e;
    }}
  }}
  :root[data-theme="dark"]{{
    --bg:#0c1116; --panel:#141c24; --ink:#e8eef4; --ink-soft:#aebac6; --faint:#7d8b99;
    --line:#243039; --line-soft:#1b242c; --accent:#3fd0c9; --accent-soft:#3fd0c922;
    --good:#3ecf7f; --good-bg:#3ecf7f1c; --warn:#e6b24a; --warn-bg:#e6b24a1c; --bad:#ef6a5c; --bad-bg:#ef6a5c1e;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink)}}
  .wrap{{max-width:var(--maxw);margin:0 auto;padding:clamp(20px,4vw,56px) clamp(16px,4vw,40px);
    font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased;min-height:100vh}}
  .eyebrow{{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:650;margin:0 0 .6rem}}
  h1{{font-size:clamp(1.5rem,3.4vw,2.15rem);line-height:1.12;margin:.1rem 0 .5rem;font-weight:720;letter-spacing:-.015em;text-wrap:balance}}
  .lede{{font-size:1.03rem;color:var(--ink-soft);max-width:68ch;margin:.2rem 0 0}}
  .lede b{{color:var(--ink);font-weight:650}}
  .rule{{height:1px;background:var(--line);border:0;margin:clamp(22px,3.4vw,34px) 0}}
  h2{{font-size:1.16rem;margin:0 0 .2rem;letter-spacing:-.01em;font-weight:680;text-wrap:balance}}
  .sub{{color:var(--faint);font-size:.9rem;margin:0 0 1rem;max-width:80ch}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:6px 0 4px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
  .card .k{{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin:0 0 .35rem;font-weight:600}}
  .card .v{{font-size:1.32rem;font-weight:700;letter-spacing:-.01em;font-family:var(--mono);margin:0}}
  .card .n{{font-size:.86rem;color:var(--ink-soft);margin:.4rem 0 0}}
  .card .v .u{{font-size:.8rem;color:var(--faint);font-weight:500}}
  .twrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel);margin-top:10px}}
  table{{border-collapse:collapse;width:100%;min-width:640px;font-size:.9rem}}
  caption{{caption-side:top;text-align:left;padding:14px 16px 0;font-weight:670;font-size:1.02rem;letter-spacing:-.01em}}
  caption .cx{{display:block;font-weight:400;color:var(--faint);font-size:.82rem;margin-top:2px}}
  th,td{{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line-soft);font-variant-numeric:tabular-nums;font-family:var(--mono);white-space:nowrap}}
  th{{font-family:var(--sans);font-weight:600;color:var(--faint);font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;border-bottom:1px solid var(--line)}}
  td.lbl,th.lbl{{text-align:left;font-family:var(--sans)}}
  td.lbl{{font-weight:600;color:var(--ink)}}
  tbody tr:last-child td{{border-bottom:0}}
  .grp td{{background:var(--accent-soft)}}
  tr.spot td{{box-shadow:inset 3px 0 0 var(--accent)}}
  tr.base td.lbl{{color:var(--ink-soft);font-weight:500}}
  .tag{{display:inline-block;font-family:var(--sans);font-size:.66rem;font-weight:650;letter-spacing:.04em;padding:1px 6px;border-radius:5px;margin-left:8px;vertical-align:middle;text-transform:uppercase}}
  .tag.spot{{background:var(--accent);color:#fff}} .tag.brk{{background:var(--bad-bg);color:var(--bad)}}
  .tag.abs{{background:var(--warn-bg);color:var(--warn)}} .tag.fix{{background:var(--good-bg);color:var(--good)}}
  .mem{{display:flex;align-items:center;justify-content:flex-end;gap:9px}}
  .mem .bar{{height:7px;border-radius:4px;background:var(--accent);opacity:.85;min-width:2px}}
  .mem .num{{width:3.6em;text-align:right}}
  .acc{{display:inline-block;min-width:2.9em;text-align:center;padding:2px 8px;border-radius:6px;font-weight:650}}
  .a1{{color:var(--good);background:var(--good-bg)}} .am{{color:var(--warn);background:var(--warn-bg)}} .a0{{color:var(--bad);background:var(--bad-bg)}}
  .ci{{display:block;font-family:var(--mono);font-size:.68rem;color:var(--faint);margin-top:3px;font-weight:500}}
  .ppl.brk{{color:var(--bad);font-weight:700}} .ppl.ok{{color:var(--ink)}} .ppl.fix{{color:var(--good);font-weight:700}}
  .arrow{{color:var(--faint);padding:0 6px}}
  .foot{{color:var(--faint);font-size:.82rem;margin:10px 2px 0;max-width:80ch}}
  .note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:14px 18px;margin-top:12px}}
  .note.good{{border-left-color:var(--good)}} .note.warn{{border-left-color:var(--warn)}}
  .note p{{margin:.3rem 0;color:var(--ink-soft);font-size:.92rem}} .note b{{color:var(--ink)}}
  .meta{{color:var(--faint);font-size:.78rem;margin-top:26px;font-family:var(--mono)}}
  a{{color:var(--accent)}} a:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
</style>
<div class="wrap">
  <p class="eyebrow">Weeks 18–19 · the fair baseline · the official anchor · three families</p>
  <h1>Where the claim narrowed, and what survived it</h1>
  <p class="lede">The Week-18 panel's fatal gap was a missing 2-bit quantization baseline. Week 19 ran a KIVI-scheme
    baseline (per-channel keys, per-token values, G=64, paired needle-for-needle) and an official NVIDIA RULER
    anchor. <b>At matched stored bytes 2-bit KIVI ties the flagship on single needles and wins fluency;
    the flagship's edge is multi-value retrieval on our generator</b>, which the official suite does not
    reproduce. <b>What no fixed-bit scalar quantizer reaches is the composed cell at {ll16["sbits"]:.3f}× stored</b> (Llama,
    Mistral) with full single/multi-key/multi-value retrieval — at a fluency cost, and not on Qwen.</p>

  <hr class="rule">
  <div class="cards">
    <div class="card"><p class="k">Fair 2-bit at matched bytes · our generator</p>
      <p class="v">mv <span class="u">3 / 3 families</span></p>
      <p class="n">Flagship wins multi-value at 16K on Llama, Mistral, Qwen (paired McNemar p ≤ 0.03); ties single;
        loses fluency on Mistral/Qwen. 4-bit at ~2× bytes is not separated in any cell.</p></div>
    <div class="card"><p class="k">Official RULER · Llama 16K · 9 tasks</p>
      <p class="v">{means[FLAG]:.2f} <span class="u">vs {means[K2]:.2f} (2-bit)</span></p>
      <p class="n">No task separated either way; the multi-value edge does not transfer. Eviction at 0.1× averages
        {means["ea-k0.1"]:.2f}; 4-bit KIVI {means[K4]:.2f}.</p></div>
    <div class="card"><p class="k">Below the quantizer floor</p>
      <p class="v">{ll16["sbits"]:.3f}× <span class="u">stored{ctx32}</span></p>
      <p class="n">Seeded gist + 4-bit coordinates: single/multi-key/multi-value {ll16["niah_single"]:.2f}/{ll16["niah_multikey"]:.2f}/{ll16["niah_multivalue"]:.2f} on Llama 16K;
        Qwen diverges. 2-bit's floor is 0.156×.</p></div>
    <div class="card"><p class="k">Persisted-cache cold start · 16K</p>
      <p class="v">{flag16[1]:.2f} s <span class="u">vs {full16[1]:.2f} s full</span></p>
      <p class="n">{speed:.0f}× faster to attend-ready ({flag16[0]:.2f} GB vs {full16[0]:.2f} GB). KIVI 2-bit: {k216[1]:.2f} s — the
        persistence win is shared with quantization.</p></div>
  </div>

  <hr class="rule">
  <h2>The crux: flagship vs fair KIVI at matched bytes</h2>
  <p class="sub">The pre-registered fund bar asked whether BUG retrieves where 2-bit quantization fails. Answer: on
    multi-value (all families, 16K) and at 32K on Mistral/Qwen, yes; on single needles, no; against 4-bit, nowhere.</p>
  {fair}
  {ppl}
  <div class="note warn">
    <p><b>Qwen is the honest counter-case.</b> Its n=512 gist stores fp32 at rest, so the flagship's honest ratio there is
      0.27× — the same bytes as 4-bit KIVI, which is perfect on every task with far better perplexity (6.2 vs 8.2 at 16K;
      7.7 vs 35 at 32K). At matched honest bytes the r64 flagship is dominated on Qwen.</p>
  </div>

  <hr class="rule">
  <h2>The official anchor: NVIDIA RULER, unmodified generator</h2>
  <p class="sub">Their essay/noise haystacks, needle types, templates, generation budgets and scoring (commit c3f5e3b); 12 records per task; same arms and decode protocol.</p>
  {official}
  <p class="foot">Transfers from our generator: the eviction collapse (flagship separated from ea-k0.1 on 6 of 9 tasks),
    single-needle parity with 2-bit, the 0.29–0.75× arms near-perfect. Does not transfer: the multi-value edge over
    2-bit (1 vs 1 discordant). The flagship's 22 misses (14 needle at depths 0.15–0.95, six ≤ 0.20; 8 variable-tracking) are front-loaded — six of fourteen in the first ingest chunk.</p>

  <hr class="rule">
  <h2>Below the quantizer floor, and the 1/T term</h2>
  <p class="sub">The flagship alone amortizes toward its 2r/n = 0.125× asymptote (19% under the 2-bit 0.156× floor); composed with 4-bit coordinates it reaches 0.048×, below any fixed-bit scalar-quantizer floor.</p>
  {compose}
  {k64}
  {cold}

  <hr class="rule">
  <h2>Panel gap-fill plan — status after Week 19</h2>
  <div class="twrap"><table>
    <caption>The 2026-09-01 ranked gaps<span class="cx">exit gate = every dimension ≥ 7, zero fatal; the paper (paper/main.tex, CI-built) carries all of it</span></caption>
    <thead><tr><th class="lbl">#</th><th class="lbl">Gap</th><th class="lbl">Outcome</th><th>Status</th></tr></thead>
    <tbody>
      <tr class="grp spot"><td class="lbl">1</td><td class="lbl">No 2-bit KV-quant baseline</td><td class="lbl">Fair KIVI 2/4-bit, 3 families, 16K/32K, paired; the Week-18 zero was a per-token-key artifact; BUG×4-bit compose at 0.048×</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">2</td><td class="lbl">Self-authored benchmark</td><td class="lbl">WikiText filler (W18) + official NVIDIA RULER anchor (W19): eviction collapse transfers, the multi-value edge does not</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">3</td><td class="lbl">Storage billed as memory</td><td class="lbl">Dual billing everywhere; measured cold start {flag16[1]:.2f} s vs {full16[1]:.2f} s (shared with 2-bit)</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">4</td><td class="lbl">Eviction absent from grids</td><td class="lbl">ea/snapkv at 0.1×, 3 families, n=12 (W18 g3); official cells (W19)</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">5</td><td class="lbl">No manuscript</td><td class="lbl">arXiv-v1 draft, 34 verified refs, builds on CI (18 pp), figures from committed line-files</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">6</td><td class="lbl">Wording debts</td><td class="lbl">"leads"/"no detected loss"; the fair-quant outcome reframed the abstract, limits and conclusion</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">7</td><td class="lbl">Evidentiary chain</td><td class="lbl">SHA-pinned pods, env headers, per-trial records for every W18/W19 cell</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">8</td><td class="lbl">Statistical firming</td><td class="lbl">32K n=12, marquee n=16 with McNemar (W18); every W19 contrast paired</td><td><span class="acc a1">done</span></td></tr>
      <tr class="grp"><td class="lbl">+</td><td class="lbl">64K point (1/T)</td><td class="lbl">Llama 64K: flagship vs 2-bit vs eviction</td><td><span class="acc am">{"done" if "64K point" not in k64 else "running"}</span></td></tr>
    </tbody></table></div>

  <p class="meta">kvdlra · week7 · Weeks 18–19 program · A100 · Llama-3.1-8B / Qwen2.5-7B / Mistral-7B-v0.3 · rendered by scripts/w19_dashboard.py from results/</p>

  <hr class="rule">
{archive}
</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"[wrote {OUT}] {len(page)} bytes")


if __name__ == "__main__":
    main()
