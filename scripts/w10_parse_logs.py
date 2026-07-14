"""Parse Week-10 GPU per-arm result lines from raw pod logs into structured JSON.

The pods stream one short line per arm (which survives vast.ai log truncation,
unlike the base64 result blocks). This harvests those lines from
``results/gpu_logs/<tag>.raw.log`` into ``results/w10-gpu-parsed.json``:

  ppl:   ``<method>  [T=<t>] ppl=<v> tok_eq/layer=<te> ratio=<r>``
  ruler: ``[<task> ctx<t>] <method>  acc=<a> recall=<rc> ratio=<r>``
  lb:    ``[<task> len~<L>] <method>  f1=<v> ratio=<r>``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PPL = re.compile(r"^\s*(\S+)\s+\[T=(\d+)\]\s+ppl=([\d.]+)\s+tok_eq/layer=([\d.]+)\s+ratio=([\d.]+)")
RULER = re.compile(r"^\[(\w+) ctx(\d+)\]\s+(\S+)\s+acc=([\d.]+)\s+recall=([\d.]+)\s+ratio=([\d.]+)")
LB = re.compile(r"^\[(\w+) len~([\d.]+)\]\s+(\S+)\s+f1=([\d.]+)\s+ratio=([\d.]+)")


def parse(tag: str, text: str) -> dict[str, list[dict[str, object]]]:
    ppl, ruler, lb = [], [], []
    for line in text.splitlines():
        if m := PPL.match(line):
            ppl.append(
                {
                    "model": tag,
                    "method": m[1],
                    "T": int(m[2]),
                    "ppl": float(m[3]),
                    "tok_eq": float(m[4]),
                    "ratio": float(m[5]),
                }
            )
        elif m := RULER.match(line):
            ruler.append(
                {
                    "model": tag,
                    "task": m[1],
                    "ctx": int(m[2]),
                    "method": m[3],
                    "acc": float(m[4]),
                    "recall": float(m[5]),
                    "ratio": float(m[6]),
                }
            )
        elif m := LB.match(line):
            lb.append(
                {
                    "model": tag,
                    "task": m[1],
                    "avg_len": float(m[2]),
                    "method": m[3],
                    "f1": float(m[4]),
                    "ratio": float(m[5]),
                }
            )
    return {"ppl": ppl, "ruler": ruler, "lb": lb}


def main() -> None:
    out: dict[str, list[dict[str, object]]] = {"ppl": [], "ruler": [], "lb": []}
    logdir = Path("results/gpu_logs")
    for log in sorted(logdir.glob("*.raw.log")):
        tag = log.stem.replace(".raw", "")
        got = parse(tag, log.read_text(errors="ignore"))
        for k in out:
            out[k].extend(got[k])
        print(f"{tag}: ppl={len(got['ppl'])} ruler={len(got['ruler'])} lb={len(got['lb'])}")
    Path("results/w10-gpu-parsed.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"[wrote results/w10-gpu-parsed.json]  totals: "
        f"ppl={len(out['ppl'])} ruler={len(out['ruler'])} lb={len(out['lb'])}"
    )


if __name__ == "__main__":
    main()
