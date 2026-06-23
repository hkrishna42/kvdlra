"""Reproduce the Week-1 singular-value-decay figure with fixed defaults.

Thin wrapper around :mod:`scripts.sigma_decay` that pins the Week-1 settings --
layers 0 / 8 / 15 and output ``figures/week1/sv_decay.pdf`` -- so that

    python scripts/week1_sv_decay.py --dump <dump_dir>

regenerates the figure without re-specifying ``--layers``/``--out``. Any extra
flags (e.g. ``--keep-sinks``, ``--ranks ...``) are forwarded verbatim to
``sigma_decay``. ``sigma_decay`` remains the single source of truth for the
plotting and reconstruction logic.
"""

from __future__ import annotations

import runpy
import sys

# Week-1 defaults, prepended so an explicit user override later on argv wins.
week1_layers = ["0", "8", "15"]
week1_out = "figures/week1/sv_decay.pdf"


def main() -> None:
    """Inject the Week-1 defaults into argv, then run ``scripts/sigma_decay``."""
    forwarded = sys.argv[1:]
    argv = ["scripts/sigma_decay.py"]
    if "--layers" not in forwarded:
        argv += ["--layers", *week1_layers]
    if "--out" not in forwarded:
        argv += ["--out", week1_out]
    argv += forwarded
    sys.argv = argv
    # Execute sigma_decay as __main__ so its argparse + main() run as usual.
    runpy.run_module("sigma_decay", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
