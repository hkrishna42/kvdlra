"""`scripts/figures.py` and `scripts/dump_kv.py`: the two remaining entrypoints.

The figure build is checked for the files the paper includes, not for pixels; the
dump verifier is checked on a two-file fake dump, which is the only part of the KV
dumper that runs without a GPU and a 4.7 GB tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args], capture_output=True, text=True, cwd=REPO_ROOT
    )


def test_figures_build_writes_every_paper_figure(tmp_path: Path) -> None:
    r = _run("figures.py", "build", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "coldstart.pdf", "coldstart.png",
        "fairquant.pdf", "fairquant.png",
        "one_over_t.pdf", "one_over_t.png",
    ]  # fmt: skip


def test_dump_kv_sha256_manifest_and_verify(tmp_path: Path) -> None:
    """`sha256` writes one line per file; `verify` exits 1 naming every file that
    changed and every one that vanished."""
    root = tmp_path / "dumps"
    (root / "doc0_len16").mkdir(parents=True)
    (root / "doc0_len16" / "layer_00.pt").write_bytes(b"\x00\x01\x02")
    (root / "doc0_len16" / "meta.json").write_text('{"model": "fake"}\n')

    r = _run("dump_kv.py", "sha256", "--dir", str(root))
    assert r.returncode == 0, r.stderr
    manifest = root.parent / "dumps.sha256"
    assert len(manifest.read_text().splitlines()) == 2
    assert _run("dump_kv.py", "verify", "--dir", str(root)).returncode == 0

    (root / "doc0_len16" / "layer_00.pt").write_bytes(b"\x00\x01\x03")
    (root / "doc0_len16" / "meta.json").unlink()
    bad = _run("dump_kv.py", "verify", "--dir", str(root))
    assert bad.returncode == 1
    assert "layer_00.pt" in bad.stdout and "meta.json" in bad.stdout
