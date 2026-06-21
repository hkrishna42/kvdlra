# Dependency pin substitutions

The PLAN (§2) pins several far-future versions. On this machine (Apple Silicon
Mac, CPU-only, no CUDA), provisioned with `uv` + CPython 3.12, all but two of
those pins resolved exactly as written against PyPI as of 2026-06-21. The two
that did not are recorded below, each also carried as an inline
`# substituted:` comment next to the dependency in `pyproject.toml`.

The substitution rule applied: for every pin that fails to resolve, relax it to
the newest installable release of that same package that satisfies the rest of
the resolution. All other pins were kept exactly as the PLAN specifies.

| Package | PLAN pin | Installed | Why |
|---|---|---|---|
| `datasets` | `4.8.5` | `2.21.0` | `kvpress==0.5.1` hard-caps `datasets>=2.21.0,<3`. `4.8.5` is incompatible. `2.21.0` is the newest `datasets` that resolves alongside `kvpress==0.5.1` (its transitive constraints pin it to the floor of the allowed range). |
| `mkdocs-material` | `9.8.0` | `9.7.6` | `mkdocs-material==9.8.0` is not published on PyPI (no matching distribution). `9.7.6` is the newest installable release. |

## Pins that were kept exactly (verified to resolve)

`torch==2.11.0` (CPU/MPS macOS arm64 wheel — there is no CUDA wheel for macOS;
`torch.cuda.is_available()` is `False` here), `transformers==5.8.0`,
`accelerate==1.13.0`, `huggingface_hub==1.14.0`, `kvpress==0.5.1`,
`hydra-core==1.3.2`, `wandb==0.26.1`, `pytest==9.0.3`, `ruff==0.15.12`,
`mypy==2.0.0`, `mkdocstrings[python]==1.0.4`.

## Notes / deviations from PLAN §2 install block

- The PLAN's `uv pip install torch==2.11.0 torchvision==0.26.0 --index-url
  https://download.pytorch.org/whl/cu128` step was **not** run: that is a CUDA
  (cu128) index and is meaningless on a CPU-only Mac. The CPU torch wheel comes
  straight from PyPI via the declarative `pyproject.toml` dependency.
- `flash-attn` was **not** installed; it remains only in the `[flash]` optional
  extra, to be built on a CUDA pod later (PLAN §6).
- `requires-python` is kept at `>=3.10`, but the virtual environment was created
  with CPython 3.12 (`uv venv --python 3.12`) to match `[tool.mypy]
  python_version = "3.12"`.

## Reproduce

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"   # exits 0
```
