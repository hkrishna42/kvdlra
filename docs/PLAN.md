# KV-DLRA + TurboQuant: a four-week launch manual

**Bottom line up front:** by Friday of week 2 you should have a single plot — singular-value decay of Llama-3.2-1B's KV cache plus Frobenius reconstruction error vs. rank for SVD, incremental SVD, and a from-scratch BUG integrator — that decides whether this project is worth pursuing. Everything below is engineered to get you to that go/no-go figure with the minimum wasted motion, then to a first compressed-generation result by end of week 4. The plan assumes you wake up Monday with a RunPod account, a GitHub account, an H100-80GB quota, and zero prior exposure to either Ceruti–Lubich DLRA or HuggingFace cache internals. Three numbers to anchor on: the BUG algorithm sits in **§3.1 of arXiv 2010.02022**, the rank-adaptive truncation criterion `(Σ σ_j²)^½ ≤ ϑ` sits in **Eq. (5)–(6) of arXiv 2104.05247**, and the entire Llama-3.2-1B KV cache at 4K context is **128 MiB in bf16** — small enough to debug everything on a single H100 without batching tricks.

---

## 1. Day-by-day reading and coding plan (Weeks 1–4)

Time budget convention: "h" = focused hours. A "day" is 6 productive hours; everything below sums to roughly 24 h/week. Doubling the calendar if you have a day job is fine — the dependency graph still holds.

### Week 1 — DLRA "hello world" + first KV capture

| Day | Reading (h) | Coding (h) | Concrete output |
|---|---|---|---|
| **Mon AM** | Lubich textbook *From Quantum to Classical MD* (EMS 2008), **Ch. II §II.1–II.2, pp. 19–34** — Dirac–Frenkel variational principle, low-rank manifold tangent space. (2 h) | Bootstrap repo per §2 below; `pre-commit run --all-files` passes; first `git push`. (2 h) | Green CI on an empty repo; `kvdlra --version` prints `0.1.0`. |
| **Mon PM** | Koch & Lubich 2007, **§1 + §4 Lemma 4.1** (the tangent-space projector formula `P(Y)Z = ZVVᵀ − UUᵀZVVᵀ + UUᵀZ`). DOI 10.1137/050639703, ~8 pages. (2 h) | In `src/kvdlra/projector.py`, implement `tangent_project(Y, U, V)` from Lemma 4.1, ~20 lines NumPy. (2 h) | Unit test: project a random low-rank matrix onto its own tangent space → identity up to 1e-12. |
| **Tue AM** | Lubich–Oseledets 2014, **arXiv 1301.1058, §3** — original KSL projector-splitting (K, S-backward, L). (2 h) | Read Ceruti–Lubich 2022 **arXiv 2010.02022, §1–§2** (the recap and motivation). (2 h) | One-page handwritten note: why the S-step minus-sign in KSL ⇒ instability ⇒ motivates BUG. |
| **Tue PM** | Ceruti–Lubich 2022, **§3.1** — the BUG algorithm box (K-step, L-step in parallel, then S-step forward in time). Memorize the three substeps. (1.5 h) | Write the synthetic-BUG script (**Script #1 in §3 below**), ~100 lines. (3 h) | `pytest tests/test_bug_synthetic.py -v` passes: BUG error vs. ode45 reference < 1e-6 at h=1e-3. |
| **Wed AM** | Ceruti–Kusch–Lubich 2022, **arXiv 2104.05247, §2** — augmented (rank-adaptive) BUG, Eqs. (3)–(6), truncation criterion `(Σ σ_j²)^½ ≤ ϑ`. (2 h) | Extend Script #1 to rank-adaptive BUG: stack `[K(t₁) | U₀]`, QR, augment S to 2r×2r, SVD-truncate by ϑ. (2 h) | Plot: mean rank vs. ϑ on the Lyapunov test; reproduce Fig. 1-style curve from the paper. |
| **Wed PM** | Skim Einkemmer et al. review, **arXiv 2412.05912, §2–§3** (ML-friendly review of DLRA). (1.5 h) | Read kvpress source: `kvpress/presses/base_press.py` and `snapkv_press.py`. (2 h) | Markdown notes in `docs/notes/2026-W1-bug-notes.md`. |
| **Thu AM** | StreamingLLM, **arXiv 2309.17453, §3 and Fig. 2** — the 4-sink-token observation. (1.5 h) | HF transformers tutorial: load `Llama-3.2-1B-Instruct`, run `model.generate` with `return_dict_in_generate=True, output_attentions=False`. (2 h) | Print `past_key_values[0][0].shape == (1, 8, T, 64)` for a 50-token prompt. |
| **Thu PM** | HF docs page "KV Cache" + `cache_utils.py` source — `DynamicCache.update()` signature. (1.5 h) | Write the KV-capture script (**Script #2 in §3**), ~70 lines. Subclass `DynamicCache`. (3 h) | `dumps/llama3.2-1b/c4_doc0/layer_00.pt` ... `layer_15.pt` on disk; one tensor per layer, shape `(8, T, 64)`. |
| **Fri AM** | ShadowKV, **arXiv 2410.21465, §3.1** — quote: "pre-RoPE keys are exceptionally low-rank … post-RoPE keys are not." (1.5 h) | Read HF `LlamaAttention.forward` in `modeling_llama.py` — find the line where `apply_rotary_pos_emb` precedes `past_key_values.update`. (1 h) | Note in `docs/notes/rope-pitfall.md`: explain the pitfall in 3 sentences for your README. |
| **Fri PM** | — | Write the rank-truncation comparison script (**Script #3 in §3**), ~90 lines. Run end-to-end. (3.5 h) | **First publishable figure**: `figs/sigma_decay_llama3.2-1b.pdf` — singular value decay per layer + reconstruction error vs. rank for SVD / incremental SVD / BUG. |

**End-of-week-1 deliverable:** a 1-page `docs/week1.md` with the figure, a paragraph of interpretation, the wandb run URL, and a tweet draft (§7). Commit hash this and tag it `v0.1-w1`.

### Week 2 — Rank-adaptive BUG on real KV streams + go/no-go pilot

| Day | Reading (h) | Coding (h) | Concrete output |
|---|---|---|---|
| **Mon** | Kieri–Lubich–Walach 2016 (SIAM J. Numer. Anal. 54(2):1020–1038), **§2 + Thm 2.1** — robust-error bound independent of σ_min. (2 h) | Refactor Script #1's BUG into a clean `kvdlra.integrators.BUG` class, with `step(Y, F, h)` API. Add type hints; mypy strict passes. (3 h) | `pytest tests/test_bug_class.py` (10 cases) green; mypy clean. |
| **Tue** | Ceruti–Lubich §3.2 + §3.3 (exactness proof Thm 3, robust error Thm 4). (2 h) | Adapt BUG to **streaming token-append**: each new (k_t, v_t) pair is a rank-1 update to a growing matrix; cast as Ẏ = F(Y) with F = projection of token onto orthogonal complement. (3 h) | Script `experiments/2026-w2-streaming-bug.py`: process 4096 tokens of one C4 doc through your streaming BUG, log mean rank vs. token index to wandb. |
| **Wed** | Lubich textbook Ch. IV §IV.1–IV.3 (pp. 105–117) — time-integration of variational approximations. (2 h) | Compare three rank policies on Layer 8 of Llama-3.2-1B: fixed r=16, fixed r=32, rank-adaptive ϑ=1e-2. (3 h) | Three-curve plot: Frobenius reconstruction error vs. seq_len for each policy. |
| **Thu** | OjaKV, **arXiv 2509.21623**, full paper. (2 h) | Implement Oja's-rule baseline (one-line update: `U ← orth(U + η·k·kᵀU)`) for comparison. (3 h) | Bar chart: BUG vs. Oja vs. truncated-SVD-oracle at matched memory budget. |
| **Fri** | — | **Pilot decision (§5 below)**: produce the exact go/no-go plot. Apply pass/fail criterion. (5 h) | `docs/week2-pilot.md` with verdict. Tag `v0.2-w2-pilot`. Push to GitHub. Tweet the figure (§7). |

### Week 3 — Plug into kvpress + generation correctness

| Day | Focus | Concrete output |
|---|---|---|
| **Mon** | Read `kvpress` `BasePress` + `ScorerPress` + `ExpectedAttentionPress` source. Read `notebooks/new_press.ipynb`. (4 h reading, 2 h prototyping) | Skeleton `BUGPress(BasePress)` class in `src/kvdlra/press/bug_press.py`. |
| **Tue** | Wire `BUGPress.forward_hook` to (a) accumulate per-layer KV into your streaming BUG state, (b) replace `cache_layer.keys`/`cache_layer.values` with the low-rank reconstruction `UₜSₜVₜᵀ` after the prefill forward. (6 h) | `python scripts/generate_with_press.py --press bug --rank 32` returns text (correctness only; quality next). |
| **Wed** | Generation parity check: compare `BUGPress` output to no-press output on 10 short prompts at greedy decode. (4 h) Read SnapKV, **arXiv 2404.14469, §3–§4**. (2 h) | Markdown table of greedy outputs side-by-side. Bug-bash any divergences. |
| **Thu** | Set up `lm-eval-harness` (v0.4.11) with `--tasks wikitext` for perplexity. Run baseline (no press), then `BUGPress` at rank 16, 32, 64. (6 h) | Wandb run group `w3-perplexity-sweep` with 4 runs; CSV in `results/w3-ppl.csv`. |
| **Fri** | Pre-RoPE keys experiment: monkey-patch `LlamaAttention.forward` to stash pre-RoPE K; rerun BUG on pre-RoPE K and compare reconstruction error to post-RoPE K. (6 h) | Plot reproducing ShadowKV's claim with your numbers. Decide whether BUGPress should operate pre- or post-RoPE. |

### Week 4 — TurboQuant residual quantization + writeup

| Day | Focus | Concrete output |
|---|---|---|
| **Mon** | Read TurboQuant, **arXiv 2504.19874** — §2 (PolarQuant rotation), §3 (QJL residual). Note: Π and RoPE do **not** commute trivially. (4 h reading, 2 h notes) | `docs/notes/turboquant-rope-interaction.md`: 1-page derivation of how Π must compose with RoPE. |
| **Tue** | Implement PolarQuant: sample fixed Π via QR of Gaussian, scalar-quantize at 3.5 bits/dim with Lloyd–Max levels. (6 h) | Unit test: quantization MSE within 2.7× of the information-theoretic floor (TurboQuant's claim). |
| **Wed** | Implement QJL residual: 1-bit sign hash, unbiased inner-product estimator. (6 h) | Test: ⟨q, k_hat⟩ estimator unbiased; variance matches §3 bound to 10%. |
| **Thu** | Compose BUG → low-rank reconstruction → quantize the (U, S, V) factors with TurboQuant. (6 h) | Run perplexity sweep at total memory budgets {0.25×, 0.5×, 1×} of full-cache. Wandb run group `w4-hybrid`. |
| **Fri** | Write `README.md` v1 with hero figure. Draft v0.1 of the arXiv-style report under `paper/main.tex` using NeurIPS template. (6 h) | Tag `v0.3-w4-hybrid`. Push README + first blog post (§7). |

---

## 2. Repository bootstrap (copy-paste sequence)

```bash
# ---------- 0. Local prerequisites ----------
# Install GitHub CLI: https://cli.github.com/  ; then:
gh auth login
# Install uv (Astral package manager):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# ---------- 1. Create the repo skeleton ----------
mkdir kvdlra && cd kvdlra
git init -b main

mkdir -p src/kvdlra/{integrators,press,utils} \
         tests \
         scripts \
         experiments \
         configs/{model,press,eval} \
         docs/notes \
         figs results dumps paper

touch src/kvdlra/__init__.py \
      src/kvdlra/integrators/__init__.py \
      src/kvdlra/press/__init__.py \
      src/kvdlra/utils/__init__.py \
      tests/__init__.py
echo '__version__ = "0.1.0"' > src/kvdlra/__init__.py

# ---------- 2. .gitignore ----------
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.mypy_cache/
.ruff_cache/
.pytest_cache/
.env
.env.*
wandb/
dumps/*.pt
dumps/**/*.pt
*.ckpt
*.safetensors
results/*.csv
figs/*.pdf
figs/*.png
!figs/.gitkeep
!dumps/.gitkeep
!results/.gitkeep
EOF
touch figs/.gitkeep dumps/.gitkeep results/.gitkeep

# ---------- 3. pyproject.toml ----------
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "kvdlra"
version = "0.1.0"
description = "Dynamical low-rank approximation for streaming KV-cache compression in LLMs."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = [
  "torch==2.11.0",
  "transformers==5.8.0",
  "accelerate==1.13.0",
  "datasets==4.8.5",
  "huggingface_hub==1.14.0",
  "kvpress==0.5.1",
  "hydra-core==1.3.2",
  "omegaconf>=2.3",
  "wandb==0.26.1",
  "python-dotenv",
  "matplotlib>=3.8",
  "numpy>=1.26",
  "scipy>=1.13",
  "tqdm",
]

[project.optional-dependencies]
dev = [
  "pytest==9.0.3", "pytest-cov",
  "ruff==0.15.12", "mypy==2.0.0", "pre-commit",
  "mkdocs-material==9.8.0", "mkdocstrings[python]==1.0.4",
  "mkdocs-jupyter", "pymdown-extensions",
]
eval  = ["lm-eval[vllm]==0.4.11"]
flash = ["flash-attn"]            # install with --no-build-isolation separately

[tool.hatch.build.targets.wheel]
packages = ["src/kvdlra"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF", "N", "C4"]

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["wandb.*", "kvpress.*", "flash_attn.*", "transformers.*", "datasets.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
EOF

# ---------- 4. pre-commit ----------
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=1024"]
      - id: check-merge-conflict
      - id: debug-statements
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.12
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v2.0.0
    hooks:
      - id: mypy
        additional_dependencies: ["torch==2.11.0", "types-PyYAML"]
EOF

# ---------- 5. README skeleton ----------
cat > README.md <<'EOF'
# kvdlra

Streaming KV-cache compression for LLMs via Dynamical Low-Rank Approximation
(Ceruti–Lubich BUG integrator) with optional TurboQuant residual quantization.

Status: **week 1**. See `docs/week1.md`.
EOF

# ---------- 6. MkDocs Material ----------
cat > mkdocs.yml <<'EOF'
site_name: kvdlra
site_url: https://YOURUSER.github.io/kvdlra/
repo_url:  https://github.com/YOURUSER/kvdlra
theme:
  name: material
  features: [navigation.tabs, navigation.sections, content.code.copy, search.suggest]
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      toggle: { icon: material/brightness-7, name: Dark }
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      toggle: { icon: material/brightness-4, name: Light }
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            members_order: source
  - mkdocs-jupyter: { include_source: true }
markdown_extensions:
  - admonition
  - attr_list
  - toc: { permalink: true }
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - pymdownx.snippets
nav:
  - Home: index.md
  - Notes: notes/
  - API: reference.md
EOF
mkdir -p docs
echo "# kvdlra" > docs/index.md
echo "# API\n\n::: kvdlra" > docs/reference.md

# ---------- 7. Environment + venv ----------
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
# CUDA 12.8 torch wheels are most compatible with flash-attn on H100:
uv pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
# Skip flash-attn on Mon AM laptop bootstrap; install it on the H100 pod (§6).

pre-commit install
pre-commit run --all-files   # may auto-fix; commit the fixes

# ---------- 8. dotenv + wandb ----------
cat > .env.example <<'EOF'
WANDB_API_KEY=
WANDB_ENTITY=YOURUSER
WANDB_PROJECT=kvdlra
HF_TOKEN=
EOF
cp .env.example .env   # then edit .env, never commit it
wandb login            # paste your key once; stored in ~/.netrc

# ---------- 9. First commit + create remote ----------
git add .
git commit -m "Initial scaffold: pyproject, pre-commit, mkdocs, package skeleton"
gh repo create kvdlra --public --source=. --remote=origin --push \
   --description "Streaming KV-cache compression via Dynamical Low-Rank Approximation + TurboQuant"
```

After this block your repo has a green tree, a working venv, a working pre-commit, a working MkDocs site (`mkdocs serve` to preview), and one commit on `origin/main`.

---

## 3. The three first-week scripts

### Script #1 — `tests/test_bug_synthetic.py`: from-scratch BUG on a matrix Lyapunov flow

This is the Ceruti–Lubich §6 numerical experiment, simplified. **Pass criterion: BUG error vs. ode45-grade reference < 1e-5** at h=1e-3 on the n=100, r=8 problem.

```python
"""From-scratch BUG integrator (Ceruti–Lubich 2022, §3.1) on a Lyapunov flow.

Reference: arXiv 2010.02022, eqs. in §3.1. Test problem: §6 / Ceruti–Kusch–Lubich §5.1.
    Y' = -H[Y],   H[Y] = (V - 0.5 D) Y + Y (V - 0.5 D)^T,
    D = tridiag(-1, 2, -1),   V = diag{1 - cos(2 pi j / n)}.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp


def build_operator(n: int) -> np.ndarray:
    D = np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    j = np.arange(-n // 2, n // 2)
    V = np.diag(1.0 - np.cos(2 * np.pi * j / n))
    return V - 0.5 * D  # B; then F(Y) = -(B Y + Y B^T)


def F(Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    return -(B @ Y + Y @ B.T)


def rk2_step(Y0: np.ndarray, rhs, h: float) -> np.ndarray:
    k1 = rhs(Y0)
    k2 = rhs(Y0 + h * k1)
    return Y0 + 0.5 * h * (k1 + k2)


def bug_step(U: np.ndarray, S: np.ndarray, V: np.ndarray, B: np.ndarray, h: float):
    """One BUG step (Ceruti–Lubich 2022, §3.1). All substep ODEs solved with RK2."""
    # K-step: K' = F(K V^T) V, K(0) = U S
    K0 = U @ S
    K1 = rk2_step(K0, lambda K: F(K @ V.T, B) @ V, h)
    U1, _ = np.linalg.qr(K1)
    M = U1.T @ U  # r x r

    # L-step: L' = F(U L^T)^T U, L(0) = V S^T
    L0 = V @ S.T
    L1 = rk2_step(L0, lambda L: F(U @ L.T, B).T @ U, h)
    V1, _ = np.linalg.qr(L1)
    N = V1.T @ V  # r x r

    # S-step (forward): S' = U1^T F(U1 S V1^T) V1, S(0) = M S N^T
    S0 = M @ S @ N.T
    S1 = rk2_step(S0, lambda Sm: U1.T @ F(U1 @ Sm @ V1.T, B) @ V1, h)
    return U1, S1, V1


def test_bug_tracks_reference() -> None:
    rng = np.random.default_rng(0)
    n, r, T, h = 100, 8, 0.1, 1e-3
    B = build_operator(n)

    # initial Y0 = U0 S0 V0^T with geometric singular values 10^{-i}
    U0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    S0 = np.diag(10.0 ** (-np.arange(r)))
    Y0 = U0 @ S0 @ V0.T

    # Dense reference (ode45-grade): solve full n^2 ODE
    def rhs(_t, y):
        Y = y.reshape(n, n)
        return F(Y, B).reshape(-1)
    sol = solve_ivp(rhs, (0.0, T), Y0.reshape(-1), method="RK45",
                    rtol=1e-10, atol=1e-12, t_eval=[T])
    Y_ref = sol.y[:, -1].reshape(n, n)

    # BUG integration
    U, S, V = U0, S0, V0
    nsteps = int(round(T / h))
    for _ in range(nsteps):
        U, S, V = bug_step(U, S, V, B, h)
    Y_bug = U @ S @ V.T

    err = np.linalg.norm(Y_bug - Y_ref, ord="fro")
    print(f"BUG error vs. reference: {err:.3e}")
    assert err < 1e-5, f"BUG diverged: {err}"


if __name__ == "__main__":
    test_bug_tracks_reference()
    print("OK")
```

Run: `pytest tests/test_bug_synthetic.py -v`. This script alone confirms you have the algorithm right.

### Script #2 — `scripts/capture_kv.py`: dump Llama-3.2-1B KV per layer to `.pt`

```python
"""Capture per-layer K/V tensors during prefill on a C4 document.

Output: dumps/llama3.2-1b/<slug>/layer_{i:02d}.pt  with dict
        {"K": (num_kv_heads, T, head_dim), "V": same, "input_ids": (T,)}.
"""
from __future__ import annotations
import argparse, os, hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset


class CapturingCache(DynamicCache):
    """DynamicCache that snapshots K, V at every layer after each update."""
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.snapshots: dict[int, dict[str, torch.Tensor]] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_full, v_full = super().update(key_states, value_states, layer_idx, cache_kwargs)
        # store CPU copies of the full prefill tensors (we only call once with full prompt)
        self.snapshots[layer_idx] = {"K": k_full.detach().to("cpu"),
                                     "V": v_full.detach().to("cpu")}
        return k_full, v_full


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--doc_idx", type=int, default=0)
    p.add_argument("--out", default="dumps/llama3.2-1b")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",  # required to keep cache plumbing simple
    )
    model.eval()

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    doc = next(x for i, x in enumerate(ds) if i == args.doc_idx)
    ids = tok(doc["text"], return_tensors="pt", truncation=True,
              max_length=args.seq_len).input_ids.to("cuda")

    cache = CapturingCache(config=model.config)
    with torch.no_grad():
        _ = model(input_ids=ids, past_key_values=cache, use_cache=True)

    slug = hashlib.md5(doc["text"][:200].encode()).hexdigest()[:8]
    out_dir = Path(args.out) / f"doc{args.doc_idx}_{slug}_len{ids.size(1)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, kv in cache.snapshots.items():
        # squeeze batch dim -> (num_kv_heads, T, head_dim)
        torch.save({"K": kv["K"].squeeze(0), "V": kv["V"].squeeze(0),
                    "input_ids": ids.squeeze(0).cpu()},
                   out_dir / f"layer_{layer_idx:02d}.pt")
    (out_dir / "meta.json").write_text(
        f'{{"model":"{args.model}","seq_len":{ids.size(1)},"doc_idx":{args.doc_idx}}}'
    )
    print(f"Wrote {len(cache.snapshots)} layer dumps to {out_dir}")


if __name__ == "__main__":
    main()
```

Run on the H100: `python scripts/capture_kv.py --seq_len 4096`. Output is ~128 MiB total.

### Script #3 — `scripts/sigma_decay.py`: SVD vs. incremental SVD vs. BUG, plot

```python
"""Compare reconstruction error on captured KV: truncated SVD vs. naive
incremental SVD vs. BUG, at fixed ranks. Produces figs/sigma_decay.pdf.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def truncated_svd_recon(M: np.ndarray, r: int) -> float:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    Mr = (U[:, :r] * s[:r]) @ Vt[:r]
    return float(np.linalg.norm(M - Mr, ord="fro") / np.linalg.norm(M, ord="fro"))


def incremental_svd_recon(M: np.ndarray, r: int) -> float:
    """Brand-style incremental SVD: process columns of M^T one by one, keep top-r."""
    U = np.zeros((M.shape[0], 0)); s = np.zeros(0); Vt = np.zeros((0, 0))
    for j in range(M.shape[1]):
        c = M[:, j:j+1]
        if U.size == 0:
            U, sj, Vtj = np.linalg.svd(c, full_matrices=False)
            s, Vt = sj, np.pad(Vtj, ((0, 0), (0, j)))[:, :j+1]
            Vt[:, -1] = Vtj[:, 0]
            continue
        # project + residual
        m = U.T @ c
        p = c - U @ m
        Ra = np.linalg.norm(p)
        P = p / (Ra + 1e-12)
        # build K
        K = np.block([[np.diag(s), m], [np.zeros((1, s.size)), np.array([[Ra]])]])
        Up, sp, Vtp = np.linalg.svd(K, full_matrices=False)
        # absorb back
        U = np.hstack([U, P]) @ Up
        s = sp
        # extend Vt
        Vt_new = np.block([[Vt, np.zeros((Vt.shape[0], 1))],
                           [np.zeros((1, Vt.shape[1])), np.array([[1.0]])]])
        Vt = Vtp @ Vt_new
        if s.size > r:
            U, s, Vt = U[:, :r], s[:r], Vt[:r, :]
    Mr = (U * s) @ Vt
    return float(np.linalg.norm(M - Mr, ord="fro") / np.linalg.norm(M, ord="fro"))


def bug_recon(M: np.ndarray, r: int, h: float = 1.0) -> float:
    """Treat columns of M as a token stream; one BUG step per token, F(Y)=Y_target-Y."""
    n, T = M.shape
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V, _ = np.linalg.qr(rng.standard_normal((T, r)))
    S = np.zeros((r, r))
    Y_target = M
    # one Euler-BUG sweep: F(Y) = (Y_target - Y)
    def F(Y): return Y_target - Y
    # K-step
    K = U @ S + h * F(U @ S @ V.T) @ V
    U1, _ = np.linalg.qr(K); Mm = U1.T @ U
    # L-step
    L = V @ S.T + h * F(U @ S @ V.T).T @ U
    V1, _ = np.linalg.qr(L); Nn = V1.T @ V
    # S-step
    S0 = Mm @ S @ Nn.T
    S1 = S0 + h * U1.T @ F(U1 @ S0 @ V1.T) @ V1
    Mr = U1 @ S1 @ V1.T
    return float(np.linalg.norm(M - Mr, ord="fro") / np.linalg.norm(M, ord="fro"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="dir of layer_XX.pt files")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--out", default="figs/sigma_decay.pdf")
    args = p.parse_args()

    blob = torch.load(Path(args.dump) / f"layer_{args.layer:02d}.pt", weights_only=False)
    # Stack heads: K is (H, T, D); flatten head dim into rows -> (H*D, T)
    K = blob["K"].float().numpy()
    H, T, D = K.shape
    M = K.transpose(0, 2, 1).reshape(H * D, T)  # rows=features, cols=tokens

    # Singular value decay
    s = np.linalg.svd(M, compute_uv=False)
    ranks = [4, 8, 16, 32, 64, 128]
    ranks = [r for r in ranks if r < min(M.shape)]

    err_svd  = [truncated_svd_recon(M, r) for r in ranks]
    err_isvd = [incremental_svd_recon(M, r) for r in ranks]
    err_bug  = [bug_recon(M, r) for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(s / s[0]); axes[0].set_xlabel("index")
    axes[0].set_ylabel(r"$\sigma_i / \sigma_1$")
    axes[0].set_title(f"Layer {args.layer} K-cache singular values")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(ranks, err_svd,  "o-", label="truncated SVD (oracle)")
    axes[1].semilogy(ranks, err_isvd, "s--", label="incremental SVD")
    axes[1].semilogy(ranks, err_bug,  "^:", label="BUG (1 step, Euler)")
    axes[1].set_xlabel("rank r"); axes[1].set_ylabel("rel. Frobenius error")
    axes[1].set_title("Reconstruction error vs. rank")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

Run: `python scripts/sigma_decay.py --dump dumps/llama3.2-1b/doc0_*_len4096 --layer 8 --out figs/sigma_decay_layer08.pdf`. **This is your week-1 figure.**

---

## 4. Experimental logging strategy

**wandb structure.** One project: `kvdlra`. One run per (script invocation × seed). Run-name convention: `{week}-{phase}-{model}_{press}_{rank|cr}_{seed}` — e.g. `w2-pilot-llama3.2-1b_bug_r32_s0`. Tags: model, phase, hardware. Always pass the full Hydra config to `wandb.init(config=...)` so you can filter post-hoc. Log scalars (`wandb.log({"frob_err": e, "mean_rank": r, "step": t})`). Log every figure as a `wandb.Image`. Log every output `.pt` over 5 MB as a `wandb.Artifact` (type `kvdump` or `reconstruction`), never as `wandb.save` (which copies them into the run dir and clutters disk).

**Lab notebook discipline.** Every experiment lives at `experiments/2026-MM-DD-<slug>/README.md` with seven mandatory headings: **Hypothesis** (one sentence), **Setup** (model, seq_len, ranks, hardware), **Command** (the literal CLI line), **Wandb URL**, **Result** (numbers and figure paths), **Interpretation** (what you concluded), **Decision** (what changes next). Keep these short — half a page each, never more than one. Treat the directory itself as a logbook: at end of week, `ls experiments/` should read like a diary.

**Config naming.** Hydra configs in `configs/`: `configs/experiment/w2-pilot.yaml` references `configs/model/llama3.2-1b.yaml`, `configs/press/bug.yaml`, `configs/eval/wikitext.yaml`. Every config gets a `version: <git-sha>` field auto-populated by a tiny resolver:

```python
# src/kvdlra/utils/hydra_resolvers.py
import subprocess
from omegaconf import OmegaConf
def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
OmegaConf.register_new_resolver("git_sha", _git_sha)
```

In every YAML: `version: ${git_sha:}`. This means **every figure traces back to a commit hash** — the single most important reproducibility hygiene rule, and the one that pays off when arXiv reviewers ask "which version of your code produced Fig. 3?"

**Seeding everything.**

```python
# src/kvdlra/utils/seed.py
import os, random, numpy as np, torch
def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

**Determinism caveats on H100:** flash-attn is **not bit-deterministic** across runs (different reduction orders); bf16 matmuls reorder differently across CUDA stream schedules; tensor-parallel sharding can change numerics. For DLRA reproducibility, run the core BUG step in **fp32 on CPU/GPU** and only the model forward in bf16. Document this everywhere.

**Where things go.** wandb: scalar metrics, small plots, run configs. Disk (`dumps/`, `results/`): large `.pt` files and CSVs that are too big or too numerous for wandb's free tier. HuggingFace Hub dataset (`you/kvdlra-kvcache-llama3.2-1b`): canonical published KV dumps for reproducibility. Git: code, configs, small `figs/*.pdf`, notebook READMEs. Never commit a tensor.

---

## 5. The week-2 go/no-go pilot

**The single plot.** X-axis: prescribed memory budget as a fraction of full-cache (0.05, 0.10, 0.20, 0.40, 0.80, 1.0). Y-axis: relative Frobenius reconstruction error on Layer 8 of Llama-3.2-1B's K-cache, averaged over 5 C4 documents at seq_len=4096. Three curves: (1) **truncated SVD oracle** — the irreducible lower bound at each rank, (2) **incremental SVD** (Brand-style) — your "naive streaming baseline", (3) **rank-adaptive BUG** with tolerance ϑ swept to match each memory budget. Shaded bands for 1-sigma over docs.

**Pass criterion (formal).** Rank-adaptive BUG reaches Frobenius reconstruction error within **1.05×** of the truncated-SVD oracle at the same memory budget, **at mean effective rank ≤ 0.9·r_oracle**, on at least 4 of 5 documents and for all budgets ≥ 0.10. If this holds, BUG is genuinely better than incremental SVD at matched memory — go.

**Fail diagnostics, in order.** (a) Print mean rank trajectory; if BUG's rank is saturating at 2r the truncation tolerance ϑ is too loose. (b) Check the K matrix is post-RoPE — if so, redo the experiment on pre-RoPE keys (monkey-patch attention to stash them) and re-test; ShadowKV's claim is that pre-RoPE is dramatically more compressible. (c) Verify F is correctly defined for the streaming setting (Y_target = current accumulated KV, not the new token alone). (d) Try a different layer (early layers are typically lower-rank). (e) If still failing on pre-RoPE keys at layer 4 — the project's core premise is weak; **pivot** to a different DLRA formulation (e.g., column-streaming instead of full-matrix-tracking) or shelve and write up the negative result.

---

## 6. Development workflow

**Editor.** VS Code (free, mature Remote-SSH) or Cursor (better AI integration; reads the same `~/.ssh/config`). Required extensions installed on the remote, not locally: `ms-python.python`, `ms-python.vscode-pylance`, `charliermarsh.ruff`, `ms-toolsai.jupyter`, `eamodio.gitlens`, `tamasfe.even-better-toml`, `redhat.vscode-yaml`, `github.copilot`. Settings: enable "format on save" with ruff, set `python.testing.pytestEnabled: true`.

**RunPod setup.** First, locally: `ssh-keygen -t ed25519 -C "you@host" -f ~/.ssh/id_ed25519` and paste the public key into RunPod **Settings → SSH Public Keys**. Then create a **Network Volume** (100 GB in your region) under **Storage → New Network Volume** — this is the trick that lets you stop the pod without losing your venv. Deploy: **Pods → Deploy → H100 PCIe 80 GB → Community Cloud (~$1.99/hr) → template `runpod/pytorch:2.11.0-py3.12-cuda12.8.0-devel-ubuntu24.04` → attach the network volume at `/workspace` → tick SSH Terminal Access → Deploy On-Demand**. RunPod shows an SSH line like `ssh root@69.48.x.x -p 25634 -i ~/.ssh/id_ed25519`. Add it to `~/.ssh/config`:

```sshconfig
Host kvdlra-h100
    HostName 69.48.x.x
    User root
    Port 25634
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

In VS Code: Cmd+Shift+P → "Remote-SSH: Connect to Host" → `kvdlra-h100` → open `/workspace`.

**First-time install on the pod** (after `git clone` into `/workspace/kvdlra`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd /workspace/kvdlra
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash-attn --no-build-isolation     # ~5 min build on H100
huggingface-cli login   # paste HF_TOKEN
wandb login
python -c "import torch; print(torch.cuda.get_device_name(0))"   # expect H100
```

**Stopping the pod.** Click **Stop** in the RunPod UI (or `runpodctl stop pod $RUNPOD_POD_ID`). Compute billing halts; storage continues at $0.20/GB/mo for the container disk plus $0.07/GB/mo for the network volume — roughly $0.50/day for a 100 GB volume. Restart with **Start**; the SSH endpoint may change, so update `~/.ssh/config` accordingly. **Never use Spot for runs longer than a checkpoint interval** — they can be reclaimed without warning.

**Sync discipline.** Code: `git push`/`git pull` only. KV dumps: push the canonical small ones to a HF Hub dataset repo (`hf upload you/kvdlra-kvcache-llama3.2-1b ./dumps . --repo-type dataset`); everything else stays on `/workspace`. Plots and CSVs: log to wandb as artifacts. Never `scp` data between laptop and pod — use HF Hub or wandb as the source of truth.

**A typical day.** Wake up, `ssh kvdlra-h100`, `tmux attach -t work` (or `tmux new -s work`), `cd /workspace/kvdlra && source .venv/bin/activate && git pull`, `uv pip install -e ".[dev]"` (no-op if unchanged), run the day's script (`python scripts/...`), watch wandb in browser, `Ctrl-b d` to detach tmux when going for lunch, come back and check the run, write your `experiments/2026-MM-DD-slug/README.md` entry, `git add . && git commit -m "..." && git push`, **Stop** the pod in the RunPod UI. Total active SSH time ≈ 30 minutes for a day with a 4-hour run.

---

## 7. The first three public artifacts

### (a) The README "what is this" section

Drop this verbatim into `README.md` once the week-1 figure exists:

> **kvdlra** explores whether *Dynamical Low-Rank Approximation* — the Ceruti–Lubich BUG integrator from numerical analysis (arXiv 2010.02022, arXiv 2104.05247) — provides a principled, streaming-friendly alternative to greedy heuristics like H2O or SnapKV for compressing the KV cache of decoder-only LLMs. DLRA tracks a moving low-rank matrix without the σ_min stiffness that plagues naive (U, S, V) ODEs, with a robust error bound *independent of the smallest singular value*. We pair the rank-adaptive BUG integrator with TurboQuant (arXiv 2504.19874) residual quantization for an end-to-end ~6× compression target.
>
> **Status.** Week-1 milestone reached: K-cache singular-value spectra and reconstruction-error curves reproducible from `scripts/sigma_decay.py`. Figure: `figs/sigma_decay_layer08.pdf`.

### (b) First blog post / Twitter thread

> 1/ The KV cache of a modern LLM is mostly low-rank — but how you *track* that low-rank structure as tokens stream in matters. Most KV compression methods are eviction heuristics. There's a 20-year-old line of numerical-analysis work that gives you a principled streaming alternative: Dynamical Low-Rank Approximation (DLRA).
>
> 2/ Here's the singular-value decay of Llama-3.2-1B's layer-8 key cache on 4K tokens of C4. The 95th percentile of mass is captured by ~r=32 out of 512 features. (figure: `figs/sigma_decay_layer08.pdf`)
>
> 3/ Naive incremental SVD works but is locally greedy. The Ceruti–Lubich "BUG" integrator (arXiv 2010.02022) tracks the low-rank manifold with a robust error bound that's *independent of σ_min* — the curvature blow-up that breaks textbook (U, S, V) ODE schemes.
>
> 4/ Week 1 of an open-source project: I implemented BUG from scratch on the Lyapunov benchmark from §6 of the paper (error < 1e-5 vs. ode45), then ran it on the Llama-3.2-1B KV cache. Code, configs, and figure all reproducible: github.com/YOURUSER/kvdlra
>
> 5/ Next: hook BUG into NVIDIA's kvpress framework, compare against SnapKV / ExpectedAttention, then layer TurboQuant residual quantization on top for ~6× total compression. Weekly updates here.

### (c) The OpenReview / HuggingFace Papers comment

Post on the OjaKV paper page (OpenReview forum `XVjgvJhLTY`):

> Nice work. Have you considered the Ceruti–Lubich "BUG" / rank-adaptive BUG integrator (arXiv 2010.02022, arXiv 2104.05247) as an alternative to Oja's rule for the online subspace update? BUG comes with a robust error bound independent of the smallest singular value (Kieri–Lubich–Walach, SIAM J. Numer. Anal. 54(2), 2016, §2) and is rank-adaptive via the Frobenius-tail truncation criterion `(Σ σ_j²)^½ ≤ ϑ`. I'm exploring this combination — including pre-RoPE keys per ShadowKV — at github.com/YOURUSER/kvdlra; happy to compare on your benchmark setup if you share the eval scripts.

---

## 8. Week-1 mistakes to avoid

The first pitfall is **getting the cache shape backwards**. HuggingFace `DynamicCache` stores K and V as `(batch, num_kv_heads, seq_len, head_dim)` for both tensors, confirmed in `cache_utils.py` and in the kvpress source: *"Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim)."* Several research libraries (vLLM, megatron) use different orderings; do not copy-paste reshape code across them without checking.

The second pitfall is **forgetting GQA**. Llama-3.2-1B has `num_attention_heads=32` but `num_key_value_heads=8` — there are only 8 KV heads, each shared by 4 query heads. Many KV-budget calculations are off by 4× because authors used `num_attention_heads`. The correct per-token KV bytes for Llama-3.2-1B in bf16 are `2 (K,V) × 16 layers × 8 KV heads × 64 head_dim × 2 bytes = 32 768 bytes/token = 32 KiB/token`.

The third pitfall is **applying SVD to post-RoPE keys**. HuggingFace's `LlamaAttention.forward` applies `apply_rotary_pos_emb` *before* `past_key_values.update(...)`, so the cached K is post-RoPE — and RoPE smears low-rank structure across positions. ShadowKV (arXiv 2410.21465 §3.1) shows pre-RoPE keys are dramatically more compressible; if you SVD post-RoPE K you'll see disappointing decay curves and wrongly conclude DLRA doesn't help. Either (a) monkey-patch attention to stash pre-RoPE K, or (b) apply the inverse rotation per position before factoring.

The fourth pitfall is **fp16 in the BUG core**. The K-step and L-step each end with a QR factorization; QR on fp16 matrices with condition numbers > 1e3 (typical for KV streams) loses 2–3 decimal digits per step. Cast the BUG factors to **fp32** (or at minimum bf16, which has fp32's exponent range) and do the QR there; cast back to bf16 only for storage. The model forward stays bf16; only the integrator's linear algebra is fp32.

The fifth pitfall is **including attention sinks in the low-rank track**. StreamingLLM (arXiv 2309.17453 Fig. 2) shows the first 4 tokens behave anomalously across all layers and heads. Their K and V vectors are outliers in any low-rank basis and inflate your reconstruction error. **Exclude the first 4 positions** from the BUG state and store them full-rank as a separate "sink buffer", exactly as StreamingLLM does for window attention.

The sixth pitfall is **convention drift between the math and the code**. Ceruti–Lubich write `Y = U S Vᵀ` with U ∈ ℝ^(m×r), S ∈ ℝ^(r×r), V ∈ ℝ^(n×r), columns-of-V being basis vectors of the row space — column-vector / right-multiplication conventions. HuggingFace's K tensor is `(num_kv_heads, T, head_dim)`; if you want a 2D matrix to factor, you must pick a convention: rows = features (`head_dim` × `num_kv_heads`), columns = tokens (`T`) is the cleanest because then "new token" = "new column", matching the streaming math. Write this translation table in `docs/notes/conventions.md` on day 1.

The seventh pitfall is **picking too big a model too soon**. Llama-3.1-8B's KV cache is **4 GiB per batch at 32K context** in bf16 (verified math: 2 × 32 × 8 × 32768 × 128 × 2). With activations, model weights (16 GiB), and the BUG state, a single 32K experiment can OOM your 80 GB H100. **Do all algorithmic development on Llama-3.2-1B at 4K (128 MiB cache)**; only scale up after BUGPress passes its parity tests in week 3.

---

## 9. Three external checkpoints

**End of week 1 — "synthetic BUG works."** Commit: tag `v0.1-w1`. README: add the singular-value-decay figure under a new "Week 1" section with a paragraph of interpretation. Tweet: post the thread in §7(b). What to include: the figure, the pytest output showing `BUG error vs. reference: 3.2e-06 OK`, and one paragraph on why this is interesting. Audience signal you're looking for: an ML researcher quote-tweets with "wait, the BUG bound is *independent of σ_min*?" — that means you have the right framing.

**End of week 2 — "KV singular value decay + pilot verdict."** Commit: tag `v0.2-w2-pilot`. README: add the go/no-go figure with axes (memory budget vs. rel. Frobenius error, three curves), the verdict ("PASS: BUG within 1.04× of oracle at 0.20 budget on 5/5 docs"), and the wandb run group URL. Blog post: 800-word writeup on Hugging Face Papers or a personal blog: "DLRA vs. incremental SVD on a real LLM KV cache" with the plot. OpenReview comment on the OjaKV paper per §7(c). What to add to the README: a "Reproducibility" section with the exact `python scripts/pilot.py --config configs/experiment/w2-pilot.yaml` invocation and the wandb URL.

**End of week 4 — "compressed-cache generation matches full-cache within X%."** Commit: tag `v0.3-w4-hybrid`. README: hero figure (perplexity vs. compression ratio for SnapKV, ExpectedAttention, BUGPress, BUGPress+TurboQuant), with the target claim "**BUG+TurboQuant achieves WikiText perplexity within 2% of full-cache at 4× compression** on Llama-3.2-1B-Instruct" (adjust the number to your actual result; if you missed 2%, report what you got — honesty beats hype). Tweet: short thread, "4 weeks in: here's where DLRA lands as a KV-compression primitive." Add a new section to README "What works, what doesn't" listing the honest limitations: pre-RoPE-only, batch size effects, perplexity-only (not yet LongBench), 1B-only. Submit a short workshop note (4 pages) to the next ICLR/NeurIPS workshop on efficient LLMs. Begin drafting the full arXiv preprint with `paper/main.tex` using the NeurIPS LaTeX template.

---

## Conclusion: the smallest viable path

The non-obvious move in this plan is **doing the synthetic Ceruti–Lubich Lyapunov test before touching a single LLM weight**. It takes ~3 hours, and it inoculates you against the most common failure mode of beginners in this area: misimplementing the BUG substep ordering (parallel K/L, then forward-in-time S) and not knowing it. Once the synthetic test passes, every later integration error is localized to the HF plumbing, not the math. The second non-obvious move is **excluding the first 4 sink tokens from day one** — most KV-compression papers had to retrofit this; you start with it. The third is **doing all of week 1–2 on Llama-3.2-1B at 4K** — 128 MiB of KV cache, fits anywhere, debugs in seconds. By the time you scale to Llama-3.1-8B in week 4 your code is correct, your conventions are documented, and your figures are reproducible from a config hash. That is the project; everything else is a sweep.
