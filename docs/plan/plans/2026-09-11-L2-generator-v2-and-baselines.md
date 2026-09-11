# L2 — Generator v2 and Baseline Honesty: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed 10-sentence cycled haystack with a generator that uses ≥ 4 real-text sources, a balanced needle-depth design and ≥ 2 code families, records everything a reviewer needs per trial, and pairs byte-identically across arms; add a faithful KIVI arm (G=32, R=128, full-precision prefill) next to the labelled streaming one; rename the SVD-oracle arm; add the matched-budget eviction/structured baselines; wire OjaKV end-to-end; and pre-register the ss2-across-families and smoke pods. The Day-1 filler-realism diagnostic is already prepared on this lane (`prereg/filler_realism.md`, `scripts/pod/w21.sh`) and waits on DECISIONS D-005.

**Architecture:** `kvdlra/eval/gen.py` is the only prompt factory: `make_trial(task_cfg, tok, task, seed, trial) -> Trial(prefill_ids, query_ids, targets, meta)` where `meta` carries `haystack_id, depth, code_family, prompt_sha256`; the trial index enumerates a balanced (haystack × depth × code-family) design deterministically, the seed drives only the random draws. Haystack corpora are materialized once per pod into `data/haystacks/<source>.jsonl` (SHA256 recorded in the manifest) by `scripts/pod.py prepare`; CPU tests use a bundled 60 KB fixture corpus. The faithful KIVI arm prefills with a plain fp16 `DynamicCache` and quantizes the store afterwards in 4096-token chunks (identical per-channel scales to whole-prompt quantization because groups are 32 consecutive tokens), keeping the last `R` tokens in fp16. Baseline presses come from kvpress 0.5.1 (`ComposedPress` for ThinK+SnapKV). Everything is a YAML arm/task under `configs/`.

**Tech Stack:** torch 2.11, transformers 5.8.0 (`DynamicCache`, `QuantizedCache`), optimum-quanto / hqq, kvpress 0.5.1, `datasets` 2.21 (streaming), omegaconf configs and `TrialRecord` from L0.

**Spec:** `docs/plan/lanes/L2_generator_v2_and_baselines.md`; `docs/plan/ICML2027_PLAN.md` §2 (2.1 generator, 2.2 faithful KIVI, 2.3 baseline honesty pass); `docs/plan/CODE_AUDIT.md` Part B §Q1 (KIVI deviations table), §Q2 (Palu = per-sequence SVD oracle), §Q5 (no run at k≈0.15; ea-k0.25 omitted), §Q9 (generator internals); `docs/plan/lanes/GATES.md` §G2.

## Global Constraints

- Worktree `.claude/worktrees/L2-generator-v2`, branch `lane/L2-generator-v2` (already exists with the two Day-1 commits; rebase onto `week7` after L0 Phase A merges before starting Task 1). Post-L0 paths are assumed (`kvdlra/eval/{config,records,ruler,runner}.py`, `scripts/pod.py`, `configs/`).
- Forbidden words in new code/configs/docs/commits: `DLRA` (word), `BUG integrator`, `honest(ly)`, `marquee`, `flagship`. The r64 configuration is `isvd_r64_h256_seed`.
- The v1 cycled-filler path (`filler: cycle`, `generator: inhouse`) stays available bit-for-bit for pairing with `results/paper-v1` records; nothing in this lane changes v1 numbers.
- Pairing invariant: same `(task, ctx, seed, trial)` → byte-identical `prefill_ids`/`query_ids` across arms and across processes; enforced by `prompt_sha256` in every record and a golden test.
- A trial that raises is recorded as `error` (L0 runner); this lane never adds a skip path.
- CPU tests only (< 90 s suite total); no network in tests (fixture corpus + the suite's tiny tokenizer fixture).
- No pod launch in this lane; Tasks 5–6 produce configs + preregs; launches are DECISIONS entries (money).
- Commit messages carry `L2.<n>`.

---

## File map

```
src/kvdlra/eval/gen.py                 generator v2 (+ the v1 inhouse builder moved here from ruler.py unchanged)
src/kvdlra/eval/haystacks.py           corpus materialization: pg19 | arxiv | wikipedia | essays -> data/haystacks/<src>.jsonl (+ .sha256)
src/kvdlra/eval/records.py             TrialRecord += code_family: str | None
src/kvdlra/eval/config.py              TaskCfg += generator "v2" fields: haystacks, code_families, design {haystacks, depths, codes}
src/kvdlra/quant/kivi.py               faithful KIVI: make_kivi(config, nbits, group=32, residual=128) + quantize_after_prefill()
src/kvdlra/quant/kivi_cache.py         kept (the streaming/chunked mixin, labelled "streaming")
src/kvdlra/baselines/svd_oracle.py     (L0 renamed it) docstring + DECISIONS D-004 entry here
src/kvdlra/baselines/presses.py        factories: snapkv/pyramidkv/expected_attention at keep k; think+snapkv ComposedPress; think alone
src/kvdlra/baselines/ojakv.py          adapter around the OjaKV repo's cache (pod-only import; skipped on CPU)
src/kvdlra/eval/ruler.py               retrieve(): "quant_faithful" kind; arm-kind dispatch table instead of if/elif chain
scripts/pod.py                         + `prepare --pod NAME` (materialize haystacks named by the pod's tasks; write SHAs)
scripts/tables.py                      + `baselines` table (ShadowKV post-fix + ea_k0.25 v1 rows from results/paper-v1 cells.jsonl)
configs/arms/kivi2_faithful.yaml, kivi4_faithful.yaml, kivi2_streaming.yaml (relabelled), snapkv_k0.10/0.15/0.25.yaml,
             pyramidkv_k0.10/0.15/0.25.yaml, ea_k0.10/0.15/0.25.yaml, think_c0.5_snapkv_k0.15.yaml, ojakv_matched.yaml
configs/tasks/ruler_v2_16k.yaml, ruler_v2_32k.yaml, ruler_v2_16k_g1.yaml (n=24 design), ruler_v2_16k_g2.yaml (n=48 design)
configs/pods/ss2_families.yaml, l2_smoke.yaml
prereg/ss2_families.md, prereg/l2_smoke.md          (filler_realism.md already committed on this lane)
tests/fixtures/haystacks_tiny.jsonl                 ~60 KB: 4 sources x 3 short documents, licence-free text
tests/test_gen_v2.py, test_kivi_faithful.py, test_presses.py, test_tables_baselines.py, tests/golden/gen_v2_prompts.json
```

---

### Task 1: `svd_oracle` — finish the rename and record the decision

**Files:**
- Modify: `src/kvdlra/baselines/svd_oracle.py` docstring (first paragraph states exactly what the arm is and is not — see below), `configs/arms/svd_oracle_r0.5.yaml` (`doc:` field), `docs/plan/DECISIONS.md` (append a D-004 evidence line: rename complete; real-Palu port deferred to Branch A/B)
- Test: `tests/test_forbidden_words.py`-style grep test `tests/test_palu_rename.py`

- [ ] **Step 1: Test**
```python
import subprocess
from pathlib import Path


def test_palu_appears_only_where_allowed() -> None:
    out = subprocess.run(["git", "grep", "-n", "-i", "palu", "--", "src", "configs", "scripts", "tests"],
                         capture_output=True, text=True).stdout.splitlines()
    allowed = ("src/kvdlra/baselines/svd_oracle.py", "configs/arms/svd_oracle_r0.5.yaml", "tests/test_palu_rename.py")
    bad = [l for l in out if not l.startswith(allowed)]
    assert not bad, "\n".join(bad)
```
- [ ] **Step 2:** Run → FAIL (or PASS if L0 already finished it; then only the docstring/DECISIONS steps apply). **Step 3:** Docstring: "Per-sequence, per-head truncated SVD of the prefill K and V (rank ratio ρ), sinks kept exact. It is an *upper bound* for static low-rank methods on this sequence (Eckart–Young in Frobenius norm on the sequence's own K/V) — it is not Palu (no weight decomposition, no grouped heads, no Fisher rank allocation, no fine-tuning, cannot be computed online). The paper-v1 records name it `palu-r0.5`; that name is retired." Remove the `legacy_name` from anything but the YAML. **Step 4:** `make test`; commit `L2.1: svd_oracle docstring + DECISIONS D-004 evidence; palu string confined to the oracle module`.

### Task 2: Faithful KIVI arm (G=32, R=128, full-precision prefill), streaming arm relabelled

**Files:**
- Create: `src/kvdlra/quant/kivi.py`, `configs/arms/kivi2_faithful.yaml`, `configs/arms/kivi4_faithful.yaml`; relabel `configs/arms/kivi2_streaming.yaml` / `kivi4_streaming.yaml` (`doc:` "transformers QuantizedCache mixin, G=64, chunked 4096 prefill — later chunks attend to 2-bit dequantized context; the arm the paper-v1 tables used")
- Modify: `src/kvdlra/eval/ruler.py:retrieve` (dispatch table `KINDS = {"bug": _run_streaming, "quant": _run_quant_streaming, "quant_faithful": _run_quant_faithful, "press": _run_press, "press_quant": _run_press_quant, "full": _run_press}`), `src/kvdlra/eval/frontier.py:_footprint` (kind `quant_faithful`: `acc.quant_footprint(..., residual_length=<actual residual tokens>, scale_words=aux_words(cache))`)
- Test: `tests/test_kivi_faithful.py`

**Interfaces:**
- Produces: `make_kivi(config, *, nbits: int, group: int = 32, residual: int = 128, backend: str = "quanto") -> QuantizedCache` (reuses `kivi_cache.make_quant_cache(scheme="kivi")`, only the defaults differ); `quantize_after_prefill(cache: QuantizedCache, dyn: DynamicCache, *, chunk: int = 4096) -> None` — moves every layer's fp16 K/V from `dyn` into `cache`: tokens `[0, T − (T mod residual) )` are quantized in `chunk`-token slices (chunk a multiple of `group`), the trailing `T mod residual` tokens (KIVI's rule: the residual holds the last `T mod R` tokens after prefill; if that is 0, the residual is empty) stay fp16 in `cache.layers[i].keys/values`; `residual_tokens(cache) -> int`.
- Prefill protocol for kind `quant_faithful`: `DynamicCache()` single-shot prefill with `logits_to_keep=1` (memory-safe, the same call the presses use), then `quantize_after_prefill`, then decode with `block=True` exactly as the streaming quant arm.

- [ ] **Step 1: Tests**
```python
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from kvdlra.quant.kivi import make_kivi, quantize_after_prefill, residual_tokens
from kvdlra.quant.kivi_cache import make_quant_cache


def _tiny():
    cfg = Qwen2Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                      intermediate_size=128, vocab_size=256, max_position_embeddings=1024)
    return Qwen2ForCausalLM(cfg).eval(), cfg


def test_faithful_prefill_never_calls_quantized_update(monkeypatch) -> None:
    model, cfg = _tiny()
    cache = make_kivi(cfg, nbits=4, backend="hqq")            # hqq: pure torch, no JIT on CPU
    calls = {"n": 0}
    orig = type(cache).update
    monkeypatch.setattr(type(cache), "update", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k))
    ids = torch.randint(0, 256, (1, 300))
    dyn = DynamicCache()
    with torch.no_grad():
        model(ids, past_key_values=dyn, use_cache=True, logits_to_keep=1)
    quantize_after_prefill(cache, dyn, chunk=64)
    assert calls["n"] == 0                                    # prefill attention saw fp16 only
    assert residual_tokens(cache) == 300 % 128                # KIVI's residual rule


def test_chunked_quantization_equals_whole_quantization() -> None:
    _, cfg = _tiny()
    torch.manual_seed(0)
    k = torch.randn(1, 2, 256, 32); v = torch.randn(1, 2, 256, 32)
    dyn = DynamicCache(); dyn.update(k, v, 0)
    a = make_kivi(cfg, nbits=4, backend="hqq", residual=0); quantize_after_prefill(a, dyn, chunk=64)
    b = make_kivi(cfg, nbits=4, backend="hqq", residual=0); quantize_after_prefill(b, dyn, chunk=256)
    ka = a.layers[0]._dequantize(a.layers[0]._quantized_keys); kb = b.layers[0]._dequantize(b.layers[0]._quantized_keys)
    assert torch.equal(ka, kb)                                 # groups are 32 consecutive tokens; chunking at multiples is exact


def test_per_channel_keys_survive_an_outlier_channel() -> None:
    _, cfg = _tiny()
    torch.manual_seed(1)
    k = torch.randn(1, 2, 256, 32); k[..., 7] *= 200.0         # one massive channel (Qwen-style key bias)
    v = torch.randn(1, 2, 256, 32)
    dyn = DynamicCache(); dyn.update(k, v, 0)
    faithful = make_kivi(cfg, nbits=2, backend="hqq", residual=0); quantize_after_prefill(faithful, dyn, chunk=256)
    token = make_quant_cache(cfg, nbits=2, scheme="token", backend="hqq", group=32, residual=0)
    token.update(k, v, 0)
    err = lambda c: float(torch.linalg.norm(c.layers[0]._dequantize(c.layers[0]._quantized_keys)[..., :31] - k[..., :31]))
    assert err(faithful) < 0.5 * err(token)


def test_bytes_include_scales_zeros_and_actual_residual() -> None:
    from kvdlra import accounting as acc
    from kvdlra.quant.kivi_cache import aux_words

    _, cfg = _tiny()
    dyn = DynamicCache(); dyn.update(torch.randn(1, 2, 300, 32), torch.randn(1, 2, 300, 32), 0)
    c = make_kivi(cfg, nbits=2, backend="hqq"); quantize_after_prefill(c, dyn, chunk=64)
    fp = acc.quant_footprint(t=300, n=64, nbits=2, group=32, residual_length=residual_tokens(c), scale_words=aux_words(c))
    assert fp.aux_words > 0 and fp.verbatim_elems == 2 * 64 * (300 % 128)
```
(`acc.quant_footprint`'s exact signature is in `src/kvdlra/accounting.py:420-447` — match it.)
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `kivi.py` (~90 lines), the dispatch table in `retrieve`, the footprint branch, the two YAMLs (`kind: quant_faithful`, `quant: {nbits: 2, group: 32, residual: 128, scheme: kivi, backend: quanto, prefill: full}`, `doc:` "KIVI's published operating point (Liu et al. 2024): per-channel keys, per-token values, G=32, R=128, full-precision prefill; quanto backend — no fused 2-bit kernel, so decode dequantizes (deviation recorded)"), relabel the streaming YAMLs. **Step 4:** `make test` → PASS; the `tests/test_w19_quant_kivi.py` pins for the streaming arm stay green. **Step 5:** Commit `L2.2: faithful KIVI arm (G=32, R=128, fp16 single-shot prefill, post-hoc chunked quantization); streaming arm relabelled`.
- [ ] **Step 6 (DECISIONS, no code):** KVQuant-style pre-RoPE 2/3-bit dense-and-sparse: append an OPEN entry "D-006 — KVQuant arm: requires pre-RoPE key quantization + an outlier sparse store + a dequant path without their kernel (~2 engineer-days); recommendation: defer to Phase 2 unless Gate 2 turns on the 3-bit point" in the task report for the orchestrator to record.

### Task 3: Generator v2

**Files:**
- Create: `src/kvdlra/eval/gen.py`, `src/kvdlra/eval/haystacks.py`, `tests/fixtures/haystacks_tiny.jsonl`, `tests/golden/gen_v2_prompts.json`, `configs/tasks/ruler_v2_16k.yaml`, `ruler_v2_32k.yaml`, `ruler_v2_16k_g1.yaml`, `ruler_v2_16k_g2.yaml`
- Modify: `src/kvdlra/eval/records.py` (`TrialRecord.code_family`), `src/kvdlra/eval/config.py` (`TaskCfg.haystacks: list[str]`, `code_families: list[str]`, `design: dict[str,int]` = `{"haystacks": 2, "depths": 3, "codes": 4}`), `src/kvdlra/eval/runner.py` (fills `haystack_id/depth/code_family/prompt_sha256` from `Trial.meta`), `scripts/pod.py` (`prepare`), `src/kvdlra/eval/ruler.py` (the v1 `build_task` + `_filler_to` move to `gen.py` unchanged as `make_trial_v1`)
- Test: `tests/test_gen_v2.py`

**Interfaces:**
- `Trial = NamedTuple(prefill_ids: Tensor, query_ids: Tensor, targets: list[str], meta: dict)` with `meta = {haystack_id: str, depth: float, code_family: str, prompt_sha256: str, n_sentences: int}`.
- `make_trial(cfg: TaskCfg, tok, task: str, seed: int, trial: int, corpora: dict[str, list[Doc]]) -> Trial`; `design_cell(cfg, trial) -> (haystack_idx, depth, code_family)` enumerates `trial` over the lexicographic product `codes × depths × haystacks` (so trials 0..n−1 with n = product cover the design exactly once; `seed` only changes the random draws); `Doc = {id: str, source: str, text: str, sha256: str}`; `load_corpora(names, root=Path("data/haystacks")) -> dict[str, list[Doc]]`.
- Haystack construction: pick `Doc = corpora[source][(haystack_idx + seed) % len]`, split into sentences (a 6-line regex splitter; no nltk), take a *contiguous* window of sentences starting at a seed-derived offset until `ctx` tokens (coherent text, not a shuffled bag); if the doc is too short, continue with the next doc of the same source (record both ids as `haystack_id="pg19:doc12+doc13"`).
- Tasks (official RULER semantics): `niah_single` (one needle at `depth`), `niah_multikey` (8 keys; the queried key at `depth`, the others evenly spread), `niah_multivalue` (4 values of one key, the first at `depth`, the rest at +0.2 steps wrapping), `niah_multiquery` (4 keys queried together — new), `vt` (4 hops = official; root at `depth`). Needle sentences use the code family: `numbers` = 7-digit integers (5-digit collided with corpus numerals — L2prep's note); `words` = adjective-noun pairs from a bundled 400-word list (`data/words.json`, SHA-pinned, RULER-style).
- `prompt_sha256 = sha256(prefill_ids.tobytes() + b"|" + query_ids.tobytes())`.
- `scripts/pod.py prepare --pod NAME`: for each `haystacks` source named by the pod's tasks, `haystacks.materialize(source, n_docs=64, out=data/haystacks/)` via `datasets` streaming (`deepmind/pg19` train, `ccdv/arxiv-summarization` (articles), `wikimedia/wikipedia` 20231101.en, `essays` = the RULER Paul Graham essays JSON at a pinned URL); writes `<source>.jsonl` + `<source>.sha256`; the runner records the SHAs in `manifest.json.dataset_sha256`.

- [ ] **Step 1: Tests**
```python
import hashlib
import json
from pathlib import Path

import pytest

from kvdlra.eval.config import TaskCfg
from kvdlra.eval.gen import design_cell, load_corpora, make_trial

FIX = Path("tests/fixtures/haystacks_tiny.jsonl")
CFG = TaskCfg(name="t", generator="v2", ctx=1024, tasks=["niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt"],
              n_trials=24, seeds=[0, 1], haystacks=["pg19", "arxiv", "wikipedia", "essays"],
              code_families=["numbers", "words"], design={"haystacks": 2, "depths": 3, "codes": 4})


@pytest.fixture(scope="module")
def corpora():
    return load_corpora(CFG.haystacks, root=FIX.parent, fixture=FIX)


def test_design_is_balanced_and_exhaustive() -> None:
    cells = [design_cell(CFG, t) for t in range(24)]
    assert len(set(cells)) == 24
    assert {c[1] for c in cells} == {0.05, 0.4, 0.95}          # 3 depths chosen evenly from the 6-point grid
    assert sorted({c[0] for c in cells}) == [0, 1] and {c[2] for c in cells} == {"numbers", "words"}


@pytest.mark.parametrize("task", CFG.tasks)
def test_prompt_is_paired_and_needle_is_present(task, tok, corpora) -> None:      # `tok` = the suite's tiny tokenizer fixture
    a = make_trial(CFG, tok, task, seed=0, trial=5, corpora=corpora)
    b = make_trial(CFG, tok, task, seed=0, trial=5, corpora=corpora)
    assert a.meta["prompt_sha256"] == b.meta["prompt_sha256"]
    assert a.prefill_ids.shape[1] >= 0.9 * CFG.ctx
    text = tok.decode(a.prefill_ids[0])
    for t in a.targets:
        assert t in text
    assert 0.0 <= a.meta["depth"] <= 1.0 and a.meta["code_family"] in CFG.code_families


def test_seed_changes_codes_not_design(tok, corpora) -> None:
    a = make_trial(CFG, tok, "niah_single", seed=0, trial=3, corpora=corpora)
    b = make_trial(CFG, tok, "niah_single", seed=1, trial=3, corpora=corpora)
    assert (a.meta["depth"], a.meta["code_family"]) == (b.meta["depth"], b.meta["code_family"])
    assert a.targets != b.targets


def test_golden_prompt_hashes(tok, corpora) -> None:
    want = json.loads(Path("tests/golden/gen_v2_prompts.json").read_text())
    for key, sha in want.items():
        task, seed, trial = key.split("|")
        got = make_trial(CFG, tok, task, seed=int(seed), trial=int(trial), corpora=corpora).meta["prompt_sha256"]
        assert got == sha, key


def test_v1_builder_is_untouched(tok) -> None:
    from kvdlra.eval.gen import make_trial_v1
    from tests.test_w10_ruler_filler import expected_v1_prefill_sha   # existing pin of the cycled path, if present; else compute once here

    pre, q, targets = make_trial_v1(tok, "niah_single", 512, trial=0, seed=0, n_keys=8, n_values=4, n_hops=3)
    assert hashlib.sha256(pre.numpy().tobytes()).hexdigest() == expected_v1_prefill_sha(tok)
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (`gen.py` ≈ 220 lines incl. the moved v1 builder; `haystacks.py` ≈ 80 lines; fixture corpus: 4 sources × 3 documents of ≥ 1,500 words each of public-domain / CC text — write them from Project Gutenberg-style public-domain passages you generate yourself, not copied from a copyrighted source; note provenance at the top of the JSONL as a comment record). Generate the golden after the tests for structure pass: `python -c "..."` over 5 tasks × 2 seeds × 3 trials → `tests/golden/gen_v2_prompts.json`. **Step 4:** `make test` → PASS. **Step 5:** Commits `L2.3a: generator v2 (4 sources, balanced depth design, 2 code families, official task semantics, prompt sha256)`, `L2.3b: haystack materialization + pod.py prepare; fixture corpus + golden`.

### Task 4: Matched-budget eviction/structured baselines + the v1 rows the paper omitted

**Files:**
- Create: `src/kvdlra/baselines/presses.py`, arms YAML `snapkv_k0.10/0.15/0.25`, `pyramidkv_k0.10/0.15/0.25`, `ea_k0.10/0.15/0.25` (ea_k0.1/0.25 exist from L0 — keep names consistent: rename to `ea_k0.10`, keep `legacy_name`), `think_c0.5_snapkv_k0.15`, `think_c0.5` (`doc:` "ablation only — ThinK's paper composes it with SnapKV/H2O")
- Modify: `scripts/tables.py` (`baselines` table: ShadowKV post-fix Llama 16K row from `results/paper-v1/w15-confirm/cells.jsonl` (1.00/1.00/—/0.00 at 0.815×) and `ea-k0.25` 16K rows from `results/paper-v1/w11-goalA-ruler/cells.jsonl` (1.00/0.88/1.00/0.50, n=8) — cells, so Wilson from `hits/n`), `docs/plan/paper-v1-tables.md` (append the new table's golden block)
- Test: `tests/test_presses.py`, `tests/test_tables_baselines.py`

- [ ] **Step 1: Tests**
```python
from kvpress import ComposedPress, ExpectedAttentionPress, SnapKVPress, ThinKPress

from kvdlra.baselines.presses import make_press
from kvdlra.eval.config import load_arm


def test_keep_fraction_maps_to_compression_ratio() -> None:
    p = make_press(load_arm("snapkv_k0.15"))
    assert isinstance(p, SnapKVPress) and abs(p.compression_ratio - 0.85) < 1e-9


def test_think_snapkv_is_composed_in_paper_order() -> None:
    p = make_press(load_arm("think_c0.5_snapkv_k0.15"))
    assert isinstance(p, ComposedPress) and [type(x) for x in p.presses] == [SnapKVPress, ThinKPress]


def test_every_eviction_arm_loads() -> None:
    for k in ("0.10", "0.15", "0.25"):
        for fam in ("snapkv", "pyramidkv", "ea"):
            assert make_press(load_arm(f"{fam}_k{k}")) is not None
```
```python
import subprocess, sys
from pathlib import Path


def test_baselines_table_has_shadow_and_ea025(tmp_path: Path) -> None:
    subprocess.run([sys.executable, "scripts/tables.py", "build", "--out", str(tmp_path)], check=True)
    md = (tmp_path / "table_baselines.md").read_text()
    assert "shadowkv_r64" in md and "ea_k0.25" in md and "0.815" in md
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `presses.py` (`make_press(arm) -> BasePress | None`: `press:` block `{family: snapkv|pyramidkv|expected_attention|think|think_snapkv, keep: float, ratio: float}`; PyramidKV = `kvpress.PyramidKVPress` if 0.5.1 ships it, else `SnapKVPress` with `PyramidKV`-style layer budgets is NOT improvised — report and mark the arm `unavailable: true` in the YAML), the table, the golden append. **Step 4:** `make test`, `make tables` diff-clean. **Step 5:** Commit `L2.4: eviction/structured baseline arms at k in {0.10,0.15,0.25}; ThinK+SnapKV composed; baselines table with the ShadowKV and ea_k0.25 v1 rows`.

### Task 5: `ss2_families` and `l2_smoke` pods — configs + pre-registrations (no launch)

**Files:**
- Create: `configs/pods/ss2_families.yaml` (Mistral-7B-v0.3 and Qwen2.5-7B at 16K: `kivi2_faithful`, `kivi4_faithful`, `kivi2_singleshot` (the ss2 replication arm: streaming mixin with chunk 0), `isvd_r64_h256_seed` paired reference; all three families at 32K; task `ruler_inhouse_*` with `filler: cycle` so every record pairs with `results/paper-v1/w19-a1-*` on `(seed, trial)`; n=12; `gpu_budget_h: 14`), `prereg/ss2_families.md` (primary contrast: r64 vs `kivi2_faithful` on multi-value, per family and pooled (McNemar, Holm over 3 families × 4 tasks = 12); reading verbatim from the brief: "the multi-value claim survives only where r64 vs single-shot KIVI-2 is separated after Holm"; budget; provenance; `STATUS: awaiting owner go`), `configs/pods/l2_smoke.yaml` (Llama-3.1-8B 16K, EVERY arm under `configs/arms/` that is not `unavailable`, task `ruler_v2_16k` with n=12 — harness validation, not paper data; `gpu_budget_h: 10`), `prereg/l2_smoke.md` (purpose: validate generator v2 + every arm end-to-end; reading: no arm with `error` records; every arm's `n` = 12; `prompt_sha256` identical across arms per `(task, seed, trial)`; the r64 config within 0.15 of its v1 cycled-filler accuracy on single-needle or the filler-realism reading applies)
- Test: `tests/test_pod_manifest.py` parametrized over the two pods (dry-run loads).

- [ ] Steps: write → dry-run test → commit `L2.5: ss2_families + l2_smoke pod configs and preregs (launch = DECISIONS)`.

### Task 6: OjaKV end-to-end (adapter, arm, version pin)

**Files:**
- Create: `src/kvdlra/baselines/ojakv.py`, `configs/arms/ojakv_matched.yaml` (`repo: <url>`, `commit: <sha>`, `arxiv: 2509.21623v2`, `doc:` which OjaKV version, their selection rule (residual-scored full-rank tokens), the rank/energy setting that lands nearest 0.15× stored state — or the nearest their code allows, stated), `scripts/pod/boot.sh` (+ `pip install git+<repo>@<commit>` guarded by `POD` naming an ojakv arm)
- Test: `tests/test_ojakv_adapter.py` — `pytest.importorskip("ojakv")`; on CPU the adapter builds their cache on the tiny Qwen2 model and runs one 64-token prefill + 4 decode steps without error; the arm YAML loads.

- [ ] **Step 1:** Read their repo (WebFetch/WebSearch allowed): the cache class, its constructor args, how it hooks attention, how it bills memory. Write the adapter so `retrieve()`'s streaming path (`cache.attach(model)` context, `_prefill_chunked`, `_footprint`) works unchanged: `OjaKVCache.attach`, `.stored_state_numel()` (their bytes: basis + coordinates + full-rank tokens; scales if any). **Step 2:** Tests (importorskip). **Step 3:** Commit `L2.6: OjaKV end-to-end adapter + matched-bytes arm (version pinned)`.

---

## Self-review

- **Spec coverage (L2 brief §1–§8):** §1 filler-realism diagnostic → already on the lane (prereg + w21.sh), launch = D-005. §2 generator v2 (≥4 sources, balanced 6-point depth grid subsampled per design, ≥2 code families, official semantics, `haystack_id/depth/code_family/seed/prompt_sha256`, pairing test + golden) → Task 3. §3 faithful KIVI (G=32, R=128, per-channel K / per-token V, fp16 full prefill, scales+zeros counted, quanto with the deviation in the config; `kivi2_streaming` kept and labelled; KVQuant → DECISIONS) → Task 2. §4 ss2 across families → Task 5. §5 rename + DECISIONS → Task 1. §6 SnapKV/PyramidKV/EA at k ∈ {0.10,0.15,0.25}, ThinK+SnapKV as intended, ThinK alone as ablation, ShadowKV + ea_k0.25 v1 rows in `make tables` → Task 4. §7 OjaKV e2e with version → Task 6. §8 smoke pod → Task 5.
- **GATES G2 mapping:** filler diagnostic harvested + DECISIONS → after the owner's go · pairing test + balanced-depth test ✓ T3 · kivi2_faithful G=32/R=128/full-precision prefill + dequant test + bytes incl. scales ✓ T2 · ss2 harvested + Holm → after launch (T5 prepares) · svd_oracle rename; grep hits only the docstring + DECISIONS ✓ T1 · SnapKV/PyramidKV/EA k∈{0.10,0.15,0.25} + ThinK+SnapKV in the smoke pod ✓ T4 + T5 · OjaKV runs on Llama 16K; version recorded ✓ T6 (pod) · ShadowKV and ea_k0.25 v1 rows in make tables ✓ T4.
- **Placeholders:** the suite's tiny tokenizer fixture (`tok`) and the v1 prefill pin helper are named by the implementer from `tests/test_w10_ruler_filler.py` / `tests/test_ruler_template_tail.py`; OjaKV's repo URL/commit are looked up in Task 6 (must be recorded, never guessed); `acc.quant_footprint`'s signature is read from accounting.py.
- **Type consistency:** `Trial.meta` keys are the `TrialRecord` fields the runner copies (`haystack_id`, `depth`, `code_family`, `prompt_sha256`); `TaskCfg.design` keys match `design_cell`; arm `kind: quant_faithful` is the key in `ruler.py`'s `KINDS` and the branch in `_footprint`.
- **Interaction with other lanes:** L3 (Gate 1 v2) consumes `ruler_v2_16k_g1.yaml` (n=24 design) and the `fd_r64_h256_seed` / `oja_r64_h256_seed` arms from L1; L5 writes `gate1_tracker_swap_v2.md`, `bf16_gist.md`, `kernel_smoke.md`; this lane owns `filler_realism.md`, `ss2_families.md`, `l2_smoke.md`.
