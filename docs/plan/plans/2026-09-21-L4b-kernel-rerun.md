# L4b — kernel_smoke re-run under Amendment 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Week-3 kernel-correctness gate READABLE — re-state `prereg/kernel_smoke.md` §4's precondition in the *relative* units the quantity is actually measured in (a per-layer bar of 4 bf16 ulp of `max|ref|`, and a near-tie margin for token mismatches), record the five fields that make it computable, and re-run one pod (`kernel_smoke2`) so the gate reads met-or-genuinely-missed instead of REFUSED-under-an-absolute-bar.

**Architecture:** Instance 51903816's REFUSED verdict is permanent and is **never re-read** (its records store neither `max|ref|` nor any logit). Amendment 2 governs only pods launched after its own commit. Task 11 adds the record fields + a $0 bf16 tiny-model calibration that fixes the near-tie margin `M` **before** the launch commit; Task 12 commits Amendment 2 (with the measured κ) + the new pod config; Task 13 launches `kernel_smoke2` and reads the outcome. All Task-11 code is CPU-tested; the Triton kernel, the arm under measurement, and the `pre_run` gate and every bar it applies are untouched (bit-identical `pytest -m gpu`).

**Tech Stack:** Python 3.12, PyTorch 2.11 (CPU for tests), Triton (GPU-only, untouched), pytest, the vast.ai pod scripts (`scripts/pod.py`, `scripts/pod/watchdog.sh`), omegaconf configs.

**Spec:** `docs/plan/reports/kernel-smoke-amendment2-draft.md` — the pre-committed draft (Amendment 2 text §2, the Task-11 table §3, the decision §4), committed at 503c754 **before any calibration ran**, so it is the pre-specification the re-run refutes-or-confirms. Also `prereg/kernel_smoke.md` §1–§11 + Amendment 1; the D-002 addendum of 2026-09-21 (the REFUSED reading); `docs/plan/cleanup/l4-ledger.md` R-L4-17/21/26/35.

## Global Constraints

- **Forbidden words** in new code/configs/docs/paper/commits: `DLRA` (as a word, case-sensitive — the repo path `kv-dlra` is not the word), `BUG integrator`, `honest`/`honestly`, `marquee`, `flagship` (case-insensitive). CI-enforced; the L6 gate greps the diff.
- **CPU-only, < 90 s CI:** every Task-11 change is pure-Python except the calibration test (item 8, the tiny model in bf16, ≤ 25 s measured). The warm suite must stay < 85 s with the calibration in (measure `make test --durations=15` and trim first; never shrink the calibration's 16 × 16). `pytest -m gpu` is untouched.
- **Instance 51903816 is immutable:** its records are cited as REFUSED under the original bar; nothing about them is revised, deleted or re-read. `kernel_smoke2` writes `results/kernel_smoke2/` — it cannot overwrite `results/kernel_smoke`.
- **Numbers as powers of two, not fitted constants:** `REL_MAX = 2**-6` (= 4 u, u = 2⁻⁸), `M_COEF = 2**-5` (= 8 u). The κ > 8 revision branch (A2.3) is the only thing that can change `M_COEF`, and only from a committed CPU calibration.
- **Provenance:** one commit per logical change, message carries the pod id + prereg path; `pod.py launch` machine-checks only the prereg's *first* commit is a strict ancestor of the launch SHA, so the launch DECISIONS entry names Amendment 2's SHA and pastes `git merge-base --is-ancestor <amendment SHA> <launch SHA>`.
- **A trial/precondition that cannot be computed is REFUSED, never a silent pass:** a record whose new fields are `None` (predates Amendment 2) reads "not computable (record predates Amendment 2)" and refuses.

**Ordering ruling (to ledger before Task 12):** the draft's A2.7 commit order (Amendment 2 → Task 11 → YAML → launch) is superseded for one reason — A2.3's κ calibration is a CPU test in Task 11 that *can revise* `M`, and a launched pod's prereg takes appended text only. So **Task 11 (with the calibration) lands first**, its printed κ is read, and **Amendment 2 (Task 12) is committed once, with the measured ρ̂ / L_tiny / κ filled into A2.3**. The pre-specification is the draft at 503c754, which precedes both.

---

### Task 11: The record fields, the relative-bar renderer, and the bf16 calibration

**Files (all under the L4 worktree; line refs are the draft §3 anchors, verify against HEAD):**
- Modify: `src/kvdlra/kernel/attention.py` (~:90-99) — `kernel_compare[layer]` stores `(max|Δ|, max|ref|)`
- Modify: `src/kvdlra/eval/kernel_check.py` (~:32-116) — `_greedy`, `check_prompt`, `failed_row`, `format_line`
- Modify: `src/kvdlra/eval/records.py` (~:106-110, 213-234, 489+) — `KERNEL_CHECK_RE`, `KernelCheckRecord`, `parse_kernel_check_lines`
- Modify: `scripts/tables.py` (~:1139, 1181-1197) — `REL_MAX`, `M_COEF`, `precondition_line`
- Modify: `src/kvdlra/eval/latency.py` (~:106-133) — the `[latency spikes …]` companion line
- Test: `tests/test_kernel_path.py` (the bf16 calibration), `tests/test_records.py` + `tests/test_kernel_smoke_pod.py` (round-trip + `precondition_line` cases)

**Interfaces:**
- Consumes: `kernel_compare(...)` returning per-layer diffs (attention.py); `_greedy` / `check_prompt` / `failed_row` / `format_line` (kernel_check.py); `KernelCheckRecord` + `KERNEL_CHECK_RE` + `parse_kernel_check_lines` (records.py); `run_latency` keeping `times_ms` (latency.py); `precondition_line` (tables.py:1181-1197 at 99ea29f).
- Produces (later tasks + the renderer rely on these exact names):
  - `KernelCheckRecord` gains `rel_max_diff: float | None`, `rel_worst_layer: int | None`, `ref_max: float | None`, `gap_at_mismatch: float | None`, `kernel_logit_for_ref_argmax: float | None` — five **optional** fields, appended **before** the `error=` tail of the `[kernel_check prompt=…]` line (exactly as `backend=` was, L4.fw1), so every archived row (51903816's among them) still parses with `None`.
  - `scripts/tables.py` module constants `REL_MAX = 2**-6` and `M_COEF = 2**-5`; `precondition_line` reads both bars and returns the split counts (attributable vs near-tie).

The nine items are the draft §3 table (implement each exactly as it reads); the load-bearing details:

- **Item 1** (attention.py): `kernel_compare[layer]` stores `(max|Δ|, max|ref|)` — one extra `.abs().max()` on the `ref` tensor `kernel_compare` already materializes.
- **Item 2** (kernel_check.py `_greedy`): return per-step `(argmax, top1, top2)`; **run the reconstruct twin FIRST**, then the kernel, so the kernel's logit at the twin's argmax is a single `gather` at the mismatching step (no full 128k×32 logit row kept). **R-L4-35 line: `del cache` before the reconstruct twin** (two 4096-ctx caches otherwise coexist; no recorded number reads that memory axis, so this is a hygiene line, not a measured change).
- **Item 3** (`check_prompt`): compute `rel_max_diff = max_l(d_l / m_l)` (skip `m_l == 0`, report it), `rel_worst_layer`, `ref_max` at that layer, and `gap_at_mismatch` (reconstruct top1−top2 at the first mismatching step; `-` if matched) / `kernel_logit_for_ref_argmax`; add `refs=` beside `diffs=` on the log-only `[kernel_check layers prompt=0 …]` line.
- **Item 4** (`failed_row`, `format_line`): the five fields (all `None` in `failed_row`), appended **before** `error=`, `-` when absent.
- **Item 5** (records.py): five optional regex groups + five `KernelCheckRecord` keys + five `_field`/float conversions; archived rows parse with `None`.
- **Item 6** (tables.py): `REL_MAX = 2**-6`, `M_COEF = 2**-5`; `precondition_line` reads BOTH bars, splits mismatches into attributable (`gap > M_COEF · top1`) / near-tie, prints `rel` with its layer + the absolute `max|Δ|` with its layer + both counts; a row whose new fields are `None` is "not computable (record predates Amendment 2)" and refuses, never silently passes. `DIFF_MAX` stays as the reported absolute number.
- **Item 7** (latency.py): one **log-only** `[latency spikes ctx=<T> arm=<key> steps=<i,j,k,…>]` companion line per cell, from the `times_ms` already held — no record field, no regex, no renderer change, no change to the measured arm.
- **Item 8** (test_kernel_path.py, the $0 calibration): the tiny model cast to **bf16**, the same 16 committed prompts × 16 greedy steps, kernel arm vs reconstruct twin, recording per step `ρ_s = max_j |L_kernel[s,j] − L_recon[s,j]| / (u · max_j |L_recon[s,j]|)` over the top-8 tokens. `κ = ρ̂ · √(32 / L_tiny)` rounded **up** to the next power of two; the test **asserts κ ≤ 8** and its failure message prints `ρ̂`, `L_tiny`, `κ` (the κ > 8 revision branch's input). Never shrink the 16 × 16.
- **Item 9** (test_records.py, test_kernel_smoke_pod.py): round-trip `format_line` → `KERNEL_CHECK_RE` with and without the new fields (an archived row still parses); `precondition_line` unit cases — met, relative-bar miss, attributable-mismatch miss, near-tie-only miss, `None`-fields refusal.
- **R-L4-35 also:** `OSError` added to the postrope snippet's `except` tuple (the handover promotes this into Task 11).
- **NOT in this task:** `do_not_specialize` on `_tiles_kernel`, any warm-up change, the post-basis early-return move (parked — A2.4 establishes no constexpr is length-dependent, so there is nothing to fix on the measured arm; the spike cause is *diagnosed* by item 7 first).

- [ ] **Step 1: Measure the suite's warm-run headroom before adding the calibration.** Run `make test --durations=15` (from the L4 worktree venv); note the warm total and the slowest tests. The suite is ~76–79 s warm against the 90 s gate; the calibration adds ≤ 25 s, so trim the slowest non-load-bearing tests until the warm suite will stay < 85 s WITH the calibration. Record what was trimmed.
- [ ] **Step 2: TDD each of items 1–7 and 9** — for each shipped-behavior change write the failing test first (round-trip parse for the fields; `precondition_line` unit cases; the `refs=`/companion-line format), watch it fail, implement the minimal change, watch it pass. Items 1–7 are < 2 s total; keep them pure-Python.
- [ ] **Step 3: Write the bf16 calibration (item 8) and RUN it** — it must pass (κ ≤ 8 is the pre-registered expectation) and PRINT `ρ̂`, `L_tiny`, `κ`. Capture the printed κ — it is Task 12's input. If κ > 8, the test fails by design; that failure's numbers are what Amendment 2's revision branch records (do not weaken the assertion — report κ > 8 to the orchestrator).
- [ ] **Step 4: Run the whole suite** — `make test` warm < 85 s, all pass; `pytest -m gpu` bit-identical (unchanged). `make tables` diff-clean. Forbidden-word grep clean over the diff.
- [ ] **Step 5: Commit** — `git commit` message: `feat(kernel_check): Amendment 2 record fields + relative-bar renderer + bf16 calibration (Task 11)`, naming `prereg/kernel_smoke.md` + the measured κ. Task-review (spec + quality) before Task 12.

---

### Task 12: Amendment 2 text + the `kernel_smoke2` pod config

**Files:**
- Modify: `prereg/kernel_smoke.md` — append **Amendment 2** (append-only, dated) verbatim from the draft §2, with A2.3's measured branch filled in from Task 11's κ
- Create: `configs/pods/kernel_smoke2.yaml` — byte-for-byte `configs/pods/kernel_smoke.yaml` with `name: kernel_smoke2`, `prereg: prereg/kernel_smoke.md` kept
- Modify: `Makefile` — a `make kernel_smoke2` target (the renderer takes `--pods results/<pod>`)

**Interfaces:** Amendment 2 is prose; it re-states §4's precondition in the fields Task 11 produced (`rel ≤ 2⁻⁶ ∧ attributable-count ≥ 14 ∧ no errors`). `kernel_smoke.yaml`'s `doc:` is inside `config_hash` and is hashed by the harvested manifest of instance 51903816 — **never edit `kernel_smoke.yaml`**; `kernel_smoke2.yaml` is a new file.

- [ ] **Step 1:** Append Amendment 2 to `prereg/kernel_smoke.md` verbatim from the draft §2 (A2.1–A2.7), filling A2.3's κ line: if κ ≤ 8, `M = 2⁻⁵ · max_j L[s,j]` stands (print the measured ρ̂/L_tiny/κ as evidence); if κ > 8, `M = κ · u · max_j L[s,j]` with ρ̂/L_tiny/κ printed. Confirm §2's "does not re-read instance 51903816" and "governs only pods after its own commit" are present. Forbidden-word grep the appended text.
- [ ] **Step 2:** `cp configs/pods/kernel_smoke.yaml configs/pods/kernel_smoke2.yaml`; change ONLY `name:` → `kernel_smoke2`; verify `prereg: prereg/kernel_smoke.md` unchanged; `diff` the two files (exactly one line differs).
- [ ] **Step 3:** Add `make kernel_smoke2` (mirror `make kernel_smoke`, `--pods results/kernel_smoke2`). `make test` still green (the pod-config parity test in `tests/test_pod_manifest.py` may need the new pod pinned — TDD it).
- [ ] **Step 4: Commit** — `prereg(kernel_smoke): Amendment 2 (relative bar 2^-6, near-tie margin M, κ=<measured>) + kernel_smoke2 pod (Task 12)`. Task-review "as a scientist" (the numbers, the append-only discipline, the byte-for-byte config).

---

### Task 13: Launch `kernel_smoke2`, harvest, read the outcome

**Files:** `docs/plan/DECISIONS.md` (D-011 addendum + the D-002 addendum re-run outcome), `docs/plan/STATE.md`, `docs/paper/tables/kernel_smoke.md` (via `make kernel_smoke2`), the harvested `results/kernel_smoke2/`.

- [ ] **Step 1:** L6 clean-clone gate on the launch SHA (Appendix B of the session prompt). Then `pod.py launch --pod kernel_smoke2 --offer <id>` on an A100 SXM4 40 GB (or PCIe if scarce — same reasoning as Stage 1), reliability ≥ 0.99; avoid the California PCIe host family 5116338x; **Mac on AC**, watchdog under `caffeinate -s -i`. If `vastai create` answers `success: False` / intended state `stopped`, destroy within minutes and relaunch on another offer (D-011 addendum 12 attempt 1). Budget ≈ $1.0–1.7 expected, ≤ $4 at the 5.0 h bar.
- [ ] **Step 2:** DECISIONS D-011 addendum (SHAs: prereg first commit 71c9be6, Amendment 2's SHA, the launch SHA; the pasted `git merge-base --is-ancestor`; offer, rate, bar 5.0 h, credit) + a STATE line.
- [ ] **Step 3:** At ALL_DONE: `scripts/pod.py check results/kernel_smoke2` (the 16 kernel_check rows + the three pre_run refusals); commit the records by explicit path; `make kernel_smoke2`; commit the table.
- [ ] **Step 4:** The D-002 addendum "Gate-3 WEEK-3 OUTCOME, re-run" — the rule decides (`rel ≤ 2⁻⁶ ∧ attributable-count ≥ 14 ∧ no errors`, then the two gate conditions at batch 1); GATES §G4 lines 2–3 ticked or annotated with evidence; the spike companion line's reading (first-touch JIT vs the 16-step absorb lattice) recorded as a performance observation, no fix on this pod. Instance 51903816's records are never re-read.

---

## After Task 13: the post-RoPE accuracy row, then the finishing menu

- **`isvd_postrope_r128`** (prereg/postrope_r128.md + configs/pods/postrope_r128.yaml, both committed, unlaunched): its own D-011 line, watchdog, harvest, `make tables` row, D-002 addendum — does r = 128 post-RoPE match r = 64 pre-RoPE at equal bytes on retrieval and perplexity (prereg §11's refusals apply)? Launch after the top-up, same A100 class.
- **Finishing:** whole-branch review (fable) + `/ponytail-review` + the L6 clean-clone gate on the final lane SHA → the finishing menu (superpowers:finishing-a-development-branch stops there). Then the L4 merge into main — **after** L3 is merged (done): STATE.md and DECISIONS.md conflict at EOF (both lanes append); keep both sides in date order, nothing dropped.

## Self-review notes (against the spec)

- Spec coverage: draft §3's nine items → Task 11; draft §2 (Amendment 2 text) → Task 12; draft §4/§10 (the outcome rule + budget) → Task 13. Cell list B stays the owner's separate amendment (§0(d): max-batch A100 first).
- The ordering ruling (calibration-first) is the one deviation from the draft's A2.7; it is ledgered before Task 12 and the pre-specification (draft at 503c754) precedes both.
- Type consistency: the five field names are identical across items 3/4/5/6 and Task 12's Amendment 2 text (`rel_max_diff`, `rel_worst_layer`, `ref_max`, `gap_at_mismatch`, `kernel_logit_for_ref_argmax`).
