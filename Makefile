PY ?= .venv/bin/python
# torch's C++ extension loader shells out to the `ninja` EXECUTABLE, and `$(PY) -m pytest`
# does not put the venv's bin/ on PATH -- without this the quant tests fail to JIT (8 of
# them on a clean clone: "Ninja is required to load C++ extensions"). `lastword` because
# PY is not always one word: `PY="uv run python"` would otherwise expand to three bogus
# entries (`$(abspath)` maps over every word), corrupting the PATH it prepends to. The
# last word names the interpreter, and `uv run` puts its own venv bin/ on PATH anyway.
export PATH := $(dir $(abspath $(lastword $(PY)))):$(PATH)

.PHONY: test tables gate1 kernel_smoke env figures check clean
# No -q here: pyproject's addopts already carries one, and a second one suppresses the
# `N passed in Ns` line -- a green run that reports no count is not a verification.
test:
	$(PY) -m pytest

tables:
	$(PY) scripts/tables.py build --out docs/paper/tables
	@cat docs/paper/tables/table*.md | diff -u docs/plan/paper-v1-tables.md - && echo "tables: diff-clean vs paper-v1"

# The Gate-1 table (prereg/gate1_tracker_swap_v2.md section 10's citability rule). It sits
# outside the `table*.md` glob `tables` diffs against paper-v1, so the two are independent:
# this reads a new pod's records, that one regenerates the frozen v1 tables.
gate1:
	$(PY) scripts/tables.py gate1 --pods results/gate1_v2_stage1_llama results/gate1_v2_stage1_qwen

# The kernel-smoke table (prereg/kernel_smoke.md sections 4, 7, 10): outside the `table*.md`
# glob like `gate1`, so `tables` stays diff-clean beside it.
kernel_smoke:
	$(PY) scripts/tables.py latency --pods results/kernel_smoke --out docs/paper/tables/kernel_smoke.md

# --allow-existing: `make env` is re-run to pick up a pin change, and uv refuses an
# existing .venv without it.
env:
	uv venv --python 3.12 --allow-existing && uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"

figures:
	$(PY) scripts/figures.py --out docs/paper/figures

# Every committed manifest: the pod directories a run writes, and the paper-v1
# archive, whose manifests `check` recognises (`converted_by`) and skips. The -f
# guard is what keeps an unmatched glob from being checked as a literal path.
check:
	@for d in results/*/manifest.json results/paper-v1/*/manifest.json; do \
	  [ -f "$$d" ] || continue; $(PY) scripts/pod.py check $$(dirname $$d) || exit 1; done

clean:
	rm -rf docs/paper/tables/*.md docs/paper/tables/*.tex docs/paper/figures/*
