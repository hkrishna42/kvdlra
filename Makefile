PY ?= .venv/bin/python
# torch's C++ extension loader shells out to the `ninja` EXECUTABLE, and `$(PY) -m pytest`
# does not put the venv's bin/ on PATH -- without this the quant tests fail to JIT (8 of
# them on a clean clone: "Ninja is required to load C++ extensions").
export PATH := $(dir $(abspath $(PY))):$(PATH)

.PHONY: test tables env figures check clean
test:
	$(PY) -m pytest -q

tables:
	$(PY) scripts/tables.py build --out docs/paper/tables
	@cat docs/paper/tables/table*.md | diff -u docs/plan/paper-v1-tables.md - && echo "tables: diff-clean vs paper-v1"

env:
	uv venv --python 3.12 && uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"

figures:
	$(PY) scripts/figures.py build --out docs/paper/figures

# Every committed manifest: the pod directories a run writes, and the paper-v1
# archive, whose manifests `check` recognises (`converted_by`) and skips. The -f
# guard is what keeps an unmatched glob from being checked as a literal path.
check:
	@for d in results/*/manifest.json results/paper-v1/*/manifest.json; do \
	  [ -f "$$d" ] || continue; $(PY) scripts/pod.py check $$(dirname $$d) || exit 1; done

clean:
	rm -rf docs/paper/tables/*.md docs/paper/tables/*.tex docs/paper/figures/*
