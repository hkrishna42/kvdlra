PY ?= .venv/bin/python

.PHONY: test tables
test:
	$(PY) -m pytest -q

tables:
	$(PY) scripts/tables.py build --out docs/paper/tables
	@cat docs/paper/tables/table*.md | diff -u docs/plan/paper-v1-tables.md - && echo "tables: diff-clean vs paper-v1"
