# `_cited-extras`

Verbatim copies of files that `paper/main.tex` cites but that are not the input
of any `scripts/tables.py convert-v1` conversion, so they have no pod of their
own. They are evidence the paper points at, kept byte-identical; nothing reads
them programmatically.

| `raw/` file | cited at | what it is |
|---|---|---|
| `w18-env-provenance.txt` | `paper/main.tex` §"Reproducibility" (`\texttt{results/w18-env-provenance.txt}`) | the Week-18 pods' `===ENV_BEGIN===`/`===ENV_END===` blocks (run SHA, `nvidia-smi`, python / torch / CUDA / transformers versions), merged from the harvested logs |
| `w19-env-provenance.txt` | same paragraph, and the Week-19 limits paragraph | the Week-19 side of the same record |

`docs/plan/cleanup/paper-source-map.md` maps every cited path in `main.tex` to
where its bytes live now.
