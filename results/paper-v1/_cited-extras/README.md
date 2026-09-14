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

## Integrity

`shasum -a 256 raw/*`, recorded 2026-09-14 (L0.11). Every other archived `raw/`
file is pinned by its pod's `manifest.json` (`source_files[].lines` plus the
parsed record counts); these two belong to no pod, so their pin is the digest.
Re-check with `cd results/paper-v1/_cited-extras && shasum -a 256 -c` against the
lines below.

```
0b59dac592283339dc51a3881e60b03382e8d2752db9759cfa4d940109059076  raw/w18-env-provenance.txt
5a121e05e5c3c6ed738d4bf65d119c1639cf7d8dff3ee65d187231b4942f7192  raw/w19-env-provenance.txt
```

This README is the index, not evidence: it carries no digest of its own.
