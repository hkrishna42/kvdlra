#!/bin/bash
# Assemble a self-contained arXiv v1 submission tarball from paper/main.tex.
#
# arXiv's AutoTeX extracts the tarball flat and runs latex -> bibtex -> latex -> latex,
# so main.tex sits at the root, figures move under figures/week19/ (the ../ prefix in the
# source is rewritten to a flat path), and refs.bib travels with it (arXiv runs bibtex).
# If paper/main.bbl exists (fetched from the `paper` CI artifact) it is included too, which
# removes arXiv's bibtex step as a failure mode -- otherwise arXiv rebuilds it from refs.bib.
#
#   bash scripts/make_arxiv.sh            # -> paper/arxiv/ + paper/arxiv-v1.tar.gz
# Then upload paper/arxiv-v1.tar.gz at arxiv.org (New Submission -> cs.LG).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=paper/arxiv
rm -rf "$OUT"; mkdir -p "$OUT/figures/week19"

# main.tex with the figure paths flattened (../figures/week19/ -> figures/week19/).
sed 's#\.\./figures/week19/#figures/week19/#g' paper/main.tex > "$OUT/main.tex"
cp paper/refs.bib "$OUT/refs.bib"
for f in fairquant one_over_t coldstart; do
  cp "figures/week19/$f.pdf" "$OUT/figures/week19/$f.pdf"
done
# Optional: a pre-built bibliography so arXiv need not run bibtex (drop it in from CI).
if [ -f paper/main.bbl ]; then cp paper/main.bbl "$OUT/main.bbl"; BBL="main.bbl (bundled)"; else BBL="none (arXiv runs bibtex from refs.bib)"; fi

# Guard: no dangling ../ path escaped the rewrite (would break on arXiv's flat extract).
if grep -q '\.\./' "$OUT/main.tex"; then echo "ERROR: ../ path remains in main.tex" >&2; exit 1; fi

tar -C "$OUT" -czf paper/arxiv-v1.tar.gz .
echo "built paper/arxiv-v1.tar.gz  (bbl: $BBL)"
tar -tzf paper/arxiv-v1.tar.gz | sed 's/^/  /'
echo "size: $(du -h paper/arxiv-v1.tar.gz | cut -f1)"
