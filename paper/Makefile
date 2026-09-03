# Build the kvdlra arXiv v1 preprint.
#   make        -> build main.pdf with latexmk if available, else print how to get it
#   make clean  -> remove latexmk aux files
#   make todos  -> list every \todo{...} left in the source
#
# Guarded: `make` is a no-op (with a message) when latexmk is not installed, so
# checking out the repo on a box without a TeX distribution does not error.

MAIN := main

.PHONY: all clean todos
all:
	@command -v latexmk >/dev/null 2>&1 && { \
	  echo "latexmk found -> building $(MAIN).pdf"; \
	  latexmk -pdf -bibtex -interaction=nonstopmode $(MAIN).tex; \
	} || { \
	  echo "latexmk not installed; skipping build (no-op)."; \
	  echo "Install a TeX distribution (e.g. MacTeX / TeX Live) then: make"; \
	}

clean:
	@command -v latexmk >/dev/null 2>&1 && latexmk -C $(MAIN).tex || \
	  rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	        $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).pdf

todos:
	@grep -n '\\todo{' $(MAIN).tex || echo "no \\todo left"
