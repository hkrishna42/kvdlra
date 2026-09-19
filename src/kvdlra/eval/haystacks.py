"""The real-text haystack sources of generator v2, materialized once per pod.

`materialize` streams the first ``n_docs`` documents of `MIN_CHARS` to `MAX_CHARS`
characters from one Hub dataset into ``data/haystacks/<source>.jsonl`` (one `gen.Doc`
per line, ids ``d<index>`` in stream order) and writes the JSONL's sha256 beside it; the
run prints that digest as a ``[stage] dataset_sha256 haystack:<source> <sha>`` line and
`scripts/pod.py harvest` writes it into ``manifest.dataset_sha256`` (the manifest the run
writes stays on the instance), so the text a cell was built on is named by the evidence
that cites it. ``data/haystacks/`` is gitignored: `scripts/pod.py prepare --pod <pod>`
(or `run`, when a source is missing) is what fills it.

Every source is pinned to a repository commit (``revision=``); the pins are the ``sha``
each repo reported at ``https://huggingface.co/api/datasets/<repo>`` on 2026-09-19:

* ``pg19``      -- ``deepmind/pg19`` train (Project Gutenberg books). A script-backed
  repo: needs ``trust_remote_code=True`` under datasets 2.21, the same fragility
  `data.load_corpus_ids` carries for its ``pg19`` corpora. The pin covers the script;
  the books come from the script's own storage bucket.
* ``arxiv``     -- ``neuralwork/arxiver`` (63K arXiv papers of 2023 as markdown, Nougat
  OCR, CC-BY-NC-SA-4.0; one parquet file, no script). Chosen over the plan's
  ``ccdv/arxiv-summarization`` -- parquet-native too at its current revision, but its
  articles are lowercased and space-tokenized (``@xmath``/``@xcite`` placeholders), so
  a capitalized needle sentence would stand out of the haystack.
* ``wikipedia`` -- ``wikimedia/wikipedia`` config ``20231101.en`` (parquet, no script).
* ``essays``    -- ``sgoel9/paul_graham_essays`` (215 essays as one CSV with a README
  naming paulgraham.com as the source and the cleaning applied; no script). The
  markdown-flavoured ``baber/paul_graham_essays`` parquet has no README and carries
  link markup, so the documented plain-text one is used.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

from kvdlra.eval.gen import HAYSTACKS, Doc

MIN_CHARS = 2_000  # a shorter row is a stub or a fragment, not a haystack document
# A longer row is a compilation, not a document -- compilations and scripture at the
# head of PG-19's train split (the KJV Bible, 4.3 M chars; a "Library of the Future"
# anthology, 5.4 M) would otherwise be the seed-0 pg19 haystacks.
MAX_CHARS = 2_000_000

# source -> (`load_dataset` keyword arguments, the text field of a row).
SOURCES: dict[str, tuple[dict[str, Any], str]] = {
    "pg19": (
        {
            "path": "deepmind/pg19",
            "split": "train",
            "revision": "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b",
            "trust_remote_code": True,
        },
        "text",
    ),
    "arxiv": (
        {
            "path": "neuralwork/arxiver",
            "split": "train",
            "revision": "698a6662e77fd5dd45dbbec988abc8123e5fa086",
        },
        "markdown",
    ),
    "wikipedia": (
        {
            "path": "wikimedia/wikipedia",
            "name": "20231101.en",
            "split": "train",
            "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        },
        "text",
    ),
    "essays": (
        {
            "path": "sgoel9/paul_graham_essays",
            "split": "train",
            "revision": "0c7155a53c25e24c9b9858314460e0ee1f5c3e4e",
        },
        "text",
    ),
}


def materialize(source: str, n_docs: int = 64, out: Path = HAYSTACKS) -> str:
    """Stream ``source`` into ``<out>/<source>.jsonl`` + ``<source>.sha256``; returns the
    sha256 of the JSONL bytes. Rows shorter than `MIN_CHARS` or longer than `MAX_CHARS`
    are skipped. Fails loud if the stream ends before ``n_docs`` documents: fewer would be
    a different corpus than the one the design names."""
    kwargs, field = SOURCES[source]
    t0 = time.perf_counter()
    docs: list[Doc] = []
    for row in load_dataset(streaming=True, **kwargs):
        text = str(row[field])
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        sha = hashlib.sha256(text.encode()).hexdigest()
        docs.append({"id": f"d{len(docs)}", "source": source, "text": text, "sha256": sha})
        if len(docs) == n_docs:
            break
    if len(docs) < n_docs:
        raise RuntimeError(
            f"{source}: {len(docs)} documents of {MIN_CHARS}..{MAX_CHARS} chars, not {n_docs}"
        )
    out.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(d, sort_keys=True) + "\n" for d in docs).encode()
    (out / f"{source}.jsonl").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (out / f"{source}.sha256").write_text(digest + "\n")
    print(
        f"[stage] materialize {source} -> {n_docs} docs ({time.perf_counter() - t0:.1f} s)",
        flush=True,
    )
    return digest


def ensure(sources: Iterable[str], out: Path | None = None) -> dict[str, str]:
    """``source -> sha256`` of ``<out>/<source>.jsonl`` (default `HAYSTACKS`) for each
    source, materializing the ones not on disk. The digest of a file already there is
    recomputed from its bytes, never read off the ``.sha256`` sidecar."""
    out = out or HAYSTACKS
    shas = {}
    for s in sources:
        p = out / f"{s}.jsonl"
        if p.is_file():
            with p.open("rb") as f:
                shas[s] = hashlib.file_digest(f, "sha256").hexdigest()
        else:
            shas[s] = materialize(s, out=out)
    return shas
