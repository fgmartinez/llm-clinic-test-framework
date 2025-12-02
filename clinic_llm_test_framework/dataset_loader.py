"""Helpers for loading evaluation data.

The evaluation framework expects two kinds of data sources:

* An evaluation dataset consisting of question–answer pairs and optional
  expected outputs.  This is stored as a JSON list of objects with
  keys ``input`` (user question), ``expected_output`` (the ideal
  assistant response), and optionally ``context_ids`` referencing
  documents in the RAG knowledge base.  See ``data/clinic_qa.json`` for
  an example.
* A RAG knowledge base containing textual documents separated by
  blank lines.  During RAG evaluation these are indexed by the
  retriever and retrieved by similarity.

By centralising data loading here we make it easy to swap in new
datasets or contexts without modifying the evaluator code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the package directory if it's not absolute."""
    p = Path(path)
    if p.is_absolute() or p.is_file():
        return p
    # Try relative to package root
    pkg_root = Path(__file__).resolve().parent
    return pkg_root / path


def load_test_cases(file_path: str) -> List[Dict[str, str]]:
    """Load a JSON list of test cases from disk.

    Each entry in the file should contain at minimum an ``input`` key.
    If ``expected_output`` is missing the evaluation will still run
    metrics that do not require it (e.g. answer relevancy).  Optionally
    a ``context_ids`` list may be provided which identifies which
    paragraphs from the knowledge base should be considered the gold
    context for RAG evaluation.  See ``data/clinic_qa.json`` for a
    concrete example.
    """
    path = _resolve_path(file_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of test cases in {file_path}")
    return data  # type: ignore[return-value]


def load_rag_context(file_path: str) -> List[str]:
    """Load a plain text knowledge base into a list of documents.

    The file is expected to contain paragraphs separated by blank
    lines.  These paragraphs are returned as a list in the order
    encountered.  The indices correspond to ``context_ids`` in test
    cases.
    """
    path = _resolve_path(file_path)
    text = path.read_text(encoding="utf-8")
    # Split on two or more newlines to separate documents.
    raw_docs = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return raw_docs