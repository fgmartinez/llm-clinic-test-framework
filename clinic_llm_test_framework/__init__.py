"""Top-level package for the clinic LLM test framework.

This package contains utilities to evaluate a conversational AI assistant
designed for a medical clinic.  The framework emphasises clarity and
extensibility: each layer (LLM provider, retrieval, prompting, metrics
and evaluation) is implemented in its own module.  This layout makes
it easy to understand how a test case flows through the system while
remaining professional enough for a portfolio piece.

Modules included:

* :mod:`config` – definitions for evaluation configuration objects.
* :mod:`llm_provider` – a wrapper for instantiating different chat models
  (OpenAI or Google) from environment variables.
* :mod:`retriever` – a simple TF‑IDF retriever for RAG experiments.
* :mod:`metrics` – wrappers around DeepEval metrics as well as basic
  heuristic scores.
* :mod:`dataset_loader` – helpers for loading test cases and RAG context
  from JSON and plain text files.
* :mod:`test_case_builder` – constructs DeepEval ``LLMTestCase`` objects
  from loaded data.
* :mod:`evaluator` – orchestrates running the LLM against each test case
  (with or without retrieval) and computing metrics.
"""

from .config import PromptEvalConfig, RAGEvalConfig
from .llm_provider import LLMProvider
from .retriever import SimpleTFIDFRetriever
from .metrics import available_metrics
from .dataset_loader import load_test_cases, load_rag_context
from .test_case_builder import build_test_cases
from .evaluator import run_prompt_tests, run_rag_tests

__all__ = [
    "PromptEvalConfig",
    "RAGEvalConfig",
    "LLMProvider",
    "SimpleTFIDFRetriever",
    "available_metrics",
    "load_test_cases",
    "load_rag_context",
    "build_test_cases",
    "run_prompt_tests",
    "run_rag_tests",
]