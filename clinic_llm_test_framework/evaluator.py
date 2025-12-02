"""Evaluation routines for prompt and RAG based testing.

This module orchestrates the end‑to‑end testing loop: it loads
configuration, instantiates the LLM, constructs prompts (with or
without retrieved context), runs the model, builds DeepEval test
cases, computes metrics, and aggregates the results.

Functions provided at module scope (rather than in a class) to
reduce complexity and make it easy to call them from scripts or
notebooks.  Each function returns both the per‑question results and
aggregate metric averages for further analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jinja2 import Template

from .config import PromptEvalConfig, RAGEvalConfig
from .llm_provider import LLMProvider
from .retriever import SimpleTFIDFRetriever
from .metrics import available_metrics
from .dataset_loader import load_test_cases, load_rag_context
from .test_case_builder import build_test_cases

try:
    from deepeval import evaluate
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the package directory if it's not absolute."""
    p = Path(path)
    if p.is_absolute() or p.is_file():
        return p
    # Try relative to package root
    pkg_root = Path(__file__).resolve().parent
    return pkg_root / path


def _invoke_llm(llm, prompt: str) -> str:
    """Invoke the LLM with the given prompt, handling both old and new LangChain APIs."""
    # Try the modern invoke() method first
    if hasattr(llm, 'invoke'):
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=prompt)])
        # Extract text from the response
        if hasattr(result, 'content'):
            return result.content
        return str(result)
    # Fall back to predict() for older versions
    elif hasattr(llm, 'predict'):
        return llm.predict(prompt)
    else:
        raise AttributeError(f"LLM object has neither 'invoke' nor 'predict' method")


def _load_template(path: str) -> Template:
    """Load a Jinja2 template file from disk."""
    resolved_path = _resolve_path(path)
    text = resolved_path.read_text(encoding="utf-8")
    return Template(text)


def _load_persona(path: str) -> str:
    """Load the system persona from a plain text file."""
    resolved_path = _resolve_path(path)
    return resolved_path.read_text(encoding="utf-8").strip()


def run_prompt_tests(config: PromptEvalConfig, dataset_path: str) -> Dict[str, Any]:
    """Execute prompt‑only tests.

    Parameters
    ----------
    config : PromptEvalConfig
        Evaluation settings including model parameters, metrics and prompt template path.
    dataset_path : str
        Path to the JSON file containing test cases.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``results`` (list of per‑case records) and
        ``average`` (aggregate metric averages).
    """
    # Load dataset and persona
    raw_cases = load_test_cases(dataset_path)
    persona = _load_persona(config.persona_path)
    prompt_template = _load_template(config.prompt_template)
    # Instantiate LLM
    provider = LLMProvider(config.model_provider, config.model_name, config.temperature, config.max_tokens)
    llm = provider.get_llm()
    # Build test case objects (without retrieval context)
    test_cases = build_test_cases(raw_cases)
    # Preload metric constructors
    metric_constructors = available_metrics()
    metrics_to_run = [metric_constructors[m]() for m in config.metrics]
    results = []
    # Iterate through each test case
    for tc_raw, tc_obj in zip(raw_cases, test_cases):
        # Build the user prompt from the template
        prompt = prompt_template.render(question=tc_raw["input"])
        full_prompt = f"{persona}\n\n{prompt}"
        # Generate output
        generated = _invoke_llm(llm, full_prompt)
        # Update the test case object with the actual output
        if _DEEPEVAL_AVAILABLE:
            tc_obj.actual_output = generated
        else:
            tc_obj["actual_output"] = generated
        # Compute metrics per test case
        case_scores: Dict[str, float] = {}
        for metric_name, metric in zip(config.metrics, metrics_to_run):
            try:
                metric.measure(tc_obj)
                score = getattr(metric, "score", None)
            except Exception as exc:
                logger.error(f"Metric {metric_name} failed: {exc}")
                score = 0.0
            case_scores[metric_name] = score if score is not None else 0.0
        results.append({
            "input": tc_raw["input"],
            "expected_output": tc_raw.get("expected_output"),
            "actual_output": generated,
            "metrics": case_scores,
        })
    # Compute averages
    averages: Dict[str, float] = {}
    for m in config.metrics:
        total = sum(r["metrics"].get(m, 0.0) for r in results)
        averages[m] = total / len(results) if results else 0.0
    return {"results": results, "average": averages}


def run_rag_tests(config: RAGEvalConfig, dataset_path: str) -> Dict[str, Any]:
    """Execute RAG based tests.

    Parameters
    ----------
    config : RAGEvalConfig
        Settings for the RAG evaluation including retrieval top_k and metrics.
    dataset_path : str
        Path to the JSON file containing test cases.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``results`` (list of per‑case records) and
        ``average`` (aggregate metric averages).
    """
    # Load dataset, persona and context
    raw_cases = load_test_cases(dataset_path)
    persona = _load_persona(config.persona_path)
    template = _load_template(config.rag_prompt_template)
    context_docs = load_rag_context(config.context_path)
    # Fit retriever on entire context
    retriever = SimpleTFIDFRetriever()
    retriever.fit(context_docs)
    # Instantiate LLM
    provider = LLMProvider(config.model_provider, config.model_name, config.temperature, config.max_tokens)
    llm = provider.get_llm()
    # Build test case objects with expected outputs for metrics requiring them
    test_cases = build_test_cases(raw_cases)
    metric_constructors = available_metrics()
    metrics_to_run = [metric_constructors[m]() for m in config.metrics]
    results = []
    for tc_raw, tc_obj in zip(raw_cases, test_cases):
        question = tc_raw["input"]
        # Retrieve top_k contexts
        retrieved = retriever.retrieve(question, top_k=config.top_k)
        context_list = [doc for doc, _score in retrieved]
        context_text = "\n\n".join(context_list)
        # Build prompt with context
        prompt = template.render(question=question, context=context_text)
        full_prompt = f"{persona}\n\n{prompt}"
        generated = _invoke_llm(llm, full_prompt)
        # Attach actual output and retrieved context to test case object
        if _DEEPEVAL_AVAILABLE:
            tc_obj.actual_output = generated
            tc_obj.retrieval_context = context_list
        else:
            tc_obj["actual_output"] = generated
            tc_obj["retrieval_context"] = context_list
        case_scores: Dict[str, float] = {}
        for metric_name, metric in zip(config.metrics, metrics_to_run):
            try:
                metric.measure(tc_obj)
                score = getattr(metric, "score", None)
            except Exception as exc:
                logger.error(f"Metric {metric_name} failed: {exc}")
                score = 0.0
            case_scores[metric_name] = score if score is not None else 0.0
        results.append({
            "input": question,
            "expected_output": tc_raw.get("expected_output"),
            "actual_output": generated,
            "retrieval_context": context_list,
            "metrics": case_scores,
        })
    averages: Dict[str, float] = {}
    for m in config.metrics:
        total = sum(r["metrics"].get(m, 0.0) for r in results)
        averages[m] = total / len(results) if results else 0.0
    return {"results": results, "average": averages}