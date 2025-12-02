"""Unit tests for the clinic LLM evaluation framework.

These tests run locally without requiring access to external LLMs.
They monkey‑patch the :class:`LLMProvider` to return a simple mock
model whose outputs are deterministic.  This allows us to verify that
the evaluation pipeline computes metric scores correctly and aggregates
them as expected.
"""

from pathlib import Path
import json
import types

import pytest

from clinic_llm_test_framework.evaluator import run_prompt_tests, run_rag_tests
from clinic_llm_test_framework.config import PromptEvalConfig, RAGEvalConfig
from clinic_llm_test_framework.llm_provider import LLMProvider


class MockModel:
    """A mock chat model used in unit tests.

    It returns the expected output for a given question based on a
    mapping loaded from the evaluation dataset.  If the question is
    unknown it returns a generic string.
    """

    def __init__(self, answer_map: dict[str, str]) -> None:
        self.answer_map = answer_map

    def predict(self, prompt: str) -> str:
        # Extract the question from the prompt by taking the last line
        # after 'Patient:' to reduce coupling to template details.
        if "Patient:" in prompt:
            question = prompt.split("Patient:")[-1].strip()
        else:
            # fallback
            question = prompt.strip()
        return self.answer_map.get(question, "MOCK_ANSWER")


@pytest.fixture
def answer_map() -> dict[str, str]:
    # Build a mapping from the input questions to expected answers
    dataset_path = Path(__file__).resolve().parents[2] / "clinic_llm_test_framework" / "data" / "clinic_qa.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    mapping = {item["input"]: item["expected_output"] for item in data}
    return mapping


def test_prompt_evaluation_exact_match(monkeypatch, answer_map) -> None:
    """Ensure that exact_match metric yields perfect scores when model returns expected outputs."""
    # Monkey patch LLMProvider.get_llm to return our MockModel
    def fake_get_llm(self):
        return MockModel(answer_map)

    monkeypatch.setattr(LLMProvider, "get_llm", fake_get_llm)
    # Configure evaluation to run only exact_match
    cfg = PromptEvalConfig(
        model_provider="openai", metrics=["exact_match"], prompt_template="prompts/prompt_template.j2",
        persona_path="prompts/system_persona.txt"
    )
    dataset_path = str(Path(__file__).resolve().parents[2] / "clinic_llm_test_framework" / "data" / "clinic_qa.json")
    results = run_prompt_tests(cfg, dataset_path)
    # Each case should have metric 1.0
    for case in results["results"]:
        assert abs(case["metrics"]["exact_match"] - 1.0) < 1e-6
    # Average should also be 1.0
    assert abs(results["average"]["exact_match"] - 1.0) < 1e-6


def test_rag_evaluation_answer_relevancy(monkeypatch, answer_map) -> None:
    """Ensure RAG evaluation runs without error and returns scores in [0,1]."""
    def fake_get_llm(self):
        return MockModel(answer_map)

    monkeypatch.setattr(LLMProvider, "get_llm", fake_get_llm)
    cfg = RAGEvalConfig(
        model_provider="openai", metrics=["exact_match"],
        rag_prompt_template="prompts/rag_prompt_template.j2",
        persona_path="prompts/system_persona.txt",
        context_path="data/clinic_context.txt",
        top_k=2,
    )
    dataset_path = str(Path(__file__).resolve().parents[2] / "clinic_llm_test_framework" / "data" / "clinic_qa.json")
    results = run_rag_tests(cfg, dataset_path)
    # Metric values should be between 0 and 1
    for case in results["results"]:
        for score in case["metrics"].values():
            assert 0.0 <= score <= 1.0