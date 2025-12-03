import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the evaluation dataset (Q&A cases)
DATASET_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]  # -> clinic_llm_test_framework/
    / "data"
    / "clinic_qa.json"
)


@pytest.mark.integration
def test_cli_prompt_mode_uses_real_api() -> None:
    """
    End-to-end test: CLI in 'prompt' mode calling the real LLM.

    This test will be skipped if OPENAI_API_KEY is not set in the environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping live prompt test.")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clinic_llm_test_framework.cli",
            "--mode",
            "prompt",
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed in prompt mode: {result.stderr}"
    assert result.stdout.strip(), "CLI prompt mode produced empty output."


@pytest.mark.integration
def test_cli_rag_mode_uses_real_api() -> None:
    """
    End-to-end test: CLI in 'rag' mode calling the real LLM.

    This test will be skipped if OPENAI_API_KEY is not set in the environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping live RAG test.")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clinic_llm_test_framework.cli",
            "--mode",
            "rag",
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed in RAG mode: {result.stderr}"
    assert result.stdout.strip(), "CLI RAG mode produced empty output."
