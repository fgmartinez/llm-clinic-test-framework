"""
Tests that hit a real LLM API.

These tests are DISABLED by default and only run if:
  1) You pass --run-real-api to pytest, AND
  2) OPENAI_API_KEY is set in the environment.

Typical usage (local):
    export OPENAI_API_KEY=sk-...
    pytest -q --run-real-api

Typical usage (Docker):
    docker run --rm \
      --env-file .env \
      llm-clinic-test-framework \
      pytest -q --run-real-api
"""

import os
import subprocess
import sys

import pytest


DATASET_PATH = "clinic_llm_test_framework/data/clinic_qa.json"


@pytest.mark.integration
def test_real_llm_prompt_mode(run_real_api):
    """
    Run the CLI in prompt mode hitting a real LLM API (OpenAI).

    This test:
      - Ensures the CLI can execute end-to-end
      - Uses the real provider configuration
      - Does NOT assert semantic quality of the answer, only that we
        get a non-empty response and a zero exit code.
    """
    if not run_real_api:
        pytest.skip("Real API tests disabled. Use --run-real-api to enable.")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real API test.")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clinic_llm_test_framework.cli",
            "--mode",
            "prompt",
            "--dataset",
            DATASET_PATH,
        ],
        capture_output=True,
        text=True,
