# Clinic LLM Test Framework

This package implements a simple yet professional framework for testing
large language models (LLMs) in the context of a medical clinic
assistant.  The framework is designed both for learning how to
structure evaluations and for inclusion in a personal portfolio.

## Features

* **Modular architecture** – separate modules handle configuration,
  LLM instantiation, retrieval, prompt rendering, metric execution and
  evaluation orchestration.
* **LLM Test Cases** – uses the [`deepeval`](https://github.com/confident-ai/deepeval)
  library to build unit tests for LLM outputs.  Test cases include
  inputs, expected outputs and retrieved contexts.
* **Retrieval Augmented Generation (RAG)** – includes a TF‑IDF
  retriever and prompt template for experiments with context injection.
* **Metrics** – supports DeepEval metrics such as answer relevancy
  and faithfulness, as well as simple heuristics like exact match.
* **Parametrisable** – configuration classes allow you to control
  model provider, model name, temperature, metrics, top‑K retrieval,
  templates and persona file.
* **Continuous Integration Ready** – includes a conda environment and
  GitHub Actions workflow to run the tests automatically.

## Directory layout

```
clinic_llm_test_framework/
├── __init__.py
├── config.py            # Evaluation configuration definitions
├── dataset_loader.py    # Helpers for loading datasets and context
├── evaluator.py         # Functions to run prompt and RAG tests
├── llm_provider.py      # LLM factory (OpenAI or Google)
├── metrics.py           # Metric constructors (DeepEval + simple)
├── prompts/
│   ├── system_persona.txt        # Persona for MedAssist
│   ├── prompt_template.j2        # Template for direct Q&A
│   └── rag_prompt_template.j2    # Template for RAG tests
├── data/
│   ├── clinic_context.txt        # RAG knowledge base
│   └── clinic_qa.json            # Evaluation test cases
├── test_case_builder.py # Builds LLMTestCase objects
├── retriever.py         # Simple TF-IDF retriever
├── environment.yml      # Conda environment specification
├── setup.py             # Package metadata for editable install
└── tests/
    └── test_evaluator.py # Unit tests with monkey-patched LLM
```

## Quick start

The framework is portable and does not depend on being run from a
specific directory.  Once installed, you can run evaluations from any
location.  We recommend using a virtual environment with Python 3.11
or newer to ensure compatibility.

1. **Create and activate a virtual environment** (if you don’t
   already have one)::

       python3.11 -m venv .venv
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate

2. **Install the package in editable mode**.  From the root of this
   repository run::

       pip install -e .

   This installs `clinic_llm_test_framework` and its dependencies and
   makes the command line interface available via `python -m`.

3. **Configure your API keys**.  DeepEval metrics and LangChain
   providers require API keys.  Create a `.env` file in the project
   root with your credentials::

       OPENAI_API_KEY=sk-…
       GOOGLE_API_KEY=…

   Alternatively export these variables in your shell.  If you do not
   require Google models you can omit `GOOGLE_API_KEY`.

4. **Run evaluations** using the CLI.  The CLI resolves dataset
   paths relative to your current working directory and falls back to
   the installed package directory, so the following commands work
   regardless of where you run them:

   *Prompt‑only (no retrieval)::*

       python -m clinic_llm_test_framework.cli --mode prompt \
         --dataset clinic_llm_test_framework/data/clinic_qa.json

   *Retrieval augmented (RAG)::*

       python -m clinic_llm_test_framework.cli --mode rag \
         --dataset clinic_llm_test_framework/data/clinic_qa.json

   To customise the LLM provider, model name, temperature or maximum
   tokens pass `--provider`, `--model-name`, `--temperature` and
   `--max-tokens`.  Run `python -m clinic_llm_test_framework.cli --help`
   for full usage.

5. **Run the tests** to verify that everything is wired correctly.
   After activating your virtual environment, run::

       pytest -q

   The tests monkey‑patch the LLM provider to avoid external API calls,
   so they run offline.

6. **Continuous integration**.  A GitHub Actions workflow (`test.yml`)
   is provided.  It uses conda to create an environment with
   Python 3.11 and runs the unit tests.  Adapt this workflow for
   your own CI server if needed.

## Contributing / Extending

This framework is intentionally simple.  To extend it you might:

* Swap the TF‑IDF retriever for an embedding based index (FAISS,
  Chroma, Milvus, etc.).
* Add new DeepEval metrics or custom heuristics in `metrics.py`.
* Expand the evaluation dataset with more realistic patient queries and
  expected outputs.
* Integrate LangChain’s agents or memory components for more complex
  conversation logic.

## References

* The DeepEval metrics **answer relevancy**, **faithfulness** and
  **contextual precision** measure different aspects of RAG pipelines【568574491278341†L210-L259】.
* Example test case definitions using DeepEval illustrate how to
  construct LLM test cases and run metrics in a PyTest style【185838769261353†L187-L246】.
