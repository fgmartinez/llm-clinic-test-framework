# Clinic LLM Test Framework

This project provides a clean, modular, and professional testing framework for evaluating Large Language Models (LLMs) in a medical-clinic assistant setting.  
It is designed both for learning and as a portfolio-grade real-world evaluation framework.

---

## 🚀 Features

- **Modular architecture**: configuration, providers, prompts, metrics, retrieval, evaluator.
- **LLM test cases** powered by [DeepEval](https://github.com/confident-ai/deepeval).
- **RAG support** using a simple TF-IDF retriever.
- **Mixed metrics**: DeepEval metrics + lightweight heuristic checks.
- **Highly configurable** via CLI and Python modules.
- **Offline tests** using monkey-patched LLM providers (no external calls).
- **Real-API tests** (OpenAI / Google) available on demand.
- **Docker support** for reproducible runs.
- **GitHub Actions CI** with Docker-based test execution.

---

## 📁 Project layout

```text
clinic_llm_test_framework/
├── __init__.py
├── config.py
├── dataset_loader.py
├── evaluator.py
├── llm_provider.py
├── metrics.py
├── prompts/
│   ├── system_persona.txt
│   ├── prompt_template.j2
│   └── rag_prompt_template.j2
├── data/
│   ├── clinic_context.txt
│   └── clinic_qa.json
├── test_case_builder.py
├── retriever.py
├── environment.yml
├── setup.py
└── tests/
    ├── test_evaluator.py       # Offline tests using monkey-patched LLM
    └── test_real_api.py        # Opt-in tests using real LLM APIs

    
## Docker and CI

This project can be tested inside a Docker container, both for prompt-only
and RAG evaluations.

### Build the Docker image

From the repository root:

```bash
docker build -t llm-clinic-test-framework .
