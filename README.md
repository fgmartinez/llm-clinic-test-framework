# Clinic LLM Test Framework

A clean, modular, production-ready evaluation framework for testing Large Language Models (LLMs) in a medical-clinic assistant setting.

The framework supports real end-to-end testing using the OpenAI API, in both **prompt-only mode** and **RAG mode** (Retrieval-Augmented Generation). It is intentionally simple, explicit, and fully runnable either locally or inside Docker.  
All tests use **real model responses** — no mocking, no synthetic outputs.

---

## 🚀 Features

- End-to-end LLM evaluation (real OpenAI API)
- Prompt-based and RAG-based evaluation modes
- TF-IDF retriever for contextual augmentation
- Modular, clean, easy-to-extend architecture
- Run locally or in Docker with identical behavior
- GitHub Actions support using Docker
- Simple, explicit CLI for manual runs

---

## 📁 Project Structure

```
clinic_llm_test_framework/
├── __init__.py
├── cli.py
├── config.py
├── dataset_loader.py
├── evaluator.py
├── llm_provider.py
├── metrics.py
├── retriever.py
├── prompts/
│   ├── system_persona.txt
│   ├── prompt_template.j2
│   └── rag_prompt_template.j2
├── data/
│   ├── clinic_context.txt
│   └── clinic_qa.json
├── test_case_builder.py
├── setup.py
└── tests/
    └── test_cli_live.py        # End-to-end test: prompt + RAG using real API
```

---

# 🔧 Installation (Local Python)

## 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

## 2. Install the framework

```bash
pip install -e .
pip install pytest
```

## 3. Configure your API key

Create a `.env` file (NOT committed to Git):

```
OPENAI_API_KEY=sk-...
```

Load it into your session:

```bash
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2-)
```

---

# ▶️ Running the Framework Locally

## Prompt-only evaluation (no RAG)

```bash
python -m clinic_llm_test_framework.cli   --mode prompt   --dataset clinic_llm_test_framework/data/clinic_qa.json
```

## RAG evaluation (TF-IDF retriever)

```bash
python -m clinic_llm_test_framework.cli   --mode rag   --dataset clinic_llm_test_framework/data/clinic_qa.json
```

---

# 🧪 Running Tests (Local)

The test suite includes two end-to-end tests:

- Running the CLI in **prompt mode** using the real LLM  
- Running the CLI in **RAG mode** using the real LLM  

Execute them with:

```bash
pytest -q
```

If `OPENAI_API_KEY` is not set, the tests are automatically skipped.

---

# 🐳 Running With Docker

Docker provides a reproducible runtime environment for local development and CI.

## 1. Build the Docker image

```bash
docker build -t llm-clinic-test-framework .
```

## 2. Run the entire test suite inside Docker (real API)

```bash
docker run --rm   -e OPENAI_API_KEY="sk-..."   llm-clinic-test-framework
```

## 3. Manually run the CLI inside Docker

### Prompt mode:

```bash
docker run --rm   -e OPENAI_API_KEY="sk-..."   llm-clinic-test-framework   python -m clinic_llm_test_framework.cli     --mode prompt     --dataset clinic_llm_test_framework/data/clinic_qa.json
```

### RAG mode:

```bash
docker run --rm   -e OPENAI_API_KEY="sk-..."   llm-clinic-test-framework   python -m clinic_llm_test_framework.cli     --mode rag     --dataset clinic_llm_test_framework/data/clinic_qa.json
```

---

# 🔄 GitHub Actions (CI)

The repository includes a Docker-based workflow (`docker-tests.yml`) that:

1. Builds the Docker image  
2. Injects your encrypted GitHub Secret:

   - `OPENAI_API_KEY`

3. Runs the full pytest suite inside Docker  
4. Validates both **prompt** and **RAG** execution using the real LLM  

No secrets are stored in the repository or inside the Docker image.

---

# 🛡️ Security Notes

- `.env` is ignored by git — never commit API keys.
- Docker images never contain secrets.
- GitHub Actions passes API keys at runtime through `secrets`.

---

# ✔️ Summary of Commands

| Task | Command |
|------|---------|
| Install locally | `pip install -e .` |
| Run prompt mode | `python -m clinic_llm_test_framework.cli --mode prompt --dataset clinic_llm_test_framework/data/clinic_qa.json` |
| Run RAG mode | `python -m clinic_llm_test_framework.cli --mode rag --dataset clinic_llm_test_framework/data/clinic_qa.json` |
| Run tests locally | `pytest -q` |
| Build Docker image | `docker build -t llm-clinic-test-framework .` |
| Run tests in Docker | `docker run --rm -e OPENAI_API_KEY=... llm-clinic-test-framework` |

---

# 🧠 Final Note

This framework is intentionally simple and explicit.  
All tests run against the **real** LLM provider, ensuring predictable, meaningful evaluation behavior both locally and in CI.
