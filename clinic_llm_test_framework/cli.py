"""
cli.py
~~~~~~

This module exposes a simple command‑line interface (CLI) for running
evaluations against the clinic LLM test framework.  It supports two
modes of operation:

* **prompt** – evaluate the model without any retrieval (pure prompt
  testing).
* **rag** – evaluate the model using retrieval augmented generation
  (RAG) by first retrieving context and then injecting it into the
  prompt.

The CLI is intentionally minimal: you specify the mode, the path to
the JSON dataset of test cases and optionally override model provider,
model name, temperature and maximum tokens.  If the dataset path
does not exist relative to your current working directory, the CLI
will attempt to resolve it relative to the installed package’s data
directory.  This makes the command portable regardless of your
current working directory.

Example usage::

    # Run prompt‑only evaluation against the built‑in dataset
    python -m clinic_llm_test_framework.cli \
        --mode prompt \
        --dataset clinic_llm_test_framework/data/clinic_qa.json

    # Run RAG evaluation using the same dataset
    python -m clinic_llm_test_framework.cli \
        --mode rag \
        --dataset clinic_llm_test_framework/data/clinic_qa.json

If you install the package into a virtual environment or as a pip
package, the relative dataset path will be resolved automatically.  You
can also provide an absolute path to your own dataset file.

"""

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import PromptEvalConfig, RAGEvalConfig
from .evaluator import run_prompt_tests, run_rag_tests


def _resolve_dataset_path(dataset: str) -> Path:
    """Resolve the dataset path.

    First checks whether the given path exists as provided.  If not,
    attempts to resolve it relative to the installed package’s root
    directory.  This allows users to run the CLI from any location
    without worrying about relative paths.  Raises SystemExit if the
    file cannot be found.

    Args:
        dataset: User supplied dataset path (relative or absolute).

    Returns:
        Path object pointing to the dataset file.
    """
    candidate = Path(dataset)
    if candidate.is_file():
        return candidate
    # Fall back: resolve relative to package
    from importlib import resources

    try:
        # Construct a resource path relative to this package
        with resources.path(__package__, "__init__.py") as pkg_path:
            pkg_root = pkg_path.parent
        fallback = pkg_root / dataset
        if fallback.is_file():
            return fallback
    except Exception:
        pass
    raise SystemExit(f"Dataset not found: {dataset}")


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI when used with ``python -m``.

    Parses command line arguments, resolves the dataset path,
    constructs a configuration object and runs the appropriate
    evaluation.  Prints the results as a JSON object.

    Args:
        argv: Optional list of arguments.  If None, defaults to
            ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Clinic LLM Test Framework – run evaluations against a dataset."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["prompt", "rag"],
        required=True,
        help="Evaluation mode: 'prompt' for prompt‑only tests or 'rag' for RAG tests.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Path to the JSON dataset of test cases.  Relative paths are resolved "
            "against the current working directory and, if not found, against the "
            "installed package’s data directory."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "google"],
        default="openai",
        help="LLM provider to use (default: openai).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name override (e.g. 'gpt-4o-mini').",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (if supported).",
    )
    args = parser.parse_args(argv)
    dataset_path = _resolve_dataset_path(args.dataset)
    if args.mode == "prompt":
        cfg = PromptEvalConfig(
            model_provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        results = run_prompt_tests(cfg, str(dataset_path))
    else:
        cfg = RAGEvalConfig(
            model_provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        results = run_rag_tests(cfg, str(dataset_path))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()