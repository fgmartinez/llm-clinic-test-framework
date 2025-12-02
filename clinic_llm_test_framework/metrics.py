"""Metric definitions and helpers for evaluation.

DeepEval provides a suite of research‑backed metrics for testing LLM
applications.  This module defines a mapping from human friendly names
to metric objects so that the evaluator can configure which metrics to
run based on a configuration file.  We también incluimos métricas de
seguridad y sesgo que dependen de LLM-as-a-judge.

Nota importante:
La función exact_match_score existe por compatibilidad, pero siempre devuelve 0.
Se recomienda utilizar métricas basadas en LLM‑as‑a‑judge (relevancy, faithfulness, toxicity, etc.).

"""

from __future__ import annotations
import logging
import string
from typing import Callable, Dict

try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        GEval,
    )
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(translator).split())


def available_metrics() -> Dict[str, Callable[[], object]]:
    """Return a mapping of metric names to constructors."""

    metrics: Dict[str, Callable[[], object]] = {}
    if _DEEPEVAL_AVAILABLE:
        # Métricas de relevancia y RAG
        metrics["answer_relevancy"] = lambda: AnswerRelevancyMetric(threshold=0.5)
        metrics["faithfulness"] = lambda: FaithfulnessMetric(threshold=0.5)
        metrics["contextual_precision"] = lambda: ContextualPrecisionMetric(threshold=0.5)

        # Métricas de seguridad y sesgo
        try:
            from deepeval.metrics import (
                ToxicityMetric,
                BiasMetric,
                NonAdviceMetric,
                PIILeakageMetric,
                RoleViolationMetric,
                MisuseMetric,
                HallucinationMetric,
            )

            metrics["toxicity"] = lambda: ToxicityMetric(threshold=0.5)
            metrics["bias"] = lambda: BiasMetric(threshold=0.5)
            metrics["non_advice"] = lambda: NonAdviceMetric(advice_types=["medical", "financial"], threshold=0.5)
            metrics["pii_leakage"] = lambda: PIILeakageMetric(threshold=0.5)
            metrics["role_violation"] = lambda role="Clinic assistant": RoleViolationMetric(role=role, threshold=0.5)
            metrics["misuse"] = lambda domain="medical": MisuseMetric(domain=domain, threshold=0.5)
            metrics["hallucination"] = lambda: HallucinationMetric(threshold=0.5)
        except ImportError:
            logger.warning("Optional safety metrics could not be imported from deepeval.")

        # Métrica GEval de corrección (ejemplo de criterio personalizado)
        metrics["geval_correctness"] = lambda: GEval(
            name="Correctness",
            model="gpt-4o-mini",
            evaluation_params=["expected_output", "actual_output"],
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradict any facts in 'expected output'.",
                "Penalize hallucinations or false statements.",
                "Penalize omissions of critical details; reward concise and accurate summaries.",
            ],
            threshold=0.5,
        )
    else:
        # Stub functions si deepeval no está instalado
        def _raise_no_deepeval(metric_name: str):
            def _ctor():
                raise ImportError(
                    f"Metric '{metric_name}' requires the deepeval package. "
                    "Install deepeval to use this metric."
                )
            return _ctor

        metrics["answer_relevancy"] = _raise_no_deepeval("answer_relevancy")
        metrics["faithfulness"] = _raise_no_deepeval("faithfulness")
        metrics["contextual_precision"] = _raise_no_deepeval("contextual_precision")
        metrics["geval_correctness"] = _raise_no_deepeval("geval_correctness")

    return metrics
