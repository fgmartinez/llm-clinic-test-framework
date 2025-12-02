"""Construct ``LLMTestCase`` objects from raw data.

DeepEval represents each unit of evaluation as an :class:`LLMTestCase`
object containing input text, the actual model output, an expected
output, and optionally a retrieval context.  This module provides
helpers for building such objects from the data structures returned
by :func:`dataset_loader.load_test_cases`.

If DeepEval is not available, the returned objects are simple
dictionaries with analogous fields; the evaluator treats them
uniformly by checking for attributes at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from deepeval.test_case import LLMTestCase
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False


def build_test_cases(
    raw_cases: List[Dict[str, Any]],
    rag_context: Optional[List[str]] = None,
) -> List[Any]:
    """Build a list of test case objects from raw dicts.

    Parameters
    ----------
    raw_cases:
        List of dictionaries with keys ``input``, ``expected_output`` and
        optionally ``context_ids``.
    rag_context:
        Optional list of documents used for retrieval.  If provided
        and a test case includes ``context_ids`` these indices will be
        used to populate the ``retrieval_context`` field.

    Returns
    -------
    list
        Instances of ``deepeval.test_case.LLMTestCase`` when DeepEval
        is available, otherwise simple dictionaries with analogous
        attributes.
    """
    cases: List[Any] = []
    for item in raw_cases:
        inp = item.get("input")
        if inp is None:
            raise ValueError("Every test case must define an 'input' field")
        expected_output = item.get("expected_output")
        context_ids: List[int] = item.get("context_ids", [])
        retrieval_context: Optional[List[str]] = None
        if rag_context is not None and context_ids:
            retrieval_context = [rag_context[i] for i in context_ids if i < len(rag_context)]
        if _DEEPEVAL_AVAILABLE:
            # Build LLMTestCase with the expected fields.  Note: expected_output
            # may be None; metrics that require it will handle accordingly.
            cases.append(
                LLMTestCase(
                    input=inp,
                    expected_output=expected_output,
                    retrieval_context=retrieval_context,
                )
            )
        else:
            # Use a simple object with matching attributes so that the
            # evaluator can access .input, .expected_output etc.
            case = {
                "input": inp,
                "expected_output": expected_output,
                "retrieval_context": retrieval_context,
            }
            cases.append(case)
    return cases