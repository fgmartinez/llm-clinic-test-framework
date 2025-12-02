"""Configuration objects for the clinic LLM test framework."""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PromptEvalConfig:
    """Configuration for prompt‑only evaluation."""
    model_provider: str = "openai"
    model_name: Optional[str] = None
    temperature: float = 0.0
    metrics: List[str] = field(default_factory=lambda: [
        "answer_relevancy",
        "role_violation",
        "non_advice",
        "toxicity",
        "bias",
    ])
    prompt_template: str = "prompts/prompt_template.j2"
    persona_path: str = "prompts/system_persona.txt"
    max_tokens: Optional[int] = None

@dataclass
class RAGEvalConfig:
    """Configuration for RAG evaluation."""
    model_provider: str = "openai"
    model_name: Optional[str] = None
    temperature: float = 0.0
    metrics: List[str] = field(default_factory=lambda: [
        "answer_relevancy",
        "faithfulness",
        "contextual_precision",
        "role_violation",
        "non_advice",
        "toxicity",
        "bias",
    ])
    rag_prompt_template: str = "prompts/rag_prompt_template.j2"
    persona_path: str = "prompts/system_persona.txt"
    context_path: str = "data/clinic_context.txt"
    top_k: int = 3
    max_tokens: Optional[int] = None
