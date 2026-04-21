from __future__ import annotations

from pydantic import BaseModel, Field


class Agent3KnowledgeResult(BaseModel):
    case_summary: str = Field(description="Compact summary of the coding-relevant clinical pattern in this case.")
    salient_clinical_patterns: list[str] = Field(
        default_factory=list,
        description="Reusable clinical patterns or cues that mattered for coding this case.",
    )
    correct_prediction_reasons: list[str] = Field(
        default_factory=list,
        description="Short lessons about why correctly predicted codes were supported.",
    )
    missed_code_lessons: list[str] = Field(
        default_factory=list,
        description="Short lessons explaining why any gold codes were missed and what clues future coding should watch for.",
    )
    unsupported_prediction_lessons: list[str] = Field(
        default_factory=list,
        description="Short lessons explaining why any predicted-only codes were weak or unsupported.",
    )
    coding_lessons: list[str] = Field(
        default_factory=list,
        description="Generalized coding heuristics distilled from this case.",
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="Short future search phrases that should retrieve this memory for similar cases.",
    )
    knowledge_summary: str = Field(
        description="One concise reusable summary that Agent 2 can later consult as knowledge."
    )
