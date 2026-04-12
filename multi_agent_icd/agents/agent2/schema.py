from __future__ import annotations

from pydantic import BaseModel, Field


class Agent2CodeCandidate(BaseModel):
    code: str = Field(description="Best supported ICD code candidate.")
    description: str = Field(description="Short description for the ICD code candidate.")
    code_system: str = Field(description="Dataset-aligned code system such as ICD-10-CM, ICD-10-PCS, or ICD-9-CM.")
    category: str = Field(description="principal_diagnosis, secondary_diagnosis, or procedure.")
    confidence: str = Field(description="high, medium, or low.")
    rationale: str = Field(description="Short evidence-grounded explanation for the code candidate.")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence ids such as E1 and E12.")
    missing_details: list[str] = Field(
        default_factory=list,
        description="Specific documentation gaps that prevent more precise coding.",
    )


class Agent2CodingResult(BaseModel):
    principal_diagnosis: Agent2CodeCandidate | None = Field(
        default=None,
        description="Best supported principal diagnosis candidate, or null if not supportable.",
    )
    secondary_diagnoses: list[Agent2CodeCandidate] = Field(
        default_factory=list,
        description="Supported secondary ICD-10-CM diagnosis candidates.",
    )
    procedures: list[Agent2CodeCandidate] = Field(
        default_factory=list,
        description="Supported ICD-10-PCS procedure candidates.",
    )
    coding_queries: list[str] = Field(
        default_factory=list,
        description="Targeted provider queries or missing documentation items.",
    )
    coding_summary: str = Field(description="Short summary of the coding recommendation.")
