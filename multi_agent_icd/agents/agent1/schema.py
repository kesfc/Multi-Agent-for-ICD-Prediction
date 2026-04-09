from __future__ import annotations

from pydantic import BaseModel, Field


class Agent1CaseSummary(BaseModel):
    gender: str = Field(description="Short normalized gender value such as male, female, or unknown.")
    chief_complaint: str = Field(description="Short cleaned chief complaint string.")
    procedure: list[str] = Field(default_factory=list, description="List of major procedures as short cleaned phrases.")
    history_present_illness: str = Field(description="Compact prose summary of the history of present illness.")
    past_medical_history: list[str] = Field(default_factory=list, description="List of past medical history items.")
    physical_exam_discharge: list[str] = Field(default_factory=list, description="List of cleaned discharge physical exam findings.")
    pertinent_results: list[str] = Field(default_factory=list, description="List of key imaging, lab, or study results.")
    hospital_course: list[str] = Field(default_factory=list, description="List of concise hospital course events.")
    discharge_diagnosis: list[str] = Field(default_factory=list, description="List of cleaned discharge diagnoses.")
