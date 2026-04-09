from __future__ import annotations

import json
from typing import Any

from multi_agent_icd.agents.agent1.prompt import build_agent1_prompts
from multi_agent_icd.providers import OpenAIResponsesLLM
from multi_agent_icd.utils.clinical_text import (
    build_evidence_index,
    extract_patient_snapshot,
    get_primary_complaint,
    normalize_clinical_text,
)
from multi_agent_icd.utils.schema import normalize_agent1_output


def _try_parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError("LLM output was neither a dict nor a JSON string.")
    return json.loads(value)


class Agent1PrimaryAnalyzer:
    def __init__(
        self,
        model_name: str | None = None,
        llm: Any | None = None,
    ) -> None:
        if llm is None and not model_name:
            raise ValueError("Agent1PrimaryAnalyzer requires a model_name or a custom llm.")

        self.model_name = model_name
        self.llm = llm or OpenAIResponsesLLM(model_name=model_name)

    def run(
        self,
        note_text: str,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_note = normalize_clinical_text(note_text)
        if not normalized_note:
            raise ValueError("note_text is required for Agent 1.")

        patient_context = patient_context or {}
        evidence_index = build_evidence_index(normalized_note)
        patient_snapshot = extract_patient_snapshot(normalized_note, patient_context)
        prompts = build_agent1_prompts(
            note_text=normalized_note,
            patient_context={**patient_context, **patient_snapshot},
            evidence_index=evidence_index,
        )

        if hasattr(self.llm, "generate_json"):
            raw_result = self.llm.generate_json(
                system_prompt=prompts["system_prompt"],
                user_prompt=prompts["user_prompt"],
                metadata={
                    "agent": "agent1",
                    "role": "primary_analyzer",
                    "model": self.model_name or "",
                },
            )
        elif hasattr(self.llm, "generateJson"):
            raw_result = self.llm.generateJson(
                systemPrompt=prompts["system_prompt"],
                userPrompt=prompts["user_prompt"],
                metadata={
                    "agent": "agent1",
                    "role": "primary_analyzer",
                    "model": self.model_name or "",
                },
            )
        else:
            raise AttributeError("llm must expose generate_json(...) or generateJson(...).")

        parsed = _try_parse_json(raw_result)
        chief_complaint = get_primary_complaint(evidence_index)
        if chief_complaint and not parsed.get("chief_complaint"):
            parsed["chief_complaint"] = chief_complaint
        return normalize_agent1_output(parsed, patient_context=patient_snapshot)


def run_agent1(
    note_text: str,
    patient_context: dict[str, Any] | None = None,
    model_name: str | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    return Agent1PrimaryAnalyzer(model_name=model_name, llm=llm).run(
        note_text=note_text,
        patient_context=patient_context,
    )
