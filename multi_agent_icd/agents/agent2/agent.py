from __future__ import annotations

import json
from typing import Any

from multi_agent_icd.agents.agent2.prompt import build_agent2_prompts, resolve_agent2_code_systems
from multi_agent_icd.agents.agent2.schema import Agent2CodingResult
from multi_agent_icd.providers import LocalQwenLLM
from multi_agent_icd.utils.clinical_text import build_evidence_index, normalize_clinical_text
from multi_agent_icd.utils.schema import normalize_agent2_output


def _try_parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError("LLM output was neither a dict nor a JSON string.")
    return json.loads(value)


class Agent2Coder:
    def __init__(
        self,
        model_name: str | None = None,
        llm: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.llm = llm or LocalQwenLLM(model_name=model_name)

    def run(self, state: Any) -> dict[str, Any]:
        note_text = normalize_clinical_text(getattr(state, "note_text", ""))
        if not note_text:
            raise ValueError("note_text is required for Agent 2.")

        patient_context = dict(getattr(state, "patient_context", {}) or {})
        _, diagnosis_code_system, procedure_code_system = resolve_agent2_code_systems(patient_context)
        shared_memory = getattr(state, "shared_memory", {}) or {}
        agent_outputs = getattr(state, "agent_outputs", {}) or {}
        structured_case_summary = shared_memory.get("structured_case_summary") or agent_outputs.get("agent1")
        if not isinstance(structured_case_summary, dict):
            raise ValueError("Agent 2 requires a structured Agent 1 summary in the pipeline state.")

        evidence_index = shared_memory.get("note_evidence_index")
        if not isinstance(evidence_index, list):
            evidence_index = build_evidence_index(note_text)

        prompts = build_agent2_prompts(
            structured_case_summary=structured_case_summary,
            patient_context=patient_context,
            evidence_index=evidence_index,
        )

        if hasattr(self.llm, "generate_json"):
            raw_result = self.llm.generate_json(
                system_prompt=prompts["system_prompt"],
                user_prompt=prompts["user_prompt"],
                metadata={
                    "agent": "agent2",
                    "role": "coder",
                    "model": self.model_name or getattr(self.llm, "model_name", ""),
                },
                response_model=Agent2CodingResult,
            )
        elif hasattr(self.llm, "generateJson"):
            raw_result = self.llm.generateJson(
                systemPrompt=prompts["system_prompt"],
                userPrompt=prompts["user_prompt"],
                metadata={
                    "agent": "agent2",
                    "role": "coder",
                    "model": self.model_name or getattr(self.llm, "model_name", ""),
                },
                response_model=Agent2CodingResult,
            )
        else:
            raise AttributeError("llm must expose generate_json(...) or generateJson(...).")

        parsed = _try_parse_json(raw_result)
        return normalize_agent2_output(
            parsed,
            diagnosis_code_system=diagnosis_code_system,
            procedure_code_system=procedure_code_system,
        )
