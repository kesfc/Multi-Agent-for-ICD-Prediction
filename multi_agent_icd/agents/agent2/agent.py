from __future__ import annotations

import json
from typing import Any

from multi_agent_icd.agents.agent2.prompt import build_agent2_prompts, resolve_agent2_code_systems
from multi_agent_icd.agents.agent2.schema import Agent2CodingResult
from multi_agent_icd.knowledge_base import KnowledgeBase
from multi_agent_icd.providers import LocalQwenLLM
from multi_agent_icd.utils.clinical_text import build_evidence_index, normalize_clinical_text
from multi_agent_icd.utils.schema import normalize_agent2_output


def _try_parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError("LLM output was neither a dict nor a JSON string.")
    return json.loads(value)


def _coerce_agent2_result(value: dict[str, Any]) -> dict[str, Any]:
    if any(
        key in value
        for key in ("principal_diagnosis", "secondary_diagnoses", "procedures", "coding_summary")
    ):
        return value
    if "code" not in value:
        return value
    category = str(value.get("category", "")).strip().lower()
    result = {
        "principal_diagnosis": None,
        "secondary_diagnoses": [],
        "procedures": [],
        "coding_queries": [
            "Model returned a single code candidate instead of the full coding result schema.",
        ],
        "coding_summary": "Single code candidate was coerced into the expected Agent 2 result schema.",
    }
    if category == "procedure":
        result["procedures"] = [value]
    elif category == "secondary_diagnosis":
        result["secondary_diagnoses"] = [value]
    else:
        result["principal_diagnosis"] = value
    return result


def _normalize_knowledge_top_k(value: Any, default: int = 3) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("knowledge_base_top_k must be a positive integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("knowledge_base_top_k must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError("knowledge_base_top_k must be a positive integer.")
    return normalized


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

        retrieved_knowledge: list[dict[str, Any]] = []
        knowledge_base_path = (
            shared_memory.get("knowledge_base_path") or patient_context.get("knowledge_base_path")
        )
        if knowledge_base_path:
            knowledge_base = KnowledgeBase(knowledge_base_path)
            retrieved_knowledge = knowledge_base.search(
                note_text=note_text,
                structured_case_summary=structured_case_summary,
                coding_version=patient_context.get("coding_version"),
                top_k=_normalize_knowledge_top_k(
                    shared_memory.get("knowledge_base_top_k")
                    or patient_context.get("knowledge_base_top_k")
                    or 3
                ),
            )
            if hasattr(state, "shared_memory"):
                state.shared_memory["agent2_retrieved_knowledge"] = retrieved_knowledge

        prompts = build_agent2_prompts(
            structured_case_summary=structured_case_summary,
            patient_context=patient_context,
            evidence_index=evidence_index,
            retrieved_knowledge=retrieved_knowledge,
            candidate_code_set=patient_context.get("candidate_code_set"),
            candidate_code_records=patient_context.get("candidate_code_records"),
            candidate_output_limit=patient_context.get("candidate_output_limit"),
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

        parsed = _coerce_agent2_result(_try_parse_json(raw_result))
        return normalize_agent2_output(
            parsed,
            diagnosis_code_system=diagnosis_code_system,
            procedure_code_system=procedure_code_system,
            allowed_codes=patient_context.get("candidate_code_set"),
            candidate_limit=patient_context.get("candidate_output_limit"),
        )
