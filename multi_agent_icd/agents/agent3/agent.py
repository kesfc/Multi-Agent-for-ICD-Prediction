from __future__ import annotations

import json
from typing import Any

from multi_agent_icd.agents.agent3.prompt import build_agent3_prompts
from multi_agent_icd.agents.agent3.schema import Agent3KnowledgeResult
from multi_agent_icd.knowledge_base import KnowledgeBase
from multi_agent_icd.providers import LocalQwenLLM
from multi_agent_icd.utils.clinical_text import build_evidence_index, normalize_clinical_text
from multi_agent_icd.utils.schema import normalize_agent3_output


def _try_parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError("LLM output was neither a dict nor a JSON string.")
    return json.loads(value)


def _normalize_code_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    codes: list[str] = []
    for item in value:
        code = str(item or "").strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _normalize_candidate_output_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("candidate_output_limit must be a positive integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_output_limit must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError("candidate_output_limit must be a positive integer.")
    return normalized


def _extract_predicted_codes(
    agent2_output: dict[str, Any] | None,
    candidate_output_limit: int | None = None,
) -> list[str]:
    if not isinstance(agent2_output, dict):
        return []

    normalized_limit = _normalize_candidate_output_limit(candidate_output_limit)
    ordered_codes: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        code = str(candidate.get("code", "")).strip().upper()
        if not code or code in seen:
            return
        if normalized_limit is not None and len(ordered_codes) >= normalized_limit:
            return
        seen.add(code)
        ordered_codes.append(code)

    add_candidate(agent2_output.get("principal_diagnosis"))
    for item in agent2_output.get("secondary_diagnoses", []):
        add_candidate(item)
    for item in agent2_output.get("procedures", []):
        add_candidate(item)

    return ordered_codes


def _resolve_source_case_id(training_context: dict[str, Any], patient_context: dict[str, Any]) -> str | None:
    for key in ("source_case_id", "case_id"):
        value = str(training_context.get(key) or patient_context.get(key) or "").strip()
        if value:
            return value

    subject_id = str(training_context.get("subject_id") or patient_context.get("subject_id") or "").strip()
    hadm_id = str(training_context.get("hadm_id") or patient_context.get("hadm_id") or "").strip()
    if subject_id and hadm_id:
        return f"{subject_id}:{hadm_id}"
    if hadm_id:
        return hadm_id
    if subject_id:
        return subject_id
    return None


class Agent3KnowledgeSynthesizer:
    def __init__(
        self,
        model_name: str | None = None,
        llm: Any | None = None,
        knowledge_base_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.llm = llm or LocalQwenLLM(model_name=model_name)
        self.knowledge_base_path = knowledge_base_path

    def run(self, state: Any) -> dict[str, Any]:
        note_text = normalize_clinical_text(getattr(state, "note_text", ""))
        if not note_text:
            raise ValueError("note_text is required for Agent 3.")

        patient_context = dict(getattr(state, "patient_context", {}) or {})
        training_context = dict(getattr(state, "training_context", {}) or {})
        shared_memory = getattr(state, "shared_memory", {}) or {}
        agent_outputs = getattr(state, "agent_outputs", {}) or {}

        gold_codes = _normalize_code_list(
            training_context.get("gold_labels") or training_context.get("gold_codes")
        )
        if not gold_codes:
            raise ValueError("Agent 3 requires gold_labels or gold_codes in training_context.")

        structured_case_summary = shared_memory.get("structured_case_summary") or agent_outputs.get("agent1")
        if not isinstance(structured_case_summary, dict):
            raise ValueError("Agent 3 requires a structured Agent 1 summary in the pipeline state.")

        agent2_output = shared_memory.get("agent2_output") or agent_outputs.get("agent2")
        if not isinstance(agent2_output, dict):
            raise ValueError("Agent 3 requires an Agent 2 coding result in the pipeline state.")

        evidence_index = shared_memory.get("note_evidence_index")
        if not isinstance(evidence_index, list):
            evidence_index = build_evidence_index(note_text)

        predicted_codes = _extract_predicted_codes(
            agent2_output,
            candidate_output_limit=patient_context.get("candidate_output_limit"),
        )
        gold_set = set(gold_codes)
        predicted_set = set(predicted_codes)
        correct_codes = [code for code in predicted_codes if code in gold_set]
        missed_codes = [code for code in gold_codes if code not in predicted_set]
        extra_codes = [code for code in predicted_codes if code not in gold_set]

        prompts = build_agent3_prompts(
            structured_case_summary=structured_case_summary,
            patient_context=patient_context,
            evidence_index=evidence_index,
            gold_codes=gold_codes,
            predicted_codes=predicted_codes,
            correct_codes=correct_codes,
            missed_codes=missed_codes,
            extra_codes=extra_codes,
        )

        if hasattr(self.llm, "generate_json"):
            raw_result = self.llm.generate_json(
                system_prompt=prompts["system_prompt"],
                user_prompt=prompts["user_prompt"],
                metadata={
                    "agent": "agent3",
                    "role": "knowledge_synthesizer",
                    "model": self.model_name or getattr(self.llm, "model_name", ""),
                },
                response_model=Agent3KnowledgeResult,
            )
        elif hasattr(self.llm, "generateJson"):
            raw_result = self.llm.generateJson(
                systemPrompt=prompts["system_prompt"],
                userPrompt=prompts["user_prompt"],
                metadata={
                    "agent": "agent3",
                    "role": "knowledge_synthesizer",
                    "model": self.model_name or getattr(self.llm, "model_name", ""),
                },
                response_model=Agent3KnowledgeResult,
            )
        else:
            raise AttributeError("llm must expose generate_json(...) or generateJson(...).")

        parsed = normalize_agent3_output(_try_parse_json(raw_result))
        knowledge_base_path = (
            training_context.get("knowledge_base_path")
            or patient_context.get("knowledge_base_path")
            or shared_memory.get("knowledge_base_path")
            or self.knowledge_base_path
        )
        if not knowledge_base_path:
            raise ValueError("Agent 3 requires knowledge_base_path so it can store synthesized memory.")

        knowledge_base = KnowledgeBase(knowledge_base_path)
        source_case_id = _resolve_source_case_id(training_context, patient_context)
        entry = knowledge_base.insert_memory(
            note_text=note_text,
            structured_case_summary=structured_case_summary,
            agent3_output=parsed,
            gold_codes=gold_codes,
            predicted_codes=predicted_codes,
            missed_codes=missed_codes,
            extra_codes=extra_codes,
            source_case_id=source_case_id,
            coding_version=str(patient_context.get("coding_version", "")).strip() or None,
            metadata={
                "subject_id": training_context.get("subject_id") or patient_context.get("subject_id"),
                "hadm_id": training_context.get("hadm_id") or patient_context.get("hadm_id"),
            },
        )

        return {
            **parsed,
            "gold_codes": gold_codes,
            "predicted_codes": predicted_codes,
            "correct_codes": correct_codes,
            "missed_codes": missed_codes,
            "extra_codes": extra_codes,
            "source_case_id": source_case_id,
            "knowledge_base_path": str(knowledge_base.db_path),
            "knowledge_entry_id": entry["id"],
            "stored": True,
        }
