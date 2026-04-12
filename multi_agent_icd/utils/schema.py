from __future__ import annotations

from typing import Any


def _as_string(value: Any, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _as_nullable_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_phrase_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_as_string(item).strip() for item in value if _as_string(item).strip()]


def _normalize_code_candidate(
    value: Any,
    default_category: str,
    default_code_system: str,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    code = _as_string(value.get("code")).strip().upper()
    description = _as_string(value.get("description")).strip()
    if not code or not description:
        return None

    confidence = _as_string(value.get("confidence")).strip().lower() or "medium"
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    return {
        "code": code,
        "description": description,
        "code_system": _as_string(value.get("code_system")).strip() or default_code_system,
        "category": _as_string(value.get("category")).strip() or default_category,
        "confidence": confidence,
        "rationale": _as_string(value.get("rationale")).strip(),
        "evidence_ids": _normalize_phrase_list(value.get("evidence_ids")),
        "missing_details": _normalize_phrase_list(value.get("missing_details")),
    }


def _normalize_code_candidate_list(
    value: Any,
    default_category: str,
    default_code_system: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        candidate = _normalize_code_candidate(item, default_category, default_code_system)
        if candidate is None:
            continue
        dedupe_key = (candidate["category"], candidate["code"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(candidate)
    return normalized


def normalize_agent1_output(
    raw_output: dict[str, Any] | None,
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw_output if isinstance(raw_output, dict) else {}
    patient_context = patient_context or {}

    return {
        "gender": _as_nullable_string(raw.get("gender"))
        or patient_context.get("sex")
        or patient_context.get("gender")
        or "unknown",
        "chief_complaint": _as_string(raw.get("chief_complaint")).strip(),
        "procedure": _normalize_phrase_list(raw.get("procedure")),
        "history_present_illness": _as_string(raw.get("history_present_illness")).strip(),
        "past_medical_history": _normalize_phrase_list(raw.get("past_medical_history")),
        "physical_exam_discharge": _normalize_phrase_list(raw.get("physical_exam_discharge")),
        "pertinent_results": _normalize_phrase_list(raw.get("pertinent_results")),
        "hospital_course": _normalize_phrase_list(raw.get("hospital_course")),
        "discharge_diagnosis": _normalize_phrase_list(raw.get("discharge_diagnosis")),
    }


def normalize_agent2_output(
    raw_output: dict[str, Any] | None,
    diagnosis_code_system: str = "ICD-10-CM",
    procedure_code_system: str = "ICD-10-PCS",
) -> dict[str, Any]:
    raw = raw_output if isinstance(raw_output, dict) else {}

    principal = _normalize_code_candidate(
        raw.get("principal_diagnosis"),
        default_category="principal_diagnosis",
        default_code_system=diagnosis_code_system,
    )

    return {
        "principal_diagnosis": principal,
        "secondary_diagnoses": _normalize_code_candidate_list(
            raw.get("secondary_diagnoses"),
            default_category="secondary_diagnosis",
            default_code_system=diagnosis_code_system,
        ),
        "procedures": _normalize_code_candidate_list(
            raw.get("procedures"),
            default_category="procedure",
            default_code_system=procedure_code_system,
        ),
        "coding_queries": _normalize_phrase_list(raw.get("coding_queries")),
        "coding_summary": _as_string(raw.get("coding_summary")).strip(),
    }
