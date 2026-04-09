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
