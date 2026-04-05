from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string(value: Any, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _as_nullable_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _sanitize_evidence_ids(value: Any, valid_ids: set[str]) -> list[str]:
    return [item for item in _as_list(value) if item in valid_ids]


def _sanitize_flat_items(
    items: Any,
    valid_ids: set[str],
    extra_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    extra_fields = extra_fields or []
    output: list[dict[str, Any]] = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        label = _as_string(item.get("label")).strip()
        if not label:
            continue
        base: dict[str, Any] = {
            "label": label,
            "evidence_ids": _sanitize_evidence_ids(item.get("evidence_ids"), valid_ids),
        }
        for field in extra_fields:
            base[field] = _as_string(item.get(field)).strip()
        output.append(base)
    return output


def normalize_agent1_output(
    raw_output: dict[str, Any] | None,
    evidence_index: list[dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    raw = raw_output if isinstance(raw_output, dict) else {}
    valid_ids = {item["id"] for item in evidence_index}
    patient_override = overrides.get("patient_snapshot", {})
    summary_override = overrides.get("note_summary", {})

    return {
        "agent_id": "agent1",
        "role": "primary_analyzer",
        "version": "0.1.0",
        "generated_at": overrides.get("generated_at")
        or datetime.now(timezone.utc).isoformat(),
        "patient_snapshot": {
            "age": _as_nullable_string(raw.get("patient_snapshot", {}).get("age"))
            or patient_override.get("age"),
            "sex": _as_nullable_string(raw.get("patient_snapshot", {}).get("sex"))
            or patient_override.get("sex"),
            "encounter_type": _as_nullable_string(
                raw.get("patient_snapshot", {}).get("encounter_type")
            )
            or patient_override.get("encounter_type"),
            "care_setting": _as_nullable_string(
                raw.get("patient_snapshot", {}).get("care_setting")
            )
            or patient_override.get("care_setting"),
        },
        "note_summary": {
            "chief_complaint": _as_nullable_string(
                raw.get("note_summary", {}).get("chief_complaint")
            )
            or summary_override.get("chief_complaint"),
            "one_liner": _as_string(raw.get("note_summary", {}).get("one_liner")),
            "coding_focus": _as_string(raw.get("note_summary", {}).get("coding_focus")),
        },
        "primary_diagnosis_candidates": _sanitize_flat_items(
            raw.get("primary_diagnosis_candidates"),
            valid_ids,
            ["section", "confidence"],
        ),
        "active_conditions": _sanitize_flat_items(
            raw.get("active_conditions"),
            valid_ids,
            ["status", "section", "confidence"],
        ),
        "symptoms_and_signs": _sanitize_flat_items(
            raw.get("symptoms_and_signs"),
            valid_ids,
            ["section"],
        ),
        "procedures_and_treatments": _sanitize_flat_items(
            raw.get("procedures_and_treatments"),
            valid_ids,
            ["section"],
        ),
        "medications": _sanitize_flat_items(
            raw.get("medications"),
            valid_ids,
            ["action"],
        ),
        "tests_and_results": _sanitize_flat_items(
            raw.get("tests_and_results"),
            valid_ids,
            ["result"],
        ),
        "risk_factors_and_history": _sanitize_flat_items(
            raw.get("risk_factors_and_history"),
            valid_ids,
            ["section"],
        ),
        "uncertainty_or_exclusions": _sanitize_flat_items(
            raw.get("uncertainty_or_exclusions"),
            valid_ids,
            ["reason"],
        ),
        "coding_clues": [
            {
                "clue": _as_string(item.get("clue")).strip(),
                "rationale": _as_string(item.get("rationale")).strip(),
                "evidence_ids": _sanitize_evidence_ids(item.get("evidence_ids"), valid_ids),
            }
            for item in _as_list(raw.get("coding_clues"))
            if isinstance(item, dict) and _as_string(item.get("clue")).strip()
        ],
        "missing_information": [
            _as_string(value).strip()
            for value in _as_list(raw.get("missing_information"))
            if _as_string(value).strip()
        ],
        "evidence_index": evidence_index,
        "raw_note_statistics": {
            "character_count": overrides.get("note_text_length", 0),
            "evidence_count": len(evidence_index),
        },
    }
