from __future__ import annotations

import json


AGENT_1_JSON_TEMPLATE = {
    "patient_snapshot": {
        "age": None,
        "sex": None,
        "encounter_type": None,
        "care_setting": None,
    },
    "note_summary": {
        "chief_complaint": None,
        "one_liner": "",
        "coding_focus": "",
    },
    "primary_diagnosis_candidates": [
        {
            "label": "",
            "section": "",
            "evidence_ids": ["E1"],
            "confidence": "high",
        }
    ],
    "active_conditions": [
        {
            "label": "",
            "status": "confirmed",
            "section": "",
            "evidence_ids": ["E1"],
            "confidence": "high",
        }
    ],
    "symptoms_and_signs": [{"label": "", "section": "", "evidence_ids": ["E2"]}],
    "procedures_and_treatments": [{"label": "", "section": "", "evidence_ids": ["E3"]}],
    "medications": [{"label": "", "action": "started", "evidence_ids": ["E4"]}],
    "tests_and_results": [{"label": "", "result": "", "evidence_ids": ["E5"]}],
    "risk_factors_and_history": [{"label": "", "section": "", "evidence_ids": ["E6"]}],
    "uncertainty_or_exclusions": [{"label": "", "reason": "", "evidence_ids": ["E7"]}],
    "coding_clues": [{"clue": "", "rationale": "", "evidence_ids": ["E8"]}],
    "missing_information": ["Need laterality for the final code."],
}


def build_agent1_prompts(
    note_text: str,
    patient_context: dict | None = None,
    evidence_index: list[dict] | None = None,
) -> dict[str, str]:
    patient_context = patient_context or {}
    evidence_index = evidence_index or []

    system_prompt = " ".join(
        [
            "You are Agent 1 in a multi-agent ICD coding system.",
            "Your role is to read raw clinical text and convert it into a coding-ready structured case summary.",
            "Do not predict ICD codes.",
            "Focus on coding-relevant facts: diagnoses, symptoms, procedures, medications, tests, timeline, chronicity, acuity, laterality, linkage language, and exclusions.",
            "Only cite evidence IDs that appear in the provided evidence index.",
            "If information is uncertain, contested, ruled out, or missing, say so explicitly.",
            "Return valid JSON only.",
        ]
    )

    user_prompt = "\n".join(
        [
            "Patient context:",
            json.dumps(patient_context, indent=2),
            "",
            "Evidence index:",
            json.dumps(evidence_index, indent=2),
            "",
            "Raw clinical note:",
            note_text,
            "",
            "Return JSON that matches this shape exactly:",
            json.dumps(AGENT_1_JSON_TEMPLATE, indent=2),
            "",
            "Rules:",
            "- Keep labels concise and clinically meaningful.",
            "- Use evidence_ids everywhere possible.",
            "- Prefer confirmed diagnoses over speculative ones in primary_diagnosis_candidates.",
            "- Put unresolved coding questions in missing_information.",
            "- If a field is unknown, use null, an empty string, or an empty array as appropriate.",
        ]
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
