from __future__ import annotations

import json


AGENT_1_JSON_TEMPLATE = {
    "gender": "<male|female|unknown>",
    "chief_complaint": "<chief complaint from the current note>",
    "procedure": [
        "<major procedure from the current note>",
    ],
    "history_present_illness": "<compact HPI summary supported only by the current note>",
    "past_medical_history": ["<past medical history item from the current note>"],
    "physical_exam_discharge": [
        "<discharge exam finding from the current note>",
    ],
    "pertinent_results": [
        "<key imaging, lab, or study result from the current note>",
    ],
    "hospital_course": [
        "<hospital course event from the current note>",
    ],
    "discharge_diagnosis": ["<discharge diagnosis from the current note>"],
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
            "You are Agent 1 in a multi-agent clinical coding system.",
            "Your job is to turn a raw discharge-style note into a compact structured JSON summary for downstream ICD coding.",
            "Return JSON only.",
            "Your first character must be { and your final character must be }.",
            "Do not explain your reasoning.",
            "Do not write a thinking process, analysis, markdown, or prose outside the JSON object.",
            "Do not predict ICD codes.",
            "Keep the output clinically faithful but concise.",
            "Preserve diagnosis, procedure, acuity, and complication cues that matter for coding.",
            "Preserve important abbreviations such as DVT, PE, CTA, EKG, GERD.",
            "Remove PHI and avoid narrative repetition.",
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
            "Return JSON that matches this structure closely:",
            json.dumps(AGENT_1_JSON_TEMPLATE, indent=2),
            "",
            "Formatting rules:",
            "- gender should be a short value like male, female, or unknown.",
            "- chief_complaint should be a short string.",
            "- procedure, past_medical_history, physical_exam_discharge, pertinent_results, hospital_course, and discharge_diagnosis should be arrays of short cleaned phrases.",
            "- history_present_illness should be a compact prose sentence or two.",
            "- Prefer normalized clinical phrases over full copied sentences.",
            "- Preserve laterality, acuity, complication, postoperative, and causal wording when supported.",
            "- The JSON structure above contains placeholders only; do not copy placeholder text as final output.",
            "- Do not invent diagnoses or procedures that are not supported by the note.",
            "- If a field is not documented in the current note, use an empty string for scalar fields or an empty array for list fields.",
        ]
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
