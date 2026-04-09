from __future__ import annotations

import json


AGENT_1_JSON_TEMPLATE = {
    "gender": "male",
    "chief_complaint": "l2 fracture, back pain",
    "procedure": [
        "l2 corpectomy retroperitoneal approach",
        "revision of posterior l1-l3 fusion",
    ],
    "history_present_illness": "patient sustained an l2 fracture after jumping from a second-floor window and had persistent back pain despite conservative treatment.",
    "past_medical_history": ["mitral valve prolapse", "headaches", "gerd"],
    "physical_exam_discharge": [
        "afebrile",
        "vital signs stable",
        "no apparent distress",
        "back incision clean dry intact",
        "strength and sensation intact",
    ],
    "pertinent_results": [
        "abdominal x-ray large bowel dilation consistent with ileus",
        "ultrasound negative for dvt",
        "cta chest negative for pulmonary embolism",
        "small pleural effusions and atelectasis",
        "spine x-ray postsurgical changes with no acute fracture",
    ],
    "hospital_course": [
        "underwent l2 corpectomy and posterior fusion revision",
        "postoperative uncontrolled back pain requiring medication adjustment",
        "large bowel ileus improved with bowel regimen",
        "tachycardia workup negative for dvt and pe",
        "transient oxygen desaturation during sleep requiring supplemental oxygen",
        "new right-sided lumbar pain with stable repeat imaging",
    ],
    "discharge_diagnosis": ["l2 fracture", "back pain"],
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
            "Your job is to wash a raw discharge-style case into a compact structured JSON summary.",
            "Return JSON only.",
            "Do not explain your reasoning.",
            "Do not predict ICD codes.",
            "Keep the output clinically faithful but concise.",
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
            "- Do not invent diagnoses or procedures that are not supported by the note.",
        ]
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
