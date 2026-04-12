from __future__ import annotations

import json


def resolve_agent2_code_systems(patient_context: dict | None = None) -> tuple[str, str, str]:
    context = patient_context or {}
    coding_version = str(context.get("coding_version", "ICD-10")).upper()
    if "ICD-9" in coding_version or coding_version.endswith("9"):
        return ("ICD-9", "ICD-9-CM", "ICD-9-CM")
    return ("ICD-10", "ICD-10-CM", "ICD-10-PCS")


def _build_agent2_json_template(coding_version: str) -> dict:
    if coding_version == "ICD-9":
        return {
            "principal_diagnosis": {
                "code": "486",
                "description": "pneumonia, organism unspecified",
                "code_system": "ICD-9-CM",
                "category": "principal_diagnosis",
                "confidence": "medium",
                "rationale": "Discharge diagnosis supports pneumonia as the main treated condition.",
                "evidence_ids": ["E3", "E18"],
                "missing_details": [],
            },
            "secondary_diagnoses": [
                {
                    "code": "401.9",
                    "description": "unspecified essential hypertension",
                    "code_system": "ICD-9-CM",
                    "category": "secondary_diagnosis",
                    "confidence": "high",
                    "rationale": "Hypertension is documented in the note history and medication list.",
                    "evidence_ids": ["E8"],
                    "missing_details": [],
                }
            ],
            "procedures": [
                {
                    "code": "96.04",
                    "description": "insertion of endotracheal tube",
                    "code_system": "ICD-9-CM",
                    "category": "procedure",
                    "confidence": "low",
                    "rationale": "Use only when the note clearly documents the procedure during the encounter.",
                    "evidence_ids": ["E10"],
                    "missing_details": ["confirm the exact inpatient procedure documented in the chart"],
                }
            ],
            "coding_queries": [
                "Clarify any missing diagnosis specificity before final code assignment.",
                "Confirm whether any inpatient procedures should be coded from this note.",
            ],
            "coding_summary": "Return only the best-supported ICD-9-CM diagnosis and procedure candidates."
        }

    return {
        "principal_diagnosis": {
            "code": "S32.029A",
            "description": "fracture of second lumbar vertebra, initial encounter for closed fracture",
            "code_system": "ICD-10-CM",
            "category": "principal_diagnosis",
            "confidence": "medium",
            "rationale": "Discharge diagnosis and hospital course support an acute L2 fracture treated during this encounter.",
            "evidence_ids": ["E3", "E18"],
            "missing_details": ["confirm whether the fracture was documented as closed or open if not explicit"],
        },
        "secondary_diagnoses": [
            {
                "code": "K56.7",
                "description": "ileus, unspecified",
                "code_system": "ICD-10-CM",
                "category": "secondary_diagnosis",
                "confidence": "high",
                "rationale": "The hospital course documents postoperative ileus that required treatment.",
                "evidence_ids": ["E22", "E24"],
                "missing_details": [],
            }
        ],
        "procedures": [
            {
                "code": "0RGA0K1",
                "description": "fusion of lumbar vertebral joint with internal fixation device, open approach",
                "code_system": "ICD-10-PCS",
                "category": "procedure",
                "confidence": "low",
                "rationale": "The note documents lumbar fusion revision but may not contain enough PCS detail for a final code.",
                "evidence_ids": ["E10"],
                "missing_details": ["confirm device, exact body part, and whether additional corpectomy PCS codes are required"],
            }
        ],
        "coding_queries": [
            "Clarify encounter specificity and fracture details if a more specific 7th character is available.",
            "Confirm the exact ICD-10-PCS procedure components for the corpectomy and fusion revision.",
        ],
        "coding_summary": "Primary focus is acute L2 fracture with treated postoperative complications and major lumbar surgery."
    }


def build_agent2_prompts(
    structured_case_summary: dict,
    patient_context: dict | None = None,
    evidence_index: list[dict] | None = None,
) -> dict[str, str]:
    patient_context = patient_context or {}
    evidence_index = evidence_index or []
    coding_version, diagnosis_code_system, procedure_code_system = resolve_agent2_code_systems(patient_context)
    json_template = _build_agent2_json_template(coding_version)

    system_prompt = " ".join(
        [
            "You are Agent 2 in a multi-agent ICD coding system.",
            "Your job is to convert a structured clinical case summary into grounded ICD code candidates.",
            "Return JSON only.",
            "Do not explain your chain-of-thought.",
            "Be conservative and evidence-based.",
            "If documentation is incomplete, choose the best supported code candidate and record the missing details explicitly.",
            "Use evidence_ids from the provided evidence index whenever possible.",
        ]
    )

    user_prompt = "\n".join(
        [
            "Patient context:",
            json.dumps(patient_context, indent=2, ensure_ascii=False),
            "",
            f"Target coding version: {coding_version}",
            f"Use {diagnosis_code_system} for diagnoses and {procedure_code_system} for procedures.",
            "",
            "Agent 1 structured case summary:",
            json.dumps(structured_case_summary, indent=2, ensure_ascii=False),
            "",
            "Evidence index from the raw note:",
            json.dumps(evidence_index, indent=2, ensure_ascii=False),
            "",
            "Return JSON that matches this structure closely:",
            json.dumps(json_template, indent=2, ensure_ascii=False),
            "",
            "Formatting rules:",
            "- principal_diagnosis should be either one code candidate object or null when unsupported.",
            "- secondary_diagnoses and procedures should contain only materially supported items.",
            f"- code_system should be {diagnosis_code_system} for diagnoses and {procedure_code_system} for procedures.",
            "- confidence should be high, medium, or low.",
            "- rationale should be short and evidence-grounded.",
            "- missing_details should list the exact documentation gaps that limit specificity.",
            "- Do not invent unsupported diagnoses, complications, or procedures.",
        ]
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
