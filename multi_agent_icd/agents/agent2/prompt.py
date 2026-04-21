from __future__ import annotations

import json

from multi_agent_icd.utils.clinical_text import compact_evidence_index_for_prompt


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
                "code": "<supported ICD-9-CM code from allowed candidates>",
                "description": "<short description of the supported diagnosis>",
                "code_system": "ICD-9-CM",
                "category": "principal_diagnosis",
                "confidence": "high|medium|low",
                "rationale": "<brief evidence-grounded reason this diagnosis is supported>",
                "evidence_ids": ["E1"],
                "missing_details": [],
            },
            "secondary_diagnoses": [
                {
                    "code": "<supported ICD-9-CM code from allowed candidates>",
                    "description": "<short description of the supported secondary diagnosis>",
                    "code_system": "ICD-9-CM",
                    "category": "secondary_diagnosis",
                    "confidence": "high|medium|low",
                    "rationale": "<brief evidence-grounded reason this diagnosis is supported>",
                    "evidence_ids": ["E2"],
                    "missing_details": [],
                }
            ],
            "procedures": [
                {
                    "code": "<supported ICD-9-CM procedure code from allowed candidates>",
                    "description": "<short description of the supported procedure>",
                    "code_system": "ICD-9-CM",
                    "category": "procedure",
                    "confidence": "high|medium|low",
                    "rationale": "<brief evidence-grounded reason this procedure is supported>",
                    "evidence_ids": ["E3"],
                    "missing_details": ["<specific missing detail if needed>"],
                }
            ],
            "coding_queries": [
                "<specific missing documentation question if needed>",
            ],
            "coding_summary": "<short summary of the selected ICD-9-CM candidates>"
        }

    return {
        "principal_diagnosis": {
            "code": "<supported ICD-10-CM code from allowed candidates>",
            "description": "<short description of the supported diagnosis>",
            "code_system": "ICD-10-CM",
            "category": "principal_diagnosis",
            "confidence": "high|medium|low",
            "rationale": "<brief evidence-grounded reason this diagnosis is supported>",
            "evidence_ids": ["E1"],
            "missing_details": ["<specific missing detail if needed>"],
        },
        "secondary_diagnoses": [
            {
                "code": "<supported ICD-10-CM code from allowed candidates>",
                "description": "<short description of the supported secondary diagnosis>",
                "code_system": "ICD-10-CM",
                "category": "secondary_diagnosis",
                "confidence": "high|medium|low",
                "rationale": "<brief evidence-grounded reason this diagnosis is supported>",
                "evidence_ids": ["E2"],
                "missing_details": [],
            }
        ],
        "procedures": [
            {
                "code": "<supported ICD-10-PCS code from allowed candidates>",
                "description": "<short description of the supported procedure>",
                "code_system": "ICD-10-PCS",
                "category": "procedure",
                "confidence": "high|medium|low",
                "rationale": "<brief evidence-grounded reason this procedure is supported>",
                "evidence_ids": ["E3"],
                "missing_details": ["<specific missing detail if needed>"],
            }
        ],
        "coding_queries": [
            "<specific missing documentation question if needed>",
        ],
        "coding_summary": "<short summary of the selected ICD-10 candidates>"
    }


def build_agent2_prompts(
    structured_case_summary: dict,
    patient_context: dict | None = None,
    evidence_index: list[dict] | None = None,
    retrieved_knowledge: list[dict] | None = None,
    candidate_code_set: list[str] | None = None,
    candidate_code_records: list[dict] | None = None,
    candidate_output_limit: int | None = None,
) -> dict[str, str]:
    patient_context = patient_context or {}
    prompt_evidence_index = compact_evidence_index_for_prompt(evidence_index)
    retrieved_knowledge = retrieved_knowledge or []
    candidate_code_set = candidate_code_set or []
    candidate_code_records = candidate_code_records or []
    coding_version, diagnosis_code_system, procedure_code_system = resolve_agent2_code_systems(patient_context)
    json_template = _build_agent2_json_template(coding_version)

    system_prompt = " ".join(
        [
            "You are Agent 2 in a multi-agent ICD coding system.",
            "Your job is to convert a structured clinical case summary into grounded ICD code candidates.",
            "Return JSON only.",
            "Your first character must be { and your final character must be }.",
            "Do not explain your chain-of-thought.",
            "Do not write a thinking process, analysis, markdown, or prose outside the JSON object.",
            "Be conservative and evidence-based.",
            "If documentation is incomplete, choose the best supported code candidate and record the missing details explicitly.",
            "When a candidate code set is provided, only choose codes from that set.",
            "Use evidence_ids from the provided evidence index whenever possible.",
            "Retrieved knowledge base memories are advisory only and must never override the current note evidence or candidate constraints.",
        ]
    )

    candidate_rules = []
    if candidate_code_records:
        candidate_rules.extend(
            [
                "",
                "Allowed ICD code candidates with descriptions:",
                json.dumps(candidate_code_records, indent=2, ensure_ascii=False),
                "",
                "Candidate-set rules:",
                "- Choose code values only from the allowed ICD code candidates above.",
                "- Use the provided description as the source of truth for each code's clinical meaning.",
                "- Return a ranked candidate list, ordered from most likely to least likely.",
                "- Do not change the clinical meaning of an allowed code to fit the note.",
                "- The output description should match or faithfully summarize the provided candidate description.",
                "- If a clinically supported condition is not represented in the allowed candidates, mention the gap in coding_queries.",
                "- Still fill the requested ranked list from the allowed candidates; mark weak lower-ranked candidates as low confidence.",
            ]
        )
    elif candidate_code_set:
        candidate_rules.extend(
            [
                "",
                "Allowed ICD code candidates:",
                json.dumps(candidate_code_set, indent=2, ensure_ascii=False),
                "",
                "Candidate-set rules:",
                "- Choose code values only from the allowed ICD code candidates above.",
                "- Return a ranked candidate list, ordered from most likely to least likely.",
                "- Do not change the clinical meaning of an allowed code to fit the note.",
                "- The description must describe the ICD code itself, not just restate the patient's symptom or diagnosis.",
                "- If a clinically supported condition is not represented in the allowed candidates, mention the gap in coding_queries.",
                "- Still fill the requested ranked list from the allowed candidates; mark weak lower-ranked candidates as low confidence.",
            ]
        )
    if candidate_output_limit is not None:
        candidate_rules.append(
            f"- Return at most {candidate_output_limit} total code candidates across principal_diagnosis, secondary_diagnoses, and procedures."
        )

    knowledge_section: list[str] = []
    if retrieved_knowledge:
        knowledge_section.extend(
            [
                "",
                "Retrieved knowledge base memories from prior reviewed cases:",
                json.dumps(retrieved_knowledge, indent=2, ensure_ascii=False),
                "",
                "Knowledge-memory rules:",
                "- Use these memories only as supportive heuristics.",
                "- The current note evidence and allowed candidate list remain the source of truth.",
                "- Do not copy a prior code just because it appears in memory.",
                "- When a memory is relevant, use it to notice patterns or documentation cues that should be checked in the current note.",
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
            json.dumps(prompt_evidence_index, indent=2, ensure_ascii=False),
            *knowledge_section,
            *candidate_rules,
            "",
            "Return JSON that matches this structure closely:",
            json.dumps(json_template, indent=2, ensure_ascii=False),
            "",
            "Formatting rules:",
            "- The JSON structure above contains placeholders only; do not copy placeholder text as final output.",
            "- principal_diagnosis should be either one code candidate object or null when unsupported.",
            "- Put the highest-ranked diagnosis candidate in principal_diagnosis when one is supportable.",
            "- Put the remaining ranked diagnosis candidates in secondary_diagnoses; use procedures only for true procedure codes documented in the note.",
            f"- code_system should be {diagnosis_code_system} for diagnoses and {procedure_code_system} for procedures.",
            "- confidence should be high, medium, or low.",
            "- rationale should be short and evidence-grounded.",
            "- missing_details should list the exact documentation gaps that limit specificity.",
            "- Do not invent unsupported diagnoses, complications, or procedures.",
            "- Do not assign an allowed code unless both the code meaning and note evidence support it.",
        ]
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
