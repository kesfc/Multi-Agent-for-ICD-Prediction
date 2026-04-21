from __future__ import annotations

import json

from multi_agent_icd.utils.clinical_text import compact_evidence_index_for_prompt


AGENT_3_JSON_TEMPLATE = {
    "case_summary": "<short reusable summary of the coding-relevant clinical pattern>",
    "salient_clinical_patterns": [
        "<important diagnosis, procedure, acuity, complication, or documentation clue>",
    ],
    "correct_prediction_reasons": [
        "<why a correctly predicted code was supported by the case>",
    ],
    "missed_code_lessons": [
        "<what the model missed and which chart clues should trigger that code next time>",
    ],
    "unsupported_prediction_lessons": [
        "<why a predicted-only code was weak or unsupported in this case>",
    ],
    "coding_lessons": [
        "<generalizable coding heuristic that can help on future similar cases>",
    ],
    "retrieval_queries": [
        "<short future search phrase for similar cases>",
    ],
    "knowledge_summary": "<one concise memory entry for later retrieval>"
}


def build_agent3_prompts(
    *,
    structured_case_summary: dict,
    patient_context: dict | None = None,
    evidence_index: list[dict] | None = None,
    gold_codes: list[str] | None = None,
    predicted_codes: list[str] | None = None,
    correct_codes: list[str] | None = None,
    missed_codes: list[str] | None = None,
    extra_codes: list[str] | None = None,
) -> dict[str, str]:
    patient_context = patient_context or {}
    prompt_evidence_index = compact_evidence_index_for_prompt(evidence_index)
    gold_codes = gold_codes or []
    predicted_codes = predicted_codes or []
    correct_codes = correct_codes or []
    missed_codes = missed_codes or []
    extra_codes = extra_codes or []

    system_prompt = " ".join(
        [
            "You are Agent 3 in a multi-agent ICD coding system.",
            "Your job is to review the case summary, gold codes, and predicted codes, then distill reusable coding knowledge for future cases.",
            "Write generalizable lessons, not patient-specific narrative.",
            "Return JSON only.",
            "Your first character must be { and your final character must be }.",
            "Do not explain your chain-of-thought.",
            "Do not write markdown or prose outside the JSON object.",
            "Ground every lesson in the provided case evidence.",
            "Avoid PHI and avoid quoting long note passages.",
            "When predictions were wrong, explain the coding clue or documentation gap that mattered.",
            "Make the output useful for later retrieval by a coding agent.",
        ]
    )

    user_prompt = "\n".join(
        [
            "Patient context:",
            json.dumps(patient_context, indent=2, ensure_ascii=False),
            "",
            "Agent 1 structured case summary:",
            json.dumps(structured_case_summary, indent=2, ensure_ascii=False),
            "",
            "Evidence index from the raw note:",
            json.dumps(prompt_evidence_index, indent=2, ensure_ascii=False),
            "",
            "Gold ICD codes:",
            json.dumps(gold_codes, indent=2, ensure_ascii=False),
            "",
            "Predicted ICD codes:",
            json.dumps(predicted_codes, indent=2, ensure_ascii=False),
            "",
            "Correctly predicted codes:",
            json.dumps(correct_codes, indent=2, ensure_ascii=False),
            "",
            "Missed gold codes:",
            json.dumps(missed_codes, indent=2, ensure_ascii=False),
            "",
            "Predicted-only codes that were not in gold:",
            json.dumps(extra_codes, indent=2, ensure_ascii=False),
            "",
            "Return JSON that matches this structure closely:",
            json.dumps(AGENT_3_JSON_TEMPLATE, indent=2, ensure_ascii=False),
            "",
            "Formatting rules:",
            "- The JSON structure above contains placeholders only; do not copy placeholder text as final output.",
            "- case_summary should be short and reusable.",
            "- salient_clinical_patterns should capture coding-relevant clues such as acuity, complications, laterality, procedure details, and discharge diagnoses.",
            "- coding_lessons should generalize beyond this single case while remaining evidence-grounded.",
            "- retrieval_queries should be short phrases that would help retrieve this memory for a similar future case.",
            "- If there were no missed or unsupported predictions, keep the corresponding arrays empty.",
            "- Do not invent facts, diagnoses, or procedures that are not supported by the case summary and evidence index.",
        ]
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
