from __future__ import annotations

import re
from typing import Any


SECTION_ALIASES = {
    "cc": "chief_complaint",
    "chief complaint": "chief_complaint",
    "chief concern": "chief_complaint",
    "reason for visit": "chief_complaint",
    "hpi": "history_present_illness",
    "history of present illness": "history_present_illness",
    "subjective": "history_present_illness",
    "ros": "review_of_systems",
    "review of systems": "review_of_systems",
    "pmh": "past_medical_history",
    "past medical history": "past_medical_history",
    "psh": "past_surgical_history",
    "past surgical history": "past_surgical_history",
    "medications": "medications",
    "home medications": "medications",
    "current medications": "medications",
    "allergies": "allergies",
    "physical exam": "physical_exam",
    "exam": "physical_exam",
    "labs": "labs",
    "laboratory": "labs",
    "laboratory data": "labs",
    "results": "results",
    "imaging": "imaging",
    "radiology": "imaging",
    "assessment": "assessment",
    "impression": "impression",
    "diagnosis": "diagnosis",
    "diagnoses": "diagnosis",
    "discharge diagnosis": "discharge_diagnosis",
    "discharge diagnoses": "discharge_diagnosis",
    "problem list": "problem_list",
    "hospital course": "hospital_course",
    "plan": "plan",
    "procedures": "procedures",
    "procedure": "procedures",
    "operations": "procedures",
    "social history": "social_history",
    "family history": "family_history",
}

SECTION_PRIORITY = {
    "discharge_diagnosis": 100,
    "assessment": 95,
    "diagnosis": 90,
    "impression": 88,
    "problem_list": 85,
    "hospital_course": 70,
    "history_present_illness": 60,
    "chief_complaint": 55,
    "plan": 50,
    "labs": 40,
    "imaging": 40,
    "results": 40,
    "past_medical_history": 35,
    "social_history": 30,
    "family_history": 25,
    "general": 10,
}


def normalize_clinical_text(note_text: str | None) -> str:
    text = str(note_text or "")
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_heading(raw_heading: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[:\-]+$", "", raw_heading.strip())).lower()


def _title_from_section(section_key: str) -> str:
    return " ".join(part.capitalize() for part in section_key.split("_"))


def detect_section(line: str) -> dict[str, str] | None:
    trimmed = line.strip()
    prefix_match = re.match(r"^([A-Za-z][A-Za-z /()\-]{1,40}):\s*(.*)$", trimmed)
    if prefix_match:
        heading = _normalize_heading(prefix_match.group(1))
        if heading in SECTION_ALIASES:
            return {
                "section": SECTION_ALIASES[heading],
                "inline_content": prefix_match.group(2).strip(),
            }

    heading_only = _normalize_heading(trimmed)
    if heading_only in SECTION_ALIASES:
        return {"section": SECTION_ALIASES[heading_only], "inline_content": ""}

    if re.match(r"^[A-Z][A-Z /()\-]{2,40}$", trimmed):
        canonical = _normalize_heading(trimmed)
        if canonical in SECTION_ALIASES:
            return {"section": SECTION_ALIASES[canonical], "inline_content": ""}

    return None


def _split_long_line(text: str) -> list[str]:
    chunks = [
        part.strip()
        for part in re.split(r"(?<=[.;])\s+|\s+\u2022\s+|\s+-\s+(?=[A-Z])", text)
        if part.strip()
    ]
    return chunks if len(chunks) > 1 else [text.strip()]


def build_evidence_index(note_text: str) -> list[dict[str, Any]]:
    normalized = normalize_clinical_text(note_text)
    lines = normalized.split("\n")
    evidence_index: list[dict[str, Any]] = []
    current_section = "general"
    running_offset = 0

    for raw_line in lines:
        line = raw_line.strip()
        start = normalized.find(raw_line, running_offset)
        if start < 0:
            start = running_offset
        running_offset = start + len(raw_line) + 1

        if not line:
            continue

        detected = detect_section(line)
        if detected:
            current_section = detected["section"]
            inline_content = detected["inline_content"]
            if not inline_content:
                continue
            for chunk in _split_long_line(inline_content):
                chunk_start = normalized.find(chunk, start)
                if chunk_start < 0:
                    chunk_start = start
                evidence_index.append(
                    {
                        "id": f"E{len(evidence_index) + 1}",
                        "section": current_section,
                        "section_label": _title_from_section(current_section),
                        "text": chunk,
                        "start_char": chunk_start,
                        "end_char": chunk_start + len(chunk),
                    }
                )
            continue

        for chunk in _split_long_line(line):
            chunk_start = normalized.find(chunk, start)
            if chunk_start < 0:
                chunk_start = start
            evidence_index.append(
                {
                    "id": f"E{len(evidence_index) + 1}",
                    "section": current_section,
                    "section_label": _title_from_section(current_section),
                    "text": chunk,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(chunk),
                }
            )

    return evidence_index


def compact_evidence_index_for_prompt(evidence_index: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    if not isinstance(evidence_index, list):
        return compacted

    min_chunk_chars = 90
    max_chunk_chars = 240
    current_id = ""
    current_section = ""
    current_texts: list[str] = []

    def normalize_prompt_text(value: Any) -> str:
        text = normalize_clinical_text(str(value or ""))
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"^[\-\u2022*]+\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_low_signal_prompt_text(text: str) -> bool:
        if not text:
            return True
        if re.fullmatch(r"[\W\d_]+", text):
            return True
        words = re.findall(r"[A-Za-z]+", text)
        if len(words) < 2 and len(text) < 8:
            return True
        return False

    def flush() -> None:
        nonlocal current_id, current_section, current_texts
        if current_id and current_texts:
            compacted.append(
                {
                    "id": current_id,
                    "text": " ".join(current_texts).strip(),
                }
            )
        current_id = ""
        current_section = ""
        current_texts = []

    for item in evidence_index:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id", "")).strip()
        section = str(item.get("section", "")).strip().lower()
        text = normalize_prompt_text(item.get("text", ""))
        if not evidence_id or is_low_signal_prompt_text(text):
            continue

        if not current_texts:
            current_id = evidence_id
            current_section = section
            current_texts = [text]
            continue

        merged_text = " ".join(current_texts + [text]).strip()
        should_merge = (
            section == current_section
            and len(merged_text) <= max_chunk_chars
            and len(" ".join(current_texts).strip()) < min_chunk_chars
        )
        if should_merge:
            current_texts.append(text)
            continue

        flush()
        current_id = evidence_id
        current_section = section
        current_texts = [text]

    flush()
    return compacted


def extract_patient_snapshot(note_text: str, patient_context: dict | None = None) -> dict[str, str | None]:
    normalized = normalize_clinical_text(note_text)
    context = patient_context or {}

    age_match = re.search(r"\b(\d{1,3})-year-old\b", normalized, re.IGNORECASE) or re.search(
        r"\bAge[: ]+(\d{1,3})\b", normalized, re.IGNORECASE
    )
    sex_match = re.search(r"\b(male|female|man|woman)\b", normalized, re.IGNORECASE) or re.search(
        r"\bSex[: ]+(male|female)\b", normalized, re.IGNORECASE
    )
    encounter_match = re.search(
        r"\b(admission|discharge|follow-up|consult|consultation|ed visit|emergency visit|inpatient stay|outpatient visit)\b",
        normalized,
        re.IGNORECASE,
    )
    setting_match = re.search(
        r"\b(ICU|ED|emergency department|inpatient|outpatient|clinic|telehealth|observation)\b",
        normalized,
        re.IGNORECASE,
    )

    return {
        "age": context.get("age") or (age_match.group(1) if age_match else None),
        "sex": context.get("sex") or (sex_match.group(1).lower() if sex_match else None),
        "encounter_type": context.get("encounter_type")
        or (encounter_match.group(1).lower() if encounter_match else None),
        "care_setting": context.get("care_setting")
        or (setting_match.group(1).lower() if setting_match else None),
    }


def get_section_priority(section: str) -> int:
    return SECTION_PRIORITY.get(section, SECTION_PRIORITY["general"])


def normalize_label(text: str) -> str:
    text = re.sub(r"^[A-Za-z /()\-]+:\s*", "", text)
    text = re.sub(r"[.;,\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_clinical_clauses(text: str) -> list[str]:
    text = normalize_label(text)
    return [part.strip() for part in re.split(r"\s*;\s*|\s{2,}|(?<=\.)\s+", text) if part.strip()]


def unique_by_normalized_label(items: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    max_items = limit if limit is not None else len(items)

    for item in items:
        key = normalize_label(str(item.get("label", ""))).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= max_items:
            break

    return output


def get_primary_complaint(evidence_index: list[dict[str, Any]]) -> str | None:
    for item in evidence_index:
        if item.get("section") == "chief_complaint":
            return normalize_label(item.get("text", ""))
    return None


def infer_action(text: str) -> str:
    lower = text.lower()
    if re.search(r"(started|initiated|began|given|received)", lower):
        return "started"
    if re.search(r"(continue|continued)", lower):
        return "continued"
    if re.search(r"(hold|held|stop|stopped|discontinue|discontinued)", lower):
        return "stopped"
    return "mentioned"


def infer_condition_status(text: str) -> str:
    lower = text.lower()
    if re.search(r"(rule out|unlikely|possible|probable|suspected|\?)", lower):
        return "uncertain"
    if re.search(r"(history of|hx of|chronic)", lower):
        return "historical_or_chronic"
    return "confirmed"
