from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "an",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "into",
    "its",
    "itself",
    "just",
    "more",
    "most",
    "not",
    "now",
    "off",
    "once",
    "only",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "until",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def _normalize_phrase_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for item in value:
        phrase = _normalize_text(item)
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(phrase)
    return normalized


def _normalize_code_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for item in value:
        code = _normalize_text(item).upper()
        if not code or code in seen:
            continue
        seen.add(code)
        normalized.append(code)
    return normalized


def _json_loads_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_note_excerpt(note_text: str, limit: int = 420) -> str:
    normalized = _normalize_text(note_text)
    if len(normalized) <= limit:
        return normalized
    truncated = normalized[:limit].rstrip()
    if " " not in truncated:
        return truncated
    return truncated.rsplit(" ", 1)[0]


def _build_query_phrases(
    note_text: str,
    structured_case_summary: dict[str, Any] | None = None,
) -> list[str]:
    structured_case_summary = structured_case_summary or {}
    phrases = [
        structured_case_summary.get("chief_complaint"),
        structured_case_summary.get("history_present_illness"),
        *structured_case_summary.get("discharge_diagnosis", []),
        *structured_case_summary.get("procedure", []),
        *structured_case_summary.get("past_medical_history", []),
        *structured_case_summary.get("hospital_course", []),
    ]
    if note_text:
        first_line = note_text.splitlines()[0].strip()
        if first_line:
            phrases.append(first_line)

    seen: set[str] = set()
    normalized: list[str] = []
    for item in phrases:
        phrase = _normalize_text(item).lower()
        if len(phrase) < 5 or phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


def _tokenize(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.]{2,}", _normalize_text(text).lower())
        if token not in _STOPWORDS
    }
    return tokens


def build_case_search_text(
    note_text: str = "",
    structured_case_summary: dict[str, Any] | None = None,
    knowledge_payload: dict[str, Any] | None = None,
) -> str:
    summary = structured_case_summary or {}
    knowledge_payload = knowledge_payload or {}
    parts = [
        summary.get("chief_complaint", ""),
        summary.get("history_present_illness", ""),
        " ".join(_normalize_phrase_list(summary.get("procedure"))),
        " ".join(_normalize_phrase_list(summary.get("past_medical_history"))),
        " ".join(_normalize_phrase_list(summary.get("pertinent_results"))),
        " ".join(_normalize_phrase_list(summary.get("hospital_course"))),
        " ".join(_normalize_phrase_list(summary.get("discharge_diagnosis"))),
        knowledge_payload.get("case_summary", ""),
        " ".join(_normalize_phrase_list(knowledge_payload.get("salient_clinical_patterns"))),
        " ".join(_normalize_phrase_list(knowledge_payload.get("coding_lessons"))),
        " ".join(_normalize_phrase_list(knowledge_payload.get("retrieval_queries"))),
        _build_note_excerpt(note_text, limit=900),
    ]
    return "\n".join(part for part in parts if _normalize_text(part))


class KnowledgeBase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_case_id TEXT,
                    coding_version TEXT,
                    note_excerpt TEXT,
                    case_summary TEXT,
                    gold_codes TEXT NOT NULL,
                    predicted_codes TEXT NOT NULL,
                    missed_codes TEXT NOT NULL,
                    extra_codes TEXT NOT NULL,
                    salient_clinical_patterns TEXT NOT NULL,
                    correct_prediction_reasons TEXT NOT NULL,
                    missed_code_lessons TEXT NOT NULL,
                    unsupported_prediction_lessons TEXT NOT NULL,
                    coding_lessons TEXT NOT NULL,
                    retrieval_queries TEXT NOT NULL,
                    knowledge_summary TEXT,
                    search_text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_entries_coding_version ON knowledge_entries(coding_version)"
            )

    def count_entries(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM knowledge_entries").fetchone()
        return int(row["count"]) if row is not None else 0

    def insert_memory(
        self,
        *,
        note_text: str,
        structured_case_summary: dict[str, Any] | None = None,
        agent3_output: dict[str, Any] | None = None,
        gold_codes: list[str] | None = None,
        predicted_codes: list[str] | None = None,
        missed_codes: list[str] | None = None,
        extra_codes: list[str] | None = None,
        source_case_id: str | None = None,
        coding_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        structured_case_summary = structured_case_summary or {}
        agent3_output = agent3_output or {}
        metadata = metadata or {}

        case_summary = _normalize_text(agent3_output.get("case_summary"))
        salient_clinical_patterns = _normalize_phrase_list(
            agent3_output.get("salient_clinical_patterns")
        )
        correct_prediction_reasons = _normalize_phrase_list(
            agent3_output.get("correct_prediction_reasons")
        )
        missed_code_lessons = _normalize_phrase_list(agent3_output.get("missed_code_lessons"))
        unsupported_prediction_lessons = _normalize_phrase_list(
            agent3_output.get("unsupported_prediction_lessons")
        )
        coding_lessons = _normalize_phrase_list(agent3_output.get("coding_lessons"))
        retrieval_queries = _normalize_phrase_list(agent3_output.get("retrieval_queries"))
        knowledge_summary = _normalize_text(agent3_output.get("knowledge_summary"))
        normalized_gold_codes = _normalize_code_list(gold_codes)
        normalized_predicted_codes = _normalize_code_list(predicted_codes)
        normalized_missed_codes = _normalize_code_list(missed_codes)
        normalized_extra_codes = _normalize_code_list(extra_codes)
        note_excerpt = _build_note_excerpt(note_text)
        search_text = build_case_search_text(
            note_text=note_text,
            structured_case_summary=structured_case_summary,
            knowledge_payload={
                "case_summary": case_summary,
                "salient_clinical_patterns": salient_clinical_patterns,
                "coding_lessons": coding_lessons,
                "retrieval_queries": retrieval_queries,
            },
        )

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO knowledge_entries (
                    created_at,
                    source_case_id,
                    coding_version,
                    note_excerpt,
                    case_summary,
                    gold_codes,
                    predicted_codes,
                    missed_codes,
                    extra_codes,
                    salient_clinical_patterns,
                    correct_prediction_reasons,
                    missed_code_lessons,
                    unsupported_prediction_lessons,
                    coding_lessons,
                    retrieval_queries,
                    knowledge_summary,
                    search_text,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    _normalize_text(source_case_id) or None,
                    _normalize_text(coding_version) or None,
                    note_excerpt,
                    case_summary,
                    json.dumps(normalized_gold_codes, ensure_ascii=False),
                    json.dumps(normalized_predicted_codes, ensure_ascii=False),
                    json.dumps(normalized_missed_codes, ensure_ascii=False),
                    json.dumps(normalized_extra_codes, ensure_ascii=False),
                    json.dumps(salient_clinical_patterns, ensure_ascii=False),
                    json.dumps(correct_prediction_reasons, ensure_ascii=False),
                    json.dumps(missed_code_lessons, ensure_ascii=False),
                    json.dumps(unsupported_prediction_lessons, ensure_ascii=False),
                    json.dumps(coding_lessons, ensure_ascii=False),
                    json.dumps(retrieval_queries, ensure_ascii=False),
                    knowledge_summary,
                    search_text,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            entry_id = int(cursor.lastrowid)

        return {
            "id": entry_id,
            "created_at": created_at,
            "source_case_id": _normalize_text(source_case_id),
            "coding_version": _normalize_text(coding_version),
            "note_excerpt": note_excerpt,
            "case_summary": case_summary,
            "gold_codes": normalized_gold_codes,
            "predicted_codes": normalized_predicted_codes,
            "missed_codes": normalized_missed_codes,
            "extra_codes": normalized_extra_codes,
            "salient_clinical_patterns": salient_clinical_patterns,
            "correct_prediction_reasons": correct_prediction_reasons,
            "missed_code_lessons": missed_code_lessons,
            "unsupported_prediction_lessons": unsupported_prediction_lessons,
            "coding_lessons": coding_lessons,
            "retrieval_queries": retrieval_queries,
            "knowledge_summary": knowledge_summary,
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "source_case_id": row["source_case_id"],
            "coding_version": row["coding_version"],
            "note_excerpt": row["note_excerpt"] or "",
            "case_summary": row["case_summary"] or "",
            "gold_codes": _normalize_code_list(_json_loads_list(row["gold_codes"])),
            "predicted_codes": _normalize_code_list(_json_loads_list(row["predicted_codes"])),
            "missed_codes": _normalize_code_list(_json_loads_list(row["missed_codes"])),
            "extra_codes": _normalize_code_list(_json_loads_list(row["extra_codes"])),
            "salient_clinical_patterns": _normalize_phrase_list(
                _json_loads_list(row["salient_clinical_patterns"])
            ),
            "correct_prediction_reasons": _normalize_phrase_list(
                _json_loads_list(row["correct_prediction_reasons"])
            ),
            "missed_code_lessons": _normalize_phrase_list(
                _json_loads_list(row["missed_code_lessons"])
            ),
            "unsupported_prediction_lessons": _normalize_phrase_list(
                _json_loads_list(row["unsupported_prediction_lessons"])
            ),
            "coding_lessons": _normalize_phrase_list(_json_loads_list(row["coding_lessons"])),
            "retrieval_queries": _normalize_phrase_list(_json_loads_list(row["retrieval_queries"])),
            "knowledge_summary": row["knowledge_summary"] or "",
            "search_text": row["search_text"] or "",
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    def search(
        self,
        *,
        note_text: str,
        structured_case_summary: dict[str, Any] | None = None,
        coding_version: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        structured_case_summary = structured_case_summary or {}
        if isinstance(top_k, bool):
            raise ValueError("top_k must be a positive integer.")
        try:
            normalized_top_k = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be a positive integer.") from exc
        if normalized_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_text = build_case_search_text(
            note_text=note_text,
            structured_case_summary=structured_case_summary,
        )
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []
        query_phrases = _build_query_phrases(
            note_text=note_text,
            structured_case_summary=structured_case_summary,
        )

        with self._connect() as connection:
            if _normalize_text(coding_version):
                rows = connection.execute(
                    "SELECT * FROM knowledge_entries WHERE coding_version = ?",
                    (_normalize_text(coding_version),),
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM knowledge_entries").fetchall()

        scored_entries: list[dict[str, Any]] = []
        for row in rows:
            entry = self._row_to_dict(row)
            entry_tokens = _tokenize(entry["search_text"])
            token_overlap = query_tokens & entry_tokens
            score = float(len(token_overlap))
            search_text_lower = entry["search_text"].lower()
            for phrase in query_phrases:
                if phrase in search_text_lower:
                    score += 2.5
            if score <= 0:
                continue
            entry["score"] = round(score, 3)
            scored_entries.append(entry)

        scored_entries.sort(key=lambda item: (-float(item["score"]), -int(item["id"])))
        return [
            {
                "id": entry["id"],
                "score": entry["score"],
                "source_case_id": entry["source_case_id"],
                "coding_version": entry["coding_version"],
                "case_summary": entry["case_summary"],
                "knowledge_summary": entry["knowledge_summary"],
                "salient_clinical_patterns": entry["salient_clinical_patterns"],
                "coding_lessons": entry["coding_lessons"],
                "retrieval_queries": entry["retrieval_queries"],
                "gold_codes": entry["gold_codes"],
                "missed_codes": entry["missed_codes"],
                "extra_codes": entry["extra_codes"],
                "note_excerpt": entry["note_excerpt"],
            }
            for entry in scored_entries[:normalized_top_k]
        ]
