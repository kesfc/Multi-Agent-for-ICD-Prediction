from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class MIMICNoteExample:
    subject_id: str
    hadm_id: str
    text: str
    labels: list[str]
    length: int | None
    raw_labels: str

    def to_patient_context(self, coding_version: str) -> dict[str, str | int]:
        context: dict[str, str | int] = {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "coding_version": coding_version,
        }
        if self.length is not None:
            context["note_length"] = self.length
        return context


def parse_label_string(value: str | None) -> list[str]:
    labels = [part.strip() for part in str(value or "").split(";")]
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not label or label in seen:
            continue
        deduped.append(label)
        seen.add(label)
    return deduped


def _normalize_code(value: str | None) -> str:
    return str(value or "").strip().upper()


def _looks_like_code(value: str | None) -> bool:
    code = _normalize_code(value)
    if not code:
        return False
    return any(char.isdigit() for char in code) and all(
        char.isalnum() or char == "." for char in code
    )


def _normalize_description(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def load_code_candidate_records(csv_path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    candidate_file = Path(csv_path)
    if not candidate_file.exists():
        raise FileNotFoundError(f"Code candidate CSV not found: {candidate_file}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than 0 when provided.")

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    with candidate_file.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(handle)
            for row in reader:
                code = _normalize_code(
                    row.get("code")
                    or row.get("codes")
                    or row.get("icd_code")
                    or row.get("ICD9_CODE")
                    or row.get("label")
                    or row.get("labels")
                )
                if not _looks_like_code(code) or code in seen:
                    continue
                description = _normalize_description(
                    row.get("description")
                    or row.get("desc")
                    or row.get("long_title")
                    or row.get("short_title")
                    or row.get("title")
                )
                records.append({"code": code, "description": description})
                seen.add(code)
                if limit is not None and len(records) >= limit:
                    return records
            return records

        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            code = _normalize_code(row[0])
            if not _looks_like_code(code) or code in seen:
                continue
            description = _normalize_description(row[1]) if len(row) > 1 else ""
            records.append({"code": code, "description": description})
            seen.add(code)
            if limit is not None and len(records) >= limit:
                return records
    return records


def load_code_candidates(csv_path: str | Path, limit: int | None = None) -> list[str]:
    return [record["code"] for record in load_code_candidate_records(csv_path, limit=limit)]


def load_hadm_ids(csv_path: str | Path) -> set[str]:
    id_file = Path(csv_path)
    if not id_file.exists():
        raise FileNotFoundError(f"HADM ID CSV not found: {id_file}")

    hadm_ids: set[str] = set()
    header_values = {"hadm_id", "hadm", "id"}
    with id_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            value = str(row[0] or "").strip()
            if not value or value.lower() in header_values:
                continue
            hadm_ids.add(value)
    return hadm_ids


def resolve_top_codes_path(dataset_path: str | Path) -> Path | None:
    path = Path(dataset_path)
    dataset_dir = path.parent if path.suffix else path
    candidate_path = dataset_dir / "TOP_50_CODES.csv"
    return candidate_path if candidate_path.exists() else None


def infer_coding_version_from_path(path: str | Path) -> str:
    normalized = str(path).lower()
    if "icd10" in normalized:
        return "ICD-10"
    if "icd9" in normalized:
        return "ICD-9"
    return "ICD-10"


def resolve_mimic_split_path(dataset_dir: str | Path, split: str) -> Path:
    normalized_split = split.lower().strip()
    if normalized_split not in {"train", "dev", "test"}:
        raise ValueError("split must be one of: train, dev, test.")
    return Path(dataset_dir) / f"{normalized_split}_full.csv"


def iter_mimic_examples(
    csv_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
) -> Iterator[MIMICNoteExample]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"MIMIC CSV not found: {csv_file}")

    emitted = 0
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index < offset:
                continue
            if limit is not None and emitted >= limit:
                break

            length_value = str(row.get("length", "")).strip()
            yield MIMICNoteExample(
                subject_id=str(row.get("subject_id", "")).strip(),
                hadm_id=str(row.get("hadm_id", "")).strip(),
                text=str(row.get("text", "")).strip(),
                labels=parse_label_string(row.get("labels")),
                length=int(length_value) if length_value.isdigit() else None,
                raw_labels=str(row.get("labels", "")).strip(),
            )
            emitted += 1


def load_mimic_examples(
    csv_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[MIMICNoteExample]:
    return list(iter_mimic_examples(csv_path=csv_path, limit=limit, offset=offset))
