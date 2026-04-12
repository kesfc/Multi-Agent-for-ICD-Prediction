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
