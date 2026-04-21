from __future__ import annotations

import ast
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


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
    labels = _coerce_label_values(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = _normalize_code(label)
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _coerce_label_values(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if converted is not value:
            return _coerce_label_values(converted)
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] in "[{(":
            parsed = _parse_structured_label_string(stripped)
            if parsed is not None:
                return _coerce_label_values(parsed)
        return [part.strip() for part in stripped.split(";") if part.strip()]
    if _is_missing_value(value):
        return []
    return [str(value).strip()]


def _parse_structured_label_string(value: str) -> Any | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return parsed
    return None


def _is_missing_value(value: Any) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False


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


def _normalize_requested_split(split: str) -> str:
    normalized = str(split or "").strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "dev": "val",
        "valid": "val",
        "validation": "val",
        "val": "val",
        "test": "test",
    }
    if normalized not in mapping:
        raise ValueError("split must be one of: train, dev, val, test.")
    return mapping[normalized]


def _normalize_dataset_split_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"dev", "valid", "validation", "val"}:
        return "val"
    if normalized in {"train", "training"}:
        return "train"
    if normalized == "test":
        return "test"
    return normalized


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas and pyarrow are required to load feather-backed datasets."
        ) from exc
    return pd


def _read_feather_schema(path: str | Path) -> list[str]:
    feather_path = Path(path)
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        pd = _import_pandas()
        return list(pd.read_feather(feather_path).columns)

    with pa.memory_map(str(feather_path), "r") as source:
        reader = ipc.open_file(source)
        return [field.name for field in reader.schema]


def _read_feather_frame(path: str | Path, desired_columns: list[str] | None = None):
    pd = _import_pandas()
    available_columns = _read_feather_schema(path)
    columns = None
    if desired_columns is not None:
        columns = [column for column in desired_columns if column in available_columns]
    return pd.read_feather(path, columns=columns)


def _infer_base_feather_from_split(split_path: str | Path) -> Path | None:
    split_file = Path(split_path)
    candidates: list[Path] = []
    stem = split_file.stem
    for suffix in ("_split", "_splits"):
        if stem.endswith(suffix):
            candidates.append(split_file.with_name(f"{stem[:-len(suffix)]}.feather"))
    if "_subsplit_" in stem:
        candidates.append(split_file.with_name(f"{stem.split('_subsplit_', 1)[0]}.feather"))

    for candidate in candidates:
        if candidate.exists() and candidate != split_file:
            return candidate
    return None


def _resolve_feather_assets(dataset_path: str | Path) -> tuple[Path, Path | None]:
    path = Path(dataset_path)
    if path.is_dir():
        direct_base = path / f"{path.name}.feather"
        canonical_split = path / f"{path.name}_split.feather"
        canonical_splits = path / f"{path.name}_splits.feather"

        split_file = canonical_split if canonical_split.exists() else None
        if split_file is None and canonical_splits.exists():
            split_file = canonical_splits
        if split_file is None:
            split_candidates = sorted(
                candidate
                for candidate in path.glob("*.feather")
                if candidate.stem.endswith("_split")
                or candidate.stem.endswith("_splits")
                or "_subsplit_" in candidate.stem
            )
            split_file = split_candidates[0] if split_candidates else None

        data_file = direct_base if direct_base.exists() else None
        if data_file is None and split_file is not None:
            data_file = _infer_base_feather_from_split(split_file)
        if data_file is None:
            for candidate in sorted(path.glob("*.feather")):
                if candidate == split_file:
                    continue
                if candidate.stem.endswith("_split") or candidate.stem.endswith("_splits") or "_subsplit_" in candidate.stem:
                    continue
                data_file = candidate
                break

        if data_file is None and split_file is not None:
            raise FileNotFoundError(
                f"Found split metadata at {split_file}, but no base feather file with note text and labels was found."
            )
        if data_file is None:
            raise FileNotFoundError(f"No supported dataset file was found in {path}")
        return data_file, split_file

    if path.suffix.lower() != ".feather":
        raise FileNotFoundError(f"Unsupported dataset path: {path}")

    if path.stem.endswith("_split") or path.stem.endswith("_splits") or "_subsplit_" in path.stem:
        data_file = _infer_base_feather_from_split(path)
        if data_file is None:
            raise FileNotFoundError(
                f"Found split metadata at {path}, but no matching base feather file with note text and labels was found."
            )
        return data_file, path

    sibling_split = path.with_name(f"{path.stem}_split.feather")
    if sibling_split.exists():
        return path, sibling_split
    sibling_splits = path.with_name(f"{path.stem}_splits.feather")
    if sibling_splits.exists():
        return path, sibling_splits
    return path, None


def _resolve_example_id_column(columns: list[str]) -> str:
    for candidate in ("hadm_id", "_id", "note_id"):
        if candidate in columns:
            return candidate
    raise ValueError("Dataset must include one of: hadm_id, _id, note_id.")


def _resolve_length_value(row: dict[str, Any]) -> int | None:
    for field_name in ("length", "num_words"):
        value = row.get(field_name)
        if _is_missing_value(value):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_labels_value(row: dict[str, Any]) -> list[str]:
    for field_name in ("labels", "target", "icd10_diag", "icd9_diag"):
        if field_name in row:
            labels = parse_label_string(row.get(field_name))
            if labels:
                return labels
    return []


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
    normalized_split = _normalize_requested_split(split)
    base_path = Path(dataset_dir)
    csv_split_name = "dev" if normalized_split == "val" else normalized_split
    csv_path = base_path / f"{csv_split_name}_full.csv"
    if csv_path.exists():
        return csv_path
    if base_path.is_dir():
        _resolve_feather_assets(base_path)
        return base_path
    raise FileNotFoundError(f"No dataset split matching {split!r} was found in {base_path}")


def iter_mimic_examples(
    csv_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    split: str | None = None,
) -> Iterator[MIMICNoteExample]:
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"MIMIC dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".csv":
        yield from _iter_csv_mimic_examples(dataset_path, limit=limit, offset=offset)
        return

    yield from _iter_feather_mimic_examples(dataset_path, limit=limit, offset=offset, split=split)


def _iter_csv_mimic_examples(
    csv_file: str | Path,
    limit: int | None = None,
    offset: int = 0,
) -> Iterator[MIMICNoteExample]:
    csv_file = Path(csv_file)

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


def _iter_feather_mimic_examples(
    dataset_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    split: str | None = None,
) -> Iterator[MIMICNoteExample]:
    normalized_split = _normalize_requested_split(split) if split is not None else None
    data_file, split_file = _resolve_feather_assets(dataset_path)
    desired_columns = [
        "subject_id",
        "hadm_id",
        "_id",
        "note_id",
        "text",
        "labels",
        "target",
        "icd10_diag",
        "icd9_diag",
        "length",
        "num_words",
        "split",
    ]
    dataframe = _read_feather_frame(data_file, desired_columns=desired_columns)
    id_column = _resolve_example_id_column(list(dataframe.columns))

    if split_file is not None:
        if normalized_split is None:
            raise ValueError(
                "split is required when loading a feather-backed dataset with separate split metadata."
            )
        split_frame = _read_feather_frame(split_file, desired_columns=["_id", "split"])
        if "_id" not in split_frame.columns or "split" not in split_frame.columns:
            raise ValueError(f"Split metadata file is missing required columns: {split_file}")
        split_frame = split_frame.copy()
        split_frame["__normalized_split"] = split_frame["split"].map(_normalize_dataset_split_value)
        filtered_ids = [
            str(value).strip()
            for value in split_frame.loc[split_frame["__normalized_split"] == normalized_split, "_id"]
            if str(value).strip()
        ]
        order_lookup = {value: index for index, value in enumerate(filtered_ids)}
        dataframe = dataframe.copy()
        dataframe["__example_id"] = dataframe[id_column].astype(str).str.strip()
        dataframe = dataframe[dataframe["__example_id"].isin(order_lookup)].copy()
        dataframe["__split_order"] = dataframe["__example_id"].map(order_lookup)
        dataframe.sort_values("__split_order", kind="stable", inplace=True)
    elif normalized_split is not None and "split" in dataframe.columns:
        dataframe = dataframe[
            dataframe["split"].map(_normalize_dataset_split_value) == normalized_split
        ].copy()

    if offset:
        dataframe = dataframe.iloc[offset:]
    if limit is not None:
        dataframe = dataframe.iloc[:limit]

    for row in dataframe.to_dict(orient="records"):
        labels = _resolve_labels_value(row)
        raw_labels = row.get("labels")
        if raw_labels is None:
            raw_labels = row.get("target")
        yield MIMICNoteExample(
            subject_id=str(row.get("subject_id", "")).strip(),
            hadm_id=str(row.get("hadm_id") or row.get("_id") or row.get("note_id") or "").strip(),
            text=str(row.get("text", "")).strip(),
            labels=labels,
            length=_resolve_length_value(row),
            raw_labels=";".join(labels) if raw_labels is None else str(raw_labels).strip(),
        )


def load_mimic_examples(
    csv_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    split: str | None = None,
) -> list[MIMICNoteExample]:
    return list(iter_mimic_examples(csv_path=csv_path, limit=limit, offset=offset, split=split))
