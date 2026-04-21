from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multi_agent_icd.datasets import (
    infer_coding_version_from_path,
    iter_mimic_examples,
    load_code_candidate_records,
    resolve_mimic_split_path,
    resolve_top_codes_path,
)
from multi_agent_icd.utils.clinical_text import normalize_clinical_text


ANSWER_FORMAT = "Answer: CODE1; CODE2; CODE3"


def build_easyr1_icd_prompt(
    *,
    note_text: str,
    coding_version: str,
    candidate_code_records: list[dict[str, str]] | None = None,
    note_id: str | None = None,
) -> str:
    candidate_code_records = candidate_code_records or []
    prompt_lines = [
        "You are an ICD coding specialist.",
        f"Predict the complete supported set of {coding_version} codes for the clinical note below.",
        "Think inside <think>...</think> tags.",
        "After thinking, end with exactly one final answer line in this format:",
        ANSWER_FORMAT,
        "Rules:",
        "- Output ICD code strings only on the final answer line.",
        "- Use uppercase codes separated by semicolons.",
        "- Do not include descriptions or extra prose after the final answer line.",
        "- If no code is supportable, output `Answer: NONE`.",
    ]
    if note_id:
        prompt_lines.append(f"- Case id: {note_id}.")

    if candidate_code_records:
        prompt_lines.extend(
            [
                "",
                "Only use codes from the allowed candidate set below:",
            ]
        )
        for record in candidate_code_records:
            code = str(record.get("code", "")).strip()
            description = str(record.get("description", "")).strip()
            if not code:
                continue
            if description:
                prompt_lines.append(f"- {code}: {description}")
            else:
                prompt_lines.append(f"- {code}")

    prompt_lines.extend(
        [
            "",
            "Clinical note:",
            normalize_clinical_text(note_text),
        ]
    )
    return "\n".join(prompt_lines).strip()


def build_easyr1_ground_truth(
    *,
    subject_id: str,
    hadm_id: str,
    coding_version: str,
    labels: list[str],
) -> str:
    payload = {
        "subject_id": str(subject_id),
        "hadm_id": str(hadm_id),
        "coding_version": str(coding_version),
        "labels": list(labels),
    }
    return json.dumps(payload, ensure_ascii=False)


def write_easyr1_split(
    *,
    dataset_path: str | Path,
    split: str,
    output_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    top_codes_path: str | Path | None = None,
    max_note_chars: int | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    split_dataset_path = resolve_mimic_split_path(dataset_path, split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coding_version = infer_coding_version_from_path(split_dataset_path)
    resolved_top_codes_path = (
        Path(top_codes_path) if top_codes_path is not None else resolve_top_codes_path(split_dataset_path)
    )
    candidate_code_records = (
        load_code_candidate_records(resolved_top_codes_path) if resolved_top_codes_path is not None else []
    )

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in iter_mimic_examples(split_dataset_path, split=split, limit=limit, offset=offset):
            note_text = example.text
            if max_note_chars is not None and max_note_chars > 0:
                note_text = note_text[:max_note_chars]

            record = {
                "prompt": build_easyr1_icd_prompt(
                    note_text=note_text,
                    coding_version=coding_version,
                    candidate_code_records=candidate_code_records,
                    note_id=example.hadm_id,
                ),
                "answer": build_easyr1_ground_truth(
                    subject_id=example.subject_id,
                    hadm_id=example.hadm_id,
                    coding_version=coding_version,
                    labels=example.labels,
                ),
                "subject_id": example.subject_id,
                "hadm_id": example.hadm_id,
                "coding_version": coding_version,
                "labels": example.labels,
                "label_count": len(example.labels),
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            written += 1

    return {
        "dataset_path": str(dataset_path),
        "resolved_dataset_path": str(split_dataset_path),
        "split": split,
        "output_path": str(output_path),
        "num_examples": written,
        "coding_version": coding_version,
        "top_codes_path": str(resolved_top_codes_path) if resolved_top_codes_path is not None else None,
        "candidate_code_count": len(candidate_code_records),
    }


def prepare_easyr1_icd_dataset(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    split_to_filename: dict[str, str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    top_codes_path: str | Path | None = None,
    max_note_chars: int | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_to_filename = split_to_filename or {
        "train": "train.jsonl",
        "dev": "val.jsonl",
        "test": "test.jsonl",
    }

    written_summaries = []
    for split, filename in split_to_filename.items():
        written_summaries.append(
            write_easyr1_split(
                dataset_path=dataset_path,
                split=split,
                output_path=output_dir / filename,
                limit=limit,
                offset=offset,
                top_codes_path=top_codes_path,
                max_note_chars=max_note_chars,
            )
        )

    return {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "splits": written_summaries,
    }
