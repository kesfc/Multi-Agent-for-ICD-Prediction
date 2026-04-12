from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_icd.datasets import infer_coding_version_from_path, iter_mimic_examples
from multi_agent_icd.run import MultiAgentController


@dataclass
class TestSetSummary:
    dataset_path: str
    coding_version: str
    num_examples: int
    true_positives: int
    total_gold_codes: int
    total_predicted_codes: int
    exact_match_count: int
    average_gold_codes: float
    average_predicted_codes: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "coding_version": self.coding_version,
            "num_examples": self.num_examples,
            "true_positives": self.true_positives,
            "total_gold_codes": self.total_gold_codes,
            "total_predicted_codes": self.total_predicted_codes,
            "exact_match_count": self.exact_match_count,
            "exact_match_rate": _safe_divide(self.exact_match_count, self.num_examples),
            "average_gold_codes": self.average_gold_codes,
            "average_predicted_codes": self.average_predicted_codes,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "output_path": self.output_path,
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def extract_predicted_codes(agent2_output: dict[str, Any] | None) -> list[str]:
    if not isinstance(agent2_output, dict):
        return []

    ordered_codes: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        code = str(candidate.get("code", "")).strip().upper()
        if not code or code in seen:
            return
        seen.add(code)
        ordered_codes.append(code)

    add_candidate(agent2_output.get("principal_diagnosis"))
    for item in agent2_output.get("secondary_diagnoses", []):
        add_candidate(item)
    for item in agent2_output.get("procedures", []):
        add_candidate(item)
    return ordered_codes


def _build_output_record(
    example: Any,
    patient_context: dict[str, Any],
    state: dict[str, Any],
    predicted_codes: list[str],
) -> dict[str, Any]:
    return {
        "subject_id": example.subject_id,
        "hadm_id": example.hadm_id,
        "coding_version": patient_context["coding_version"],
        "gold_labels": example.labels,
        "predicted_codes": predicted_codes,
        "text_length": example.length,
        "agent1_output": state.get("agent_outputs", {}).get("agent1"),
        "agent2_output": state.get("agent_outputs", {}).get("agent2"),
        "execution_trace": state.get("execution_trace", []),
    }


def run_testset(
    csv_path: str | Path,
    controller: MultiAgentController,
    limit: int | None = None,
    offset: int = 0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(csv_path)
    coding_version = infer_coding_version_from_path(dataset_path)
    output_file = Path(output_path) if output_path else None

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_gold_codes = 0
    total_predicted_codes = 0
    true_positives = 0
    exact_match_count = 0

    writer = output_file.open("w", encoding="utf-8") if output_file is not None else None
    try:
        for example in iter_mimic_examples(dataset_path, limit=limit, offset=offset):
            total_examples += 1
            patient_context = example.to_patient_context(coding_version=coding_version)
            state = controller.run(note_text=example.text, patient_context=patient_context)
            agent2_output = state.get("agent_outputs", {}).get("agent2")
            predicted_codes = extract_predicted_codes(agent2_output)

            gold_set = set(example.labels)
            predicted_set = set(predicted_codes)
            total_gold_codes += len(gold_set)
            total_predicted_codes += len(predicted_set)
            true_positives += len(gold_set & predicted_set)
            if gold_set == predicted_set:
                exact_match_count += 1

            if writer is not None:
                writer.write(
                    json.dumps(
                        _build_output_record(
                            example=example,
                            patient_context=patient_context,
                            state=state,
                            predicted_codes=predicted_codes,
                        ),
                        ensure_ascii=False,
                    )
                )
                writer.write("\n")
    finally:
        if writer is not None:
            writer.close()

    precision = _safe_divide(true_positives, total_predicted_codes)
    recall = _safe_divide(true_positives, total_gold_codes)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    summary = TestSetSummary(
        dataset_path=str(dataset_path),
        coding_version=coding_version,
        num_examples=total_examples,
        true_positives=true_positives,
        total_gold_codes=total_gold_codes,
        total_predicted_codes=total_predicted_codes,
        exact_match_count=exact_match_count,
        average_gold_codes=_safe_divide(total_gold_codes, total_examples),
        average_predicted_codes=_safe_divide(total_predicted_codes, total_examples),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        output_path=str(output_file) if output_file is not None else None,
    )
    return summary.to_dict()
