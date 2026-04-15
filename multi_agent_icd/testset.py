from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_icd.datasets import (
    infer_coding_version_from_path,
    iter_mimic_examples,
    load_code_candidate_records,
    load_hadm_ids,
    resolve_top_codes_path,
)
from multi_agent_icd.run import MultiAgentController


@dataclass
class TestSetSummary:
    dataset_path: str
    coding_version: str
    num_examples: int
    failed_examples: int
    true_positives: int
    total_gold_codes: int
    total_predicted_codes: int
    exact_match_count: int
    precision_at_k_total: float
    precision_at_k_covered_total: float
    recall_at_k_top_code_total: float
    covered_examples: int
    top_code_gold_codes: int
    average_gold_codes: float
    average_predicted_codes: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    candidate_output_limit: int
    candidate_code_count: int = 0
    top_codes_path: str | None = None
    hadm_ids_path: str | None = None
    hadm_id_filter_count: int = 0
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        precision_at_k_all = _safe_divide(self.precision_at_k_total, self.num_examples)
        precision_at_k_covered = _safe_divide(
            self.precision_at_k_covered_total,
            self.covered_examples,
        )
        return {
            "dataset_path": self.dataset_path,
            "coding_version": self.coding_version,
            "num_examples": self.num_examples,
            "failed_examples": self.failed_examples,
            "hadm_id_filter_count": self.hadm_id_filter_count,
            "covered_examples": self.covered_examples,
            "case_coverage": _safe_divide(self.covered_examples, self.num_examples),
            "true_positives": self.true_positives,
            "total_gold_codes": self.total_gold_codes,
            "top_code_gold_codes": self.top_code_gold_codes,
            "label_coverage": _safe_divide(self.top_code_gold_codes, self.total_gold_codes),
            "total_predicted_codes": self.total_predicted_codes,
            "exact_match_count": self.exact_match_count,
            "exact_match_rate": _safe_divide(self.exact_match_count, self.num_examples),
            f"precision_at_{self.candidate_output_limit}": precision_at_k_all,
            f"precision_at_{self.candidate_output_limit}_all": precision_at_k_all,
            f"precision_at_{self.candidate_output_limit}_covered": precision_at_k_covered,
            f"recall_at_{self.candidate_output_limit}_top_codes": _safe_divide(
                self.recall_at_k_top_code_total,
                self.covered_examples,
            ),
            "average_gold_codes": self.average_gold_codes,
            "average_predicted_codes": self.average_predicted_codes,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "candidate_output_limit": self.candidate_output_limit,
            "candidate_code_count": self.candidate_code_count,
            "top_codes_path": self.top_codes_path,
            "hadm_ids_path": self.hadm_ids_path,
            "output_path": self.output_path,
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_candidate_output_limit(candidate_output_limit: int) -> int:
    if isinstance(candidate_output_limit, bool):
        raise ValueError("candidate_output_limit must be a positive integer.")
    try:
        value = int(candidate_output_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_output_limit must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("candidate_output_limit must be a positive integer.")
    return value


def extract_predicted_codes(
    agent2_output: dict[str, Any] | None,
    candidate_output_limit: int | None = None,
    allowed_codes: list[str] | None = None,
    fill_to_limit: bool = False,
) -> list[str]:
    if not isinstance(agent2_output, dict):
        return []

    ordered_codes: list[str] = []
    seen: set[str] = set()
    allowed_code_set = {str(code).strip().upper() for code in allowed_codes or [] if str(code).strip()}
    normalized_limit = (
        _validate_candidate_output_limit(candidate_output_limit)
        if candidate_output_limit is not None
        else None
    )

    def add_candidate(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        code = str(candidate.get("code", "")).strip().upper()
        if not code or code in seen:
            return
        if allowed_code_set and code not in allowed_code_set:
            return
        if normalized_limit is not None and len(ordered_codes) >= normalized_limit:
            return
        seen.add(code)
        ordered_codes.append(code)

    add_candidate(agent2_output.get("principal_diagnosis"))
    for item in agent2_output.get("secondary_diagnoses", []):
        add_candidate(item)
    for item in agent2_output.get("procedures", []):
        add_candidate(item)

    if fill_to_limit and normalized_limit is not None and allowed_code_set:
        for code in allowed_codes or []:
            normalized_code = str(code).strip().upper()
            if not normalized_code or normalized_code in seen:
                continue
            if len(ordered_codes) >= normalized_limit:
                break
            seen.add(normalized_code)
            ordered_codes.append(normalized_code)

    return ordered_codes


def _build_output_record(
    example: Any,
    patient_context: dict[str, Any],
    state: dict[str, Any],
    predicted_codes: list[str],
    candidate_output_limit: int,
    top_codes_path: Path | None,
    top_code_gold_labels: list[str],
) -> dict[str, Any]:
    return {
        "subject_id": example.subject_id,
        "hadm_id": example.hadm_id,
        "coding_version": patient_context["coding_version"],
        "gold_labels": example.labels,
        "top_code_gold_labels": top_code_gold_labels,
        "predicted_codes": predicted_codes,
        "candidate_output_limit": candidate_output_limit,
        "top_codes_path": str(top_codes_path) if top_codes_path is not None else None,
        "text_length": example.length,
        "agent1_output": state.get("agent_outputs", {}).get("agent1"),
        "agent2_output": state.get("agent_outputs", {}).get("agent2"),
        "execution_trace": state.get("execution_trace", []),
    }


def _build_failed_output_record(
    example: Any,
    patient_context: dict[str, Any],
    state: dict[str, Any],
    error_message: str,
    candidate_output_limit: int,
    top_codes_path: Path | None,
    top_code_gold_labels: list[str],
) -> dict[str, Any]:
    return {
        "subject_id": example.subject_id,
        "hadm_id": example.hadm_id,
        "coding_version": patient_context["coding_version"],
        "gold_labels": example.labels,
        "top_code_gold_labels": top_code_gold_labels,
        "predicted_codes": [],
        "candidate_output_limit": candidate_output_limit,
        "top_codes_path": str(top_codes_path) if top_codes_path is not None else None,
        "text_length": example.length,
        "status": "failed",
        "error": error_message,
        "agent1_output": state.get("agent_outputs", {}).get("agent1") if isinstance(state, dict) else None,
        "agent2_output": state.get("agent_outputs", {}).get("agent2") if isinstance(state, dict) else None,
        "execution_trace": state.get("execution_trace", []) if isinstance(state, dict) else [],
    }


def _validate_two_agent_state(state: dict[str, Any]) -> None:
    agent_outputs = state.get("agent_outputs", {})
    missing = [agent_name for agent_name in ("agent1", "agent2") if agent_name not in agent_outputs]
    if missing:
        raise RuntimeError(f"Pipeline did not produce required agent output(s): {', '.join(missing)}")

    failed: list[str] = []
    for item in state.get("execution_trace", []):
        agent_name = item.get("agent_name", "")
        if agent_name not in {"agent1", "agent2"} or item.get("status") == "completed":
            continue
        trace_message = str(item.get("message", "")).strip()
        output_message = ""
        agent_output = agent_outputs.get(agent_name)
        if isinstance(agent_output, dict):
            output_message = str(agent_output.get("message", "")).strip()
        message = trace_message or output_message or "no error message was returned"
        failed.append(f"{agent_name}: {message}")
    if failed:
        raise RuntimeError(f"Required agent(s) did not complete successfully: {'; '.join(failed)}")


def run_testset(
    csv_path: str | Path,
    controller: MultiAgentController,
    limit: int | None = None,
    offset: int = 0,
    output_path: str | Path | None = None,
    top_codes_path: str | Path | None = None,
    hadm_ids_path: str | Path | None = None,
    candidate_output_limit: int = 5,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(csv_path)
    coding_version = infer_coding_version_from_path(dataset_path)
    output_file = Path(output_path) if output_path else None
    normalized_candidate_output_limit = _validate_candidate_output_limit(candidate_output_limit)
    top_code_file = Path(top_codes_path) if top_codes_path is not None else resolve_top_codes_path(dataset_path)
    candidate_code_records = load_code_candidate_records(top_code_file) if top_code_file is not None else []
    candidate_code_set = [record["code"] for record in candidate_code_records]
    candidate_code_lookup = set(candidate_code_set)
    hadm_id_file = Path(hadm_ids_path) if hadm_ids_path is not None else None
    hadm_id_filter = load_hadm_ids(hadm_id_file) if hadm_id_file is not None else None
    if top_code_file is not None and not candidate_code_set:
        raise ValueError(f"No code candidates were found in {top_code_file}")
    if hadm_id_file is not None and not hadm_id_filter:
        raise ValueError(f"No HADM IDs were found in {hadm_id_file}")

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    failed_examples = 0
    total_gold_codes = 0
    total_predicted_codes = 0
    true_positives = 0
    exact_match_count = 0
    precision_at_k_total = 0.0
    precision_at_k_covered_total = 0.0
    recall_at_k_top_code_total = 0.0
    covered_examples = 0
    top_code_gold_codes = 0
    skipped_by_hadm_filter = 0
    filtered_seen = 0

    writer = output_file.open("w", encoding="utf-8") if output_file is not None else None
    try:
        for example in iter_mimic_examples(dataset_path):
            if hadm_id_filter is not None and example.hadm_id not in hadm_id_filter:
                skipped_by_hadm_filter += 1
                continue
            if filtered_seen < offset:
                filtered_seen += 1
                continue
            if limit is not None and total_examples >= limit:
                break

            filtered_seen += 1
            total_examples += 1
            patient_context = example.to_patient_context(coding_version=coding_version)
            patient_context["candidate_output_limit"] = normalized_candidate_output_limit
            if candidate_code_set:
                patient_context["candidate_code_set"] = candidate_code_set
                patient_context["candidate_code_records"] = candidate_code_records
            state: dict[str, Any] = {}
            gold_set = set(example.labels)
            top_code_gold_set = gold_set & candidate_code_lookup if candidate_code_lookup else gold_set
            top_code_gold_labels = [code for code in example.labels if code in top_code_gold_set]
            try:
                state = controller.run(
                    note_text=example.text,
                    patient_context=patient_context,
                    requested_agents=["agent1", "agent2"],
                )
                _validate_two_agent_state(state)
            except Exception as exc:
                if not continue_on_error:
                    raise
                failed_examples += 1
                if top_code_gold_set:
                    covered_examples += 1
                total_gold_codes += len(gold_set)
                top_code_gold_codes += len(top_code_gold_set)
                if writer is not None:
                    writer.write(
                        json.dumps(
                            _build_failed_output_record(
                                example=example,
                                patient_context=patient_context,
                                state=state,
                                error_message=str(exc),
                                candidate_output_limit=normalized_candidate_output_limit,
                                top_codes_path=top_code_file,
                                top_code_gold_labels=top_code_gold_labels,
                            ),
                            ensure_ascii=False,
                        )
                    )
                    writer.write("\n")
                continue
            agent2_output = state.get("agent_outputs", {}).get("agent2")
            predicted_codes = extract_predicted_codes(
                agent2_output,
                candidate_output_limit=normalized_candidate_output_limit,
                allowed_codes=candidate_code_set,
                fill_to_limit=bool(candidate_code_set),
            )

            predicted_set = set(predicted_codes)
            top_code_hits = set(predicted_codes[:normalized_candidate_output_limit]) & top_code_gold_set
            if top_code_gold_set:
                covered_examples += 1
                precision_at_k_covered_total += _safe_divide(
                    len(top_code_hits),
                    normalized_candidate_output_limit,
                )
                recall_at_k_top_code_total += _safe_divide(len(top_code_hits), len(top_code_gold_set))

            precision_at_k_total += _safe_divide(
                len(gold_set & set(predicted_codes[:normalized_candidate_output_limit])),
                normalized_candidate_output_limit,
            )
            total_gold_codes += len(gold_set)
            top_code_gold_codes += len(top_code_gold_set)
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
                            candidate_output_limit=normalized_candidate_output_limit,
                            top_codes_path=top_code_file,
                            top_code_gold_labels=top_code_gold_labels,
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
        failed_examples=failed_examples,
        true_positives=true_positives,
        total_gold_codes=total_gold_codes,
        total_predicted_codes=total_predicted_codes,
        exact_match_count=exact_match_count,
        precision_at_k_total=precision_at_k_total,
        precision_at_k_covered_total=precision_at_k_covered_total,
        recall_at_k_top_code_total=recall_at_k_top_code_total,
        covered_examples=covered_examples,
        top_code_gold_codes=top_code_gold_codes,
        average_gold_codes=_safe_divide(total_gold_codes, total_examples),
        average_predicted_codes=_safe_divide(total_predicted_codes, total_examples),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        candidate_output_limit=normalized_candidate_output_limit,
        candidate_code_count=len(candidate_code_set),
        top_codes_path=str(top_code_file) if top_code_file is not None else None,
        hadm_ids_path=str(hadm_id_file) if hadm_id_file is not None else None,
        hadm_id_filter_count=len(hadm_id_filter) if hadm_id_filter is not None else 0,
        output_path=str(output_file) if output_file is not None else None,
    )
    return summary.to_dict()
