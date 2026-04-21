from __future__ import annotations

import ast
import json
import re
from typing import Any


REWARD_NAME = "icd_f1"
REWARD_TYPE = "batch"


def _normalize_code(value: Any) -> str:
    code = str(value or "").strip().upper()
    code = re.sub(r"^[^A-Z0-9]+", "", code)
    code = re.sub(r"[^A-Z0-9.]+$", "", code)
    return code


def _looks_like_icd_code(value: str) -> bool:
    return bool(value) and any(char.isdigit() for char in value) and all(
        char.isalnum() or char == "." for char in value
    )


def _dedupe_codes(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        normalized = _normalize_code(value)
        if not _looks_like_icd_code(normalized) or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _coerce_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("labels", "gold_labels", "codes", "target"):
            if key in value:
                return _coerce_codes(value[key])
        return []
    if isinstance(value, (list, tuple, set)):
        return _dedupe_codes([str(item).strip() for item in value if str(item).strip()])
    if not isinstance(value, str):
        return _dedupe_codes([str(value).strip()])

    stripped = value.strip()
    if not stripped:
        return []
    if stripped.upper() == "NONE":
        return []

    if stripped[0] in "[{(":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            return _coerce_codes(parsed)

    parts = re.split(r"[;,\n]+", stripped)
    if len(parts) == 1 and " " in stripped:
        parts = re.split(r"\s+", stripped)

    tokens = []
    for part in parts:
        for token in re.findall(r"[A-Za-z0-9.]+", part):
            if token:
                tokens.append(token)
    return _dedupe_codes(tokens)


def _extract_answer_segment(response: str) -> str:
    matches = re.findall(r"(?is)answer\s*:\s*(.+)", response)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_predicted_codes(response: str) -> list[str]:
    return _coerce_codes(_extract_answer_segment(response))


def extract_ground_truth_codes(ground_truth: str) -> list[str]:
    return _coerce_codes(ground_truth)


def compute_f1(predicted_codes: list[str], gold_codes: list[str]) -> dict[str, float]:
    predicted = set(predicted_codes)
    gold = set(gold_codes)
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold) if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        predicted_codes = extract_predicted_codes(response)
        gold_codes = extract_ground_truth_codes(reward_input["ground_truth"])
        metrics = compute_f1(predicted_codes, gold_codes)
        format_score = 1.0 if re.search(r"(?im)^answer\s*:", response) else 0.0
        scores.append(
            {
                "overall": metrics["f1"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "format": format_score,
                "predicted_count": float(len(predicted_codes)),
                "gold_count": float(len(gold_codes)),
            }
        )

    return scores
