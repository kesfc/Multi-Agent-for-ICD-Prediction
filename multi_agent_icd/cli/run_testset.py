from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_agent_icd import DEFAULT_QWEN_MODEL_NAME, MultiAgentController
from multi_agent_icd.datasets import resolve_mimic_split_path, resolve_top_codes_path
from multi_agent_icd.testset import run_testset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-agent ICD pipeline on a MIMIC-style test split.")
    parser.add_argument(
        "--dataset-dir",
        default="data/mimic4_icd10",
        help="Directory containing train_full.csv, dev_full.csv, and test_full.csv.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Which split file to run.",
    )
    parser.add_argument(
        "--csv-path",
        help="Optional explicit CSV path. Overrides --dataset-dir and --split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of examples to run.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many rows before starting.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=DEFAULT_QWEN_MODEL_NAME,
        help="Local Qwen model id or local path for both agents.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional JSONL file for per-example predictions.",
    )
    parser.add_argument(
        "--top-codes-path",
        help=(
            "Optional CSV containing allowed ICD code candidates. "
            "Defaults to TOP_50_CODES.csv next to the selected split when present."
        ),
    )
    parser.add_argument(
        "--hadm-ids-path",
        help=(
            "Optional one-column CSV of HADM IDs to include. "
            "Use this to run/evaluate only admissions containing top-code labels."
        ),
    )
    parser.add_argument(
        "--num-candidates",
        "--candidate-output-limit",
        dest="candidate_output_limit",
        type=int,
        default=5,
        help=(
            "Number of ranked ICD code candidates to output per example. "
            "Defaults to 5 for P@5 evaluation. "
            "--candidate-output-limit is kept as a backwards-compatible alias."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else resolve_mimic_split_path(args.dataset_dir, args.split)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path("outputs") / csv_path.parent.name / f"{csv_path.stem}_predictions.jsonl"
    )
    top_codes_path = Path(args.top_codes_path) if args.top_codes_path else resolve_top_codes_path(csv_path)

    controller = MultiAgentController(
        agent_models={
            "agent1": args.model_name,
            "agent2": args.model_name,
        }
    )
    summary = run_testset(
        csv_path=csv_path,
        controller=controller,
        limit=args.limit,
        offset=args.offset,
        output_path=output_path,
        top_codes_path=top_codes_path,
        hadm_ids_path=Path(args.hadm_ids_path) if args.hadm_ids_path else None,
        candidate_output_limit=args.candidate_output_limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
