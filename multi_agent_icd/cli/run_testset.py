from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_agent_icd import DEFAULT_QWEN_MODEL_NAME, MultiAgentController
from multi_agent_icd.datasets import resolve_mimic_split_path
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
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
