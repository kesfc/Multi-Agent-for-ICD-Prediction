from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_agent_icd.rl import prepare_easyr1_icd_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an EasyR1-compatible ICD RL dataset with prompt/answer JSONL files."
    )
    parser.add_argument(
        "--dataset-dir",
        default="multi_agent_icd/datasets/mimiciv_icd10",
        help="Dataset directory or split-aware dataset path supported by multi_agent_icd.datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/easyr1/icd_rl",
        help="Directory where train/val/test JSONL files will be written.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Dataset split name to use for the EasyR1 train file.",
    )
    parser.add_argument(
        "--val-split",
        default="dev",
        help="Dataset split name to use for the EasyR1 val file.",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Dataset split name to use for the EasyR1 test file.",
    )
    parser.add_argument(
        "--top-codes-path",
        help="Optional allowed-candidate CSV. When present, the prompt will include those code candidates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of examples to export per split.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many examples inside each split before exporting.",
    )
    parser.add_argument(
        "--max-note-chars",
        type=int,
        help="Optional character cap applied to each note before writing the EasyR1 prompt.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not export the test split.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    split_to_filename = {
        args.train_split: "train.jsonl",
        args.val_split: "val.jsonl",
    }
    if not args.skip_test:
        split_to_filename[args.test_split] = "test.jsonl"

    summary = prepare_easyr1_icd_dataset(
        dataset_path=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        split_to_filename=split_to_filename,
        limit=args.limit,
        offset=args.offset,
        top_codes_path=Path(args.top_codes_path) if args.top_codes_path else None,
        max_note_chars=args.max_note_chars,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
