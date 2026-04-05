from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_agent_icd import run_agent1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agent 1 on a clinical note.")
    parser.add_argument("note_path", help="Path to a UTF-8 text file containing the clinical note.")
    parser.add_argument(
        "--model",
        required=True,
        dest="model_name",
        help="Required OpenAI model name, for example gpt-5 or gpt-4.1.",
    )
    parser.add_argument(
        "--context",
        dest="context_path",
        help="Optional path to patient context JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    note_text = Path(args.note_path).read_text(encoding="utf-8")
    patient_context = (
        json.loads(Path(args.context_path).read_text(encoding="utf-8"))
        if args.context_path
        else {}
    )

    result = run_agent1(
        note_text=note_text,
        patient_context=patient_context,
        model_name=args.model_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
