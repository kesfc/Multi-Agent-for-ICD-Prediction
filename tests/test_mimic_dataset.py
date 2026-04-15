from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_icd.datasets import (
    infer_coding_version_from_path,
    load_code_candidate_records,
    load_code_candidates,
    load_hadm_ids,
    load_mimic_examples,
    parse_label_string,
)
from multi_agent_icd.testset import extract_predicted_codes, run_testset


class StubController:
    def run(self, note_text, patient_context=None, requested_agents=None):
        del note_text, requested_agents
        coding_version = (patient_context or {}).get("coding_version", "ICD-10")
        code_system = "ICD-9-CM" if "9" in str(coding_version) else "ICD-10-CM"
        return {
            "agent_outputs": {
                "agent1": {"chief_complaint": "demo"},
                "agent2": {
                    "principal_diagnosis": {
                        "code": "J18.9",
                        "description": "pneumonia, unspecified organism",
                        "code_system": code_system,
                        "category": "principal_diagnosis",
                        "confidence": "high",
                        "rationale": "Supported by the note.",
                        "evidence_ids": ["E1"],
                        "missing_details": [],
                    },
                    "secondary_diagnoses": [
                        {
                            "code": "I10",
                            "description": "essential hypertension",
                            "code_system": code_system,
                            "category": "secondary_diagnosis",
                            "confidence": "medium",
                            "rationale": "Supported by history.",
                            "evidence_ids": ["E2"],
                            "missing_details": [],
                        }
                    ],
                    "procedures": [],
                    "coding_queries": [],
                    "coding_summary": "Demo output.",
                },
            },
            "execution_trace": [],
        }


class FailingSecondController(StubController):
    def __init__(self):
        self.calls = 0

    def run(self, note_text, patient_context=None, requested_agents=None):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic failure")
        return super().run(note_text, patient_context=patient_context, requested_agents=requested_agents)


class MIMICDatasetTests(unittest.TestCase):
    def test_parse_label_string_deduplicates(self):
        self.assertEqual(parse_label_string("J18.9;I10;J18.9;;"), ["J18.9", "I10"])

    def test_infer_coding_version_from_path(self):
        self.assertEqual(infer_coding_version_from_path("data/mimic4_icd10/test_full.csv"), "ICD-10")
        self.assertEqual(infer_coding_version_from_path("data/mimic4_icd9/test_full.csv"), "ICD-9")

    def test_load_mimic_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_full.csv"
            csv_path.write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "1,2,example note,J18.9;I10,42\n",
                encoding="utf-8",
            )

            examples = load_mimic_examples(csv_path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].subject_id, "1")
        self.assertEqual(examples[0].labels, ["J18.9", "I10"])
        self.assertEqual(examples[0].length, 42)

    def test_load_code_candidates_reads_one_code_per_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "TOP_50_CODES.csv"
            csv_path.write_text("code\n486\n401.9\n486\n", encoding="utf-8")

            codes = load_code_candidates(csv_path)

        self.assertEqual(codes, ["486", "401.9"])

    def test_load_code_candidate_records_reads_descriptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "TOP_50_CODES.csv"
            csv_path.write_text(
                "code,description\n"
                "274.9,\"Gout, unspecified\"\n"
                "585.6,End stage renal disease\n",
                encoding="utf-8",
            )

            records = load_code_candidate_records(csv_path)
            codes = load_code_candidates(csv_path)

        self.assertEqual(
            records,
            [
                {"code": "274.9", "description": "Gout, unspecified"},
                {"code": "585.6", "description": "End stage renal disease"},
            ],
        )
        self.assertEqual(codes, ["274.9", "585.6"])

    def test_load_hadm_ids_reads_one_column_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_50_hadm_ids.csv"
            csv_path.write_text("hadm_id\n123\n456\n123\n", encoding="utf-8")

            hadm_ids = load_hadm_ids(csv_path)

        self.assertEqual(hadm_ids, {"123", "456"})

    def test_extract_predicted_codes(self):
        predicted = extract_predicted_codes(
            {
                "principal_diagnosis": {"code": "J18.9"},
                "secondary_diagnoses": [{"code": "I10"}, {"code": "J18.9"}],
                "procedures": [{"code": "0BH17EZ"}],
            }
        )

        self.assertEqual(predicted, ["J18.9", "I10", "0BH17EZ"])

    def test_extract_predicted_codes_filters_and_limits(self):
        predicted = extract_predicted_codes(
            {
                "principal_diagnosis": {"code": "J18.9"},
                "secondary_diagnoses": [{"code": "I10"}, {"code": "E11.9"}],
                "procedures": [{"code": "0BH17EZ"}],
            },
            candidate_output_limit=2,
            allowed_codes=["J18.9", "E11.9", "0BH17EZ"],
        )

        self.assertEqual(predicted, ["J18.9", "E11.9"])

    def test_extract_predicted_codes_fills_to_limit_from_allowed_codes(self):
        predicted = extract_predicted_codes(
            {
                "principal_diagnosis": {"code": "J18.9"},
                "secondary_diagnoses": [{"code": "I10"}],
                "procedures": [],
            },
            candidate_output_limit=5,
            allowed_codes=["J18.9", "I10", "E11.9", "N17.9", "I50.9"],
            fill_to_limit=True,
        )

        self.assertEqual(predicted, ["J18.9", "I10", "E11.9", "N17.9", "I50.9"])

    def test_run_testset_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_dir = base / "mimic4_icd10"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dataset_dir / "test_full.csv"
            output_path = base / "predictions.jsonl"
            csv_path.write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "1,2,example note one,J18.9;I10,42\n"
                "3,4,example note two,J18.9,11\n",
                encoding="utf-8",
            )

            summary = run_testset(
                csv_path=csv_path,
                controller=StubController(),
                output_path=output_path,
            )

            written_lines = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(summary["coding_version"], "ICD-10")
        self.assertEqual(summary["num_examples"], 2)
        self.assertEqual(summary["true_positives"], 3)
        self.assertEqual(summary["total_predicted_codes"], 4)
        self.assertEqual(len(written_lines), 2)

    def test_run_testset_uses_top_codes_and_candidate_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_dir = base / "mimic4_icd10"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dataset_dir / "test_full.csv"
            top_codes_path = dataset_dir / "TOP_50_CODES.csv"
            output_path = base / "predictions.jsonl"
            csv_path.write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "1,2,example note one,J18.9;I10,42\n",
                encoding="utf-8",
            )
            top_codes_path.write_text("J18.9\nI10\nE11.9\nN17.9\nI50.9\n", encoding="utf-8")

            summary = run_testset(
                csv_path=csv_path,
                controller=StubController(),
                output_path=output_path,
                top_codes_path=top_codes_path,
                candidate_output_limit=5,
            )
            written = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(summary["candidate_output_limit"], 5)
        self.assertEqual(summary["candidate_code_count"], 5)
        self.assertEqual(summary["total_predicted_codes"], 5)
        self.assertEqual(summary["precision_at_5"], 0.4)
        self.assertEqual(written["predicted_codes"], ["J18.9", "I10", "E11.9", "N17.9", "I50.9"])
        self.assertEqual(written["candidate_output_limit"], 5)

    def test_run_testset_filters_by_hadm_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_dir = base / "mimic4_icd10"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dataset_dir / "test_full.csv"
            top_codes_path = dataset_dir / "TOP_50_CODES.csv"
            hadm_ids_path = dataset_dir / "test_50_hadm_ids.csv"
            output_path = base / "predictions.jsonl"
            csv_path.write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "1,2,example note one,J18.9;I10,42\n"
                "3,4,example note two,E11.9,11\n",
                encoding="utf-8",
            )
            top_codes_path.write_text("code,description\nJ18.9,Pneumonia\nI10,Hypertension\n", encoding="utf-8")
            hadm_ids_path.write_text("2\n", encoding="utf-8")

            summary = run_testset(
                csv_path=csv_path,
                controller=StubController(),
                output_path=output_path,
                top_codes_path=top_codes_path,
                hadm_ids_path=hadm_ids_path,
                candidate_output_limit=5,
            )
            written_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            written = json.loads(written_lines[0])

        self.assertEqual(summary["num_examples"], 1)
        self.assertEqual(summary["hadm_id_filter_count"], 1)
        self.assertEqual(summary["covered_examples"], 1)
        self.assertEqual(summary["top_code_gold_codes"], 2)
        self.assertEqual(summary["precision_at_5_covered"], 0.4)
        self.assertEqual(summary["recall_at_5_top_codes"], 1.0)
        self.assertEqual(len(written_lines), 1)
        self.assertEqual(written["hadm_id"], "2")
        self.assertEqual(written["top_code_gold_labels"], ["J18.9", "I10"])

    def test_run_testset_continues_after_example_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_dir = base / "mimic4_icd10"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dataset_dir / "test_full.csv"
            top_codes_path = dataset_dir / "TOP_50_CODES.csv"
            output_path = base / "predictions.jsonl"
            csv_path.write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "1,2,example note one,J18.9;I10,42\n"
                "3,4,example note two,J18.9,11\n"
                "5,6,example note three,I10,9\n",
                encoding="utf-8",
            )
            top_codes_path.write_text("J18.9\nI10\nE11.9\nN17.9\nI50.9\n", encoding="utf-8")

            summary = run_testset(
                csv_path=csv_path,
                controller=FailingSecondController(),
                output_path=output_path,
                top_codes_path=top_codes_path,
                candidate_output_limit=5,
            )
            written = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").strip().splitlines()
            ]

        self.assertEqual(summary["num_examples"], 3)
        self.assertEqual(summary["failed_examples"], 1)
        self.assertEqual(len(written), 3)
        self.assertEqual(written[1]["status"], "failed")
        self.assertEqual(written[1]["predicted_codes"], [])
        self.assertIn("synthetic failure", written[1]["error"])


if __name__ == "__main__":
    unittest.main()
