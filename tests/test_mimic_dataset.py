from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multi_agent_icd.datasets import (
    infer_coding_version_from_path,
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

    def test_extract_predicted_codes(self):
        predicted = extract_predicted_codes(
            {
                "principal_diagnosis": {"code": "J18.9"},
                "secondary_diagnoses": [{"code": "I10"}, {"code": "J18.9"}],
                "procedures": [{"code": "0BH17EZ"}],
            }
        )

        self.assertEqual(predicted, ["J18.9", "I10", "0BH17EZ"])

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


if __name__ == "__main__":
    unittest.main()
