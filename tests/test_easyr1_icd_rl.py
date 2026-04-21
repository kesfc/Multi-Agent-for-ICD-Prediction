from __future__ import annotations

import importlib.util
import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from multi_agent_icd.rl import build_easyr1_icd_prompt, prepare_easyr1_icd_dataset


@contextmanager
def workspace_tempdir():
    base = Path.cwd() / ".tmp_test"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _load_reward_module():
    reward_path = (
        Path.cwd()
        / "EasyR1-main"
        / "examples"
        / "reward_function"
        / "icd_f1.py"
    )
    spec = importlib.util.spec_from_file_location("icd_f1_reward", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EasyR1ICDTests(unittest.TestCase):
    def test_build_prompt_includes_answer_contract(self):
        prompt = build_easyr1_icd_prompt(
            note_text="Discharge diagnosis: pneumonia",
            coding_version="ICD-10",
            candidate_code_records=[{"code": "J18.9", "description": "Pneumonia, unspecified"}],
            note_id="123",
        )

        self.assertIn("Answer: CODE1; CODE2; CODE3", prompt)
        self.assertIn("Only use codes from the allowed candidate set below", prompt)
        self.assertIn("J18.9: Pneumonia, unspecified", prompt)
        self.assertIn("Case id: 123.", prompt)

    def test_prepare_easyr1_dataset_writes_jsonl_files(self):
        with workspace_tempdir() as tmpdir:
            dataset_dir = tmpdir / "mimic4_icd10"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "train_full.csv").write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "1,10,train note,J18.9;I10,42\n",
                encoding="utf-8",
            )
            (dataset_dir / "dev_full.csv").write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "2,20,dev note,E11.9,12\n",
                encoding="utf-8",
            )
            (dataset_dir / "test_full.csv").write_text(
                "subject_id,hadm_id,text,labels,length\n"
                "3,30,test note,N17.9,9\n",
                encoding="utf-8",
            )
            (dataset_dir / "TOP_50_CODES.csv").write_text(
                "code,description\nJ18.9,Pneumonia\nI10,Hypertension\n",
                encoding="utf-8",
            )

            output_dir = tmpdir / "easyr1"
            summary = prepare_easyr1_icd_dataset(
                dataset_path=dataset_dir,
                output_dir=output_dir,
            )

            train_record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").strip())
            val_record = json.loads((output_dir / "val.jsonl").read_text(encoding="utf-8").strip())

        self.assertEqual(summary["splits"][0]["num_examples"], 1)
        self.assertEqual(train_record["hadm_id"], "10")
        self.assertEqual(train_record["labels"], ["J18.9", "I10"])
        self.assertIn("Only use codes from the allowed candidate set below", train_record["prompt"])
        self.assertEqual(json.loads(train_record["answer"])["labels"], ["J18.9", "I10"])
        self.assertEqual(val_record["labels"], ["E11.9"])

    def test_icd_f1_reward_scores_predictions(self):
        reward = _load_reward_module()
        scores = reward.compute_score(
            [
                {
                    "response": "<think>reason</think>\nAnswer: J18.9; I10; E11.9",
                    "response_length": 12,
                    "ground_truth": json.dumps({"labels": ["J18.9", "I10"]}),
                }
            ]
        )

        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0]["precision"], 2 / 3)
        self.assertAlmostEqual(scores[0]["recall"], 1.0)
        self.assertAlmostEqual(scores[0]["f1"], 0.8)
        self.assertAlmostEqual(scores[0]["overall"], 0.8)


if __name__ == "__main__":
    unittest.main()
