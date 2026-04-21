# Multi-Agent for ICD Prediction

This repository now ships a multi-agent ICD coding pipeline built around a local Qwen model:

- `agent1`: turns a raw clinical note into a compact coding-ready case summary
- `agent2`: converts that summary into grounded ICD-10-CM / ICD-10-PCS code candidates
- `agent3`: during training, reviews `gold answer + pred + case evidence`, distills reusable lessons, and writes them into a SQLite knowledge base for future Agent 2 retrieval

The default inference pipeline is:

```text
agent1 -> agent2
```

The optional training-time memory update pipeline is:

```text
agent1 -> agent2 -> agent3
```

## Recommended local Qwen model

As of April 11, 2026, the best practical Qwen choice for a single A100 in this text-only workflow is:

- `Qwen/Qwen3-30B-A3B-Instruct-2507`

In this repo, the default is the FP8 variant for safer single-GPU usage:

- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`

Why this default:

- it is a newer Qwen instruct release suitable for structured text tasks
- it is realistic on a single A100 even when the card is 40 GB
- it keeps output formatting much more stable than a thinking-mode setup for JSON-heavy agents

If your A100 is 80 GB and you want the highest quality, set:

```bash
set QWEN_MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507
```

## Project structure

```text
multi_agent_icd/
  agents/
    agent1/
      agent.py
      prompt.py
      schema.py
    agent2/
      agent.py
      prompt.py
      schema.py
    agent3/
      agent.py
      prompt.py
      schema.py
  knowledge_base.py
  providers/
    local_qwen.py
  utils/
    clinical_text.py
    schema.py
  cli/
    run_agent1.py
    run_testset.py
  run.py
tests/
  test_pipeline.py
  test_mimic_dataset.py
```

## Agent 1 output

`agent1` does not predict codes. It produces a cleaned structured summary like:

```json
{
  "gender": "male",
  "chief_complaint": "l2 fracture, back pain",
  "procedure": [
    "l2 corpectomy retroperitoneal approach",
    "revision of posterior l1-l3 fusion"
  ],
  "history_present_illness": "patient sustained an l2 fracture after trauma and continued to have persistent back pain despite treatment.",
  "past_medical_history": [
    "mitral valve prolapse",
    "headaches",
    "gerd"
  ],
  "physical_exam_discharge": [
    "afebrile",
    "vital signs stable",
    "back incision clean dry intact",
    "strength and sensation intact"
  ],
  "pertinent_results": [
    "abdominal x-ray large bowel dilation consistent with ileus",
    "ultrasound negative for dvt",
    "cta chest negative for pulmonary embolism"
  ],
  "hospital_course": [
    "postoperative uncontrolled back pain requiring medication adjustment",
    "large bowel ileus improved with bowel regimen",
    "tachycardia workup negative for dvt and pe"
  ],
  "discharge_diagnosis": [
    "l2 fracture",
    "back pain"
  ]
}
```

## Agent 2 output

`agent2` consumes the structured summary plus note evidence and returns grounded coding candidates:

```json
{
  "principal_diagnosis": {
    "code": "J18.9",
    "description": "pneumonia, unspecified organism",
    "code_system": "ICD-10-CM",
    "category": "principal_diagnosis",
    "confidence": "high",
    "rationale": "Discharge diagnosis states pneumonia.",
    "evidence_ids": ["E2"],
    "missing_details": []
  },
  "secondary_diagnoses": [],
  "procedures": [],
  "coding_queries": [],
  "coding_summary": "Pneumonia is the principal coded condition."
}
```

## Agent 3 output

`agent3` is intended for training or hindsight review. It compares the case with gold labels and Agent 2 predictions, then writes reusable coding knowledge into a SQLite database:

```json
{
  "case_summary": "Pneumonia discharge diagnosis supported by cough and fever.",
  "salient_clinical_patterns": [
    "explicit discharge diagnosis of pneumonia",
    "infectious respiratory symptoms"
  ],
  "correct_prediction_reasons": [
    "The principal diagnosis was correctly supported by the discharge diagnosis."
  ],
  "missed_code_lessons": [
    "Review chronic comorbidities in the summary because hypertension can be easy to miss."
  ],
  "unsupported_prediction_lessons": [],
  "coding_lessons": [
    "When the note names pneumonia at discharge, prioritize it as the principal condition if no competing reason for admission is documented."
  ],
  "retrieval_queries": [
    "pneumonia discharge diagnosis cough fever"
  ],
  "knowledge_summary": "Cases with pneumonia at discharge and infectious respiratory symptoms should keep pneumonia front-of-mind during code selection."
}
```

Agent 2 can later retrieve the most relevant memory entries from that knowledge base and inject them into its coding prompt as advisory knowledge.

## Quickstart

Install the local-model dependencies from `pyproject.toml`, then run:

```bash
python -m multi_agent_icd.cli.run_agent1 path/to/note.txt
```

Optional patient context:

```bash
python -m multi_agent_icd.cli.run_agent1 path/to/note.txt --context path/to/patient-context.json
```

Override the model:

```bash
python -m multi_agent_icd.cli.run_agent1 path/to/note.txt --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

## Run directly on your dataset

The repository now supports the MIMIC-style CSV format you added:

```text
subject_id,hadm_id,text,labels,length
14815480,22040094,<note text>,Z515;K7200;C3490,76
```

You can run the full multi-agent pipeline directly on a split like this:

```bash
python -m multi_agent_icd.cli.run_testset --dataset-dir data/mimic4_icd10 --split test --limit 10
```

Or point at one explicit file:

```bash
python -m multi_agent_icd.cli.run_testset --csv-path data/mimic4_icd10/test_full.csv --limit 10
```

To build a knowledge base from the training split:

```bash
python -m multi_agent_icd.cli.run_testset \
  --dataset-dir data/mimic4_icd10 \
  --split train \
  --knowledge-base-path outputs/mimic4_icd10/train_knowledge.sqlite3 \
  --update-knowledge-base
```

To reuse that memory during evaluation or inference:

```bash
python -m multi_agent_icd.cli.run_testset \
  --dataset-dir data/mimic4_icd10 \
  --split test \
  --knowledge-base-path outputs/mimic4_icd10/train_knowledge.sqlite3 \
  --knowledge-base-top-k 3
```

When `TOP_50_CODES.csv` exists next to the selected split, the testset runner automatically uses it as the allowed ICD candidate set for Agent 2. You can also pass it explicitly and control how many code candidates are emitted per note:

`TOP_50_CODES.csv` may contain either one code per line or a richer two-column format:

```csv
code,description
274.9,"Gout, unspecified"
585.6,End stage renal disease
584.9,"Acute kidney failure, unspecified"
```

The two-column format is recommended because Agent 2 uses the descriptions as the source of truth for each code's clinical meaning.

```bash
python -m multi_agent_icd.cli.run_testset \
  --csv-path multi_agent_icd/datasets/mimic4_icd9/test_full.csv \
  --top-codes-path multi_agent_icd/datasets/mimic4_icd9/TOP_50_CODES.csv \
  --hadm-ids-path multi_agent_icd/datasets/mimic4_icd9/test_50_hadm_ids.csv \
  --num-candidates 5
```

The command prints a summary JSON and writes per-case predictions to:

```text
outputs/<dataset_name>/<split_name>_predictions.jsonl
```

Each JSONL row includes:

- `subject_id`
- `hadm_id`
- `gold_labels`
- `top_code_gold_labels`
- `predicted_codes` (an ordered list of ICD code candidates; with a top-code file, the runner fills this to `--num-candidates`, default 5)
- `candidate_output_limit`
- `top_codes_path`
- `agent1_output`
- `agent2_output`
- `agent3_output` when `--update-knowledge-base` is enabled
- `execution_trace`

The summary JSON includes `precision_at_5` by default, computed as the mean per-case `hits_in_top_5 / 5`. When using a top-code candidate file, it also reports top-code label-space metrics such as `covered_examples`, `case_coverage`, `label_coverage`, `precision_at_5_covered`, and `recall_at_5_top_codes`.

When a knowledge base is configured, the summary also includes:

- `knowledge_base_path`
- `knowledge_entries_written`

For the bundled ICD-9 top-50 task, use `test_50_hadm_ids.csv` to evaluate only admissions known to contain at least one top-50 gold label.

## Controller example

```python
from multi_agent_icd import MultiAgentController

controller = MultiAgentController(
    agent_models={
        "agent1": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "agent2": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "agent3": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    }
)

state = controller.run(note_text="...", patient_context={})

print(state["agent_outputs"]["agent1"])
print(state["agent_outputs"]["agent2"])
print(state["execution_trace"])
```

Training-time memory writing example:

```python
from multi_agent_icd import MultiAgentController

controller = MultiAgentController(
    knowledge_base_path="outputs/train_knowledge.sqlite3",
    agent_models={
        "agent1": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "agent2": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "agent3": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    },
)

state = controller.run(
    note_text="...",
    patient_context={"coding_version": "ICD-10"},
    training_context={"gold_labels": ["J18.9", "I10"]},
    requested_agents=["agent1", "agent2", "agent3"],
)

print(state["agent_outputs"]["agent3"])
```

## Custom LLM contract

Both agents can also accept a custom `llm` object as long as it exposes:

```python
class ExampleLLM:
    def generate_json(self, system_prompt: str, user_prompt: str, metadata: dict):
        return {
            "principal_diagnosis": None,
            "secondary_diagnoses": [],
            "procedures": [],
            "coding_queries": [],
            "coding_summary": ""
        }
```

## Next steps

- add batching / resume support for very large test runs
- add de-identified regression notes for Agent 2
- plug Agent 2 into a local ICD validation dictionary if stricter code checking is needed
