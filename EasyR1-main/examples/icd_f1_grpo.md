# ICD F1 GRPO Recipe

1. Export the ICD dataset into EasyR1 JSONL files:

```bash
python -m multi_agent_icd.cli.prepare_easyr1_dataset \
  --dataset-dir multi_agent_icd/datasets/mimiciv_icd10 \
  --output-dir outputs/easyr1/icd_rl
```

2. Start EasyR1 training from the EasyR1 folder:

```bash
cd EasyR1-main
bash examples/qwen2_5_1_5b_icd_f1_grpo.sh
```

Notes:

- The reward is example-level ICD F1. `overall = f1`.
- The model is prompted to end with `Answer: CODE1; CODE2; ...`.
- If a `TOP_50_CODES.csv` file exists next to the source split, the exported prompt will include the allowed candidate set.
- The current `mimiciii_50` directory in this repo only contains split metadata. To actually train on it, add the matching base note table feather file with note text and labels.
