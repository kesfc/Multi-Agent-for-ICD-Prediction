#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${ROOT_DIR}/outputs/easyr1/icd_rl/train.jsonl}
VAL_FILE=${VAL_FILE:-${ROOT_DIR}/outputs/easyr1/icd_rl/val.jsonl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_1_5b_icd_f1_grpo}

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python3 -m verl.trainer.main \
    config=examples/icd_f1_grpo.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    trainer.experiment_name="${EXPERIMENT_NAME}"
