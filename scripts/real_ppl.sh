#!/bin/bash

set -e



DATA_DIR="./data/"
RESULT_DIR="./results/"
SEQ_LEN=(4096)
TRAINING_LEN=2048

echo "================================================="
echo "  Finding LlaMA's Sliding Critical Band"
echo "================================================="


for seq in "${SEQ_LEN[@]}"; do
    TARGET_SCALE=$(awk "BEGIN {print $seq/$TRAINING_LEN}")
    echo "target scale: ${TARGET_SCALE}"

    python src/real_llama.py \
        --model_id "huggyllama/llama-7b" \
        --data_dir "${DATA_DIR}" \
        --result_dir "${RESULT_DIR}" \
        --seq_len ${seq} \
        --num_samples 48 \
        --batch_size 4 \
        --target_scale ${TARGET_SCALE}

    echo "Length ${seq} finished."
    echo "-----------------------------------------"
done

echo "================================================="
echo "  Raw result data ${RESULT_DIR}"
echo "================================================="
            