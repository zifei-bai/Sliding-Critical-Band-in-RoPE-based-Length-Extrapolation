#!/bin/bash
set -e


DATA_PATH="./data/"
WORKING_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"



echo "========================================="
echo "  Extract Attention Matrix  "
echo "========================================="
python src/get_attn.py \
    --working_dir "${WORKING_PATH}" \
    --data_dir "${DATA_PATH}" \
    --result_dir "${RESULT_PATH}" \
    --original 50 \
    --pct 1.5 \
    --rope_base 10000 

echo "========================================="
echo "  Read data, draw attention map     "
echo "========================================="
python src/draw_attn.py \
    --result_dir "${RESULT_PATH}" \
    --graph_dir "${GRAPH_PATH}" \
    --original 50 \
    --pct 1.5

echo "Process Done. "