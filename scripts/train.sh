#!/bin/bash
set -e


DATA_DIR="./data/"
WORKING_DIR="./models/"


echo "================================================="
echo "  Pretraining Model"
echo "================================================="


echo "-------------------------------------------------"
echo "   Training Original Length = 100"
echo "-------------------------------------------------"

# 调用 Python 脚本
python src/train.py \
    --data_dir "${DATA_DIR}" \
    --working_dir "${WORKING_DIR}" \
    --original 100 \
    --rope_base 10000 \
    --max_iters 9000 \
    --batch_size 1000 \
    --block_size 256 \
    --lr 5e-4
    
echo "Pretraining Done"
