#!/bin/bash
set -e


DATA_PATH="./data/"
WORKING_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"



echo "========================================="
echo "  Step 1: 运行模型，提取 Attention 矩阵  "
echo "========================================="
python get_attn.py \
    --working_dir "${WORKING_PATH}" \
    --data_dir "${DATA_PATH}" \
    --result_dir "${RESULT_PATH}" \
    --original 50 \
    --pct 1.5 \
    --rope_base 10000 

echo "========================================="
echo "  Step 2: 读取数据，绘制 PDF 矢量图      "
echo "========================================="
python draw_attn.py \
    --result_dir "${RESULT_PATH}" \
    --graph_dir "${GRAPH_PATH}" \
    --original 50 \
    --pct 1.5

echo "🎉 全部流程执行完毕！"