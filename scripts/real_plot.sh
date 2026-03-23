#!/bin/bash

set -e

RESULT_DIR="./results/"
GRAPH_DIR="./graphs/"

echo "================================================="
echo "  Visualizing LLaMA Real Model UB/LB graph on dimension"
echo "  Data path: ${RESULT_DIR}"
echo "================================================="


python src/real_plot.py \
    --result_dir "${RESULT_DIR}" \
    --graph_dir "${GRAPH_DIR}"

echo "Graph saved to ${GRAPH_DIR}"