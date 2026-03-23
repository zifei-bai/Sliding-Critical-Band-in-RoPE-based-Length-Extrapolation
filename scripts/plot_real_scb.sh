#!/bin/bash

set -e


RESULT_DIR="./results/"
GRAPH_DIR="./graphs/"

echo "================================================="
echo "  Visualizing LLaMA Sliding Critical Band"
echo "  Data path: ${RESULT_DIR}"
echo "================================================="


python src/plot_real_scb.py \
    --result_dir "${RESULT_DIR}" \
    --graph_dir "${GRAPH_DIR}"

echo "Graph saved to ${GRAPH_DIR}"