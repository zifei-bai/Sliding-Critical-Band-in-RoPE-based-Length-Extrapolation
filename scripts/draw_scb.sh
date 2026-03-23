#!/bin/bash

set -e 

DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"

    
python draw_SCB.py \
    --result_dir ${RESULT_PATH} \
    --graph_dir ${GRAPH_PATH} 
    
    
echo "SCB Figure saved"