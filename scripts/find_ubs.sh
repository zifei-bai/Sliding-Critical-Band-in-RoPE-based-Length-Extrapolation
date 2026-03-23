#!/bin/bash

set -e 


ORIGINAL_LEN=(50 100 500)

DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
PCTS=(1.1 1.2 1.5 2 4 8)


echo "Find UB"
for ori_len in "${ORIGINAL_LEN[@]}"; do

   
    for pct in "${PCTS[@]}"; do
        echo "=================================================="
        echo "Starting Evaluating: original=${ori_len}, rope_base=10000, pct=${pct}"
        echo "=================================================="
        
       
        python eval.py \
            --data_dir ${DATA_PATH} \
            --working_dir ${SAVE_PATH} \
            --result_dir ${RESULT_PATH} \
            --original ${ori_len} \
            --n_embd 384 \
            --n_head 2 \
            --n_layer 4 \
            --dropout 0.0 \
            --rope_base 10000 \
            --batch_size 100 \
            --block_size 8192 \
            --is_from \
            --from_where 0 \
            --pct ${pct} \
            --is_save \
            --need_acc
            
        echo "Original_len=${ori_len}, rb=10000, pct=${pct} finished. "
        echo " "
        
    done
done

for pct in "${PCTS[@]}"; do
        echo "=================================================="
        echo "Starting Evaluating: original=${ori_len}, rope_base=10000, pct=${pct}"
        echo "=================================================="
        
    
        
        python eval.py \
            --data_dir ${DATA_PATH} \
            --working_dir ${SAVE_PATH} \
            --result_dir ${RESULT_PATH} \
            --original 500 \
            --n_embd 384 \
            --n_head 2 \
            --n_layer 4 \
            --dropout 0.0 \
            --rope_base 100000 \
            --batch_size 100 \
            --block_size 8192 \
            --is_from \
            --from_where 0 \
            --pct ${pct} \
            --is_save \
            --need_acc
            
        echo "Original_len=500, , rb=100000, pct=${pct} finished. "
        echo " "
        
    done

echo "UB found"