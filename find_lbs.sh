#!/bin/bash
# 第一行叫 shebang，告诉系统用 bash 来运行这个脚本

# 遇到错误即刻停止脚本，防止上一个没跑成功，下一个接着瞎跑
set -e 

# 定义一些固定的变量
ORIGINAL_LEN=100
MAX_ITERS=500
DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
PCTS=(1.1 1.2 1.5 2 4 8)


echo "🚀 开始自动化批量实验...LB"

# 开始循环遍历数组中的每一个值
for pct in "${PCTS[@]}"; do
    echo "=================================================="
    echo "Starting Evaluating LB: original=${ORIGINAL_LEN}, rope_base=10000, pct=${pct}"
    echo "=================================================="
    
    # 1. 运行训练脚本 (调用你刚刚写好的 argparse)
    # python train.py \
    #     --original ${ORIGINAL_LEN} \
    #     --rope_base ${rb} \
    #     --max_iters ${MAX_ITERS}
    
    # 2. 训练完紧接着跑评估脚本 (开启保存和准确率计算)
    
    python eval.py \
        --data_dir ${DATA_PATH} \
        --working_dir ${SAVE_PATH} \
        --result_dir ${RESULT_PATH} \
        --original ${ORIGINAL_LEN} \
        --n_embd 384 \
        --n_head 2 \
        --n_layer 4 \
        --dropout 0.0 \
        --rope_base 10000 \
        --batch_size 100 \
        --block_size 8192 \
        --pct ${pct} \
        --is_save \
        --need_acc
        
    echo "LB found"
    echo " "
    
done
