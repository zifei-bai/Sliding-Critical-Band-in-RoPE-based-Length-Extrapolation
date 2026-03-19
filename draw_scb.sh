#!/bin/bash
# 第一行叫 shebang，告诉系统用 bash 来运行这个脚本

# 遇到错误即刻停止脚本，防止上一个没跑成功，下一个接着瞎跑
set -e 

# 定义一些固定的变量
DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"

echo "开始画图..."

    
python draw_SCB.py \
    --result_dir ${RESULT_PATH} \
    --graph_dir ${GRAPH_PATH} 
    
    
echo "SCB Figure saved"