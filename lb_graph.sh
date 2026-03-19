#!/bin/bash
# 第一行叫 shebang，告诉系统用 bash 来运行这个脚本

# 遇到错误即刻停止脚本，防止上一个没跑成功，下一个接着瞎跑
set -e 

# 定义一些固定的变量
ORIGINAL_LEN=100
DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"

echo "开始画图..."

    
python draw_graphs.py \
    --result_dir ${RESULT_PATH} \
    --graph_dir ${GRAPH_PATH} \
    --original 100 \
    --rope_base 10000 \
    --d_extra -1 \
    --zoom_widths "{1.1:21, 1.2:17, 1.5:18, 2.0:17, 4.0:10, 8.0:6}" \
    --ublb "lb"
    
    
echo "Figure saved"