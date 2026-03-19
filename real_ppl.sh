#!/bin/bash
# 第一行叫 shebang，告诉系统用 bash 来运行这个脚本

# 遇到错误即刻停止脚本，防止上一个没跑成功，下一个接着瞎跑
set -e 

# 定义一些固定的变量
DATA_PATH="./data/"
RESULT_PATH="./results/"


echo "🚀 开始自动化批量实验...prepare C4 data"


        
python real_llama.py \
    --data_dir ${DATA_PATH} \
    --result_dir ${RESULT_PATH} 
            