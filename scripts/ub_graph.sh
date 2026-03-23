#!/bin/bash
# 第一行叫 shebang，告诉系统用 bash 来运行这个脚本

# 遇到错误即刻停止脚本，防止上一个没跑成功，下一个接着瞎跑
set -e 

# 定义一些固定的变量
DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"
ORIGINAL_LEN=(50 100 500)

echo "开始画图..."

for ori in "${ORIGINAL_LEN[@]}"; do
    
    # 🌟 核心：根据当前的 ori 值，硬编码分配不同的参数
    case $ori in
        50)
            # 填入 50 对应的具体参数
            ZW="{1.1:15, 1.2:15, 1.5:15, 2.0:15, 4.0:15, 8.0:15}" 
            D_EXTRA=30
            ;;
        100)
            # 填入 100 对应的具体参数
            ZW="{1.1:21, 1.2:17, 1.5:18, 2.0:17, 4.0:10, 8.0:6}"
            D_EXTRA=37
            ;;
        500)
            # 填入 500 对应的具体参数
            ZW="{1.1:10, 1.2:10, 1.5:10, 2.0:10, 4.0:10, 8.0:10}"
            D_EXTRA=53
            ;;
        *)
            # 防御性编程：如果意外跑了其他长度，给个默认值并报错提示
            echo "⚠️ 未知的 original_len: $ori"
            continue
            ;;
    esac

    # 运行 Python 脚本，将刚刚分配好的变量传进去
    # 注意："${ZW}" 外面一定要加双引号，防止字典里的空格被终端截断
    python src/draw_graphs.py \
        --result_dir "${RESULT_PATH}" \
        --graph_dir "${GRAPH_PATH}" \
        --original ${ori} \
        --rope_base 10000 \
        --d_extra ${D_EXTRA} \
        --zoom_widths "${ZW}" \
        --ublb "ub"
        
    echo "✅ Figure saved for original=${ori}"
done

python src/draw_graphs.py \
        --result_dir "${RESULT_PATH}" \
        --graph_dir "${GRAPH_PATH}" \
        --original 500 \
        --rope_base 100000 \
        --d_extra 43 \
        --zoom_widths "{1.1:25, 1.2:25, 1.5:25, 2:22, 4:24, 8:20}" \
        --ublb "ub"
        
echo "✅ Figure saved for original=500, rb=100000"