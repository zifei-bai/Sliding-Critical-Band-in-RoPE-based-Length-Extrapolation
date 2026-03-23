#!/bin/bash
# 开启错误检测：任何一行报错立即停止运行
set -e



# 设定数据和结果存放的绝对路径
DATA_DIR="./data/"
RESULT_DIR="./results/"
SEQ_LEN=(4096)
TRAINING_LEN=2048

echo "================================================="
echo "  🚀 启动 LLaMA RoPE 端到端真实场景验证 (Real)"
echo "================================================="


for seq in "${SEQ_LEN[@]}"; do
    TARGET_SCALE=$(awk "BEGIN {print $seq/$TRAINING_LEN}")
    echo "target scale: ${TARGET_SCALE}"

    python src/real_llama.py \
        --model_id "huggyllama/llama-7b" \
        --data_dir "${DATA_DIR}" \
        --result_dir "${RESULT_DIR}" \
        --seq_len ${seq} \
        --num_samples 48 \
        --batch_size 4 \
        --target_scale ${TARGET_SCALE}

    echo "✅ Length ${seq} finished."
    echo "-----------------------------------------"
done

echo "================================================="
echo "  ✅ 所有端到端验证任务已圆满结束！"
echo "  请前往 ${RESULT_DIR} 查看生成的 CSV 结果文件。"
echo "================================================="
            