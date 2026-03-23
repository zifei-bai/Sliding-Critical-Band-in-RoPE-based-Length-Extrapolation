#!/bin/bash
# 开启错误检测
set -e

# 定义数据输入和图表输出的路径
RESULT_DIR="./results/"
GRAPH_DIR="./graphs/"

echo "================================================="
echo "  📊 正在绘制 LLaMA Real Model UB/LB 2x3 曲线图"
echo "  数据目录: ${RESULT_DIR}"
echo "================================================="


# 调用绘图脚本
python src/real_plot.py \
    --result_dir "${RESULT_DIR}" \
    --graph_dir "${GRAPH_DIR}"

echo "🎉 绘图完成！请前往 ${GRAPH_DIR} 查看 PDF 文件。"