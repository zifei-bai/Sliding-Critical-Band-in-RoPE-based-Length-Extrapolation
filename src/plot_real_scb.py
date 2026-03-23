import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot Sliding Critical Band from CSVs")
    parser.add_argument('--result_dir', type=str, required=True, help="存放 CSV 数据的文件夹")
    parser.add_argument('--graph_dir', type=str, required=True, help="输出图表的文件夹")
    args = parser.parse_args()

    os.makedirs(args.graph_dir, exist_ok=True)
    print("📥 正在读取数据并提取滑动边界...")

    # 1. 定义实验配置映射
    # 配置: (文件后缀长度, 对应的 s 标签, 颜色)
    configs = [
        (2304, 's=1.125', '#1f77b4'), # 专业的蓝色
        (2560, 's=1.25', '#55A868'),  # 专业的绿色
        (4096, 's=2.0', '#C44E52')    # 专业的红色
    ]

    s_labels = []
    ranges = []
    colors = []

    # 2. 动态读取数据并提取 UB/LB 边界
    for seq_len, label, color in configs:
        csv_path = os.path.join(args.result_dir, f'real_ppl_ub_lb_{seq_len}.csv')
        
        if not os.path.exists(csv_path):
            print(f"⚠️ 警告: 找不到文件 {csv_path}，已跳过该项！")
            continue
            
        df = pd.read_csv(csv_path)
        
        # 提取 UB (左边界) 和 LB (右边界) 的最小 PPL 对应的索引
        left_bound = df['PPL_ub'].idxmin()
        right_bound = df['PPL_lb'].idxmin()
        
        s_labels.append(label)
        ranges.append([left_bound, right_bound])
        colors.append(color)
        
        print(f"  [{label}] 提取成功: 左边界(UB)={left_bound}, 右边界(LB)={right_bound}")

    if not ranges:
        raise ValueError("❌ 没有找到任何有效的 CSV 数据，无法绘图！")

    print("🎨 开始绘制滑动临界带 (Sliding Critical Band)...")

    # 3. 创建画布 - 扁平比例，极其适合插在论文正文中
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=300)
    
    # 学术期刊级字体设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['STIXGeneral']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    y_positions = np.arange(len(s_labels))
    bar_height = 0.55 # 条形厚度

    # 4. 绘制阴影区间
    for j, (start, end) in enumerate(ranges):
        # 绘制主体色块
        ax.barh(y_positions[j], end - start, left=start,
                height=bar_height, color=colors[j], alpha=0.6,
                edgecolor='none')

        # 绘制左右边界线，增强视觉硬度
        ax.vlines([start, end], y_positions[j] - bar_height/2,
                  y_positions[j] + bar_height/2,
                  colors=colors[j], linewidth=2.5, alpha=0.9)

    # 5. 图表修饰
    ax.set_yticks(y_positions)
    ax.set_yticklabels(s_labels, fontsize=11, fontweight='bold')
    
    # 翻转 Y 轴，让最小的 s=1.125 排在最上面，符合常规阅读习惯

    ax.set_xlabel('Dimension Index ($d$)', fontsize=11, fontweight='bold')
    ax.set_title('Sliding Critical Band (C4 Task)', fontsize=12, fontweight='bold', pad=12)

    # 坐标范围设置 (RoPE 维度总共有 64 个，即 0-63)
    ax.set_xlim(0, 64)
    ax.set_xticks(range(0, 65, 8))
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)

    # 移除上方、右方、左方的边框，使图表更现代、更透气
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 隐藏左侧 Y 轴的刻度小短线（因为已经没有边框了，留着字就行）
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    
    # 保存为 PDF
    save_path = os.path.join(args.graph_dir, 'real_scb.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表已成功保存至: {save_path}")

if __name__ == '__main__':
    main()