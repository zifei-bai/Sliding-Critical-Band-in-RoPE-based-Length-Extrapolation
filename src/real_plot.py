import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter

def main():
    parser = argparse.ArgumentParser(description="Plot 2x3 UB/LB Grid for Real LLaMA Validation")
    parser.add_argument('--result_dir', type=str, required=True, help="存放 CSV 数据的文件夹")
    parser.add_argument('--graph_dir', type=str, required=True, help="输出图表的文件夹")
    args = parser.parse_args()

    os.makedirs(args.graph_dir, exist_ok=True)

    print("📥 正在加载 CSV 数据...")
    # 1. 加载数据 
    # s=1.125 (2304), s=1.25 (2560), s=2.0 (4096)
    df_2304 = pd.read_csv(os.path.join(args.result_dir, 'real_ppl_ub_lb_2304.csv'))
    df_2560 = pd.read_csv(os.path.join(args.result_dir, 'real_ppl_ub_lb_2560.csv'))
    df_4096 = pd.read_csv(os.path.join(args.result_dir, 'real_ppl_ub_lb_4096.csv'))

    # 数据集按列排列: [Col 0, Col 1, Col 2]
    dfs = [df_2304, df_2560, df_4096]
    
    # 获取最大维度长度，动态生成 x 轴
    max_len = max(len(df_2304), len(df_2560), len(df_4096))
    x = np.arange(0, max_len)

    # 布局配置
    search_types = ['PPL_ub', 'PPL_lb']
    colors = [['#1f77b4', '#1f77b4', '#1f77b4'],  # 第一行 (UB) 全是蓝色
              ['#d62728', '#d62728', '#d62728']]  # 第二行 (LB) 全是红色
              
    titles = [
        ['Exclusive Sweep (UB), $s=1.125$', 'Exclusive Sweep (UB), $s=1.25$', 'Exclusive Sweep (UB), $s=2.0$'],
        ['Inclusive Sweep (LB), $s=1.125$', 'Inclusive Sweep (LB), $s=1.25$', 'Inclusive Sweep (LB), $s=2.0$']
    ]

    print("🎨 开始绘制 2x3 画布...")
    # 2. 创建 2x3 画布 (稍微加宽以适应3列)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    
    # 学术期刊级字体设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['STIXGeneral']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    for r in range(2):       # 行 (0: UB, 1: LB)
        for c in range(3):   # 列 (0: 1.125, 1: 1.25, 2: 2.0)
            ax = axes[r, c]
            col_name = search_types[r]
            df = dfs[c]
            
            # 安全截取真实数据
            y = df[col_name] if col_name in df.columns else pd.Series([np.nan]*max_len)

            # --- 核心改动：动态 X 轴范围设置 ---
            if r == 1:  # 如果是 LB (第二行)
                first_valid = y.first_valid_index()
                if first_valid is not None:
                    # X 轴最左侧不要从0开始，而是从有数值的 -5 开始
                    start_x = max(0, first_valid - 5)
                    ax.set_xlim([start_x, max_len + 2])
            else:       # 如果是 UB (第一行)
                ax.set_xlim([-2, max_len + 2])
                
            # --- 绘制主图曲线 ---
            ax.plot(x[:len(y)], y, color=colors[r][c], linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_title(titles[r][c], fontsize=11, fontweight='bold', pad=15)
            ax.set_xlabel('Dimension Index ($d$)', fontsize=12)
            if c == 0:  # 只在最左侧显示 Y 轴标签
                ax.set_ylabel('Perplexity', fontsize=12)

            ax.grid(True, linestyle=':', alpha=0.4)

            # --- UB (第一行) d=41 垂直虚线标注 ---
            if r == 0 and 41 < len(y):
                ax.axvline(x=41, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                val_41 = y.iloc[41]
                if not np.isnan(val_41):
                    # 动态偏移，避免文字撞车
                    offset = (y.max() - y.min()) * 0.15 if c != 2 else (y.max() - y.min()) * -0.15
                    x_text_offset = 41 + 5 if c == 0 else 41 - 25
                    
                    ax.annotate(f'PPL@41: {val_41:.4f}', xy=(41, val_41),
                                xytext=(x_text_offset, val_41 + offset),
                                fontsize=9, color='black', fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))

            # --- 寻找最小值并绘制放大镜 (Inset Zoom) ---
            valid_y = y.dropna()
            if not valid_y.empty:
                min_idx = valid_y.idxmin()
                min_x, min_val = x[min_idx], y[min_idx]

                # 设置缩放范围 
                zoom_w = 12 if c == 0 else 8
                ax.axvspan(min_x - zoom_w, min_x + zoom_w, color='red', alpha=0.05)

                # 确定放大镜摆放位置 (UB放左上，LB放右上)
                loc = 'upper right' if r == 1 else 'upper left'
                axins = inset_axes(ax, width="45%", height="35%", loc=loc,
                                   bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
                                   bbox_transform=ax.transAxes)

                axins.plot(x[:len(y)], y, color=colors[r][c], linewidth=2.0)
                axins.plot(min_x, min_val, 'r*', markersize=12, zorder=5) # 极小值标注

                # 放大镜坐标轴缩放
                inset_x_lim = [max(0, min_x - zoom_w), min(max_len, min_x + zoom_w)]
                mask = (x >= inset_x_lim[0]) & (x <= inset_x_lim[1])
                # 对齐掩码长度
                mask = mask[:len(y)] 
                y_zoom_data = y[mask].dropna()

                if not y_zoom_data.empty:
                    local_y_min, local_y_max = y_zoom_data.min(), y_zoom_data.max()
                    local_range = max(local_y_max - local_y_min, 0.001)
                    axins.set_xlim(inset_x_lim)
                    axins.set_ylim(local_y_min - local_range * 0.1, local_y_min + local_range * 1.5)

                # 放大镜内标注
                annotated_x = -45 if r == 1 else 0
                axins.annotate(f'd = {int(min_x)}\nMin PPL: {min_val:.4f}',
                               xy=(min_x, min_val),
                               xytext=(annotated_x, 25),
                               textcoords='offset points',
                               ha='center', va='bottom',
                               color='red', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', lw=1, alpha=0.9),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                               zorder=10)

                axins.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                axins.tick_params(axis='both', labelsize=8)
                axins.grid(True, linestyle=':', alpha=0.5)

    # 自动整理布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    # 保存为 PDF
    save_path = os.path.join(args.graph_dir, 'real_model_ub_lb_2x3.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表已成功保存至: {save_path}")

if __name__ == '__main__':
    main()