import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
import os

import argparse
import ast
import torch
from config import VOCAB # 确保导入了 VOCAB

def parse_dict(arg_str):
    try:
        # literal_eval 可以将类似 "{'a': 1}" 的字符串安全地转换成真实字典
        parsed = ast.literal_eval(arg_str)
        if not isinstance(parsed, dict):
            raise ValueError("Input is not a dictionary")
        return parsed
    except Exception as e:
        raise argparse.ArgumentTypeError(f"无效的字典格式! 请确认格式如: \"{{1.1: 21, 1.2: 17}}\"\\n错误信息: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Draw UBs Script")
    
    # 路径参数
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--graph_dir', type=str, default=None)
    parser.add_argument("--original", type=int, default=None)
    parser.add_argument("--rope_base", type=int, default=10000)
    parser.add_argument("--d_extra", type=int, default=None)
    parser.add_argument("--zoom_widths", type=parse_dict, default=None)
    parser.add_argument("--ublb", type=str, default=None)
    
    return parser.parse_args()


def draw_evaluation_plots(
    ppl_csv_path, 
    original_length, 
    vline_x, 
    zoom_widths_dict, 
    save_path, 
    acc_csv_path=None
):
    """
    绘制 PPL (和可选的 ACC) 随维度 d 变化的图表。
    
    参数:
    - ppl_csv_path: PPL 数据的 CSV 路径
    - original_length: 原始长度 (如 50, 100, 500)
    - vline_x: 垂直参考线所在的 x 坐标 (如 37, 43)
    - zoom_widths_dict: 字典，映射 s_val 到放大镜宽度，如 {1.1: 21, 1.2: 17, ...}
    - save_path: 图片保存的完整路径
    - acc_csv_path: ACC 数据的 CSV 路径。如果不传或为 None，则只画 PPL 单 Y 轴。
    """
    
    # 1. 加载数据
    if not os.path.exists(ppl_csv_path):
        print(f"❌ 找不到 PPL 数据文件: {ppl_csv_path}")
        return
        
    df_ppl = pd.read_csv(ppl_csv_path)
    x_col = df_ppl.columns[0]
    x = df_ppl[x_col]
    
    # 自动解析列名和 s_vals
    cols = df_ppl.columns[1:].tolist()
    s_vals = [float(col) / original_length for col in cols]
    
    # 尝试加载 Accuracy 数据
    df_acc = None
    if acc_csv_path and os.path.exists(acc_csv_path):
        df_acc = pd.read_csv(acc_csv_path)

    # 2. 动态创建画布
    num_plots = len(cols)
    ncols = 3
    nrows = int(np.ceil(num_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), dpi=300)
    axes = axes.flatten() if num_plots > 1 else [axes]
    plt.rcParams['font.family'] = 'serif'

    ppl_color = '#1f77b4'
    acc_color = '#2ca02c'

    for i, col in enumerate(cols):
        ax = axes[i]
        y_ppl = df_ppl[col]
        s_val = s_vals[i]
        
        valid_x = x[y_ppl.notna()]
        start_x = valid_x.min() if not valid_x.empty else 0
        end_x = valid_x.max() if not valid_x.empty else 100

        # 获取指定的放大宽度，如果没有配置则默认 20
        w = zoom_widths_dict.get(s_val, 20)

        # 获取最小值索引和坐标
        min_idx = y_ppl.idxmin()
        min_x, min_val = df_ppl.loc[min_idx, x_col], y_ppl[min_idx]

        # --- 绘制主图 PPL ---
        ln1 = ax.plot(x, y_ppl, color=ppl_color, linestyle='--', linewidth=1.5, alpha=0.8, label='Perplexity')
        ax.set_title(f'Extrapolation Ratio $s = {s_val:g}$', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Dimensions ($d$)', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12, color=ppl_color)
        ax.tick_params(axis='y', labelcolor=ppl_color)
        
        
        ax.set_xlim([start_x-5, max(end_x, 100)]) # 保证至少显示到 100，也可以根据需要调整
        ax.grid(True, linestyle=':', alpha=0.4)

        # --- 处理 Accuracy (双坐标轴) ---
        lns = ln1
        if df_acc is not None and col in df_acc.columns:
            ax2 = ax.twinx()
            # 确保 x 轴对齐
            acc_x_col = df_acc.columns[0]
            y_acc_aligned = df_acc.set_index(acc_x_col).reindex(x).reset_index()[col]
            
            ln2 = ax2.plot(x, y_acc_aligned, color=acc_color, linestyle='-', linewidth=1.5, alpha=0.8, label='Accuracy')
            ax2.set_ylabel('Accuracy', fontsize=12, color=acc_color)
            ax2.tick_params(axis='y', labelcolor=acc_color)
            ax2.set_ylim(-0.05, 1.05)
            lns = ln1 + ln2

        # 图例
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='lower right', fontsize=6, framealpha=0.8)

       
        
        if vline_x != -1:
             # --- 绘制自定义垂直虚线及标注 ---
            ax.axvline(x=vline_x, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
            # PPL 垂直线交点
            val_vline_ppl_series = df_ppl[df_ppl[x_col] == vline_x][col]
            if not val_vline_ppl_series.empty:
                val_vline_ppl = val_vline_ppl_series.iloc[0]
                if not np.isnan(val_vline_ppl):
                    ax.plot(vline_x, val_vline_ppl, 'ko', markersize=4)
                    ax.annotate(f'PPL@{vline_x}: {val_vline_ppl:.4f}',
                                xy=(vline_x, val_vline_ppl),
                                xytext=(vline_x+2, val_vline_ppl + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05),
                                fontsize=8, color='black', fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color='black'))

            # ACC 垂直线交点
            if df_acc is not None and col in df_acc.columns:
                val_vline_acc_series = df_acc[df_acc[acc_x_col] == vline_x][col]
                if not val_vline_acc_series.empty:
                    val_vline_acc = val_vline_acc_series.iloc[0]
                    if not np.isnan(val_vline_acc):
                        ax2.plot(vline_x, val_vline_acc, 'go', markersize=4)
                        ax2.annotate(f'Acc@{vline_x}: {val_vline_acc:.2f}',
                                    xy=(vline_x, val_vline_acc),
                                    xytext=(vline_x-22, val_vline_acc-0.15),
                                    fontsize=8, color=acc_color, fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color=acc_color))

        # 使用指定的宽度 w 绘制主图阴影
        ax.axvspan(min_x - w, min_x + w, color='red', alpha=0.05)

        # --- 绘制放大镜 ---
        inset_x_lim = [max(start_x, min_x - w), min(end_x, min_x + w)]

        axins = inset_axes(ax, width="50%", height="33%", loc='right',
                           bbox_to_anchor=(0, 0, 1.0, 1.0),
                           bbox_transform=ax.transAxes,
                           borderpad=0.5)

        axins.plot(x, y_ppl, color=ppl_color, linewidth=2.0)
        axins.plot(min_x, min_val, 'r*', markersize=10, zorder=5)

        y_inset_data = y_ppl[(x >= inset_x_lim[0]) & (x <= inset_x_lim[1])]
        if not y_inset_data.dropna().empty:
            local_y_min, local_y_max = y_inset_data.min(), y_inset_data.max()
            local_range = max(local_y_max - local_y_min, 0.0001)
            axins.set_xlim(inset_x_lim)
            axins.set_ylim(local_y_min - local_range * 0.1, local_y_min + local_range * 1.5)

        axins.annotate(f'$\\mathbf{{d = {int(min_x)}}}$\nMin PPL: {min_val:.4f}',
                       xy=(min_x, min_val),
                       xytext=(0, 15),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       color='red', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', lw=0.5, alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1),
                       zorder=10)

        axins.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axins.tick_params(axis='both', labelsize=7)
        axins.grid(True, linestyle=':', alpha=0.5)

        if y_ppl.max() > 1000:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 清除多余的空坐标轴
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
    plt.close() # 释放内存




if __name__ == '__main__':
    
    args = parse_args()
    # ==========================================
    # 1. 路径与文件夹设置 (在此处修改为你想要的路径)
    # ==========================================
    # 指向你要读取的 csv 文件
    d_extra=args.d_extra
    zoom_width_dict = args.zoom_widths
    
    original = args.original
    rb = args.rope_base
    ublb=args.ublb

    graph_path = args.graph_dir
    result_path = args.result_dir

    graph_filename = f"{ublb}_original={original}_rb={rb}.pdf"
    graph_save_path = os.path.join(graph_path, graph_filename)

    ppl_result_filename = f"raw_ppl_orig{original}_rb{rb}_{ublb}.csv"
    ppl_result_file_path = os.path.join(result_path, ppl_result_filename)
    
    acc_result_filename = f"raw_acc_orig{original}_rb{rb}_{ublb}.csv"
    acc_result_file_path = os.path.join(result_path, acc_result_filename)


    # 强烈建议在这里把你刚刚写好的函数调用放进来
    # 例如画 original = 100 的图：
    draw_evaluation_plots(
        ppl_csv_path = ppl_result_file_path,
        acc_csv_path = acc_result_file_path, 
        original_length = original,
        vline_x = d_extra,  
        zoom_widths_dict = zoom_width_dict,
        save_path = graph_save_path
    )