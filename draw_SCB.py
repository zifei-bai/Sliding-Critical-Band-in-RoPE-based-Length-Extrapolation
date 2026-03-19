import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse
import ast


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
    
    return parser.parse_args()

def plot_sliding_critical_band(
    result_dir, 
    graph_dir, 
    compare_mode='original', # 'original' 或 'rope_base'
    original_lengths=None,   # 当 mode='original' 时传列表，否则传单个整数
    rope_bases=None,         # 当 mode='rope_base' 时传列表，否则传单个整数
    save_filename='sliding_critical_band.pdf'
):
    """
    统一的滑动临界带绘制函数，支持比较不同的 original_length 或不同的 rope_base。
    """
    print(f"📥 当前模式: 比较 {compare_mode}...")
    
    # 1. 动态读取并组装数据
    settings = []
    s_data = {} 
    target_pcts = [1.1, 1.2, 1.5, 2.0, 4.0, 8.0]
    
    for pct in target_pcts:
        s_data[f's={pct:g}'] = []

    # 确定要遍历的列表以及图例名称
    if compare_mode == 'original':
        iterate_list = original_lengths
        legend_title = "String-Copying Task"
    elif compare_mode == 'rope_base':
        iterate_list = rope_bases
        legend_title = "RoPE-Base"
    else:
        raise ValueError("compare_mode 必须是 'original' 或 'rope_base'")

    for val in iterate_list:
        # 动态分配当前循环的 orig 和 rb 参数
        orig = val if compare_mode == 'original' else original_lengths
        rb = rope_bases if compare_mode == 'original' else val
        
        # 动态生成标签名称 (把 10000 转成 1e4, 100000 转成 1e5 等)
        if compare_mode == 'original':
            settings.append(f'{orig} digit')
        else:
            label = f'1e{int(np.log10(rb))}' if rb >= 10000 else str(rb)
            settings.append(label)
        
        # 读取文件
        ub_path = os.path.join(result_dir, f"orig{orig}_rb{rb}_ub.csv")
        lb_path = os.path.join(result_dir, f"orig{orig}_rb{rb}_lb.csv")
        
        if not os.path.exists(ub_path) or not os.path.exists(lb_path):
            print(f"⚠️ 找不到 orig={orig}, rb={rb} 的 ub/lb 文件，此项跳过！")
            for pct in target_pcts:
                s_data[f's={pct:g}'].append([None, None])
            continue
            
        df_ub = pd.read_csv(ub_path)
        df_lb = pd.read_csv(lb_path)
        
        for pct in target_pcts:
            ub_row = df_ub[df_ub['pct'] == pct]
            lb_row = df_lb[df_lb['pct'] == pct]
            
            start_val = int(ub_row['min_index'].iloc[0]) if not ub_row.empty else None
            end_val = int(lb_row['min_index'].iloc[0]) if not lb_row.empty else None
            
            s_data[f's={pct:g}'].append([start_val, end_val])
            
    # 转换为 DataFrame
    df = pd.DataFrame({'Setting': settings, **s_data}).set_index('Setting')
    
    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    plt.rcParams['font.family'] = 'serif'

    s_labels = df.columns.tolist()
    y_positions = np.arange(len(s_labels))
    
    # 支持最多 5 种设置的颜色搭配 (红、绿、蓝、紫、黄)
    # 你刚才用的是红绿搭配，这里把它们放在前面
    palette = ['#C44E52', '#55A868', '#4C72B0', '#8172B3', '#CCB974']
    colors = palette[:len(iterate_list)]
    
    num_settings = len(iterate_list)
    bar_height = 0.6 / num_settings 

    # 3. 绘制阴影区间
    for i, task in enumerate(df.index):
        offset = ((num_settings - 1) / 2.0 - i) * (bar_height + 0.02)

        for j, s_col in enumerate(s_labels):
            vals = df.loc[task, s_col]
            if vals[0] is None or vals[1] is None:
                continue
                
            start, end = vals
            ax.barh(y_positions[j] + offset, end - start, left=start,
                    height=bar_height, color=colors[i], alpha=0.7,
                    edgecolor='none', label=task if j == 0 else "")
            ax.vlines([start, end], y_positions[j] + offset - bar_height/2,
                      y_positions[j] + offset + bar_height/2,
                      colors=colors[i], linewidth=1.2, alpha=0.9)

    # 4. 图表修饰
    ax.set_yticks(y_positions)
    ax.set_yticklabels(s_labels, fontsize=11)
    ax.set_xlabel('Dimension Index ($d$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Extrapolation Ratio ($s$)', fontsize=12, fontweight='bold')
    ax.set_title('Sliding Critical Band Across $s$', fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)

    # 图例
    legend_elements = [Line2D([0], [0], color=colors[i], lw=9, label=task, alpha=0.7) for i, task in enumerate(df.index)]
    ax.legend(
        handles=legend_elements,
        title=legend_title, # 动态获取图例标题
        loc='upper left',
        frameon=True,
        shadow=True,
        fontsize=9,           
        title_fontsize=9,    
        handlelength=1.5,     
        handleheight=1.0,     
        labelspacing=0.4      
    )

    plt.tight_layout()
    os.makedirs(graph_dir, exist_ok=True)
    save_path = os.path.join(graph_dir, save_filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
    plt.show()
    
    
    
    
    
if __name__ == '__main__':
    
    args = parse_args()

    result_dir = args.result_dir
    graph_dir = args.graph_dir


    # 直接调用函数
    plot_sliding_critical_band(
        result_dir=result_dir,
        graph_dir=graph_dir,
        compare_mode='original',       # 设定为比较 length
        original_lengths=[50, 100, 500], # 这里传列表
        rope_bases=10000,               # 这里传单个固定值
        save_filename='sliding_band_by_original.pdf'
    )
    
    plot_sliding_critical_band(
        result_dir=result_dir,
        graph_dir=graph_dir,
        compare_mode='rope_base',      # 设定为比较 rope_base
        original_lengths=500,          # 这里传单个固定值
        rope_bases=[10000, 100000],    # 这里传列表
        save_filename='sliding_band_by_rope_base.pdf'
    )