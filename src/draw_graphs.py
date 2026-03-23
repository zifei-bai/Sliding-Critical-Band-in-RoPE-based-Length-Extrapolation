import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
import os

import argparse
import ast
import torch
from config import VOCAB 

def parse_dict(arg_str):
    try:
        parsed = ast.literal_eval(arg_str)
        if not isinstance(parsed, dict):
            raise ValueError("Input is not a dictionary")
        return parsed
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Error {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Draw UBs Script")
    
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
    Draw PPL (optional ACC) change with dimention (d)
    
    参数:
    - ppl_csv_path: PPL CSV data path
    - original_length: training length (50, 100, 500)
    - vline_x: d_extra
    - zoom_widths_dict: subgraph width {1.1: 21, 1.2: 17, ...}
    - save_path: save path
    - acc_csv_path: ACC CSV data path. If None, single PPL y axis
    """
    
    if not os.path.exists(ppl_csv_path):
        print(f"Can't find PPL data file: {ppl_csv_path}")
        return
        
    df_ppl = pd.read_csv(ppl_csv_path)
    x_col = df_ppl.columns[0]
    x = df_ppl[x_col]
    
    cols = df_ppl.columns[1:].tolist()
    s_vals = [float(col) / original_length for col in cols]
    
    df_acc = None
    if acc_csv_path and os.path.exists(acc_csv_path):
        df_acc = pd.read_csv(acc_csv_path)

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

        w = zoom_widths_dict.get(s_val, 20)

        min_idx = y_ppl.idxmin()
        min_x, min_val = df_ppl.loc[min_idx, x_col], y_ppl[min_idx]

        ln1 = ax.plot(x, y_ppl, color=ppl_color, linestyle='--', linewidth=1.5, alpha=0.8, label='Perplexity')
        ax.set_title(f'Extrapolation Ratio $s = {s_val:g}$', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Dimensions ($d$)', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12, color=ppl_color)
        ax.tick_params(axis='y', labelcolor=ppl_color)
        
        
        ax.set_xlim([start_x-5, max(end_x, 100)]) 
        ax.grid(True, linestyle=':', alpha=0.4)

        lns = ln1
        if df_acc is not None and col in df_acc.columns:
            ax2 = ax.twinx()
            acc_x_col = df_acc.columns[0]
            y_acc_aligned = df_acc.set_index(acc_x_col).reindex(x).reset_index()[col]
            
            ln2 = ax2.plot(x, y_acc_aligned, color=acc_color, linestyle='-', linewidth=1.5, alpha=0.8, label='Accuracy')
            ax2.set_ylabel('Accuracy', fontsize=12, color=acc_color)
            ax2.tick_params(axis='y', labelcolor=acc_color)
            ax2.set_ylim(-0.05, 1.05)
            lns = ln1 + ln2

        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='lower right', fontsize=6, framealpha=0.8)

       
        
        if vline_x != -1:
            ax.axvline(x=vline_x, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
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

        ax.axvspan(min_x - w, min_x + w, color='red', alpha=0.05)

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

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")
    plt.close()




if __name__ == '__main__':
    
    args = parse_args()

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

    draw_evaluation_plots(
        ppl_csv_path = ppl_result_file_path,
        acc_csv_path = acc_result_file_path, 
        original_length = original,
        vline_x = d_extra,  
        zoom_widths_dict = zoom_width_dict,
        save_path = graph_save_path
    )