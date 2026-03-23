import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot Sliding Critical Band from CSVs")
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--graph_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.graph_dir, exist_ok=True)


   
    configs = [
        (2304, 's=1.125', '#1f77b4'), 
        (2560, 's=1.25', '#55A868'), 
        (4096, 's=2.0', '#C44E52')   
    ]

    s_labels = []
    ranges = []
    colors = []

    
    for seq_len, label, color in configs:
        csv_path = os.path.join(args.result_dir, f'real_ppl_ub_lb_{seq_len}.csv')
        
        if not os.path.exists(csv_path):
            print(f"Can't find file {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        left_bound = df['PPL_ub'].idxmin()
        right_bound = df['PPL_lb'].idxmin()
        
        s_labels.append(label)
        ranges.append([left_bound, right_bound])
        colors.append(color)
        
        print(f"  [{label}] (UB)={left_bound}, (LB)={right_bound}")

    if not ranges:
        raise ValueError("Can't find file")

    print("Visualizing Sliding Critical Band...")

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=300)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['STIXGeneral']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    y_positions = np.arange(len(s_labels))
    bar_height = 0.55

    for j, (start, end) in enumerate(ranges):
        
        ax.barh(y_positions[j], end - start, left=start,
                height=bar_height, color=colors[j], alpha=0.6,
                edgecolor='none')

        ax.vlines([start, end], y_positions[j] - bar_height/2,
                  y_positions[j] + bar_height/2,
                  colors=colors[j], linewidth=2.5, alpha=0.9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(s_labels, fontsize=11, fontweight='bold')
    

    ax.set_xlabel('Dimension Index ($d$)', fontsize=11, fontweight='bold')
    ax.set_title('Sliding Critical Band (C4 Task)', fontsize=12, fontweight='bold', pad=12)

    ax.set_xlim(0, 64)
    ax.set_xticks(range(0, 65, 8))
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    
    save_path = os.path.join(args.graph_dir, 'real_scb.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Graph saved to {save_path}")

if __name__ == '__main__':
    main()