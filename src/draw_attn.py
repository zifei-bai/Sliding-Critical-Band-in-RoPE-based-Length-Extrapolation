import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Attention Matrices from NPZ")
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--graph_dir', type=str, required=True)
    parser.add_argument('--original', type=int, default=50)
    parser.add_argument('--pct', type=float, default=1.5)
    args = parser.parse_args()

    print("Drawing Attention Graph...")

    data_file = os.path.join(args.result_dir, f"attention_matrices_orig{args.original}_pct{args.pct}.npz")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Can't find data file: {data_file}. ")
    
    att_data = np.load(data_file)
    target_heads = [(0, 0), (0, 1), (1, 0), (3, 1)]
    
    indss = [20, 63] 
    row_names = ["Original", f"Interpolated {indss}"]


    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=200)
    plt.rcParams['font.family'] = 'serif'    
    for row_idx in range(2):
        prefix = "orig" if row_idx == 0 else "interp"
        
        for col_idx, (l, h) in enumerate(target_heads):
            ax = axes[row_idx, col_idx]
            
            att_np = att_data[f"{prefix}_L{l}H{h}"]

            im = ax.imshow(att_np, aspect='equal', interpolation='nearest', cmap='viridis')
            ax.set_box_aspect(1)

            ax.set_title(f"{row_names[row_idx]}\nL{l}H{h}", fontsize=13, fontweight='bold', pad=10)
            if row_idx == 1: 
                ax.set_xlabel("Token Position Index", fontsize=11)
            if col_idx == 0: 
                ax.set_ylabel("Token Position Index", fontsize=11)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(args.graph_dir, exist_ok=True)
    save_pdf = os.path.join(args.graph_dir, f"attention_compare_{args.original}*{args.pct}.pdf")
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"Graph saved to: {save_pdf}")
    
    plt.show()