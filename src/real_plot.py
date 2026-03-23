import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter

def main():
    parser = argparse.ArgumentParser(description="Plot 2x3 UB/LB Grid for Real LLaMA Validation")
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--graph_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.graph_dir, exist_ok=True)

    # s=1.125 (2304), s=1.25 (2560), s=2.0 (4096)
    df_2304 = pd.read_csv(os.path.join(args.result_dir, 'real_ppl_ub_lb_2304.csv'))
    df_2560 = pd.read_csv(os.path.join(args.result_dir, 'real_ppl_ub_lb_2560.csv'))
    df_4096 = pd.read_csv(os.path.join(args.result_dir, 'real_ppl_ub_lb_4096.csv'))

    dfs = [df_2304, df_2560, df_4096]

    max_len = max(len(df_2304), len(df_2560), len(df_4096))
    x = np.arange(0, max_len)

    search_types = ['PPL_ub', 'PPL_lb']
    colors = [['#1f77b4', '#1f77b4', '#1f77b4'],  
              ['#d62728', '#d62728', '#d62728']] 
              
    titles = [
        ['Exclusive Sweep (UB), $s=1.125$', 'Exclusive Sweep (UB), $s=1.25$', 'Exclusive Sweep (UB), $s=2.0$'],
        ['Inclusive Sweep (LB), $s=1.125$', 'Inclusive Sweep (LB), $s=1.25$', 'Inclusive Sweep (LB), $s=2.0$']
    ]

   
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['STIXGeneral']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    for r in range(2):       
        for c in range(3):   
            ax = axes[r, c]
            col_name = search_types[r]
            df = dfs[c]
            
           
            y = df[col_name] if col_name in df.columns else pd.Series([np.nan]*max_len)

            
            if r == 1:  
                first_valid = y.first_valid_index()
                if first_valid is not None:
                    
                    start_x = max(0, first_valid - 5)
                    ax.set_xlim([start_x, max_len + 2])
            else:     
                ax.set_xlim([-2, max_len + 2])
                
            
            ax.plot(x[:len(y)], y, color=colors[r][c], linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_title(titles[r][c], fontsize=11, fontweight='bold', pad=15)
            ax.set_xlabel('Dimension Index ($d$)', fontsize=12)
            if c == 0: 
                ax.set_ylabel('Perplexity', fontsize=12)

            ax.grid(True, linestyle=':', alpha=0.4)

          
            if r == 0 and 41 < len(y):
                ax.axvline(x=41, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                val_41 = y.iloc[41]
                if not np.isnan(val_41):
                    
                    offset = (y.max() - y.min()) * 0.15 if c != 2 else (y.max() - y.min()) * -0.15
                    x_text_offset = 41 + 5 if c == 0 else 41 - 25
                    
                    ax.annotate(f'PPL@41: {val_41:.4f}', xy=(41, val_41),
                                xytext=(x_text_offset, val_41 + offset),
                                fontsize=9, color='black', fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))

           
            valid_y = y.dropna()
            if not valid_y.empty:
                min_idx = valid_y.idxmin()
                min_x, min_val = x[min_idx], y[min_idx]

              
                zoom_w = 12 if c == 0 else 8
                ax.axvspan(min_x - zoom_w, min_x + zoom_w, color='red', alpha=0.05)

               
                loc = 'upper right' if r == 1 else 'upper left'
                axins = inset_axes(ax, width="45%", height="35%", loc=loc,
                                   bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
                                   bbox_transform=ax.transAxes)

                axins.plot(x[:len(y)], y, color=colors[r][c], linewidth=2.0)
                axins.plot(min_x, min_val, 'r*', markersize=12, zorder=5) 

                
                inset_x_lim = [max(0, min_x - zoom_w), min(max_len, min_x + zoom_w)]
                mask = (x >= inset_x_lim[0]) & (x <= inset_x_lim[1])
               
                mask = mask[:len(y)] 
                y_zoom_data = y[mask].dropna()

                if not y_zoom_data.empty:
                    local_y_min, local_y_max = y_zoom_data.min(), y_zoom_data.max()
                    local_range = max(local_y_max - local_y_min, 0.001)
                    axins.set_xlim(inset_x_lim)
                    axins.set_ylim(local_y_min - local_range * 0.1, local_y_min + local_range * 1.5)

              
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


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    # 保存为 PDF
    save_path = os.path.join(args.graph_dir, 'real_model_ub_lb_2x3.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")

if __name__ == '__main__':
    main()