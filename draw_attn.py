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

    print("🎨 开始绘制 Attention 模式图...")

    # 1. 加载打包好的数据
    data_file = os.path.join(args.result_dir, f"attention_matrices_orig{args.original}_pct{args.pct}.npz")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到数据文件: {data_file}，请先运行 extract 脚本！")
    
    att_data = np.load(data_file)
    target_heads = [(0, 0), (0, 1), (1, 0), (3, 1)]
    
    # 因为你硬编码了，这里我们图例保持一致
    indss = [20, 63] 
    row_names = ["Original", f"Interpolated {indss}"]

    # 2. 准备画布
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=200)
    plt.rcParams['font.family'] = 'serif'    
    # 3. 开始遍历画图
    for row_idx in range(2):
        prefix = "orig" if row_idx == 0 else "interp"
        
        for col_idx, (l, h) in enumerate(target_heads):
            ax = axes[row_idx, col_idx]
            
            # 从 npz 字典中拿出对应矩阵
            att_np = att_data[f"{prefix}_L{l}H{h}"]

            im = ax.imshow(att_np, aspect='equal', interpolation='nearest', cmap='viridis')
            ax.set_box_aspect(1)

            # 动态生成标题和标签
            ax.set_title(f"{row_names[row_idx]}\nL{l}H{h}", fontsize=13, fontweight='bold', pad=10)
            if row_idx == 1: 
                ax.set_xlabel("Token Position Index", fontsize=11)
            if col_idx == 0: 
                ax.set_ylabel("Token Position Index", fontsize=11)

            # Colorbar 侧边栏
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 4. 保存图表
    os.makedirs(args.graph_dir, exist_ok=True)
    save_pdf = os.path.join(args.graph_dir, f"attention_compare_{args.original}*{args.pct}.pdf")
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_pdf}")
    
    plt.show()