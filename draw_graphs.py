import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter

from google.colab import drive
import os
drive.mount('/content/drive')

working_dir = "/content/drive/MyDrive/Research/RoPE_NoPE/final_for_paper/"

# 1. 加载数据 (假设文件在当前工作目录)
# 请确保 working_dir 的路径正确，或者直接读取本地文件
df_ppl = pd.read_csv(f'{working_dir}raw_data_ppl_500digit_rb1e5_ub.csv')
# df_acc = pd.read_csv(f'{working_dir}raw_data_acc_500digit_ub.csv')
x = df_ppl['Unnamed: 0']
cols = ['550', '600', '750', '1000', '2000', '4000']
s_vals = [1.1, 1.2, 1.5, 2.0, 4.0, 8.0]

# --- 【在此处逐一调整每个图的放大宽度】 ---
# 列表中的 6 个数字分别对应 s = 1.1, 1.2, 1.5, 2.0, 4.0, 8.0
# 例如：如果您想让第一个图看宽一点，第二个图窄一点，就改这里
zoom_widths = [25, 25, 25, 22, 24, 20]

# 2. 创建画布
fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=300)
axes = axes.flatten()
plt.rcParams['font.family'] = 'serif'

ppl_color = '#1f77b4'
acc_color = '#2ca02c'

for i, col in enumerate(cols):
    ax = axes[i]
    y_ppl = df_ppl[col]

    # 获取当前子图指定的放大宽度
    w = zoom_widths[i]

    # 获取最小值索引和坐标
    min_idx = y_ppl.idxmin()
    min_x, min_val = df_ppl.loc[min_idx, 'Unnamed: 0'], y_ppl[min_idx]

    # --- 绘制主图 PPL ---
    ln1 = ax.plot(x, y_ppl, color=ppl_color, linestyle='--', linewidth=1.5, alpha=0.8, label='Perplexity')
    ax.set_title(f'Extrapolation Ratio $s = {s_vals[i]}$', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dimensions ($d$)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12, color=ppl_color)
    ax.set_xlim([-5, 100])
    ax.tick_params(axis='y', labelcolor=ppl_color)
    ax.grid(True, linestyle=':', alpha=0.4)

    # --- 处理 Accuracy (双坐标轴) ---
    # lns = ln1
    # if col in df_acc.columns:
    #     ax2 = ax.twinx()
    #     y_acc_aligned = df_acc.set_index('Unnamed: 0').reindex(x).reset_index()[col]
    #     ln2 = ax2.plot(x, y_acc_aligned, color=acc_color, linestyle='-', linewidth=1.5, alpha=0.8, label='Accuracy')
    #     ax2.set_ylabel('Accuracy', fontsize=12, color=acc_color)
    #     ax2.tick_params(axis='y', labelcolor=acc_color)
    #     ax2.set_ylim(-0.05, 1.05)
    #     lns = ln1 + ln2

    # 图例
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right', fontsize=6, framealpha=0.8)

    # --- 绘制 d=30 垂直虚线及标注 ---
    ax.axvline(x=43, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
    val_37_ppl = df_ppl[df_ppl['Unnamed: 0'] == 43][col].iloc[0]
    if not np.isnan(val_37_ppl):
        ax.plot(43, val_37_ppl, 'ko', markersize=4)
        if i >= 3:
            ax.annotate(f'PPL@43: {val_37_ppl:.4f}',
                        xy=(43, val_37_ppl),
                        xytext=(-3, val_37_ppl + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.08),
                        fontsize=8, color='black', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'))
        else:
            ax.annotate(f'PPL@43: {val_37_ppl:.4f}',
                        xy=(43, val_37_ppl),
                        xytext=(-3, val_37_ppl + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.08),
                        fontsize=8, color='black', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'))

    # if col in df_acc.columns:
    #     val_37_acc = df_acc[df_acc['Unnamed: 0'] == 53][col].iloc[0]
    #     if not np.isnan(val_37_acc):
    #         ax2.plot(53, val_37_acc, 'go', markersize=4)
    #         ax2.annotate(f'Acc@30: {val_37_acc:.2f}',
    #                     xy=(53, val_37_acc),
    #                     xytext=(53-22, val_37_acc-0.15),
    #                     fontsize=8, color=acc_color, fontweight='bold',
    #                     arrowprops=dict(arrowstyle='->', color=acc_color))

    # 使用当前指定的宽度 w 绘制主图阴影
    ax.axvspan(min_x - w, min_x + w, color='red', alpha=0.05)

    # --- 绘制放大镜 (紧贴右侧) ---
    # 使用当前指定的宽度 w 设置放大镜 X 轴范围
    inset_x_lim = [max(0, min_x - w), min(x.max(), min_x + w)]

    # bbox_to_anchor 设为 (0, 0, 1.0, 1.0) 结合 borderpad=0 实现紧贴右侧和顶部
    axins = inset_axes(ax, width="50%", height="33%", loc='right',
                       bbox_to_anchor=(0, 0, 1.0, 1.0),
                       bbox_transform=ax.transAxes,
                       borderpad=0.5)

    axins.plot(x, y_ppl, color=ppl_color, linewidth=2.0)
    axins.plot(min_x, min_val, 'r*', markersize=10, zorder=5)

    # 放大镜 Y 轴自动缩放
    y_inset_data = y_ppl[(x >= inset_x_lim[0]) & (x <= inset_x_lim[1])]
    if not y_inset_data.dropna().empty:
        local_y_min, local_y_max = y_inset_data.min(), y_inset_data.max()
        local_range = max(local_y_max - local_y_min, 0.0001)
        axins.set_xlim(inset_x_lim)
        axins.set_ylim(local_y_min - local_range * 0.1, local_y_min + local_range * 1.5)

    # 放大镜内标注
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

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(f'{working_dir}ppl_acc_plot_ub_500digi_rb1e5.pdf', bbox_inches='tight')
plt.show()