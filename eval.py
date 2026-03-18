import pandas as pd
from collections import Counter
import math
import random
import inspect
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import wandb
import matplotlib.pyplot as plt
import os

from model import GPT_RoPE
# from model import generate_greedy
from utilities import get_wrong_ans_acc, estimate_ppl

import argparse
import torch
from config import VOCAB # 确保导入了 VOCAB

def parse_args():
    parser = argparse.ArgumentParser(description="RoPE Model Evaluation Script")
    
    # 路径参数
    parser.add_argument('--working_dir', type=str, default="/content/drive/MyDrive/Research/RoPE_NoPE/final_for_paper/")
    parser.add_argument('--data_dir', type=str, default="/content/drive/MyDrive/Research/train_test_data/")
    parser.add_argument('--result_dir', type=str, default=None)
    
    # 模型架构参数
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rope_base', type=int, default=10000)
    # bias 默认为 True，如果传入 --no_bias 则变为 False
    parser.add_argument('--no_bias', action='store_false', dest='bias') 
    
    # 实验与数据参数
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--block_size', type=int, default=8192)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--original', type=int, default=500)
    parser.add_argument('--inverse_t', type=float, default=1.0)
    parser.add_argument('--from_where', type=int, default=0)
    parser.add_argument('--pct', type=float, default=1.0) # 建议用 float，以防你传 8.5
    parser.add_argument('--ramp', type=int, default=1)
    
    # 布尔控制开关 (如果不传该参数，默认是 False；在命令行加上该参数，就是 True)
    parser.add_argument('--is_from', action='store_true')
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--need_acc', action='store_true')
    
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# --- 为了不修改你后面的代码，这里将 args 拆解回你原来的变量名 ---
batch_size = args.batch_size
block_size = args.block_size
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout
inverse_t = args.inverse_t
bias = args.bias
eval_iters = args.eval_iters
working_dir = args.working_dir
data_dir = args.data_dir
result_dir = args.result_dir
original = args.original
rope_base = args.rope_base
is_from = args.is_from
from_where = args.from_where
pct = args.pct
ramp = args.ramp
is_save = args.is_save
need_acc = args.need_acc

# 动态推断和无需外部传入的变量保留原样
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(VOCAB)
indices = None


# batch_size = 100
# block_size = 8192
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n_embd = 384
# n_head = 2
# n_layer = 4
# dropout = 0.0
# inverse_t = 1.0
# bias = True
# indices = None
# eval_iters = 10
# vocab_size = len(vocab)
# working_dir = "/content/drive/MyDrive/Research/RoPE_NoPE/final_for_paper/"
# data_dir = "/content/drive/MyDrive/Research/train_test_data/"

# original = 500
# rope_base = 100000
# is_from = False
# from_where = 29
# pct = 8
# ramp = 1

# is_save = False
# need_acc = False
# usage_list = []




ind_acc_list = []
ood8_acc_list = []
ppl_test = round(original * pct)
ppls = {ppl_test:[]}
print(f"original is {original}")
print(f"rope_base is {rope_base}")
print(f"ppl_test is {ppl_test}")
print("------------------------------")
if is_from:
    iter_list = list(range(0, 97))
    prefix = "From_"
    special_i = 96
else:
    iter_list = list(range(from_where-1, 96))
    prefix = "To_"
    # special_i = -1
    special_i = from_where-1

for i in iter_list:
    print("----------------------------------------------")
    if i == special_i:
        print("First one")
        m0 = GPT_RoPE(vocab_size, block_size, n_embd, n_layer, n_head, dropout, inverse_t=inverse_t, rope_base=rope_base, start=0, end=0, gemma=1, indices = None, bias=True)
        m0.to(device)

        checkpoint_path = f"{working_dir}mlp_rope0_orig{original}_rb{rope_base}.pt"
        m0.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(sum(p.numel() for p in m0.parameters())/1e6, 'M parameters')

        m0.eval()
        if need_acc:
            ood8_acc_list.append(get_wrong_ans_acc(m0, ppl_test,
                                                   data_dir=data_dir,
                                                    block_size=block_size,
                                                    batch_size=100))


        # get ppl on original-original+2
        for num_digits in [ppl_test]:
            data=[]
            with open(f"{data_dir}teach_force_{num_digits}.txt", "r", encoding="utf-8") as f:
                data = f.readlines()
                data = [line.strip() for line in data]
                data = data[:100]

            ppl, val_count = estimate_ppl(data, m0, eval_iters=1,
                                            batch_size=batch_size,
                                            block_size=block_size)
            print(f"ppl is {ppl}")
            ppls[num_digits].append(ppl)

        continue

    if is_from:
        indss = [i, 96]
        # indss = [25, 68]
    else:
        indss = [from_where, i]

    print(f"PI {indss} PLANES")

    m0 = GPT_RoPE(vocab_size, block_size, n_embd, n_layer, n_head, dropout, inverse_t=inverse_t, rope_base=rope_base, start=0, end=0, gemma=1, indices = indss, bias=True)
    m0.to(device)

    checkpoint_path = f"{working_dir}mlp_rope0_orig{original}_rb{rope_base}.pt"
    m0.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # print(sum(p.numel() for p in m0.parameters())/1e6, 'M parameters')

    m0.transformer.h[0].attn.set_head_pi_mask([True, True])
    m0.transformer.h[0].attn.enable_headwise_pi()

    m0.transformer.h[1].attn.set_head_pi_mask([True, True])
    m0.transformer.h[1].attn.enable_headwise_pi()


    # if original == 1000:
    m0.transformer.h[2].attn.set_head_pi_mask([True, True])
    m0.transformer.h[2].attn.enable_headwise_pi()

    m0.transformer.h[3].attn.set_head_pi_mask([True, True])
    m0.transformer.h[3].attn.enable_headwise_pi()

    m0.eval()
    m0.transformer.h[0].attn.set_gemma((original*2+3)/((ppl_test)*2+3), indice=indss, ramp=ramp)
    m0.transformer.h[1].attn.set_gemma((original*2+3)/((ppl_test)*2+3), indice=indss, ramp=ramp)
    # if original == 1000:
    m0.transformer.h[2].attn.set_gemma((original*2+3)/((ppl_test)*2+3), indice=indss, ramp=ramp)
    m0.transformer.h[3].attn.set_gemma((original*2+3)/((ppl_test)*2+3), indice=indss, ramp=ramp)
    if need_acc:
        ood8_acc_list.append(get_wrong_ans_acc(m0, ppl_test,
                                               data_dir=data_dir,
                                                block_size=block_size,
                                                batch_size=100))

        # ood8_acc_list.append(get_wrong_ans_acc(m0, 150,
        #                                        block_size=block_size,
        #                                        batch_size=100))


    # get ppl on original-original+2
    for num_digits in [ppl_test]:
        data=[]
        with open(f"{data_dir}teach_force_{num_digits}.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
            data = [line.strip() for line in data]
            data = data[:100]
        ppl, val_count = estimate_ppl(data, m0, eval_iters=1,
                                            batch_size=batch_size,
                                            block_size=block_size)
        print(f"ppl is {ppl}")
        ppls[num_digits].append(ppl)


if is_save:
    # ========================================================
    # 任务一：保存完整的原始数据 (和你上传的表格格式完全一致)
    # ========================================================
    suffix = "ub" if is_from else "lb"
    raw_csv_filename = f"raw_ppl_orig{original}_rb{rope_base}_{suffix}.csv"
    raw_csv_path = os.path.join(result_dir, raw_csv_filename) # 如果你那边叫 result_dir 就改一下
    
    if os.path.exists(raw_csv_path):
        df_raw = pd.read_csv(raw_csv_path, index_col=0)
    else:
        df_raw = pd.DataFrame()
        
    # 2. 将这次跑出来的结果（比如 key 是 110, 120），作为“新的一列”挂到右边
    for k, v in ppls.items():
        df_raw[str(k)] = pd.Series(v)
        
    # 3. 覆盖保存文件，带上左边的 0, 1, 2... 索引
    df_raw.to_csv(raw_csv_path, index=True) 
    print(f"✅ pct={pct} 的数据已作为【新列】加入到: {raw_csv_filename}")

    # ========================================================
    # 任务二：提取并追加最小值的位置(min_index)及具体数值(min_ppl)
    # ========================================================
    min_csv_filename = f"orig{original}_rb{rope_base}_{suffix}.csv"
    min_csv_path = os.path.join(result_dir, min_csv_filename)
    
    # 找到列表中最小值对应的索引 (例如 [3,1,2] 返回 1)
    ppl_arr = np.array(ppls[ppl_test])
    min_idx = int(np.argmin(ppl_arr)) 
    min_ppl = float(ppl_arr[min_idx])
    
    # 构造极简的单行记录
    df_min = pd.DataFrame({
        "min_index": [min_idx],
        "min_ppl": [min_ppl]
    }, index=[pct])
    
    df_min.index.name = "pct" # 指定行名为 pct
    
    # 追加模式写入：文件不存在则带表头新建，存在则只追加数据
    if not os.path.exists(min_csv_path):
        df_min.to_csv(min_csv_path, mode='w', header=True)
    else:
        df_min.to_csv(min_csv_path, mode='a', header=False)
        
    print(f"✅ pct={pct} 的极小值 (min_index={min_idx}, min_ppl={min_ppl}) 已追加至 {min_csv_filename}")