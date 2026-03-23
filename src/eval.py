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
from config import VOCAB

def parse_args():
    parser = argparse.ArgumentParser(description="RoPE Model Evaluation Script")
    
    parser.add_argument('--working_dir', type=str, default="/content/drive/MyDrive/Research/RoPE_NoPE/final_for_paper/")
    parser.add_argument('--data_dir', type=str, default="/content/drive/MyDrive/Research/train_test_data/")
    parser.add_argument('--result_dir', type=str, default=None)
    
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rope_base', type=int, default=10000)
    parser.add_argument('--no_bias', action='store_false', dest='bias') 
    
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--block_size', type=int, default=8192)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--original', type=int, default=500)
    parser.add_argument('--inverse_t', type=float, default=1.0)
    parser.add_argument('--from_where', type=int, default=0)
    parser.add_argument('--pct', type=float, default=1.0) 
    parser.add_argument('--ramp', type=int, default=1)
    
    parser.add_argument('--is_from', action='store_true')
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--need_acc', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(VOCAB)
    indices = None
    
    if not is_from:  
        ub_csv_path = os.path.join(result_dir, f"orig{original}_rb{rope_base}_ub.csv")
        
        if os.path.exists(ub_csv_path):
            df_ub = pd.read_csv(ub_csv_path)
           
            match_row = df_ub[df_ub['pct'] == pct]
            if not match_row.empty:
             
                from_where = int(match_row['min_index'].iloc[0])
                print(f"pct={pct} best start, from_where={from_where}")
            else:
                print(f"can't find ub pct={pct}, default from_where={from_where}")
        else:
            print(f"can't find {ub_csv_path} path, default from_where={from_where}")

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
        
        ppls[ppl_test] = [None] * (from_where - 1)
        ood8_acc_list = [None] * (from_where - 1) if (pct < 2 and original <= 100) else []
        print(f"filled {from_where - 1} empty block to CSV")

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
            if need_acc and pct < 2 and original <= 100:
                ood8_acc_list.append(get_wrong_ans_acc(m0, ppl_test,
                                                    data_dir=data_dir,
                                                        block_size=block_size,
                                                        batch_size=100))

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
        if need_acc and pct < 2 and original <= 100:
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


    if is_save:
      
        suffix = "ub" if is_from else "lb"
        raw_csv_filename = f"raw_ppl_orig{original}_rb{rope_base}_{suffix}.csv"
        raw_csv_path = os.path.join(result_dir, raw_csv_filename) 
        
        if os.path.exists(raw_csv_path):
            df_raw = pd.read_csv(raw_csv_path, index_col=0)
        else:
            df_raw = pd.DataFrame()
            
      
        for k, v in ppls.items():
            df_raw[str(k)] = pd.Series(v)
            
       
        df_raw.to_csv(raw_csv_path, index=True) 
        print(f"pct={pct} added: {raw_csv_filename}")


        # if is_from:
        min_csv_filename = f"orig{original}_rb{rope_base}_{suffix}.csv"
        min_csv_path = os.path.join(result_dir, min_csv_filename)
        
        clean_list = [np.nan if x is None else x for x in ppls[ppl_test]]
        ppl_arr = np.array(clean_list, dtype=float)
        
        min_idx = int(np.nanargmin(ppl_arr)) 
        min_ppl = float(ppl_arr[min_idx])
        
        df_min = pd.DataFrame({
            "min_index": [min_idx],
            "min_ppl": [min_ppl]
        }, index=[pct])
        
        df_min.index.name = "pct"
        
        if not os.path.exists(min_csv_path):
            df_min.to_csv(min_csv_path, mode='w', header=True)
        else:
            df_min.to_csv(min_csv_path, mode='a', header=False)
            
        print(f"✅ pct={pct} (min_index={min_idx}, min_ppl={min_ppl}) added to {min_csv_filename}")
        
        if need_acc and len(ood8_acc_list) > 0:
            acc_csv_filename = f"raw_acc_orig{original}_rb{rope_base}_{suffix}.csv"
            acc_csv_path = os.path.join(result_dir, acc_csv_filename)
            
            if os.path.exists(acc_csv_path):
                df_acc = pd.read_csv(acc_csv_path, index_col=0)
            else:
                df_acc = pd.DataFrame()
                
            df_acc[str(ppl_test)] = pd.Series(ood8_acc_list)
                
            df_acc.to_csv(acc_csv_path, index=True) 
            print(f"✅ pct={pct} ACC added to: {acc_csv_filename}")