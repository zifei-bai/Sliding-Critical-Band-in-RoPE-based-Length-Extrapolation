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
import os


from model import GPT_RoPE
from model import generate_greedy
from model import encode, decode

import argparse
import torch
from config import VOCAB, PADDING_TOKEN_INDEX, END_TOKEN_INDEX, DEVICE 





@torch.no_grad()
def estimate_loss(data, model, eval_iters, batch_size, block_size):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch_random(data, batch_size, block_size)
       
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['loss'] = losses.mean()
    model.train()
    return out



eval_iters = 10
@torch.no_grad()
def estimate_ppl(data, model, eval_iters, batch_size, block_size):
    model.eval()
    losses = torch.zeros(eval_iters)
    ppl_list, valid_count_list = [], []
    for k in range(eval_iters):
        X, Y = get_batch_random(data, batch_size, block_size) 
        ppl, valid_count = model.forward_with_ppl(X, Y)
        ppl_list.append(ppl)
        valid_count_list.append(valid_count)

    return sum(ppl_list)/len(ppl_list), sum(valid_count_list)/len(valid_count_list)


def set_seeds(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      


def get_batch(data, iters, batch_size, block_size):
    """data is combined dataset, get combined dataset in train loop"""
    i = int(iters % (len(data) // batch_size))
    final_sample = data[i*batch_size:(i+1)*batch_size]
   
    final_sample = [line.strip() for line in final_sample]

    x_list, y_list = [], []
    for x_str in final_sample:
        
        x_encoded = encode(x_str)
        len_xencoded = len(x_encoded)
        

        x_padded = x_encoded + [PADDING_TOKEN_INDEX] * (block_size - len(x_encoded))
        x_list.append(torch.tensor(x_padded, dtype=torch.int64))
        y_encoded = encode(x_str)[1:]
        
        y_padded = y_encoded + [PADDING_TOKEN_INDEX] * (block_size - len(y_encoded))

       
        y_list.append(torch.tensor(y_padded, dtype=torch.int64))

    x_tensor = torch.stack(x_list).to(DEVICE)
    y_tensor = torch.stack(y_list).to(DEVICE)
    return x_tensor, y_tensor


def get_batch_random(data, batch_size, block_size):
    """data is combined dataset, get combined dataset in train loop"""
    final_sample = random.sample(data, batch_size)
    final_sample = [line.strip() for line in final_sample]

    x_list, y_list = [], []
    for x_str in final_sample:
      
        x_encoded = encode(x_str)
        len_xencoded = len(x_encoded)
      

        x_padded = x_encoded + [PADDING_TOKEN_INDEX] * (block_size - len(x_encoded))
        x_list.append(torch.tensor(x_padded, dtype=torch.int64))
        y_encoded = encode(x_str)[1:]
       
        y_padded = y_encoded + [PADDING_TOKEN_INDEX] * (block_size - len(y_encoded))

     
        y_list.append(torch.tensor(y_padded, dtype=torch.int64))

    x_tensor = torch.stack(x_list).to(DEVICE)
    y_tensor = torch.stack(y_list).to(DEVICE)
    return x_tensor, y_tensor



def get_wrong_ans_acc(model, num_digits, data_dir, block_size, batch_size, wrong_file_path=None, correct_file_path=None):
    data=[]
  
    file_path = os.path.join(data_dir, f"test_{num_digits}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [line.strip() for line in data]
        data = data[:100]

    correct = 0
    correct_ans = []
    num_batches = len(data) // batch_size
    print(f"There are {num_batches} batches")
    
    wrong = 0
    wrong_ans = []
    for x in range(num_batches):
        prompts = data[x*batch_size : (x+1)*batch_size]

        context = torch.tensor([encode(inp) for inp in prompts], dtype=torch.long, device=DEVICE)

      
        output_batch = generate_greedy(model=model, idx=context, max_new_tokens=block_size)

        targets = [p + p[1:-1] + "&" for p in prompts]


        for output, target in zip(output_batch, targets):
            if output == target:
                correct_ans.append(f"{output}")
                correct += 1
            else:
                wrong_ans.append(f"{output}")
                wrong += 1

        if wrong_file_path is not None and correct_file_path is not None:
            with open(wrong_file_path, "w") as f:
                f.write("\n".join(wrong_ans))

            with open(correct_file_path, "w") as f:
                f.write("\n".join(correct_ans))

    acc = correct / len(data)
    print(f"Accuracy for {num_digits} digits: {acc}")
    print("----------------------------------")
   
    print("----------------------------------")
    return acc




def create_optimizer_and_scheduler(model, total, warm, decay, lr):
    # AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,              # learning rate
        betas=(0.9, 0.99),
        eps=1e-12,
        weight_decay=0.1
    )

    # LR Scheduler
    total_steps = total 
    warmup_steps = warm
    decay_steps = decay
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  
        elif step < warmup_steps + stable_steps:
            return 1.0               
        else:
           
            decay_ratio = (step - warmup_steps - stable_steps) / decay_steps
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler