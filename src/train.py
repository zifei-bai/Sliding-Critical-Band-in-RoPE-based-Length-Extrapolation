import os
import argparse
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
import wandb
from datetime import datetime

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config import VOCAB, PADDING_TOKEN_INDEX, END_TOKEN_INDEX, DEVICE
from model import GPT_RoPE
from utilities import get_wrong_ans_acc, get_batch, estimate_loss, create_optimizer_and_scheduler, set_seeds

def parse_args():
    parser = argparse.ArgumentParser(description="RoPE Model Training Script")
    
    # path config
    parser.add_argument('--working_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    
    # model arch config
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rope_base', type=int, default=10000)
    parser.add_argument('--no_bias', action='store_false', dest='bias')
    
    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--max_iters', type=int, default=9000)
    parser.add_argument('--warm', type=int, default=1000)
    parser.add_argument('--train_decay', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--inverse_t', type=float, default=1.0)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    # experiment config
    parser.add_argument('--original', type=int, default=100)
    
    return parser.parse_args()

if __name__ == '__main__':
   
    args = parse_args()
    
    batch_size = args.batch_size
    block_size = args.block_size
    max_iters = args.max_iters
    warm = args.warm
    train_decay = args.train_decay
    lr = args.lr
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    inverse_t = args.inverse_t
    rope_base = args.rope_base
    original = args.original
    bias = args.bias
    eval_iters = args.eval_iters
    working_dir = args.working_dir
    data_dir = args.data_dir
    seed_num = args.seed


    os.makedirs(working_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(VOCAB)

    print(f"original is {original}")
    print(f"rope_base is {rope_base}")
    print("------------------------------")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wandb.init(project="sliding_critical_band",
              config={
                "batch_size": batch_size,
                "block_size": block_size,
                "optimizer": "AdamW",
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "max_iter": max_iters,
                "warm_up": warm,
                "decay": train_decay,
                "lr": lr,
                "seed": seed_num,
                },
              name= f"base 1-{original}_{timestamp}"
    )
    
    set_seeds(seed_num)
    print(f"Start run pretrain train loop with {max_iters} steps and {warm} warm, {train_decay} decay, {lr} learning rate")

    
    data_path = os.path.join(data_dir, f"origin_ds_bos_{original}.txt")
    print(f"📥 Loading data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    
    model = GPT_RoPE(vocab_size, block_size, n_embd, n_layer, n_head, dropout, 
                     inverse_t=inverse_t, rope_base=rope_base, start=0, end=0, 
                     gemma=1, indices=None, bias=bias)
    model = model.to(device)
    

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print(f'Model config: {model.n_layer} layers, {model.n_head} heads, {model.n_embd} embeddings,\n'
          f'dropout is {model.dropout}, bias is {model.bias}, inverse_t is {model.inverse_t}, rope_base is {model.rope_base}\n'
          f'block size {model.block_size}, batch size {batch_size}')
    print("------------------------------")

    optimizer, scheduler = create_optimizer_and_scheduler(model, max_iters, warm, train_decay, lr)
    scaler = GradScaler('cuda')
    loss_list = []

  
    for iter in tqdm(range(max_iters), desc="Training Progress"):
        
      
        if iter % 500 == 0 or iter == max_iters - 1:
            model.eval()
            with torch.no_grad():
                losses1 = estimate_loss(data, model, eval_iters=10, batch_size=batch_size, block_size=block_size)['loss']
            print(f"step {iter}: loss {losses1:.4f}")
            log_dict = {"Loss": losses1}
            loss_list.append(round(losses1.item(), 4))
            wandb.log(log_dict)
            
        if iter % 3000 == 0 and iter > 0:
            model.eval()
            with torch.no_grad():
                acc7 = get_wrong_ans_acc(model, original, data_dir=data_dir,
                                         block_size=block_size, batch_size=100)
               

   
        model.train()
        xb, yb = get_batch(data, iters=iter, batch_size=batch_size, block_size=block_size)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits1, loss1 = model(xb, yb)

   
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss1).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

 
    print(f"Training finished for pretrain.\nEvaluating {original}-digit accuracy...")
    model.eval()
    with torch.no_grad():
        acc7 = get_wrong_ans_acc(model, original, block_size=block_size, data_dir=data_dir, batch_size=100)
    print(f"Average accuracy: {acc7}")

    filename = f"mlp_rope0_orig{original}_rb{rope_base}_{timestamp}.pt"
    save_path = os.path.join(working_dir, filename)
    print(f"Saving final model to {save_path}")
    torch.save(model.state_dict(), save_path)

    print("----------------------------------")
    wandb.finish()