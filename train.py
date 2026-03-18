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
import argparse

from config import VOCAB, PADDING_TOKEN_INDEX, END_TOKEN_INDEX, DEVICE
from model import GPT_RoPE
# from model import generate_greedy
from utilities import get_wrong_ans_acc, get_batch, get_batch_random, estimate_loss, estimate_ppl, create_optimizer_and_scheduler, set_seeds

def parse_args():
    parser = argparse.ArgumentParser(description="RoPE Model Training Script")
    
    # 路径参数
    parser.add_argument('--working_dir', type=str, default="/content/drive/MyDrive/Research/RoPE_NoPE/analyze_1120/")
    parser.add_argument('--data_dir', type=str, default="/content/drive/MyDrive/Research/train_test_data/")
    
    # 模型架构参数
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rope_base', type=int, default=10000)
    parser.add_argument('--no_bias', action='store_false', dest='bias')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--warm', type=int, default=0)
    parser.add_argument('--train_decay', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--inverse_t', type=float, default=1.0)
    parser.add_argument('--eval_iters', type=int, default=10)
    
    # 实验与数据参数
    parser.add_argument('--original', type=int, default=500)
    
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# --- 为了不修改你后面的代码，这里将 args 拆解回你原来的变量名 ---
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

# 动态推断和无需外部传入的变量保留原样
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(VOCAB) # 注意这里改成了从 config 引入的大写 VOCAB
indices = None


# This is a base training loop for producing base model
train_digits = [25, 50, 100, 200, 500, 1000]
for i in range(1):
    original = 500
    rope_base = 100000
    print(f"original is {original}")
    print(f"rope_base is {rope_base}")
    print("------------------------------")
    wandb.init(project="research",
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
                },
              name= f"base 1-{original}"
    )
    set_seeds(47+i)
    print(f"Start run pretrain train loop with {max_iters} steps and {warm} warm, {train_decay} decay, {lr} learning rate")

    data = []
    # INITIALIZE MODEL, OPTIMIZER, SHCEDULER
    model = GPT_RoPE(vocab_size, block_size, n_embd, n_layer, n_head, dropout, inverse_t=inverse_t, rope_base=rope_base, start=0, end=0, gemma=1, indices = None, bias=True)
    model = model.to(device)
    working_dir = "/content/drive/MyDrive/Research/RoPE_NoPE/analyze_1120/"

    checkpoint_path = f"{working_dir}mlp_rope{0}_orig{original}_rb{rope_base}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    with open(f"{data_dir}origin_ds_bos_{original}.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    optimizer, scheduler = create_optimizer_and_scheduler(model, max_iters, warm, train_decay, lr)

    # TRAINNG LOOP:
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print(f'model has {model.n_layer} layers, {model.n_head} heads, {model.n_embd} embeddings,\n use {model.dropout}, bias is {model.bias}, inverse_t is {model.inverse_t}, rope_base is {model.rope_base}')
    print(f'block size {model.block_size}, batch size {batch_size}')
    print("------------------------------")
    loss_list = []

    scaler = GradScaler('cuda')
    for iter in tqdm(range(max_iters), desc="Training Progress"):
        # sample a batch of data
        # every once in a while evaluate the loss on train and val sets
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
                acc7 = get_wrong_ans_acc(model, original,
                                         block_size=block_size, batch_size=100)
                # acc8 = get_wrong_ans_acc(model, original+200, 100)

        model.train()
        xb, yb = get_batch(data, iters=iter, batch_size=batch_size, block_size=block_size)

        # evaluate the loss
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits1, loss1 = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss1).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

    print(f"Training finished for pretrain.\nEvaluating {original}-digit accuracy...")
    # evaluate final performance on digit addition
    model.eval()
    with torch.no_grad():
        acc7 = get_wrong_ans_acc(model, original, data_dir=data_dir,
                         block_size=block_size, batch_size=100)
    print(f"Average accuracy: {acc7}")

    filename = f"mlp_rope{i}_orig{original}_rb{rope_base}.pt"
    save_path = f"{working_dir}{filename}"
    torch.save(model.state_dict(), save_path)


    # print(f"Training finished for pretrain.\nEvaluating {original+2}-digit accuracy...")
    # model.eval()
    # with torch.no_grad():
    #     acc9 = get_wrong_ans_acc(model, 17, batch_size)
    # print(f"Average accuracy: {acc9}")

    # print(f"Training finished for pretrain.\nEvaluating {original+3}-digit accuracy...")
    # model.eval()
    # with torch.no_grad():
    #     acc10 = get_wrong_ans_acc(model, 10, batch_size)
    # print(f"Average accuracy: {acc10}")



    print("----------------------------------")
    print("----------------------------------")

    wandb.finish()