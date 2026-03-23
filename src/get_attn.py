import os
import argparse
import numpy as np
import torch
from model import GPT_RoPE, encode 
from config import VOCAB 

def make_idx(s_list: list, encode_fn, device):
    x_list = []
    for x_str in s_list:
        x_encoded = encode_fn(x_str)
        x_list.append(torch.tensor(x_encoded, dtype=torch.int64))
    x_tensor = torch.stack(x_list).to(device)
    return x_tensor

def get_attention_matrix(model, idx, layer, head):
    model.eval()
    with torch.no_grad():
        _, attn_list, _, _, _, _, _, _, _ = model.forward_with_cache(idx)
        att_matrix = attn_list[layer][:, head].mean(dim=0).cpu().float().numpy()
    return att_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and save Attention Matrices")
    parser.add_argument('--working_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--original', type=int, default=50)
    parser.add_argument('--pct', type=float, default=1.5)
    parser.add_argument('--rope_base', type=int, default=10000)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--block_size', type=int, default=8192)
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    block_size = args.block_size
    inverse_t = 1.0
    vocab_size = len(VOCAB) 
    
    ppl_test = int(args.original * args.pct)
    indss = [20, 63] 
    ramp = 1
    
    print(f"original={args.original}, pct={args.pct}, rope_base={args.rope_base}")

    data_path = os.path.join(args.data_dir, f"teach_force_{ppl_test}.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()][:100]

    idx_data = make_idx(data, encode_fn=encode, device=device)

    m0 = GPT_RoPE(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_layer=n_layer, 
                  n_head=n_head, dropout=dropout, inverse_t=inverse_t, rope_base=args.rope_base, 
                  start=0, end=96, gemma=1.0, indices=None, bias=True) 
    m0.to(device)
    
    checkpoint_path = os.path.join(args.working_dir, f"mlp_rope0_orig{args.original}_rb{args.rope_base}.pt")
    m0.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    for l in range(n_layer):
        m0.transformer.h[l].attn.set_head_pi_mask([True, True])
        m0.transformer.h[l].attn.enable_headwise_pi()
    m0.eval()

    target_heads = [(0, 0), (0, 1), (1, 0), (3, 1)]
    gemma_val = (args.original * 2 + 3) / (ppl_test * 2 + 3)
    attention_data_dict = {}

    print("Original Attention...")
    for l, h in target_heads:
        att_np = get_attention_matrix(m0, idx_data, layer=l, head=h)
        attention_data_dict[f"orig_L{l}H{h}"] = att_np

    print("Interpolated Attention...")
    for l in range(n_layer):
        m0.transformer.h[l].attn.set_gemma(gemma_val, indice=indss, ramp=ramp)
        
    for l, h in target_heads:
        att_np = get_attention_matrix(m0, idx_data, layer=l, head=h)
        attention_data_dict[f"interp_L{l}H{h}"] = att_np

    os.makedirs(args.result_dir, exist_ok=True)
    save_file = os.path.join(args.result_dir, f"attention_matrices_orig{args.original}_pct{args.pct}.npz")
    np.savez_compressed(save_file, **attention_data_dict)
    print(f"Raw result data saved to: {save_file}")