import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def get_c4_data(tokenizer, seq_len, num_samples=100):
    """Load allenai/c4 data, truncate to length"""
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    input_ids_list = []
    count = 0
    for entry in dataset:
        text = entry['text']
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        if tokens.shape[1] >= seq_len:
            input_ids_list.append(tokens[:, :seq_len])
            count += 1
            if count >= num_samples:
                break
    return torch.cat(input_ids_list, dim=0)

def load_or_generate_data(tokenizer, seq_len, num_samples, cache_file):
    if os.path.exists(cache_file):
        print(f"Local data found: {cache_file}...")
        return torch.load(cache_file)
    else:
        print(f"Can;t find data, processing {seq_len} data...")
        data = get_c4_data(tokenizer, seq_len, num_samples=num_samples)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        print(f"Cached data saved to: {cache_file}...")
        torch.save(data, cache_file)
        return data


class FlexibleRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, custom_scale_map=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        if custom_scale_map is not None:
            inv_freq = inv_freq / custom_scale_map.to(device)
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids=None, seq_len=None):
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max() + 1
            else:
                seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype)
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        return cos, sin

def apply_scaling_map(model, scale_map, max_len):
    target_rotary = None
    if hasattr(model.model, "rotary_emb"):
        target_rotary = model.model.rotary_emb
    elif hasattr(model.model.layers[0].self_attn, "rotary_emb"):
        target_rotary = model.model.layers[0].self_attn.rotary_emb
    else:
        raise AttributeError("Can't find RoPE block")

    derived_dim = target_rotary.inv_freq.shape[0] * 2
    derived_base = getattr(target_rotary, "base", 10000.0)

    new_rotary = FlexibleRotaryEmbedding(
        dim=derived_dim, max_position_embeddings=max_len, base=derived_base,
        device=model.device, custom_scale_map=scale_map
    ).to(model.dtype)

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = new_rotary
    else:
        for layer in model.model.layers:
            layer.self_attn.rotary_emb = new_rotary

def calculate_ppl(model, dataloader):
    model.eval()
    total_nll, total_tokens = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating PPL", leave=False):
            input_ids = batch[0].to(model.device)
            outputs = model(input_ids, labels=input_ids)
            total_nll += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    avg_nll = total_nll / total_tokens
    return torch.exp(torch.tensor(avg_nll)).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-End Real Validation for LLaMA RoPE")
    parser.add_argument('--model_id', type=str, default="huggyllama/llama-7b")
    parser.add_argument('--data_dir', type=str, default="./data", help="cached data and training data")
    parser.add_argument('--result_dir', type=str, default="./results", help="result files")
    parser.add_argument('--seq_len', type=int, default=8192)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--target_scale', type=float)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nInitializing Tokenizer and Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    
    cache_file = os.path.join(args.data_dir, f"c4_data_{args.seq_len}.pt")
    data_tensor = load_or_generate_data(tokenizer, args.seq_len, 100, cache_file)
    
    dataset = TensorDataset(data_tensor[:args.num_samples]) 
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    DIM_HEAD = 128
    NUM_FREQS = DIM_HEAD // 2 # 64

    print("\n" + "="*50)
    print("Find UB (Upper Bound)")
    print("="*50)
    results_stage_1 = []
    
    for k in range(NUM_FREQS + 1):
        scale_map = torch.ones(NUM_FREQS).to(device)
        scale_map[:k] = 1.0               
        scale_map[k:] = args.target_scale

        apply_scaling_map(model, scale_map, args.seq_len)
        ppl = calculate_ppl(model, loader)
        results_stage_1.append(ppl)
        print(f"Cutoff k={k}: PPL = {ppl:.4f}")

    best_k = torch.argmin(torch.tensor(results_stage_1)).item()
    print(f"Best UB: k = {best_k}")


    print("\n" + "="*50)
    print(f"Find LB (Start from k={best_k})")
    print("="*50)
    results_stage_2 = []
    max_w = NUM_FREQS - best_k

    for w in range(max_w):
        scale_map = torch.ones(NUM_FREQS).to(device)
        scale_map[:best_k] = 1.0
        scale_map[best_k: best_k + w] = args.target_scale
        
        apply_scaling_map(model, scale_map, args.seq_len)
        ppl = calculate_ppl(model, loader)
        results_stage_2.append(ppl)
        print(f"  Window width w={w} (Indices {best_k} to {best_k+w}): PPL = {ppl:.4f}")

    print("\nData saved...")
    
    new_results_stage_2 = [0.0] * best_k + results_stage_2
    
    if len(new_results_stage_2) < len(results_stage_1):
        new_results_stage_2 += [0.0] * (len(results_stage_1) - len(new_results_stage_2))

    df = pd.DataFrame({
        'PPL_ub': results_stage_1,
        'PPL_lb': new_results_stage_2
    })

    df = df.replace(0.0, np.nan)

    # 保存为 CSV
    csv_filename = f'real_ppl_ub_lb_{args.seq_len}.csv'
    save_path = os.path.join(args.result_dir, csv_filename)
    df.to_csv(save_path, index=False)
    
    print(f"Raw reuslt data saved to: {save_path}")



