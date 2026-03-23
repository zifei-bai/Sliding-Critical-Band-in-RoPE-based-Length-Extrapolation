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
    """从 allenai/c4 加载数据，并截取到指定长度"""
    print(f"📥 正在从 Hugging Face 下载并处理 C4 数据 (长度: {seq_len})...")
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
    """如果有本地缓存则读取，没有则在线下载并处理"""
    if os.path.exists(cache_file):
        print(f"📦 发现本地数据缓存，正在加载: {cache_file}...")
        return torch.load(cache_file)
    else:
        print(f"🌐 未找到本地缓存。准备生成长度为 {seq_len} 的数据...")
        data = get_c4_data(tokenizer, seq_len, num_samples=num_samples)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        print(f"💾 保存数据缓存至: {cache_file}...")
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
    """将官方 RoPE 替换为我们的 Flexible RoPE"""
    target_rotary = None
    if hasattr(model.model, "rotary_emb"):
        target_rotary = model.model.rotary_emb
    elif hasattr(model.model.layers[0].self_attn, "rotary_emb"):
        target_rotary = model.model.layers[0].self_attn.rotary_emb
    else:
        raise AttributeError("找不到 rotary_emb 模块！")

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

# ==========================================
# 3. 评估函数
# ==========================================
def calculate_ppl(model, dataloader):
    """计算当前 DataLoader 内文本的困惑度 (Perplexity)"""
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

# ==========================================
# 4. 主程序入口
# ==========================================
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

    # 模型常数推断 (LLaMA-7B default)
    DIM_HEAD = 128
    NUM_FREQS = DIM_HEAD // 2 # 64

    # ------------------------------------------
    # Stage 1: 寻找高频截断点 k (即 UB)
    # ------------------------------------------
    print("\n" + "="*50)
    print("   Stage 1: Find UB (Upper Bound)")
    print("="*50)
    results_stage_1 = []
    
    for k in range(NUM_FREQS + 1):
        scale_map = torch.ones(NUM_FREQS).to(device)
        scale_map[:k] = 1.0               # [0, k) 保持原样 (Gold)
        scale_map[k:] = args.target_scale # [k, 64) 执行插值 (PI)

        apply_scaling_map(model, scale_map, args.seq_len)
        ppl = calculate_ppl(model, loader)
        results_stage_1.append(ppl)
        print(f"  Cutoff k={k}: PPL = {ppl:.4f}")

    best_k = torch.argmin(torch.tensor(results_stage_1)).item()
    print(f"Stage 1 Best Cutoff (UB): k = {best_k}")

    # ------------------------------------------
    # Stage 2: 探索低频/平滑窗口 (即 LB)
    # ------------------------------------------
    print("\n" + "="*50)
    print(f"   Stage 2: Find LB (Start from k={best_k})")
    print("="*50)
    results_stage_2 = []
    max_w = NUM_FREQS - best_k

    for w in range(max_w):
        scale_map = torch.ones(NUM_FREQS).to(device)
        scale_map[:best_k] = 1.0
        scale_map[best_k: best_k + w] = args.target_scale # 扫略插值窗口
        
        apply_scaling_map(model, scale_map, args.seq_len)
        ppl = calculate_ppl(model, loader)
        results_stage_2.append(ppl)
        print(f"  Window width w={w} (Indices {best_k} to {best_k+w}): PPL = {ppl:.4f}")

    # ------------------------------------------
    # 数据对齐与保存 CSV
    # ------------------------------------------
    print("\n📦 正在对齐并保存 CSV 数据...")
    
    # 前面补充 best_k 个 0.0 以对齐索引
    new_results_stage_2 = [0.0] * best_k + results_stage_2
    
    # 自动对齐长度保护机制
    if len(new_results_stage_2) < len(results_stage_1):
        new_results_stage_2 += [0.0] * (len(results_stage_1) - len(new_results_stage_2))

    df = pd.DataFrame({
        'PPL_ub': results_stage_1,
        'PPL_lb': new_results_stage_2
    })

    # 将占位的 0.0 统一替换为 NaN
    df = df.replace(0.0, np.nan)

    # 保存为 CSV
    csv_filename = f'real_ppl_ub_lb_{args.seq_len}.csv'
    save_path = os.path.join(args.result_dir, csv_filename)
    df.to_csv(save_path, index=False)
    
    print(f"🎉 真实场景验证完成！结果已完美对齐并保存至: {save_path}")



