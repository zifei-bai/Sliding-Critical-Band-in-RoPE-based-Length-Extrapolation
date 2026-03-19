import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import copy

# 1. 模型与Tokenizer加载
# 使用 huggyllama/llama-7b 作为 LLaMA-1 的替代 (ctx=2048)
model_id = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 数据预处理函数
def get_c4_data(seq_len, num_samples=100):
    """
    从 allenai/c4 加载数据，并截取到指定长度。
    """
    print(f"Loading C4 data for length {seq_len}...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    input_ids_list = []
    count = 0

    for entry in dataset:
        text = entry['text']
        # 编码，不截断，先拿到完整token
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]

        # 只要长度足够的样本
        if tokens.shape[1] >= seq_len:
            # 截取前 seq_len 个 token
            input_ids_list.append(tokens[:, :seq_len])
            count += 1
            if count >= num_samples:
                break

    # 堆叠成一个 Batch: [num_samples, seq_len]
    return torch.cat(input_ids_list, dim=0)

def load_or_generate_data(seq_len, num_samples, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        return torch.load(cache_file)
    else:
        print(f"Cache not found. Generating data for length {seq_len}...")
        # 调用你之前定义的 get_c4_data 函数
        data = get_c4_data(seq_len, num_samples=num_samples)

        print(f"Saving data to {cache_file}...")
        torch.save(data, cache_file)
        return data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Extract and save Attention Matrices")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    result_dir = args.result_dir

    # 定义缓存文件名
    cache_file_2560 = "c4_data_2560.pt"
    cache_file_4096 = "c4_data_4096.pt"
    cache_file_8192 = "c4_data_8192.pt"

    test_file_path2560 = os.path.join(args.data_dir, cache_file_2560)
    test_file_path4096 = os.path.join(args.data_dir, cache_file_4096)
    test_file_path8192 = os.path.join(args.data_dir, cache_file_8192)
    

    # --- 执行加载 ---
    # 这样下次运行脚本时，只要目录里有 .pt 文件，就会瞬间完成
    data_2560 = load_or_generate_data(2560, 100, test_file_path2560)
    data_4096 = load_or_generate_data(4096, 100, test_file_path4096)
    data_8192 = load_or_generate_data(8192, 100, test_file_path8192)
