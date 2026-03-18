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

from config import VOCAB, PADDING_TOKEN_INDEX, END_TOKEN_INDEX, DEVICE

# create a mapping from chars to ints
stoi = {ch:i for i, ch in enumerate(VOCAB)}
itos = {i:ch for i, ch in enumerate(VOCAB)}
encode = lambda s:[stoi[c] for c in s] # encoder: take a string, output a list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of ints, output a string

print(encode("@12=12&"))
print(decode(encode("@12=12&")))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False): # class constructor
        super().__init__()
        # nn.Parameter, pytorch optimize will update the value of this parameter during training
        self.weight = nn.Parameter(torch.ones(ndim)) # trainable parameter
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None # trainable parameter

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)

class RotaryEmbedding(nn.Module):
    """
    Full RoPE on all head_dim (must be even).
    base: usually 10000.0 ; can tweak for NTK scaling.
    """
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.base = base
        # 预计算各维的频率 inv_freq: [D/2]
        t = torch.arange(0, head_dim, 2).float()  # 0,2,4,...

        inv_freq = 1.0 / (base ** (t / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        angle_scale = torch.ones_like(inv_freq)
        self.register_buffer("angle_scale", angle_scale, persistent=False)


    def set_scale_indices(self, indices, gamma: float, ramp: int = 0):
        """
        将给定频率平面的 angle_scale 调整为带可选“平滑过渡”的 PI。
        - 若 ramp <= 0，或者 indices 不是形如 [a, b] 的两个端点：
            直接把 indices 对应的 index 的 angle_scale 设为 gamma（老逻辑）。
        - 若给定两个端点 [a, b] 且 ramp > 0：
            [a, b] 区间完全 PI；
            [a-ramp, a) 线性从 0→1 过渡；
            (b, b+ramp] 线性从 1→0 过渡。
          其它平面保持原值不变。
          这里 [a, b] 可以退化成单点（a == b），比如 [0, 0]、[95, 95]。
        """
        device = self.angle_scale.device
        gamma = float(gamma)

        # 统一成 Python list + 1D long tensor
        if isinstance(indices, int):
            idx_list = [indices]
        elif isinstance(indices, torch.Tensor):
            idx_list = indices.tolist()
        else:
            idx_list = list(indices)

        idx = torch.as_tensor(idx_list, dtype=torch.long, device=device)

        P = self.angle_scale.numel()
        ramp = int(ramp)

        # =========== 情况 1：不使用平滑过渡（老逻辑） ===========
        # 条件：
        #   - ramp <= 0
        #   - 或者 idx 不是两个元素（一个 index 或多于两个）
        if ramp <= 0 or len(idx_list) != 2:
            s = self.angle_scale.clone()
            s[idx.clamp_(0, P - 1)] = gamma   # 防越界
            self.angle_scale = s
            return

        # =========== 情况 2：两个端点 + 平滑过渡（a 和 b 可以相等） ===========
        a = min(idx_list)
        b = max(idx_list)

        # 边界裁剪
        a = max(0, min(P - 1, a))
        b = max(0, min(P - 1, b))

        # α 权重：默认 0（不 PI）
        alpha = torch.zeros(P, device=device)

        # 中心区间 [a, b]：α = 1
        # 注意：当 a == b 时，这里就是只把第 a 个平面设为 1
        alpha[a:b+1] = 1.0

        # 左侧过渡 [a - ramp, a)：0 -> 1
        left_end = a - 1
        if left_end >= 0:
            left_start = max(0, a - ramp)
            if left_start <= left_end:
                L = left_end - left_start + 1
                # 在 [left_start, a) 上从 0 递增到接近 1（但 <1，中心 a 自己是 1）
                vals = torch.linspace(0.0, 1.0, steps=L + 1, device=device)[:-1]
                alpha[left_start:a] = vals

        # 右侧过渡 (b, b + ramp]：1 -> 0
        right_start = b + 1
        if right_start < P:
            right_end = min(P - 1, b + ramp)
            if right_start <= right_end:
                L = right_end - right_start + 1
                # 在 (b, right_end] 上从接近 1 递减到 0
                vals = torch.linspace(1.0, 0.0, steps=L + 1, device=device)[1:]
                alpha[right_start:right_end + 1] = vals

        # 用 α 在「原 angle_scale」和「gamma」之间线性插值：
        #   alpha = 0  → 原值不变
        #   alpha = 1  → 完全变为 gamma
        orig = self.angle_scale
        self.angle_scale = orig * (1.0 - alpha) + gamma * alpha
        # print(f"ALPHA {alpha}")


    # def set_scale_indices(self, indices, gamma: float):
    #     """
    #     将给定频率平面（基于 [0, D/2) 的索引）统一乘以 gamma。
    #     indices: 可为 int、list[int]、torch.LongTensor
    #     gamma: 缩放系数（角度乘子）
    #     """
    #     if isinstance(indices, int):
    #         indices = [indices]
    #     idx = torch.as_tensor(indices, dtype=torch.long, device=self.inv_freq.device)
    #     s = self.angle_scale.clone()
    #     s[idx] = float(gamma)
    #     self.angle_scale = s

    def set_scale_slice(self, start: int, end: int, gamma: float):
        """
        将频率区间 [start, end) 统一乘以 gamma。
        """
        s = self.angle_scale.clone()
        s[start:end] = float(gamma)
        self.angle_scale = s

    def _cos_sin(self, seq_len: int, device, dtype):
        """
        返回 cos, sin 形状 [seq_len, head_dim]，已按偶/奇位展开（便于直接广播到 (..., T, Dh)）
        """
        # 位置 p: [T, 1] ; 频率: [1, D/2]
        # pos = torch.arange(seq_len, device=device).float().unsqueeze(1)     # [T,1]
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # 位置 m: [T,1]，从 0 到 T-1
        # 把每个位置乘以每个频率（含 angle_scale），得到相位矩阵 [T, D/2]
        freqs = pos * self.inv_freq.unsqueeze(0)
        cos = torch.cos(freqs).to(dtype)                      # cos 部分 [T, D/2]
        sin = torch.sin(freqs).to(dtype)                      # sin 部分 [T, D/2]
        # 把 [D/2] 交错复制到 [D]：偶位放 cos[i]，奇位也放 cos[i]（后面与 rotate_half 组合成复数旋转）
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, -1)  # [T, D]
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, -1)  # [T, D]
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        """
        把 (..., D) 的向量做 (x_even, x_odd) -> (-x_odd, x_even) 的 90° 旋转辅助。
        等价于把每对 (2t,2t+1) 变换为 (-y, x)。
        """
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        x_rot_even = -x_odd
        x_rot_odd  =  x_even
        # 重新交错回原布局
        return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)

    def apply_rotary(self, x: torch.Tensor, seq_len: int):
        """
        x: [B, H, T, Dh] 或 [*, T, Dh]，最后两维是 (T, Dh)
        返回：同形状旋转后的张量
        """
        B = x.shape[0]
        T = x.shape[-2]
        Dh = x.shape[-1]
        assert Dh == self.head_dim

        cos, sin = self._cos_sin(T, x.device, x.dtype)         # [T, Dh]
        # 广播到 [B,H,T,Dh] / [*,T,Dh]
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        # 公式：rot(x) = x * cos + rotate90(x) * sin
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin


    def _cos_sin_with_plane_scale(self, positions: torch.Tensor, s_plane: torch.Tensor, device, dtype):
        # positions: [T], s_plane: [D/2]
        freqs = positions.to(device, torch.float32).unsqueeze(1) \
              * self.inv_freq.to(device, torch.float32).unsqueeze(0) \
              * s_plane.to(device, torch.float32).unsqueeze(0)                      # [T, D/2]
        cos = torch.cos(freqs).to(dtype); sin = torch.sin(freqs).to(dtype)
        T = positions.numel()
        cos = torch.stack([cos, cos], dim=-1).reshape(T, -1)                        # [T, D]
        sin = torch.stack([sin, sin], dim=-1).reshape(T, -1)
        return cos, sin

    def apply_rotary_pair_headwise_pi(self, q: torch.Tensor, k: torch.Tensor,
                                      head_pi_mask: torch.Tensor,
                                      *,           # 仅限关键字参数，防误用
                                      use_length_ratio: bool = False,
                                      length_ratio: float = 1.0):
        """
        q,k           : [B,H,T,Dh]
        head_pi_mask  : [H]，True 的 head 走 PI；False 的 head 走普通 RoPE
        use_length_ratio: 是否再乘 (L_train/L_target)
        length_ratio  : 上面的数值；只有 use_length_ratio=True 时才使用
        规则：
          - 普通 head：用 m（不乘 gemma）
          - PI 的 head：未被选平面 => 用 m（不乘 gemma）；被选平面 => 用 m * (gemma) * (length_ratio 可选)
            这里“被选平面”的 gemma 由 set_scale_slice/indices 写进 self.angle_scale
        """
        B, H, T, Dh = q.shape
        device, dtype = q.device, q.dtype
        assert Dh == self.head_dim

        # 普通 cos/sin（所有平面尺度=1）
        ones = torch.ones_like(self.angle_scale)                                     # [D/2]
        m = torch.arange(T, device=device, dtype=torch.float32)                      # [T]
        cos_n, sin_n = self._cos_sin_with_plane_scale(m, ones, device, dtype)        # [T,D]

        # PI cos/sin：每平面尺度 = angle_scale(=gemma on selected planes, else 1) * (length_ratio if enabled)
        if use_length_ratio:
            s_plane = self.angle_scale * float(length_ratio)                         # [D/2]
        else:
            s_plane = self.angle_scale
        cos_i, sin_i = self._cos_sin_with_plane_scale(m, s_plane, device, dtype)     # [T,D]

        # 广播
        def _b(t):
            while t.dim() < q.dim(): t = t.unsqueeze(0)
            return t
        cos_n_b, sin_n_b = _b(cos_n), _b(sin_n)
        cos_i_b, sin_i_b = _b(cos_i), _b(sin_i)

        # 先给所有 head 用“普通 m”
        q_rot = self._rotate_half(q); k_rot = self._rotate_half(k)
        q_out = q * cos_n_b + q_rot * sin_n_b
        k_out = k * cos_n_b + k_rot * sin_n_b

        # 再覆盖需要 PI 的 head
        if head_pi_mask is not None and head_pi_mask.any():
            mask = head_pi_mask.to(device=device, dtype=torch.bool)
            q_sel, k_sel = q[:, mask, :, :], k[:, mask, :, :]
            q_sel_rot, k_sel_rot = self._rotate_half(q_sel), self._rotate_half(k_sel)
            q_pi = q_sel * cos_i_b + q_sel_rot * sin_i_b
            k_pi = k_sel * cos_i_b + k_sel_rot * sin_i_b
            q_out[:, mask, :, :] = q_pi
            k_out[:, mask, :, :] = k_pi
        return q_out, k_out


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, inverse_t, rope_base, start, end, gemma, indices, bias=True):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by the number of heads."

        # Store hyperparameters
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.block_size = block_size
        self.inverse_t = inverse_t
        self.rope_base = rope_base

        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if rope_base is not None:
            self.rope = RotaryEmbedding(n_embd // n_head, rope_base)
            F = n_embd // n_head // 2
            # self.rope.set_scale_slice(start=start, end=end, gamma=gemma)
            if indices is None:
                self.rope.set_scale_slice(start=start, end=end, gamma=gemma)
                # self.rope.set_scale_slice(start=32, end=96, gamma=0)
            else:
                self.rope.set_scale_indices(indices=indices, gamma=gemma)

                # Check for Flash Attention availability
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for slow attention

        self.register_buffer(
        "casual_mask_full",
        torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1, 1, block_size, block_size),
        persistent=False
        )

        self.register_buffer("head_pi_mask", torch.zeros(n_head, dtype=torch.bool), persistent=False)
        self._pi_enabled = False
        self._pi_use_length_ratio = False         ### NEW: 是否乘长度比（默认 False=只用 gemma）
        self._pi_L_train = block_size
        self._pi_L_target = None


    def set_gemma(self, gemma: float, indice, ramp):
        self.rope.set_scale_indices(indices=indice, gamma=gemma, ramp=ramp)


    def set_head_pi_mask(self, mask_bool_list_or_tensor):
        m = mask_bool_list_or_tensor
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, dtype=torch.bool, device=self.head_pi_mask.device)  # 放到相同设备
        assert m.numel() == self.n_head
        self.head_pi_mask.copy_(m)                                                 # 原地更新 buffer

    # 开启 head-wise PI，并可设置 L_train/L_target
    def enable_headwise_pi(self, L_train: int = None, L_target: int = None):
        if L_train is not None:
            self._pi_L_train = int(L_train)       # 比例分子
        if L_target is not None:
            self._pi_L_target = int(L_target)     # 比例分母
        self._pi_enabled = True                   # 打开 PI 开关

    # 关闭 head-wise PI（恢复普通 RoPE）
    def disable_headwise_pi(self):
        self._pi_enabled = False
        self._pi_L_target = None



    def forward(self, x, use_t, return_cache):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # Split into Q, K, V (B, T, n_embd)

        H = self.n_head
        head_dim = C // H

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)

        if self._pi_enabled and self.head_pi_mask.any():
                # 计算是否使用长度比
                use_len = self._pi_use_length_ratio and (self._pi_L_target is not None)
                length_ratio = (float(self._pi_L_train) / float(self._pi_L_target)) if use_len else 1.0
                q, k = self.rope.apply_rotary_pair_headwise_pi(
                    q, k,
                    head_pi_mask=self.head_pi_mask,
                    use_length_ratio=use_len,
                    length_ratio=length_ratio
                )
        else:
            q = self.rope.apply_rotary(q, T)
            k = self.rope.apply_rotary(k, T)

        if return_cache:
            # ----- 手动注意力：拿到权重 -----
            # (B, H, T, T)
            att_raw = torch.matmul(q, k.transpose(-2, -1)) * (self.inverse_t / math.sqrt(head_dim))

            # causal mask（只取 [:, :, :T, :T]）
            mask = self.casual_mask_full[:, :, :T, :T]
            att = att_raw.masked_fill(~mask, float('-inf'))

            att = F.softmax(att, dim=-1)  # 注意力权重
            # 可视化时一般不加 dropout，保证权重干净
            y = torch.matmul(att, v)      # (B, H, T, head_dim)

        # Flash Attention or fallback to manual implementation
        elif self.flash and not use_t:
            # print("flash Attn")
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # assert False
            att = torch.matmul(q, k.transpose(-2, -1)) * (self.inverse_t / math.sqrt(head_dim))
            # causal mask（只取 [:, :, :T, :T]）
            mask = self.casual_mask_full[:, :, :T, :T]
            att = att.masked_fill(~mask, float('-inf'))

            att = F.softmax(att, dim=-1)  # 注意力权重
            y = torch.matmul(att, v)      # (B, H, T, head_dim)

        # Reshape back to original format
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble heads

        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))

        if return_cache:
            return y, att, q, k, v, att_raw
        else:
            return y



    # @torch.no_grad()
    # def analyze_plane_usage_norms(
    #     self,
    #     x: torch.Tensor,
    #     *,
    #     heads: int | list[int] | torch.Tensor | None = None,
    #     agg: str = "mean",            # "mean" 或 "sum"：跨 (B,T) 的聚合
    #     combine_qk: str = "product",  # "product"= E||q|| * E||k||；"sum"= E||q|| + E||k||
    #     per_head: bool = False        # True: 额外返回 (H_sel, P) 的每head曲线
    # ):
    #     """
    #     频率依赖分析（不改动模型原逻辑）：
    #       - 把 head_dim 拆成 P 个 RoPE 平面 (Dh=2P)，计算每平面的 ||q||_2 与 ||k||_2。
    #       - 与论文一致，用每平面的范数（或 q/k 范数的乘积）作为“频率使用度”的 proxy。
    #       - 返回 (P,) 的曲线（必要时也返回 (H_sel,P)）。

    #     输入:
    #       x: (B,T,C) —— 注意力前的输入，建议传该层的 att_in = block.ln_1(x)
    #       heads: 统计哪些 head（None=全部）
    #       agg: 跨 (B,T) 的聚合方式："mean" 或 "sum"
    #       combine_qk:
    #          - "product": usage_p := (聚合后的 q_norm_p) * (聚合后的 k_norm_p)  [常用]
    #          - "sum":     usage_p := (聚合后的 q_norm_p) + (聚合后的 k_norm_p)
    #       per_head: True 时额外返回每个 head 的 (P,) 曲线

    #     返回:
    #       dict:
    #         'usage': (P,)                 # 归一化到和=1
    #         'q_norm': (P,)                # 归一化前的聚合 q_norm（按 agg 聚合且已在 head 上合并）
    #         'k_norm': (P,)                # 同上
    #         可选 'usage_per_head': (H_sel,P)  # 每个 head 的曲线（各行独立归一化到和=1）
    #     """
    #     self.eval()
    #     B, T, C = x.shape
    #     H = self.n_head
    #     Dh = C // H
    #     assert Dh % 2 == 0, "head_dim 必须为偶数（每两维为 1 个 RoPE 平面）"
    #     P = Dh // 2

    #     # 线性投影
    #     q, k, _ = self.c_attn(x).split(self.n_embd, dim=2)
    #     q = q.view(B, T, H, Dh).transpose(1, 2).contiguous()  # (B,H,T,Dh)
    #     k = k.view(B, T, H, Dh).transpose(1, 2).contiguous()

    #     # 与 forward 保持一致地施加 RoPE/PI（范数不变，但保持一致更稳妥）
    #     if self._pi_enabled and self.head_pi_mask.any():
    #         use_len = self._pi_use_length_ratio and (self._pi_L_target is not None)
    #         length_ratio = (float(self._pi_L_train) / float(self._pi_L_target)) if use_len else 1.0
    #         q, k = self.rope.apply_rotary_pair_headwise_pi(
    #             q, k,
    #             head_pi_mask=self.head_pi_mask,
    #             use_length_ratio=use_len,
    #             length_ratio=length_ratio
    #         )
    #     else:
    #         q = self.rope.apply_rotary(q, T)
    #         k = self.rope.apply_rotary(k, T)

    #     # 选择要统计的 head
    #     if heads is None:
    #         hmask = torch.arange(H, device=x.device)
    #     else:
    #         hmask = torch.as_tensor(heads, device=x.device)
    #     q = q[:, hmask, :, :]   # (B,H_sel,T,Dh)
    #     k = k[:, hmask, :, :]
    #     H_sel = q.shape[1]

    #     # reshape -> (B,H_sel,T,P,2)
    #     q2 = q.view(B, H_sel, T, P, 2)
    #     k2 = k.view(B, H_sel, T, P, 2)

    #     # 每个平面的 2-范数： (B,H_sel,T,P)
    #     qn = torch.linalg.vector_norm(q2, ord=2, dim=-1)
    #     kn = torch.linalg.vector_norm(k2, ord=2, dim=-1)

    #     # 跨 (B,T) 聚合
    #     if agg == "mean":
    #         qn_bt = qn.mean(dim=(0, 2))  # (H_sel, P)
    #         kn_bt = kn.mean(dim=(0, 2))  # (H_sel, P)
    #     elif agg == "sum":
    #         qn_bt = qn.sum(dim=(0, 2))
    #         kn_bt = kn.sum(dim=(0, 2))
    #     else:
    #         raise ValueError("agg must be 'mean' or 'sum'.")

    #     # 合并 q/k
    #     if combine_qk == "product":
    #         usage_by_head = qn_bt * kn_bt   # (H_sel, P)
    #     elif combine_qk == "sum":
    #         usage_by_head = qn_bt + kn_bt
    #     else:
    #         raise ValueError("combine_qk must be 'product' or 'sum'.")

    #     # 合并 head -> (P,)
    #     usage = usage_by_head.sum(dim=0)

    #     # 统一归一化到和=1
    #     eps = torch.finfo(usage.dtype).eps
    #     usage = usage / (usage.sum() + eps)

    #     # 同时给出聚合后的 q/k（先按 head 合并，再各自归一化，便于诊断）
    #     qn_merged = qn_bt.sum(dim=0)
    #     kn_merged = kn_bt.sum(dim=0)
    #     qn_merged = qn_merged / (qn_merged.sum() + eps)
    #     kn_merged = kn_merged / (kn_merged.sum() + eps)

    #     out = {
    #         "usage": usage,     # (P,), sum=1
    #         "q_norm": qn_merged,
    #         "k_norm": kn_merged,
    #     }

    #     if per_head:
    #         # 每个 head 单独归一化（各行和=1）
    #         usage_ph = usage_by_head / (usage_by_head.sum(dim=1, keepdim=True) + eps)
    #         out["usage_per_head"] = usage_ph  # (H_sel, P)

    #     return out



class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        d_ff = int((8/3) * n_embd)
        self.fc1 = nn.Linear(n_embd, 2 * d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        swish = x1 * torch.sigmoid(x1)
        x = swish * x2
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, inverse_t, rope_base, start, end, gemma, indices, bias=True):
        super().__init__()
        # LayerNorm and CausalSelfAttention with explicit parameters
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, inverse_t=inverse_t, rope_base=rope_base, start=start, end=end, gemma=gemma, indices=indices, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        # self.mlp = MLP(n_embd, dropout, bias=bias)  # MLP with explicit parameters
        self.mlp = SwiGLUFFN(n_embd, dropout) #bias=bias)

    def forward(self, x, use_t=False, return_cache=False):
        if return_cache:
            att_in = self.ln_1(x)
            y, att, q, k, v, att_raw = self.attn(att_in, use_t=use_t, return_cache=return_cache)
            x = x + y
            x = x + self.mlp(self.ln_2(x))
            return x, att, q, k, v, att_raw
        else:
            # Apply residual connection and pre-normalization
            x = x + self.attn(self.ln_1(x), use_t=use_t, return_cache=return_cache)  # Apply LayerNorm before attention
            x = x + self.mlp(self.ln_2(x))
            return x



class GPT_RoPE(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout, inverse_t, rope_base, start, end, gemma, indices, bias=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.inverse_t = inverse_t
        self.rope_base = rope_base

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd), # token embeddings
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout, block_size, inverse_t=inverse_t, rope_base=rope_base, start=start, end=end, gemma=gemma, indices=indices, bias=bias) for _ in range(n_layer)]),
                              #  Block(n_embd, n_head, dropout, block_size, inverse_t=inverse_t, rope_base=rope_base, start=start, end=end, gemma=gemma, indices=indices, bias=bias)]),
                             #  Block(n_embd, n_head, dropout, block_size, inverse_t=inverse_t, rope_base=rope_base, start=start, end=end, gemma=gemma, indices=indices, bias=bias)]), # a stack of n_layer blocks
            ln_f = LayerNorm(n_embd, bias=bias), # final layer norm
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False) # projects the final transformer output to the vocab size

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_inverse_t(self, inverse_t: float):
        """Set k for all attention modules"""
        self.inverse_t = inverse_t
        for blk in self.transformer.h:
            blk.attn.inverse_t = self.inverse_t

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)# + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training/Evaluation path
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=encode("*")[0])
        else:
            # Inference path
            # 只计算最后一个时间步的 logits 以提高效率
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


    @torch.no_grad()
    def forward_with_inverse_t(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)# + pos_emb)
        for block in self.transformer.h:
            x = block(x, use_t=True)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training/Evaluation path
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=encode("*")[0])
        else:
            # Inference path
            # 只计算最后一个时间步的 logits 以提高效率
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def forward_with_cache(self, idx, targets=None):
        """
        仅用于可视化：返回
          x: 最终隐状态 (B, T, C)
          attn_list: 长度为 n_layer 的列表，每个元素是 (B, H, T, T)
        """
        self.eval()
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"seq len {t} > block_size {self.block_size}"

        x = self.transformer.wte(idx)   # (B, T, C)
        x = self.transformer.drop(x)

        attn_list = []
        q_list = []
        k_list = []
        v_list = []
        out_list = []
        att_raw_list = []

        for block in self.transformer.h:
            x, att, q, k, v, att_raw  = block(x, return_cache=True)
            attn_list.append(att)       # (B, H, T, T)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
            out_list.append(x)
            att_raw_list.append(att_raw)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training/Evaluation path
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=encode("*")[0])
        else:
            # Inference path
            # 只计算最后一个时间步的 logits 以提高效率
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return x, attn_list, q_list, k_list, v_list, out_list, logits, loss, att_raw_list



    @torch.no_grad()
    def forward_with_ppl(self, idx: torch.Tensor, targets: torch.Tensor):
        """
        计算只在唯一 '=' 之后（且非 padding）的 perplexity。
        假设每个序列恰有一个 '='。
        返回: (ppl: float, valid_count: int)
        """
        self.eval()

        # 取全序列 logits（不改你原 forward 逻辑）
        logits, _ = self.forward(idx, targets=targets)  # (B, T, V)

        pad_id = encode("*")[0]
        eq_id  = encode("=")[0]

        B, T = idx.shape

        # 找到每个样本中 '=' 的位置（唯一）
        eq_mask = (idx == eq_id)                          # (B, T) bool
        # 恰有一个 True，argmax 即该位置
        eq_pos = torch.argmax(eq_mask.to(torch.int64), dim=1)   # (B,)

        # 构造“等号之后”的位置掩码（不含 '=' 自身）
        ar = torch.arange(T, device=idx.device)           # (T,)
        after_eq = ar.unsqueeze(0) > eq_pos.unsqueeze(1)  # (B, T) bool

        # 去掉 padding（以 targets 是否为 pad 判断被监督位置）
        not_pad = (targets != pad_id)                     # (B, T) bool

        valid_mask = after_eq & not_pad                   # (B, T) bool

        # 负对数似然
        log_probs = F.log_softmax(logits, dim=-1)         # (B, T, V)
        nll_all = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, T)

        total_nll = (nll_all * valid_mask.float()).sum()
        valid_count = int(valid_mask.sum().item())

        # 题设保证一定有等号 -> 必然有有效 token，无需额外判断
        avg_nll = total_nll / valid_count
        ppl = torch.exp(avg_nll).item()
        return ppl, valid_count


    # @torch.no_grad()
    # def analyze_plane_usage_norms(
    #     self,
    #     idx: torch.Tensor,
    #     *,
    #     layers: int | list[int] | None = None,   # None=全部层；int或list指定
    #     heads: int | list[int] | None = None,    # None=全部head；子集时只统计这些head
    #     agg: str = "mean",                       # 传到子函数："mean" 或 "sum"
    #     combine_qk: str = "product",             # "product" 或 "sum"
    #     per_layer: bool = False,                 # True: 返回每层 (P,)；否则跨层等权平均
    #     per_head: bool = False                   # True: 单层时返回 (H_sel,P)
    # ):
    #     """
    #     频率依赖（论文范数法）整模分析：不改变模型行为。
    #       - 默认对所有层的 (P,) 曲线做等权平均，返回 {'usage': (P,)}（和=1）。
    #       - 指定单层时，可额外返回每 head 的曲线：{'usage_per_head': (H_sel,P)}。
    #       - 指定多层且 per_layer=True 时，返回 {'usage_per_layer': (L,P)}（每行和=1）。

    #     统计位置：每层注意力的输入处（att_in = ln_1(x)），与你的 forward 对齐。
    #     """
    #     self.eval()
    #     device = idx.device
    #     B, T = idx.shape
    #     assert T <= self.block_size, f"seq len {T} > block_size {self.block_size}"

    #     # 词嵌入 + dropout
    #     x = self.transformer.drop(self.transformer.wte(idx))  # (B,T,C)

    #     # 规范层列表
    #     if layers is None:
    #         layer_ids = list(range(len(self.transformer.h)))
    #     elif isinstance(layers, int):
    #         layer_ids = [layers]
    #     else:
    #         layer_ids = list(layers)
    #     for li in layer_ids:
    #         assert 0 <= li < len(self.transformer.h), f"layer {li} 超界"

    #     per_layer_usages = []
    #     last_single_layer_per_head = None
    #     qn_list, kn_list = [], []

    #     for li, block in enumerate(self.transformer.h):
    #         att_in = block.ln_1(x)  # 与 forward 对齐

    #         if li in layer_ids:
    #             out = block.attn.analyze_plane_usage_norms(
    #                 att_in,
    #                 heads=heads,
    #                 agg=agg,
    #                 combine_qk=combine_qk,
    #                 per_head=per_head and (len(layer_ids) == 1)
    #             )
    #             per_layer_usages.append(out["usage"])  # (P,)
    #             qn_list.append(out["q_norm"])          # (P,)
    #             kn_list.append(out["k_norm"])          # (P,)

    #             if (per_head and len(layer_ids) == 1) and ("usage_per_head" in out):
    #                 last_single_layer_per_head = out["usage_per_head"]  # (H_sel,P)

    #         # 正常前向（不改变原模型行为）
    #         y = block.attn(att_in, use_t=False, return_cache=False)
    #         x = x + y

    #     # 组织输出
    #     layer_stack = torch.stack(per_layer_usages, dim=0)  # (L_sel,P)

    #     if per_layer:
    #         # 每层各自已经和=1，这里直接返回
    #         result = {"usage_per_layer": layer_stack}
    #     else:
    #         # 跨层等权平均，再归一化（数值稳健）
    #         merged = layer_stack.mean(dim=0)  # (P,)
    #         eps = torch.finfo(merged.dtype).eps
    #         merged = merged / (merged.sum() + eps)
    #         result = {"usage": merged}

    #     # 附上 q/k 的平均曲线（跨层平均，再归一化），便于诊断
    #     qn_avg = torch.stack(qn_list, dim=0).mean(dim=0)
    #     kn_avg = torch.stack(kn_list, dim=0).mean(dim=0)
    #     eps = torch.finfo(qn_avg.dtype).eps
    #     result["q_norm"] = qn_avg / (qn_avg.sum() + eps)
    #     result["k_norm"] = kn_avg / (kn_avg.sum() + eps)

    #     if last_single_layer_per_head is not None:
    #         result["usage_per_head"] = last_single_layer_per_head

    #     return result
    
@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, stop_token_id=END_TOKEN_INDEX):
    """
    Generate a sequence of tokens using greedy decoding.

    Parameters:
        model (nn.Module): The model used for generation.
        idx (torch.Tensor or list): Initial sequence of indices (LongTensor of shape (b,t)).
        max_new_tokens (int): Number of new tokens to generate.
        stop_token_id (int, optional): The token ID that signals the end of generation.

    Returns:
        list[str]: A list of decoded generated texts.
    """
    # --- Setup and Input Handling ---
    # [修复] 正确获取设备
    device = next(model.parameters()).device
    # [优化] 增强输入处理的鲁棒性
    if not isinstance(idx, torch.Tensor):
        idx = torch.tensor(idx)
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)
    idx = idx.to(device)

    batch_size, _ = idx.shape
    is_active = torch.ones(batch_size, dtype=torch.bool, device=device)

    # --- Generation Loop ---
    for _ in range(max_new_tokens):
        if not is_active.any():
            break

        # 确保上下文长度不超过模型的block_size
        # block_size = getattr(model, 'block_size', getattr(model.config, 'block_size', 1024)) # Original line
        block_size = model.block_size # Fixed line
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # 前向传播获取logits
        logits, _ = model.forward_with_inverse_t(idx_cond)

        # 只关心最后一个时间步的logits
        logits_last = logits[is_active, -1, :]

        # --- [核心改动] Greedy Decoding ---
        # 直接使用 argmax 获取 logits 最高分的 token，无需 temperature, top_k, 或 softmax
        idx_next_active = torch.argmax(logits_last, dim=-1, keepdim=True)

        # 创建一个完整的下一token张量，以便与原始idx拼接
        idx_next = torch.full((batch_size, 1), 0, dtype=torch.long, device=device) # 用0或其他占位符填充
        idx_next[is_active] = idx_next_active

        # 将新生成的token拼接到序列中
        idx = torch.cat((idx, idx_next), dim=1)

        # --- [优化] 向量化的停止条件检查 ---
        if stop_token_id is not None:
            # 检查当前活跃的序列是否生成了停止符
            finished_now = is_active & (idx_next.squeeze(1) == stop_token_id)
            is_active[finished_now] = False

    # --- Decoding and Post-processing ---
    decoded_texts = []
    for i in range(batch_size):
        seq = idx[i].tolist()
        # 如果有停止符，截断到停止符之前
        if stop_token_id is not None:
            try:
                stop_pos = seq.index(stop_token_id)
                seq = seq[:stop_pos+1]
            except ValueError:
                pass # 如果没找到停止符，就使用整个序列

        text = decode(seq)  # 假设 decode 和 encode 函数已在全局定义
        decoded_texts.append(text)

    return decoded_texts