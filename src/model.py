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

       
        if ramp <= 0 or len(idx_list) != 2:
            s = self.angle_scale.clone()
            s[idx.clamp_(0, P - 1)] = gamma   
            self.angle_scale = s
            return

    
        a = min(idx_list)
        b = max(idx_list)

       
        a = max(0, min(P - 1, a))
        b = max(0, min(P - 1, b))

       
        alpha = torch.zeros(P, device=device)

     
        alpha[a:b+1] = 1.0

      
        left_end = a - 1
        if left_end >= 0:
            left_start = max(0, a - ramp)
            if left_start <= left_end:
                L = left_end - left_start + 1
               
                vals = torch.linspace(0.0, 1.0, steps=L + 1, device=device)[:-1]
                alpha[left_start:a] = vals

     
        right_start = b + 1
        if right_start < P:
            right_end = min(P - 1, b + ramp)
            if right_start <= right_end:
                L = right_end - right_start + 1
            
                vals = torch.linspace(1.0, 0.0, steps=L + 1, device=device)[1:]
                alpha[right_start:right_end + 1] = vals

        orig = self.angle_scale
        self.angle_scale = orig * (1.0 - alpha) + gamma * alpha
        # print(f"ALPHA {alpha}")


  
    def set_scale_slice(self, start: int, end: int, gamma: float):
        """
        将频率区间 [start, end) 统一乘以 gamma。
        """
        s = self.angle_scale.clone()
        s[start:end] = float(gamma)
        self.angle_scale = s

    def _cos_sin(self, seq_len: int, device, dtype):
       
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  
        freqs = pos * self.inv_freq.unsqueeze(0)
        cos = torch.cos(freqs).to(dtype)                    
        sin = torch.sin(freqs).to(dtype)                  
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, -1)  # [T, D]
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, -1)  # [T, D]
        return cos, sin

    @staticmethod
    def _rotate_half(x):
      
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        x_rot_even = -x_odd
        x_rot_odd  =  x_even
       
        return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)

    def apply_rotary(self, x: torch.Tensor, seq_len: int):
       
        B = x.shape[0]
        T = x.shape[-2]
        Dh = x.shape[-1]
        assert Dh == self.head_dim

        cos, sin = self._cos_sin(T, x.device, x.dtype)        
      
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
      
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin


    def _cos_sin_with_plane_scale(self, positions: torch.Tensor, s_plane: torch.Tensor, device, dtype):
       
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
                                      *,           
                                      use_length_ratio: bool = False,
                                      length_ratio: float = 1.0):
       
        B, H, T, Dh = q.shape
        device, dtype = q.device, q.dtype
        assert Dh == self.head_dim

        
        ones = torch.ones_like(self.angle_scale)                                     # [D/2]
        m = torch.arange(T, device=device, dtype=torch.float32)                      # [T]
        cos_n, sin_n = self._cos_sin_with_plane_scale(m, ones, device, dtype)        # [T,D]

     
        if use_length_ratio:
            s_plane = self.angle_scale * float(length_ratio)                         # [D/2]
        else:
            s_plane = self.angle_scale
        cos_i, sin_i = self._cos_sin_with_plane_scale(m, s_plane, device, dtype)     # [T,D]

        
        def _b(t):
            while t.dim() < q.dim(): t = t.unsqueeze(0)
            return t
        cos_n_b, sin_n_b = _b(cos_n), _b(sin_n)
        cos_i_b, sin_i_b = _b(cos_i), _b(sin_i)

       
        q_rot = self._rotate_half(q); k_rot = self._rotate_half(k)
        q_out = q * cos_n_b + q_rot * sin_n_b
        k_out = k * cos_n_b + k_rot * sin_n_b

       
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
        self._pi_use_length_ratio = False      
        self._pi_L_train = block_size
        self._pi_L_target = None


    def set_gemma(self, gemma: float, indice, ramp):
        self.rope.set_scale_indices(indices=indice, gamma=gemma, ramp=ramp)


    def set_head_pi_mask(self, mask_bool_list_or_tensor):
        m = mask_bool_list_or_tensor
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, dtype=torch.bool, device=self.head_pi_mask.device) 
        assert m.numel() == self.n_head
        self.head_pi_mask.copy_(m)                                                
        
    def enable_headwise_pi(self, L_train: int = None, L_target: int = None):
        if L_train is not None:
            self._pi_L_train = int(L_train)       
        if L_target is not None:
            self._pi_L_target = int(L_target)     
        self._pi_enabled = True                   

    
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
          
            # (B, H, T, T)
            att_raw = torch.matmul(q, k.transpose(-2, -1)) * (self.inverse_t / math.sqrt(head_dim))

           
            mask = self.casual_mask_full[:, :, :T, :T]
            att = att_raw.masked_fill(~mask, float('-inf'))

            att = F.softmax(att, dim=-1)  
            y = torch.matmul(att, v)     
       
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
           
            mask = self.casual_mask_full[:, :, :T, :T]
            att = att.masked_fill(~mask, float('-inf'))

            att = F.softmax(att, dim=-1)  
            y = torch.matmul(att, v)      

        # Reshape back to original format
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble heads

        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))

        if return_cache:
            return y, att, q, k, v, att_raw
        else:
            return y



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
                            
            ln_f = LayerNorm(n_embd, bias=bias), 
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

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
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


    @torch.no_grad()
    def forward_with_inverse_t(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
      
        tok_emb = self.transformer.wte(idx) 
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, use_t=True)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training/Evaluation path
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=encode("*")[0])
        else:
            # Inference path
            
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def forward_with_cache(self, idx, targets=None):
       
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
           
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return x, attn_list, q_list, k_list, v_list, out_list, logits, loss, att_raw_list



    @torch.no_grad()
    def forward_with_ppl(self, idx: torch.Tensor, targets: torch.Tensor):
        
        self.eval()

       
        logits, _ = self.forward(idx, targets=targets)  # (B, T, V)

        pad_id = encode("*")[0]
        eq_id  = encode("=")[0]

        B, T = idx.shape

        
        eq_mask = (idx == eq_id)                          # (B, T) bool
        
        eq_pos = torch.argmax(eq_mask.to(torch.int64), dim=1)   # (B,)

        
        ar = torch.arange(T, device=idx.device)           # (T,)
        after_eq = ar.unsqueeze(0) > eq_pos.unsqueeze(1)  # (B, T) bool

        
        not_pad = (targets != pad_id)                     # (B, T) bool

        valid_mask = after_eq & not_pad                   # (B, T) bool

        
        log_probs = F.log_softmax(logits, dim=-1)         # (B, T, V)
        nll_all = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, T)

        total_nll = (nll_all * valid_mask.float()).sum()
        valid_count = int(valid_mask.sum().item())

       
        avg_nll = total_nll / valid_count
        ppl = torch.exp(avg_nll).item()
        return ppl, valid_count


  
    
@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, stop_token_id=END_TOKEN_INDEX):
    
    device = next(model.parameters()).device
    
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

        
        block_size = model.block_size 
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

       
        logits, _ = model.forward_with_inverse_t(idx_cond)

        
        logits_last = logits[is_active, -1, :]

        
        idx_next_active = torch.argmax(logits_last, dim=-1, keepdim=True)

        
        idx_next = torch.full((batch_size, 1), 0, dtype=torch.long, device=device) 
        idx_next[is_active] = idx_next_active

        
        idx = torch.cat((idx, idx_next), dim=1)

        
        if stop_token_id is not None:
            
            finished_now = is_active & (idx_next.squeeze(1) == stop_token_id)
            is_active[finished_now] = False

   
    decoded_texts = []
    for i in range(batch_size):
        seq = idx[i].tolist()
       
        if stop_token_id is not None:
            try:
                stop_pos = seq.index(stop_token_id)
                seq = seq[:stop_pos+1]
            except ValueError:
                pass 

        text = decode(seq)  
        decoded_texts.append(text)

    return decoded_texts