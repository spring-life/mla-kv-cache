"""
MLA / KV Cache 扩展实验
在原作业（MLA的实现.py）基础上补三件事：

  1. 补 causal mask —— 原实现是双向注意力，不是解码器该有的样子
  2. 扫描 latent_dim —— 证明压缩比是超参的函数，不是 MLA 的固有常数
  3. 外推到 Llama-3-8B —— 把玩具实验接到真实工程问题上

运行：
  python mla_kv_cache.py
"""

import math
import sys

import torch
import torch.nn as nn

# Windows 控制台默认编码不是 UTF-8，不处理的话中文输出会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(0)


def build_causal_mask(S, device):
    """上三角（不含对角线）为 True，表示这些位置在预测时看不到。"""
    return torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)


class MHA(nn.Module):
    """标准多头注意力，可开 KV Cache、可开 causal mask。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_cache()

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False, causal=False):
        b, T, _ = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k, v = self.cache_k, self.cache_v

        def reshape(t):
            return t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal:
            S = att.size(-1)
            mask = build_causal_mask(S, att.device)
            if q.size(-2) != S:                      # 增量解码时 q 比 k 短
                mask = mask[S - q.size(-2):, :]
            att = att.masked_fill(mask, float("-inf"))

        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.out_proj(out)


class MLA(nn.Module):
    """多头潜变量注意力：缓存的是压缩后的 latent，不是完整的 K/V。"""

    def __init__(self, d_model, num_heads, latent_dim):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.latent_dim = latent_dim

        self.W_q = nn.Linear(d_model, d_model)
        self.W_dkv = nn.Linear(d_model, latent_dim)      # 压缩
        self.W_uk = nn.Linear(latent_dim, d_model)       # 还原 K
        self.W_uv = nn.Linear(latent_dim, d_model)       # 还原 V
        self.out_proj = nn.Linear(d_model, d_model)
        self.reset_cache()

    def reset_cache(self):
        self.cache_latent = None

    def forward(self, x, use_cache=False, causal=False):
        b, T, _ = x.shape
        q = self.W_q(x)
        latent = self.W_dkv(x)

        if use_cache:
            if self.cache_latent is None:
                self.cache_latent = latent
            else:
                self.cache_latent = torch.cat([self.cache_latent, latent], dim=1)
            latent = self.cache_latent

        k, v = self.W_uk(latent), self.W_uv(latent)

        def reshape(t):
            return t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal:
            S = att.size(-1)
            mask = build_causal_mask(S, att.device)
            if q.size(-2) != S:
                mask = mask[S - q.size(-2):, :]
            att = att.masked_fill(mask, float("-inf"))

        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.out_proj(out)


def kb(tensor):
    return tensor.numel() * tensor.element_size() / 1024


def bar(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ---------------------------------------------------------------- 实验 1
bar("实验 1：补上 causal mask，验证前面 token 的输出不受后面 token 影响")

d_model, n_heads, T = 512, 8, 16
mha = MHA(d_model, n_heads)
x1 = torch.randn(1, T, d_model)
x2 = x1.clone()
x2[:, -1, :] += 100.0          # 只改最后一个 token

o1 = mha(x1, causal=True)
o2 = mha(x2, causal=True)

diff_masked = (o1[:, :-1] - o2[:, :-1]).abs().max().item()
diff_all = (o1 - o2).abs().max().item()

mha.reset_cache()
o3 = mha(x1, causal=False)
mha.reset_cache()
o4 = mha(x2, causal=False)
diff_bidir = (o3[:, :-1] - o4[:, :-1]).abs().max().item()

print(f"causal=True 时，修改末位 token 对前 {T-1} 个位置输出的影响：{diff_masked:.3e}")
print(f"双向注意力（causal=False）时，同样的影响：          {diff_bidir:.3e}")
print(f"最后一个位置自身的变化（应该明显 > 0）：            {diff_all:.3e}")
print()
print("结论：causal 版本前 N-1 个输出完全不变（差值为 0），")
print("      双向版本发生了泄漏 —— 说明 mask 确实生效了。")


# ---------------------------------------------------------------- 实验 2
bar("实验 2：扫描 latent_dim，看压缩比是怎么来的")

d_model, n_heads, b, T = 512, 8, 2, 64
x = torch.randn(b, T, d_model)

mha2 = MHA(d_model, n_heads)
mha2.reset_cache()
mha2(x, use_cache=True)
mha_kb = kb(mha2.cache_k) + kb(mha2.cache_v)

print(f"{'latent_dim':>12} {'MLA缓存(KB)':>14} {'MHA缓存(KB)':>14} {'压缩比':>10}")
print("-" * 54)

rows = []
for ratio in (2, 4, 8, 16):
    ld = d_model // ratio
    mla = MLA(d_model, n_heads, ld)
    mla.reset_cache()
    mla(x, use_cache=True)
    mla_kb = kb(mla.cache_latent)
    comp = mla_kb / mha_kb
    rows.append((ld, mla_kb, comp))
    print(f"{ld:>12} {mla_kb:>14.2f} {mha_kb:>14.2f} {comp:>9.4%}")

print()
print(f"拟合关系：压缩比 = latent_dim / (2 * d_model)")
for ld, mla_kb, comp in rows:
    predicted = ld / (2 * d_model)
    print(f"  latent_dim={ld:>3} → 实测 {comp:.4%}，公式 {predicted:.4%}")


# ---------------------------------------------------------------- 实验 3
bar("实验 3：外推到真实模型规模")

def kv_gb(n_layers, d_model, T, batch=1, bytes_per=2):
    """MHA 的 KV Cache：2(K和V) * batch * T * 层数 * d_model * 字节数"""
    return 2 * batch * T * n_layers * d_model * bytes_per / 1024 ** 3


n_layers, d_model_real = 32, 4096      # Llama-3-8B 的量级
print(f"模型设定：{n_layers} 层，d_model={d_model_real}，batch=1，fp16")
print()
print(f"{'上下文长度':>10} {'KV Cache':>14}")
print("-" * 28)
for T_real in (4096, 32768, 131072):
    print(f"{T_real:>10} {kv_gb(n_layers, d_model_real, T_real):>11.2f} GB")

print()
print("推导：2 × 1 × 4096 × 32 × 4096 × 2 字节 = 2.1 GB")
print("这就是业界常说「4K 上下文要吃 2 GB KV Cache」的来源。")
print("也解释了为什么长上下文推理的瓶颈在显存带宽，不在算力。")
