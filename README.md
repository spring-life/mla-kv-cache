# MLA 与 KV Cache 显存对比

用 PyTorch 从零实现多头注意力（MHA）与多头潜变量注意力（MLA），对比两者在自回归推理时 KV Cache 的显存占用，并把结论外推到真实模型规模。

## 背景

长上下文推理的开销很大一部分来自 KV Cache。标准 MHA 每生成一个 token，都要在该层缓存完整的 K 和 V 两个张量，缓存大小正比于 `2 × d_model`。MLA 的做法是先把 K/V 压缩成一个低维 latent 向量再缓存，需要时用线性投影还原，缓存大小降到 `latent_dim`。

## 三个实验与结论

### 1. 压缩比由 latent 维度决定

扫描 `latent_dim` 取 `d_model/2` 到 `d_model/16`：

| latent_dim | MLA 缓存 | MHA 缓存 | 压缩比 | 公式 `latent_dim/(2×d_model)` |
|---|---|---|---|---|
| 256 | 128.00 KB | 512.00 KB | 25.0000% | 25.0000% |
| 128 | 64.00 KB | 512.00 KB | 12.5000% | 12.5000% |
| 64 | 32.00 KB | 512.00 KB | 6.2500% | 6.2500% |
| 32 | 16.00 KB | 512.00 KB | 3.1250% | 3.1250% |

四组实测与公式精确吻合。压缩比只取决于 `latent_dim / d_model` 的比值，与序列长度无关。

### 2. causal mask 生效验证

只修改序列最后一个 token，观察前面位置的输出是否变化：

| 设置 | 前 N-1 个位置的最大变化 |
|---|---|
| causal = True | `0.000e+00` |
| causal = False（双向） | `1.167e+02` |
| 最后一个位置自身 | `1.094e+02` |

causal 版本前序输出完全不变，双向版本发生严重泄漏，说明掩码确实生效。

### 3. 外推到 Llama-3-8B 规模

32 层、`d_model=4096`、`batch=1`、fp16：

| 上下文长度 | KV Cache |
|---|---|
| 4K | 2.00 GB |
| 32K | 16.00 GB |
| 128K | 64.00 GB |

推导：`2 × 1 × 4096 × 32 × 4096 × 2 字节 = 2.1 GB`

128K 上下文时 KV Cache 达到 64 GB，超过模型权重本身的量级。长上下文推理的开销主要落在显存容量与带宽上。

## 运行

需要 PyTorch（CPU 版即可，实验不依赖 GPU）。

```bash
pip install torch
python mla_kv_cache.py
```

## 已知简化

这是一个简化实现，和 DeepSeek-V2 论文中的 MLA 有以下差异：

1. 没有解耦的 RoPE 分量（真实 MLA 需要单独保留位置编码键，因为压缩会破坏旋转结构）
2. 上投影是先还原到完整 `d_model` 再切头，论文中的实现是按头投影
3. `latent_dim` 是手动设定的超参，没有做端到端的精度-显存权衡实验
4. 只测了单层、注意力本身，没有接入完整的前馈网络和层归一化

因此本实现给出的压缩比比真实 MLA 更乐观，实际部署中还需考虑精度损失。

## 参考

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
