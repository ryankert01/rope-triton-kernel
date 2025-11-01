# API Reference

## Core Functions

### `rope_triton(q, k, seq_len, theta=10000.0)`

Apply Rotary Position Embedding using the optimized Triton GPU kernel.

**Parameters:**
- `q` (torch.Tensor): Query tensor of shape `(batch, seq_len, n_heads, head_dim)`
  - Must be on CUDA device
  - head_dim must be even
- `k` (torch.Tensor): Key tensor of shape `(batch, seq_len, n_heads, head_dim)`
  - Must be on CUDA device
  - Must have same shape as q
- `seq_len` (int): Sequence length (kept for API compatibility)
- `theta` (float, optional): Base for frequency computation (default: 10000.0)

**Returns:**
- Tuple of `(q_rotated, k_rotated)` with same shapes and dtypes as inputs

**Example:**
```python
import torch
from rope_triton import rope_triton

q = torch.randn(2, 128, 8, 64, device="cuda")
k = torch.randn(2, 128, 8, 64, device="cuda")
q_rot, k_rot = rope_triton(q, k, 128)
```

---

### `rope_pytorch(q, k, seq_len, theta=10000.0)`

PyTorch reference implementation of RoPE.

**Parameters:**
Same as `rope_triton`, but works on both CPU and CUDA.

**Returns:**
Same as `rope_triton`.

**Example:**
```python
import torch
from rope_pytorch import rope_pytorch

q = torch.randn(2, 128, 8, 64)  # CPU or CUDA
k = torch.randn(2, 128, 8, 64)
q_rot, k_rot = rope_pytorch(q, k, 128)
```

---

### `precompute_freqs_cis(dim, seq_len, theta=10000.0, device="cuda")`

Precompute frequency tensor for RoPE.

**Parameters:**
- `dim` (int): Dimension of embeddings (must be even)
- `seq_len` (int): Maximum sequence length
- `theta` (float, optional): Base for frequency computation (default: 10000.0)
- `device` (str, optional): Device to create tensors on (default: "cuda")

**Returns:**
- Complex tensor of shape `(seq_len, dim // 2)` containing rotation frequencies

**Example:**
```python
from rope_pytorch import precompute_freqs_cis

freqs = precompute_freqs_cis(dim=64, seq_len=128, device="cuda")
# Shape: (128, 32), dtype: complex
```

---

### `apply_rotary_emb(x, freqs_cis)`

Apply rotary embeddings to input tensor using precomputed frequencies.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape `(batch, seq_len, n_heads, head_dim)`
- `freqs_cis` (torch.Tensor): Precomputed frequencies of shape `(seq_len, head_dim // 2)`

**Returns:**
- Rotated tensor with same shape as input

**Example:**
```python
from rope_pytorch import precompute_freqs_cis, apply_rotary_emb
import torch

freqs = precompute_freqs_cis(64, 128)
x = torch.randn(2, 128, 8, 64)
x_rotated = apply_rotary_emb(x, freqs)
```

---

## Classes

### `RoPEAttention`

Multi-head attention layer with integrated Rotary Position Embeddings.

**Constructor Parameters:**
- `d_model` (int): Model dimension (total dimension across all heads)
- `n_heads` (int): Number of attention heads
  - d_model must be divisible by n_heads
- `dropout` (float, optional): Dropout probability (default: 0.1)
- `theta` (float, optional): RoPE theta parameter (default: 10000.0)

**Forward Method:**
```python
forward(x, mask=None) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input of shape `(batch, seq_len, d_model)`
- `mask` (torch.Tensor, optional): Attention mask of shape `(batch, seq_len, seq_len)`
  - Values of 0 indicate positions to mask out
  - Values of 1 indicate positions to attend to

**Returns:**
- Output tensor of shape `(batch, seq_len, d_model)`

**Example:**
```python
from demo_attention import RoPEAttention
import torch

model = RoPEAttention(d_model=512, n_heads=8).cuda()
x = torch.randn(2, 128, 512, device="cuda")
output = model(x)

# With causal mask
causal_mask = torch.tril(torch.ones(128, 128, device="cuda"))
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
output = model(x, mask=causal_mask)
```

---

## Utility Functions

### `benchmark_rope(...)`

Benchmark RoPE implementations.

**Parameters:**
- `batch_size` (int): Batch size
- `seq_len` (int): Sequence length
- `n_heads` (int): Number of attention heads
- `head_dim` (int): Dimension per head
- `n_warmup` (int, optional): Number of warmup iterations (default: 10)
- `n_iterations` (int, optional): Number of benchmark iterations (default: 100)

**Returns:**
- Dictionary with benchmark results containing:
  - `batch_size`, `seq_len`, `n_heads`, `head_dim`: Configuration
  - `pytorch_ms`: PyTorch implementation time in milliseconds
  - `triton_ms`: Triton implementation time in milliseconds
  - `speedup`: Speedup ratio (pytorch_ms / triton_ms)

**Example:**
```python
from benchmark import benchmark_rope

result = benchmark_rope(
    batch_size=4,
    seq_len=512,
    n_heads=8,
    head_dim=64
)
print(f"Speedup: {result['speedup']:.2f}x")
```

---

## Data Types

RoPE implementations support the following data types:
- `torch.float32` (recommended for correctness testing)
- `torch.float16` (may offer better performance on modern GPUs)

Complex numbers are used internally in the PyTorch reference but are transparent to users.

---

## Shape Requirements

All tensors must follow these shape conventions:

**Query/Key Input:**
- Shape: `(batch_size, seq_len, n_heads, head_dim)`
- `head_dim` must be even (required for pair-wise rotation)

**Attention Input:**
- Shape: `(batch_size, seq_len, d_model)`
- `d_model = n_heads * head_dim`

**Attention Mask (optional):**
- Shape: `(batch_size, seq_len, seq_len)` or `(1, 1, seq_len, seq_len)`
- Broadcasting is supported

---

## Performance Considerations

1. **Batch Size**: Larger batches utilize GPU better (4-32 recommended)
2. **Sequence Length**: Performance scales linearly
3. **Head Dimension**: Powers of 2 (32, 64, 128) perform best
4. **Data Type**: FP16 may be 2x faster on modern GPUs with Tensor Cores
5. **Device**: Triton kernel requires CUDA; PyTorch reference works on CPU/CUDA

---

## Error Handling

Common errors and solutions:

**"Inputs must be on CUDA device"**
- Solution: Move tensors to CUDA with `.cuda()` or `.to('cuda')`

**"Q and K must have same shape"**
- Solution: Ensure query and key have identical dimensions

**"Head dimension must be even"**
- Solution: Use even head dimensions (32, 64, 128, etc.)

**"d_model must be divisible by n_heads"**
- Solution: Ensure `d_model % n_heads == 0`

---

## Testing

Run the test suite:
```bash
# With pytest
pytest test_rope.py -v

# Manual execution
python test_rope.py
```

Run benchmarks:
```bash
python benchmark.py
```

Run examples:
```bash
python example.py
python demo_attention.py
```

---

## References

- [RoFormer Paper](https://arxiv.org/abs/2104.09864)
- [Triton Documentation](https://triton-lang.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
