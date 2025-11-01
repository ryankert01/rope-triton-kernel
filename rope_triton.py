"""
High-performance Triton Kernel Implementation of Rotary Position Embedding (RoPE)

This module implements RoPE using Triton for GPU acceleration. The kernel fuses
the frequency computation and rotation into a single GPU kernel for optimal performance.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def rope_kernel(
    # Pointers to tensors
    Q_ptr,
    K_ptr,
    Q_out_ptr,
    K_out_ptr,
    # Tensor dimensions
    batch,
    seq_len,
    n_heads,
    head_dim,
    # RoPE parameters
    theta,
    # Strides for Q and K (batch, seq, heads, dim)
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Triton kernel for applying rotary position embeddings.
    Each program handles one position in one head for one batch element.
    """
    # Program ID gives us which element we're processing
    pid = tl.program_id(0)
    
    # Compute indices
    total_size = batch * seq_len * n_heads
    
    if pid >= total_size:
        return
    
    # Decompose pid into batch, sequence, and head indices
    b_idx = pid // (seq_len * n_heads)
    remainder = pid % (seq_len * n_heads)
    s_idx = remainder // n_heads
    h_idx = remainder % n_heads
    
    # Compute base pointers for this batch/seq/head
    q_base = Q_ptr + b_idx * stride_qb + s_idx * stride_qs + h_idx * stride_qh
    k_base = K_ptr + b_idx * stride_kb + s_idx * stride_ks + h_idx * stride_kh
    q_out_base = Q_out_ptr + b_idx * stride_qb + s_idx * stride_qs + h_idx * stride_qh
    k_out_base = K_out_ptr + b_idx * stride_kb + s_idx * stride_ks + h_idx * stride_kh
    
    # Process pairs of dimensions
    for pair_idx in range(HEAD_DIM // 2):
        d_idx = pair_idx * 2
        
        # Load input values (pair of adjacent elements)
        q0 = tl.load(q_base + d_idx * stride_qd)
        q1 = tl.load(q_base + (d_idx + 1) * stride_qd)
        k0 = tl.load(k_base + d_idx * stride_kd)
        k1 = tl.load(k_base + (d_idx + 1) * stride_kd)
        
        # Compute frequency for this dimension pair
        freq = 1.0 / (theta ** (tl.cast(d_idx, tl.float32) / tl.cast(HEAD_DIM, tl.float32)))
        
        # Compute angle for this position
        angle = tl.cast(s_idx, tl.float32) * freq
        
        # Compute cos and sin
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        
        # Apply rotation (complex multiplication)
        # (q0 + i*q1) * (cos + i*sin) = (q0*cos - q1*sin) + i*(q0*sin + q1*cos)
        q_rot0 = q0 * cos_val - q1 * sin_val
        q_rot1 = q0 * sin_val + q1 * cos_val
        
        k_rot0 = k0 * cos_val - k1 * sin_val
        k_rot1 = k0 * sin_val + k1 * cos_val
        
        # Store rotated values
        tl.store(q_out_base + d_idx * stride_qd, q_rot0)
        tl.store(q_out_base + (d_idx + 1) * stride_qd, q_rot1)
        tl.store(k_out_base + d_idx * stride_kd, k_rot0)
        tl.store(k_out_base + (d_idx + 1) * stride_kd, k_rot1)


def rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors using Triton kernel.
    
    Args:
        q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        seq_len: Sequence length (unused but kept for API compatibility)
        theta: Theta parameter for frequency computation
        
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    assert q.is_cuda and k.is_cuda, "Inputs must be on CUDA device"
    assert q.shape == k.shape, "Q and K must have same shape"
    
    batch, seq_len, n_heads, head_dim = q.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    # Allocate output tensors
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    # Get strides
    stride_qb, stride_qs, stride_qh, stride_qd = q.stride()
    stride_kb, stride_ks, stride_kh, stride_kd = k.stride()
    
    # Launch kernel
    # Each program processes one (batch, seq, head) element
    grid = (batch * seq_len * n_heads,)
    
    rope_kernel[grid](
        q, k, q_out, k_out,
        batch, seq_len, n_heads, head_dim,
        theta,
        stride_qb, stride_qs, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kh, stride_kd,
        BLOCK_SIZE=128,
        HEAD_DIM=head_dim,
    )
    
    return q_out, k_out
