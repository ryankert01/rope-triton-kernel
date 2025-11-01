"""
PyTorch Reference Implementation of Rotary Position Embedding (RoPE)

RoPE applies rotary embeddings to query and key tensors by rotating adjacent pairs
of elements in the feature dimension. This implementation serves as a reference
for correctness testing of the Triton kernel.
"""

import torch
import math


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device: str = "cuda"):
    """
    Precompute the frequency tensor for complex exponentials (cis) for RoPE.
    
    Args:
        dim: Dimension of embeddings (must be even)
        seq_len: Maximum sequence length
        theta: Theta parameter for frequency computation (default: 10000.0)
        device: Device to create tensors on
        
    Returns:
        Tensor of shape (seq_len, dim // 2) containing complex frequencies
    """
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE"
    
    # Compute frequencies for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Create position indices
    t = torch.arange(seq_len, device=device)
    
    # Compute outer product to get frequencies for each position
    freqs = torch.outer(t, freqs).float()
    
    # Convert to complex exponentials (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequency tensor of shape (seq_len, head_dim // 2)
        
    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    # Reshape input to complex numbers by pairing adjacent elements
    # x shape: (batch, seq_len, n_heads, head_dim)
    # We need to view pairs as complex: (batch, seq_len, n_heads, head_dim // 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Reshape freqs_cis to broadcast correctly
    # freqs_cis shape: (seq_len, head_dim // 2)
    # Expand to: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation by complex multiplication
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real representation
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    
    return x_out.type_as(x)


def rope_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors (PyTorch reference implementation).
    
    Args:
        q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        seq_len: Sequence length (unused but kept for API compatibility)
        theta: Theta parameter for frequency computation
        
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    batch, seq_len, n_heads, head_dim = q.shape
    
    # Precompute frequencies
    freqs_cis = precompute_freqs_cis(head_dim, seq_len, theta, device=q.device)
    
    # Apply rotary embeddings
    q_rotated = apply_rotary_emb(q, freqs_cis)
    k_rotated = apply_rotary_emb(k, freqs_cis)
    
    return q_rotated, k_rotated
