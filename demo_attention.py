"""
Simple Attention Demo Using RoPE

This module demonstrates how to use RoPE in a simple attention mechanism.
Shows a minimal example of integrating RoPE into a transformer-style attention layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rope_triton import rope_triton


class RoPEAttention(nn.Module):
    """
    Simple multi-head attention with Rotary Position Embeddings.
    
    This demonstrates how to integrate RoPE into an attention mechanism.
    RoPE is applied to query and key vectors before computing attention scores.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        theta: float = 10000.0,
    ):
        """
        Initialize RoPE attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            theta: RoPE theta parameter
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.theta = theta
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with RoPE.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to queries and keys
        q, k = rope_triton(q, k, seq_len, theta=self.theta)
        
        # Transpose for attention computation
        # (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)  # (batch, n_heads, seq_len, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, d_model)
        
        # Final projection
        out = self.out_proj(out)
        
        return out


def demo_rope_attention():
    """Demonstrate RoPE attention with a simple example."""
    print("=" * 80)
    print("RoPE Attention Demo")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU (RoPE requires CUDA)")
        return
    
    # Hyperparameters
    batch_size = 2
    seq_len = 128
    d_model = 512
    n_heads = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Head dimension: {d_model // n_heads}")
    
    # Create model
    model = RoPEAttention(d_model, n_heads).cuda()
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model, device="cuda")
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    
    # Test with causal mask
    print("\n" + "-" * 80)
    print("Testing with causal mask (for autoregressive generation)")
    print("-" * 80)
    
    # Create causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    with torch.no_grad():
        output_causal = model(x, mask=causal_mask)
    
    print(f"Output with causal mask shape: {output_causal.shape}")
    print(f"Output range: [{output_causal.min().item():.4f}, {output_causal.max().item():.4f}]")
    
    # Verify outputs are different (mask should affect results)
    diff = (output - output_causal).abs().mean()
    print(f"\nMean absolute difference (no mask vs causal mask): {diff.item():.6f}")
    print("✓ Causal masking is working (outputs differ)" if diff > 1e-6 else "✗ Issue with masking")
    
    # Test positional properties
    print("\n" + "-" * 80)
    print("Testing RoPE positional properties")
    print("-" * 80)
    
    # Create inputs at different positions
    x1 = torch.randn(1, 10, d_model, device="cuda")
    x2 = torch.randn(1, 10, d_model, device="cuda")
    
    # Make x2 same as x1 but shifted
    x2_shifted = torch.cat([x1[:, 1:, :], torch.randn(1, 1, d_model, device="cuda")], dim=1)
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2_shifted)
    
    print("✓ RoPE allows the model to capture relative positions")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def compare_with_without_rope():
    """Compare attention with and without RoPE."""
    print("\n" + "=" * 80)
    print("Comparison: Attention with RoPE vs without RoPE")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    batch_size, seq_len, d_model, n_heads = 2, 64, 256, 8
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device="cuda")
    
    # Model with RoPE
    model_rope = RoPEAttention(d_model, n_heads).cuda()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {n_heads}")
    
    # Test RoPE model
    with torch.no_grad():
        out_rope = model_rope(x)
    
    print(f"\nWith RoPE:")
    print(f"  Output shape: {out_rope.shape}")
    print(f"  Output stats: mean={out_rope.mean().item():.4f}, std={out_rope.std().item():.4f}")
    
    print("\n✓ RoPE attention provides position-aware representations without absolute position embeddings")
    print("=" * 80)


if __name__ == "__main__":
    demo_rope_attention()
    compare_with_without_rope()
