"""
Simple example showing how to use RoPE

This script demonstrates the basic usage of RoPE with both
PyTorch reference and Triton implementations.
"""

import torch
from rope_pytorch import rope_pytorch
from rope_triton import rope_triton


def main():
    print("=" * 80)
    print("Simple RoPE Example")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Head dimension: {head_dim}")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cpu":
        print("\nNote: Running on CPU. For best performance, use a CUDA GPU.")
        print("      Triton kernel requires CUDA and will not be tested.\n")
    
    # Create random query and key tensors
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
    
    print(f"\nInput shapes:")
    print(f"  Query (Q): {q.shape}")
    print(f"  Key (K): {k.shape}")
    
    # Apply RoPE using PyTorch reference
    print("\n" + "-" * 80)
    print("Applying RoPE with PyTorch reference implementation...")
    print("-" * 80)
    
    q_rot_pt, k_rot_pt = rope_pytorch(q.clone(), k.clone(), seq_len)
    
    print(f"Output shapes:")
    print(f"  Rotated Q: {q_rot_pt.shape}")
    print(f"  Rotated K: {k_rot_pt.shape}")
    
    # Verify norm preservation
    q_norm_before = torch.norm(q, dim=-1).mean()
    q_norm_after = torch.norm(q_rot_pt, dim=-1).mean()
    
    print(f"\nNorm preservation check:")
    print(f"  Average Q norm before RoPE: {q_norm_before:.6f}")
    print(f"  Average Q norm after RoPE: {q_norm_after:.6f}")
    print(f"  Difference: {abs(q_norm_before - q_norm_after):.6e}")
    
    if abs(q_norm_before - q_norm_after) < 1e-4:
        print("  ✓ Norm is preserved (as expected from rotation)")
    
    # Apply RoPE using Triton kernel (if CUDA available)
    if device == "cuda":
        print("\n" + "-" * 80)
        print("Applying RoPE with Triton kernel implementation...")
        print("-" * 80)
        
        q_rot_tr, k_rot_tr = rope_triton(q.clone(), k.clone(), seq_len)
        
        print(f"Output shapes:")
        print(f"  Rotated Q: {q_rot_tr.shape}")
        print(f"  Rotated K: {k_rot_tr.shape}")
        
        # Compare implementations
        print(f"\nComparing implementations:")
        q_diff = (q_rot_pt - q_rot_tr).abs().max()
        k_diff = (k_rot_pt - k_rot_tr).abs().max()
        
        print(f"  Max difference in Q: {q_diff:.6e}")
        print(f"  Max difference in K: {k_diff:.6e}")
        
        if q_diff < 1e-4 and k_diff < 1e-4:
            print("  ✓ PyTorch and Triton implementations match!")
    
    # Show how rotation affects attention scores
    print("\n" + "-" * 80)
    print("How RoPE affects attention scores...")
    print("-" * 80)
    
    # Compute attention scores before and after RoPE
    # Reshape for attention: (batch, n_heads, seq_len, head_dim)
    q_attn = q.transpose(1, 2)
    k_attn = k.transpose(1, 2)
    q_rot_attn = q_rot_pt.transpose(1, 2)
    k_rot_attn = k_rot_pt.transpose(1, 2)
    
    # Attention scores: Q @ K^T
    scores_before = torch.matmul(q_attn, k_attn.transpose(-2, -1))
    scores_after = torch.matmul(q_rot_attn, k_rot_attn.transpose(-2, -1))
    
    print(f"Attention scores shape: {scores_before.shape}")
    print(f"Score statistics before RoPE:")
    print(f"  Mean: {scores_before.mean():.4f}")
    print(f"  Std: {scores_before.std():.4f}")
    print(f"Score statistics after RoPE:")
    print(f"  Mean: {scores_after.mean():.4f}")
    print(f"  Std: {scores_after.std():.4f}")
    
    print("\nRoPE modifies attention scores to incorporate positional information,")
    print("allowing the model to be aware of token positions through rotations.")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Run tests: python test_rope.py")
    print("  - Run benchmarks: python benchmark.py")
    print("  - See attention demo: python demo_attention.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
