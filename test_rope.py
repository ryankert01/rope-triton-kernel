"""
Correctness Tests for RoPE Implementation

This module tests the Triton kernel implementation against the PyTorch reference
to ensure correctness across various input shapes and parameters.
"""

import torch
import pytest
from rope_pytorch import rope_pytorch
from rope_triton import rope_triton


def allclose_with_info(a, b, rtol=1e-5, atol=1e-5, name="tensor"):
    """Helper function to check if tensors are close and print debug info if not."""
    close = torch.allclose(a, b, rtol=rtol, atol=atol)
    if not close:
        diff = (a - b).abs()
        print(f"\n{name} mismatch:")
        print(f"  Max absolute difference: {diff.max().item()}")
        print(f"  Mean absolute difference: {diff.mean().item()}")
        print(f"  Median absolute difference: {diff.median().item()}")
        print(f"  PyTorch range: [{a.min().item()}, {a.max().item()}]")
        print(f"  Triton range: [{b.min().item()}, {b.max().item()}]")
    return close


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRoPECorrectness:
    """Test suite for RoPE correctness."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    @pytest.mark.parametrize("n_heads", [8, 16])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_rope_correctness(self, batch_size, seq_len, n_heads, head_dim):
        """Test that Triton kernel matches PyTorch reference."""
        # Create random input tensors
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        
        # Apply RoPE with both implementations
        q_torch, k_torch = rope_pytorch(q.clone(), k.clone(), seq_len)
        q_triton, k_triton = rope_triton(q.clone(), k.clone(), seq_len)
        
        # Check correctness
        assert allclose_with_info(q_torch, q_triton, rtol=1e-4, atol=1e-4, name="Q"), \
            f"Q mismatch for shape {q.shape}"
        assert allclose_with_info(k_torch, k_triton, rtol=1e-4, atol=1e-4, name="K"), \
            f"K mismatch for shape {k.shape}"
    
    @pytest.mark.parametrize("theta", [1000.0, 10000.0, 100000.0])
    def test_rope_different_theta(self, theta):
        """Test RoPE with different theta values."""
        batch_size, seq_len, n_heads, head_dim = 2, 256, 8, 64
        
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        
        q_torch, k_torch = rope_pytorch(q.clone(), k.clone(), seq_len, theta=theta)
        q_triton, k_triton = rope_triton(q.clone(), k.clone(), seq_len, theta=theta)
        
        assert allclose_with_info(q_torch, q_triton, rtol=1e-4, atol=1e-4, name="Q"), \
            f"Q mismatch for theta={theta}"
        assert allclose_with_info(k_torch, k_triton, rtol=1e-4, atol=1e-4, name="K"), \
            f"K mismatch for theta={theta}"
    
    def test_rope_preserves_norm(self):
        """Test that RoPE preserves the norm of vectors (rotation property)."""
        batch_size, seq_len, n_heads, head_dim = 2, 128, 4, 64
        
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        
        q_norm_before = torch.norm(q, dim=-1)
        k_norm_before = torch.norm(k, dim=-1)
        
        q_triton, k_triton = rope_triton(q, k, seq_len)
        
        q_norm_after = torch.norm(q_triton, dim=-1)
        k_norm_after = torch.norm(k_triton, dim=-1)
        
        # Rotation should preserve norm
        assert torch.allclose(q_norm_before, q_norm_after, rtol=1e-4, atol=1e-4), \
            "RoPE should preserve Q norm"
        assert torch.allclose(k_norm_before, k_norm_after, rtol=1e-4, atol=1e-4), \
            "RoPE should preserve K norm"
    
    def test_rope_dtype_support(self):
        """Test RoPE with different data types."""
        batch_size, seq_len, n_heads, head_dim = 1, 128, 4, 64
        
        for dtype in [torch.float32, torch.float16]:
            torch.manual_seed(42)
            q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=dtype)
            
            # Should not raise errors
            q_triton, k_triton = rope_triton(q, k, seq_len)
            
            assert q_triton.dtype == dtype, f"Output dtype should match input dtype {dtype}"
            assert k_triton.dtype == dtype, f"Output dtype should match input dtype {dtype}"


if __name__ == "__main__":
    # Run tests manually
    if torch.cuda.is_available():
        print("Running RoPE correctness tests...")
        test_suite = TestRoPECorrectness()
        
        # Test basic correctness
        print("\n1. Testing basic correctness...")
        test_suite.test_rope_correctness(batch_size=2, seq_len=128, n_heads=8, head_dim=64)
        print("✓ Basic correctness test passed")
        
        # Test different theta values
        print("\n2. Testing different theta values...")
        test_suite.test_rope_different_theta(theta=10000.0)
        print("✓ Different theta test passed")
        
        # Test norm preservation
        print("\n3. Testing norm preservation...")
        test_suite.test_rope_preserves_norm()
        print("✓ Norm preservation test passed")
        
        # Test dtype support
        print("\n4. Testing dtype support...")
        test_suite.test_rope_dtype_support()
        print("✓ Dtype support test passed")
        
        print("\n✅ All tests passed!")
    else:
        print("CUDA not available, skipping tests")
