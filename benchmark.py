"""
Performance Benchmarks for RoPE Implementation

This module benchmarks the Triton kernel against the PyTorch reference
implementation to measure speedup and performance characteristics.
"""

import torch
import time
from rope_pytorch import rope_pytorch
from rope_triton import rope_triton
import triton


def benchmark_rope(
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    n_warmup: int = 10,
    n_iterations: int = 100,
):
    """
    Benchmark RoPE implementations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        n_heads: Number of attention heads
        head_dim: Dimension per head
        n_warmup: Number of warmup iterations
        n_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary with benchmark results
    """
    # Create input tensors
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
    
    # Warmup
    for _ in range(n_warmup):
        _ = rope_pytorch(q.clone(), k.clone(), seq_len)
        _ = rope_triton(q.clone(), k.clone(), seq_len)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(n_iterations):
        _ = rope_pytorch(q.clone(), k.clone(), seq_len)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / n_iterations * 1000  # ms
    
    # Benchmark Triton
    start = time.time()
    for _ in range(n_iterations):
        _ = rope_triton(q.clone(), k.clone(), seq_len)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iterations * 1000  # ms
    
    # Calculate speedup
    speedup = pytorch_time / triton_time
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "pytorch_ms": pytorch_time,
        "triton_ms": triton_time,
        "speedup": speedup,
    }


def print_benchmark_table(results):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("RoPE Performance Benchmark Results")
    print("=" * 100)
    print(f"{'Config':<40} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 100)
    
    for result in results:
        config = f"B={result['batch_size']}, S={result['seq_len']}, H={result['n_heads']}, D={result['head_dim']}"
        print(f"{config:<40} {result['pytorch_ms']:<15.4f} {result['triton_ms']:<15.4f} {result['speedup']:<10.2f}x")
    
    print("=" * 100)
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Time (ms)',
        plot_name='rope-performance-vs-seqlen',
        args={'batch_size': 4, 'n_heads': 8, 'head_dim': 64},
    )
)
def benchmark_seqlen(seq_len, provider, batch_size, n_heads, head_dim):
    """Benchmark performance vs sequence length."""
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rope_pytorch(q, k, seq_len),
            quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rope_triton(q, k, seq_len),
            quantiles=quantiles
        )
    
    return ms, min_ms, max_ms


def run_benchmarks():
    """Run comprehensive benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    print("Running RoPE Performance Benchmarks...")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Define benchmark configurations
    configs = [
        # Vary batch size
        (1, 512, 8, 64),
        (2, 512, 8, 64),
        (4, 512, 8, 64),
        (8, 512, 8, 64),
        
        # Vary sequence length
        (4, 128, 8, 64),
        (4, 256, 8, 64),
        (4, 512, 8, 64),
        (4, 1024, 8, 64),
        (4, 2048, 8, 64),
        
        # Vary number of heads
        (4, 512, 4, 64),
        (4, 512, 8, 64),
        (4, 512, 16, 64),
        (4, 512, 32, 64),
        
        # Vary head dimension
        (4, 512, 8, 32),
        (4, 512, 8, 64),
        (4, 512, 8, 128),
        (4, 512, 8, 256),
    ]
    
    results = []
    for batch_size, seq_len, n_heads, head_dim in configs:
        result = benchmark_rope(batch_size, seq_len, n_heads, head_dim)
        results.append(result)
    
    print_benchmark_table(results)
    
    # Generate performance plot
    print("Generating performance vs sequence length plot...")
    try:
        benchmark_seqlen.run(print_data=True, save_path='.')
        print("Plot saved as 'rope-performance-vs-seqlen.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    run_benchmarks()
