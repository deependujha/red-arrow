---
title: Triton Testing & Benchmarking
type: docs
math: true
sidebar:
  open: false
weight: 603
---

## 1. Correctness Testing: `triton.testing.assert_close`

When writing GPU kernels, floating-point math can vary slightly from the CPU or standard PyTorch due to different execution orders or optimization paths (like Fused Multiply-Add). `assert_close` is Triton’s specialized utility to check if your kernel's output matches a trusted reference (like standard PyTorch).

### Key Concepts

* **`atol` (Absolute Tolerance):** The maximum allowable absolute difference between elements ($|a - b| \le \text{atol}$).
* **`rtol` (Relative Tolerance):** The maximum allowable difference relative to the magnitude of the reference value ($|a - b| \le \text{rtol} \times |b|$).
* **Detailed Error Output:** If a mismatch occurs, it doesn't just fail; it prints out the max absolute error, max relative error, and the exact index where things went wrong.

### Deep-Dive Code Example

```python
import torch
import triton
import triton.language as tl

# Imagine you just ran your custom Triton vector addition kernel
# triton_output = ...
# reference_output = tensor_a + tensor_b

def test_kernel_correctness():
    # Creating dummy data
    ref = torch.randn(1024, device='cuda', dtype=torch.float16)
    out = ref + 1e-4 * torch.randn(1024, device='cuda', dtype=torch.float16) # slight noise
    
    try:
        # Check if they are close within standard FP16 tolerances
        triton.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
        print("✅ Correctness check passed!")
    except AssertionError as e:
        print("❌ Mismatch detected!")
        print(e)

test_kernel_correctness()

```

---

## 2. Core Benchmarking: `triton.testing.do_bench`

You can't just use Python's standard `time.time()` or basic `timeit` for GPU kernels. GPUs are asynchronous; the CPU can finish executing code and moving to the next line while the GPU is still crunching numbers.

`do_bench` handles all the heavy lifting of GPU benchmarking for you automatically:

* **Warm-up iterations:** Runs the kernel a few times first so things like JIT compilation, caching, and GPU clocks stabilize.
* **Stream Synchronization:** Explicitly waits for the GPU to finish (`torch.cuda.synchronize()`) before stopping the timer.
* **Statistical Rigor:** Runs the kernel dozens of times to give you stable metrics (median, 20th percentile, 80th percentile runtime).

### Deep-Dive Code Example

```python
import torch
import triton

# Dummy function mimicking a kernel launch + setup
def py_reference_ops(x):
    return torch.softmax(x, dim=-1)

x = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)

# Benchmark the function
# Returns the median runtime in milliseconds (ms)
median_ms = triton.testing.do_bench(lambda: py_reference_ops(x))

print(f"Median execution time: {median_ms:.4f} ms")

```

### Critical Parameters to Know:

* `warmup` (int): Number of iterations to discard before timing begins (default is usually around 25).
* `rep` (int): Number of repetitions to measure to get reliable statistics (default around 100).
* `fast_flush` (bool): Flushes the L2 cache between iterations by reading a dummy tensor, ensuring you measure actual DRAM bandwidth rather than lucky cache hits.

---

## 3. Advanced Benchmarking: Graphs & Profiling

As kernels scale, infrastructure overhead can mask true kernel execution speeds. Triton provides specialized variations of the benchmarking tool to isolate performance.

### `triton.testing.do_bench_cudagraph`

If your kernel launch overhead (CPU overhead of invoking the driver) is taking up a significant portion of your runtime, you can wrap it in a **CUDA Graph**. This captures the entire execution sequence once and replays it on the GPU with minimal CPU intervention.

* **When to use:** For incredibly fast, low-latency kernels where Python/host overhead distorting the actual kernel runtime is a concern.

### `triton.testing.do_bench_proton` & `do_bench_cudagraph_proton`

* **Proton** is Triton's dedicated, lightweight profiling infrastructure.
* Instead of just giving you a raw time metric, these functions record detailed telemetry and hardware metric traces during the benchmark run.
* **When to use:** When a kernel is slow and you need a granular breakdown of hardware bottlenecks (e.g., instructions, memory stalls) without setting up heavy external visual profilers.

---

## 4. End-to-End Automated Profiling: `Benchmark` & `@triton.testing.perf_report`

Instead of running individual benchmarks manually for every input size, Triton includes a brilliant declarative system to generate comprehensive scaling plots (e.g., Matrix Size vs. TFLOPS or GB/s).

### Concrete Blueprint

Here is exactly how you structure a file to automatically sweep parameters and generate beautiful performance reports.

```python
import torch
import triton
import triton.testing

# Define the sweeping experiment using triton.testing.Benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SIZE'],               # Argument name that changes on the X-axis
        x_vals=[128 * i for i in range(1, 9)],  # X-axis values to sweep through [128, 256, ..., 1024]
        line_arg='provider',            # The argument that distinguishes different lines on the plot
        line_vals=['triton_op', 'torch_op'],   # Label identifiers for the lines
        line_names=['Triton Kernel', 'PyTorch Native'], # Human-readable line names
        styles=[('blue', '-'), ('green', '--')],       # Aesthetics: colors and line types
        ylabel='Execution Time (ms)',   # Y-axis label
        plot_name='vector-add-performance', # Name of the saved plot file
        args={}                         # Optional extra constant arguments to pass to the function
    )
)
def benchmark(SIZE, provider):
    # Setup inputs dynamically based on the current sweep configuration
    x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    y = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    
    # Quantify the exact operation being run
    if provider == 'torch_op':
        return triton.testing.do_bench(lambda: x + y)
    if provider == 'triton_op':
        # Replace this lambda with your actual custom Triton kernel launch wrapper
        # e.g., lambda: triton_vector_add_kernel_launch(x, y, SIZE)
        return triton.testing.do_bench(lambda: x + y * 1.0) 

# Running this command prints a beautifully formatted Markdown table 
# and automatically saves a line plot image to your local directory!
if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)

```

---

## Summary Cheat Sheet

| Tool | True Intent / Purpose |
| --- | --- |
| **`assert_close`** | Ensures GPU kernel outputs are numerically correct against reference tensors (manages precision thresholds). |
| **`do_bench`** | Obtains accurate GPU timing by managing asynchronous execution, warm-ups, and cache flushing. |
| **`do_bench_cudagraph`** | Eliminates CPU-side launch overhead from timing calculations via CUDA Graphs. |
| **`do_bench_proton`** | Captures execution speed along with deep Triton Proton profiling metrics. |
| **`perf_report` / `Benchmark**` | Automates sweeping sizes/configs to spit out markdown tables and line graphs of performance scaling. |

You are now fully locked and loaded on the theory, syntax, language constraints, and verification tools.

Which kernel out of your 17 are you planning to tackle first?