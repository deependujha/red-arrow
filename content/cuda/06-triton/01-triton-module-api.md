---
title: Triton Module APIs
type: docs
prev: cuda/06-triton-index
sidebar:
  open: false
weight: 610
---

The top-level `triton` module provides the structural decorators and configuration objects necessary to compile, automatically tune, and optimize custom hardware-backed execution routines. 

---

## Core Module Components

As shown in the official documentation reference from **Screenshot 2026-06-11 at 3.49.58 PM.png**, the root module exposes four essential building blocks:

### 1. `@triton.jit`
* **Purpose:** A decorator used to Just-In-Time (JIT) compile a standard Python function using the Triton compiler chain.
* **Under the Hood:** It overrides normal Python execution for the function. When invoked, it intercepts the inputs, parses the Python Abstract Syntax Tree (AST), translates it into Triton MLIR dialects, and triggers the hardware-specific compiler backend to generate optimized machine code binary streams (e.g., `.cubin`).

### 2. `@triton.autotune`
* **Purpose:** A decorator designed for auto-tuning a function that has already been wrapped with `@triton.jit`.
* **Under the Hood:** High-performance hardware kernels depend heavily on finding the ideal hyper-parameters (such as choosing block sizes or unrolling factors) for a specific GPU architecture. `@triton.autotune` executes benchmarks across different configurations at runtime to discover which one delivers the absolute highest memory bandwidth or FLOPS efficiency.

### 3. `@triton.heuristics`
* **Purpose:** A decorator used to define conditions or explicit logic specifying how the values of certain meta-parameters should be dynamically calculated at runtime.
* **Under the Hood:** Instead of searching blindly or hardcoding parameters, you can use heuristics to dynamically calculate configurations (e.g., determining optimal block tiling parameters based directly on the actual size dimensions of the runtime input tensor matrices).

### 4. `triton.Config`
* **Purpose:** An object that represents an individual, discrete kernel configuration for the auto-tuner to evaluate.
* **Under the Hood:** A `Config` object packages a bundle of optimization choices together—specifying variable definitions like block dimensions (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`), internal compilation variables (like `num_warps`), or target stages for memory pipeline streaming.

---

## Technical Context: Auto-Tuning & Metaprogramming

Understanding how these primitives work together is essential for writing production-ready code:

### The Optimization Workflow
When optimization engineers write code in Triton, they don't guess the best tile shapes. Instead, they pair `Config` with `autotune` to map out the hardware's sweet spot:

```python
# Conceptual architectural combination
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.heuristics({
    'BLOCK_SIZE_K': lambda args: max(16, min(32, args['K'] // 4))
})
@triton.jit
def matmul_kernel(..., M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Kernel computation logic here
    pass

```

### Key Differences: Autotune vs. Heuristics

* Use **`@triton.autotune`** when the best option depends heavily on real-world hardware profiles (like shared memory latency vs. instruction pipelines) and requires empirical profiling to verify execution speed.
* Use **`@triton.heuristics`** when the optimal configuration value is deterministic and can be computed directly from mathematical bounds or constraints in the input tensor shapes.

---

## Interview Questions & Answers

### Q: Why does Triton require compiling functions using `@triton.jit` rather than running pure Python?

**Answer:** Pure Python code runs sequentially on the host CPU and cannot execute natively across thousands of parallel GPU cores. The `@triton.jit` decorator acts as a compiler entry-point. It intercepts the function, bypasses the standard Python interpreter, and compiles the code through an MLIR and LLVM-based system. This transforms the high-level Python code directly into highly optimized machine assembly code (PTX/AMDGPU binary) optimized for the GPU hardware.

### Q: What is the overhead of using `@triton.autotune`? Does it slow down production training?

**Answer:** Auto-tuning introduces a minor compilation warm-up overhead the very first time a kernel is launched with unique input dimensions. Triton compiles and profiles the variant configurations defined in the `triton.Config` objects, benchmarking them directly on the hardware. Once the fastest configuration is identified, Triton caches the optimal compiled binary wrapper. Subsequent calls read straight from the cache with zero execution overhead, making it highly efficient for prolonged production training runs.
