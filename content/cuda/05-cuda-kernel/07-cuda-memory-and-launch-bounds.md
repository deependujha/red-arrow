---
title: CUDA memory & launch bounds
type: docs
math: true
sidebar:
open: false
weight: 507
---

## 1. CUDA Memory Hierarchy & Latency

To effectively optimize kernels or understand Nsight Compute/Systems profiles, you must memorize the physical location, bandwidth, and latency characteristics of each memory layer.

### Latency & Bandwidth Cheat Sheet

| Memory Type | Physical Location | Scope | Latency (Clock Cycles) | Relative Speed |
| --- | --- | --- | --- | --- |
| **Registers** | On-Chip (per SM) | Per Thread | ~1 cycle | Immediate |
| **Shared Memory / L1** | On-Chip (per SM) | Per Thread Block | ~20 - 30 cycles | ~10x slower than registers |
| **L2 Cache** | On-Chip (Shared across SMs) | Device-wide | ~200 - 300 cycles | ~10x slower than L1 |
| **Global Memory (VRAM)** | Off-Chip (HBM / GDDR) | Device-wide | ~400 - 800+ cycles | **The Ultimate Bottleneck** |

---

### Deep-Dive Per Layer

### A. Registers

* **What it is:** The fastest storage on the GPU, allocated dynamically to active threads.
* **Capacity:** Extremely limited (typically 64KB per SM on modern architectures like Ampere/Hopper). Each individual thread can address a maximum of 255 registers.
* **The Trap:** **Register Spilling.** If your kernel uses more variables than the allocated registers per thread allow, the compiler silently "spills" the excess variables into **Local Memory** (which physically resides in slow Global VRAM, cached in L1/L2). This destroys performance.

### B. Shared Memory (SMem) & L1 Cache

* **What it is:** On-chip memory shared by all threads within a single thread block. On modern architectures, SMem and the L1 cache share the same physical hardware block, and the split can often be configured.
* **Key Behavior:** Managed explicitly by the programmer (`__shared__`). Excellent for caching frequently accessed data (e.g., matrix tiles) to avoid redundant global memory trips.
* **The Profiling Catch:** **Bank Conflicts.** Shared memory is divided into 32 equally sized memory banks that can be accessed simultaneously. If two or more threads in a warp access different addresses within the *same* bank, the hardware serializes the accesses.

### C. L2 Cache

* **What it is:** A device-wide cache shared by all SMs. It caches reads and writes to global memory.
* **Why it matters in profiling:** When Nsight Compute shows a high **L2 Hit Rate**, it means your grid-stride loops or memory access patterns are localized enough that data didn't have to be fetched from external VRAM.

### D. Global Memory (VRAM)

* **What it is:** The massive off-chip pool of memory (GDDR6 or HBM).
* **The Golden Rule:** **Coalescing.** Global memory operations are serviced in 32-byte, 64-byte, or 128-byte memory transactions. If threads in a warp access contiguous memory locations, those 32 threads can be served with a single memory transaction. If accesses are scattered (strided or random), the GPU issues multiple transactions for the same warp, wasting massive amounts of bandwidth.

> **Profiling Mental Rule:** If your profiler says you are **Memory Bound**, look at your L2 hit rate and global memory coalescing efficiency. If you are **Compute Bound**, look at warp execution efficiency or register/shared memory limitations.

---

## 2. Kernel Tuning: `__launch_bounds__`

### The Problem it Solves

The compiler (`nvcc`) is aggressive. If you don't give it constraints, it will try to maximize the speed of a single thread by giving it as many registers as it needs (up to 255).

However, register allocation is a zero-sum game per SM. If one thread uses 64 registers, fewer total threads can run concurrently on that SM, crashing your **Occupancy** (the ratio of active warps to maximum supported warps).

### Syntax & Mechanics

You apply `__launch_bounds__` directly to a global kernel definition to explicitly tell the compiler the constraints of your execution configuration:

```cpp
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
my_optimized_kernel(const float* input, float* output) {
    // Kernel code here
}

```

* **`maxThreadsPerBlock`** *(Required)*: The absolute maximum number of threads per block you intend to launch this kernel with.
* **`minBlocksPerMultiprocessor`** *(Optional, Defaults to 1)*: The minimum number of concurrent thread blocks you want the compiler to guarantee can fit on a single Streaming Multiprocessor (SM).

---

### How the Compiler Changes Behavior

When you provide these numbers, the compiler calculates exactly how many registers it can allow *per thread* without violating your block layout goals.

For example, if you declare:

```cpp
__global__ void __launch_bounds__(1024, 2) my_kernel(...)

```

The compiler reasons: *"The user wants 2 blocks of 1024 threads to fit on one SM. That means $1024 \times 2 = 2048$ threads must live on this SM simultaneously. The hardware has a fixed pool of registers per SM. Therefore, I am strictly capped to $X$ registers per thread. If the code tries to use more, I must spill them to local memory rather than lowering occupancy."*

### Strategic Tuning Guide

Use `__launch_bounds__` to fine-tune the delicate balance between **Instruction-Level Parallelism (ILP)** and **Thread-Level Parallelism (TLP)**:

1. **For Compute-Heavy Kernels (High Register Pressure):** If your kernel does heavy math per thread, it needs more registers to prevent local memory spilling. Use lower occupancy targets (e.g., lower `minBlocksPerMultiprocessor`) to let the compiler give each thread more breathing room.
2. **For Memory-Bound Kernels (Latency Hiding):** If your kernel spends time waiting for VRAM, you need high occupancy to hide that latency. Use `__launch_bounds__` to force a higher block count per SM. This restricts register usage per thread, freeing up space to host more warps that can take over execution while other warps wait on memory fetches.

---

## 3. Quick Reference Template for Code Integration

```cpp
#include <cuda_runtime.h>

// Tuning targets: 256 threads per block, aiming for 4 blocks per SM 
// to maximize latency hiding during global memory sweeps.
#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM     4

__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
vector_add_tuned(const float* __restrict__ A, const float* __restrict__ B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Using restricted pointers ensures the compiler can leverage 
    // read-only cache paths efficiently.
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

```
