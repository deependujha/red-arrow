---
title: CUDA Architecture
type: docs
prev: docs/
sidebar:
  open: false
weight: 301
---

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model for general-purpose GPU computing.

The key insight: GPUs are optimized for **throughput** — executing millions of operations simultaneously — while CPUs are optimized for **latency** — completing a single operation as fast as possible.

---

## Mental Model

CPU and GPU work as a team. The CPU orchestrates; the GPU crunches.

```
CPU (Host)                        GPU (Device)
──────────────────                ──────────────────────────────
1. Allocate GPU memory   ──────►  VRAM allocated
2. Copy data to GPU      ──────►  Data ready
3. Launch kernel         ──────►  Thousands of threads execute
4. (CPU continues or             Results written to VRAM
   waits)
5. Copy results back     ◄──────  Done
```

The philosophical split:

```
CPU → a few powerful cores, fast at sequential logic
GPU → thousands of lightweight cores, fast at parallel math
```

---

## Execution Hierarchy

Every piece of work on the GPU lives somewhere in this hierarchy:

```
Grid
└── Block  (group of threads; runs on one SM)
    └── Warp  (32 threads; true hardware unit)
        └── Thread  (single execution lane)
```

### Thread

The smallest unit of execution. Each thread runs the same kernel code but on **different data**, identified by its index.

```cpp
__global__ void add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
```

```
Thread 0 → C[0] = A[0] + B[0]
Thread 1 → C[1] = A[1] + B[1]
Thread 2 → C[2] = A[2] + B[2]
...
```

### Block

A group of threads that share **shared memory** and can synchronize with each other.

```cpp
__shared__ float tile[256];   // visible to all threads in this block
__syncthreads();               // barrier: all threads in block wait here
```

Blocks are **independent** — they can run in any order on any SM. This is what allows CUDA to scale across GPUs of different sizes.

### Grid

The collection of all blocks launched for a single kernel call.

```cpp
// Launch 4096 blocks, each with 256 threads
// Total: 4096 × 256 = 1,048,576 threads
kernel<<<4096, 256>>>(args);
```

### Warp

The **true hardware scheduling unit** — 32 threads that physically execute together.

```
1 warp = 32 threads

All 32 threads execute the same instruction simultaneously.
This model is called SIMT: Single Instruction, Multiple Threads.
```

Warps are invisible to your code, but they dominate performance. When you write 256 threads per block, CUDA silently splits that into 8 warps of 32.

---

## Key Components

### Host

The **CPU + system RAM**. Responsible for launching kernels, managing GPU memory, and transferring data.

```cpp
// Allocate on GPU
cudaMalloc(&d_A, size);

// Transfer host → device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

// Launch kernel (asynchronous — CPU doesn't wait)
kernel<<<blocks, threads>>>(d_A);

// Wait for GPU to finish
cudaDeviceSynchronize();

// Transfer device → host
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
```

> **Important:** Kernel launches are **asynchronous** — the CPU continues immediately after launch unless you explicitly synchronize.

### Device

The **GPU + VRAM**. Executes kernels across thousands of threads in parallel.

### Kernel

A function defined with `__global__` that runs on the GPU and is called from the CPU.

```cpp
__global__ void scale(float *data, float factor, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        data[i] *= factor;
}
```

Launched with the `<<<grid, block>>>` syntax, which specifies how many blocks and how many threads per block to use.

### Streaming Multiprocessor (SM)

The **core compute unit** of a GPU. Each SM is a self-contained processor with its own resources:

```
SM
├── CUDA cores          (arithmetic: add, mul, fma)
├── Tensor cores        (matrix math, on Volta+)
├── Warp schedulers     (pick which warp runs next)
├── Register file       (per-thread fast storage)
├── Shared memory       (per-block fast storage)
└── L1 cache
```

When a kernel launches, CUDA distributes blocks across all available SMs. Each SM then splits its blocks into warps and schedules them for execution.

```
Grid
 └── Blocks assigned to SMs
      └── Each SM splits blocks into warps
           └── Warp schedulers issue instructions to CUDA cores
```

### CUDA Core

The **ALU** (arithmetic logic unit) inside an SM. Executes one instruction per clock for a warp. Threads are logical; CUDA cores are the physical silicon that executes them.

---

## Memory Hierarchy

Memory access is where most CUDA performance is won or lost.

```
Speed      Type               Scope          Size
──────     ────────────────   ───────────    ────────
Fastest    Registers          Per thread     ~256 KB/SM
  │        Shared Memory      Per block      16–164 KB/SM
  │        L1 Cache           Per SM         (shared with SMEM)
  │        L2 Cache           Whole GPU      4–50 MB
Slowest    Global Memory      All threads    8–80 GB
```

### Registers

The fastest possible storage — but private to each thread and limited in count.

```
- Zero latency access
- Used automatically for local variables
- Overflow spills to global memory → big performance hit
```

### Shared Memory

Explicitly managed cache shared among all threads in a block. The go-to tool for data reuse.

```cpp
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Each thread loads one element
tile[threadIdx.y][threadIdx.x] = input[row * N + col];

// Sync before using data loaded by other threads
__syncthreads();

// Now all threads can safely read the full tile
```

Common uses: matrix tiling, reductions, prefix sums.

### Global Memory (VRAM)

Main GPU memory — large but high latency (~400–800 cycles).

Performance is dominated by **memory coalescing**: whether threads in a warp access contiguous addresses that can be served in a single transaction.

```cpp
// ✅ Coalesced — threads access consecutive addresses
// thread 0 → A[0], thread 1 → A[1], thread 2 → A[2]...
int i = blockIdx.x * blockDim.x + threadIdx.x;
float val = A[i];

// ❌ Strided — each access is a separate transaction
// thread 0 → A[0], thread 1 → A[1024], thread 2 → A[2048]...
float val = A[threadIdx.x * 1024];
```

### L1 and L2 Cache

L1 is per-SM and shared with shared memory (you can configure the split on some architectures). L2 is shared across the entire GPU — all SMs check L2 before going to VRAM.

---

## Warp Divergence

When threads within the same warp take different code paths, those paths are **serialized**:

```cpp
// ⚠️ This causes divergence within a warp
if (threadIdx.x % 2 == 0)
    A[i] = heavy_computation();   // even threads execute this
else
    A[i] = other_computation();   // odd threads execute this
```

CUDA's SIMT model handles this by masking off threads that don't take a branch:

```
Step 1: run even-thread branch (odd threads idle)
Step 2: run odd-thread branch (even threads idle)
```

Both branches still take time. Half your warp is idle in each step → 2× slower in the worst case.

**Minimize divergence** by structuring control flow so all threads in a warp take the same path, or by ensuring divergent branches are short.

---

## Latency Hiding

GPUs don't try to make any single operation faster. Instead, they **hide latency** by switching to another warp whenever the current warp stalls (e.g., waiting for a memory load).

```
warp 1 → issues memory load (stalls for ~400 cycles)
warp 2 → executes arithmetic instructions
warp 3 → executes arithmetic instructions
warp 4 → executes arithmetic instructions
warp 1 → memory arrives, resumes
```

This means a GPU needs **many active warps** to stay busy. A GPU with 80 SMs, 64 warps per SM, and 32 threads per warp can have:

```
80 × 64 × 32 = 163,840 active threads
```

Keeping those execution units fed is the job of the warp schedulers — and why occupancy matters so much for performance.

---

## Full Execution Flow

```
CPU program starts
      │
      ▼
cudaMalloc  →  allocate GPU memory
      │
      ▼
cudaMemcpy  →  copy host → device
      │
      ▼
kernel<<<G, B>>>()  →  launch (async, CPU continues immediately)
      │
      ▼
CUDA runtime distributes blocks across SMs
      │
      ▼
Each SM splits its blocks into warps (32 threads each)
      │
      ▼
Warp schedulers issue instructions each clock cycle
      │
      ▼
Results written to global memory (VRAM)
      │
      ▼
cudaDeviceSynchronize()  →  CPU waits for GPU to finish
      │
      ▼
cudaMemcpy  →  copy device → host
```

---

## Cuda performance Guide

understanding **why your kernels are slow**, comes down to six concepts:

| Concept | What it means |
|---|---|
| **Occupancy** | How many warps are active vs. the SM maximum |
| **Memory Coalescing** | Whether warp memory accesses form a single transaction |
| **Warp Divergence** | How often threads in a warp take different branches |
| **Bank Conflicts** | Shared memory access patterns that serialize loads |
| **Register Pressure** | Too many registers per thread → fewer warps → less latency hiding |
| **Launch Configuration** | Choosing the right grid/block sizes for your workload |

These six ideas explain 90% of real-world CUDA performance problems.
