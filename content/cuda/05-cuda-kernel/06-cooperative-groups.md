---
title: Cooperative Groups
type: docs
math: true
sidebar:
open: false
weight: 506
---

## The Core Philosophy: Why Cooperative Groups Exist

Before Cooperative Groups (introduced in CUDA 9), the CUDA execution model treated thread synchronization as a rigid, hardware-enforced block boundary. You had `__syncthreads()`, which forced *every* thread in a block to sync, or you had to exit the kernel completely to sync the entire grid.

This created three severe architectural issues:

1. **Divergence Deadlocks:** If you called `__syncthreads()` inside a conditional branch (`if (threadIdx.x < 16)`), the GPU would deadlock because the remaining threads in the block could never reach the barrier.
2. **Brittle Interfaces:** You couldn't write safe, modular library functions. If a function needed to synchronize its threads, it had to assume it owned the entire block, making it impossible to reuse safely in different execution contexts.
3. **Implicit Warp Dangers:** Since the Volta architecture, NVIDIA introduced **Independent Thread Scheduling**. Threads inside the same warp no longer share a single program counter automatically. Relying on "implicit warp synchronization" (omitting sync steps because "threads run in lockstep anyway") now causes intermittent race conditions.

Cooperative Groups explicitly solve this by making thread groups **first-class programming objects**. You define exactly which threads are cooperating, and the synchronization barrier is bound strictly to that explicit group object.

For more details, check here: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html

---

## The Abstract Base: `thread_group`

Every group type in the Cooperative Groups API inherits from or implements the `thread_group` interface. No matter the size of the group (2 threads or 50,000 threads), they all share these foundational API methods:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Common methods available on all group instances:
unsigned int size()        // Total number of threads contained within this group.
unsigned int thread_rank() // The 0-indexed rank of the calling thread *inside this group*.
bool is_valid()            // Returns true if the group was constructed successfully.
void sync()                // Barrier synchronization strictly for the threads in this group.

```

---

## 1. Thread-Block Scope (`thread_block`)

This explicitly captures the standard thread block context. It is a clean, modern replacement for legacy block variables.

```cpp
cg::thread_block block = cg::this_thread_block();

// API Mapping to legacy CUDA:
dim3 block_idx = block.group_index();  // Replaces blockIdx
dim3 thread_idx = block.thread_index(); // Replaces threadIdx
block.sync();                          // Replaces __syncthreads()

```

---

## 2. Warp-Subdivision & Tiling (`tiled_partition`)

### Why Tiling is Useful

In massive parallel workloads, forcing an entire block of 256 or 512 threads to wait for each other at a `__syncthreads()` barrier creates heavy execution stalls.

**Tiling** allows you to carve a thread block into smaller, independent sub-teams (tiles) of size 2, 4, 8, 16, or 32. These tiles can complete local computations (like a small row reduction or a local matrix dot-product) and synchronize *among themselves* without stalling the rest of the block.

### Dynamic Tiles vs. Static Tiles

There are two ways to create tiles. **Always prefer Static Tiles** for performance-critical kernels.

#### Dynamic Tiles (Evaluated at Runtime)

Created by passing a runtime integer size. They are versatile but carry slight overhead because sizes aren't known at compile time.

```cpp
// Runtime partitioning
cg::thread_group tile_rt = cg::tiled_partition(cg::this_thread_block(), 32);

```

#### Static Tiles (Evaluated at Compile Time)

Created using template parameters. This returns a `cg::thread_block_tile<Size>` object.

```cpp
cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
cg::thread_block_tile<8>  tile8  = cg::tiled_partition<8>(cg::this_thread_block());

```

### Why Static Tiles are Superior: Built-in Warp Collectives

When the compiler knows the tile size at compile time, it optimizes register allocations and unrolls loops automatically. Furthermore, static tiles grant you direct access to highly optimized, hardware-level warp primitives **without needing to pass explicit active masks or calculate Lane IDs manually**:

```cpp
int lane_rank = tile32.thread_rank(); // Automatically maps to 0-31 inside this tile

// Direct register communication (No shared memory needed!)
int upper_val = tile32.shfl_down(my_val, 4); 
int bcast_val = tile32.shfl(my_val, 0);

// Vote and Match directly on the tile object
bool any_match = tile32.any(my_val > 10);
unsigned int match_mask = tile32.match_any(my_val);

```

---

## 3. Grid-Level Scope (`grid_group`) & Global Sync

The `grid_group` enables synchronization **across different thread blocks** inside the same kernel launch.

### The Syntax

```cpp
cg::grid_group grid = cg::this_grid();

// Syncs every single block launched in this kernel
grid.sync(); 

```

### Host-Side Launch Mechanics

You **cannot** launch a kernel using standard `<<<blocks, threads>>>` syntax if it invokes `grid.sync()`. If you do, it will trigger a runtime error or crash. You must launch it from the host CPU using the explicit `cudaLaunchCooperativeKernel` C-API.

Here is the exact boilerplate required to configure and launch a cooperative grid safely:

```cpp
#include <cuda_runtime.h>
#include <iostream>

// The Device Kernel
__global__ void my_cooperative_kernel(float* data, int N) {
    cg::grid_group grid = cg::this_grid();
    
    // Perform work phase 1...
    
    grid.sync(); // Guarantees ALL blocks have finished phase 1 before any block moves to phase 2
    
    // Perform work phase 2...
}

// The Host Launch Function
void launch_cooperative_work(float* d_data, int N) {
    int threads_per_block = 256;
    
    // 1. Calculate how many blocks can physically fit on the GPU simultaneously
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, 
        my_cooperative_kernel, 
        threads_per_block, 
        0 // dynamic shared memory size in bytes
    );
    
    // Total blocks must NEVER exceed total physical capacity
    int blocks_per_grid = num_sms * max_blocks_per_sm;

    // 2. Set up kernel arguments array
    void* kernel_args[] = { (void*)&d_data, (void*)&N };

    // 3. Launch via the Cooperative C-API
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)my_cooperative_kernel,
        blocks_per_grid,
        threads_per_block,
        kernel_args,
        0,    // Shared memory bytes
        nullptr // Stream (nullptr defaults to the legacy default stream)
    );
    
    if (err != cudaSuccess) {
        std::cerr << "Cooperative launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}

```

### The Critical Grid-Sync Hardware Constraint

Standard CUDA kernels allow you to over-subscribe the GPU (e.g., launching 50,000 blocks on a GPU that only has room to run 80 blocks concurrently). The hardware scheduler just queues the remaining blocks and processes them as older blocks finish.

**This is strictly prohibited with `grid.sync()**`. Because `grid.sync()` acts as a global barrier where every block waits for every other block to check in, **all launched blocks must reside on the physical GPU hardware at the exact same moment.** If you launch more blocks than the hardware can hold, the first set of blocks will occupy the SMs and wait infinitely at `grid.sync()` for the queued blocks to start. But the queued blocks can never start because the active blocks are refusing to yield the SMs. **Result: Instant, unrecoverable hardware deadlock.**

---

## 4. Real-World Application: Solving the Softmax / RMSNorm Multi-Pass Problem

To see why `grid_group` is revolutionary for deep learning compilers, look at the **Softmax** activation function formula:

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

To safely calculate this over a massive array that spans across multiple thread blocks, your algorithm requires three consecutive phases:

1. **Pass 1:** Find the maximum value ($x_{\max}$) across the *entire input array* (to prevent numeric overflow during exponentiation).
2. **Pass 2:** Subtract $x_{\max}$ from each element, calculate $e^{x_i - x_{\max}}$, and compute the *global sum* of all these exponentials.
3. **Pass 3:** Divide each individual exponential by that global sum to generate the final normalized probability.

### The Legacy Problem (Without Cooperative Groups)

Because Pass 2 absolutely depends on the final global maximum from Pass 1, you cannot let threads start exponentiating until you know *every single block* has finished searching for its local maximum.

Without Cooperative Groups, you were forced to break this up into **three completely separate kernel launches**:

```text
Host Launch Kernel 1 (Find Global Max) -> Global Memory write -> GPU Synchronize to Host ->
Host Launch Kernel 2 (Compute Global Sum) -> Global Memory write -> GPU Synchronize to Host ->
Host Launch Kernel 3 (Compute Final Division)

```

This strategy incurs severe host-side launch overhead penalties and constantly forces intermediate data out of fast L1/L2 caches back into slow global VRAM.

### The Cooperative Groups Solution

With `grid.sync()`, you can fuse all three stages into a **single kernel launch**. Data stays resident in registers and local cache lanes across all three passes.

```cpp
__global__ void fused_softmax_kernel(float* d_in, float* d_out, float* d_global_scratch, int N) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // --- PASS 1: FIND GLOBAL MAX (Grid-Stride Loop) ---
    float local_max = -GIGANTIC_NUMBER;
    for (int i = idx; i < N; i += stride) {
        local_max = max(local_max, d_in[i]);
    }
    
    // (Reduce local_max down to block-level using static tiles/shared memory here...)
    
    if (threadIdx.x == 0) {
        // Use atomicMax to commit this block's max to a single global staging variable
        atomicMaxFloat(&d_global_scratch[0], block_max);
    }

    // CRITICAL GLOBAL BARRIER: Wait for every block to finish updating the global max
    grid.sync(); 
    
    // Past this line, d_global_scratch[0] is guaranteed to hold the true global maximum!
    float global_max = d_global_scratch[0];

    // --- PASS 2: COMPUTE GLOBAL EXPONENTIAL SUM ---
    float local_sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        local_sum += expf(d_in[i] - global_max);
    }

    // (Reduce local_sum down to block-level here...)

    if (threadIdx.x == 0) {
        atomicAdd(&d_global_scratch[1], block_sum);
    }

    // CRITICAL GLOBAL BARRIER: Wait for every block to finish updating the global sum
    grid.sync();

    // Past this line, d_global_scratch[1] is guaranteed to hold the true global sum!
    float global_sum = d_global_scratch[1];

    // --- PASS 3: FINALIZE NORMALIZATION AND WRITE BACK ---
    for (int i = idx; i < N; i += stride) {
        d_out[i] = expf(d_in[i] - global_max) / global_sum;
    }
}

```

---

## Summary Cheat Sheet for Interview Scopes

| Group Type | Creation Primitive | Synchronization Boundary | HW Version Required |
| --- | --- | --- | --- |
| `thread_block` | `cg::this_thread_block()` | Threads within the same Block. | Kepler (CC 3.0+) |
| `thread_block_tile<N>` | `cg::tiled_partition<N>(block)` | Fixed team of size $N$ within a Warp. | Kepler (CC 3.0+) |
| `coalesced_group` | `cg::coalesced_threads()` | Only the threads currently active on that line. | Kepler (CC 3.0+) |
| `grid_group` | `cg::this_grid()` | Every block across the entire launch grid. | Pascal (CC 6.0+) |

Does this deeper breakdown provide the execution context and structural understanding you need to comfortably begin writing your reduction kernels?