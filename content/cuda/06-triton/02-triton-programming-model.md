---
title: Triton Programming Model & Core APIs
type: docs
sidebar:
  open: false
weight: 602
math: true
---

## The Triton Programming Model

Triton abstracts away the low-level thread-block execution model of CUDA. Instead of writing kernel code from the perspective of an individual thread (like CUDA's single-instruction multiple-threads or SIMT), you write sequential code from the perspective of a **Program Instance** (analogous to a CUDA thread block) operating on **Tensors of Blocks**.

### Key Differences from CUDA

```mermaid
graph TD
    subgraph CUDA Programming Model
        A["Thread (Scalar)"] --> B["Warp (32 Threads)"]
        B --> C["Block (Shared Memory/Sync)"]
        C --> D["Grid (GPU-wide Launch)"]
    end
    subgraph Triton Programming Model
        E["Program Instance (Block-level Operations)"] --> F["Execution Grid (Parallel Instances)"]
    end
```

* **No Thread-Level Indexing:** You do not manage individual thread IDs (`threadIdx.x`), warp lane IDs, or manual shared memory offsets.
* **Block-Level Operations:** The fundamental unit of execution is a block tensor (e.g., a $128 \times 64$ float matrix). Operations on these blocks (like element-wise addition or matrix multiplication) are written as single-line array operations.
* **Tensors of Pointers:** Instead of loading a scalar per thread, you construct an array of pointers (a pointer offset from a base address) and load/store the entire block in a single API call.

---

## The Block Sizing Rule: Powers of Two

> [!IMPORTANT]
> In standard Triton, all block dimensions (e.g., `BLOCK_SIZE_M`, `BLOCK_SIZE_N`) **must be compile-time constants (`tl.constexpr`) and must be powers of two** (e.g., 16, 32, 64, 128, 256...).

### Why this is a Hardware Requirement:
1. **Instruction Alignment:** GPUs fetch and process memory in chunks of 32 elements (warps). Power-of-two block dimensions guarantee that memory transactions align perfectly with hardware load/store units.
2. **Compiler Optimization:** Knowing the size is a power of two allows the Triton compiler to unroll loops, partition registers, and optimize memory layouts (like tiling and software pipelining) at compile time.
3. **Tensor Core Layouts:** Tensor Cores (MMA - Matrix Multiply-Accumulate instructions) require inputs in specific dimensions (e.g., $16 \times 16 \times 16$). Power-of-two shapes map cleanly to these matrix math engines.

### Masking & Boundary Conditions
Since block sizes are forced to be powers of two, your inputs will rarely fit perfectly. If your input size is $N = 100$, and your block size is $128$, you must use **boundary masks** to guard against out-of-bounds memory accesses.

---

## Core Triton Language (`tl`) APIs

The `triton.language` module (typically imported as `tl`) provides the building blocks for memory, math, and shaping.

### Memory & Pointer Operations

| API | Signature / Description | Example |
| --- | --- | --- |
| **`tl.arange`** | `tl.arange(start, end)`<br>Generates a 1D range of integers. | `offsets = tl.arange(0, 128)` |
| **`tl.load`** | `tl.load(pointer, mask=None, other=0.0)`<br>Loads block from global memory. | `x = tl.load(x_ptr + offsets, mask=mask)` |
| **`tl.store`** | `tl.store(pointer, value, mask=None)`<br>Writes block to global memory. | `tl.store(z_ptr + offsets, z, mask=mask)` |

### Matrix & Mathematical Operations

| API | Description | Constraints |
| --- | --- | --- |
| **`tl.dot(a, b)`** | Hardware-accelerated matrix multiplication ($C = A \times B$). | Inputs `a` and `b` must be 2D; dimensions must be powers of 2. |
| **`tl.reduce`** | Reduction operations (sum, max, min) across axes. | `row_sum = tl.sum(matrix, axis=1)` |
| **`tl.where`** | Element-wise selection: `where(cond, x, y)`. | Selects from `x` if `cond` is true, else `y`. |

### Shape Manipulation

* **`tl.expand_dims(x, axis)`**: Inserts a dimension of size 1 at the specified axis.
* **`tl.trans(x)`**: Transposes a 2D tensor.
* **`tl.view(x, shape)`**: Reshapes a tensor.

---

## CUDA vs. Triton Side-by-Side: Vector Addition

To see how the programming paradigms shift, here is a simple Vector Add kernel ($Z = X + Y$) written in both languages:

### 1. The CUDA C++ Approach (Thread-Centric)

```cpp
__global__ void vector_add_cuda(const float* x, const float* y, float* z, int N) {
    // 1. Thread gets its unique index in the global grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Perform scalar load, add, and store if within bounds
    if (idx < N) {
        z[idx] = x[idx] + y[idx];
    }
}
```

### 2. The Triton Python Approach (Block-Centric)

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_triton(
    x_ptr, y_ptr, z_ptr, N, 
    BLOCK_SIZE: tl.constexpr # Block size is marked as compile-time constant
):
    # 1. Identify which program instance we are in (analogous to blockIdx)
    pid = tl.program_id(axis=0)
    
    # 2. Compute the memory offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. Create a boundary mask to prevent reading out-of-bounds
    mask = offsets < N
    
    # 4. Load the entire block vector from global memory using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 5. Perform the vector operation (compiled to vector assembly instructions)
    z = x + y
    
    # 6. Store the resulting block back to global memory
    tl.store(z_ptr + offsets, z, mask=mask)
```

---

## Behind the Scenes: Triton Compiler Optimizations

When you run `@triton.jit`, the compiler automates the hardest parts of GPU optimization:

### 1. Automatic Memory Coalescing
In CUDA, you must carefully design thread indexes to access contiguous global memory to achieve coalesced reads (32 contiguous threads reading 32 contiguous floats). In Triton, as long as you construct pointer offsets using contiguous ranges (e.g., `base_ptr + tl.arange(0, BLOCK_SIZE)`), the compiler **automatically coalesces the memory accesses** into wide 128-bit load/store instructions.

### 2. Automatic Shared Memory Management
In CUDA, to reuse data (e.g., in matrix multiplication), you must allocate shared memory (`__shared__`), load data into it, synchronize threads (`__syncthreads()`), perform computations, and synchronize again. 
Triton manages this completely under the hood:
* The compiler analyzes the lifetime of intermediate block tensors.
* It automatically allocates the required SRAM (Shared Memory) inside the SM.
* It inserts necessary memory fences and warp synchronizations at the MLIR level, eliminating race conditions.

### 3. Software Pipelining
For operations like matrix multiply (`tl.dot`), Triton automatically sets up **double buffering** or **multi-buffering**. While the Tensor Cores are computing on the current tile in shared memory, the memory controllers are pre-fetching the next tile from global memory (VRAM) directly into shared memory. This hides global memory latency almost entirely.

---

## Interview Questions & Answers

### Q: Why does Triton require block sizes to be compile-time constants (`tl.constexpr`)?

**Answer:** If block sizes were dynamic runtime variables, the compiler could not determine register allocation, shared memory requirements, or hardware mapping at compilation time. By enforcing compile-time constants (via `tl.constexpr`), the Triton compiler can unroll loops, decide how to tile memory across L1/L2 caches, configure the exact amount of shared memory per SM, and generate specialized machine assembly code targeting the GPU's registers directly.

### Q: How does Triton handle bank conflicts in shared memory?

**Answer:** In CUDA, if multiple threads in a warp access the same memory bank in shared memory, it causes a bank conflict, serializing the accesses. Triton avoids this by analyzing layout descriptors of tensors in its MLIR optimization passes. It automatically applies layout transformations (like swizzling) to memory allocations in SRAM, ensuring memory accesses from different warps and lanes are routed to different banks without requiring manual padding from the developer.

### Q: What is the risk of selecting a block size (`BLOCK_SIZE`) that is too large or too small?

**Answer:** 
* **Too Large:** A block size that is too large (e.g., 1024 or 2048) requires a massive number of registers and shared memory. This can lead to **register spilling** (variables overflowing to slow local memory) or reduce occupancy (fewer blocks running concurrently on the SM because of shared memory limits).
* **Too Small:** A block size that is too small (e.g., 8 or 16) fails to saturate the GPU's memory bandwidth or compute pipelines. It prevents memory coalescing (which requires contiguous accesses of at least 32 or 64 bytes) and underutilizes Tensor Cores, which require larger tiling dimensions to run efficiently.
