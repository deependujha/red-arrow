---
title: Atomic Operations
type: docs
sidebar:
    open: false
weight: 503
---

## The Problem with Race Conditions

When multiple threads concurrently read, modify, and write to the same memory address, a race condition occurs.

```cpp
// BAD: Multiple threads writing to the same counter results in lost updates
__global__ void naive_count(int* counter) {
    (*counter)++; // Read-Modify-Write is not thread-safe
}

```

Standard memory operations are not atomic. Without protection, updates overwrite each other, leading to incorrect, non-deterministic results.

---

## What Atomic Operations Do

Atomic operations perform a **Read-Modify-Write (RMW)** sequence as a single, uninterrupted operation. No other thread can access the memory address until the operation completes.

```cpp
// GOOD: Safe concurrent updates
__global__ void atomic_count(int* counter) {
    atomicAdd(counter, 1); 
}

```

* **Hardware-level lock:** Handled directly by the memory controllers (L2 cache or global memory).
* **Return Value:** Every atomic API returns the **old value** stored at that address *before* the modification.

---

## Common Atomic APIs

CUDA provides built-in atomic functions for arithmetic and bitwise operations.

### Arithmetic Atomics

| Function | Supported Types |
| --- | --- |
| `atomicAdd(ptr, val)` | `int`, `uint`, `long long`, `float`, `double`, `__half` |
| `atomicSub(ptr, val)` | `int`, `uint` *(For floats/doubles, use `atomicAdd` with negative values)* |
| `atomicExch(ptr, val)` | `int`, `uint`, `long long`, `float` *(Swaps value, returns old)* |
| `atomicMin(ptr, val)` | `int`, `uint`, `long long`, `ulonglong` |
| `atomicMax(ptr, val)` | `int`, `uint`, `long long`, `ulonglong` |
| `atomicInc(ptr, val)` | `uint` *(Clamps to `val`: rolls back to 0 if `old >= val`)* |
| `atomicDec(ptr, val)` | `uint` *(Clamps to `val`: rolls back to `val` if `old == 0` or `old > val`)* |

### Bitwise Atomics

Available only for integer types (`int`, `uint`, `long long`, `ulonglong`):

* `atomicAnd(ptr, val)`
* `atomicOr(ptr, val)`
* `atomicXor(ptr, val)`

---

## Performance Considerations & Scopes

Atomics inherently serialize execution for threads targeting the exact same address. If 32 threads in a warp hit the same address, they will execute sequentially, hurting throughput.

### Memory Scopes (CUDA 9+)

To optimize performance, you can limit the scope of the atomic visibility using `atomicXX_system`, `atomicXX_block`, or `atomicXX_device`.

```cpp
// Only atomic relative to threads within the same thread block (Fastest)
atomicAdd_block(&shared_counter, 1);

// Atomic relative to all threads on the GPU device (Default behavior)
atomicAdd(&global_counter, 1);

// Atomic across Host and Device (for unified/pinned memory across CPU/GPU)
atomicAdd_system(&managed_counter, 1);
```

### Shared Memory vs. Global Memory Atomics (Reduction Pattern)

A common performance pattern is to first accumulate results locally using fast shared memory atomics, and then write the final block-level sum to global memory with a single global atomic operation. This dramatically reduces contention.

```cpp
__global__ void block_atomic_sum(const float* input, float* global_sum, int N) {
    // Shared memory allocated for the block
    __shared__ float s_sum;
    
    // Thread 0 initializes the shared counter
    if (threadIdx.x == 0) {
        s_sum = 0.0f;
    }
    __syncthreads(); // Ensure initialization is visible
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Accumulate locally using block-scoped atomic additions
        atomicAdd_block(&s_sum, input[idx]);
    }
    __syncthreads(); // Wait for all block updates to complete
    
    // Thread 0 commits the block's total to the global counter
    if (threadIdx.x == 0 && s_sum != 0.0f) {
        atomicAdd(global_sum, s_sum);
    }
}
```

---

## Important Patterns for Interviews

### 1. Global Counter / Dynamic Indexing

When threads filter data and need to write to a packed output array sequentially:

```cpp
__global__ void filter_kernel(float* input, float* output, int* d_count, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (input[idx] > 0.5f) {
        // Get a unique index for this thread and increment the global counter
        int write_idx = atomicAdd(d_count, 1); 
        output[write_idx] = input[idx];
    }
}

```

### 2. Implementing Custom Atomics with CAS (Compare-And-Swap)

If a type or operation is not natively supported (e.g., `atomicMin` for `float` on older hardware), you must use `atomicCAS`.

- `atomicCAS(ptr, compare, val)` checks if `*ptr == compare`. If true, it sets `*ptr = val`.
- It always returns the old value.
- `atomicCAS` only works on `integers (int, unsigned int, etc.)`.

```cpp
// Implementing a custom atomicMax for float using CAS
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        // Use __int_as_float and __float_as_int to safely compare bits
        float old_float = __int_as_float(assumed);
        float new_max = (val > old_float) ? val : old_float;
        
        // Attempt to swap the bits
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_max));
        
    // If old != assumed, another thread updated the memory first. Loop and retry.
    } while (assumed != old);

    return __int_as_float(old);
}
```


### 3. Warp-Aggregated Atomics (The Ultimate Optimization)

If all threads in a warp are performing `atomicAdd` on the same global pointer (e.g., writing filtered results to an array), they will serialize, destroying throughput. 

**Warp Aggregation** resolves this:
1. Active threads within the warp query who is active using `__activemask()`.
2. They compute their rank/offset within the active threads using bitwise count-leading-zeros/popcount.
3. The "leader" thread of the warp (first active lane) performs a *single* `atomicAdd` for the whole warp.
4. The leader broadcasts the returned base offset to all other threads in the warp using `__shfl_sync()`.
5. Each thread adds its rank to the base offset to find its unique global write position.

This reduces global atomic collisions by up to 32x.

```cpp
__device__ int warp_aggregate_increment(int* counter) {
    unsigned int active = __activemask();
    
    // Find lane ID (0-31) of the current thread
    int laneId = threadIdx.x % 32;
    
    // Mask out lanes greater than or equal to ours
    unsigned int leader_mask = (1U << laneId) - 1;
    
    // Count how many active threads in the warp are before us
    int rank = __popc(active & leader_mask);
    
    // Count total active threads in the warp
    int total_active = __popc(active);
    
    int base_offset = 0;
    // The first active thread (leader) performs the atomic increment for the whole warp
    if (rank == 0) {
        base_offset = atomicAdd(counter, total_active);
    }
    
    // Broadcast the base offset from the leader thread (lowest set bit index) to everyone else
    int leader_lane = __ffs(active) - 1;
    base_offset = __shfl_sync(active, base_offset, leader_lane);
    
    // Return unique index for each thread
    return base_offset + rank;
}

__global__ void filter_kernel_optimized(float* input, float* output, int* d_count, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (input[idx] > 0.5f) {
        // Warp aggregation replaces simple atomicAdd(d_count, 1)
        int write_idx = warp_aggregate_increment(d_count); 
        output[write_idx] = input[idx];
    }
}
```

---

## What Atomics Do Not Guarantee

* **No Global Ordering:** Atomics guarantee that operations occur without interference, but they **do not** guarantee the order of thread execution. If Thread A and Thread B call `atomicAdd`, which one completes first is non-deterministic.
* **Memory Visibility (Implicit Fences):** Standard CUDA atomics do *not* act as full memory barriers and do not guarantee that other threads immediately see the updated value without explicit fencing (like `__threadfence()`, `__threadfence_block()`, or `__threadfence_system()`).

---

## Modern C++ Atomics in CUDA (`cuda::atomic`)

Starting in CUDA 10.2, the C++ Standard Library in CUDA (`<cuda/std/atomic>`) introduces `cuda::atomic` and `cuda::atomic_ref`. These are modern replacements for legacy C-style atomic functions, allowing C++11-style memory ordering (`acquire`, `release`, `relaxed`) and explicit scoping.

```cpp
#include <cuda/std/atomic>

// Flag synchronization between CPU and GPU (using unified memory and system-wide scope)
__global__ void producer_kernel(cuda::atomic<int, cuda::thread_scope_system>* flag, int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] = 42;
        // Release semantics ensure the write to data[0] is visible before flag is updated
        flag->store(1, cuda::memory_order_release);
    }
}
```

> [!TIP]
> To minimize global memory conflict, use shared memory atomics (`atomicAdd_block`) within a block first, then use a single global `atomicAdd` at the end of the block to commit the shared total to global memory.
