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

`atomicCAS(ptr, compare, val)` checks if `*ptr == compare`. If true, it sets `*ptr = val`. It always returns the old value.

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

---

## What Atomics Do Not Guarantee

* **No Global Ordering:** Atomics guarantee that operations occur without interference, but they **do not** guarantee the order of thread execution. If Thread A and Thread B call `atomicAdd`, which one completes first is non-deterministic.
* **No Implicit Fences:** Atomics ensure read-modify-write atomicity, but they do not act as structural memory barriers (`__syncthreads()`) across the grid.

> [!TIP]
> To minimize global memory conflict, use shared memory atomics (`atomicAdd_block`) within a block first, then use a single global `atomicAdd` at the end of the block to commit the shared total to global memory.
