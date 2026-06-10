---
title: Warp-Level Primitives
type: docs
sidebar:
    open: false
weight: 504
math: true
---

## The Core Concept: Active vs. Inactive Threads

A warp consists of 32 threads executing in Lock-Step (SIMT). However, due to conditional branching (`if-else`), not all threads in a warp execute the same instruction at the same time.

* **Active Threads:** Threads currently executing the current instruction.
* **Inactive Threads:** Threads skipped due to a branch, or threads out-of-bounds (`idx >= N`).

Warp primitives use a **32-bit bitmask (Active Mask)** where each bit represents a thread in the warp (Bit 0 for Lane 0, Bit 31 for Lane 31). A `1` means the thread is active; `0` means it is inactive.

---

## The Essential Primitive: `__activemask()`

Before doing any warp-level communication, you often need to know exactly which threads are currently alive and executing alongside you.

```cpp
unsigned int mask = __activemask();

```

* Returns a 32-bit unsigned integer representing the calling threads' warp state.
* **Crucial for modern CUDA:** Older CUDA versions assumed all threads were synchronized automatically. Post-Volta independent thread scheduling requires explicit masks for safe warp communication.

---

## Warp Vote Primitives

Vote primitives allow threads within a warp to query the status of other active threads and pool their conditions together in a single cycle.

### 1. `__any_sync(mask, predicate)`

Returns non-zero (true) if **at least one** active thread in the mask evaluates the `predicate` to true.

```cpp
// Check if any thread in the warp encountered an error condition
bool warp_has_error = __any_sync(__activemask(), error_flag);

```

### 2. `__all_sync(mask, predicate)`

Returns non-zero (true) if **every single** active thread in the mask evaluates the `predicate` to true.

```cpp
// Verify if all threads are ready to proceed with a specific optimization block
bool entire_warp_ready = __all_sync(__activemask(), data_loaded == true);

```

### 3. `__ballot_sync(mask, predicate)`

Returns a 32-bit integer where the $N$-th bit is set to `1` if the $N$-th thread of the warp is active and its `predicate` evaluates to true.

```cpp
// Collect a bitfield of all threads matching a filtering criteria
unsigned int positive_match_mask = __ballot_sync(__activemask(), value > 0.0f);

```

---

## What is a "Lane ID" and How to Find It?

Inside a warp, threads are indexed from 0 to 31. This index is called the **Lane ID**. Unlike `threadIdx.x`, CUDA does not provide a built-in variable for it, so you calculate it via bitwise math:

```cpp
int lane_id = threadIdx.x % 32; // Standard way
int lane_id = threadIdx.x & 31; // Faster bitwise equivalent (since 32 is a power of 2)

```

---

## Interview Pattern: Work Convergence / Compaction

A classic interview application for vote primitives is finding out how many threads ahead of you met a certain condition. This is used to pack data or assign unique dynamic indices without using heavy global atomics.

```cpp
__global__ void warp_filter_kernel(float* input, float* output, int* global_counter, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int lane_id = threadIdx.x & 31;
    bool match = (input[idx] > 0.5f);

    // 1. Get the mask of all threads in this warp that matched the condition
    unsigned int active_mask = __activemask();
    unsigned int match_mask = __ballot_sync(active_mask, match);

    if (match) {
        // 2. Count how many threads with a LOWER lane ID also matched.
        // We mask out bits greater than or equal to our lane_id.
        unsigned int lower_lanes_mask = (1U << lane_id) - 1;
        int warp_offset = __popc(match_mask & lower_lanes_mask); 
        
        // __popc counts the number of set bits (population count)
        
        // 3. The lowest active lane in the warp allocates global space for the WHOLE warp
        int base_idx;
        int total_warp_matches = __popc(match_mask);
        
        if (lane_id == __ffs(match_mask) - 1) { // __ffs finds the first set bit (1-indexed)
            base_idx = atomicAdd(global_counter, total_warp_matches);
        }
        
        // (Note: To broadcast base_idx to the rest of the warp safely, we use Shuffles, which we cover next!)
    }
}

```

---

## What Warp Primitives Do Not Do

* **They do not synchronize the whole block:** Warp primitives only operate on the 32 threads within the exact same warp. They completely ignore other warps in the block. If you need block-wide coordination, you still need `__syncthreads()`.
* **They do not work across divergent execution lines safely without a mask:** Never pass a hardcoded `0xFFFFFFFF` mask if there is any chance some threads in the warp are inactive due to an outer `if` statement. Always prefer `__activemask()`.

> [!CAUTION]
> **Independent Thread Scheduling (Volta and Newer)**
> Before Volta architecture, warps shared a single program counter, guaranteeing lockstep execution. Volta introduced independent program counters per thread. This means threads in the same warp can diverge and merge dynamically.
> **Why this matters for interviews:** Because of this, modern CUDA *requires* the `_sync` suffix functions (e.g., `__all_sync`) paired with an explicit execution mask. The older versions without `_sync` are deprecated because they cause intermittent deadlocks on modern GPUs.
