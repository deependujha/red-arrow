---
title: Warp Shuffles & Advanced Primitives
type: docs
sidebar:
    open: false
weight: 505
---

## What are Warp Shuffles?

Warp shuffles allow threads within the same warp to **directly read each other's registers**.

Historically, for Thread A to talk to Thread B, Thread A had to write to Shared Memory, execute `__syncthreads()`, and then Thread B could read it. Shuffles bypass shared memory completely, moving data across lanes in a single clock cycle.

### The Shuffle API Signature

Every shuffle primitive follows this general signature:

```cpp
int __shfl_X_sync(unsigned int mask, T var, int srcLane, int width=warpSize);

```

* **`mask`**: 32-bit active thread mask (use `__activemask()`).
* **`var`**: The register variable you want to share/read.
* **`srcLane` / `delta**`: The target lane index or offset.
* **`width`**: Must be a power of 2 $\le$ 32 (default is 32). If set to 16, the warp treats itself as two independent 16-thread sub-warps (logical warps).

---

## The Four Shuffle Primitives

### 1. `__shfl_sync` (Direct Broadcast)

Every thread reads the value of `var` from one specific, absolute lane (`srcLane`).

```cpp
// Lane 5 broadcasts its value of 'val' to every thread in the warp
float broadcast_val = __shfl_sync(__activemask(), val, 5);

```

### 2. `__shfl_up_sync` (Shift Right)

Shifts data up the warp by `delta` places. Thread's new value comes from `lane_id - delta`.

```cpp
// Thread 4 gets value from Thread 2. First 'delta' threads (0 and 1) keep their original 'val'
float shifted_val = __shfl_up_sync(__activemask(), val, 2);

```

### 3. `__shfl_down_sync` (Shift Left)

Shifts data down the warp by `delta` places. Thread's new value comes from `lane_id + delta`. **This is the foundational primitive for parallel reductions.**

```cpp
// Thread 0 gets value from Thread 2. Last 'delta' threads keep their original 'val'
float shifted_val = __shfl_down_sync(__activemask(), val, 2);

```

### 4. `__shfl_xor_sync` (Butterfly Exchange)

Calculates a source lane ID by performing a bitwise XOR (`lane_id ^ mask`). **Perfect for tree-structured reductions.**

```cpp
// Pairwise exchange between neighbors: 0<=>1, 2<=>3, 4<=>5...
float neighbor_val = __shfl_xor_sync(__activemask(), val, 1); 

// Exchange across half-warps (distance 16): 0<=>16, 1<=>17...
float half_warp_val = __shfl_xor_sync(__activemask(), val, 16);

```

---

## Cheat Sheet: Vote vs. Match vs. Shuffle

Here is how all the primitives you listed break down by intent:

| Category | Primitive | Return Value | Common Use Case |
| --- | --- | --- | --- |
| **Vote** | `__any_sync` | `bool` (True if any thread is true) | Error checking, early exit conditions. |
| **Vote** | `__all_sync` | `bool` (True if all threads are true) | Convergence verification. |
| **Vote** | `__ballot_sync` | `uint32_t` (Bitmask of matching threads) | Compaction, stream filtering. |
| **Match** | `__match_any_sync` | `uint32_t` (Bitmask of threads with *same value*) | Grouping identical keys (e.g., hash tables). |
| **Match** | `__match_all_sync` | `uint32_t` (Bitmask if *all* match, else 0) | Asserting warp uniformity for a value. |
| **Shuffle** | `__shfl_sync` | `T` (Value from absolute lane) | Broadcasting coefficients or scalar values. |
| **Shuffle** | `__shfl_down_sync` | `T` (Value from higher lane) | Inline warp reduction loops. |
| **Shuffle** | `__shfl_up_sync` | `T` (Value from lower lane) | Prefix sums / cumulative scans. |
| **Shuffle** | `__shfl_xor_sync` | `T` (Value from flipped bit lane) | Butterfly reduction, sorting networks. |

> **Note on `__uni_sync**`: There is no official standard `__uni_sync` primitive in CUDA. If you encounter it in codebases, it is typically a custom helper macro wrapping `__all_sync(mask, pred)` or a shorthand used to check if a predicate is **uniform** across all active threads.

---

## Interview Pattern: Warp-Level Reduction (No Shared Memory)

This is a classic question: *"How do you sum 32 elements inside a warp without using shared memory?"* You use a unrolled loop with `__shfl_down_sync`.

```cpp
__device__ inline float warpSum(float val) {
    unsigned int mask = __activemask();
    
    // Each step folds the warp in half
    val += __shfl_down_sync(mask, val, 16); // 16 lanes away: 0+16, 1+17...
    val += __shfl_down_sync(mask, val, 8);  // 8 lanes away
    val += __shfl_down_sync(mask, val, 4);  // 4 lanes away
    val += __shfl_down_sync(mask, val, 2);  // 2 lanes away
    val += __shfl_down_sync(mask, val, 1);  // 1 lane away

    // Lane 0 now contains the grand total for the entire warp
    return val; 
}

```

---

## Advanced: Match Primitives (Volta+)

`__match_any_sync` and `__match_all_sync` are specialized primitives for string/key matching. Threads pass a variable, and the hardware returns a mask of which *other* threads hold the exact same value.

```cpp
// Example: Value grouping
// Thread 0 has key=A, Thread 1 has key=B, Thread 2 has key=A, Thread 3 has key=A
int my_key = ...; 

unsigned int match_mask = __match_any_sync(__activemask(), my_key);

// For Threads 0, 2, and 3, match_mask will return binary 1101 (lanes 0, 2, 3 are set)
// For Thread 1, match_mask will return binary 0010 (only lane 1 matches itself)

```

---

## Crucial Alignment Guard for Shuffles

> [!CAUTION]
> If a thread attempts to read from an **inactive** lane or an **out-of-bounds** lane (e.g., `lane_id + delta >= 32` during a `__shfl_down_sync`), the primitive does not crash. Instead, it returns the calling thread's **own original value** of `var`.
> When writing reductions, ensure you only read the returned result on lanes that actually received valid shifted data.

---

With Atomics, Warp Primitives, and Shuffles now locked into your notes, you have the entire structural toolkit ready. Are you ready to dive into the core implementation of the parallel block reduction algorithm next?
