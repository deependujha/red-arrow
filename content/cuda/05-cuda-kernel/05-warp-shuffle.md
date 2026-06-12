---
title: Warp Shuffles & Advanced Primitives
type: docs
math: true
sidebar:
    open: false
weight: 505
math: true
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

## Match Primitives

Introduced in the Volta architecture, `match primitives allow threads within a warp to instantly map out which other threads are holding the **exact same value** in a specific register`.

Instead of comparing values across threads manually using shared memory loops or multiple shuffle steps, the hardware performs a warp-wide broadcast comparison in a single cycle.

---

## 1. `__match_any_sync`

```cpp
unsigned int mask = __match_any_sync(unsigned int mask, T value);

```

* **Behavior:** Every thread passes its own `value`. The hardware looks across the warp and returns a 32-bit bitmask to each thread, showing every lane that passed an identical value.
* **Result:** Threads with the same value get the exact same bitmask. Threads with a unique value get a mask with only their own Lane ID bit set.

### Visual Example

If four threads in a warp pass the following values:

```text
Lane 0: "Apple"  → returns mask 1001 (Lanes 0 and 3 match)
Lane 1: "Banana" → returns mask 0010 (Only Lane 1 matches itself)
Lane 2: "Orange" → returns mask 0100 (Only Lane 2 matches itself)
Lane 3: "Apple"  → returns mask 1001 (Lanes 0 and 3 match)

```

---

## 2. `__match_all_sync`

```cpp
unsigned int mask = __match_all_sync(unsigned int mask, T value, int* pred);

```

* **Behavior:** Tests if **every single active thread** in the warp holds the exact same value.
* **Result:** * If all threads match, it returns the full active `mask` and sets the predicate `pred` to `true` (non-zero).
* If even one thread diverges, it returns `0` and sets `pred` to `false` (zero).



---

## Primary Interview Use Case: Warp-Level Aggregation

The absolute classic application for match primitives is **coalescing global atomic updates**.

If multiple threads in a warp want to increment the same index in a global array (e.g., building a histogram or updating a hash table), hitting global memory with 32 distinct `atomicAdd` instructions causes massive serialization.

Using `__match_any_sync`, the warp can group itself by target index, elect a single leader per group to perform one atomic instruction, and scale performance dramatically.

```cpp
__global__ void aggregate_histogram(int* d_bins, int* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int my_bin = d_data[idx]; // The value we want to match on

    unsigned int active = __activemask();
    // 1. Find all peer threads in this warp updating the SAME bin
    unsigned int match_mask = __match_any_sync(active, my_bin);

    // 2. Elect a leader (the lowest lane ID in the matching group)
    int leader_lane = __ffs(match_mask) - 1;
    int my_lane     = threadIdx.x & 31;

    // 3. Count how many total threads in this warp are hitting this exact bin
    int num_matches = __popc(match_mask);

    // 4. Only the leader updates global memory for the whole cohort
    if (my_lane == leader_lane) {
        atomicAdd(&d_bins[my_bin], num_matches);
    }
}

```

---

## Key Hardware Constraints

* **Type Restrictions:** Only supported for 32-bit and 64-bit integer types (`int`, `unsigned int`, `long long`, `unsigned long long`).
* **Floating-Point Workaround:** To use them on `float` or `double`, you must bit-cast them first using `__float_as_int()` or `reinterpret_cast` to avoid compiler errors.

> [!TIP]
> Think of `__match_any_sync` as a hardware-accelerated `GROUP BY` clause for a warp. It is highly optimized for irregular data processing patterns like graph algorithms, sparse matrix operations, and database engines.

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

