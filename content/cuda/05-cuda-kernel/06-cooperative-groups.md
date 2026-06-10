---
title: Cooperative Groups
type: docs
sidebar:
    open: false
weight: 507
math: true
---

## Why Use It: Modularity and Safety

Before CUDA 9, `__syncthreads()` was the only way to synchronize threads, but it only worked for an **entire thread block**.

If a helper function (like a reduction utility) included `__syncthreads()`, calling it inside a divergent branch (`if-else`) would cause an instant, silent **deadlock** if even one thread skipped the call.

### The Composition Fix

Cooperative Groups treat groups of threads as **first-class objects**. Passing an explicit group object to a device function makes its synchronization requirements clear, eliminating implicit assumptions and avoiding race conditions.

```cpp
// Explicit: The function signature forces you to pass the valid thread block
__device__ int sum(cg::thread_block block, int *x, int n) {
    // ...
    block.sync(); // Perfectly safe because the block handle guarantees entry
    // ...
}

```

---

## Core Types and Properties

To use the API, include the header file and namespace:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

```

### The `thread_group` Base Interface

Every group type in Cooperative Groups implements a standard handle interface:

* `g.size()`: Total number of threads in group `g`.
* `g.thread_rank()`: The unique index of the calling thread within group `g` ($0$ to `size() - 1`).
* `g.sync()`: Performs barrier synchronization among all threads in group `g`.

---

## Group Creations & Partitioning

### 1. `thread_block`

Explicitly represents a standard CUDA thread block.

```cpp
cg::thread_block block = cg::this_thread_block();

block.sync();               // Equivalent to __syncthreads()
dim3 idx = block.group_index();  // Equivalent to blockIdx
dim3 tid = block.thread_index(); // Equivalent to threadIdx

```

### 2. Dynamic vs. Static Tiles (`tiled_partition`)

You can subdivide an existing group into smaller sub-groups (tiles) of sizes like 32, 16, 4, or 2.

#### Dynamic Partition (Evaluated at Runtime)

```cpp
// Divides the thread block into dynamic 32-thread sub-groups
cg::thread_group tile32 = cg::tiled_partition(cg::this_thread_block(), 32);

```

#### Static Partition (Evaluated at Compile Time)

Using template parameters returns a `thread_block_tile<size>`. This lets the compiler aggressively optimize code, like unrolling reduction loops (`#pragma unroll`) and optimizing registers.

```cpp
cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());

```

> **Warp-Level Collectives:** Statically sized tiles expose built-in collective methods directly on the object, mimicking raw warp intrinsics without requiring explicit active masks:
> `tile32.shfl_down(val, delta)`, `tile32.any(pred)`, `tile32.ballot(pred)`, `tile32.match_any(val)`.

---

## Advanced: `coalesced_threads()`

```cpp
cg::coalesced_group active = cg::coalesced_threads();

```

Creates a dynamic group consisting exclusively of all threads in the current warp that are **currently active** (coalesced) at this exact instruction line. This is incredibly useful inside highly divergent execution branches.

### Interview Pattern: Warp-Aggregated Atomics

When many threads in a warp attempt to increment a single global atomic counter, it creates massive hardware serialization. Warp aggregation intercepts this by electing a single active thread to perform one combined atomic instruction for the entire cohort.

```cpp
__device__ int atomicAggInc(int *ptr) {
    // Group all threads currently executing this path
    cg::coalesced_group g = cg::coalesced_threads();
    int prev_global_val;

    // Elect the first active thread (rank 0 within this dynamic group)
    if (g.thread_rank() == 0) {
        // Increment by the total number of active matching threads at once
        prev_global_val = atomicAdd(ptr, g.size());
    }

    // Broadcast the base index to everyone else in the coalesced group
    prev_global_val = g.shfl(prev_global_val, 0);
    
    // Each thread gets its unique position by adding its rank to the base
    return prev_global_val + g.thread_rank();
}

```

---

## Summary of Synchronization Scopes

| Group Type | Creation API | Synchronization Scope |
| --- | --- | --- |
| **`thread_block`** | `cg::this_thread_block()` | Threads inside the **same thread block**. |
| **`thread_block_tile<N>`** | `cg::tiled_partition<N>(parent)` | Sub-groups of size $N$ within a warp. |
| **`coalesced_group`** | `cg::coalesced_threads()` | Only the **active threads** at the current instruction line. |
| **`grid_group`** | `cg::this_grid()` | **Every single thread** launched in the kernel grid (Requires cooperative launch API). |

---

## 💡 Quick Interview Takeaways

* **Hardware Support:** Basic Cooperative Groups (intra-block partitions, tiles, and coalesced groups) work on **Kepler and newer** architectures (Compute Capability 3.0+). Grid-level and multi-GPU synchronization require **Pascal or newer**.
* **Zero Allocation Overhead:** Inter-block synchronization via `grid.sync()` **does not** invalidate or flush registers, local memory, or shared memory. It acts as a standard execution barrier.
* **Safety Over Shortcuts:** Never rely on "implicit warp synchronization" (omitting sync steps assuming warp threads stay perfectly aligned). Modern architectures use independent thread scheduling, making explicit synchronization using `.sync()` or `_sync` variants mandatory.
