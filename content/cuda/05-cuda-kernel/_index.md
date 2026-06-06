---
title: CUDA kernels
type: docs
prev: cuda/04-profiler/01-basics.md
sidebar:
  open: false
weight: 500
---

## Using math functions in CUDA

- For `float` computations, prefer CUDA's fast intrinsic functions such as `__expf`, `__sinf`, `__cosf`, and `__logf` when maximum numerical precision is not required. 

- These intrinsics are typically faster than their standard counterparts (`expf`, `sinf`, `cosf`, `logf`) because they map more directly to hardware implementations, trading a small amount of accuracy for improved performance.
- This tradeoff is often acceptable in machine learning and other throughput-oriented workloads.

- `fminf`, `fmaxf`, `fmaf` (x,y,z -> x*y+z), etc for math operations in cuda kernel.

```cpp
  output[idx] = fmaxf(fminf(input[idx], hi), lo);
```


---

## Todo:

Good place to be — these are exactly the things that separate "I read about CUDA" from "I can actually read kernel code."
Here's what I'd consider the core vocabulary you need to be fluent in:
Things you already have

* Grid-stride loops
* `reinterpret_cast` + vectorized loads
* Occupancy mental model
* Memory coalescing
What's likely still fuzzy
Shared memory patterns — declaring `__shared__`, tiling (loading a chunk of global memory into smem so a block can reuse it), and why this matters for reducing global memory traffic. You've seen bank conflicts mentioned but probably haven't built the full mental model yet.
Warp-level primitives — `__shfl_sync`, `__ballot_sync`, `__reduce_add_sync`. These let threads within a warp communicate without shared memory. Reduction kernels live here. Very common in LeetGPU problems.
Atomic operations — `atomicAdd`, `atomicMax`, `atomicCAS`. When multiple threads write to the same address. `atomicCAS` (compare-and-swap) is the primitive everything else is built on.
Thread synchronization — `__syncthreads()` vs `__syncwarp()`. When you need one, when you need the other, and what happens if you get it wrong (deadlock or data race).
Reduction pattern — probably the single most important algorithmic pattern in CUDA. Parallel sum, max, etc. Starts naive, gets progressively optimized through warp shuffles. Shows up everywhere.
Memory hierarchy mental model — registers → shared memory → L1/L2 cache → global memory, with rough latency numbers attached. Without this, profiling output doesn't make sense.
`__launch_bounds__` — hints to the compiler about max threads/block and min blocks/SM, affects register allocation. You'll see it in optimized kernels.
Cooperative Groups — modern replacement for raw `__syncthreads`. Lets you reason about sync at block, warp, or tile granularity explicitly. Newer LeetGPU problems use this.
Priority order I'd suggest

1. Shared memory + tiling (foundational, everything builds on it)
2. Reduction pattern (algorithmic core, tests warp primitives + smem together)
3. Warp shuffles (once you do reduction, shuffles click naturally)
4. Atomics (simpler, but important for correctness)
5. Memory hierarchy latency numbers (makes profiling readable)
6. `__launch_bounds__` + Cooperative Groups (polish, useful but not blocking)
Want to go through these one by one the same way — I explain, you ask questions, then we shape notes? Shared memory + tiling would be the natural next one.

reply back and then we will slowly start learning and being comfortable with them