---
title: CUDA kernels
type: docs
math: true
prev: cuda/04-profiler/01-basics.md
sidebar:
  open: false
weight: 500
---

## Using math functions in CUDA

- Use `cuda::ceil_div` instead of `(totalElements + threadsPerBlock - 1) / threadsPerBlock`.

```cpp
#include <cuda/std/utility>

// Replaces the older cub::DivideAndRoundUp utility
int blocksPerGrid = cuda::ceil_div(totalElements, threadsPerBlock); 
```
- For `float` computations, prefer CUDA's fast intrinsic functions such as `__expf`, `__sinf`, `__cosf`, and `__logf` when maximum numerical precision is not required. 

> - These intrinsics are typically faster than their standard counterparts (`expf`, `sinf`, `cosf`, `logf`) because they map more directly to hardware implementations, trading a small amount of accuracy for improved performance.
> - This tradeoff is often acceptable in machine learning and other throughput-oriented workloads.

- `fminf`, `fmaxf`, `fmaf` (x,y,z -> x*y+z), etc for math operations in cuda kernel.

```cpp
  output[idx] = fmaxf(fminf(input[idx], hi), lo); // clamps the value to be between lo and hi
```


---

## `__syncthreads()` v/s `syncwarp()`

| Feature           | `__syncthreads()`                    | `syncwarp()`                         |
| ----------------- | ------------------------------------ | ------------------------------------ |
| **Scope**         | All threads in the block (e.g. 1024) | All threads in the warp (32)         |
| **Barrier Type**  | Block-level                          | Warp-level                           |
| **Hardware**      | Explicit Block Barrier               | Implicit Warp-Wide Barrier           |
| **Latency Impact**| Higher latency (waits for full block)  | Lower latency (waits for warp)     |
| **Hardware Cost** | Higher (transistor-level barrier)    | Lower (implicit execution pipeline)  |
| **Use Case**      | Block-wide data synchronization      | Warp-wide communication, reductions  |
| **Error**         | No error on partial participation    | **Undefined** if not all threads sync |

> **Key Insight:**
> `__syncthreads()` forces the entire block (e.g., 1024 threads) to wait. This is computationally expensive because you're waiting for all 1024 threads to arrive, even if only 32 are needed for the synchronization.
>
> `syncwarp()` is much cheaper because it only synchronizes within a warp (32 threads). In Volta and later architectures, this synchronization is often implicit, meaning it doesn't require a heavy hardware barrier but is handled by the warp scheduler.
