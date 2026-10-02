---
title: Basics
type: docs
math: true
sidebar:
  open: false
weight: 1001
---

## Locating a thread

Every thread can figure out where it sits in the execution hierarchy:
**lane → warp → block (CTA) → cluster → grid**. CuTe exposes all of these through `cute.arch`.

### Lane : `cute.arch.lane_idx()`
A warp is a group of 32 threads that execute in lockstep. A thread's position inside its warp (0–31) is its **lane id**.

> [!IMPORTANT]
> - For a 1D block this is just `tidx % 32`. For 2D/3D blocks, linearize first:
> - `tid = tidx + tidy * bdimx + tidz * bdimx * bdimy`, then `lane = tid % 32`.
> - Lane ids matter for warp-level ops (shuffles, reductions, MMA fragment layouts), where each lane owns a specific slice of data.

### Warp : `cute.arch.warp_idx()`
The **logical** warp index within the block, i.e. `tid // 32`.

> [!IMPORTANT]
> - This is what you use for warp specialization, e.g. "warp 0 issues TMA loads, warps 1–3 do math."
> - Don't confuse it with `cute.arch.physical_warp_id()`, which reads PTX `%warpid`. That is the hardware warp *slot*, is meant for diagnostics, and can change mid-execution if the warp gets rescheduled. For indexing, always use `warp_idx()`.

### Thread & block : `thread_idx()`, `block_dim()`, `block_idx()`
- `cute.arch.thread_idx()` → `(x, y, z)` index of the thread within its block
- `cute.arch.block_dim()` → `(x, y, z)` size of the block (threads per block)
- `cute.arch.block_idx()` → `(x, y, z)` index of the block within the grid

> [!IMPORTANT]
> - The classic global 1D index is `bidx * bdimx + tidx`. A block (CTA) is the unit that shares shared memory and can `__syncthreads()` together.

### Cluster (Hopper / sm_90+) : `cluster_*`, `block_in_cluster_*`
A **cluster** is a group of blocks guaranteed to be co-scheduled on the same **GPC** (Graphics Processing Cluster, a hardware grouping of several TPCs / SMs on the chip). It's the software counterpart of a GPC, the same way a block is the software counterpart of an SM.

Because the blocks are physically close and running at the same time, they get two extra capabilities:
- **Distributed shared memory (DSMEM):** a block can read/write another block's shared memory in the same cluster, without going through global memory.
- **TMA multicast:** one global → shared memory load can be delivered to several blocks in the cluster at once. In a GEMM, blocks that need the same tile of A or B load it once instead of each fetching it separately, which saves L2/HBM bandwidth.

APIs:
- `cute.arch.cluster_idx()` / `cluster_dim()` → which cluster in the grid / how many clusters
- `cute.arch.block_in_cluster_idx()` / `block_in_cluster_dim()` → this block's `(x, y, z)` position inside its cluster / cluster shape
- `cute.arch.block_idx_in_cluster()` → the same position, linearized to a single int
- `cute.arch.cluster_size()` → number of blocks in the cluster

> [!IMPORTANT]
> - Cluster shape is chosen at launch: `.launch(grid=..., block=..., cluster=[2, 1, 1])`. Grid dims must be divisible by cluster dims.
> - Without a cluster shape, every block is effectively a cluster of size 1.
> - Size limit: 8 blocks is portable; H100 allows up to 16 with a non-portable opt-in. A cluster must fit within one GPC.
> - Blocks in a cluster can synchronize with each other (cluster-wide barrier: `cute.arch.cluster_arrive()` + `cute.arch.cluster_wait()`), similar to `__syncthreads()` but across blocks. Do this before touching another block's smem, so it's guaranteed to be initialized and still alive.
> - Bigger clusters aren't free: the scheduler has to find enough free SMs in one GPC at once, which can hurt occupancy.


### Grid : `cute.arch.grid_dim()`
`(x, y, z)` number of blocks in the grid.

> [!IMPORTANT] Useful for grid-stride loops: `for i in range(global_tid, N, grid_dim_x * bdimx)`.

### Bonus : `cute.arch.dynamic_smem_size()`
Returns the dynamic shared memory size requested at launch.

```python
@cute.kernel
def kernel():
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdim = cute.arch.block_dim()
    gdim = cute.arch.grid_dim()

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    # Print once per warp, from lane 0
    if lane == 0:
        cute.printf("block (%d, %d, %d) | warp %d | thread (%d, %d, %d)",
                    bidx, bidy, bidz, warp, tidx, tidy, tidz)

    # Print once per block, from thread (0, 0, 0)
    if tidx == 0 and tidy == 0 and tidz == 0:
        cute.printf("block dim: %d, %d, %d", bdim[0], bdim[1], bdim[2])
        cute.printf("grid dim: %d, %d, %d", gdim[0], gdim[1], gdim[2])
```

---

## PTX & SASS of the kernel

