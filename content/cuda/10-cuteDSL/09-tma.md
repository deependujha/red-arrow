---
title: TMA
type: docs
math: true
sidebar:
  open: false
weight: 1009
---

**Hopper (sm_90) and newer only.** An L4 (Ada, sm_89) has `cp.async` but no TMA.

The **Tensor Memory Accelerator** is a copy engine in each SM. With `cp.async`, every thread moves 16 B and computes its own address. With TMA, **one thread** says "copy tile `(bm, bn)`", and the hardware does the address math, the edge clipping and the swizzle.

![cp.async vs TMA](/10-cuteDSL/tma-vs-cpasync.svg)

| | `cp.async` | TMA |
|---|---|---|
| Who issues | every thread, 16 B each | **one warp** (one elected thread inside it) |
| Address math | threads, via the TV layout | hardware, from a **descriptor** |
| Tile edges | you predicate | hardware **zero-fills** loads, **clips** stores |
| Swizzle | any smem layout you like | 4 hardware modes: none, 32, 64, 128 B |
| "Is it done?" | commit → `wait_group` → barrier | an **mbarrier** that counts bytes |
| Direction | gmem → smem | gmem → smem **and** smem → gmem |

---

## 1. The pieces

| Piece | Where | What it is |
|---|---|---|
| **descriptor** (tensor map) | built on the host, passed to the kernel | gmem address, shape, strides, dtype, box size, swizzle mode |
| **TMA atom** | `make_tiled_tma_atom(...)` in `@cute.jit` | "the TMA instruction + its descriptor" |
| **TMA tensor** | returned with the atom | the gmem tensor, but holding **coordinates**, not addresses |
| **smem tile** | `SmemAllocator` | where the box lands; its layout must match the descriptor |
| **mbarrier** | 8 B in smem | counts arriving bytes; tells everyone when the tile is in |

The same five steps as the copy page, with the mbarrier added:

```python
tma_atom, mA_tma = cpasync.make_tiled_tma_atom(op, mA, sA_layout, TILE)     # 1. host: descriptor + atom
gA = cute.local_tile(mA_tma, TILE, (None, None))                             # 2. tile the TMA tensor
tAsA, tAgA = cpasync.tma_partition(tma_atom, 0, cute.make_layout(1),         # 3. partition smem/gmem
                                   cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
cute.copy(tma_atom, tAgA[(None, bm, bn)], tAsA, tma_bar_ptr=mbar)            # 4. one warp issues
cute.arch.mbarrier_wait(mbar, phase)                                         # 5. everyone waits
```

---

## 2. Host side: the descriptor

```python
tma_load, mA_tma = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),   # load; CopyBulkTensorTileS2GOp() for store
    mA,                                  # the gmem tensor
    sA_layout,                           # the smem layout the box lands in
    TILE,                                # the CTA tile, e.g. (32, 128)
)
```

- It runs in `@cute.jit` (host). The descriptor travels to the kernel as a kernel parameter inside the atom.
- CuTe picks the **box** (the rectangle one TMA instruction moves) from `sA_layout` and `TILE`. If one box can't cover the tile, `cute.copy` issues several. Example 2 below gets 2 boxes per tile.
- It returns a `TmaInfo`. Unpack it as `atom, tma_tensor = ...`.

### Rules the descriptor must follow

From the CUDA driver docs for `cuTensorMapEncodeTiled`:

| Rule | Why you care |
|---|---|
| every gmem stride except the contiguous one is a **multiple of 16 B** | fp16 rows need `N % 8 == 0` |
| each box dimension ≤ **256** elements | a `(32, 128)` box is fine |
| with a swizzle, **inner box bytes ≤ swizzle size** | 128 B swizzle → at most 64 fp16 per box row |
| gmem base 16 B aligned | `assumed_align=16` |

The 16 B stride rule is why the tensors are wrapped like this:

```python
from_dlpack(t, assumed_align=16)
    .mark_layout_dynamic(leading_dim=1)                 # shape known at run time, dim 1 contiguous
    .mark_compact_shape_dynamic(mode=1, divisibility=8) # promise: N % 8 == 0 -> row stride is a multiple of 16 B
```

Without the divisibility promise, the compiler can't prove the stride rule and refuses to build the descriptor.

---

## 3. The TMA tensor holds coordinates

`mA_tma` looks like `mA`, but indexing it gives a **coordinate** like `(row, col)`, not an address. TMA instructions take coordinates; the descriptor turns them into addresses. Everything you know still works on it: `local_tile`, slicing, partitioning.

---

## 4. `tma_partition`

```python
gA = cute.local_tile(mA_tma, TILE, (None, None))   # (32, 128, tiles_m, tiles_n)
tAsA, tAgA = cpasync.tma_partition(
    tma_load,
    0, cute.make_layout(1),                         # CTA id and cluster layout: no cluster here
    cute.group_modes(sA, 0, 2),                     # ((32, 128),)
    cute.group_modes(gA, 0, 2),                     # ((32, 128), tiles_m, tiles_n)
)
tAgA[(None, bm, bn)]                                # this CTA's tile
```

- `group_modes(x, 0, 2)` bundles the first two modes into **mode 0 = "one tile"**. `tma_partition` requires that.
- `None` keeps the whole tile; `bm, bn` pick which one.
- There's no `tidx` anywhere. TMA has no per-thread split.

---

## 5. The mbarrier

A barrier object in smem (8 bytes) with three pieces of state:

- **pending arrivals**: how many `arrive`s this round still needs (starts at the init count)
- **tx count**: how many bytes this round still expects
- **phase bit**: 0 or 1, flips each time a round completes

A round (**phase**) completes when **pending arrivals = 0 and tx count = 0**. The barrier then flips its phase bit and re-arms itself for the next round.

![mbarrier phase](/10-cuteDSL/tma-mbarrier-phase.svg)

| Call | Who | Effect |
|---|---|---|
| `mbarrier_init(mbar, 1)` | **one thread** | arrivals per round = 1 |
| `mbarrier_init_fence()` + `barrier()` | all | everyone sees the initialized barrier |
| `mbarrier_arrive_and_expect_tx(mbar, nbytes)` | **one thread** | tx += nbytes, then arrive (arrivals −1) |
| `cute.copy(tma_atom, ..., tma_bar_ptr=mbar)` | one warp | hardware subtracts bytes from tx as they land |
| `mbarrier_wait(mbar, phase)` | everyone who reads the tile | returns once round `phase` (0 or 1) is complete |

> [!IMPORTANT]
> `mbarrier_wait` takes a **parity**, not a round number. Round 0 → wait(0), round 1 → wait(1), round 2 → wait(0) again. Keep a `phase` variable and flip it every round.

> [!WARNING]
> **Parity can't tell round 0 from round 2.** If a slow thread is still waiting for round 0 while the barrier has already finished rounds 0 and 1, it sees parity 0 again and waits forever. Don't let the producer get two rounds ahead of anyone: a CTA `barrier()` per round (example 3), or an "empty" barrier (the pipelining page).

The tx count can briefly go negative if bytes land before `expect_tx`. The round still can't complete early, because the arrival hasn't happened yet. That's why "arrive_and_expect_tx, then copy" and "copy, then arrive_and_expect_tx" both work.

---

## 6. Who runs what

| Operation | Run by | How |
|---|---|---|
| `mbarrier_init`, `arrive_and_expect_tx` | exactly **one thread** | `if warp_idx == 0:` + `with cute.arch.elect_one():` |
| TMA `cute.copy` | exactly **one warp** | `if warp_idx == 0:`, **no** `elect_one` |
| `mbarrier_wait` | every thread that reads the tile | plain call |

- `elect_one()` picks one thread **per warp**. With 4 warps and no `warp_idx` guard, you'd get 4 inits.
- The TMA copy already elects one lane inside (you can see `elect.sync` before `cp.async.bulk.tensor` in the PTX). Wrapping it in `elect_one()` can deadlock.
- `make_warp_uniform(warp_idx())` tells the compiler the value is the same across a warp, so the `if` doesn't split the warp.

---

## 7. Example 1: TMA load → TMA store

The whole round trip with no threads touching the data. `A` has 65 rows on purpose (section 8).

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE = (32, 128)                                   # (rows, cols) per CTA
TILE_BYTES = TILE[0] * TILE[1] * 2                 # fp16 -> 8192 B


@cute.struct
class SharedStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 1]                       # one mbarrier (8 B)
    sA: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, TILE[0] * TILE[1]], 128]


@cute.kernel
def tma_copy_kernel(
    tma_load: cute.CopyAtom, mA_tma: cute.Tensor,
    tma_store: cute.CopyAtom, mB_tma: cute.Tensor,
    sA_layout: cute.Layout,
):
    bm, bn, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---- 1. shared memory: one mbarrier + one tile ----
    smem = cutlass.memory.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    mbar = storage.mbar.data_ptr()
    sA = storage.sA.get_tensor(sA_layout)

    # ---- 2. init the mbarrier: ONE thread, then fence + CTA barrier ----
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar, 1)       # expects 1 arrival per phase
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # ---- 3. partition: the CTA's tile in gmem <-> the whole smem tile ----
    gA = cute.local_tile(mA_tma, TILE, (None, None))     # (32, 128, tiles_m, tiles_n)
    gB = cute.local_tile(mB_tma, TILE, (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_load, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
    )
    tBsA, tBgB = cpasync.tma_partition(
        tma_store, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gB, 0, 2),
    )

    # ---- 4. TMA load: warp 0 issues, the mbarrier counts the bytes ----
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, TILE_BYTES)
        cute.copy(tma_load, tAgA[(None, bm, bn)], tAsA, tma_bar_ptr=mbar)

    # ---- 5. everyone waits until all 8192 bytes have landed ----
    cute.arch.mbarrier_wait(mbar, 0)                     # phase 0

    # ---- 6. TMA store: warp 0 issues, then waits until smem has been read ----
    if warp_idx == 0:
        cute.copy(tma_store, tBsA, tBgB[(None, bm, bn)])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def launch(mA: cute.Tensor, mB: cute.Tensor):
    sA_layout = cute.make_ordered_layout(TILE, order=(1, 0))        # plain row-major
    tma_load, mA_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA, sA_layout, TILE
    )
    tma_store, mB_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mB, sA_layout, TILE
    )
    tma_copy_kernel(tma_load, mA_tma, tma_store, mB_tma, sA_layout).launch(
        grid=[cute.ceil_div(mA.shape[0], TILE[0]), cute.ceil_div(mA.shape[1], TILE[1]), 1],
        block=[128, 1, 1],
    )


def wrap(t):
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=8)   # row stride: multiple of 8 fp16 = 16 B
    )


def main():
    A = torch.randn(65, 256, dtype=torch.float16, device="cuda")   # 65: not a multiple of 32
    B = torch.empty_like(A)
    launch(wrap(A), wrap(B))
    torch.cuda.synchronize()
    print("correct:", torch.equal(A, B))


if __name__ == "__main__":
    main()
```

What to notice:

- **No `tidx`, no TV layout.** The 128 threads only wait.
- `TILE_BYTES` is what the mbarrier expects. If it's wrong, the wait **hangs forever** (too many bytes) or returns early (too few).
- The store ends with `commit` + `wait_group(0, read=True)`. `read=True` means "TMA has finished **reading** smem". After that, smem can be reused or the CTA can exit.
- The PTX has one `cp.async.bulk.tensor ... mbarrier::complete_tx::bytes` (the load) and one `cp.async.bulk.tensor.2d.global.shared::cta` (the store): the (32, 128) tile fits in one box.

---

## 8. Tile edges are free

`A` is 65 × 256, so the third row of CTAs covers rows 64–95, and only row 64 exists.

- **Load:** rows that don't exist arrive as **zeros**. The mbarrier still gets the full 8192 bytes, because TMA always writes the whole box.
- **Store:** rows that don't exist are **not written**.

There's no predicate tensor and no identity tensor. Compare that with the tail section of the copy page.

---

## 9. TMA store after threads write smem

Two "proxies" access smem:

- the **generic proxy**: normal `ld.shared` / `st.shared` from threads
- the **async proxy**: TMA

Writes made through one proxy aren't automatically visible to the other. Before TMA reads what threads wrote:

```python
cute.arch.fence_proxy("async.shared", space="cta")   # every writing thread: make MY writes visible to TMA
cute.arch.barrier()                                  # wait until EVERY thread has written + fenced
if warp_idx == 0:
    cute.copy(tma_store, tBsA, tBgB[(None, bm, bn)])
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=True)
```

The other direction (TMA load, then threads read) needs no fence: the mbarrier wait already makes the bytes visible.

---

## 10. Swizzle with TMA

TMA writes smem in one of four fixed patterns. You don't pick `B, M, S`; you pick a mode, and the smem layout must use the matching swizzle.

| Mode | Swizzle on **byte** addresses | Same thing on fp16 offsets | Max inner box |
|---|---|---|---|
| none | – | – | 256 elements |
| 32 B | `Swizzle<1,4,3>` | `<1,3,3>` | 32 B |
| 64 B | `Swizzle<2,4,3>` | `<2,3,3>` | 64 B |
| 128 B | `Swizzle<3,4,3>` | `<3,3,3>` | 128 B |

In every mode, `M = 4` keeps 16 B chunks whole and `S = 3` keys the XOR off the row inside a 128 B-row atom. The 128 B mode is what `ldmatrix` and WGMMA want.

### Building the layout

```python
atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
# S<3,4,3> o 0 o (8,64):(64,1)      one atom = 8 rows x 128 B = 1 KB
sA_layout = cute.tile_to_shape(atom, (32, 128), order=(0, 1))
# S<3,4,3> o 0 o ((8,4),(64,2)):((64,512),(1,2048))
sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
```

- `K_SW128` means "rows are the contiguous dimension, 128 B swizzle".
- `tile_to_shape(..., order=(0, 1))` stacks atoms **down the rows first**: cols 0–63 of all 32 rows (4 KB), then cols 64–127 (4 KB).
- `get_tensor(outer, swizzle=inner)` attaches the swizzle to the **pointer**, so it acts on byte addresses. That's the convention these atoms use. Don't pass the composed layout as the layout here; that would apply `<3,4,3>` to element offsets.
- Align the buffer to **1024 B** (one atom), as CUTLASS does for swizzled TMA buffers.

![SW128 layout](/10-cuteDSL/tma-sw128-layout.svg)

The 128 B inner-box limit splits the (32, 128) tile into **two boxes** of (32, 64). You don't handle that: the PTX simply has two loads and two stores.

### Which reads are conflict-free?

The swizzle is fixed, so check your thread reads against it (same method as the swizzle page, checked here with a model of this exact layout):

| Thread read (16 B per thread) | No swizzle | 128 B swizzle |
|---|---|---|
| `(8,16)` column-major: a column of vectors | 8-way | **1-way** |
| `(16,8)` row-major: a phase reads one 128 B row | 1-way | 1-way |
| `(32,4)` row-major: a phase reads half rows of 2 rows | 2-way | 2-way |

With TMA, you choose the **read pattern** to fit the swizzle, not the other way round.

---

## 11. Example 2: swizzled TMA load → threads compute → TMA store

`B = A * A`. The threads read a column of vectors (conflict-free under the 128 B swizzle), square it, and write it back. TMA stores it.

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack

TILE = (32, 128)
TILE_BYTES = TILE[0] * TILE[1] * 2


@cute.struct
class SharedStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 1]
    sA: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, TILE[0] * TILE[1]], 1024]


def make_sA_layout():
    # 128 B swizzle atom: 8 rows x 64 fp16 (= 128 B per row), Swizzle<3,4,3> in BYTES
    atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
    # repeat the atom to cover (32, 128): down the rows first, then the 2 column halves
    return cute.tile_to_shape(atom, TILE, order=(0, 1))


@cute.kernel
def tma_square_kernel(
    tma_load: cute.CopyAtom, mA_tma: cute.Tensor,
    tma_store: cute.CopyAtom, mB_tma: cute.Tensor,
    sA_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.memory.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    mbar = storage.mbar.data_ptr()
    # outer = logical (row, col) -> element offset; inner = swizzle applied to the BYTE address
    sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gA = cute.local_tile(mA_tma, TILE, (None, None))
    gB = cute.local_tile(mB_tma, TILE, (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_load, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
    )
    tBsA, tBgB = cpasync.tma_partition(
        tma_store, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gB, 0, 2),
    )

    # ---- TMA load (hardware writes the tile already swizzled) ----
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, TILE_BYTES)
        cute.copy(tma_load, tAgA[(None, bm, bn)], tAsA, tma_bar_ptr=mbar)
    cute.arch.mbarrier_wait(mbar, 0)

    # ---- threads: smem -> registers -> square -> smem (a column of vectors per phase) ----
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=128)
    rw = cute.make_tiled_copy_tv(
        atom, cute.make_ordered_layout((8, 16), order=(0, 1)), cute.make_layout((1, 8))
    )
    thr = rw.get_slice(tidx)
    tRsA = thr.partition_S(sA)
    tRrA = cute.make_rmem_tensor_like(tRsA)
    cute.copy(rw, tRsA, tRrA)
    x = tRrA.load()
    tRrA.store(x * x)
    cute.copy(rw, tRrA, thr.partition_D(sA))

    # ---- hand smem from the threads (generic proxy) to TMA (async proxy) ----
    cute.arch.fence_proxy("async.shared", space="cta")   # each thread: my smem writes -> visible to TMA
    cute.arch.barrier()                                  # wait for ALL threads' writes + fences

    # ---- TMA store (hardware reads the swizzled tile, un-swizzles into gmem) ----
    if warp_idx == 0:
        cute.copy(tma_store, tBsA, tBgB[(None, bm, bn)])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def launch(mA: cute.Tensor, mB: cute.Tensor):
    sA_layout = make_sA_layout()
    tma_load, mA_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA, sA_layout, TILE
    )
    tma_store, mB_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mB, sA_layout, TILE
    )
    tma_square_kernel(tma_load, mA_tma, tma_store, mB_tma, sA_layout).launch(
        grid=[cute.ceil_div(mA.shape[0], TILE[0]), cute.ceil_div(mA.shape[1], TILE[1]), 1],
        block=[128, 1, 1],
    )


def wrap(t):
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=8)
    )


def main():
    A = torch.randn(65, 256, dtype=torch.float16, device="cuda")
    B = torch.empty_like(A)
    launch(wrap(A), wrap(B))
    torch.cuda.synchronize()
    print("correct:", torch.equal(B, A * A))


if __name__ == "__main__":
    main()
```

What to notice:

- The **only** changes for the swizzle are `make_sA_layout()`, `get_tensor(..., swizzle=...)` and the 1024 B alignment. `make_tiled_tma_atom` reads the swizzle from `sA_layout` and puts the matching mode in the descriptor.
- Threads index `sA` by logical `(row, col)`. The swizzle is invisible to them, just as on the swizzle page.
- `fence_proxy` → `barrier` → TMA store: section 9.

---

## 12. Example 3: many tiles, one buffer, flipping the phase

Each CTA owns a 32-row strip and walks across it one tile at a time, reusing one smem buffer and one mbarrier.

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE = (32, 128)
TILE_BYTES = TILE[0] * TILE[1] * 2


@cute.struct
class SharedStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 1]
    sA: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, TILE[0] * TILE[1]], 128]


@cute.kernel
def tma_row_strip_kernel(
    tma_load: cute.CopyAtom, mA_tma: cute.Tensor,
    tma_store: cute.CopyAtom, mB_tma: cute.Tensor,
    sA_layout: cute.Layout,
):
    bm, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.memory.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    mbar = storage.mbar.data_ptr()
    sA = storage.sA.get_tensor(sA_layout)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gA = cute.local_tile(mA_tma, TILE, (None, None))
    gB = cute.local_tile(mB_tma, TILE, (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_load, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
    )
    tBsA, tBgB = cpasync.tma_partition(
        tma_store, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gB, 0, 2),
    )

    n_tiles = cute.size(tAgA, mode=[2])        # column tiles in this row strip
    phase = cutlass.Int32(0)

    for k in cutlass.range(n_tiles):
        # load tile k into the (single) smem buffer
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, TILE_BYTES)
            cute.copy(tma_load, tAgA[(None, bm, k)], tAsA, tma_bar_ptr=mbar)

        cute.arch.mbarrier_wait(mbar, phase)   # phase 0, 1, 0, 1, ...

        # (here, threads would compute on sA)

        # store tile k, and wait until TMA has finished READING sA
        if warp_idx == 0:
            cute.copy(tma_store, tBsA, tBgB[(None, bm, k)])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        cute.arch.barrier()                    # nobody runs ahead into the next phase
        phase = phase ^ 1


@cute.jit
def launch(mA: cute.Tensor, mB: cute.Tensor):
    sA_layout = cute.make_ordered_layout(TILE, order=(1, 0))
    tma_load, mA_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA, sA_layout, TILE
    )
    tma_store, mB_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mB, sA_layout, TILE
    )
    tma_row_strip_kernel(tma_load, mA_tma, tma_store, mB_tma, sA_layout).launch(
        grid=[cute.ceil_div(mA.shape[0], TILE[0]), 1, 1],   # one CTA per 32-row strip
        block=[128, 1, 1],
    )


def wrap(t):
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=8)
    )


def main():
    A = torch.randn(65, 1000, dtype=torch.float16, device="cuda")   # 1000 cols: 8 tiles, last one partial
    B = torch.empty_like(A)
    launch(wrap(A), wrap(B))
    torch.cuda.synchronize()
    print("correct:", torch.equal(A, B))


if __name__ == "__main__":
    main()
```

What to notice:

- `phase` starts at 0 and flips every round: the waits go 0, 1, 0, 1, …
- `wait_group(0, read=True)` before the next load: the store must finish **reading** `sA` before the next load overwrites it.
- The `barrier()` at the end of each round stops anyone falling two rounds behind (section 5).
- This version is **serial**: load, wait, store, wait, repeat. The fix is two buffers and two mbarriers, so tile `k+1` loads while tile `k` is used. That's the pipelining page.

---

## 13. Checking your work

**Compile without a GPU.** `cute.compile` only needs a target architecture, and fake tensors stand in for real ones:

```python
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
# run with: CUTE_DSL_ARCH=sm_90a python check.py
A = make_fake_compact_tensor(cutlass.Float16, (65, 256), stride_order=(1, 0), assumed_align=16)
B = make_fake_compact_tensor(cutlass.Float16, (65, 256), stride_order=(1, 0), assumed_align=16)
ptx = cute.compile(launch, A, B, options="--keep-ptx").__ptx__
```

**What to look for in the PTX:**

| Look for | Means |
|---|---|
| `cp.async.bulk.tensor.2d.shared::cta.global ... complete_tx::bytes` | TMA load; count them to see how many boxes per tile |
| `cp.async.bulk.tensor.2d.global.shared::cta` | TMA store |
| `elect.sync` right before the copy | the atom electing its issuing thread |
| `mbarrier.try_wait.parity` | your `mbarrier_wait` |
| `fence.proxy.async.shared::cta` | the generic → async fence |
| `ld.shared.v4` / `st.shared.v4` | your thread reads and writes are 128-bit |

On a real GPU, `torch.equal` checks correctness. For conflicts, ncu's shared-memory metrics from the bank-conflict section still apply to your thread reads and writes.

---

## 14. Pitfalls

1. **Wrong `expect_tx` byte count.** Too high: the wait never returns. Too low: you read a half-filled tile.
2. **Initializing the mbarrier from more than one thread**, or skipping `mbarrier_init_fence()` + `barrier()`.
3. **Wrapping the TMA `cute.copy` in `elect_one()`.** It already elects; this can deadlock.
4. **Waiting on the wrong parity** in a loop, or letting the producer run two rounds ahead.
5. **No `fence_proxy` before a TMA store** of thread-written smem: TMA may read stale data.
6. **Reusing smem after a TMA store** without `cp_async_bulk_wait_group(0, read=True)`.
7. **Row stride not a multiple of 16 B**, or no divisibility promise on a dynamic shape: the descriptor can't be built.
8. **Swizzle units.** TMA atoms put `<B,4,3>` on the pointer (bytes). A swizzle you derived in element offsets goes in the layout instead.

---

## 15. Recap

| Step | Code |
|---|---|
| descriptor + atom | `make_tiled_tma_atom(op, mA, sA_layout, TILE)` |
| smem | struct with an `Int64` mbarrier + the tile (128 B aligned, 1024 B if swizzled) |
| mbarrier init | warp 0, `elect_one`: `mbarrier_init(mbar, 1)`; then `mbarrier_init_fence()` + `barrier()` |
| partition | `tma_partition(atom, 0, make_layout(1), group_modes(sA,0,2), group_modes(gA,0,2))` |
| load | warp 0: `arrive_and_expect_tx(mbar, bytes)` (elect_one), then `cute.copy(atom, tAgA[(None,bm,bn)], tAsA, tma_bar_ptr=mbar)` |
| wait | everyone: `mbarrier_wait(mbar, phase)`; flip `phase` each round |
| store | `fence_proxy` + `barrier` if threads wrote smem; warp 0: `cute.copy(store_atom, tBsA, tBgB[...])`, commit, `wait_group(0, read=True)` |

> [!NOTE]
> All three examples compile for `sm_90a` (checked through the PTX), but they haven't been run on a GPU here. Run each `main()` on your H100 and check that it prints `correct: True`.

**Not covered yet:** double buffering with full/empty barriers (pipelining page), clusters and **multicast** (one load delivered to several CTAs), descriptor prefetch (`cpasync.prefetch_descriptor`), and im2col mode for convolutions.
