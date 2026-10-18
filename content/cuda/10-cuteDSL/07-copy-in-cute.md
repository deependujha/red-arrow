---
title: Copy Atoms, Tiled Copy & cp.async
type: docs
math: true
sidebar:
  open: false
weight: 1007
---

## The big picture

Moving data in CuTe always goes through the same five steps. Each step adds one piece of information, and the final `cute.copy` just executes the plan.

![copy layers](/10-cutedsl/copy-layers.svg)

```python
atom       = cute.make_copy_atom(op, dtype, num_bits_per_copy=128)   # 1. what ONE instruction does
tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)   # 2. the whole block's plan
thr_copy   = tiled_copy.get_slice(tidx)                              # 3. this thread's spot in the plan
tXgA       = thr_copy.partition_S(gA)                                # 4. what this thread reads...
tXsA       = thr_copy.partition_D(sA)                                #    ...and where it writes
cute.copy(tiled_copy, tXgA, tXsA)                                    # 5. go
```

---

## 1. Copy atom: what one instruction does

A copy atom describes the **smallest copy the hardware does in one instruction**:

- **which instruction** (the op),
- **which element type**,
- **how many bits** move per instruction.

```python
dtype = input_tensor.element_type # or, `input_tensor.dtype`, or specify directly `cutlass.Float16`
dtype_width = dtype.width # gives the number of bits per element

atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),   # which instruction family: plain load/store
    dtype,                          # element type
    num_bits_per_copy=128,          # bits per instruction: 128 bits => no. of vectorized elements per instruction = 128 / dtype_width
)
```

### The op picks the hardware path

| Op | What it is | Route |
|---|---|---|
| `cute.nvgpu.CopyUniversalOp()` | normal load/store (LDG/STG, LDS/STS) | anything → anything |
| `cpasync.CopyG2SOp()` | `cp.async`, Ampere+ | gmem → smem only, asynchronous |
| TMA ops (`cpasync.CopyBulkTensorTile...`) | Tensor Memory Accelerator, Hopper+ | whole tiles, gmem ↔ smem |
| `warp.LdMatrix...` / `warp.StMatrix...` | warp-wide smem ↔ register shuffles for MMA | smem ↔ registers |

#### Simple COPY ops

for more info, [check here](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_common.html#cutlass.cute.nvgpu.CopyUniversalOp)

- `cutlass.cute.nvgpu.CopyUniversalOp`: The universal Copy Operation.

This operation is equivalent to the `a = b` assignment without any extra memory attributes. For advanced memory features (memory order, memory scope, cache eviction priority, invariant loads, etc.) please use the specialized copy operations instead:
- **`CopyG2ROp`** : global memory to register
- **`CopyR2GOp`** : register to global memory
- **`CopyS2ROp`** : shared memory to register
- **`CopyR2SOp`** : register to shared memory

```python
op = cute.nvgpu.CopyG2ROp()
atom = cute.make_copy_atom(
    op,
    tensor_dtype,
    num_bits_per_copy=64,
    memory_order=cute.nvgpu.MemoryOrder.VOLATILE,
    memory_scope=cute.nvgpu.MemoryScope.SYS,
    l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE,
    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
    load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
    shared_space=cute.nvgpu.SharedSpace.CTA,
    invariant=False,
)
```

> [!warning]
> - As you can see in simple copy atoms, `G <-> R`, `R <-> G`, `S <-> R`, `R <-> S`; but we can't directly go from `G <-> S` or `S <-> G`.
> - So, if you need to move data between global and shared memory, you will have to go through registers as an intermediate step.
> - `cp.async` can be used for asynchronous copies from global memory to shared memory, typically on Ampere+ GPUs.

### `num_bits_per_copy` = vector width

- `128` means one 128-bit instruction (LDG.128 / STG.128). That's the widest normal load, and the one you want for bandwidth.
- `64` or `32` means smaller instructions. More of them are needed to move the same data.

### Single-thread vs warp-wide atoms

- Most atoms are executed by **one thread**: "thread t moves its 16 bytes".
- Some atoms need a **whole warp cooperating** on one instruction (`ldmatrix`, `stmatrix`). Because of these, an atom carries its own small thread/value layout. You won't need to think about that until the MMA page.

> [!CAUTION]
> - **Alignment:** A 128-bit atom requires addresses aligned to **16 bytes** (128 bits = 16 bytes).
> - When wrapping the tensor, tell CuTe about its alignment:
> ```python
> from_dlpack(t, assumed_align=16)
> ```
>   CUDA allocations have at least 256-byte alignment under the documented [CUDA allocation guarantees](https://docs.nvidia.com/cuda/archive/9.2/cuda-c-programming-guide), so `assumed_align=16` is safe for CUDA-allocated tensors. This is a compiler hint, not a runtime check.
> - Without sufficient alignment information, the compiler may reject the operation or generate narrower copies.

---

## 2. Tiled copy: the atom repeated over threads and values

```python
thr = cute.make_ordered_layout((8, 16), order=(1, 0))   # 128 threads, ids go along N first
val = cute.make_layout((1, 8))                          # each thread: 1 row × 8 fp16 = 16 bytes
tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)    # covers an (8, 128) patch per pass
```

- This is **`make_layout_tv(thr, val)` from the partitioning page, plus the atom**.
  - `thr` and `val` say *which thread owns which values* (exactly like before).
  - The atom says *which instruction moves them*.
- **One pass** covers `thr × val` per mode: `(8·1, 16·8) = (8, 128)`.
- Pick `val` so it **matches the atom's width**: 8 fp16 = 128 bits = exactly one atom call per thread per pass.

> [!TIP]
> - Choosing `thr` with ids along the contiguous dimension (**`order=(1,0)`** for a row-major tensor) means neighbouring threads read neighbouring 16-byte chunks.
> - One row of 16 threads × 16 B = 256 contiguous bytes, so loads are **`coalesced`** **and** **`vectorized`**.

![tiled copy layout](/10-cutedsl/tiled-copy.png)

---

## 3–4. `get_slice` and `partition_S` / `partition_D`

```python
thr_copy = tiled_copy.get_slice(tidx)   # "you're thread tidx"
tXgA = thr_copy.partition_S(gA)         # source view: what this thread reads
tXsA = thr_copy.partition_D(sA)         # destination view: where this thread writes
```

### The shape you'll see everywhere: `(CPY, CPY_M, CPY_N)`

```
( CPY , CPY_M , CPY_N )
   │      └──────┴── how many times the atom repeats to cover the tile
   └── the values ONE atom call moves
```

Example: the CTA tile `gA` is `(32, 128)`, but one pass covers only `(8, 128)`.

![copy partition](/10-cutedsl/copy-partition.svg)

- Thread 17 sits at thr coord `(1,1)`, so in **every pass** it moves the same relative 8-value chunk.
- `tXgA` has shape ≈ `((8), 4, 1)`: 8 values per atom call, 4 passes down M, 1 pass across N.
- `cute.copy` will fire the atom 4 times for this thread.

> [!NOTE]
> This is the big upgrade over `composition(gA, tv)[(tidx, None)]` from the previous page:
> - The tile can be **bigger than one pass**. The `CPY_M`/`CPY_N` modes handle the repeats automatically.
> - The **instruction is fixed up front** (a 128-bit atom), instead of hoping `.load()` vectorizes.

| Partitioning page | Copy page |
|---|---|
| `make_layout_tv(thr, val)` | `make_tiled_copy_tv(atom, thr, val)` |
| `composition(gA, tv)[(tidx, None)]` | `tiled_copy.get_slice(tidx).partition_S(gA)` |
| `tA.load()` / `.store()` | `cute.copy(tiled_copy, src, dst)` |
| tile = exactly one pass | tile = any number of passes |

### Why two partition functions (S and D)?

Because for some instructions, **the thread that reads a value isn't the thread that ends up holding it.**

![partition S vs D](/10-cutedsl/partition-s-vs-d.svg)

- **Plain copy** (`CopyUniversalOp`, `cp.async`): thread 37 reads 8 values, and thread 37 holds those 8 values. Source and destination use the **same** map, so `partition_S` and `partition_D` cut identically.
- **Shuffling copy** (`ldmatrix`, `stmatrix`): one warp-wide instruction reads rows in one arrangement and hands the values out in a **different** arrangement (the one the tensor cores want). So a tiled copy carries **two** TV maps:
  - `partition_S` uses the **source** map: "which values does thread t read?"
  - `partition_D` uses the **destination** map: "which values does thread t receive?"

In the story: normally each kid carries their own box to their own desk. With a **conveyor belt**, kid #37 loads blocks from row 37 onto the belt, but the belt drops blocks at desks in a different order. So each kid needs two lists: "where I load from" and "what lands on my desk".

> [!TIP]
> Always use `partition_S` for the source and `partition_D` for the destination, even when they'd give the same result.
> Then swapping the atom for `ldmatrix` later doesn't silently break your code.

---

## 5. `cute.copy`: execute the plan

```python
cute.copy(tiled_copy, tXgA, tXrA)   # src, dst
```

- It loops over `CPY_M × CPY_N` and fires the atom on each `CPY` chunk.
- Pass the **tiled copy** (the full plan), not the bare atom.
- **Source and destination partitions must match in shape.** The easy way is to create the register buffer from the source partition:

```python
  tXrA = cute.make_rmem_tensor_like(tXgA)
```

> [!CAUTION]
> `cute.copy` has **no bounds check**. For a tile hanging off the edge of the tensor, pass a predicate tensor (same shape as the partition, one bool per element), built from an identity tensor partitioned the **same way**.
> (Tail handling gets its own section later.)

### Run it: gmem → registers → gmem with 128-bit copies

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def copy_kernel(mA: cute.Tensor, mB: cute.Tensor, cA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()

    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type, num_bits_per_copy=128)
    thr = cute.make_ordered_layout((8, 16), order=(1, 0))      # 128 threads
    val = cute.make_layout((1, 8))                             # 8 fp16 = 16 B per thread
    tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)       # (8,128) per pass

    gA = cute.local_tile(mA, (32, 128), (bm, bn))              # CTA tile = 4 passes
    gB = cute.local_tile(mB, (32, 128), (bm, bn))
    gC = cute.local_tile(cA, (32, 128), (bm, bn))                   # same tile, but coordinates


    thr_copy = tiled_copy.get_slice(tidx)
    tXgA = thr_copy.partition_S(gA)
    tXgB = thr_copy.partition_D(gB)
    tXcA = thr_copy.partition_S(gC)
    tXrA = cute.make_rmem_tensor_like(tXgA)

    # one bool per atom call, shaped like the copy expects: (1, rest of tXgA)
    tXpA = cute.make_rmem_tensor((1, tXgA.shape[1]), cutlass.Boolean)
    cpy = cute.size(tXcA, mode=[0])                   # 8 values per atom call
    for j in cutlass.range_constexpr(cute.size(tXpA)):
        # linear index j*cpy = first element of the j-th atom call
        tXpA[j] = cute.elem_less(tXcA[j * cpy], mA.shape)
    
    print(f"{tXgA=};\n\n{thr_copy=}")
    print(f"{tXrA=}")

    cute.copy(tiled_copy, tXgA, tXrA, pred=tXpA)   # masked load: skips rows past the edge)
    val = tXrA.load()
    new_val = (val*val)
    tXrA.store(new_val)
    cute.copy(tiled_copy, tXrA, tXgB, pred=tXpA)          # registers -> gmem (128-bit stores)


@cute.jit
def launch(mA: cute.Tensor, mB: cute.Tensor):
    cA = cute.make_identity_tensor(mA.shape)       # coordinate "tensor", no memory

    grid_x = cute.ceil_div(mA.shape[0], 32)
    grid_y = cute.ceil_div(mA.shape[1], 128)

    copy_kernel(mA, mB, cA).launch(
        grid=[grid_x, grid_y, 1],
        block=[128, 1, 1],
    )

def main():
    A = torch.randn(150, 256, dtype=torch.float16, device="cuda")
    B = torch.empty_like(A)
    launch(from_dlpack(A, assumed_align=16), from_dlpack(B, assumed_align=16))
    torch.cuda.synchronize()
    print("correct:", torch.allclose(B, A**2))

main()

# ==== print ptx ====
A = torch.randn(128, 256, dtype=torch.float16, device="cuda")
B = torch.empty_like(A)

compiled = cute.compile(launch, from_dlpack(A, assumed_align=16), from_dlpack(B, assumed_align=16), options="--keep-ptx")   # host_fn is the @cute.jit function

print(compiled.__ptx__)
```

> Through PTX, we can verify it uses `vectorized float4` for loading & saving:
> ```ptx
> ld.global.v4.b32 	{%r13, %r14, %r15, %r16}, [%rd4];
> ld.global.v4.b32 	{%r17, %r18, %r19, %r20}, [%rd4+4096];
> ld.global.v4.b32 	{%r21, %r22, %r23, %r24}, [%rd4+8192];
> ld.global.v4.b32 	{%r25, %r26, %r27, %r28}, [%rd4+12288];
> ```

> [!IMPORTANT]
> **How to design a copy kernel for a 2D tensor**
>
> Threads are **1D** (`block=[128, 1, 1]`) and get their 2D meaning from `thr_layout`. The grid is **2D**, so `(bm, bn)` can be used directly as tile coordinates.
>
> **0. Know your tensor.** Which dtype, and which dimension is contiguous?
>   A PyTorch tensor is row-major, so **N (the last dim)** is contiguous.
>
> **1. Per-thread vector: aim for 16 bytes along the contiguous dimension.**
>   That's one 128-bit load per atom call: `val = (1, 16 // bytes_per_elem)`.
>   - fp16/bf16 → `(1, 8)` · fp32 → `(1, 4)` · fp8 → `(1, 16)`
>   - Match the atom: `num_bits_per_copy=128`.
>
> **2. Thread arrangement: thread ids along the contiguous dimension.**
>   `thr = make_ordered_layout((rows, cols), order=(1, 0))`, with `rows × cols` = threads per block (128 or 256).
>   - Neighbouring threads then read neighbouring 16-byte chunks, so loads are **coalesced and vectorized**.
>   - Aim for at least a warp's worth of threads along N: 32 × 16 B = 512 contiguous bytes per row.
>   - **One pass** = `thr × val` per mode. Here: `(8, 16) × (1, 8) = (8, 128)`.
>
> **3. CTA tile: a whole number of passes.**
>   - The tile **must be a multiple of one pass**: `(32, 128)` = 4 passes of `(8, 128)` ✓, but `(32, 64)` ✗.
>   - Values per thread = `size(tile) / threads`. Here `4096 / 128 = 32`, i.e. 4 atom calls of 8.
>   - Bigger tiles give each thread more loads in flight and need fewer CTAs. But with a small tensor you may end up with fewer CTAs than SMs, leaving the GPU half idle.
>
> **4. Grid: cover the tensor with tiles.**
>   `grid = [ceil_div(M, tile_M), ceil_div(N, tile_N), 1]`.
>   - Use `ceil_div`, not `//`. Floor division silently drops the tail rows.
>   - Put the dimension with more tiles on **x**: grid y and z are capped at 65535.
>
> **5. Tail: if the tensor isn't a multiple of the tile, predicate.**
>   `ceil_div` gives the tail a CTA, but that CTA's tile hangs off the edge.
>   Mask both copies with an identity tensor tiled and partitioned the same way (see the tail section), or you get silent corruption or an illegal memory access.
>
> **6. Verify.**
>   - Print the partition: `tXgA` should be `((vals_per_atom), passes_M, passes_N)`.
>   - Check the PTX for `ld.global.v4` / `st.global.v4` (128-bit vector access).
>   - Compare with PyTorch, and run `compute-sanitizer` once with a non-multiple shape.
>
> **Worked example (this page):** fp16, row-major → `val = (1, 8)` → `thr = (8, 16)`, ids along N → pass `(8, 128)` → tile `(32, 128)` = 4 passes, 32 values per thread → grid `(ceil_div(M, 32), ceil_div(N, 128))`.


---

## `cp.async`: copy straight into shared memory, asynchronously

`cp.async` (Ampere+, so it works on the L4) keeps **all five steps**. It changes **where the data goes** and **when it arrives**, which adds a "wait for it" step at the end.

![cp.async route](/10-cutedsl/cp-async-route.svg)

| | Plain copy (`CopyUniversalOp`) | `cp.async` (`cpasync.CopyG2SOp`) |
|---|---|---|
| Route | gmem → **registers** → smem | gmem → **smem directly** (registers skipped) |
| When `cute.copy` returns | when the data is in registers | **immediately**; the data arrives later |
| Destination | anything | **must be shared memory** |
| Before you can read the data | nothing extra | **commit → wait → barrier** |
| Bytes per instruction | any supported width | 4, 8 or 16 |
| `partition_S` vs `partition_D` | same map | same map (still one thread per instruction) |

### The story version

Before, each kid **walked the box over** and stood there until it was on their desk.

With `cp.async`, each kid **places a delivery order**. The truck brings the blocks straight to the **shared table** in the middle of the room (smem), and the kid walks away immediately to do other work. That comes with three rules:

1. **Commit:** staple your orders into one **batch**. "These belong together."
2. **Wait:** "Wait until no more than **N** of *my* batches are still on the road." N = 0 means wait for all of them.
3. **Barrier:** **everyone waits for everyone.** The shared table holds blocks *other kids* ordered, and step 2 only covers **your own** deliveries.

### In code

```python
from cutlass.cute.nvgpu import cpasync

atom = cute.make_copy_atom(
    cpasync.CopyG2SOp(),            # cp.async: gmem -> smem
    cutlass.Float16,
    num_bits_per_copy=128,          # 16 B per instruction
)
tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)
thr_copy   = tiled_copy.get_slice(tidx)

tXgA = thr_copy.partition_S(gA)     # source: gmem
tXsA = thr_copy.partition_D(sA)     # destination: MUST be smem

cute.copy(tiled_copy, tXgA, tXsA)   # 1. place the orders (returns right away)
cute.arch.cp_async_commit_group()   # 2. staple them into one batch
cute.arch.cp_async_wait_group(0)    # 3. wait until 0 of MY batches are still in flight
cute.arch.barrier()                 # 4. wait for EVERY thread's deliveries
# sA is now safe to read
```

### Cache mode (optional)

```python
cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL)
```

- `GLOBAL` is `cp.async.cg`: caches in L2 only and skips L1. It's good for streaming data you'll only read once from smem. It only works with **16-byte** copies.
- The default caches in L1 as well (`.ca`) and works with 4, 8 or 16 bytes.

> [!CAUTION]
> - **Forgetting the barrier** is the classic bug. `wait_group` only covers *your* thread's copies, but you'll usually read smem that **other threads** loaded. Without the barrier, you read half-arrived data, sometimes, depending on timing.
> - The destination must be **smem**. `cp.async` can't write to registers or global memory.
> - 16-byte copies need 16-byte alignment on **both** sides (gmem and smem).

### Why bother? Overlap.

Done once like above, `cp.async` is **no faster** than a plain load: you order and then immediately wait. The win comes from **ordering the next tile while computing on the current one**. This is called **double buffering**:

![double buffering](/10-cutedsl/cp-async-double-buffer.svg)

```
order tile 0 -> buf 0, commit
loop k:
    order tile k+1 -> other buf, commit   # truck is on the road
    wait_group(1)                         # tile k arrived; only tile k+1 may still be travelling
    barrier
    compute on tile k
    barrier                               # nobody may overwrite a buffer someone is still reading
```

- `wait_group(N)` takes a number for exactly this reason: **keep N batches in flight** while you compute.
- With 2 buffers, use `wait_group(1)`. With 3 buffers (3 stages), use `wait_group(2)`, and so on.
- On the last iteration there's nothing new to order, so use `wait_group(0)`.
- This is the first step of **pipelining**. The pipelining page builds on exactly this loop.

### Run it: gmem → smem with `cp.async`, read back by *different* threads

This example loads with one thread arrangement and reads smem with **another**, so each thread reads values that other threads loaded. That's exactly the situation where the barrier is required.

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE = (32, 128)


@cute.kernel
def cpasync_kernel(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()

    # one CTA tile in shared memory, row-major
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type, cute.make_ordered_layout(TILE, order=(1, 0)), byte_alignment=16
    )

    gA = cute.local_tile(mA, TILE, (bm, bn))
    gB = cute.local_tile(mB, TILE, (bm, bn))

    # LOAD plan: cp.async, threads as (8,16), 8 fp16 each
    load_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), mA.element_type, num_bits_per_copy=128)
    load = cute.make_tiled_copy_tv(
        load_atom, cute.make_ordered_layout((8, 16), order=(1, 0)), cute.make_layout((1, 8))
    )
    thr_load = load.get_slice(tidx)
    cute.copy(load, thr_load.partition_S(gA), thr_load.partition_D(sA))
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()                  # remove this and the result can break

    # READ plan: a DIFFERENT arrangement (32,4), so threads read what others loaded
    read_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type, num_bits_per_copy=128)
    read = cute.make_tiled_copy_tv(
        read_atom, cute.make_ordered_layout((32, 4), order=(1, 0)), cute.make_layout((1, 8))
    )
    thr_read = read.get_slice(tidx)
    tRsA = thr_read.partition_S(sA)
    tRgB = thr_read.partition_D(gB)
    tRrA = cute.make_rmem_tensor_like(tRsA)
    cute.copy(read, tRsA, tRrA)          # smem -> registers
    cute.copy(read, tRrA, tRgB)          # registers -> gmem


@cute.jit
def launch(mA: cute.Tensor, mB: cute.Tensor):
    cpasync_kernel(mA, mB).launch(
        grid=[mA.shape[0] // TILE[0], mA.shape[1] // TILE[1], 1],
        block=[128, 1, 1],
        smem=TILE[0] * TILE[1] * 2 + 128,   # fp16 tile + alignment slack
    )


if __name__ == "__main__":
    A = torch.randn(64, 256, dtype=torch.float16, device="cuda")
    B = torch.empty_like(A)
    launch(from_dlpack(A, assumed_align=16), from_dlpack(B, assumed_align=16))
    torch.cuda.synchronize()
    print("correct:", torch.equal(A, B))

# --- output ---
# correct: True
```

> [!information]
> `SmemAllocator` does not create memory; it suballocates the dynamic shared-memory buffer reserved at launch with `smem=...`. When you request a tensor aligned to 16 bytes, it advances its pointer to the next 16-byte boundary before placing the tensor.
>
> A `(32, 128)` tile of `fp16` values occupies `32 × 128 × 2 = 8192` bytes. The launch reserves an additional 128 bytes to allow for alignment padding. For example, if the buffer starts at an address ending in `0x08`, the allocator skips 8 bytes to reach the next 16-byte boundary:
> 
> ```
> buffer start      0x...08
> aligned start     0x...10   ← skipped 8 bytes
> tile ends at      0x...10 + 8192
> ```
>
> If `smem=...` is omitted or too small, the kernel may access invalid shared memory at runtime; this is not necessarily caught at compile time.

> [!NOTE]
> The read side here (32 threads down a row-major `(32,128)` smem tile) is likely to cause **bank conflicts**, because many threads hit the same banks.
> That's not a bug in this toy example. It's the problem **swizzling** solves, coming in the next section.

---

## Bank conflict in shared memory

The `cp.async` example above is correct, but its read side is slow in a way that `torch.equal` will never show you: it has a **2-way bank conflict**. This section builds the mental model to see why just by reading the code.

---

#### 1. What `sA` really is

- Following last example, `TILE = (32, 128)`; Each CTA reserves one `(32, 128)` fp16 tile in shared memory: `32 × 128 × 2 = 8192` bytes.
- Shared memory has **no rows or columns**. It's one flat run of bytes. The layout decides how the 2D tile is flattened.
- `make_ordered_layout(TILE, order=(1, 0))` is row-major, i.e. the layout `(32,128):(128,1)`. Row 0's 128 elements come first, then row 1, and so on.

So element `(r, c)` lives at:

$$
\text{byte}(r, c) = \underbrace{(r \cdot 128 + c)}_{\text{element index}} \cdot \underbrace{2}_{\text{bytes per fp16}}
$$

- Example: `(1, 8)` → `(128 + 8) · 2 = 272`.
- **One row = 256 bytes.** Remember this number.

---

#### 2. Banks

Shared memory is split into **32 banks**, each **4 bytes** wide. Consecutive 4-byte words go to consecutive banks:

$$
\text{bank} = \left(\frac{\text{byte}}{4}\right) \bmod 32
$$

- Bytes 0–3 → bank 0, bytes 4–7 → bank 1, …, bytes 124–127 → bank 31, then byte 128 wraps back to bank 0.
- Each bank serves **one word per pass**, so one pass delivers at most `32 × 4 = 128` bytes.

Draw `sA` 128 bytes per line and each column *is* a bank:

![smem lines and banks](/10-cutedsl/bank-smem-lines.svg)

> [!IMPORTANT]
> A row of `sA` is 256 B = 64 words = **exactly two laps** of the banks. So **every row starts at bank 0**, and column `c` of every row sits in the **same bank**.
> Any row size that's a multiple of 128 B has this property. That alone isn't a bug, but it's the warning sign.

##### When is it a conflict?

Look at the threads being served **in the same pass**:

| Situation | Result |
|---|---|
| all hit different banks | 1 pass ✓ |
| several read the **same address** | broadcast, no conflict ✓ |
| several hit the **same bank, different addresses** | **conflict**: the bank serves them one after another |

**n-way conflict** = the busiest bank gets n different addresses = n passes instead of 1.

---

#### 3. Phases: who counts as "the same pass"

One pass moves at most 128 B. If every thread in a warp asks for 16 B, that's 512 B, so the hardware splits the warp into groups that each fit in 128 B. Each group is a **phase**.

| Bytes per thread (atom width) | Warp needs | Phases | Threads per phase |
|---|---|---|---|
| 4 (fp32, 32-bit) | 128 B | 1 | 32 |
| 8 (64-bit) | 256 B | 2 | 16 |
| 16 (`num_bits_per_copy=128`) | 512 B | 4 | **8** |

In the story: a counter with **32 windows**, each handing out one 4-byte packet per round. Kids who each want 4 packets come up **8 at a time**. Two kids in the **same group** who need the same window cause a repeated round. Kids in different groups never compete.

> [!warning]
> - Conflicts are only counted **within a phase**. Threads 0 and 8 hitting the same bank is fine, because they're served in different rounds.
> - The phase rule for 64/128-bit accesses isn't officially documented by NVIDIA, but it's the standard working model and matches what ncu reports.

---

#### 4. The tiled copy decides the access pattern

Two things together decide which banks get hit:

1. **`thr_layout` + `val_layout`** of the tiled copy → which `(r, c)` each thread touches **in one instruction**.
2. **The smem layout** (`sA`) → which byte, and so which bank, each `(r, c)` lives at.

In the kernel above, `sA` and the data are identical for both copies. Only (1) changes, and that alone decides conflict vs no conflict:

![phase 0: load vs read](/10-cutedsl/bank-phase0-load-vs-read.svg)

**Load `(8,16)`**: `thread t → (t // 16, (t % 16)·8)`

- Phase 0 = threads 0–7 = row 0, cols 0–63 = bytes 0–127, contiguous.
- Every bank is hit exactly once → **1 pass**.

**Read `(32,4)`**: `thread t → (t // 4, (t % 4)·8)`

| Thread | (r, c) | byte | banks |
|---|---|---|---|
| 0–3 | (0, 0 / 8 / 16 / 24) | 0–63 | 0–15 |
| 4–7 | (1, 0 / 8 / 16 / 24) | 256–319 | **0–15 again** |

- Only 4 threads per row → each row contributes just 64 B (half a lap).
- The phase spans two rows, and row 1 restarts at bank 0.
- So banks 0–15 get two addresses each, and banks 16–31 sit idle → **2-way, 8 passes instead of 4**.

---

#### 5. Which arrangements work for this tile?

Constraint: `threads_per_row × 8 elements ≤ 128 columns`, and 128 threads in total.

| Threads | Values / thread | Fits 128 cols? | Phase 0 | Result |
|---|---|---|---|---|
| (8,16) | 8 (16 B) | 16·8 = 128 ✓ | one row, 128 B contiguous | ✓ conflict-free |
| (16,8) | 8 (16 B) | 8·8 = 64 ✓ | one row, 128 B contiguous | ✓ conflict-free |
| (32,4) | 8 (16 B) | 4·8 = 32 ✓ | two rows, both on banks 0–15 | ✗ 2-way |
| (4,32) | 8 (16 B) | 32·8 = 256 ✗ | doesn't tile `(32,128)` | invalid |
| (4,32) | 4 (8 B, 64-bit atom) | 32·4 = 128 ✓ | 16 threads × 8 B, one row | ✓ conflict-free, but 2× more instructions |

---

#### 6. Spotting conflicts fast

The one-sentence rule: **threads in the same phase must not want different addresses in the same bank.**

Quick checklist, no table needed:

1. **Atom width → phase size.** (phase size = 128/atom_width): 16 B → 8 threads, 8 B → 16, 4 B → 32.
2. **Row size a multiple of 128 B?** Then every row starts at bank 0. Danger if a phase spans more than one row.
3. **Does each phase read one contiguous 128 B chunk?** Yes → conflict-free, done.
4. Otherwise, work out `byte → bank` for one phase, as in the table above.

> [!TIP]
> The precise condition is "the phase covers 32 **different** banks". Contiguous bytes within a row guarantee that. Merely being in the same row doesn't, if threads are spread out with gaps.

---

#### 7. Fixes

1. **Change the thread arrangement** so each phase reads contiguous bytes, e.g. `(16,8)` or `(8,16)`.
2. **Padding**: make each row longer so consecutive rows start at different banks.
3. **Swizzling**: XOR some row bits into the column bits of the address. No extra memory, and alignment stays intact.

![padding vs swizzle](/10-cutedsl/bank-fixes-pad-vs-swizzle.svg)

| | Padding | Swizzle |
|---|---|---|
| For this read | +32 fp16 per row → 320 B rows → row 1 starts at bank 16 | `Swizzle<1,5,2>`: odd rows swap their two 64 B halves |
| Extra smem | 25% here | none |
| Gotchas | padding must keep 16 B alignment; +8 fp16 only shifts 4 banks (still 2-way) | layout must use the same swizzle for writes and reads |
| Used by | quick hand-written kernels | CUTLASS / CuTe, TMA |

> [!NOTE]
> The classic textbook case: fp32 `float[32][32]`, each thread reads down a column → all 32 threads on one bank → **32-way**. Pad to `float[32][33]` and it's conflict-free; the swizzle equivalent is `Swizzle<5,0,5>`.

---

#### 8. Verifying

**By hand / script (exact):**

```python
def check(name, thread_to_coord, row_elems=128, elem_bytes=2, vec_bytes=16):
    phase = 32 // (vec_bytes // 4) if vec_bytes > 4 else 32
    for p in range(32 // phase):                         # warp 0
        banks = {}
        for t in range(p * phase, (p + 1) * phase):
            r, c = thread_to_coord(t)
            word = (r * row_elems + c) * elem_bytes // 4
            for w in range(max(1, vec_bytes // 4)):
                banks.setdefault((word + w) % 32, set()).add(word + w)
        print(name, f"phase {p}:", max(len(s) for s in banks.values()), "-way")

check("load (8,16)", lambda t: (t // 16, (t % 16) * 8))   # 1-way
check("read (32,4)", lambda t: (t // 4,  (t % 4)  * 8))   # 2-way
```

**With ncu:**

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (loads) and `..._op_st.sum` (stores) count conflicts.
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` counts passes ("wavefronts"). Compare it to the ideal: instructions × phases.

**By timing (A/B):** repeat the smem read in a loop with a dependency so it isn't optimized away, then compare `(32,4)` against `(16,8)`. Expect a clear gap, though not exactly 2×.

> [!IMPORTANT]
> **Interview version:** 32 banks × 4 B; `bank = (addr / 4) % 32`. A warp's request is served in phases of 128 B (8 threads for 16-byte loads). Same bank + different address + same phase = serialized. The tiled copy's thread layout decides which addresses share a phase, and the smem layout decides which bank each address lands in. Fix with a contiguous-per-phase thread layout, padding, or (best) a swizzled smem layout.

---

## Quick recap

| Piece | Answers |
|---|---|
| copy atom | which instruction, which dtype, how many bits per call |
| `make_tiled_copy_tv` | the whole block's plan: atom × threads × values, one pass at a time |
| `get_slice(tidx)` | this thread's spot in the plan |
| `partition_S` / `partition_D` | what this thread reads / where it writes, shaped `(CPY, CPY_M, CPY_N)` |
| `cute.copy` | fire the atom `CPY_M × CPY_N` times |
| `cp.async` | gmem → smem directly; returns at once; **commit → wait → barrier** before reading |
| double buffering | order tile k+1 while computing tile k; `wait_group(N)` keeps N in flight |

---

## Next: TMA & swizzling

*(to be continued)*