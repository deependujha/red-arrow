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

CuTe DSL compiles your Python kernel to MLIR → PTX → CUBIN (which holds the SASS). You can dump each stage.

### Option 1 : env vars (no code change)

```bash
export CUTE_DSL_KEEP_PTX=1       # keep the generated .ptx
export CUTE_DSL_KEEP_CUBIN=1     # keep the generated .cubin
export CUTE_DSL_PRINT_IR=1       # print the MLIR IR while compiling
export CUTE_DSL_LINEINFO=1       # embed Python line info (maps SASS/PTX back to your source)
export CUTE_DSL_DUMP_DIR=./dump;mkdir -p dump  # where the dumped files go (default: cwd)

python my_kernel.py
```

### Option 2 : from Python, on the compiled kernel

```python
import cutlass.cute as cute

compiled = cute.compile(host_fn, *args)   # host_fn is the @cute.jit function

print(compiled.__ptx__)                   # PTX as a string

with open("kernel.ptx", "w") as f:
    f.write(compiled.__ptx__)
with open("kernel.cubin", "wb") as f:
    f.write(compiled.__cubin__)
```

> [!IMPORTANT]
> - `__ptx__` / `__cubin__` are only populated if the matching `KEEP` option is on (env var above, or `cute.compile(host_fn, *args, options="--keep-ptx --keep-cubin")`).
> - Only a compiled (not purely JIT-called) function exposes them, so go through `cute.compile`.
> - Dumped files are named after the kernel, with the target arch in the name. Names and flags can shift between versions, so check `ls` in the dump dir and the docs for your installed version.

### Getting SASS

PTX is virtual ISA; **SASS** is the real machine code for your GPU, and it lives inside the cubin. Disassemble it with the CUDA toolkit:

```bash
cuobjdump -sass kernel.cubin     # or: nvdisasm kernel.cubin
cuobjdump -ptx  kernel.cubin     # PTX embedded in the cubin, if present
```

> [!TIP]
> - With `CUTE_DSL_LINEINFO=1`, `nvdisasm -g` / `cuobjdump -sass` can show which source line produced each instruction.
> - Quick checks to look for in SASS: `LDG.E.128` (vectorized loads), `LDGSTS` / `UTMALDG` (cp.async / TMA), `HMMA` / `WGMMA` / `UTCMMA` (tensor-core ops), and `STL`/`LDL` (register spills).

---

## `print` vs `cute.printf`

The two run at **different times**, which is the most common source of confusion.

| | `print(...)` | `cute.printf(...)` |
|---|---|---|
| Runs when | **Compile time** (while the DSL traces your Python) | **Runtime**, on the GPU |
| Runs how often | Once per compilation (not on cached re-runs) | Once per thread that reaches it, every launch |
| Values shown | Static info: Python ints, types, shapes. Dynamic values show up as abstract IR values like `?` / `%arg0` | Actual runtime values |
| Format | Normal Python | C-style (`%d`, `%f`, `%s`, ...) |
| Usable in kernel? | Yes, but only tells you what the compiler sees | Yes, this is how you inspect real data |

```python
@cute.kernel  # or, @cute.jit
def kernel(n: cutlass.Int32):
    print("compile time:", n)            # -> prints something like "?" once, at compile time
    cute.printf("runtime: %d", n)        # -> prints the real value, from every thread
    cute.printf("runtime: {}", n)        # -> prints the real value using Python-style formatting
```

> [!IMPORTANT]
> - Because `cute.printf` fires per thread, guard it: `if tidx == 0 and bidx == 0:`, otherwise you get a flood of output.
> - Device printf output is buffered and shows up after the kernel finishes (e.g. after a sync), and the buffer is limited, so very heavy printing can drop lines.
> - `print` is perfect for checking *what the compiler sees* (is this value static or dynamic?). If you want actual values, use `cute.printf`.
> - A Python `if` on a dynamic value is lowered to a runtime branch, but `print` inside it still only fires once at compile time.

---

## Datatypes

Numeric types live in the `cutlass` namespace. They're used for annotating kernel arguments, casting, and picking the element type of data.

| Kind | Types |
|---|---|
| Signed int | `cutlass.Int8`, `Int16`, `Int32`, `Int64` |
| Unsigned int | `cutlass.Uint8`, `Uint16`, `Uint32`, `Uint64` |
| Float | `cutlass.Float16`, `BFloat16`, `Float32`, `Float64`, `TFloat32` |
| Low-precision float | `cutlass.Float8E4M3FN`, `Float8E5M2`, plus newer FP6/FP4 types on Blackwell |
| Bool | `cutlass.Boolean` |

```python
import cutlass

x = cutlass.Int32(5)             # construct a typed scalar
y = x.to(cutlass.Float32)        # cast
z = cutlass.Float16(1.5)

print(cutlass.Float16.width)     # 16 -> bit width of the type
```

> [!IMPORTANT]
> - Plain Python `int` / `float` are **compile-time constants**; `cutlass.Int32(...)` etc. are **dynamic runtime values** in the generated code. Annotating a kernel/jit argument as `cutlass.Int32` makes it a runtime value, while an unannotated Python int is baked in (a change causes a recompile).
> - `TFloat32` is a 19-bit format stored in 32 bits, mainly used as an input type for tensor-core MMA. `BFloat16` keeps Float32's exponent range with fewer mantissa bits, while `Float16` has more precision but a smaller range.
> - Exact set of low-precision types depends on your CUTLASS version, so check `dir(cutlass)` if something is missing.

---

## `from_dlpack` : bridging torch → CuTe

A `@cute.jit` function can take torch tensors in two ways, and **what you pass decides whether the kernel is specialized on shape**:

| How you call it | Conversion | Layout | Specialization |
|---|---|---|---|
| `add(x, y, z, N)` (raw torch tensors) | implicit, via DLPack | **dynamic**: `(?):(1)`, only the stride-1 dim stays static | one compile, any shape |
| `add(from_dlpack(x), ...)` | explicit | **static**: `(4096):(1)` | specialized; new shape = recompile |

The parameter annotation can be `torch.Tensor` (accepts raw torch tensors) or `cute.Tensor` (accepts `from_dlpack` output). The layout comes from **what you pass**, not from the annotation.

> [!NOTE]
> To verify: with a `cute.Tensor` annotation, does passing a raw torch tensor also work and give `(?):(1)`? Expected yes (implicit DLPack conversion). If so, the annotation doesn't affect the layout at all.

### Evidence

```python
@cute.jit
def add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, N: cutlass.Int32):
    print(f"{x=}; {N=}")
    ...

add(x, y, z, N)   # raw torch tensors
# x=tensor<ptr<f32, gmem> o (?):(1)>; N=Int32(?)
```

```python
@cute.jit
def add(x: cute.Tensor, y: cute.Tensor, z: cute.Tensor, N: cutlass.Int32):
    print(f"{x=}; {N=}")
    ...

add(from_dlpack(x, assumed_align=16), ...)
# x=tensor<ptr<f32, gmem, align<16>> o (4096):(1)>; N=Int32(?)
```

### What `from_dlpack` is

- Wraps an existing DLPack-compatible tensor (torch, numpy, jax…) as a `cute.Tensor`.
- **Zero copy.** It allocates no memory. The `cute.Tensor` is the same device pointer plus shape, strides, and dtype, so kernel writes show up in torch directly. Keep the torch tensor alive while the kernel uses it.
- `assumed_align=16` promises the pointer is 16B-aligned (shows up as `align<16>` in the type), so the compiler can emit vectorized loads and stores. It can't infer this on its own, and slices/views may not be aligned even when the base allocation is.

### Why use it: control

The implicit path gives you dynamic with no knobs. `from_dlpack` is static by default, and you can choose what to make dynamic:

```python
t = from_dlpack(x, assumed_align=16)                      # fully static
t = t.mark_layout_dynamic(leading_dim=1)                  # all runtime except the stride-1 dim
t = t.mark_compact_shape_dynamic(mode=0, divisibility=8)  # only dim 0 runtime, promised % 8 == 0
```

> [!IMPORTANT]
> - Static gives the fastest code (loop bounds fold, index math simplifies, vectorization is provable), at the cost of one compile per distinct shape.
> - Dynamic gives one compile for any shape and slightly less optimized code.
> - `divisibility` is a promise that recovers some vectorization in the dynamic case.
> - Rule of thumb: static while tuning a fixed problem size; dynamic (or partially dynamic with divisibility) for real workloads with varying batch/seq lengths.

### Scalars

- `N: cutlass.Int32` is a **runtime** value (`Int32(?)`) even when the tensors are static, so the grid size and bounds check are computed at runtime.
- `N: cutlass.Constexpr` bakes the value in at compile time, which means a recompile per distinct value.
- With a static layout, `N` is redundant: `cute.size(x)` gives `4096` as a compile-time constant.

### Alignment only matters once you vectorize

A one-element-per-thread kernel does scalar loads, so `align<16>` buys nothing. It pays off when each thread loads multiple elements (e.g. 4×f32 = 16B per load via tiling with `cute.zipped_divide`).

> [!TIP]
> Always check with `print(t)` on the host or inside the jit function. Static dims print as numbers and dynamic ones as `?`.
