---
title: Tiling & Partitioning
type: docs
math: true
sidebar:
  open: false
weight: 1006
---

## The big picture

A kernel doesn't work on the whole tensor. The work gets split in steps, and each step hands out a smaller piece:

```
Full tensor in global memory             (M, N)
        │
        │  local_tile        -> each BLOCK takes its own tile
        ▼
Block's tile                             (TileM, TileN)
        │
        │  local_partition   -> each THREAD takes its own elements
        │  make_layout_tv    -> (or) describe "which thread gets which values"
        ▼
Thread's piece                           (a few values)
        │
        │  make_rmem_tensor(_like) -> make a register buffer to hold them
        ▼
Registers: do the math here, then write back
```

| Step | Function | Question it answers |
|---|---|---|
| Pick parts of a coord/layout | `slice_`, `dice` | "Keep these modes, drop those" |
| Thread ↔ value map | `make_layout_tv` | "Thread `t`, value `v`: which element is that?" |
| Block-level split | `local_tile` | "Which tile does **this block** work on?" |
| Thread-level split | `local_partition` | "Which elements does **this thread** work on?" |
| Register buffer | `make_rmem_tensor`, `make_rmem_tensor_like` | "Where do I keep my piece while computing?" |

> [!TIP]
> - `None` in a coordinate means **"keep this whole mode, don't pick a value"**. It plays the same role as `:` in NumPy.
> - `tensor[(tidx, None)]` means "fix mode 0 to `tidx` and keep all of mode 1".
> - It shows up in almost every function on this page.

---

## `slice_` & `dice`

Both take a coordinate with some `None`s in it. They do **opposite** things with it:

| | `None` mode | integer mode |
|---|---|---|
| `slice_` | **kept** | used as an index, then **removed** |
| `dice` | **removed** | **kept** (the number itself is ignored) |

### `slice_`: "fix some modes, keep the rest"

Like NumPy's `x[2, :]`, where `None` plays the role of `:`.

```python
L = cute.make_layout((8, 8))         # (8,8):(1,8), column-major (CuTe default)
x = from_dlpack(torch.arange(64).reshape(8, 8).cuda())   # (8,8):(8,1), row-major (PyTorch)

cute.slice_(L, (None, 5))   # (8):(1)   one column, contiguous in col-major
cute.slice_(L, (2, None))   # (8):(8)   one row, jumps of 8 in col-major
cute.slice_(x, (6, None))   # (8):(1)   one row, contiguous in row-major
cute.slice_(x, (None, 3))   # (8):(8)   one column, jumps of 8 in row-major
cute.slice_(L, (5, 5))      # ():()     everything fixed, a single element
```

> [!NOTE]
> The same slice gives different strides on `L` and `x` because one is column-major and the other is row-major.
> Contiguous (stride 1) means "neighbours in memory". So a row is contiguous in PyTorch, and a column is contiguous in CuTe's default layout.

> [!CAUTION]
> **Slicing a layout loses the starting offset.**
>
> ```python
> cute.slice_(L, (None, 2))   # (8):(1) -> offsets 0..7
> cute.slice_(L, (None, 5))   # (8):(1) -> offsets 0..7, the same!
> ```
>
> Column 2 really starts at offset `16`, and column 5 at `40`, but a layout has nowhere to store that. You only get the **shape and stride of what's left**, not **where it starts**.
>
> Slicing a **tensor** keeps the offset, because a tensor = pointer + layout and the shift goes into the pointer. The printed layout looks the same, but the pointer has moved.
> This is the same reason `local_tile` needs an identity **tensor**, not an identity layout.

### `dice`: "keep the modes that have a number"

```python
L = cute.make_layout((3, 4, 5, 6))  # 8x8 grid of 8-bit elements

print(f"{L=}")  # (3, 4, 5, 6):(1, 3, 12, 60)
print(f"{cute.dice(L, (1, None, None, 1))=}")  # (3, 6):(1, 60)
print(f"{cute.dice(L, (None, 1, 1, 1))=}")  # (4, 5, 6):(3, 12, 60)
print(f"{cute.dice(L, (None, 1, None, 1))=}")  # (4, 6):(3, 60)
```

- `dice` does **not** index. The integer is just a marker meaning "keep this mode". By convention people write `1` there.
- Think of it as a **mode picker**: `(1, None, 1)` means "give me modes 0 and 2, drop mode 1".

#### So where is `dice` actually used?

Mostly hidden inside `local_tile`, through its `proj` argument. Take a GEMM, where each block works on a `(BM, BN, BK)` tile of work. Each matrix only cares about some of those modes:

```
A is (M, K) -> needs modes 0 and 2 of (BM, BN, BK) -> proj = (1, None, 1)
B is (N, K) -> needs modes 1 and 2                 -> proj = (None, 1, 1)
C is (M, N) -> needs modes 0 and 1                 -> proj = (1, 1, None)
```

```python
tiler = (128, 128, 64)              # (BM, BN, BK)
coord = (bidx, bidy, None)          # block's position in (M, N, K)

gA = cute.local_tile(mA, tiler, coord, proj=(1, None, 1))
# internally: tiler -> dice -> (128, 64)
#             coord -> dice -> (bidx, None)
```

So you write **one** tiler and **one** block coordinate for the whole GEMM, and `dice` picks the right parts for A, B, and C.

---

---

## `make_layout_tv`

- You describe two small things, and it builds the full mapping:
  - **`thr_layout`**: how threads are arranged across the tile, and how thread ids are numbered.
  - **`val_layout`**: how each thread's own values are arranged, and how value ids are numbered.
- It returns two things:
  - **`tiler_mn`**: the tile shape that all threads together cover in one go.
  - **`tv_layout`**: a mapping from `(thread_id, value_id)` to a position inside that tile.

```python
thr_layout = cute.make_layout((2, 3), stride=(3, 1))   # 2 threads along M, 3 along N, ids go along N first
val_layout = cute.make_layout((2, 2), stride=(2, 1))   # each thread: 2x2 values, ids go along N first
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

# tiler_mn  = (4, 6)
# tv_layout = ((3,2),(2,2)):((8,2),(4,1))
```

### Tile size = thread shape × value shape (per mode)

```
tiler_mn = (2*2, 3*2) = (4, 6)
```

| | count |
|---|---|
| threads | 2 × 3 = 6 |
| values per thread | 2 × 2 = 4 |
| tile elements | 4 × 6 = 24 = 6 × 4 ✓ |

Another example: `thr (3,4)` with `val (6,7)` gives a tile of `(18, 28)`, covered by 12 threads × 42 values = 504 elements.

### The picture: "who owns this element?"

- Threads arrangement `thr_layout: (2,3):(3,1)`:
```
T0 T1 T2
T3 T4 T5
```
- Value arrangement `val_layout: (2,2):(2,1)`:
```
V0 V1
V2 V3
```

- The TV layout is made by **repeating the `val_layout` over the `thr_layout`**

![tv layout](/10-cutedsl/tv_layout.png)

> [!NOTE]
> This table goes **(m, n) → (thread, value)**, which is the easy way for us to read it.
> `tv_layout` goes the **other way**: **(thread, value) → position in the tile**. It's the same information, viewed from the other side.

### Gotchas

> [!CAUTION]
> - `tv_layout` points into the **tile**, not into memory. It knows nothing about strides in gmem or smem. You only get real memory offsets after combining it with a real tensor.
> - The printed shape of `tv_layout` won't match `thr_layout`/`val_layout` one-to-one. Its modes are ordered **by id** (thread id, value id), not by M/N.
> - `thr_layout` and `val_layout` must be **compact** (no gaps).
> - Use `size(tv_layout, mode=[0])` for the **threads per block** and `size(tv_layout, mode=[1])` for the **values per thread**.

---

## `local_tile`

- **"Which tile does this CTA work on?"**
- It cuts the tensor into `tiler`-sized tiles, then hands you the one at `coord`.

```python
gA = cute.local_tile(A, tiler, coord)

# same as:
gA = cute.zipped_divide(A, tiler)[(None, coord)]
#     ((TileM, TileN), (RestM, RestN))
#       ^ keep this     ^ pick one tile
```

### Typical use: tiler from `make_layout_tv`

- `tiler_mn` from `make_layout_tv` is **exactly the area that one CTA's threads cover in one pass** (threads × values).
- If you use it as the tile size, each CTA's tile fits its threads perfectly: every element gets exactly one (thread, value).
- `tv_layout` is applied to that tile in the next step.

```python
# host side (@cute.jit)
thr_layout = cute.make_layout(256)            # 256 threads
val_layout = cute.make_layout(4)              # 4 values each
tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)   # tiler = (1024,)

grid = cute.size(cute.zipped_divide(C, tiler), mode=[1])         # number of tiles = number of CTAs

# kernel side (@cute.kernel)
bidx, _, _ = cute.arch.block_idx()
gA = cute.local_tile(A, tiler, (bidx,))       # this CTA's 1024 elements
```

> [!NOTE]
> The tiler doesn't **have** to come from `make_layout_tv`. In GEMMs the tile shape usually comes from the block/MMA design (e.g. `(128, 128, 64)`), and the threads may cover the tile in several passes. "TV tiler = CTA tile" is the simple case where one pass covers the whole tile.


![local tile](/10-cutedsl/cute-local-tile.svg)

### Key points

- **`coord` is a tile coordinate, not an element coordinate.** `(1,2)` means "2nd tile down, 3rd tile across", which starts at element `(4, 8)`.
- **It returns a tensor**, with the pointer moved to the tile's start. This is the same "slicing a tensor keeps the offset" rule from `slice_`.
- **It only picks the CTA's tile.** It does not split anything among threads. That's the next step.
- **Number of CTAs to launch** = the number of tiles = `size(zipped_divide(A, tiler), mode=[1])`.

### Gotcha: the last tile hangs off the edge

> [!CAUTION]
> If the tensor size isn't a multiple of the tile, the tile count is **rounded up**, but every tile still has the **full tiler shape**.
>
> Example: `N = 2500`, `tiler = (1024,)`, so there are 3 tiles.
> - Tile 2 covers `2048..3071`.
> - Only `2048..2499` actually exist.
>
> The view happily lets you index past the end, and nothing stops you, so you must predicate.
> - Tile the identity tensor the same way (`local_tile(cC, tiler, coord)`) and check `elem_less(coord, A.shape)` before each load/store. [check example](/cuda/10-cutedsl/05-cute-layouts/#identity-layout-vs-identity-tensor)
> ```python
> # Check if the current thread's tile is within bounds using the identity tensor
>    if cute.elem_less(tCrd[cute.size(tCrd) - 1], C.shape):
>        tC[None] = tA.load() + tB.load()
>    else:
>        for i in cutlass.range_constexpr(cute.size(tC)):
>            if cute.elem_less(tCrd[i], C.shape):     # tail: predicated
>                tC[i] = tA[i] + tB[i]
> ```

---

## `local_partition`

- **"Which elements does this thread get?"**
- You give it a thread layout and a thread id, and it gives that thread its share of a tensor.

```python
thr = cute.make_layout((2, 4), stride=(4, 1))   # 8 threads, ids go along cols first
tA  = cute.local_partition(gA, thr, tidx)       # gA: (8, 8) -> tA: (4, 2)

# same as:
gA = cute.zipped_divide(gA, thr)[(tidx, None)]
```

### `local_tile` vs `local_partition`: same cut, opposite pick

Both cut the tensor into tiles with `zipped_divide`. Then they keep **opposite** parts:

```python
# local_tile: keep the inside of ONE tile
zipped_divide(gA, tiler)[(None, tile_coord)]

# local_partition: keep ONE cell from EVERY tile
zipped_divide(gA, thr_shape)[(thr_coord, None)]
```

| | picks | keeps |
|---|---|---|
| `local_tile` | which tile | everything inside it |
| `local_partition` | which cell (the thread's spot) | that cell from all tiles |

### Step by step (thread 5)

1. Cut `gA (8,8)` into `(2,4)` tiles, one cell per thread. That gives a `(4,2)` grid of tiles.
2. Thread 5's spot in `thr` is `(1,1)`, because `1*4 + 1*1 = 5`.
3. Take cell `(1,1)` from every tile. That's rows `{1,3,5,7}` × cols `{1,5}`.

![local partition visualization](/10-cutedsl/local_partition_viz.png)

- Result shape = number of tiles = `gA.shape / thr.shape = (4, 2)`.
- A thread's values are **spread out**: they repeat every `thr` shape.

> [!CAUTION]
> - Call it on the **CTA tile** (`gA` from `local_tile`), not on the whole global tensor. Otherwise every CTA gets the same slices and does the full tensor's work.
> - The **values per thread are decided by the tensor size**, not by you: `size(gA) / size(thr)`.
> - Thread id → spot depends on `thr`'s **stride**. `(2,4):(4,1)` and `(2,4):(1,2)` give thread 5 different spots.
> - Values are **not next to each other** in memory, so you get no vector loads. Each value is its own load.

### Verify: partition an identity tensor

```python
@cute.jit
def check():
    c   = cute.make_identity_tensor((8, 8))
    thr = cute.make_layout((2, 4), stride=(4, 1))
    p   = cute.local_partition(c, thr, 5)
    for i in cutlass.range_constexpr(cute.size(p)):
        cute.printf("v{} -> {}", i, p[i])
# v0 (1,1)  v1 (3,1)  v2 (5,1)  v3 (7,1)  v4 (1,5)  v5 (3,5)  v6 (5,5)  v7 (7,5)
# (values go down the rows first, because colex order makes the first mode fastest)
```

---

## The pattern you'll use most: `composition` + `[(tidx, None)]`

```python
# host
tiler, tv = cute.make_layout_tv(thr_layout, val_layout)

# kernel
gA = cute.local_tile(A, tiler, (bidx,))          # 1. this CTA's tile
tA = cute.composition(gA, tv)[(tidx, None)]      # 2. this thread's values
```

### Why it works

- `tv` maps `(thread, value)` → **position in the tile**. It knows nothing about memory.
- `gA` maps a **position in the tile** → **real memory**.
- `composition(gA, tv)` chains the two: `(thread, value)` → position → memory. The result is a tensor shaped like `tv`: `((threads), (values))`.
- `[(tidx, None)]` fixes the thread and keeps all of its values. That's this thread's view.

```
composition(gA, tv)      : ((threads), (values)) -> memory
           [(tidx, None)]:              (values) -> memory   (one thread)
```

![tv composition visualization](/10-cutedsl/tv_composition_viz.png)


> [!IMPORTANT]
> `gA` must be cut with the **same tiler** that `make_layout_tv` returned. `tv` assumes the tile is exactly `tiler_mn`. With any other tile size, the positions point to the wrong elements.

### How the two compare

| | `local_partition(gA, thr, t)` | `composition(gA, tv)[(t, None)]` |
|---|---|---|
| A thread's values | spread out, every `thr` shape | packed in one `val` block |
| You describe | threads only | threads + values |
| Values per thread | `size(gA) / size(thr)` (decided by the tensor) | `size(val)` (decided by you) |
| Vector loads | no (values aren't adjacent) | yes, if the val block is contiguous in memory |

### Same work, different loads (1D, 1024 floats, 256 threads)

```
local_partition : T0 -> 0, 256, 512, 768     4 separate 4-byte loads
TV (val = 4)    : T0 -> 0, 1, 2, 3           1 single 16-byte load
```

- Both are **coalesced**: a warp touches neighbouring addresses each time.
- Only TV is **vectorized**. A thread's 4 values sit next to each other in memory, so they can be fetched with one 16-byte load. With `local_partition` they're 256 apart, so each one needs its own load.
- Vectorizing depends on the **layout**, not the API: any layout that gives a thread neighbouring values can vectorize.
- That's why the vector-add kernel uses `val_layout = make_layout(4)`: 4 fp32 = 16 bytes = one 128-bit load.

### Which one to use

- **TV + composition**: the default choice. You control exactly which values each thread owns, and you get vectorized loads.
- **`local_partition`**: when you *want* a cyclic split, for example a thread grid sweeping a tile in passes, as in the classic CuTe sgemm tutorial.

> [!TIP]
> Tail handling works exactly the same way: run the identity tensor through the **same** steps (`local_tile` → `composition` → `[(tidx, None)]`) to get each value's global coordinate, then `elem_less(coord, A.shape)`.

---

---

## Registers: rmem tensors, fragments & TensorSSA

Every kernel ends the same way: get the thread's values into **registers**, do the math there, then write the results back.
There are two kinds of "things" involved, and mixing them up is the most common source of confusion.

| | **Tensor** (gmem / smem / rmem) | **TensorSSA** |
|---|---|---|
| What it is | storage: pointer + layout | just values, a result of math |
| Can you index it, `t[i]`? | yes | treat it as one whole value |
| Lives at an address? | yes | no, it lives in registers the compiler picks |
| You get it from | `make_tensor`, `local_tile`, `make_rmem_tensor`, ... | `tensor.load()`, math on SSA values |
| You change it by | `t[i] = x`, `t.store(ssa)` | you don't; math makes a new one |

### `load()` and `store()`: moving between the two

```python
vals = tA.load()          # Tensor -> TensorSSA (reads every element of the view)
out  = vals + tB.load()   # math on whole TensorSSA values, elementwise
tC.store(out)             # TensorSSA -> Tensor (writes every element)

tC[None] = tA.load() + tB.load()   # same as tC.store(...)
```

- `load()`/`store()` work on **any** tensor view: gmem, smem, or rmem. Your vector-add loads straight from a gmem slice.
- They read/write the **whole view** at once. If the layout puts a thread's values next to each other in memory, this becomes a vectorized load (see the TV section).
- Math on TensorSSA is **elementwise**: `+ - * /`, comparisons, `.to(dtype)` for casts, and functions from `cute.math`.

```python
x = tA.load().to(cutlass.Float32)      # fp16 -> fp32 for accurate math
y = (x * scale).to(cutlass.Float16)    # back to fp16 before writing
tC.store(y)
```

> [!CAUTION]
> `load()`/`store()` have **no bounds check**. On the last tile they happily read/write past the end of the tensor.
> For tails, loop over the elements and check each coordinate with `elem_less` (as in the vector-add kernel).

> [!WARNING]
> **TensorSSA math needs the same shape *and* nesting, not just the same size.**
>
> ```python
> # tA : ((2,2)):((1,6))   <- a TV slice is nested
> acc = cute.make_rmem_tensor(cute.size(tA), cutlass.Float32)   # 4:1, flat
> acc.load() + tA.load()   # TypeError: profile of input tuples doesn't match: (4, (2, 2))
> ```
>
> Build buffers **from the tensor you'll combine them with**, so the shapes match automatically (see below).

### Fragment = rmem tensor

- A **fragment** is a thread's own small piece of a tile, held in **registers**.
- In CuTe DSL you create one with `make_rmem_tensor`. The old name was `make_fragment`, and you'll still see "fragment" in CUTLASS code and docs.

```python
# from a shape (or layout) + dtype: compact col-major
acc  = cute.make_rmem_tensor(tA.shape, cutlass.Float32)          # ((2,2)) -> ((2,2)):((1,2))

# from an existing tensor: same shape, nesting & stride order (dtype optional)
rA   = cute.make_rmem_tensor_like(tA)                            # same dtype as tA
racc = cute.make_rmem_tensor_like(tA, cutlass.Float32)           # same shape, fp32
racc = cute.make_rmem_tensor_like(tA, dtype=cutlass.Float32)     # keyword works too

# flat buffer: fine when you only index it with [i], e.g. a mask
pred = cute.make_rmem_tensor(cute.size(tA), cutlass.Boolean)
```

> [!NOTE]
> Tested on `nvidia-cutlass-dsl 4.8.0`:
> - `make_rmem_tensor_like` keeps the source's layout order: a row-major `(2,4):(4,1)` source gives `(2,4):(4,1)` in registers. The changelog says "compact col-major", but that's not what 4.8.0 does.
> - `make_rmem_tensor` with a plain shape gives compact col-major.
> - Both keep the **nesting** when you pass `tA` or `tA.shape`, which is what the SSA-math rule above needs.

### When do you actually need an rmem tensor?

For plain `load → math → store`, you **don't**: the TensorSSA already lives in registers.
You need a real register **buffer** when:

- **The values must survive a loop.** Example: a GEMM accumulator added to across every K tile.
- **Another op writes into it.** `cute.copy` (gmem → rmem) and `cute.gemm` (MMA) need a destination tensor, not an SSA value.
- **You build something element by element.** Example: a predicate mask filled in a loop, then used for a copy.
- **You need to index an SSA result.** Store it into a small rmem tensor first, then use `frag[i]`.

```python
# accumulator pattern
acc = cute.make_rmem_tensor_like(tC, cutlass.Float32)        # same shape as tC, but fp32
for i in cutlass.range_constexpr(cute.size(acc)):
    acc[i] = 0.0
for k in range(num_k_tiles):
    acc.store(acc.load() + tA_k.load().to(cutlass.Float32))   # keeps adding into the same registers
tC.store(acc.load().to(tC.element_type))
```

> [!WARNING]
> **Only index rmem tensors with compile-time indices.** Registers can't be addressed with a runtime index.
> If you do `frag[i]` with a runtime `i`, the compiler moves the fragment to **local memory**, which is slow and off-chip.
> - Loop over fragments with `cutlass.range_constexpr`, which unrolls the loop so every index is a constant.
> - Keep fragment sizes **static**. Their shape must be known at compile time.

### Naming convention you'll see in real CUTLASS code

```
t  A  s  A
│  │  │  └─ the data           : which tensor (A, B, C)
│  │  └──── where it lives     : m / g / s / r / c / p
│  └─────── partitioned by     : which thread arrangement split it
└────────── t = per-thread view: this is one thread's share
```

| letter | lives in | example |
|---|---|---|
| `m` | whole tensor in gmem (before tiling) | `mA` |
| `g` | CTA's tile in gmem | `gA` |
| `s` | tile in smem | `sA` |
| `r` | registers (fragment) | `tArA`, `tCrC` |
| `c` | coordinates (identity tensor) | `tAcA` |
| `p` | predicate mask | `tApA` |

- **Partitioned by** = which thread layout cut the tensor. The **same** tensor can be cut differently for different jobs:
  - `tAsA`: smem A, cut for **copying A in** (which elements this thread writes).
  - `tCsA`: the **same** smem A, cut for the **MMA computing C** (which elements this thread reads).
- The MMA side uses `C` for all its operands (`tCsA`, `tCsB`, `tCrC`), because the MMA computes C.
- With only one partitioner (e.g. a single TV layout in an elementwise kernel), use one letter for all of them: `tXgA`, `tXgB`, `tXgC`, `tXcC`.

> [!TIP]
> Seeing `tAsA` and `tCsA` together means data enters smem with one thread mapping and leaves with another.
> That handoff is exactly where a `__syncthreads()` / barrier is needed.

> [!TIP]
> Typical flow: `mA` → `local_tile` → `gA` → partition → `tAgA` → copy → `tAsA` / `tArA` → math / MMA → `tCrC` → `tCgC`.

---

## Quick recap

| Function / idea | Answers |
|---|---|
| `slice_`, `dice` | keep some modes, drop others |
| `make_layout_tv` | thread & value → position in the tile, plus the tile size |
| `local_tile` | which tile does this CTA work on? |
| `local_partition` | this thread's cell from every tile (spread out) |
| `composition(gA, tv)[(tidx, None)]` | this thread's packed values (vectorizable) |
| `make_rmem_tensor(_like)` | a register buffer that sticks around |
| `.load()` / `.store()` | Tensor ↔ TensorSSA (storage ↔ values) |
