---
title: CuTe layout basics
type: docs
math: true
sidebar:
  open: false
weight: 1002
---

## 0. Mental model (read this first)

```
logical coordinate ──Layout──► index (offset into memory / thread id / anything)
   (i, j)                         i*stride_i + j*stride_j
```

- A **Layout is just a function**: `coordinate → integer`. Nothing more.
- It is written `Shape : Stride`. The **shape** says what coordinates are legal, the **stride** says how far each coordinate step moves the output.
- **Layout algebra** = a small set of operations that take layouts and return layouts (compose, divide, product, complement, inverse). Because layouts are functions, these are just function-level tricks: "do A after B", "split this domain into tile + rest", "repeat this pattern", and so on.
- **Everything else in CuTe is built on it.** Tiling a tensor, handing each thread its slice, describing how an MMA spreads a tile over 32 lanes, swizzling smem: all are layouts plus layout algebra.

```
Tensor   = pointer (engine) + Layout
Tile     = a divide of a Layout
Partition= a divide + pick my slice
TV layout= a Layout mapping (thread, value) → position in tile
Fragment = a small register Tensor, one per thread, holding that thread's values
```

> [!IMPORTANT]
> - If you only remember one thing: **a layout answers "where does element #k (or coordinate c) live?"**. The algebra answers "how do I build the layout I want out of simpler ones?".

---

## 1. Layout basics

### 1.1 Shape, stride, coordinate → index

```python
import cutlass.cute as cute

@cute.jit
def demo():
    L = cute.make_layout((4, 8), stride=(8, 1))     # row-major 4x8
    print(L)                                        # (4,8):(8,1)
    print(cute.crd2idx((2, 3), L))                  # 2*8 + 3*1 = 19
    print(cute.idx2crd(19, L.shape))                # (19%4, (19//4)%8) = (3, 4)

if __name__ == "__main__":
    demo()
```
> [!CAUTION]
> - **`crd2idx` (coordinate → index):** The returned `idx` represents the linear offset calculated from the coordinate and the layout's strides (i.e., the coordinate–stride dot product).
>
> - **`idx2crd` (index → coordinate):** Unlike `crd2idx`, this does not necessarily represent a physical memory offset. When provided only a shape, it interprets the index using the shape's default compact, lexicographic coordinate ordering, without considering an arbitrary layout's strides.
>
> - **`idx2crd(crd2idx(input))` is not guaranteed to equal `input`**, because `crd2idx` uses the layout's strides to calculate the linear index, whereas `idx2crd` uses the shape's default coordinate enumeration.
>
> - In other words, `idx2crd` answers *"What is the coordinate of the nth element in the default compact ordering?"*, whereas `crd2idx` answers *"At what linear offset is this coordinate located according to the given layout?"*
>
> - The naming can be misleading: `idx` in `crd2idx` is effectively a layout-dependent linear offset, while `idx` in `idx2crd` is a logical linear index. A name like `crd2offset` would arguably communicate the former's behavior more clearly.

| Layout | Meaning | `(i, j) → index` |
|---|---|---|
| `(4,8):(8,1)` | row-major | `8i + j` |
| `(4,8):(1,4)` | column-major | `i + 4j` |
| `(4,8):(0,1)` | row broadcast (every row reads the same 8 elements) | `j` |
| `(4,8):(8,2)` | every other column | `8i + 2j` |

> [!IMPORTANT]
> - **Stride 0 = broadcast.** Many threads / rows can map to the same element. Used a lot for bias and scale tensors.
> - `make_layout((4,8))` with no stride gives the **compact column-major** layout `(4,8):(1,4)`, i.e. the **left-most mode is fastest**. This is the opposite of NumPy's default. It is `column-major`.
> - Row-major, explicitly: `stride=(8,1)` or `cute.make_ordered_layout((4,8), order=(1,0))` (order lists dims from fastest to slowest).

### 1.2 1D coordinates also work (colexicographic)

Any layout can be indexed with a **single integer**. It gets unfolded left-to-right, leftmost mode first:

```python
cute.idx2crd(11, (5, 4))     # (11 % 5, 11 // 5 % 4) = (1, 2)
```

So a `(4,8)` layout is also a 32-element 1D layout. `L(19)` and `L((3, 4))` are the same thing, since `19 % 4 = 3` and `19 // 4 = 4`.

> [!TIP]
> This is why a layout can be treated as "a list of `size(L)` positions" regardless of rank. Divide, product and TV layouts all lean on this.

### 1.3 Hierarchical (nested) layouts

Shapes can nest. A nested mode means "this one logical dimension is really several sub-dimensions":

```
((2,2),3) : ((1,6),2)
  ^^^^^   ^
  mode 0 is itself a 2x2, mode 1 is plain 3
```

Why bother? Because it lets one mode represent *"a tile of data"* and another *"how many tiles"*, without losing either. Tile/rest, thread/value, and warp/lane/value are all nested layouts.

### 1.4 Vocabulary

| Term | Meaning | API |
|---|---|---|
| `rank` | number of top-level modes | `cute.rank(L)` |
| `depth` | nesting level | `cute.depth(L)` |
| `size` | number of coordinates (product of shape) | `cute.size(L)` |
| `cosize` | size of the output range, i.e. `max index + 1`. **This is the allocation size.** | `cute.cosize(L)` |
| mode | one top-level entry of the shape | `cute.shape(L, mode=0)` |
| congruent | same nesting profile (tuples match tuples, ints match ints) | `cute.is_congruent(a, b)` |
| weakly congruent | `a` can be broadcast up to `b` (an int may stand in for a tuple) | `cute.is_weakly_congruent(a, b)` |
| static / dynamic | known at compile time or runtime | `cute.is_static(x)` |

```python
import cutlass
import cutlass.cute as cute


@cute.jit
def demo():
    L = cute.make_layout(((2,2), 8), stride=((4,2), 1))     # row-major 4x8
    print(L)                                         # (4,8):(8,1)
    helpers = ("rank", "depth", "size", "cosize", "is_static")
    for hlp_method in helpers:
        mthd = getattr(cute, hlp_method)
        print(f"{hlp_method}: {mthd(L)}")
    
    print("-"*70)
    for mod in cutlass.range_constexpr(cute.rank(L)):
        print(f"mode: {mod}; shape: {cute.shape(L, mode=mod)}")
    
    print("-"*70)
    l1 = cute.make_layout((3,4))
    l2 = cute.make_layout(((2,2),8))

    print(f"{cute.is_congruent(l1,l2)=}")
    print(f"{cute.is_weakly_congruent(l1,l2)=}")

if __name__ == "__main__":
    demo()

# ⚡ ~/cute-layouts python main.py 
# ((2,2),8):((4,2),1)
# rank: 2
# depth: 2
# size: 32
# cosize: 14
# is_static: True
# ------------------------------------
# mode: 0; shape: (2, 2)
# mode: 1; shape: 8
# ------------------------------------
# cute.is_congruent(l1,l2)=False
# cute.is_weakly_congruent(l1,l2)=True
```

> [!NOTE]
> `size` vs `cosize`: `(4,8):(8,1)` has `size = 32` and `cosize = 32`. `(4,8):(16,1)` still has `size = 32`, but `cosize = 56` because the rows are separated by a gap. So allocate `cosize` elements, not `size`.
>
> To compute it, find the last valid coordinate: `(3,7)` for shape `(4,8)`, then take the dot product with the stride to get the maximum offset. That gives the allocation size: `3*stride[0] + 7*stride[1] + 1` (since the base address is 0).

#### congruence

- **`is_congruent(a, b)`**: Requires an exact structural match. Every tuple level must have the same rank, and scalars must match with scalars.

- **`is_weakly_congruent(a, b)`**: Allows a to be flatter than b. A scalar element in a is allowed to match against a nested tuple in b.

### 1.5 Structure-only helpers (don't change the mapping)

| Op | What it does | Example |
|---|---|---|
| `group_modes(L, b, e)` | wrap modes `[b, e)` into one nested mode | `(2,3,4,5)` → `(2,(3,4),5)` |
| `flatten(L)` | remove all nesting | `((2,2),3)` → `(2,2,3)` |
| `slice_(L, coord)` | fix some coords, keep `None` modes | `slice_(L, (1, None))` |
| `dice(L, dicer)` | opposite of slice: keep modes where dicer has an int | |
| `append / prepend` | add a mode at the end / front | |
| `coalesce(L)` | merge neighbouring modes when it doesn't change the function | `(2,(1,6)):(1,(6,2))` → `12:1` |
| `filter_zeros(L)` | drop stride-0 modes | |

`coalesce` is the "simplify" button: same coordinate → index mapping (on the 1D view), minimal shape.

```python
import cutlass
import cutlass.cute as cute
import torch


@cute.jit
def demo(x: torch.Tensor):
    l1 = cute.make_layout((3,4,5,(6,7),8,9))
    print(f"{cute.group_modes(l1, 2, 5)=}")
    print(f"{cute.flatten(l1)=}")

    print("-"*50)
    # Layout slicing
    l2 = cute.make_layout((24,44))

    # Select 1st index of first mode and keep all elements in second mode
    sub_layout = cute.slice_(l2, (1, None))
    print(f"{sub_layout=}; {sub_layout(3)=}")

    sub_tensor = cute.slice_(x, (2, None))

    for i in cutlass.range_constexpr(5):
        cute.printf("i: {} => val: {}", i, sub_tensor[i])

    print("-"*50)

    l3 = cute.make_layout((4,4),stride=(24, 26))
    sub_l3 = cute.dice(l3, (1, None))
    print(f"{sub_l3=}")
    print(f"{sub_l3(3)=}")
    print("-"*50)

    print(f"{l2=}; {l3=}")
    print(f"{cute.shape_div(l2.shape, l3.shape)=}")




if __name__ == "__main__":
    x = torch.arange(20).reshape(4,5)
    demo(x)

# ⚡ ~/cute-layouts python 04-step_one.py
# cute.group_modes(l1, 2, 5)=(3,4,(5,(6,7),8),9):(1,3,(12,(60,360),2520),20160)
# cute.flatten(l1)=(3,4,5,6,7,8,9):(1,3,12,60,360,2520,20160)
# --------------------------------------------------
# sub_layout=(44):(24); sub_layout(3)=72
# --------------------------------------------------
# sub_l3=(4):(24)
# sub_l3(3)=72
# --------------------------------------------------
# l2=(24,44):(1,24); l3=(4,4):(24,26)
# cute.shape_div(l2.shape, l3.shape)=(6, 11)
# --------------------------------------------------
# (below is the output of slice, since it used cute.printf, it's printed at the runtime, not compile time)
# i: 0 => val: 10
# i: 1 => val: 11
# i: 2 => val: 12
# i: 3 => val: 13
# i: 4 => val: 14
```

---

## 3. Layouts in practice: tiles, partitions, TV layouts

Here the algebra turns into a kernel. We follow data from global memory to registers.

```
global tensor  (M, N)
   │  zipped_divide / local_tile       "which tile is mine?"        → block-level
   ▼
CTA tile       (BM, BN)
   │  local_partition / partition_S    "which elements are mine?"   → thread-level
   ▼
thread slice   (V, ...)      ← in gmem / smem
   │  cute.copy / cute.gemm
   ▼
fragment       (V, ...)      ← in registers (rmem)
```

### 3.1 Tiles (block level): `local_tile`

`local_tile` is `zipped_divide` + "slice out the tile this block owns".

```python
# gA : (M, K) global tensor,   tile = (BM, BK)
tiler = (BM, BK)
gA_tiles = cute.zipped_divide(gA, tiler)          # ((BM,BK), (M/BM, K/BK))
my_tile  = gA_tiles[(None, (bidx, kblk))]         # (BM, BK)  <- slice rest by tile coord

# same thing in one call:
my_tile = cute.local_tile(gA, tiler, coord=(bidx, kblk))
```

Using `None` for a coord keeps that mode (an unsliced "loop over these" dimension):

```python
gA_blk = cute.local_tile(gA, (BM, BK), coord=(bidx, None))   # (BM, BK, num_k_tiles)
for k in range(cute.size(gA_blk, mode=[2])):
    tile_k = gA_blk[None, None, k]                           # (BM, BK)
```

**GEMM-style: one tiler for M,N,K, project per operand.** `proj` says which dims of the tiler an operand actually has.

```python
tiler_mnk = (BM, BN, BK)
coord_mnk = (bidx, bidy, None)

gA = cute.local_tile(mA, tiler_mnk, coord_mnk, proj=(1, None, 1))   # (BM, BK, k)   uses M,K
gB = cute.local_tile(mB, tiler_mnk, coord_mnk, proj=(None, 1, 1))   # (BN, BK, k)   uses N,K
gC = cute.local_tile(mC, tiler_mnk, coord_mnk, proj=(1, 1, None))   # (BM, BN)      uses M,N
```

> [!IMPORTANT]
> - `proj=(1, None, 1)` means "use dims M and K of the tiler/coord, skip N". `1` = keep, `None` = drop. One tiler, three consistent tile views.
> - This is "layout algebra as bookkeeping": no data moves, `my_tile` is just a **view** with a new layout and a shifted pointer.

### 3.2 Partitions (thread level): `local_partition`

Block-level gave us a `(BM, BN)` tile. Now we split it across threads. Describe the **thread arrangement** as a layout, and divide the tile by it.

```python
thr_layout = cute.make_layout((4, 32), stride=(32, 1))      # 128 threads, as a 4x32 grid

# every thread gets the elements at its "position" in each repeat of the thread grid
tCgC = cute.local_partition(gC, thr_layout, tidx)
# shape: (BM/4, BN/32)   -> this thread's elements across the whole tile
```

Under the hood: `zipped_divide(gC, thr_layout)` → `((4,32), (BM/4, BN/32))`, then slice mode 0 with this thread's coordinate. **Mode 0 of the divide is "which thread", mode 1 is "my elements".**

Note this is the *thread-interleaved* partition (thread `t` owns `t, t+128, ...`), the natural one for scalar coalesced accesses (neighbouring threads touch neighbouring addresses). For *vectorized* ownership (each thread owns a contiguous chunk) we need a TV layout.

**Naming convention** (you'll see this everywhere in CUTLASS): `t` + `<which thread layout>` + `<what it's a view of>`:

| Name | Reads as |
|---|---|
| `gA` | global memory tile of A |
| `sA` | shared memory tile of A |
| `rA` / `tArA` | register fragment of A |
| `tAgA` | **t**hread's partition (by the A-copy) of **g**lobal **A** |
| `tAsA` | thread's partition (by the A-copy) of **s**hared **A** |
| `tCrC` | thread's partition (by the MMA) of **r**egister **C** (accumulator) |

### 3.3 TV layouts: the thread-value layout

> **TV layout = a layout `(thread, value) → position in tile`.**
> Mode 0 is "which thread", mode 1 is "which of that thread's values". The output is the (1D-unfolded) coordinate inside the tile.

This one idea describes **both** vectorized copies and tensor-core MMA operand fragments.

**Toy example.** 8-element tile, 2 threads, 4 values each.

```
contiguous (vectorized)   (2,4):(4,1)     thread t, value v → 4t + v
    pos:   0 1 2 3 | 4 5 6 7
    owner: T0 T0 T0 T0 | T1 T1 T1 T1

interleaved (coalesced)   (2,4):(1,2)     thread t, value v → t + 2v
    pos:   0  1  2  3  4  5  6  7
    owner: T0 T1 T0 T1 T0 T1 T0 T1
```

Same tile, same count of work, different ownership. The first lets a thread issue one 128-bit load; the second makes every warp-level access perfectly coalesced with scalar loads.

**Building one for a copy:**

```python
thr_layout = cute.make_layout((4, 32), stride=(32, 1))   # 128 threads
val_layout = cute.make_layout((1, 4))                    # each thread: 4 values along N

# helper that does raked_product(thr, val), computes the tile, and builds the TV layout
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
# tiler_mn = (4, 128)    -> one pass of 128 threads x 4 values covers a 4x128 tile
# tv_layout: (128 threads, 4 values) -> position in that 4x128 tile
```

```
tile (4 x 128), each thread owns 4 contiguous elements along N:

 T0: 0..3 | T1: 4..7 | T2: 8..11 | ...   (a row of 128 cols = 32 threads * 4)
```

`thr_layout` fixes *how threads are arranged*, `val_layout` fixes *what each thread grabs*, and the raked product spreads them so a thread's values are neighbours in memory (vectorizable).

**MMA fragments are TV layouts too.** In an m16n8k16 style MMA, the A/B/C operand tile is spread over 32 lanes with a fixed pattern. For example an 8x8 tile (column-major index `row + 8*col`), where lane `l` holds row `l/4` and columns `2*(l%4)` and `+1`:

```
tv = ((4,8),2) : ((16,1),8)         # ((lane%4, lane/4), v) → row + 8*col
      ^^^^^^^   ^
      lane      the 2 values per lane
 index = 16*(lane%4) + 1*(lane/4) + 8*v
```

That's what the hardware requires, and it's why `ldmatrix` loads land in the right registers. You never write these by hand for MMA; the **MMA atom** carries its TV layouts, and `make_tiled_mma` repeats them across warps.

### 3.4 Tiled copy: putting copy atom + TV layout together

```python
# 1. a copy atom: "one thread moves N bits with this instruction"
copy_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128,
)

# 2. a tiled copy: the atom spread across threads/values by a TV layout
tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

# 3. in the kernel: this thread's view
thr_copy = tiled_copy.get_slice(tidx)

tAgA = thr_copy.partition_S(gA)        # S = source tensor, as this thread sees it
tAsA = thr_copy.partition_D(sA)        # D = destination tensor, as this thread sees it

# 4. copy: loops over the repeats, each does one vectorized instruction per thread
cute.copy(tiled_copy, tAgA, tAsA)
```

Shape of `tAgA`: `(CPY, CPY_M, CPY_N, ...)`:
- `CPY` = what **one copy instruction** moves for this thread (e.g. 4 floats). This mode is consumed by the atom.
- `CPY_M, CPY_N` = how many times this thread repeats the instruction to cover the tile.

> [!IMPORTANT]
> - `partition_S` and `partition_D` use the **same TV layout**, so thread `t` reads and writes the *same logical positions*, just in different memory (global, shared). That's what makes the copy correct regardless of the memory layouts on either side.
> - Whether the copy is really 128-bit depends on `max_common_vector` of the two layouts and the pointer alignment (`assumed_align=16`, see [basics](../01-basics)). If the compiler can't prove it, you silently get narrower accesses.
> - Variants: `make_tiled_copy(atom, layout_tv, tiler_mn)` if you already have a TV layout and tile; `make_tiled_copy_A/B/C(atom, tiled_mma)` build a copy whose layout matches the MMA's operand, so smem → register loads land exactly in MMA fragment order.

### 3.5 Fragments: the register tensors

A **fragment** is a small tensor in **registers** (`rmem`) holding *one thread's* values. Its layout is just the shape of the partition, but compact (no gaps), since registers aren't addressable memory.

```python
# make a register tensor with the same shape as the thread's partition
rA = cute.make_fragment_like(tAgA)           # (CPY, CPY_M, CPY_N), compact in registers

cute.copy(tiled_copy, tAgA, rA)              # gmem → registers
# ... compute on rA elementwise ...
cute.copy(tiled_copy, rA, tAgC)              # registers → gmem
```

For an MMA accumulator:

```python
tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk=(2, 2, 1))
thr_mma   = tiled_mma.get_slice(tidx)

tCgC = thr_mma.partition_C(gC)               # my slice of C in gmem
tCrC = tiled_mma.make_fragment_C(tCgC)       # register accumulator, same shape
tCrC.fill(0.0)
```

> [!NOTE]
> Register tensors **must have static layouts** so the compiler can keep them in registers and unroll every index. A dynamic index into a register fragment forces local memory (spills). Check names against your installed version: `make_fragment_like`, `make_fragment` and `make_rmem_tensor` have moved around between releases.

> [!IMPORTANT]
> - A register fragment loop like `for i in range(cute.size(rA)):` must have a **compile-time** trip count so it unrolls (`cutlass.range_constexpr` or a static `size`).
> - `cute.size(rA)` is a Python int here, which is why loops over fragments are free.

### 3.6 Predication: when the tile doesn't divide the tensor

If `M % BM != 0` the last tile hangs off the end. Build an **identity tensor** (each element holds its own coordinate), partition it exactly like the data, and compare coordinates to the bounds:

```python
cA   = cute.make_identity_tensor(gA_full.shape)         # element (i,j) holds the coord (i,j)
cA_t = cute.local_tile(cA, tiler, coord)                 # tile it the same way as the data
tAcA = thr_copy.partition_S(cA_t)                        # partition it the same way, too

pred = cute.make_fragment(tAcA.shape, cutlass.Boolean)
for i in range(cute.size(pred)):
    pred[i] = cute.elem_less(tAcA[i], gA_full.shape)     # coord < shape in every dim

cute.copy(tiled_copy, tAgA, rA, pred=pred)
```

This is the neat part of "layout = function": the **identity layout** (`make_identity_layout`, stride `(1@0, 1@1)`) maps coordinate → coordinate, so after any tile/partition you can ask each element *"what logical coordinate are you?"*.

---

## 4. Putting it together: vectorized elementwise add

One block handles a tile, each thread owns 4 contiguous floats, copies via 128-bit loads.

```python
@cute.kernel
def add_kernel(gA, gB, gC, tiled_copy, tiler_mn):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 1. block-level: which tile of the global tensor is mine?
    blk = (None, bidx)                                   # slice the "rest" mode
    bA, bB, bC = gA[blk], gB[blk], gC[blk]               # each (tile_m, tile_n)

    # 2. thread-level: which elements within the tile are mine?
    thr = tiled_copy.get_slice(tidx)
    tAgA, tBgB, tCgC = thr.partition_S(bA), thr.partition_S(bB), thr.partition_D(bC)

    # 3. fragments in registers
    rA = cute.make_fragment_like(tAgA)
    rB = cute.make_fragment_like(tBgB)
    rC = cute.make_fragment_like(tCgC)

    cute.copy(tiled_copy, tAgA, rA)                      # gmem → rmem (128-bit)
    cute.copy(tiled_copy, tBgB, rB)
    rC.store(rA.load() + rB.load())                      # compute in registers
    cute.copy(tiled_copy, rC, tCgC)                      # rmem → gmem


@cute.jit
def add(mA, mB, mC):
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((1, 4))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type,
                               num_bits_per_copy=128)
    tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
    tiler_mn, _ = cute.make_layout_tv(thr_layout, val_layout)      # (4, 128)

    gA = cute.zipped_divide(mA, tiler_mn)                # ((4,128), (M/4, N/128))
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    num_tiles = cute.size(gA, mode=[1])                  # number of "rest" elements
    add_kernel(gA, gB, gC, tiled_copy, tiler_mn).launch(
        grid=[num_tiles, 1, 1], block=[cute.size(thr_layout), 1, 1],
    )
```

Every line is layout algebra:

| Step | Algebra | Effect |
|---|---|---|
| `zipped_divide(mA, tiler_mn)` | divide | `((tile),(which tile))` |
| `gA[(None, bidx)]` | slice | pick this block's tile |
| `make_layout_tv` | raked product + right inverse | `(thread, value) → tile position` |
| `partition_S/D` | compose with TV layout | this thread's view |
| `make_fragment_like` | compact layout | register copy of that view |

> [!TIP]
> When something is confusing, **`print` the layout at each step** (compile-time, static). The shapes tell the whole story: `print(gA)`, `print(tAgA)`, `print(rA)`. Static dims show as numbers, dynamic as `?`. See [basics](../01-basics#print-vs-cuteprintf).

---

## 5. Debugging & gotchas

- **Rest mode vs tile mode:** with `zipped_divide`, mode 0 = inside the tile, mode 1 = which tile. Mixing them up is the most common bug. Print the shape.
- **`None` keeps, an int slices.** `gA[(None, 3)]` fixes the *second* mode to 3 and keeps the whole first mode.
- **Non-divisible shapes** need predication (§3.6). `zipped_divide` happily gives a partial last tile.
- **Dynamic shapes break vectorization**: if the compiler can't prove divisibility/alignment, you silently get scalar loads. Fix with `from_dlpack(...).mark_compact_shape_dynamic(mode=..., divisibility=...)`.
- **Layout is not data.** Slicing, dividing, partitioning only build new views. The only things that touch memory are `copy`, `gemm`, `load/store` and indexing a *fully-sliced* element.
- **Allocate `cosize`**, not `size`, for anything with gaps (padding, swizzle).
- **Swizzle** composes with a layout: `make_composed_layout(swizzle, 0, layout)`. It permutes the *output index* of the layout and never the coordinates. See [shared memory](../03-shared-memory#3-swizzle).
