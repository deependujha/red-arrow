---
title: Cute Layouts
type: docs
math: true
sidebar:
  open: false
weight: 1005
---

## What is a layout?

- A layout is just a **function**: you give it a coordinate and it gives you back a **memory offset** (a single number).
- It is made of two things:
  - **shape**: how big each dimension is.
  - **stride**: how many steps you jump in memory when you move by 1 in that dimension.
- It's written as `shape:stride`, for example `(3,4):(4,1)`.

```
offset = coord[0]*stride[0] + coord[1]*stride[1] + ...
```

Example: for the layout `(3,4):(4,1)`, the coordinate `(1,2)` gives offset `1*4 + 2*1 = 6`.

---

## Colexicographical isomorphism

### Row-major vs column-major

- **Row-major (lexicographical)**: the **last** dimension changes fastest. Python, C++, and NumPy use this.
- **Column-major (colexicographical, or "colex")**: the **first** dimension changes fastest. CuTe uses this.

Take a `(3,4)` array and number its elements in the order you walk through them.

Row-major walks along each row first:

```
[[0, 1,  2,  3],
 [4, 5,  6,  7],
 [8, 9, 10, 11]]
```

```
0  - (0,0)      6  - (1,2)
1  - (0,1)      7  - (1,3)
2  - (0,2)      8  - (2,0)
3  - (0,3)      9  - (2,1)
4  - (1,0)      10 - (2,2)
5  - (1,1)      11 - (2,3)
```

Colex walks down each column first. It's the **same** `(3,4)` array, just numbered in a different order:

```
[[0, 3, 6,  9],
 [1, 4, 7, 10],
 [2, 5, 8, 11]]
```

```
0  - (0,0)      6  - (0,2)
1  - (1,0)      7  - (1,2)
2  - (2,0)      8  - (2,2)
3  - (0,1)      9  - (0,3)
4  - (1,1)      10 - (1,3)
5  - (2,1)      11 - (2,3)
```

> [!NOTE]
> Index `6` in a `(3,4)` array:
> - In row-major, it is `(1,2)`, because `6 // 4 = 1` and `6 % 4 = 2`.
> - In colex, it is `(0,2)`, because `6 % 3 = 0` and `6 // 3 = 2`.

### Index → coordinate (colex)

The first dim is peeled off first:

```
i0 = index % shape[0]
i1 = (index // shape[0]) % shape[1]
i2 = (index // (shape[0] * shape[1])) % shape[2]
...

coord = (i0, i1, i2, ...)
```

In CuTe this is `cute.idx2crd(index, shape)`. Note that it takes a **shape**, not a layout. If you have a layout, pass `layout.shape`.

```python
import cutlass.cute as cute

@cute.jit
def foo():
    coord = cute.idx2crd(11, (5, 4))
    # colex: (11 % 5, 11 // 5 % 4) = (1, 2)
    cute.printf("coord: {}", coord)

foo()  # (1, 2)
```

> [!WARNING]
> - The official docs say "idx2crd is always lexicographical", but the formula is actually **colex** (the first dim is fastest). Trust the formula, not the word.
> - `lexicographical` means row-major, but CuTe uses `colexicographical` (first dim fastest, column-major order).

### Coordinate → index (colex)

```
index = i0 + i1*shape[0] + i2*shape[0]*shape[1] + ...
```

### So what is the "isomorphism"?

- A scalar index and a coordinate are **two names for the same element**. In a `(3,4)` shape, `6` and `(0,2)` point to the same spot.
- Because of this, a layout accepts **either one**. If you pass a scalar, CuTe first turns it into a coordinate using colex, then applies the strides.
- "Isomorphism" just means "same thing, written differently".

```python
L = make_layout((3,4), stride=(4,1))   # row-major memory

L((0,2))   # 0*4 + 2*1 = 2
L(6)       # 6 -> (0,2) using colex -> 2  (same answer)
```

> [!IMPORTANT]
> - The scalar index you pass in (`6`) is **not** the memory offset you get out (`2`).
> - The index is "which element, counted in colex order".
> - The offset is "where that element sits in memory".

---

## `cute.make_layout`

- Creates a layout from a shape and an optional stride.
- If you don't give a stride, CuTe makes a **compact column-major** one, where the first dim gets stride 1 and there are no gaps.

```python
# (4,4) with default stride -> (4,4):(1,4), column-major
layout = cute.make_layout((4,4))

# Get the memory offset of the element at (0,0)
offset = layout((0,0))

# Same thing using crd2idx
offset = cute.crd2idx((0,0), layout)

# Custom strides
layout = cute.make_layout((4,4), stride=(1,4))       # column-major (same as default)
layout = cute.make_layout((4,4), stride=(4,1))       # row-major

# 3D, default stride -> (32,16,8):(1,32,512)
layout = cute.make_layout((32,16,8))

# Any strides you want
layout = cute.make_layout((2,2,2), stride=(4,1,2))

# Glue layouts together: each one becomes one mode (dimension)
mode0 = cute.make_layout(64, stride=1)
mode1 = cute.make_layout(128, stride=64)
combined = cute.make_layout([mode0, mode1])          # (64,128):(1,64)
```

> [!NOTE]
> `stride` is keyword-only. You must write `make_layout((3,4), stride=(1,4))`, not `make_layout((3,4), (1,4))`, because the second form looks too much like a nested shape.
> - stride: `(1,4) => column-major (default)`
> - stride: `(4,1) => row-major`


### `crd2idx` vs `idx2crd`, and why the names are confusing

```python
L = cute.make_layout((5, 4), stride=(4, 1))
cute.crd2idx((2, 3), L)    # 2*4 + 3*1 = 11  -> memory offset
cute.idx2crd(11, (5, 4))   # (1, 2)          -> colex coordinate
```

> [!CAUTION]
> The word `idx` means different things in these two functions:
> - In `crd2idx`, `idx` is the **memory offset**: coordinate · stride. It uses the layout's strides.
> - In `idx2crd`, `idx` is the **colex index**: "which element, counted in colex order". It uses only the shape and ignores strides.
>
> So `crd2idx` and `idx2crd` are **not** inverses of each other unless the layout is compact column-major.

---

## `cute.make_identity_layout`

- A normal layout turns a coordinate into **one number** (an offset).
- An identity layout turns a coordinate into **the same coordinate**. You put `(2,3)` in and get `(2,3)` back.

```python
L = cute.make_identity_layout((4,4))   # (4,4):(1@0,1@1)
L((2,3))   # -> (2,3), not a number
```

- What `1@0` and `1@1` mean:
  - `1@0` means "1 step along mode 0", so it's like the vector `(1,0)`.
  - `1@1` means "1 step along mode 1", so it's like the vector `(0,1)`.
  - Applying the layout to `(i,j)` gives `i*(1,0) + j*(0,1) = (i,j)`. Each coordinate maps to itself.

> [!NOTE]
> - Why you'd want a layout that does "nothing":
>   - When you split a big tensor into tiles and threads, each thread gets some **values**, but it no longer knows **where** those values came from.
>   - If you split the identity layout exactly the same way, each thread gets the **original (row, col)** of every element it holds.
>   - That's what you need for bounds checks: if a tile hangs past the edge of the matrix, the thread checks `row < M and col < N` before loading.

> [!TIP]
> Think of the identity layout as a "coordinate tracker". Split it the same way as your data, and it tells each thread where its data lives in the full tensor.

### Identity layout vs identity tensor

- **Tensor = iterator (start point) + layout (how to step)**
  - A normal tensor's iterator is a pointer, so reading it gives a **value**.
  - An identity tensor's iterator is a coordinate counter starting at `(0,)`, so reading it gives a **coordinate**. It uses no memory.
- `make_identity_layout(shape)` is just the function `coord -> coord`. It has no start point.
- `make_identity_tensor(shape)` is that identity layout plus a start point. It behaves like a "fake tensor full of coordinates".

> [!IMPORTANT]
> - In kernels, use `make_identity_tensor`, not the layout.
> - `local_tile` / `partition` only accept tensors ("views"). They also shift the tensor's start point to the tile's origin, which is what makes the coordinates **global** (block 1 starts at 1024, not 0).
> - A bare layout can't hold that shift, so every tile would think it starts at 0, and the bounds checks would break.

```python
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vector_addition_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    cC: cute.Tensor,
    N: cute.Uint32,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    tv_tiler: cute.typing.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    gA = cute.local_tile(A, tv_tiler, (bidx,))
    gB = cute.local_tile(B, tv_tiler, (bidx,))
    gC = cute.local_tile(C, tv_tiler, (bidx,))
    gCrd = cute.local_tile(cC, tv_tiler, (bidx,))


    tA = cute.composition(gA, tv_layout)[(tidx, None)]
    tB = cute.composition(gB, tv_layout)[(tidx, None)]
    tC = cute.composition(gC, tv_layout)[(tidx, None)]
    tCrd = cute.composition(gCrd, tv_layout)[(tidx, None)]

    # Check if the current thread's tile is within bounds using the identity tensor
    if cute.elem_less(tCrd[cute.size(tCrd) - 1], C.shape):
        tC[None] = tA.load() + tB.load()
    else:
        for i in cutlass.range_constexpr(cute.size(tC)):
            if cute.elem_less(tCrd[i], C.shape):     # tail: predicated
                tC[i] = tA[i] + tB[i]


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    thr_layout = cute.make_layout(256)  # 256 threads along the vector
    val_layout = cute.make_layout(4)  # 4 fp32 = 16B per thread

    tv_tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    num_tiles = cute.zipped_divide(C, tv_tiler)

    # CREATE identity tensor FOR TRACKING IF EACH ELEMENT IS WITHIN BOUNDS
    # This tensor maps each element to its coordinate, allowing us to check if it's within the valid range of C.
    cC = cute.make_identity_tensor(C.shape)  # element i -> coordinate (i,)

    BLOCKS = [cute.size(tv_layout, mode=[0]), 1, 1]
    GRIDS = [cute.size(num_tiles, mode=[1]), 1, 1]

    print(f"{BLOCKS=}; {GRIDS=}")
    print(f"{tv_tiler=}")
    vector_addition_kernel(
        A, B, C, cC, N, thr_layout, val_layout, tv_tiler, tv_layout
    ).launch(
        grid=GRIDS,
        block=BLOCKS,
    )
```

---

## `cute.make_ordered_layout`

- Same as `make_layout(shape)`, except **you pick which dimension changes fastest** in memory. CuTe works out the strides for you.
- `order[i]` says how fast dim `i` changes:
  - `0` means it changes fastest (stride 1).
  - Bigger numbers change slower.

```python
make_ordered_layout((4,4), order=(0,1))   # dim0 fastest -> stride (1,4)  column-major
make_ordered_layout((4,4), order=(1,0))   # dim1 fastest -> stride (4,1)  row-major
```

- How the strides get built, using `(32,16,8)` with `order=(2,0,1)` as the example:
  1. dim1 has order 0, so it's fastest and gets stride `1`.
  2. dim2 has order 1, so it comes next. Its stride is `1 * 16` (the size of dim1), which is `16`.
  3. dim0 has order 2, so it's slowest. Its stride is `16 * 8` (the size of dim2), which is `128`.
  4. The result is stride `(128, 1, 16)`.

> [!CAUTION]
> `order` is **not** a list of dims from fastest to slowest. It is "the rank of each dim".
> `order=(2,0,1)` means dim0 is the slowest, dim1 is the fastest, and dim2 is in the middle.
> It does **not** mean "dim2 first, then dim0, then dim1".

---

## `cute.make_layout_like`

- You give it a layout (or a tensor). It returns a **compact** layout with:
  - the **same shape**,
  - the **same "which dim is fastest" order** as the input,
  - but **no gaps or padding**. The strides are packed tight.

```python
src = cute.make_layout((4,4), stride=(8,1))   # row-major, but each row is padded to 8
dst = cute.make_layout_like(src)              # (4,4):(4,1), still row-major, padding removed
```

- Why it's useful: you often have a slice of a big global tensor with large, padded strides. You want a small buffer (in registers or shared memory) that holds the same data in the same order, without the gaps.
- `make_layout_like` gives you the layout for that buffer.

> [!NOTE]
> The Python API's behavior is not documented in detail. This description follows the C++ CuTe `make_ordered_layout(shape, stride)` behavior: preserve the relative stride order while making the layout compact.

For shape `(4, 3)`, these examples show the compact strides produced for
different input stride orders:

| Input strides | Compact strides | Faster mode |
| --- | --- | --- |
| `(1, 6)` | `(1, 4)` | Mode 0 |
| `(10, 6)` | `(3, 1)` | Mode 1 |
| `(7, 6)` | `(3, 1)` | Mode 1 |
| `(5, 6)` | `(1, 4)` | Mode 0 |
| `(6, 6)` | `(1, 1)` | Tie; the modes overlap |

For example:

```python
src = cute.make_layout((4, 3), stride=(10, 6))
dst = cute.make_layout_like(src)  # shape (4, 3), strides (3, 1)
```

The returned layout is compact and preserves the input's stride order when that order is unambiguous. It describes how data would be indexed; it does not copy or rearrange tensor data.

---

## `cute.make_composed_layout`

- Stacks transformations in a chain: `result(c) = inner(offset + outer(c))`.
- They run **right to left**:
  1. `outer` turns the coordinate into a normal offset.
  2. `offset` adds a fixed shift.
  3. `inner` transforms that number again.

```python
outer = cute.make_layout((4,4), stride=(4,1))   # normal row-major
sw    = cute.make_swizzle(2, 0, 2)              # scrambles some bits of the offset
L     = cute.make_composed_layout(sw, 0, outer) # row-major first, then scramble
```

- The main real use is **swizzling shared memory**:
  - Shared memory is split into 32 banks. If many threads hit the same bank at once, their accesses get serialized. This is called a **bank conflict**.
  - The normal layout (`outer`) gives the "obvious" offset.
  - The swizzle (`inner`) XORs a few bits of that offset so that neighbouring rows land in different banks.
  - You still index with normal coordinates. The scrambling happens automatically underneath.

> [!TIP]
> In short: `outer` decides **where** data would normally go, and `inner` **shuffles** it to avoid bank conflicts. You write normal code, and the hardware gets a conflict-free pattern.

---

---

## `cute.recast_layout`

- Same memory, **different element size**. You tell CuTe "these bytes used to be X-bit elements, now read them as Y-bit elements", and it gives you the layout in the new units.
- Argument order is **new bits first, then old bits**: `cute.recast_layout(new_bits, old_bits, layout)`.

### Intuition

Think of a row of 8 bytes:

```
as 8-bit  : [a][b][c][d][e][f][g][h]     -> 8 elements
as 16-bit : [ab ][cd ][ef ][gh ]         -> 4 elements
```

Nothing in memory moved. You're just counting it in bigger (or smaller) chunks.

- **Bigger type** (for example 8 → 16 bits, or upcast by 2):
  - The fastest dim (the one with stride 1) gets **2x fewer elements**.
  - Every other stride gets **2x smaller**, because each jump now covers the same bytes using fewer, bigger elements.
- **Smaller type** (for example 16 → 8 bits, or downcast by 2):
  - The fastest dim gets **2x more elements**.
  - Every other stride gets **2x bigger**.

### Examples

```python
L = cute.make_layout((8, 8), stride=(1, 8))   # 8x8 grid of 8-bit elements

cute.recast_layout(16, 8, L)   # (4,8):(1,4)    pairs of bytes become one 16-bit element
cute.recast_layout(8, 16, L)   # (16,8):(1,16)  each 16-bit element splits into two bytes
```

Check that it's the same memory. A column of the original is 8 bytes. After the upcast it's 4 elements × 2 bytes, which is still 8 bytes.

> [!NOTE]
> Only the **stride-1 mode** changes its shape. That's the dim where neighbouring elements sit right next to each other in memory, so those are the elements that get merged or split. The other modes keep their shape and only rescale their strides.

### Why you'd use it

- **Vectorized copies.** You can view an fp16 buffer as 128-bit chunks, so one instruction moves 8 halves at once.
- **Packed types.** fp8, int4, and fp4 data is often loaded as bigger words and then unpacked.
- In all of these, you want to work with the same data in different chunk sizes without rewriting the layout by hand.

> [!CAUTION]
> The upcast only works cleanly if the stride-1 dim is divisible by the ratio. For 8 → 16 bits that ratio is 2, so the stride-1 dim must be even. You can't merge half an element.

> [!TIP]
> - Common pattern: write `val_layout` in **bytes** (e.g. 16 bytes per thread = one 128-bit load), then call
> - `recast_layout(dtype.width, 8, val_layout)` to turn it into elements. The kernel then does vectorized
> - Loads for any dtype (fp32 → 4 elems, fp16 → 8, fp8 → 16) without hardcoding anything.

- [reference](https://github.com/NVIDIA/cutlass/blob/ee98dbdfd5727085ef86a018009215c2230904e3/examples/python/CuTeDSL/cute/notebooks/elementwise_add.ipynb)

```python
    coalesced_ldst_bytes = 16

    # Compile time validation: expect same element type for all input tensors
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0)) # 256 threads
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0)) # each thread handles 16 rows of 16 bytes
    val_layout = cute.recast_layout(dtype.width, 8, val_layout) # whatever is the element size, view it as elements instead of bytes
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
```

## Quick recap

| Function | Input → Output | One-line intuition |
|---|---|---|
| `cute.make_layout` | coord → offset | shape + stride = how to find each element in memory |
| `cute.make_identity_layout` | coord → same coord | tracks "where did this element come from" |
| `cute.make_ordered_layout` | shape + order → layout | pick which dim is fastest, strides get computed |
| `cute.make_layout_like` | layout → compact layout | same shape and order, gaps removed |
| `cute.make_composed_layout` | coord → outer → +offset → inner | normal layout, then a twist (usually a swizzle) |
| `cute.recast_layout` | new_bits, old_bits, layout → layout | same memory, different element size |
| `cute.crd2idx` | coord → offset | uses the strides |
| `cute.idx2crd` | colex index → coord | uses only the shape |
