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
> Tile the identity tensor the same way (`local_tile(cC, tiler, coord)`) and check `elem_less(coord, A.shape)` before each load/store.
