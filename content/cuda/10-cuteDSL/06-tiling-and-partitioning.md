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

