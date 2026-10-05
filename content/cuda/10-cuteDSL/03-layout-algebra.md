---
title: CuTe layout algebra
type: docs
math: true
sidebar:
  open: false
weight: 1003
---

The goal of this page: when you see something like `((2,4),(4,2)):((8,1),(16,4))`, you should be able to *read it out loud* and know what it means. Formulas are secondary.

Companion code lives in `~/cl-algebra/`:

| file | what it's for |
|---|---|
| `videos/S0*.mp4` | 7 short animations, one per idea below (re-render with `./render.sh`) |
| `explore.py` | every example on this page, run through the real CuTe DSL, printed as grids. Predict first, then run. |
| `layout.py` | the whole algebra in ~250 lines of plain Python. Read this when something feels like magic. |
| `verify.py` | checks `layout.py` against CuTe on ~1500 cases (all agree) |
| `gpu_partition.py` | divide on the L4: CTAs and threads partition a matrix, checked against torch |

---

## Composition

```python
R = cute.composition(A, B)  # roughly: R(c) = A(B(c))
```

![cute composition](/10-cutedsl/cute-composition.png)

Think:

> **B picks, A places.**

`B` produces coordinates into `A`; `A` maps those coordinates to memory.

```text
c → B → coordinate in A → A → memory
```

So `composition(A, B)` means **"A viewed through B."**

- **B determines how you index the result.**
- **A provides the underlying memory mapping.**

```python
A = 12:2
B = 4:3

R = cute.composition(A, B)
# R = 4:6
```

### With tilers

Composition can apply a tiler to an existing layout:

```python
A = (4, 6):(6, 1)

composition(A, (2, 3))
# → (2, 3):(6, 1)

composition(A, (2:2, 3:2))
# → (2, 3):(12, 2)
```

A plain integer `n` means `n:1` — take the first `n` positions.

This is the basic idea behind how higher-level operations such as **divide/tiling** manipulate layouts.

### Nested modes

Composition can produce nested modes when the coordinates from `B` cross a boundary in `A`.

```text
A = (6,2):(8,2)
B = (4,3):(3,1)

R = ((2,2),3):((24,2),8)
```

The nested `(2,2)` still represents **4 positions**. The nesting is just CuTe's way of representing a mapping that cannot be expressed with one constant stride.

![cute multimode](/10-cutedsl/cute-multimode.png)

> **Nested mode ≠ different logical size.** It is a structured representation of the same coordinate space.

### In real kernels

You will usually encounter composition **indirectly through higher-level APIs** such as `tiling/division & partitioning` rather than writing `cute.composition()` yourself.

Explicit composition is most useful when you already have mappings that need to be combined/reinterpreted — **swizzles are an important example**, especially for shared-memory layouts.

---

## Complement

```python id="y4s0q1"
C = cute.complement(A, n)
```

Think:

> **A describes one copy; C describes where the other copies go.**

`C` gives the offsets needed to shift copies of `A` so that together they cover the target space `[0, n)` exactly once.

```text id="l4c0qw"
        A = one tile
             │
             ▼
        ┌─────────┐
        │  copy   │
        └─────────┘
             │
        C gives the
        copy offsets
             │
             ▼
┌─────────┬─────────┬─────────┐
│ copy 0  │ copy 1  │ copy 2  │ ...
└─────────┴─────────┴─────────┘
```

The important relationship is:

```text id="5mkrqf"
A(c) + C(c*)
```

Together, `A` and `C` uniquely cover the target offsets.

So mentally:

> **A = which element inside a copy**  
> **C = which copy / where that copy starts**

For example:

```python
complement(4:1, 24) => 6:4
```

`A` describes 4 contiguous elements, and the complement says the copies start every 4 elements:

```text id="x6q8ve"
A:       [0 1 2 3]
C:       0 4 8 12 16 20
```
> if every element of A is added with every element of C, you get the full target space `[0, n)`

### A useful example

```python id="d3g7fq"
A = (2,2):(1,6)
C = cute.complement(A, 24)
# C = (3,2):(2,12)
```

Here `A` describes the footprint:

```text id="9m0qfa"
{0, 1, 6, 7}
```

and `C` describes how to shift that footprint to cover the larger target.

![Complement visualization](/10-cutedsl/cute-complement.png)


### In real kernels

You won't usually call `complement()` directly when writing a kernel.

It's more useful as a **fundamental building block behind tiling/division operations**.

When you see it, think:

> **"What coordinate space is missing from this layout so that together they cover the target?"**

This is especially useful for understanding how CuTe derives **tile coordinates, copy positions, and partitioned tensor layouts**.

---

## 3. Divide: re-name A as *(inside tile, which tile)*

```
logical_divide(A, T) = composition(A, make_layout(T, complement(T, size(A))))
                                                  ^         ^
                                    inside one tile    which tile
```

Composition intuition: we build a lens **B = (T, complement(T))** and view A through it. T says which coordinates form tile #0, and complement(T) says where all the other tiles start. The result has two modes:

- **mode 0: position inside a tile** (shape of T)
- **mode 1: which tile** (shape of the complement)

Nothing moves in memory. It's the same elements, re-indexed as `(elem, tile)`.

```
logical_divide(24:1, 4:1) = (4,6):(1,4)
      0  4  8 12 16 20        column t = tile t: plain chunks of 4
      1  5  9 13 17 21
      ...

logical_divide(24:1, 4:2) = (4,(2,3)):(2,(1,8))
      0  1  8  9 16 17        tile = 4 elements, 2 apart
      2  3 10 11 18 19        tiles start at +1, then +8 (complement of 4:2)
      4  5 12 13 20 21
      6  7 14 15 22 23
```

Read `(4,(2,3)):(2,(1,8))` out loud: "*4 per tile, step 2 · 6 tiles: next tile +1, and after two of those, +8*".

> [!IMPORTANT]
> **The tiler acts on A's *coordinates*, not on memory.** That's why it's called *logical* divide. `gpu_partition.py` partitions a row-major and a column-major tensor with the same tiler and gets **identical owners** for every `(i,j)`; only the strides inside the result differ. Think "cut the index space"; memory just comes along.

### 2D: tuple tilers + the four flavours

A tuple tiler divides each mode separately. The flavours are **the same function**; they differ only in how the resulting modes are grouped:

```
A = (8,8):(8,1), tiler (4,4)

logical_divide  ((4,2),(4,2)):((8,32),(1,4))     each dim split in place: (rows: in-tile, which-tile), (cols: ...)
zipped_divide   ((4,4),(2,2)):((8,1),(32,4))     ((tile modes), (rest modes))   <- the one you'll use
tiled_divide    ((4,4),2,2):((8,1),32,4)         (tile, rest flattened)
flat_divide     (4,4,2,2):(8,1,32,4)             everything flat
```

**Reading zipped_divide**: mode 0 `(4,4):(8,1)` = inside a tile you move **exactly like in A** (A's own strides). Mode 1 `(2,2):(32,4)` = jumping tiles: `32 = 4 rows × 8` and `4 = 4 cols × 1`, i.e. *tile extent × A's stride*. You can predict these strides without running anything.

### The payoff: slicing a zipped divide

`Z = zipped_divide(A, tile)` has shape `((tile), (rest))`. Fixing one of the two modes gives the two most common partition patterns in every kernel:

| slice | gives | used for |
|---|---|---|
| `Z[((None,None), t)]` | all of tile `t` | **a CTA grabs its block** |
| `Z[(e, None)]` | element `e` of *every* tile (strided) | **a thread grabs its share** when "tile" = the thread layout |

```
Z = zipped_divide((8,8):(8,1), (2,4)) = ((2,4),(4,2)):((8,1),(16,4))
Z[((None,None), 5)]  = (2,4):(8,1)      + offset 20   -> rows 2-3, cols 4-7
Z[(3, None)]         = (4,2):(16,4)     + offset 9    -> one element per tile
```

Two things that bite:
- `Z[(None, t)]` keeps mode 0 as **one nested mode** `((2,4))`, which is rank 1. Spell it `((None,None), t)` if you want a rank-2 tile you can divide again (the kernel in `gpu_partition.py` needs this).
- Slicing a *layout* returns a sub-layout **plus an offset** (`cute.slice_and_offset`). On a *tensor*, the offset is folded into the pointer, so you never see it.

`gpu_partition.py` does exactly this on the GPU: `zipped_divide(mA, (8,16))[((None,None),(bx,by))]` per CTA, then inside the tile either:
- `zipped_divide(tile, (4,8))[(tid, None)]`: "strided": thread `t` gets position `t` of every 4×8 sub-tile, or
- `zipped_divide(tile, (2,2))[(None, tid)]`: "chunked": thread `t` gets the whole `t`-th 2×2 sub-tile.

Same divide, other mode fixed.

> [!NOTE]
> Sizes that don't divide evenly: `cute.ceil_div(shape, tiler)` = the rest-mode shape, and the last tile hangs off the edge. Mask it with an identity tensor (predication).

🎬 `S04_LogicalDivide1D.mp4`, `S05_ZippedDivide2D.mp4`

---

## 4. Product: the mirror image of divide

Divide takes a **big** thing and names it *(tile, which tile)*. Product takes a **small** thing and *builds* the big one: *(block, which copy)*.

```
logical_product(A, B) = make_layout(A, composition(complement(A, size(A)*cosize(B)), B))
                                    ^              ^                                   ^
                               the block    all the places a copy of A could start     arranged in B's pattern
```

Step by step with `A = (2,2):(1,2)` (a 2×2 block) and `B = (3,2):(1,3)` (a 3×2 grid of copies):

1. `complement(A, 4*6=24) = 6:4`: copy #n starts at `4n`. The complement automatically skips over A's own footprint.
2. `composition(6:4, B) = (3,2):(4,12)`: lay those 6 copies out as B's 3×2 grid.
3. `P = ((2,2),(3,2)):((1,2),(4,12))`: mode 0 = inside block, mode 1 = which copy.

Why `size(A)*cosize(B)`? B's outputs number the copies `0..cosize(B)-1`, and each copy occupies `size(A)` slots.

### blocked vs raked: same product, regrouped per dimension

`logical_product` gives `(block, copies)`. For a 2D picture, you want *row* and *column* modes instead. Pair the row part of the block with the row part of the copies, and so on. **The order inside each pair is the only difference:**

```
blocked_product(A,B) = ((2,3),(2,2)):((1,4),(2,12))     row = (inside block, which block)
     0  2 12 14
     1  3 13 15          each copy's 4 values (4n..4n+3) sit together as a 2x2 block
     4  6 16 18
     5  7 17 19
     ...

raked_product(A,B)   = ((3,2),(2,2)):((4,1),(12,2))     row = (which block, inside block)
     0 12  2 14
     4 16  6 18          elements dealt out like cards:
     8 20 10 22          neighbours belong to DIFFERENT copies
     1 13  3 15
     ...
```

How to read them: in `(2,3)` the *fast* sub-mode comes first. In blocked, the fast part is "inside block", so a block's elements are adjacent. In raked, the fast part is "which block", so you cycle through copies first.

Intuition for kernels: "copies" are usually **threads**. *Blocked* = each thread's values are contiguous. *Raked* = threads take turns, so adjacent elements belong to adjacent threads (coalesced).

`tile_to_shape(atom, shape)` is "blocked_product until it fills `shape`". It's how a small swizzled smem atom grows into a full tile.

🎬 `S06_Product.mp4`

---

## 5. Inverses: from memory slot back to coordinate

`L` answers "*coordinate k → which slot?*". An inverse answers "*slot i → which coordinate?*". For a bijection (a permutation) there's only one answer:

```
L = (2,4):(4,1)                  memory order: k0 k2 k4 k6 k1 k3 k5 k7
right_inverse(L) = (4,2):(2,1)   i = a + 4b  ->  k = 2a + b
```

When L has **holes**, there are two different questions:

```
L = (2,2):(1,4)                  hits slots {0,1,4,5} of 8

left_inverse(L)  = (4,2):(1,2)   Li(L(k)) == k for every k: undoes L on the slots L touches.
                                 On slots L never touches (2,3,6,7) its output is junk.
right_inverse(L) = 2:1           L(Ri(i)) == i, but only for the contiguous run 0,1,2,... L covers.
                                 Here: slots 0,1 -> size 2.
```

**`size(right_inverse(L))` = how many consecutive elements, starting at 0, L lays out contiguously = the widest vector you can load.** That's the idea behind `max_common_vector(src, dst)` (roughly: `coalesce(composition(src, right_inverse(dst)))`, then its leading stride-1 run). It decides whether a copy is 128-bit or scalar.

🎬 `S07_Inverses.mp4`

---

## 6. Everything at once: `make_layout_tv`

Verified in the DSL:

```python
thr = (4,8):(8,1)      # 32 threads
val = (2,2):(2,1)      # 4 values each
tiler, tv = cute.make_layout_tv(thr, val)                 # (8,16),  ((8,4),(2,2)):((16,2),(8,1))

mn = cute.raked_product(thr, val)                          # (m,n) -> (thread,value) id, interleaved
tv == cute.composition(cute.right_inverse(mn), cute.make_layout((32, 4)))   # same layout
```

Read it as a sentence: **product** spreads threads over the tile in a raked way, mapping `(m,n) → (thread, value)`. The **right inverse** flips that into `(thread, value) → (m,n)`, which is what a thread needs ("which element is my v-th value?"). **Composition** with the layout `make_layout((32,4))` just reshapes that 1D inverse into a `(thread, value)` 2-mode layout.

---

## 7. Reading checklist (any output)

1. **Split at the top-level modes.** For a divide/product: mode 0 = *inside*, mode 1 = *which*.
2. **For each mode: size, then stride.** "s positions, each +d."
3. **Nested mode?** Read sub-modes fast→slow: "first `s0` steps of `d0`, then `s1` steps of `d1`." It appeared because steps crossed a boundary or because two things were zipped together.
4. **Size-1 modes** have meaningless strides (CuTe often prints them as 0).
5. **Sanity check** strides: inside-tile strides = the original's; between-tile strides = tile extent × original stride.
6. Still confused? `python explore.py <section>` prints it as a grid.

## 8. Cheat sheet

| want to… | use | read the result as |
|---|---|---|
| index into a layout | `L(c)`, `cute.crd2idx(c, L)` | |
| linear index → coordinate | `cute.idx2crd(k, shape)` | column-first unfold |
| view A through B / take a sub-block | `cute.composition(A, B)` | shape of B, strides from A |
| other copies of a footprint | `cute.complement(A, n)` | offsets of the copies |
| split into tiles | `cute.zipped_divide(L, tiler)` | `((inside tile), (which tile))` |
| a CTA's tile / a thread's share | `Z[((None,None), t)]` / `Z[(e, None)]` | fix *which* / fix *inside* |
| number of tiles | `cute.ceil_div(shape, tiler)` | = rest-mode shape |
| repeat a block | `blocked_product` / `raked_product` | `(inside, which)` per dim / `(which, inside)` per dim |
| grow an atom to a shape | `cute.tile_to_shape(atom, shape, order)` | |
| slot → coordinate | `right_inverse` / `left_inverse` | contiguous prefix / undo-on-image |
| widest legal vector copy | `cute.max_common_vector(a, b)` | |
| same function, simpler | `cute.coalesce(L)` | |
