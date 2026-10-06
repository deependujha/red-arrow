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

## Division / Tiling

Division is how CuTe **re-indexes a layout into tiles**.

```python
Z = cute.zipped_divide(A, T)
```

Think:

> **A = the whole space**  
> **T = what one tile looks like**  
> **divide = expose `(inside_tile, which_tile)` coordinates**

Nothing is moved in memory. The same elements are simply given a new coordinate system.

```text
A
│
│ divide by T
▼
(inside tile, which tile)
```

### The fundamental operation

```python
cute.logical_divide(A, T)
```

is essentially:

```python
composition(
    A,
    make_layout(T, complement(T, size(A)))
)
```

So the first mode represents the **tile**, while the complementary part represents the **rest / tile positions**.

You don't usually need to construct this manually.

---

### The variants

They all perform the same basic division; they mainly differ in **how the resulting modes are grouped**.

For a 2-D tensor and 2-D tiler:

```text
logical_divide
    ((TileM, RestM), (TileN, RestN))

zipped_divide
    ((TileM, TileN), (RestM, RestN))

tiled_divide
    ((TileM, TileN), RestM, RestN)

flat_divide
    (TileM, TileN, RestM, RestN)
```

Think of them as different **views of the same tiling**:

```text
logical → preserves the original modes
zipped  → groups all tile modes + all rest modes
tiled   → keeps tile together, rest modes separate
flat    → flattens everything
```

### The one to remember

**`zipped_divide` is the important practical one.**

It gives the clean mental model:

```text
((tile), (rest))
```

For example:

```python
Z = cute.zipped_divide(A, (128, 128))
```

means:

```text
Z[tile_element, tile_coordinate]
```

So you can naturally:

```text
select a tile
        ↓
select elements within that tile
        ↓
partition those elements across threads
```

This is why it appears so often in real kernels.

---

### `local_tile`

When you don't need the entire divided layout and simply want **one tile**, use:

```python
tile = cute.local_tile(A, (128, 128), coord)
```

Conceptually:

```text
zipped_divide(A, tiler)
        ↓
      choose
    one tile
        ↓
   local_tile(...)
```

This is commonly used for **CTA/block-level tiles**.

---

### Real-kernel mental model

When you see:

```python
zipped_divide(...)
```

think:

> **"Expose the tile coordinates."**

When you see:

```python
local_tile(...)
```

think:

> **"Give me this particular tile."**

When you see:

```python
local_partition(...)
```

think:

> **"Now distribute this tile across threads/warps."**

So a common kernel flow is:

```text
global tensor
      │
      ▼
  zipped_divide
      │
      ▼
   CTA tiles
      │
      ▼
  local_tile
      │
      ▼
 thread/warp partition
      │
      ▼
 registers / shared memory / MMA
```

The exact divide variant matters mainly when you need a particular **mode organization** for subsequent layout operations.

> **Don't memorize the four variants as four different algorithms. They're mostly different ways of arranging the same `(tile, rest)` decomposition.**

---

## Product

Think of **product as the mirror image of divide**.

> **Divide:** take a big layout → `(inside tile, which tile)`  
> **Product:** take a small layout → `(inside block, which copy)` → build the bigger layout

```text
Divide:
big space
   ↓
(tile, which tile)

Product:
small block
   ↓
(block, copies)
   ↓
bigger space
```

### The basic intuition

> [!INFO]Real-world example: apartment building
>
> Imagine an apartment building with:
> - 4 floors
> - 3 apartments per floor
> 
> You could identify an apartment using:
> - `(floor, apartment)`
>
> So:
> ```text
> Floor 0:  A0  A1  A2
> Floor 1:  A0  A1  A2
> Floor 2:  A0  A1  A2
> Floor 3:  A0  A1  A2
> ```
>
> There are 4 × 3 = 12 apartments.

Suppose `A` is a small block:

```text
A = one 2×2 block

0 1
2 3
```

and `B` describes **where/how many copies** of that block we want:

```text
B = 3 × 2 arrangement of copies
```

Then:

```python
logical_product(A, B)
```

means:

> **Take A and replicate it according to B.**

Conceptually:

```text
A       B
block × layout-of-copies
          ↓
      bigger layout
```

So remember:

> **A = what each copy looks like**  
> **B = how the copies are arranged**

This is why `complement` appears underneath the implementation: it gives the available positions where copies of `A` can be placed.

---

### Logical product vs the practical variants

`logical_product(A, B)` fundamentally produces:

```text
(block, which_copy)
```

Just like `logical_divide` produces:

```text
(tile, which_tile)
```

But for multidimensional layouts, `(block, copy)` isn't always the most useful way to look at the result.

The other product variants mainly **regroup those modes**.

### `blocked_product`

Think:

> **Keep each block together.**

```text
block 0: [A A A A]
block 1: [A A A A]
block 2: [A A A A]
```

The elements belonging to one copy stay together.

Useful when you want a **block/thread's values to be contiguous**.

---

### `raked_product`

Think:

> **Interleave the copies.**

Instead of:

```text
AAAA BBBB CCCC
```

you get something conceptually like:

```text
ABCABCABCABC
```

The copies take turns.

This is useful when you want **different threads/copies to take interleaved elements**, e.g. a cyclic/raked distribution.

---

### The other variants

Just like divide:

```text
logical_product
    → fundamental product

zipped_product
    → group the original modes together
      and the copy/tile modes together

tiled_product
    → keep the original block together,
      copy modes separate

flat_product
    → flatten the modes
```

You don't need to memorize the exact shapes yet.

The important distinction is:

```text
logical  → what is the fundamental result?
zipped / tiled / flat
         → how do I want that result organized?
```

---

## What to expect in real kernels

You probably won't spend much time manually constructing `logical_product()`.

You'll encounter the product family when **building a larger layout from a smaller layout/atom**.

A particularly important pattern is:

```text
small layout / atom
        ↓
product
        ↓
replicated across threads / tiles
        ↓
larger layout
```

For example, a small **shared-memory atom** can be replicated into a larger tensor layout.

`tile_to_shape(...)` is essentially a higher-level convenience for this kind of operation:

> **"Take this small layout/atom and grow it to this target shape."**

### Mental vocabulary

Keep these four words:

```text
composition → "view this through that"

complement  → "where can the other copies go?"

divide      → "big → inside tile + which tile"

product     → "small → inside block + which copy"
```

And for product:

```text
blocked → copies stay together

raked   → copies are interleaved
```

That's enough to recognize what's happening when you encounter product-related code in a real kernel.

---

## Inverses

A layout normally answers:

> **“Given a logical coordinate, where is it in memory?”**

```text
logical coordinate → layout → memory offset
```

An inverse reasons in the opposite direction:

```text
memory offset → inverse → logical coordinate
```

### Bijection

If a layout is a **bijection** — every coordinate maps to a unique offset and every offset in the target space is covered — the inverse is straightforward:

e.g.: `Layout = (x, y): (y, 1)`

```text
logical coordinate ↔ memory offset
```

### `left_inverse`

`left_inverse(L)` gives a layout that **undoes the mapping of `L`**.

Think:

```text
L:
coordinate → offset

left_inverse(L):
offset → coordinate
```

For offsets that `L` actually produces:

```text
L(2) = 4
left_inverse(L)(4) = 2
```

If `L` has holes, `left_inverse` is still itself a CuTe layout, so its behavior outside the offsets produced by `L` is determined by that layout rather than being a simple dictionary lookup.

### `right_inverse`

`right_inverse(L)` is **not simply `offset → coordinate`**.

It constructs an **inverse layout on the other side of the layout composition**.

The important distinction is:

```text
left_inverse  → think “undo L”
right_inverse → think “construct the inverse layout”

- original_layout(right_inverse_layout(x)) = x

```

Both return **layouts**, not ordinary lookup functions.

The left/right terminology comes from **which side of a layout composition the inverse operates on**, rather than simply meaning:

```text
left  = forward
right = backward
```

You generally don't need to derive the algebra by hand at first.

---

### Example

```python
import cutlass
import cutlass.cute as cute


@cute.jit
def foo():
    L = cute.make_layout((2, 2), stride=(1, 4))

    for i in cutlass.range_constexpr(4):
        print(f"{i=}: {L(i)=}")

    print("-" * 60)

    rl = cute.right_inverse(L)
    ll = cute.left_inverse(L)

    for offset in cutlass.range_constexpr(10):
        print(f"{offset=}; {ll(offset)=}; {rl(offset)=}")


if __name__ == "__main__":
    foo()
```

Output:

```text
i=0: L(i)=0
i=1: L(i)=1
i=2: L(i)=4
i=3: L(i)=5
------------------------------------------------------------
offset=0; ll(offset)=0; rl(offset)=0
offset=1; ll(offset)=1; rl(offset)=1
offset=2; ll(offset)=2; rl(offset)=2
offset=3; ll(offset)=3; rl(offset)=3
offset=4; ll(offset)=2; rl(offset)=4
offset=5; ll(offset)=3; rl(offset)=5
offset=6; ll(offset)=4; rl(offset)=6
offset=7; ll(offset)=5; rl(offset)=7
offset=8; ll(offset)=4; rl(offset)=8
offset=9; ll(offset)=5; rl(offset)=9
```

The original layout is:

```text
L = (2, 2):(1, 4)
```

so:

```text
coordinate    offset

    0    →      0
    1    →      1
    2    →      4
    3    →      5
```

Notice the holes:

```text
memory:

0  1  2  3  4  5
A  B  .  .  C  D
```

For the offsets actually produced by `L`:

```text
L(0) = 0  → left_inverse(0) = 0
L(1) = 1  → left_inverse(1) = 1
L(2) = 4  → left_inverse(4) = 2
L(3) = 5  → left_inverse(5) = 3
```

So `left_inverse` behaves like the intuitive **“undo the layout”** operation.

The important observation from the experiment is that:

```text
right_inverse(L)(4) = 4
right_inverse(L)(5) = 5
```

Therefore, **do not interpret `right_inverse(L)` as simply `offset → logical coordinate`**. It is an inverse **layout**, whose meaning becomes clear when considering layout composition rather than treating it as a lookup table.

---

### Why care in real kernels?

Inverses become useful when CuTe needs to reason about how layouts map onto each other — particularly for **copy operations, layout compatibility, and vectorization**.

For example, CuTe may need to determine whether two layouts allow consecutive elements to be moved together:

```text
contiguous:

[A B C D E F G H]
 ^^^^^^^^
 consecutive → potentially vectorizable
```

versus:

```text
[A . B . C . D]
```

where the accesses are not contiguous.

This is related to APIs such as:

```python
cute.max_common_vector(...)
```

which helps determine the largest common vectorizable access between layouts.

### When you'll encounter it

You probably won't manually call:

```python
cute.left_inverse(...)
cute.right_inverse(...)
```

very often.

You're more likely to encounter inverse-related logic indirectly when working with:

- `copy` / `cute.copy`
- `CopyAtom`
- copy layouts
- vectorized global/shared-memory transfers
- layout compatibility
- alignment
- `max_common_vector`

So the practical mental model is:

```text
Layout:
    logical coordinate → memory

left_inverse:
    memory offset → coordinate
    “undo the layout”

right_inverse:
    construct the inverse layout
    used in layout composition
```

> **You mainly care about inverses when CuTe needs to reason about the relationship between a layout and its reverse mapping — especially for copies, layout compatibility, and vectorization.**

---

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
