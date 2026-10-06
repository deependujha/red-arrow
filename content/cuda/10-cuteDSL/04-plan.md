---
title: Plan `Kernel building primitives`
type: docs
math: true
sidebar:
  open: false
weight: 1004
---

> **Recognize the common CuTe building blocks in real kernels, understand what they are doing intuitively, and be able to follow unfamiliar code.**

I would **not** jump directly into TMA/WGMMA/etc. First get comfortable with the small set of concepts that appear everywhere.

## The next chapter I'd do

### Phase 2.5 — CuTe kernel-building primitives

Learn these **in this order**:

#### 1. Memory access patterns ⭐⭐⭐

Before atoms, understand what you're actually trying to achieve.

Be comfortable looking at:

```text
thread 0 → A[0,0], A[0,1], A[0,2], ...
thread 1 → A[1,0], A[1,1], A[1,2], ...
```

and recognizing:

- contiguous access
- strided access
- coalesced vs non-coalesced access
- per-thread access
- per-warp access
- tiled access
- vectorized access

You don't need to study CUDA memory architecture deeply here.

Just be able to look at a layout and say:

> "Ah, these threads are loading adjacent elements."

or:

> "These threads are jumping by 8, so this is strided."

This becomes the foundation for understanding **TV layouts and copy atoms**.

---

#### 2. Thread-Value (TV) layouts ⭐⭐⭐

This is probably the **most important thing to learn next**.

Understand:

```text
(thread, value) → tensor coordinate
```

Think:

> **Which elements does each thread own?**

For example:

```text
          values
       0   1   2   3
T0     A   B   C   D
T1     E   F   G   H
T2     I   J   K   L
...
```

You should become comfortable with:

```python
cute.make_layout_tv(...)
cute.make_layout_tv(...)
```

and especially:

```python
layout_tv(thread, value)
```

Don't study the algebra behind TV layouts further.

Just understand:

> **TV layout = assignment of tensor elements to threads and values.**

---

#### 3. `local_partition` ⭐⭐⭐

Once TV layouts make sense, learn:

```python
cute.local_partition(...)
```

Mental model:

> **Take a big tensor and give each thread the pieces assigned to it.**

So:

```text
tensor
   ↓
TV layout
   ↓
local_partition
   ↓
thread 0 → its elements
thread 1 → its elements
thread 2 → its elements
...
```

This is where TV layouts stop being abstract and become useful.

You should be able to look at:

```python
tAg = cute.local_partition(...)
```

and immediately think:

> "This is the portion of tensor `A` that belongs to the current thread."

---

#### 4. `local_tile` ⭐⭐⭐

You've already seen this conceptually.

Now just make it second nature:

```python
cute.local_tile(...)
```

Mental model:

> **Select the tile belonging to this CTA/thread/warp coordinate.**

Think:

```text
global tensor
      ↓
    tiles
      ↓
   tile[cta_id]
      ↓
local tile
```

Don't revisit `zipped_divide` mathematically.

Just remember:

```text
local_tile = "give me this tile"
local_partition = "give me my portion of this tile"
```

That's enough.

---

#### 5. Tensor slices / fragments ⭐⭐⭐

Now learn what CuTe means when you see things like:

```python
tAg
tAs
tCrA
tCrB
tCgA
```

and slicing:

```python
tAg(_, 0)
tAg(0, _)
```

Mental model:

> **A fragment is simply the portion of a tensor that some execution unit currently owns.**

For example:

```text
Global A
   ↓
CTA tile
   ↓
Warp tile
   ↓
Thread fragment
   ↓
Registers
```

This is extremely important for reading CUTLASS code.

Don't worry about formally defining "fragment."

Just recognize:

```text
fragment = my local piece of the tensor
```

---

# Then: Atoms

Only **after** the above should you learn atoms.

This is where I think your current instinct is exactly right.

### 6. What is an Atom? ⭐⭐⭐

The simplest useful definition:

> **An atom is one small, predefined operation + its data-access pattern.**

Examples:

```text
CopyAtom
MMA Atom
TMA Atom
```

Think of an atom as a **hardware/instruction-shaped building block**.

For example:

```text
CopyAtom
    ↓
"How does this particular copy operation move data?"

MMA Atom
    ↓
"How does this particular matrix-multiply instruction consume/produce data?"

TMA Atom
    ↓
"How does this TMA transfer move a multidimensional tensor?"
```

Don't study every atom yet.

Just understand:

```text
Atom = primitive operation
Tile = how we repeat/use the operation
Layout = how data is assigned
```

That's the key relationship.

---

# 7. Copy atoms ⭐⭐⭐

This should be your **first actual atom**.

Learn:

```python
cute.make_copy_atom(...)
```

and understand:

```text
CopyAtom
    +
TiledCopy
    +
partitioning
    =
actual data movement
```

The mental model:

```text
CopyAtom
"What does ONE copy operation look like?"

TiledCopy
"How do I repeat that operation across threads?"
```

For example:

```text
         CopyAtom
            ↓
       one thread's copy
            ↓
      ┌─────────────┐
      │ TiledCopy   │
      └─────────────┘
       ↓ ↓ ↓ ↓ ↓ ↓
     threads cooperate
       ↓ ↓ ↓ ↓ ↓ ↓
        tensor tile
```

---

# 8. `make_tiled_copy` ⭐⭐⭐

This is where CuTe starts looking like real kernel code.

Learn:

```python
cute.make_tiled_copy(...)
```

and the basic flow:

```text
CopyAtom
    ↓
TiledCopy
    ↓
partition_S(...)
partition_D(...)
    ↓
each thread's source/destination fragment
    ↓
cute.copy(...)
```

You should eventually be able to see:

```python
copy_tiled = cute.make_tiled_copy(...)
thr_copy = copy_tiled.get_slice(tid)
tS = thr_copy.partition_S(...)
tD = thr_copy.partition_D(...)
cute.copy(...)
```

and understand the **story** without understanding every layout expression.

That's the milestone I want you to aim for.

---

# 9. Copy partitioning ⭐⭐⭐

Understand:

```python
get_slice(tid)
partition_S(...)
partition_D(...)
```

Mental model:

```text
TiledCopy
    ↓
"How should the whole copy be distributed?"

get_slice(tid)
    ↓
"What is thread tid responsible for?"

partition_S
    ↓
"What source elements does this thread read?"

partition_D
    ↓
"What destination elements does this thread write?"
```

This is one of the most useful things for reading existing CuTe code.

---

# 10. Vectorized copies ⭐⭐

Now connect this to your earlier memory-access intuition.

Understand:

```text
scalar copy
    ↓
vectorized copy
```

and why CuTe cares about:

- alignment
- contiguous elements
- vector width
- `max_common_vector`

You don't need to study the implementation.

Just recognize:

> **CuTe wants each thread to move multiple adjacent elements when the layout/hardware allows it.**

---

# 11. Shared-memory layouts + swizzle ⭐⭐⭐

Now you're ready for shared memory.

Learn:

```text
global memory
      ↓
shared memory
```

and understand:

> **Shared memory isn't just an array — its layout can be deliberately designed to avoid bank conflicts.**

Then:

```python
cute.make_swizzle(...)
```

Mental model:

> **Swizzle = rearrange the mapping of logical elements to shared-memory addresses so accesses behave better.**

You already understand composition, so you don't need to go back and study composition again.

Just recognize:

```text
logical tensor
      ↓
swizzled layout
      ↓
shared memory
```

---

# 12. Bank conflicts ⭐⭐⭐

Only learn enough to answer:

> "Why is this shared-memory layout swizzled?"

You should understand:

```text
32 banks
    ↓
threads access addresses
    ↓
multiple threads hit same bank
    ↓
bank conflict
```

and:

```text
swizzle
    ↓
change address mapping
    ↓
reduce conflicts
```

That's enough for now.

---

# Then stop.

Seriously.

At this point you should **not** jump into TMA/WGMMA/tcgen05.

Instead, take **real CUTLASS/CuTe kernels** and read them.

---

# Your actual learning path

I'd make your next chapter exactly this:

```text
Memory access patterns
        ↓
TV layouts
        ↓
local_tile
        ↓
local_partition
        ↓
fragments / tensor slices
        ↓
Atoms
        ↓
CopyAtom
        ↓
TiledCopy
        ↓
copy partitioning
        ↓
vectorized copies
        ↓
shared-memory layouts
        ↓
swizzle
        ↓
bank conflicts
```

And then:

```text
                  ┌──────────────┐
                  │  REAL KERNEL │
                  └──────┬───────┘
                         ↓
                  global tensor
                         ↓
                     local_tile
                         ↓
                  thread partition
                         ↓
                    TV layout
                         ↓
                     fragment
                         ↓
                    CopyAtom
                         ↓
                    TiledCopy
                         ↓
                 shared memory
                         ↓
                    swizzle
```

If you can follow that pipeline, **you'll be able to read a huge amount of CuTe code without understanding every algebraic detail.**

---

## And only after that

Then I'd do the hardware atoms separately:

```text
CopyAtom
   ↓
MMA Atom
   ↓
Tensor Core layouts
   ↓
TiledMMA
   ↓
MMA partitioning
```

Then later:

```text
cp.async
   ↓
TMA
   ↓
WGMMA
   ↓
tcgen05
   ↓
cluster / remote shared memory
```

Those are **different chapters**. Don't mix them into your current one.

### Your immediate goal

Don't aim for:

> "I understand CuTe."

Aim for:

> **"When I open a CuTe kernel, I know what every major object is trying to represent."**

If you see:

```python
cta_tiler
thr_mma
tCrA
tCrB
tCgC
copy_atom
tiled_copy
local_tile
local_partition
```

you should be able to say:

```text
cta_tiler       → which CTA tile?
local_tile      → give me that tile
TV layout       → who owns which elements?
local_partition → give this thread its elements
fragment        → my local piece
CopyAtom        → one copy primitive
TiledCopy       → distribute copies across threads
tCrA/tCrB       → my register fragments for MMA
```

**That is the level I'd target now.** Once you reach that, stop studying fundamentals and start learning by reading kernels. That's where your CuTe knowledge will compound much faster.

Later learn about `warp-specializations`, `pipelining` & more advanced concepts & patterns.
