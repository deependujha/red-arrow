---
title: L1 & L2 Cache
type: docs
math: true
prev: docs/
weight: 50
sidebar:
  open: false
---


> **You don't directly "use" L1/L2 like shared memory. You arrange your memory accesses so the hardware naturally gets useful cache behavior.**

With shared memory, you explicitly say:

```cpp
__shared__ float tile[...];
```

With L1/L2, you mostly say:

```cpp
x = input[idx];
```

and then **access pattern + reuse + cache policy** determine whether the request hits L1, L2, or HBM.

---

# 1. The hierarchy you already know

Think:

```text
                 SM
                  │
             ┌────┴────┐
             │ Registers│
             └────┬────┘
                  │
              L1 / SMEM
                  │
                  ▼
                 L2
                  │
                  ▼
              HBM / VRAM
```

The rough latency/capacity relationship is:

```text
register   → tiny, extremely fast
L1         → small, very fast
shared     → small, very fast, explicitly managed
L2         → larger, slower
HBM        → huge, much slower
```

But there's an important distinction:

### Registers / shared memory

**You explicitly control allocation and usage.**

### L1 / L2

**Hardware-controlled caches.**

Your job is to create access patterns that make caching effective.

---

# 2. What does "using L1" actually mean?

Suppose:

```cpp
for (int i = 0; i < N; i++)
    y[i] = x[i] * 2;
```

Every element of `x` is loaded once.

There's basically no reuse:

```text
x[0] → load → use once
x[1] → load → use once
x[2] → load → use once
...
```

L1 can't magically help much.

You are fundamentally doing:

```text
HBM → L2 → L1 → register
```

for each element.

This is **memory bandwidth bound**.

---

# 3. Now introduce reuse

Imagine:

```cpp
for (int i = 0; i < N; i++) {
    y[i] = x[i] + x[i + 1];
}
```

Now:

```text
x[1]
```

is used by both:

```text
y[0]
y[1]
```

So the hardware can potentially do:

```text
first access:

HBM → L2 → L1 → register

second access:

L1 → register
```

That second access is an **L1 hit**.

You didn't explicitly tell the GPU:

> "Put x[1] in L1."

The hardware figured it out.

---

# 4. L2 becomes especially interesting with multiple SMs

Suppose:

```text
SM0 → reads x[0:1024]
SM1 → reads x[1024:2048]
SM2 → reads x[2048:3072]
```

Those accesses don't necessarily have much reuse between SMs.

But consider:

```text
SM0 ──┐
SM1 ──┤
SM2 ──┼──> same data
SM3 ──┘
```

L1 is generally local to an SM, so:

```text
SM0 → L1
SM1 → L1
SM2 → L1
```

would be separate caches.

But they can share the **L2 cache**:

```text
             L2
          /   |   \
        SM0  SM1  SM2
```

So:

```text
SM0 → HBM → L2
SM1 → L2 hit
SM2 → L2 hit
```

This is one reason L2 can be extremely valuable.

---

# 5. The biggest thing to understand: locality

There are two types.

### Spatial locality

You access nearby addresses.

```text
x[0]
x[1]
x[2]
x[3]
...
```

This is excellent for GPUs.

A memory transaction fetches a cache line containing multiple nearby bytes.

So if a warp does:

```text
lane 0 → x[0]
lane 1 → x[1]
lane 2 → x[2]
...
lane 31 → x[31]
```

you've got excellent spatial locality.

---

### Temporal locality

You reuse the same data.

```text
x[0]
x[1]
x[2]

... computation ...

x[0]
x[1]
x[2]
```

Now caching can help significantly.

---

# 6. This is where your Triton knowledge becomes useful

Consider:

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

x = tl.load(x_ptr + offs)
y = tl.load(y_ptr + offs)

tl.store(out_ptr + offs, x + y)
```

Your current vector-add kernel has:

```text
x → read once
y → read once
out → write once
```

There is almost **zero temporal reuse**.

So trying to "optimize L1" here is mostly pointless.

Your main concern is:

```text
Are my global memory accesses coalesced?
```

And they are.

This is why vector addition is a great example of a **bandwidth-bound kernel**.

---

# 7. Where caches become extremely important

Consider matrix multiplication.

Naive:

```text
C[i,j] = Σ A[i,k] * B[k,j]
```

Suppose many threads need the same:

```text
A[i,k]
```

or:

```text
B[k,j]
```

Now you have massive reuse.

You have two choices:

### Let cache handle it

```text
HBM
 ↓
L2
 ↓
L1
 ↓
register
```

### Explicitly stage it

```text
HBM
 ↓
L2
 ↓
shared memory
 ↓
register
```

The second approach is what traditional CUDA GEMM kernels do heavily.

---

# 8. Why not just rely on L1/L2?

Because caches are **not deterministic storage that you control**.

Suppose L1 has limited capacity.

You load:

```text
A
B
C
D
E
F
G
H
```

and the cache gets filled.

Your previously useful `A` might get evicted.

Shared memory gives you:

> "I want this exact tile to stay here until I'm done."

That's much more powerful.

So:

```text
Cache:
    hardware-managed

Shared:
    programmer/compiler-managed
```

This distinction is huge for performance engineering.

---

# 9. How you actually optimize cache behavior

There are roughly four knobs.

### ① Increase spatial locality

Bad:

```text
x[0]
x[1024]
x[2048]
x[3072]
```

Good:

```text
x[0]
x[1]
x[2]
x[3]
```

For GPU kernels, this also connects directly to **coalescing**.

---

### ② Increase temporal reuse

Bad:

```text
load A
use A
discard
```

Better:

```text
load A

use A
use A
use A
use A

discard
```

You get more value from every memory transaction.

---

### ③ Reduce working-set size

Suppose your kernel touches:

```text
1 MB
```

and L2 can easily keep the relevant portion hot.

Great.

If you're streaming through:

```text
10 GB
```

with no reuse, cache doesn't save you.

---

### ④ Structure work so reuse happens close together

This is subtle and extremely important.

Suppose:

```text
SM0 accesses A
...
10 million cycles later
...
SM0 accesses A again
```

Even if there is temporal reuse conceptually, the data might have been evicted.

But:

```text
load A
use A
use A
use A
```

is much more cache-friendly.

This is why **tiling** works.

---

# 10. A really useful hierarchy for performance work

When you see a memory-bound kernel, ask these questions in order:

```text
1. Are accesses coalesced?
        ↓
2. Is there spatial locality?
        ↓
3. Is there temporal reuse?
        ↓
4. Can reuse happen within registers?
        ↓
5. Can reuse happen in shared memory?
        ↓
6. Is L1 helping?
        ↓
7. Is L2 helping?
        ↓
8. Am I ultimately limited by HBM bandwidth?
```

You generally shouldn't start with:

> "How do I maximize L1 utilization?"

Start with:

> **"What data is reused, by whom, and when?"**

Then decide which level of the hierarchy should capture that reuse.

---

# 11. One very important GPU-specific concept

You will often see metrics like:

```text
L1/TEX hit rate
L2 hit rate
dram throughput
```

Don't blindly optimize for:

```text
L1 hit rate = 100%
```

That's not necessarily good.

Example:

```text
Kernel A:
L1 hit rate = 90%
HBM bandwidth = 200 GB/s

Kernel B:
L1 hit rate = 40%
HBM bandwidth = 800 GB/s
```

Kernel B might be dramatically faster.

**Cache hit rate is a diagnostic, not an optimization target.**

What you ultimately care about is:

```text
execution time
throughput
latency
```

---

# 12. And this connects directly to Nsight Compute

Since you're learning NCU, start looking at:

```text
Memory Workload Analysis
```

and metrics around:

```text
L1/TEX
L2
DRAM
Global Load/Store
```

A particularly useful experiment is to write three kernels:

```text
Kernel A:
    read array once

Kernel B:
    read same array twice

Kernel C:
    read same array many times
```

Then profile them.

You'll start seeing the transition:

```text
             A             B             C

HBM       ███████       █████         ██
L2        ███████       ██████        ███
L1        █             █████         ███████
compute   █             ██            ███████
```

The exact behavior depends on working-set size and access pattern, but you'll develop the intuition you're currently missing.

---

## The mental model I'd use

Don't think:

> **"How do I utilize L1/L2?"**

Think:

> **"Where does reuse occur in my algorithm?"**

Then:

```text
same thread, immediate reuse
        ↓
registers

different threads, same block
        ↓
shared memory

nearby/recent accesses on same SM
        ↓
L1 may capture them

reuse across SMs
        ↓
L2 may capture them

no reuse / streaming
        ↓
HBM bandwidth
```

That's the mental model that will serve you much better when you start optimizing Triton/CUDA kernels seriously.
