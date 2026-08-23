---
title: L1 & L2 Cache
type: docs
math: true
prev: docs/
weight: 50
sidebar:
  open: false
---

> [!NOTE] The one-line version
> You don't "use" L1/L2 the way you use shared memory. You arrange **access patterns and reuse** so the hardware naturally produces useful cache behavior — and only then reach for the few explicit knobs that do exist.

With shared memory, you say exactly what lives where:

```cpp
__shared__ float tile[BLOCK_M][BLOCK_K];
```

With L1/L2, you mostly just say:

```cpp
x = input[idx];
```

and **access pattern + reuse distance + cache policy** decide whether that request is served by L1, L2, or HBM.

---

## 1. The hierarchy

```text
              SM
               │
        ┌──────┴──────┐
        │  Registers  │   per-thread, compiler-allocated
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  L1 / SMEM  │   per-SM, one physical SRAM, split two ways
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │     L2      │   device-wide, shared by all SMs
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  HBM / VRAM │
        └─────────────┘
```

Rough ordering (numbers are order-of-magnitude, not spec):

| Level | Scope | Latency | Capacity | Managed by |
|---|---|---|---|---|
| Register | thread | ~1 cycle | 255 regs/thread max | compiler |
| Shared | thread block | ~20–30 cycles | tens of KB / SM | **you** |
| L1 | SM | ~30 cycles | tens of KB / SM | hardware |
| L2 | whole GPU | ~200 cycles | several MB–tens of MB | hardware (+ a persistence hint) |
| HBM | whole GPU | ~400–800 cycles | GBs | — |

The distinction that matters:

- **Registers / shared memory** — you control allocation and lifetime explicitly.
- **L1 / L2** — hardware-managed. Your job is to create access patterns that make caching effective.

### L1 and shared memory are the same SRAM

Since Volta, L1 and shared memory are one unified block per SM, split at runtime. Asking for more shared memory leaves less L1, and vice versa:

| Arch | Unified L1+SMEM per SM | Max shared per block |
|---|---|---|
| Volta (sm_70) | 128 KB | 96 KB |
| Turing (sm_75) | 96 KB | 64 KB |
| Ampere A100 (sm_80) | 192 KB | 164 KB |
| Ampere GA10x / Ada (sm_86/89) | 128 KB | 100 KB |
| Hopper (sm_90) | 256 KB | 227 KB |

The split can be nudged with `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, pct)`, and dynamic shared memory beyond the default opt-in limit requires `cudaFuncAttributeMaxDynamicSharedMemorySize`.

### L1 is not coherent, L2 is

L1 is private to an SM and **not** coherent with other SMs' L1s. L2 is the device-wide coherence point for global memory — every SM's misses funnel through it. This is why cross-block communication has to go through global memory (i.e. L2), and why an `atomicAdd` on a hot address serializes at an L2 slice.

---

## 2. Granularity: lines and sectors

Caches never move single floats. A cache line is **128 B**, made of four **32 B sectors**, and traffic between L2 and DRAM is counted in sectors.

So the real question behind "is my access coalesced?" is:

```text
Of the 32 B I dragged in, how many bytes did I actually use?
```

A warp of 32 lanes reading consecutive `float`s touches 128 B = 4 sectors, all fully used. The same warp reading with stride 32 touches 32 separate lines — 32× the traffic for the same useful bytes.

---

## 3. What "using L1" actually means

Take a pure streaming kernel:

```cpp
for (int i = 0; i < N; i++)
    y[i] = x[i] * 2;
```

Every element of `x` is loaded once and used once:

```text
x[0] → load → use once
x[1] → load → use once
x[2] → load → use once
```

There is no reuse for a cache to capture, so every element pays the full path:

```text
HBM → L2 → L1 → register
```

This is a **bandwidth-bound** kernel. Cache tuning cannot help it; only reducing bytes moved can.

---

## 4. Introducing reuse

```cpp
y[i] = x[i] + x[i + 1];
```

`x[1]` feeds both `y[0]` and `y[1]`, so the second access can be served without another trip to DRAM:

```text
first access:   HBM → L2 → L1 → register
second access:  L1 → register
```

You never told the GPU to put `x[1]` in L1 — the hardware did it.

> [!WARNING] Careful with this example on a GPU
> On a CPU this is textbook temporal locality. On a GPU, `x[i]` and `x[i+1]` are read by *adjacent lanes of the same warp in the same instruction*, so the "reuse" is mostly absorbed by the **same 128 B line being fetched once**. That's spatial locality inside a single transaction, not a hit on a later instruction. Real temporal reuse means coming back to the same line **later in time** — which is the case tiling creates deliberately.

---

## 5. L2 and multiple SMs

If blocks partition the data, there is little to share:

```text
SM0 → reads x[0:1024]
SM1 → reads x[1024:2048]
SM2 → reads x[2048:3072]
```

But when many SMs read the *same* data — a weight matrix, a lookup table, a broadcast tensor — L1 can't help them, because each SM has its own:

```text
SM0 → its own L1
SM1 → its own L1
SM2 → its own L1
```

They do, however, share L2:

```text
             L2
          /   |   \
        SM0  SM1  SM2
```

so the cost is paid once:

```text
SM0 → miss → HBM → L2
SM1 → L2 hit
SM2 → L2 hit
```

This is where a large L2 earns its area: V100 had 6 MB, A100 40 MB, H100 50 MB, and consumer Ada parts up to 72–96 MB. On the bigger chips, an entire small model's weights or a full activation tile can stay resident across the whole grid.

---

## 6. Locality, the two kinds

### Spatial locality — nearby addresses

```text
lane 0 → x[0]
lane 1 → x[1]
...
lane 31 → x[31]
```

Excellent: one instruction, four sectors, everything used. On GPUs this is the same property as **coalescing**.

### Temporal locality — same address, again, soon

```text
x[0], x[1], x[2]
   ... computation ...
x[0], x[1], x[2]
```

Caching helps here *only if the reuse distance is short enough* that the line survives eviction.

---

## 7. Why vector add can't be cache-tuned

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

x = tl.load(x_ptr + offs)
y = tl.load(y_ptr + offs)

tl.store(out_ptr + offs, x + y)
```

The traffic profile is:

```text
x   → read once
y   → read once
out → written once
```

**Zero temporal reuse.** Chasing L1 hit rate here is wasted effort; the only question worth asking is whether the accesses are coalesced — and they are. Arithmetic intensity is ~1 flop per 12 bytes, so the kernel is pinned to HBM bandwidth by construction. That's what makes it the canonical bandwidth-bound example.

---

## 8. Where caches start to matter: GEMM

```text
C[i,j] = Σ_k A[i,k] * B[k,j]
```

Every element of `A[i,k]` is needed by every thread computing row `i`; every `B[k,j]` by every thread in column `j`. That's O(N) reuse per element — enormous. Two ways to capture it:

**Let the cache handle it**

```text
HBM → L2 → L1 → register
```

**Stage it explicitly**

```text
HBM → L2 → shared memory → register
```

Real GEMM kernels do the second, because reuse this valuable is too important to leave to an eviction policy.

---

## 9. Why not just rely on L1/L2?

Because a cache is **not storage you control**. With limited capacity, loading

```text
A B C D E F G H
```

may evict the `A` you were about to reuse. Shared memory instead lets you say:

> "This exact tile stays here until I'm done with it."

```text
Cache:   hardware-managed, best-effort, capacity-and-conflict evictable
Shared:  programmer-managed, guaranteed resident for the block's lifetime
```

That guarantee is the whole point. It converts a *probabilistic* hit rate into a *deterministic* one — which is what lets you actually reason about a kernel's memory traffic.

---

## 10. The knobs that do exist

Cache behavior is mostly implicit, but not entirely.

### ① Spatial locality / coalescing

Bad → good:

```text
x[0], x[1024], x[2048], x[3072]     ⟶     x[0], x[1], x[2], x[3]
```

Fix layout (AoS → SoA), transpose on the way in, or pad to kill conflicts.

### ② Temporal reuse

```text
load A → use once → discard        ⟶     load A → use A four times → discard
```

More value extracted per memory transaction. This is arithmetic intensity going up.

### ③ Smaller working set

If the hot set fits in L2 (a few MB), it stays hot. If you stream 10 GB with no reuse, no cache saves you — reduce bytes instead (fp16/bf16/fp8, fusion, recompute).

### ④ Reuse close together in time

```text
SM0 touches A
   ... 10M cycles ...
SM0 touches A again        ← conceptually reuse, practically a miss
```

versus

```text
load A → use, use, use     ← survives in cache
```

Restructuring so reuse happens *nearby in time* is exactly what **tiling** does — and why launch order / swizzling of block IDs measurably changes L2 hit rate in GEMM.

### ⑤ L2 persistence window (CC 8.0+)

You *can* tell L2 to favor a region — useful for data re-read by every block:

```cpp
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = ptr;
attr.accessPolicyWindow.num_bytes = bytes;   // ≤ cudaDeviceProp::accessPolicyMaxWindowSize
attr.accessPolicyWindow.hitRatio  = 0.6f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

Reset it with `cudaCtxResetPersistingL2Cache()` when done, or you'll penalize the next kernel.

### ⑥ Per-instruction cache hints

| PTX | Meaning | Use for |
|---|---|---|
| `ld.global.ca` | cache at all levels (default) | normal reuse |
| `ld.global.cg` | cache in L2, bypass L1 | data other SMs also read |
| `ld.global.cs` | streaming, mark evict-first | one-shot reads |
| `ld.global.lu` | last use, don't retain | final read of a value |
| `ld.global.nc` / `__ldg` | read-only path | data not written by the kernel |

In Triton these are exposed directly:

```python
x = tl.load(x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
```

Use `evict_last` for the tile you keep coming back to, `evict_first` for the one you stream past once.

---

## 11. A diagnostic order for memory-bound kernels

```text
1. Are accesses coalesced?              (bytes requested vs bytes fetched)
        ↓
2. Is there spatial locality?
        ↓
3. Is there temporal reuse?             ← "what data, by whom, when?"
        ↓
4. Can that reuse live in registers?
        ↓
5. Can it live in shared memory?
        ↓
6. Is L1 capturing it?
        ↓
7. Is L2 capturing it?
        ↓
8. Am I simply at HBM bandwidth?        ← if yes, stop tuning, reduce bytes
```

Never open with *"how do I maximize L1 utilization?"* Open with:

> **"What data is reused, by whom, and when?"**

Then decide which level of the hierarchy should capture that reuse.

---

## 12. Hit rate is a diagnostic, not a target

Profilers report things like:

```text
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

A 100% L1 hit rate is not automatically good:

```text
Kernel A:  L1 hit rate = 90%,  DRAM throughput = 200 GB/s
Kernel B:  L1 hit rate = 40%,  DRAM throughput = 800 GB/s
```

Kernel B may be dramatically faster — it's saturating the machine, while A may be latency-bound with too little work in flight. High hit rate can even mean you're re-reading data you should have kept in registers.

**What you actually care about is execution time, achieved throughput, and where the roofline says you sit.**

---

## 13. Seeing it in Nsight Compute

Start in the **Memory Workload Analysis** section and follow the flow diagram: Global Load/Store → L1/TEX → L2 → DRAM. The numbers on each arrow are the request/sector counts, so you can see exactly where traffic is amplified.

A good calibration experiment — three kernels over the same array:

```text
Kernel A: read the array once
Kernel B: read it twice
Kernel C: read it many times
```

Profile all three and watch the traffic migrate up the hierarchy:

```text
             A             B             C

HBM       ███████       █████         ██
L2        ███████       ██████        ███
L1        █             █████         ███████
compute   █             ██            ███████
```

Exact behavior depends on working-set size versus L2 capacity and on access pattern — which is precisely the intuition the experiment is meant to build.

---

## The mental model

Don't ask *"how do I utilize L1/L2?"* Ask *"where does reuse occur in my algorithm?"* — then map it:

```text
same thread, immediate reuse            →  registers
different threads, same block           →  shared memory
nearby/recent accesses on the same SM   →  L1 may capture it
reuse across SMs                        →  L2 may capture it
no reuse / pure streaming               →  HBM bandwidth, and that's the ceiling
```

Registers and shared memory are decisions. L1 and L2 are consequences.
