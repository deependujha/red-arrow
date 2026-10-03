---
title: Shared Memory and Warp-level Performance
type: docs
math: true
sidebar:
  open: false
weight: 1003
---

## 0. Mental model (read this first)

```
GMEM ──(LDG / cp.async / TMA)──► SMEM ──(LDS / ldmatrix)──► RMEM ──► compute
           alignment matters          bank conflicts matter
           (vector width)             (swizzle fixes these)
```

- **Alignment** decides *how wide* each load can be: 32, 64, or 128 bits.
- **Swizzle** decides *whether a warp's SMEM access serializes*.
- **Layouts** decide *which thread touches which address*. Everything above is just a property of the layout plus the pointer.

In CuTe DSL, nearly every perf property is something the **compiler must be able to prove** from static layout info plus pointer alignment. If it can't prove it, it silently falls back to scalar or narrow accesses.

---

## 1. Shared memory basics

### Hardware facts to memorize
| Fact | Value |
|---|---|
| Banks | 32 |
| Bank width | 4 bytes |
| One full "bank sweep" | 32 × 4 B = **128 B** |
| Bank of byte address `a` | `(a // 4) % 32` |
| 128-bit (16 B) accesses | warp served in **4 phases of 8 threads**; conflicts only counted *within* a phase |
| 64-bit (8 B) accesses | 2 phases of 16 threads |
| Static smem limit | 48 KB per block |
| Dynamic smem max | ~227 KB (H100), ~163 KB (A100), ~99 KB (consumer Ampere/Ada) |

**Bank conflict:** threads in the same phase hit the same bank at *different* words, so the accesses serialize. Hitting the same word is a broadcast and costs nothing.

**Rule of thumb for 16 B vector access:** within each group of 8 threads, the 8 addresses must land in 8 *different* 16 B slots of a 128 B span.

### Allocating SMEM in CuTe DSL

All DSL smem is carved out of **dynamic shared memory**. You allocate inside the kernel with `SmemAllocator` and pass the total size at launch.

```python
import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator   # or cutlass.utils.SmemAllocator

@cute.kernel
def kernel(gA: cute.Tensor):
    smem = SmemAllocator()

    # Option A: tensor directly from a layout
    sA_layout = cute.make_layout((64, 64), stride=(64, 1))      # row-major
    sA = smem.allocate_tensor(
        element_type=cutlass.Float16,
        layout=sA_layout,
        byte_alignment=128,      # see §4
        swizzle=None,            # see §3
    )

    # Option B: raw 1D array, e.g. reduction scratch
    red = smem.allocate_array(cutlass.Float32, num_elems=32)
```

### Option C: struct (preferred for real kernels)

The struct is the single source of truth for size, and that size is what you pass to launch.

```python
@cute.struct
class SharedStorage:
    sA:   cute.struct.Align[cute.struct.MemRange[cutlass.Float16, 64 * 64], 1024]
    sB:   cute.struct.Align[cute.struct.MemRange[cutlass.Float16, 64 * 64], 1024]
    mbar: cute.struct.MemRange[cutlass.Int64, 2 * STAGES]   # mbarriers (TMA pipelines)

@cute.kernel
def kernel(...):
    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = storage.sA.get_tensor(sA_layout)            # optionally swizzle=...
    mbar_ptr = storage.mbar.data_ptr()
```

**Size helpers:**
```python
cute.cosize(layout)                      # #elements the layout's codomain spans → allocation size
cute.size_in_bytes(cutlass.Float16, layout)
SharedStorage.size_in_bytes()
```

> ⚠ For a **swizzled or composed layout**, allocate `cosize(layout)` elements, not `size(layout)`. They differ when strides have gaps or padding.

---

## 2. Dynamic shared memory

### Launch side

```python
@cute.jit
def host(mA: cute.Tensor, stream):
    smem_bytes = SharedStorage.size_in_bytes()
    kernel(mA).launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
        smem=smem_bytes,        # dynamic smem in BYTES
        stream=stream,
    )
```

### Things to remember
- Forgetting `smem=`, or passing too small a value, gives garbage or illegal-address errors, not a compile error.
- Above 48 KB, CUDA C++ needs `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`. **⚠ verify** whether your DSL version sets this automatically (recent ones do).
- Lower-level escape hatch: `cute.arch.get_dyn_smem(dtype, alignment=...)` and `cute.arch.get_dyn_smem_size()` give the raw dynamic smem pointer and size. **⚠ verify**
- **Occupancy math:** blocks/SM ≤ `smem_per_SM // smem_per_block`. Doubling stages doubles smem, which can halve occupancy. Decide deliberately.

### Padding vs swizzle (interview question)
| | Padding (`stride=(65,1)`) | Swizzle |
|---|---|---|
| Wastes smem | yes (1 elem/row) | no |
| Breaks 16 B alignment of rows | often yes | no |
| Works with TMA / ldmatrix / wgmma | no / awkward | yes, the native path |
| Simplicity | trivial | needs layout understanding |

Use padding for quick scalar kernels. Use swizzle for anything feeding tensor cores or vector loads.

---

## 3. Swizzle

### 3.1 The problem

- SMEM = **32 banks × 4 B**. Bank of a byte address = `(addr // 4) % 32`.
- A warp whose threads hit 32 different banks finishes in one step. Different words in the **same** bank serialize, and that's a **bank conflict**. The same word is a broadcast, which is free.
- **fp32 32×32 row-major example:** element `(r, c)` is at offset `r*32 + c`, so its bank is just `c`. Reading a column means every thread hits the same bank: a **32-way conflict**.
- **fp16 8×64 example (128 B rows):** every row starts at bank 0. Reading one 16 B vector from each of 8 rows (what `ldmatrix` / MMA tile loads do) hits the same 4 banks 8 times: an **8-way conflict**.

**Fix:** change where elements are stored so a column spreads across banks, while rows still spread well.

### 3.2 What `Swizzle<B, M, S>` does

It's a function on an integer offset.

| Param | Name | Meaning |
|---|---|---|
| **B** | BBits | how many bits in the XOR mask, giving 2^B slots to rotate between |
| **M** | MBase | how many low bits to leave alone, so a vector of 2^M elements stays whole |
| **S** | SShift | how far down to shift the mask before XORing |

**Rule:** take B bits starting at bit `M+S`, shift them right by S, and XOR them into the B bits starting at bit M.

```
offset bits:   ... [ yyy ] [ ... S bits ... ] [ zzz ] [ M low bits ]
                       |                         ^
                       +------ shift right S ----+  XOR
```

```python
def swizzle(off, b, m, s):
    mask = ((1 << b) - 1) << (m + max(0, s))
    bits = off & mask
    return off ^ (bits >> s if s >= 0 else bits << -s)   # negative S shifts left (CuTe allows it)

def bank(off, elem_bytes):
    return (off * elem_bytes // 4) % 32
```

**Why it's safe:**
- Only bits `[M, M+B)` change, and only by XORing with other bits of the same offset. So it's a **permutation**, and no two elements collide.
- Every result stays inside the same `2^(M+B+|S|)`-aligned block, so nothing leaves the allocation.
- The one way to break it is a tile whose size isn't a multiple of that block: results can land past the end. Example: a 5×1 tile with `<2,0,2>` maps offset 4 → 5, which is past the end.
- The usual choice is `S ≥ B`, so the source and target bits don't overlap.

### 3.3 Picking B, M, S

**Inputs:**
- `E` = bytes per element
- `vector_bits` = one thread's load width (usually 128)
- `X` = elements in the contiguous (fast) dim, a power of 2

```python
N = (vector_bits // 8) // E     # elements per vector
M = log2(N)                     # keep each vector whole
B = log2(128 // E) - M          # 128 here = BYTES (32 banks × 4 B), not the 128-bit vector!
S = log2(X) - M                 # row length measured in vectors
```

**Worked example: fp16, 128-bit vectors, X = 64**
1. **N:** 128 bits = 16 B per load, and 16 / 2 = **8 elements**.
2. **M = log2(8) = 3.** The 8 elements of a vector are the low 3 offset bits. Touching them tears the vector apart, so it can't be loaded in one 128-bit instruction.
3. **B = log2(64) − 3 = 3.** 128 B (one full bank sweep) = 64 fp16 = 6 offset bits. The low 3 are inside a vector, so the remaining 3 pick which **16 B slot** you're in. That gives 8 slots, each on a different group of 4 banks.
4. **S = log2(64) − 3 = 3.** The row number starts at bit 6, and the slot bits are `[3, 6)`. Shifting row bits `[6, 9)` down by 3 lines them up with the slot bits, so the row index becomes the slot rotation.
5. **Result: `Swizzle<3,3,3>`.** Row `r` stores its vectors rotated by `r` slots: `physical_slot = slot XOR (row % 8)`.

```
logical slot:          physical slot:
row0: 0 1 2 3 4 5 6 7     0 1 2 3 4 5 6 7
row1: 0 1 2 3 4 5 6 7     1 0 3 2 5 4 7 6
row2: 0 1 2 3 4 5 6 7     2 3 0 1 6 7 4 5
row3: 0 1 2 3 4 5 6 7     3 2 1 0 7 6 5 4
```
- **Column read:** slot 0 across rows 0–7 lands in physical slots 0–7. Together those cover all 32 banks, so it's conflict-free.
- **Row read:** still conflict-free, because XOR with a fixed value just shuffles slots within the row.

**In one line:** keep vectors whole (**M**), rotate over every slot in a 128 B span (**B**), and key the rotation off the row (**S**).

### 3.4 Cheat table

| Case | N | M | B | S | Swizzle |
|---|---|---|---|---|---|
| fp32, scalar, X=32 | 1 | 0 | 5 | 5 | `<5,0,5>` |
| fp32, 128-bit, X=32 | 4 | 2 | 3 | 3 | `<3,2,3>` |
| fp16, 128-bit, X=64 | 8 | 3 | 3 | 3 | `<3,3,3>` |
| int8, 128-bit, X=128 | 16 | 4 | 3 | 3 | `<3,4,3>` |
| fp16, 128-bit, X=128 (256 B row) | 8 | 3 | 3 | **4** | `<3,3,4>` |

**Why CUTLASS always uses S=3:** it builds a swizzle **atom exactly 128 B wide** (e.g. fp16 8×64), then `tile_to_shape`s it over the full tile. Inside the atom, X is always 128 B, so S = 3. There are two equivalent options:
- **Swizzle the whole tile** with one function: S = log2(X) − M, giving `<3,3,4>` for a 256 B row.
- **Swizzle a 128 B atom and tile it:** always `<3,M,3>`. Prefer this, since it matches what TMA and wgmma expect.

Applying `<3,3,3>` directly to a 256 B row is a bug. Rows 0 and 4 get the same rotation, which gives a 2-way conflict.

**Rows narrower than 128 B:** the formula gives S < B, so the source and target bits overlap. That still works, but CUTLASS shrinks B instead:
- **Byte domain:** SW128 = `<3,4,3>`, SW64 = `<2,4,3>`, SW32 = `<1,4,3>`.
- **Element domain:** subtract `log2(E)` from M (e.g. fp16 SW64 = `<2,3,3>`).

### 3.5 CuTe DSL code

```python
import cutlass, cutlass.cute as cute

@cute.jit
def demo():
    layout   = cute.make_layout((8, 64), stride=(64, 1))     # fp16, row-major, 128 B rows
    swizzle  = cute.make_swizzle(3, 3, 3)                     # Swizzle<3,3,3>
    swizzled = cute.make_composed_layout(swizzle, 0, layout)  # swizzle(layout(coord))
    cute.printf("{}", swizzled(2, 5))                         # swizzled offset of (2, 5)
    # swizzled.inner = swizzle, swizzled.outer = layout
    # ⚠ verify: some docs show the offset arg as (0, 0)

# Real kernel: 128 B atom → tile to full shape → allocate
atom      = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0,
                                      cute.make_layout((8, 64), stride=(64, 1)))
sA_layout = cute.tile_to_shape(atom, (128, 64), order=(0, 1))
sA        = smem.allocate_tensor(cutlass.Float16, sA_layout, byte_alignment=1024)
```

For wgmma / tcgen05 / TMA, use the arch helpers instead of hand-rolling, so the swizzle mode matches what the hardware expects. Candidates are `make_smem_layout_a` and `warpgroup.make_smem_layout_atom` (**⚠ verify** names).

### 3.6 Gotchas
- **Writer and reader must use the same swizzled layout.** Otherwise the result is wrong, not just slow.
- **Atom needs ≥ 2^B rows** (8 for B=3), or the pattern is cut off.
- **B too small:** conflicts come back, because the slots must cover all 32 banks.
- **M too small:** vectors get torn apart, so 128-bit loads become scalar.
- **TMA alignment:** SW128 needs a 1024 B-aligned smem base, SW64 512 B, SW32 256 B.
- **Element vs byte domain is the #1 bug.** `<3,4,3>` means SW128 in bytes, but int8 128 B in elements.
  - **⚠ verify** the units of the `swizzle=` kwarg on `allocate_tensor` / `get_tensor`.
  - The composed-layout path is in elements, so it's unambiguous.
- **Profile:** in `ncu`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (and `_st`) should be ≈ 0.

---

## 4. Alignment: `assumed_align` and `byte_alignment`

Two knobs, one idea: **tell the compiler what alignment it can rely on**, so it can emit wide loads and stores.

| Knob | Where | Applies to |
|---|---|---|
| `assumed_align` | pointer / tensor **creation** (host → kernel, or `make_ptr`) | **GMEM** pointers mostly |
| `byte_alignment` | `SmemAllocator.allocate_tensor/array` | **SMEM** placement |
| `cute.struct.Align[T, N]` | smem struct fields | SMEM placement |

### `assumed_align`
```python
from cutlass.cute.runtime import from_dlpack

a_ = from_dlpack(torch_tensor, assumed_align=16)          # promise: base ptr is 16 B aligned
a_ = a_.mark_layout_dynamic(leading_dim=1)                 # dynamic shape, stride-1 dim declared

ptr = cute.make_ptr(cutlass.Float16, raw_int_ptr,
                    cute.AddressSpace.gmem, assumed_align=16)
```

- **Default is about sizeof(element)**, so 2 B for fp16. The compiler then emits 16-bit loads even if your copy atom asks for 128 bits, or it errors on the atom/alignment mismatch.
- **Slices aren't aligned.** torch allocations are 256 B-aligned at the base, but slices like `x[:, 1:]` are not. If you promise 16 and lie, you get misaligned-address faults or silent corruption.
- **Tile offsets need alignment too, not just the base.** This holds if the stride-1 extent per tile × sizeof(T) is a multiple of 16, and the leading stride is too. Use `mark_compact_shape_dynamic(mode=..., divisibility=N)` to promise divisibility of dynamic shapes.

### `byte_alignment`
```python
sA = smem.allocate_tensor(cutlass.Float16, layout, byte_alignment=128)
```
| Use case | Required alignment |
|---|---|
| 16 B vector LDS/STS, cp.async 16 B | 16 |
| TMA destination (no swizzle) | 128 |
| TMA / wgmma with SW32 / SW64 / SW128 | 256 / 512 / **1024** |
| mbarrier | 8 |

Default to **1024 for any tensor-core operand buffer**. It costs at most ~1 KB of padding.

---

## 5. Vectorized access: the CuTe DSL "float4"

There's no `float4` type. You get 128-bit LDG/STS/LDS when **all three** of these hold:

1. **Contiguity:** the per-thread value layout has ≥ 16 B contiguous along a stride-1 mode, *statically known*.
2. **Alignment:** pointer alignment ≥ 16 (`assumed_align` / `byte_alignment`).
3. **Atom width:** the copy atom asks for 128 bits, or autovec decides it can use them.

### 5.1 Simple elementwise: tile, then load as TensorSSA
```python
@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # gX were zipped_divide'd on host with tiler (1, 8) -> 8 fp16 = 16 B per thread
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    t = bidx * bdim + tidx

    m, n = gA.shape[1]             # mode 1 = tile coords after zipped_divide
    mi, ni = t // n, t % n

    a = gA[(None, (mi, ni))].load()   # TensorSSA of 8 elems -> one 128-bit load
    b = gB[(None, (mi, ni))].load()
    gC[(None, (mi, ni))].store(a + b)

@cute.jit
def add(mA, mB, mC):
    tiler = (1, 8)
    gA = cute.zipped_divide(mA, tiler)   # ((1,8), (M, N/8))
    gB = cute.zipped_divide(mB, tiler)
    gC = cute.zipped_divide(mC, tiler)
    ...launch...
```

### 5.2 TiledCopy: the general, reusable pattern
```python
# 128-bit copy atom
atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16,
                           num_bits_per_copy=128)

thr_layout = cute.make_layout((32, 4), stride=(4, 1))   # 128 threads
val_layout = cute.make_layout((1, 8))                    # 8 fp16 per thread = 16 B (contiguous!)
tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)

# in kernel
thr_copy = tiled_copy.get_slice(tidx)
tAgA = thr_copy.partition_S(gA_tile)     # this thread's view of the source
tAsA = thr_copy.partition_D(sA)          # this thread's view of the destination
cute.copy(tiled_copy, tAgA, tAsA)
```

**Async GMEM→SMEM (Ampere cp.async):**
```python
atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
    cutlass.Float16, num_bits_per_copy=128)
...
cute.copy(tiled_copy, tAgA, tAsA)
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)     # wait until ≤0 groups in flight
cute.arch.sync_threads()             # make smem visible to other threads
```

### 5.3 Register fragments
```python
frag = cute.make_fragment_like(tAgA)     # rmem tensor with matching shape (⚠ newer: make_rmem_tensor)
cute.autovec_copy(tAgA, frag)            # compiler picks widest legal vector
x = frag.load().to(cutlass.Float32)      # TensorSSA math
```

### 5.4 Verify you actually got 128-bit
- **Dump PTX / SASS:** `CUTE_DSL_KEEP_PTX=1`, `CUTE_DSL_KEEP_CUBIN=1`, `CUTE_DSL_LINEINFO=1` (**⚠ verify** env var names).
- **What to look for:**
  - PTX: `ld.global.v4.b32` / `ld.global.v4.f32`
  - SASS: `LDG.E.128`, `STS.128`, `LDS.128`, `LDGSTS.E.BYPASS.128`
- **If you see `.v2` or scalar loads, check:**
  1. `assumed_align`
  2. whether the val_layout is contiguous along stride-1
  3. whether the shape is dynamic, which hides divisibility

### 5.5 Coalescing reminder
- Vector width is per thread, but coalescing is across the warp.
- Ideally consecutive lanes take consecutive 16 B chunks: 32 lanes × 16 B = 512 B = 4 full 128 B transactions.
- So make `thr_layout`'s fastest-varying mode walk the stride-1 GMEM dim.

---

## 6. Warp-level primitives

### 6.1 Indices
```python
tidx, _, _ = cute.arch.thread_idx()
lane  = cute.arch.lane_idx()                                # tidx % 32
warp  = cute.arch.make_warp_uniform(cute.arch.warp_idx())   # tells compiler it's uniform → scalar regs
```
`make_warp_uniform` matters because branches on it become uniform branches. That means no divergence bookkeeping, and values stay in uniform registers.

### 6.2 Shuffles
```python
cute.arch.shuffle_sync(val, offset=src_lane)        # read from absolute lane
cute.arch.shuffle_sync_bfly(val, offset=k)          # read from lane ^ k   (butterfly / xor)
cute.arch.shuffle_sync_down(val, offset=k)          # read from lane + k
cute.arch.shuffle_sync_up(val, offset=k)            # read from lane - k
# optional: mask=0xFFFFFFFF (default full warp), mask_and_clamp for sub-warp groups
```

### 6.3 Warp reduction (the one you'll write 100 times)
```python
@cute.jit
def warp_reduce_sum(val):
    # butterfly: every lane ends with the full sum
    for i in cutlass.range_constexpr(5):            # unrolled at compile time
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << (4 - i))  # 16,8,4,2,1
    return val

@cute.jit
def warp_reduce_max(val):
    for i in cutlass.range_constexpr(5):
        val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1 << (4 - i)))
    return val
```
- **Butterfly (`bfly`) gives all lanes the result.** That's what you want for softmax and LayerNorm. With `down`, only lane 0 has it.
- **Sub-warp groups** (e.g. 8 threads per row): loop `log2(8) = 3` times with offsets 4, 2, 1.
- **`range_constexpr` for loops that must unroll.** Plain `range` or `cutlass.range` gives a runtime loop.

### 6.4 Block reduction (warp → smem → warp)
```python
@cute.jit
def block_reduce_sum(val, red_smem, num_warps: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    val = warp_reduce_sum(val)
    if lane == 0:
        red_smem[warp] = val
    cute.arch.sync_threads()

    val = red_smem[lane] if lane < num_warps else 0.0   # ⚠ may need cutlass.select_/where for dynamic cond
    val = warp_reduce_sum(val)                           # every warp redundantly reduces; all get result
    return val
```

### 6.5 Votes / predicates
```python
cute.arch.vote_ballot_sync(pred)     # 32-bit mask of lanes where pred is true
cute.arch.vote_any_sync(pred)
cute.arch.vote_all_sync(pred)

with cute.arch.elect_one():          # exactly one thread in the warp executes (e.g. issue TMA / mbarrier arrive)
    ...
```

### 6.6 Barriers and fences
```python
cute.arch.sync_threads()                                    # __syncthreads
cute.arch.sync_warp()                                       # __syncwarp
cute.arch.barrier(barrier_id=1, number_of_threads=256)      # named barrier (warp specialization)
cute.arch.fence_proxy(...)                                  # generic ↔ async proxy (before TMA store reads smem you wrote) ⚠ verify signature
```

**When you need `sync_threads` around smem:**
- **RAW:** after writing smem, before *other* threads read it.
- **WAR:** before overwriting a smem buffer others may still be reading. This is the one people forget in multi-stage loops.

---

## 7. DSL control-flow traps

| Want | Write |
|---|---|
| Compile-time constant arg | `x: cutlass.Constexpr` |
| Compile-time `if` (branch pruned) | `if cutlass.const_expr(cond):` |
| Unrolled loop | `for i in cutlass.range_constexpr(N):` |
| Runtime loop | `for i in cutlass.range(n):` (or `range(n)` with dynamic `n`) |
| Runtime `if` on dynamic value | plain `if` (lowered to `scf.if`) |

- **Python-side values exist only at trace time.** Lists and dicts can't be mutated per-thread at runtime.
- **Define before branching.** A variable assigned inside a runtime `if`/`for` and used after it must be defined before the block, with a matching type.
- **Static shapes enable vectorization.** Python int shapes give the compiler divisibility and contiguity info. Dynamic shapes hide it unless you declare `divisibility`.

---

## 8. Pre-flight checklist when authoring a kernel

```
[ ] GMEM tensors: assumed_align=16 (or more) — and is that actually true for my inputs?
[ ] Dynamic dims: divisibility declared so tiles stay 16 B aligned?
[ ] Per-thread value layout: ≥16 B contiguous along stride-1?
[ ] Thread layout: consecutive lanes → consecutive 16 B chunks (coalesced)?
[ ] SMEM: allocated cosize(layout), byte_alignment matches use (16/128/1024)?
[ ] launch(smem=...) == storage.size_in_bytes()?
[ ] Swizzle: <3, log2(16/E), 3> on a 128 B-wide atom, tiled to full shape?
[ ] Same swizzled layout used by writer AND reader?
[ ] Atom ≥ 2^B rows in swizzled dim?
[ ] sync_threads after smem write, before cross-thread read; and before buffer reuse?
[ ] cp.async: commit_group / wait_group / sync in the right order?
[ ] Reductions: bfly if all lanes need the result; range_constexpr for unroll?
[ ] Verified in SASS: LDG.128 / LDS.128 present; ncu bank-conflict counters ≈ 0?
```

---

## 9. Interview one-liners

- **Why 128 B matters:** one full sweep of 32 banks × 4 B. It's also the L1/L2 cache line and TMA's natural row width.
- **Swizzle in one sentence:** XOR the row index into the 16 B-slot index, so a column of vectors spreads across all bank groups. It's a permutation, so nothing is wasted.
- **`Swizzle<B,M,S>`:** keep 2^M elements whole, rotate among 2^B 16 B slots, and use the row number (found S bits above the slot bits) as the rotation. With a 128 B atom that's always `<3, log2(16/E), 3>`.
- **The "128" in B's formula** is bytes (one bank sweep), not the 128-bit vector width. The match is a coincidence.
- **Why 1024 B alignment for SW128:** the pattern repeats every 8 rows × 128 B, and the hardware computes the XOR from absolute address bits. So the base must sit on a pattern boundary.
- **Why vector loads need alignment:** a 128-bit load must be naturally aligned, and the compiler only emits it if it can *prove* alignment. That's what `assumed_align` is for.
- **Butterfly vs down shuffle:** butterfly gives every lane the result in log2(32) = 5 steps; down concentrates it in lane 0.
- **Padding vs swizzle:** padding breaks alignment and wastes space. Swizzle is free and is what tensor-core and TMA paths expect.
