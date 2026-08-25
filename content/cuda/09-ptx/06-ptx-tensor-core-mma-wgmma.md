---
title: PTX Tensor Core (from `mma` to `wgmma`)
type: docs
math: true
sidebar:
  open: false
weight: 906
---

## 0. The one-paragraph orientation

Tensor cores are per-SM hardware units that execute small matrix multiply-accumulates (`D = A×B + C`) as single instructions. They are **collective**: no single thread owns the operation — a warp (32 threads, Volta→Ada) or a warp group (128 threads, Hopper) cooperatively supplies operands and receives results. Everything hard about programming them is **data choreography**: getting operand tiles into the exact register/shared-memory layouts the instruction demands, fast enough that the math units never starve. PTX is where those layouts stop being abstract.

The instruction lineage you must be able to place on sight:

| PTX instruction | Arch (sm) | Collective unit | Operands from | Async? |
|---|---|---|---|---|
| `wmma.*` | 70+ (Volta) | warp | registers (opaque fragments) | no |
| `mma.*` | 75/80+ (Turing/Ampere) | warp | registers (documented layouts) | no |
| `wgmma.*` | 90a (Hopper only) | warp group (128 thr) | shared memory via descriptors (A optionally regs) | yes |
| `tcgen05.*` | 100a+ (Blackwell) | single thread issues; CTA-scoped | shared mem / tensor memory | yes |

Modern hand-written code targets `mma` (portable, Ampere+) and `wgmma` (Hopper peak). `wmma` is legacy — read-only knowledge. [`tcgen05`](/cuda/09-ptx/07-ptx-tcgen-tma/) is covered in next page.

---

## 1. Prerequisite mental model: fragments and who-holds-what

A tensor core instruction computes a **tile** — e.g. `m16n8k16 (m,n are dimensions of output, and k is inner dimension of both inputs)` means A is 16×16, B is 16×8, D/C are 16×8. Those tiles don't live in one thread; they're **scattered across the warp's registers in a fixed interleaved pattern**. Each thread's slice is called its **fragment**.

> D = A*B + C is the canonical tensor core operation.

The single most important thing to internalize: **the mapping from (thread lane, register) → (row, col) of the tile is fixed by the ISA and is public documentation** (PTX ISA manual, "Matrix Fragments for mma"). You are expected to either:

1. Load operands with an instruction that *produces* that layout (`ldmatrix`), or
2. Compute the layout by hand when writing epilogues (writing D back to memory).

For the workhorse `mma.m16n8k16` with f16 inputs / f32 accumulate:

- **A** (16×16 f16): 8 elements per thread, held as 4 × `.b32` registers (2 f16 packed per reg)
- **B** (16×8 f16): 4 elements per thread, 2 × `.b32` regs
- **C/D** (16×8 f32): 4 elements per thread, 4 × `.f32` regs

Example of the fixed mapping (for D, f32): thread with lane id `T` holds elements
`d[0]`→(row = T/4, col = 2*(T%4)), `d[1]`→(same row, col+1), `d[2]`→(row+8, col), `d[3]`→(row+8, col+1).
You don't memorize these — you memorize *that they exist and where to look them up*, and that epilogue code (`store D to gmem`) is essentially "invert this mapping."

---

## 2. `mma` — the Ampere workhorse

### 2.1 Syntax anatomy

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%d0, %d1, %d2, %d3},      // D: 4 x .f32 per thread
    {%a0, %a1, %a2, %a3},      // A: 4 x .b32 (8 f16) per thread
    {%b0, %b1},                // B: 2 x .b32 (4 f16) per thread
    {%c0, %c1, %c2, %c3};      // C: 4 x .f32 per thread
```

Reading the suffix left to right:

- `sync.aligned` — all 32 threads of the warp execute this together, converged. Mandatory. Divergent warps = undefined behavior.
- `m16n8k16` — tile shape M×N×K. Common shapes: `m16n8k8`, `m16n8k16` (f16/bf16), `m16n8k32` (int8), `m16n8k4` (tf32).
- `row.col` — A is row-major, B is column-major *as interpreted by the instruction*. This pairing (row.col) is the only one supported for most shapes — meaning **B effectively needs a transpose somewhere upstream** if your data is row-major. That transpose is usually absorbed by `ldmatrix.trans` (below), not by an explicit kernel.
- `.f32.f16.f16.f32` — types of D, A, B, C. bf16: `.f32.bf16.bf16.f32`. int8: `.s32.s8.s8.s32`. tf32: `.f32.tf32.tf32.f32`.

Accumulation is in-place in practice: you pass the same registers for C and D, looping over K:

```ptx
// K-loop skeleton: acc += A_k x B_k
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%acc0,%acc1,%acc2,%acc3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%acc0,%acc1,%acc2,%acc3};
```

### 2.2 A note on tf32

tf32 is fp32 with a truncated 10-bit mantissa, chewed by tensor cores at ~8× fp32 FMA rate. In PTX you must *explicitly* convert (`cvt.rna.tf32.f32`) and use `mma...tf32` — it is opt-in at PTX level even though cuBLAS/PyTorch flip it on by default. Shape is `m16n8k4`/`m16n8k8` (small K because tf32 is 4 bytes).

### 2.3 `ldmatrix` — the layout-producing load

You *could* compute per-thread fragment addresses manually with regular `ld.shared`, but the addressing math is miserable and slow. `ldmatrix` exists to load 8×8 matrix tiles from **shared memory** directly into the exact fragment layout `mma` expects:

```ptx
// Each of the 4 threads-groups supplies a row address; x4 loads four 8x8 tiles
ldmatrix.sync.aligned.m8n8.x4.shared.b16
    {%a0, %a1, %a2, %a3}, [%smem_addr];
```

Mechanics worth remembering:

- Granularity is 8×8 tiles of 16-bit elements. `.x1/.x2/.x4` = load 1/2/4 tiles in one go (`x4` fills A of m16n8k16 exactly).
- **Address supply is the weird part**: threads 0–7 supply the 8 row start-addresses of tile 0, threads 8–15 for tile 1, etc. Each "row" is 16 contiguous bytes. The *data* each thread receives is unrelated to the address it supplied — the hardware shuffles across lanes.
- `.trans` variant transposes each 8×8 tile during the load — this is how row-major B in smem becomes the col-major fragment `mma` wants, for free.
- sm_75+ only. On Blackwell there's a `ldmatrix` extension for larger tiles, same idea.

### 2.4 Bank conflicts and swizzle at the PTX level

`ldmatrix` reads 16-byte rows per thread; with a naive row-major smem layout, the 8 addresses of a tile column-step hit the same banks. The fix is storing tiles **swizzled**: address bits XOR-permuted, canonically

```
smem_col_16B = col_16B XOR (row mod 8)      // "128B swizzle" family
```

so consecutive rows shift their 16B chunks across banks. In PTX you implement this by XOR-ing into the address you pass to `st.shared`/`cp.async` when *writing* the tile, and the same XOR when computing `ldmatrix` addresses. On Hopper, TMA applies the swizzle in hardware on the write side (`CU_TENSOR_MAP_SWIZZLE_128B`), and `wgmma` descriptors declare it on the read side — the concept is identical, the labor moves to hardware.

**Revision hook:** *swizzle is not an optimization you add later; the layout choice is made once, at tile-store time, and every reader must agree on it.*

### 2.5 Minimal but complete warp-tile fragment loop (Ampere pattern)

The canonical inner structure of every Ampere-class GEMM/attention kernel:

```ptx
// Per K-iteration, per warp:
// 1. (earlier: cp.async brought gmem tile -> smem, swizzled, double-buffered)
cp.async.wait_group 1;            // wait until buffer k is landed
bar.sync 0;

// 2. smem -> registers in mma layout
ldmatrix.sync.aligned.m8n8.x4.shared.b16       {%a0..%a3}, [%smemA_swz];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%b0,%b1},  [%smemB_swz];

// 3. math
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%acc0..%acc3}, {%a0..%a3}, {%b0,%b1}, {%acc0..%acc3};

// 4. meanwhile the *next* gmem tile is already in flight:
cp.async.cg.shared.global [next_smem], [next_gmem], 16;
cp.async.commit_group;
```

The pipeline (`cp.async` k+1 overlapping `mma` on k) is the whole game: **tensor cores are only fast if the loads for the next tile were issued before the math on this tile began.**

Equivalent CUDA C++ (what you'd actually write; nvcc emits the above):

```cuda
uint32_t a[4], b[2]; float acc[4] = {0};
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(smemA_addr));
// ... ldmatrix for b ...
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
```

---

## 3. `wgmma` — Hopper's warp-group MMA

### 3.1 What changed and why

Three structural shifts from `mma`:

1. **Collective unit is a warp group (128 threads)**, because Hopper-scale tiles (up to m64n256k16) need more accumulator registers than one warp owns. D is spread across all 128 threads' registers.
2. **Operands come from shared memory directly** (A optionally from registers, B always from smem). No `ldmatrix` step — the instruction reads smem itself. Operands are described by **matrix descriptors**, not addresses.
3. **It's asynchronous.** `wgmma.mma_async` is issued into a pipeline; you continue issuing, then explicitly wait. This is what lets one warp group overlap math with other work.

### 3.2 The matrix descriptor

A 64-bit value packing: smem start address, leading-dimension byte offset, stride byte offset, base offset, and the **swizzle mode** (none/32B/64B/128B). Built by shifting/OR-ing bitfields:

```cuda
// The idiom you'll see in CUTLASS/hand-rolled Hopper code:
uint64_t desc = 0;
desc |= (matrix_smem_addr >> 4);            // bits 0..13: addr / 16
desc |= (uint64_t)(lbo >> 4) << 16;         // leading byte offset
desc |= (uint64_t)(sbo >> 4) << 32;         // stride byte offset
desc |= (uint64_t)swizzle_mode << 62;       // 1=128B, 2=64B, 3=32B
```

**Revision hook:** the descriptor is "tensormap's little sibling" — tensormap describes a tensor in *global* memory for TMA; the wgmma descriptor describes a tile in *shared* memory for the tensor core. TMA writes swizzled; the descriptor's swizzle field tells wgmma how to unscramble. The two must declare the same mode — mismatches produce garbage numbers, not errors.

### 3.3 Instruction and fence choreography

```ptx
wgmma.fence.sync.aligned;                  // (1) order prior register writes vs wgmma

wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%d0, ..., %d63},                      // 64 f32 accumulators per thread
    %descA, %descB,                        // 64-bit smem descriptors
    1,                                     // scale-d: 1 => D = A*B + D, 0 => D = A*B
    1, 1,                                  // scale-a, scale-b (negation flags)
    0, 1;                                  // trans-a, trans-b

wgmma.commit_group.sync.aligned;           // (2) close the batch
wgmma.wait_group.sync.aligned 0;           // (3) wait until <=0 groups pending
```

The fence/commit/wait trio, in plain words:

- `wgmma.fence` — must precede the first `mma_async` after any thread wrote registers/smem the wgmma will read. Forgetting it is the classic "results are nondeterministically wrong" bug.
- `commit_group` — batches all `mma_async` issued since the last commit into one completion unit (same mental model as `cp.async.commit_group`).
- `wait_group N` — block until at most N groups are still in flight. `wait_group 0` = drain everything. Keeping 1 in flight while computing on previous results is the software-pipelining knob.
- Accumulators are readable **only after** the wait; touching them earlier is UB.

Note `scale-d`: accumulate-or-overwrite is an *operand bit*, not separate instructions — a K-loop sets it to 0 on the first iteration, 1 afterward (or just zeros the accumulators once and always passes 1).

### 3.4 Where the data comes from: the full Hopper pipeline

`wgmma` never touches global memory, so a complete Hopper mainloop is TMA + mbarrier + wgmma glued together. Skeleton, producer/consumer specialized:

```
producer warp (1 thread active):
  loop k:
    wait until buffer[k % STAGES] is free            (mbarrier "empty" phase)
    cp.async.bulk.tensor.2d ... [buf], [tmap,{x,y}], [full_bar[k%STAGES]]
    mbarrier.arrive.expect_tx full_bar, TILE_BYTES

consumer warp group (128 threads):
  loop k:
    mbarrier.try_wait full_bar[k % STAGES]           (data landed, swizzled, by TMA)
    wgmma.fence
    wgmma.mma_async ... descA(buf), descB(buf) ...   (multiple, covering the tile's K)
    wgmma.commit_group
    wgmma.wait_group 1                                (keep one group in flight)
    mbarrier.arrive empty_bar[k % STAGES]             (release buffer to producer)
```

Plus the proxy fence rule from the memory-model side: after *generic* stores initialize mbarriers (or smem the TMA will read), `fence.proxy.async.shared::cta` before the async-proxy operations see them.

### 3.5 Register economics: `setmaxnreg`

Producer warps issue TMA (needs ~no registers); consumers hold giant accumulators. Hopper lets warps re-partition the register file at runtime:

```ptx
setmaxnreg.dec.sync.aligned.u32 40;    // producer warps shrink to 40 regs/thread
setmaxnreg.inc.sync.aligned.u32 232;   // consumer warps grow to 232
```

This is why warp-specialized kernels can afford m64n256 accumulators without occupancy collapse — and why you'll see these odd instructions at the top of FA3/CUTLASS SASS.

---

## 4. Triton's relationship to all of this

Triton makes you express the *tile program*; its backend chooses everything above per-architecture:

```python
acc = tl.zeros((64, 128), dtype=tl.float32)
for k in range(0, K, BK):
    a = tl.load(...)   # -> cp.async (sm_80) or TMA via descriptors (sm_90)
    b = tl.load(...)
    acc = tl.dot(a, b, acc)   # -> mma.m16n8k16 loop (sm_80) or wgmma (sm_90)
```

What it decides for you: fragment layouts, `ldmatrix` usage, swizzle pattern, pipeline depth (`num_stages`), and on sm_90 whether to emit wgmma + TMA + mbarriers (warp specialization included in newer versions). What you still control: tile shapes, `num_warps` (4 warps = 1 warp group — the reason sm_90 kernels like `num_warps=4` multiples), `num_stages`, and memory-access patterns that make coalescing/swizzling possible.

**Practical implication:** read the PTX Triton emits (`triton.compile(...).asm["ptx"]` or `TRITON_CACHE_DIR`) — you'll now recognize every instruction in it, which is exactly the skill needed to answer "why is my tl.dot kernel slow" with ncu.

---

## 5. Numeric shapes & types quick reference (revision table)

| Input type | Acc type | mma shape (Ampere) | wgmma shapes (Hopper) |
|---|---|---|---|
| f16 / bf16 | f32 | m16n8k16 (also k8) | m64nNk16, N ∈ {8,16,...,256} |
| tf32 | f32 | m16n8k4 / k8 | m64nNk8 |
| int8 (s8/u8) | s32 | m16n8k32 | m64nNk32 |
| fp8 (e4m3/e5m2) | f32 | — (Ada has mma fp8) | m64nNk32 |
| b1 (binary) | s32 | m16n8k256 | m64nNk256 |

K scales inversely with element width (K × elem_size ≈ 32 bytes of depth); N on Hopper is the free axis you crank for arithmetic intensity.

---

## 6. Debug / correctness checklist (the bugs everyone hits)

1. **Garbage numbers, no error** → swizzle mode mismatch between writer (TMA/manual store) and reader (descriptor/ldmatrix addressing). Verify both declare the same mode.
2. **Nondeterministic wrong results on Hopper** → missing `wgmma.fence` after register writes, or reading accumulators before `wait_group`.
3. **Hang** → mbarrier `expect_tx` byte count doesn't match what TMA actually transfers, or phase parity bug in `try_wait`.
4. **Wrong results only for some tiles** → epilogue fragment mapping wrong; re-derive thread→(row,col) from the PTX ISA fragment tables instead of guessing.
5. **Illegal instruction at runtime** → wgmma on non-sm_90a compile target (`-arch=sm_90a`, the `a` suffix matters), or ldmatrix pre-sm_75.
6. **Slow despite tensor cores** → check ncu "Pipe Tensor" utilization: if low, the bottleneck is feeding (bank conflicts, no pipelining, launch-bound), not math. `mma` throughput is almost never the actual limiter in real kernels.

---

## 7. Suggested deep-dive path from here

1. PTX ISA manual §9.7.13–14 (mma fragment layouts, ldmatrix) — skim the tables so you know their shape.
2. Write one warp-tile m16n8k16 GEMM in raw CUDA+inline PTX; verify against cuBLAS. (One evening; permanently demystifies fragments.)
3. Read the PTX of a Triton `tl.dot` kernel on sm_80, map every instruction to these notes.
4. CUTLASS `examples/48_hopper_warp_specialized_gemm` + read emitted SASS for wgmma choreography.
5. GTC talks: "Developing CUDA Kernels for Hopper" (TMA/wgmma), FlashAttention-3 paper §3 (warp specialization + pingpong scheduling).
