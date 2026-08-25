---
title: tcgen05 + TMA Notes — Blackwell Tensor Cores and the Data Path That Feeds Them
type: docs
math: true
sidebar:
  open: false
weight: 907
---

# tcgen05 + TMA Notes — Blackwell Tensor Cores and the Data Path That Feeds Them

*Companion to the PTX tensor core notes (`mma`/`wgmma`). Same contract: understandable first time, revisable months later. Covers sm_100a (datacenter Blackwell: B200/GB200). Consumer Blackwell (sm_120, RTX 50-series / RTX PRO 6000) does NOT have tcgen05 or TMEM — it uses Ampere-style `mma`. Check your target before applying anything here.*

---

## 0. Orientation: what generation 5 actually changes

Each tensor core generation has answered one question: *who owns the MMA?*

| Gen | Instruction | Issued by | Accumulator lives in |
|---|---|---|---|
| Ampere | `mma.sync` | whole warp (32 thr, converged) | warp's registers (fragments) |
| Hopper | `wgmma.mma_async` | warp group (128 thr) | warp group's registers |
| Blackwell | `tcgen05.mma` | **ONE thread** | **Tensor Memory (TMEM)** — not registers at all |

Two structural breaks:

1. **Tensor core execution is fully decoupled from warp execution.** A single thread issues the MMA (exactly like one thread issues a TMA copy); the tensor core runs it independently and signals completion through an **mbarrier**. Threads are no longer "inside" the MMA — they're clients of an engine. TMA and tcgen05.mma now have *the same programming shape*: descriptor in, one-thread issue, mbarrier out. This symmetry is the single best thing to hold onto.
2. **Accumulators move out of the register file into TMEM**, a new dedicated on-chip memory. Registers are freed for epilogue/softmax/whatever; occupancy stops being hostage to accumulator size; and `setmaxnreg` register-juggling from Hopper becomes largely unnecessary.

Throughput motivation: tcgen05.mma delivers roughly 2–4× wgmma throughput depending on dtype, with new formats (fp8, fp6, fp4, and microscaled `mxf4`/`mxf8` block-scaled types).

---

## 1. TMEM — Tensor Memory

### 1.1 Physical picture

- **256 KB per SM**, separate from smem/registers/L1.
- Addressed as a 2D grid: **128 lanes × 512 columns**, each cell 32 bits. A TMEM address is `(lane, column)` packed into a `.b32` (lane in the upper 16 bits, column in the lower 16).
- "Lane" here is an **address coordinate, not a thread lane ID** — easy first-time confusion. But the mapping to threads comes back at load time (§1.3).
- Purpose: hold MMA accumulators (D), and optionally operand A. B never lives in TMEM.

### 1.2 Allocation — dynamic, column-granular, and warp-collective

```ptx
// One full warp executes this (warp-collective, NOT single-thread):
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [smem_slot], 128;
```

Rules that bite:

- Allocation unit is a **column** (all 128 lanes of it). `n_cols` ∈ {32, 64, 128, 256, 512} — power of two, minimum 32.
- The instruction **writes the allocated base address into a shared-memory slot** you provide. The kernel then loads that slot (after a CTA sync + fence) so all warps learn the TMEM base. There is no "return register" — the detour through smem is mandatory.
- `alloc` may **block** until columns are free.
- Must be paired with `tcgen05.dealloc` **from the same warp**, and typically `tcgen05.relinquish_alloc_permit` when done allocating, so the allocation unit is released for other/peer CTAs.

```ptx
tcgen05.dealloc.cta_group::1.sync.aligned.b32  %tmem_base, 128;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

### 1.3 Access — `tcgen05.ld` / `tcgen05.st` and the warp-chunk rule

Threads read/write TMEM only through dedicated instructions:

```ptx
// Load 128 b32 values from TMEM into this warp's registers
tcgen05.ld.sync.aligned.32x32b.x128.b32  {%r0, ..., %r127}, [%tmem_addr];
tcgen05.wait::ld.sync.aligned;            // loads are async: wait before using regs
```

- Shapes like `.16x64b`, `.32x32b`, `.16x128b`, `.16x256b` describe the lane×bit access pattern; `.xN` is the repeat count.
- **The warp-chunk restriction (exam question):** TMEM's 128 lanes are split into 4 chunks of 32; warp *i* of a warp group may only access lanes `32*i .. 32*i+31`. So reading a full 128-lane accumulator requires all 4 warps of a warp group, each reading its own lane chunk. One warp cannot see the whole tile. This shapes every epilogue.
- `tcgen05.st` is the reverse (registers → TMEM, e.g. pre-loading a bias); `tcgen05.wait::st` to fence it.
- `tcgen05.cp` copies **smem → TMEM** asynchronously via a descriptor (used when staging operand A in TMEM, or accumulator init from smem).

### 1.4 What TMEM buys (the "why" for revision)

Hopper's pain: m64n256 f32 accumulators eat ~128 registers/thread across 128 threads → occupancy collapse → `setmaxnreg` gymnastics → epilogue and mainloop fight over the register file. Blackwell: accumulator sits in TMEM across the whole K-loop, registers hold only transient epilogue data, and the MMA engine reads/writes TMEM directly. Register pressure stops being the axis around which kernel design bends.

---

## 2. `tcgen05.mma` — the instruction

### 2.1 Syntax anatomy

```ptx
tcgen05.mma.cta_group::1.kind::f16
    [%d_tmem],        // accumulator: TMEM address (b32), NOT registers
    %a_desc,          // 64-bit smem descriptor (or TMEM address if A is in TMEM)
    %b_desc,          // 64-bit smem descriptor (B is always smem)
    %idesc,           // 32-bit instruction descriptor
    %enable_d;        // predicate: 1 => D += A*B, 0 => D = A*B (zero-init trick)
```

Piece by piece:

- **`.cta_group::1` or `::2`** — whether one CTA or a **CTA pair** (two CTAs on the two SMs of a TPC, in the same cluster) executes the MMA cooperatively. §4 below.
- **`.kind::*`** — dtype family: `kind::f16` (f16/bf16), `kind::tf32`, `kind::f8f6f4` (all 8/6/4-bit float combos), `kind::i8`, `kind::mxf8f6f4` / `kind::mxf4` / `kind::mxf4nvf4` (block-scaled microscaling formats — each tile column group carries a shared scale factor; the hardware applies scales during MMA).
- **`d_tmem`** — where results accumulate. The same TMEM region persists across the whole K-loop.
- **`a_desc` / `b_desc`** — *same 64-bit shared-memory descriptor format as wgmma* (start address, LBO/SBO strides in 16-byte units, swizzle mode). Everything you know from Hopper descriptors transfers unchanged. Tiles are expected K-major in smem; transpose is available via descriptor/idesc bits.
- **`idesc`** — new vs Hopper: a 32-bit **instruction descriptor** packing what used to be instruction-name suffixes: dtypes of A/B/D, MMA_M, MMA_N, transpose/negate bits for A and B, sparsity, scale-vector mode. Built once before the K-loop (shapes don't change), e.g.:

```cuda
constexpr uint32_t idesc =
      (1u << 4)                    // D dtype = f32
    | (1u << 7)                    // A dtype = bf16
    | (1u << 10)                   // B dtype = bf16
    | ((BLOCK_N >> 3) << 17)       // MMA_N / 8
    | ((BLOCK_M >> 4) << 24);      // MMA_M / 16
```

- **`enable_d`** — accumulate-vs-overwrite as an operand (like wgmma's scale-d): pass 0 on the first K-iteration to skip zero-initializing TMEM.
- Optional `disable_output_lane` mask — suppress writing selected accumulator lanes (used by split-K / partial-tile tricks; know it exists).

Shapes: M ∈ {64, 128} per CTA (256 with cta_group::2), N up to 256, K per instruction 16–64 depending on dtype (e.g. k16 f16, k32 fp8, k64 fp4). One instruction covers a tile so large that **there is no MMA tiling loop inside a block anymore** — a K-loop of single tcgen05.mma calls is the whole mainloop.

### 2.2 Issue and completion — the mbarrier contract

Single-thread issue, mbarrier completion — same protocol as TMA:

```ptx
// ONE elected thread:
tcgen05.mma.cta_group::1.kind::f16  [%d_tmem], %adesc, %bdesc, %idesc, %p;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64  [%mma_bar];
// commit = "when all MMAs issued so far complete, arrive on this mbarrier"

// Everyone else (and the issuer) later:
mbarrier.try_wait.parity.shared::cta.b64  %done, [%mma_bar], %phase;
```

- `tcgen05.commit` batches all previously issued (uncommitted) MMAs from this thread and attaches their completion to an mbarrier — the exact analogue of TMA's `complete_tx` mechanism, but arrival-count-based rather than byte-count-based.
- For `cta_group::2` there's a **multicast commit** that signals mbarriers in *both* CTAs of the pair.
- Ordering fences: `tcgen05.fence::before_thread_sync` / `after_thread_sync` order async tcgen05 ops against ordinary synchronization (e.g. before a `bar.sync` that "publishes" the completion to other warps, or before the epilogue's `tcgen05.ld`). Forgetting the fence between mbarrier-wait and `tcgen05.ld` = epilogue reads TMEM while the accumulator is still being written — the Blackwell equivalent of the missing `wgmma.fence` bug.

### 2.3 Contrast table (revision anchor)

| | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma` |
|---|---|---|---|
| Issue width | 32 threads | 128 threads | 1 thread |
| A source | regs | smem desc / regs | smem desc / TMEM |
| B source | regs | smem desc | smem desc |
| D lives in | regs | regs | TMEM |
| Shape config | instruction name | instruction name | idesc operand |
| Completion | synchronous | commit_group / wait_group | commit → mbarrier |
| Accumulate flag | implicit (C operand) | scale-d operand | enable_d predicate |
| Multi-SM | — | — | cta_group::2 |

---

## 3. TMA recap and how the two engines interlock

### 3.1 TMA in 6 lines (details in the previous notes)

- Host builds a **`CUtensorMap`**: global tensor's base/shape/strides + box (tile) size + **swizzle mode** + OOB fill.
- Kernel: ONE thread issues `cp.async.bulk.tensor.{1..5}d` with the tensormap + tile *coordinates*; the TMA engine does all address math, bounds handling, and **writes the tile into smem already swizzled**.
- Completion: TMA counts bytes into an mbarrier (`mbarrier::complete_tx::bytes` + `expect_tx`).
- Reverse direction (`smem → gmem`) for epilogue stores; also `cp.reduce.async.bulk` for atomic-reduce stores.
- Because TMA is an async-proxy engine, generic-proxy writes it must observe need `fence.proxy.async.shared::cta` first (mbarrier init, manually-written smem).

### 3.2 The full Blackwell dataflow

```
GMEM --TMA(cp.async.bulk.tensor)--> SMEM (swizzled) --tcgen05.mma reads via desc--> TMEM (acc)
                                                                                     |
GMEM <--st.global or TMA store-- registers <--tcgen05.ld (per-warp lane chunks)------+
```

Three independent hardware engines — TMA, tensor core, and the SM's regular execution units — each driven by descriptors and synchronized *only* through mbarriers. The kernel becomes a scheduler:

```
// Skeleton: one CTA, STAGES-deep smem ring buffer
init (warp 0):    mbarriers (full[s], empty[s], mma_done); fence.proxy.async
alloc (warp 1):   tcgen05.alloc -> tmem_base via smem slot
barrier (all)

LOAD thread (1 thread):
  for k in 0..K_TILES:
    wait empty[k % S]                       // buffer free?
    cp.async.bulk.tensor.2d [bufA], [tmapA,{...}], [full[k%S]]
    cp.async.bulk.tensor.2d [bufB], [tmapB,{...}], [full[k%S]]
    mbarrier.arrive.expect_tx full[k%S], BYTES_A + BYTES_B

MMA thread (1 thread):
  for k in 0..K_TILES:
    wait full[k % S]                        // tile landed?
    tcgen05.mma [tmem], adesc(bufA), bdesc(bufB), idesc, (k>0)
    tcgen05.commit [empty[k % S]]           // MMA done => buffer reusable
  tcgen05.commit [mma_done]

EPILOGUE (all 128 threads, after mma_done + fence):
  tcgen05.ld {regs...}, [tmem + warp's lane chunk]   // 4 warps cover 128 lanes
  tcgen05.wait::ld
  st.global / TMA store to output
dealloc (warp 1): tcgen05.dealloc + relinquish_alloc_permit
```

Read that skeleton twice — it *is* Blackwell kernel programming. Notice what vanished vs Hopper: no 128-thread wgmma choreography, no `wgmma.fence`/`commit_group`/`wait_group` trio, no register accumulator management. Notice what appeared: TMEM lifecycle, idesc, per-warp TMEM lane chunks in the epilogue.

Also notice the deep symmetry: the LOAD thread and the MMA thread run the **same pattern** (wait mbarrier → issue descriptor-based async op → signal mbarrier). On Blackwell, "warp specialization" reduces to electing a couple of single threads as engine drivers — which is why it's *simpler* than Hopper's, not more complex. (Full treatment in the warp-specialization notes.)

### 3.3 TMA multicast — the cluster-level feed trick

`cp.async.bulk.tensor` has a `.multicast::cluster` variant: one TMA fetch delivers the same gmem tile into the smem of **multiple CTAs in a cluster** simultaneously (a bitmask picks the recipients). Classic use: in a GEMM, CTAs in a cluster row share the same B tile — multicast fetches it once from HBM/L2 instead of per-CTA. Cuts DRAM traffic; pairs naturally with cta_group::2 (§4). Completion: each recipient CTA's mbarrier gets its own byte count.

---

## 4. `cta_group::2` — CTA pairs / 2-SM MMA

The biggest tcgen05 tile (m256) exceeds one CTA's TMEM+smem budget, so Blackwell lets **two CTAs on the two SMs of a TPC execute one MMA jointly**:

- Requirements: launched as a **cluster** (pairs are cluster ranks {0,1}, {2,3}, ...); `cta_group::2` on alloc/mma/commit. (Consumer sm_120 has no clusters → no CTA pairs → no tcgen05 at all there.)
- The accumulator is **split across the two CTAs' TMEM**: even CTA holds logical rows 0–127, odd CTA rows 128–255. Each CTA's epilogue reads only its own half.
- Only the **leader CTA's elected thread issues the MMA**; the multicast commit signals mbarriers in both CTAs.
- Operands: the pair can share smem via cluster addressing (`shared::cluster`), halving per-CTA smem bandwidth needs — combined with TMA multicast, this is how the full tensor core throughput becomes feedable at all. That's the honest reason CTA pairs exist: one SM's smem bandwidth cannot keep gen-5 tensor cores busy alone.

Mental model: Hopper scaled the MMA collective from warp → warp group. Blackwell scales it from CTA → CTA pair, while paradoxically shrinking the *issuing* collective to one thread. Compute widens; control narrows.

---

## 5. Numeric formats quick reference

| kind | input types | acc | notes |
|---|---|---|---|
| `kind::f16` | f16, bf16 | f32 (f16 out possible) | the default path |
| `kind::tf32` | tf32 | f32 | k8-ish depth |
| `kind::f8f6f4` | e4m3/e5m2/e3m2/e2m3/e2m1 mixes | f32 | 8/6/4-bit floats, A and B types can differ |
| `kind::i8` | s8/u8 | s32 | |
| `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4` | microscaled blocks (MXFP) | f32 | per-block scale factors (stored in TMEM, applied in-MMA); `nvf4` = NVIDIA's fp4 block format |

Block-scaled (`mx*`) kinds are why Blackwell inference numbers look the way they do: fp4 storage with hardware-applied per-block scales, no separate dequant kernel. Scale-factor layouts are their own rabbit hole (PTX ISA §9.7.17.10.7) — know they exist, look up when needed.

---

## 6. Debug checklist (Blackwell-specific bugs)

1. **`tcgen05.alloc` hangs** → previous allocation never `dealloc`ed / permit never relinquished, or you issued alloc from one thread instead of a full warp (it's warp-collective).
2. **All-zero / stale accumulator in epilogue** → missing `tcgen05.fence::after_thread_sync` (or the before-variant) between mbarrier wait and `tcgen05.ld`, or epilogue ran before the *commit's* mbarrier actually fired (waiting on the wrong barrier/phase).
3. **Garbage results** → idesc dtype/shape bits wrong (build it with named constants, not magic numbers), or smem descriptor swizzle mode ≠ tensormap swizzle mode (same bug as Hopper, same fix).
4. **Illegal instruction** → compiled for sm_100 without the `a` suffix features, or running on sm_120 consumer Blackwell (no tcgen05, no TMEM, no clusters).
5. **One warp reads wrong tile quadrant** → violated the TMEM warp-chunk rule; each warp of the group must address its own 32-lane chunk.
6. **cta_group::2 deadlock** → cluster not launched (`cudaLaunchKernelEx` with cluster dims), or commit not multicast to both CTAs' mbarriers, or both CTAs issued the MMA (only the leader should).
7. **Correct but slow** → check smem feed: without TMA multicast / CTA-pair operand sharing, tcgen05's throughput out-runs a single CTA's smem bandwidth. ncu: tensor pipe idle + smem throughput pegged = feeding problem, not MMA problem.

---

## 7. What transfers from what you already know

- **Descriptors**: wgmma's smem descriptor → identical format for a_desc/b_desc. Tensormap → unchanged. idesc is the only new descriptor.
- **mbarrier protocol**: TMA's expect_tx/arrive/try_wait → identical; tcgen05.commit is just a new *producer* of arrivals.
- **Proxy fences**: same rule, same places.
- **Swizzle**: same modes, same writer-reader agreement requirement.
- **What's genuinely new**: TMEM lifecycle (alloc/ld/st/cp/dealloc + lane-chunk rule), idesc, single-thread MMA issue, cta_group::2 semantics, block-scaled dtypes.

Deep-dive path: PTX ISA §9.7.17 (skim structure) → gau-nernst's "tcgen05 for dummies" blog + repo (best plain-CUDA walkthrough, reaches ~98% cuBLAS) → Colfax "GEMM kernels using Tensor Memory" CUTLASS tutorial → CUTLASS sm100 collective mainloops for how production code organizes the ring buffers.
