---
title: Writing Inline PTX (CUDA & Triton)
type: docs
math: true
sidebar:
  open: false
weight: 905
---

Reading PTX is the common case. Writing it is the escape hatch — for instructions with no intrinsic, for precise control the compiler won't give you, and for new hardware features before CUDA exposes them.

> [!IMPORTANT]
> **Reach for inline PTX last.** In order: (1) a CUDA intrinsic, (2) a CUB/CUTLASS/`cuda::` primitive that already wraps it, (3) a compiler hint (`__restrict__`, `__launch_bounds__`, `#pragma unroll`), (4) inline PTX. Hand-written `asm` blocks defeat optimization passes across the boundary, break silently on new architectures, and don't get type-checked until `ptxas` runs.

---

## 1. `asm()` in CUDA — the whole syntax

```cpp
asm("template-string" : "constraint"(output) : "constraint"(input) : "clobbers");
```

```cpp
asm("membar.gl;");                                          // no operands at all
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));      // d = a + b
asm("mov.s32 %0, 2;" : "=r"(i));                            // no inputs → drop last colon
asm("st.global.u32 [%0], %1;" :: "l"(p), "r"(x) : "memory"); // no outputs → adjacent colons
```

**Operand numbering**: `%0`, `%1`, … index the flat operand list in text order, **outputs first**. They may appear in any order and may repeat.

```cpp
asm("add.s32 %0, %1, %1;" : "=r"(i) : "r"(k));   // i = k + k
```

**Escaping `%`**: PTX special registers start with `%`, so double it.

```cpp
asm volatile("mov.u32 %0, %%clock;" : "=r"(x));    // %%clock → %clock
```

### Constraints

| Letter | PTX register type | C/C++ type |
|---|---|---|
| `h` | `.u16` | `short`, `unsigned short`, `__half` (via bit-cast) |
| `r` | `.u32` | `int`, `unsigned` |
| `l` | `.u64` | `long long`, pointers |
| `q` | `.u128` | `__int128` (only where supported) |
| `f` | `.f32` | `float` |
| `d` | `.f64` | `double` |
| `n` | immediate integer, value known at compile time | `constexpr` int |
| `C` | `constexpr const char[]` — **spliced into the template as text** | compile-time instruction-mode selection |

Modifiers on the constraint:

| Modifier | Meaning |
|---|---|
| `"=r"` | **write-only** output |
| `"+r"` | **read-write** output — required if the asm *conditionally* updates it |
| `"r"` | input |

> [!WARNING]
> There is **no 8-bit constraint**. For `.b8` operations, use a 32-bit register — PTX permits operands wider than the instruction type:
> ```cpp
> int d;
> asm("ld.u8 %0, [%1];" : "=r"(d) : "l"(in) : "memory");
> ```
> Similarly, `__half` and `__nv_bfloat16` go through `"h"` with a bit-cast, and `__half2`/`__nv_bfloat162` through `"r"`.

### `volatile` and `"memory"`

The compiler assumes an `asm` block has **no side effects except writing its outputs**, and may delete, duplicate, or hoist it.

| Situation | What you need |
|---|---|
| Result depends on time/state, not just inputs (`%clock`, `%globaltimer`, `%smid`) | `asm volatile` |
| The block reads/writes memory the compiler can't see | `: "memory"` clobber |
| The block must not move across surrounding loads/stores | both |
| Pure function of its inputs (`add`, `cvt`, `prmt`) | **neither** — let it be optimized |

```cpp
asm volatile("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
asm         ("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));  // pure: no volatile
```

### Multi-line blocks, temporaries, and scope

Use C++ implicit string concatenation, and end every line but the last with `\n\t` so the emitted PTX stays readable.

```cpp
__device__ int cube(int x) {
    int y;
    asm("{\n\t"                              // ◄── braces give a PRIVATE scope
        " .reg .u32 t1;\n\t"                 //     so t1 doesn't collide when inlined twice
        " mul.lo.u32 t1, %1, %1;\n\t"
        " mul.lo.u32 %0, t1, %1;\n\t"
        "}"
        : "=r"(y) : "r"(x));
    return y;
}
```

Predicates need the same treatment — a `.pred` register must be declared inside the block:

```cpp
__device__ int is34(int x) {
    int y = 0;                               // note the initialization
    asm("{\n\t"
        " .reg .pred %%p;\n\t"
        " setp.eq.s32 %%p, %1, 34;\n\t"
        " @%%p mov.s32 %0, 1;\n\t"           // conditional write → "+r", not "=r"
        "}"
        : "+r"(y) : "r"(x));
    return y;
}
```

> [!CAUTION]
> Without the `{ }` scope, an `asm` block declaring `.reg .u32 t1;` produces **"duplicate definition of t1"** from `ptxas` as soon as the function is inlined more than once. Labels inside `asm` need the same treatment (or unique names via `__COUNTER__`).

### Compile-time instruction selection with `"C"`

```cpp
template<> struct rmode<RN> { static constexpr const char m[] = ".rn"; };
template<> struct rmode<RZ> { static constexpr const char m[] = ".rz"; };

template <int M>
__device__ float add_r(float a, float b) {
    float r;
    asm("add.f32%1 %0, %2, %3;" : "=f"(r) : "C"(rmode<M>::m), "f"(a), "f"(b));
    return r;                                 // → add.f32.rn  or  add.f32.rz
}
```

The `"C"` operand's contents are **spliced as text** into the template. This is how you template over rounding modes, scopes, cache hints, or MMA shapes without writing N copies of the asm string.

---

## 2. The four pitfalls (straight from NVIDIA's guide)

| Pitfall | Symptom | Fix |
|---|---|---|
| **Namespace conflicts** | `ptxas` error: duplicate definition of `t1` | wrap the block in `{ }` |
| **Memory space confusion** | wrong data / crash | any pointer passed to `asm` arrives as a **generic** address. Use `ld.f32` (generic) or convert first with `cvta.to.global`/`__cvta_generic_to_shared()` |
| **Incorrect optimization** | asm block vanishes or moves | `asm volatile` and/or `"memory"` |
| **Incorrect PTX** | cryptic `ptxas` parse error, wrong answer | the front end never parses your string. Compile early and often; check the generated `.ptx` |

Errors the compiler *does* catch:

```cpp
asm("add.s32 %0,%1,%2;" : "=r"(i) : "rf"(j), "r"(k));  // ✗ one constraint letter per operand
asm("add.s32 %0,%1,%2;" : "=r"(i4) : ...);             // ✗ aggregates (int4) not allowed
asm("add.s32 %0,%1,%2;" : "=r"(ci) : ...);             // ✗ char: size 1 ≠ size implied by 'r'
asm("add.s32 %0,%1,%2;" : "=r"(fi) : ...);             // ✗ float with 'r' constraint
```

The shared-memory address pitfall deserves its own line, because it is the most common real bug:

```cpp
__shared__ float tile[256];

// ✗ WRONG — tile decays to a 64-bit generic pointer, st.shared wants a 32-bit smem offset
asm volatile("st.shared.f32 [%0], %1;" :: "l"(&tile[i]), "f"(v));

// ✓ RIGHT
uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tile[i]));
asm volatile("st.shared.f32 [%0], %1;" :: "r"(smem_addr), "f"(v));
```

---

## 3. Architecture gating

Inline PTX for a newer architecture will not assemble for an older one. Guard it:

```cpp
__device__ void tma_load(...) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  #if defined(__CUDA_ARCH_FEAT_SM90_ALL)          // set by -arch=sm_90a
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global"
                 ".mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem), "l"(tmap), "r"(x), "r"(y), "r"(bar) : "memory");
  #else
    #error "This kernel requires -arch=sm_90a"
  #endif
#endif
}
```

| Macro | Set when |
|---|---|
| `__CUDA_ARCH__` | device compilation; `900` = sm_90, `1000` = sm_100 |
| `__CUDA_ARCH_FEAT_SM90_ALL` | `-arch=sm_90a` — enables `wgmma`, Hopper TMA forms, 228 KB smem |
| `__CUDA_ARCH_FEAT_SM100_ALL` | `-arch=sm_100a` — enables `tcgen05` |
| `__CUDA_ARCH_FEAT_SM120_ALL` etc. | corresponding `a` targets |

Remember `__CUDA_ARCH__` is undefined during **host** compilation, so always test `defined(__CUDA_ARCH__)` first.

---

## 4. Recipes

```cpp
// ── timing ────────────────────────────────────────────────────────────
__device__ __forceinline__ uint64_t clock64_ptx() {
    uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t;
}
__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// ── read-only cache load (when the compiler won't prove it) ───────────
__device__ __forceinline__ float ldg_f32(const float* p) {
    float v; asm("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v;
}

// ── streaming store: don't pollute L1 with write-once output ──────────
__device__ __forceinline__ void stream_f32(float* p, float v) {
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}

// ── vectorized 128-bit load ───────────────────────────────────────────
__device__ __forceinline__ float4 ldg128(const float4* p) {
    float4 v;
    asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

// ── elect one lane of a warp ──────────────────────────────────────────
__device__ __forceinline__ bool elect_one() {
    uint32_t pred;
    asm volatile("{ .reg .pred %%p; .reg .b32 %%r;\n\t"
                 "  elect.sync %%r|%%p, 0xffffffff;\n\t"
                 "  selp.b32 %0, 1, 0, %%p;\n\t"
                 "}" : "=r"(pred));
    return pred != 0;
}

// ── warp reduction in one instruction (sm_80+) ────────────────────────
__device__ __forceinline__ int warp_sum(int v) {
    int s; asm("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(s) : "r"(v)); return s;
}

// ── fp8 pack: two f32 → one packed e4m3x2 ─────────────────────────────
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float a, float b) {
    uint16_t d;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(a), "f"(b));
    return d;
}

// ── async copy global → shared (Ampere) ───────────────────────────────
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(gmem) : "memory");
}
template<int N>                                  // N must be an immediate → template, not a param
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.commit_group;\n\t"
                 "cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// ── mbarrier ──────────────────────────────────────────────────────────
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t ok;
    asm volatile("{ .reg .pred %%p;\n\t"
                 "  mbarrier.try_wait.parity.shared::cta.b64 %%p, [%1], %2;\n\t"
                 "  selp.b32 %0, 1, 0, %%p;\n\t"
                 "}" : "=r"(ok) : "r"(b), "r"(phase) : "memory");
    return ok != 0;
}

// ── register budget for warp specialization (sm_90+) ──────────────────
template<int N> __device__ __forceinline__ void set_max_nreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
}

// ── low-power spin ────────────────────────────────────────────────────
__device__ __forceinline__ void nap(uint32_t ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}
```

> [!TIP]
> Before writing any of these by hand, check whether CUDA already ships it: `__ldg`, `__nanosleep`, `__reduce_add_sync`, `__cvta_generic_to_shared`, `cuda::memcpy_async`, `cuda::barrier`, `cuda::ptx::*` (libcu++ has typed wrappers for a large fraction of modern PTX, including TMA and mbarrier), and CUTLASS/CuTe's `cute::copy`/`cute::gemm` atoms. Hand-rolled inline PTX should be the residual.

---

## 5. Inline PTX in Triton

Triton exposes inline assembly through **`tl.inline_asm_elementwise`**. It is *elementwise only* — the asm block sees a few scalars per invocation, not the whole tile.

```python
tl.inline_asm_elementwise(
    asm: str,          # the PTX, using $0, $1, ... for operands (NOT %0)
    constraints: str,  # comma-separated LLVM constraints: outputs first, then inputs
    args: Sequence,    # input tensors, implicitly broadcast to a common shape
    dtype,             # output dtype, or a tuple of dtypes for multiple outputs
    is_pure: bool,     # True → compiler may CSE/DCE it (like NOT using `volatile`)
    pack: int,         # elements processed per asm invocation
)
```

Four differences from CUDA `asm()` that will trip you up:

| | CUDA | Triton |
|---|---|---|
| Operand placeholder | `%0` | **`$0`** |
| Special register escape | `%%clock` | `%%clock` (still doubled) |
| Constraints | one string per operand | **one comma-separated string for all**, outputs first |
| Side-effect control | `volatile` keyword | `is_pure=False` |
| Empty output | allowed | **not allowed** — return a dummy tensor if you don't need one |

```python
@triton.jit
def clamped_exp(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.0)
    # ex2.approx: fast base-2 exponential straight from the SFU
    y = tl.inline_asm_elementwise(
        asm="ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Y + offs, y, mask=offs < N)
```

Multiple outputs, and `pack` > 1 to process several elements per invocation:

```python
@triton.jit
def fp8_pack(X, Y, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(X + offs, mask=offs < N, other=0.0)
    b = tl.load(Y + offs, mask=offs < N, other=0.0)
    # two f32 → one packed e4m3x2 (returned in a 32-bit register, low 16 bits valid)
    packed = tl.inline_asm_elementwise(
        asm="cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        constraints="=h,f,f",
        args=[a, b],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )
    tl.store(OUT + offs, packed, mask=offs < N)
```

> [!IMPORTANT]
> **`pack` and sub-4-byte types.** Triton packs elements smaller than 4 bytes into 32-bit registers before handing them to the asm block. With `pack=4` and a `uint8` input, `$N` holds **four** bytes at once and you must unpack them yourself (`mov.b32 {t0,t1,t2,t3}, $N;` then `cvt.u32.u8`). Getting `pack` wrong produces silently wrong results, not an error.

**What Triton inline asm cannot do:** anything cross-lane or memory-touching. No `shfl`, no `bar.sync`, no `mma`, no `ld.global`, no `mbarrier`. The block operates on values already in registers, and Triton owns the layout. For those, your options are:

| Need | Use |
|---|---|
| Elementwise math not in `tl.*` | `tl.inline_asm_elementwise` |
| Device-library math (`erf`, `rint`, …) | `tl.extra.cuda.libdevice.*` |
| Explicit layouts, shared memory, warp specialization, TMA, `mma` control | **Gluon** (Triton's low-level dialect) |
| Full control | a real CUDA kernel + `torch.utils.cpp_extension` / a custom op |

---

## 6. Verification workflow

Inline PTX is untyped text until `ptxas` sees it. Build the feedback loop before you write the second line.

```bash
# 1. does it even assemble?
nvcc -arch=sm_90a -ptx k.cu -o k.ptx        # front end + your string, verbatim
nvcc -arch=sm_90a -cubin k.cu               # ← ptxas runs HERE; this is where errors appear

# 2. did it land where you meant?
grep -n -A3 -B3 'cp.async.bulk' k.ptx

# 3. what did ptxas make of it?
nvcc -arch=sm_90a -Xptxas -v -cubin k.cu    # registers, spills — inline asm often adds both
cuobjdump -sass k.cubin | grep -i utmaldg

# 4. assemble a standalone PTX file directly (fastest iteration on a snippet)
ptxas -arch=sm_90a -O3 snippet.ptx -o snippet.cubin
```

```python
# Triton: verify the asm reached the output
compiled = kernel[(1,)](...)
assert "cvt.rn.satfinite.e4m3x2.f32" in compiled.asm["ptx"]
print(compiled.n_regs, compiled.n_spills)
```

Then, always: **a numerical test against a reference implementation**, and a run under `compute-sanitizer`.

```bash
compute-sanitizer --tool memcheck  ./a.out
compute-sanitizer --tool racecheck ./a.out    # catches missing fences/barriers around asm
```

> [!CAUTION]
> Inline PTX that touches memory or synchronization is exactly the code `racecheck` was built for. A missing `"memory"` clobber or a missing `fence.proxy.async` produces a race that passes every test on your machine and fails in production under load.

---

## 7. Condensed reference

### Instruction families

| Family | Representative opcodes |
|---|---|
| Integer | `add sub mul{.lo,.hi,.wide} mad sad div rem abs neg min max popc clz bfind brev bfe bfi bmsk szext dp4a dp2a clmad` |
| Float | `add sub mul fma mad div rcp sqrt rsqrt sin cos lg2 ex2 tanh abs neg min max copysign testp` |
| Compare/select | `set setp selp slct` |
| Logic/shift | `and or xor not cnot lop3 shf shl shr` |
| Data movement | `mov ld st ldu prefetch isspacep cvta cvt cvt.pack prmt mapa getctarank` |
| Async copy | `cp.async cp.async.bulk{.tensor} cp.reduce.async.bulk st.async tensormap.replace` |
| Control | `bra brx.idx call ret exit @ { }` |
| Sync/comm | `bar barrier barrier.cluster membar fence atom red vote.sync match.sync activemask redux.sync elect.sync mbarrier.* griddepcontrol` |
| Warp MMA | `wmma.* mma.sync mma.sp ldmatrix stmatrix movmatrix` |
| Warpgroup MMA | `wgmma.mma_async wgmma.fence wgmma.commit_group wgmma.wait_group` |
| 5th-gen TC | `tcgen05.{alloc,dealloc,mma,ld,st,cp,shift,wait,fence,commit}` |
| Misc | `brkpt nanosleep pmevent trap setmaxnreg` |

### Qualifier cheatsheet

| Qualifier group | Values |
|---|---|
| State space | `.reg .sreg .const .global .local .param{::entry,::func} .shared{::cta,::cluster} .tex` |
| Types | `.s8/16/32/64 .u8/16/32/64 .f16 .f16x2 .f32 .f64 .b8/16/32/64/128 .pred` |
| Alt FP | `bf16 tf32 e4m3 e5m2 e3m2 e2m3 e2m1 ue8m0 ue4m3` (+ `x2`/`x4` packed forms) |
| Rounding | `.rn .rz .rm .rp` · `.rni .rzi .rmi .rpi` · `.rna` · `.rs` |
| Saturation | `.sat .satfinite .relu` |
| Load cache | `.ca .cg .cs .lu .cv` · `.nc` |
| Store cache | `.wb .cg .cs .wt` |
| Eviction | `.L1::evict_{normal,first,last,unchanged} .L1::no_allocate .L2::evict_{normal,first,last}` |
| Semantics | `.weak .volatile .relaxed .acquire .release .acq_rel .sc .mmio` |
| Scope | `.cta .cluster .gpu .sys` |
| Proxy | `.alias .async .async.global .async.shared::{cta,cluster} .tensormap::generic` |
| Vector | `.v2 .v4 .v8` |

### CUDA → PTX map

| CUDA | PTX |
|---|---|
| `threadIdx.x` / `blockIdx.x` / `blockDim.x` / `gridDim.x` | `%tid.x` / `%ctaid.x` / `%ntid.x` / `%nctaid.x` |
| `__syncthreads()` / `__syncwarp()` | `bar.sync 0` / `bar.warp.sync -1` |
| `__threadfence{,_block,_system}()` | `fence.sc.{gpu,cta,sys}` |
| `atomicAdd` (used / unused result) | `atom.*.add` / `red.*.add` |
| `__ldg` / `const __restrict__` | `ld.global.nc` |
| `__shfl_{,up,down,xor}_sync` | `shfl.sync.{idx,up,down,bfly}.b32` |
| `__ballot_sync` / `__activemask` | `vote.sync.ballot.b32` / `activemask.b32` |
| `__reduce_add_sync` | `redux.sync.add.s32` |
| `__fmaf_rn` | `fma.rn.f32` |
| `__float2half_rn` | `cvt.rn.f16.f32` |
| `__launch_bounds__(N, M)` | `.maxntid N,1,1` + `.minnctapersm M` |
| `__cluster_dims__(x,y,z)` | `.reqnctapercluster` / `.explicitcluster` |
| `extern __shared__` | `.extern .shared .align 4 .b8 name[]` |
| `cuda::memcpy_async` | `cp.async.*` |
| `cuda::barrier` | `mbarrier.*` |
| `cuTensorMapEncodeTiled` + copy | `cp.async.bulk.tensor.*` |
| `nvcuda::wmma` | `wmma.*` |
| register spill | `ld.local` / `st.local` |

---

## Interview Questions & Answers

### Q: When should an `asm()` block be `volatile`, and what does the `"memory"` clobber add on top?

**Answer:** `volatile` tells the compiler the block has side effects beyond writing its declared outputs, so it must not be deleted (even if the outputs are unused), duplicated, or hoisted out of a loop. You need it whenever the result isn't a pure function of the inputs — reading `%clock`, `%globaltimer`, `%smid`, `activemask`, or anything with an architectural side effect like `setmaxnreg` or `mbarrier.arrive`. The `"memory"` clobber is a separate, stronger statement: it tells the compiler this block may read or write memory it can't see, so it must not reorder surrounding loads and stores across it and must not keep memory values cached in registers. Timing reads want both (`volatile` so it isn't moved, `"memory"` so surrounding accesses aren't reordered around the measurement). A pure computation like `prmt.b32` or a `cvt` wants **neither** — marking it volatile just blocks CSE and costs performance.

### Q: You write inline PTX that stores to `__shared__` memory and it corrupts data. What's the most likely cause?

**Answer:** Address-space mismatch. Any pointer handed to an `asm()` block arrives as a **generic** 64-bit address, but `st.shared`/`ld.shared` expect a 32-bit offset into the shared window. Passing `&tile[i]` with the `"l"` constraint to a `st.shared.f32` writes to a garbage shared address. The fix is `__cvta_generic_to_shared(ptr)`, truncated to `uint32_t` and passed with `"r"` — or use the generic form `st.f32` and pay the window-resolution cost. The same class of bug hits mbarrier and `cp.async` code, where the shared operand is always a 32-bit smem address while the global operand is a 64-bit pointer.

### Q: Why is Triton's `tl.inline_asm_elementwise` restricted to elementwise operations?

**Answer:** Because Triton owns the data layout. A Triton tensor of shape `(128, 64)` is distributed across threads according to a layout the compiler chose (and may change during optimization) — a given lane's registers hold an arbitrary, non-contiguous subset of elements. A cross-lane instruction like `shfl` or `mma` needs to know exactly which lane holds which element, and Triton deliberately doesn't expose that at the `tl.*` level. Elementwise is the one class of operation whose meaning is layout-independent: apply f to each value wherever it happens to live. For anything layout-dependent, Triton's answer is **Gluon**, the lower-level dialect that does make layouts, shared memory, and warp specialization explicit.

### Q: A colleague's hand-written `wgmma` inline PTX gives correct results in a unit test but wrong results in the full model. Where would you look?

**Answer:** Missing cross-proxy synchronization is the top suspect. `wgmma` reads its A/B operands from shared memory through the **async proxy**, so shared memory written by ordinary `st.shared` (generic proxy) needs `fence.proxy.async.shared::cta` before the MMA — and `wgmma.fence.sync.aligned` before it as well, to order accesses to the *accumulator registers*. Those two fences cover different things and both are required. A unit test with one tile and one warpgroup often has enough incidental serialization to hide the race; the full model, with a deep pipeline and multiple stages in flight, does not. Second suspect: the swizzle mode encoded in bits 63–62 of the shared-memory descriptor not matching the swizzle the TMA descriptor actually wrote with — that produces plausible-but-wrong numbers rather than a crash. `compute-sanitizer --tool racecheck` and a single-stage (`num_stages=1`) run are the fastest ways to separate the two.
