---
title: PTX (parallel thread execution)
type: docs
math: true
prev: cuda/08-onnx
sidebar:
  open: false
weight: 900
---

**PTX** is NVIDIA's **virtual ISA** — the stable, documented, human-readable assembly language that sits between your CUDA/Triton source and the GPU's real machine code (SASS).

```text
CUDA C++ ──nvcc/cicc──┐
                      ├──► PTX ──ptxas──► SASS ──► cubin ──► fatbin
Triton ──MLIR─►LLVM ──┘        ▲                    ▲
                               │                    │
                    fully documented,      real ISA, arch-specific,
                    forward-compatible,    mostly undocumented
                    JIT-able by driver
```

> Prerequisite: [CUDA Binary Toolchain](/cuda/01-basics/03-cuda-toolchain/) covers where PTX sits in the compilation pipeline, `compute_XX` vs `sm_XX`, `cuobjdump`, `nvdisasm` and JIT caching. This chapter is about the **language itself**.

---

## Why bother with PTX in 2026

You can write good CUDA for years without reading PTX. You cannot do the following without it:

| Goal | Why PTX is unavoidable |
|---|---|
| **Understand tensor cores** | `mma.sync`, `wgmma.mma_async`, `tcgen05.mma` are PTX-level instructions. There is no C++ intrinsic for most of them. CUTLASS/CuTe is a template wrapper around inline PTX. |
| **Low-precision / quantization** | fp8 (`e4m3`/`e5m2`), fp6 (`e2m3`/`e3m2`), fp4 (`e2m1`), and the `ue8m0` microscaling exponents exist as PTX `cvt` types and mma `.kind::mxf4` variants long before they get friendly C++ types. |
| **Hopper/Blackwell async pipelines** | TMA (`cp.async.bulk.tensor`), `mbarrier`, clusters + distributed shared memory, `setmaxnreg` warp specialization — all PTX. |
| **Reading Triton/torch.compile output** | Triton has no C++ to inspect. PTX *is* the readable artifact between Triton IR and SASS. |
| **Debugging "why is this slow/wrong"** | "Did my load vectorize?", "did the mask become a branch or a predicate?", "did `tl.dot` actually emit an mma?" — one `grep` away in PTX. |
| **Escaping the compiler** | When no intrinsic exists, inline `asm()` is the only door. |

> [!IMPORTANT]
> **PTX is not what runs.** `ptxas` re-does register allocation, scheduling, and instruction selection. PTX tells you **what the compiler intended** (semantics, memory space, precision, tensor-core usage). SASS tells you **what the hardware does** (register pressure, spills, actual latency). Use PTX for *semantics*, SASS + Nsight Compute for *performance*.

---

## Versioning: `.version`, `.target`, and the `a`/`f` suffixes

Every PTX module starts with two directives that pin its dialect and its hardware assumptions:

```ptx
.version 9.3          // PTX ISA version — a language version, tied to the CUDA toolkit
.target   sm_90a      // virtual architecture whose feature set this code assumes
.address_size 64
```

- **`.version`** bumps when NVIDIA adds instructions. Newer `.version` needs a newer `ptxas`/driver. This is why an old driver rejects a new wheel's PTX.
- **`.target sm_XX`** follows an **onion-layer model**: each generation keeps everything the previous one had, so `sm_80` PTX runs on `sm_90` hardware.

Two suffixes break the onion model:

| Suffix | Meaning | Example |
|---|---|---|
| *(none)* — `sm_90` | Portable subset. Runs on this arch **and all later ones**. | `sm_80`, `sm_90` |
| **`a`** — arch-specific | Features exposed on **that architecture only**. Does *not* forward-run on later GPUs. | `sm_90a` (wgmma, TMA intrinsics), `sm_100a` (tcgen05) |
| **`f`** — family-specific | Features shared within one **architecture family**; runs on later devices *in the same family*. | `sm_100f`, `sm_120f` |

Families (PTX ISA 9.x): `sm_10x` = {`sm_100f`, `sm_103f`, …}, `sm_11x` = {`sm_110f`, `sm_101f`, …}, `sm_12x` = {`sm_120f`, `sm_121f`, …}.
Note `sm_101*` was **renamed to `sm_110*`** in PTX ISA 9.0.

> [!TIP]
> This is why CUTLASS and FlashAttention build with `-arch=sm_90a` rather than `sm_90`: `wgmma` and the TMA instructions only assemble under the `a` target. nvcc defines `__CUDA_ARCH_FEAT_SM90_ALL` (and `__CUDA_ARCH_FEAT_SM100_ALL`, …) so you can `#ifdef` around them.

---

## Getting PTX out of anything

Everything below is a one-liner you will use constantly while working through these notes.

```bash {filename="from CUDA C++"}
nvcc -arch=sm_90 -ptx kernel.cu -o kernel.ptx        # source → PTX, stop there
nvcc -arch=sm_90 -ptx -lineinfo -src-in-ptx k.cu     # interleave CUDA source as comments
nvcc -arch=sm_90 -cubin k.cu && cuobjdump -sass k.cubin   # the SASS it becomes
cuobjdump -ptx a.out                                  # PTX embedded in an existing binary
```

```python {filename="from Triton"}
import triton, triton.language as tl

@triton.jit
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr): ...

compiled = k[(1,)](x, y, n, BLOCK=1024)   # launch once to force compilation
print(compiled.asm["ttir"])   # Triton IR
print(compiled.asm["ttgir"])  # Triton GPU IR (layouts appear here)
print(compiled.asm["llir"])   # LLVM IR
print(compiled.asm["ptx"])    # ← PTX
print(compiled.asm["cubin"])  # raw bytes
```

```bash {filename="dump every stage to disk"}
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump TRITON_ALWAYS_COMPILE=1 python bench.py
# → ./dump/<hash>/{*.ttir,*.ttgir,*.llir,*.ptx,*.cubin}
# TRITON_ALWAYS_COMPILE=1 defeats the ~/.triton/cache hit that would skip codegen
```

```bash {filename="from torch.compile (Inductor generates Triton)"}
TORCH_LOGS=output_code python train.py      # print the generated Triton kernels
TORCH_COMPILE_DEBUG=1 python train.py       # full debug dir incl. generated code
# then use the Triton dump env vars above to get their PTX
```

For quick experiments with no GPU: [godbolt.org](https://godbolt.org) has nvcc with side-by-side PTX and SASS panes.

---

## How these notes are organized

Read them in order the first time; after that they work as reference.

| # | Note | What it gives you |
|---|---|---|
| **01** | [PTX Language Fundamentals](/cuda/09-ptx/01-ptx-language-fundamentals/) | Module structure, state spaces, types, virtual registers, instruction grammar, predication, control flow, special registers, directives. **The grammar you need before anything else parses.** |
| **02** | [Reading Emitted PTX](/cuda/09-ptx/02-reading-emitted-ptx/) | Line-by-line walkthroughs of real nvcc and Triton output. The idiom catalog, the review checklist, and what you must *not* conclude from PTX. |
| **03** | [Memory Model & Synchronization](/cuda/09-ptx/03-memory-model-and-synchronization/) | Addressing, cache qualifiers, the memory consistency model (`.relaxed`/`.acquire`/scopes/proxies), atomics, barriers, clusters + DSMEM, `cp.async`, **`mbarrier`**, **TMA**. |
| **04** | [Compute & Tensor Cores](/cuda/09-ptx/04-compute-and-tensor-cores/) | Arithmetic and rounding, `cvt` for **fp8/fp6/fp4 quantization**, warp primitives, and the tensor-core lineage: `wmma` → `mma.sync` → `wgmma` → **`tcgen05`**. |
| **05** | [Writing Inline PTX](/cuda/09-ptx/05-writing-inline-ptx/) | `asm()` in CUDA, `tl.inline_asm_elementwise` in Triton, constraints, pitfalls, recipes, verification workflow, plus a condensed reference cheatsheet. |

---

## A reading strategy that survives six months

1. **Never read PTX top-to-bottom.** Skip the `.reg` declarations and the parameter prologue. Find the loop body — it's the block between a label and a `bra` back to that label.
2. **Grep before you read.** `ld.global`, `st.global`, `bar.sync`, `mma`, `cp.async`, `atom`, `ld.local`/`st.local` (spills!) tell you the shape of a kernel in five seconds.
3. **Read qualifiers right-to-left.** `ld.global.nc.v4.f32` = "load, from global, non-coherent (read-only cache), four-wide vector, of f32". The type is always last.
4. **Keep a diff handy.** The highest-value use of PTX is `diff before.ptx after.ptx` after a source change.
5. **Stop at the right layer.** Semantics → PTX. Performance → SASS + `ncu`.

## References

- [PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) — the source of truth; ~1000 pages, but chapters 5 (state spaces/types), 9.7 (instruction set) and 10 (special registers) are the ones you'll live in.
- [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/) — short, read it once, covered in [chapter 05](./05-writing-inline-ptx/).
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) — `cuobjdump`, `nvdisasm`, per-arch SASS opcode lists & in [notes](/cuda/01-basics/03-cuda-toolchain/)
