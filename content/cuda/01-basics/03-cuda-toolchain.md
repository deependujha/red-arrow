---
title: CUDA Binary Toolchain- PTX, SASS, cubin, fatbin & Inspection Tools
type: docs
math: true
prev: docs/
weight: 30
sidebar:
  open: false
---


Reference notes on what nvcc actually produces, how the pieces relate, and how to inspect them with `cuobjdump`, `nvdisasm`, and friends.

---

## 1. The Big Picture: Compilation Pipeline

```
foo.cu
  │
  ├── host code ──► host compiler (gcc/clang/MSVC) ──► host object code
  │
  └── device code ──► cicc (NVVM/LLVM frontend) ──► PTX (virtual assembly)
                                                       │
                                                       ▼
                                              ptxas (PTX assembler)
                                                       │
                                                       ▼
                                              SASS (real machine code)
                                                       │
                                                       ▼
                                              cubin (ELF container, one arch)
                                                       │
                            (one or more cubins + optional PTX)
                                                       ▼
                                              fatbin (fat binary container)
                                                       │
                                    embedded into host object → final executable/.so
```

Key mental model: **PTX is a virtual ISA; SASS is the real ISA.** Everything else (cubin, fatbin) is packaging.

- **Virtual ISA**: Describes what the program wants the GPU to do, while abstracting away the exact hardware implementation. e.g., assumes unlimited registers, no scheduling constraints, and a stable instruction set across future hardware.

- **Real ISA**: Describes the concrete instructions the target hardware actually executes. e.g., has a fixed register count matching the hardware, scheduling constraints, and is specific to a particular GPU hardware.

> ISA: `Instruction Set Architecture` — the set of instructions a processor understands. e.g., `add.f32`, `ld.global.f32`, `mma.sync` are all SASS instructions.

---

## 2. PTX (Parallel Thread eXecution)

- A **virtual instruction set** — stable, documented, forward-compatible. Think of it as CUDA's "LLVM IR meets assembly."
- Text-based, human-readable. Uses virtual registers (`%r1`, `%f2`, `%rd3` — unlimited, SSA-ish), typed instructions (`add.s32`, `ld.global.f32`, `mma.sync.aligned...`).
- Targets a **virtual architecture**: `compute_80`, `compute_90`, etc.
- **Forward compatibility mechanism**: PTX embedded in a binary can be **JIT-compiled by the driver** (`ptxas` inside the driver) for GPUs newer than any SASS you shipped. PTX for `compute_80` runs on sm_90 hardware via JIT. SASS does *not* have this property.
- PTX is *not* what executes. Register counts, instruction scheduling, spills — all decided later by `ptxas`. Never draw performance conclusions from PTX register usage.

```ptx
.visible .entry _Z6vecAddPfS_S_i(...)
{
    .reg .f32   %f<4>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<11>;

    ld.param.u64    %rd1, [_Z6vecAddPfS_S_i_param_0];
    ...
    ld.global.f32   %f1, [%rd8];
    ld.global.f32   %f2, [%rd9];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rd10], %f3;
    ret;
}
```

## 3. SASS (Streaming ASSembler / Shader ASSembly)

- The **actual machine code** executed by the GPU. Architecture-specific and mostly undocumented (opcodes listed in the CUDA Binary Utilities doc, semantics largely reverse-engineered by the community).
- Fixed physical register file (255 general-purpose registers per thread max), real scheduling, real predication.
- **Not portable across major architectures** — sm_80 SASS won't run on sm_90. Binary compatibility only holds within a major compute capability family (sm_80 SASS runs on sm_86, sm_89... mostly).
- What you read when you actually care about performance: memory instruction widths (`LDG.E.128`), tensor core usage (`HMMA`, `IMMA`, `QGMMA`), predicates (`@P0`), bank conflicts don't show directly but shared memory access patterns do (`LDS`, `STS`).

Common SASS instructions worth recognizing:

| Instruction | Meaning |
|---|---|
| `LDG` / `STG` | Load/store global memory (`LDG.E.128` = 128-bit vectorized load) |
| `LDS` / `STS` | Load/store shared memory |
| `LDGSTS` | Async copy global→shared (`cp.async`, Ampere+) |
| `HMMA` / `IMMA` | Tensor core MMA (half / int) |
| `FFMA` | Fused multiply-add (fp32) |
| `IMAD` | Integer multiply-add (often used for address math, even `IMAD.MOV`) |
| `BAR.SYNC` | `__syncthreads()` |
| `EXIT` | Thread exit |
| `@P0`, `@!P1` prefix | Predicated execution (how branches often compile) |
| `S2R` | Special register → register (e.g. reading `threadIdx`) |
| `UTMALDG` / TMA ops | Tensor Memory Accelerator (Hopper+) |

## 4. cubin

- An **ELF file (executable & linkable format)** containing SASS for **one specific architecture** (e.g. sm_90), plus metadata: kernel symbols, register counts, shared memory sizes, constant banks, relocation info.
- Produced by `ptxas`. `file foo.cubin` → "ELF 64-bit LSB relocatable, NVIDIA CUDA architecture".
- Can be loaded directly at runtime via the driver API (`cuModuleLoad`).

## 5. fatbin (fat binary)

- A **container** bundling multiple cubins (for different archs) and/or PTX (for JIT forward-compat), possibly compressed.
- Embedded in the host object file in sections like `.nv_fatbin` / `__nv_relfatbin`.
- At runtime, the CUDA runtime picks the best match for the current GPU: exact SASS if present → else compatible SASS → else JIT from PTX → else `cudaErrorNoKernelImageForDevice` / "no kernel image available".

---

## 6. `compute_XX` vs `sm_XX` — the thing everyone confuses

- `compute_XX` = **virtual** architecture → controls what PTX is generated (which features you can use).
- `sm_XX` = **real** architecture → controls what SASS is generated.
- `-arch` sets the virtual arch, `-code` sets the real outputs, `-gencode` lets you specify pairs.

```bash
# Shorthand: PTX for compute_90 + SASS for sm_90, both embedded
nvcc -arch=sm_90 foo.cu -o foo

# Explicit multi-arch fatbin: SASS for Ampere + Hopper, PTX for Hopper (future-proofing)
nvcc foo.cu -o foo \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90    # code=compute_XX → embed PTX

# "native": build only for the GPU in this machine
nvcc -arch=native foo.cu -o foo
```

Rules of thumb:
- `code=sm_XX` → embeds SASS.
- `code=compute_XX` → embeds PTX.
- Ship SASS for every arch you care about **plus** PTX for the newest, so future GPUs still work (at a JIT cost on first run).
- Each `-gencode` pair multiplies compile time and binary size (why PyTorch wheels are huge).

Selected arch values: `70` Volta (V100) · `75` Turing · `80` Ampere (A100) · `86` Ampere consumer (30xx) · `89` Ada (40xx, L4) · `90` Hopper (H100) · `90a` Hopper w/ arch-specific features (wgmma, TMA intrinsics) · `100`/`120` Blackwell (B200 / 50xx).

{{< callout type="important" >}} 

  1. `nvcc` first generates PTX for the virtual architecture (`compute_XX`) you requested.
  2. `ptxas` then compiles that PTX into SASS for the real architecture (`sm_XX`) you requested.
  3. `nvcc -gencode` lets you specify multiple virtual→real pairs, producing a fat binary with multiple cubins and/or PTX.
  4. `arch` specifies to the nvcc compiler which virtual architecture to target for PTX generation.
  5. `code` specified to nvcc what all to embed in the fat binary (SASS for real archs, PTX for virtual archs).

  #. `arch` can only accept a single value and must be `compute_XX` or `native`. Whereas, `code` can accept multiple values and must be `sm_XX` or `compute_XX`. and it basically tells the compiler, to first emit PTX for the virtual architecture and then compile that PTX into SASS for the real architecture and embed that SASS into the fat binary. If you specify `code=compute_XX`, it will embed PTX for that virtual architecture into the fat binary, which can be JIT compiled by the driver at runtime if no matching SASS is found.

  #. when we only specify `arch=sm_XX`, it expands to `arch=compute_XX,code=sm_XX` and emits PTX for the virtual architecture and SASS for the real architecture. If we specify `arch=compute_XX`, it expands to `arch=compute_XX,code=compute_XX` and emits PTX for the virtual architecture and embeds that PTX into the fat binary.
{{< /callout >}}

---

## 7. nvcc — useful flags for inspection & tuning

```bash
nvcc -ptx foo.cu                 # stop after PTX → foo.ptx
nvcc -cubin -arch=sm_90 foo.cu   # emit foo.cubin
nvcc -fatbin -arch=sm_90 foo.cu  # emit foo.fatbin
nvcc -c foo.cu                   # object file with embedded fatbin

nvcc -keep foo.cu                # keep ALL intermediates (.ptx, .cubin, .fatbin, .ii, ...)
nvcc -keep -keep-dir build/ foo.cu

nvcc --dryrun foo.cu             # print every sub-command nvcc would run (great for learning the pipeline)
nvcc -v foo.cu                   # verbose: actually run + show commands

# Resource usage report from ptxas — registers, spills, smem, cmem:
nvcc -Xptxas -v -arch=sm_90 foo.cu
#   ptxas info: Used 40 registers, 0 bytes spill stores, 0 bytes spill loads,
#               16384 bytes smem, 384 bytes cmem[0]

nvcc -lineinfo foo.cu            # embed source line info (essential for ncu/nsys source view; ~no perf cost)
nvcc -G foo.cu                   # device debug: disables optimizations (never benchmark with this)
nvcc -Xptxas -O3 foo.cu          # ptxas opt level (default already 3)
nvcc --maxrregcount=64 foo.cu    # cap registers per thread (occupancy vs spill tradeoff)
                                 # (prefer __launch_bounds__ per-kernel over a global cap)
nvcc -src-in-ptx -lineinfo -ptx foo.cu   # interleave CUDA source as comments in PTX
```

Things `ptxas -v` tells you and why you care:
- **Registers/thread** → occupancy limiter (regs/SM ÷ regs/thread bounds resident threads).
- **Spill stores/loads** → register pressure overflowing to local memory (global mem!) — a red flag.
- **smem** → static shared memory per block (dynamic smem from `<<<..., smemBytes>>>` not shown here).
- **cmem[0]** → kernel params + constants in constant memory.

---

## 8. cuobjdump — inspecting what's inside a binary

Works on executables, `.so`/`.a`, `.o`, `.cubin`, `.fatbin`. Answers "what did the compiler actually put in this file?"

```bash
cuobjdump a.out                    # summary of embedded ELF images + PTX
cuobjdump -lelf a.out              # list embedded cubins (name + arch) — quick "what archs does this ship?"
cuobjdump -lptx a.out              # list embedded PTX
cuobjdump -ptx a.out               # dump all embedded PTX text
cuobjdump -sass a.out              # disassemble all embedded SASS
cuobjdump -elf foo.cubin           # dump ELF sections/metadata of a cubin
cuobjdump -symbols a.out           # device symbol table (mangled kernel names)
cuobjdump -res-usage a.out         # per-kernel register/smem/cmem usage from the binary

# Filter to one architecture (essential on fat binaries like PyTorch's):
cuobjdump -sass -arch sm_90 a.out

# Extract embedded images to standalone files:
cuobjdump -xelf all a.out          # extract every cubin
cuobjdump -xelf foo.sm_90.cubin a.out   # extract a specific one (name from -lelf)
cuobjdump -xptx all a.out          # extract PTX files
```

Real-world example — see what a PyTorch install actually ships:

```bash
cuobjdump -lelf $(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_cuda.so'))") | awk '{print $NF}' | sort -u
# → lists sm_80, sm_86, sm_90 ... i.e. the archs in the wheel

# Dump SASS of one kernel from a library (use -fun with the mangled name from -symbols):
cuobjdump -sass -fun _Z9my_kernelPfS_i libmylib.so
```

Demangle names with `c++filt` (or `cu++filt` for CUDA-specific mangling):

```bash
cuobjdump -symbols a.out | grep STT_FUNC | c++filt
```

## 9. nvdisasm — the heavier disassembler

Only takes **cubin** input (not executables/fatbins — extract first with `cuobjdump -xelf`). More capable than `cuobjdump -sass`:

```bash
nvdisasm foo.cubin                    # disassemble
nvdisasm -c foo.cubin                 # annotate with control-flow info
nvdisasm -g foo.cubin                 # interleave source lines (needs -lineinfo at compile)
nvdisasm -cfg foo.cubin > cfg.dot     # emit control-flow graph in DOT
dot -Tpng cfg.dot -o cfg.png          # render it (graphviz)
nvdisasm --print-life-ranges foo.cubin   # register live ranges
nvdisasm --print-line-info foo.cubin
```

**cuobjdump vs nvdisasm**: `cuobjdump -sass` is the quick look and works on anything; `nvdisasm` is for deep analysis of a single cubin (CFGs, live ranges, source interleave).

## 10. Related tools

```bash
nvprune --arch sm_90 fat.a -o slim.a   # strip a fat library down to one arch (smaller deploys)
ptxas -arch=sm_90 -O3 foo.ptx -o foo.cubin   # run the PTX→SASS step manually
ptxas -arch=sm_90 -v foo.ptx           # resource usage without full nvcc invocation
fatbinary ...                          # (internal) builds fatbins; you rarely call it directly
readelf -S foo.o | grep nv_fatbin      # see the fatbin section in a host object
```

---

## 11. JIT compilation & the compute cache

- If the runtime falls back to PTX JIT, the driver compiles PTX→SASS **at first kernel/module load** — can add seconds (or minutes for big libraries) to startup.
- Results cached in `~/.nv/ComputeCache` (override dir with `CUDA_CACHE_PATH`, size with `CUDA_CACHE_MAXSIZE`, disable with `CUDA_CACHE_DISABLE=1`).
- `CUDA_FORCE_PTX_JIT=1` forces JIT even when SASS exists — handy to test the forward-compat path.
- Slow first-run on a brand-new GPU arch with an older wheel? That's PTX JIT. Fix by installing a build with native SASS for that arch.

---

## 12. Practical workflows (recipes)

**"Did my kernel vectorize loads?"**
```bash
nvcc -arch=sm_90 -cubin k.cu && cuobjdump -sass k.cubin | grep LDG
# want LDG.E.128 (float4-width), not four LDG.E.32
```

**"Is my kernel actually using tensor cores?"**
```bash
cuobjdump -sass k.cubin | grep -E "HMMA|IMMA|GMMA"
```

**"Why did occupancy drop?"**
```bash
nvcc -Xptxas -v ...        # check registers/thread and smem/block
cuobjdump -res-usage a.out # same info from an existing binary
```

**"Do I have register spills?"**
```bash
nvcc -Xptxas -v ... 2>&1 | grep spill      # non-zero spill bytes
cuobjdump -sass k.cubin | grep -E "LDL|STL" # local loads/stores in SASS confirm it
```

**"What archs does this .so support / why 'no kernel image available'?"**
```bash
cuobjdump -lelf libfoo.so     # embedded SASS archs
cuobjdump -lptx libfoo.so     # any PTX for JIT fallback?
nvidia-smi --query-gpu=compute_cap --format=csv   # what your GPU needs
```

**"Show me source ↔ SASS mapping"**
```bash
nvcc -arch=sm_90 -lineinfo -cubin k.cu
nvdisasm -g k.sm_90.cubin | less
# (or just open the kernel in Nsight Compute's Source page — same lineinfo)
```

**Quick iteration without local hardware:** [godbolt.org](https://godbolt.org) has CUDA (nvcc + PTX/SASS views) — fastest way to see how a code change alters SASS.

**Triton note:** Triton bypasses nvcc but converges on the same backend — it emits PTX (via LLVM NVPTX) and drives `ptxas` to produce a cubin. `triton_kernel.asm["ptx"]` / `.asm["cubin"]` expose the artifacts, and everything in sections 8–9 applies to the cubin.

---

## 13. Mental model cheat sheet

| Artifact | What | Portable? | Executes? | Inspect with |
|---|---|---|---|---|
| `.cu` | CUDA C++ source | — | no | your eyes |
| PTX | virtual ISA (text) | forward (via JIT) | no (JIT'd first) | `nvcc -ptx`, `cuobjdump -ptx` |
| SASS | real machine code | same major arch only | **yes** | `cuobjdump -sass`, `nvdisasm` |
| cubin | ELF of SASS, one arch | no | loaded directly | `cuobjdump -elf/-sass`, `nvdisasm` |
| fatbin | bundle of cubins + PTX | best-match at runtime | container only | `cuobjdump -lelf/-lptx` |

## 14. References

- CUDA Binary Utilities docs (cuobjdump / nvdisasm / nvprune) — includes per-arch SASS instruction lists
- PTX ISA reference (the virtual ISA is fully documented; SASS is not)
- NVCC docs, "GPU Compilation" chapter — virtual vs real archs, `-gencode` semantics
