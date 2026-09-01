---
title: CuteDSL
type: docs
math: true
prev: cuda/09-ptx
sidebar:
  open: false
weight: 1000
---

```bash
# For CUDA Toolkit 12.9:
uv pip install nvidia-cutlass-dsl

# For CUDA Toolkit 13.3:
# if on 13.1, update to 13.3 first, then:
uv pip install "nvidia-cutlass-dsl[cu13]"
```

---

## `kernel` v/s `jit`

**CuTe DSL** is a Python DSL for JIT-compiling GPU kernels — the CUTLASS C++ CuTe abstractions, exposed as decorators. Zero-cost (hybrid DSL: Python is the metaprogramming layer, everything hot lowers to IR), DLPack interop with PyTorch/JAX, and cached IR modules across calls.

## The two decorators

| | `@cute.jit` | `@cute.kernel` |
|---|---|---|
| Runs on | host | GPU |
| Called from | Python, `@jit`, `@kernel` | `@jit` only |
| Launch syntax | plain call | `f(args).launch(grid=..., block=...)` |

Both take `preprocessor=True` (default), which rewrites Python `for`/`if` into IR ops. Set it to `False` and you must avoid or hand-write flow control.

```python
import cutlass.cute as cute

@cute.kernel
def my_kernel(x: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    x[tidx] = x[tidx] * 2.0

@cute.jit
def entry(x: cute.Tensor):
    my_kernel(x).launch(grid=[1, 1, 1], block=[256, 1, 1])
```

`@jit` call sites accept `no_cache=True` to force recompilation.

### Launch parameters

`grid`, `block`, `cluster` are `[x, y, z]` lists. `smem` defaults to `None` — the size is derived from `cutlass.memory.SmemAllocator` usage, which is what you want unless you have a reason not to.

The rest are tuning knobs, all optional:

| Parameter | Default | Purpose |
|---|---|---|
| `fallback_cluster` | `None` | Makes `cluster` a *preferred* size; degrades to this minimum if the hardware can't satisfy it. |
| `max_number_threads` | `[0,0,0]` | `maxntid`. Default derives `reqntid` from `block`. |
| `min_blocks_per_mp` | `0` | `minctasm` occupancy hint. |
| `use_pdl` | `False` | Programmatic Dependent Launch — overlap dependent launches in one stream. |
| `cooperative` | `False` | Cooperative launch (grid-wide sync). |
| `smem_merge_branch_allocs` | `False` | Let mutually exclusive `if`/`else` branches reuse smem instead of summing. Experimental; for mega-kernels. |
| `preferred_smem_carveout` | `None` | % of on-chip memory for smem vs L1. Auto-computed when `min_blocks_per_mp > 1`. |
| `hint_smem_base_uniform` | `True` | Keep the dynamic-smem base pointer in a warp-uniform register. Experimental. |

## Calling conventions

```text
Python ──✅──► @jit ──✅──► @kernel     (dynamic launch via driver)
                │             │
                ├─✅─► @jit   ├─✅─► @jit      (inlined at compile time)
                └─✅─► py fn  └─✅─► py fn     (inlined at compile time)

Python ──❌──► @kernel      @kernel ──❌──► @kernel
```

Two rules cover it:
> - **a kernel can only be launched from `@jit`**,
> - and **a kernel cannot launch another kernel**.

Everything else — `@jit`, `@kernel`, or plain Python calling `@jit` or plain Python — is inlined at compile time and costs nothing at runtime.
