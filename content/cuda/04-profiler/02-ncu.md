---
title: Nsight Compute
type: docs
math: true
prev: cuda/03-cuda-cpp/01-basics.md
sidebar:
  open: false
weight: 402
---

**NVIDIA Nsight Compute** (`ncu`) is a *kernel-level* profiler. Where `nsys` shows the whole timeline (where does time go *between* kernels), `ncu` answers what's happening *inside* one kernel: memory-bound? compute-bound? occupancy-limited?

- Workflow: `nsys` first → pick the suspect kernel → aim `ncu` at it.

> [!IMPORTANT]
> `nsys` **observes**, `ncu` **intervenes**. ncu hijacks each targeted kernel and **replays it many times** (different hardware counter group per pass — the GPU has too few counter registers to collect everything in one go). Consequences:
> - profiled runs are 10–1000× slower than normal
> - ncu timings ≠ nsys timings (cache flushes, clock locking between passes) — don't try to reconcile them
> - **run the workload ONCE inside the capture range** — ncu does the repetition itself. Looping 5 steps just profiles the same kernels 5× redundantly (5 steps × 4 kernels × ~35 passes = ~700 kernel executions for nothing).

## Minimal example

```python
"""command:
ncu --set full --profile-from-start off \
        -k "regex:gemm" --launch-count 3 -f \
        -o deependu_ncu_report \
        python main.py
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.layers(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = SimpleModel().to(device)
    inputs = torch.randn(4, 8).to(device)

    with torch.inference_mode():
        # warmup — lazy init, cuBLAS handle creation etc. stay out of the capture
        for _ in range(20):
            model(inputs)
        torch.cuda.synchronize()

        # ONE step is enough — ncu replays each kernel itself
        with torch.cuda.profiler.profile():
            torch.cuda.nvtx.range_push("profiled_step")
            _ = model(inputs)
            torch.cuda.nvtx.range_pop()

if __name__ == "__main__":
    main()
```

Differences from the nsys script:
- **single forward pass** in the capture range (see replay note above)
- no `emit_nvtx()` — that was for per-op attribution on the nsys timeline; ncu targets kernels by name, and emit_nvtx just adds overhead here. Keep plain `nvtx.range_push` though — it enables `--nvtx-include` filtering.
- `--profile-from-start off` is ncu's equivalent of nsys's `--capture-range cudaProfilerApi`: wait for `cudaProfilerStart()`, i.e. the same `with torch.cuda.profiler.profile():` block works unchanged.

## Key flags

| Flag | Purpose |
|---|---|
| `--profile-from-start off` | only profile after `cudaProfilerStart()` |
| `-o <file>` | write `<file>.ncu-rep` |
| `-f` / `--force-overwrite` | overwrite existing report |
| `-k <name>` / `--kernel-name` | filter kernels by name (exact, or `regex:` prefix) |
| `-s N` / `--launch-skip N` | skip the first N *matching* launches |
| `-c M` / `--launch-count M` | profile M launches after skipping, then detach |
| `--set basic\|detailed\|full` | how many counter sections to collect |
| `--nvtx --nvtx-include "range/"` | only profile kernels inside an NVTX range |
| `--kernel-id ctx:stream:name:invocation` | surgical targeting of one exact invocation |

### `--launch-skip` / `--launch-count`

ncu counts matching kernel launches in order, then slices that sequence: skip N, profile M, stop. The count applies **after** name filtering — `-k regex:gemm -s 2 -c 1` = "the 3rd gemm in the program", not the 3rd kernel overall.

Standard idiom: **skip past warmup, count 1** — instances of the same launch are near-identical, one representative is enough.

```bash
# only gemms, one instance, full detail:
ncu --set full --profile-from-start off \
    -k regex:gemm -c 1 \
    -f -o gemm_report \
    python main.py
```

### `--set` (cost model)

A *set* = bundle of counter *sections*. More sections → more replay passes → slower.

| Set | Contents | Cost |
|---|---|---|
| `basic` (default) | Speed of Light, launch stats, occupancy | ~9 passes/kernel |
| `full` | + memory workload analysis, roofline, scheduler stats, instruction mix | ~35 passes/kernel |

`ncu --list-sets` / `ncu --list-sections` shows the menu. Total cost = (instances profiled) × (passes per set) → control both knobs.

### Mangled vs demangled names

By default `-k` matches against the **mangled base name**. If a regex mysteriously matches nothing:

```bash
ncu --kernel-name-base demangled -k "regex:at::native::vectorized" ...
```

## Reading the report

Open in `ncu-ui`, or without GUI:

```bash
ncu --import report.ncu-rep --page summary            # one line per kernel: duration, SM %, mem %
ncu --import report.ncu-rep --page details            # full sections
```

Reading order (the 90/10):
1. **GPU Speed of Light (SOL)** — `Compute (SM) %` and `Memory %` vs hardware peak. High mem/low compute → memory-bound. High compute → compute-bound. **Both low → latency-bound or kernel too small to saturate the GPU.**
2. **Roofline** (`--set full`) — arithmetic intensity vs the memory-slope/compute-ceiling.
3. **Occupancy** — theoretical vs achieved, and *which resource limits it* (registers / shared mem / block size).
4. **Memory Workload Analysis** — L1/L2/DRAM traffic, sectors per request → uncoalesced access & bank conflicts become measurable here.

> [!IMPORTANT]
> Tiny kernels (like this toy model's) show **single-digit SOL on both axes** — 2 blocks on a 40-SM GPU leaves 38 SMs idle. ncu is the wrong tool for launch-bound workloads; that's nsys territory. Scale the problem until the kernels can actually saturate something before trusting ncu's verdict.

## Gotchas

- `ERR_NVGPUCTRPERM` → driver restricts counters to admin (`NVreg_RestrictProfilingToAdminUsers`), needs elevated access or a driver setting change.
- ncu attaches to **child processes** too (e.g. Inductor compile workers show up as a second `==PROF== Connected`).
- Kernel names change with shape — cuBLAS picks different tile variants for big matrices, so don't hard-code a regex from a small-shape run; do a quick unfiltered pass (or check nsys) to learn the names first.
