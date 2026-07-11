---
title: Nsight System
type: docs
math: true
prev: cuda/03-cuda-cpp/01-basics.md
sidebar:
  open: false
weight: 401
---

**NVIDIA Nsight Systems** (`nsys`) is a system-wide timeline profiler. It shows CPU and GPU activity together: CUDA kernels, memory transfers, CUDA API calls, OS threads, and NVTX annotations, all on a single timeline.

- Use `nsys` first to understand *where* time is going (CPU-bound? kernel-launch overhead? PCIe transfers?), then use `ncu` to deep-dive into individual kernels.

```python
"""command:
nsys profile \
  --trace cuda,nvtx,cublas,osrt \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  --output deependu_nsys_profile_report \
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
    model = torch.compile(SimpleModel(),mode="max-autotune-no-cudagraphs").to(device)
    inputs = torch.randn(4, 8).to(device)

    with torch.inference_mode():
        # warmup
        for _ in range(20):
            model(inputs)
        torch.cuda.synchronize()

        with torch.cuda.profiler.profile(), torch.autograd.profiler.emit_nvtx():
            for step in range(5):
                torch.cuda.nvtx.range_push(f"step_{step}")
                _ = model(inputs)
                torch.cuda.nvtx.range_pop()

if __name__ == "__main__":
    main()
```

![nsys sample](/04-profiler/nsys-sample.png)

---

## PyTorch dev-discuss

- [source](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)

```python
import torch
import torch.nn as nn
import torchvision.models as models

# setup
device = 'cuda:0'
model = models.resnet18().to(device)
data = torch.randn(64, 3, 224, 224, device=device)
target = torch.randint(0, 1000, (64,), device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nb_iters = 20
warmup_iters = 10
for i in range(nb_iters):
    optimizer.zero_grad()

    # start profiling after 10 warmup iterations
    if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()

    # push range for current iteration
    if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))

    # push range for forward
    if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
    output = model(data)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    loss = criterion(output, target)

    if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    if i >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
    optimizer.step()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    # pop iteration range
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()
```

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -o my_profile python main.py
```



---

## Profiling a PyTorch Script with `nsys`

### 1. Emit every autograd kernel as an NVTX range

```python
with torch.autograd.profiler.emit_nvtx():
    for step in range(5):
        torch.cuda.nvtx.range_push(f"step_{step}")
        _ = model(inputs)
        torch.cuda.nvtx.range_pop()
```

### 2. Annotate PyTorch Code with NVTX

Adding NVTX ranges makes the timeline far easier to read — each range appears as a labelled band on the CPU row, aligned with the GPU kernels it launched.

![pytorch nvtx](/04-profiler/pytorch-nvtx.png)

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        return self.fc2(self.fc1(x))

device = "cuda"
model = SimpleModel().to(device)
x = torch.randn(64, 1024, device=device)

# Warmup — keeps lazy init out of the profiled range
for _ in range(3):
    _ = model(x)

torch.cuda.synchronize()

for step in range(5):
    torch.cuda.nvtx.range_push(f"step_{step}")

    torch.cuda.nvtx.range_push("forward")
    y = model(x)
    torch.cuda.nvtx.range_pop()  # forward

    torch.cuda.nvtx.range_push("backward")
    y.sum().backward()
    torch.cuda.nvtx.range_pop()  # backward

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()  # step_N
```

### 3. Mark the Start and End of the Profiling Session

- There're multiple way to mark the start and end of the profiling session. All of them are equivalent, and basically doing the same thing: calling `cudaProfilerStart()` and `cudaProfilerStop()` under the hood.

#### Option 1: Use `torch.cuda.profiler.start()` / `torch.cuda.profiler.stop()`

```python
import torch

torch.cuda.profiler.start()
for step in range(5):
    torch.cuda.nvtx.range_push(f"forward_{step}")
    y = model(x)
    torch.cuda.nvtx.range_pop()  # forward
torch.cuda.profiler.stop()
```

#### Option 2: Use `torch.cuda.cudart()`

```python
import torch
from torch.cuda import check_error, cudart

check_error(cudart().cudaProfilerStart())
for step in range(5):
    torch.cuda.nvtx.range_push(f"forward_{step}")
    y = model(x)
    torch.cuda.nvtx.range_pop()  # forward
check_error(cudart().cudaProfilerStop())
```

#### ✅ Option 3: Use `with torch.cuda.profiler.profile()`

```python
import torch

with torch.cuda.profiler.profile():
    for step in range(5):
        torch.cuda.nvtx.range_push(f"forward_{step}")
        y = model(x)
        torch.cuda.nvtx.range_pop()  # forward
```

> [!IMPORTANT]
> The truth is, all of the above options are same lol, so prefer `with torch.cuda.profiler.profile()` for simplicity.
> check code here: https://github.com/pytorch/pytorch/blob/v2.13.0/torch/cuda/profiler.py

---

### 3. Run the Profiler

```bash
nsys profile \
  --trace cuda,nvtx,cudnn,cublas,osrt \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  --output profile_report \
  python train.py
```

This produces `profile_report.nsys-rep`, which can be opened in the **Nsight Systems GUI**.

- We can use `-t` for `--trace`, & `-o` for `--output` to shorten the command.
- `capture-range` and `capture-range-end` tell nsys to not profile the entire process, but only the region between `cudaProfilerStart()` and `cudaProfilerStop()`.

### 3. Key Flags

| Flag | Purpose |
|---|---|
| `--trace cuda,nvtx,osrt` | Collect CUDA API calls, NVTX ranges, and OS runtime (thread scheduling) |
| `--trace cuda,nvtx,cudnn,cublas` | Also include cuDNN and cuBLAS library traces |
| `--capture-range cudaProfilerApi` | Only record between `cudaProfilerStart` / `cudaProfilerStop` calls |
| `--capture-range-end stop` | Stop the entire profiling session on `cudaProfilerStop` |
| `--output <file>` | Write a `.nsys-rep` report file |
| `--stats true` | Print a kernel summary table to stdout after profiling |
| `--gpu-metrics-device all` | Collect GPU hardware counters (SM active, DRAM BW) alongside the timeline |

### 4. Get a Quick Summary Without the GUI

```bash
nsys stats profile_report.nsys-rep
```

This prints ranked tables of CUDA kernel time, CUDA API time, and memory operation time directly to the terminal — useful on remote servers.

---

## Recommended Checklist

- **Always warmup** — PyTorch's caching allocator, cuDNN auto-tuner, and `torch.compile` all run on the first iteration. Profile the steady state.
- **Use `--capture-range`** — Profiling the full process (including Python import and model init) buries the useful signal under noise.
- **Add NVTX ranges** — Without them, the timeline shows hundreds of unlabelled kernels. Ranges let you instantly correlate GPU work to your Python code.
- **Check the CPU row first** — If GPU kernels have large gaps, the bottleneck is CPU-side (Python overhead, DataLoader, kernel launch latency), not the kernels themselves.

---

## Reading the Report in the GUI

Open `profile_report.nsys-rep` in the **NVIDIA Nsight Systems** desktop application. Key rows in the timeline:

- **NVTX** — Your labelled ranges; click to select the time window and filter the kernel list below.
- **CUDA HW (GPU)** — Actual kernel execution bars on the GPU.
- **CUDA API (CPU)** — `cudaLaunchKernel`, `cudaMemcpy`, etc. — gaps here indicate CPU-side overhead.
- **Threads** — OS thread scheduling; useful for diagnosing DataLoader worker contention.

Key workflow in the GUI:
1. Select an NVTX range (e.g. `forward`) to zoom in.
2. Right-click → **Filter to Selection** to see only kernels in that window.
3. Open **Analysis → CUDA Summary** for an aggregated kernel table sorted by total GPU time.

---

## 🚀 setup `nsys` on lightning studio

- check `nvcc` version:
```bash
nvcc --version
```

- note its version, for me: `Build cuda_13.0.r13.0/compiler.36424714_0`

- now search for `nsight-systems` package for that version:
```bash
⚡ ~ apt-cache search nsight-systems | grep 13

cuda-nsight-systems-13-0 - NVIDIA Nsight Systems
cuda-nsight-systems-13-1 - NVIDIA Nsight Systems
cuda-nsight-systems-13-2 - NVIDIA Nsight Systems
cuda-nsight-systems-13-3 - NVIDIA Nsight Systems
```

- so i'll download `cuda-nsight-systems-13-0` package and install it:
```bash
⚡ ~ sudo apt install -y cuda-nsight-systems-13-0
```

- and finally check if `nsys` is installed:
```bash
⚡ ~ nsys --version
```

---

## Case Study

```python

cases = [{"outputDim":1, "bias":True}, {"outputDim":1, "bias":False}, {"outputDim":8, "bias":True}]

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1, bias=True) # update based on cases
        )

    def forward(self, x):
        return self.layers(x)
```

## 🧪 Case Study: 3 ops → 4 kernels

Profiled a tiny model (`Linear(8,16) → ReLU → Linear(16,1)`, batch 4) and the trace showed **4 kernel launches per step**, not 3. Everything below came from chasing that one extra kernel.

| # | Kernel in nsys | What it actually is |
|---|---|---|
| 1 | `gemmSN_TN_kernel` | Linear 1 matmul (cuBLASLt), **bias fused in epilogue** |
| 2 | `vectorized_elementwise_kernel` | ReLU (float4 loads) |
| 3 | `elementwise_kernel` | **bias broadcast-copy** for Linear 2 (the mystery kernel) |
| 4 | `gemvx::kernel` | Linear 2 matmul — a **gemv**, not gemm, because N=1 |

### Reading kernel names

Demangled cuBLAS names look like garbage:

```
std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, ...>
```

Recipe:
1. **Strip C++ noise** — `std::enable_if<...>::type`, `void`, `internal::` are template machinery, zero perf meaning.
2. **Find the family** — `gemm*` (matrix-matrix), `gemv*` (matrix-vector), `elementwise_kernel`, `vectorized_elementwise_kernel`, `reduce_kernel`, `triton_poi_fused_*` (Inductor pointwise), `cutlass::`, `flash_*`.
3. **Read the dtypes** in template args — catches accidental fp32 fallbacks.
4. **Namespace = source** — `at::native::` = PyTorch, `internal::`/`cublas*` = cuBLAS, `cudnn::` = cuDNN, `triton_*` = Inductor codegen.

Also: `TN` in `gemmSN_TN` = transpose config (op(A)=T, op(B)=N). PyTorch stores weight as `(out, in)` and computes `x @ Wᵀ` via a **stride-swapped view** — no transpose kernel ever launches, the "T" is just metadata.

### Bias: epilogue fusion vs the beta trick

Every BLAS matmul computes `C = alpha·(A@B) + beta·C`, not just `A@B`. Two ways `nn.Linear` gets its `+ bias`:

**Epilogue fusion (1 kernel)** — cuBLASLt (`cublasLtMatmul`) supports epilogues: cheap elementwise ops (bias add, relu) applied in the kernel's final write-out phase, while the result is still in registers. Free — the memory write was happening anyway.

**Beta trick (2 kernels)** — classic cuBLAS gemv has no epilogue. So:
1. small `elementwise_kernel` broadcast-copies bias into the output buffer
2. gemv runs with `beta=1` → accumulates `x@Wᵀ` on top of the bias already sitting there

`aten::addmm` picks the path at dispatch time based on shape/layout. My Linear 2 was `(4,16)@(16,1)` → N=1 → routed to gemv → no epilogue available → beta trick → 4th kernel.

### Experiments (change one variable, re-trace)

| Config | Linear 2 route | Bias handling | Kernels/step |
|---|---|---|---|
| `(16,1)`, bias=True | gemv | copy kernel + beta=1 | **4** |
| `(16,1)`, bias=False | gemv | — | **3** |
| `(16,8)`, bias=True | gemm | Lt epilogue, fused | **3** |

So it was never "bias costs a kernel" — it's "**the gemv path costs a kernel when bias is present**". Same ops, one dim changed, one kernel gone.

> [!IMPORTANT]
> `vectorized_elementwise_kernel` vs `elementwise_kernel`: the vectorized one uses `float4` loads (4 floats per instruction) — requires contiguous, 16-byte-aligned, size divisible by 4. TensorIterator picks per launch. ReLU on 64 contiguous floats qualified; the 4-float bias copy didn't.

### torch.compile findings

| Mode | What it bundles | What I saw |
|---|---|---|
| `default` | Inductor codegen | still 3 kernels — matmuls go **extern** to cuBLAS, Triton only generates the pointwise glue (`triton_poi_fused_addmm_relu_0` = Linear 1's bias + relu, NOT the matmul) |
| `reduce-overhead` | default + **cudagraphs** | whole step = 1 `cudaGraphLaunch` replay + 1 `multi_tensor_apply` input-copy |
| `max-autotune` | default + Triton GEMM templates + **cudagraphs too** | on T4: `Not enough SMs to use max_autotune_gemm mode` (needs ≥68 SMs, T4 has 40) → effectively reduce-overhead |
| `max-autotune-no-cudagraphs` | autotune without graphs | **back to 4 kernels lol** — see below |

Key facts:
- `triton_poi_fused_addmm_relu_0` naming: lists the source ops that contributed *fragments*. The `addmm` part is only the bias-add residue — the mm went extern. Kernel does NOT do a matmul.
- cudagraphs replay reads from **frozen static buffers** → inputs get copied in each step (`aten::_foreach_copy` → `multi_tensor_apply_kernel`). Cost scales with input size. Shape change = re-capture.
- The autotune log is readable:
  ```
  AUTOTUNE addmm(4x8, 4x16, 16x8)
    addmm      0.0864 ms  100.0%   ← winner
    bias_addmm 0.1249 ms   69.2%
  ```
  Both are cuBLAS variants (not Triton). It picked plain `addmm` — the **2-kernel** beta-trick path — over the 1-kernel epilogue variant, because it measured faster at this shape. **The autotuner optimizes time, not kernel count. Kernel count is a proxy metric, never the objective.**

### The receipt: `TORCH_LOGS="output_code"`

```bash
TORCH_LOGS="output_code" python main.py 2> output_code.log
```

Dumps the Python that Inductor actually generated. Everything above is verifiable in ~10 lines of the `call()` function:

```python
extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (8, 16), (1, 8), 0), out=buf0)  # L1 mm, stride-swap = transpose
triton_poi_fused_addmm_relu_0.run(buf1, arg1_1, 64, stream=stream0)                  # L1 bias + relu, in-place
extern_kernels.addmm(reinterpret_tensor(arg4_1, (4, 8), (0, 1), 0), buf1, ..., beta=1, out=buf2)  # L2
```

- The L2 bias is passed as a **stride `(0, 1)` broadcast view** — non-contiguous `self` → aten can't use the Lt epilogue → beta trick → 2 kernels from 1 extern call. An extern call promises *at least* one launch, not exactly one.
- The `# Source Nodes: [...] Original ATen: [...]` comments are Inductor's fusion ledger: rich annotation = fused Triton kernel, empty annotation = library passthrough (trace has the final word on its cost).
- `assert_size_stride(...)` at the top of `call()` = what Dynamo guards look like; per-call cost that cudagraphs mode skips.

### Lessons

1. Python code says **what** to compute; dispatch (aten/cuBLAS/Inductor) decides **how many kernels** that becomes; only the trace shows the truth.
2. Shape and layout (strides!) decide routing — N=1 vs N=8 flipped gemv↔gemm, a stride-0 view killed epilogue fusion.
3. Unfused epilogues have a trace signature: **suspicious tiny elementwise kernel glued to a big compute kernel**. That pattern = fusion opportunity.
4. Tiny-kernel workloads are **launch-bound**: kernels ~7 µs, steps ~100+ µs. Signature: sparse slivers on the kernel row, CPU NVTX ≫ GPU NVTX. Fix is cudagraphs, not faster kernels.
5. Debug loop: anomaly in trace → hypothesis → one-variable experiment or `TORCH_LOGS="output_code"` → confirm. Repeat.