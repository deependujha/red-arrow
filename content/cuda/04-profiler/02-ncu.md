---
title: Nsight Compute
type: docs
math: true
prev: cuda/03-cuda-cpp/01-basics.md
sidebar:
  open: false
weight: 402
---

**NVIDIA Nsight Compute** (`ncu`) is a kernel-level profiler that collects detailed hardware performance counters for individual CUDA kernels — SM occupancy, memory throughput, warp efficiency, Tensor Core utilization, and more. Unlike Nsight Systems (`nsys`), which gives a system-wide timeline, `ncu` re-runs each kernel multiple times (replay) to collect all hardware metrics without instrumentation overhead.

---

## Profiling a PyTorch Model with `ncu`

Profiling an entire training loop would generate an enormous and unmanageable report. The standard workflow is to annotate PyTorch code with **NVTX ranges**, then tell `ncu` to only capture kernels within those ranges.

### 1. Annotate PyTorch Code with NVTX

Use `torch.cuda.nvtx.range_push()` / `torch.cuda.nvtx.range_pop()` to bracket the section you want to profile. Always include a warmup iteration *before* the profiled range to avoid capturing PyTorch's lazy initialization and caching-allocator overhead.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

device = "cuda"
model = SimpleModel().to(device)
x = torch.randn(64, 1024, device=device)

# Warmup — keeps initialization out of the profiled range
for _ in range(3):
    _ = model(x)

torch.cuda.synchronize()

# Mark the region ncu should capture
torch.cuda.nvtx.range_push("profile_region")
y = model(x)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
```

### 2. Run the Profiler

```bash
ncu \
  --target-processes all \
  --nvtx \
  --nvtx-include "profile_region" \
  --set full \
  --replay-mode kernel \
  --export profile_report \
  python train.py
```

This produces `profile_report.ncu-rep`, which can be opened in the **Nsight Compute GUI**.

### 3. Key Flags

| Flag | Purpose |
|---|---|
| `--nvtx` | Enable NVTX range filtering |
| `--nvtx-include "name"` | Only record kernels inside the named NVTX range |
| `--target-processes all` | Also profile child processes (needed for `torch.multiprocessing`) |
| `--set full` | Collect the complete set of hardware metrics (SM occupancy, memory BW, Tensor Core utilization, etc.) |
| `--replay-mode kernel` | Re-run each kernel transparently to gather multiple counter sets |
| `--export <file>` | Write a `.ncu-rep` report file for the GUI |

### 4. Isolating Specific Kernels (Optional)

If the report still contains too many PyTorch-internal operations (helper memcopies, index selections, etc.), filter by kernel name with a regex:

```bash
ncu \
  --target-processes all \
  --nvtx \
  --nvtx-include "profile_region" \
  --set full \
  --kernel-name "volta_sgemm|ampere_fp16" \
  --export profile_report \
  python train.py
```

`--kernel-name` accepts a regex matched against the full mangled kernel name.

---

## Recommended Checklist

- **Warmup first** — PyTorch's caching allocator and lazy compilation (`torch.compile`) distort results on the first run.
- **Disable other profiling tools** — Running `torch.profiler` or autograd tracing at the same time causes CUPTI conflicts (`CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`).
- **Isolate the GPU** — Display servers or background CUDA processes on the same GPU corrupt L2 cache metrics.
- **Use `--replay-mode kernel`** — This is the default and most accurate mode; `--replay-mode application` is needed only when kernel state depends on prior kernels.

---

## Reading the Report in the GUI

Open `profile_report.ncu-rep` in the **NVIDIA Nsight Compute** desktop application. Key views:

- **Summary** — per-kernel runtime and throughput at a glance.
- **Roofline** — plots each kernel against the hardware's arithmetic and memory bandwidth ceilings to identify whether it is compute-bound or memory-bound.
- **SM Utilization / Warp State** — shows stall reasons (memory dependency, pipe busy, synchronization, etc.).
- **Memory Workload** — L1/L2 hit rates and DRAM bandwidth utilization.

To profile a PyTorch model at the CUDA kernel level using NVIDIA Nsight Compute (), you must capture specific execution ranges because profiling an entire deep learning training loop will generate massive, unmanageable report files. [1, 2]  
The industry standard workflow relies on  and  to pinpoint exact iterations, which are then isolated via the  command line. [3, 4]  
1. Annotate PyTorch Code 
Isolate the specific forward or backward pass you want to profile. This prevents  from gathering data during the slow model initialization phase. [5]  
2. Execute the Profiler Command 
Launch your script using the  command line interface. Use the  flag to instruct  to ignore all code except what resides between the start and stop lines. [3, 4, 6]  
3. Key Configuration Flags 

• : Tells the profiler to only record kernels executed between  and . 
• : Collects complete hardware metrics including SM occupancy, memory throughput, and tensor core utilization. 
• : Automatically re-runs the kernel transparently to collect multiple hardware performance counters. 
• : Generates a  report file that can be natively opened inside the NVIDIA Nsight Compute GUI. [1, 3, 5, 7, 8]  

4. Isolating Specific Kernels (Optional) 
If your script still records too many internal PyTorch operations (such as helper memory copies or index selections), you can target a specific kernel by matching its string name using regex: [8, 9, 10, 11]  
5. Recommended Optimization Checklist 

• Always include a warmup phase: PyTorch uses caching allocators and lazy compilations () which distort your results on the first run. 
• Turn off tracking tools: Ensure that other performance observers, like the built-in  or PyTorch autograd trace, are disabled to prevent conflict errors (). 
• Isolate your GPU: Make sure no display servers or background processes are running on the target GPU to protect your L2 cache metric calculations. [19]  

Would you like help rewriting your specific PyTorch loop to support NVTX ranges, or do you need assistance reading the Roofline Analysis chart generated inside the Nsight Compute UI? [1, 20]  

AI responses may include mistakes.

[1] https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/
[2] https://www.youtube.com/watch?v=F_BazucyCMw
[3] https://github.com/eth-easl/orion/blob/main/PROFILE.md
[4] https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
[5] https://forums.developer.nvidia.com/t/profiling-pytorch-model-using-ncu/219445
[6] https://help.nscc.sg/wp-content/uploads/ASPIRE2APLUS_ADVANCED_WORKSHOP_PROFILING-AND-DEBUGGING_TRAINING-GUIDE.pdf
[7] https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
[8] https://forums.developer.nvidia.com/t/nsight-compute-with-pytorch/302353
[9] https://forums.developer.nvidia.com/t/how-to-profile-all-attention-layers-in-bert-inference/310494
[10] https://discuss.pytorch.org/t/profiling-and-tracing-pytorch-code-for-cuda-kernels/217985
[11] https://forums.developer.nvidia.com/t/question-about-profiling-kernel-using-ncu/212544
[12] https://www.youtube.com/watch?v=LuhJEEJQgUM
[13] https://towardsdatascience.com/simple-ways-to-speed-up-your-pytorch-model-training-9c9d4899313d/
[14] https://medium.com/data-science/how-to-accelerate-your-pytorch-training-with-xla-on-aws-3d599bc8f6a9
[15] https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.3/release-notes/index.html
[16] https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/profiling-pytorch-code
[17] https://discuss.pytorch.org/t/nsight-compute-cupti-error/192883
[18] https://forums.developer.nvidia.com/t/i-get-different-time-in-ncu-and-pytorch-prolifer/237921
[19] https://forums.developer.nvidia.com/t/ncu-profiling-with-cache-control/246113
[20] https://docs.alcf.anl.gov/polaris/data-science/profiling_dl/

