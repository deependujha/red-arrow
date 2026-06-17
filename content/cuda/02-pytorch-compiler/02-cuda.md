---
title: CUDA APIs
type: docs
math: true
prev: docs/02-pytorch-compiler/01-concepts
next: docs/02-pytorch-compiler/03-compiler_apis
sidebar:
  open: false
weight: 202
---

- PyTorch has a rich set of CUDA APIs to interact with NVIDIA GPUs for tensor computations.

```md
5.x Maxwell (old)

6.x Pascal

7.0 Volta

7.5 Turing

8.0 Ampere datacenter (A100)

8.6 Ampere consumer (RTX 30-series)

8.9 Ada datacenter (L4, L40S)

9.0 Hopper (H100)

10.0 Blackwell (B200) (newest)

---

> 5 < 6 < 7 < 8 < 9 < 10 (newest)

> Within the same major:
> - .0 = datacenter flagship
> - .5 or .6 = consumer / variation
> - .9 = fancy recent Ada datacenter bump
```

## Commonly used CUDA APIs in PyTorch

```python
import torch

# Check if CUDA is available & initialized
torch.cuda.is_available() # no CUDA op (torch.cuda.initialized() == False)
torch.cuda.is_initialized() # only after first CUDA op


# Get total number of GPUs
torch.cuda.device_count()

# Get current device index
torch.cuda.current_device()
torch.cuda.set_device(device_index)
torch.cuda.get_device_name(device_index) # default device_index=0

# Get device capability. Returns (major, minor) tuple cuda capability of the device
torch.cuda.get_device_capability(device_index)
```

---

## **Memory Management**

```python
# memory usage stats
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.empty_cache()

# for profiling memory usage
torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
```

- Allocated = currently used by tensors.
- Reserved = grabbed by PyTorch allocator.
- `empty_cache()` clears the **reserved** pool, not the VRAM in use by tensors (common misunderstanding).

---

## **Synchronization / Barriers**

```python
torch.cuda.synchronize()
```

Wait for GPU to finish all work in the queue. Essential for **timing** and debugging async weirdness.

```python
torch.cuda.stream(stream)
```

Used when juggling multiple streams (parallel GPU work).

---

## **Streams**

- To create a new stream, use `torch.cuda.Stream()` and use context manager `with torch.cuda.stream(s):` to enqueue ops on this stream.

```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    ... # enqueue ops on this stream
```

> [!TIP]
> after with block is completed, it doesn't mean gpu operations are completed. It simply means, all the gpu tasks have been queued on stream, and you need to specifically ensure synchronisation before accessing the results.

- To get the current active CUDA stream in PyTorch, use **`torch.cuda.current_stream()`**. If you haven't explicitly set a custom stream, this function automatically returns the **default stream** (also known as the legacy stream or Stream 0) for your current GPU device.

```python
import torch

# Ensure CUDA is available
if torch.cuda.is_available():
    # 1. Get the current active stream
    current_stream = torch.cuda.current_stream()
    print(current_stream)
    # Output example: <torch.cuda.Stream device=cuda:0 id=0x0>

    # 2. Check its properties
    print(f"Device: {current_stream.device}")
    print(f"CUDA pointer/handle: {current_stream.cuda_stream}")

    # 3. Verify it is the default stream
    default_stream = torch.cuda.default_stream()
    print(f"Is current stream the default? {current_stream == default_stream}")
    # Output: True

```

---

### Important Multi-GPU Gotcha

`torch.cuda.current_stream()` is **device-specific**. If you are working on a multi-GPU machine and switch devices, you must pass the device index (or a `torch.device` object) to get the correct stream for that specific card.

```python
# Get the active stream for GPU 1 instead of the default GPU 0
gpu_1_stream = torch.cuda.current_stream(device=1) 

```

### Context: Why this matters for Triton

In Triton, kernels are submitted directly to a CUDA stream. When you launch a Triton kernel without explicitly specifying a stream, Triton automatically sniffs out whatever `torch.cuda.current_stream()` is active at that millisecond and queues your kernel there.

If you ever need to pass it explicitly to a raw driver call or external library, you use `.cuda_stream`:

```python
# Raw underlying CUDA stream pointer integer
stream_handle = torch.cuda.current_stream().cuda_stream 
```

---

## **Events (Timing)**

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# GPU ops
end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)  # ms
```

Correct way to measure GPU execution, not `time.time()`.

---

## **Graphs (CUDA Graphs)**

```python
g = torch.cuda.CUDAGraph()

with torch.cuda.graph(g):
    out = model(static_input)

g.replay()
```

Capture-and-replay to kill Python overhead. Static shapes. Used in modern LLM stacks.

---

## **Manual Tensor Device Moves**

```python
tensor.to("cuda")
tensor.cuda()
tensor.cpu()
```

`.to("cuda")` preferred — consistent API and flexible.

---

## **Seed**

```python
torch.cuda.manual_seed(123)
```

Make GPU RNG predictable. Useful in interviews when they ask “how do you make training reproducible?”

---

## **Low-level goodies you’ll hear but rarely type**

```python
torch.cuda.device(...)
torch.cuda.nvtx.range_push(...)
torch.cuda.nvtx.range_pop()
```

NVTX ranges — marking regions for profilers like Nsight.

---

## 🔥 One-sentence mental model cheat

* **is_available / device / count** → “where am I running?”
* **to("cuda") / cuda()** → move tensors
* **memory_allocated / reserved / empty_cache** → VRAM inspection
* **synchronize / events** → timing & correctness
* **streams / graphs** → advanced perf control
