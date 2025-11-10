---
title: CUDA APIs
type: docs
prev: docs/02-pytorch-compiler
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

```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    ... # enqueue ops on this stream
```

Manual GPU scheduling. Interview gold if you can explain when you'd use it
(e.g., overlap env step + inference in RL).

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
