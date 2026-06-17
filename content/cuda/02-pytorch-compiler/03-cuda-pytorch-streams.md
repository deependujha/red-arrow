---
title: CUDA Events & Streams in PyTorch
type: docs
math: true
prev: docs/02-pytorch-compiler/01-concepts
next: docs/02-pytorch-compiler/03-compiler_apis
sidebar:
  open: false
weight: 203
---

A **CUDA Stream** is a linear sequence of execution commands sent to the GPU. Commands queued within the same stream always execute in order (First-In, First-Out). Commands queued in *different* streams can execute concurrently, allowing you to overlap independent computation and data transfers.

check pr: https://github.com/Lightning-AI/pytorch-lightning/pull/21746/changes

---

## 1. Core Mechanics: Stream Behavior

By default, PyTorch executes all operations on a single global stream per device.

```text
Default Stream (Stream 0):

┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Memory Copy (H2D) │ ──> │   Triton Kernel   │ ──> │ Memory Copy (D2H) │
└───────────────────┘     └───────────────────┘     └───────────────────┘

              Executes in strict sequential order
```

### Multi-Stream Concurrency

When you create custom streams, the GPU scheduler can run operations in parallel if hardware resources (Streaming Multiprocessors, Copy Engines) are available.

```text
Stream 1:

┌───────────────────────┐
│     Matrix Mult 1     │
└───────────────────────┘
            ▲
            │ Concurrent Execution
            ▼

┌───────────────────────┐
│ Triton Custom Kernel  │
└───────────────────────┘

Stream 2
```

Important:

- Operations inside a stream always execute in order.
- Operations across streams may execute concurrently.
- Concurrency is not guaranteed; it depends on GPU resources and kernel characteristics.

---

## 2. Managing Streams in PyTorch

### Fetching the Active Stream

```python
import torch

if torch.cuda.is_available():
    # Active stream for current device
    current_stream = torch.cuda.current_stream()

    # Default stream (Stream 0)
    default_stream = torch.cuda.default_stream()

    # Active stream for GPU 1
    gpu_1_stream = torch.cuda.current_stream(device=1)
```

---

### Creating and Activating Custom Streams

The recommended approach is using a context manager.
> - capital Stream (class) => create new stream
> - small stream (function) => context manager to use that stream

```python
custom_stream = torch.cuda.Stream()

with torch.cuda.stream(custom_stream):
    x = torch.randn(4096, 4096, device="cuda")
    y = torch.matmul(x, x)

# Previous stream is automatically restored
```

Internally this is equivalent to `cuda code`:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaSetDevice(...);
cudaLaunchKernel(..., stream);
```

---

## 3. Synchronization

CUDA execution is asynchronous.

The CPU submits work and immediately continues execution.

```text
CPU:
Launch Kernel ──> Continue Running

GPU:
              Kernel Executes
```

Because of this, synchronization is often required.

---

### Host-to-Device Synchronization (Blocks CPU)

#### Synchronize Entire Device

```python
torch.cuda.synchronize()
```

Equivalent:

```cpp
cudaDeviceSynchronize();
```

This blocks the CPU until **all streams on the device** complete.

---

#### Synchronize a Specific Stream

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    y = torch.matmul(x, x)

stream.synchronize()
```

Equivalent:

```cpp
cudaStreamSynchronize(stream);
```

Only waits for that stream.

---

## 4. CUDA Events

A CUDA Event is a lightweight synchronization marker inserted into a stream.

Once all work before the event finishes, the event becomes complete.

Think of an event as a checkpoint.

```text
Stream 1:

Kernel A ──> Kernel B ──> [ Event ] ──> Kernel C
                              ▲
                              │
                              │ Wait
                              ▼

Stream 2:

                  Kernel D ──> Kernel E
```

Events are used for:

- Stream-to-stream dependencies
- Measuring execution time
- Fine-grained synchronization

---

### Creating and Recording Events

```python
stream = torch.cuda.Stream()
event = torch.cuda.Event()

with torch.cuda.stream(stream):
    output = torch.matmul(x, x)

    event.record()
```

Equivalent:

```cpp
cudaEventRecord(event, stream);
```

The event becomes complete only after all previous work in that stream finishes.

---

### Waiting on an Event from Another Stream

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

event = torch.cuda.Event()

with torch.cuda.stream(stream1):
    tensor = torch.matmul(x, x)
    event.record()

stream2.wait_event(event)

with torch.cuda.stream(stream2):
    result = tensor + 1
```

Equivalent:

```cpp
cudaStreamWaitEvent(stream2, event);
```

This creates a **GPU-side dependency**.

The CPU never blocks.

---

### Waiting on an Event from the CPU

```python
event = torch.cuda.Event()

with torch.cuda.stream(stream):
    y = torch.matmul(x, x)
    event.record()

event.synchronize()
```

Equivalent:

```cpp
cudaEventSynchronize(event);
```

Only waits until that specific event completes.

---

### Measuring Kernel Execution Time

Events are the standard CUDA timing mechanism.

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

y = torch.matmul(x, x)

end.record()
end.synchronize()

print(start.elapsed_time(end), "ms")
```

Equivalent:

```cpp
cudaEventElapsedTime(...);
```

---

## 5. Stream-to-Stream Dependencies

### wait_stream()

PyTorch provides a convenience API.

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    tensor = torch.matmul(x, x)

stream2.wait_stream(stream1)

with torch.cuda.stream(stream2):
    result = tensor + 1
```

Important:

`wait_stream()` does **not** create a permanent dependency between streams.

Internally, PyTorch records an event on `stream1` and then makes `stream2` wait for that event.

Conceptually:

```python
event = torch.cuda.Event()

event.record(stream1)
stream2.wait_event(event)
```

For fine-grained control, production systems often use explicit events.

---

## 6. Event vs Stream Synchronization

| API | CPU Blocks? | Scope |
|------|------|------|
| `torch.cuda.synchronize()` | Yes | Entire device |
| `stream.synchronize()` | Yes | One stream |
| `event.synchronize()` | Yes | One event |
| `stream.wait_stream(other)` | No | Stream dependency |
| `stream.wait_event(event)` | No | Event dependency |

---

### Rule of Thumb

```text
Need host synchronization?
    -> synchronize()

Need GPU-to-GPU dependency?
    -> wait_event()

Need timing?
    -> Event(enable_timing=True)
```

> [!IMPORTANT]
> Commands in the same stream execute in order. Commands in different streams may execute concurrently. Actual parallel execution depends on available GPU resources.
>
> Modern PyTorch uses per-thread default streams, so the default stream is not automatically a global synchronization point. Explicit synchronization should be expressed with events or stream dependencies.

---

## 7. Overlapping Data Transfer and Computation

One of the most important CUDA optimization patterns.

To overlap CPU→GPU transfers with compute:

1. Use pinned memory
2. Use non_blocking=True
3. Use a separate stream

```python
compute_stream = torch.cuda.current_stream()
copy_stream = torch.cuda.Stream()

pinned_cpu_tensor = torch.randn(2048, 2048).pin_memory()

with torch.cuda.stream(copy_stream):
    gpu_tensor = pinned_cpu_tensor.to(
        "cuda",
        non_blocking=True,
    )

with torch.cuda.stream(compute_stream):
    torch.matmul(
        existing_gpu_data,
        existing_gpu_data,
    )

compute_stream.wait_stream(copy_stream)

output = gpu_tensor + 5
```

Without streams:

```text
Copy -> Compute -> Copy -> Compute
```

With streams:

```text
Copy Req2
          \
           \

Compute Req1
            \
             \
              Compute Req2
```

This improves overall throughput.

---

## 8. Interfacing with Triton and CUDA Extensions

Triton automatically launches kernels on the currently active PyTorch stream.

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    my_triton_kernel[grid](...)
```

Sometimes low-level APIs require the raw CUDA stream handle.

```python
stream = torch.cuda.current_stream()

raw_stream_pointer = stream.cuda_stream
```

Example:

```python
my_custom_extension.launch(raw_stream_pointer)
```

This is common when integrating:

- CUDA extensions
- Triton kernels
- NCCL communication
- Custom drivers

---

## 9. Real-World Interview Example: LLM Inference Pipeline

A common interview question:

> How would you use CUDA streams to improve inference throughput?

Consider an LLM server.

Each request requires:

1. Copy tokens CPU → GPU
2. Run attention kernels
3. Run MLP kernels
4. Copy outputs GPU → CPU

Naive execution:

```text
Time ─────────────────────────►

Copy Req1
           Compute Req1
                          Copy Back Req1
                                         Copy Req2
                                                    Compute Req2
```

The GPU spends time idle waiting for transfers.

---

### Stream-Based Pipeline

Use separate streams:

```text
Copy Stream:

Req2 H2D ─────────────────────►


Compute Stream:

Req1 Attention
Req1 MLP
                     Req2 Attention
                     Req2 MLP


Output Stream:

Req1 D2H
                          Req2 D2H
```

Implementation:

```python
copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()

copy_done = torch.cuda.Event()

with torch.cuda.stream(copy_stream):
    next_batch = cpu_batch.to(
        "cuda",
        non_blocking=True,
    )
    copy_done.record()

compute_stream.wait_event(copy_done)

with torch.cuda.stream(compute_stream):
    logits = model(next_batch)
```

Benefits:

- PCIe transfer overlaps with compute
- Higher GPU utilization
- Higher request throughput
- Lower latency under load
- CPU remains available for scheduling

This pattern appears in:

- vLLM
- TensorRT-LLM
- DeepSpeed Inference
- Recommendation systems
- Large-scale inference servers

---

## 10. Production Gotchas Checklist

### Multi-GPU Hazard

`current_stream()` is device specific.

```python
with torch.cuda.device(1):
    stream = torch.cuda.current_stream()
```

Always ensure you're operating on the intended device.

---

### Pinned Memory Requirement

This:

```python
tensor.to(
    "cuda",
    non_blocking=True,
)
```

does not guarantee overlap.

For asynchronous H2D transfers:

```python
tensor = tensor.pin_memory()
```

must be used.

---

### Implicit Synchronization

The following operations force synchronization:

```python
tensor.item()
tensor.cpu()
print(tensor)
```

These can destroy stream overlap and pipeline efficiency.

---

### Avoid Global Synchronization

Avoid:

```python
torch.cuda.synchronize()
```

inside performance-critical code.

Prefer:

```python
stream.wait_event(event)
```

or

```python
stream.wait_stream(other_stream)
```

so synchronization remains on the GPU.

---

## Interview Takeaways

A strong interview answer should include:

1. Operations inside a stream execute in order.
2. Different streams may execute concurrently.
3. CUDA execution is asynchronous with respect to the CPU.
4. Events are used to express dependencies between streams.
5. `wait_event()` is generally preferred over host-side synchronization.
6. Pinned memory + non-blocking copies allow overlap of transfers and compute.
7. **`Real systems often separate copy, compute, and communication work into different streams`**.

A concise summary:

> Streams provide concurrency. Events provide coordination. High-performance CUDA systems minimize CPU synchronization and express dependencies directly on the GPU.

