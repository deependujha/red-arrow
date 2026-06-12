---
title: Grid-Stride Loop
type: docs
math: true
sidebar:
  open: false
weight: 501
---

## The Problem with Monolithic Kernels

```cpp
__global__ void saxpy(int n, float a, float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = a * x[idx] + y[idx];
}
```

Requires launching exactly enough threads to cover `N`. Doesn't scale well and wastes the opportunity to reuse threads.

---

## Grid-Stride Loop

```cpp
__global__ void saxpy(int n, float a, float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // total threads in grid

    for (int i = idx; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}
```

Each thread handles multiple elements, striding by the total thread count. With 4 threads and N=12:

```text
Thread 0 → 0, 4, 8
Thread 1 → 1, 5, 9
Thread 2 → 2, 6, 10
Thread 3 → 3, 7, 11
```

---

## Why Use It

**Scalability** - works for any `N`, even beyond the max grid size.

**Thread reuse** - instead of `ceil(N / blockDim.x)` blocks, launch relative to SMs:

```cpp
int device, numSMs;
cudaGetDevice(&device);
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

kernel<<<32 * numSMs, 256>>>(args...);
```

Fewer blocks means lower scheduling overhead and amortized setup cost.

**Debugging** - serialize to a single thread with `<<<1, 1>>>` for deterministic output and CPU comparison.

**Memory coalescing** - threads in a warp still access consecutive addresses each iteration, so coalescing is fully preserved.

---

## What It Doesn't Do

Grid-stride loops don't affect occupancy. Occupancy is determined by registers/thread, shared memory/block, and block size - not how many elements each thread processes.

---

## Template

```cpp
__global__ void kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        // work on element i
    }
}
```

> [!TIP]
> Use this for essentially all element-wise kernels. It's the NVIDIA-recommended default.
