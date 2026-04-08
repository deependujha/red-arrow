---
title: CUDA Memory Model
type: docs
prev: docs/
sidebar:
  open: false
weight: 303
---

Unified Memory is CUDA’s attempt to make CPU–GPU memory look like **one shared address space**. The pointer you get from `cudaMallocManaged` works everywhere. But the real machinery underneath is **page migration**. Memory moves between CPU RAM and GPU VRAM in chunks (pages, typically 4 KB or 64 KB depending on architecture).

Left alone, the system migrates pages **only when they are touched**. That creates GPU page faults, which stall execution. `cudaMemPrefetchAsync` exists to move the data **before the kernel needs it**.

Below is a cleaned-up set of notes suitable for a markdown reference.

---

# CUDA Unified Memory Notes

## 1. Allocate Unified Memory

`cudaMallocManaged` allocates memory accessible from both CPU and GPU.

```cpp
int *data;
int N = 1024;

cudaMallocManaged(&data, N * sizeof(int));
```

Same pointer works everywhere.

```cpp
__global__ void add_one(int* data) {
    int i = threadIdx.x;
    data[i] += 1;
}

int main() {
    add_one<<<1, 1024>>>(data);
    cudaDeviceSynchronize();
}
```

Free memory normally:

```cpp
cudaFree(data);
```

---

# 2. Why Prefetch Exists

Without prefetch:

1. Kernel accesses memory
2. GPU triggers **page fault**
3. Driver migrates page CPU → GPU
4. Kernel resumes

This can happen **thousands of times**.

Prefetch moves memory **ahead of execution**, avoiding faults.

---

# 3. cudaMemPrefetchAsync

Prefetch unified memory to a specific device.

```cpp
cudaMemPrefetchAsync(ptr, size, device_id);
```

Example:

```cpp
int device;
cudaGetDevice(&device);

cudaMemPrefetchAsync(data, N * sizeof(int), device);

add_one<<<1, N>>>(data);
cudaDeviceSynchronize();
```

Meaning:

Move the pages to the **GPU’s VRAM before the kernel runs**.

---

# 4. Prefetch Back to CPU

After GPU work finishes, you can migrate memory back.

```cpp
cudaMemPrefetchAsync(data, N*sizeof(int), cudaCpuDeviceId);
```

Now CPU reads will not trigger page faults.

---

# 5. Multi-GPU Prefetch

Unified memory supports migration between GPUs.

Example system:

```
GPU0
GPU1
CPU
```

Move memory to GPU1:

```cpp
cudaMemPrefetchAsync(data, size, 1);
```

Then run a kernel on GPU1:

```cpp
cudaSetDevice(1);
kernel<<<grid, block>>>(data);
```

The runtime migrates pages **GPU0 → GPU1** if needed.

---

# 6. Multi-GPU Example

```cpp
int *data;
size_t size = N * sizeof(int);

cudaMallocManaged(&data, size);

// initialize on CPU
for(int i = 0; i < N; i++)
    data[i] = i;

// move memory to GPU1
cudaMemPrefetchAsync(data, size, 1);

cudaSetDevice(1);
kernel<<<grid, block>>>(data);

cudaDeviceSynchronize();
```

---

# 7. cudaMemAdvise (placement hints)

You can guide CUDA's migration strategy.

Example: prefer GPU memory.

```cpp
cudaMemAdvise(data,
              size,
              cudaMemAdviseSetPreferredLocation,
              device);
```

Useful hints:

| Advice                              | Meaning                           |
| ----------------------------------- | --------------------------------- |
| `cudaMemAdviseSetPreferredLocation` | preferred device for pages        |
| `cudaMemAdviseSetReadMostly`        | optimize for read-heavy workloads |
| `cudaMemAdviseSetAccessedBy`        | allow access from another GPU     |

---

# 8. Multi-GPU Access Hint

Allow multiple GPUs to access the same memory.

```cpp
cudaMemAdvise(data,
              size,
              cudaMemAdviseSetAccessedBy,
              gpu_id);
```

This avoids repeated migrations in some scenarios.

---

# 9. When Unified Memory Works Well

Good cases:

* complex pointer structures (trees, graphs)
* quick CUDA prototyping
* multi-GPU research workloads
* datasets larger than VRAM

---

# 10. When It’s Not Ideal

High-performance kernels often avoid it because:

* page faults stall warps
* migration latency is unpredictable
* frameworks prefer explicit memory control

Production systems often use:

```
cudaMalloc
cudaMemcpyAsync
custom memory pools
```

for deterministic performance.
