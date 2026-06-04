---
title: Learning Nsight
type: docs
sidebar:
  open: false
weight: 400
---

## 1. Visualize memory access and kernel execution

```cpp
// file name: memory_test.cu
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdio.h>

#define MATRIX_SIZE 4096

// BAD: Each thread reads with a stride of MATRIX_SIZE
__global__ void uncoalesced_read(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Stride access patterns force multiple memory requests
        out[idx] = in[(idx * 32) % N]; 
    }
}

// GOOD: Consecutive threads read consecutive memory addresses
__global__ void coalesced_read(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];
    }
}

int main() {
    int N = MATRIX_SIZE * MATRIX_SIZE;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Profile Uncoalesced
    nvtxRangePushA("Uncoalesced Kernel Run");
    uncoalesced_read<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Profile Coalesced
    nvtxRangePushA("Coalesced Kernel Run");
    coalesced_read<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    nvtxRangePop();

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
```

- Compile the code and use `-lineinfo` for getting the line number information in the trace.

```bash
nvcc -O3 -lineinfo memory_test.cu -o memory_test
```

- generate trace files for analyzing the profilers.

```bash
# Get the broad timeline with NVTX markers
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=timeline_study ./memory_test

# Profile the exact hardware metrics of both kernels
ncu --set full -o kernel_study ./memory_test
```

- download and load in `nsys` & `ncu`.

![nsy](/04-profiler/nsys-01.png)
![ncu](/04-profiler/ncu-01.png)

---

## 2. Visualize shared memory and bank conflict

```cpp
// filename: shared_test.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

// BAD: Stride causes multiple threads to hit the exact same shared memory banks
__global__ void conflict_kernel(float *out, float *in) {
    __shared__ float s_data[BLOCK_SIZE * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    // Load data cleanly
    s_data[tid] = in[tid];
    __syncthreads();
    
    // Read with a stride of 32 -> Bank Conflict!
    // Thread 0 reads s_data[0] (Bank 0)
    // Thread 1 reads s_data[32] (Bank 0)
    // Thread 2 reads s_data[64] (Bank 0)... 32-way conflict!
    out[tid] = s_data[tid * BLOCK_SIZE];
}

// GOOD: Linear access utilizes parallel banks perfectly
__global__ void clean_kernel(float *out, float *in) {
    __shared__ float s_data[BLOCK_SIZE * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    s_data[tid] = in[tid];
    __syncthreads();
    
    // Sequential read -> No conflicts
    out[tid] = s_data[tid];
}

int main() {
    size_t bytes = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    conflict_kernel<<<1, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    clean_kernel<<<1, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

- commands:

```bash
# compile
nvcc -O3 -lineinfo shared_test.cu -o shared_test

# Generate trace files for analyzing the profilers
# Get the broad timeline with NVTX markers
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=02_timeline_study ./shared_test

# Profile the exact hardware metrics of both kernels
ncu --set full -o 02_kernel_study ./shared_test
```

![ncu bank conflict 1](/04-profiler/ncu-02-01.png)
![ncu bank conflict 2](/04-profiler/ncu-02-02.png)

> - we have 32-way bank conflict for `conflict_kernel` and bank-conflict for `clean_kernel` is 0.
> - we can check them in, `details > memory workload analysis`.