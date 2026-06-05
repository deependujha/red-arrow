---
title: Learning Nsight
type: docs
sidebar:
  open: false
weight: 400
---

## 1. Visualize memory access and kernel execution

```cpp
// filename: memory_test.cu
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

---

## 3. CUDA Streams & Memory Overlap Visualization

```cpp
// filename: stream_test.cu

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdio.h>

#define VECTOR_SIZE (1 << 24) // Large vector
#define NUM_STREAMS 4

__global__ void intense_math_kernel(float *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Just some math to keep the GPU busy for a moment
        float val = in[idx];
        for(int i = 0; i < 200; i++) {
            val = rsqrtf(val + 1.0f) + 0.5f;
        }
        out[idx] = val;
    }
}

int main() {
    size_t bytes = VECTOR_SIZE * sizeof(float);
    
    // Page-locked (pinned) host memory is REQUIRED for async memcpys
    float *h_in, *h_out;
    cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault);
    
    // Initialize host data
    for(int i = 0; i < VECTOR_SIZE; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    int threads = 256;
    int blocks = (VECTOR_SIZE + threads - 1) / threads;

    // ==========================================
    // APPROACH 1: Synchronous (The Default Way)
    // ==========================================
    nvtxRangePushA("APPROACH_1_SYNCHRONOUS");
    
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    intense_math_kernel<<<blocks, threads>>>(d_out, d_in, VECTOR_SIZE);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    nvtxRangePop();

    // ==========================================
    // APPROACH 2: Asynchronous (The Streams Way)
    // ==========================================
    nvtxRangePushA("APPROACH_2_STREAMS");

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&streams[i]);

    int chunk_size = VECTOR_SIZE / NUM_STREAMS;
    size_t chunk_bytes = chunk_size * sizeof(float);

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        
        // Use cudaMemcpyAsync and pass the specific stream
        cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        
        int intense_blocks = (chunk_size + threads - 1) / threads;
        intense_math_kernel<<<intense_blocks, threads, 0, streams[i]>>>(d_out + offset, d_in + offset, chunk_size);
        
        cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    return 0;
}
```

- compile and generate nsys profile (since we're not interested in invididual kernel metric, but rather overlap of kernel and memory transfer)

```bash
# 1. Compile
nvcc -O3 -lineinfo stream_test.cu -o stream_test

# 2. Profile with nsys
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=streams_study ./stream_test
```

![nsys stream overlap](/04-profiler/nsys-stream-overlap.png)

---

## 4. Occupancy and Resource Constraints

- `Occupancy` is the ratio of the number of active warps per Streaming Multiprocessor (SM) to the maximum number of possible active warps the SM can physically support.
> If your kernel asks for too many registers or too much shared memory per block, the GPU cannot fit many blocks onto the hardware at the same time, leaving performance on the table.

```cpp
// filename: occupancy_test.cu
#include <cuda_runtime.h>
#include <stdio.h>

// A kernel that forces the compiler to use a ton of registers 
// by declaring many variables and doing dependent math.
__global__ void high_register_kernel(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a1 = in[idx];
        float a2 = a1 * 2.0f; float a3 = a2 + 3.0f; float a4 = a3 / 4.0f;
        float a5 = a4 - 5.0f; float a6 = a5 * 6.0f; float a7 = a6 + 7.0f;
        float a8 = a7 / 8.0f; float a9 = a8 - 9.0f; float a10 = a9 * 10.0f;
        
        // Volatile array forces registers to stay alive and not be optimized away
        volatile float local_arr[10];
        local_arr[0] = a1; local_arr[1] = a2; local_arr[2] = a3; local_arr[3] = a4;
        local_arr[4] = a5; local_arr[5] = a6; local_arr[6] = a7; local_arr[7] = a8;
        local_arr[8] = a9; local_arr[9] = a10;

        out[idx] = local_arr[idx % 10] + a10;
    }
}

int main() {
    int N = 1 << 20;

    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Launching with large block size to maximize resource pressure per SM
    // int threads = 256;
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    high_register_kernel<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

- compile once with `256` threads per block and once with `1024` threads per block.

```bash
# 1. Compile
# For 256 threads per block
nvcc -O3 -lineinfo occupancy_test.cu -o occupancy_256

# For 1024 threads per block
# nvcc -O3 -lineinfo occupancy_test.cu -o occupancy_1024

# 2. Profile with nsys (use the one with 256 threads for best results)
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=occupancy_study_256 ./occupancy_256
```

- check for `launch statistics` & `occupancy`

![occupancy 256](/04-profiler/occupancy_256.png)

![occupancy 1024](/04-profiler/occupancy_1024.png)

| Block Size (Threads) | Warps Per Block (block_size/32) | Max Blocks Per SM | Achieved Occupancy | Why? |
| --- | --- | --- | --- | --- |
| 256 | 8 | 4 | 88.07% | Smaller blocks give the hardware scheduler more flexibility to hide latency. |
| 1024 | 32 | 1 | 77.62% | One giant block monolithicly dominates the SM; any tail latency drops occupancy. |

> A Tesla T4 SM can physically hold a maximum of 32 active warps simultaneously, so if there're 8 warps per block, a single SM can hold at max (32/8=4) blocks simultaneously. For 1024 threads per block, a single SM can hold at max (32/32=1) block simultaneously.

> [!INFO]
> - **Registers Per Thread**: In both runs, it says 16. Even though we wrote a bunch of code variables to force register pressure, the NVCC compiler is incredibly smart. It looked at the math, realized many variables were just intermediate steps, and aggressively optimized them away, packing everything into just 16 registers per thread.

> [!CAUTION]
> - **High theoretical occupancy doesn't always equal max performance**
> - but a block size of 128, 256, or 512 threads is generally the "sweet spot" for modern GPUs because it gives the hardware scheduler enough distinct blocks to play with.**

