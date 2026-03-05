---
title: Basics of CUDA C++
type: docs
prev: docs/
sidebar:
  open: false
weight: 302
---

![nvidia software stack](/03-cuda-cpp/nvidia-software-stack.png)

Key libraries: **cuDNN** (deep learning), **cuBLAS** (linear algebra), **cuFFT** (Fourier transforms), **Thrust** (general-purpose STL-like parallel algorithms/data structures).

---

## Fundamentals

`__global__` marks a function as a GPU kernel:

```cpp
__global__ void myKernel() { ... }
```

Launch with the triple chevron (`<<<blocks, threads>>>`):

```cpp
myKernel<<<numBlocks, threadsPerBlock>>>();
```

Grid/block sizes can be 1D, 2D, or 3D via `dim3`:

```cpp
myKernel<<<4, 8>>>();                        // 1D
myKernel<<<dim3(2,2), dim3(4,4)>>>();        // 2D
myKernel<<<dim3(2,2,2), dim3(4,4,4)>>>();   // 3D
```

Built-in index variables inside a kernel:

```cpp
threadIdx.{x,y,z}   // thread index within block
blockIdx.{x,y,z}    // block index within grid
blockDim.{x,y,z}    // block dimensions
gridDim.{x,y,z}     // grid dimensions
```

Global index formulas:
```cpp
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
```

## `triple chevron operator` & `thread/block` indices

```cpp
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(){
    int gridX = gridDim.x;
    int gridY = gridDim.y;
    int gridZ = gridDim.z;

    int blockX = blockDim.x;
    int blockY = blockDim.y;
    int blockZ = blockDim.z;

    int blockIdX = blockIdx.x;
    int blockIdY = blockIdx.y;
    int blockIdZ = blockIdx.z;

    int threadIdX = threadIdx.x;
    int threadIdY = threadIdx.y;
    int threadIdZ = threadIdx.z;

    printf("Grid:(%d,%d,%d) Block:(%d,%d,%d) BlockId:(%d,%d,%d) ThreadId:(%d,%d,%d)\n",
        gridX, gridY, gridZ,
        blockX, blockY, blockZ,
        blockIdX, blockIdY, blockIdZ,
        threadIdX, threadIdY, threadIdZ);
}

int main(){
    std::cout << "Hello World!\n";

    myKernel<<<2,4>>>();

    cudaDeviceSynchronize();   // wait for GPU

    return 0;
}

// ---
// output:
// ⚡ ~/cpp_codes nvcc 01-main.cu 
// ⚡ ~/cpp_codes ./a.out 
// Hello World!
// Grid:(2,1,1) Block:(4,1,1) BlockId:(1,0,0) ThreadId:(0,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(1,0,0) ThreadId:(1,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(1,0,0) ThreadId:(2,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(1,0,0) ThreadId:(3,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(0,0,0) ThreadId:(0,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(0,0,0) ThreadId:(1,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(0,0,0) ThreadId:(2,0,0)
// Grid:(2,1,1) Block:(4,1,1) BlockId:(0,0,0) ThreadId:(3,0,0)
```

---

## Memory

**Allocate on GPU:**
```cpp
int* deviceArray;
cudaMalloc((void**)&deviceArray, 100 * sizeof(int));
// void** because cudaMalloc writes the pointer back into your variable
```

**Copy between CPU ↔ GPU:**
```cpp
cudaMemcpy(dst, src, size_bytes, direction);
// directions: cudaMemcpyHostToDevice | cudaMemcpyDeviceToHost | cudaMemcpyDeviceToDevice
```

**Free:**
```cpp
cudaFree(deviceArray);
```

---

## Sum kernel example

```cpp
#include <iostream>
#include <cuda_runtime.h>

// GPU kernel
// Each thread computes the sum of one element
__global__ void sumKernel(int* arr1, int* arr2, int* sum_arr, int limit){

    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard condition so threads don't go out of bounds
    if (idx < limit){
        sum_arr[idx] = arr1[idx] + arr2[idx];
    }
}

int main(){

    // Host (CPU) arrays
    int arr1[5] = {1,2,3,4,5};
    int arr2[5] = {6,7,8,9,10};
    int sum_arr[5];

    // Device (GPU) pointers
    int* cuda_arr1;
    int* cuda_arr2;
    int* cuda_sum_arr;

    int N = 5;
    int size = N * sizeof(int);

    // Allocate memory on the GPU
    // cudaMalloc requires void** because it writes the pointer value
    cudaMalloc((void**)&cuda_arr1, size);
    cudaMalloc((void**)&cuda_arr2, size);
    cudaMalloc((void**)&cuda_sum_arr, size);

    // Copy input data from CPU → GPU
    cudaMemcpy(cuda_arr1, arr1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_arr2, arr2, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    // dim3 is a CUDA built-in type for 3D dimensions (x,y,z)
    dim3 blockSize(4);
    dim3 gridSize(2);

    // Launch kernel on GPU
    sumKernel<<<gridSize, blockSize>>>(cuda_arr1, cuda_arr2, cuda_sum_arr, N);

    // Check for launch configuration errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cout << "Launch error: " << cudaGetErrorString(launchErr) << std::endl;
    }

    // Wait for GPU to finish and catch runtime errors
    cudaError_t runtimeErr = cudaDeviceSynchronize();
    if (runtimeErr != cudaSuccess) {
        std::cout << "Runtime error: " << cudaGetErrorString(runtimeErr) << std::endl;
    }

    // Copy result from GPU → CPU
    cudaMemcpy(sum_arr, cuda_sum_arr, size, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Sum of arrays: ";
    for (int i = 0; i < N; i++){
        std::cout << sum_arr[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(cuda_arr1);
    cudaFree(cuda_arr2);
    cudaFree(cuda_sum_arr);

    return 0;
}
```

---

## Launch Configuration

```
total_threads = gridSize × blockSize
```

**Standard pattern** to cover N elements:
```cpp
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;  // ceiling division
kernel<<<gridSize, blockSize>>>(...);
```

Always add a **bounds guard** inside the kernel:
```cpp
if (idx < N) { ... }
```

---

## Thread Hierarchy

```
Grid → Blocks → Threads
```

Each block is scheduled onto a single Streaming Multiprocessor (SM).
Multiple blocks can run on the same SM simultaneously depending on resource limits
(registers, shared memory, threads). Blocks are queued and scheduled dynamically as SMs free up. Launching too few blocks leaves SMs idle.

```md
# example
blockSize = 256
threads per SM = 2048
→ SM can run 8 blocks concurrently
```

---

## Block & Grid Limits

| Constraint | Limit |
|---|---|
| Max threads per block | 1024 (architecture dependent) |
| `gridDim.x` | 2³¹ − 1 |
| `gridDim.y/z` | 65535 |

For multidimensional blocks: `blockDim.x * blockDim.y * blockDim.z ≤ 1024`.

---

## Warps

- Threads are grouped into `warps of 32 threads`.
- The GPU scheduler issues instructions at the warp level.
- Block sizes should be **multiples of 32** to avoid wasted lanes in the last warp.
- Common sizes: 128, 256, 512.
- Recommended block sizes are usually 128–256 threads. Most real kernels use `256`. This gives enough warps per block to keep SMs busy while avoiding resource limits.
is this section fine:


## `Synchronization` between threads

- `__syncthreads()`: synchronizes threads within a block. All threads must reach this point before any can proceed. Useful for coordinating shared memory access.

- example: shift each element to the left by one position within a block:

```cpp
__global__ void shiftLeft(int* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >0 && idx < N) {
        int temp = arr[idx];
        __syncthreads(); // ensure all threads have read their value
        arr[idx - 1] = temp; // write to left neighbor
        __syncthreads(); // ensure all threads have written before any read
    }
}
```

---

## Error Checking

CUDA errors can occur in two stages:

1. **Kernel launch errors** (invalid configuration, too many threads per block, etc.)
2. **Runtime errors** (illegal memory access, out-of-bounds, etc.)

Use `cudaGetLastError()` to catch launch errors immediately, and `cudaDeviceSynchronize()` to catch runtime errors.

```cpp
sumKernel<<<gridSize, blockSize>>>(...);

// Check for launch configuration errors
cudaError_t launchErr = cudaGetLastError();
if (launchErr != cudaSuccess) {
    std::cout << "Launch error: " << cudaGetErrorString(launchErr) << std::endl;
}

// Wait for GPU to finish and catch runtime errors
cudaError_t runtimeErr = cudaDeviceSynchronize();
if (runtimeErr != cudaSuccess) {
    std::cout << "Runtime error: " << cudaGetErrorString(runtimeErr) << std::endl;
}
```

Without these checks, CUDA errors may appear much later in the program, which can make debugging difficult.

---

## CUDA error checking `Template` 🔥

A minimal CUDA error-checking macro is basically just **“run a CUDA call → check if it failed → print error and stop.”**

Here’s the smallest useful one.

```cpp
#define CUDA_CHECK(call)                                           \
{                                                                  \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__          \
                  << std::endl;                                    \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}
```

---

### How to use it

Instead of writing:

```cpp
cudaMalloc((void**)&ptr, size);
```

you write:

```cpp
CUDA_CHECK(cudaMalloc((void**)&ptr, size));
```

Example:

```cpp
CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));
```

If anything fails, you'll see something like:

```
CUDA error: invalid argument at main.cu:42
```

The `__FILE__` and `__LINE__` are handy because they show **exactly where the failure happened**.

---

### Handling kernel launches

Kernel launches are a little special because they don't return a `cudaError_t`.

So you check them like this:

```cpp
sumKernel<<<gridSize, blockSize>>>(...);

CUDA_CHECK(cudaGetLastError());       // launch error
CUDA_CHECK(cudaDeviceSynchronize());  // runtime error
```

---

### Why macros are used

Without the macro you'd write this everywhere:

```cpp
cudaError_t err = cudaMemcpy(...);
if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
}
```

Which becomes **painfully repetitive** in real CUDA code.

The macro turns it into one line.

---

### Minimal mental rule

In CUDA programs you usually do:

```
CUDA_CHECK(cudaMalloc(...))
CUDA_CHECK(cudaMemcpy(...))

kernel<<<...>>>(...)

CUDA_CHECK(cudaGetLastError())
CUDA_CHECK(cudaDeviceSynchronize())
```

That pattern alone eliminates **most silent CUDA bugs**.

---

## `Sum Kernel` with Error Checking macro

```cpp
#include <iostream>
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                           \
{                                                                  \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__          \
                  << std::endl;                                    \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

// GPU kernel
// Each thread computes the sum of one element
__global__ void sumKernel(int* arr1, int* arr2, int* sum_arr, int limit){

    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard condition so threads don't go out of bounds
    if (idx < limit){
        sum_arr[idx] = arr1[idx] + arr2[idx];
    }
}

int main(){

    // Host (CPU) arrays
    int arr1[5] = {1,2,3,4,5};
    int arr2[5] = {6,7,8,9,10};
    int sum_arr[5];

    // Device (GPU) pointers
    int* cuda_arr1;
    int* cuda_arr2;
    int* cuda_sum_arr;

    int N = 5;
    int size = N * sizeof(int);

    // Allocate memory on the GPU
    // cudaMalloc requires void** because it writes the pointer value
    CUDA_CHECK(cudaMalloc((void**)&cuda_arr1, size));
    CUDA_CHECK(cudaMalloc((void**)&cuda_arr2, size));
    CUDA_CHECK(cudaMalloc((void**)&cuda_sum_arr, size));

    // Copy input data from CPU → GPU
    CUDA_CHECK(cudaMemcpy(cuda_arr1, arr1, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_arr2, arr2, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    // dim3 is a CUDA built-in type for 3D dimensions (x,y,z)
    dim3 blockSize(4);
    dim3 gridSize(2);

    // Launch kernel on GPU
    sumKernel<<<gridSize, blockSize>>>(cuda_arr1, cuda_arr2, cuda_sum_arr, N);

    // Check for launch configuration errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish and catch runtime errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result from GPU → CPU
    CUDA_CHECK(cudaMemcpy(sum_arr, cuda_sum_arr, size, cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Sum of arrays: ";
    for (int i = 0; i < N; i++){
        std::cout << sum_arr[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    CUDA_CHECK(cudaFree(cuda_arr1));
    CUDA_CHECK(cudaFree(cuda_arr2));
    CUDA_CHECK(cudaFree(cuda_sum_arr));

    return 0;
}
```
