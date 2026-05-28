---
title: CUDA C++
type: docs
prev: docs/
next: docs/03-cuda-cpp/01-basics.md
sidebar:
  open: false
weight: 300
---

- Get hardware architecture

```bash
# x86_64: 64-bit Intel/AMD
# aarch64/arm64: 64-bit ARM (Apple Silicon, Graviton, many ARM servers)
# armv7l: 32-bit ARM
# i686: 32-bit x86
uname -m
```

- Get Ubuntu version

```bash
cat /etc/os-release
```

- download CUDA toolkit

![cuda download](/03-cuda-cpp/cuda-download.png)

**deb (network)** `preferred`
> Installs CUDA via NVIDIA’s apt repo; integrates cleanly with Ubuntu and supports normal `apt upgrade/remove`. Best default for reproducible setups.

**deb (local)**
> Downloads a local CUDA repo snapshot and installs via apt; useful for offline or restricted environments.

**runfile (local)**
> Standalone installer that bypasses apt and installs CUDA directly; more control, doesn’t integrate with your package manager. Harder to uninstall cleanly. Can conflict with distro drivers.

---

## C++ installation

```bash
# on ubuntu
sudo apt update && sudo apt install -y build-essential
```

## Check & Specify C++ standard

- 201103L → C++11
- 201402L → C++14
- 201703L → C++17
- 202002L → C++20
- 202302L → C++23

```cpp
// To see which standard macro is active, do this:
#include <iostream>

int main() {
    std::cout << __cplusplus << "\n";
}
```

- To specify the C++ standard:

```bash
# use std flag

g++ -std=c++20 test.cpp -o test
```

---

## Compute Capability

Every NVIDIA GPU has a `Compute Capability (CC)` number, which indicates what features are supported by that GPU and specifies some hardware parameters for that GPU.

> Compute capability is denoted as a major and minor version number in the format X.Y where X is the major version number and Y is the minor version number. For example, CC 12.0 has a major version of 12 and a minor version of 0. The compute capability directly corresponds to the version number of the SM. For example, the SMs within a GPU of CC 12.0 have SM version sm_120. This version is used to label binaries.

```bash
# get compute capability of your GPU
nvidia-smi --query-gpu=name,compute_cap
```

- At runtime, the compute capability can be obtained using:
    - the CUDA Runtime API `cudaDeviceGetAttribute()`,
    - the CUDA Driver API `cuDeviceGetAttribute()`, or
    - the NVML API `nvmlDeviceGetCudaComputeCapability()`:

```cpp
#include <iostream>
#include <cuda_runtime_api.h>

int main(){
    int device_id = 0;
    int computeCapabilityMajor, computeCapabilityMinor;
    cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id);
    std::cout << "Compute Capability: " << computeCapabilityMajor << "." << computeCapabilityMinor << std::endl;
}
```

```cpp
#include <cuda.h>

int computeCapabilityMajor, computeCapabilityMinor;
cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);
```

```cpp
#include <nvml.h> // required linking with -lnvidia-ml

int computeCapabilityMajor, computeCapabilityMinor;
nvmlDeviceGetCudaComputeCapability(nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor);
```

## Kernel Launch & Occupancy

The **`cudaGetDeviceProperties`** function allows an application to query the limits of each SM via device properties. `Note that there are limits per SM and per thread block`.

- **`maxBlocksPerMultiProcessor`**: The maximum number of resident blocks per SM.

- **`sharedMemPerMultiprocessor`**: The amount of shared memory available per SM in bytes.

- **`regsPerMultiprocessor`**: The number of 32-bit registers available per SM.

- **`maxThreadsPerMultiProcessor`**: The maximum number of resident threads per SM.

- **`sharedMemPerBlock`**: The maximum amount of shared memory that can be allocated by a thread block in bytes.

- **`regsPerBlock`**: The maximum number of 32-bit registers that can be allocated by a thread block.

- **`maxThreadsPerBlock`**: The maximum number of threads per thread block.

> **`The occupancy of a CUDA kernel is the ratio of the number of active warps to the maximum number of active warps supported by the SM.`**
> In general, it’s a good practice to have occupancy as high as possible which hides latency and increases performance.


```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0); // GPU 0

    std::cout << "GPU: " << prop.name << "\n\n";

    std::cout << "maxBlocksPerMultiProcessor: "
              << prop.maxBlocksPerMultiProcessor << "\n";

    std::cout << "sharedMemPerMultiprocessor: "
              << prop.sharedMemPerMultiprocessor << " bytes\n";

    std::cout << "regsPerMultiprocessor: "
              << prop.regsPerMultiprocessor << "\n";

    std::cout << "maxThreadsPerMultiProcessor: "
              << prop.maxThreadsPerMultiProcessor << "\n";

    std::cout << "sharedMemPerBlock: "
              << prop.sharedMemPerBlock << " bytes\n";

    std::cout << "regsPerBlock: "
              << prop.regsPerBlock << "\n";

    std::cout << "maxThreadsPerBlock: "
              << prop.maxThreadsPerBlock << "\n";

    return 0;
}
```

> If a kernel was launched as testKernel<<<512, 768>>>(), i.e., 768 threads per block, each SM would only be able to execute 2 thread blocks at a time. The scheduler cannot assign more than 2 thread blocks per SM because the maxThreadsPerMultiProcessor is 2048. So the occupancy would be (768 * 2) / 2048, or 75%.
> 
> If a kernel was launched as testKernel<<<512, 32>>>(), i.e., 32 threads per block, each SM would not run into a limit on maxThreadsPerMultiProcessor, but since the maxBlocksPerMultiProcessor is 32, the scheduler would only be able to assign 32 thread blocks to each SM. Since the number of threads in the block is 32, the total number of threads resident on the SM would be 32 blocks * 32 threads per block, or 1024 total threads. Since a compute capability 10.0 SM has a maximum value of 2048 resident threads per SM, the occupancy in this case is 1024 / 2048, or 50%.
> 
> The same analysis can be done with shared memory. If a kernel uses 100KB of shared memory, for example, the scheduler would only be able to assign 2 thread blocks to each SM, because the third thread block on that SM would require another 100KB of shared memory for a total of 300KB, which is more than the 233472 bytes available per SM.