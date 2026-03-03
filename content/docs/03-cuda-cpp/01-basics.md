---
title: Basics of CUDA C++
type: docs
prev: docs/
sidebar:
  open: false
weight: 301
---

- GPUs are specialized hardwares, (also called accelerators) that can perform many computations in parallel.
- They are designed to handle large amounts of data and are particularly effective for tasks that can be parallelized, such as matrix operations, image processing, and machine learning.
- They can't perform network operations, file I/O, or other tasks that require interaction with the operating system. They are designed to perform computations on data that is already in memory.

> To use a GPU, first a CPU program (called the host) sends data to the GPU (called the device), then launches a kernel (a function that runs on the GPU) to perform computations on that data, and finally retrieves the results back to the CPU.

![nvidia software stack](/03-cuda-cpp/nvidia-software-stack.png)

> CuDNN (for deep learning), cuBLAS (for linear algebra), and cuFFT (for Fourier transforms) are libraries that provide optimized implementations of common operations on NVIDIA GPUs.

- `Thrust` is more general-purpose and provides a C++ template library for parallel algorithms (like, `sort`, `reduce`, `transform`) and data structures, similar to the C++ Standard Template Library (STL), but optimized for CUDA.
