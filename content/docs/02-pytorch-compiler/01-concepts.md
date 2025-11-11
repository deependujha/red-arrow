---
title: PyTorch Compiler Concepts
type: docs
prev: docs/02-pytorch-compiler
next: docs/02-pytorch-compiler/02-cuda
sidebar:
  open: false
weight: 201
---

![deep learning compiler](/02-pytorch-compiler/deep_learning_compiler.webp)

> - A deep learning compiler translates high-level code written in deep learning frameworks into optimized lower level hardware specific code to accelerate training and inference.
>
> - It finds opportunities in deep learning models to optimize for performance by performing `layer and operator fusion`, better `memory planning`, and generating target specific `optimized fused kernels` to reduce function call overhead.

---

## High level architecture

![high level architecture](/02-pytorch-compiler/high_level_architecture.webp)

1. **Graph capture**: Computational graph representation for your models and functions. PyTorch technologies: `TorchDynamo`, `Torch FX`, `FX IR`
2. **Automatic differentiation**: Backward graph tracing using automatic differentiation and lowering to primitives operators. PyTorch technologies: `AOTAutograd`, `Aten IR`
3. **Optimizations**: Forward and backward graph-level optimizations and operator fusion. PyTorch technologies: `TorchInductor` (default) or other compilers
4. **Code generation**: Generating hardware specific C++/GPU Code. PyTorch technologies: `TorchInductor`, `OpenAI Triton` (default) other compilers

---

## What actually happens (`roughly`)?

- `torchdynamo` captures the model's computation graph during execution by tracing Python bytecode. It records operations performed on tensors to create a static representation of the model's computation, and generates an intermediate representation (IR) of the computation graph in `FX IR`.
- `AOTAutograd` takes the captured computation graph and traces the forward and backward graph ahead of time, i.e. prior to execution, and generates a joint forward and backward graph. It then partitions the forward and the backward graph into two separate graphs. Both the forward and the backward graphs are stored in the `FX graph data structure (FX IR)`.
- `AOTAutograd` then lowers the forward and backward graphs to primitive operators in `Aten IR`.
- `TorchInductor` takes the lowered forward and backward graphs in `Aten IR` and applies various graph-level optimizations, such as **operator fusion**, **memory planning**, **kernel scheduling**, etc, to improve performance.
- Finally, `TorchInductor` generates optimized target-specific code (e.g., C++ for CPU, CUDA or Triton for GPU) and compiles it into executable kernels that can be run on the target hardware.

---

## Different IRs in PyTorch Compiler

![pytorch compiler irs](/02-pytorch-compiler/ir_in_pytorch_compiler.webp)

-  **`ATen`** stands for `A Tensor library`, which is a low level library with a C++ interface that implements many of the fundamental operations that run on CPU and GPU.
In eager mode operation, your PyTorch operations are routed to this library which then calls the appropriate CPU or GPU implementation.
> AOTAutograd automatically generates code that replaces the higher level PyTorch API with ATen IR for the forward and backward graph which you can see in the output below:
> Let’s say you are designing a processor and want to support PyTorch code acceleration on your hardware. It’d be near impossible to support the full list of PyTorch API in hardware, so what you can do is build a compiler that only supports the smaller subset of fundamental operators defined in Core Aten IR or Prims IR, and let AOTAutograd decompose compound operators into the core operators

- ATen IR is a list of operators supported by the ATen library.
> Core Aten IR is a subset of the broader Aten IR and Prims IR and an even smaller subset of Core Aten IR

- Core Aten IR (formerly canonical Aten IR) is a subset of the Aten IR that can be used to compose all other operators in the Aten IR. Compilers that target specific hardware accelerators can focus on supporting only the Core Aten IR and mapping it to their low level hardware API. This makes it easier to add hardware support to PyTorch since they don’t have to implement support for the full PyTorch API which will continue to grow with more and more abstractions.

- Prims IR is an even smaller subset of the Core Aten IR that further decomposes Core Aten IR ops into fundamental operations making it even easier for compilers that target specific hardware to support PyTorch. But decomposing operators into lower and lower operations will most definitely lead to performance degradation due to excess memory writes and function call overhead. But the expectation is that hardware compilers can take these operators and fuse them back together to support hardware API to get back performance.

---

## Graph optimization: `Layer and operator fusion` and C++/GPU code generation

Deep learning is composed of many fundamental operations such as matrix-matrix and matrix-vector multiplications. In PyTorch eager mode of execution each operation will result in separate function calls or kernel launches on hardware. This leads to CPU overhead of launching kernels and results in more memory reads and writes between kernel launches. A deep learning optimizing compiler like TorchInductor can fuse multiple operations into a single compound operator in python and generate low-level GPU kernels or C++/OpenMP code for it. This results in faster computation due to fewer kernel launches and fewer memory read/writes.

The computational graph from the output of AOTAutograd in the previous section is composed of many Aten operators represented in an FX graph. TorchInductor optimizations doesn’t change the underlying computation in the graph but merely restructures it with operator and layer fusion, and generates CPU or GPU code for it. Since TorchInductor can see the full forward and backward computational graph ahead of time, it can take decisions on out-of-order execution of operations that don’t have dependence on each other, and maximize hardware resource utilization.

Under the hood, for GPU targets, TorchInductor uses OpenAI’s Triton to generate fused GPU kernels. Triton itself is a separate Python based framework and compiler for writing optimized low-level GPU code which is otherwise written in CUDA C/C++. But the only difference is that TorchInductor will generate Triton code which is compiled into low level PTX code.

For multi-core CPU targets, TorchInductor generates C++ code and injects OpenMP pragma directives to generate parallel kernels. From the PyTorch user level world view, this is the IR transformation flow:

![full flow](/02-pytorch-compiler/full_flow_pytorch_compiler.webp
)