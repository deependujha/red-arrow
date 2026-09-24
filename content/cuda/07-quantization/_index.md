---
title: Quantization
type: docs
math: true
sidebar:
  open: false
weight: 700
---

**torchAO**: `PyTorch's open-source quantization toolkit`. `Provides tools for reducing model size and memory usage while maintaining accuracy`.

> **`ao: architecture optimization`**

## Precision v/s Quantization

- **Precision** refers to the number of bits used to store a numerical value in a model,
- while **quantization** is the specific `process of converting higher-precision values` (like 32-bit floats) `into lower-precision formats` (like 8-bit or 4-bit integers)

> `FP32`, `FP16`, `INT8`, `INT4`, `NVFP4`, `MXFP8`, `MXFP4`, `INT2`, etc are some common precision formats.
> - quantization: `process of converting higher-precision values` into lower-precision formats
> - dequantization: `process of converting lower-precision values` back `into higher-precision formats`

## Sub-Byte representation ⭐️

A **sub-byte representation** means storing or processing data using **fewer than 8 bits (1 byte) per value**.

In standard computing, the byte is the smallest directly addressable unit of memory. Datatypes like FP32 (4 bytes), FP16 (2 bytes), or INT8 (1 byte) all occupy whole bytes.

When you go below 8 bits—into formats like **INT4, FP4, INT2, or 1-bit (BitNet)**—a single value no longer fills an entire byte.

### How it works under the hood

Because modern hardware (CPUs, GPUs) cannot address individual bits directly in memory, sub-byte values must be **packed together** into standard byte containers (like `uint8` or `int32`).

For example, with **INT4** (4 bits per value):

* A single `uint8` byte (8 bits) holds **two** INT4 numbers side-by-side.
* **Packing:** To fit two 4-bit numbers ($A$ and $B$) into one byte:

$$\text{Byte} = (A \text{ \& } 0\text{xF}) \mid (B \ll 4)$$


* **Unpacking:** To retrieve them during computation, bitwise operations extract the lower 4 bits and upper 4 bits.

### Why it matters

1. **Memory Footprint:** Loading a 70B parameter model in FP16 takes ~140 GB of VRAM. Packing it into sub-byte INT4 reduces that footprint to ~35 GB, allowing it to fit on a single GPU.
2. **Bandwidth Savings:** LLM inference is usually memory-bandwidth bound (waiting for weights to travel from VRAM to compute cores). Transferring 4-bit or 2-bit packed bytes speeds up token generation dramatically.

---

## Quantization Techniques


- pytorch dispatch method and subclassing torch.tensor
https://github.com/albanD/subclass_zoo/tree/main

- understand ptq, gptq, awq, qat, 

https://github.com/jeffhammond/pytorch-china-2026-nccl-tutorial/tree/main

https://gemini.google.com/app/f62deea7cbae2589

https://github.com/meta-pytorch/MSLK