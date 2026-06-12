---
title: reinterpret_cast <>
type: docs
math: true
sidebar:
  open: false
weight: 502
---

## What It Does

Tells the compiler: "treat this block of memory as a different type : don't convert, just reinterpret." The bits stay exactly the same, you're just changing the lens through which you read them.

```cpp
float x = 1.0f;
int* p = reinterpret_cast<int*>(&x);
// *p is now 0x3F800000 : the IEEE 754 bits of 1.0f
```

Zero runtime cost : purely a compile-time directive.

> `reinterpret_cast` is a core C++ feature, not CUDA-specific. CUDA just uses it heavily for performance patterns.

---

## Why It Matters in CUDA : Vectorized Loads

The most common pattern: load multiple floats in a single memory instruction.

```cpp
float4* in4  = reinterpret_cast<float4*>(input);
float4* out4 = reinterpret_cast<float4*>(output);

int idx  = blockIdx.x * blockDim.x + threadIdx.x;
float4 val = in4[idx]; // one 128-bit load (LDG.128 in PTX)

// reading values from `reinterpret_cast` pointer
float val1 = val.x;
float val2 = val.y;
float val3 = val.z;
float val4 = val.w;

// writing to `reinterpret_cast` pointer at once
out4[idx] = {1.0f, 2.0f, 3.0f, 4.0f};
```

Instead of 4 separate 32-bit loads, you get one 128-bit load. This is one of the most common micro-optimizations in CUDA kernels.

---

## Vector Types Available

CUDA provides built-in vector types for `char`, `short`, `int`, `long`, `float`, `double`, and unsigned variants. The pattern is `type + width`:

| Base type | Variants |
|-----------|----------|
| `char` | `char1`, `char2`, `char3`, `char4` |
| `uchar` | `uchar1`, `uchar2`, `uchar3`, `uchar4` |
| `short` | `short1`, `short2`, `short3`, `short4` |
| `int` | `int1`, `int2`, `int3`, `int4` |
| `uint` | `uint1`, `uint2`, `uint3`, `uint4` |
| `long` | `long1`, `long2`, `long3`, `long4` |
| `float` | `float1`, `float2`, `float3`, `float4` |
| `double` | `double1`, `double2` |

Access fields via `.x`, `.y`, `.z`, `.w`.

**For vectorized memory access, prefer width-4 for 32-bit types (`float4`, `int4`) and width-2 for 64-bit types (`double2`).** Both map to a 128-bit load : the widest single instruction the memory system supports.

There's no `double4` or `float8` : 128 bits is the hardware ceiling for a single load.

---

## Alignment Rule

The pointer you cast to must satisfy the alignment requirement of the target type:

| Type | Alignment required |
|------|--------------------|
| `float2` | 8 bytes |
| `float4` | 16 bytes |
| `double2` | 16 bytes |

`cudaMalloc` always returns 256-byte aligned memory so you're safe by default. But if you're offsetting into a buffer manually, verify alignment before casting.

---

## Handling Non-Multiple-of-4 Sizes

If `N` isn't a multiple of 4, pad the allocation and guard the writeback:

```cpp
// host: pad allocation
int padded_n = (N + 3) / 4 * 4;
cudaMalloc(&buf, padded_n * sizeof(float));

// kernel: safe vectorized load, guarded store
float4 val = in4[idx];
int base = idx * 4;
if (base + 0 < N) out[base + 0] = val.x;
if (base + 1 < N) out[base + 1] = val.y;
if (base + 2 < N) out[base + 2] = val.z;
if (base + 3 < N) out[base + 3] = val.w;
```

The load is always safe because the buffer is padded. The store is guarded so you don't write garbage past `N`.

---

## Other Common Uses in CUDA

**Half precision buffers:**
```cpp
__half* h = reinterpret_cast<__half*>(fp16_buffer);
```

**Raw shared memory : declare once, cast to whatever you need:**
```cpp
extern __shared__ char smem[];
float* fs = reinterpret_cast<float*>(smem);
int*   is = reinterpret_cast<int*>(smem + offset);
```

**Byte-level inspection:**
```cpp
uint8_t* bytes = reinterpret_cast<uint8_t*>(some_struct);
```

> [!CAUTION]
> > **128 bits (16 bytes) is the hardware ceiling for a single load**
>
> - `double` only goes up to `double2` because `double2` is already 128 bits : the hardware load width ceiling.
> - Same reason there's no `float8`.
> - Worth remembering for interviews.
