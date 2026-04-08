---
title: Tensors in PyTorch
type: docs
prev: docs/01-basics/01-data/01-dataset-and-dataloader.md
next: docs/01-basics/02-concepts/02-learning-rate-scheduler.md
sidebar:
  open: false
weight: 22
---

## Creation

```python
import torch

# from array
torch.tensor([1, 2, 3])
torch.tensor([[1, 2], [3, 4]])

# Random
torch.rand(2, 3)        # Uniform [0, 1)
torch.randn(2, 3)       # Normal (mean=0, std=1)

torch.empty(size)       # uninitialized memory (fast but contains garbage values).

# torch.randint(low, high, shape)
torch.randint(0, 10, (2, 3))  # Integers [0, 10)

# Constants
torch.zeros(2, 3) # shape
torch.ones(2, 3)

# torch.full(size, fill_value)
torch.full((2, 3), 7) # tensor of shape (2, 3) initialized with 7

# Sequences
torch.arange(0, 10, 2)     # 0, 2, 4, 6, 8
torch.linspace(0, 1, 5)    # 5 points between 0 and 1
torch.eye(3)               # Identity matrix
```

### `torch.empty` v/s `torch.rand`

`torch.empty` and `torch.rand` are very different beasts even though both give you a tensor “full” of numbers.

* **`torch.empty(shape)`**

  * Allocates memory **without** initializing it.
  * The contents are whatever random bits happened to be in RAM/VRAM — garbage values.
  * Very fast because it skips filling the tensor.
  * Useful if you plan to immediately overwrite all elements.
  * Dangerous if you accidentally use it before assignment (can produce NaNs, huge values, etc.).

* **`torch.rand(shape)`**

  * Allocates **and** fills with random values from **uniform distribution \[0, 1)**.
  * Safe to use directly in computations.
  * Slightly slower because it does actual initialization.

Example:

```python
torch.empty(2, 3)
# tensor([[ 5.1396e-02,  0.0000e+00,  3.9894e-08],
#         [ 0.0000e+00,  1.2867e+31, -1.0094e+25]])  # garbage data

torch.rand(2, 3)
# tensor([[0.4728, 0.7289, 0.1710],
#         [0.9804, 0.3112, 0.6945]])  # nice [0, 1) random numbers
```

Rule of thumb:
Use `empty` only if you **know** you’ll overwrite everything before reading. Otherwise, `rand` (or `zeros`/`ones`) is the safe choice.

---

## Properties

```python
t = torch.randn(2, 3)
t.shape        # torch.Size([2, 3])
t.dtype        # e.g. torch.float32
t.device       # 'cpu' or 'cuda:0'
```

## Device/Dtype

```python
t = torch.ones(2, 3, device='cuda', dtype=torch.float64)
t_cpu = t.to('cpu')
t_float = t.to(torch.float32)
```

## Indexing & Slicing

```python
t[0]          # First row
t[:, 1]       # Second column
t[0, 1]       # Single element
```

## Reshape

```python
t.view(6)                  # Flatten
t.reshape(6)               # Flatten
t.view(3, 2)               # New shape
t.reshape(3, 2)            # New shape
t.permute(1, 0)            # Swap dimensions
t.unsqueeze(0)             # Add dim at pos 0
t.squeeze()                # Remove dims of size 1
```

## Math Ops

```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
a + b
a * b                      # Elementwise
torch.matmul(a.view(1, -1), b.view(-1, 1))  # Matrix mult
```

## Interop

```python
import numpy as np

np_arr = np.array([1, 2])
t = torch.from_numpy(np_arr)
np_back = t.numpy()
```

## Saving/Loading

```python
torch.save(t, 'tensor.pt')
t_loaded = torch.load('tensor.pt')
```

---

## `torch.tensor` vs `torch.Tensor`


The key differences between `torch.tensor` and `torch.Tensor`:

**`torch.tensor`** (lowercase 't'):

- A **function** that creates a new tensor from data
- Always copies the input data
- Infers the dtype from the input by default
- More flexible - accepts lists, arrays, scalars, etc.

```python
x = torch.tensor([1, 2, 3])  # Creates tensor with dtype inferred (int64)
y = torch.tensor([1.0, 2.0])  # Creates float32 tensor
```

**`torch.Tensor`** (uppercase 'T'):

- A **class** (specifically an alias for `torch.FloatTensor`)
- The default tensor constructor
- Always creates a float32 tensor
- Can create uninitialized tensors when given just a size

```python
x = torch.Tensor([1, 2, 3])  # Creates float32 tensor [1.0, 2.0, 3.0]
y = torch.Tensor(3, 4)       # Creates uninitialized 3x4 float32 tensor
```

**Practical recommendations:**

- Use `torch.tensor()` when creating tensors from data - it's more intuitive and safer
- Use `torch.Tensor()` mainly for creating empty tensors of a specific size (though `torch.empty()` is clearer)
- For explicit dtype control, both support the `dtype` parameter

Example showing the difference:
```python
data = [1, 2, 3]
a = torch.tensor(data)   # tensor([1, 2, 3], dtype=int64)
b = torch.Tensor(data)   # tensor([1., 2., 3.])  - note the floats!
```

---

## `torch.view()` vs `torch.reshape()`

The key differences between `view()` and `reshape()`:

**`view()`**:

- Requires the tensor to be **contiguous** in memory
- Returns a view of the original tensor (shares the same underlying data)
- Fails if the tensor is not contiguous
- Faster when it works

```python
x = torch.randn(4, 3)
y = x.view(12)        # Works - shares memory with x
y[0] = 100            # Modifies x as well!

z = x.t()             # Transpose makes it non-contiguous
w = z.view(12)        # ❌ RuntimeError: view size is not compatible
```

**`reshape()`**:

- Works on **both contiguous and non-contiguous** tensors
- Returns a view when possible, but copies data if necessary
- More flexible and safer to use
- Slightly slower due to the contiguity check

```python
x = torch.randn(4, 3)
z = x.t()             # Non-contiguous
w = z.reshape(12)     # ✅ Works! Copies data if needed
```

**Checking contiguity:**

```python
x = torch.randn(3, 4)
print(x.is_contiguous())      # True

y = x.t()
print(y.is_contiguous())      # False
print(y.contiguous().is_contiguous())  # True
```

**Practical recommendations:**

- Use `reshape()` as the default - it's more robust and handles both cases
- Use `view()` only when you specifically need to ensure no copying occurs (for performance or memory-sharing reasons)
- If you get an error with `view()`, either use `reshape()` or call `.contiguous()` first: `x.contiguous().view(...)`

**Memory sharing example:**

```python
x = torch.tensor([1, 2, 3, 4])
y = x.view(2, 2)      # Shares memory
y[0, 0] = 999
print(x)              # tensor([999, 2, 3, 4]) - x changed too!
```

---

## TensorDataset

In PyTorch, `TensorDataset` is a simple dataset wrapper from `torch.utils.data` that stores one or more tensors of the same length and lets you index them together.

Think of it as a zip for tensors — you give it, say, your inputs and labels, and it returns matching slices when you index it.

Example:

```python
from torch.utils.data import TensorDataset, DataLoader
import torch

# Two tensors of equal length
features = torch.randn(5, 3)  # 5 samples, 3 features each
labels = torch.randint(0, 2, (5,))  # 5 labels

dataset = TensorDataset(features, labels)

# Indexing returns a tuple
x, y = dataset[0]
print(x, y)

# You can wrap it in a DataLoader for batching
loader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch_x, batch_y in loader:
    print(batch_x, batch_y)
```

Key points:

* All tensors passed in must have the same first dimension size (number of samples).
* Useful when your entire dataset already fits in memory as tensors.
* Minimal — no transformations, lazy loading, or file handling; just indexing and length.

It’s perfect for small to medium datasets, prototyping, or unit tests — but for large datasets, you’d usually subclass `Dataset` to load data on the fly instead of keeping it all in memory.
