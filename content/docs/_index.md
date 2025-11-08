---
title: Red Arrow Documentation
type: docs
next: docs/01-basics/01-data/
weight: 1
---

- Contains notes and helper codes on PyTorch Compile, CUDA, and Triton.

## Example Code

```py {filename="main.py"}
import torch

def f(x):
    return torch.sin(x)**2 + torch.cos(x)**2
```
