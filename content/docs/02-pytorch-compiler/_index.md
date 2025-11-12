---
title: PyTorch Compiler
type: docs
prev: docs/
next: docs/02-pytorch-compiler/01-concepts.md
sidebar:
  open: false
weight: 200
---

```bash
TORCH_COMPILE_DEBUG=1 python my_script.py
```

- `TORCH_COMPILE_DEBUG=1` enables debug logging for `torch.compile`, providing insights into the compilation process, optimizations applied, and any potential issues encountered during graph capture and execution.
