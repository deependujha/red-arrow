---
title: PyTorch Compiler
type: docs
prev: docs/
next: docs/02-pytorch-compiler/01-concepts.md
sidebar:
  open: false
weight: 200
---

![pytorch2 backend](/02-pytorch-compiler/pt2-backend.png)

```bash
TORCH_COMPILE_DEBUG=1 python my_script.py
```

- `TORCH_COMPILE_DEBUG=1` enables debug logging for `torch.compile`, providing insights into the compilation process, optimizations applied, and any potential issues encountered during graph capture and execution.
- Generated `triton kernel` will be available in `torchinductor/model_*/output_code.py` file. It'll also have fx graphs saved in `torchinductor/model_*/fx_graphs.py` file, which can be used to understand the transformations applied to the original PyTorch code during compilation.

---

- we can also set logging options to control the verbosity and format of the debug output, such as:

```python
import torch

torch._logging.set_logs(graph_code = True, graph_breaks = True)
```

- for more details, [check here](https://docs.pytorch.org/docs/stable/generated/torch._logging.set_logs.html).

---

```python
torch._dynamo.reset()
```

- `torch._dynamo.reset()` is a function that resets the state of the PyTorch Dynamo compiler, clearing any cached compiled graphs and allowing for a fresh start in subsequent compilations. This can be useful when you want to ensure that changes to your code are reflected in the compiled output without interference from previously cached results.
