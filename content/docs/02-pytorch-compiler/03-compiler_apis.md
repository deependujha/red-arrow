---
title: Compiler APIs
type: docs
prev: docs/02-pytorch-compiler/02-cuda
sidebar:
  open: false
weight: 203
---

`torch.compile` leverages the following underlying technologies:

- **`TorchDynamo (torch._dynamo)`** is an internal API that uses a CPython feature called the **`Frame Evaluation API`** to safely capture PyTorch graphs.

- **`TorchInductor`** is the default torch.compile deep learning compiler that generates fast code for multiple accelerators and backends. You need to use a backend compiler to make speedups through torch.compile possible. For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key building block.

- **`AOT Autograd`** captures not only the user-level code, but also backpropagation, which results in capturing the backwards pass “ahead-of-time”. This enables acceleration of both forwards and backwards pass using TorchInductor.

---

## Backends supported by `torch.compile`

- Different backends can result in various optimization gains.
- The default backend is called **`TorchInductor`**, also known as *`inductor`*.
- TorchDynamo has a list of supported backends:

```python
torch.compiler.list_backends() # each of which with its optional dependencies.
```

#### Training & Inference backends

| Backend Name      | Description                                      |
|-------------------|--------------------------------------------------|
| `torch.compile(m, backend="inductor")`       | Uses the TorchInductor backend.      |
| `torch.compile(m, backend="cudagraphs")`       | CUDA graphs with AOT Autograd.           |
| `torch.compile(m, backend="ipex")`       | Uses IPEX on CPU.     |

#### Inference-only backends

| Backend Name      | Description                                      |
|-------------------|--------------------------------------------------|
| `torch.compile(m, backend="tensorrt")`       | Uses Torch-TensorRT for inference optimizations. Requires `import torch_tensorrt` in the calling script to register backend.      |
| `torch.compile(m, backend="ipex")`       | Uses IPEX for inference on CPU.        |
| `torch.compile(m, backend="tvm")`       | Uses Apache TVM for inference optimizations.  |
| `torch.compile(m, backend="openvino")`       | Uses OpenVINO for inference optimizations.     |

---

## Guards in `torch.compile`

- `TorchDynamo` uses **guards** to ensure that the assumptions made during graph capture remain valid during execution.
- Guards are runtime checks that verify whether the conditions under which the graph was captured still hold true.
- If a guard fails, `TorchDynamo` will recompile the graph to accommodate the new conditions.
- Guards can check various aspects of the program state, such as **`tensor shapes`**, **`data types`**, and **`control flow`**.

### Skipping guards

- `skip_guard_on_inbuilt_nn_modules_unsafe`
- `skip_guard_on_all_nn_modules_unsafe`
- `keep_tensor_guards_unsafe`
- `skip_guard_on_globals_unsafe`

- Usage example:

```python
import torch

opt_mod = torch.compile(
  mod,
  options={
    "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe,
  }
)
```

> `guards` are not about weights & biases changing. They are about shapes, types, control flow, etc changing in ways that would make the compiled graph invalid.

- if you changes model size or input shapes/types, the guards will fail and the graph will be recompiled.
- Skipping guards can lead to incorrect results if the assumptions made during graph capture are violated. Use with caution.

- But, in most cases, this assumption is true, so skipping guards can lead to performance improvements.

---

## Helpful apis in `torch.compile`

```python
import torch

# compile a model
opt_mod = torch.compile(mod, backend="inductor")

# get all available backends
torch.compiler.list_backends()

# reset compilation caches and restores the system to its initial state
# recommended to call this function, especially after using operations
# like torch.compile(…) to ensure a clean state before another unrelated compilation
torch.compile.reset()

# check if compiling, compiling by _dynamo, and is exporting
torch.compiler.is_compiling()
torch.compiler.is_dynamo_compiling()
torch.compiler.is_exporting()
```

---

## `set_stance` in `torch.compile`

- Set the current stance of the compiler.
- Can be used as a `function`, `context manager`, or `decorator`.
- **`Do not use this function inside a torch.compile region`** - an error will be raised otherwise.

```python
import torch

@torch.compile
def foo(x): ...


@torch.compiler.set_stance("force_eager")
def bar():
    # will not be compiled
    foo(...)


bar()

with torch.compiler.set_stance("force_eager"):
    # will also not be compiled
    foo(...)

torch.compiler.set_stance("force_eager")
# will also not be compiled
foo(...)
torch.compiler.set_stance("default")

# will be compiled
foo(...)
```

- for more details, [refer here](https://docs.pytorch.org/docs/main/generated/torch.compiler.set_stance.html#torch.compiler.set_stance)

---

## `allow` & `substitute` in graph

### Allow in graph

```python
torch.compiler.allow_in_graph(my_custom_function)

@torch.compile(...)
def fn(x):
    x = torch.add(x, 1)
    x = my_custom_function(x)
    x = torch.add(x, 1)
    return x

fn(...)
```

- for more details, [refer here](https://docs.pytorch.org/docs/main/generated/torch.compiler.allow_in_graph.html#torch-compiler-allow-in-graph)

### Substitute in graph

- Register a polyfill handler for a function, usually a C function from the C extension, to be used in place of the original function when inlining the original function in the graph.

> The polyfill handler is a function that will be called in place of the original function when inlining the original function. The polyfill handler should have the same signature and the same behavior as the original function.


```python
>>> import operator
>>> operator.indexOf([1, 2, 3, 4, 5], 3)
2
>>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
... # xdoctest: +SKIP("Long tracebacks")
Traceback (most recent call last):
...
torch._dynamo.exc.Unsupported: ...

>>> @torch.compiler.substitute_in_graph(operator.indexOf)
>>> def indexOf(a, b, /):
>>>     for i, item in enumerate(a):
>>>         if item is b or item == b:
>>>             return i
>>>     raise ValueError("sequence.index(x): x not in sequence")
>>>
>>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
2
```

- for more details, [refer here](https://docs.pytorch.org/docs/main/generated/torch.compiler.substitute_in_graph.html)

