---
title: torch.fx
type: docs
math: true
sidebar:
  open: false
weight: 206
---

`fx: functional transformations`

`torch.fx` is a program transformation toolkit within the PyTorch ecosystem designed to capture, inspect, and transform the computational graph of an `nn.Module` or function.

It consists of three decoupled components:
1. **Symbolic Tracer**: Captures execution flow by running the code with mock symbol inputs.
2. **Intermediate Representation (IR)**: A graph-based data structure tracking operations.
3. **Python Code Generation**: Produces valid, human-readable Python source code from the IR.

---

## 1. Overview and Core Architecture

At its core, `torch.fx` is a **Python-to-Python (Module-to-Module) program transformation toolkit**. Given an `nn.Module` (or a plain function), it runs the code once using symbolic inputs to record the operations, generates an intermediate representation (IR) graph, allows graph manipulation, and then **regenerates standard Python source code** from the updated graph.

Unlike frameworks that compile models into opaque bytecode or custom runtimes, FX outputs a readable `def forward(self, x): ...` Python method. This makes the transformed program fully inspectable and debuggable.

```
Symbolic Tracing  ──►  Intermediate Representation (Graph)  ──►  Python Codegen
   (Capture)                 (Editable Nodes)                 (Regenerate Source)
```

Each component can be used independently. For example, a module can be traced purely to analyze its layers without transformation, or a new `GraphModule` can be constructed entirely from custom config files without ever running a tracer.

### Context: Relation to `torch.compile`
Conceptual understanding of FX is a prerequisite for debugging `torch.compile` internals. Conceptually, TorchDynamo captures Python bytecode and parses it into FX graphs, which are then handed over to compiler backends like TorchInductor. Familiarity with FX nodes, tracers, and graph transformation patterns directly translates to understanding Dynamo's graph breaks and backend lowerings.

---

## 2. Canonical Example

The following code illustrates the end-to-end tracing and recompilation process:

```python
import torch
from torch.fx import symbolic_trace

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()
symbolic_traced = symbolic_trace(module)  # Returns a GraphModule

# Inspect the IR Graph
print(symbolic_traced.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %param : [num_users=1] = get_attr[target=param]
#     %add : [num_users=1] = call_function[target=operator.add](args=(%x, %param), kwargs={})
#     %linear : [num_users=1] = call_module[target=linear](args=(%add,), kwargs={})
#     %clamp : [num_users=1] = call_method[target=clamp](args=(%linear,), kwargs={min:0.0,max:1.0})
#     return clamp

# Inspect the generated Python code
print(symbolic_traced.code)
# def forward(self, x):
#     param = self.param
#     add = x + param;  x = param = None
#     linear = self.linear(add);  add = None
#     clamp = linear.clamp(min=0.0, max=1.0);  linear = None
#     return clamp
```

### Key Takeaways
* **GraphModule**: The returned `symbolic_traced` object is an instance of `GraphModule` (a subclass of `nn.Module`). It exposes the editable `.graph` representation, the recompiled `.code` string, and can be invoked directly as a standard PyTorch module.
* **Symbolic Execution**: The tracer replaces actual tensor arguments with `Proxy` objects during execution. These proxies do not carry physical data. Instead, they intercept operations and record them as nodes in the `Graph` object.

---

## 3. The Graph IR and Node Structure

A `Graph` is structured as a doubly-linked list of `Node`s. Every `Node` is categorized by its `op` (opcode), indicating the operation type:

| Opcode | Description | `target` representation | `args` / `kwargs` content |
| :--- | :--- | :--- | :--- |
| `placeholder` | Function input parameter | Parameter name as string | Empty, or default value |
| `get_attr` | Fetch attribute from module hierarchy | Fully-qualified attribute path | Unused |
| `call_function` | Execute free functions (`torch.add`, etc.) | The function object | Arguments passed to the function |
| `call_module` | Execute submodule's `forward` method | Fully-qualified submodule path | Arguments passed to submodule |
| `call_method` | Invoke method on a tracked value | Method name as string | Receiver object (`args[0]`) and arguments |
| `output` | Terminating return statement | The string `"output"` | Value being returned (`args[0]`) |

Use `Graph.print_tabular()` to display this information in a structured, readable grid format when debugging.

### Key Concept: Leaf Modules and Inlining
If a submodule is marked as a **leaf module**, the tracer does not inspect its internals; it records a single `call_module` node. By default, all standard `torch.nn` modules (e.g., `Linear`, `Conv2d`) are treated as leaf modules, whereas custom module classes are traced through (inlined). This behavior can be controlled by overriding `Tracer.is_leaf_module()`.

---

## 4. Graph Transformation Strategies

Every graph transformation follows a standard workflow: trace the input module to retrieve its graph, mutate the graph structure, and wrap the modified graph inside a new `GraphModule`.

```python
def transform(m: torch.nn.Module, tracer_class=torch.fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(m)
    # ... Step 2: Modify the graph IR ...
    return torch.fx.GraphModule(m, graph)
```

The three primary paradigms for graph modification are detailed below.

### 4a. Direct Node Manipulation
This approach involves iterating through `graph.nodes` and modifying targets or properties in-place.

```python
def transform(m: torch.nn.Module, tracer_class=torch.fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(m)
    for node in graph.nodes:
        # Swap all occurrences of addition with multiplication
        if node.op == 'call_function' and node.target == torch.add:
            node.target = torch.mul
    graph.lint()  # Ensures the mutated graph is topologically sound
    return torch.fx.GraphModule(m, graph)
```

To insert new operations, use `inserting_after` or `inserting_before` context managers combined with `replace_all_uses_with`:

```python
with traced.graph.inserting_after(node):
    new_node = traced.graph.call_function(torch.relu, args=(node,))
    node.replace_all_uses_with(new_node)
```

> [!WARNING]
> Avoid utilizing Python `set` collections to store `Node` objects when order affects codegen (e.g., inserting them into a graph). Python sets are unordered, causing non-deterministic code generation across runs. Use an insertion-ordered `dict` as an ordered set instead.

### 4b. Subgraph Find and Replace (`subgraph_rewriter`)
This method handles replacement of specific multi-node patterns using structural matching.

```python
import torch
from torch.fx import symbolic_trace, subgraph_rewriter

class M(torch.nn.Module):
    def forward(self, x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w1, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

def pattern(w1, w2):
    return torch.cat([w1, w2])

def replacement(w1, w2):
    return torch.stack([w1, w2])

traced = symbolic_trace(M())
subgraph_rewriter.replace_pattern(traced, pattern, replacement)
```

#### Matching and Replacement Rules
* **Use-Def Matching**: Patterns are matched based on data-flow dependencies (use-def chains) rather than string variable names.
* **Return Anchor**: Only the output node returned by `pattern` serves as the anchor point for the replacement.
* **Parameters**: The replacement function must accept the exact parameter signature defined in the pattern function.
* **Overlap Handling**: If multiple matched subgraphs overlap, only the topologically earliest match is replaced.

### 4c. Proxy Retracing (Decompositions)
Retracing is ideal for complex transformations (e.g., operator decompositions). You write replacement patterns in standard Python, wrapping nodes in `Proxy` objects to record operations dynamically.

```python
import torch.nn.functional as F

def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {F.relu: relu_decomposition}

def decompose(model, tracer_class=torch.fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(model)
    new_graph = torch.fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # Map existing nodes to Proxy objects in the new graph
            proxy_args = [torch.fx.Proxy(env[a.name], tracer) if isinstance(a, torch.fx.Node) else a
                          for a in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)
            env[node.name] = output_proxy.node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
            
    return torch.fx.GraphModule(model, new_graph)
```

#### Transformation Strategy Decision Matrix
* **Direct Node Manipulation**: Best for single-node swaps, simple insertions (e.g., identity inserts), or dead-code elimination.
* **Pattern Rewriting (`replace_pattern`)**: Best for matching static, multi-node subgraphs (e.g., fusing Conv + BatchNorm + ReLU).
* **Proxy Retracing**: Best for complex math transformations (e.g., lowering high-level operators to basic math primitives).

---

## 5. The Interpreter Pattern: Executing Graphs

The Interpreter pattern runs the graph node-by-node. This allows tracing metadata, propagating shapes, or registering custom hook logic during execution without modifying the graph.

```python
class ShapeProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env = {}
        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = self._fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args_ = load_arg(node.args)
                result = getattr(self_obj, node.target)(*args_, **load_arg(node.kwargs))
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            if isinstance(result, torch.Tensor):
                node.shape, node.dtype = result.shape, result.dtype
            env[node.name] = result
        return load_arg(self.graph.result)
```

Instead of hand-rolling dispatch loops, use `torch.fx.Interpreter` and override individual execution hooks:

```python
from torch.fx import Interpreter

class NegSigmSwapInterpreter(Interpreter):
    def call_function(self, target, args, kwargs):
        if target is torch.sigmoid:
            return torch.neg(*args, **kwargs)
        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        if target == "neg":
            call_self, *rest = args
            return call_self.sigmoid(*rest, **kwargs)
        return super().call_method(target, args, kwargs)

result = NegSigmSwapInterpreter(gm).run(input_tensor)
```

### Interpreter vs. Transformer
* **`Interpreter`**: Executes a graph using concrete tensor values, returning execution outputs (e.g., shapes or profiles).
* **`Transformer`**: Executes a graph using `Proxy` objects to record operations, outputting a new, transformed `GraphModule`.

---

## 6. Debugging FX Graphs and Generated Code

### Verification and Numerical Correctness
* Always check outputs using `torch.allclose(actual, expected, rtol=..., atol=...)`. Do not use standard equality (`==`) on float tensors; it raises runtime exceptions and fails to account for floating-point tolerance.

### Inspecting Dynamically Generated Code
Because code generated from graphs is compiled in memory, traditional breakpoints do not map to it immediately:
* **Interactive Debugging**: Insert `import pdb; pdb.set_trace()` prior to calling the `GraphModule`, then step (`s` or `step`) directly into the call to enter the generated `forward` method.
* **Source Export**: Call `gm.to_folder("output_path", "ModuleName")` to export the module and its generated source code as standard `.py` files. You can then insert breakpoints or custom logs directly inside the exported files.

### Inspecting the Graph IR
* `print(gm.graph)`: Displays the raw node list.
* `gm.graph.print_tabular()`: Prints a clean layout of all nodes, displaying opcode, target, arguments, and custom metadata.

---

## 7. Practical Limitations of Symbolic Tracing

Understand these limitations to diagnose tracing errors effectively:

### 7a. Dynamic Control Flow
Tracer execution fails if a branch condition evaluates a dynamic tensor value:

```python
def dynamic_func(x):
    if x.sum() > 0:  # TraceError: symbolic variables cannot evaluate to a boolean
        return torch.relu(x)
    return torch.neg(x)
```
* **Reason**: The input is a `Proxy` object without concrete value data during trace time.
* **Workarounds**: Specialize the parameter using `concrete_args` if the branch is static at evaluation time, or isolate the branching logic using `@torch.fx.wrap` to treat it as an opaque leaf function.

### 7b. Static Control Flow
Branching based on static hyperparameters (e.g., module configuration flags set during `__init__`) is fully supported:

```python
class StaticModule(torch.nn.Module):
    def __init__(self, use_relu=True):
        super().__init__()
        self.use_relu = use_relu
        
    def forward(self, x):
        if self.use_relu:  # Evaluated statically during tracing
            return torch.relu(x)
        return x
```

### 7c. Non-Tensor Built-ins and Standard Libraries
Standard library functions (such as `len()` or `math.sqrt()`) do not trigger the custom `__torch_function__` dispatch mechanism, resulting in trace errors.
* **Workaround**: Register these functions explicitly using `torch.fx.wrap` before tracing.

### 7d. Tensor Constructors and Randomness
* **Non-random Constructors (`torch.zeros`, `torch.ones`)**: Supported, but shapes must not depend on dynamic tensor properties (e.g., `x.shape[0]`). If dynamic shapes are required, use `torch.zeros_like(x)` instead.
* **Random Constructors (`torch.rand`, `torch.randn`)**: The random execution is evaluated exactly once during trace time, freezing the random output tensor into the graph as a **constant**. This results in the same static tensor being reused across all future module executions. Wrap these calls in a wrapped function using `@torch.fx.wrap` to ensure they execute dynamically.

### 7e. The `self.training` State Gotcha
Branching directly on `self.training` inside functional calls (e.g., `F.dropout(x, training=self.training)`) evaluates `self.training` at trace time and freezes it as a literal boolean constant. Setting the model to evaluation mode (`gm.eval()`) later will have no effect on this constant.
* **Workaround**: Use the object-oriented submodule equivalent (`nn.Dropout`), which behaves as a leaf module and queries the training state dynamically at runtime.

---

## 8. Customizing Tracing Behavior

To modify how a graph is generated, subclass `torch.fx.Tracer`. The most common customization hook is overriding `is_leaf_module`:

```python
class CustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # Prevent tracing into custom activation layers or blocks
        if isinstance(m, CustomActivationBlock):
            return True
        return super().is_leaf_module(m, module_qualified_name)
```

---

## 9. API Reference Map

| Class / Function | Purpose | Key Attributes / Methods |
| :--- | :--- | :--- |
| `symbolic_trace` | Traces a module or function and outputs a `GraphModule`. | `concrete_args` |
| `GraphModule` | A subclass of `nn.Module` backed by an FX Graph. | `.graph`, `.code`, `.recompile()` |
| `Graph` | A doubly-linked list representing the execution graph. | `.nodes`, `.lint()`, `.print_tabular()`, `.eliminate_dead_code()` |
| `Node` | A single entry inside a `Graph` representing an operation. | `.op`, `.target`, `.args`, `.kwargs`, `.replace_all_uses_with()` |
| `Proxy` | An execution wrapper recording operations onto a Graph. | `.node` |
| `Tracer` | Subclassable tracer engine. | `is_leaf_module`, `trace()` |
| `Interpreter` | Executor for node-by-node runtime logic. | `run()`, `call_function()`, `call_method()` |
| `Transformer` | Modifies graph structure using interpreter dispatch hooks. | `transform()` |
| `replace_pattern` | Subgraph rewriter using structural matches. | `subgraph_rewriter.replace_pattern()` |
| `fx.wrap` | Registers functions as opaque leaves in the graph. | Decorator or manual function call |

---

## 10. Design Patterns and Best Practices

* **Layer Fusion and Quantization**: Implement using `replace_pattern` (§4b) or direct node modification (§4a) to merge adjacent operators (e.g., Conv-BatchNorm-ReLU).
* **Metadata Propagation**: Use the `Interpreter` pattern (§5) to walk the graph and append attributes (like shapes, strides, and dtypes) to nodes prior to code generation.
* **Operator Decomposition**: Use Proxy Retracing (§4c) to decompose high-level composite operators into target-specific primitive math instructions.
* **Diagnosing Tracing Errors**: Verify traceback details for data-dependent branching, unregistered standard library calls, or frozen evaluation states (e.g., `self.training`). Export the generated files using `gm.to_folder` to debug them interactively.
