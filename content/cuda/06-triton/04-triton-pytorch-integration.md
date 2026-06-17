---
title: Triton & PyTorch Integration
type: docs
math: true
sidebar:
    open: false
weight: 604
---

## Why `torch.autograd.Function` Exists

PyTorch automatically computes gradients for native operations using a built-in graph. However, when you launch a custom Triton kernel, PyTorch cannot inspect the compiled GPU code to determine how gradients flow.

```python
# Native PyTorch: Differentiable automatically
y = x * x
y.backward() 

# Custom Triton Kernel: PyTorch does not know how to compute gradients
y = my_triton_kernel(x) 

```

`torch.autograd.Function` acts as the bridge, allowing you to explicitly teach PyTorch's autograd engine how to differentiate through custom Triton operations.

---

## High-Level Architecture

The custom `Function` maps high-level PyTorch graph nodes directly to your optimized hardware kernels:

```text
       Model Graph
            │
            ▼
    PyTorch Autograd
            │
            ▼
 torch.autograd.Function
      ╱           ╲
     ▼             ▼
Triton Forward   Triton Backward
    Kernel            Kernel

```

---

## Basic Structure

Custom autograd layers inherit from `torch.autograd.Function`. They require a static `forward` method and a static `backward` method.

```python
import torch

class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        # 1. Allocate output tensors
        # 2. Launch Triton forward kernel
        # 3. Save state for backward pass using ctx
        y = ...
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve saved context tensors
        # 2. Launch Triton backward kernel
        # 3. Calculate and return input gradients
        grad_x = ...
        return grad_x

```

> **CRITICAL:** Always invoke custom autograd functions using the `.apply()` method. Never instantiate the class directly.

```python
# Correct
y = MyOp.apply(x)

# Incorrect
y = MyOp()(x)

```

---

## Managing Context via `ctx`

The `ctx` (context) object acts as a scratchpad to pass information from the forward pass to the backward pass.

### 1. Saving Tensors

Use `ctx.save_for_backward` to track tensors involved in the gradient formula.

```python
ctx.save_for_backward(x, y)

```

Retrieve them during the backward pass via `ctx.saved_tensors`:

```python
x, y = ctx.saved_tensors

```

### 2. Saving Arbitrary Metadata

Non-tensor values (like Python scalars, dimensions, or hyperparameters) must be attached directly to `ctx` as attributes.

```python
ctx.block_size = 1024
ctx.dropout_p = 0.1

```

---

## End-to-End Triton Integration Pattern

### The Forward Pass

The forward pass is responsible for configuring kernel dimensions, preparing output allocations, and launching the Triton kernel grid.

```python
class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        
        my_forward_kernel[grid](
            x, y, x.numel(), BLOCK_SIZE=1024
        )
        
        ctx.save_for_backward(x)
        return y

```

### The Backward Pass

The backward pass receives incoming upstream gradients ($grad\_output$, representing $\frac{\partial L}{\partial y}$) and computes downstream gradients ($\frac{\partial L}{\partial x}$).

```python
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        
        my_backward_kernel[grid](
            x, grad_output, grad_x, x.numel(), BLOCK_SIZE=1024
        )
        
        return grad_x

```

---

## Mapping Inputs to Outputs

The number of arguments returned by `backward()` must **exactly match** the number of arguments accepted by `forward()` (excluding `ctx`).

* If an input variable requires a gradient, return its calculated tensor.
* If an input variable is non-differentiable (like metadata or integer flags), return `None`.

```python
@staticmethod
def forward(ctx, x, weight, bias, block_size):
    ...
    return y

@staticmethod
def backward(ctx, grad_output):
    ...
    # Must return 4 elements matching the 4 forward inputs
    return grad_x, grad_weight, grad_bias, None

```

> [!important]
> **`Why do we need to match no. of forward method arguments with no. of backward method return? first we do forward then backward, so shouldn't it be opposite?`**
> 
> Every return value from the `backward()` method corresponds to the gradient of a specific input variable from the `forward()` method.
> 
> Think of your forward pass as a mathematical function $f(x, w, b)$ that takes three inputs and produces an output $y$.
> 
> During training, the loss function calculates a final error $L$. When PyTorch runs backpropagation, its goal is to update **every single input variable** that contributed to that loss. To do that, it needs to find the gradient for each input:
> 
> 1. How does changing $x$ affect $L$? (We need $\frac{\partial L}{\partial x}$)
> 2. How does changing $w$ affect $L$? (We need $\frac{\partial L}{\partial w}$)
> 3. How does changing $b$ affect $L$? (We need $\frac{\partial L}{\partial b}$)
> 
> Because your forward pass took **three** variables, your backward pass must return **three** gradients.

---

### How PyTorch Matches Them Under the Hood

PyTorch doesn't use variable names to match gradients to inputs; it uses **positional order**.

If your forward pass looks like this:

```python
@staticmethod
def forward(ctx, x, weight, bias):
    # Position 0: x
    # Position 1: weight
    # Position 2: bias
    ...

```

PyTorch's autograd engine expects the `backward()` method to return a tuple where the items line up exactly with those exact same positions:

```python
@staticmethod
def backward(ctx, grad_output):
    ...
    # Position 0 matches x
    # Position 1 matches weight
    # Position 2 matches bias
    return grad_x, grad_weight, grad_bias

```

### What happens if an input doesn't change (like a configuration flag)?

Even if a forward input is a static configuration flag (like `block_size = 1024`) that doesn't have a gradient, PyTorch *still* requires you to pass a placeholder back to keep the positions aligned. You simply pass `None`.

```python
@staticmethod
def forward(ctx, x, block_size):
    ...

@staticmethod
def backward(ctx, grad_output):
    ...
    return grad_x, None  # None tells PyTorch "block_size has no gradient"

```

If the number of returns didn't match the number of inputs, PyTorch wouldn't know which gradient belonged to which variable, and the autograd engine would break.

---

## The Wrapper Pattern

To hide the internal `.apply()` semantics from end users, always wrap your autograd function in a clean Python definition. This mirrors the pattern used by production libraries like Unsloth and FlashAttention.

```python
class _FastLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w): ...
    
    @staticmethod
    def backward(ctx, grad): ...

# User-facing API
def fast_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _FastLinear.apply(x, weight)

```

Integrating this wrapper into standard deep learning code is straightforward using `torch.nn.Module`:

```python
class CustomLinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return fast_linear(x, self.weight)

```

---

## Memory vs. Compute Optimization Trade-offs

When writing Triton kernels, the largest performance gains come from optimizing memory bandwidth rather than computing operations. Consider a Custom Cross-Entropy loss implementation:

$$\text{softmax}(x)_i = \exp(x_i - \text{logsumexp}(x))$$

You face a fundamental design decision regarding how to handle intermediate states:

| Strategy | Implementation Profile | Memory Footprint | Compute Profile |
| --- | --- | --- | --- |
| **Naive Saving** | `ctx.save_for_backward(softmax_tensor)` | High ($B \times S \times V$) | Fast backward pass. |
| **Smart Saving** | `ctx.save_for_backward(logsumexp_tensor)` | Low ($B \times S$) | Recomputes softmax values in-place inside the backward Triton kernel using the small saved profile. |

### Rule of Thumb for LLMs

> **Save small reductions; recompute large activations.** Memory bandwidth is almost always the bottleneck on modern GPUs. Recomputing values like element-wise activations inside a Triton block is significantly faster than reading large matrices back out of High Bandwidth Memory (HBM).

---

## Advanced Autograd Features

### 1. `ctx.needs_input_grad`

Avoid wasting GPU cycles calculating gradients for parameters that do not need them (e.g., when `frozen_layer` or `requires_grad=False` is active).

```python
@staticmethod
def backward(ctx, grad_output):
    grad_x, grad_weight = None, None
    
    # Only execute kernels if PyTorch requests the gradient
    if ctx.needs_input_grad[0]:
        grad_x = launch_input_kernel(grad_output)
    if ctx.needs_input_grad[1]:
        grad_weight = launch_weight_kernel(grad_output)
        
    return grad_x, grad_weight

```

### 2. `ctx.mark_non_differentiable`

If your Triton kernel generates secondary discrete values (like top-k indices or boolean masks), explicitly mark them to eliminate autograd tracking overhead.

```python
@staticmethod
def forward(ctx, x):
    values, indices = launch_custom_topk_kernel(x)
    ctx.mark_non_differentiable(indices)
    ctx.save_for_backward(indices)
    return values, indices

```

### 3. Automatic Mixed Precision (AMP) Support

Ensure your custom operation handles mixed-precision types gracefully by decorating your methods with PyTorch's AMP decorators.

```python
class MyAMPBlock(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x):
        # Runs safely within autocast environments (FP16/BF16)
        return launch_kernel(x)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return launch_grad_kernel(grad_output)

```

---

## Validating Gradients with `gradcheck`

A backward kernel can contain subtle mathematical bugs that still allow models to run without throwing structural errors. Always validate your math using numerical differentiation checks.

`torch.autograd.gradcheck` compares your **analytical gradients** (from your Triton backward kernel) against **numerical gradients** (calculated via finite differences).

```python
import torch

# 1. ALWAYS use float64. Lower precision types (FP16/BF16/FP32) 
# fail gradcheck due to numerical instability in finite differences.
x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

# 2. Run the check
is_correct = torch.autograd.gradcheck(MyOp.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Gradient verification passed: {is_correct}")

```

### Common Gradcheck Failure Points

* **Shape Mismatches:** The shape returned by `backward` must be identical to the shape of the tensor passed into `forward`.
* **Missing Elements:** Returning `None` when an input tensor actually has `requires_grad=True`.
* **Incorrect Scale Factors:** Forgetting a scalar multiplier (e.g., missing a $2\times$ or a $\frac{1}{N}$ scaling parameter in the gradient chain rule).

---

## Implementation Checklist

* [ ] Write Triton forward kernel.
* [ ] Write Triton backward kernel using in-place writing to output buffers (`torch.empty_like`) to avoid temporary allocations.
* [ ] Create your `torch.autograd.Function` wrapper class.
* [ ] Profile and select memory-saving vs compute trade-offs via `ctx.save_for_backward`.
* [ ] Match all `forward` parameters to `backward` gradient returns.
* [ ] Test numerical correctness using `torch.autograd.gradcheck` in `float64`.
* [ ] Verify outputs exactly match native PyTorch baseline values using `torch.testing.assert_close`.

---

## Mental Model Summary

```text
       PyTorch Architecture                       Your Kernel Responsibility
┌────────────────────────────────┐            ┌────────────────────────────────┐
│  "Hey, I need to know how to   │            │ "No problem, here is the exact │
│   differentiate this block."   │───────────>│  Triton backward kernel rules."│
└────────────────────────────────┘            └────────────────────────────────┘
                                                              │
                                                              ▼
┌────────────────────────────────┐            ┌────────────────────────────────┐
│ "What raw context parameters   │            │ "Just hold onto these small    │
│  do you need me to save?"      │<───────────│  reductions I gave you earlier"│
└────────────────────────────────┘            └────────────────────────────────┘

```

Without a custom `torch.autograd.Function`, Triton operates outside PyTorch's awareness. With it, your optimized kernels become fully integrated elements of the neural network graph.
