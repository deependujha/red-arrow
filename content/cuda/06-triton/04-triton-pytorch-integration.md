---
title: Triton & PyTorch Integration
type: docs
math: true
sidebar:
  open: false
weight: 604
---

## Why `torch.autograd.Function` Exists

PyTorch can automatically compute gradients for operations it understands:

```python
y = x * x
y.backward()
```

because multiplication already has a registered backward implementation.

However, when we launch a custom Triton kernel, PyTorch only sees:

```python
y = my_triton_kernel(x)
```

and has no idea how gradients should be computed.

`torch.autograd.Function` is the mechanism that allows us to teach PyTorch how to differentiate through custom operations.

---

## High Level Architecture

```text
Model
  ↓
PyTorch Autograd
  ↓
torch.autograd.Function
  ↓
Triton Forward Kernel
Triton Backward Kernel
```

The custom `Function` acts as a bridge between PyTorch's autograd engine and Triton's GPU kernels.

---

## Basic Structure

```python
import torch

class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = ...
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = ...
        return grad_x
```

Usage:

```python
y = MyOp.apply(x)
```

**Important:** Custom autograd functions are invoked through `.apply()`, not by creating an instance.

---

## The Context Object (`ctx`)

The `ctx` object allows data to be shared between forward and backward passes.

### Saving tensors

```python
ctx.save_for_backward(x, y)
```

Retrieving later:

```python
x, y = ctx.saved_tensors
```

### Saving metadata

```python
ctx.block_size = 1024
ctx.dropout_p = 0.1
```

This is useful for kernel launch parameters or configuration values that are needed during backward.

---

## Triton Forward Integration

A typical forward implementation launches a Triton kernel:

```python
class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        y = torch.empty_like(x)

        my_forward_kernel[grid](
            x,
            y,
            ...
        )

        ctx.save_for_backward(x)

        return y
```

Forward is responsible for:

1. Launching Triton kernels.
2. Producing output tensors.
3. Saving information required for gradient computation.

---

## Triton Backward Integration

Backward receives gradients from the next layer:

```python
@staticmethod
def backward(ctx, grad_output):

    (x,) = ctx.saved_tensors

    grad_x = torch.empty_like(x)

    my_backward_kernel[grid](
        x,
        grad_output,
        grad_x,
        ...
    )

    return grad_x
```

`grad_output` represents:

```text
dL/dy
```

where:

```text
L = final loss
y = output of this operation
```

The backward kernel computes:

```text
dL/dx
```

using the chain rule.

---

## Mapping Forward Inputs to Backward Returns

Every input of `forward()` requires a corresponding return value from `backward()`.

Example:

```python
@staticmethod
def forward(ctx, x, weight, bias):
    ...
```

Backward must return:

```python
return grad_x, grad_weight, grad_bias
```

If an argument is non-differentiable:

```python
return grad_x, None, None
```

The number of returned values must exactly match the number of forward inputs.

---

## Typical Wrapper Pattern

Users generally should not call `.apply()` directly.

Instead, create a Python wrapper:

```python
class MyOp(torch.autograd.Function):
    ...

def my_op(x):
    return MyOp.apply(x)
```

Usage:

```python
y = my_op(x)
```

This is the pattern used by many Triton projects including Unsloth.

---

## Example: Custom Cross Entropy

A common implementation structure:

```python
class FastCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels):

        losses = ...

        triton_cross_entropy_forward[grid](
            logits,
            losses,
            ...
        )

        ctx.save_for_backward(
            logits,
            logsumexp,
            labels,
        )

        return losses

    @staticmethod
    def backward(ctx, grad_losses):

        logits, logsumexp, labels = \
            ctx.saved_tensors

        triton_cross_entropy_backward[grid](
            logits,
            grad_losses,
            logsumexp,
            labels,
            ...
        )

        return grad_logits, None
```

---

## Why Save Intermediate Results?

Many operations compute expensive intermediate values.

Cross entropy is a good example:

```text
softmax(x) = exp(x - logsumexp(x))
```

Computing:

```text
logsumexp(x)
```

requires a reduction across the vocabulary dimension.

Instead of recomputing it during backward, we can save it:

```python
ctx.save_for_backward(logsumexp)
```

and reuse it later.

Benefits:

* Less computation
* Faster backward pass
* Lower memory bandwidth usage

This is one of the most common optimization techniques in custom autograd implementations.

---

## In-place Gradient Buffers

A common performance pattern is:

```python
grad_x = torch.empty_like(x)
```

and letting the Triton kernel write directly into the output buffer.

```python
my_backward_kernel[grid](
    x,
    grad_output,
    grad_x,
)
```

This avoids unnecessary allocations and copies.

---

## Relationship with `nn.Module`

`autograd.Function` is not a layer.

It only defines differentiation rules.

For reusable layers, combine it with `nn.Module`.

```python
class MyLayer(torch.nn.Module):

    def forward(self, x):
        return MyOp.apply(x)
```

Typical hierarchy:

```text
nn.Module
    ↓
autograd.Function
    ↓
Triton Kernels
```

---

## Checklist for Custom Triton Ops

1. Write Triton forward kernel.
2. Write Triton backward kernel.
3. Create `torch.autograd.Function`.
4. Save tensors needed for backward.
5. Return gradients for every forward input.
6. Wrap `.apply()` in a user-facing function.
7. Validate using:

```python
torch.autograd.gradcheck(...)
```

before using in production.

---

## Mental Model

Think of `torch.autograd.Function` as the contract between PyTorch and Triton:

```text
PyTorch:
"How do I compute gradients?"

Your Function:
"Run this Triton backward kernel."

PyTorch:
"Which tensors do you need?"

Your Function:
"Use the tensors I saved during forward."
```

Without `torch.autograd.Function`, PyTorch can execute Triton kernels but cannot differentiate through them.

I'd add a separate page later on **gradcheck**, **higher-order gradients**, and **memory-saving techniques (`ctx.mark_non_differentiable`, recomputation vs save_for_backward, custom AMP)** because those start showing up quickly in real Triton contributions.

---

# Advanced Triton + PyTorch Autograd

## Gradcheck, Higher-Order Gradients, and Memory Optimization

Once a custom Triton operation works, the next challenge is ensuring it is:

1. Correct
2. Differentiable
3. Memory efficient
4. Compatible with the PyTorch ecosystem

This page covers the most important concepts beyond basic `torch.autograd.Function`.

---

# 1. Verifying Gradients with `gradcheck`

A backward kernel can easily be wrong while appearing to work during training.

PyTorch provides a numerical gradient checker:

```python
torch.autograd.gradcheck(...)
```

which compares:

```text
Analytical Gradient
(from your backward kernel)

vs

Numerical Gradient
(finite differences)
```

---

## Example

Suppose:

```python
y = x²
```

Then:

```text
dy/dx = 2x
```

PyTorch can verify your implementation automatically.

```python
import torch

x = torch.randn(
    10,
    dtype=torch.float64,
    requires_grad=True,
)

torch.autograd.gradcheck(
    MySquare.apply,
    (x,),
)
```

Expected:

```text
True
```

---

## Why Double Precision?

Finite differences are sensitive to numerical error.

Always use:

```python
dtype=torch.float64
```

for gradcheck.

Using FP16 or BF16 usually produces meaningless results.

---

## Common Gradcheck Failures

### Wrong derivative

Forward:

```python
y = x * x
```

Backward:

```python
return grad_output * x
```

Correct answer should be:

```python
return grad_output * 2 * x
```

Gradcheck immediately detects this.

---

### Incorrect tensor shape

Forward input:

```python
x.shape = (128,)
```

Backward output:

```python
grad_x.shape = (64,)
```

Autograd will fail.

Gradient tensors must exactly match input shapes.

---

### Missing gradients

Forward:

```python
forward(ctx, x, weight)
```

Backward:

```python
return grad_x
```

Wrong.

Must return:

```python
return grad_x, grad_weight
```

or:

```python
return grad_x, None
```

if the parameter is not differentiable.

---

# 2. Higher-Order Gradients

Most training only requires:

```text
First derivative
```

Example:

```text
dL/dx
```

Some algorithms require:

```text
Second derivative
```

or even higher.

Examples:

* Meta-learning
* Implicit layers
* PINNs
* Some optimization research

---

## First Derivative

```python
y = x**2
```

Gradient:

```python
grad = torch.autograd.grad(
    y,
    x,
    create_graph=True,
)
```

Result:

```text
2x
```

---

## Second Derivative

```python
second_grad = torch.autograd.grad(
    grad,
    x,
)
```

Result:

```text
2
```

---

## Problem with Most Triton Ops

Most custom Triton operations only implement:

```python
forward()
backward()
```

which gives:

```text
First-order gradients only
```

Second derivatives will fail because PyTorch cannot differentiate through your backward kernel.

---

## How PyTorch Builtins Handle It

For operations like:

```python
torch.exp()
torch.sin()
torch.softmax()
```

PyTorch also knows how to differentiate the backward pass itself.

This is why higher-order gradients work automatically.

---

## Practical Advice

For most Triton kernels:

```text
Forward + Backward
```

is sufficient.

Only implement higher-order gradients when a real use case requires them.

---

# 3. Memory vs Compute Tradeoff

One of the biggest design decisions:

```text
Save Intermediate Results
          vs
Recompute Intermediate Results
```

---

## Saving Everything

Forward:

```python
ctx.save_for_backward(
    x,
    y,
    z,
    softmax,
    logsumexp,
)
```

Advantages:

```text
Fast backward
```

Disadvantages:

```text
High memory usage
```

---

## Recomputation

Forward:

```python
ctx.save_for_backward(x)
```

Backward:

```python
softmax = recompute_softmax(x)
```

Advantages:

```text
Low memory usage
```

Disadvantages:

```text
More computation
```

---

## Example: Cross Entropy

Common approach:

Save:

```python
logsumexp
```

instead of:

```python
softmax
```

because:

```text
softmax = exp(x - logsumexp)
```

can be reconstructed cheaply.

This reduces memory usage significantly.

---

# 4. `ctx.mark_non_differentiable`

Sometimes outputs should never receive gradients.

Example:

```python
indices = torch.argmax(x)
```

Indices are integers.

Gradients do not make sense.

Tell autograd:

```python
ctx.mark_non_differentiable(indices)
```

Example:

```python
@staticmethod
def forward(ctx, x):

    values, indices = ...

    ctx.mark_non_differentiable(indices)

    return values, indices
```

Benefits:

* Less memory
* Less autograd overhead
* Clearer semantics

---

# 5. `ctx.needs_input_grad`

Backward often computes gradients for every input:

```python
grad_x
grad_w
grad_b
```

But maybe only:

```python
x.requires_grad = True
```

and:

```python
weight.requires_grad = False
```

PyTorch exposes:

```python
ctx.needs_input_grad
```

Example:

```python
if ctx.needs_input_grad[0]:
    compute_grad_x()

if ctx.needs_input_grad[1]:
    compute_grad_weight()
```

Benefits:

* Less computation
* Faster backward

---

# 6. Avoid Saving Huge Tensors

Bad:

```python
ctx.save_for_backward(
    logits,
    probabilities,
    softmax,
    exp_logits,
)
```

Memory explodes for:

```text
Batch Size × Sequence Length × Vocabulary
```

especially in LLM training.

Prefer:

```python
ctx.save_for_backward(
    logits,
    logsumexp,
)
```

and reconstruct when possible.

Rule of thumb:

```text
Save small reductions,
recompute large activations.
```

---

# 7. Mixed Precision Considerations

Many Triton kernels run in:

```text
FP16
BF16
```

But gradients are often accumulated in:

```text
FP32
```

Example:

```python
x = tl.load(...)
x = x.to(tl.float32)
```

Common pattern:

```text
Load FP16
Compute FP32
Store FP16
```

to improve numerical stability.

---

# 8. Custom AMP Support

For Automatic Mixed Precision (AMP), custom ops can be annotated.

Modern PyTorch provides:

```python
torch.amp.custom_fwd
torch.amp.custom_bwd
```

Example:

```python
class MyOp(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x):
        ...

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        ...
```

Benefits:

* Proper autocast handling
* Better integration with AMP training

---

# 9. Common Performance Pattern

A high-performance Triton op usually follows:

```text
Forward
 ├─ Compute output
 ├─ Save minimal state
 └─ Return result

Backward
 ├─ Load saved state
 ├─ Reconstruct cheap intermediates
 ├─ Launch Triton kernel
 └─ Return gradients
```

The goal is:

```text
Minimal Saved Memory
+
Minimal Recomputation
+
Correct Gradients
```

---

# Debugging Checklist

When a custom Triton operation behaves incorrectly:

### Step 1

Verify forward output:

```python
torch.testing.assert_close(...)
```

against a PyTorch reference implementation.

---

### Step 2

Run:

```python
torch.autograd.gradcheck(...)
```

---

### Step 3

Compare backward output:

```python
custom_grad
```

against:

```python
reference_grad
```

from native PyTorch.

---

### Step 4

Benchmark:

```python
Forward latency
Backward latency
Peak memory
```

Only then evaluate whether the custom kernel is actually an improvement.

---

# Mental Model

```text
Beginner:
Save everything

Intermediate:
Save only expensive-to-recompute values

Advanced:
Choose the optimal point on the
memory ↔ compute tradeoff curve
for your workload
```

Most high-performance Triton kernels are not just faster because of better GPU code—they are faster because they carefully choose what to save and what to recompute during backward.
