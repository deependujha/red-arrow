---
title: Quantization Concepts in Depth
type: docs
math: true
sidebar:
  open: false
weight: 701
---

This page builds quantization up from first principles. It is deliberately verbose: every formula is derived, every claim has a worked example, and the "why" is spelled out before the "how". Read it top to bottom once, then use the cheat sheet at the end as a refresher.

> **Roadmap**
> 1. The core problem: mapping floats onto a small integer grid
> 2. The affine formula (scale $S$ and zero-point $Z$)
> 3. Asymmetric quantization, derived and worked
> 4. Symmetric quantization, derived and worked, and *why it wastes half the codes*
> 5. Quantization error, clipping vs rounding, calibration
> 6. Granularity: per-tensor, per-channel, per-group
> 7. Packing sub-byte integers (nibble packing)
> 8. Low-precision **floating point** formats: FP8, FP4, sign/exponent/mantissa
> 9. Why floats handle outliers gracefully ("every value has its own ruler")
> 10. Quantizing FP32 → FP4 with block scaling; MXFP4 vs NVFP4
> 11. What to actually remember

---

## 1. The core problem

A neural network stores its weights and activations as `float32` (32 bits per value). Quantization stores them as `int8` (8 bits), `int4` (4 bits), or a low-precision float such as `fp8` / `fp4`. Going from 32 → 8 bits is a **4× memory reduction**; 32 → 4 bits is **8×**. Since LLM inference is usually memory-bandwidth bound, this also makes token generation faster.

But an 8-bit integer can only take 256 distinct values, and a 4-bit integer only 16. A `float32` can take about 4 billion. So we cannot store floats exactly; we need a **mapping** between a real number $r$ (the float) and a quantized code $q$ (the small integer), plus a way to go back.

There are two families of answers to "how do we map":

| Family | Idea | Examples |
|---|---|---|
| **Integer (uniform) quantization** | Lay a **uniform grid** of integers over the data, stretched by a shared scale | `int8`, `int4`, `uint8`, `int2` |
| **Floating-point (non-uniform) quantization** | Store each value as a tiny float with its own exponent; grid is **dense near 0, sparse far out** | `fp8` (E4M3 / E5M2), `fp4` (E2M1), `MXFP4`, `NVFP4` |

Sections 2–7 cover the integer family. Sections 8–10 cover the float family. They solve the same problem with very different tools, so it is worth understanding both.

---

## 2. The affine quantization formula

The general integer mapping is called **affine** (or linear) quantization. "Affine" just means "a linear transform plus a shift", i.e. $y = mx + b$.

**Quantize** (float → integer):

$$q = \operatorname{clamp}\Big(\operatorname{round}\big(\tfrac{r}{S}\big) + Z,\; q_{min},\; q_{max}\Big)$$

**Dequantize** (integer → approximate float):

$$\hat{r} = S \cdot (q - Z)$$

Where:

- $S$ = **scale**. A positive float. It is the *step size* between two neighbouring representable values. Small $S$ means fine steps (good precision) but a narrow range; large $S$ means coarse steps but a wide range.
- $Z$ = **zero-point**. An *integer* in $[q_{min}, q_{max}]$. It is the code that represents the real value $0.0$. Storing it as an integer guarantees that $0.0$ round-trips exactly, which matters a lot (padding, zero-initialised weights, ReLU outputs).
- $[q_{min}, q_{max}]$ = the integer range of the target type.
- $\hat{r}$ = the reconstructed value. It is not exactly $r$; the difference is the quantization error (Section 5).

> Think of $S$ as the spacing of the tick marks on a ruler, and $Z$ as *which tick mark is labelled zero*. Quantization = "which tick is closest to my value?". Dequantization = "read the label off that tick".

### The integer ranges you will see

For an $n$-bit integer:

| Bits | Signed range $[-2^{n-1}, 2^{n-1}-1]$ | Unsigned range $[0, 2^n - 1]$ |
|---|---|---|
| 8-bit | $[-128, 127]$ | $[0, 255]$ |
| 4-bit | $[-8, 7]$ | $[0, 15]$ |
| 2-bit | $[-2, 1]$ | $[0, 3]$ |

- **Asymmetric** quantization typically uses the **unsigned** range.
- **Symmetric** quantization typically uses the **signed** range.

The two schemes differ only in how they pick $S$ and $Z$. Let's derive each.

---

## 3. Asymmetric quantization (the full affine form)

**Goal:** map the real range $[r_{min}, r_{max}]$ *exactly* onto the integer range $[q_{min}, q_{max}]$, so that no integer code is wasted.

### Derivation

We want the two endpoints to line up. Using the dequantize formula $r = S(q - Z)$, that gives two equations:

$$r_{min} = S\,(q_{min} - Z) \qquad (1)$$
$$r_{max} = S\,(q_{max} - Z) \qquad (2)$$

Subtract (1) from (2). The $Z$ cancels:

$$r_{max} - r_{min} = S\,(q_{max} - q_{min})$$

$$\boxed{S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}}$$

This says: *the total real span, divided by the number of integer steps available, is the size of one step.* Which is exactly what a scale should be.

Now plug $S$ back into (1) and solve for $Z$:

$$q_{min} - Z = \frac{r_{min}}{S} \;\Rightarrow\; \boxed{Z = \operatorname{round}\Big(q_{min} - \frac{r_{min}}{S}\Big)}$$

then clamp $Z$ to $[q_{min}, q_{max}]$.

**Why round and clamp $Z$?** $Z$ must be an integer (it is a code), but $r_{min}/S$ is generally not an integer. Rounding introduces a tiny shift, so one endpoint may be very slightly off the exact integer boundary. Clamping guarantees $Z$ is a valid code. Strict implementations recompute $S$ after rounding $Z$; most just accept the tiny error.

**A useful special case:** if $r_{min} = 0$ (e.g. post-ReLU activations) then $Z = q_{min} = 0$ for the unsigned range.

### Worked example: $[-2.5,\ 3.7]$ → `uint8`

$q_{min} = 0,\ q_{max} = 255$.

$$S = \frac{3.7 - (-2.5)}{255 - 0} = \frac{6.2}{255} \approx 0.024314$$

$$Z = \operatorname{round}\Big(0 - \frac{-2.5}{0.024314}\Big) = \operatorname{round}(102.82) = 103$$

Check a few values with $q = \operatorname{round}(r/S) + Z$:

| $r$ | $r/S$ | $q = \operatorname{round}(r/S) + 103$ | dequant $\hat r = S(q-103)$ |
|---|---|---|---|
| $-2.5$ | $-102.82$ | $-103 + 103 = 0$ | $-2.504$ |
| $0.0$ | $0$ | $103$ | $0.0$ (exact) |
| $1.0$ | $41.13$ | $144$ | $0.997$ |
| $3.7$ | $152.18$ | $255$ | $3.696$ |

The whole $[0, 255]$ range is used, $0.0$ is exact, and every reconstructed value is within $S/2 \approx 0.012$ of the original.

### Properties of asymmetric

- ✅ **Tight fit.** Every integer code is used. Best when the data is skewed / one-sided (e.g. ReLU activations that are always $\ge 0$).
- ✅ **Exact zero** via $Z$.
- ❌ **Extra metadata.** Must store both $S$ and $Z$ per tensor (or per channel / group).
- ❌ **Slightly more compute.** The $(q - Z)$ subtraction in dequantization, and cross terms in integer matmul. When you multiply two asymmetric tensors, $(q_a - Z_a)(q_b - Z_b)$ expands to four terms; the kernel has to handle the $Z$ cross terms.

---

## 4. Symmetric quantization

**Idea:** force the zero-point to be exactly $Z = 0$. To make that work, the real range is *made* symmetric around zero:

$$[-|r|_{max},\ +|r|_{max}] \quad\text{where}\quad |r|_{max} = \max(|r_{min}|,\ |r_{max}|)$$

### Derivation

With $Z = 0$ the affine formula collapses to:

$$q = \operatorname{round}\Big(\frac{r}{S}\Big), \qquad \hat r = S \cdot q$$

We want $+|r|_{max}$ to land on the largest positive code:

$$\boxed{S = \frac{|r|_{max}}{q_{max}}} \qquad\text{with}\qquad q_{max} = 2^{n-1} - 1 \;\; (=127 \text{ for int8},\ 7 \text{ for int4})$$

### Two flavours: restricted range vs full range

A signed $n$-bit integer is lopsided: `int8` goes from $-128$ to $+127$. There is one more negative code than positive. So:

- **Restricted range** (most common): use $[-127, 127]$ and $S = |r|_{max} / 127$. The code $-128$ is simply never produced. Clean and symmetric; some hardware / older tensor-core paths prefer this.
- **Full range**: use $[-128, 127]$ and $S = |r|_{max} / 128$ (or $|r|_{max}/127.5$ in some libraries). Slightly finer step, but $-|r|_{max}$ maps to $-128$ while $+|r|_{max}$ would need $+128$, which does not exist, so it clips to $127$. Marginal gain, slight asymmetry.

For this page, assume restricted range unless stated.

### Worked example: same $[-2.5,\ 3.7]$ → `int8` symmetric

$|r|_{max} = \max(2.5, 3.7) = 3.7$.

$$S = \frac{3.7}{127} \approx 0.029134$$

| $r$ | $r/S$ | $q$ | dequant $\hat r = S q$ |
|---|---|---|---|
| $0.0$ | $0$ | $0$ | $0.0$ (exact by construction) |
| $3.7$ | $127.0$ | $127$ | $3.7$ |
| $-2.5$ | $-85.81$ | $-86$ | $-2.506$ |
| $-3.7$ | $-127$ | $-127$ | (never occurs in this data) |

Notice the step is $0.0291$ instead of the asymmetric $0.0243$: about **20% coarser**. Codes $-127 \ldots -87$ are reserved for values in $[-3.7, -2.53]$ which don't exist in this data. That is the "wasted codes" effect.

### Why symmetric "wastes half the codes" (the extreme case)

The short version: **symmetric forces the representable range to be centred on zero, whether or not your data is.**

Take post-ReLU activations living in $[0, 6.0]$ (all non-negative).

- $|r|_{max} = 6.0$, so $S = 6/127 \approx 0.0472$.
- Positive values use codes $[0, 127]$. ✅
- Negative codes $[-127, -1]$ represent values in $[-6.0, -0.047]$. **No such values exist.** ❌

So 127 out of 255 codes (about half) are never used. You paid for them in the scale but got nothing back.

Compare asymmetric `uint8` on the same data: $S = 6/255 \approx 0.0235$. **Half the step size**, every code used, rounding error halved.

> Symmetric does not just "waste codes" in the abstract. It **doubles the step size** on one-sided data, which **doubles the rounding error**.

The rule: symmetric wastes codes in proportion to how one-sided the data is. For zero-centred data (typical of weights) nothing is wasted and symmetric matches asymmetric in accuracy while being simpler.

### Properties of symmetric

- ✅ **No zero-point to store.** Only $S$.
- ✅ **Simpler kernels.** $\hat r = S q$; integer matmul has no cross terms: $(S_a q_a)(S_b q_b) = S_a S_b (q_a q_b)$, so you accumulate plain integer products and multiply by one float at the end.
- ✅ **Exact zero**, by construction.
- ❌ **Wastes range** on skewed data, which increases error.

---

## 5. Side-by-side comparison

| Property | Symmetric | Asymmetric |
|---|---|---|
| Zero-point $Z$ | Always $0$ | Computed and stored |
| Integer range | Signed $[-2^{n-1}, 2^{n-1}-1]$ (usually restricted to $\pm(2^{n-1}-1)$) | Unsigned $[0, 2^n - 1]$ |
| Scale $S$ | $\dfrac{\lvert r\rvert_{max}}{2^{n-1}-1}$ | $\dfrac{r_{max} - r_{min}}{2^n - 1}$ |
| Zero exact? | Yes | Yes (via $Z$) |
| Range utilisation | Wastes codes if data is skewed | Full |
| Metadata | $S$ only | $S$ and $Z$ |
| Dequant | $\hat r = S q$ | $\hat r = S (q - Z)$ |
| Typical use | **Weights** (roughly zero-centred) | **Activations** (skewed, e.g. post-ReLU) |

### A concrete numerical comparison

**Post-ReLU activations in $[0, 6.0]$, 8-bit:**

| Scheme | $S$ | Max rounding error $S/2$ | Codes used |
|---|---|---|---|
| Asymmetric `uint8` | $6/255 = 0.0235$ | $0.0118$ | 256 / 256 |
| Symmetric `int8` | $6/127 = 0.0472$ | $0.0236$ (**2× worse**) | 128 / 255 |

**Weights in roughly $[-0.5, 0.5]$, 8-bit:**

| Scheme | $S$ |
|---|---|
| Symmetric `int8` | $0.5/127 = 0.003937$ |
| Asymmetric `uint8` | $1.0/255 = 0.003922$ |

Essentially identical. So for zero-centred data, symmetric wins: same accuracy, less metadata, simpler kernels. That is why the default recipe is **symmetric for weights, asymmetric for activations**.

---

## 6. Quantization error, clipping, and calibration

### The rounding error bound

For round-to-nearest, the reconstructed value is at most half a step away from the original:

$$|\epsilon| = |r - \hat r| \le \frac{S}{2}$$

(assuming $r$ was inside the representable range). So **smaller $S$ ⇒ smaller error**. This is the single most useful sentence about integer quantization.

### But smaller $S$ means a narrower range

Range covered $= S \times (\text{number of codes})$. With a fixed number of codes, shrinking $S$ shrinks the range. Anything outside gets **clipped** to $q_{min}$ or $q_{max}$, which can be a *huge* error for that value.

So there are two competing error sources:

| Choose | Rounding error | Clipping error |
|---|---|---|
| Small range (small $S$) | Low for everyone | High for the tail / outliers |
| Large range (large $S$) | High for everyone | None |

This is a bias-variance style trade-off, and it is the reason **calibration** exists.

### Calibration

Calibration is the process of choosing $r_{min}, r_{max}$ (equivalently $S, Z$) so that total error is minimised. Common strategies:

- **Min/max**: use the observed extremes. Zero clipping, but one outlier can inflate $S$ for the whole tensor.
- **Percentile**: use e.g. the 99.99th percentile as $r_{max}$. Clip a handful of outliers, gain precision for the other 99.99%.
- **MSE / entropy (KL) search**: sweep candidate ranges and pick the one that minimises reconstruction error or distribution mismatch on a calibration dataset.

For **weights**, you know all the values up front, so calibration is trivial (they are static). For **activations**, values depend on the input, so you either run a calibration dataset through the model and record ranges (**static** quantization) or compute $S, Z$ on the fly per batch (**dynamic** quantization).

---

## 7. Granularity: per-tensor, per-channel, per-group

Everything so far assumed one $(S, Z)$. The question is: *one per what?*

### Per-tensor

One $(S, Z)$ pair for the **entire tensor**. You compute the global $r_{min}, r_{max}$ across every element; everyone shares the same ruler.

```
Tensor: [0.1, 0.2, 5.0, 0.15, -0.3, 0.05, ...]
         └──────────── one S, one Z ──────────┘
```

Simple, minimal metadata, but **one outlier ruins the scale for everybody**.

### Per-channel

One $(S, Z)$ pair **per index along a chosen axis** (the "channel" axis). Each slice perpendicular to that axis gets its own ruler.

```
Weight, 4 output channels:
  ch0: [0.10, 0.20, 0.15, ...]   → S₀, Z₀
  ch1: [0.05, 0.10, 0.08, ...]   → S₁, Z₁
  ch2: [5.00, 4.80, 5.20, ...]   → S₂, Z₂   ← loud channel, only hurts itself
  ch3: [-0.30, -0.20, ...]       → S₃, Z₃
```

**Why it matters — the outlier problem.** Suppose channel 2 has values around $5.0$ and channel 1 has values around $0.05$.

- *Per-tensor*: $S \approx 5/127 \approx 0.039$. Channel 1's values ($0.05, 0.08, 0.10$) round to codes $1, 2, 3$. Precision destroyed.
- *Per-channel*: channel 1 gets $S_1 \approx 0.1/127 \approx 0.0008$. Its values keep full 8-bit resolution. Channel 2's big scale only affects channel 2.

### Counting the scales: how many $(S, Z)$ pairs?

This trips people up, so let's be precise. **Per-channel along axis $k$ means one $(S, Z)$ per index of axis $k$. The number of pairs equals the size of axis $k$**, not "total elements divided by something".

Take a tensor of shape $(5, 4, 3, 2, 7)$, which has $5 \cdot 4 \cdot 3 \cdot 2 \cdot 7 = 840$ elements.

| Granularity | # of $(S, Z)$ pairs | Elements sharing each pair |
|---|---|---|
| Per-tensor | $1$ | $840$ |
| Per-channel, axis 0 (size 5) | $5$ | $840 / 5 = 168$ each |
| Per-channel, axis 4 (size 7) | $7$ | $840 / 7 = 120$ each |
| Per-channel, axis 3 (size 2) | $2$ | $420$ each |

So if axis 0 is the output-channel axis (the PyTorch convention for both `Linear` weights `(out, in)` and conv weights `(out, in, kH, kW)`), you store **5** scales, and each scale covers a slice of 168 elements.

> Which axis is "the channel" is a convention, not something the shape tells you. Frameworks almost always only support per-tensor or per-channel along the **output-channel axis (axis 0)** for weights.

### Per-group (a.k.a. block-wise)

Per-channel can still be too coarse for 4-bit. A single `Linear` row might have 4096 elements sharing one scale. **Per-group** quantization chops each row into contiguous groups of, say, 32, 64, or 128 elements, and gives each group its own $(S, Z)$.

```
Weight (out=4, in=128), group_size=32:
  row 0: [g0: 32 elems][g1: 32][g2: 32][g3: 32]   → 4 scales
  row 1: [g0][g1][g2][g3]                          → 4 scales
  ...
  total = 4 rows × 4 groups = 16 (S, Z) pairs
```

Metadata cost per element: with `int4` weights and a 16-bit scale per group of 128, that's $16/128 = 0.125$ extra bits per weight, so effective bit-width $\approx 4.125$. Cheap for a large accuracy win. Group size 128 is the common GPTQ/AWQ default; block size 32 (MXFP4) and 16 (NVFP4) appear in the float formats below.

### Trade-offs

| | Per-tensor | Per-channel | Per-group |
|---|---|---|---|
| Metadata | 1 scale | 1 per channel | 1 per group |
| Accuracy | Worst (outliers hurt all) | Good | Best |
| Kernel complexity | Simplest | Index into a scale vector | Index scale per group inside the inner loop |
| Typical use | Activations | Weights (int8) | Weights (int4 and below) |

> **One-line intuition:** per-tensor = one ruler for everything. Per-channel = one ruler per channel, so a loud channel can't drown out a quiet one. Per-group = one ruler per small block, so even *within* a channel a local outlier can't hurt its far-away neighbours.

---

## 8. Packing sub-byte integers (nibble packing)

`int8` values are one byte each; storage is trivial. Below 8 bits, memory is still byte-addressable, so multiple values must be **packed** into one byte or word. (The index page covers *why*; this section covers the exact bit gymnastics with the symmetric/asymmetric examples.)

### The nibble convention

For `int4`, each value is a **nibble** (4 bits), and a byte holds two. Following the convention from the index page (first value in the **low** nibble):

```
byte = (q0 & 0xF) | (q1 << 4)

       ┌── high nibble ──┐┌── low nibble ──┐
byte:  [ b7  b6  b5  b4 ][ b3  b2  b1  b0 ]
             q1                  q0

unpack:
  q0 = byte & 0x0F
  q1 = (byte >> 4) & 0x0F
```

Element $i$ lives at byte index $i \,//\, 2$, in the low nibble if $i$ is even and the high nibble if $i$ is odd. Whether the first value goes low or high is just a convention; every kernel picks one and sticks with it.

**Sign matters when unpacking.** `byte & 0x0F` yields an *unsigned* nibble in $[0, 15]$. For symmetric `int4` you must reinterpret it as two's-complement $[-8, 7]$ (e.g. `q = q - 16 if q >= 8`, or shift left 4 and arithmetic-shift right 4). For asymmetric `uint4`, the unsigned nibble is already the code.

### Worked int4 example (symmetric)

Weights: $[-1.2,\ 0.4,\ 0.9,\ -0.3]$. $|r|_{max} = 1.2$, $q_{max} = 7$.

$$S = 1.2 / 7 \approx 0.1714$$

| $r$ | $r/S$ | $q$ | two's-complement nibble |
|---|---|---|---|
| $-1.2$ | $-7.0$ | $-7$ | `1001` |
| $0.4$ | $2.33$ | $2$ | `0010` |
| $0.9$ | $5.25$ | $5$ | `0101` |
| $-0.3$ | $-1.75$ | $-2$ | `1110` |

Pack (first value in low nibble):

```
byte 0 = (q1 << 4) | q0 = 0010 1001 = 0x29
byte 1 = (q3 << 4) | q2 = 1110 0101 = 0xE5
```

Dequantize: $\hat r = 0.1714 \cdot q$ → $[-1.2,\ 0.343,\ 0.857,\ -0.343]$.

Codes used: $-7, -2, 2, 5$. Codes $-8, 6, 7$ and most of the range go unused because the data is lopsided ($-1.2$ vs $+0.9$).

### Worked int4 example (asymmetric)

Same values, `uint4` $[0, 15]$. $r_{min} = -1.2,\ r_{max} = 0.9$.

$$S = \frac{0.9 - (-1.2)}{15} = \frac{2.1}{15} = 0.14$$
$$Z = \operatorname{round}\Big(0 - \frac{-1.2}{0.14}\Big) = \operatorname{round}(8.571) = 9$$

| $r$ | $r/S + Z$ | $q$ | nibble |
|---|---|---|---|
| $-1.2$ | $-8.571 + 9 = 0.43$ | $0$ | `0000` |
| $0.4$ | $2.857 + 9 = 11.857$ | $12$ | `1100` |
| $0.9$ | $6.43 + 9 = 15.43$ | $15$ | `1111` |
| $-0.3$ | $-2.143 + 9 = 6.857$ | $7$ | `0111` |

Pack:

```
byte 0 = (12 << 4) | 0  = 1100 0000 = 0xC0
byte 1 = (7 << 4)  | 15 = 0111 1111 = 0x7F
```

Dequantize: $\hat r = 0.14 \cdot (q - 9)$ → $[-1.26,\ 0.42,\ 0.84,\ -0.28]$.

Now both ends of the code range ($0$ and $15$) are used, and the step $0.14$ is smaller than the symmetric $0.1714$. Asymmetric fits skewed data better, at the price of storing $Z = 9$ and doing a subtraction on unpack.

### Packing and the zero-point

With per-channel or per-group asymmetric `int4`, adjacent nibbles in one byte may belong to different groups with different $Z$. The kernel must track which $Z$ applies to which nibble, which complicates the fused unpack-and-multiply path. Symmetric packing has no $Z$, so the unpacked nibble can go straight into an integer dot-product instruction. This is one reason **symmetric per-channel / per-group is the default for weights**.

---

## 9. Low-precision floating point: FP8, FP4

Everything above is **integer** quantization: a uniform grid stretched by a scale. **Float** formats are a fundamentally different family. There is no separate $S$ and $Z$ baked into the *format*; instead every value carries an exponent that acts as its own scale.

### The layout: sign | exponent | mantissa

Every IEEE-style float is three bit-fields:

```
[ s | e e e e | m m m ]     ← FP8 E4M3: 1 sign, 4 exponent, 3 mantissa
[ s | e e | m ]             ← FP4 E2M1: 1 sign, 2 exponent, 1 mantissa
```

The naming is **E**(exponent bits)**M**(mantissa bits). The sign bit is always there and not counted in the name.

| Format | Total bits | Sign | Exponent | Mantissa | Bias | Notes |
|---|---|---|---|---|---|---|
| FP32 | 32 | 1 | 8 | 23 | 127 | Baseline |
| FP16 | 16 | 1 | 5 | 10 | 15 | Standard half |
| BF16 | 16 | 1 | 8 | 7 | 127 | Same range as FP32, less precision |
| FP8 E4M3 | 8 | 1 | 4 | 3 | 7 | More precision, less range; **weights & activations** |
| FP8 E5M2 | 8 | 1 | 5 | 2 | 15 | More range, less precision; **gradients** |
| FP4 E2M1 | 4 | 1 | 2 | 1 | 1 | Only 16 values |

### Decoding: getting a number out of the bits

For a **normal** number (exponent field not all-zeros and not the reserved all-ones pattern):

$$\text{value} = (-1)^{s} \times 2^{\,(E - \text{bias})} \times \Big(1 + \frac{M}{2^{m}}\Big)$$

Where:

- $s$ = sign bit. $0$ → positive, $1$ → negative.
- $E$ = the exponent field read as an **unsigned** integer.
- $\text{bias} = 2^{(\text{exp bits} - 1)} - 1$. It lets the unsigned $E$ represent *negative* powers of 2 without a second sign bit. (E4 → bias 7, E5 → bias 15, E2 → bias 1, E8 → bias 127.)
- $M$ = the mantissa field read as an unsigned integer, and $m$ = the number of mantissa bits.
- The "$1 +$" is the **implicit leading one**. Since every normal number in binary scientific notation starts with a $1$, the format doesn't bother storing it. That gives you one extra bit of precision for free.

#### Why is the mantissa divided by $2^m$?

Because the mantissa bits are a **binary fraction**: digits *after* the binary point, not an integer.

```
    1 . m₂ m₁ m₀        (E4M3: three mantissa bits after the point)
    ↑
implicit 1
```

With bits `101`:

$$0.101_2 = \frac{1}{2} + \frac{0}{4} + \frac{1}{8} = \frac{4 + 0 + 1}{8} = \frac{5}{8} = 0.625$$

And "read `101` as the integer 5, then divide by $2^3 = 8$" gives exactly the same $5/8$. That is all the division is doing: converting "integer spelled by the bits" into "fraction in $[0, 1)$".

| Mantissa bits $m$ | Divide by $2^m$ | Possible fractions |
|---|---|---|
| 1 | 2 | $\{0, 0.5\}$ |
| 2 | 4 | $\{0, 0.25, 0.5, 0.75\}$ |
| 3 | 8 | $\{0, 0.125, \ldots, 0.875\}$ |
| 10 (FP16) | 1024 | steps of $1/1024$ |

#### Worked example: FP8 E4M3

Decode the byte `0 1000 101`:

- $s = 0$ → positive
- $E = 1000_2 = 8$; bias $= 7$; exponent $= 8 - 7 = 1$
- $M = 101_2 = 5$; $m = 3$; fraction $= 5/8 = 0.625$

$$\text{value} = (+1) \times 2^{1} \times (1 + 0.625) = 2 \times 1.625 = 3.25$$

#### Worked example: FP4 E2M1

Decode `1 10 1`:

- $s = 1$ → negative
- $E = 10_2 = 2$; bias $= 1$; exponent $= 1$
- $M = 1$; $m = 1$; fraction $= 1/2 = 0.5$

$$\text{value} = (-1) \times 2^{1} \times (1 + 0.5) = -3.0$$

### Special bit patterns

Every float format reserves some patterns:

| Exponent field | Mantissa | Meaning |
|---|---|---|
| all zeros | $0$ | $\pm 0$ |
| all zeros | $\ne 0$ | **Subnormal** (see below) |
| all ones | $0$ | $\pm\infty$ |
| all ones | $\ne 0$ | NaN |

**FP8 E4M3 is deliberately non-standard.** Because 8 bits are so scarce, it has **no infinity**; the all-ones exponent patterns are used as ordinary numbers, extending the max to $448 = 2^{8} \times 1.75$. Only one pattern per sign, `S.1111.111`, is reserved for NaN. E5M2 follows the IEEE rules (has inf and NaN, max $57344$). **FP4 E2M1 has no inf or NaN at all**; all 16 patterns are numbers.

### Subnormals: no cliff next to zero

Normal numbers have the implicit $1$, so the smallest normal is $2^{(1 - \text{bias})} \times 1.0$. Below that there would be a **gap** straight down to $0$ with nothing representable in between.

Subnormals fill that gap. When the exponent field is **all zeros**:

- Drop the implicit 1: the significand becomes $0.M$ instead of $1.M$.
- Fix the exponent at $1 - \text{bias}$ (the same as the smallest normal, *not* $0 - \text{bias}$; this keeps the spacing continuous).

$$\text{value}_{\text{subnormal}} = (-1)^{s} \times 2^{\,(1 - \text{bias})} \times \frac{M}{2^{m}}$$

**Example, E4M3 (bias 7):**

- Smallest normal: $E = 1, M = 0$ → $2^{-6} \times 1.0 = 0.015625$
- Subnormals ($E = 0$): $2^{-6} \times \{1/8, 2/8, \ldots, 7/8\}$ = $0.00195,\ 0.00391,\ \ldots,\ 0.01367$
- Then $0$.

So values fade smoothly toward zero instead of snapping. **Cost:** some hardware handles subnormals on a slow path, so kernels often flush them to zero (FTZ / DAZ flags). For quantization you mostly just need to know they exist and that E2M1 *relies* on them for its $\pm 0.5$ values.

### The complete FP4 E2M1 table (all 16 values)

Bias $= 1$. With $E=0$ the value is subnormal: $2^{0} \times M/2$.

| bits `s e m` | value | | bits `s e m` | value |
|---|---|---|---|---|
| `0 00 0` | $+0.0$ | | `1 00 0` | $-0.0$ |
| `0 00 1` | $+0.5$ (subnormal) | | `1 00 1` | $-0.5$ |
| `0 01 0` | $+1.0$ | | `1 01 0` | $-1.0$ |
| `0 01 1` | $+1.5$ | | `1 01 1` | $-1.5$ |
| `0 10 0` | $+2.0$ | | `1 10 0` | $-2.0$ |
| `0 10 1` | $+3.0$ | | `1 10 1` | $-3.0$ |
| `0 11 0` | $+4.0$ | | `1 11 0` | $-4.0$ |
| `0 11 1` | $+6.0$ | | `1 11 1` | $-6.0$ |

Positive side: $0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6$. Look at the gaps: $0.5, 0.5, 0.5, 0.5, 1, 1, 2$. **Dense near zero, sparse far out.** That non-uniform spacing is the entire point of using a float format, and it matches how neural-net weights are distributed (most small, a few large).

### A reference decoder

```python
def decode_fp(s, E, M, exp_bits, man_bits):
    bias = (1 << (exp_bits - 1)) - 1
    sign = -1.0 if s else 1.0
    if E == 0:                                   # zero or subnormal
        return sign * 2.0 ** (1 - bias) * (M / 2 ** man_bits)
    if E == (1 << exp_bits) - 1 and exp_bits >= 5:  # IEEE-style inf/nan (not E4M3/E2M1)
        return float("inf") * sign if M == 0 else float("nan")
    return sign * 2.0 ** (E - bias) * (1 + M / 2 ** man_bits)

decode_fp(0, 0b1000, 0b101, 4, 3)   # 3.25
decode_fp(1, 0b10,   0b1,   2, 1)   # -3.0
```

### Nibble packing vs bit-field packing

Both put several values in one byte / word. The difference is what the bits *mean* once extracted.

- **Nibble packing (int4):** extract 4 bits with mask + shift, and you are done; the nibble *is* the integer code.
- **Bit-field packing (fp4, fp8, mixed widths):** extract the value's bits, then **decode the sub-fields** (sign, exponent, mantissa) to get a number. Widths may also be non-4 (6-bit FP6, 3-bit, etc.), so values need not align to nibbles.

```
32-bit word holding 8 × FP4:
  [v7][v6][v5][v4][v3][v2][v1][v0]      each 4 bits

v = (word >> (4 * i)) & 0xF     # extract
s = (v >> 3) & 1                 # sign
e = (v >> 1) & 0b11              # exponent (2 bits)
m =  v       & 1                 # mantissa (1 bit)
value = decode_fp(s, e, m, 2, 1)
```

In practice FP4 kernels skip the arithmetic and use a 16-entry **lookup table** (index by the 4-bit code, get the value), which is faster than decoding.

> Nibble packing is a special case of bit-field packing where the width is 4 and no sub-field decoding is needed.

---

## 10. Why floats handle outliers gracefully

This is the sentence to internalise: **in a float, every value carries its own scale in its exponent.**

### The exponent is a per-value power-of-two multiplier

$$\text{value} = 2^{(E - \text{bias})} \times (1 + M/2^m)$$

The factor $2^{(E - \text{bias})}$ is baked into each value individually. Different values have different $E$, so different multipliers. Look at how the **gap** between neighbouring representable values changes with $E$ in E4M3:

| $E$ | $2^{E-7}$ | Representable values (mantissa steps of $1/8$) | Gap |
|---|---|---|---|
| 7 | $1$ | $1.0,\ 1.125,\ 1.25,\ \ldots,\ 1.875$ | $0.125$ |
| 10 | $8$ | $8,\ 9,\ 10,\ \ldots,\ 15$ | $1.0$ |
| 4 | $1/8$ | $0.125,\ 0.1406,\ \ldots,\ 0.2344$ | $0.0156$ |

Same mantissa bits, different gap, because the exponent scales everything. In general:

$$\text{gap} = 2^{(E - \text{bias})} \times \frac{1}{2^{m}}$$

Small values get a small exponent → small gap → fine precision. Large values get a large exponent → large gap → coarse precision. **The exponent plays exactly the role that the shared scale $S$ plays in integer quantization, except it is chosen per value, not per tensor.**

### Contrast with the integer case

Suppose a tensor holds $[0.01,\ 0.02,\ 500.0]$.

**Integer `int8` (one shared ruler):**
- $S = 500/127 \approx 3.94$
- $0.01 → \operatorname{round}(0.0025) = 0$; $0.02 → 0$; $500 → 127$
- The two small values are **wiped out**. The outlier ate the entire range.

**Float `fp8` E4M3 (each value has its own ruler):**
- $0.01$ → falls in the subnormal range ($2^{-6} \times 5/8$); nearest representable $\approx 0.00977$. Fine.
- $0.02$ → exponent $2^{-6}$; nearest $\approx 0.0195$. Fine.
- $500$ → clips to $448$ (E4M3 max). Coarse, but present.
- All three coexist. The outlier only affects **its own** precision.

> **Integer quantization = one shared ruler.** One outlier forces a large step for everyone.
> **Float = every value picks its own ruler via the exponent.** Outliers don't steal precision from small values.

### Why this matters for LLMs

LLM activations have massive outliers: a few channels can be 100× larger than the rest. With `int8` you need tricks like SmoothQuant (migrate activation outliers into the weights) to survive that. With `fp8`, the exponent absorbs the outliers with no calibration dataset. Combined with native FP8 tensor cores on Hopper and FP4 on Blackwell, that is why float formats became the default for LLM inference.

### The cost

Float trades precision in the tails for range. A value near $400$ in E4M3 has a gap of $2^{8}/8 = 32$ between neighbours. That is fine for neural nets, which tolerate relative error and rarely need large values to be exact.

---

## 11. Quantizing FP32 → FP4 (block scaling)

Raw FP4 E2M1 can only represent $[-6, 6]$, and only 16 distinct values. Real tensors don't fit. So real systems use **block scaling**: split the tensor into small blocks and store one higher-precision scale per block. Each element is then stored as an FP4 code *relative to* its block's scale.

### The pipeline

1. **Reshape** the tensor into blocks of $B$ contiguous elements ($B = 32$ for MXFP4, $16$ for NVFP4).
2. **Compute a per-block scale** so the block's largest magnitude lands on the largest FP4 value:
   $$s_{block} = \frac{\max_{\text{block}} |x|}{6.0}$$
3. **Normalise** the block: $x' = x / s_{block}$, now in $[-6, 6]$.
4. **Round each $x'$ to the nearest FP4 value**. The grid is non-uniform, so this is a lookup against the 8 positive values $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ (and their negatives), not a simple divide-and-round. Practically: compare against the midpoints $\{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0\}$.
5. **Emit the 4-bit code** for that FP4 value and **pack** two per byte (or 8 per 32-bit word).
6. **Store $s_{block}$** next to the packed data. Dequantize as $\hat x = s_{block} \times \text{decode\_fp4}(\text{code})$.

### Worked numerical example (block of 8, for brevity)

Block: $[0.12,\ -0.05,\ 0.33,\ 0.90,\ -0.60,\ 0.02,\ 0.45,\ -0.21]$

- $\max |x| = 0.90$, so $s_{block} = 0.90 / 6 = 0.15$

| $x$ | $x' = x / 0.15$ | nearest FP4 | code `s e m` | $\hat x = 0.15 \times \text{FP4}$ | error |
|---|---|---|---|---|---|
| $0.12$ | $0.80$ | $1.0$ | `0 01 0` = `0x2` | $0.150$ | $0.030$ |
| $-0.05$ | $-0.33$ | $-0.5$ | `1 00 1` = `0x9` | $-0.075$ | $0.025$ |
| $0.33$ | $2.20$ | $2.0$ | `0 10 0` = `0x4` | $0.300$ | $0.030$ |
| $0.90$ | $6.00$ | $6.0$ | `0 11 1` = `0x7` | $0.900$ | $0$ |
| $-0.60$ | $-4.00$ | $-4.0$ | `1 11 0` = `0xE` | $-0.600$ | $0$ |
| $0.02$ | $0.13$ | $0.0$ | `0 00 0` = `0x0` | $0.000$ | $0.020$ |
| $0.45$ | $3.00$ | $3.0$ | `0 10 1` = `0x5` | $0.450$ | $0$ |
| $-0.21$ | $-1.40$ | $-1.5$ | `1 01 1` = `0xB` | $-0.225$ | $0.015$ |

Packed (first element in low nibble): `0x92, 0x74, 0x0E, 0xB5`, plus the scale $0.15$.

Storage: 8 elements × 4 bits = 32 bits, plus one scale. With a 16-element block and an 8-bit scale (NVFP4), that's $4 + 8/16 = 4.5$ bits per element.

### FP4 vs INT4 on the same block

With **symmetric `int4`** ($S = 0.9/7 = 0.1286$, uniform steps), the small values $0.02$ and $-0.05$ would both round to $0$, and $0.12$ would round to code $1$ ($=0.129$). Roughly comparable here, but as the spread inside a block grows (say a block containing both $0.01$ and $0.9$), FP4's dense-near-zero grid keeps resolving the small values while int4's uniform grid flattens them. In exchange FP4 has only 15 distinct non-zero magnitudes vs int4's 15 uniform ones, so on *evenly spread* data int4 can be slightly better. Which wins is empirical; for LLM weights FP4 with fine block scaling tends to.

### Pseudocode

```python
FP4_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def quantize_fp32_to_fp4(x, block_size=16):
    blocks = x.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1) / 6.0          # one scale per block
    normalized = blocks / scales[:, None]              # now in [-6, 6]
    codes = nearest_fp4_code(normalized)               # lookup on the 16-value grid
    packed = pack_nibbles(codes)                       # 2 codes per byte
    return packed, scales

def dequantize(packed, scales, block_size=16):
    codes = unpack_nibbles(packed).reshape(-1, block_size)
    return fp4_lookup(codes) * scales[:, None]
```

### MXFP4 vs NVFP4

Both use **exactly the same** E2M1 element format and the same 16 values. They differ only in the scaling scheme.

| | MXFP4 (OCP Microscaling) | NVFP4 (NVIDIA) |
|---|---|---|
| Element format | E2M1 | E2M1 |
| Block size | 32 | **16** (finer) |
| Block scale format | **E8M0** (8-bit exponent only, powers of 2) | **E4M3** (8-bit float, fractional) |
| Per-tensor scale | none | **FP32** $s_{global}$ |
| Decode | $\hat x = \text{fp4} \times s_{block}$ | $\hat x = \text{fp4} \times s_{block} \times s_{global}$ |

**Why E4M3 block scales are better than E8M0.** E8M0 can only be a power of two: $1, 2, 4, 8, \ldots$. If a block's max is $0.9$, the ideal scale is $0.15$, but E8M0 has to pick $0.125$ or $0.25$. Pick $0.25$ and the block's max lands at $3.6$ instead of $6.0$, so the top part of the FP4 range is wasted. E4M3 can represent $0.15$ (approximately $0.15625 = 2^{-3} \times 1.25$), fitting the block much more tightly. Halving the block size to 16 doubles the number of chances to adapt locally.

**Where does $s_{global}$ come from?** The block scale is stored as E4M3, whose max is $448$. So the largest value the two-level scheme can encode is $448 \times 6$. The global scale exists to shrink the whole tensor so that its largest element fits under that ceiling:

$$s_{global} = \frac{\max_{\text{tensor}} |x|}{448 \times 6}$$

It is computed **once, from the whole tensor's absolute maximum**, before any block scale. Then each block's scale is computed relative to it:

$$s_{block} = \frac{\max_{\text{block}} |x| \,/\, 6}{s_{global}} \quad\text{(then rounded to E4M3)}$$

and each element is encoded as $x / (s_{block} \cdot s_{global})$ rounded to FP4. The global scale answers "how much must I shrink the whole tensor so the biggest block scale is representable?"; the block scales fill in the local detail. For weights, $s_{global}$ is computed offline; for activations it is either calibrated ahead of time or computed dynamically from the batch's amax.

> **Summary of the two families of scaling:**
> - Integer: $\hat r = S (q - Z)$, uniform grid, one scale per tensor / channel / group.
> - Block-scaled float: $\hat x = \text{fp4}(\text{code}) \times s_{block}\ [\times s_{global}]$, non-uniform grid, one scale per small block, no zero-point (the sign bit handles sign, and $0$ is native).

---

## 12. What to actually remember

You need to **reason** about these formats, not compute them by hand. Libraries (`torch.float8_e4m3fn`, `torchao`, CUTLASS) do the bit manipulation.

**Memorise (comes up in interviews and code review constantly):**

- Affine: $q = \operatorname{round}(r/S) + Z$, $\hat r = S(q - Z)$. Error $\le S/2$.
- Asymmetric: $S = (r_{max} - r_{min})/(2^n - 1)$, $Z = \operatorname{round}(q_{min} - r_{min}/S)$. Full range, exact zero, stores $Z$.
- Symmetric: $S = |r|_{max}/(2^{n-1} - 1)$, $Z = 0$. Simpler, wastes codes on one-sided data (2× error on post-ReLU).
- Weights → symmetric per-channel / per-group. Activations → asymmetric per-tensor (or fp8).
- Per-channel along axis $k$ ⇒ (size of axis $k$) scales. Per-group ⇒ one scale per $G$ contiguous elements.
- Float decode: $(-1)^s \times 2^{E - \text{bias}} \times (1 + M/2^m)$, bias $= 2^{e-1} - 1$.
- Exponent = per-value scale ⇒ dense near 0, sparse far out ⇒ outliers don't hurt small values.
- E4M3 for weights/activations (precision), E5M2 for gradients (range). E4M3 has no inf, max 448.
- FP4 E2M1 = 16 values, $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ and negatives; needs block scaling.
- MXFP4: block 32, E8M0 scale. NVFP4: block 16, E4M3 scale + FP32 global scale.

**Know the concept, look up the details:**

- Subnormals exist (no cliff near zero); E2M1's $\pm 0.5$ *are* subnormals.
- Restricted vs full symmetric range ($\pm 127$ vs $[-128, 127]$).
- Exact bit patterns and tables; the bias for a given format.

**Next topics:** GPTQ / AWQ (Hessian- and activation-aware weight rounding), SmoothQuant (migrate activation outliers into weights), NF4 / QLoRA (non-uniform 4-bit "normal float"), rounding modes (nearest-even vs stochastic), and the straight-through estimator for quantization-aware training.
