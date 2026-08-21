---
title: CUDA Architecture Suffixes

type: docs
math: true
prev: docs/
weight: 40
sidebar:
  open: false
---

> [!NOTE] CUDA Architecture Suffixes: `a`, `f`, and the `compute_/sm_` Namespaces
> `a` = architecture-specific, `f` = family
> - suffixes (`f` & `a`) was introduced from **`Hopper series (9.x)`**


Companion notes to the binary-toolchain doc. Covers arch-specific (`a`) and family (`f`) suffixes, why they exist on *both* the virtual and real side, compatibility rules, and the "why would anyone embed PTX for an `a` arch at all?" question.

---

## 1. Two namespaces, one suffix system

The suffix is **not a PTX-only concept**. It exists in both namespaces because it describes a *feature tier of the target*, and both stages of the pipeline need to know the target's tier:

| Stage | Namespace | Examples | Controls |
|---|---|---|---|
| Frontend → PTX | virtual: `compute_XX[a\|f]` | `compute_100`, `compute_100f`, `compute_100a` | which instructions are *legal to generate* |
| ptxas → SASS | real: `sm_XX[a\|f]` | `sm_100`, `sm_100f`, `sm_100a` | which instructions are *emitted*, and how the cubin is labeled |

A cubin built for `sm_100a` is stamped as `sm_100a` in its ELF metadata — the loader will not treat it as a generic `sm_100` image, because it may contain instructions other 10.x chips physically lack. That's why the suffix must exist on the `sm_` side, not just in PTX land.

**Consistency rule:** virtual feature tier ≤ real feature tier.

```
compute_100  → sm_100 ✓   sm_100f ✓   sm_100a ✓
compute_100f → sm_100 ✗   sm_100f ✓   sm_100a ✓
compute_100a → sm_100 ✗   sm_100f ✗   sm_100a ✓
```

You can't compile `compute_100a` PTX to plain `sm_100` — the specialized instructions have nowhere to go.

---

## 2. The three tiers

### Plain (`sm_100` / `compute_100`) — baseline, portable
- Features guaranteed across the **whole major version**, all minors, forever within 10.x.
- SASS binary-compatible with any 10.x chip where minor ≥ compiled minor (sm_100 cubin runs on sm_103).
- PTX (`compute_100`) is **forward-compatible via JIT** onto future majors too — the classic PTX story.

### `f` — family (CUDA 12.9+)
- `sm_100f` / `compute_100f`: most of the accelerated, architecture-class features (tcgen05-tier tensor core ops etc.) while staying **compatible across the family** — sm_100, sm_103, and future same-major chips that support the family feature set.
- The answer to "I want the fast instructions but I refuse to build one cubin per die."
- This is what serious datacenter-Blackwell kernels (CUTLASS-class, FlashAttention-class) mostly target.
- `f` PTX is family-portable, **not** portable to other majors.

### `a` — architecture-specific
- `sm_100a` / `compute_100a`: **everything** the chip has, including instructions that exist only on that exact compute capability.
- Runs on that one CC and nothing else. sm_100a → 10.0 silicon only. Not 10.3, not future 10.x, nothing.
- No forward compatibility in *either* representation: `compute_100a` PTX is exact-match only — the driver will not JIT it onto sm_103 or anything else.
- Price of admission for the last few percent (or the newest instructions before they graduate to a family/baseline tier).

Mental shorthand: **suffix = feature tier, prefix = pipeline stage.**

---

## 3. Compatibility matrix (datacenter Blackwell as the example)

| Build target | 10.0 (B200) | 10.3 (B300) | future 10.x | RTX 50xx (12.0) |
|---|---|---|---|---|
| `sm_100` (SASS) | ✓ | ✓ | ✓ (minor≥0) | ✗ |
| `sm_100f` (SASS) | ✓ | ✓ | ✓ (if family-compatible) | ✗ |
| `sm_100a` (SASS) | ✓ | ✗ | ✗ | ✗ |
| `sm_103a` (SASS) | ✗ | ✓ | ✗ | ✗ |
| `compute_100` (PTX) | ✓ JIT | ✓ JIT | ✓ JIT (even future majors) | ✗* |
| `compute_100f` (PTX) | ✓ JIT | ✓ JIT | family only | ✗ |
| `compute_100a` (PTX) | ✓ JIT | ✗ | ✗ | ✗ |

Traps worth memorizing:

- **"Blackwell" is marketing, compute capability is what matters.** RTX 50-series is `sm_120` — major 12, a different binary family from `sm_100` despite the shared codename. Datacenter and consumer Blackwell need separate arch-list entries. (*Plain PTX for a lower major can generally JIT upward across majors, but you should ship native sm_120/sm_120f for consumer cards rather than rely on that.)
- **The 10.x family is sm_100 and sm_103.** `sm_101` was Thor (automotive), renamed `sm_110` in CUDA 13.0 — you'll see both names depending on toolkit version. `sm_121` is GB10 (DGX Spark).
- Same suffix logic applies on Hopper: `sm_90a` (wgmma, TMA intrinsics) is 9.0-only; there was no `f` tier yet in the Hopper era, which is a big part of why `f` was introduced.

---

## 4. "Why would anyone embed `compute_100a` PTX? SASS-only would be the same, but faster"

Short answer: **you're right, and so does NVIDIA — official guidance for `a` targets is ship SASS (`code=sm_100a`), skip the PTX.** `a`-suffixed PTX has no forward-compat value (exact-match only), so embedding it just adds binary size and a pointless JIT path.

Why `compute_100a` still shows up everywhere anyway:

1. **The virtual arch is required at compile time regardless.**
   `-gencode arch=compute_100a,code=sm_100a` needs `compute_100a` as the frontend target just to make the arch-specific instructions *legal in your source*. Using the virtual arch ≠ embedding PTX. Only `code=compute_100a` embeds PTX; `code=sm_100a` uses it transiently and ships SASS.

2. **JIT-native pipelines have no choice.**
   Triton, NVRTC, anything generating code at runtime — their interchange format *is* PTX. A Triton kernel using Blackwell-specific features emits `compute_100a` PTX and runs ptxas on the spot. The PTX isn't there for forward compat; it's just the on-the-way-to-a-cubin representation.

3. **Re-JIT with a newer driver's ptxas** (marginal).
   Embedded PTX can be recompiled by a later driver's improved/bug-fixed ptxas. Real benefit, rarely worth the first-load JIT cost.

Decision tree:

```
Need only baseline features?        → sm_100 SASS + compute_100 PTX (future-proof)
Need accelerated, family-portable?  → sm_100f SASS (± compute_100f PTX for future family members)
Need absolute max on one chip?      → sm_100a SASS only. No PTX. You're welded to the die anyway.
```

---

## 5. How this looks in real build systems

```bash
# nvcc, explicit:
nvcc foo.cu -o foo \
  -gencode arch=compute_100f,code=sm_100f \   # datacenter Blackwell family
  -gencode arch=compute_120,code=sm_120 \     # consumer Blackwell
  -gencode arch=compute_120,code=compute_120  # PTX for future majors

# nvcc shorthand — expands to compute_100a PTX (transient) + sm_100a SASS:
nvcc -arch=sm_100a foo.cu

# PyTorch / CMake style arch lists — where you'll usually *see* these names:
TORCH_CUDA_ARCH_LIST="9.0a;10.0f;12.0"
CMAKE_CUDA_ARCHITECTURES="90a;100f;120"
```

Verifying what actually got embedded:

```bash
cuobjdump -lelf libfoo.so    # cubins are labeled with real arch: look for sm_100a vs sm_100f vs sm_100
cuobjdump -lptx libfoo.so    # if compute_100a shows up HERE, someone embedded pointless PTX
```


That last line is the practical audit for the question in [Q4](#4-why-would-anyone-embed-compute_100a-ptx-sass-only-would-be-the-same-but-faster): `compute_100a` appearing in `-lptx` output is dead weight; `sm_100a` in `-lelf` output is fine and intended.

- Trying on tesla T4:

```bash
nvcc main.cu -o meow \
        -gencode arch=compute_75,code=sm_75,compute_75 \
        -gencode arch=compute_90a,code=sm_90a,compute_90a \
        -gencode arch=compute_89,code=sm_89
```
- and then verify with:
```bash
cuobjdump -lelf meow
cuobjdump -lptx meow
```

---

## 6. Cheat sheet

| Name | Side | Tier | Portability |
|---|---|---|---|
| `compute_100` | virtual | baseline | JIT to all 10.x **and** future majors |
| `compute_100f` | virtual | family | JIT within 10.x family only |
| `compute_100a` | virtual | arch-specific | JIT to 10.0 only (i.e. useless to embed) |
| `sm_100` | real | baseline | runs on 10.x, minor ≥ 0 |
| `sm_100f` | real | family | runs on family-compatible 10.x |
| `sm_100a` | real | arch-specific | runs on 10.0 only |

- Suffix = feature tier (`∅` < `f` < `a`), prefix = stage (virtual vs real).
- Virtual tier must be ≤ real tier at compile time.
- `a` kills forward compat in both representations → SASS-only is the right way to ship it.
- `f` (CUDA 12.9+) exists precisely to escape the "portable OR fast, pick one" trap that `a` created on Hopper.

--

> [!IMPORTANT] **Family compatibility in simple terms:**
> Within one major CC version, plain `sm_XX` binaries are guaranteed to run on every chip of that major, forever — even if the chips are actually different GPU generations. Ada proves this: it's CC 8.9, same major as Ampere (8.0, 8.6, 8.7), so all plain `sm_80` cubins run on a 4090 even though Ada and Ampere are different architectures. The `f` tier doesn't get this blanket promise — `sm_100f` runs only on chips NVIDIA explicitly declares members of the sm_100 family (today: sm_100, sm_103), and family membership can't be derived from the number, it's NVIDIA's per-chip call. So if a hypothetical sm_105 launched tomorrow as a different design sharing major 10 (an "Ada move"), `sm_100` SASS would be guaranteed to run on it, while `sm_100f` would run only if NVIDIA put it in the family. Same logic on the PTX side: `compute_100` JITs forward onto any newer chip including future majors, `compute_100f` only within the family. Bottom line: plain tier = unconditional guarantee across the major; `f` tier = same reach today, but a bet on NVIDIA's family roster tomorrow. That's why the safe shipping recipe is `sm_100f` SASS for speed plus `compute_100` PTX as the unconditional fallback.
