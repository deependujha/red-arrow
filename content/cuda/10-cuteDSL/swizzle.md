---
title: Swizzle
type: docs
math: true
sidebar:
  open: false
weight: 1008
---

This page derives a swizzle from scratch for **any** smem tile and **any** tiled copy that reads or writes it. Every assumption is stated where it's used. For the same steps done with real numbers, see the [worked example](#10-worked-example).

The method in one line:

> Find the phases → find each thread's **slot** → if two threads in a phase share a slot, find the bits that **separate** them (source) and the slot bits **nobody uses** (destination) → XOR source into destination. That XOR is `Swizzle<B, M, S>`.

---

## 0. Setup and assumptions

| Symbol | Meaning |
|---|---|
| `R × C` | smem tile shape, in elements |
| `E` | bytes per element (fp16 → 2) |
| `(Tr, Tc)` | the tiled copy's thread layout |
| `(1, V)` | the tiled copy's value layout: each thread copies `V` contiguous elements of one row per atom call |
| `W = V · E` | bytes per thread per atom call (128-bit copy → 16) |

Assumptions:

- **A1.** The smem tile is **row-major** before swizzling, i.e. layout `(R, C):(C, 1)`. For a column-major tile, swap the roles of rows and columns everywhere.
- **A2.** `C` and `V` are powers of two. That's what makes "row bits" and "column bits" separate bit ranges.
- **A3.** `W` is 4, 8 or 16 bytes, and each thread's `V` elements start at a column that's a multiple of `V`. Every access is then `W`-aligned. (Accesses smaller than 4 B share words between threads and follow different rules; this page doesn't cover them.)
- **A4.** The smem base address is **128 B-aligned** (`byte_alignment=128` or more), so offset-based bank math equals real bank math.
- **A5.** Phases: a warp's smem request moves at most 128 B per pass, so it's served in groups of `P = 128 / W` **consecutive lanes**. This is the standard working model; NVIDIA doesn't officially document it.
- **A6.** The swizzle acts on the **element offset**, which is how CuTe applies a swizzle composed with an element layout.

---

## 1. Offset bits and their three zones

Element `(r, c)` sits at offset

$$
\text{off}(r, c) = r \cdot C + c
$$

Because `C` is a power of two (A2), the low $\log_2 C$ bits are the column and everything above is the row.

Split the offset bits into three zones. Define

$$
v = \log_2 V \qquad L = \log_2 \frac{128}{E}
$$

| Zone | Bits | What it decides |
|---|---|---|
| **vector** | `[0, v)` | position inside one thread's copy. Never touch these. |
| **slot** | `[v, L)` | which `W`-byte slot of a 128 B line the copy lands in |
| **line** | `[L, ∞)` | which 128 B line. This has **no effect on banks**. |

![offset bit zones](/10-cutedsl/swizzle-bit-zones.svg)

Why "slot": a 128 B line covers all 32 banks once. Cut it into `W`-byte slots and each slot is a fixed group of `W/4` banks. There are exactly `128 / W = P` slots, one per thread in a phase. Because of A3, each thread's copy fills exactly one slot.

> [!IMPORTANT]
> **The bank of a whole copy depends only on its slot bits.** This turns "check 32 banks" into "check one small number per thread".

Common cases:

| dtype, copy width | v | L | slot bits | P |
|---|---|---|---|---|
| fp16, 16 B | 3 | 6 | 3–5 | 8 |
| fp32, 16 B | 2 | 5 | 2–4 | 8 |
| int8, 16 B | 4 | 7 | 4–6 | 8 |
| fp32, 4 B | 0 | 5 | 0–4 (slot = bank) | 32 |

Rows start at bit $\log_2 C$. If a row is at least 128 B (`C · E ≥ 128`), the row bits all sit in the line zone. If a row is shorter, the low row bits fall into the slot zone; that's fine, just read off the bits as they are.

---

## 2. Phases

$$
P = \frac{128}{W} \text{ threads per phase}
$$

Phase `p` of warp `w` is threads `32w + pP` … `32w + pP + P − 1`. Conflicts only exist **inside** a phase.

---

## 3. Coordinates of each thread in a phase

From the thread layout (`order` in `make_ordered_layout`):

| Thread order | thread `t` → `(r, c)` |
|---|---|
| row-major, `order=(1,0)` | `(t // Tc, (t % Tc) · V)` |
| column-major, `order=(0,1)` | `(t % Tr, (t // Tr) · V)` |

One pass covers `(Tr, Tc · V)`. If the tile is bigger, the copy repeats in **steps**, each adding `(i · Tr, j · Tc · V)` to every coordinate. Every step is a separate instruction, so check each one.

---

## 4. The conflict test

For each thread in the phase:

$$
\text{slot} = \left\lfloor \frac{\text{off}}{V} \right\rfloor \bmod P \quad\text{(= the slot bits)}
$$

- **Conflict-free** ⟺ the `P` threads land on `P` **different** slots. Threads with the exact same offset count once (broadcast).
- **n-way conflict** ⟺ the busiest slot holds `n` different offsets.

---

## 5. Read the bits: source and destination

If there's a conflict, write the phase's offsets in binary and classify every bit by whether it **varies** across the `P` threads:

| Bit is in… | Varies across the phase? | Name | Role |
|---|---|---|---|
| vector zone | no (always 0) | | never touch |
| slot zone | yes | **used** | already spreading threads apart |
| slot zone | no | **free** | **destination**: flipping it moves threads to unused slots |
| line zone | yes | **source** | separates threads that collide on the same slot |
| line zone | no | | irrelevant |

Why it works:
- Colliding threads have the **same slot bits** but **different line bits**. The source bits are what tell them apart.
- XOR the source bits into the free slot bits. Threads that differed only in the line zone now also differ in the slot zone, so they land on different slots.
- XORing into a **used** bit doesn't help. It only shuffles threads among slots that are already occupied.

To fully remove the conflict, you need at least as many free bits as source bits.

---

## 6. Build `Swizzle<B, M, S>`

Definition: XOR bits `[M + S, M + S + B)` into bits `[M, M + B)`:

```python
def swizzle(off, B, M, S):
    if B == 0:
        return off
    mask = ((1 << B) - 1) << (M + S)
    return off ^ ((off & mask) >> S)
```

Read the parameters off step 5:

| Param | Rule |
|---|---|
| **B** | number of source bits |
| **M** | lowest destination bit |
| **S** | lowest source bit − M |

These constraints must hold, or a single swizzle can't express the fix:

- **C1.** The source bits are **contiguous**, the destination bits are **contiguous**, and there are `B` of each. Swizzle pairs them in order: bit `M + S + i` → bit `M + i`.
- **C2.** `M ≥ v`, so vectors stay whole. Otherwise a 128-bit copy can't be issued.
- **C3.** `S ≥ B`, so the source and destination ranges don't overlap. The swizzle then never changes its own source bits, which makes it reversible (no two elements collide).
- **C4.** `R · C` is a multiple of $2^{M+B}$. The swizzle only changes bits below `M + B`, so every result stays inside the tile.

If C1 fails, change the thread layout, use padding, or accept the conflict.

---

## 7. Verify

1. Redo the conflict test (step 4) with swizzled offsets for **every phase** and **every step**.
2. Do the same for **every other copy** that reads or writes this smem tile. There's only one smem layout, so one swizzle must suit all of them. If no single swizzle does, favor the access that runs most often.
3. Check C2–C4.

A small checker built on the slot test:

```python
def worst_conflict(coord_of, n_threads, steps, C, V, E, B=0, M=0, S=0):
    """coord_of(t) -> (row, col) in the first pass.
    steps = list of (d_row, d_col), one per instruction."""
    v = V.bit_length() - 1          # log2(V), V is a power of 2
    P = 128 // (V * E)              # threads per phase
    worst = 1
    for dr, dc in steps:
        for p0 in range(0, n_threads, P):
            slots = {}
            for t in range(p0, p0 + P):
                r, c = coord_of(t)
                off = swizzle((r + dr) * C + (c + dc), B, M, S)
                slots.setdefault((off >> v) % P, set()).add(off)
            worst = max(worst, max(len(s) for s in slots.values()))
    return worst                    # 1 = conflict-free, n = n-way
```

---

## 8. Special case: a column of vectors (Lei Mao's formula)

A common pattern (`ldmatrix`, MMA tile loads) has extra assumptions:

- **A7.** Every thread in a phase reads the **same column chunk** from a **different row**, and those rows are `P` consecutive rows.
- **A8.** Rows are at least 128 B (`C · E ≥ 128`).

Then steps 4–6 come out the same every time:

- Same chunk → all slot bits are equal → **every slot bit is free** → destination = `[v, L)`, so `M = v`, `B = L − v`.
- `P` consecutive rows differ in the lowest $\log_2 P$ row bits, starting at $\log_2 C$ → `S = log₂C − v`.

$$
M = v, \qquad B = L - v = \log_2 P, \qquad S = \log_2 C - v
$$

That's the formula that appears in Lei Mao's post (with `N = V`, `X = C`):

```
N = (vector_bits / 8) / E
M = log2(N)
B = log2(128 / E) - M
S = log2(X) - M
```

> [!WARNING]
> Use this formula **only** when A7 and A8 hold. If a phase reads several chunks of the same row, some slot bits are already used, and the formula XORs into them for no benefit. That's exactly what happens with the `(32,4)` read in the [worked example](#10-worked-example).

---

## 9. Using it in CuTe DSL

```python
sA_layout = cute.make_composed_layout(
    cute.make_swizzle(B, M, S), 0, cute.make_ordered_layout((R, C), order=(1, 0))
)
sA = smem.allocate_tensor(dtype, sA_layout, byte_alignment=128)
```

> [!WARNING]
> **Units.** Everything on this page is derived in **element offsets**. Passing the composed layout to `allocate_tensor` keeps that. The `swizzle=` keyword applies the swizzle to the **byte address** instead, so the same `<B,M,S>` is a different swizzle there. To use it, convert to bytes: for `E`-byte elements, `M` goes up by `log2(E)` (fp16: `M+1`). A swizzle passed as `swizzle=<1,5,2>` for fp16 is not the one derived here; the byte-unit equivalent is `<1,6,2>`. The two forms were compared in the generated PTX: element `(1, 0)` lands at byte 256 with `swizzle=` and byte 320 with the composed layout.

- `partition_S` / `partition_D` / `cute.copy` don't change. The swizzle lives inside `sA`, and every `(r, c) → address` lookup goes through it.
- So every copy that touches `sA` goes through the same swizzle automatically. Writes and reads always agree on where `(r, c)` lives.

> [!IMPORTANT]
> **The swizzle belongs to the smem layout, not to the copy.**

> [!NOTE]
> The composed-layout form was compiled to PTX and still issues 16 B `cp.async` and 128-bit `ld.shared.v4`. It has not been run on a GPU.

Credit: the column-of-vectors formula and examples follow [Lei Mao's CuTe Swizzle post](https://leimao.github.io/blog/CuTe-Swizzle/).

---

## 10. Worked example

This part runs the method above on the `cp.async` kernel from the copy notes: tile `(32,128)` fp16, write threads `(8,16)`, read threads `(32,4)`. The read side has a 2-way bank conflict, and we derive the swizzle that removes it. Section numbers `10.n` mirror sections `n` above.

---
### 10.0 Inputs, and checking the assumptions

| | Value |
|---|---|
| smem tile `R × C` | `32 × 128`, row-major, `(32,128):(128,1)` |
| element size `E` | 2 B (fp16) |
| **read** copy | threads `(32, 4)` row-major, values `(1, 8)`, 128-bit `CopyUniversalOp` |
| **write** copy | threads `(8, 16)` row-major, values `(1, 8)`, 128-bit `cp.async` |
| `V`, `W` | 8 elements, `W = 8 · 2 = 16` B, for both copies |

- **A1** row-major ✓
- **A2** `C = 128`, `V = 8`, both powers of two ✓
- **A3** `W = 16`; columns start at multiples of 8 ✓
- **A4** allocate `sA` with `byte_alignment=128` ✓
- **A5** phase model assumed
- **A6** swizzle on element offsets (CuTe composed layout) ✓

---

### 10.1 Bit zones

$$
v = \log_2 8 = 3, \qquad L = \log_2 \frac{128}{2} = 6, \qquad \log_2 C = 7
$$

```
bit:    9  8  7 |  6  | 5  4  3 | 2  1  0
        row     |     |  slot   | vector
        ←──────── line ────────→
```

- Bits 0–2: the 8 elements of one copy.
- Bits 3–5: the slot, i.e. which 16 B piece of the 128 B line.
- Bit 6: which half of the row (cols 0–63 or 64–127). It's in the line zone, because a 256 B row spans two 128 B lines.
- Bits 7 and up: the row number. Bit 7 is the row's parity (even/odd).

---

### 10.2 Phases

$$
P = \frac{128}{16} = 8
$$

Phase 0 is threads 0–7, phase 1 is threads 8–15, and so on.

---

### 10.3 Coordinates (read plan)

Row-major `(32, 4)`: `t → (t // 4, (t % 4) · 8)`.

- One pass covers `(32, 4 · 8) = (32, 32)`.
- The tile is `(32, 128)`, so there are 4 steps, adding `(0, 32j)` for `j = 0…3`.

Phase 0, step 0:

| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `(r, c)` | (0,0) | (0,8) | (0,16) | (0,24) | (1,0) | (1,8) | (1,16) | (1,24) |
| off = `128r + c` | 0 | 8 | 16 | 24 | 128 | 136 | 144 | 152 |

---

### 10.4 Conflict test

`slot = (off // 8) % 8`:

| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| slot | 0 | 1 | 2 | 3 | **0** | **1** | **2** | **3** |

Slots 0–3 each hold 2 different offsets, and slots 4–7 are empty. That's a **2-way conflict**.

---

### 10.5 Source and destination bits

![phase 0 before and after](/10-cutedsl/swizzle-152-phase0.svg)

Going bit by bit across the 8 offsets:

| Bit | Zone | Varies? | Role |
|---|---|---|---|
| 0–2 | vector | no | never touch |
| 3, 4 | slot | yes | used |
| **5** | slot | no (always 0) | **free → destination** |
| 6 | line | no | irrelevant |
| **7** | line | yes (row 0 vs row 1) | **source** |
| 8+ | line | no | irrelevant |

---

### 10.6 Build the swizzle

| Param | Rule | Value |
|---|---|---|
| B | number of source bits | **1** |
| M | lowest destination bit | **5** |
| S | source − M = 7 − 5 | **2** |

So the swizzle is **`Swizzle<1, 5, 2>`**: if the row is odd, flip bit 5.

Constraints:
- **C1** one source bit, one destination bit, trivially contiguous ✓
- **C2** `M = 5 ≥ v = 3`, vectors stay whole ✓
- **C3** `S = 2 ≥ B = 1` ✓
- **C4** `32 · 128 = 4096` is a multiple of $2^{5+1} = 64$ ✓

In memory terms: on **odd rows**, the two 64 B halves of each 128 B line swap places.

![padding vs swizzle](/10-cutedsl/bank-fixes-pad-vs-swizzle.svg)

---

### 10.7 Verify

**Read, phase 0 after the swizzle:** t4–t7 have bit 5 flipped, so their offsets become 160, 168, 176, 184 and their slots become 4, 5, 6, 7. All 8 slots are distinct, so it's conflict-free ✓

**Read, every other phase:** phase `p` is rows `2p` and `2p + 1`. These again differ only in bit 7, so the same argument holds ✓

**Read, every step:** step `j` adds `32j` to the column, which sets bits 5–6 to `j`. All threads in the phase share that value, and bit 7 still flips bit 5 for the odd row, so the two rows still land in opposite halves ✓

**Write (`cp.async`, `(8,16)`):** phase 0 is threads 0–7 = row 0, cols 0–56, so slots 0–7 are distinct and the row is one fixed parity. The swizzle flips bit 5 either for none of them or for all of them, which only swaps the two halves. The slots stay distinct ✓

Checked with the checker from the swizzle page:

```python
read  = lambda t: (t // 4,  (t % 4)  * 8)
write = lambda t: (t // 16, (t % 16) * 8)
read_steps  = [(0, 32 * j) for j in range(4)]
write_steps = [(8 * i, 0)  for i in range(4)]
args = dict(n_threads=128, C=128, V=8, E=2)

worst_conflict(read,  steps=read_steps,  **args)                 # 2
worst_conflict(read,  steps=read_steps,  **args, B=1, M=5, S=2)  # 1
worst_conflict(write, steps=write_steps, **args)                 # 1
worst_conflict(write, steps=write_steps, **args, B=1, M=5, S=2)  # 1
```

---

### 10.8 Why Lei Mao's formula gives the wrong answer here

The formula gives `M = 3, B = 3, S = 4`, i.e. `Swizzle<3,3,4>`. It assumes the column-of-vectors pattern (A7): every slot bit is free. Here, bits 3 and 4 are **used**, since four threads read four different chunks of the same row. The formula XORs the row into them anyway, which only shuffles rows 0 and 1 among slots 0–3. It stays 2-way.

| Read pattern | No swizzle | `<1,5,2>` | `<3,3,4>` |
|---|---|---|---|
| `(32,4)` row-major (this page) | 2-way | **1-way** | 2-way |
| `(8,16)` column-major (column of vectors) | 8-way | 4-way | **1-way** |

A swizzle is fitted to an access pattern; no single one fixes every pattern.
