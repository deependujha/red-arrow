### Fixing the read with a swizzle

The `(32,4)` read above has a 2-way bank conflict. A swizzle removes it without changing the copy code: only the way `sA` is allocated changes.

For this tile and this read, the swizzle is **`Swizzle<1,5,2>`**: on odd rows, flip bit 5 of the element offset, which swaps the two 64 B halves of each 128 B line. For how it's derived, see the [worked example](../swizzle-example/). For the general method, see [Swizzle](../swizzle/).

```python
# before: plain row-major tile
sA = smem.allocate_tensor(
    mA.element_type, cute.make_ordered_layout(TILE, order=(1, 0)), byte_alignment=16
)

# after: the same tile, composed with Swizzle<1,5,2> (element offsets -> swizzle goes IN the layout)
sA_layout = cute.make_composed_layout(
    cute.make_swizzle(1, 5, 2), 0, cute.make_ordered_layout(TILE, order=(1, 0))
)
sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=128)
```

Everything else in the kernel stays as it is: the load plan, the read plan, `partition_S` / `partition_D`, and `cute.copy`.

| Copy | No swizzle | `Swizzle<1,5,2>` |
|---|---|---|
| write: `cp.async`, threads `(8,16)` | 1-way | 1-way |
| read: threads `(32,4)` | **2-way** | **1-way** |

> [!IMPORTANT]
> - **The swizzle belongs to `sA`, not to the copy.** Both copies partition the same `sA`, so writes and reads always agree on where `(r, c)` lives.
> - One smem layout means **one swizzle for every copy that touches it**. Always re-check the other copies.
> - `torch.equal(A, B)` stays `True`: the swizzle changes where data is stored, not the values.
> - The `+128` slack in `launch` now matters: with `byte_alignment=128`, the allocator may skip up to 127 bytes.

> [!WARNING]
> Don't pass the swizzle as `allocate_tensor(..., swizzle=...)`: that form applies it to **byte** addresses, which for fp16 would need `Swizzle<1,6,2>`. See [Swizzle, section 9](../swizzle/). Checked in PTX: this version still issues 16 B `cp.async` and 128-bit `ld.shared.v4`.
