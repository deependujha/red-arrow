
### 3.4 Axis 3 — proxy: which path to memory

This axis has no C++ analogue; it exists because a modern SM has **several independent hardware engines that can touch the same byte of memory, each through its own path with its own buffering**. PTX calls each path a **proxy**:

```text
                      ┌────────────────────┐
                      │   shared memory    │
                      └───▲───────────▲────┘
              generic     │           │     async proxy
              proxy       │           │
        (LSU: ld/st/atom) │           │  (TMA engine: cp.async.bulk,
                          │           │   wgmma/tcgen05 reading operands)
                     warp lanes    copy / tensor engines
```

| Proxy | Who uses it |
|---|---|
| **generic** | ordinary `ld` / `st` / `atom` / `red` — everything a lane issues through the LSU |
| **async** | `cp.async.bulk*` (TMA), `cp.reduce.async.bulk`, and the tensor pipes' own reads of shared memory (`wgmma`, `tcgen05`) |
| **tensormap** | `tensormap.replace` writing a TMA descriptor, vs. the TMA engine later reading it |
| *alias* | the special case of two different virtual addresses mapping to the same physical location |

The crucial fact: **everything in §3.2–3.3 — barriers included — orders operations *within one proxy* only.** A `bar.sync` makes all *generic*-proxy writes of the block visible to all *generic*-proxy reads of the block. It says nothing about when those writes become visible to the async engine's view of the same address. Crossing paths requires an explicit **cross-proxy fence**:

```ptx
fence.proxy.async;                    // generic ↔ async, all state spaces
fence.proxy.async.shared::cta;        // narrowed: only shared memory (cheaper, the common one)
fence.proxy.async.global;             // narrowed to global
```

The canonical Hopper bug, spelled out:

```ptx
st.shared.f32   [%r_buf], %f1;        // lanes fill a tile   (generic proxy)
bar.sync 0;                           // orders generic proxy only!
// ── MISSING: fence.proxy.async.shared::cta; ──
wgmma.mma_async...  [%r_buf] ...      // tensor pipe reads it (async proxy)  → RACE
```

The correct sequence is `st.shared` → `bar.sync` (so all lanes are done) → `fence.proxy.async.shared::cta` (so the async proxy sees it) → `wgmma`. Omit the fence and you get the worst failure mode in the trade: a race that only reproduces at full speed, and vanishes the moment you add a `printf`.

The tensormap proxy works the same way with its own fence pair — you'll see this pattern in §7.3 when a kernel patches a TMA descriptor on-device:

```ptx
tensormap.replace.tile.global_address.global.b1024.b64 [%rd_tmap], %rd_new;
fence.proxy.tensormap::generic.release.gpu;               // publish my generic-proxy write
...
fence.proxy.tensormap::generic.acquire.gpu [%rd_t], 128;  // acquire before the TMA reads it
cp.async.bulk.tensor.2d... [%r_smem], [%rd_t, {...}], [bar];
```

> **Mental model:** scope asks "*which threads* can see it?"; proxy asks "*through which pipe* are they looking?" Two observers can be in scope and still disagree, if they're looking through different pipes and you never fenced between them.

### 3.5 Reading a fully-qualified instruction

With the three axes in hand, any qualifier soup decodes mechanically:

| Instruction | Strength | Scope | Proxy | Plain English |
|---|---|---|---|---|
| `ld.global.f32 %f1,[%rd]` | `.weak` | — | generic | plain load, no promises |
| `ld.volatile.shared.u32` | ≈`.relaxed` | `.sys` | generic | really perform it; no ordering of neighbors |
| `atom.relaxed.cta.shared.add.u32` | `.relaxed` | `.cta` | generic | indivisible add my block agrees on; orders nothing else |
| `st.release.gpu.global.u32` | `.release` | `.gpu` | generic | publish everything above to the whole device |
| `ld.acquire.cluster.shared::cluster.u32` | `.acquire` | `.cluster` | generic | receive a peer CTA's publication via DSMEM |
| `fence.sc.sys` | `.sc` | `.sys` | generic | total order the CPU also agrees on |
| `fence.proxy.async.shared::cta` | — | (cta) | generic↔async | let the TMA/tensor pipes see my smem stores |

### 3.6 Fences: ordering without an access

A **fence** is Axis 1 + Axis 2 detached from any particular load or store: it orders the operations *around* it instead of riding on one.

```ptx
fence.acq_rel.cta;        // ops before it can't sink below; ops after can't hoist above (block)
fence.acq_rel.gpu;        //   same, device-wide
fence.sc.gpu;             // + joins the single total order → this is __threadfence()
fence.sc.cta;             //   __threadfence_block()
fence.sc.sys;             //   __threadfence_system()
membar.cta; membar.gl; membar.sys;   // legacy spellings ≈ fence.sc.{cta,gpu,sys}
```

Two things worth internalizing:

**When a fence instead of qualified accesses?** When the ordering point doesn't coincide with one access — e.g. you wrote a whole struct with plain stores and want *one* publication point before the flag; or a library boundary forces plain accesses. `fence.release` + plain flag store + plain flag load + `fence.acquire` is equivalent to the qualified pair in §3.2 (this is exactly C++'s fence formulation).

**Why `.sc` exists — the store→load hole.** Acquire/release never forbids a *store* to one address being reordered with a *later load* of a different address. Dekker-style mutual exclusion needs exactly that ordering:

```text
   thread A                thread B
   st  a_wants = 1         st  b_wants = 1
   ld  b_wants  → 0?       ld  a_wants  → 0?
```

With only release stores and acquire loads, **both** loads may execute before either store is visible — both threads read 0, both enter the critical section. Placing `fence.sc.gpu` between each thread's store and load closes the hole: the two fences are ordered against each other in the single total order, so at least one thread must observe the other's store. If you ever wonder why `__threadfence()` compiles to `fence.sc` rather than `fence.acq_rel` — this is why; it's the strongest and safest default.

**Special-purpose fences** you'll meet later in the chapter:

```ptx
fence.mbarrier_init.release.cluster;   // publish mbarrier.init to the cluster (§7.2 setup)
fence.proxy.async.shared::cta;         // the cross-proxy fence from §3.4
fence.proxy.tensormap::generic...;     // descriptor-patching pair from §3.4 / §7.3
```

### 3.7 Section gotchas

| Gotcha | Reality |
|---|---|
| "`volatile` synchronizes" | It only pins the compiler. It is `.relaxed`-like: no acquire/release edge, ever. |
| "`relaxed` is ordered" | Only that one address has an agreed modification order. Neighbors move freely. |
| "acquire/release is enough for mutual exclusion" | It's enough for *message passing* (one-way). Store→load ordering (Dekker) needs `fence.sc`. |
| "scope is a performance hint" | Scope is **correctness**: an observer outside the scope gets *no* guarantee at all. |
| "`bar.sync` flushed my smem for TMA/wgmma" | It ordered the generic proxy only. Cross-proxy needs `fence.proxy.async.shared::cta`. |
| "`.cg`/`.cv` gave me fresh data, so I'm synchronized" | Cache ops are hints on a separate layer; they can never substitute for Axis-1/2 qualifiers. |
| "the plain data next to my release must also be atomic" | No — plain `.weak` data rides the release→acquire edge for free. That's the whole point. |

### Interview Questions & Answers (section 3 additions)

#### Q: What's the difference between `.volatile` and `.relaxed`, and when does picking the wrong one bite you?

**Answer:** `.volatile` is a *compiler* directive: the access is really performed, not cached in a register, not reordered at compile time; for hardware ordering PTX treats it like `.relaxed` at `.sys` scope. `.relaxed` is a *memory-model* property: the access is atomic at a chosen scope with an agreed per-address modification order, and the scope can be narrowed (`.cta`) for performance. It bites in two directions: a spin loop on a plain variable needs at least `volatile`-like behavior or the compiler hoists the load and the loop never exits; but a flag protecting data needs `.acquire`/`.release`, and `volatile` silently fails to provide the edge — the loop exits and the data is stale. `volatile` fixes the *compile-time* half of a race and leaves the *hardware* half, which is why volatile-based "locks" pass light tests and fail under load.

#### Q: Walk through message passing between two blocks. Which qualifiers, which scope, and why is the payload store allowed to stay `.weak`?

**Answer:** Producer: plain stores for the payload, then `st.release.gpu` on the flag. Consumer: spin on `ld.acquire.gpu` of the flag, then plain loads of the payload. Scope must be `.gpu` because the observer is in another block — `.cta` would create no synchronizes-with edge across SMs (L1s aren't coherent; the guarantee must be anchored at L2). The payload stays `.weak` because the release/acquire pair on the flag transfers visibility of *everything* sequenced before the release to *everything* sequenced after the acquire, provided the acquire reads the released value. Making the payload atomic too would add cost and zero correctness.

#### Q: C++ has memory orders but no scopes. Why does PTX need scopes?

**Answer:** CPUs present a single coherence domain — every core snoops the same coherent cache fabric, so "visible" means visible to everyone, and the ISA can afford one implicit system scope. A GPU is a hierarchy of *non-coherent* islands: per-SM L1s that never snoop each other, cluster fabrics, then a device-wide L2, then the system link. Making a write visible one level further up costs a physically longer round trip, and most synchronization on a GPU is local (a block cooperating through shared memory). Scopes let the program state the actual observer set so the hardware can satisfy a `.cta` acquire inside the SM instead of paying an L2 or system round trip. It's the memory-model expression of the same locality principle that gives you shared memory in the first place — and it's why `atomicAdd` (default `.gpu`) vs `atomicAdd_block` (`.cta`) can differ by an order of magnitude under contention.

#### Q: You added `__syncthreads()` before a `wgmma` that reads shared memory, and the kernel still races. What's missing and why?

**Answer:** A cross-proxy fence. `__syncthreads()`/`bar.sync` orders the *generic* proxy — the LSU path lanes use for `st.shared`. `wgmma` reads its operands through the *async* proxy, a different hardware path with its own view of shared memory; nothing in the barrier's contract makes generic-proxy writes visible to it. The sequence must be: lanes store the tile, `bar.sync` (all lanes done), `fence.proxy.async.shared::cta` (generic→async visibility edge), then `wgmma`. Same rule for TMA stores reading smem written by `st.shared`, and the tensormap analogue (`fence.proxy.tensormap::generic.*`) when a kernel patches a TMA descriptor it's about to use.

#### Q: Why does `__threadfence()` lower to `fence.sc.gpu` and not `fence.acq_rel.gpu`?

**Answer:** Because acquire/release leaves the store→load hole: a store to X followed by a load of Y may still be reordered, which breaks flag-based mutual exclusion (Dekker) — both threads can store their flag, both load the other's as 0, both enter. `.sc` fences additionally join a single total order all threads in scope agree on, which forces at least one thread to observe the other's store. CUDA's `__threadfence` predates the fine-grained model and promises the strong behavior, so it must lower to the `.sc` form; code that only needs one-way publication can use `cuda::atomic_thread_fence(memory_order_release/acquire, ...)` and get the cheaper `fence.acq_rel`/qualified accesses instead.