# RIIR + GPU plan: integer IQIF in Rust with a PyTorch-style backend switch

## Goal

Rewrite the IQIF spiking network in Rust (done for CPU), then add a **wgpu GPU
backend that preserves the exact integer dynamics**, exposed to Python so you
can switch backends at runtime the way PyTorch switches device:

```python
from iqif_rs import iqnet
net = iqnet("params.txt", "conn.txt", device="cpu")   # bit-exact CPU reference
net = iqnet("params.txt", "conn.txt", device="gpu")    # same math, on the GPU
```

`device` is internal state of a single `iqnet` type — mirroring how a PyTorch
`Tensor` is one type with a `.device` field, not two separate classes.

## Why integer-on-GPU is bit-exact-able

The per-timestep IQIF update is pure 32-bit integer math, and WGSL's `i32`
matches C++/Rust on the three operations that matter:

- `>>` on `i32` is an **arithmetic** (sign-extending) shift,
- `/` and `%` **truncate toward zero**,
- `+`/`-`/`*` wrap modulo 2^32 (and these sims stay within range).

The only floating-point in IQIF (`log2`/`log10` in `recalculate_params`) runs
**once at setup on the CPU**; only the resulting integers are uploaded. So the
GPU hot path never touches floats. **Validation step 0** (before trusting any
of this): a WGSL kernel that checks `>>`, `/`, `%` on *negative* operands
against the CPU on the actual device — implemented in `iqif-gpu/src/sanity.rs`
(+`sanity.wgsl`), run via the `relied_on_ops_are_bit_exact_on_every_backend`
test.

### Phase-2 finding: hardware signed `%` is NOT portable (use `a-(a/b)*b`)

Running the sanity kernel on the dev NVIDIA RTX 3060 surfaced a real driver
divergence from the WGSL spec, **not** confined to edge cases:

| backend | `>>` `/` `*` | signed `%` |
|---------|:---:|:---:|
| **DX12**   | exact | **exact** |
| **Vulkan** | exact | **WRONG for negative dividend** (85/155 pairs) |
| **GL**     | exact | **WRONG for negative dividend** (85/155 pairs) |

NVIDIA's Vulkan/GL paths compute `a % b` with the *divisor's* sign (Euclidean-ish)
instead of truncating toward zero — e.g. `-1000000007 % 3` gives `2`, not `-2`;
with both operands negative it returns the dividend untouched. Ordinary ~1e9
operands trigger it, so "the sims stay in range" does **not** save us.

This is the NVIDIA driver's shader codegen, **not OS-specific**: the same RTX
3060 reproduces the broken Vulkan `%` under Linux as well as Windows. So it is
not something a platform switch escapes — only the `a-(a/b)*b` form does.

**Rule for all GPU kernels (Phase 3+): never emit raw signed `%`.** Compute
remainder as `rem = a - (a/b)*b`. `/` is correct on every backend, so this is
bit-exact everywhere — validated on-device (the `rem_via_div` lane passes on
Vulkan/GL/DX12). The allow-list lives in `iqif_gpu::RELIED_ON_OPS`. (For the
current network this is belt-and-suspenders: the only hot-path `%` is
`rng.next() % noise`, both operands non-negative — but the rule keeps us correct
if that ever changes, and keeps us off DX12-only.)

**Why not just pin DX12?** It is Windows-only — the RADV/Linux box has no DX12,
only Vulkan, where raw `%` is exactly the broken path. Portability (the whole
"works on every device" premise) requires the `a-(a/b)*b` form, not a backend
bet. DX12 may still be *preferred* on Windows for perf, but correctness must not
depend on it. The workaround is ~free: GPUs synthesize both `/` and `%` from the
same division anyway, and integer div/rem doesn't appear in the per-step hot
path (only `>>`, `+`, compare do).

**Still TODO on the RADV box:** re-run this same test there to confirm RADV's
Vulkan `%` behavior and that the relied-on ops are exact.

## Architecture: Cargo workspace under `rust/`

```
rust/
  Cargo.toml                 # virtual workspace manifest
  pyproject.toml             # maturin -> builds the iqif-py crate as module `iqif_rs`
  crates/
    iqif-core/               # pure Rust, no deps: integer dynamics + setup
      src/lib.rs             #   (single source of truth for f_min/decay/CSR params)
    iqif-py/                 # PyO3 bindings; `iqnet` with Backend dispatch + device=
      src/lib.rs             #   optional dep on iqif-gpu behind `gpu` feature
    iqif-gpu/                # wgpu backend (skeleton now; kernels later)
      src/lib.rs
```

- **Workspace = source organization.** Keeps wgpu's large dependency tree out
  of the lean CPU Python wheel.
- **Backend switching = a property of the built wheel.** For `device="gpu"` to
  work at runtime from one `import`, the GPU crate must be compiled *into* the
  wheel — so `iqif-py` takes an **optional** dependency on `iqif-gpu` behind a
  `gpu` feature. No feature -> tiny CPU-only wheel that errors cleanly on
  `device="gpu"`. With the feature -> one module, runtime switch, PyTorch-style.

### Backend dispatch

`iqnet` holds `Box<dyn Backend + Send>`. `Backend` is an object-safe trait
implemented by both `iqif_core::IqNetwork` (CPU) and `iqif_gpu::GpuNetwork`
(GPU). Both construct from the **same `iqif_core` setup**, so parity is
structural, not coincidental.

## The one real GPU design problem: spike propagation

`send_synapse` is a *scatter* (each firing neuron `+=` weights into its targets'
accumulators) — GPUs race on that. Reframe as a **gather over the transposed
adjacency (CSC)**: each post-neuron pulls and sums weights of incoming edges
whose source fired last step. No atomics, deterministic, and bit-exact (integer
add is commutative). The timestep becomes **two compute dispatches** —
propagate, then update_state — the dispatch boundary giving the phase sync for
free. `is_firing` double-buffers across the two.

(Alternative: keep CSR + `atomicAdd` on `atomic<i32>`. Also exact, simpler port,
slightly slower. Gather is the plan.)

## Performance caveat (the PyTorch lesson)

GPU only wins at scale (fly-connectome size, à la neurilium); for the small test
configs the CPU port is faster. And like PyTorch, **state must stay resident on
the GPU across `send_synapse()` calls**, reading back only in **bulk**
(`get_all_*`) on demand. Per-neuron scalar getters (`potential(i)`,
`get_is_firing(i)`) in the GPU hot path mean a PCIe round-trip each call and will
dominate runtime. `test_rust_parity.py` deliberately does exactly that — correct
for *correctness* checking, wrong for *benchmarking*.

## Validation

Reuse `tests/test_rust_parity.py` (already compares Rust CPU vs the C++ `iqif`
reference, 46367 step-by-step integer checks). Extend it to also run
`device="gpu"` and assert identical output against the same C++ reference.

## Roadmap / status

- [x] **Phase 0** — Bit-exact CPU port (`iqif_core`) + PyO3 module `iqif_rs`,
      parity-proven vs C++.
- [x] **Phase 1** — Workspace refactor + PyTorch-style `device=` plumbing.
      CPU arm fully working; GPU arm behind the `gpu` feature as a stub.
- [x] **Phase 2** — WGSL integer sanity kernel (`>>`, `/`, `%` on negatives),
      run on-device per backend. **Found NVIDIA Vulkan/GL break signed `%`;**
      adopted `rem = a-(a/b)*b` as the portable rule (see finding above). RADV
      re-run still pending.
- [ ] **Phase 3** — Build CSC (transposed adjacency) in `iqif_core`; upload
      buffers; `propagate` + `update_state` WGSL kernels; state stays on-device.
- [ ] **Phase 4** — Bulk readback API; wire GPU arm into `Backend`; extend the
      parity test to `device="gpu"`.
- [ ] **Phase 5** — Benchmark at connectome scale; tune workgroup sizing.
