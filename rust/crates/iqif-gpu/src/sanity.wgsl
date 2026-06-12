// Phase-2 integer-semantics sanity kernel (see rust/PLAN.md).
//
// The IQIF hot path is pure 32-bit integer math. Bit-exact parity with the
// C++/Rust CPU reference hinges on three WGSL `i32` operations behaving the way
// they do on the CPU, *especially for negative operands*:
//
//   * `>>` is an arithmetic (sign-extending) right shift,
//   * `/`  truncates toward zero,
//   * `%`  takes the sign of the dividend (so `a == (a/b)*b + a%b`).
//
// These are guaranteed by the WGSL spec, but the point of this kernel is to
// confirm the *actual driver* on the *actual device* honors them before we
// trust 46k step-by-step parity checks to the GPU. The host fills `pairs` with
// sign-mixed operands, this kernel applies the four ops, and the host compares
// every lane against Rust's own i32 result.

struct Pair {
    a: i32,
    b: i32,
    shift: u32,  // kept in 0..=31; WGSL leaves out-of-range shifts undefined
    _pad: u32,
};

// Mirrors the host `[i32; 8]` result stride.
struct Out {
    shr: i32,
    div: i32,
    rem: i32,          // hardware `%` — buggy on NVIDIA+Vulkan for negatives
    mul: i32,
    rem_via_div: i32,  // portable workaround: a - (a/b)*b
    _p0: i32,
    _p1: i32,
    _p2: i32,
};

@group(0) @binding(0) var<storage, read>       pairs: array<Pair>;
@group(0) @binding(1) var<storage, read_write> results: array<Out>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&pairs)) {
        return;
    }
    let p = pairs[i];
    // Callers exclude the only WGSL-indeterminate integer cases (b == 0, and
    // INT_MIN / -1) up front, so every lane below is well-defined per spec.
    var o: Out;
    o.shr = p.a >> p.shift;
    o.div = p.a / p.b;
    o.rem = p.a % p.b;
    o.mul = p.a * p.b;
    o.rem_via_div = p.a - (p.a / p.b) * p.b;
    o._p0 = 0;
    o._p1 = 0;
    o._p2 = 0;
    results[i] = o;
}
