//! Generates the Metal Shading Language source for the SHA-256d search kernel.
//!
//! The kernel is emitted from Rust rather than kept as a static `.metal` file
//! because it is fully unrolled (125 rounds with the working variables renamed
//! every round), which is far less error-prone to generate than to hand-write.
//!
//! # What the kernel computes
//!
//! A Bitcoin block header is 80 bytes, hashed as `SHA256(SHA256(header))`.
//! SHA-256 consumes 64-byte blocks, so the first hash is two compressions:
//!
//! ```text
//! block 1: header[0..64]                       -> "midstate" (nonce independent)
//! block 2: header[64..80] + padding            -> first hash
//!          w0 = merkle tail, w1 = ntime, w2 = nbits, w3 = NONCE,
//!          w4 = 0x80000000, w5..w14 = 0, w15 = 640
//! block 3: first hash (8 words) + padding      -> final hash
//! ```
//!
//! # Optimisations
//!
//! * **Midstate** - block 1 is compressed once per work unit on the CPU.
//! * **Round skipping** - the nonce is message word 3, so rounds 0-2 of block 2
//!   are nonce independent and also run on the CPU. Round 3 is reduced to two
//!   additions because everything except `+ nonce` is folded into `pre[3]` and
//!   `pre[7]`.
//! * **Schedule folding** - most of block 2 and half of block 3 are padding
//!   constants. Schedule words that do not depend on the nonce (`w16`, `w17`)
//!   and the constant parts of others are precomputed; zero terms are dropped.
//! * **Early exit** - a valid share needs the last state word of the final hash
//!   to be zero. That word is `IV[7] + e` where `e` is produced by round 60, so
//!   rounds 61-63 and the final feed-forward additions are never computed.
//!
//! Each GPU thread tests a single nonce: on Apple GPUs that measured ~15%
//! faster than looping over several nonces per thread. The `CH`/`MAJ`
//! formulations below were likewise picked by benchmark (M1 Max).
//!
//! Candidates that pass the 32-bit zero test (one in ~4.29 billion hashes) are
//! appended to a small result buffer and fully verified on the CPU.

use crate::sha256::{IV, K};
use std::fmt::Write;

/// Maximum number of candidate nonces a single dispatch can report.
pub const MAX_RESULTS: usize = 63;

/// log2 of the dispatch grid width, in threads.
pub const GRID_SHIFT: u32 = 14;

/// Working-variable names for round `r`, in role order `a..h`.
///
/// Instead of shuffling eight variables every round, the roles rotate.
fn vars(r: usize) -> [char; 8] {
    const NAMES: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    std::array::from_fn(|i| NAMES[(i + 8 - r % 8) % 8])
}

fn round(out: &mut String, r: usize, kw: &str) {
    let [a, b, c, d, e, f, g, h] = vars(r);
    writeln!(
        out,
        "    RND({a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}, {kw});"
    )
    .unwrap();
}

/// `K[r] + constant` folded into a single literal.
fn k_plus(r: usize, constant: u32) -> String {
    format!("0x{:08x}u", K[r].wrapping_add(constant))
}

/// Emits `p{i} = s1(p{i-2}) + p{i-7} + s0(p{i-15}) + p{i-16}`.
fn schedule(out: &mut String, prefix: char, i: usize) {
    writeln!(
        out,
        "    uint {p}{i} = s1({p}{}) + {p}{} + s0({p}{}) + {p}{};",
        i - 2,
        i - 7,
        i - 15,
        i - 16,
        p = prefix
    )
    .unwrap();
}

/// Builds the kernel source. Each GPU thread tests exactly one nonce.
#[allow(clippy::needless_range_loop)] // rounds are indexed by number throughout
pub fn source() -> String {
    let mut s = String::with_capacity(32 * 1024);
    write!(
        s,
        r#"#include <metal_stdlib>
using namespace metal;

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define S0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define s1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
#define CH(e, f, g) (((e) & (f)) ^ (~(e) & (g)))
#define MAJ(a, b, c) ((b) ^ (((a) ^ (b)) & ((b) ^ (c))))
#define RND(a, b, c, d, e, f, g, h, kw) {{ \
    uint t1 = h + S1(e) + CH(e, f, g) + (kw); \
    d += t1; \
    h = t1 + S0(a) + MAJ(a, b, c); }}

struct Params {{
    uint mid[8];   // state after header[0..64]
    uint pre[8];   // state after rounds 0-2 of block 2, round 3 partially applied
    uint w16;
    uint w17;
    uint c18;      // w18 = c18 + s0(nonce)
    uint c19;      // w19 = c19 + nonce
    uint c31;      // constant part of w31
    uint c32;      // constant part of w32
    uint base;     // first nonce word of this dispatch
    uint pad;
}};

struct Results {{
    atomic_uint count;
    uint nonces[{max_results}];
}};

kernel void search(constant Params& p [[buffer(0)]],
                   device Results& results [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]])
{{
    const uint m0 = p.mid[0], m1 = p.mid[1], m2 = p.mid[2], m3 = p.mid[3];
    const uint m4 = p.mid[4], m5 = p.mid[5], m6 = p.mid[6], m7 = p.mid[7];
    const uint p0 = p.pre[0], p1 = p.pre[1], p2 = p.pre[2], p3 = p.pre[3];
    const uint p4 = p.pre[4], p5 = p.pre[5], p6 = p.pre[6], p7 = p.pre[7];
    const uint w16 = p.w16, w17 = p.w17, c18 = p.c18, c19 = p.c19;
    const uint c31 = p.c31, c32 = p.c32;
    const uint n = p.base + ((gid.y << {grid_shift}) | gid.x);

    // ---- Hash 1, block 2: resume after round 3 ------------------------------
"#,
        max_results = MAX_RESULTS,
        grid_shift = GRID_SHIFT,
    )
    .unwrap();

    // The CPU hands over the canonical state after round 3 (minus the nonce).
    // Map canonical positions onto the variable names round 3 would have used.
    let v = vars(3);
    for (i, name) in v.iter().enumerate() {
        let nonce = if i == 3 || i == 7 { " + n" } else { "" };
        writeln!(s, "    uint {name} = p{i}{nonce};").unwrap();
    }

    // Rounds 4..15 consume padding only.
    round(&mut s, 4, &k_plus(4, 0x8000_0000));
    for r in 5..15 {
        round(&mut s, r, &k_plus(r, 0));
    }
    round(&mut s, 15, &k_plus(15, 640));

    // Message schedule with zero terms dropped (w5..w14 = 0, w4 and w15 const).
    s.push_str(
        "    uint w18 = c18 + s0(n);\n\
         \x20   uint w19 = c19 + n;\n\
         \x20   uint w20 = s1(w18) + 0x80000000u;\n\
         \x20   uint w21 = s1(w19);\n\
         \x20   uint w22 = s1(w20) + 640u;\n\
         \x20   uint w23 = s1(w21) + w16;\n\
         \x20   uint w24 = s1(w22) + w17;\n\
         \x20   uint w25 = s1(w23) + w18;\n\
         \x20   uint w26 = s1(w24) + w19;\n\
         \x20   uint w27 = s1(w25) + w20;\n\
         \x20   uint w28 = s1(w26) + w21;\n\
         \x20   uint w29 = s1(w27) + w22;\n\
         \x20   uint w30 = s1(w28) + w23 + s0(640u);\n\
         \x20   uint w31 = s1(w29) + w24 + c31;\n\
         \x20   uint w32 = s1(w30) + w25 + c32;\n",
    );
    for i in 33..64 {
        schedule(&mut s, 'w', i);
    }
    for r in 16..64 {
        round(&mut s, r, &format!("0x{:08x}u + w{r}", K[r]));
    }

    // Feed-forward: the first hash becomes message words x0..x7 of block 3.
    s.push_str(
        "\n    // ---- Hash 2 -----------------------------------------------------------\n",
    );
    for (i, name) in vars(0).iter().enumerate() {
        writeln!(s, "    const uint x{i} = m{i} + {name};").unwrap();
    }
    for (i, name) in vars(0).iter().enumerate() {
        writeln!(s, "    {name} = 0x{:08x}u;", IV[i]).unwrap();
    }
    for r in 0..8 {
        round(&mut s, r, &format!("0x{:08x}u + x{r}", K[r]));
    }
    round(&mut s, 8, &k_plus(8, 0x8000_0000));
    for r in 9..15 {
        round(&mut s, r, &k_plus(r, 0));
    }
    round(&mut s, 15, &k_plus(15, 256));

    // x8 = 0x80000000, x9..x14 = 0, x15 = 256.
    s.push_str(
        "    uint x16 = s0(x1) + x0;\n\
         \x20   uint x17 = s1(256u) + s0(x2) + x1;\n\
         \x20   uint x18 = s1(x16) + s0(x3) + x2;\n\
         \x20   uint x19 = s1(x17) + s0(x4) + x3;\n\
         \x20   uint x20 = s1(x18) + s0(x5) + x4;\n\
         \x20   uint x21 = s1(x19) + s0(x6) + x5;\n\
         \x20   uint x22 = s1(x20) + 256u + s0(x7) + x6;\n\
         \x20   uint x23 = s1(x21) + x16 + s0(0x80000000u) + x7;\n\
         \x20   uint x24 = s1(x22) + x17 + 0x80000000u;\n\
         \x20   uint x25 = s1(x23) + x18;\n\
         \x20   uint x26 = s1(x24) + x19;\n\
         \x20   uint x27 = s1(x25) + x20;\n\
         \x20   uint x28 = s1(x26) + x21;\n\
         \x20   uint x29 = s1(x27) + x22;\n\
         \x20   uint x30 = s1(x28) + x23 + s0(256u);\n\
         \x20   uint x31 = s1(x29) + x24 + s0(x16) + 256u;\n",
    );
    for i in 32..61 {
        schedule(&mut s, 'x', i);
    }
    for r in 16..60 {
        round(&mut s, r, &format!("0x{:08x}u + x{r}", K[r]));
    }

    // Round 60 produces the `e` that ends up as the last word of the hash.
    // Only that word is needed: hash[7] == 0  <=>  e == -IV[7].
    let [_, _, _, d, e, f, g, h] = vars(60);
    write!(
        s,
        r#"    const uint last = {d} + {h} + S1({e}) + CH({e}, {f}, {g}) + 0x{k60:08x}u + x60;
    if (last == 0x{target:08x}u) {{
        const uint slot = atomic_fetch_add_explicit(&results.count, 1u, memory_order_relaxed);
        if (slot < {max_results}u) {{
            results.nonces[slot] = n;
        }}
    }}
}}
"#,
        k60 = K[60],
        target = 0u32.wrapping_sub(IV[7]),
        max_results = MAX_RESULTS,
    )
    .unwrap();
    s
}
