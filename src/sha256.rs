//! Minimal SHA-256.
//!
//! The miner needs access to the raw compression function (for midstates and
//! for pre-computing the nonce-independent rounds handed to the GPU), which
//! general-purpose hashing crates do not expose, so it is implemented here.

pub const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[inline(always)]
pub fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
#[inline(always)]
pub fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
#[inline(always)]
pub fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
#[inline(always)]
pub fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}
#[inline(always)]
pub fn ch(e: u32, f: u32, g: u32) -> u32 {
    g ^ (e & (f ^ g))
}
#[inline(always)]
pub fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) | (c & (a | b))
}

/// One SHA-256 round on a canonical `[a, b, c, d, e, f, g, h]` state.
#[inline(always)]
pub fn round(s: &mut [u32; 8], k_plus_w: u32) {
    let t1 = s[7]
        .wrapping_add(big_sigma1(s[4]))
        .wrapping_add(ch(s[4], s[5], s[6]))
        .wrapping_add(k_plus_w);
    let t2 = big_sigma0(s[0]).wrapping_add(maj(s[0], s[1], s[2]));
    *s = [
        t1.wrapping_add(t2),
        s[0],
        s[1],
        s[2],
        s[3].wrapping_add(t1),
        s[4],
        s[5],
        s[6],
    ];
}

/// The SHA-256 compression function.
pub fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes(chunk.try_into().unwrap());
    }
    for i in 16..64 {
        w[i] = small_sigma1(w[i - 2])
            .wrapping_add(w[i - 7])
            .wrapping_add(small_sigma0(w[i - 15]))
            .wrapping_add(w[i - 16]);
    }
    let mut s = *state;
    for i in 0..64 {
        round(&mut s, K[i].wrapping_add(w[i]));
    }
    for i in 0..8 {
        state[i] = state[i].wrapping_add(s[i]);
    }
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = IV;
    let mut chunks = data.chunks_exact(64);
    for block in &mut chunks {
        compress(&mut state, block.try_into().unwrap());
    }
    // Padding: 0x80, zeros, then the bit length as a big-endian u64.
    let rest = chunks.remainder();
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let tail_len = if rest.len() < 56 { 64 } else { 128 };
    tail[tail_len - 8..tail_len].copy_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in tail[..tail_len].chunks_exact(64) {
        compress(&mut state, block.try_into().unwrap());
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Bitcoin's double SHA-256.
pub fn sha256d(data: &[u8]) -> [u8; 32] {
    sha256(&sha256(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(
            hex::encode(sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex::encode(sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Two-block message (56 bytes forces the length into a second block).
        assert_eq!(
            hex::encode(sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn genesis_block_hash() {
        let header = hex::decode(crate::work::GENESIS_HEADER_HEX).unwrap();
        let mut hash = sha256d(&header);
        hash.reverse();
        assert_eq!(
            hex::encode(hash),
            "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
        );
    }
}
