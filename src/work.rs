//! Turning pool jobs into block headers, targets and GPU work units.

use crate::gpu::KernelParams;
use crate::sha256::{self, IV, K, big_sigma0, big_sigma1, ch, maj, small_sigma0, small_sigma1};

/// The Bitcoin genesis block header - used as a known-answer test.
pub const GENESIS_HEADER_HEX: &str = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c";

/// A 256-bit proof-of-work threshold, stored big-endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Target([u8; 32]);

/// 2^208, the scale of the difficulty-1 target (`0xFFFF * 2^208`).
const TWO_POW_208: f64 = 4.113_761_393_303_015e62;

impl Target {
    /// The target a hash must not exceed to count as a share of `difficulty`.
    pub fn from_difficulty(difficulty: f64) -> Self {
        if !(difficulty.is_finite() && difficulty > 0.0) {
            return Target([0xff; 32]);
        }
        // target = 0xFFFF * 2^208 / difficulty. Split the quotient into a
        // 53-bit integer mantissa and a power of two, then place the bits.
        let quotient = 65535.0 / difficulty;
        let bits = quotient.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        if exponent == 0 {
            return Target([0; 32]); // subnormal: absurdly high difficulty
        }
        let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
        let shift = exponent - 1075 + 208;
        if shift + 53 > 256 {
            return Target([0xff; 32]);
        }
        let mut out = [0u8; 32];
        for bit in 0..53 {
            let position = bit + shift;
            if mantissa >> bit & 1 == 1 && position >= 0 {
                out[31 - (position / 8) as usize] |= 1 << (position % 8);
            }
        }
        Target(out)
    }

    /// Decodes the compact `nbits` encoding from a block header.
    pub fn from_nbits(nbits: u32) -> Self {
        let exponent = (nbits >> 24) as i32;
        let mantissa = nbits & 0x007f_ffff;
        let mut out = [0u8; 32];
        for i in 0..3 {
            let position = 32 - exponent + i;
            if (0..32).contains(&position) {
                out[position as usize] = (mantissa >> (16 - 8 * i)) as u8;
            }
        }
        Target(out)
    }

    /// Whether a double-SHA256 digest (as produced, i.e. little-endian when
    /// read as a number) satisfies this target.
    pub fn is_met_by(&self, hash: &[u8; 32]) -> bool {
        hash.iter().rev().le(self.0.iter())
    }

    pub fn difficulty(&self) -> f64 {
        difficulty_of(self.0.iter().copied())
    }
}

/// Difficulty of a 256-bit number given as big-endian bytes.
fn difficulty_of(big_endian: impl Iterator<Item = u8>) -> f64 {
    let value = big_endian.fold(0.0f64, |acc, byte| acc * 256.0 + byte as f64);
    65535.0 * TWO_POW_208 / value
}

/// Difficulty achieved by a double-SHA256 digest.
pub fn hash_difficulty(hash: &[u8; 32]) -> f64 {
    difficulty_of(hash.iter().rev().copied())
}

/// A unit of work announced by the pool (`mining.notify`), together with the
/// session parameters needed to turn it into block headers.
#[derive(Clone, Debug)]
pub struct Job {
    /// The pool connection that issued this job.
    pub session: u64,
    pub id: String,
    /// Previous block hash in header byte order.
    pub prev_hash: [u8; 32],
    pub coinbase1: Vec<u8>,
    pub coinbase2: Vec<u8>,
    pub merkle_branch: Vec<[u8; 32]>,
    pub version: u32,
    pub nbits: u32,
    pub ntime: u32,
    /// The pool wants work on earlier jobs abandoned.
    pub clean: bool,
    pub extranonce1: Vec<u8>,
    pub extranonce2_size: usize,
    /// Difficulty the pool wants shares to meet.
    pub share_difficulty: f64,
}

impl Job {
    /// Encodes an extranonce2 counter at the width the pool asked for.
    pub fn extranonce2(&self, counter: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; self.extranonce2_size];
        for (slot, byte) in bytes.iter_mut().rev().zip(counter.to_le_bytes()) {
            *slot = byte;
        }
        bytes
    }

    /// Builds the 80-byte header for a given extranonce2, with a zero nonce.
    pub fn header(&self, extranonce2: &[u8]) -> [u8; 80] {
        let mut coinbase = Vec::with_capacity(
            self.coinbase1.len()
                + self.extranonce1.len()
                + extranonce2.len()
                + self.coinbase2.len(),
        );
        coinbase.extend_from_slice(&self.coinbase1);
        coinbase.extend_from_slice(&self.extranonce1);
        coinbase.extend_from_slice(extranonce2);
        coinbase.extend_from_slice(&self.coinbase2);

        // Fold the coinbase hash up the left edge of the merkle tree.
        let mut root = sha256::sha256d(&coinbase);
        let mut pair = [0u8; 64];
        for branch in &self.merkle_branch {
            pair[..32].copy_from_slice(&root);
            pair[32..].copy_from_slice(branch);
            root = sha256::sha256d(&pair);
        }

        let mut header = [0u8; 80];
        header[0..4].copy_from_slice(&self.version.to_le_bytes());
        header[4..36].copy_from_slice(&self.prev_hash);
        header[36..68].copy_from_slice(&root);
        header[68..72].copy_from_slice(&self.ntime.to_le_bytes());
        header[72..76].copy_from_slice(&self.nbits.to_le_bytes());
        header
    }

    pub fn network_target(&self) -> Target {
        Target::from_nbits(self.nbits)
    }

    /// Block height, read from the BIP34 push at the start of the coinbase
    /// script. Returns `None` if the coinbase does not look as expected.
    pub fn height(&self) -> Option<u64> {
        // version (4) + input count (1) + previous output (36) + script length (1)
        let script = self.coinbase1.get(42..)?;
        let len = *script.first()? as usize;
        if !(1..=8).contains(&len) {
            return None;
        }
        let bytes = script.get(1..1 + len)?;
        Some(bytes.iter().rev().fold(0u64, |acc, &b| acc << 8 | b as u64))
    }
}

/// Precomputes everything about a header that does not depend on the nonce.
/// See [`crate::gpu::kernel`] for how the kernel uses each field.
pub fn kernel_params(header: &[u8; 80]) -> KernelParams {
    let mut mid = IV;
    sha256::compress(&mut mid, header[..64].try_into().unwrap());

    let word = |i: usize| u32::from_be_bytes(header[64 + 4 * i..68 + 4 * i].try_into().unwrap());
    let (w0, w1, w2) = (word(0), word(1), word(2));

    // Rounds 0-2 of the second block only touch w0..w2.
    let mut s = mid;
    for (i, w) in [w0, w1, w2].into_iter().enumerate() {
        sha256::round(&mut s, K[i].wrapping_add(w));
    }
    // Round 3 with the nonce left out: the kernel adds it to slots 3 and 7.
    let t1 = s[7]
        .wrapping_add(big_sigma1(s[4]))
        .wrapping_add(ch(s[4], s[5], s[6]))
        .wrapping_add(K[3]);
    let mut pre = s;
    pre[3] = s[3].wrapping_add(t1);
    pre[7] = t1
        .wrapping_add(big_sigma0(s[0]))
        .wrapping_add(maj(s[0], s[1], s[2]));

    let w16 = small_sigma0(w1).wrapping_add(w0);
    let w17 = small_sigma1(640)
        .wrapping_add(small_sigma0(w2))
        .wrapping_add(w1);
    KernelParams {
        mid,
        pre,
        w16,
        w17,
        c18: small_sigma1(w16).wrapping_add(w2),
        c19: small_sigma1(w17).wrapping_add(small_sigma0(0x8000_0000)),
        c31: small_sigma0(w16).wrapping_add(640),
        c32: small_sigma0(w17).wrapping_add(w16),
        base: 0,
        pad: 0,
    }
}

/// Writes a GPU nonce word into a header. The kernel iterates over the
/// big-endian view of header bytes 76..80, so the word is stored as-is.
pub fn set_nonce_word(header: &mut [u8; 80], word: u32) {
    header[76..80].copy_from_slice(&word.to_be_bytes());
}

/// The nonce as Stratum expects it in `mining.submit`.
pub fn nonce_hex(word: u32) -> String {
    format!("{:08x}", word.swap_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_pow_208() {
        assert_eq!(TWO_POW_208, 2f64.powi(208));
    }

    #[test]
    fn difficulty_one_target() {
        let target = Target::from_difficulty(1.0);
        assert_eq!(
            hex::encode(target.0),
            "00000000ffff0000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(target, Target::from_nbits(0x1d00ffff));
        assert_eq!(target.difficulty(), 1.0);
    }

    #[test]
    fn fractional_and_large_difficulties() {
        assert_eq!(
            hex::encode(Target::from_difficulty(0.5).0),
            "00000001fffe0000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            hex::encode(Target::from_difficulty(65536.0).0),
            "000000000000ffff000000000000000000000000000000000000000000000000"
        );
        let hard = Target::from_difficulty(1.5e14);
        assert!((hard.difficulty() / 1.5e14 - 1.0).abs() < 1e-9);
        assert_eq!(Target::from_difficulty(1e-30).0, [0xff; 32]);
    }

    #[test]
    fn nbits_decoding() {
        // Block 800,000.
        let target = Target::from_nbits(0x17053894);
        assert_eq!(
            hex::encode(target.0),
            "0000000000000000000538940000000000000000000000000000000000000000"
        );
        assert!((target.difficulty() / 53_911_173_001_054.59 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn genesis_meets_its_own_target() {
        let header = hex::decode(GENESIS_HEADER_HEX).unwrap();
        let hash = sha256::sha256d(&header);
        let target = Target::from_nbits(0x1d00ffff);
        assert!(target.is_met_by(&hash));
        assert!(!Target::from_difficulty(3000.0).is_met_by(&hash));
        assert!((hash_difficulty(&hash) - 2536.43).abs() < 0.01);
    }

    /// Rebuilds block 1's header the way a Stratum pool would have served it.
    #[test]
    fn header_assembly() {
        // Block 1 coinbase transaction, split around an imaginary extranonce.
        let coinbase = hex::decode("01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0704ffff001d0104ffffffff0100f2052a0100000043410496b538e853519c726a2c91e61ec11600ae1390813a627c66fb8be7947be63c52da7589379515d4e0a604f8141781e62294721166bf621e73a82cbf2342c858eeac00000000").unwrap();
        let mut prev_hash =
            hex::decode("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f")
                .unwrap();
        prev_hash.reverse();
        let job = Job {
            session: 0,
            clean: true,
            id: "1".into(),
            prev_hash: prev_hash.try_into().unwrap(),
            coinbase1: coinbase[..44].to_vec(),
            coinbase2: coinbase[49..].to_vec(),
            merkle_branch: vec![],
            version: 1,
            nbits: 0x1d00ffff,
            ntime: 1231469665,
            extranonce1: coinbase[44..47].to_vec(),
            extranonce2_size: 2,
            share_difficulty: 1.0,
        };
        let mut header = job.header(&coinbase[47..49]);
        header[76..80].copy_from_slice(&2573394689u32.to_le_bytes());
        let mut hash = sha256::sha256d(&header);
        hash.reverse();
        assert_eq!(
            hex::encode(hash),
            "00000000839a8e6886ab5951d76f411475428afc90947ee320161bbf18eb6048"
        );
        assert_eq!(
            nonce_hex(u32::from_be_bytes(header[76..80].try_into().unwrap())),
            "9962e301"
        );
        assert_eq!(job.extranonce2(0x0102), vec![1, 2]);
    }

    #[test]
    fn bip34_height() {
        let mut coinbase1 = vec![0u8; 42];
        coinbase1.extend_from_slice(&[0x03, 0x40, 0x0d, 0x0c, 0xaa]);
        let job = Job {
            session: 0,
            clean: true,
            id: String::new(),
            prev_hash: [0; 32],
            coinbase1,
            coinbase2: vec![],
            merkle_branch: vec![],
            version: 0,
            nbits: 0,
            ntime: 0,
            extranonce1: vec![],
            extranonce2_size: 4,
            share_difficulty: 1.0,
        };
        assert_eq!(job.height(), Some(789_824));
    }
}
