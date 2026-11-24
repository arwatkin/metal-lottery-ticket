//! A minimal decoder for Apple G13 GPU machine code (the M1 family's "AGX").
//!
//! Just enough to support [`super::patch`]: identify every instruction and
//! its length using [`agx_table`](super::agx_table), and, for the handful of
//! ALU instructions that make up the SHA-256 rounds, say exactly which
//! registers they read and write. Everything else is reported as opaque, and
//! the patcher treats opaque instructions as barriers.
//!
//! Operand layouts follow applegpu (<https://github.com/dougallj/applegpu>,
//! BSD-3-Clause, Dougall Johnson); see `tools/gen_agx_table.py`.

use super::agx_table::ENCODINGS;

/// Longest instruction, in bytes.
pub const MAX_LEN: usize = 12;

/// The general-purpose registers an operand covers, in 16-bit halves:
/// half `2n` is `rNl`, `2n + 1` is `rNh`, and a 32-bit `rN` covers both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Regs {
    pub first: u16,
    pub halves: u16,
}

impl Regs {
    pub fn overlaps(self, other: Regs) -> bool {
        self.first < other.first + other.halves && other.first < self.first + self.halves
    }

    /// `Some(n)` if this is exactly the 32-bit register `rN`.
    pub fn as_r32(self) -> Option<u16> {
        (self.halves == 2 && self.first.is_multiple_of(2)).then_some(self.first / 2)
    }
}

/// A decoded source operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// An immediate or a uniform register: no general-purpose register read.
    Constant,
    Reg(Regs),
}

/// One decoded instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instr {
    pub offset: usize,
    pub len: usize,
    pub name: &'static str,
    /// Raw little-endian instruction bits.
    pub word: u128,
    /// `None` for instructions whose operands are not modelled (opaque).
    pub alu: Option<Alu>,
}

/// Register effects of a modelled ALU instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alu {
    pub dst: Regs,
    /// Sources in operand order (e.g. `A, B, C` for `extr`).
    pub srcs: Vec<Src>,
}

fn field(word: u128, start: u32, size: u32) -> u32 {
    ((word >> start) & ((1u128 << size) - 1)) as u32
}

/// Reads up to [`MAX_LEN`] bytes at `offset`, zero-padded past the end.
fn word_at(code: &[u8], offset: usize) -> u128 {
    let mut bytes = [0u8; 16];
    let end = (offset + MAX_LEN).min(code.len());
    bytes[..end - offset].copy_from_slice(&code[offset..end]);
    u128::from_le_bytes(bytes)
}

/// Identifies the instruction at `offset`. `None` if no known encoding matches.
pub fn decode(code: &[u8], offset: usize) -> Option<Instr> {
    let raw = word_at(code, offset);
    for e in ENCODINGS {
        let len = if (raw >> e.length_bit) & 1 == 1 {
            e.sizes.1
        } else {
            e.sizes.0
        } as usize;
        let word = raw & ((1u128 << (len * 8)) - 1);
        if word & e.mask == e.bits {
            if offset + len > code.len() {
                return None;
            }
            return Some(Instr {
                offset,
                len,
                name: e.name,
                word,
                alu: alu_effects(e.name, word),
            });
        }
    }
    None
}

/// Destination operand (applegpu `ALUDstDesc`, or `ALUDst64Desc` if `allow64`).
fn dst(word: u128, ex: u32, allow64: bool) -> Regs {
    let flags = field(word, 7, 2);
    let value = (field(word, 9, 6) | field(word, ex, 2) << 6) as u16;
    if flags & 2 != 0 {
        if value & 1 != 0 && allow64 {
            Regs {
                first: (value >> 1) * 2,
                halves: 4,
            }
        } else {
            Regs {
                first: (value >> 1) * 2,
                halves: 2,
            }
        }
    } else {
        Regs {
            first: value,
            halves: 1,
        }
    }
}

/// Source operand (applegpu `ALUSrcDesc` and its 16/64-bit variants).
/// `None` for encodings applegpu does not define.
fn src(word: u128, off: u32, ex: u32, allow32: bool, allow64: bool) -> Option<Src> {
    let flags = field(word, off + 6, 4);
    let value = (field(word, off, 6) | field(word, ex, 2) << 6) as u16;
    if flags == 0 || flags & 0b1100 == 0b0100 {
        return Some(Src::Constant);
    }
    if flags & 0b11 == 0 {
        return None;
    }
    let size = flags & 0b1100;
    let regs = if allow64 && size == 0b1100 {
        Regs {
            first: (value >> 1) * 2,
            halves: 4,
        }
    } else if allow32 && (size == 0b1000 || size == 0b1100) {
        Regs {
            first: (value >> 1) * 2,
            halves: 2,
        }
    } else if size == 0b0000 || size == 0b1000 || size == 0b1100 {
        Regs {
            first: value,
            halves: 1,
        }
    } else {
        return None;
    };
    Some(Src::Reg(regs))
}

/// Compare-and-select source (applegpu `CmpselSrcDesc`); its width follows
/// the destination's.
fn cmpsel_src(word: u128, off: u32, ex: u32, dst32: bool) -> Option<Src> {
    let flags = field(word, off + 6, 3);
    let value = (field(word, off, 6) | field(word, ex, 2) << 6) as u16;
    match flags {
        0b100 | 0b110 | 0b111 => Some(Src::Constant),
        0b001..=0b011 if dst32 => Some(Src::Reg(Regs {
            first: (value >> 1) * 2,
            halves: 2,
        })),
        0b001..=0b011 => Some(Src::Reg(Regs {
            first: value,
            halves: 1,
        })),
        _ => None,
    }
}

fn alu_effects(name: &str, w: u128) -> Option<Alu> {
    let alu = |dst: Regs, srcs: Vec<Option<Src>>| {
        srcs.into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|srcs| Alu { dst, srcs })
    };
    match name {
        "bitop" => alu(
            dst(w, 44, false),
            vec![src(w, 16, 42, true, false), src(w, 28, 40, true, false)],
        ),
        "iadd" => alu(
            dst(w, 44, true),
            vec![src(w, 16, 42, true, true), src(w, 28, 40, true, true)],
        ),
        "imadd" => alu(
            dst(w, 60, true),
            vec![
                src(w, 16, 58, true, false),
                src(w, 28, 56, true, false),
                src(w, 40, 54, true, true),
            ],
        ),
        "extr" | "bfeil" | "bfi" => alu(
            dst(w, 60, false),
            vec![
                src(w, 16, 58, true, false),
                src(w, 28, 56, true, false),
                src(w, 40, 54, true, false),
            ],
        ),
        // Bit 8 selects the 32-bit immediate form, whose extension bits sit higher.
        "mov_imm" => alu(dst(w, if w >> 8 & 1 == 0 { 44 } else { 60 }, false), vec![]),
        "icmpsel" => {
            let d = dst(w, 76, false);
            let dst32 = field(w, 7, 2) & 2 != 0;
            alu(
                d,
                vec![
                    src(w, 16, 74, true, false),
                    src(w, 28, 72, true, false),
                    cmpsel_src(w, 40, 70, dst32),
                    cmpsel_src(w, 52, 68, dst32),
                ],
            )
        }
        "simd_shuffle_xor" => alu(
            dst(w, 44, false),
            vec![src(w, 16, 42, true, false), src(w, 28, 40, false, false)],
        ),
        _ => None,
    }
}

/// Truth table of a `bitop`, as applegpu prints it (`"0101"` is `mov`).
pub fn bitop_truth_table(word: u128) -> [u32; 4] {
    [
        field(word, 26, 1),
        field(word, 27, 1),
        field(word, 38, 1),
        field(word, 39, 1),
    ]
}

/// Bit positions of `extr` source operands `A` and `B`: (value, type, extension).
const EXTR_A: (u32, u32, u32) = (16, 22, 58);
const EXTR_B: (u32, u32, u32) = (28, 34, 56);

/// Rewrites an `extr` so its `B` operand reads exactly what `A` reads
/// (register, size and cache hint). The instruction keeps its length.
pub fn extr_copy_a_into_b(word: u128) -> u128 {
    let get =
        |(v, t, x): (u32, u32, u32)| (field(word, v, 6), field(word, t, 4), field(word, x, 2));
    let (v, t, x) = get(EXTR_A);
    let mut out = word;
    for (start, size, value) in [(EXTR_B.0, 6, v), (EXTR_B.1, 4, t), (EXTR_B.2, 2, x)] {
        out &= !(((1u128 << size) - 1) << start);
        out |= (value as u128) << start;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes one instruction given as hex, as the disassembler prints it.
    fn one(hex: &str) -> Instr {
        let code = hex::decode(hex).unwrap();
        let instr = decode(&code, 0).expect("decodes");
        assert_eq!(instr.len, code.len(), "length of {hex}");
        instr
    }

    fn r32(n: u16) -> Src {
        Src::Reg(Regs {
            first: 2 * n,
            halves: 2,
        })
    }

    // Expected values below are applegpu's disassembly of these exact bytes,
    // taken from the compiled SHA-256 kernels.

    #[test]
    fn rotate_and_copy() {
        // extr r24.cache, r6.cache, r12.cache, 2
        let extr = one("ae618c8629020000");
        assert_eq!(extr.name, "extr");
        let alu = extr.alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(24));
        assert_eq!(alu.srcs, vec![r32(6), r32(12), Src::Constant]);

        // mov r8.cache, r12.cache
        let mov = one("fe21980a8000");
        assert_eq!(mov.name, "bitop");
        assert_eq!(bitop_truth_table(mov.word), [0, 1, 0, 1]);
        let alu = mov.alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(8));
        assert_eq!(alu.srcs, vec![r32(12), Src::Constant]);
    }

    #[test]
    fn arithmetic_and_logic() {
        // iadd r23.cache, r23.discard, r21.discard
        let alu = one("8e5deea22e000000").alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(23));
        assert_eq!(alu.srcs, vec![r32(23), r32(21)]);
        // iadd r1.cache, u23, r14.cache  (uniform source)
        let alu = one("8e05aec129000000").alu.unwrap();
        assert_eq!(alu.srcs, vec![Src::Constant, r32(14)]);
        // xor r12.cache, r24.discard, r28.discard
        let x = one("fe31f08a6f00");
        assert_eq!(bitop_truth_table(x.word), [0, 1, 1, 0]);
        assert_eq!(x.alu.unwrap().srcs, vec![r32(24), r32(28)]);
        // bfeil r22.cache, 0, r18, 3
        let alu = one("aed9004026030000").alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(22));
        assert_eq!(alu.srcs, vec![Src::Constant, r32(18), Src::Constant]);
        // mov_imm r20.cache, 1538233109
        let m = one("e2511593af5b");
        assert_eq!((m.name, m.alu.unwrap().dst.as_r32()), ("mov_imm", Some(20)));
        // imadd r17.cache, r4l.discard, u105, r3.discard
        let alu = one("9e45c82019c60203").alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(17));
        assert_eq!(
            alu.srcs,
            vec![
                Src::Reg(Regs {
                    first: 8,
                    halves: 1
                }),
                Src::Constant,
                r32(3)
            ]
        );
        // icmpsel seq, r8.cache, r2, 0, r16, r20
        let alu = one("9221440200608086").alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(8));
        assert_eq!(alu.srcs, vec![r32(2), Src::Constant, r32(16), r32(20)]);
    }

    #[test]
    fn opaque_and_short_instructions() {
        let stop = one("8800");
        assert_eq!((stop.name, stop.alu), ("stop", None));
        // device_load 0, i32, xyzw, r0_r1_r2_r3, u0_u1, 4, signed, lsl 2
        let load = one("0501400d00c8f200");
        assert_eq!((load.name, load.alu), ("device_load", None));
        assert_eq!(one("3800").name, "wait");
    }

    #[test]
    fn extr_rewrite_reads_a_twice() {
        // extr r22.cache, r12.cache, r8.cache, 2  ->  extr r22.cache, r12.cache, r12.cache, 2
        let before = one("ae59980629020000");
        let after = extr_copy_a_into_b(before.word);
        let bytes = &after.to_le_bytes()[..8];
        let again = decode(bytes, 0).unwrap();
        assert_eq!(again.name, "extr");
        let alu = again.alu.unwrap();
        assert_eq!(alu.dst.as_r32(), Some(22));
        assert_eq!(alu.srcs[0], alu.srcs[1]);
        assert_eq!(alu.srcs[0], r32(12));
    }
}
