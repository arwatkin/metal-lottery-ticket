//! Post-compilation patch for the search kernels: removes register copies
//! that Apple's shader compiler inserts before rotates.
//!
//! The compiler lowers a rotate to `extr rD, rX, rY, n` (extract from the
//! register pair `rY:rX`) and, to rotate `rX`, first copies it: `mov rY, rX`.
//! The hardware is happy to read the same register twice (`extr rD, rX, rX,
//! n`), so those copies, about 325 per hash, are pure overhead in a kernel
//! that is limited by instruction issue. Removing them is worth ~5%.
//!
//! The patch works on a serialised `MTLBinaryArchive` and is deliberately
//! conservative:
//!
//! * Every instruction of each kernel's main program must decode to a known
//!   encoding and appear on a short list of instructions these kernels use.
//!   Anything else and the archive is left alone.
//! * A copy `mov rD, rS` is removed only if, until `rD` is next written, every
//!   read of `rD` is the second source of an `extr` whose first source is
//!   `rS`, `rS` is unchanged, and every instruction in between is a fully
//!   modelled ALU instruction. Those `extr`s then read `rS` twice.
//! * Copies are only removed ahead of the first control-flow instruction, so
//!   no branch spans the code that moves.
//!
//! The caller still runs the genesis self-test on the patched kernels and
//! falls back to the compiler's code if anything about this goes wrong.

use super::agx::{self, Instr, Regs, Src};
use super::kernel::{CANARY, CANARY_PATCHED};
use anyhow::{Context, Result, bail, ensure};

/// What a patch changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub kernels: usize,
    pub copies_removed: usize,
    pub rotates_rewritten: usize,
}

/// Instructions that may appear in a kernel's main program. ALU instructions
/// are modelled; the others may only occur before the first copy (loads,
/// thread ids) or at the end (the result check and store).
const ALU: &[&str] = &[
    "bitop",
    "iadd",
    "imadd",
    "extr",
    "bfeil",
    "bfi",
    "mov_imm",
    "icmpsel",
    "simd_shuffle_xor",
];
const PROLOGUE: &[&str] = &["get_sr", "device_load", "wait"];
const EPILOGUE: &[&str] = &[
    "if_icmp",
    "jmp_exec_none",
    "pop_exec",
    "device_store",
    "atomic",
    "icmp_ballot",
    "ffs",
    "bitrev",
    "simd_shuffle",
    "stop",
];

/// Patches every kernel in a serialised Metal binary archive in place: the
/// canary kernel gets its constant flipped (so the caller can tell that the
/// patched code is what runs), every other kernel loses its rotate copies.
pub fn patch_archive(archive: &mut [u8]) -> Result<Stats> {
    let kernels = locate_kernels(archive)?;
    let mut stats = Stats::default();
    let mut canaries = 0;
    for k in kernels {
        let text = &mut archive[k.text.clone()];
        let context = || format!("kernel at offset {:#x}", k.text.start);
        if flip_canary(text, k.main).with_context(context)? {
            canaries += 1;
            continue;
        }
        let (patched, removed, rewritten) =
            remove_rotate_copies(text, k.main).with_context(context)?;
        text.copy_from_slice(&patched);
        stats.kernels += 1;
        stats.copies_removed += removed;
        stats.rotates_rewritten += rewritten;
    }
    ensure!(
        canaries == 1,
        "expected one canary kernel, found {canaries}"
    );
    ensure!(
        stats.copies_removed > 0,
        "found no register copies to remove"
    );
    Ok(stats)
}

/// If this is the canary kernel (its main program XORs with the immediate
/// [`CANARY`]), rewrites that immediate to [`CANARY_PATCHED`] and returns true.
fn flip_canary(text: &mut [u8], main: usize) -> Result<bool> {
    // (value field, type field, extension field) of a bitop's A and B sources.
    const SOURCES: [(u32, u32, u32); 2] = [(16, 22, 42), (28, 34, 40)];
    let get = |w: u128, start: u32, size: u32| ((w >> start) & ((1u128 << size) - 1)) as u32;
    let program = decode_program(text, main)?;
    let mut hits = Vec::new();
    for instr in &program {
        if instr.name != "bitop" || agx::bitop_truth_table(instr.word) != [0, 1, 1, 0] {
            continue;
        }
        for (value, kind, ex) in SOURCES {
            let immediate = get(instr.word, kind, 4) == 0;
            let n = get(instr.word, value, 6) | get(instr.word, ex, 2) << 6;
            if immediate && n == CANARY {
                hits.push((instr.offset, instr.len, instr.word, value, ex));
            }
        }
    }
    match hits.as_slice() {
        [] => Ok(false),
        [(at, len, word, value, ex)] => {
            let mut w = *word & !(0x3fu128 << value) & !(0x3u128 << ex);
            w |= ((CANARY_PATCHED & 0x3f) as u128) << value | ((CANARY_PATCHED >> 6) as u128) << ex;
            text[*at..at + len].copy_from_slice(&w.to_le_bytes()[..*len]);
            Ok(true)
        }
        _ => bail!("canary immediate found more than once"),
    }
}

/// Removes rotate copies from one kernel's `__text` section, whose main
/// program starts at `main`. Returns the new section (same length, padded
/// with zeros) and the number of copies removed and rotates rewritten.
pub fn remove_rotate_copies(text: &[u8], main: usize) -> Result<(Vec<u8>, usize, usize)> {
    let program = decode_program(text, main)?;
    let control = program
        .iter()
        .position(|i| !ALU.contains(&i.name) && !PROLOGUE.contains(&i.name))
        .unwrap_or(program.len());

    let mut remove = vec![false; program.len()];
    let mut rewrite = vec![false; program.len()];
    for (i, instr) in program[..control].iter().enumerate() {
        let Some((d, s)) = as_copy(instr) else {
            continue;
        };
        if let Some(uses) = copy_uses(&program, i, d, s) {
            remove[i] = true;
            for j in uses {
                rewrite[j] = true;
            }
        }
    }

    let mut out = Vec::with_capacity(text.len());
    out.extend_from_slice(&text[..main]);
    for (i, instr) in program.iter().enumerate() {
        if remove[i] {
            continue;
        }
        let bytes = &text[instr.offset..instr.offset + instr.len];
        if rewrite[i] {
            let new = agx::extr_copy_a_into_b(instr.word).to_le_bytes();
            out.extend_from_slice(&new[..instr.len]);
        } else {
            out.extend_from_slice(bytes);
        }
    }
    // Whatever followed the program (normally nothing) keeps its place.
    let end = program.last().map_or(main, |i| i.offset + i.len);
    out.extend_from_slice(&text[end..]);
    out.resize(text.len(), 0);
    let removed = remove.iter().filter(|&&r| r).count();
    let rewritten = rewrite.iter().filter(|&&r| r).count();
    Ok((out, removed, rewritten))
}

/// Decodes the main program up to and including its `stop`, insisting that
/// every instruction is known and expected.
fn decode_program(text: &[u8], main: usize) -> Result<Vec<Instr>> {
    let mut program = Vec::new();
    let mut offset = main;
    loop {
        let instr = agx::decode(text, offset)
            .with_context(|| format!("unknown instruction at {offset:#x}"))?;
        let name = instr.name;
        ensure!(
            ALU.contains(&name) || PROLOGUE.contains(&name) || EPILOGUE.contains(&name),
            "unexpected `{name}` at {offset:#x}"
        );
        ensure!(
            !ALU.contains(&name) || instr.alu.is_some(),
            "unmodelled `{name}` operands at {offset:#x}"
        );
        offset += instr.len;
        program.push(instr);
        if name == "stop" {
            return Ok(program);
        }
        ensure!(
            offset < text.len(),
            "program runs off the end of its section"
        );
    }
}

/// `Some((rD, rS))` if `instr` is `mov rD, rS` between distinct 32-bit registers.
fn as_copy(instr: &Instr) -> Option<(Regs, Regs)> {
    if instr.name != "bitop" || agx::bitop_truth_table(instr.word) != [0, 1, 0, 1] {
        return None;
    }
    // `mov` is the bitop alias whose second operand is the immediate 0.
    let b_raw =
        (instr.word >> 28) & 0x3f | ((instr.word >> 40) & 3) << 6 | ((instr.word >> 34) & 0xf) << 8;
    if b_raw != 0 {
        return None;
    }
    let alu = instr.alu.as_ref()?;
    let Src::Reg(s) = alu.srcs[0] else {
        return None;
    };
    let d = alu.dst;
    (d.as_r32().is_some() && s.as_r32().is_some() && !d.overlaps(s)).then_some((d, s))
}

/// The `extr`s to rewrite if the copy `program[at]` (`mov d, s`) is removed,
/// or `None` if removing it is not provably safe.
fn copy_uses(program: &[Instr], at: usize, d: Regs, s: Regs) -> Option<Vec<usize>> {
    let mut uses = Vec::new();
    let mut s_changed = false;
    for (j, instr) in program.iter().enumerate().skip(at + 1) {
        let alu = instr.alu.as_ref()?; // opaque instruction: cannot see its reads
        let reads_d = alu
            .srcs
            .iter()
            .any(|src| matches!(src, Src::Reg(r) if r.overlaps(d)));
        if reads_d {
            let rotate_of_s = instr.name == "extr"
                && alu.srcs[0] == Src::Reg(s)
                && alu.srcs[1] == Src::Reg(d)
                && !matches!(alu.srcs[2], Src::Reg(r) if r.overlaps(d));
            if !rotate_of_s || s_changed {
                return None;
            }
            uses.push(j);
        }
        if alu.dst.overlaps(s) {
            s_changed = true;
        }
        if alu.dst.overlaps(d) {
            break; // the copy is dead from here
        }
    }
    (!uses.is_empty()).then_some(uses)
}

// ---- Archive container ------------------------------------------------------

/// Where one kernel's code sits in the archive.
struct KernelCode {
    /// Byte range of the kernel's `__text` section within the archive.
    text: std::ops::Range<usize>,
    /// Offset of `_agc.main` within that section.
    main: usize,
}

fn u32_at(buf: &[u8], off: usize, big_endian: bool) -> Result<u32> {
    let bytes: [u8; 4] = buf
        .get(off..off + 4)
        .context("archive truncated")?
        .try_into()
        .unwrap();
    Ok(if big_endian {
        u32::from_be_bytes(bytes)
    } else {
        u32::from_le_bytes(bytes)
    })
}

fn u64_at(buf: &[u8], off: usize) -> Result<u64> {
    let bytes: [u8; 8] = buf
        .get(off..off + 8)
        .context("archive truncated")?
        .try_into()
        .unwrap();
    Ok(u64::from_le_bytes(bytes))
}

const MH_MAGIC_64: u32 = 0xfeed_facf;
const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
/// `CPU_TYPE_APPLE_GPU | CPU_ARCH_ABI64`.
const CPU_APPLE_GPU: u32 = 0x0100_0013;

struct Section {
    segment: String,
    name: String,
    offset: usize,
    size: usize,
}

struct MachO {
    sections: Vec<Section>,
    symbols: Vec<(String, u64)>,
    /// One past the last byte the image uses.
    extent: usize,
}

fn cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Parses the 64-bit Mach-O image at the start of `m`.
fn parse_macho(m: &[u8]) -> Result<MachO> {
    ensure!(
        u32_at(m, 0, false)? == MH_MAGIC_64,
        "not a 64-bit Mach-O image"
    );
    let ncmds = u32_at(m, 16, false)? as usize;
    let mut sections = Vec::new();
    let mut symbols = Vec::new();
    let mut extent = 32 + u32_at(m, 20, false)? as usize;
    let mut p = 32;
    for _ in 0..ncmds {
        let cmd = u32_at(m, p, false)?;
        let size = u32_at(m, p + 4, false)? as usize;
        ensure!(size >= 8, "malformed load command");
        if cmd == LC_SEGMENT_64 {
            let nsects = u32_at(m, p + 64, false)? as usize;
            for j in 0..nsects {
                let s = p + 72 + 80 * j;
                let raw = m.get(s..s + 80).context("archive truncated")?;
                let section = Section {
                    name: cstr(&raw[0..16]),
                    segment: cstr(&raw[16..32]),
                    size: u64_at(m, s + 40)? as usize,
                    offset: u32_at(m, s + 48, false)? as usize,
                };
                extent = extent.max(section.offset + section.size);
                sections.push(section);
            }
        } else if cmd == LC_SYMTAB {
            let symoff = u32_at(m, p + 8, false)? as usize;
            let nsyms = u32_at(m, p + 12, false)? as usize;
            let stroff = u32_at(m, p + 16, false)? as usize;
            let strsize = u32_at(m, p + 20, false)? as usize;
            extent = extent.max(symoff + 16 * nsyms).max(stroff + strsize);
            for k in 0..nsyms {
                let e = symoff + 16 * k;
                let strx = u32_at(m, e, false)? as usize;
                let name = cstr(m.get(stroff + strx..).context("archive truncated")?);
                symbols.push((name, u64_at(m, e + 8)?));
            }
        }
        p += size;
    }
    Ok(MachO {
        sections,
        symbols,
        extent,
    })
}

/// Finds every kernel's code: the archive is a fat binary whose Apple GPU
/// slice holds a `__TEXT,__compute` section of back-to-back Mach-O images,
/// one per pipeline, each with a `__text` section and an `_agc.main` symbol.
fn locate_kernels(archive: &[u8]) -> Result<Vec<KernelCode>> {
    let magic = u32_at(archive, 0, true)?;
    ensure!(
        matches!(magic, 0xcafe_babe | 0xcbfe_babe),
        "not a fat binary ({magic:#x})"
    );
    let nslices = u32_at(archive, 4, true)? as usize;
    let mut gpu = None;
    for i in 0..nslices {
        let e = 8 + 20 * i;
        if u32_at(archive, e, true)? == CPU_APPLE_GPU {
            ensure!(gpu.is_none(), "several GPU slices");
            let offset = u32_at(archive, e + 8, true)? as usize;
            let size = u32_at(archive, e + 12, true)? as usize;
            gpu = Some(offset..offset + size);
        }
    }
    let slice_range = gpu.context("no Apple GPU slice")?;
    let slice = archive
        .get(slice_range.clone())
        .context("archive truncated")?;
    let outer = parse_macho(slice)?;
    let compute = outer
        .sections
        .iter()
        .find(|s| s.segment == "__TEXT" && s.name == "__compute")
        .context("no __TEXT,__compute section")?;
    let compute_start = slice_range.start + compute.offset;
    let compute_bytes = archive
        .get(compute_start..compute_start + compute.size)
        .context("archive truncated")?;

    let mut kernels = Vec::new();
    let mut at = 0;
    while at < compute_bytes.len() {
        let image = parse_macho(&compute_bytes[at..])?;
        let text = image
            .sections
            .iter()
            .find(|s| s.segment == "__TEXT" && s.name == "__text")
            .context("kernel image without __text")?;
        let main = image
            .symbols
            .iter()
            .find(|(name, _)| name == "_agc.main")
            .context("kernel image without _agc.main")?
            .1 as usize;
        ensure!(main < text.size, "_agc.main outside __text");
        let start = compute_start + at + text.offset;
        kernels.push(KernelCode {
            text: start..start + text.size,
            main,
        });
        // The next image follows, possibly after zero padding.
        at += image.extent;
        while at < compute_bytes.len() && compute_bytes[at] == 0 {
            at += 1;
        }
    }
    if kernels.len() > 16 {
        bail!("implausible number of kernel images ({})", kernels.len());
    }
    Ok(kernels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(hexes: &[&str]) -> Vec<u8> {
        hexes.iter().flat_map(|h| hex::decode(h).unwrap()).collect()
    }

    /// Instructions from the real single-lane kernel: a copy feeding three
    /// rotates (`mov r8, r12`, then `extr` by 2, 13 and 22), the next write to
    /// `r8` (which ends the copy's life), and `stop`.
    const ROTATES: &[&str] = &[
        "fe21980a8000",     // mov  r8.cache, r12.cache
        "8e51ea822e000000", // iadd r20.cache, r21.discard, r20.discard
        "ae59980629020000", // extr r22.cache, r12.cache, r8.cache, 2
        "ae659806290d0000", // extr r25.cache, r12.cache, r8.cache, 13
        "ae5598062d160000", // extr r21.cache, r12.cache, r8.discard, 22
        "fe218e0a8000",     // mov  r8.cache, r7.cache
        "8800",             // stop
    ];

    #[test]
    fn removes_a_copy_and_rewrites_its_rotates() {
        let text = code(ROTATES);
        let (out, removed, rewritten) = remove_rotate_copies(&text, 0).unwrap();
        assert_eq!((removed, rewritten), (1, 3));
        assert_eq!(out.len(), text.len());
        assert!(
            out[out.len() - 6..].iter().all(|&b| b == 0),
            "freed space is zeroed"
        );
        let program = decode_program(&out, 0).unwrap();
        assert_eq!(program.len(), ROTATES.len() - 1);
        assert_eq!(program[0].name, "iadd", "the copy is gone");
        for instr in program.iter().filter(|i| i.name == "extr") {
            let srcs = &instr.alu.as_ref().unwrap().srcs;
            assert_eq!(srcs[0], srcs[1], "rotate reads r12 twice");
            assert_eq!(
                srcs[0],
                Src::Reg(Regs {
                    first: 24,
                    halves: 2
                })
            );
        }
    }

    #[test]
    fn keeps_a_copy_with_any_other_reader() {
        // Same, but the copy is also XORed: it must stay.
        let mut hexes = ROTATES.to_vec();
        hexes.insert(2, "fe1df20a6d00"); // xor r7.cache, r25.discard, r8.discard
        let text = code(&hexes);
        let program = decode_program(&text, 0).unwrap();
        let xor = &program[2];
        assert!(matches!(xor.alu.as_ref().unwrap().srcs[1], Src::Reg(r) if r.as_r32() == Some(8)));
        let (out, removed, _) = remove_rotate_copies(&text, 0).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(out, text);
    }

    #[test]
    fn keeps_a_copy_whose_source_changes() {
        // `iadd r12, ...` between the copy and its rotates changes the source.
        let mut hexes = ROTATES.to_vec();
        hexes.insert(1, "8e31d8622f000000"); // iadd r12.cache, r12.discard, r27.discard
        let text = code(&hexes);
        let (out, removed, _) = remove_rotate_copies(&text, 0).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(out, text);
    }

    #[test]
    fn keeps_a_copy_still_live_at_the_end() {
        // Without the redefinition the copy reaches `stop`, past which its
        // reads cannot be seen: it must stay.
        let mut hexes = ROTATES.to_vec();
        hexes.remove(ROTATES.len() - 2);
        let text = code(&hexes);
        assert_eq!(remove_rotate_copies(&text, 0).unwrap().1, 0);
    }

    #[test]
    fn refuses_unknown_code() {
        let mut text = code(ROTATES);
        text.splice(0..0, [0xff, 0xff]);
        assert!(remove_rotate_copies(&text, 0).is_err());
        // Missing `stop`.
        let text = code(&ROTATES[..ROTATES.len() - 1]);
        assert!(remove_rotate_copies(&text, 0).is_err());
    }
}
