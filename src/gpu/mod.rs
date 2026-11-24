//! Metal compute backend.
//!
//! Owns the device, the compiled search pipelines and a small ring of
//! parameter/result buffers so several dispatches can be queued at once and the
//! GPU never idles while the CPU inspects the previous batch.
//!
//! The pipelines normally run the compiler's machine code with one fix-up
//! applied ([`patch`]); if that cannot be done safely they run it unchanged.

mod agx;
#[rustfmt::skip]
mod agx_table;
pub mod kernel;
mod patch;

use anyhow::{Context, Result, anyhow, bail};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSError, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLBuffer, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLPipelineOption,
    MTLResourceOptions, MTLSize,
};
use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use kernel::{GRID_SHIFT, MAX_RESULTS};
pub use kernel::{GROUP_SHIFT, GROUP_SIZE, Kernel};

// `MTLCreateSystemDefaultDevice` lives in Metal but needs CoreGraphics loaded.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Threads (and hashes) in one grid row. Each thread computes one hash.
pub const HASHES_PER_ROW: u64 = 1u64 << GRID_SHIFT;

/// Number of dispatches kept in flight.
pub const QUEUE_DEPTH: usize = 2;

/// Per-work-unit constants consumed by the kernel. Layout must match the
/// `Params` struct in [`kernel::source`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelParams {
    pub mid: [u32; 8],
    pub pre: [u32; 8],
    pub w16: u32,
    pub w17: u32,
    pub c18: u32,
    pub c19: u32,
    pub c31: u32,
    pub c32: u32,
    pub base: u32,
    pub pad: u32,
    /// [`Kernel::Pair`] only: midstate and round-3 state of the second
    /// (version-rolled) header, hashed by the odd lanes.
    pub mid_b: [u32; 8],
    pub pre_b: [u32; 8],
    /// The SHA-256 round constants. Read through the parameter buffer they
    /// sit in uniform registers, which saves an instruction per use compared
    /// with literals. Two halves because `Default` stops at 32-element arrays.
    pub kc: [[u32; 32]; 2],
}

/// Per-version midstates and round-3 states for [`Kernel::Group`], stored in
/// structure-of-arrays order so SIMD lanes coalesce into contiguous cache lines.
/// Layout must match `struct GroupVersions` in [`kernel::source`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GroupVersions {
    pub mid: [[u32; GROUP_SIZE]; 8],
    pub pre: [[u32; GROUP_SIZE]; 8],
}

impl Default for GroupVersions {
    fn default() -> Self {
        Self {
            mid: [[0; GROUP_SIZE]; 8],
            pre: [[0; GROUP_SIZE]; 8],
        }
    }
}

#[repr(C)]
struct KernelResults {
    count: AtomicU32,
    nonces: [u32; MAX_RESULTS],
    lanes: [u32; MAX_RESULTS],
}

/// A nonce the GPU flagged, and which header version (lane) produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Big-endian view of header bytes 76..80.
    pub nonce: u32,
    /// 0 for the `mid`/`pre` header, 1 for the `mid_b`/`pre_b` header.
    pub lane: u32,
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Function = Retained<ProtocolObject<dyn MTLFunction>>;
type PipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Which machine code the kernels run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Codegen {
    /// The compiler's code with its rotate register copies removed.
    Patched(patch::Stats),
    /// The compiler's code as is, and why it was not patched.
    Compiler(String),
}

impl Codegen {
    pub fn is_patched(&self) -> bool {
        matches!(self, Codegen::Patched(_))
    }
}

impl fmt::Display for Codegen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Codegen::Patched(stats) => write!(
                f,
                "hand-patched ({} register copies removed)",
                stats.copies_removed
            ),
            Codegen::Compiler(reason) => write!(f, "compiler output ({reason})"),
        }
    }
}

struct Pipeline {
    state: PipelineState,
    /// Largest power-of-two threadgroup the pipeline supports, capped at the
    /// grid width so the grid is always a whole number of groups wide.
    group_width: usize,
}

struct Slot {
    params: Buffer,
    results: Buffer,
    versions: Buffer,
}

struct InFlight<T> {
    slot: Slot,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    hashes: u64,
    tag: T,
}

/// The outcome of one finished dispatch.
pub struct Batch<T> {
    /// Caller-supplied value identifying what this dispatch was searching.
    pub tag: T,
    pub candidates: Vec<Candidate>,
    /// Number of hashes the dispatch covered.
    pub hashes: u64,
}

pub struct Gpu<T> {
    name: String,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: [Pipeline; 3],
    supports_group: bool,
    codegen: Codegen,
    free: Vec<Slot>,
    in_flight: VecDeque<InFlight<T>>,
}

fn ns_error(error: Retained<NSError>) -> anyhow::Error {
    anyhow!("{}", error.localizedDescription())
}

/// A file in the temporary directory, deleted on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!(
            "bitcoin_miner-{}-{nanos}-{tag}.metallib",
            std::process::id()
        );
        TempFile(std::env::temp_dir().join(name))
    }

    fn url(&self) -> Retained<NSURL> {
        NSURL::fileURLWithPath(&NSString::from_str(&self.0.to_string_lossy()))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Pipelines straight from the compiler.
fn compiled_pipelines(device: &Device, functions: &[Function; 3]) -> Result<[PipelineState; 3]> {
    let build = |function: &Function| {
        device
            .newComputePipelineStateWithFunction_error(function)
            .map_err(|e| anyhow!("pipeline creation failed: {}", e.localizedDescription()))
    };
    Ok([
        build(&functions[0])?,
        build(&functions[1])?,
        build(&functions[2])?,
    ])
}

/// Hidden command-line flag: compile the kernels into a binary archive at the
/// given path and exit. See [`patched_pipelines`] for why this runs in a
/// separate process.
pub const EMIT_ARCHIVE_FLAG: &str = "--emit-kernel-archive";

fn compile_library(device: &Device) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let source = NSString::from_str(&kernel::source());
    device
        .newLibraryWithSource_options_error(&source, Some(&MTLCompileOptions::new()))
        .map_err(|e| anyhow!("kernel compilation failed: {}", e.localizedDescription()))
}

fn function(library: &ProtocolObject<dyn MTLLibrary>, entry: &str) -> Result<Function> {
    library
        .newFunctionWithName(&NSString::from_str(entry))
        .with_context(|| format!("kernel entry point `{entry}` missing"))
}

fn pipeline_descriptor(function: &Function) -> Retained<MTLComputePipelineDescriptor> {
    let d = MTLComputePipelineDescriptor::new();
    d.setComputeFunction(Some(function));
    d
}

/// Entry points stored in the archive: all three search kernels, then the canary.
fn archive_entries() -> [&'static str; 4] {
    [
        Kernel::Single.entry_point(),
        Kernel::Pair.entry_point(),
        Kernel::Group.entry_point(),
        kernel::CANARY_ENTRY,
    ]
}

/// Body of [`EMIT_ARCHIVE_FLAG`]: compiles every kernel into a binary archive.
pub fn emit_compiled_archive(path: &std::path::Path) -> Result<()> {
    let device = MTLCreateSystemDefaultDevice().context("no Metal device found")?;
    let library = compile_library(&device)?;
    let archive = device
        .newBinaryArchiveWithDescriptor_error(&MTLBinaryArchiveDescriptor::new())
        .map_err(ns_error)
        .context("creating a binary archive")?;
    for entry in archive_entries() {
        archive
            .addComputePipelineFunctionsWithDescriptor_error(&pipeline_descriptor(&function(
                &library, entry,
            )?))
            .map_err(ns_error)
            .context("adding a pipeline to the archive")?;
    }
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    archive
        .serializeToURL_error(&url)
        .map_err(ns_error)
        .context("serialising the archive")
}

/// Pipelines running the compiler's code after [`patch::patch_archive`].
///
/// The compiled archive comes from a child process: once a process has
/// compiled a pipeline, Metal answers later requests for the same function
/// from its in-memory cache, even when told to use a binary archive, so
/// compiling here would silently bring back the unpatched code. The canary
/// kernel then proves which code actually runs.
fn patched_pipelines(
    device: &Device,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Result<([PipelineState; 3], patch::Stats)> {
    let original = TempFile::new("compiled");
    let exe = std::env::current_exe().context("locating the miner executable")?;
    let child = std::process::Command::new(exe)
        .arg(EMIT_ARCHIVE_FLAG)
        .arg(&original.0)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .context("starting the kernel compiler process")?;
    if !child.status.success() {
        bail!(
            "kernel compiler process failed: {}",
            String::from_utf8_lossy(&child.stderr).trim()
        );
    }
    let mut bytes = std::fs::read(&original.0).context("reading the compiled archive")?;

    let stats = patch::patch_archive(&mut bytes)?;

    let patched = TempFile::new("patched");
    std::fs::write(&patched.0, &bytes).context("writing the patched archive")?;
    let archive_descriptor = MTLBinaryArchiveDescriptor::new();
    archive_descriptor.setUrl(Some(&patched.url()));
    let archive = device
        .newBinaryArchiveWithDescriptor_error(&archive_descriptor)
        .map_err(ns_error)
        .context("loading the patched archive")?;
    let archives = NSArray::from_retained_slice(&[archive]);
    let build = |entry: &str| -> Result<PipelineState> {
        let d = pipeline_descriptor(&function(library, entry)?);
        d.setBinaryArchives(Some(&archives));
        device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &d,
                MTLPipelineOption::FailOnBinaryArchiveMiss,
                None,
            )
            .map_err(ns_error)
            .context("building a pipeline from the patched archive")
    };
    let [single, pair, group, canary] = archive_entries();
    check_canary(device, queue, &build(canary)?)?;
    Ok(([build(single)?, build(pair)?, build(group)?], stats))
}

/// Runs the canary kernel: `io[1] = io[0] ^ constant`, where the patch
/// replaced the constant. Errors unless the patched constant was used.
fn check_canary(
    device: &Device,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    state: &PipelineState,
) -> Result<()> {
    let buffer = device
        .newBufferWithLength_options(8, MTLResourceOptions::StorageModeShared)
        .context("failed to allocate a Metal buffer")?;
    // SAFETY: an 8-byte CPU-visible buffer that no GPU work is using yet.
    unsafe { buffer.contents().cast::<[u32; 2]>().write([7, 0]) };
    autoreleasepool(|_| -> Result<()> {
        let command = queue
            .commandBuffer()
            .context("failed to create a command buffer")?;
        let encoder = command
            .computeCommandEncoder()
            .context("failed to create a compute encoder")?;
        encoder.setComputePipelineState(state);
        // SAFETY: index 0 matches the canary's signature; `buffer` outlives the wait.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0) };
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() == MTLCommandBufferStatus::Error {
            bail!("canary dispatch failed");
        }
        Ok(())
    })?;
    // SAFETY: the dispatch has completed.
    let out = unsafe { buffer.contents().cast::<[u32; 2]>().read() }[1];
    if out == 7 ^ kernel::CANARY_PATCHED {
        Ok(())
    } else if out == 7 ^ kernel::CANARY {
        bail!("Metal ran the unpatched code instead of the patched archive")
    } else {
        bail!("canary kernel returned {out:#x}")
    }
}

impl<T> Gpu<T> {
    /// Opens the default Metal device and compiles the search kernels,
    /// patched unless `KERNEL_PATCH=off` is set.
    pub fn new() -> Result<Self> {
        let off = std::env::var("KERNEL_PATCH").is_ok_and(|v| v.eq_ignore_ascii_case("off"));
        Self::open(off.then(|| "patch disabled by KERNEL_PATCH=off".to_string()))
    }

    /// Opens the default Metal device and compiles the search kernels. With
    /// `skip_patch`, the compiler's code runs unchanged for the reason given.
    pub fn open(skip_patch: Option<String>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .context("no Metal device found - this miner needs an Apple Silicon GPU")?;
        let name = device.name().to_string();

        let library = compile_library(&device)?;
        let queue = device
            .newCommandQueue()
            .context("failed to create a Metal command queue")?;
        let compiled = || -> Result<[PipelineState; 3]> {
            compiled_pipelines(
                &device,
                &[
                    function(&library, Kernel::Single.entry_point())?,
                    function(&library, Kernel::Pair.entry_point())?,
                    function(&library, Kernel::Group.entry_point())?,
                ],
            )
        };
        let (states, codegen) = match skip_patch {
            Some(reason) => (compiled()?, Codegen::Compiler(reason)),
            None => match patched_pipelines(&device, &queue, &library) {
                Ok((states, stats)) => (states, Codegen::Patched(stats)),
                Err(error) => (
                    compiled()?,
                    Codegen::Compiler(format!("patch skipped: {error:#}")),
                ),
            },
        };
        let supports_group =
            states[Kernel::Group as usize].maxTotalThreadsPerThreadgroup() >= GROUP_SIZE;
        let [s_single, s_pair, s_group] = states;
        let make_pipeline = |state: PipelineState, fixed_width: Option<usize>| {
            let max_group = state.maxTotalThreadsPerThreadgroup().max(1);
            let group_width = fixed_width
                .unwrap_or_else(|| (1usize << max_group.ilog2()).clamp(2, 1 << GRID_SHIFT));
            Pipeline { state, group_width }
        };
        let pipelines = [
            make_pipeline(s_single, None),
            make_pipeline(s_pair, None),
            make_pipeline(s_group, Some(GROUP_SIZE)),
        ];
        let mut free = Vec::with_capacity(QUEUE_DEPTH);
        for _ in 0..QUEUE_DEPTH {
            let alloc = |len: usize| {
                device
                    .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                    .context("failed to allocate a Metal buffer")
            };
            free.push(Slot {
                params: alloc(size_of::<KernelParams>())?,
                results: alloc(size_of::<KernelResults>())?,
                versions: alloc(size_of::<GroupVersions>())?,
            });
        }

        Ok(Self {
            name,
            queue,
            pipelines,
            supports_group,
            codegen,
            free,
            in_flight: VecDeque::new(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// True if the device supports [`Kernel::Group`]'s threadgroup size.
    pub fn supports_group(&self) -> bool {
        self.supports_group
    }

    /// Which machine code the kernels run.
    pub fn codegen(&self) -> &Codegen {
        &self.codegen
    }

    /// True if another dispatch can be queued without waiting.
    pub fn has_capacity(&self) -> bool {
        !self.free.is_empty()
    }

    pub fn pending(&self) -> usize {
        self.in_flight.len()
    }

    /// Queues `rows * HASHES_PER_ROW` hashes with `kernel`, covering
    /// `rows * HASHES_PER_ROW / kernel.threads_per_nonce()` nonces from
    /// `params.base`. For [`Kernel::Group`], pass `versions`. Panics if
    /// [`has_capacity`](Self::has_capacity) is false.
    pub fn submit(
        &mut self,
        kernel: Kernel,
        params: &KernelParams,
        versions: Option<&GroupVersions>,
        rows: u32,
        tag: T,
    ) -> Result<()> {
        debug_assert!(
            kernel != Kernel::Group || versions.is_some(),
            "Kernel::Group requires GroupVersions"
        );
        let slot = self.free.pop().expect("submit called without capacity");
        // SAFETY: all slot buffers are CPU-visible (shared storage), were
        // allocated with exactly these struct sizes, and no dispatch is using
        // this slot.
        unsafe {
            slot.params.contents().cast::<KernelParams>().write(*params);
            if let Some(v) = versions {
                slot.versions.contents().cast::<GroupVersions>().write(*v);
            }
            slot.results
                .contents()
                .cast::<KernelResults>()
                .as_ref()
                .count
                .store(0, Ordering::Relaxed);
        }

        // Metal hands back autoreleased objects (the compute encoder, at least).
        // The miner thread has no run loop to drain them, so without a pool of
        // our own every dispatch leaks one: millions over a long run, until
        // malloc runs out of address space.
        let command = match autoreleasepool(|_| self.encode(kernel, &slot, rows)) {
            Ok(command) => command,
            Err(error) => {
                self.free.push(slot);
                return Err(error);
            }
        };

        self.in_flight.push_back(InFlight {
            slot,
            command,
            hashes: rows as u64 * HASHES_PER_ROW,
            tag,
        });
        Ok(())
    }

    /// Encodes and commits one dispatch reading and writing `slot`.
    fn encode(
        &self,
        kernel: Kernel,
        slot: &Slot,
        rows: u32,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let command = self
            .queue
            .commandBuffer()
            .context("failed to create a command buffer")?;
        let encoder = command
            .computeCommandEncoder()
            .context("failed to create a compute encoder")?;
        let pipeline = &self.pipelines[kernel as usize];
        encoder.setComputePipelineState(&pipeline.state);
        // SAFETY: indices match the kernel signature and all buffers outlive
        // the command buffer (they are held by the in-flight entry).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&slot.params), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&slot.results), 0, 1);
            if kernel == Kernel::Group {
                encoder.setBuffer_offset_atIndex(Some(&slot.versions), 0, 2);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (1usize << GRID_SHIFT) / pipeline.group_width,
                height: rows as usize,
                depth: 1,
            },
            MTLSize {
                width: pipeline.group_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        Ok(command)
    }

    /// Blocks until the oldest queued dispatch finishes and returns its result.
    pub fn wait(&mut self) -> Result<Option<Batch<T>>> {
        let Some(job) = self.in_flight.pop_front() else {
            return Ok(None);
        };
        let failure = autoreleasepool(|_| {
            job.command.waitUntilCompleted();
            (job.command.status() == MTLCommandBufferStatus::Error).then(|| {
                job.command
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into())
            })
        });
        if let Some(reason) = failure {
            self.free.push(job.slot);
            bail!("GPU dispatch failed: {reason}");
        }
        // SAFETY: the dispatch has completed, so the GPU no longer writes here.
        let results = unsafe { job.slot.results.contents().cast::<KernelResults>().as_ref() };
        let count = (results.count.load(Ordering::Relaxed) as usize).min(MAX_RESULTS);
        let candidates = (0..count)
            .map(|i| Candidate {
                nonce: results.nonces[i],
                lane: results.lanes[i],
            })
            .collect();

        self.free.push(job.slot);
        Ok(Some(Batch {
            tag: job.tag,
            candidates,
            hashes: job.hashes,
        }))
    }

    /// Waits for every queued dispatch, discarding the results.
    pub fn drain(&mut self) {
        while !self.in_flight.is_empty() {
            let _ = self.wait();
        }
    }
}
