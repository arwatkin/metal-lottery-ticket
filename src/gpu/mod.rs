//! Metal compute backend.
//!
//! Owns the device, the compiled search pipeline and a small ring of
//! parameter/result buffers so several dispatches can be queued at once and the
//! GPU never idles while the CPU inspects the previous batch.

pub mod kernel;

use anyhow::{Context, Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};

use kernel::{GRID_SHIFT, MAX_RESULTS};

// `MTLCreateSystemDefaultDevice` lives in Metal but needs CoreGraphics loaded.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Hashes covered by one grid row (one nonce per GPU thread).
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
}

#[repr(C)]
struct KernelResults {
    count: AtomicU32,
    nonces: [u32; MAX_RESULTS],
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

struct Slot {
    params: Buffer,
    results: Buffer,
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
    /// Candidate nonce words (big-endian view of header bytes 76..80).
    pub candidates: Vec<u32>,
    /// Number of hashes the dispatch covered.
    pub hashes: u64,
}

pub struct Gpu<T> {
    name: String,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    group_width: usize,
    free: Vec<Slot>,
    in_flight: VecDeque<InFlight<T>>,
}

impl<T> Gpu<T> {
    /// Opens the default Metal device and compiles the search kernel.
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .context("no Metal device found - this miner needs an Apple Silicon GPU")?;
        let name = device.name().to_string();

        let options = MTLCompileOptions::new();
        let source = NSString::from_str(&kernel::source());
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| anyhow!("kernel compilation failed: {}", e.localizedDescription()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("search"))
            .context("kernel entry point `search` missing")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("pipeline creation failed: {}", e.localizedDescription()))?;
        let queue = device
            .newCommandQueue()
            .context("failed to create a Metal command queue")?;

        // Largest power-of-two threadgroup the pipeline supports, capped at the
        // grid width so the grid is always a whole number of groups wide.
        let max_group = pipeline.maxTotalThreadsPerThreadgroup().max(1);
        let group_width = (1usize << max_group.ilog2()).min(1 << GRID_SHIFT);

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
            });
        }

        Ok(Self {
            name,
            queue,
            pipeline,
            group_width,
            free,
            in_flight: VecDeque::new(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// True if another dispatch can be queued without waiting.
    pub fn has_capacity(&self) -> bool {
        !self.free.is_empty()
    }

    pub fn pending(&self) -> usize {
        self.in_flight.len()
    }

    /// Queues a search of `rows * HASHES_PER_ROW` nonces starting at
    /// `params.base`. Panics if [`has_capacity`](Self::has_capacity) is false.
    pub fn submit(&mut self, params: &KernelParams, rows: u32, tag: T) -> Result<()> {
        let slot = self.free.pop().expect("submit called without capacity");
        // SAFETY: both buffers are CPU-visible (shared storage), were allocated
        // with exactly these struct sizes, and no dispatch is using this slot.
        unsafe {
            slot.params.contents().cast::<KernelParams>().write(*params);
            slot.results
                .contents()
                .cast::<KernelResults>()
                .as_ref()
                .count
                .store(0, Ordering::Relaxed);
        }

        let command = self
            .queue
            .commandBuffer()
            .context("failed to create a command buffer")?;
        let encoder = command
            .computeCommandEncoder()
            .context("failed to create a compute encoder")?;
        encoder.setComputePipelineState(&self.pipeline);
        // SAFETY: indices 0 and 1 match the kernel signature and both buffers
        // outlive the command buffer (they are held by the in-flight entry).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&slot.params), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&slot.results), 0, 1);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (1usize << GRID_SHIFT) / self.group_width,
                height: rows as usize,
                depth: 1,
            },
            MTLSize {
                width: self.group_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();

        self.in_flight.push_back(InFlight {
            slot,
            command,
            hashes: rows as u64 * HASHES_PER_ROW,
            tag,
        });
        Ok(())
    }

    /// Blocks until the oldest queued dispatch finishes and returns its result.
    pub fn wait(&mut self) -> Result<Option<Batch<T>>> {
        let Some(job) = self.in_flight.pop_front() else {
            return Ok(None);
        };
        job.command.waitUntilCompleted();
        if job.command.status() == MTLCommandBufferStatus::Error {
            let reason = job
                .command
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown error".into());
            self.free.push(job.slot);
            bail!("GPU dispatch failed: {reason}");
        }
        // SAFETY: the dispatch has completed, so the GPU no longer writes here.
        let results = unsafe { job.slot.results.contents().cast::<KernelResults>().as_ref() };
        let count = (results.count.load(Ordering::Relaxed) as usize).min(MAX_RESULTS);
        let candidates = results.nonces[..count].to_vec();

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
