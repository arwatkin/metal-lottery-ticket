//! The mining loop: feeds work to the GPU and checks what comes back.

use crate::gpu::{Batch, Candidate, Codegen, Gpu, HASHES_PER_ROW, Kernel, KernelParams};
use crate::sha256::sha256d;
use crate::state::{Level, Shared};
use crate::stratum::{Client, Share};
use crate::ui::format;
use crate::work::{self, Job, Target};
use anyhow::{Result, bail};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Size of the nonce space of one header.
const NONCE_SPACE: u64 = 1 << 32;
/// How much work to hand the GPU per dispatch. Short dispatches keep new jobs
/// from waiting and leave the window server room to draw the desktop.
const DISPATCH_SECONDS: f64 = 0.03;
const STARTUP_BENCHMARK: Duration = Duration::from_millis(1500);
/// Length of a hashing burst when INTENSITY is below 100.
const THROTTLE_BURST: Duration = Duration::from_secs(1);

/// One header's worth of work: a job with a fixed extranonce2.
struct Unit {
    job: Arc<Job>,
    extranonce2: Vec<u8>,
    header: [u8; 80],
    /// With version rolling: the version bit the odd GPU lanes add, and the
    /// header they hash.
    rolled: Option<(u32, [u8; 80])>,
    kernel: Kernel,
    params: KernelParams,
    share_target: Target,
    network_target: Target,
}

impl Unit {
    fn new(job: Arc<Job>, counter: u64) -> Arc<Self> {
        let extranonce2 = job.extranonce2(counter);
        let header = job.header(&extranonce2);
        let rolled = work::rolled_version(job.version, job.version_mask)
            .map(|version| (version ^ job.version, work::with_version(&header, version)));
        let (kernel, params) = match &rolled {
            Some((_, rolled_header)) => (
                Kernel::Pair,
                work::kernel_params_pair(&header, rolled_header),
            ),
            None => (Kernel::Single, work::kernel_params(&header)),
        };
        Arc::new(Self {
            rolled,
            kernel,
            params,
            share_target: Target::from_difficulty(job.share_difficulty),
            network_target: job.network_target(),
            extranonce2,
            header,
            job,
        })
    }
}

type Tag = Option<Arc<Unit>>;

/// Miner thread entry point.
pub fn run(shared: Arc<Shared>, client: Arc<Client>) {
    if let Err(error) = mine(&shared, &client) {
        shared.fail(format!("GPU error: {error:#}"));
    }
}

/// Nonces one grid row covers with `kernel`.
fn nonces_per_row(kernel: Kernel) -> u64 {
    HASHES_PER_ROW / kernel.threads_per_nonce() as u64
}

fn genesis() -> [u8; 80] {
    hex::decode(work::GENESIS_HEADER_HEX)
        .expect("valid hex")
        .try_into()
        .expect("80 bytes")
}

/// Genesis with a version bit set: a header whose nonce is unknown.
fn genesis_rolled() -> [u8; 80] {
    let header = genesis();
    let version = u32::from_le_bytes(header[0..4].try_into().unwrap());
    work::with_version(&header, version | 1 << 13)
}

/// Opens the GPU and checks its kernels with [`self_test`]. Hand-patched
/// kernels that fail it are replaced by the compiler's unpatched ones, which
/// must then pass.
pub fn open_tested_gpu<T: Default>() -> Result<Gpu<T>> {
    let mut gpu = Gpu::new()?;
    match self_test(&mut gpu) {
        Ok(()) => Ok(gpu),
        Err(error) if gpu.codegen().is_patched() => {
            drop(gpu);
            let mut plain = Gpu::open(Some(format!(
                "patched kernels failed the self-test: {error:#}"
            )))?;
            self_test(&mut plain)?;
            Ok(plain)
        }
        Err(error) => Err(error),
    }
}

/// Checks both kernels against the genesis block: the GPU must find its
/// nonce, in either lane of the paired kernel.
pub fn self_test<T: Default>(gpu: &mut Gpu<T>) -> Result<()> {
    let (real, other) = (genesis(), genesis_rolled());
    expect_genesis(gpu, Kernel::Single, work::kernel_params(&real), 0)?;
    expect_genesis(
        gpu,
        Kernel::Pair,
        work::kernel_params_pair(&real, &other),
        0,
    )?;
    expect_genesis(
        gpu,
        Kernel::Pair,
        work::kernel_params_pair(&other, &real),
        1,
    )
}

fn expect_genesis<T: Default>(
    gpu: &mut Gpu<T>,
    kernel: Kernel,
    mut params: KernelParams,
    lane: u32,
) -> Result<()> {
    let nonce = u32::from_be_bytes(genesis()[76..80].try_into().unwrap());
    let rows = 64;
    let span = rows as u64 * nonces_per_row(kernel);
    params.base = nonce & !(span as u32 - 1);
    gpu.submit(kernel, &params, rows, T::default())?;
    let batch = gpu.wait()?.expect("a dispatch was just queued");
    let expected = Candidate { nonce, lane };
    if batch.candidates != [expected] {
        bail!(
            "self-test failed ({kernel:?} kernel): expected {expected:?}, GPU reported {:?}",
            batch.candidates
        );
    }
    Ok(())
}

/// Measures raw hashrate (H/s) of `kernel` on a fixed header.
pub fn benchmark<T: Default>(gpu: &mut Gpu<T>, kernel: Kernel, duration: Duration) -> Result<f64> {
    let mut params = match kernel {
        Kernel::Single => work::kernel_params(&genesis()),
        Kernel::Pair => work::kernel_params_pair(&genesis(), &genesis_rolled()),
    };
    let rows = 1024;
    let step = (rows as u64 * nonces_per_row(kernel)) as u32;
    let start = Instant::now();
    let mut hashes = 0u64;
    while start.elapsed() < duration {
        while gpu.has_capacity() {
            gpu.submit(kernel, &params, rows, T::default())?;
            params.base = params.base.wrapping_add(step);
        }
        hashes += gpu.wait()?.map_or(0, |batch| batch.hashes);
    }
    while gpu.pending() > 0 {
        hashes += gpu.wait()?.map_or(0, |batch| batch.hashes);
    }
    Ok(hashes as f64 / start.elapsed().as_secs_f64())
}

fn mine(shared: &Shared, client: &Client) -> Result<()> {
    let mut gpu: Gpu<Tag> = open_tested_gpu()?;
    shared.details().gpu_name = gpu.name().to_string();
    if let Codegen::Compiler(reason) = gpu.codegen() {
        shared.log(
            Level::Warn,
            format!("Using unpatched GPU kernels  ·  {reason}"),
        );
    }
    // Most pools allow version rolling, so measure the paired kernel.
    let hashrate = benchmark(&mut gpu, Kernel::Pair, STARTUP_BENCHMARK)?;
    shared.log(
        Level::Good,
        format!(
            "{} ready  ·  self-test passed  ·  {} kernels  ·  {}",
            gpu.name(),
            if gpu.codegen().is_patched() {
                "hand-patched"
            } else {
                "compiler"
            },
            format::hashrate(hashrate)
        ),
    );
    shared
        .benchmark
        .store(hashrate.max(1.0) as u64, Ordering::Relaxed);

    let rows =
        ((hashrate * DISPATCH_SECONDS / HASHES_PER_ROW as f64).round() as u64).clamp(16, 16384);
    let intensity = shared.config.intensity as f64 / 100.0;
    let mut burst = Instant::now();

    let mut generation = u64::MAX;
    let mut live_job: Option<Arc<Job>> = None;
    let mut unit: Option<Arc<Unit>> = None;
    let mut counter = 0u64;
    let mut offset = 0u64;
    let mut was_paused = false;

    while !shared.quitting() {
        let current = shared.generation.load(Ordering::Acquire);
        if current != generation {
            generation = current;
            live_job = shared.current_job();
            counter = 0;
            offset = 0;
            unit = live_job.clone().map(|job| Unit::new(job, counter));
        }

        let paused = shared.paused.load(Ordering::Relaxed);
        if paused != was_paused {
            was_paused = paused;
            shared.log(
                Level::Info,
                if paused {
                    "Mining paused"
                } else {
                    "Mining resumed"
                },
            );
        }

        if !paused {
            while gpu.has_capacity() {
                let Some(current_unit) = unit.clone() else {
                    break;
                };
                let per_row = nonces_per_row(current_unit.kernel);
                let rows_now = rows.min((NONCE_SPACE - offset) / per_row) as u32;
                let params = KernelParams {
                    base: offset as u32,
                    ..current_unit.params
                };
                gpu.submit(
                    current_unit.kernel,
                    &params,
                    rows_now,
                    Some(current_unit.clone()),
                )?;
                offset += rows_now as u64 * per_row;
                if offset >= NONCE_SPACE {
                    // Every nonce tried: change the coinbase to get a fresh header.
                    counter += 1;
                    offset = 0;
                    unit = Some(Unit::new(current_unit.job.clone(), counter));
                }
            }
        }

        let Some(batch) = gpu.wait()? else {
            // Nothing to do: paused, or waiting for the pool to send work.
            std::thread::sleep(Duration::from_millis(20));
            burst = Instant::now();
            continue;
        };
        finish(shared, client, batch, live_job.as_ref());

        // Below full intensity, alternate bursts of hashing with idle time.
        // Bursts are long enough for the GPU to stay at full clock speed.
        if intensity < 1.0 && burst.elapsed() >= THROTTLE_BURST {
            while let Some(batch) = gpu.wait()? {
                finish(shared, client, batch, live_job.as_ref());
            }
            shared.sleep(burst.elapsed().mul_f64((1.0 - intensity) / intensity));
            burst = Instant::now();
        }
    }
    gpu.drain();
    Ok(())
}

/// Accounts for a finished dispatch and checks any candidates it produced.
fn finish(shared: &Shared, client: &Client, batch: Batch<Tag>, live_job: Option<&Arc<Job>>) {
    shared.hashes.fetch_add(batch.hashes, Ordering::Relaxed);
    if let Some(source) = &batch.tag {
        for &candidate in &batch.candidates {
            check(shared, client, source, candidate, live_job);
        }
    }
}

/// Verifies a GPU candidate on the CPU and submits it if the pool wants it.
fn check(
    shared: &Shared,
    client: &Client,
    unit: &Unit,
    candidate: Candidate,
    live_job: Option<&Arc<Job>>,
) {
    let word = candidate.nonce;
    let (mut header, version_bits) = match (candidate.lane, &unit.rolled) {
        (0, _) => (unit.header, None),
        (1, Some((bits, rolled))) => (*rolled, Some(*bits)),
        _ => {
            shared.log(
                Level::Error,
                format!("GPU reported an impossible lane ({candidate:?})"),
            );
            return;
        }
    };
    work::set_nonce_word(&mut header, word);
    let hash = sha256d(&header);
    if hash[28..] != [0; 4] {
        shared.log(
            Level::Error,
            format!("GPU reported a bad nonce ({word:08x}) - hardware fault?"),
        );
        return;
    }

    let difficulty = work::hash_difficulty(&hash);
    shared.candidates.fetch_add(1, Ordering::Relaxed);
    {
        let mut details = shared.details();
        if difficulty > details.best_difficulty {
            details.best_difficulty = difficulty;
        }
    }

    // Work on a superseded job is only still useful if the pool did not ask
    // for a clean start and the job came from the current connection.
    let job = &unit.job;
    let usable = live_job
        .is_some_and(|live| Arc::ptr_eq(live, job) || (!live.clean && live.session == job.session));
    let is_block = unit.network_target.is_met_by(&hash);
    if !usable || !(is_block || unit.share_target.is_met_by(&hash)) {
        return;
    }

    if is_block {
        let mut display = hash;
        display.reverse();
        shared.details().blocks_found += 1;
        shared.log(
            Level::Good,
            format!("BLOCK FOUND!  {}", hex::encode(display)),
        );
    }
    let sent = client.submit(
        &shared.config.username(),
        Share {
            session: job.session,
            job_id: job.id.clone(),
            extranonce2: hex::encode(&unit.extranonce2),
            ntime: format!("{:08x}", job.ntime),
            nonce: work::nonce_hex(word),
            version_bits,
            difficulty,
            is_block,
        },
    );
    if !sent {
        shared.log(Level::Warn, "Found a share but the pool connection is down");
    }
}
