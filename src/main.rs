//! A Bitcoin miner for Apple Silicon GPUs.
//!
//! * [`gpu`] - Metal compute backend and the SHA-256d search kernel
//! * [`work`] - block header assembly, targets, per-header precomputation
//! * [`stratum`] - Stratum v1 pool client
//! * [`miner`] - the loop that ties GPU and pool together
//! * [`ui`] - ratatui dashboard and headless output
//! * [`config`] - `.env` configuration

#[cfg(not(target_os = "macos"))]
compile_error!("bitcoin_miner uses Apple's Metal API and only builds on macOS (Apple Silicon).");

mod config;
mod gpu;
mod miner;
mod sha256;
mod state;
mod stratum;
mod ui;
mod work;

use anyhow::{Result, bail};
use state::{Level, Shared};
use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

const HELP: &str = concat!(
    env!("CARGO_PKG_NAME"),
    " ",
    env!("CARGO_PKG_VERSION"),
    " - ",
    env!("CARGO_PKG_DESCRIPTION"),
    "

USAGE:
    bitcoin_miner [OPTIONS]

OPTIONS:
    --benchmark [SECONDS]   Measure GPU hashrate (default 10 s) and exit
    --headless              Print log lines instead of drawing the dashboard
    -h, --help              Show this help
    -V, --version           Show the version

CONFIGURATION (.env file or environment):
    BTC_ADDRESS     Address that receives the block reward (required)
    POOL            Stratum pool as host:port   [default: solo.ckpool.org:3333]
    WORKER          Optional name for this machine
    POOL_PASSWORD   Pool password               [default: x]
    INTENSITY       GPU duty cycle, 1-100       [default: 100]

KEYS:
    q / Esc   quit        p / Space   pause or resume
"
);

enum Mode {
    Mine { headless: bool },
    Benchmark(Duration),
}

fn parse_args() -> Result<Option<Mode>> {
    let mut mode = Mode::Mine { headless: false };
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--headless" => mode = Mode::Mine { headless: true },
            "--benchmark" => {
                let seconds = match args.next_if(|next| !next.starts_with('-')) {
                    None => 10,
                    Some(raw) => match raw.parse::<u64>() {
                        Ok(seconds @ 1..=3600) => seconds,
                        _ => bail!("--benchmark takes a number of seconds (1-3600), got {raw:?}"),
                    },
                };
                mode = Mode::Benchmark(Duration::from_secs(seconds));
            }
            other => bail!("unknown option {other:?} (try --help)"),
        }
    }
    Ok(Some(mode))
}

fn run_benchmark(duration: Duration) -> Result<()> {
    let mut gpu = gpu::Gpu::<()>::new()?;
    println!("GPU        {}", gpu.name());
    miner::self_test(&mut gpu)?;
    println!("Self-test  passed (found the genesis block nonce)");
    println!("Hashing for {} s ...", duration.as_secs());
    let hashrate = miner::benchmark(&mut gpu, duration)?;
    println!("Hashrate   {}", ui::format::hashrate(hashrate));
    Ok(())
}

fn run_miner(headless: bool) -> Result<()> {
    let config = config::Config::load()?;
    let shared = Arc::new(Shared::new(config));
    let client = Arc::new(stratum::Client::new());

    // Without a terminal there is nothing to draw on (e.g. output piped to a file).
    let headless = headless || !std::io::stdout().is_terminal();
    shared.echo_logs.store(headless, Ordering::Relaxed);
    shared.log(
        Level::Info,
        format!(
            "Mining to {}  ·  intensity {}%",
            shared.config.address, shared.config.intensity
        ),
    );

    let miner_thread = std::thread::Builder::new().name("miner".into()).spawn({
        let (shared, client) = (shared.clone(), client.clone());
        move || miner::run(shared, client)
    })?;
    let pool_thread = std::thread::Builder::new().name("stratum".into()).spawn({
        let (shared, client) = (shared.clone(), client.clone());
        move || stratum::run(shared, client)
    })?;

    let interface = if headless {
        ui::run_headless(&shared);
        Ok(())
    } else {
        ui::run(&shared)
    };

    shared.quit.store(true, Ordering::Relaxed);
    miner_thread.join().ok();
    pool_thread.join().ok();
    interface?;

    let details = shared.details();
    if let Some(message) = &details.fatal {
        bail!("{message}");
    }
    let hashes = shared.hashes.load(Ordering::Relaxed);
    let uptime = shared.started.elapsed();
    println!(
        "Mined for {}  ·  {} hashes ({} average)  ·  {} accepted, {} rejected  ·  best share {}  ·  {} blocks",
        ui::format::clock(uptime),
        ui::format::count(hashes),
        ui::format::hashrate(hashes as f64 / uptime.as_secs_f64().max(1.0)),
        shared.accepted.load(Ordering::Relaxed),
        shared.rejected.load(Ordering::Relaxed),
        ui::format::difficulty(details.best_difficulty),
        details.blocks_found,
    );
    Ok(())
}

fn main() -> ExitCode {
    let outcome = match parse_args() {
        Ok(None) => Ok(()),
        Ok(Some(Mode::Benchmark(duration))) => run_benchmark(duration),
        Ok(Some(Mode::Mine { headless })) => run_miner(headless),
        Err(error) => Err(error),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
