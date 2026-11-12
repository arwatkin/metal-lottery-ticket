//! State shared between the miner, the pool connection and the interface.

use crate::config::Config;
use crate::work::Job;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const LOG_CAPACITY: usize = 500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Good,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
pub struct LogLine {
    pub time: String,
    pub level: Level,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    Connecting,
    Online,
    Offline,
}

/// What the interface shows about the job currently being mined.
#[derive(Clone, Debug)]
pub struct JobInfo {
    pub id: String,
    pub height: Option<u64>,
    pub network_difficulty: f64,
    pub share_difficulty: f64,
    pub prev_hash: String,
    pub received: Instant,
}

pub struct Details {
    pub gpu_name: String,
    pub link: Link,
    pub job: Option<JobInfo>,
    pub best_difficulty: f64,
    pub last_accepted: Option<Instant>,
    pub blocks_found: u64,
    pub logs: VecDeque<LogLine>,
    /// Set when the miner cannot continue; the interface exits and prints it.
    pub fatal: Option<String>,
}

pub struct Shared {
    pub config: Config,
    pub started: Instant,
    pub quit: AtomicBool,
    pub paused: AtomicBool,
    /// Print log lines to stdout as they happen (headless mode).
    pub echo_logs: AtomicBool,

    /// Total hashes computed since start.
    pub hashes: AtomicU64,
    /// Hashrate measured by the start-up benchmark; 0 until the GPU is ready.
    pub benchmark: AtomicU64,
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
    /// Hashes that met difficulty 1 (whether or not the pool wanted them).
    pub candidates: AtomicU64,
    pub jobs: AtomicU64,

    /// Bumped whenever `job` is replaced, so the miner can poll cheaply.
    pub generation: AtomicU64,
    pub job: Mutex<Option<Arc<Job>>>,
    details: Mutex<Details>,
}

impl Shared {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            started: Instant::now(),
            quit: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            echo_logs: AtomicBool::new(false),
            hashes: AtomicU64::new(0),
            benchmark: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            candidates: AtomicU64::new(0),
            jobs: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            job: Mutex::new(None),
            details: Mutex::new(Details {
                gpu_name: String::new(),
                link: Link::Connecting,
                job: None,
                best_difficulty: 0.0,
                last_accepted: None,
                blocks_found: 0,
                logs: VecDeque::with_capacity(LOG_CAPACITY),
                fatal: None,
            }),
        }
    }

    pub fn details(&self) -> MutexGuard<'_, Details> {
        self.details
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn quitting(&self) -> bool {
        self.quit.load(Ordering::Relaxed)
    }

    pub fn log(&self, level: Level, text: impl Into<String>) {
        let line = LogLine {
            time: clock(),
            level,
            text: text.into(),
        };
        if self.echo_logs.load(Ordering::Relaxed) {
            println!("{} {}", line.time, line.text);
        }
        let mut details = self.details();
        if details.logs.len() == LOG_CAPACITY {
            details.logs.pop_front();
        }
        details.logs.push_back(line);
    }

    /// Records an unrecoverable error and asks every thread to stop.
    pub fn fail(&self, message: String) {
        self.details().fatal = Some(message);
        self.quit.store(true, Ordering::Relaxed);
    }

    /// Installs (or clears) the job the miner should work on.
    pub fn set_job(&self, job: Option<Arc<Job>>) {
        *self
            .job
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = job;
        self.generation.fetch_add(1, Ordering::Release);
    }

    pub fn current_job(&self) -> Option<Arc<Job>> {
        self.job
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Sleeps for `duration`, waking early if the miner is shutting down.
    pub fn sleep(&self, duration: Duration) {
        let deadline = Instant::now() + duration;
        while !self.quitting() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            std::thread::sleep(left.min(Duration::from_millis(50)));
        }
    }
}

/// Local wall-clock time as `HH:MM:SS`.
fn clock() -> String {
    // SAFETY: `time` accepts a null pointer and `localtime_r` only writes to
    // the `tm` we hand it.
    let tm = unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut tm);
        tm
    };
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

/// Turns the monotonically increasing hash counter into a hashrate.
pub struct RateMeter {
    samples: VecDeque<(Instant, u64)>,
    window: Duration,
}

impl RateMeter {
    pub fn new(window: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            window,
        }
    }

    /// Records the current total and returns hashes per second over the window.
    pub fn sample(&mut self, total: u64) -> f64 {
        let now = Instant::now();
        self.samples.push_back((now, total));
        while self.samples.len() > 2 && now.duration_since(self.samples[1].0) >= self.window {
            self.samples.pop_front();
        }
        let (then, before) = self.samples[0];
        let seconds = now.duration_since(then).as_secs_f64();
        if seconds < 0.2 {
            0.0
        } else {
            (total - before) as f64 / seconds
        }
    }
}
