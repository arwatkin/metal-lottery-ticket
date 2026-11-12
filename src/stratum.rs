//! Stratum v1 pool client.
//!
//! One thread owns the connection and reads newline-delimited JSON from the
//! pool; the miner thread writes share submissions through [`Client::submit`].

use crate::state::{JobInfo, Level, Link, Shared};
use crate::work::Job;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Pools send work at least every minute or so; longer silence means a dead link.
const SILENCE_LIMIT: Duration = Duration::from_secs(180);
const MAX_BACKOFF: Duration = Duration::from_secs(30);
/// Aim for roughly one share this often when suggesting a difficulty.
const SHARE_INTERVAL_SECS: f64 = 20.0;

const ID_SUBSCRIBE: u64 = 1;
const ID_AUTHORIZE: u64 = 2;
const ID_SUGGEST: u64 = 3;
const ID_FIRST_SHARE: u64 = 100;

/// A solved share, ready to be sent to the pool.
pub struct Share {
    pub session: u64,
    pub job_id: String,
    pub extranonce2: String,
    pub ntime: String,
    pub nonce: String,
    pub difficulty: f64,
    pub is_block: bool,
}

struct Pending {
    difficulty: f64,
    is_block: bool,
    sent: Instant,
}

/// Handle shared between the connection thread and the miner.
pub struct Client {
    writer: Mutex<Option<TcpStream>>,
    pending: Mutex<HashMap<u64, Pending>>,
    next_id: AtomicU64,
    /// Identifies the current connection; shares from older ones are dropped.
    session: AtomicU64,
}

impl Client {
    pub fn new() -> Self {
        Self {
            writer: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(ID_FIRST_SHARE),
            session: AtomicU64::new(0),
        }
    }

    fn send(&self, message: &Value) -> Result<()> {
        let mut line = message.to_string();
        line.push('\n');
        let mut writer = self.writer.lock().unwrap_or_else(|p| p.into_inner());
        let stream = writer.as_mut().context("not connected")?;
        stream
            .write_all(line.as_bytes())
            .context("failed to send to pool")
    }

    /// Submits a share. Returns `false` if it could not be sent (for example
    /// because the connection that issued the job is gone).
    pub fn submit(&self, username: &str, share: Share) -> bool {
        if share.session != self.session.load(Ordering::Acquire) {
            return false;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(
                id,
                Pending {
                    difficulty: share.difficulty,
                    is_block: share.is_block,
                    sent: Instant::now(),
                },
            );
        let message = json!({
            "id": id,
            "method": "mining.submit",
            "params": [username, share.job_id, share.extranonce2, share.ntime, share.nonce],
        });
        let sent = self.send(&message).is_ok();
        if !sent {
            self.pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(&id);
        }
        sent
    }

    fn attach(&self, stream: Option<TcpStream>) {
        *self.writer.lock().unwrap_or_else(|p| p.into_inner()) = stream;
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }
}

/// Per-connection protocol state.
struct Session {
    id: u64,
    extranonce1: Vec<u8>,
    extranonce2_size: usize,
    difficulty: f64,
    authorized: bool,
}

/// Connection thread: keeps a session alive until the miner shuts down.
pub fn run(shared: Arc<Shared>, client: Arc<Client>) {
    // Wait for the GPU so the difficulty suggestion can use its real speed.
    while shared.benchmark.load(Ordering::Relaxed) == 0 {
        if shared.quitting() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let mut backoff = Duration::from_secs(1);
    while !shared.quitting() {
        shared.details().link = Link::Connecting;
        shared.log(Level::Info, format!("Connecting to {}", shared.config.pool));

        let started = Instant::now();
        let outcome = session(&shared, &client);

        client.attach(None);
        shared.set_job(None);
        {
            let mut details = shared.details();
            details.link = Link::Offline;
            details.job = None;
        }
        if shared.quitting() {
            break;
        }
        if let Err(error) = outcome {
            shared.log(Level::Error, format!("Pool connection lost: {error:#}"));
        }
        // A session that lasted a while was healthy: retry quickly.
        if started.elapsed() > Duration::from_secs(60) {
            backoff = Duration::from_secs(1);
        }
        shared.log(
            Level::Warn,
            format!("Reconnecting in {}s", backoff.as_secs()),
        );
        shared.sleep(backoff);
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}

fn connect(pool: &str) -> Result<TcpStream> {
    let addresses = pool
        .to_socket_addrs()
        .with_context(|| format!("cannot resolve {pool}"))?;
    let mut last_error = anyhow!("{pool} did not resolve to any address");
    for address in addresses {
        match TcpStream::connect_timeout(&address, CONNECT_TIMEOUT) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = anyhow!("cannot connect to {address}: {error}"),
        }
    }
    Err(last_error)
}

fn session(shared: &Shared, client: &Client) -> Result<()> {
    let stream = connect(&shared.config.pool)?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(POLL_INTERVAL))?;
    stream.set_write_timeout(Some(CONNECT_TIMEOUT))?;

    let mut state = Session {
        id: client.session.fetch_add(1, Ordering::AcqRel) + 1,
        extranonce1: Vec::new(),
        extranonce2_size: 0,
        difficulty: 1.0,
        authorized: false,
    };
    client.attach(Some(stream.try_clone()?));
    client
        .send(&json!({"id": ID_SUBSCRIBE, "method": "mining.subscribe", "params": [USER_AGENT]}))?;

    let mut reader = BufReader::new(stream);
    let mut line = Vec::with_capacity(4096);
    let mut last_heard = Instant::now();
    while !shared.quitting() {
        // On a timeout `read_until` keeps the partial line in `line`, so the
        // next call simply continues where it left off.
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => bail!("pool closed the connection"),
            Ok(_) if line.ends_with(b"\n") => {
                last_heard = Instant::now();
                let text = String::from_utf8_lossy(&line);
                let text = text.trim();
                if !text.is_empty() {
                    match serde_json::from_str::<Value>(text) {
                        Ok(message) => handle(shared, client, &mut state, &message)?,
                        Err(_) => {
                            shared.log(Level::Warn, "Ignored a malformed message from the pool")
                        }
                    }
                }
                line.clear();
            }
            Ok(_) => {} // partial line at EOF; the next read reports the close
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if last_heard.elapsed() > SILENCE_LIMIT {
                    bail!("no data from pool for {}s", SILENCE_LIMIT.as_secs());
                }
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error).context("read failed"),
        }
    }
    Ok(())
}

fn handle(shared: &Shared, client: &Client, state: &mut Session, message: &Value) -> Result<()> {
    match message.get("method").and_then(Value::as_str) {
        Some(method) => {
            let params = message.get("params").cloned().unwrap_or(Value::Null);
            notification(shared, client, state, method, &params, message.get("id"))
        }
        None => response(shared, client, state, message),
    }
}

fn response(shared: &Shared, client: &Client, state: &mut Session, message: &Value) -> Result<()> {
    let id = message.get("id").and_then(Value::as_u64).unwrap_or(0);
    let result = message.get("result").unwrap_or(&Value::Null);
    let error = message.get("error").filter(|e| !e.is_null());
    let ok = error.is_none() && result.as_bool() == Some(true);

    match id {
        ID_SUBSCRIBE => {
            let parsed = (|| {
                let fields = result.as_array()?;
                let extranonce1 = hex::decode(fields.get(1)?.as_str()?).ok()?;
                let size = fields.get(2)?.as_u64()? as usize;
                Some((extranonce1, size))
            })();
            let Some((extranonce1, size)) = parsed else {
                bail!(
                    "pool rejected mining.subscribe: {}",
                    describe(error, result)
                );
            };
            if !(1..=16).contains(&size) {
                bail!("pool asked for an unsupported extranonce2 size ({size})");
            }
            state.extranonce1 = extranonce1;
            state.extranonce2_size = size;
            let config = &shared.config;
            client.send(&json!({
                "id": ID_AUTHORIZE,
                "method": "mining.authorize",
                "params": [config.username(), config.password],
            }))?;
        }
        ID_AUTHORIZE => {
            if !ok {
                // Wrong address or login: retrying will not help.
                shared.fail(format!(
                    "The pool refused to authorize {:?}: {}\nCheck BTC_ADDRESS (and POOL) in your .env file.",
                    shared.config.username(),
                    describe(error, result)
                ));
                bail!("authorization refused");
            }
            state.authorized = true;
            shared.details().link = Link::Online;
            shared.log(
                Level::Good,
                format!("Authorized as {}", shared.config.username()),
            );

            // Ask for a difficulty this GPU can meet every few seconds. Pools
            // are free to ignore the request or clamp it to their minimum.
            let hashrate = shared.benchmark.load(Ordering::Relaxed) as f64;
            let suggested = (hashrate * SHARE_INTERVAL_SECS / 4_294_967_296.0).max(1.0);
            client.send(&json!({
                "id": ID_SUGGEST,
                "method": "mining.suggest_difficulty",
                "params": [(suggested * 100.0).round() / 100.0],
            }))?;
        }
        ID_SUGGEST => {} // optional extension; pools without it reply with an error
        _ => {
            let pending = client
                .pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(&id);
            let Some(share) = pending else { return Ok(()) };
            let latency = share.sent.elapsed().as_millis();
            if ok {
                shared.accepted.fetch_add(1, Ordering::Relaxed);
                shared.details().last_accepted = Some(Instant::now());
                shared.log(
                    Level::Good,
                    format!(
                        "Share accepted  ·  difficulty {}  ·  {latency} ms",
                        crate::ui::format::difficulty(share.difficulty)
                    ),
                );
                if share.is_block {
                    shared.log(
                        Level::Good,
                        "The pool accepted your BLOCK. Congratulations!",
                    );
                }
            } else {
                shared.rejected.fetch_add(1, Ordering::Relaxed);
                shared.log(
                    Level::Warn,
                    format!("Share rejected  ·  {}", describe(error, result)),
                );
            }
        }
    }
    Ok(())
}

fn notification(
    shared: &Shared,
    client: &Client,
    state: &mut Session,
    method: &str,
    params: &Value,
    id: Option<&Value>,
) -> Result<()> {
    match method {
        "mining.notify" => {
            let job = parse_notify(params, state).context("invalid mining.notify")?;
            let info = JobInfo {
                id: job.id.clone(),
                height: job.height(),
                network_difficulty: job.network_target().difficulty(),
                share_difficulty: job.share_difficulty,
                prev_hash: hex::encode(job.prev_hash.iter().rev().copied().collect::<Vec<u8>>()),
                received: Instant::now(),
            };
            let new_block = {
                let mut details = shared.details();
                let previous = details.job.replace(info.clone());
                previous.is_none_or(|p| p.prev_hash != info.prev_hash)
            };
            if new_block {
                let height = info.height.map_or("?".into(), |h| h.to_string());
                shared.log(
                    Level::Info,
                    format!(
                        "New block to mine  ·  height {height}  ·  network difficulty {}",
                        crate::ui::format::difficulty(info.network_difficulty)
                    ),
                );
            }
            shared.jobs.fetch_add(1, Ordering::Relaxed);
            shared.set_job(Some(Arc::new(job)));
        }
        "mining.set_difficulty" => {
            let difficulty = params
                .get(0)
                .and_then(Value::as_f64)
                .filter(|d| d.is_finite() && *d > 0.0)
                .context("invalid mining.set_difficulty")?;
            if difficulty != state.difficulty {
                shared.log(
                    Level::Info,
                    format!(
                        "Pool set share difficulty to {}",
                        crate::ui::format::difficulty(difficulty)
                    ),
                );
            }
            state.difficulty = difficulty;
        }
        "mining.set_extranonce" => {
            let extranonce1 = params
                .get(0)
                .and_then(Value::as_str)
                .and_then(|s| hex::decode(s).ok());
            let size = params.get(1).and_then(Value::as_u64);
            if let (Some(extranonce1), Some(size @ 1..=16)) = (extranonce1, size) {
                state.extranonce1 = extranonce1;
                state.extranonce2_size = size as usize;
            }
        }
        "mining.ping" => {
            if let Some(id) = id {
                client.send(&json!({"id": id, "result": "pong", "error": null}))?;
            }
        }
        "client.show_message" => {
            if let Some(text) = params.get(0).and_then(Value::as_str) {
                let text: String = text.chars().filter(|c| !c.is_control()).take(200).collect();
                shared.log(Level::Info, format!("Pool says: {text}"));
            }
        }
        "client.reconnect" => bail!("pool asked us to reconnect"),
        _ => {}
    }
    Ok(())
}

fn parse_notify(params: &Value, state: &Session) -> Option<Job> {
    if state.extranonce2_size == 0 {
        return None; // work before a successful subscribe is unusable
    }
    let text = |i: usize| params.get(i).and_then(Value::as_str);
    let word = |i: usize| u32::from_str_radix(text(i)?, 16).ok();

    // Stratum sends the previous hash as eight 32-bit words, each byte-swapped
    // relative to the order they appear in the block header.
    let mut prev_hash: [u8; 32] = hex::decode(text(1)?).ok()?.try_into().ok()?;
    for chunk in prev_hash.chunks_exact_mut(4) {
        chunk.reverse();
    }
    let merkle_branch = params
        .get(4)?
        .as_array()?
        .iter()
        .map(|v| hex::decode(v.as_str()?).ok()?.try_into().ok())
        .collect::<Option<Vec<[u8; 32]>>>()?;

    Some(Job {
        session: state.id,
        id: text(0)?.to_string(),
        prev_hash,
        coinbase1: hex::decode(text(2)?).ok()?,
        coinbase2: hex::decode(text(3)?).ok()?,
        merkle_branch,
        version: word(5)?,
        nbits: word(6)?,
        ntime: word(7)?,
        clean: params.get(8).and_then(Value::as_bool).unwrap_or(true),
        extranonce1: state.extranonce1.clone(),
        extranonce2_size: state.extranonce2_size,
        share_difficulty: state.difficulty,
    })
}

/// Human-readable reason from a Stratum error (`[code, "message", ...]`).
fn describe(error: Option<&Value>, result: &Value) -> String {
    match error {
        Some(Value::Array(fields)) => match (fields.first(), fields.get(1).and_then(Value::as_str))
        {
            (Some(code), Some(text)) => format!("{text} (code {code})"),
            _ => Value::Array(fields.clone()).to_string(),
        },
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None if result.as_bool() == Some(false) => "request denied".into(),
        None => format!("unexpected reply {result}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_parsing() {
        let state = Session {
            id: 7,
            extranonce1: vec![0xab, 0xcd],
            extranonce2_size: 4,
            difficulty: 2.0,
            authorized: true,
        };
        let params = json!([
            "job1",
            "0a8ce26f72b3f1b646a2a6c14ff763ae65831e939c085ae10019d66800000000",
            "01000000",
            "ffffffff",
            ["00".repeat(32)],
            "20000000",
            "1d00ffff",
            "495fab29",
            false
        ]);
        let job = parse_notify(&params, &state).unwrap();
        // The genesis hash, as it appears in the header of block 1.
        let mut display = job.prev_hash;
        display.reverse();
        assert_eq!(
            hex::encode(display),
            "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
        );
        assert_eq!(
            (job.version, job.nbits, job.ntime),
            (0x20000000, 0x1d00ffff, 0x495fab29)
        );
        assert_eq!(
            (job.session, job.clean, job.share_difficulty),
            (7, false, 2.0)
        );
        assert_eq!(job.merkle_branch.len(), 1);
        assert!(parse_notify(&json!(["job1", "zz"]), &state).is_none());
    }

    #[test]
    fn error_descriptions() {
        assert_eq!(
            describe(Some(&json!([21, "Job not found", null])), &Value::Null),
            "Job not found (code 21)"
        );
        assert_eq!(describe(Some(&json!("nope")), &Value::Null), "nope");
        assert_eq!(describe(None, &json!(false)), "request denied");
        assert_eq!(describe(None, &json!(7)), "unexpected reply 7");
    }
}
