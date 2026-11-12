//! Configuration, read from a `.env` file (or the process environment).

use anyhow::{Context, Result, bail};

pub const DEFAULT_POOL: &str = "solo.ckpool.org:3333";

#[derive(Clone, Debug)]
pub struct Config {
    /// Payout address. For solo pools this is also the login.
    pub address: String,
    /// Optional rig name, appended to the address as `address.worker`.
    pub worker: String,
    /// Pool endpoint as `host:port`.
    pub pool: String,
    pub password: String,
    /// Percentage of time the GPU spends hashing (1-100).
    pub intensity: u32,
}

impl Config {
    pub fn load() -> Result<Self> {
        // A missing .env is fine as long as the variables come from somewhere.
        if let Err(error) = dotenvy::dotenv()
            && !error.not_found()
        {
            bail!("could not parse .env: {error}");
        }
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |key: &str| {
            get(key)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };

        let address = get("BTC_ADDRESS").context(
            "BTC_ADDRESS is not set.\n\n\
             Create a .env file next to where you run the miner:\n\n    \
             cp .env.example .env\n\n\
             and set BTC_ADDRESS to the Bitcoin address that should receive the reward.",
        )?;
        if address.contains(char::is_whitespace) || address.starts_with("your_") {
            bail!("BTC_ADDRESS does not look like a Bitcoin address: {address:?}");
        }

        let intensity = match get("INTENSITY") {
            None => 100,
            Some(raw) => match raw.trim_end_matches('%').parse::<u32>() {
                Ok(value @ 1..=100) => value,
                _ => bail!("INTENSITY must be a whole number from 1 to 100, got {raw:?}"),
            },
        };

        Ok(Self {
            address,
            worker: get("WORKER").unwrap_or_default(),
            pool: parse_pool(&get("POOL").unwrap_or_else(|| DEFAULT_POOL.into()))?,
            password: get("POOL_PASSWORD").unwrap_or_else(|| "x".into()),
            intensity,
        })
    }

    /// The Stratum login: `address` or `address.worker`.
    pub fn username(&self) -> String {
        if self.worker.is_empty() {
            self.address.clone()
        } else {
            format!("{}.{}", self.address, self.worker)
        }
    }
}

/// Normalises a pool setting to `host:port`.
fn parse_pool(raw: &str) -> Result<String> {
    if raw.starts_with("stratum+ssl://") || raw.starts_with("stratum+tls://") {
        bail!(
            "POOL uses TLS ({raw}), which is not supported - use the pool's stratum+tcp endpoint"
        );
    }
    let host = raw
        .trim_start_matches("stratum+tcp://")
        .trim_start_matches("tcp://")
        .trim_end_matches('/');
    if host.is_empty() || host.contains('/') || host.contains(char::is_whitespace) {
        bail!("POOL must look like host:port, got {raw:?}");
    }
    match host.rsplit_once(':') {
        Some((name, port)) if !name.is_empty() && port.parse::<u16>().is_ok() => Ok(host.into()),
        Some(_) => bail!("POOL must look like host:port, got {raw:?}"),
        None => Ok(format!("{host}:3333")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(vars: &[(&str, &str)]) -> Result<Config> {
        Config::from_lookup(|key| {
            vars.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        })
    }

    #[test]
    fn only_address_is_required() {
        let c = config(&[("BTC_ADDRESS", "bc1qexample")]).unwrap();
        assert_eq!(c.pool, DEFAULT_POOL);
        assert_eq!(c.username(), "bc1qexample");
        assert_eq!(c.password, "x");
        assert_eq!(c.intensity, 100);
        assert!(config(&[]).is_err());
        assert!(config(&[("BTC_ADDRESS", "your_bitcoin_address_here")]).is_err());
    }

    #[test]
    fn optional_settings() {
        let c = config(&[
            ("BTC_ADDRESS", "bc1qexample"),
            ("WORKER", "macbook"),
            ("POOL", "stratum+tcp://public-pool.io:21496"),
            ("INTENSITY", "60%"),
        ])
        .unwrap();
        assert_eq!(c.username(), "bc1qexample.macbook");
        assert_eq!(c.pool, "public-pool.io:21496");
        assert_eq!(c.intensity, 60);
        assert!(config(&[("BTC_ADDRESS", "a"), ("INTENSITY", "0")]).is_err());
        assert!(config(&[("BTC_ADDRESS", "a"), ("INTENSITY", "fast")]).is_err());
    }

    #[test]
    fn pool_parsing() {
        assert_eq!(parse_pool("example.com").unwrap(), "example.com:3333");
        assert_eq!(
            parse_pool("tcp://example.com:4444/").unwrap(),
            "example.com:4444"
        );
        assert!(parse_pool("stratum+ssl://example.com:443").is_err());
        assert!(parse_pool("example.com:port").is_err());
    }
}
