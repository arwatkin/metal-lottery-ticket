//! Human-friendly number formatting.

use std::time::Duration;

/// Splits a hashrate into a number and a unit, e.g. `("923.4", "MH/s")`.
pub fn hashrate_parts(rate: f64) -> (String, &'static str) {
    const UNITS: [&str; 8] = [
        "H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s", "ZH/s",
    ];
    let mut value = rate.max(0.0);
    let mut unit = 0;
    while value >= 999.95 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    let digits = if unit == 0 { 0 } else { 1 };
    (format!("{value:.digits$}"), UNITS[unit])
}

pub fn hashrate(rate: f64) -> String {
    let (value, unit) = hashrate_parts(rate);
    format!("{value} {unit}")
}

/// Difficulty with an SI suffix: `3.20`, `125.4K`, `53.91T`.
pub fn difficulty(value: f64) -> String {
    const SUFFIXES: [&str; 7] = ["", "K", "M", "G", "T", "P", "E"];
    if !value.is_finite() {
        return "∞".into();
    }
    let mut scaled = value.max(0.0);
    let mut suffix = 0;
    while scaled >= 999.995 && suffix < SUFFIXES.len() - 1 {
        scaled /= 1000.0;
        suffix += 1;
    }
    format!("{scaled:.2}{}", SUFFIXES[suffix])
}

/// Integer with thousands separators: `1,234,567`.
pub fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A large quantity in words: `12.3 billion`.
pub fn magnitude(value: f64) -> String {
    const NAMES: [&str; 6] = [
        "",
        " thousand",
        " million",
        " billion",
        " trillion",
        " quadrillion",
    ];
    if !value.is_finite() {
        return "∞".into();
    }
    let mut scaled = value.max(0.0);
    let mut name = 0;
    while scaled >= 999.95 && name < NAMES.len() - 1 {
        scaled /= 1000.0;
        name += 1;
    }
    if name == 0 && scaled < 100.0 {
        format!("{scaled:.1}")
    } else if name == NAMES.len() - 1 && scaled >= 999.95 {
        format!("{scaled:.1e}{}", NAMES[name])
    } else {
        format!("{scaled:.1}{}", NAMES[name])
    }
}

/// Elapsed time as `HH:MM:SS`, with a day count once it is needed.
pub fn clock(duration: Duration) -> String {
    let total = duration.as_secs();
    let (days, hours, minutes, seconds) = (
        total / 86_400,
        total / 3600 % 24,
        total / 60 % 60,
        total % 60,
    );
    if days > 0 {
        format!("{days}d {hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    }
}

/// A short "time since" label: `8s ago`, `3m ago`, `2h ago`.
pub fn ago(duration: Duration) -> String {
    match duration.as_secs() {
        s @ 0..60 => format!("{s}s ago"),
        s @ 60..3600 => format!("{}m ago", s / 60),
        s => format!("{}h ago", s / 3600),
    }
}

/// An expected waiting time in the largest sensible unit.
pub fn span(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "forever".into();
    }
    const YEAR: f64 = 365.25 * 86_400.0;
    if seconds >= YEAR {
        format!("{} years", magnitude(seconds / YEAR))
    } else if seconds >= 86_400.0 {
        format!("{:.1} days", seconds / 86_400.0)
    } else if seconds >= 3600.0 {
        format!("{:.1} hours", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} minutes", seconds / 60.0)
    } else {
        format!("{seconds:.0} seconds")
    }
}

/// Shortens long identifiers such as addresses: `bc1qxy2k…x0wlh`.
pub fn shorten(text: &str, head: usize, tail: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= head + tail + 1 {
        return text.to_string();
    }
    let start: String = chars[..head].iter().collect();
    let end: String = chars[chars.len() - tail..].iter().collect();
    format!("{start}…{end}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashrates() {
        assert_eq!(hashrate(0.0), "0 H/s");
        assert_eq!(hashrate(950.0), "950 H/s");
        assert_eq!(hashrate(923_400_000.0), "923.4 MH/s");
        assert_eq!(hashrate(999_990_000.0), "1.0 GH/s");
        assert_eq!(hashrate(8.1e20), "810.0 EH/s");
    }

    #[test]
    fn difficulties() {
        assert_eq!(difficulty(1.0), "1.00");
        assert_eq!(difficulty(125_432.0), "125.43K");
        assert_eq!(difficulty(53.91e12), "53.91T");
    }

    #[test]
    fn counts_and_magnitudes() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1_234_567), "1,234,567");
        assert_eq!(magnitude(42.0), "42.0");
        assert_eq!(magnitude(12_300_000_000.0), "12.3 billion");
        assert_eq!(magnitude(5.0e20), "5.0e5 quadrillion");
    }

    #[test]
    fn times() {
        assert_eq!(clock(Duration::from_secs(3_725)), "01:02:05");
        assert_eq!(clock(Duration::from_secs(90_000)), "1d 01:00:00");
        assert_eq!(ago(Duration::from_secs(8)), "8s ago");
        assert_eq!(ago(Duration::from_secs(200)), "3m ago");
        assert_eq!(span(30.0), "30 seconds");
        assert_eq!(span(86_400.0 * 3.0), "3.0 days");
        assert_eq!(span(365.25 * 86_400.0 * 2.5e6), "2.5 million years");
    }

    #[test]
    fn shortening() {
        assert_eq!(shorten("short", 4, 4), "short");
        assert_eq!(
            shorten("bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", 8, 5),
            "bc1qxy2k…x0wlh"
        );
    }
}
