//! Terminal dashboard (ratatui) and the plain-text headless mode.

pub mod format;

use crate::state::{JobInfo, Level, Link, LogLine, RateMeter, Shared};
use anyhow::Result;
use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Padding, Paragraph};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const ORANGE: Color = Color::Rgb(247, 147, 26);
const GREEN: Color = Color::Rgb(94, 201, 122);
const RED: Color = Color::Rgb(235, 90, 90);
const YELLOW: Color = Color::Rgb(240, 200, 80);
const DIM: Color = Color::DarkGray;

const TICK: Duration = Duration::from_millis(250);
const RATE_WINDOW: Duration = Duration::from_secs(4);
const HISTORY_SECONDS: usize = 600;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Starting,
    Connecting,
    Mining,
    Paused,
    Offline,
}

impl Status {
    fn label(self) -> (&'static str, Color) {
        match self {
            Status::Starting => ("STARTING GPU", YELLOW),
            Status::Connecting => ("CONNECTING", YELLOW),
            Status::Mining => ("MINING", GREEN),
            Status::Paused => ("PAUSED", YELLOW),
            Status::Offline => ("OFFLINE", RED),
        }
    }
}

/// Everything one frame needs, copied out of the shared state so drawing
/// never holds a lock (and so the layout can be tested with fixed data).
pub struct View {
    pub gpu: String,
    pub status: Status,
    pub uptime: Duration,
    pub hashrate: f64,
    pub average: f64,
    pub history: Vec<f64>,
    pub pool: String,
    pub address: String,
    pub worker: String,
    pub intensity: u32,
    pub job: Option<JobInfo>,
    pub jobs: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub candidates: u64,
    pub best_difficulty: f64,
    pub last_accepted: Option<Duration>,
    pub blocks_found: u64,
    pub total_hashes: u64,
    pub logs: Vec<LogLine>,
}

impl View {
    fn capture(shared: &Shared, hashrate: f64, history: &VecDeque<f64>) -> Self {
        let details = shared.details();
        let ready = shared.benchmark.load(Ordering::Relaxed) > 0;
        let status = if !ready {
            Status::Starting
        } else if shared.paused.load(Ordering::Relaxed) {
            Status::Paused
        } else {
            match (details.link, &details.job) {
                (Link::Online, Some(_)) => Status::Mining,
                (Link::Offline, _) => Status::Offline,
                _ => Status::Connecting,
            }
        };
        let uptime = shared.started.elapsed();
        let total_hashes = shared.hashes.load(Ordering::Relaxed);
        Self {
            gpu: details.gpu_name.clone(),
            status,
            uptime,
            hashrate,
            average: total_hashes as f64 / uptime.as_secs_f64().max(1.0),
            history: history.iter().copied().collect(),
            pool: shared.config.pool.clone(),
            address: shared.config.address.clone(),
            worker: shared.config.worker.clone(),
            intensity: shared.config.intensity,
            job: details.job.clone(),
            jobs: shared.jobs.load(Ordering::Relaxed),
            accepted: shared.accepted.load(Ordering::Relaxed),
            rejected: shared.rejected.load(Ordering::Relaxed),
            candidates: shared.candidates.load(Ordering::Relaxed),
            best_difficulty: details.best_difficulty,
            last_accepted: details.last_accepted.map(|at| at.elapsed()),
            blocks_found: details.blocks_found,
            total_hashes,
            logs: details.logs.iter().cloned().collect(),
        }
    }
}

/// Runs the full-screen dashboard until the user quits or the miner stops.
pub fn run(shared: &Shared) -> Result<()> {
    let mut terminal = ratatui::init();
    let result = (|| -> Result<()> {
        let mut meter = RateMeter::new(RATE_WINDOW);
        let mut history: VecDeque<f64> = VecDeque::with_capacity(HISTORY_SECONDS);
        let mut last_history = Instant::now();
        while !shared.quitting() {
            let hashrate = meter.sample(shared.hashes.load(Ordering::Relaxed));
            if last_history.elapsed() >= Duration::from_secs(1) {
                last_history = Instant::now();
                if history.len() == HISTORY_SECONDS {
                    history.pop_front();
                }
                history.push_back(hashrate);
            }
            let view = View::capture(shared, hashrate, &history);
            terminal.draw(|frame| draw(frame, &view))?;

            if event::poll(TICK)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                let ctrl_c =
                    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    _ if ctrl_c => break,
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('p') | KeyCode::Char(' ') => {
                        shared.paused.fetch_xor(true, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    ratatui::restore();
    result
}

/// Runs without a dashboard: log lines plus a periodic one-line summary.
pub fn run_headless(shared: &Shared) {
    const SUMMARY_EVERY: Duration = Duration::from_secs(30);
    let mut meter = RateMeter::new(Duration::from_secs(10));
    let mut last_summary = Instant::now();
    while !shared.quitting() {
        std::thread::sleep(TICK);
        let hashrate = meter.sample(shared.hashes.load(Ordering::Relaxed));
        if last_summary.elapsed() >= SUMMARY_EVERY {
            last_summary = Instant::now();
            let best = shared.details().best_difficulty;
            shared.log(
                Level::Info,
                format!(
                    "{}  ·  accepted {}  ·  rejected {}  ·  best share {}",
                    format::hashrate(hashrate),
                    shared.accepted.load(Ordering::Relaxed),
                    shared.rejected.load(Ordering::Relaxed),
                    format::difficulty(best),
                ),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

pub fn draw(frame: &mut Frame, view: &View) {
    let area = frame.area();
    if area.width < 50 || area.height < 12 {
        let text = format!(
            "{}  (enlarge the terminal for the dashboard)",
            format::hashrate(view.hashrate)
        );
        frame.render_widget(Paragraph::new(text).fg(ORANGE), area);
        return;
    }

    // Wide terminals get the four panels side by side; narrow ones a 2x2 grid.
    let wide = area.width >= 110;
    let panels_height = if wide { 7 } else { 14 };
    // The large hashrate display needs room; fall back to a single line.
    let hero_height = if area.height >= 9 + panels_height + 7 {
        9
    } else {
        3
    };

    let [header, hero, panels, log, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(hero_height),
        Constraint::Length(panels_height),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_header(frame, header, view);
    if hero_height > 3 {
        let [rate, chart] =
            Layout::horizontal([Constraint::Length(48), Constraint::Min(20)]).areas(hero);
        draw_hashrate(frame, rate, view);
        draw_history(frame, chart, view);
    } else {
        draw_hashrate_compact(frame, hero, view);
    }
    draw_panels(frame, panels, view, wide);
    draw_log(frame, log, view);
    draw_footer(frame, footer, view);
}

fn panel(title: &str) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(DIM))
        .title(Span::styled(
            format!(" {title} "),
            Style::new().fg(ORANGE).bold(),
        ))
        .padding(Padding::horizontal(1))
}

fn draw_header(frame: &mut Frame, area: Rect, view: &View) {
    let left = Line::from(vec![
        Span::styled(" ₿ ", Style::new().fg(Color::Black).bg(ORANGE).bold()),
        Span::styled(
            format!(" {} ", env!("CARGO_PKG_NAME")),
            Style::new().fg(ORANGE).bold(),
        ),
        Span::styled(
            concat!("v", env!("CARGO_PKG_VERSION")),
            Style::new().fg(DIM),
        ),
        Span::raw("   "),
        Span::raw(view.gpu.clone()),
    ]);
    let (label, color) = view.status.label();
    let mut right = Vec::new();
    if view.blocks_found > 0 {
        right.push(Span::styled(
            format!(" ★ BLOCK FOUND ×{} ★ ", view.blocks_found),
            Style::new().fg(Color::Black).bg(GREEN).bold(),
        ));
        right.push(Span::raw("  "));
    }
    right.push(Span::styled(
        format!("● {label}"),
        Style::new().fg(color).bold(),
    ));
    right.push(Span::styled(
        format!("   up {} ", format::clock(view.uptime)),
        Style::new().fg(DIM),
    ));

    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(
        Paragraph::new(Line::from(right)).alignment(Alignment::Right),
        area,
    );
}

/// 3x5 pixel digits, drawn two cells wide per pixel so they look square.
fn glyph(ch: char) -> [&'static str; 5] {
    match ch {
        '0' => ["###", "# #", "# #", "# #", "###"],
        '1' => [" # ", "## ", " # ", " # ", "###"],
        '2' => ["###", "  #", "###", "#  ", "###"],
        '3' => ["###", "  #", "###", "  #", "###"],
        '4' => ["# #", "# #", "###", "  #", "  #"],
        '5' => ["###", "#  ", "###", "  #", "###"],
        '6' => ["###", "#  ", "###", "# #", "###"],
        '7' => ["###", "  #", "  #", "  #", "  #"],
        '8' => ["###", "# #", "###", "# #", "###"],
        '9' => ["###", "# #", "###", "  #", "###"],
        '.' => [" ", " ", " ", " ", "#"],
        _ => ["   ", "   ", "   ", "   ", "   "],
    }
}

fn big_text(text: &str) -> [String; 5] {
    let mut rows: [String; 5] = Default::default();
    for (i, ch) in text.chars().enumerate() {
        for (row, pattern) in rows.iter_mut().zip(glyph(ch)) {
            if i > 0 {
                row.push_str("  ");
            }
            for pixel in pattern.chars() {
                row.push_str(if pixel == '#' { "██" } else { "  " });
            }
        }
    }
    rows
}

fn rate_color(view: &View) -> Color {
    if view.status == Status::Mining {
        ORANGE
    } else {
        DIM
    }
}

fn draw_hashrate(frame: &mut Frame, area: Rect, view: &View) {
    let (number, unit) = format::hashrate_parts(view.hashrate);
    let color = rate_color(view);
    let rows = big_text(&number);
    let mut lines = vec![Line::raw("")];
    for (i, row) in rows.into_iter().enumerate() {
        let mut spans = vec![Span::styled(row, Style::new().fg(color))];
        if i == 4 {
            spans.push(Span::styled(
                format!("  {unit}"),
                Style::new().fg(color).bold(),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(vec![
        Span::styled("average ", Style::new().fg(DIM)),
        Span::raw(format::hashrate(view.average)),
        Span::styled("  ·  intensity ", Style::new().fg(DIM)),
        Span::raw(format!("{}%", view.intensity)),
    ]));
    frame.render_widget(Paragraph::new(lines).block(panel("Hashrate")), area);
}

fn draw_hashrate_compact(frame: &mut Frame, area: Rect, view: &View) {
    let line = Line::from(vec![
        Span::styled(
            format::hashrate(view.hashrate),
            Style::new().fg(rate_color(view)).bold(),
        ),
        Span::styled("   average ", Style::new().fg(DIM)),
        Span::raw(format::hashrate(view.average)),
        Span::styled("   intensity ", Style::new().fg(DIM)),
        Span::raw(format!("{}%", view.intensity)),
    ]);
    frame.render_widget(Paragraph::new(line).block(panel("Hashrate")), area);
}

fn draw_history(frame: &mut Frame, area: Rect, view: &View) {
    // Braille gives two samples per cell; show as many seconds as fit.
    let capacity = (area.width.saturating_sub(12) as usize * 2).max(2);
    let start = view.history.len().saturating_sub(capacity);
    let points: Vec<(f64, f64)> = view.history[start..]
        .iter()
        .enumerate()
        .map(|(i, &rate)| (i as f64, rate))
        .collect();
    let peak = points.iter().map(|p| p.1).fold(1.0, f64::max);
    let top = peak * 1.15;

    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(ORANGE))
        .data(&points);
    let dim = Style::new().fg(DIM);
    let chart = Chart::new(vec![dataset])
        .block(panel("History").padding(Padding::ZERO))
        .x_axis(
            Axis::default()
                .bounds([0.0, (capacity - 1) as f64])
                .style(dim),
        )
        .y_axis(Axis::default().bounds([0.0, top]).style(dim).labels([
            Line::raw("0"),
            Line::raw(format::hashrate(top / 2.0)),
            Line::raw(format::hashrate(top)),
        ]));
    frame.render_widget(chart, area);
}

/// A `label  value` row.
fn row<'a>(label: &'a str, value: impl Into<Span<'a>>) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{label:<11}"), Style::new().fg(DIM)),
        value.into(),
    ])
}

fn draw_panels(frame: &mut Frame, area: Rect, view: &View, wide: bool) {
    let cells: [Rect; 4] = if wide {
        Layout::horizontal([Constraint::Ratio(1, 4); 4]).areas(area)
    } else {
        let [top, bottom] = Layout::vertical([Constraint::Ratio(1, 2); 2]).areas(area);
        let [a, b] = Layout::horizontal([Constraint::Ratio(1, 2); 2]).areas(top);
        let [c, d] = Layout::horizontal([Constraint::Ratio(1, 2); 2]).areas(bottom);
        [a, b, c, d]
    };
    let dash = || Span::styled("—", Style::new().fg(DIM));
    let job = view.job.as_ref();

    // Pool
    let (label, color) = view.status.label();
    let pool = vec![
        row("Server", view.pool.as_str()),
        row("Status", Span::styled(label, Style::new().fg(color))),
        row("Address", format::shorten(&view.address, 10, 6)),
        row(
            "Worker",
            if view.worker.is_empty() {
                dash()
            } else {
                view.worker.as_str().into()
            },
        ),
        row(
            "Share diff",
            job.map_or(dash(), |j| format::difficulty(j.share_difficulty).into()),
        ),
    ];
    frame.render_widget(Paragraph::new(pool).block(panel("Pool")), cells[0]);

    // Block being mined
    let block = vec![
        row(
            "Height",
            job.and_then(|j| j.height)
                .map_or(dash(), |h| format::count(h).into()),
        ),
        row(
            "Difficulty",
            job.map_or(dash(), |j| format::difficulty(j.network_difficulty).into()),
        ),
        row(
            "Previous",
            job.map_or(dash(), |j| {
                format::shorten(j.prev_hash.trim_start_matches('0'), 8, 6).into()
            }),
        ),
        row(
            "Job",
            job.map_or(dash(), |j| format::shorten(&j.id, 10, 4).into()),
        ),
        row(
            "Updated",
            job.map_or(dash(), |j| {
                format!(
                    "{}  ({} jobs)",
                    format::ago(j.received.elapsed()),
                    view.jobs
                )
                .into()
            }),
        ),
    ];
    frame.render_widget(Paragraph::new(block).block(panel("Block")), cells[1]);

    // Shares
    let rejected_style = if view.rejected > 0 {
        Style::new().fg(RED)
    } else {
        Style::new()
    };
    let shares = vec![
        row(
            "Accepted",
            Span::styled(format::count(view.accepted), Style::new().fg(GREEN).bold()),
        ),
        row(
            "Rejected",
            Span::styled(format::count(view.rejected), rejected_style),
        ),
        row(
            "Last share",
            view.last_accepted.map_or(dash(), |d| format::ago(d).into()),
        ),
        row(
            "Best diff",
            if view.best_difficulty > 0.0 {
                Span::styled(
                    format::difficulty(view.best_difficulty),
                    Style::new().fg(ORANGE).bold(),
                )
            } else {
                dash()
            },
        ),
        row(
            "Near hits",
            format!("{}  (diff ≥ 1)", format::count(view.candidates)),
        ),
    ];
    frame.render_widget(Paragraph::new(shares).block(panel("Shares")), cells[2]);

    // Lottery: how likely this machine is to solve a block.
    let rate = if view.hashrate > 0.0 {
        view.hashrate
    } else {
        view.average
    };
    let odds = job.filter(|_| rate > 0.0).map(|j| {
        // A block takes on average difficulty * 2^32 hashes.
        let seconds = j.network_difficulty * 4_294_967_296.0 / rate;
        (seconds, j.network_difficulty * 4_294_967_296.0 / 600.0)
    });
    let found_style = if view.blocks_found > 0 {
        Style::new().fg(Color::Black).bg(GREEN).bold()
    } else {
        Style::new()
    };
    let lottery = vec![
        row(
            "Network",
            odds.map_or(dash(), |(_, network)| format::hashrate(network).into()),
        ),
        row(
            "Per block",
            odds.map_or(dash(), |(s, _)| {
                format!("1 in {}", format::magnitude(s / 600.0)).into()
            }),
        ),
        row(
            "Per year",
            odds.map_or(dash(), |(s, _)| {
                format!("1 in {}", format::magnitude(s / 31_557_600.0)).into()
            }),
        ),
        row(
            "Expected",
            odds.map_or(dash(), |(s, _)| format::span(s).into()),
        ),
        row(
            "Found",
            Span::styled(format!(" {} ", view.blocks_found), found_style),
        ),
    ];
    frame.render_widget(Paragraph::new(lottery).block(panel("Lottery")), cells[3]);
}

fn draw_log(frame: &mut Frame, area: Rect, view: &View) {
    let visible = area.height.saturating_sub(2) as usize;
    let start = view.logs.len().saturating_sub(visible);
    let lines: Vec<Line> = view.logs[start..]
        .iter()
        .map(|entry| {
            let (mark, color) = match entry.level {
                Level::Info => ("·", DIM),
                Level::Good => ("✓", GREEN),
                Level::Warn => ("!", YELLOW),
                Level::Error => ("✗", RED),
            };
            let text_style = match entry.level {
                Level::Info => Style::new(),
                _ => Style::new().fg(color),
            };
            Line::from(vec![
                Span::styled(entry.time.as_str(), Style::new().fg(DIM)),
                Span::styled(
                    format!(" {mark} "),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(entry.text.as_str(), text_style),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines).block(panel("Log")), area);
}

fn draw_footer(frame: &mut Frame, area: Rect, view: &View) {
    let key = |k: &'static str| Span::styled(k, Style::new().fg(ORANGE).bold());
    let dim = |t: &'static str| Span::styled(t, Style::new().fg(DIM));
    let pause = if view.status == Status::Paused {
        " resume   "
    } else {
        " pause   "
    };
    let left = Line::from(vec![
        Span::raw(" "),
        key("q"),
        dim(" quit   "),
        key("p"),
        dim(pause),
    ]);
    let right = Line::from(Span::styled(
        format!("{} hashes computed ", format::count(view.total_hashes)),
        Style::new().fg(DIM),
    ));
    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(Paragraph::new(right).alignment(Alignment::Right), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn sample_view() -> View {
        View {
            gpu: "Apple M1 Max".into(),
            status: Status::Mining,
            uptime: Duration::from_secs(3725),
            hashrate: 923.4e6,
            average: 918.2e6,
            history: (0..300).map(|i| 900e6 + (i % 7) as f64 * 5e6).collect(),
            pool: "solo.ckpool.org:3333".into(),
            address: "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh".into(),
            worker: "macbook".into(),
            intensity: 100,
            job: Some(JobInfo {
                id: "6a1f00000a2c".into(),
                height: Some(923_456),
                network_difficulty: 1.5e14,
                share_difficulty: 4.3,
                prev_hash: format!("{}{}", "0".repeat(20), "1a2b3c4d5e6f".repeat(4)),
                received: Instant::now(),
            }),
            jobs: 42,
            accepted: 17,
            rejected: 1,
            candidates: 812,
            best_difficulty: 48_211.0,
            last_accepted: Some(Duration::from_secs(12)),
            blocks_found: 0,
            total_hashes: 3_439_665_000_000,
            logs: (0..12)
                .map(|i| LogLine {
                    time: format!("12:00:{i:02}"),
                    level: [Level::Info, Level::Good, Level::Warn, Level::Error][i % 4],
                    text: format!("Log line number {i}"),
                })
                .collect(),
        }
    }

    fn render(width: u16, height: u16, view: &View) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, view)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..height {
            for x in 0..width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn wide_layout_shows_everything() {
        let screen = render(140, 40, &sample_view());
        println!("{screen}");
        for needle in [
            "MINING",
            "MH/s",
            "Pool",
            "Block",
            "Shares",
            "Lottery",
            "923,456",
            "Log line number 11",
        ] {
            assert!(screen.contains(needle), "missing {needle:?}");
        }
    }

    #[test]
    fn standard_terminal_layout() {
        let screen = render(80, 24, &sample_view());
        println!("{screen}");
        for needle in ["923.4 MH/s", "Accepted", "Expected", "Log line number 11"] {
            assert!(screen.contains(needle), "missing {needle:?}");
        }
    }

    #[test]
    fn survives_any_size() {
        let mut view = sample_view();
        view.job = None;
        view.history.clear();
        view.status = Status::Connecting;
        for (width, height) in [
            (1, 1),
            (20, 5),
            (50, 12),
            (60, 30),
            (109, 31),
            (110, 23),
            (300, 100),
        ] {
            render(width, height, &view);
        }
    }
}
