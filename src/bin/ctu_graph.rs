use std::collections::BTreeMap;
use std::io::Write;

use clap::Parser;
use is_terminal::IsTerminal;
use terminal_size::{Width, terminal_size};

use ctu::{DeduplicatedRecords, PeriodAgg, fmt_cost, fmt_num, get_cost, get_log_dirs, find_jsonl_files};

#[derive(Parser)]
#[command(name = "ctu-graph", version = "1.0.0", about = "Claude token usage bar chart")]
struct Cli {
    /// Hourly drill-down for YYYY-MM-DD
    #[arg(short = 'd', long = "day")]
    day: Option<String>,

    /// Filter from date (daily mode)
    #[arg(short = 's', long)]
    since: Option<String>,

    /// Filter until date (daily mode)
    #[arg(short = 'u', long)]
    until: Option<String>,

    /// Show last N days (0 = all time)
    #[arg(short = 'n', long, default_value = "30")]
    days: usize,

    /// Two sub-bars per row
    #[arg(long)]
    split: bool,

    /// Disable ANSI colors
    #[arg(long = "no-color")]
    no_color: bool,
}

// ── ANSI helpers ─────────────────────────────────────────────────────────────

struct Colors {
    reset: &'static str,
    label: &'static str,
    bar_in:  &'static str,
    bar_out: &'static str,
    dim:  &'static str,
    cost: &'static str,
    hdr:  &'static str,
}

const COLORS_ON: Colors = Colors {
    reset:   "\x1b[0m",
    label:   "\x1b[1;37m",
    bar_in:  "\x1b[34m",
    bar_out: "\x1b[32m",
    dim:     "\x1b[2;37m",
    cost:    "\x1b[33m",
    hdr:     "\x1b[1;36m",
};

const COLORS_OFF: Colors = Colors {
    reset: "", label: "", bar_in: "", bar_out: "", dim: "", cost: "", hdr: "",
};

// ── Scale helpers ─────────────────────────────────────────────────────────────

fn nice_scale(raw: f64) -> u64 {
    if raw <= 0.0 { return 1; }
    let mag = 10f64.powf(raw.log10().floor());
    let scaled = raw / mag;
    let factor = if scaled <= 1.0 { 1.0 }
                 else if scaled <= 2.0 { 2.0 }
                 else if scaled <= 5.0 { 5.0 }
                 else { 10.0 };
    (factor * mag).ceil() as u64
}

fn render_bar(val: u64, scale: u64, bar_area: usize) -> String {
    let filled = ((val as f64 / scale as f64) as usize).min(bar_area);
    let mut bar = String::with_capacity(bar_area * 3);
    for _ in 0..filled       { bar.push('█'); }
    for _ in filled..bar_area { bar.push('░'); }
    bar
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    let dirs = get_log_dirs();
    if dirs.is_empty() {
        eprintln!("No Claude log directories found.");
        std::process::exit(1);
    }
    let files = find_jsonl_files(&dirs);
    if files.is_empty() {
        eprintln!("No JSONL files found.");
        std::process::exit(1);
    }

    let hourly_mode = cli.day.is_some();
    let day_str = cli.day.clone().unwrap_or_default();

    // Aggregate
    let mut periods: BTreeMap<String, PeriodAgg> = BTreeMap::new();
    let mut grand = PeriodAgg::default();

    let data = DeduplicatedRecords::collect(&files);
    for rec in &data.records {
        let ts = &rec.timestamp;
        if ts.len() < 13 { continue; }

        let period: &str;
        if hourly_mode {
            if !ts.starts_with(&day_str) { continue; }
            period = &ts[11..13];
        } else {
            let date = &ts[..10];
            if let Some(ref s) = cli.since { if date < s.as_str() { continue; } }
            if let Some(ref u) = cli.until { if date > u.as_str() { continue; } }
            period = &ts[..10];
        }

        let cost = get_cost(&rec.model, rec.input, rec.output, rec.cache_create, rec.cache_read);

        let agg = periods.entry(period.to_string()).or_default();
        agg.input        += rec.input;
        agg.output       += rec.output;
        agg.cache_create += rec.cache_create;
        agg.cache_read   += rec.cache_read;
        agg.cost         += cost;

        grand.input        += rec.input;
        grand.output       += rec.output;
        grand.cache_create += rec.cache_create;
        grand.cache_read   += rec.cache_read;
        grand.cost         += cost;
    }

    // Gap-fill hours 00-23
    if hourly_mode {
        for h in 0..24u32 {
            periods.entry(format!("{:02}", h)).or_default();
        }
    }

    // Determine visible slice
    let all_keys: Vec<String> = periods.keys().cloned().collect();
    let start = if !hourly_mode && cli.days > 0 && all_keys.len() > cli.days {
        all_keys.len() - cli.days
    } else { 0 };
    let visible: Vec<&String> = all_keys[start..].iter().collect();

    // Layout
    let label_w: usize = if hourly_mode { 2 } else { 10 };
    let stats_w: usize = 33;

    let term_w = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| terminal_size().map(|(Width(w), _)| w as usize))
        .unwrap_or(80);

    let bar_area = (term_w.saturating_sub(label_w + 1 + stats_w)).clamp(10, 60);

    // Color detection
    let use_color = !cli.no_color
        && std::env::var("NO_COLOR").map(|v| v.is_empty()).unwrap_or(true)
        && std::io::stdout().is_terminal();
    let c = if use_color { &COLORS_ON } else { &COLORS_OFF };

    // Scale: max of inc (and out in split mode)
    let max_val = visible.iter().map(|p| {
        let agg = &periods[*p];
        let inc = agg.input + agg.cache_create + agg.cache_read;
        if cli.split { inc.max(agg.output) } else { inc }
    }).max().unwrap_or(0);

    let scale = nice_scale(max_val as f64 / bar_area as f64).max(1);

    // Header
    let hdr_txt = if hourly_mode {
        format!("Hourly breakdown · {}", day_str)
    } else if cli.days == 0 {
        "Daily token usage · all time".to_string()
    } else {
        format!("Daily token usage · last {} days", cli.days)
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}{} · scale: █ = {} tokens{}\n",
        c.hdr, hdr_txt, fmt_num(scale), c.reset).unwrap();

    // Data rows
    for p in &visible {
        let agg = &periods[*p];
        let inc  = agg.input + agg.cache_create + agg.cache_read;
        let cost = agg.cost;

        if cli.split {
            let bar1 = render_bar(inc, scale, bar_area);
            let bar2 = render_bar(agg.output, scale, bar_area);
            writeln!(out, "{}{:<label_w$}{}  {}{}{} {:>6} in+cache",
                c.label, p, c.reset,
                c.bar_in, bar1, c.reset,
                fmt_num(inc),
                label_w = label_w).unwrap();
            writeln!(out, "{:<label_w$}  {}{}{} {:>6} output  {}{:>7}{}",
                "", c.bar_out, bar2, c.reset,
                fmt_num(agg.output),
                c.cost, fmt_cost(cost), c.reset,
                label_w = label_w + 1).unwrap();
        } else {
            let bar = render_bar(inc, scale, bar_area);
            writeln!(out, "{}{:<label_w$}{}  {}{}{} {}{:>6} in+c {:>6} out{}  {}{:>7}{}",
                c.label, p, c.reset,
                c.bar_in, bar, c.reset,
                c.dim, fmt_num(inc), fmt_num(agg.output), c.reset,
                c.cost, fmt_cost(cost), c.reset,
                label_w = label_w).unwrap();
        }
    }

    // Separator
    let sep_len = label_w + 1 + bar_area + stats_w;
    writeln!(out, "{}", "─".repeat(sep_len)).unwrap();

    // Total row
    let t_inc = grand.input + grand.cache_create + grand.cache_read;
    if cli.split {
        writeln!(out, "{:<w$}  {}{:>6} in+cache{}",
            "TOTAL", c.dim, fmt_num(t_inc), c.reset,
            w = label_w + 1 + bar_area).unwrap();
        writeln!(out, "{:<w$}  {}{:>6} output{}  {}{:>7}{}",
            "", c.dim, fmt_num(grand.output), c.reset,
            c.cost, fmt_cost(grand.cost), c.reset,
            w = label_w + 1 + bar_area).unwrap();
    } else {
        writeln!(out, "{:<w$}  {}{:>6} in+c {:>6} out{}  {}{:>7}{}",
            "TOTAL",
            c.dim, fmt_num(t_inc), fmt_num(grand.output), c.reset,
            c.cost, fmt_cost(grand.cost), c.reset,
            w = label_w + 1 + bar_area).unwrap();
    }
}
