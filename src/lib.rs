use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use serde::Deserialize;

// ── Public data types ────────────────────────────────────────────────────────

pub struct RawRecord {
    pub timestamp:    String,
    pub req_id:       String,
    pub msg_id:       String,
    pub model:        String,
    pub input:        u64,
    pub output:       u64,
    pub cache_create: u64,
    pub cache_read:   u64,
}

#[derive(Default, Clone)]
pub struct PeriodAgg {
    pub input:        u64,
    pub output:       u64,
    pub cache_create: u64,
    pub cache_read:   u64,
    pub cost:         f64,
}

// ── JSONL deserialization ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct JsonLine {
    #[serde(rename = "type")]
    kind:       String,
    timestamp:  String,
    #[serde(rename = "requestId", default)]
    request_id: Option<String>,
    message:    Option<JsonMessage>,
}

#[derive(Deserialize)]
struct JsonMessage {
    id:    Option<String>,
    model: Option<String>,
    usage: Option<JsonUsage>,
}

#[derive(Deserialize)]
struct JsonUsage {
    input_tokens:                  Option<u64>,
    output_tokens:                 Option<u64>,
    cache_creation_input_tokens:   Option<u64>,
    cache_read_input_tokens:       Option<u64>,
}

// ── Pricing table ────────────────────────────────────────────────────────────

// (prefix, input, output, cache_create, cache_read) per token.
// Prices are USD per million tokens / 1_000_000; cache_create is the 5-minute
// cache-write rate. Verified against the official Anthropic API pricing docs.
// Longest/most-specific prefixes are not required to precede shorter ones here
// because no prefix below is itself a prefix of another model's id.
static PRICING: &[(&str, f64, f64, f64, f64)] = &[
    // Opus — $5 / $25 per MTok (Opus 4.5 through 4.8)
    ("claude-opus-4-8",          0.000005,  0.000025,  0.00000625, 0.0000005),
    ("claude-opus-4-7",          0.000005,  0.000025,  0.00000625, 0.0000005),
    ("claude-opus-4-6",          0.000005,  0.000025,  0.00000625, 0.0000005),
    ("claude-opus-4-5",          0.000005,  0.000025,  0.00000625, 0.0000005),
    // Opus — $15 / $75 per MTok (Opus 4.1 and Opus 4)
    ("claude-opus-4-1",          0.000015,  0.000075,  0.00001875, 0.0000015),
    ("claude-opus-4-20250514",   0.000015,  0.000075,  0.00001875, 0.0000015),
    // Sonnet — $3 / $15 per MTok (Sonnet 4, 4.5, 4.6)
    ("claude-sonnet-4-6",        0.000003,  0.000015,  0.00000375, 0.0000003),
    ("claude-sonnet-4-5",        0.000003,  0.000015,  0.00000375, 0.0000003),
    ("claude-sonnet-4-20250514", 0.000003,  0.000015,  0.00000375, 0.0000003),
    // Haiku 4.5 — $1 / $5 per MTok
    ("claude-haiku-4-5",         0.000001,  0.000005,  0.00000125, 0.0000001),
    // Haiku 3.5 — $0.80 / $4 per MTok (retired on the first-party API)
    ("claude-3-5-haiku",         0.0000008, 0.000004,  0.000001,   0.00000008),
];

// ── Public functions ─────────────────────────────────────────────────────────

pub fn normalize_model(m: &str) -> String {
    let m = m.strip_prefix("anthropic.").unwrap_or(m);
    let m = if let Some(pos) = m.find("-v") {
        let rest = &m[pos + 2..];
        if rest.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            // check pattern -vN:N
            if rest.contains(':') { &m[..pos] } else { m }
        } else { m }
    } else { m };
    let m = m.split('@').next().unwrap_or(m);
    m.to_string()
}

pub fn get_cost(model: &str, inp: u64, out: u64, cc: u64, cr: u64) -> f64 {
    let norm = normalize_model(model);
    // Unknown / unpriced models (e.g. "<synthetic>") contribute $0 so they
    // are obvious in reports rather than silently billed at some default rate.
    match PRICING.iter().find(|(pfx, ..)| norm.starts_with(pfx)) {
        Some(&(_, ip, op, cp, rp)) => {
            inp as f64 * ip + out as f64 * op + cc as f64 * cp + cr as f64 * rp
        }
        None => 0.0,
    }
}

pub fn fmt_num(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000     { format!("{:.1}k", n as f64 / 1e3) }
    else                   { n.to_string() }
}

pub fn fmt_cost(c: f64) -> String {
    if c >= 1000.0      { format!("${:.0}", c) }
    else if c >= 100.0  { format!("${:.1}", c) }
    else                { format!("${:.2}", c) }
}

pub fn get_log_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(env) = std::env::var("CLAUDE_CONFIG_DIR") {
        for part in env.split(',') {
            let part = part.trim();
            if part.is_empty() { continue; }
            let p = PathBuf::from(part);
            if p.file_name().and_then(|n| n.to_str()) == Some("projects") {
                dirs.push(p);
            } else {
                dirs.push(p.join("projects"));
            }
        }
    } else {
        if let Some(home) = dirs_home() {
            dirs.push(home.join(".config/claude/projects"));
            dirs.push(home.join(".claude/projects"));
        }
    }

    dirs.into_iter().filter(|d| d.is_dir()).collect()
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

pub fn find_jsonl_files(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in dirs {
        collect_jsonl(dir, &mut out);
    }
    out
}

fn collect_jsonl(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let rd = match std::fs::read_dir(dir) { Ok(r) => r, Err(_) => return };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
}

pub fn iter_file_records(path: &PathBuf) -> impl Iterator<Item = RawRecord> {
    let file = File::open(path).ok();
    let lines: Vec<String> = match file {
        None => vec![],
        Some(f) => BufReader::new(f).lines().flatten().collect(),
    };
    lines.into_iter().filter_map(|line| {
        let jl: JsonLine = serde_json::from_str(&line).ok()?;
        if jl.kind != "assistant" { return None; }
        let msg = jl.message.as_ref()?;
        let usage = msg.usage.as_ref()?;
        Some(RawRecord {
            timestamp:    jl.timestamp,
            req_id:       jl.request_id.unwrap_or_default(),
            msg_id:       msg.id.clone().unwrap_or_default(),
            model:        msg.model.clone().unwrap_or_else(|| "unknown".into()),
            input:        usage.input_tokens.unwrap_or(0),
            output:       usage.output_tokens.unwrap_or(0),
            cache_create: usage.cache_creation_input_tokens.unwrap_or(0),
            cache_read:   usage.cache_read_input_tokens.unwrap_or(0),
        })
    })
}

/// Deduplication key
pub fn dedup_key(r: &RawRecord) -> Option<(String, String)> {
    if r.msg_id.is_empty() || r.req_id.is_empty() { None }
    else { Some((r.msg_id.clone(), r.req_id.clone())) }
}

pub struct DeduplicatedRecords {
    pub records: Vec<RawRecord>,
}

impl DeduplicatedRecords {
    pub fn collect(files: &[PathBuf]) -> Self {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut records = Vec::new();
        for path in files {
            for rec in iter_file_records(path) {
                if let Some(key) = dedup_key(&rec) {
                    if !seen.insert(key) { continue; }
                }
                records.push(rec);
            }
        }
        Self { records }
    }
}
