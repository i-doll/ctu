use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;

// ── Public data types ────────────────────────────────────────────────────────

pub struct RawRecord {
    pub timestamp: String,
    pub req_id: String,
    pub msg_id: String,
    pub provider: String,
    pub model: String,
    pub cost_usd: Option<f64>,
    pub input: u64,
    pub output: u64,
    pub cache_create: u64,
    pub cache_read: u64,
}

#[derive(Default, Clone)]
pub struct PeriodAgg {
    pub input: u64,
    pub output: u64,
    pub cache_create: u64,
    pub cache_read: u64,
    pub cost: f64,
}

// ── JSONL deserialization ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct JsonLine {
    #[serde(rename = "type")]
    kind: String,
    timestamp: String,
    #[serde(rename = "requestId", default)]
    request_id: Option<String>,
    message: Option<JsonMessage>,
    payload: Option<CodexPayload>,
}

#[derive(Deserialize)]
struct JsonMessage {
    id: Option<String>,
    model: Option<String>,
    usage: Option<JsonUsage>,
}

#[derive(Deserialize)]
struct JsonUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct CodexPayload {
    #[serde(rename = "type")]
    kind: Option<String>,
    model: Option<String>,
    model_provider: Option<String>,
    info: Option<CodexTokenInfo>,
}

#[derive(Deserialize)]
struct CodexTokenInfo {
    last_token_usage: Option<CodexUsage>,
}

#[derive(Deserialize)]
struct CodexUsage {
    input_tokens: Option<u64>,
    cached_input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct OpenCodeMessage {
    id: Option<String>,
    #[serde(rename = "sessionID")]
    session_id: Option<String>,
    role: String,
    #[serde(rename = "modelID")]
    model_id: Option<String>,
    #[serde(rename = "providerID")]
    provider_id: Option<String>,
    cost: Option<f64>,
    tokens: Option<OpenCodeTokens>,
    time: Option<OpenCodeTime>,
}

#[derive(Deserialize)]
struct OpenCodeTokens {
    input: Option<u64>,
    output: Option<u64>,
    reasoning: Option<u64>,
    cache: Option<OpenCodeCache>,
}

#[derive(Deserialize)]
struct OpenCodeCache {
    read: Option<u64>,
    write: Option<u64>,
}

#[derive(Deserialize)]
struct OpenCodeTime {
    created: Option<i64>,
}

// ── Pricing table ────────────────────────────────────────────────────────────

// Standard processing prices in USD per million tokens. More-specific model
// prefixes must precede their families. `long_context` is `(threshold, output
// multiplier)`: above `threshold` input tokens the whole request bills at 2x
// input/cache and the given output multiplier. OpenAI charges 1.5x output over
// 272K and Gemini Pro the same over 200K; xAI doubles every rate over 200K.
// Providers whose cache writes carry no separate charge (OpenAI, Google, xAI,
// Moonshot) set `cache_create` equal to `input` — those tokens bill as
// ordinary input.
//
// A model whose price changes on a date gets one entry per window, chained with
// `.until(date)` / `.from(date)`. Bounds are `YYYY-MM-DD` strings compared
// lexically against the record's date, `from` inclusive and `until` exclusive,
// so adjacent windows share a boundary date without overlapping:
//
//     price!("claude-sonnet-5", 2.00, 10.00, 2.50, 0.20).until("2026-09-01"),
//     price!("claude-sonnet-5", 3.00, 15.00, 3.75, 0.30).from("2026-09-01"),
//
// Order windows oldest-first and keep them adjacent, since lookup takes the
// first entry whose prefix and date both match. An undated entry matches every
// date, so it must come last within a model's group — anything after it is
// unreachable. A date outside every window for a prefix falls through to the
// next matching entry, or to $0 if there is none.
struct ModelPricing {
    prefix: &'static str,
    input: f64,
    output: f64,
    cache_create: f64,
    cache_read: f64,
    long_context: Option<(u64, f64)>,
    /// Inclusive lower bound, `YYYY-MM-DD`. `None` = no lower bound.
    from: Option<&'static str>,
    /// Exclusive upper bound, `YYYY-MM-DD`. `None` = no upper bound.
    until: Option<&'static str>,
}

impl ModelPricing {
    /// Restrict this entry to records dated on or after `date` (inclusive).
    const fn from(mut self, date: &'static str) -> Self {
        self.from = Some(date);
        self
    }

    /// Restrict this entry to records dated strictly before `date`.
    const fn until(mut self, date: &'static str) -> Self {
        self.until = Some(date);
        self
    }

    /// Whether this entry applies on `date` (`YYYY-MM-DD`). An entry with no
    /// bounds applies to every date, so undated tables behave as before.
    fn covers(&self, date: &str) -> bool {
        if let Some(from) = self.from {
            if date < from {
                return false;
            }
        }
        if let Some(until) = self.until {
            if date >= until {
                return false;
            }
        }
        true
    }
}

macro_rules! price {
    ($prefix:expr, $input:expr, $output:expr, $cache_create:expr, $cache_read:expr) => {
        ModelPricing {
            prefix: $prefix,
            input: $input,
            output: $output,
            cache_create: $cache_create,
            cache_read: $cache_read,
            long_context: None,
            from: None,
            until: None,
        }
    };
    ($prefix:expr, $input:expr, $output:expr, $cache_create:expr, $cache_read:expr, long) => {
        price!($prefix, $input, $output, $cache_create, $cache_read, long 272_000, 1.5)
    };
    ($prefix:expr, $input:expr, $output:expr, $cache_create:expr, $cache_read:expr, long $threshold:expr, $output_factor:expr) => {
        ModelPricing {
            prefix: $prefix,
            input: $input,
            output: $output,
            cache_create: $cache_create,
            cache_read: $cache_read,
            long_context: Some(($threshold, $output_factor)),
            from: None,
            until: None,
        }
    };
}

static PRICING: &[ModelPricing] = &[
    // OpenAI GPT-5.6 — cache writes are 1.25x uncached input.
    price!("gpt-5.6-terra", 2.50, 15.00, 3.125, 0.25, long),
    price!("gpt-5.6-luna", 1.00, 6.00, 1.25, 0.10, long),
    price!("gpt-5.6-sol", 5.00, 30.00, 6.25, 0.50, long),
    price!("gpt-5.6", 5.00, 30.00, 6.25, 0.50, long),
    // Current general models available in Codex.
    price!("gpt-5.5-pro", 30.00, 180.00, 30.00, 30.00, long),
    price!("gpt-5.5", 5.00, 30.00, 5.00, 0.50, long),
    price!("gpt-5.4-mini", 0.75, 4.50, 0.75, 0.075),
    price!("gpt-5.4-nano", 0.20, 1.25, 0.20, 0.02),
    price!("gpt-5.4-pro", 30.00, 180.00, 30.00, 30.00, long),
    price!("gpt-5.4", 2.50, 15.00, 2.50, 0.25, long),
    // Legacy Codex model families.
    price!("gpt-5.3-codex", 1.75, 14.00, 1.75, 0.175),
    price!("gpt-5.2-codex", 1.75, 14.00, 1.75, 0.175),
    price!("gpt-5.2", 1.75, 14.00, 1.75, 0.175),
    price!("gpt-5.1-codex-mini", 0.25, 2.00, 0.25, 0.025),
    price!("gpt-5.1-codex-max", 1.25, 10.00, 1.25, 0.125),
    price!("gpt-5.1-codex", 1.25, 10.00, 1.25, 0.125),
    price!("gpt-5.1", 1.25, 10.00, 1.25, 0.125),
    price!("gpt-5-codex", 1.25, 10.00, 1.25, 0.125),
    price!("gpt-5-mini", 0.25, 2.00, 0.25, 0.025),
    price!("gpt-5-nano", 0.05, 0.40, 0.05, 0.005),
    price!("gpt-5", 1.25, 10.00, 1.25, 0.125),
    price!("codex-mini-latest", 1.50, 6.00, 1.50, 0.375),
    price!("o4-mini", 1.10, 4.40, 1.10, 0.275),
    price!("o3", 2.00, 8.00, 2.00, 0.50),
    price!("gpt-4.1-mini", 0.40, 1.60, 0.40, 0.10),
    price!("gpt-4.1-nano", 0.10, 0.40, 0.10, 0.025),
    price!("gpt-4.1", 2.00, 8.00, 2.00, 0.50),
    price!("gpt-4o-2024-05-13", 5.00, 15.00, 5.00, 5.00),
    price!("gpt-4o-mini", 0.15, 0.60, 0.15, 0.075),
    price!("gpt-4o", 2.50, 10.00, 2.50, 1.25),
    // Open-weight; the rate depends entirely on who hosts it (roughly
    // $0.04-$0.15 in / $0.15-$0.75 out per MTok). Priced at the mid-range
    // OpenRouter rate rather than left unpriced.
    price!("gpt-oss-120b", 0.10, 0.50, 0.10, 0.10),
    // Google Gemini — cache reads are 10% of input; the per-hour storage fee on
    // explicit caches is not billed per token, so it is out of scope here.
    // Only Pro carries a long-context surcharge (>200K), Flash bills flat.
    price!("gemini-3.6-flash", 1.50, 7.50, 1.50, 0.15),
    price!("gemini-3.5-flash-lite", 0.30, 2.50, 0.30, 0.03),
    price!("gemini-3.5-flash", 1.50, 9.00, 1.50, 0.15),
    price!("gemini-3.1-flash-lite", 0.25, 1.50, 0.25, 0.025),
    price!("gemini-3.1-pro", 2.00, 12.00, 2.00, 0.20, long 200_000, 1.5),
    price!("gemini-3-flash", 0.50, 3.00, 0.50, 0.05),
    // The proxy's "gemini-pro-agent" alias tracks the current Pro tier.
    price!("gemini-pro", 2.00, 12.00, 2.00, 0.20, long 200_000, 1.5),
    // xAI Grok — every rate doubles above 200K input tokens.
    price!("grok-4.5", 2.00, 6.00, 2.00, 0.50, long 200_000, 2.0),
    price!("grok-4.3", 1.25, 2.50, 1.25, 0.20, long 200_000, 2.0),
    price!("grok-4.20", 1.25, 2.50, 1.25, 0.20, long 200_000, 2.0),
    price!("grok-build", 1.00, 2.00, 1.00, 0.20, long 200_000, 2.0),
    price!("grok-composer-2.5", 1.25, 2.50, 1.25, 0.20, long 200_000, 2.0),
    price!("grok-3-mini", 0.30, 0.50, 0.30, 0.075),
    // Moonshot Kimi — cache hits are ~10% of input; no long-context surcharge.
    price!("kimi-k3", 3.00, 15.00, 3.00, 0.30),
    price!("kimi-k2.7-code", 0.95, 4.00, 0.95, 0.19),
    price!("kimi-k2.6", 0.95, 4.00, 0.95, 0.19),
    price!("kimi-k2.5", 0.60, 3.00, 0.60, 0.19),
    price!("kimi-k2-thinking", 0.60, 2.50, 0.60, 0.15),
    price!("kimi-k2", 0.55, 2.20, 0.55, 0.15),
    // Fable 5 — $10 / $50 per MTok
    price!("claude-fable-5", 10.00, 50.00, 12.50, 1.00),
    // Opus 5 — $5 / $25 per MTok
    price!("claude-opus-5", 5.00, 25.00, 6.25, 0.50),
    // Sonnet 5 — introductory $2 / $10 per MTok through 2026-08-31, then the
    // $3 / $15 list price from 2026-09-01. Cache rates scale with input
    // (write 1.25x, read 0.1x), so they shift with it.
    price!("claude-sonnet-5", 2.00, 10.00, 2.50, 0.20).until("2026-09-01"),
    price!("claude-sonnet-5", 3.00, 15.00, 3.75, 0.30).from("2026-09-01"),
    // Opus — $5 / $25 per MTok (Opus 4.5 through 4.8)
    price!("claude-opus-4-8", 5.00, 25.00, 6.25, 0.50),
    price!("claude-opus-4-7", 5.00, 25.00, 6.25, 0.50),
    price!("claude-opus-4-6", 5.00, 25.00, 6.25, 0.50),
    price!("claude-opus-4-5", 5.00, 25.00, 6.25, 0.50),
    // Opus — $15 / $75 per MTok (Opus 4.1 and Opus 4)
    price!("claude-opus-4-1", 15.00, 75.00, 18.75, 1.50),
    price!("claude-opus-4-20250514", 15.00, 75.00, 18.75, 1.50),
    // Sonnet — $3 / $15 per MTok (Sonnet 4, 4.5, 4.6)
    price!("claude-sonnet-4-6", 3.00, 15.00, 3.75, 0.30),
    price!("claude-sonnet-4-5", 3.00, 15.00, 3.75, 0.30),
    price!("claude-sonnet-4-20250514", 3.00, 15.00, 3.75, 0.30),
    // Haiku 4.5 — $1 / $5 per MTok
    price!("claude-haiku-4-5", 1.00, 5.00, 1.25, 0.10),
    // Haiku 3.5 — $0.80 / $4 per MTok (retired on the first-party API)
    price!("claude-3-5-haiku", 0.80, 4.00, 1.00, 0.08),
    // Sonnet 3.7 — $3 / $15 per MTok (retired on the first-party API)
    price!("claude-3-7-sonnet", 3.00, 15.00, 3.75, 0.30),
    // Bare tier aliases that harnesses log instead of a resolved model id
    // (subagent short names, workflow model defaults). Each is priced at the
    // current model in its tier, so they read as approximately right rather
    // than as $0. Keep last: these prefixes are short and would otherwise
    // shadow the specific ids above.
    price!("gemini-default", 1.50, 9.00, 1.50, 0.15),
    price!("fable", 10.00, 50.00, 12.50, 1.00),
    price!("opus", 5.00, 25.00, 6.25, 0.50),
    price!("sonnet", 3.00, 15.00, 3.75, 0.30),
    price!("haiku", 1.00, 5.00, 1.25, 0.10),
];

// ── Public functions ─────────────────────────────────────────────────────────

pub fn normalize_model(m: &str) -> String {
    let m = if let Some(pos) = m.find(".anthropic.") {
        &m[pos + 1..]
    } else {
        m
    };
    let m = m.strip_prefix("anthropic.").unwrap_or(m);
    let m = if let Some(pos) = m.find("-v") {
        let rest = &m[pos + 2..];
        if rest
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            // check pattern -vN:N
            if rest.contains(':') {
                &m[..pos]
            } else {
                m
            }
        } else {
            m
        }
    } else {
        m
    };
    let m = m.split('@').next().unwrap_or(m);
    m.to_string()
}

/// Sentinel date meaning "the most recent price window". Sorts after any real
/// `YYYY-MM-DD`, so it selects the open-ended final window rather than baking a
/// build-time date into the binary.
const LATEST_RATES: &str = "9999-12-31";

/// Cost of a request priced at the latest known rates. Prefer [`get_cost_on`]
/// for stored records: a model with dated price windows should bill at
/// whichever window covers the record's own date, not the current one.
pub fn get_cost(model: &str, inp: u64, out: u64, cc: u64, cr: u64) -> f64 {
    get_cost_on(model, LATEST_RATES, inp, out, cc, cr)
}

/// Cost of a request made on `date` (`YYYY-MM-DD`), honoring dated price
/// windows. A date matching no window for the model contributes $0, same as an
/// unpriced model.
pub fn get_cost_on(model: &str, date: &str, inp: u64, out: u64, cc: u64, cr: u64) -> f64 {
    let norm = normalize_model(model);
    // Unknown / unpriced models (e.g. "<synthetic>") contribute $0 so they
    // are obvious in reports rather than silently billed at some default rate.
    match PRICING
        .iter()
        .find(|pricing| norm.starts_with(pricing.prefix) && pricing.covers(date))
    {
        Some(pricing) => {
            let input_tokens = inp.saturating_add(cc).saturating_add(cr);
            let long = pricing
                .long_context
                .filter(|(threshold, _)| input_tokens > *threshold);
            let input_factor = if long.is_some() { 2.0 } else { 1.0 };
            let output_factor = long.map_or(1.0, |(_, factor)| factor);
            (inp as f64 * pricing.input * input_factor
                + out as f64 * pricing.output * output_factor
                + cc as f64 * pricing.cache_create * input_factor
                + cr as f64 * pricing.cache_read * input_factor)
                / 1_000_000.0
        }
        None => 0.0,
    }
}

pub fn get_record_cost(record: &RawRecord) -> f64 {
    record.cost_usd.unwrap_or_else(|| {
        // Price against the record's own date so historical rows keep billing
        // at the rate that was in effect when they were made. Timestamps are
        // ISO-8601, so the leading 10 chars are the `YYYY-MM-DD` date; a
        // malformed timestamp falls back to the latest rates.
        let date = record.timestamp.get(..10).unwrap_or(LATEST_RATES);
        get_cost_on(
            &record.model,
            date,
            record.input,
            record.output,
            record.cache_create,
            record.cache_read,
        )
    })
}

pub fn fmt_num(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

pub fn fmt_cost(c: f64) -> String {
    if c >= 1000.0 {
        format!("${:.0}", c)
    } else if c >= 100.0 {
        format!("${:.1}", c)
    } else {
        format!("${:.2}", c)
    }
}

pub fn get_log_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(env) = std::env::var("CLAUDE_CONFIG_DIR") {
        for part in env.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let p = PathBuf::from(part);
            if p.file_name().and_then(|n| n.to_str()) == Some("projects") {
                dirs.push(p);
            } else {
                dirs.push(p.join("projects"));
            }
        }
    } else if let Some(home) = dirs_home() {
        dirs.push(home.join(".config/claude/projects"));
        dirs.push(home.join(".claude/projects"));
    }

    if let Some(env) = std::env::var("CODEX_HOME")
        .ok()
        .filter(|value| !value.is_empty())
    {
        let p = PathBuf::from(env);
        if p.file_name().and_then(|n| n.to_str()) == Some("sessions") {
            dirs.push(p);
        } else {
            dirs.push(p.join("sessions"));
        }
    } else if let Some(home) = dirs_home() {
        dirs.push(home.join(".codex/sessions"));
    }

    let mut seen = HashSet::new();
    dirs.into_iter()
        .filter(|d| d.is_dir() && seen.insert(d.clone()))
        .collect()
}

pub fn get_opencode_files() -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Some(db) = std::env::var("OPENCODE_DB")
        .ok()
        .filter(|value| !value.is_empty())
    {
        let path = PathBuf::from(db);
        if path.is_file() {
            files.push(path);
        }
    }

    let data_dir = std::env::var("OPENCODE_DATA_DIR")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("XDG_DATA_HOME")
                .ok()
                .filter(|value| !value.is_empty())
                .map(|value| PathBuf::from(value).join("opencode"))
        })
        .or_else(|| dirs_home().map(|home| home.join(".local/share/opencode")));

    if let Some(data_dir) = data_dir {
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if path.is_file()
                    && path.extension().and_then(|ext| ext.to_str()) == Some("db")
                    && name.starts_with("opencode")
                {
                    files.push(path);
                }
            }
        }

        collect_extension_files(&data_dir.join("storage/message"), "json", &mut files);
    }

    let mut seen = HashSet::new();
    files.retain(|path| seen.insert(path.clone()));
    files
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
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
}

fn collect_extension_files(dir: &Path, extension: &str, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_extension_files(&path, extension, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
            out.push(path);
        }
    }
}

pub fn iter_file_records(path: &PathBuf) -> impl Iterator<Item = RawRecord> {
    let file = File::open(path).ok();
    let lines: Vec<String> = match file {
        None => vec![],
        Some(f) => BufReader::new(f).lines().map_while(Result::ok).collect(),
    };
    parse_lines(lines, path.to_string_lossy().as_ref()).into_iter()
}

fn parse_lines(lines: Vec<String>, source_id: &str) -> Vec<RawRecord> {
    let mut codex_model = "unknown".to_string();
    let mut codex_provider = "openai".to_string();
    let mut records = Vec::new();

    for (line_no, line) in lines.into_iter().enumerate() {
        let Ok(jl) = serde_json::from_str::<JsonLine>(&line) else {
            continue;
        };
        if jl.kind == "assistant" {
            let Some(msg) = jl.message.as_ref() else {
                continue;
            };
            let Some(usage) = msg.usage.as_ref() else {
                continue;
            };
            records.push(RawRecord {
                timestamp: jl.timestamp,
                req_id: jl.request_id.unwrap_or_default(),
                msg_id: msg.id.clone().unwrap_or_default(),
                provider: claude_provider(msg.model.as_deref().unwrap_or("unknown")).to_string(),
                model: msg.model.clone().unwrap_or_else(|| "unknown".into()),
                cost_usd: None,
                input: usage.input_tokens.unwrap_or(0),
                output: usage.output_tokens.unwrap_or(0),
                cache_create: usage.cache_creation_input_tokens.unwrap_or(0),
                cache_read: usage.cache_read_input_tokens.unwrap_or(0),
            });
            continue;
        }

        let Some(payload) = jl.payload.as_ref() else {
            continue;
        };
        if jl.kind == "session_meta" {
            if let Some(provider) = payload.model_provider.as_ref() {
                codex_provider.clone_from(provider);
            }
            continue;
        }
        if jl.kind == "turn_context" {
            if let Some(model) = payload.model.as_ref() {
                codex_model.clone_from(model);
            }
            continue;
        }
        if jl.kind != "event_msg" || payload.kind.as_deref() != Some("token_count") {
            continue;
        }
        let Some(usage) = payload
            .info
            .as_ref()
            .and_then(|i| i.last_token_usage.as_ref())
        else {
            continue;
        };
        let cached = usage.cached_input_tokens.unwrap_or(0);
        records.push(RawRecord {
            timestamp: jl.timestamp.clone(),
            req_id: source_id.to_string(),
            msg_id: format!("{}:{}", jl.timestamp, line_no + 1),
            provider: codex_provider.clone(),
            model: codex_model.clone(),
            cost_usd: None,
            input: usage.input_tokens.unwrap_or(0).saturating_sub(cached),
            output: usage.output_tokens.unwrap_or(0),
            cache_create: 0,
            cache_read: cached,
        });
    }

    records
}

fn claude_provider(model: &str) -> &'static str {
    if model.contains('@') {
        "google-vertex"
    } else if model.starts_with("anthropic.") || model.contains(".anthropic.") {
        "amazon-bedrock"
    } else if is_openai_model(model) {
        "openai"
    } else if is_anthropic_model(model) {
        "anthropic"
    } else if is_google_model(model) {
        "google"
    } else if is_xai_model(model) {
        "xai"
    } else if is_moonshot_model(model) {
        "moonshotai"
    } else {
        "unknown"
    }
}

fn is_openai_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.starts_with("gpt-")
        || model.starts_with("chatgpt-")
        || model.starts_with("codex-")
        || model_family(&model, "o1")
        || model_family(&model, "o3")
        || model_family(&model, "o4")
}

fn is_anthropic_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.starts_with("claude-") || matches!(model.as_str(), "opus" | "sonnet" | "haiku" | "fable")
}

fn is_google_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    // Gemma ids vary in separator ("gemma-3-27b-it", "gemma3:4b"), so match the
    // bare family name; no other vendor uses it.
    model_family(&model, "gemini") || model.starts_with("gemma")
}

fn is_xai_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model_family(&model, "grok")
}

fn is_moonshot_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model_family(&model, "kimi") || model_family(&model, "moonshot")
}

fn model_family(model: &str, family: &str) -> bool {
    model == family
        || model
            .strip_prefix(family)
            .map(|suffix| suffix.starts_with('-'))
            .unwrap_or(false)
}

fn read_opencode_records(path: &Path) -> Vec<RawRecord> {
    if path.extension().and_then(|ext| ext.to_str()) == Some("db") {
        read_opencode_database(path)
    } else {
        read_opencode_json(path).into_iter().collect()
    }
}

fn read_opencode_database(path: &Path) -> Vec<RawRecord> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let connection = match Connection::open_with_flags(path, flags) {
        Ok(connection) => connection,
        Err(_) => return Vec::new(),
    };
    let mut statement = match connection
        .prepare("SELECT id, session_id, time_created, data FROM message ORDER BY time_created, id")
    {
        Ok(statement) => statement,
        Err(_) => return Vec::new(),
    };
    let rows = match statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
        ))
    }) {
        Ok(rows) => rows,
        Err(_) => return Vec::new(),
    };

    rows.filter_map(Result::ok)
        .filter_map(|(id, session_id, created, data)| {
            parse_opencode_message(&data, Some(&id), Some(&session_id), Some(created))
        })
        .collect()
}

fn read_opencode_json(path: &Path) -> Option<RawRecord> {
    let data = std::fs::read_to_string(path).ok()?;
    let fallback_id = path.file_stem().and_then(|name| name.to_str());
    let fallback_session = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str());
    parse_opencode_message(&data, fallback_id, fallback_session, None)
}

fn parse_opencode_message(
    data: &str,
    fallback_id: Option<&str>,
    fallback_session: Option<&str>,
    fallback_created: Option<i64>,
) -> Option<RawRecord> {
    let message: OpenCodeMessage = serde_json::from_str(data).ok()?;
    if message.role != "assistant" {
        return None;
    }
    let tokens = message.tokens?;
    let cache = tokens.cache.unwrap_or(OpenCodeCache {
        read: Some(0),
        write: Some(0),
    });
    let created = message
        .time
        .and_then(|time| time.created)
        .or(fallback_created)?;

    Some(RawRecord {
        timestamp: timestamp_from_millis(created),
        req_id: message
            .session_id
            .or_else(|| fallback_session.map(str::to_string))
            .unwrap_or_default(),
        msg_id: message
            .id
            .or_else(|| fallback_id.map(str::to_string))
            .unwrap_or_default(),
        provider: message
            .provider_id
            .unwrap_or_else(|| "opencode".to_string()),
        model: message.model_id.unwrap_or_else(|| "unknown".to_string()),
        cost_usd: message.cost,
        input: tokens.input.unwrap_or(0),
        output: tokens
            .output
            .unwrap_or(0)
            .saturating_add(tokens.reasoning.unwrap_or(0)),
        cache_create: cache.write.unwrap_or(0),
        cache_read: cache.read.unwrap_or(0),
    })
}

fn timestamp_from_millis(milliseconds: i64) -> String {
    let seconds = milliseconds.div_euclid(1_000);
    let millis = milliseconds.rem_euclid(1_000);
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    // Howard Hinnant's civil_from_days algorithm, with 1970-01-01 as day zero.
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Deduplication key
pub fn dedup_key(r: &RawRecord) -> Option<(String, String)> {
    if r.msg_id.is_empty() || r.req_id.is_empty() {
        None
    } else {
        Some((r.msg_id.clone(), r.req_id.clone()))
    }
}

pub struct DeduplicatedRecords {
    pub records: Vec<RawRecord>,
}

impl DeduplicatedRecords {
    pub fn collect(files: &[PathBuf]) -> Self {
        Self::collect_with_opencode(files, &[])
    }

    pub fn collect_with_opencode(jsonl_files: &[PathBuf], opencode_files: &[PathBuf]) -> Self {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut records = Vec::new();
        for path in jsonl_files {
            for rec in iter_file_records(path) {
                if let Some(key) = dedup_key(&rec) {
                    if !seen.insert(key) {
                        continue;
                    }
                }
                records.push(rec);
            }
        }
        for path in opencode_files {
            for rec in read_opencode_records(path) {
                if let Some(key) = dedup_key(&rec) {
                    if !seen.insert(key) {
                        continue;
                    }
                }
                records.push(rec);
            }
        }
        Self { records }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    #[test]
    fn prices_current_and_legacy_openai_models() {
        assert_close(get_cost("gpt-5.6-sol", 100_000, 100_000, 0, 0), 3.5);
        assert_close(get_cost("gpt-5.3-codex", 1_000_000, 1_000_000, 0, 0), 15.75);
        assert_close(get_cost("gpt-5", 1_000_000, 1_000_000, 0, 0), 11.25);
        assert_close(get_cost("gpt-4o", 1_000_000, 1_000_000, 0, 0), 12.50);
    }

    #[test]
    fn prices_cached_input_and_long_context() {
        assert_close(get_cost("gpt-5.6", 0, 0, 1_000_000, 1_000_000), 13.50);
        // More than 272K total input applies 2x input/cache and 1.5x output.
        assert_close(get_cost("gpt-5.6-terra", 272_001, 10_000, 0, 0), 1.585005);
    }

    #[test]
    fn prices_google_xai_and_moonshot_models() {
        assert_close(
            get_cost("gemini-3.5-flash", 1_000_000, 1_000_000, 0, 0),
            10.50,
        );
        assert_close(get_cost("gemini-3.5-flash-low", 1_000_000, 0, 0, 0), 1.50);
        // Kept under each provider's long-context threshold to test base rates;
        // the surcharges have their own test below.
        assert_close(get_cost("gemini-3.1-pro", 100_000, 100_000, 0, 0), 1.40);
        assert_close(get_cost("grok-4.5", 100_000, 100_000, 0, 0), 0.80);
        assert_close(get_cost("grok-4.5-build-free", 100_000, 0, 0, 0), 0.20);
        assert_close(get_cost("kimi-k3", 1_000_000, 1_000_000, 0, 0), 18.00);
        assert_close(get_cost("kimi-k2.6", 1_000_000, 1_000_000, 0, 0), 4.95);
    }

    #[test]
    fn prices_bare_tier_aliases_at_current_tier_rates() {
        assert_close(get_cost("opus", 1_000_000, 0, 0, 0), 5.00);
        assert_close(get_cost("sonnet", 1_000_000, 0, 0, 0), 3.00);
        assert_close(get_cost("haiku", 1_000_000, 0, 0, 0), 1.00);
        assert_close(get_cost("fable", 1_000_000, 0, 0, 0), 10.00);
        assert_close(get_cost("gemini-default", 1_000_000, 0, 0, 0), 1.50);
        // The aliases must not shadow fully-qualified ids that share a prefix.
        assert_close(get_cost("claude-opus-4-1", 1_000_000, 0, 0, 0), 15.00);
        assert_close(get_cost("gemini-3.5-flash-lite", 1_000_000, 0, 0, 0), 0.30);
    }

    #[test]
    fn prices_dated_windows_by_record_date() {
        // Sonnet 5 introductory rate: $2 / $10 per MTok through 2026-08-31.
        assert_close(
            get_cost_on("claude-sonnet-5", "2026-07-25", 1_000_000, 1_000_000, 0, 0),
            12.00,
        );
        // Last day of the intro window still bills at the intro rate.
        assert_close(
            get_cost_on("claude-sonnet-5", "2026-08-31", 1_000_000, 0, 0, 0),
            2.00,
        );
        // `until` is exclusive, so the boundary date itself is list price.
        assert_close(
            get_cost_on("claude-sonnet-5", "2026-09-01", 1_000_000, 1_000_000, 0, 0),
            18.00,
        );
        assert_close(
            get_cost_on("claude-sonnet-5", "2027-03-14", 1_000_000, 0, 0, 0),
            3.00,
        );
        // Cache rates shift with the window too.
        assert_close(
            get_cost_on("claude-sonnet-5", "2026-07-25", 0, 0, 1_000_000, 1_000_000),
            2.70,
        );
        assert_close(
            get_cost_on("claude-sonnet-5", "2026-09-01", 0, 0, 1_000_000, 1_000_000),
            4.05,
        );
        // A dateless call bills at the latest window, not the intro rate.
        assert_close(get_cost("claude-sonnet-5", 1_000_000, 0, 0, 0), 3.00);
    }

    #[test]
    fn undated_models_ignore_the_record_date() {
        for date in ["2024-01-01", "2026-07-25", "2099-12-31"] {
            assert_close(get_cost_on("claude-opus-5", date, 1_000_000, 0, 0, 0), 5.00);
            // Kept under 272K so the long-context surcharge stays out of it.
            assert_close(get_cost_on("gpt-5.6-sol", date, 100_000, 0, 0, 0), 0.50);
            assert_close(get_cost_on("kimi-k3", date, 1_000_000, 0, 0, 0), 3.00);
        }
    }

    #[test]
    fn prices_records_at_their_own_date() {
        let dated = |timestamp: &str| RawRecord {
            timestamp: timestamp.to_string(),
            req_id: "req".into(),
            msg_id: "msg".into(),
            provider: "anthropic".into(),
            model: "claude-sonnet-5".into(),
            cost_usd: None,
            input: 1_000_000,
            output: 0,
            cache_create: 0,
            cache_read: 0,
        };

        // A record from the intro window keeps billing at the intro rate even
        // after the list price takes effect.
        assert_close(get_record_cost(&dated("2026-07-25T10:00:00Z")), 2.00);
        assert_close(get_record_cost(&dated("2026-09-01T10:00:00Z")), 3.00);
        // A malformed timestamp falls back to the latest rates rather than $0.
        assert_close(get_record_cost(&dated("bogus")), 3.00);

        // A recorded cost still wins over the table, dated windows included.
        let mut recorded = dated("2026-07-25T10:00:00Z");
        recorded.cost_usd = Some(42.0);
        assert_close(get_record_cost(&recorded), 42.0);
    }

    #[test]
    fn prices_claude_5_family_and_retired_sonnet() {
        assert_close(get_cost("claude-opus-5", 1_000_000, 1_000_000, 0, 0), 30.00);
        assert_close(
            get_cost("claude-sonnet-5", 1_000_000, 1_000_000, 0, 0),
            18.00,
        );
        assert_close(
            get_cost("claude-3-7-sonnet-20250219", 1_000_000, 0, 0, 0),
            3.00,
        );
    }

    #[test]
    fn applies_provider_specific_long_context_thresholds() {
        // Gemini Pro bills the surcharge above 200K, not OpenAI's 272K.
        assert_close(get_cost("gemini-3.1-pro", 200_001, 0, 0, 0), 0.800004);
        assert_close(get_cost("gemini-3.1-pro", 200_000, 0, 0, 0), 0.400000);
        // xAI doubles output too, rather than applying OpenAI's 1.5x.
        assert_close(get_cost("grok-4.5", 200_001, 100_000, 0, 0), 2.000004);
        // Gemini Flash bills flat at any size.
        assert_close(get_cost("gemini-3.5-flash", 1_000_000, 0, 0, 0), 1.50);
    }

    #[test]
    fn parses_claude_and_codex_records() {
        let lines = vec![
            r#"{"type":"assistant","timestamp":"2026-07-09T10:00:00Z","requestId":"req","message":{"id":"msg","model":"claude-sonnet-4-6","usage":{"input_tokens":10,"output_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}}"#.to_string(),
            r#"{"type":"session_meta","timestamp":"2026-07-10T09:59:59Z","payload":{"model_provider":"openai"}}"#.to_string(),
            r#"{"type":"turn_context","timestamp":"2026-07-10T10:00:00Z","payload":{"model":"gpt-5.6-sol"}}"#.to_string(),
            r#"{"type":"event_msg","timestamp":"2026-07-10T10:00:01Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":999999},"last_token_usage":{"input_tokens":100,"cached_input_tokens":60,"output_tokens":7}}}}"#.to_string(),
        ];

        let records = parse_lines(lines, "codex-session.jsonl");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].provider, "anthropic");
        assert_eq!(records[0].model, "claude-sonnet-4-6");
        assert_eq!(
            (
                records[0].input,
                records[0].cache_create,
                records[0].cache_read
            ),
            (10, 3, 4)
        );
        assert_eq!(records[1].model, "gpt-5.6-sol");
        assert_eq!(records[1].provider, "openai");
        assert_eq!(
            (records[1].input, records[1].output, records[1].cache_read),
            (40, 7, 60)
        );
        assert_eq!(records[1].req_id, "codex-session.jsonl");
    }

    #[test]
    fn attributes_openai_models_in_claude_logs_to_openai() {
        let lines = vec![
            r#"{"type":"assistant","timestamp":"2026-07-18T10:00:00Z","requestId":"req","message":{"id":"msg","model":"gpt-5.6-sol","usage":{"input_tokens":10,"output_tokens":2}}}"#.to_string(),
        ];

        let records = parse_lines(lines, "claude-session.jsonl");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model, "gpt-5.6-sol");
        assert_eq!(records[0].provider, "openai");
    }

    #[test]
    fn infers_claude_log_provider_from_hosting_or_model() {
        assert_eq!(claude_provider("claude-sonnet-4-6"), "anthropic");
        assert_eq!(claude_provider("sonnet"), "anthropic");
        assert_eq!(
            claude_provider("us.anthropic.claude-sonnet-4-6-v1:0"),
            "amazon-bedrock"
        );
        assert_eq!(
            normalize_model("us.anthropic.claude-sonnet-4-6-v1:0"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            claude_provider("claude-sonnet-4-6@20260217"),
            "google-vertex"
        );
        assert_eq!(claude_provider("gpt-5.6-sol"), "openai");
        assert_eq!(claude_provider("gpt-4o"), "openai");
        assert_eq!(claude_provider("o3"), "openai");
        assert_eq!(claude_provider("o4-mini"), "openai");
        assert_eq!(claude_provider("codex-mini-latest"), "openai");
        assert_eq!(claude_provider("gemini-3.5-flash-low"), "google");
        assert_eq!(claude_provider("gemini-default"), "google");
        assert_eq!(claude_provider("gemma-3-27b-it"), "google");
        assert_eq!(claude_provider("grok-4.5"), "xai");
        assert_eq!(claude_provider("grok-4.5-build-free"), "xai");
        assert_eq!(claude_provider("kimi-k3"), "moonshotai");
        assert_eq!(claude_provider("moonshot-v1-8k"), "moonshotai");
        assert_eq!(claude_provider("<synthetic>"), "unknown");
    }

    #[test]
    fn parses_opencode_usage_and_recorded_cost() {
        let data = r#"{
            "id":"msg_test",
            "sessionID":"ses_test",
            "role":"assistant",
            "modelID":"custom-model",
            "providerID":"openrouter",
            "cost":0.42,
            "tokens":{
                "input":6,
                "output":100,
                "reasoning":5,
                "cache":{"write":20,"read":30}
            },
            "time":{"created":0}
        }"#;

        let record = parse_opencode_message(data, None, None, None).unwrap();
        assert_eq!(record.timestamp, "1970-01-01T00:00:00.000Z");
        assert_eq!(record.provider, "openrouter");
        assert_eq!(record.model, "custom-model");
        assert_eq!((record.input, record.output), (6, 105));
        assert_eq!((record.cache_create, record.cache_read), (20, 30));
        assert_close(get_record_cost(&record), 0.42);
    }

    #[test]
    fn reads_opencode_sqlite_messages() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ctu-opencode-{unique}.db"));
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE message (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    time_created INTEGER NOT NULL,
                    data TEXT NOT NULL
                );",
            )
            .unwrap();
        let data = r#"{
            "role":"assistant",
            "modelID":"gpt-5",
            "providerID":"openai",
            "cost":0.01,
            "tokens":{"input":10,"output":2,"reasoning":1,"cache":{"write":3,"read":4}}
        }"#;
        connection
            .execute(
                "INSERT INTO message (id, session_id, time_created, data) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params!["msg_db", "ses_db", 1_000_i64, data],
            )
            .unwrap();
        drop(connection);

        let records = read_opencode_database(&path);
        std::fs::remove_file(path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].msg_id, "msg_db");
        assert_eq!(records[0].req_id, "ses_db");
        assert_eq!(records[0].output, 3);
    }
}
