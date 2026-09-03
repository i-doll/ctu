use std::collections::BTreeMap;
use std::io::Write;

use clap::Parser;
use serde_json::json;

use ctu::{
    find_jsonl_files, fmt_num, get_log_dirs, get_opencode_files, get_record_cost,
    DeduplicatedRecords, PeriodAgg,
};

#[derive(Parser)]
#[command(
    name = "ctu",
    version,
    about = "Claude, Codex, OpenCode, and Pi token usage scanner"
)]
struct Cli {
    /// Show daily breakdown (default)
    #[arg(short = 'd', long)]
    daily: bool,

    /// Include per-model breakdown
    #[arg(short = 'm', long = "by-model")]
    by_model: bool,

    /// Include per-provider breakdown by day
    #[arg(short = 'p', long = "by-provider")]
    by_provider: bool,

    /// Show only total summary
    #[arg(short = 't', long)]
    total: bool,

    /// Filter from date (YYYY-MM-DD)
    #[arg(short = 's', long)]
    since: Option<String>,

    /// Filter until date (YYYY-MM-DD)
    #[arg(short = 'u', long)]
    until: Option<String>,

    /// Output as JSON
    #[arg(short = 'j', long)]
    json: bool,
}

#[derive(Clone, Copy)]
struct OutputOptions {
    daily: bool,
    total: bool,
    model: bool,
    provider: bool,
}

fn main() {
    let cli = Cli::parse();

    // Determine display mode
    let show_total = cli.total;
    let show_daily = !show_total || cli.daily;

    let dirs = get_log_dirs();
    let opencode_files = get_opencode_files();
    if dirs.is_empty() && opencode_files.is_empty() {
        eprintln!("No Claude, Codex, OpenCode, or Pi usage data found.");
        std::process::exit(1);
    }

    let files = find_jsonl_files(&dirs);
    if files.is_empty() && opencode_files.is_empty() {
        eprintln!("No usage records found.");
        std::process::exit(1);
    }

    eprintln!(
        "Scanning {} JSONL files and {} OpenCode sources...",
        files.len(),
        opencode_files.len()
    );

    // Aggregate
    let mut daily: BTreeMap<String, PeriodAgg> = BTreeMap::new();
    let mut by_model: BTreeMap<(String, String), PeriodAgg> = BTreeMap::new();
    let mut by_provider: BTreeMap<(String, String), PeriodAgg> = BTreeMap::new();
    let mut total = PeriodAgg::default();

    let data = DeduplicatedRecords::collect_with_opencode(&files, &opencode_files);
    for rec in &data.records {
        let Some(date) = rec.timestamp.get(..10) else {
            continue;
        };

        // Date filters
        if let Some(ref s) = cli.since {
            if date < s.as_str() {
                continue;
            }
        }
        if let Some(ref u) = cli.until {
            if date > u.as_str() {
                continue;
            }
        }

        let cost = get_record_cost(rec);

        let day = daily.entry(date.to_string()).or_default();
        day.input += rec.input;
        day.output += rec.output;
        day.cache_create += rec.cache_create;
        day.cache_read += rec.cache_read;
        day.cost += cost;

        let dm = by_model
            .entry((date.to_string(), rec.model.clone()))
            .or_default();
        dm.input += rec.input;
        dm.output += rec.output;
        dm.cache_create += rec.cache_create;
        dm.cache_read += rec.cache_read;
        dm.cost += cost;

        let dp = by_provider
            .entry((date.to_string(), rec.provider.clone()))
            .or_default();
        dp.input += rec.input;
        dp.output += rec.output;
        dp.cache_create += rec.cache_create;
        dp.cache_read += rec.cache_read;
        dp.cost += cost;

        total.input += rec.input;
        total.output += rec.output;
        total.cache_create += rec.cache_create;
        total.cache_read += rec.cache_read;
        total.cost += cost;
    }

    let options = OutputOptions {
        daily: show_daily,
        total: show_total,
        model: cli.by_model,
        provider: cli.by_provider,
    };

    if cli.json {
        print_json(&daily, &by_model, &by_provider, &total, options);
    } else {
        print_text(&daily, &by_model, &by_provider, &total, options);
    }
}

fn print_text(
    daily: &BTreeMap<String, PeriodAgg>,
    by_model: &BTreeMap<(String, String), PeriodAgg>,
    by_provider: &BTreeMap<(String, String), PeriodAgg>,
    total: &PeriodAgg,
    options: OutputOptions,
) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if options.daily {
        writeln!(
            out,
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Date", "Input", "Output", "Cache-R", "Cache-C", "Total", "Cost"
        )
        .unwrap();
        writeln!(out, "{}", "─".repeat(70)).unwrap();

        for (date, agg) in daily {
            let t = agg.input + agg.output + agg.cache_create + agg.cache_read;
            writeln!(
                out,
                "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.2}",
                date,
                fmt_num(agg.input),
                fmt_num(agg.output),
                fmt_num(agg.cache_read),
                fmt_num(agg.cache_create),
                fmt_num(t),
                agg.cost
            )
            .unwrap();
        }

        writeln!(out, "{}", "─".repeat(70)).unwrap();
    }

    if options.model {
        writeln!(out, "\nPer-Model Breakdown:").unwrap();
        writeln!(
            out,
            "{:<12} {:<28} {:>8} {:>8} {:>10}",
            "Date", "Model", "Input", "Output", "Cost"
        )
        .unwrap();
        writeln!(out, "{}", "─".repeat(70)).unwrap();

        for ((date, model), agg) in by_model {
            let m = if model.len() > 26 {
                format!("{}...", &model[..23])
            } else {
                model.clone()
            };
            writeln!(
                out,
                "{:<12} {:<28} {:>8} {:>8} {:>9.2}",
                date,
                m,
                fmt_num(agg.input),
                fmt_num(agg.output),
                agg.cost
            )
            .unwrap();
        }
        writeln!(out, "{}", "─".repeat(70)).unwrap();
    }

    if options.provider {
        writeln!(out, "\nPer-Provider Breakdown:").unwrap();
        writeln!(
            out,
            "{:<12} {:<16} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Date", "Provider", "Input", "Output", "Cache-R", "Cache-C", "Cost"
        )
        .unwrap();
        writeln!(out, "{}", "─".repeat(86)).unwrap();

        for ((date, provider), agg) in by_provider {
            writeln!(
                out,
                "{:<12} {:<16} {:>8} {:>8} {:>8} {:>8} {:>9.2}",
                date,
                provider,
                fmt_num(agg.input),
                fmt_num(agg.output),
                fmt_num(agg.cache_read),
                fmt_num(agg.cache_create),
                agg.cost
            )
            .unwrap();
        }
        writeln!(out, "{}", "─".repeat(86)).unwrap();
    }

    if options.daily || options.total {
        let t = total.input + total.output + total.cache_create + total.cache_read;
        writeln!(
            out,
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.2}",
            "TOTAL",
            fmt_num(total.input),
            fmt_num(total.output),
            fmt_num(total.cache_read),
            fmt_num(total.cache_create),
            fmt_num(t),
            total.cost
        )
        .unwrap();
    }

    if options.total && !options.daily {
        let t = total.input + total.output + total.cache_create + total.cache_read;
        writeln!(out, "\nTotal Token Usage:").unwrap();
        writeln!(out, "  Input tokens:         {:>12}", fmt_num(total.input)).unwrap();
        writeln!(out, "  Output tokens:        {:>12}", fmt_num(total.output)).unwrap();
        writeln!(
            out,
            "  Cache read tokens:    {:>12}",
            fmt_num(total.cache_read)
        )
        .unwrap();
        writeln!(
            out,
            "  Cache create tokens:  {:>12}",
            fmt_num(total.cache_create)
        )
        .unwrap();
        writeln!(out, "  {}", "─".repeat(33)).unwrap();
        writeln!(out, "  Total tokens:         {:>12}", fmt_num(t)).unwrap();
        writeln!(out, "  Estimated cost:       ${:>11.2}", total.cost).unwrap();
    }
}

fn print_json(
    daily: &BTreeMap<String, PeriodAgg>,
    by_model: &BTreeMap<(String, String), PeriodAgg>,
    by_provider: &BTreeMap<(String, String), PeriodAgg>,
    total: &PeriodAgg,
    options: OutputOptions,
) {
    let daily_arr: Vec<_> = daily
        .iter()
        .map(|(date, agg)| {
            let t = agg.input + agg.output + agg.cache_create + agg.cache_read;
            json!({
                "date": date,
                "input": agg.input,
                "output": agg.output,
                "cache_read": agg.cache_read,
                "cache_create": agg.cache_create,
                "total": t,
                "cost_usd": (agg.cost * 10000.0).round() / 10000.0
            })
        })
        .collect();

    let t = total.input + total.output + total.cache_create + total.cache_read;
    let total_obj = json!({
        "input": total.input,
        "output": total.output,
        "cache_read": total.cache_read,
        "cache_create": total.cache_create,
        "total": t,
        "cost_usd": (total.cost * 10000.0).round() / 10000.0
    });

    let mut root = serde_json::Map::new();
    root.insert("daily".into(), json!(daily_arr));

    if options.model {
        let model_arr: Vec<_> = by_model
            .iter()
            .map(|((date, model), agg)| {
                json!({
                    "date": date,
                    "model": model,
                    "input": agg.input,
                    "output": agg.output,
                    "cache_read": agg.cache_read,
                    "cache_create": agg.cache_create,
                    "cost_usd": (agg.cost * 10000.0).round() / 10000.0
                })
            })
            .collect();
        root.insert("by_model".into(), json!(model_arr));
    }

    if options.provider {
        let provider_arr: Vec<_> = by_provider
            .iter()
            .map(|((date, provider), agg)| {
                json!({
                    "date": date,
                    "provider": provider,
                    "input": agg.input,
                    "output": agg.output,
                    "cache_read": agg.cache_read,
                    "cache_create": agg.cache_create,
                    "cost_usd": (agg.cost * 10000.0).round() / 10000.0
                })
            })
            .collect();
        root.insert("by_provider".into(), json!(provider_arr));
    }

    root.insert("total".into(), total_obj);
    println!("{}", serde_json::to_string_pretty(&root).unwrap());
}
