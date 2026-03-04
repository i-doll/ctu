use std::collections::BTreeMap;
use std::io::Write;

use clap::Parser;
use serde_json::json;

use ctu::{DeduplicatedRecords, PeriodAgg, fmt_num, get_cost, get_log_dirs, find_jsonl_files};

#[derive(Parser)]
#[command(name = "ctu", version = "1.0.0", about = "Claude token usage scanner")]
struct Cli {
    /// Show daily breakdown (default)
    #[arg(short = 'd', long)]
    daily: bool,

    /// Include per-model breakdown
    #[arg(short = 'm', long = "by-model")]
    by_model: bool,

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

fn main() {
    let cli = Cli::parse();

    // Determine display mode
    let show_total = cli.total;
    let show_daily = !show_total || cli.daily;

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

    eprintln!("Scanning {} JSONL files...", files.len());

    // Aggregate
    let mut daily: BTreeMap<String, PeriodAgg> = BTreeMap::new();
    let mut by_model: BTreeMap<(String, String), PeriodAgg> = BTreeMap::new();
    let mut total = PeriodAgg::default();

    let data = DeduplicatedRecords::collect(&files);
    for rec in &data.records {
        let date = &rec.timestamp[..10];

        // Date filters
        if let Some(ref s) = cli.since { if date < s.as_str() { continue; } }
        if let Some(ref u) = cli.until { if date > u.as_str() { continue; } }

        let cost = get_cost(&rec.model, rec.input, rec.output, rec.cache_create, rec.cache_read);

        let day = daily.entry(date.to_string()).or_default();
        day.input        += rec.input;
        day.output       += rec.output;
        day.cache_create += rec.cache_create;
        day.cache_read   += rec.cache_read;
        day.cost         += cost;

        let dm = by_model.entry((date.to_string(), rec.model.clone())).or_default();
        dm.input        += rec.input;
        dm.output       += rec.output;
        dm.cache_create += rec.cache_create;
        dm.cache_read   += rec.cache_read;
        dm.cost         += cost;

        total.input        += rec.input;
        total.output       += rec.output;
        total.cache_create += rec.cache_create;
        total.cache_read   += rec.cache_read;
        total.cost         += cost;
    }

    if cli.json {
        print_json(&daily, &by_model, &total, cli.by_model);
    } else {
        print_text(&daily, &by_model, &total, show_daily, show_total, cli.by_model);
    }
}

fn print_text(
    daily:    &BTreeMap<String, PeriodAgg>,
    by_model: &BTreeMap<(String, String), PeriodAgg>,
    total:    &PeriodAgg,
    show_daily: bool,
    show_total: bool,
    show_model: bool,
) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if show_daily {
        writeln!(out, "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Date", "Input", "Output", "Cache-R", "Cache-C", "Total", "Cost").unwrap();
        writeln!(out, "{}", "─".repeat(70)).unwrap();

        for (date, agg) in daily {
            let t = agg.input + agg.output + agg.cache_create + agg.cache_read;
            writeln!(out, "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.2}",
                date,
                fmt_num(agg.input), fmt_num(agg.output),
                fmt_num(agg.cache_read), fmt_num(agg.cache_create),
                fmt_num(t), agg.cost).unwrap();
        }

        writeln!(out, "{}", "─".repeat(70)).unwrap();
    }

    if show_model {
        writeln!(out, "\nPer-Model Breakdown:").unwrap();
        writeln!(out, "{:<12} {:<28} {:>8} {:>8} {:>10}", "Date", "Model", "Input", "Output", "Cost").unwrap();
        writeln!(out, "{}", "─".repeat(70)).unwrap();

        for ((date, model), agg) in by_model {
            let m = if model.len() > 26 { format!("{}...", &model[..23]) } else { model.clone() };
            writeln!(out, "{:<12} {:<28} {:>8} {:>8} {:>9.2}",
                date, m, fmt_num(agg.input), fmt_num(agg.output), agg.cost).unwrap();
        }
        writeln!(out, "{}", "─".repeat(70)).unwrap();
    }

    if show_daily || show_total {
        let t = total.input + total.output + total.cache_create + total.cache_read;
        writeln!(out, "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.2}",
            "TOTAL",
            fmt_num(total.input), fmt_num(total.output),
            fmt_num(total.cache_read), fmt_num(total.cache_create),
            fmt_num(t), total.cost).unwrap();
    }

    if show_total && !show_daily {
        let t = total.input + total.output + total.cache_create + total.cache_read;
        writeln!(out, "\nTotal Token Usage:").unwrap();
        writeln!(out, "  Input tokens:         {:>12}", fmt_num(total.input)).unwrap();
        writeln!(out, "  Output tokens:        {:>12}", fmt_num(total.output)).unwrap();
        writeln!(out, "  Cache read tokens:    {:>12}", fmt_num(total.cache_read)).unwrap();
        writeln!(out, "  Cache create tokens:  {:>12}", fmt_num(total.cache_create)).unwrap();
        writeln!(out, "  {}", "─".repeat(33)).unwrap();
        writeln!(out, "  Total tokens:         {:>12}", fmt_num(t)).unwrap();
        writeln!(out, "  Estimated cost:       ${:>11.2}", total.cost).unwrap();
    }
}

fn print_json(
    daily:    &BTreeMap<String, PeriodAgg>,
    by_model: &BTreeMap<(String, String), PeriodAgg>,
    total:    &PeriodAgg,
    show_model: bool,
) {
    let daily_arr: Vec<_> = daily.iter().map(|(date, agg)| {
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
    }).collect();

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

    if show_model {
        let model_arr: Vec<_> = by_model.iter().map(|((date, model), agg)| {
            json!({
                "date": date,
                "model": model,
                "input": agg.input,
                "output": agg.output,
                "cache_read": agg.cache_read,
                "cache_create": agg.cache_create,
                "cost_usd": (agg.cost * 10000.0).round() / 10000.0
            })
        }).collect();
        root.insert("by_model".into(), json!(model_arr));
    }

    root.insert("total".into(), total_obj);
    println!("{}", serde_json::to_string_pretty(&root).unwrap());
}
