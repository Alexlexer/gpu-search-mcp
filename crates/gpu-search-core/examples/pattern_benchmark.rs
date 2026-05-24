use std::env;
use std::path::PathBuf;
use std::time::Instant;

use gpu_search_core::{IndexOptions, PatternSearchOptions, discover_files, search_files};
use serde::Serialize;

const DEFAULT_QUERIES: &[&str] = &[
    "handleError",
    "createTextModel",
    "addEventListener",
    "disposeOnReturn",
    "ICodeEditor",
];

#[derive(Debug)]
struct Args {
    repo: PathBuf,
    runs: usize,
    queries: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    implementation: &'static str,
    repo: String,
    runs: usize,
    indexed_files: usize,
    indexed_size_mb: f64,
    index_build_ms: f64,
    results: Vec<QueryResult>,
}

#[derive(Debug, Serialize)]
struct QueryResult {
    query: String,
    matches: usize,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn main() {
    let args = parse_args().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: cargo run --release --example pattern_benchmark -- --repo <path> [--runs 10] [--query needle]"
        );
        std::process::exit(2);
    });

    let index_start = Instant::now();
    let files = discover_files(&args.repo, &IndexOptions::default()).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
    let index_build_ms = index_start.elapsed().as_secs_f64() * 1000.0;
    let indexed_size_mb = files.iter().map(|file| file.size).sum::<u64>() as f64 / 1024.0 / 1024.0;

    let mut results = Vec::new();
    let options = PatternSearchOptions::default();
    for query in &args.queries {
        let mut timings_ms = Vec::with_capacity(args.runs);
        let mut matches = 0;
        for _ in 0..args.runs {
            let started = Instant::now();
            let found = search_files(&files, query, &options).unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(1);
            });
            timings_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            matches = found.len();
        }

        results.push(QueryResult {
            query: query.clone(),
            matches,
            p50_ms: percentile(&timings_ms, 0.50),
            p95_ms: percentile(&timings_ms, 0.95),
            p99_ms: percentile(&timings_ms, 0.99),
        });
    }

    let output = BenchmarkOutput {
        implementation: "rust-core-pattern",
        repo: args.repo.display().to_string(),
        runs: args.runs,
        indexed_files: files.len(),
        indexed_size_mb: round_one(indexed_size_mb),
        index_build_ms: round_two(index_build_ms),
        results,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("benchmark output should serialize")
    );
}

fn parse_args() -> Result<Args, String> {
    let mut repo = PathBuf::from(".");
    let mut runs = 10;
    let mut queries = Vec::new();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repo" => {
                repo = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--repo requires a path".to_string())?;
            }
            "--runs" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--runs requires a positive integer".to_string())?;
                runs = raw
                    .parse::<usize>()
                    .map_err(|_| "--runs requires a positive integer".to_string())?;
                if runs == 0 {
                    return Err("--runs must be greater than zero".to_string());
                }
            }
            "--query" => {
                queries.push(
                    args.next()
                        .ok_or_else(|| "--query requires a value".to_string())?,
                );
            }
            "--help" | "-h" => {
                return Err(String::new());
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if queries.is_empty() {
        queries = DEFAULT_QUERIES
            .iter()
            .map(|query| (*query).to_string())
            .collect();
    }

    Ok(Args {
        repo,
        runs,
        queries,
    })
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("timings should not be NaN"));
    let idx = (((sorted.len() - 1) as f64) * p).ceil() as usize;
    round_two(sorted[idx.min(sorted.len() - 1)])
}

fn round_one(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn round_two(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}
