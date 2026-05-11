//! Ad-hoc timing harness: pull one HF repo and print wall-clock +
//! throughput. Useful when tuning `GENIEX_DL_*` knobs, sanity-checking
//! proxy setup, or eyeballing throughput after a transport change.
//!
//! Usage (from the workspace root):
//!
//!   cargo run --release --example bench_pull -- <repo> [<quant>]
//!
//! `<repo>` is an HF "org/repo" string; `<quant>` narrows the manifest
//! inference to a single quantization (e.g. `Q4_K_M`) so small runs
//! don't fetch every GGUF in the repo.
//!
//! Env vars honoured:
//!   GENIEX_DATADIR                    Output directory (defaults to a
//!                                     temp dir that is wiped on start).
//!   GENIEX_HFTOKEN                    HF bearer token (optional).
//!   GENIEX_DL_FILE_CONCURRENCY        Executor tuning knobs.
//!   GENIEX_DL_CHUNK_CONCURRENCY
//!   GENIEX_DL_CHUNK_SIZE
//!   HTTPS_PROXY / HTTP_PROXY / NO_PROXY   Read by reqwest automatically.
//!
//! Not a benchmark suite — no statistics, no warm-up, no locked-in
//! baseline. It just prints numbers so you can decide for yourself.

use std::path::PathBuf;
use std::time::Instant;

use model_manager_core::config::StoreConfig;
use model_manager_core::executor::{FileProgress, ProgressCallback};
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::{pull, PullIntent, PullRequest};
use model_manager_core::store::Store;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bench_pull <repo> [<quant>]");
        std::process::exit(2);
    }
    let repo = args[1].clone();
    let quant = args.get(2).cloned();

    let dest_dir: PathBuf = std::env::var("GENIEX_DATADIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let t = std::env::temp_dir().join("geniex_bench");
            let _ = std::fs::remove_dir_all(&t);
            std::fs::create_dir_all(&t).unwrap();
            t
        });
    std::fs::create_dir_all(&dest_dir).unwrap();

    let token = std::env::var("GENIEX_HFTOKEN").ok();
    let store = Store::new(StoreConfig::new(dest_dir.clone())).expect("build store");

    let start = Instant::now();
    let last_print = std::sync::Mutex::new(Instant::now());
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| -> bool {
        let mut lp = last_print.lock().unwrap();
        if lp.elapsed().as_millis() < 500 {
            return true;
        }
        *lp = Instant::now();
        let elapsed = start.elapsed().as_secs_f64();
        let done: i64 = files.iter().map(|f| f.downloaded_bytes).sum();
        let total: i64 = files.iter().map(|f| f.total_bytes.max(0)).sum();
        let mbps = if elapsed > 0.0 {
            (done as f64) / 1_048_576.0 / elapsed
        } else {
            0.0
        };
        eprintln!(
            "  [{:5.1}s] {:>10} / {:>10} bytes  ({:5.1} MiB/s)",
            elapsed, done, total, mbps
        );
        true
    });

    eprintln!("# bench_pull: repo={repo} quant={quant:?}");
    eprintln!("# dest_dir={}", dest_dir.display());

    let req = PullRequest {
        model_name: repo.clone(),
        intent: PullIntent::HuggingFace {
            repo: repo.clone(),
            token,
        },
        on_progress: Some(cb),
        hint: ManifestHint {
            quant,
            ..ManifestHint::default()
        },
    };
    let t0 = Instant::now();
    pull(&store, req).await.expect("pull failed");
    let elapsed = t0.elapsed();

    let secs = elapsed.as_secs_f64();
    println!();
    println!("=== RESULT ===");
    println!("elapsed_s  : {secs:.2}");
}
