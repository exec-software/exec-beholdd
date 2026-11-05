//! Explicit bounded native-kernel qualification; never a serving startup action.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use beholdd_backend::hag::linear_probe;
    use exec_hag::budget::MemoryBudget;
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let [flag, path] = arguments.as_slice() else {
        return Err(
            "usage: hag_linear_probe --execute|--diagnose-first /dev/dri/renderD128".into(),
        );
    };
    if flag != "--execute" && flag != "--diagnose-first" {
        return Err("GPU execution requires explicit --execute or --diagnose-first".into());
    }
    let budget = MemoryBudget::new(256 * 1024);
    let report = if flag == "--diagnose-first" {
        linear_probe::diagnose_first(path, &budget)?
    } else {
        linear_probe::run(path, &budget)?
    };
    if budget.current_bytes() != 0 {
        return Err("native linear proof left charged allocations".into());
    }
    println!("PASS native packed linear: cases={} dispatches={} checked_rows={} peak_queue_bytes={} max_cpu_absolute_error={} budget_after=0",
        report.cases, report.dispatches, report.checked_rows, report.peak_queue_bytes,
        report.max_cpu_absolute_error);
    if flag == "--diagnose-first" {
        println!("Single BF16 diagnostic only. Q8_0 and queue reuse are not qualified.");
    } else {
        println!(
            "Serial BF16/Q8_0 qualification only. No serving registration or throughput claim."
        );
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native linear qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
