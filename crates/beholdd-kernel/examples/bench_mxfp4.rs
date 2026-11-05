//! Gate M0.4 -- the MXFP4 matvec, scalar definition against the vpermps path.
//!
//! SINGLE-THREADED on purpose: beholdd has no scheduler yet, and mixing an unbuilt
//! threading policy into a kernel measurement is how you end up attributing a 9.5x
//! threading effect to arithmetic. The claim here is the RATIO between two paths doing
//! identical work on one core.
//!
//! usage: cargo run -p beholdd-kernel --release --example bench_mxfp4
use std::time::Instant;

use beholdd_kernel::mxfp4;

fn fill(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

fn main() {
    let (rows, n_in, group) = (3072usize, 3584usize, 32usize);
    let reps = 7;

    let packed = fill(rows * n_in / 2, 0xBEE5);
    let scales: Vec<u8> = fill(rows * n_in / group, 0x5CA1)
        .into_iter()
        .map(|b| 118 + (b % 20))
        .collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
    let mut ys = vec![0.0f32; rows];
    let mut yv = vec![0.0f32; rows];

    let flop = 2.0 * rows as f64 * n_in as f64;
    let bytes = (rows * (n_in / 2 + n_in / group)) as f64;

    let mut best_s = f64::INFINITY;
    let mut best_v = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        mxfp4::matvec_scalar(&mut ys, &x, &packed, &scales, n_in, rows, group);
        best_s = best_s.min(t.elapsed().as_secs_f64());

        let t = Instant::now();
        mxfp4::matvec(&mut yv, &x, &packed, &scales, n_in, rows, group);
        best_v = best_v.min(t.elapsed().as_secs_f64());
    }

    println!("MXFP4 matvec [{rows} x {n_in}], group {group}, 1 thread, best of {reps}\n");
    println!(
        "  {:<34} {:>9} {:>10} {:>9}",
        "path", "ms", "GFLOP/s", "GB/s"
    );
    for (name, t) in [
        ("scalar definition", best_s),
        ("vpermps decode + f32 acc", best_v),
    ] {
        println!(
            "  {:<34} {:>9.2} {:>10.1} {:>9.1}",
            name,
            t * 1e3,
            flop / t / 1e9,
            bytes / t / 1e9
        );
    }
    println!("\n  speedup {:.2}x", best_s / best_v);

    // A benchmark that does not check its own output is a benchmark of the wrong thing.
    let rms = (ys.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / rows as f64).sqrt();
    let worst = ys
        .iter()
        .zip(&yv)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max);
    println!(
        "  agreement: worst {:.3e} against signal rms {:.3e} = {:.2e} relative",
        worst,
        rms,
        worst / rms
    );
}
