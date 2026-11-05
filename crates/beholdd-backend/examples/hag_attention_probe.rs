//! Explicit finite-fixture RoPE and causal GQA gates. Never selected by serving.
//! CPU oracles are computed before native device access; external process, thermal
//! and crash-capture supervision is required. No host arithmetic fallback exists.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{ensure, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::{CommandList, ExecutionDevice, RecordedBatch};
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::{Device, IncrementalAttentionConfig, RopeLayout};
    use std::path::Path;

    const GUARD: f32 = -12345.5;
    const BUDGET: u64 = 1024 * 1024;
    const ROUNDS: usize = 3;
    const ATOL: f64 = 2e-5;
    const RTOL: f64 = 2e-5;

    fn config() -> NativeExecutionConfig {
        NativeExecutionConfig {
            weight_capacity_bytes: 4,
            tensor_capacity_bytes: 256 * 1024,
            auxiliary_capacity_bytes: 4096,
        }
    }

    fn check_submission(
        device: &NativeExecutionDevice,
        batch: &RecordedBatch,
        dispatches: u64,
    ) -> Result<()> {
        let before = device.stats()?;
        let submission = device.submit(batch)?;
        device.wait(&submission)?;
        let after = device.stats()?;
        ensure!(
            after.dispatch_count == before.dispatch_count + dispatches
                && after.dispatch_attempts == before.dispatch_attempts + dispatches
                && after.logical_submissions == before.logical_submissions + 1,
            "dispatch accounting differs"
        );
        ensure!(
            after.host_read_bytes == before.host_read_bytes
                && after.host_write_bytes == before.host_write_bytes
                && after.uploaded_weight_bytes == before.uploaded_weight_bytes
                && after.allocation_bytes == before.allocation_bytes
                && after.weight_live_bytes == before.weight_live_bytes
                && after.tensor_live_bytes == before.tensor_live_bytes
                && after.auxiliary_live_bytes == before.auxiliary_live_bytes
                && after.weight_peak_bytes == before.weight_peak_bytes
                && after.tensor_peak_bytes == before.tensor_peak_bytes
                && after.auxiliary_peak_bytes == before.auxiliary_peak_bytes
                && !after.poisoned,
            "submission transferred payloads, grew resources or poisoned the queue"
        );
        Ok(())
    }

    fn retired(device: &NativeExecutionDevice) -> Result<()> {
        let stats = device.stats()?;
        ensure!(
            stats.weight_live_bytes == 0
                && stats.tensor_live_bytes == 0
                && stats.auxiliary_live_bytes == 0
                && !stats.poisoned,
            "case retained resources"
        );
        Ok(())
    }

    fn compare(actual: &[f32], expected: &[f32], exact: bool) -> Result<f64> {
        ensure!(actual.len() == expected.len() + 2, "guarded extent differs");
        ensure!(
            actual[0].to_bits() == GUARD.to_bits()
                && actual[actual.len() - 1].to_bits() == GUARD.to_bits(),
            "output guard changed"
        );
        let mut maximum = 0.0_f64;
        for (index, (&a, &b)) in actual[1..actual.len() - 1].iter().zip(expected).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            ensure!(a.is_finite() && b.is_finite(), "nonfinite output {index}");
            ensure!(
                a.is_sign_negative() == b.is_sign_negative() && (a == 0.0) == (b == 0.0),
                "sign/zero mismatch at {index}: {a:?} != {b:?}"
            );
            ensure!(
                if exact {
                    a.to_bits() == b.to_bits()
                } else {
                    error <= ATOL + RTOL * f64::from(b).abs()
                },
                "numerical mismatch at {index}: {a:?} != {b:?}, error={error}"
            );
            maximum = maximum.max(error);
        }
        Ok(maximum)
    }

    fn guarded(values: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(values.len() + 2);
        result.push(GUARD);
        result.extend_from_slice(values);
        result.push(GUARD);
        result
    }

    type FixtureRound = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    struct RopeFixture {
        layout: RopeLayout,
        inverse: bool,
        rounds: Vec<FixtureRound>,
    }

    fn rope_fixtures() -> Result<Vec<RopeFixture>> {
        let cpu = Cpu::new();
        let mut fixtures = vec![RopeFixture {
            layout: RopeLayout::Interleaved,
            inverse: false,
            rounds: vec![(vec![2.0, 3.0], vec![0.5], vec![0.25], vec![0.25, 2.0])],
        }];
        for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
            for inverse in [false, true] {
                let mut rounds = Vec::new();
                for round in 0..ROUNDS {
                    let x: Vec<_> = (0..130)
                        .map(|i| ((i * 7 + round * 3) % 29) as f32 / 8.0 - 1.75)
                        .collect();
                    let cos: Vec<_> = (0..65)
                        .map(|i| ((i * 3 + round) % 17) as f32 / 8.0 - 1.0)
                        .collect();
                    let sin: Vec<_> = (0..65)
                        .map(|i| ((i * 5 + round * 7) % 19) as f32 / 16.0 - 0.5)
                        .collect();
                    let mut expected = x.clone();
                    cpu.rope(&mut expected, &cos, &sin, layout, inverse)?;
                    rounds.push((x, cos, sin, expected));
                }
                fixtures.push(RopeFixture {
                    layout,
                    inverse,
                    rounds,
                });
            }
        }
        Ok(fixtures)
    }

    struct AttentionFixture {
        config: IncrementalAttentionConfig,
        rounds: Vec<FixtureRound>,
    }

    fn attention_fixtures() -> Result<Vec<AttentionFixture>> {
        let cpu = Cpu::new();
        let mut fixtures = vec![AttentionFixture {
            config: IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: 0,
                n_kv_positions: 1,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                scale: 1.0,
            },
            rounds: vec![(
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
                vec![5.0, 6.0],
            )],
        }];
        for (queries, start, positions, dim) in [(3, 0, 4, 4), (2, 63, 66, 128)] {
            let config = IncrementalAttentionConfig {
                n_query_positions: queries,
                query_start: start,
                n_kv_positions: positions,
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: dim,
                scale: 1.0 / (dim as f32).sqrt(),
            };
            let mut rounds = Vec::new();
            for round in 0..ROUNDS {
                let q: Vec<_> = (0..queries * 4 * dim)
                    .map(|i| ((i * 7 + round * 3) % 31) as f32 / 64.0 - 0.234375)
                    .collect();
                let mut k: Vec<_> = (0..positions * 2 * dim)
                    .map(|i| ((i * 11 + round * 5) % 37) as f32 / 32.0 - 0.5)
                    .collect();
                let mut v: Vec<_> = (0..positions * 2 * dim)
                    .map(|i| ((i * 13 + round * 7) % 41) as f32 / 32.0 + 0.125)
                    .collect();
                // This future position is inside the declared cache but outside every
                // query's causal prefix. Accidental reads must fail numerical checks.
                k[(positions - 1) * 2 * dim..].fill(f32::NAN);
                v[(positions - 1) * 2 * dim..].fill(f32::NAN);
                let mut expected = vec![0.0; q.len()];
                cpu.incremental_causal_attention(&mut expected, &q, &k, &v, config)?;
                ensure!(expected.iter().all(|x| x.is_finite()), "nonfinite oracle");
                rounds.push((q, k, v, expected));
            }
            fixtures.push(AttentionFixture { config, rounds });
        }
        Ok(fixtures)
    }

    struct Expectations {
        execution: &'static str,
        dispatches: u64,
        submissions: u64,
        checked: usize,
        guards: usize,
        exact: bool,
    }

    fn finish(
        device: NativeExecutionDevice,
        budget: MemoryBudget,
        expected: u64,
        specification: Expectations,
        error: f64,
    ) -> Result<()> {
        let Expectations {
            execution,
            dispatches,
            submissions,
            checked,
            guards,
            exact,
        } = specification;
        retired(&device)?;
        let stats: NativeExecutionStats = device.stats()?;
        ensure!(
            stats.allocation_bytes == expected
                && budget.current_bytes() == expected
                && stats.dispatch_count == dispatches
                && stats.dispatch_attempts == dispatches
                && stats.logical_submissions == submissions
                && stats.uploaded_weight_bytes == 0,
            "final accounting differs"
        );
        ensure!(
            stats.host_read_bytes == ((checked + guards) * 4) as u64,
            "readback traffic differs"
        );
        drop(device);
        ensure!(budget.current_bytes() == 0, "HAG charge retained");
        println!("{{\"execution\":\"{execution}\",\"physical_dispatches\":{dispatches},\"logical_submissions\":{submissions},\"checked_values\":{checked},\"guard_checks\":{guards},\"max_absolute_error\":{error},\"exact\":{exact},\"absolute_tolerance\":{},\"relative_tolerance\":{},\"actual_hag_allocation_bytes\":{expected},\"expected_hag_allocation_bytes\":{expected},\"hag_budget_bytes\":{BUDGET},\"verification_readback_bytes\":{},\"submission_tensor_transfer_bytes\":0,\"budget_after\":0,\"required_hag_trace_submission\":\"0\",\"hag_trace_requirement_verified\":true}}", if exact {0.0} else {ATOL}, if exact {0.0} else {RTOL}, stats.host_read_bytes);
        Ok(())
    }

    pub(super) fn run(path: &Path, rope: bool) -> Result<()> {
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(std::ffi::OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly 0 before native access"
        );
        // Compute every CPU oracle before opening native resources.
        let rope_cases = if rope { rope_fixtures()? } else { Vec::new() };
        let attention_cases = if rope {
            Vec::new()
        } else {
            attention_fixtures()?
        };
        let expected = config().allocation_bytes()?;
        ensure!(expected <= BUDGET, "queue exceeds diagnostic HAG budget");
        let budget = MemoryBudget::new(BUDGET);
        let device = NativeExecutionDevice::open(path, &budget, config())?;
        let initial = device.stats()?;
        ensure!(
            initial.allocation_bytes == expected
                && initial.dispatch_attempts == 0
                && initial.logical_submissions == 0
                && !initial.poisoned,
            "unexpected initial state"
        );
        let mut maximum = 0.0_f64;
        if rope {
            for (case, fixture) in rope_cases.iter().enumerate() {
                let width = fixture.rounds[0].0.len();
                let x = device.allocate_f32(width + 2)?;
                let cos = device.allocate_f32(width / 2)?;
                let sin = device.allocate_f32(width / 2)?;
                let mut commands = CommandList::new();
                commands.rope(
                    &x.view(1, width)?,
                    &cos.full_view(),
                    &sin.full_view(),
                    fixture.layout,
                    fixture.inverse,
                )?;
                let batch = device.record(&commands)?;
                drop(commands);
                for (round, (values, c, s, expected)) in fixture.rounds.iter().enumerate() {
                    device.write_f32(&x.full_view(), &guarded(values))?;
                    device.write_f32(&cos.full_view(), c)?;
                    device.write_f32(&sin.full_view(), s)?;
                    check_submission(&device, &batch, 1)?;
                    let mut actual = vec![f32::NAN; width + 2];
                    device.read_f32(&x.full_view(), &mut actual)?;
                    maximum = maximum.max(compare(&actual, expected, true)?);
                    println!("PASS rope case={case} round={round} checked_values={width} guards=2");
                }
                drop((batch, x, cos, sin));
                retired(&device)?;
            }
            finish(
                device,
                budget,
                expected,
                Expectations {
                    execution: "native-hag-rope-diagnostic",
                    dispatches: 13,
                    submissions: 13,
                    checked: 1562,
                    guards: 26,
                    exact: true,
                },
                maximum,
            )
        } else {
            for (case, fixture) in attention_cases.iter().enumerate() {
                let width = fixture.rounds[0].0.len();
                let q = device.allocate_f32(width)?;
                let k = device.allocate_f32(fixture.rounds[0].1.len())?;
                let v = device.allocate_f32(fixture.rounds[0].2.len())?;
                let out = device.allocate_f32(width + 2)?;
                let mut commands = CommandList::new();
                commands.incremental_causal_attention(
                    &out.view(1, width)?,
                    &q.full_view(),
                    &k.full_view(),
                    &v.full_view(),
                    fixture.config,
                )?;
                let batch = device.record(&commands)?;
                drop(commands);
                let dispatches =
                    (fixture.config.n_query_positions * fixture.config.n_heads * 3) as u64;
                for (round, (queries, keys, values, expected)) in fixture.rounds.iter().enumerate()
                {
                    device.write_f32(&q.full_view(), queries)?;
                    device.write_f32(&k.full_view(), keys)?;
                    device.write_f32(&v.full_view(), values)?;
                    device.write_f32(&out.full_view(), &guarded(&vec![f32::NAN; width]))?;
                    check_submission(&device, &batch, dispatches)?;
                    let mut actual = vec![f32::NAN; width + 2];
                    device.read_f32(&out.full_view(), &mut actual)?;
                    maximum = maximum.max(compare(&actual, expected, case == 0)?);
                    println!(
                        "PASS attention case={case} round={round} checked_values={width} guards=2"
                    );
                }
                drop((batch, q, k, v, out));
                retired(&device)?;
            }
            finish(
                device,
                budget,
                expected,
                Expectations {
                    execution: "native-hag-attention-diagnostic",
                    dispatches: 183,
                    submissions: 7,
                    checked: 3218,
                    guards: 14,
                    exact: false,
                },
                maximum,
            )
        }
    }

    mod context2048 {
        include!("hag_attention_probe/context2048.rs");
    }

    pub(super) fn run_context2048(path: &Path) -> Result<()> {
        context2048::run(path)
    }

    mod context16384 {
        include!("hag_attention_probe/context16384.rs");
    }

    pub(super) fn run_context16384(path: &Path) -> Result<()> {
        context16384::run(path)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn diagnostic_geometries_counters_and_changed_replays_are_preregistered() {
            let ropes = rope_fixtures().unwrap();
            assert_eq!(ropes.iter().map(|f| f.rounds.len()).sum::<usize>(), 13);
            assert_eq!(
                ropes
                    .iter()
                    .flat_map(|f| &f.rounds)
                    .map(|r| r.3.len())
                    .sum::<usize>(),
                1562
            );
            let attention = attention_fixtures().unwrap();
            assert_eq!(attention.iter().map(|f| f.rounds.len()).sum::<usize>(), 7);
            assert_eq!(
                attention
                    .iter()
                    .flat_map(|f| &f.rounds)
                    .map(|r| r.3.len())
                    .sum::<usize>(),
                3218
            );
            assert_eq!(
                attention
                    .iter()
                    .map(|f| f.rounds.len() * f.config.n_query_positions * f.config.n_heads * 3)
                    .sum::<usize>(),
                183
            );
            for rounds in ropes
                .iter()
                .map(|f| &f.rounds)
                .chain(attention.iter().map(|f| &f.rounds))
                .filter(|r| r.len() > 1)
            {
                assert_ne!(rounds[0].3, rounds[1].3);
                assert_ne!(rounds[1].3, rounds[2].3);
            }
        }
        #[test]
        fn future_cache_poison_is_causally_excluded_and_first_fixture_is_analytic() {
            for fixture in attention_fixtures().unwrap() {
                for (q, k, v, expected) in fixture.rounds {
                    let mut cpu = vec![0.0; q.len()];
                    Cpu::new()
                        .incremental_causal_attention(&mut cpu, &q, &k, &v, fixture.config)
                        .unwrap();
                    assert_eq!(cpu, expected);
                    assert!(cpu.iter().all(|x| x.is_finite()));
                }
            }
            let mut x = [2.0, 3.0];
            Cpu::new()
                .rope(&mut x, &[0.5], &[0.25], RopeLayout::Interleaved, false)
                .unwrap();
            assert_eq!(x, [0.25, 2.0]);
        }
        #[test]
        fn guarded_comparison_refuses_stale_nonfinite_sign_zero_and_extent_errors() {
            assert!(compare(&guarded(&[1.0]), &[1.0], true).is_ok());
            for actual in [
                vec![GUARD, 2.0, GUARD],
                vec![GUARD, f32::NAN, GUARD],
                vec![GUARD, -1.0, GUARD],
                vec![GUARD, 1.0, 0.0],
                vec![1.0],
            ] {
                assert!(compare(&actual, &[1.0], false).is_err());
            }
            assert!(compare(&guarded(&[-0.0]), &[0.0], false).is_err());
            assert!(compare(&guarded(&[0.0]), &[1e-8], false).is_err());
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.is_empty() || arguments == ["--help"] || arguments == ["-h"] {
        println!(
            "Usage: hag_attention_probe --diagnose-rope|--diagnose-attention|--diagnose-attention2048|--diagnose-attention16384 /dev/dri/renderD128"
        );
        println!("Requires HAG_TRACE_SUBMISSION=0 and external process/thermal/crash supervision.");
        return Ok(());
    }
    let [flag, path] = arguments.as_slice() else {
        anyhow::bail!("explicit diagnostic mode and device path required");
    };
    if flag == "--diagnose-attention2048" {
        return proof::run_context2048(std::path::Path::new(path));
    }
    if flag == "--diagnose-attention16384" {
        return proof::run_context16384(std::path::Path::new(path));
    }
    let rope = match flag.to_str() {
        Some("--diagnose-rope") => true,
        Some("--diagnose-attention") => false,
        _ => anyhow::bail!("unknown diagnostic mode"),
    };
    proof::run(std::path::Path::new(path), rope)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native diagnostics require Linux");
    std::process::exit(1);
}
