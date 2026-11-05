// Additive attention-only proof. Never opens a checkpoint or changes serving.
use super::*;

const PREFIXES: [usize; 8] = [511, 512, 513, 1023, 1024, 1025, 2047, 2048];
const CAPACITY: usize = 2048;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 128;
const REPLAYS: usize = 2;
const LIMIT: u64 = 16 * 1024 * 1024;
const SUBMISSIONS: u64 = (PREFIXES.len() * REPLAYS) as u64;
const DISPATCHES: u64 = SUBMISSIONS * HEADS as u64 * 3;
const VALUES: usize = PREFIXES.len() * REPLAYS * HEADS * DIM;
const GUARDS: usize = PREFIXES.len() * REPLAYS * 8;

fn configuration() -> NativeExecutionConfig {
    NativeExecutionConfig {
        weight_capacity_bytes: 4,
        tensor_capacity_bytes: 8 * 1024 * 1024,
        auxiliary_capacity_bytes: 64 * 1024,
    }
}

fn fixtures() -> Result<Vec<AttentionFixture>> {
    let cpu = Cpu::new();
    let mut fixtures = Vec::new();
    for visible in PREFIXES {
        let config = IncrementalAttentionConfig {
            n_query_positions: 1,
            query_start: visible - 1,
            n_kv_positions: CAPACITY,
            n_heads: HEADS,
            n_kv_heads: KV_HEADS,
            head_dim: DIM,
            scale: 1.0 / (DIM as f32).sqrt(),
        };
        let mut rounds = Vec::new();
        for round in 0..REPLAYS {
            let q: Vec<_> = (0..HEADS * DIM)
                .map(|i| ((i * 7 + round * 3) % 31) as f32 / 64.0 - 0.234375)
                .collect();
            let mut k: Vec<_> = (0..CAPACITY * KV_HEADS * DIM)
                .map(|i| ((i * 11 + round * 5) % 37) as f32 / 32.0 - 0.5)
                .collect();
            let mut v: Vec<_> = (0..CAPACITY * KV_HEADS * DIM)
                .map(|i| ((i * 13 + round * 7) % 41) as f32 / 32.0 + 0.125)
                .collect();
            k[visible * KV_HEADS * DIM..].fill(f32::NAN);
            v[visible * KV_HEADS * DIM..].fill(f32::NAN);
            let mut expected = vec![0.0; q.len()];
            cpu.incremental_causal_attention(&mut expected, &q, &k, &v, config)?;
            ensure!(
                expected.iter().all(|x| x.is_finite()),
                "C2048 nonfinite CPU oracle"
            );
            rounds.push((q, k, v, expected));
        }
        ensure!(
            rounds[0].3.iter().zip(&rounds[1].3).any(|(&old, &new)| {
                (f64::from(old) - f64::from(new)).abs() > ATOL + RTOL * f64::from(new).abs()
            }),
            "changed replay oracle is indistinguishable at the declared tolerance"
        );
        fixtures.push(AttentionFixture { config, rounds });
    }
    Ok(fixtures)
}

pub(super) fn run(path: &Path) -> Result<()> {
    ensure!(
        std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(std::ffi::OsStr::new("0")),
        "payload tracing must be disabled"
    );
    ensure!(
        std::env::var_os("HAG_ATTENTION2048_GATE").as_deref() == Some(std::ffi::OsStr::new("1")),
        "external C2048 supervision is required"
    );
    // All sixteen CPU references finish before the sole native device open.
    let cases = fixtures()?;
    let charge = configuration().allocation_bytes()?;
    ensure!(
        charge <= LIMIT,
        "C2048 diagnostic hardware charge exceeds budget"
    );
    let budget = MemoryBudget::new(LIMIT);
    let device = NativeExecutionDevice::open(path, &budget, configuration())?;
    let initial = device.stats()?;
    ensure!(
        initial.allocation_bytes == charge
            && !initial.poisoned
            && initial.dispatch_attempts == 0
            && initial.logical_submissions == 0,
        "C2048 initial state differs"
    );
    let mut maximum = 0.0_f64;
    for (case, fixture) in cases.iter().enumerate() {
        let width = HEADS * DIM;
        let cache = CAPACITY * KV_HEADS * DIM;
        let q = device.allocate_f32(width + 2)?;
        let k = device.allocate_f32(cache + 2)?;
        let v = device.allocate_f32(cache + 2)?;
        let out = device.allocate_f32(width + 2)?;
        // Generic command construction accepts this geometry; the native
        // backend must refuse C16385 before recording writes or dispatch.
        let mut oversized = CommandList::new();
        oversized.incremental_causal_attention(
            &out.view(1, 2)?,
            &q.view(1, 2)?,
            &k.view(1, 16385 * 2)?,
            &v.view(1, 16385 * 2)?,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: 16384,
                n_kv_positions: 16385,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                scale: 1.0,
            },
        )?;
        let before_refusal = device.stats()?;
        ensure!(
            device.record(&oversized).is_err(),
            "C16385 unexpectedly admitted"
        );
        ensure!(
            device.stats()? == before_refusal,
            "C16385 refusal mutated native state"
        );
        drop(oversized);

        let mut commands = CommandList::new();
        commands.incremental_causal_attention(
            &out.view(1, width)?,
            &q.view(1, width)?,
            &k.view(1, cache)?,
            &v.view(1, cache)?,
            fixture.config,
        )?;
        let batch = device.record(&commands)?;
        drop(commands);
        for (round, (queries, keys, values, expected)) in fixture.rounds.iter().enumerate() {
            device.write_f32(&q.full_view(), &guarded(queries))?;
            device.write_f32(&k.full_view(), &guarded(keys))?;
            device.write_f32(&v.full_view(), &guarded(values))?;
            device.write_f32(&out.full_view(), &guarded(&vec![f32::NAN; width]))?;
            check_submission(&device, &batch, HEADS as u64 * 3)?;
            let mut actual = vec![f32::NAN; width + 2];
            device.read_f32(&out.full_view(), &mut actual)?;
            maximum = maximum.max(compare(&actual, expected, false)?);
            for tensor in [&q, &k, &v] {
                let mut first = [0.0];
                let mut last = [0.0];
                device.read_f32(&tensor.view(0, 1)?, &mut first)?;
                device.read_f32(&tensor.view(tensor.len() - 1, 1)?, &mut last)?;
                ensure!(
                    first[0].to_bits() == GUARD.to_bits() && last[0].to_bits() == GUARD.to_bits(),
                    "C2048 input guard changed"
                );
            }
            println!("PASS attention2048 prefix={} case={case} round={round} checked_values={width} guards=8",
                fixture.config.query_start + 1);
        }
        drop((batch, q, k, v, out));
        retired(&device)?;
    }
    let stats = device.stats()?;
    ensure!(
        stats.dispatch_count == DISPATCHES
            && stats.dispatch_attempts == DISPATCHES
            && stats.logical_submissions == SUBMISSIONS
            && stats.uploaded_weight_bytes == 0
            && stats.host_read_bytes == ((VALUES + GUARDS) * 4) as u64
            && stats.allocation_bytes == charge
            && budget.current_bytes() == charge,
        "C2048 final counters differ"
    );
    drop(device);
    ensure!(
        budget.current_bytes() == 0,
        "C2048 final hardware charge retained"
    );
    println!("{{\"execution\":\"native-hag-attention2048-diagnostic\",\"capacity\":{CAPACITY},\"visible_prefixes\":{:?},\"changed_replays_per_prefix\":{REPLAYS},\"physical_dispatches\":{DISPATCHES},\"logical_submissions\":{SUBMISSIONS},\"checked_values\":{VALUES},\"guard_checks\":{GUARDS},\"max_absolute_error\":{maximum},\"absolute_tolerance\":{ATOL},\"relative_tolerance\":{RTOL},\"actual_hag_allocation_bytes\":{charge},\"hag_budget_bytes\":{LIMIT},\"verification_readback_bytes\":{},\"submission_tensor_transfer_bytes\":0,\"capacity_refusals\":{},\"budget_after\":0}}",
        PREFIXES, stats.host_read_bytes, PREFIXES.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_boundaries_replay_and_causal_oracles_precede_native_open() {
        let cases = fixtures().unwrap();
        assert_eq!(cases.len(), 8);
        assert_eq!(
            (SUBMISSIONS, DISPATCHES, VALUES, GUARDS),
            (16, 192, 8192, 128)
        );
        assert_eq!(configuration().allocation_bytes().unwrap(), 8_495_104);
        for (case, visible) in cases.into_iter().zip(PREFIXES) {
            assert_eq!(case.config.query_start + 1, visible);
            assert_eq!(case.config.n_kv_positions, 2048);
            assert_eq!(case.rounds.len(), 2);
            for (_, keys, values, oracle) in case.rounds {
                let visible_len = visible * KV_HEADS * DIM;
                assert!(keys[..visible_len].iter().all(|v| v.is_finite()));
                assert!(values[..visible_len].iter().all(|v| v.is_finite()));
                assert!(keys[visible_len..].iter().all(|v| v.is_nan()));
                assert!(values[visible_len..].iter().all(|v| v.is_nan()));
                assert_eq!(oracle.len(), HEADS * DIM);
                assert!(oracle.iter().all(|v| v.is_finite()));
            }
        }
    }
}
