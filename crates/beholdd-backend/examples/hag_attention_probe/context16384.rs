// Bounded late-position attention only; no checkpoint or full-model qualification.
use super::*;

const PREFIXES: [usize; 3] = [8192, 16383, 16384];
const CAPACITY: usize = 16384;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const DIM: usize = 128;
const REPLAYS: usize = 2;
const WIDTH: usize = HEADS * DIM;
const CACHE: usize = CAPACITY * KV_HEADS * DIM;
const LIMIT: u64 = 32 * 1024 * 1024;
const CHARGE: u64 = 25_468_928;
const SUBMISSIONS: u64 = (PREFIXES.len() * REPLAYS) as u64;
const DISPATCHES: u64 = SUBMISSIONS * HEADS as u64 * 3;
const VALUES: usize = PREFIXES.len() * REPLAYS * WIDTH;
const GUARDS: usize = PREFIXES.len() * REPLAYS * 8;
const TENSOR_BYTES: u64 = ((2 * WIDTH + 2 * CACHE + 8) * 4) as u64;
const FIXTURE_BYTES: usize = PREFIXES.len() * REPLAYS * (2 * WIDTH + 2 * CACHE) * 4;

fn configuration() -> NativeExecutionConfig {
    NativeExecutionConfig {
        weight_capacity_bytes: 4,
        tensor_capacity_bytes: 24 * 1024 * 1024,
        auxiliary_capacity_bytes: 256 * 1024,
    }
}

fn auxiliary_bytes(visible: usize) -> u64 {
    // Shared F64 scores/F32 probabilities workspace, alignment word, scale word.
    (visible * 12 + 8) as u64
}

fn host_write_bytes() -> u64 {
    // Zero each allocation once, upload each replay, and record each workspace/scale.
    PREFIXES.len() as u64 * (1 + REPLAYS as u64) * TENSOR_BYTES
        + PREFIXES.into_iter().map(auxiliary_bytes).sum::<u64>()
}

fn fixtures() -> Result<Vec<AttentionFixture>> {
    ensure!(
        FIXTURE_BYTES <= 128 * 1024 * 1024,
        "C16384 CPU fixture bound exceeded"
    );
    let cpu = Cpu::new();
    let mut fixtures = Vec::with_capacity(PREFIXES.len());
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
        let mut rounds = Vec::with_capacity(REPLAYS);
        for round in 0..REPLAYS {
            let q: Vec<_> = (0..WIDTH)
                .map(|i| ((i * 7 + round * 3) % 31) as f32 / 64.0 - 0.234375)
                .collect();
            let mut k: Vec<_> = (0..CACHE)
                .map(|i| ((i * 11 + round * 5) % 37) as f32 / 32.0 - 0.5)
                .collect();
            // The replay offset keeps stale output distinguishable even when a
            // long prefix averages away differences in the periodic fixture.
            let mut v: Vec<_> = (0..CACHE)
                .map(|i| ((i * 13 + round * 7) % 41) as f32 / 32.0 + 0.125 + round as f32 * 0.25)
                .collect();
            k[visible * KV_HEADS * DIM..].fill(f32::NAN);
            v[visible * KV_HEADS * DIM..].fill(f32::NAN);
            let mut expected = vec![0.0; WIDTH];
            cpu.incremental_causal_attention(&mut expected, &q, &k, &v, config)?;
            ensure!(
                expected.iter().all(|x| x.is_finite()),
                "C16384 nonfinite CPU oracle at prefix {visible} replay {round}"
            );
            rounds.push((q, k, v, expected));
        }
        ensure!(
            rounds[0].3.iter().zip(&rounds[1].3).any(|(&old, &new)| {
                (f64::from(old) - f64::from(new)).abs() > ATOL + RTOL * f64::from(new).abs()
            }),
            "C16384 changed replay oracle is indistinguishable at the declared tolerance"
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
        std::env::var_os("HAG_ATTENTION16384_GATE").as_deref() == Some(std::ffi::OsStr::new("1")),
        "external C16384 supervision is required"
    );
    // All six single-query CPU references finish before the sole native open.
    // Both CPU and GPU work are O(C); no C-by-C matrix is allocated or evaluated.
    let cases = fixtures()?;
    let charge = configuration().allocation_bytes()?;
    ensure!(
        charge == CHARGE && charge <= LIMIT,
        "C16384 hardware charge differs"
    );
    let budget = MemoryBudget::new(LIMIT);
    let device = NativeExecutionDevice::open(path, &budget, configuration())?;
    let initial = device.stats()?;
    ensure!(
        initial.allocation_bytes == charge
            && budget.current_bytes() == charge
            && !initial.poisoned
            && initial.dispatch_count == 0
            && initial.dispatch_attempts == 0
            && initial.logical_submissions == 0
            && initial.host_read_bytes == 0
            && initial.host_write_bytes == 0
            && initial.uploaded_weight_bytes == 0,
        "C16384 initial state differs"
    );
    retired(&device)?;
    let mut maximum = 0.0_f64;
    for (case, fixture) in cases.iter().enumerate() {
        let visible = fixture.config.query_start + 1;
        let q = device.allocate_f32(WIDTH + 2)?;
        let k = device.allocate_f32(CACHE + 2)?;
        let v = device.allocate_f32(CACHE + 2)?;
        let out = device.allocate_f32(WIDTH + 2)?;
        // Valid generic views, but a capacity beyond the native attention limit.
        // Refusal must happen before recording writes or any queue activity.
        let mut oversized = CommandList::new();
        oversized.incremental_causal_attention(
            &out.view(1, 2)?,
            &q.view(1, 2)?,
            &k.view(1, (CAPACITY + 1) * 2)?,
            &v.view(1, (CAPACITY + 1) * 2)?,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: CAPACITY,
                n_kv_positions: CAPACITY + 1,
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
            &out.view(1, WIDTH)?,
            &q.view(1, WIDTH)?,
            &k.view(1, CACHE)?,
            &v.view(1, CACHE)?,
            fixture.config,
        )?;
        let batch = device.record(&commands)?;
        drop(commands);
        let recorded = device.stats()?;
        ensure!(
            recorded.weight_live_bytes == 0
                && recorded.tensor_live_bytes == TENSOR_BYTES
                && recorded.auxiliary_live_bytes == auxiliary_bytes(visible),
            "C16384 recorded resource sizes differ"
        );
        for (round, (queries, keys, values, expected)) in fixture.rounds.iter().enumerate() {
            device.write_f32(&q.full_view(), &guarded(queries))?;
            device.write_f32(&k.full_view(), &guarded(keys))?;
            device.write_f32(&v.full_view(), &guarded(values))?;
            device.write_f32(&out.full_view(), &guarded(&vec![f32::NAN; WIDTH]))?;
            check_submission(&device, &batch, HEADS as u64 * 3)?;
            let mut actual = vec![f32::NAN; WIDTH + 2];
            device.read_f32(&out.full_view(), &mut actual)?;
            maximum = maximum.max(compare(&actual, expected, false)?);
            for tensor in [&q, &k, &v] {
                let mut first = [0.0];
                let mut last = [0.0];
                device.read_f32(&tensor.view(0, 1)?, &mut first)?;
                device.read_f32(&tensor.view(tensor.len() - 1, 1)?, &mut last)?;
                ensure!(
                    first[0].to_bits() == GUARD.to_bits() && last[0].to_bits() == GUARD.to_bits(),
                    "C16384 input guard changed"
                );
            }
            println!("PASS attention16384 prefix={visible} case={case} round={round} checked_values={WIDTH} guards=8");
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
            && stats.host_write_bytes == host_write_bytes()
            && stats.weight_peak_bytes == 0
            && stats.tensor_peak_bytes == TENSOR_BYTES
            && stats.auxiliary_peak_bytes == auxiliary_bytes(CAPACITY)
            && stats.allocation_bytes == charge
            && budget.current_bytes() == charge,
        "C16384 final counters differ"
    );
    drop(device);
    ensure!(
        budget.current_bytes() == 0,
        "C16384 final hardware charge retained"
    );
    println!("{{\"execution\":\"native-hag-attention16384-diagnostic\",\"capacity\":{CAPACITY},\"visible_prefixes\":{:?},\"query_positions_per_prefix\":1,\"query_heads\":{HEADS},\"kv_heads\":{KV_HEADS},\"head_dimension\":{DIM},\"changed_replays_per_prefix\":{REPLAYS},\"physical_dispatches\":{DISPATCHES},\"logical_submissions\":{SUBMISSIONS},\"checked_values\":{VALUES},\"guard_checks\":{GUARDS},\"max_absolute_error\":{maximum},\"absolute_tolerance\":{ATOL},\"relative_tolerance\":{RTOL},\"actual_hag_allocation_bytes\":{charge},\"hag_budget_bytes\":{LIMIT},\"verification_readback_bytes\":{},\"preparation_write_bytes\":{},\"tensor_peak_bytes\":{},\"auxiliary_peak_bytes\":{},\"submission_tensor_transfer_bytes\":0,\"capacity_refusals\":{},\"budget_after\":0,\"full_model_qualified\":false}}",
        PREFIXES, stats.host_read_bytes, stats.host_write_bytes, stats.tensor_peak_bytes,
        stats.auxiliary_peak_bytes, PREFIXES.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_late_positions_replays_poison_and_cpu_oracles_precede_native_open() {
        let cases = fixtures().unwrap();
        assert_eq!(cases.len(), 3);
        assert_eq!((SUBMISSIONS, DISPATCHES, VALUES, GUARDS), (6, 36, 1536, 48));
        assert_eq!(configuration().allocation_bytes().unwrap(), CHARGE);
        assert_eq!(CHARGE, 25_468_928);
        assert_eq!(TENSOR_BYTES, 16_779_296);
        assert_eq!(FIXTURE_BYTES, 100_675_584);
        assert_eq!(auxiliary_bytes(CAPACITY), 196_616);
        assert_eq!(host_write_bytes(), 151_505_196);
        assert!(TENSOR_BYTES <= u64::from(configuration().tensor_capacity_bytes));
        assert!(auxiliary_bytes(CAPACITY) <= u64::from(configuration().auxiliary_capacity_bytes));
        assert!(CHARGE <= LIMIT);
        for (case, visible) in cases.into_iter().zip(PREFIXES) {
            assert_eq!(case.config.n_query_positions, 1);
            assert_eq!(case.config.query_start + 1, visible);
            assert_eq!(case.config.n_kv_positions, CAPACITY);
            assert_eq!(case.config.n_heads, HEADS);
            assert_eq!(case.config.n_kv_heads, KV_HEADS);
            assert_eq!(case.config.head_dim, DIM);
            assert_eq!(case.rounds.len(), REPLAYS);
            for (query, keys, values, oracle) in case.rounds {
                assert_eq!(query.len(), WIDTH);
                assert_eq!((keys.len(), values.len()), (CACHE, CACHE));
                let visible_len = visible * KV_HEADS * DIM;
                assert!(query.iter().all(|v| v.is_finite()));
                assert!(keys[..visible_len].iter().all(|v| v.is_finite()));
                assert!(values[..visible_len].iter().all(|v| v.is_finite()));
                assert!(keys[visible_len..].iter().all(|v| v.is_nan()));
                assert!(values[visible_len..].iter().all(|v| v.is_nan()));
                assert_eq!(oracle.len(), WIDTH);
                assert!(oracle.iter().all(|v| v.is_finite()));
            }
        }
    }
}
