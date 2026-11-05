//! Explicit, bounded composed native-execution proof; never a serving startup action.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{ensure, Context, Result};
    use beholdd_backend::execution::{
        CommandList, CpuExecutionDevice, ExecutionDevice, F32Tensor, LinearMode, RecordedBatch,
    };
    use beholdd_backend::hag::native_execution::{NativeExecutionConfig, NativeExecutionDevice};
    use beholdd_backend::Weight;
    use beholdd_format::GgmlType;
    use exec_hag::budget::MemoryBudget;
    use std::path::Path;

    const INPUTS: usize = 32;
    const INTERMEDIATES: usize = 32;
    const OUTPUTS: usize = 4;
    const ROUNDS: usize = 3;
    const COMMANDS: usize = 2;
    const BF16: GgmlType = GgmlType(30);
    const Q8_0: GgmlType = GgmlType(8);
    const BF16_BYTES: usize = INTERMEDIATES * INPUTS * 2;
    const Q8_BYTES: usize = OUTPUTS * 34;
    const PACKED_BYTES: usize = BF16_BYTES + Q8_BYTES;
    const TENSOR_BYTES: usize = (INPUTS + INTERMEDIATES + OUTPUTS) * 4;
    const BUDGET_BYTES: u64 = 256 * 1024;
    const Q8_SCALE_NUMERATORS: [i32; OUTPUTS] = [1, -2, 4, 8];
    const Q8_SCALE_BITS: [u16; OUTPUTS] = [0x3000, 0xb400, 0x3800, 0x3c00];
    const OUTPUT_POISON: f32 = f32::from_bits(0x7fc0_4847);

    struct Fixture {
        bf16: Vec<u8>,
        q8: Vec<u8>,
    }

    struct Chain {
        input: F32Tensor,
        output: F32Tensor,
        batch: RecordedBatch,
    }

    fn bf16_numerator(row: usize, column: usize) -> i32 {
        ((row * 7 + column * 3) % 17) as i32 - 8
    }

    fn q8_integer(row: usize, column: usize) -> i32 {
        ((row * 11 + column * 5) % 31) as i32 - 15
    }

    fn input_numerator(round: usize, column: usize) -> i32 {
        ((column * 5 + round * 7) % 19) as i32 - 9
    }

    fn input_values(round: usize) -> Vec<f32> {
        (0..INPUTS)
            .map(|column| input_numerator(round, column) as f32 / 4.0)
            .collect()
    }

    fn fixture() -> Result<Fixture> {
        let mut bf16 = Vec::with_capacity(BF16_BYTES);
        for row in 0..INTERMEDIATES {
            for column in 0..INPUTS {
                let bits = (bf16_numerator(row, column) as f32 / 8.0).to_bits();
                ensure!(bits & 0xffff == 0, "fixture requires exact BF16 values");
                bf16.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
        }
        let mut q8 = Vec::with_capacity(Q8_BYTES);
        for (row, scale_bits) in Q8_SCALE_BITS.into_iter().enumerate() {
            q8.extend_from_slice(&scale_bits.to_le_bytes());
            for column in 0..INTERMEDIATES {
                q8.push(q8_integer(row, column) as i8 as u8);
            }
        }
        ensure!(bf16.len() == BF16_BYTES && q8.len() == Q8_BYTES);
        Ok(Fixture { bf16, q8 })
    }

    // Independent integer arithmetic avoids sharing the shader's packed decoder or
    // its floating-point reduction implementation. All fixture products and partial
    // sums are exactly representable in f32: stage one has denominator 32, stage two
    // has denominator 256, and every numerator remains well below 2^24.
    fn exact_reference(round: usize) -> Vec<f32> {
        let intermediate: Vec<i64> = (0..INTERMEDIATES)
            .map(|row| {
                (0..INPUTS)
                    .map(|column| {
                        i64::from(bf16_numerator(row, column))
                            * i64::from(input_numerator(round, column))
                    })
                    .sum()
            })
            .collect();
        (0..OUTPUTS)
            .map(|row| {
                let numerator: i64 = intermediate
                    .iter()
                    .enumerate()
                    .map(|(column, value)| {
                        i64::from(Q8_SCALE_NUMERATORS[row])
                            * i64::from(q8_integer(row, column))
                            * value
                    })
                    .sum();
                numerator as f32 / 256.0
            })
            .collect()
    }

    fn record_chain(device: &dyn ExecutionDevice, data: &Fixture) -> Result<Chain> {
        let encoded_bf16 = device.upload_weight(&Weight {
            bytes: &data.bf16,
            kind: BF16,
            rows: INTERMEDIATES,
            cols: INPUTS,
        })?;
        let encoded_q8 = device.upload_weight(&Weight {
            bytes: &data.q8,
            kind: Q8_0,
            rows: OUTPUTS,
            cols: INTERMEDIATES,
        })?;
        let prepared_bf16 = device.prepare_weight(&encoded_bf16)?;
        let prepared_q8 = device.prepare_weight(&encoded_q8)?;
        ensure!(encoded_bf16.byte_len() == BF16_BYTES);
        ensure!(prepared_bf16.byte_len() == BF16_BYTES);
        ensure!(encoded_q8.byte_len() == Q8_BYTES);
        ensure!(prepared_q8.byte_len() == Q8_BYTES);

        let input = device.upload_f32(&input_values(0))?;
        let intermediate = device.allocate_f32(INTERMEDIATES)?;
        let output = device.allocate_f32(OUTPUTS)?;
        let mut commands = CommandList::new();
        commands.linear(
            &intermediate.full_view(),
            &input.full_view(),
            &prepared_bf16,
            LinearMode::Overwrite,
        )?;
        commands.linear(
            &output.full_view(),
            &intermediate.full_view(),
            &prepared_q8,
            LinearMode::Overwrite,
        )?;
        let batch = device.record(&commands)?;
        ensure!(batch.len() == COMMANDS);

        // The source list is mutable, but its already-recorded snapshot is not.
        // This third command is never submitted by the proof.
        commands.linear(
            &output.full_view(),
            &intermediate.full_view(),
            &prepared_q8,
            LinearMode::Overwrite,
        )?;
        ensure!(commands.len() == COMMANDS + 1 && batch.len() == COMMANDS);

        // Returning only these handles deliberately drops the source list, both
        // encoded/prepared weights, and the intermediate's original host handle.
        // The immutable batch must retain its complete resident resource graph.
        Ok(Chain {
            input: input.clone(),
            output,
            batch,
        })
    }

    fn verify_output(round: usize, actual: &[f32]) -> Result<()> {
        let expected = exact_reference(round);
        ensure!(actual.len() == expected.len());
        for (row, (&observed, &reference)) in actual.iter().zip(&expected).enumerate() {
            ensure!(
                observed.is_finite() && observed.to_bits() == reference.to_bits(),
                "round {round} row {row}: got {observed:?} ({:#010x}), expected {reference:?} ({:#010x})",
                observed.to_bits(),
                reference.to_bits()
            );
        }
        Ok(())
    }

    fn verify_cpu_fixture(data: &Fixture) -> Result<()> {
        let cpu = CpuExecutionDevice::new();
        let chain = record_chain(&cpu, data)?;
        for round in 0..ROUNDS {
            cpu.write_f32(&chain.input.full_view(), &input_values(round))?;
            cpu.write_f32(&chain.output.full_view(), &[OUTPUT_POISON; OUTPUTS])?;
            let submission = cpu.submit(&chain.batch)?;
            cpu.wait(&submission)?;
            let mut actual = [OUTPUT_POISON; OUTPUTS];
            cpu.read_f32(&chain.output.full_view(), &mut actual)?;
            verify_output(round, &actual).context("CPU fixture/reference disagreement")?;
        }
        Ok(())
    }

    pub(super) fn run(path: &Path) -> Result<()> {
        let data = fixture()?;
        verify_cpu_fixture(&data)?;
        let budget = MemoryBudget::new(BUDGET_BYTES);
        let device = NativeExecutionDevice::open(
            path,
            &budget,
            NativeExecutionConfig {
                weight_capacity_bytes: 4096,
                tensor_capacity_bytes: 4096,
                ..Default::default()
            },
        )?;
        let resident_bytes = budget.current_bytes();
        ensure!(resident_bytes > 0 && resident_bytes <= BUDGET_BYTES);
        let chain = record_chain(&device, &data)?;
        let alias = device.clone();
        drop(device);

        let initial = alias.stats()?;
        ensure!(initial.allocation_bytes == resident_bytes);
        ensure!(initial.weight_live_bytes == PACKED_BYTES as u64);
        ensure!(initial.tensor_live_bytes == TENSOR_BYTES as u64);
        ensure!(initial.weight_peak_bytes == PACKED_BYTES as u64);
        ensure!(initial.tensor_peak_bytes == TENSOR_BYTES as u64);
        ensure!(initial.auxiliary_live_bytes == 0 && initial.auxiliary_peak_bytes == 0);
        ensure!(initial.uploaded_weight_bytes == PACKED_BYTES as u64);
        ensure!(initial.host_read_bytes == 0);
        ensure!(
            initial.dispatch_attempts == 0
                && initial.dispatch_count == 0
                && initial.logical_submissions == 0
        );
        ensure!(!initial.poisoned);
        for round in 0..ROUNDS {
            // Only activations change between submissions; packed weights and the
            // recorded batch are uploaded/created once and stay resident.
            if round != 0 {
                alias.write_f32(&chain.input.full_view(), &input_values(round))?;
            }
            alias.write_f32(&chain.output.full_view(), &[OUTPUT_POISON; OUTPUTS])?;
            let before = alias.stats()?;
            println!(
                "RUN composed BF16->Q8_0 round={round} commands={COMMANDS} resident_bytes={resident_bytes} packed_bytes={PACKED_BYTES}"
            );
            let submission = alias.submit(&chain.batch)?;
            alias.wait(&submission)?;
            // Clone ownership and repeated waits must not resubmit completed work.
            let completed_alias = submission.clone();
            alias.wait(&completed_alias)?;
            let after = alias.stats()?;
            ensure!(after.dispatch_attempts == before.dispatch_attempts + COMMANDS as u64);
            ensure!(after.dispatch_count == before.dispatch_count + COMMANDS as u64);
            ensure!(after.logical_submissions == before.logical_submissions + 1);
            ensure!(after.host_read_bytes == before.host_read_bytes);
            ensure!(after.host_write_bytes == before.host_write_bytes);
            ensure!(after.uploaded_weight_bytes == PACKED_BYTES as u64);
            ensure!(after.weight_live_bytes == PACKED_BYTES as u64);
            ensure!(after.tensor_live_bytes == TENSOR_BYTES as u64);
            ensure!(after.auxiliary_live_bytes == 0 && after.auxiliary_peak_bytes == 0);
            ensure!(after.allocation_bytes == resident_bytes);
            ensure!(budget.current_bytes() == resident_bytes && !after.poisoned);

            // This is verification readback after completion, not an intermediate
            // tensor transfer during submission. Only four final values leave GPU.
            let mut actual = [OUTPUT_POISON; OUTPUTS];
            alias.read_f32(&chain.output.full_view(), &mut actual)?;
            verify_output(round, &actual)?;
            let verified = alias.stats()?;
            ensure!(verified.host_read_bytes == after.host_read_bytes + (OUTPUTS * 4) as u64);
            println!(
                "PASS round={round} exact_final_rows={OUTPUTS} submission_tensor_transfer_bytes=0"
            );
        }
        let final_stats = alias.stats()?;
        ensure!(final_stats.dispatch_attempts == (ROUNDS * COMMANDS) as u64);
        ensure!(final_stats.dispatch_count == (ROUNDS * COMMANDS) as u64);
        ensure!(final_stats.logical_submissions == ROUNDS as u64);
        ensure!(final_stats.host_read_bytes == (ROUNDS * OUTPUTS * 4) as u64);
        ensure!(final_stats.weight_peak_bytes == initial.weight_peak_bytes);
        ensure!(final_stats.tensor_peak_bytes == initial.tensor_peak_bytes);
        drop(chain);
        let retired = alias.stats()?;
        ensure!(retired.weight_live_bytes == 0 && retired.tensor_live_bytes == 0);
        ensure!(budget.current_bytes() == resident_bytes);
        drop(alias);
        ensure!(
            budget.current_bytes() == 0,
            "native execution proof retained allocations"
        );
        println!(
            "PASS composed native execution: rounds={ROUNDS} dispatches={} checked_final_rows={} packed_bytes={PACKED_BYTES} verification_readback_bytes={} peak_resident_bytes={resident_bytes} budget_after=0",
            ROUNDS * COMMANDS,
            ROUNDS * OUTPUTS,
            ROUNDS * OUTPUTS * 4
        );
        println!(
            "Synchronous per-operator fences; GPU-resident intermediate; no single-IB, throughput, or serving claim."
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn dyadic_composed_fixture_matches_cpu_reference_without_gpu() {
            verify_cpu_fixture(&fixture().unwrap()).unwrap();
        }

        #[test]
        fn changed_inputs_have_distinct_finite_nontrivial_results() {
            let expected: Vec<_> = (0..ROUNDS).map(exact_reference).collect();
            for (index, rows) in expected.iter().enumerate() {
                assert!(rows.iter().all(|value| value.is_finite() && *value != 0.0));
                assert!(!expected[..index].contains(rows));
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && (arguments[0] == "--help" || arguments[0] == "-h") {
        println!("Usage: hag_execution_probe --execute /dev/dri/renderD128");
        println!(
            "Explicit bounded BF16->Q8_0 native GPU proof; 6 dispatches, 256 KiB allocation cap."
        );
        println!("Run under an external process/runtime limit. No GPU access without --execute.");
        return Ok(());
    }
    let [flag, path] = arguments.as_slice() else {
        anyhow::bail!("usage: hag_execution_probe --execute /dev/dri/renderD128");
    };
    anyhow::ensure!(
        flag == "--execute",
        "GPU execution requires explicit --execute"
    );
    proof::run(std::path::Path::new(path))
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native composed-execution qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
