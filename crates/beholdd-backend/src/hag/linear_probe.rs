//! Bounded BF16/Q8_0 matvec qualification on the experimental native HAG queue.
//!
//! This is an opt-in test harness, not a `Device`, `ExecutionDevice` or serving
//! backend. Kernels consume the original packed bytes on-device. Deliberately
//! small serial arithmetic favors an independently verifiable reference over
//! throughput; no model or checkpoint is loaded.

use super::{fixtures, kernels};
use crate::DeviceError;
use exec_hag::{
    budget::MemoryBudget,
    linux::compute::{ComputeQueue, KernelAbi, GFX1151_SERIAL_ABI},
};
use std::path::Path;

const INPUT_BYTES: usize = 256 * 4;
const OUTPUT_BYTES: usize = 64 * 4;
const UNUSED: u8 = 0xa5;

/// Summary of fully completed, byte/numerically verified submissions.
#[derive(Debug)]
pub struct LinearProofReport {
    /// Distinct packed matrix shapes/formats exercised.
    pub cases: usize,
    /// Completed dispatches; each fixture runs twice with changed input.
    pub dispatches: usize,
    /// Total output rows checked against both references.
    pub checked_rows: usize,
    /// Largest resident queue charge, including allocation padding and guards.
    pub peak_queue_bytes: u64,
    /// Largest observed absolute difference from Beholdd's optimized CPU oracle.
    pub max_cpu_absolute_error: f64,
}

fn backend(error: impl std::fmt::Display) -> DeviceError {
    DeviceError::Backend {
        device: "hag-linear-proof",
        message: error.to_string(),
    }
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Execute only the fixed finite fixture suite on a caller-selected render node.
///
/// The two immutable shaders each retain one queue across all their fixtures.
/// Every fixture runs twice, changing host input between submissions. Every result
/// must match the ordered scalar oracle bitwise and satisfy the separately
/// declared CPU-roundoff envelope. Full input/weight readback and unused output
/// checks supplement the queue's allocation guards. No retries or fallback occur.
pub fn run(
    path: impl AsRef<Path>,
    budget: &MemoryBudget,
) -> Result<LinearProofReport, DeviceError> {
    run_bounded(path.as_ref(), budget, false)
}

/// Submit only the first BF16 fixture once for an explicitly supervised diagnosis.
/// This does not qualify queue reuse, Q8_0, or the remaining shapes.
pub fn diagnose_first(
    path: impl AsRef<Path>,
    budget: &MemoryBudget,
) -> Result<LinearProofReport, DeviceError> {
    run_bounded(path.as_ref(), budget, true)
}

fn run_bounded(
    path: &Path,
    budget: &MemoryBudget,
    first_only: bool,
) -> Result<LinearProofReport, DeviceError> {
    if GFX1151_SERIAL_ABI
        != (KernelAbi {
            program_resources: [kernels::RSRC1, kernels::RSRC2, kernels::RSRC3],
            raw_descriptor: kernels::RAW_SRD_WORD3,
            threads: kernels::THREADS,
            max_groups: exec_hag::linux::compute::DEFAULT_DISPATCH_GROUPS,
        })
    {
        return Err(backend(
            "HAG command ABI differs from the audited shader ABI",
        ));
    }
    let baseline = budget.current_bytes();
    let mut report = LinearProofReport {
        cases: 0,
        dispatches: 0,
        checked_rows: 0,
        peak_queue_bytes: 0,
        max_cpu_absolute_error: 0.0,
    };
    let cases = fixtures::linear_fixtures();
    for (kind, code) in [(30, kernels::bf16_bytes()), (8, kernels::q8_0_bytes())] {
        let mut selected: Vec<_> = cases
            .iter()
            .filter(|fixture| fixture.kind.0 == kind)
            .collect();
        let capacity = selected
            .iter()
            .map(|fixture| fixture.weight_bytes.len())
            .max()
            .ok_or_else(|| backend("missing native proof fixtures"))?;
        // All scalar references validate finite values, exact packed lengths and
        // shape limits before any fixture can reach an unsafe device submission.
        let format = if kind == 30 {
            kernels::KernelKind::Bf16
        } else {
            kernels::KernelKind::Q8_0
        };
        for fixture in &selected {
            if format
                .packed_bytes(fixture.rows, fixture.cols)
                .map_err(backend)?
                != fixture.weight_bytes.len()
            {
                return Err(backend("shader admission and fixture storage disagree"));
            }
            fixture.scalar_reference()?;
        }
        // Retain the original queue capacities while reducing only submission
        // count, so a diagnostic does not silently change the failed VA layout.
        if first_only {
            selected.truncate(1);
        }
        // SAFETY: these checked-in LLVM-verified GFX1151 programs implement the
        // fixed three-SRD/two-scalar ABI, eight-VGPR limit and bounded loops.
        // No user/model code is supplied; shape validation below keeps one group
        // of 64 lanes inside the exact weight/input/output descriptors.
        let mut queue = unsafe {
            ComputeQueue::open(
                path,
                budget,
                &code,
                [capacity as u32, INPUT_BYTES as u32, OUTPUT_BYTES as u32],
            )
        }
        .map_err(backend)?;
        let charge = queue.allocation_bytes();
        report.peak_queue_bytes = report.peak_queue_bytes.max(charge);
        for original in selected {
            let mut fixture = original.clone();
            let mut weights = vec![UNUSED; capacity];
            weights[..fixture.weight_bytes.len()].copy_from_slice(&fixture.weight_bytes);
            queue.write(0, 0, &weights).map_err(backend)?;
            for round in 0..if first_only { 1 } else { 2 } {
                if round == 1 {
                    for input in &mut fixture.input {
                        *input *= -0.5;
                    }
                    fixture.input[0] += 0.25;
                }
                // Validates dimensions/packed payload and poisons every live row.
                let poisoned = fixture.unwritten_output()?;
                let mut input = vec![UNUSED; INPUT_BYTES];
                let packed_input = bytes(&fixture.input);
                input[..packed_input.len()].copy_from_slice(&packed_input);
                let mut output = vec![UNUSED; OUTPUT_BYTES];
                let packed_output = bytes(&poisoned);
                output[..packed_output.len()].copy_from_slice(&packed_output);
                queue.write(1, 0, &input).map_err(backend)?;
                queue.write(2, 0, &output).map_err(backend)?;
                eprintln!(
                    "RUN {} round={} rows={} cols={} resident_bytes={}",
                    fixture.name, round, fixture.rows, fixture.cols, charge
                );
                // SAFETY: fixture validation guarantees rows1..64, cols1..256,
                // complete32-element Q8_0 blocks, exact logical byte lengths and
                // one 64-thread group. Each active lane writes precisely one row.
                let fence = unsafe {
                    queue.dispatch(
                        [
                            fixture.weight_bytes.len() as u32,
                            packed_input.len() as u32,
                            packed_output.len() as u32,
                        ],
                        [fixture.cols as u32, fixture.rows as u32],
                        kernels::GROUPS[0],
                    )
                }
                .map_err(backend)?;
                let mut observed = vec![0; OUTPUT_BYTES];
                queue.read(2, 0, &mut observed).map_err(backend)?;
                if observed[packed_output.len()..] != output[packed_output.len()..] {
                    return Err(backend(format!(
                        "{}: shader changed inactive output rows",
                        fixture.name
                    )));
                }
                let actual: Vec<f32> = observed[..packed_output.len()]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let parity = fixture.verify_serial_output(&actual)?;
                let mut observed_weights = vec![0; capacity];
                let mut observed_input = vec![0; INPUT_BYTES];
                queue.read(0, 0, &mut observed_weights).map_err(backend)?;
                queue.read(1, 0, &mut observed_input).map_err(backend)?;
                if observed_weights != weights || observed_input != input {
                    return Err(backend(format!(
                        "{}: shader changed a read-only input",
                        fixture.name
                    )));
                }
                if budget.current_bytes() != baseline + charge || queue.allocation_bytes() != charge
                {
                    return Err(backend(
                        "resident queue allocation charge changed during reuse",
                    ));
                }
                report.dispatches += 1;
                report.checked_rows += parity.checked_rows;
                report.max_cpu_absolute_error = report
                    .max_cpu_absolute_error
                    .max(parity.max_cpu_absolute_error);
                println!("PASS {} round={} fence={} rows={} serial=bitwise cpu_abs_error={} cpu_bound={} resident_bytes={}",
                    fixture.name, round, fence, parity.checked_rows, parity.max_cpu_absolute_error,
                    parity.max_cpu_allowed_error, charge);
            }
            report.cases += 1;
        }
        queue.close().map_err(backend)?;
        if budget.current_bytes() != baseline {
            return Err(backend("native queue cleanup did not release its budget"));
        }
        if first_only {
            break;
        }
    }
    Ok(report)
}
