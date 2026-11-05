//! Inspect or explicitly execute one real dense-Qwen3 FFN through owned resources.
//!
//! This is a CPU correctness gate for the model-to-backend integration. It does not
//! select a GPU or claim that the bounded native HAG adapter supports model shapes.
//! Run real execution under the documented thermal/resource supervisor.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, ensure, Context, Result};
use beholdd_backend::cpu::Cpu;
use beholdd_backend::execution::{CpuExecutionDevice, ExecutionDevice};
use beholdd_backend::{Device, Weight};
use beholdd_format::{CheckpointId, Gguf, WeightRange};
use beholdd_memory::{
    check_checkpoint_storage, ResidentSource, ResidentWeight, StorageOverride, WeightAccess,
    WeightSource,
};
use beholdd_model::qwen3moe::{DenseFfnBlockLimits, DenseFfnLimits};
use beholdd_model::QwenModel;
use serde_json::json;

const USAGE: &str = "recorded_dense_ffn --model PATH [--layer N] [--block] [--execute-cpu]\n\
    Default: inspect bound metadata only. --execute-cpu explicitly runs three CPU parity rounds.\n\
    --block includes RMSNorm and in-place residual addition, but not attention.\n\
    GPU execution and serving are not enabled by this example.";

struct CountingSource<'a> {
    inner: ResidentSource<'a>,
    reads: AtomicU64,
    bytes: AtomicU64,
}

impl WeightSource for CountingSource<'_> {
    fn checkpoint_id(&self) -> CheckpointId {
        self.inner.checkpoint_id()
    }

    fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
        let weight = self.inner.load_range(range, access)?;
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes
            .fetch_add(weight.bytes().len() as u64, Ordering::Relaxed);
        Ok(weight)
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut path = None;
    let mut layer = None;
    let mut execute = false;
    let mut block = false;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--help") => {
                println!("{USAGE}");
                return Ok(());
            }
            Some("--model") if path.is_none() => {
                path = Some(PathBuf::from(args.next().context("--model needs a path")?));
            }
            Some("--layer") if layer.is_none() => {
                layer = Some(
                    args.next()
                        .context("--layer needs a nonnegative index")?
                        .to_str()
                        .context("layer index is not Unicode")?
                        .parse::<usize>()
                        .context("--layer needs a nonnegative index")?,
                );
            }
            Some("--execute-cpu") if !execute => execute = true,
            Some("--block") if !block => block = true,
            _ => bail!("unknown or repeated argument {arg:?}\n{USAGE}"),
        }
    }
    let path = path.with_context(|| format!("missing --model\n{USAGE}"))?;
    let layer = layer.unwrap_or(0);
    check_checkpoint_storage([&path], StorageOverride::RefuseUnsafe)?;
    let checkpoint = Gguf::open(&path)?;
    check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
    let model = QwenModel::load(&checkpoint)?;
    let plan = model.dense_ffn_plan(layer)?;
    let memory = plan.memory();
    let block_plan = if block {
        Some(model.dense_ffn_block_plan(layer)?)
    } else {
        None
    };
    let mut report = json!({
        "architecture": "qwen3",
        "scope": if block { "dense_ffn_with_norm_and_residual" } else { "dense_ffn" },
        "layer": layer,
        "input_width": plan.dim(),
        "ffn_width": plan.feed_forward_length(),
        "logical_memory": {
            "packed_weight_bytes": memory.packed_weight_bytes,
            "largest_staging_bytes": memory.largest_staging_bytes,
            "scratch_bytes": memory.scratch_bytes,
            "boundary_tensor_bytes": (plan.dim() as u64) * 8
        },
        "backend_allocation_overhead_included": false,
        "execution": "not_requested"
    });
    if let Some(block_plan) = &block_plan {
        let memory = block_plan.memory();
        report["logical_memory"] = json!({
            "packed_weight_bytes": memory.packed_weight_bytes,
            "largest_staging_bytes": memory.largest_staging_bytes,
            "resident_tensor_bytes": memory.resident_tensor_bytes,
            "boundary_tensor_bytes": (plan.dim() as u64) * 4
        });
    }
    if !execute {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    // Keep the opt-in fixture bounded even when supplied a different checkpoint.
    // These are logical component limits; the outer supervisor bounds all process
    // memory, including the CPU oracle's temporary decoded matrix and allocator.
    let limits = DenseFfnLimits {
        packed_weight_bytes: 64 * 1024 * 1024,
        staging_bytes: 32 * 1024 * 1024,
        scratch_bytes: 1024 * 1024,
    };
    let device = CpuExecutionDevice::new();
    let source = CountingSource {
        inner: ResidentSource::new(&checkpoint),
        reads: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
    };
    let (input, output, batch, expected_reads, expected_read_bytes) =
        if let Some(block_plan) = block_plan {
            let prepared = block_plan.prepare(
                &device,
                &source,
                DenseFfnBlockLimits {
                    packed_weight_bytes: limits.packed_weight_bytes,
                    largest_staging_bytes: limits.staging_bytes,
                    resident_tensor_bytes: 1024 * 1024,
                },
            )?;
            let hidden = device.allocate_f32(plan.dim())?;
            let batch = prepared.record(&device, &hidden.full_view())?;
            ensure!(
                batch.len() == 5,
                "expected norm/gate/up/SwiGLU/residual commands"
            );
            (
                hidden.clone(),
                hidden,
                batch,
                4,
                memory.packed_weight_bytes + plan.dim() as u64 * 4,
            )
        } else {
            let prepared = plan.prepare(&device, &source, limits)?;
            let input = device.allocate_f32(plan.dim())?;
            let output = device.allocate_f32(plan.dim())?;
            let batch = prepared.record(&device, &input.full_view(), &output.full_view())?;
            ensure!(batch.len() == 4, "expected gate/up/SwiGLU/down commands");
            (input, output, batch, 3, memory.packed_weight_bytes)
        };
    ensure!(
        source.reads.load(Ordering::Relaxed) == expected_reads,
        "unexpected preparation read count"
    );
    ensure!(
        source.bytes.load(Ordering::Relaxed) == expected_read_bytes,
        "prepared reads differ from the plan"
    );
    let mut checksum = 0u64;
    for round in 0..3 {
        // Deterministic inputs exercise reuse; these are not captured activations.
        let values: Vec<f32> = (0..plan.dim())
            .map(|index| (((index % 31) * 7 + round * 3) % 31) as f32 / 32.0 - 0.5)
            .collect();
        let expected = if block {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.ffn_norm.weight"))
                .context("missing reference FFN norm")?;
            let storage = ResidentWeight::load(&checkpoint, tensor.locator())?;
            let norm: Vec<f32> = storage
                .bytes()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            let mut normalized = vec![0.0; plan.dim()];
            let cpu = Cpu::new();
            cpu.rmsnorm(&mut normalized, &values, Some(&norm), model.config().eps())?;
            let contribution =
                immediate_reference(&checkpoint, layer, &normalized, plan.feed_forward_length())?;
            let mut residual = values.clone();
            cpu.add_in_place(&mut residual, &contribution)?;
            residual
        } else {
            immediate_reference(&checkpoint, layer, &values, plan.feed_forward_length())?
        };
        device.write_f32(&input.full_view(), &values)?;
        let submission = device.submit(&batch)?;
        device.wait(&submission)?;
        let mut actual = vec![0.0; plan.dim()];
        device.read_f32(&output.full_view(), &mut actual)?;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            ensure!(
                actual.is_finite() && actual.to_bits() == expected.to_bits(),
                "round {round}, output {index}: recorded={actual}, immediate={expected}"
            );
            checksum = checksum.wrapping_add(u64::from(actual.to_bits()));
        }
    }
    ensure!(
        source.reads.load(Ordering::Relaxed) == expected_reads,
        "recorded reuse reread weights"
    );
    report["execution"] = json!("cpu-recorded");
    report["rounds"] = json!(3);
    report["recorded_commands"] = json!(batch.len());
    report["preparation_reads"] = json!(expected_reads);
    report["matches_immediate_cpu_bits"] = json!(true);
    report["output_checksum"] = json!(checksum);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn immediate_reference(
    checkpoint: &Gguf,
    layer: usize,
    input: &[f32],
    width: usize,
) -> Result<Vec<f32>> {
    let cpu = Cpu::new();
    let project = |name: &str, input: &[f32], output: &mut [f32]| -> Result<()> {
        let tensor = checkpoint
            .get(&format!("blk.{layer}.{name}.weight"))
            .with_context(|| format!("missing reference weight {name}"))?;
        let storage = ResidentWeight::load(checkpoint, tensor.locator())?;
        cpu.matvec(
            output,
            input,
            &Weight {
                bytes: storage.bytes(),
                kind: tensor.kind(),
                rows: output.len(),
                cols: input.len(),
            },
        )?;
        Ok(())
    };
    let mut gate = vec![0.0; width];
    let mut up = vec![0.0; width];
    project("ffn_gate", input, &mut gate)?;
    project("ffn_up", input, &mut up)?;
    cpu.swiglu(&mut gate, &up, 0.0)?;
    let mut output = vec![0.0; input.len()];
    project("ffn_down", &gate, &mut output)?;
    Ok(output)
}
