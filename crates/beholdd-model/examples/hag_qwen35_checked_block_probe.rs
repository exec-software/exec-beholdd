//! Staged, unqualified CHECKED actual-weight Qwen3.5 block diagnostic.
//! Intended as a beholdd-model example with required-features=["hag-native"].
//! No CPU arithmetic fallback or model discovery. External OOM/thermal/crash and
//! hard-timeout supervision is mandatory. Only explicit post-FFN hidden outputs
//! and finite/Delta control status are read; this is not a production admission
//! path. One mandatory shared checker guards every checked component boundary.

#[cfg(target_os = "linux")]
mod proof {
    use std::env;
    use std::ffi::{OsStr, OsString};
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, Read, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use anyhow::{bail, ensure, Context, Result};
    use beholdd_backend::execution::{
        ExecutionDevice, F32Tensor, F32TensorView, FiniteCheck, FiniteCheckSpec,
    };
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionProfile,
        NativeExecutionStats, NativeRecordingFootprint,
    };
    use beholdd_format::{CheckpointId, Gguf, GgufTensor, RangeReader, WeightRange};
    use beholdd_memory::{ResidentSource, ResidentWeight, WeightAccess, WeightSource};
    use beholdd_model::qwen35::config::{Attention, Binding};
    use beholdd_model::qwen35::recorded_attention::{
        AttentionLimits, RecordedAttentionPlan, ResidentAttention,
    };
    use beholdd_model::qwen35::recorded_delta::{DeltaLimits, RecordedDeltaPlan, ResidentDelta};
    use beholdd_model::qwen35::recorded_ffn::{FfnLimits, RecordedFfnPlan, ResidentFfn};
    use beholdd_model::qwen35::recorded_weights::{PackedRowPlan, StreamStats};
    use beholdd_model::qwen35::weights::ReadLimits;
    use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
    use serde::Serialize;

    const MIB: u64 = 1024 * 1024;
    const MAX_JSON: usize = 1024 * 1024;
    const MAX_POSITIONS: usize = 5;
    const WIDTH: usize = 5120;
    const GUARD: u32 = 0x4b12_3456;
    // These are explicit conservative ceilings, not exact process/RSS forecasts.
    // The external scope must independently allow GGUF metadata, page cache, driver
    // memory and runtime overhead; streamed tiles do not cap those costs.
    const HOST_HEADROOM: u64 = 16 * MIB;

    const USAGE: &str = r#"native_qwen35_checked_block_probe --native /dev/dri/renderD... \
  --model MODEL.gguf --layer 0|3 --input INPUT.json --output OUTPUT.json \
  --max-positions 1..5 --gpu-budget-mib N --host-payload-budget-mib N \
  --deadline-secs 1..180 [--tile-rows 1..64] [--encoded-mib 1..4] [--decoded-mib 1..8]

Input: {"schema_version":1,"hidden":[[...5120 finite f32 values...], ...]}.
Both JSON files are bounded to 1 MiB; output must not exist. Defaults: tile64,
encoded4MiB, decoded8MiB. Explicit GPU budget<=64MiB, host payload budget<=128MiB.
Require HAG_TRACE_SUBMISSION=0 and external OOM/thermal/crash/hard-timeout guards.
No automatic model lookup, no unchecked/CPU fallback, no serving admission. Main sequence
capacity equals input row count, then a no-effect capacity refusal and reset/first
replay are checked. Output rows are diagnostic post-FFN activations for independent
frozen-oracle comparison. A successful receipt proves completion, not parity.
Deadline is cooperative between reads, tiles, submissions and output operations;
it cannot interrupt blocked I/O or an in-flight GPU dispatch. GGUF metadata, OS
page cache and driver/runtime allocations are outside the host payload ceiling."#;

    struct Args {
        native: PathBuf,
        model: PathBuf,
        layer: usize,
        input: PathBuf,
        output: PathBuf,
        max_positions: usize,
        gpu_budget: u64,
        host_budget: u64,
        deadline_secs: u64,
        limits: ReadLimits,
    }

    fn number<T: std::str::FromStr>(value: OsString, name: &str) -> Result<T> {
        value
            .to_str()
            .context("numeric option is not UTF-8")?
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid number for {name}"))
    }

    fn args() -> Result<Option<Args>> {
        let mut options = std::collections::BTreeMap::<String, OsString>::new();
        let mut values = env::args_os().skip(1);
        while let Some(flag) = values.next() {
            if flag == "--help" || flag == "-h" {
                println!("{USAGE}");
                return Ok(None);
            }
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option is not UTF-8"))?;
            ensure!(
                matches!(
                    flag.as_str(),
                    "--native"
                        | "--model"
                        | "--layer"
                        | "--input"
                        | "--output"
                        | "--max-positions"
                        | "--gpu-budget-mib"
                        | "--host-payload-budget-mib"
                        | "--deadline-secs"
                        | "--tile-rows"
                        | "--encoded-mib"
                        | "--decoded-mib"
                ),
                "unknown option {flag}"
            );
            let value = values
                .next()
                .with_context(|| format!("missing value for {flag}"))?;
            ensure!(
                options.insert(flag.clone(), value).is_none(),
                "duplicate option {flag}"
            );
        }
        let mut take = |name: &str| {
            options
                .remove(name)
                .with_context(|| format!("required option {name}; use --help"))
        };
        let native = PathBuf::from(take("--native")?);
        let model = PathBuf::from(take("--model")?);
        let layer = number(take("--layer")?, "--layer")?;
        let input = PathBuf::from(take("--input")?);
        let output = PathBuf::from(take("--output")?);
        let max_positions = number(take("--max-positions")?, "--max-positions")?;
        let gpu_mib: u64 = number(take("--gpu-budget-mib")?, "--gpu-budget-mib")?;
        let host_mib: u64 = number(
            take("--host-payload-budget-mib")?,
            "--host-payload-budget-mib",
        )?;
        let deadline_secs = number(take("--deadline-secs")?, "--deadline-secs")?;
        let mut optional = |name: &str, default: usize| -> Result<usize> {
            options
                .remove(name)
                .map(|v| number(v, name))
                .transpose()
                .map(|v| v.unwrap_or(default))
        };
        let tile_rows = optional("--tile-rows", 64)?;
        let encoded_mib = optional("--encoded-mib", 4)?;
        let decoded_mib = optional("--decoded-mib", 8)?;
        ensure!(
            matches!(layer, 0 | 3),
            "this first diagnostic admits only layers 0 and 3"
        );
        ensure!(
            (1..=MAX_POSITIONS).contains(&max_positions),
            "positions must be 1..=5"
        );
        ensure!((1..=64).contains(&tile_rows), "tile rows must be 1..=64");
        ensure!(
            (1..=4).contains(&encoded_mib) && (1..=8).contains(&decoded_mib),
            "encoded/decoded limits must be 1..4/1..8 MiB"
        );
        ensure!(
            (1..=64).contains(&gpu_mib) && (1..=128).contains(&host_mib),
            "GPU/host payload budgets must be 1..64/1..128 MiB"
        );
        ensure!(
            (1..=180).contains(&deadline_secs),
            "deadline must be 1..=180 seconds"
        );
        let suffix = native
            .file_name()
            .and_then(OsStr::to_str)
            .and_then(|s| s.strip_prefix("renderD"))
            .unwrap_or("");
        ensure!(
            native.parent() == Some(Path::new("/dev/dri"))
                && !suffix.is_empty()
                && suffix.bytes().all(|b| b.is_ascii_digit()),
            "explicit /dev/dri/renderD... required"
        );
        Ok(Some(Args {
            native,
            model,
            layer,
            input,
            output,
            max_positions,
            gpu_budget: gpu_mib * MIB,
            host_budget: host_mib * MIB,
            deadline_secs,
            limits: ReadLimits {
                tile_rows,
                encoded_bytes: encoded_mib * MIB as usize,
                decoded_bytes: decoded_mib * MIB as usize,
            },
        }))
    }

    fn check_deadline(deadline: Instant) -> Result<()> {
        ensure!(
            Instant::now() < deadline,
            "native block cooperative deadline exceeded"
        );
        Ok(())
    }

    fn regular_file(path: &Path) -> Result<()> {
        ensure!(
            fs::metadata(path)?.is_file(),
            "{} is not a regular file",
            path.display()
        );
        Ok(())
    }

    fn read_input(path: &Path) -> Result<Vec<u8>> {
        let file = File::open(path).context("open explicit input")?;
        ensure!(file.metadata()?.is_file(), "input is not a regular file");
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(MAX_JSON + 1)
            .context("allocate bounded input JSON")?;
        file.take((MAX_JSON + 1) as u64).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= MAX_JSON, "input JSON exceeds 1 MiB");
        Ok(bytes)
    }

    // Seeded visitors preserve the CPU block probe's direct serde-f32 boundary and
    // reject excess rows/width, unknown/duplicate keys and nonfinite values in parse.
    struct InputSeed {
        width: usize,
        max_positions: usize,
    }
    struct HiddenSeed {
        width: usize,
        max_positions: usize,
    }
    struct RowSeed {
        width: usize,
        permitted: bool,
    }

    impl<'de> DeserializeSeed<'de> for InputSeed {
        type Value = Vec<Vec<f32>>;
        fn deserialize<D: de::Deserializer<'de>>(
            self,
            d: D,
        ) -> std::result::Result<Self::Value, D::Error> {
            d.deserialize_map(self)
        }
    }
    impl<'de> Visitor<'de> for InputSeed {
        type Value = Vec<Vec<f32>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("schema_version=1 and bounded hidden rows")
        }
        fn visit_map<A: MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut version = None;
            let mut hidden = None;
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "schema_version" => {
                        if version.is_some() {
                            return Err(de::Error::duplicate_field("schema_version"));
                        }
                        let value = map.next_value::<u32>()?;
                        if value != 1 {
                            return Err(de::Error::custom("schema_version must be 1"));
                        }
                        version = Some(value);
                    }
                    "hidden" => {
                        if hidden.is_some() {
                            return Err(de::Error::duplicate_field("hidden"));
                        }
                        hidden = Some(map.next_value_seed(HiddenSeed {
                            width: self.width,
                            max_positions: self.max_positions,
                        })?);
                    }
                    _ => {
                        return Err(de::Error::unknown_field(
                            &key,
                            &["schema_version", "hidden"],
                        ))
                    }
                }
            }
            version.ok_or_else(|| de::Error::missing_field("schema_version"))?;
            hidden.ok_or_else(|| de::Error::missing_field("hidden"))
        }
    }
    impl<'de> DeserializeSeed<'de> for HiddenSeed {
        type Value = Vec<Vec<f32>>;
        fn deserialize<D: de::Deserializer<'de>>(
            self,
            d: D,
        ) -> std::result::Result<Self::Value, D::Error> {
            d.deserialize_seq(self)
        }
    }
    impl<'de> Visitor<'de> for HiddenSeed {
        type Value = Vec<Vec<f32>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("1..5 hidden rows")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut rows = Vec::new();
            rows.try_reserve_exact(self.max_positions)
                .map_err(de::Error::custom)?;
            while let Some(row) = seq.next_element_seed(RowSeed {
                width: self.width,
                permitted: rows.len() < self.max_positions,
            })? {
                rows.push(row);
            }
            if rows.is_empty() {
                return Err(de::Error::custom("hidden must contain a row"));
            }
            Ok(rows)
        }
    }
    impl<'de> DeserializeSeed<'de> for RowSeed {
        type Value = Vec<f32>;
        fn deserialize<D: de::Deserializer<'de>>(
            self,
            d: D,
        ) -> std::result::Result<Self::Value, D::Error> {
            if !self.permitted {
                return Err(de::Error::custom("too many hidden rows"));
            }
            d.deserialize_seq(self)
        }
    }
    impl<'de> Visitor<'de> for RowSeed {
        type Value = Vec<f32>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("5120 finite f32 values")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut row = Vec::new();
            row.try_reserve_exact(self.width)
                .map_err(de::Error::custom)?;
            while let Some(value) = seq.next_element::<f32>()? {
                if row.len() == self.width {
                    return Err(de::Error::custom("hidden row too wide"));
                }
                if !value.is_finite() {
                    return Err(de::Error::custom("nonfinite hidden value"));
                }
                row.push(value);
            }
            if row.len() != self.width {
                return Err(de::Error::custom("hidden width differs"));
            }
            Ok(row)
        }
    }

    #[derive(Clone, Copy, Default, PartialEq, Eq, Serialize)]
    struct ReadStats {
        attempts: u64,
        calls: u64,
        bytes: u64,
        largest_read_bytes: u64,
    }
    struct MeasuredSource<'a> {
        direct: ResidentSource<'a>,
        stats: Mutex<ReadStats>,
        deadline: Instant,
        max_read: u64,
        max_total: u64,
    }
    impl MeasuredSource<'_> {
        fn snapshot(&self) -> Result<ReadStats> {
            Ok(*self
                .stats
                .lock()
                .map_err(|_| anyhow::anyhow!("read statistics poisoned"))?)
        }
    }
    impl WeightSource for MeasuredSource<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.direct.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            check_deadline(self.deadline)?;
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| anyhow::anyhow!("read statistics poisoned"))?;
            stats.attempts = stats
                .attempts
                .checked_add(1)
                .context("read count overflow")?;
            let total = stats
                .bytes
                .checked_add(range.nbytes())
                .context("read bytes overflow")?;
            ensure!(
                range.nbytes() <= self.max_read && total <= self.max_total,
                "weight read exceeds admitted slice/total byte bound"
            );
            let weight = self.direct.load_range(range, access)?;
            ensure!(
                weight.bytes().len() as u64 == range.nbytes(),
                "source slice length differs"
            );
            stats.calls += 1;
            stats.bytes = total;
            stats.largest_read_bytes = stats.largest_read_bytes.max(range.nbytes());
            check_deadline(self.deadline)?;
            Ok(weight)
        }
    }

    enum CorePlan {
        Full(RecordedAttentionPlan),
        Delta(RecordedDeltaPlan),
    }
    enum Core {
        Full(ResidentAttention),
        Delta(ResidentDelta),
    }
    impl CorePlan {
        fn memory(&self) -> (u64, u64, u64) {
            match self {
                Self::Full(p) => {
                    let m = p.memory();
                    (
                        m.resident_tensor_bytes,
                        m.packed_tile_bytes,
                        m.host_staging_bytes,
                    )
                }
                Self::Delta(p) => {
                    let m = p.memory();
                    (
                        m.resident_tensor_bytes,
                        m.packed_tile_bytes,
                        m.host_staging_bytes,
                    )
                }
            }
        }
        fn finite_requirements(&self, spec: Option<FiniteCheckSpec>) -> Result<u64> {
            Ok(match self {
                Self::Full(p) => p.finite_check_requirements(spec)?.checks_per_step,
                Self::Delta(p) => p.finite_check_requirements(spec)?.checks_per_step,
            })
        }
        fn prepare_checked(
            &self,
            device: &NativeExecutionDevice,
            source: &dyn WeightSource,
            control: &mut dyn FnMut() -> Result<()>,
        ) -> Result<Core> {
            let (resident_tensor_bytes, packed_tile_bytes, host_staging_bytes) = self.memory();
            Ok(match self {
                Self::Full(p) => Core::Full(p.prepare_checked(
                    device,
                    source,
                    AttentionLimits {
                        resident_tensor_bytes,
                        packed_tile_bytes,
                        host_staging_bytes,
                    },
                    control,
                )?),
                Self::Delta(p) => Core::Delta(p.prepare_checked(
                    device,
                    source,
                    DeltaLimits {
                        resident_tensor_bytes,
                        packed_tile_bytes,
                        host_staging_bytes,
                    },
                    control,
                )?),
            })
        }
    }
    impl Core {
        fn positions(&self) -> usize {
            match self {
                Self::Full(v) => v.positions(),
                Self::Delta(v) => v.positions(),
            }
        }
        fn is_poisoned(&self) -> bool {
            match self {
                Self::Full(v) => v.is_poisoned(),
                Self::Delta(v) => v.is_poisoned(),
            }
        }
        fn reset_bytes(&self) -> u64 {
            match self {
                Self::Full(v) => v.reset_write_bytes(),
                Self::Delta(v) => {
                    v.outside_reset_write_bytes() + v.sequence_stats().reset_write_bytes
                }
            }
        }
        fn status_reads(&self) -> (u64, u64) {
            match self {
                Self::Full(_) => (0, 0),
                Self::Delta(v) => {
                    let s = v.sequence_stats();
                    (s.status_reads, s.status_read_bytes)
                }
            }
        }
        fn step_checked(
            &mut self,
            device: &NativeExecutionDevice,
            source: &dyn WeightSource,
            hidden: &F32TensorView,
            checker: &mut FiniteCheck,
            stats: &mut StreamStats,
            control: &mut dyn FnMut() -> Result<()>,
        ) -> Result<()> {
            match self {
                Self::Full(v) => v.step_checked(device, source, hidden, checker, stats, control),
                Self::Delta(v) => v.step_checked(source, hidden, checker, stats, control),
            }
        }
        fn reset(
            &mut self,
            device: &NativeExecutionDevice,
            control: &mut dyn FnMut() -> Result<()>,
        ) -> Result<()> {
            match self {
                Self::Full(v) => v.reset(device, control),
                Self::Delta(v) => v.reset(control),
            }
        }
    }

    #[derive(Serialize)]
    struct Forecast {
        resident_tensor_bytes: u64,
        weight_capacity_bytes: u32,
        tensor_capacity_bytes: u32,
        auxiliary_capacity_bytes: u32,
        allocation_bytes: u64,
        model_host_staging_bytes: u64,
        host_payload_bound_bytes: u64,
        gpu_budget_bytes: u64,
        host_payload_budget_bytes: u64,
        preparation_read_bytes: u64,
        per_step_packed_bytes: u64,
        per_step_packed_tiles: u64,
        expected_packed_bytes: u64,
        expected_packed_tiles: u64,
        expected_source_bytes: u64,
        expected_source_calls: u64,
        largest_source_slice: u64,
        expected_reset_bytes: u64,
        checker_resident_bytes: u64,
        checker_initialization_write_bytes: u64,
        checker_host_scratch_bytes: u64,
        finite_checks_per_step: u64,
        expected_finite_checks: u64,
        expected_finite_pending_bytes: u64,
        expected_finite_status_bytes: u64,
        expected_delta_status_reads: u64,
        expected_delta_status_bytes: u64,
        expected_output_read_bytes: u64,
        expected_host_read_bytes: u64,
        expected_host_write_bytes: u64,
        expected_preparation_write_bytes: u64,
        expected_physical_dispatches: u64,
        expected_logical_submissions: u64,
        expected_auxiliary_peak_bytes: u64,
        recording_metadata_bound_bytes: u64,
        source_file_bytes: u64,
        source_cache_bytes: u64,
        source_resident_bytes: u64,
    }
    fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
        values
            .into_iter()
            .try_fold(0u64, |a, b| a.checked_add(b).context("forecast overflow"))
    }
    fn round4(value: u64) -> Result<u32> {
        u32::try_from(value.checked_add(3).context("arena alignment overflow")? & !3)
            .context("arena exceeds u32 bounds")
    }
    fn times(a: u64, b: u64) -> Result<u64> {
        a.checked_mul(b).context("forecast product overflow")
    }

    fn forecast(
        binding: &Binding,
        checkpoint: &Gguf,
        args: &Args,
        positions: usize,
        core: &CorePlan,
        ffn: &RecordedFfnPlan,
        spec: FiniteCheckSpec,
    ) -> Result<(Forecast, NativeExecutionConfig)> {
        let finite_checks_per_step = sum([
            core.finite_requirements(Some(spec))?,
            ffn.finite_check_requirements(Some(spec))?.checks_per_step,
        ])?;
        let block = &binding.blocks()[args.layer];
        let mut matrices: Vec<&GgufTensor> = vec![&block.ffn.gate, &block.ffn.up, &block.ffn.down];
        let mut parameters = vec![&block.attention_norm, &block.post_attention_norm];
        match &block.attention {
            Attention::Full(v) => {
                matrices.extend([&v.query_gate, &v.key, &v.value, &v.output]);
                parameters.extend([&v.query_norm, &v.key_norm]);
            }
            Attention::DeltaNet(v) => {
                matrices.extend([&v.qkv, &v.gate, &v.alpha, &v.beta, &v.output]);
                parameters.extend([&v.conv, &v.a, &v.dt_bias, &v.norm]);
            }
        }
        let preparation_read_bytes = sum(parameters.iter().map(|t| t.nbytes()))?;
        let preparation_read_calls = parameters.len() as u64;
        let per_step_packed_bytes = sum(matrices.iter().map(|t| t.nbytes()))?;
        let mut per_step_packed_tiles = 0u64;
        let mut scaled_tiles = 0u64;
        let mut scaled_rows = 0u64;
        let mut max_scaled_tile_rows = 0u64;
        let mut largest_source_slice = parameters.iter().map(|t| t.nbytes()).max().unwrap_or(0);
        for tensor in matrices {
            let plan = PackedRowPlan::new(tensor, args.limits)?;
            per_step_packed_tiles = sum([
                per_step_packed_tiles,
                plan.rows().div_ceil(plan.tile_rows()) as u64,
            ])?;
            largest_source_slice = largest_source_slice.max(plan.peak_read_bytes() as u64);
            let core_output = match &block.attention {
                Attention::Full(v) => &v.output,
                Attention::DeltaNet(v) => &v.output,
            };
            if tensor.locator() == block.ffn.down.locator()
                || tensor.locator() == core_output.locator()
            {
                scaled_tiles = sum([scaled_tiles, plan.rows().div_ceil(plan.tile_rows()) as u64])?;
                scaled_rows = sum([scaled_rows, plan.rows() as u64])?;
                max_scaled_tile_rows =
                    max_scaled_tile_rows.max(plan.rows().min(plan.tile_rows()) as u64);
            }
        }
        let (core_resident, core_packed, core_host) = core.memory();
        let fm = ffn.memory();
        let resident_tensor_bytes = sum([
            core_resident,
            fm.resident_tensor_bytes,
            ((WIDTH + 2) * 4) as u64,
            spec.resident_tensor_bytes,
        ])?;
        let weight_capacity_bytes = round4(core_packed.max(fm.packed_tile_bytes))?;
        let tensor_capacity_bytes = round4(resident_tensor_bytes)?;
        // Checked Full records four arithmetic batches only after the entry checker
        // has retired. Normalization/QK/attention scalar+workspace resources coexist
        // at that point; normalization drops before any matrix tile is recorded.
        // FFN retains norm+activation together; Delta parity recordings have no aux.
        let expected_auxiliary_peak_bytes = match core {
            CorePlan::Full(p) => sum([
                (WIDTH * 4 + 4) as u64,
                (p.shape().head_dim * 4 + 4) as u64,
                (positions * 12 + 8) as u64,
            ])?,
            CorePlan::Delta(_) => (WIDTH * 4 + 8) as u64,
        }
        .max((WIDTH * 4 + 8) as u64)
        .max(sum([times(max_scaled_tile_rows, 4)?, 4])?);
        let config = NativeExecutionConfig {
            weight_capacity_bytes,
            tensor_capacity_bytes,
            auxiliary_capacity_bytes: round4(expected_auxiliary_peak_bytes)?,
        };
        let allocation_bytes =
            config.allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?;
        ensure!(
            allocation_bytes <= args.gpu_budget,
            "GPU plan needs {allocation_bytes} bytes"
        );
        let model_host_staging_bytes = core_host.max(fm.host_staging_bytes);
        // Inputs, retained outputs, one replay/read buffer, both bounded JSON buffers,
        // backend conversion (one largest source/upload slice), and explicit metadata
        // headroom. Backend never retains a decoded packed tile on this native path.
        let (
            parities,
            core_tensor_allocations,
            recording_dispatches,
            largest_commands,
            largest_dispatches,
        ) = match core {
            CorePlan::Full(p) => (0, 1, 183, p.retained_commands() as u64, 84),
            CorePlan::Delta(_) => (2, 2, 16, 5, 5),
        };
        let recording_metadata_bound_bytes = NativeRecordingFootprint {
            batches: 4,
            dispatches: recording_dispatches,
            weight_allocations: 1,
            tensor_allocations: core_tensor_allocations + 3,
            auxiliary_allocations: 6,
            retained_allocation_references: parities + 32,
            noop_operands: 0,
            max_batch_commands: largest_commands,
            max_batch_dispatches: largest_dispatches,
        }
        .host_bytes()?;
        let host_payload_bound_bytes = sum([
            HOST_HEADROOM,
            model_host_staging_bytes,
            largest_source_slice,
            2 * MAX_JSON as u64 + 1,
            ((2 * positions + 2) * WIDTH * 4) as u64,
            2 * 65536,
            spec.host_scratch_bytes,
            recording_metadata_bound_bytes,
        ])?;
        ensure!(
            host_payload_bound_bytes <= args.host_budget,
            "host payload plan needs {host_payload_bound_bytes} bytes"
        );
        let runs = positions as u64 + 1;
        let expected_packed_bytes = per_step_packed_bytes
            .checked_mul(runs)
            .context("read forecast overflow")?;
        let expected_packed_tiles = per_step_packed_tiles
            .checked_mul(runs)
            .context("tile forecast overflow")?;
        let expected_source_bytes = sum([preparation_read_bytes, expected_packed_bytes])?;
        let expected_source_calls = sum([preparation_read_calls, expected_packed_tiles])?;
        // Reset excludes immutable norms and coefficients. The diagnostic hidden
        // guards survive reset; only its WIDTH mutable coordinates are zeroed.
        let core_reset = match core {
            CorePlan::Full(p) => {
                let shape = p.shape();
                let immutable = ((WIDTH + 2 * shape.head_dim) * 4) as u64
                    + (shape.capacity * shape.rope_dim * 4) as u64;
                core_resident
                    .checked_sub(immutable)
                    .context("attention reset forecast underflow")?
            }
            CorePlan::Delta(p) => u64::from(p.layout().frame_bytes) + (WIDTH * 4) as u64,
        };
        let expected_reset_bytes = sum([
            core_reset,
            fm.resident_tensor_bytes - (WIDTH * 4) as u64,
            (WIDTH * 4) as u64,
        ])?;
        let expected_finite_checks = times(finite_checks_per_step, runs)?;
        let expected_finite_pending_bytes =
            times(expected_finite_checks, spec.pending_write_bytes)?;
        let expected_finite_status_bytes = times(expected_finite_checks, spec.status_read_bytes)?;
        let (
            core_dispatches,
            core_submissions,
            core_recording_writes,
            core_immutable_writes,
            expected_delta_status_reads,
            expected_delta_status_bytes,
        ) = match core {
            CorePlan::Full(p) => {
                let s = p.shape();
                // Native Q/K RMS2+RoPE1 per head, attention3 per query head,
                // gate1 per query head, and one wide weighted hidden RMS3.
                let dispatches =
                    3 + 3 * (s.query_heads + s.kv_heads) as u64 + 4 * s.query_heads as u64;
                // Main visible positions1..N followed by reset replay at visible1.
                let visible_sum = sum([times(positions as u64, positions as u64 + 1)? / 2, 1])?;
                let writes = sum([
                    times(runs, (WIDTH * 4 + 4 + s.head_dim * 4 + 4 + 8) as u64)?,
                    times(12, visible_sum)?,
                ])?;
                let immutable = ((WIDTH + 2 * s.head_dim + positions * s.rope_dim) * 4) as u64;
                (dispatches, 4, writes, immutable, 0, 0)
            }
            CorePlan::Delta(p) => (
                7,
                2,
                times(runs, (WIDTH * 4 + 4) as u64)?,
                sum([(WIDTH * 4) as u64, p.layout().parameter_bytes as u64, 24])?,
                runs,
                times(runs, (p.shape().value_heads * 4) as u64)?,
            ),
        };
        let ffn_dispatches = 3 + ffn.activation_commands() as u64;
        let expected_physical_dispatches = sum([
            times(
                runs,
                sum([
                    per_step_packed_tiles,
                    scaled_tiles,
                    core_dispatches,
                    ffn_dispatches,
                ])?,
            )?,
            times(expected_finite_checks, spec.physical_dispatches)?,
        ])?;
        let expected_logical_submissions = sum([
            times(runs, sum([per_step_packed_tiles, core_submissions, 2])?)?,
            times(expected_finite_checks, spec.logical_submissions)?,
        ])?;
        let expected_output_read_bytes = times(runs, ((WIDTH + 2) * 4) as u64)?;
        let expected_host_read_bytes = sum([
            expected_output_read_bytes,
            expected_delta_status_bytes,
            expected_finite_status_bytes,
        ])?;
        // Tensor zeroing includes checker payload once; its remaining initialization
        // is guards+PENDING writes. No finite check runs during parameter preparation.
        let expected_preparation_write_bytes = sum([
            resident_tensor_bytes,
            core_immutable_writes,
            (WIDTH * 4) as u64,
            8,
            spec.initialization_write_bytes
                .checked_sub(spec.resident_tensor_bytes)
                .context("checker initialization smaller than its zeroed packet")?,
        ])?;
        let ffn_recording_writes = times(runs, (WIDTH * 4 + 8) as u64)?;
        let scaled_recording_writes = times(
            runs,
            sum([times(scaled_rows, 4)?, times(scaled_tiles, 4)?])?,
        )?;
        let expected_host_write_bytes = sum([
            expected_preparation_write_bytes,
            expected_packed_bytes,
            times(runs, (WIDTH * 4) as u64)?,
            core_recording_writes,
            ffn_recording_writes,
            scaled_recording_writes,
            expected_finite_pending_bytes,
            expected_reset_bytes,
        ])?;
        let source_file_bytes = (0..checkpoint.shard_count()).try_fold(0u64, |total, i| {
            sum([
                total,
                checkpoint
                    .shard_len(i)
                    .context("missing opened shard extent")?,
            ])
        })?;
        Ok((
            Forecast {
                resident_tensor_bytes,
                weight_capacity_bytes,
                tensor_capacity_bytes,
                auxiliary_capacity_bytes: config.auxiliary_capacity_bytes,
                allocation_bytes,
                model_host_staging_bytes,
                host_payload_bound_bytes,
                gpu_budget_bytes: args.gpu_budget,
                host_payload_budget_bytes: args.host_budget,
                preparation_read_bytes,
                per_step_packed_bytes,
                per_step_packed_tiles,
                expected_packed_bytes,
                expected_packed_tiles,
                expected_source_bytes,
                expected_source_calls,
                largest_source_slice,
                expected_reset_bytes,
                checker_resident_bytes: spec.resident_tensor_bytes,
                checker_initialization_write_bytes: spec.initialization_write_bytes,
                checker_host_scratch_bytes: spec.host_scratch_bytes,
                finite_checks_per_step,
                expected_finite_checks,
                expected_finite_pending_bytes,
                expected_finite_status_bytes,
                expected_delta_status_reads,
                expected_delta_status_bytes,
                expected_output_read_bytes,
                expected_host_read_bytes,
                expected_host_write_bytes,
                expected_preparation_write_bytes,
                expected_physical_dispatches,
                expected_logical_submissions,
                expected_auxiliary_peak_bytes,
                recording_metadata_bound_bytes,
                source_file_bytes,
                source_cache_bytes: 0,
                source_resident_bytes: 0,
            },
            config,
        ))
    }

    // Whole-block publication is distinct from each component's commit. In
    // particular an FFN/readback failure after attention commits leaves this owner
    // poisoned. No retry or partial rewind is performed on a failed native attempt.
    struct BlockRunner {
        core: Core,
        ffn: ResidentFfn,
        hidden: F32Tensor,
        checker: FiniteCheck,
        positions: usize,
        capacity: usize,
        poisoned: bool,
        guard_checks: u64,
        output_read_bytes: u64,
        hidden_reset_bytes: u64,
    }
    impl BlockRunner {
        fn hidden_view(&self) -> Result<F32TensorView> {
            Ok(self.hidden.view(1, WIDTH)?)
        }
        fn step_checked(
            &mut self,
            device: &NativeExecutionDevice,
            source: &dyn WeightSource,
            input: &[f32],
            stream: &mut StreamStats,
            deadline: Instant,
        ) -> Result<Vec<f32>> {
            ensure!(
                !self.poisoned && !self.core.is_poisoned() && !self.ffn.is_poisoned(),
                "block is poisoned"
            );
            ensure!(self.positions < self.capacity, "block capacity exhausted");
            ensure!(
                self.core.positions() == self.positions
                    && self.ffn.completed_steps() == self.positions as u64,
                "component position mismatch"
            );
            ensure!(
                input.len() == WIDTH && input.iter().all(|v| v.is_finite()),
                "invalid hidden input"
            );
            check_deadline(deadline)?;
            let hidden = self.hidden_view()?;
            self.poisoned = true;
            device.write_f32(&hidden, input)?;
            self.core.step_checked(
                device,
                source,
                &hidden,
                &mut self.checker,
                stream,
                &mut || check_deadline(deadline),
            )?;
            self.ffn.step_checked(
                device,
                source,
                &hidden,
                &mut self.checker,
                stream,
                &mut || check_deadline(deadline),
            )?;
            check_deadline(deadline)?;
            let mut guarded = Vec::new();
            guarded
                .try_reserve_exact(WIDTH + 2)
                .context("allocate diagnostic output")?;
            guarded.resize(WIDTH + 2, 0.0);
            device.read_f32(&self.hidden.full_view(), &mut guarded)?;
            self.output_read_bytes += ((WIDTH + 2) * 4) as u64;
            ensure!(
                guarded[0].to_bits() == GUARD && guarded[WIDTH + 1].to_bits() == GUARD,
                "diagnostic hidden guard changed"
            );
            self.guard_checks += 2;
            ensure!(
                guarded[1..WIDTH + 1].iter().all(|v| v.is_finite()),
                "nonfinite diagnostic block output"
            );
            check_deadline(deadline)?;
            ensure!(
                self.core.positions() == self.positions + 1
                    && self.ffn.completed_steps() == self.positions as u64 + 1
                    && !self.core.is_poisoned()
                    && !self.ffn.is_poisoned(),
                "component commit mismatch"
            );
            // Remove only guards, preserving each output's exact F32 bits.
            guarded.remove(0);
            guarded.pop();
            self.positions += 1;
            self.poisoned = false;
            Ok(guarded)
        }
        fn reset(&mut self, device: &NativeExecutionDevice, deadline: Instant) -> Result<()> {
            self.poisoned = true;
            self.core.reset(device, &mut || check_deadline(deadline))?;
            self.ffn.reset(device, &mut || check_deadline(deadline))?;
            check_deadline(deadline)?;
            let mut zeros = Vec::new();
            zeros
                .try_reserve_exact(WIDTH)
                .context("allocate bounded hidden reset")?;
            zeros.resize(WIDTH, 0.0);
            device.write_f32(&self.hidden_view()?, &zeros)?;
            self.hidden_reset_bytes += (WIDTH * 4) as u64;
            check_deadline(deadline)?;
            ensure!(
                self.core.positions() == 0
                    && self.ffn.completed_steps() == 0
                    && !self.core.is_poisoned()
                    && !self.ffn.is_poisoned(),
                "incomplete component reset"
            );
            self.positions = 0;
            self.poisoned = false;
            Ok(())
        }
    }

    #[derive(Default, Serialize)]
    struct Receipt {
        execution: &'static str,
        passed: bool,
        parity_compared: bool,
        generic_intermediate_finite_checks: bool,
        exact_accounting_verified: bool,
        expected_finite_checks: u64,
        expected_finite_pending_bytes: u64,
        expected_finite_status_bytes: u64,
        expected_physical_dispatches: u64,
        expected_logical_submissions: u64,
        expected_host_write_bytes: u64,
        expected_host_read_bytes: u64,
        positions: usize,
        final_positions_after_replay: usize,
        capacity_refusal_without_effects: bool,
        reset_replay_exact: bool,
        reset_write_bytes: u64,
        guard_checks: u64,
        output_read_bytes: u64,
        status_reads: u64,
        status_read_bytes: u64,
        streamed_read_attempts: u64,
        streamed_read_bytes: u64,
        streamed_uploaded_bytes: u64,
        completed_tiles: u64,
        main_reads: ReadStats,
        reads: ReadStats,
        physical_dispatches: u64,
        dispatch_attempts: u64,
        logical_submissions: u64,
        uploaded_weight_bytes: u64,
        host_write_bytes: u64,
        host_read_bytes: u64,
        weight_peak_bytes: u64,
        tensor_peak_bytes: u64,
        auxiliary_peak_bytes: u64,
        weight_live_bytes_after_drop: u64,
        tensor_live_bytes_after_drop: u64,
        auxiliary_live_bytes_after_drop: u64,
        allocation_bytes: u64,
        expected_allocation_bytes: u64,
        budget_current_bytes: u64,
        budget_after: u64,
        queue_poisoned: bool,
        block_poisoned_on_error: bool,
        elapsed_seconds: f64,
        error: Option<String>,
    }
    impl Receipt {
        fn queue(&mut self, s: NativeExecutionStats) {
            self.physical_dispatches = s.dispatch_count;
            self.dispatch_attempts = s.dispatch_attempts;
            self.logical_submissions = s.logical_submissions;
            self.uploaded_weight_bytes = s.uploaded_weight_bytes;
            self.host_write_bytes = s.host_write_bytes;
            self.host_read_bytes = s.host_read_bytes;
            self.weight_peak_bytes = s.weight_peak_bytes;
            self.tensor_peak_bytes = s.tensor_peak_bytes;
            self.auxiliary_peak_bytes = s.auxiliary_peak_bytes;
            self.weight_live_bytes_after_drop = s.weight_live_bytes;
            self.tensor_live_bytes_after_drop = s.tensor_live_bytes;
            self.auxiliary_live_bytes_after_drop = s.auxiliary_live_bytes;
            self.allocation_bytes = s.allocation_bytes;
            self.queue_poisoned = s.poisoned;
        }
    }

    #[derive(Serialize)]
    struct Output<'a> {
        schema_version: u32,
        layer: usize,
        block_kind: &'static str,
        positions: usize,
        hidden: &'a [Vec<f32>],
        forecast: &'a Forecast,
        receipt: &'a Receipt,
        tile_rows: usize,
    }
    struct BoundedOutput(Vec<u8>);
    impl Write for BoundedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > MAX_JSON - self.0.len() {
                return Err(io::Error::other("output exceeds 1 MiB"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn write_output(path: &Path, output: &Output<'_>, deadline: Instant) -> Result<()> {
        check_deadline(deadline)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(MAX_JSON)
            .context("allocate bounded output JSON")?;
        let mut buffer = BoundedOutput(bytes);
        serde_json::to_writer(&mut buffer, output)?;
        buffer.write_all(b"\n")?;
        check_deadline(deadline)?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let result = (|| -> Result<()> {
            file.write_all(&buffer.0)?;
            file.sync_all()?;
            check_deadline(deadline)
        })();
        if result.is_err() {
            drop(file);
            let _ = fs::remove_file(path);
        }
        result
    }

    pub fn run() -> Result<()> {
        let Some(args) = args()? else {
            return Ok(());
        };
        let started = Instant::now();
        let deadline = started
            .checked_add(Duration::from_secs(args.deadline_secs))
            .context("deadline overflow")?;
        ensure!(
            env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly 0 before device access"
        );
        match fs::symlink_metadata(&args.output) {
            Ok(_) => bail!("output already exists"),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).context("check output path"),
        }
        let json = read_input(&args.input)?;
        regular_file(&args.model)?;
        check_deadline(deadline)?;
        let gguf = Gguf::open(&args.model).context("read checkpoint headers")?;
        let binding = Binding::load(&gguf).context("strict qwen35 binding")?;
        let c = binding.config();
        ensure!(
            c.dim == WIDTH
                && c.feed_forward == 17408
                && c.n_layers == 64
                && c.n_heads == 24
                && c.n_kv_heads == 4
                && c.key_head_dim == 256
                && c.rope_dim == 64
                && c.ssm_group_count == 16
                && c.ssm_time_step_rank == 48
                && c.ssm_state_size == 128,
            "this first gate requires the explicit OpenMythos geometry"
        );
        ensure!(
            c.is_full_attention(args.layer)? == (args.layer == 3),
            "layer kind differs from gate"
        );
        let mut deserializer = serde_json::Deserializer::from_slice(&json);
        let input = InputSeed {
            width: WIDTH,
            max_positions: args.max_positions,
        }
        .deserialize(&mut deserializer)
        .context("parse bounded input")?;
        deserializer.end().context("trailing input JSON")?;
        let core = if args.layer == 3 {
            CorePlan::Full(RecordedAttentionPlan::new(
                &binding,
                args.layer,
                input.len(),
                args.limits,
            )?)
        } else {
            CorePlan::Delta(RecordedDeltaPlan::new(
                &binding,
                args.layer,
                input.len(),
                args.limits,
            )?)
        };
        let ffn = RecordedFfnPlan::new(&binding, args.layer, args.limits)?;
        // Query the actual typed immutable library without opening a queue. An
        // absent slot17 or unsupported boundary refuses before the large allocation
        // and before any parameter payload read; no diagnostic path is substituted.
        let spec = NativeExecutionProfile::Qwen35Probe
            .finite_check_spec()
            .context("checked probe requires native finite-validation slot17")?;
        let (forecast, config) = forecast(&binding, &gguf, &args, input.len(), &core, &ffn, spec)?;
        check_deadline(deadline)?;
        let source = MeasuredSource {
            direct: ResidentSource::new(&gguf),
            stats: Mutex::default(),
            deadline,
            max_read: forecast.largest_source_slice,
            max_total: forecast.expected_source_bytes,
        };
        let mut receipt = Receipt {
            execution: "native-qwen35-checked-block",
            generic_intermediate_finite_checks: true,
            expected_finite_checks: forecast.expected_finite_checks,
            expected_finite_pending_bytes: forecast.expected_finite_pending_bytes,
            expected_finite_status_bytes: forecast.expected_finite_status_bytes,
            expected_physical_dispatches: forecast.expected_physical_dispatches,
            expected_logical_submissions: forecast.expected_logical_submissions,
            expected_host_write_bytes: forecast.expected_host_write_bytes,
            expected_host_read_bytes: forecast.expected_host_read_bytes,
            expected_allocation_bytes: forecast.allocation_bytes,
            ..Receipt::default()
        };
        let budget = MemoryBudget::new(args.gpu_budget);
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(input.len())
            .context("allocate bounded output rows")?;
        let result = (|| -> Result<()> {
            let device = NativeExecutionDevice::open_with_profile(
                &args.native,
                &budget,
                config,
                NativeExecutionProfile::Qwen35Probe,
            )?;
            let execution = (|| -> Result<()> {
                ensure!(
                    device.finite_check_spec() == Some(spec),
                    "opened native finite capability differs from metadata admission"
                );
                ensure!(
                    device.stats()?
                        == NativeExecutionStats {
                            allocation_bytes: forecast.allocation_bytes,
                            ..NativeExecutionStats::default()
                        }
                        && budget.current_bytes() == forecast.allocation_bytes,
                    "initial queue accounting differs"
                );
                check_deadline(deadline)?;
                let checker = device.prepare_finite_check()?;
                let resident_core =
                    core.prepare_checked(&device, &source, &mut || check_deadline(deadline))?;
                let m = ffn.memory();
                let resident_ffn = ffn.prepare_checked(
                    &device,
                    &source,
                    FfnLimits {
                        resident_tensor_bytes: m.resident_tensor_bytes,
                        packed_tile_bytes: m.packed_tile_bytes,
                        host_staging_bytes: m.host_staging_bytes,
                    },
                    &mut || check_deadline(deadline),
                )?;
                let hidden = device.allocate_f32(WIDTH + 2)?;
                device.write_f32(&hidden.view(0, 1)?, &[f32::from_bits(GUARD)])?;
                device.write_f32(&hidden.view(WIDTH + 1, 1)?, &[f32::from_bits(GUARD)])?;
                let mut runner = BlockRunner {
                    core: resident_core,
                    ffn: resident_ffn,
                    hidden,
                    checker,
                    positions: 0,
                    capacity: input.len(),
                    poisoned: false,
                    guard_checks: 0,
                    output_read_bytes: 0,
                    hidden_reset_bytes: 0,
                };
                let mut stream = StreamStats::default();
                let steps = (|| -> Result<()> {
                    ensure!(
                        source.snapshot()?.bytes == forecast.preparation_read_bytes,
                        "preparation read byte forecast differs"
                    );
                    ensure!(
                        device.stats()?.host_write_bytes
                            == forecast.expected_preparation_write_bytes,
                        "preparation host write forecast differs"
                    );
                    for row in &input {
                        outputs.push(runner.step_checked(
                            &device,
                            &source,
                            row,
                            &mut stream,
                            deadline,
                        )?);
                    }
                    receipt.positions = runner.positions;
                    receipt.main_reads = source.snapshot()?;
                    let before_queue = device.stats()?;
                    let before_stream = stream;
                    // Exercise the real attention/Delta component capacity check,
                    // bypassing only the outer wrapper's matching preflight guard.
                    let hidden = runner.hidden_view()?;
                    let refusal = runner.core.step_checked(
                        &device,
                        &source,
                        &hidden,
                        &mut runner.checker,
                        &mut stream,
                        &mut || check_deadline(deadline),
                    );
                    ensure!(
                        refusal.is_err()
                            && device.stats()? == before_queue
                            && source.snapshot()? == receipt.main_reads
                            && stream == before_stream
                            && runner.positions == input.len()
                            && runner.core.positions() == input.len()
                            && runner.ffn.completed_steps() == input.len() as u64
                            && !runner.poisoned
                            && !runner.core.is_poisoned()
                            && !runner.ffn.is_poisoned(),
                        "capacity refusal changed reads, queue, positions or poison"
                    );
                    receipt.capacity_refusal_without_effects = true;
                    runner.reset(&device, deadline)?;
                    let replay =
                        runner.step_checked(&device, &source, &input[0], &mut stream, deadline)?;
                    ensure!(
                        replay
                            .iter()
                            .zip(&outputs[0])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "reset first-position replay differs bitwise"
                    );
                    receipt.reset_replay_exact = true;
                    receipt.final_positions_after_replay = runner.positions;
                    check_deadline(deadline)?;
                    Ok(())
                })();
                receipt.block_poisoned_on_error = steps.is_err()
                    && (runner.poisoned || runner.core.is_poisoned() || runner.ffn.is_poisoned());
                receipt.guard_checks = runner.guard_checks;
                receipt.output_read_bytes = runner.output_read_bytes;
                receipt.reset_write_bytes = runner.core.reset_bytes()
                    + runner.ffn.reset_write_bytes()
                    + runner.hidden_reset_bytes;
                (receipt.status_reads, receipt.status_read_bytes) = runner.core.status_reads();
                receipt.streamed_read_attempts = stream.read_attempts;
                receipt.streamed_read_bytes = stream.returned_read_bytes;
                receipt.streamed_uploaded_bytes = stream.uploaded_bytes;
                receipt.completed_tiles = stream.completed_tiles;
                drop(runner); // Drops private Delta batches/device clones before queue retirement.
                steps
            })();
            let stats = device.stats()?;
            receipt.queue(stats);
            if execution.is_ok() {
                ensure!(
                    !stats.poisoned
                        && stats.weight_live_bytes == 0
                        && stats.tensor_live_bytes == 0
                        && stats.auxiliary_live_bytes == 0,
                    "resident allocations did not retire"
                );
                ensure!(
                    stats.tensor_peak_bytes == forecast.resident_tensor_bytes
                        && stats.weight_peak_bytes == u64::from(config.weight_capacity_bytes)
                        && stats.auxiliary_peak_bytes == forecast.expected_auxiliary_peak_bytes,
                    "arena peak forecast differs"
                );
                ensure!(
                    receipt.completed_tiles == forecast.expected_packed_tiles
                        && receipt.streamed_read_attempts == forecast.expected_packed_tiles
                        && receipt.streamed_read_bytes == forecast.expected_packed_bytes
                        && receipt.streamed_uploaded_bytes == forecast.expected_packed_bytes
                        && stats.uploaded_weight_bytes == forecast.expected_packed_bytes
                        && source.snapshot()?.bytes == forecast.expected_source_bytes
                        && source.snapshot()?.calls == forecast.expected_source_calls
                        && source.snapshot()?.attempts == forecast.expected_source_calls,
                    "packed/source accounting differs"
                );
                ensure!(
                    receipt.output_read_bytes == forecast.expected_output_read_bytes
                        && receipt.status_reads == forecast.expected_delta_status_reads
                        && receipt.status_read_bytes == forecast.expected_delta_status_bytes
                        && stats.host_read_bytes == forecast.expected_host_read_bytes
                        && stats.host_read_bytes
                            == receipt.output_read_bytes
                                + receipt.status_read_bytes
                                + forecast.expected_finite_status_bytes,
                    "unexpected readback beyond diagnostic output and finite/Delta control"
                );
                ensure!(
                    stats.host_write_bytes == forecast.expected_host_write_bytes,
                    "exact host write forecast differs"
                );
                ensure!(
                    receipt.guard_checks == (input.len() as u64 + 1) * 2
                        && receipt.reset_write_bytes == forecast.expected_reset_bytes,
                    "guard/reset accounting differs"
                );
                ensure!(
                    stats.dispatch_count == stats.dispatch_attempts
                        && stats.dispatch_count == forecast.expected_physical_dispatches
                        && stats.logical_submissions == forecast.expected_logical_submissions
                        && stats.allocation_bytes == forecast.allocation_bytes
                        && budget.current_bytes() == forecast.allocation_bytes,
                    "queue accounting differs"
                );
                receipt.exact_accounting_verified = true;
            }
            drop(device);
            execution
        })();
        receipt.reads = source.snapshot()?;
        receipt.budget_current_bytes = budget.current_bytes();
        receipt.budget_after = budget.current_bytes();
        let result = result.and_then(|()| {
            ensure!(
                budget.current_bytes() == 0,
                "native queue budget remains charged after drop"
            );
            check_deadline(deadline)
        });
        receipt.elapsed_seconds = started.elapsed().as_secs_f64();
        receipt.passed = result.is_ok();
        receipt.error = result.as_ref().err().map(|e| format!("{e:#}"));
        if result.is_ok() {
            let output = Output {
                schema_version: 1,
                layer: args.layer,
                block_kind: if args.layer == 3 {
                    "full_attention"
                } else {
                    "delta_net"
                },
                positions: input.len(),
                hidden: &outputs,
                forecast: &forecast,
                receipt: &receipt,
                tile_rows: args.limits.tile_rows,
            };
            if let Err(error) = write_output(&args.output, &output, deadline) {
                receipt.passed = false;
                receipt.error = Some(format!("{error:#}"));
                println!("{}", serde_json::to_string(&receipt)?);
                return Err(error);
            }
        }
        println!("{}", serde_json::to_string(&receipt)?);
        result
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::run()
}
#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("checked native qwen35 block probe requires Linux")
}
