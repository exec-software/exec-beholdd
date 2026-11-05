// Additive C2048 qualification with separate supervised CPU/native jobs. The
// native path consumes completed references and never creates a CPU KV owner.
use super::*;
use beholdd_backend::hag::native_execution::NativeRecordingFootprint;
use beholdd_memory::{available_memory, cgroup_memory};

pub(super) const CONTEXT: usize = 2048;
const CHECK_POSITIONS: [usize; 12] = [
    0, 127, 340, 341, 510, 511, 512, 1022, 1023, 1024, 2046, 2047,
];
const TENSOR_BYTES: u64 = 501_720_576;
const PERSISTENT_AUX: u64 = 119_124;
const PEAK_AUX: u32 = 143_716;
const HAG_BYTES: u64 = 1_135_140_864;
const PROPOSED_HAG_BUDGET: u64 = 1152 * 1024 * 1024;
const PROCESS_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const EARLY_STOP: u64 = 1536 * 1024 * 1024;
const CPU_KV: u64 = (LAYERS * 2 * CONTEXT * 8 * 128 * 4) as u64;
// Twelve retained checkpoints and one working/reset vocabulary vector.
const REFERENCE_BYTES: u64 = 13 * VOCAB as u64 * 4;
const MEMORY_POLICY: &str = "separate-cpu-native-processes-v1";

mod reference {
    include!("demand2048_reference.rs");
}
mod execution {
    include!("demand2048_execution.rs");
}

pub(super) fn run(options: &Options) -> Result<()> {
    execution::run(options)
}

fn footprint() -> NativeRecordingFootprint {
    NativeRecordingFootprint {
        batches: 31,
        dispatches: 364,
        weight_allocations: 234,
        tensor_allocations: 452,
        auxiliary_allocations: 119,
        retained_allocation_references: 404,
        noop_operands: 0,
        max_batch_commands: 54,
        max_batch_dispatches: 127,
    }
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |total, value| {
        total
            .checked_add(*value)
            .context("C2048 phase forecast overflows")
    })
}

#[derive(Debug, PartialEq, Eq)]
struct PhaseCosts {
    cpu_total: u64,
    native_host: u64,
    native_total: u64,
    all_overlap: u64,
}

fn forecast(checkpoint: u64, prepared_metadata: u64) -> Result<PhaseCosts> {
    // Both phases count the complete checkpoint independently. The 16-MiB owner
    // allowance and 128-MiB general headroom cover unmodeled process allocations.
    // C2048 coefficient pair plus backend conversion is 1,572,864 bytes, below
    // the existing packed-matrix staging maximum. Zero initialization is 64 KiB.
    // Streaming reference I/O and its header/record metadata fit inside the
    // separate 16-MiB owner allowance; no second complete artifact is retained.
    let common = sum(&[checkpoint, STAGING, REFERENCE_BYTES, 16 << 20, 128 << 20])?;
    let cpu_total = sum(&[common, CPU_KV])?;
    let native_host = sum(&[common, footprint().host_bytes()?, prepared_metadata])?;
    let native_total = sum(&[native_host, HAG_BYTES])?;
    let all_overlap = sum(&[native_total, CPU_KV])?;
    Ok(PhaseCosts {
        cpu_total,
        native_host,
        native_total,
        all_overlap,
    })
}

fn validate_phase_bounds(costs: &PhaseCosts, available: u64) -> Result<()> {
    // Arithmetic is not a live admission ticket. Each supervised process must
    // resample its own host/ancestor/cgroup headroom. The external supervisor
    // additionally proves CPU process retirement before launching a native job.
    ensure!(
        costs.cpu_total < EARLY_STOP && costs.native_host < EARLY_STOP,
        "C2048 host forecast exceeds unchanged early-stop bound"
    );
    ensure!(
        costs.cpu_total <= PROCESS_LIMIT
            && costs.native_total <= PROCESS_LIMIT
            && costs.cpu_total <= available
            && costs.native_total <= available,
        "C2048 phase exceeds process or available-memory bound"
    );
    ensure!(
        HAG_BYTES <= PROPOSED_HAG_BUDGET,
        "C2048 proposed device budget is insufficient"
    );
    Ok(())
}

fn validate_scope(unit: &str, stage: &str, membership: &str, status: &str) -> Result<PathBuf> {
    ensure!(
        matches!(stage, "cpu" | "first-token" | "context"),
        "unknown C2048 phase"
    );
    let prefix = format!("hag-demand2048-{stage}-");
    let suffix = unit.strip_prefix(&prefix).unwrap_or("");
    ensure!(
        !suffix.is_empty() && suffix.bytes().all(|v| v.is_ascii_digit()),
        "C2048 service identity differs"
    );
    let mut memberships = membership.lines();
    let path = memberships
        .next()
        .and_then(|v| v.strip_prefix("0::"))
        .context("C2048 requires unified cgroup membership")?;
    ensure!(
        memberships.next().is_none()
            && path.starts_with('/')
            && path.ends_with(&format!("/{unit}.service"))
            && path
                .split('/')
                .skip(1)
                .all(|p| !p.is_empty() && p != "." && p != ".."),
        "C2048 service does not own the current cgroup"
    );
    let mut nnp = status.lines().filter_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == "NoNewPrivs").then_some(value.trim())
    });
    ensure!(
        nnp.next() == Some("1") && nnp.next().is_none(),
        "C2048 requires NoNewPrivs=1"
    );
    Ok(std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/')))
}

fn require_scope(stage: &str) -> Result<()> {
    ensure!(
        std::env::var_os("HAG_DEMAND2048_GATE").as_deref() == Some(OsStr::new("1")),
        "C2048 requires its frozen external supervisor"
    );
    let path = validate_scope(
        &std::env::var("HAG_DEMAND2048_UNIT").context("missing C2048 service identity")?,
        stage,
        &std::fs::read_to_string("/proc/self/cgroup")?,
        &std::fs::read_to_string("/proc/self/status")?,
    )?;
    ensure!(
        path.canonicalize()? == path,
        "C2048 cgroup path is not canonical"
    );
    // The process verifies actual controls as well as the supervisor's receipt.
    for (name, expected) in [
        ("memory.max", "2147483648"),
        ("memory.high", "1610612736"),
        ("memory.swap.max", "0"),
        ("memory.swap.current", "0"),
        ("memory.swap.peak", "0"),
        ("pids.max", "64"),
        ("cpu.max", "75000 100000"),
    ] {
        ensure!(
            std::fs::read_to_string(path.join(name))?.trim() == expected,
            "C2048 cgroup {name} differs"
        );
    }
    let events = std::fs::read_to_string(path.join("memory.events"))?;
    ensure!(!events.is_empty(), "C2048 memory events missing");
    for line in events.lines() {
        let mut parts = line.split_whitespace();
        ensure!(
            parts.next().is_some() && parts.next() == Some("0") && parts.next().is_none(),
            "C2048 observed memory pressure"
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ChargeSample {
    current: u64,
    peak: u64,
}
#[derive(Clone, Copy)]
struct AdmissionMemory {
    host_available: u64,
    ancestor_remaining: Option<u64>,
    before: Option<ChargeSample>,
    after: Option<ChargeSample>,
}

fn admit_sample(required: u64, additional: u64, memory: AdmissionMemory) -> Result<()> {
    let before = memory
        .before
        .context("missing pre-admission private charge")?;
    let after = memory
        .after
        .context("missing post-admission private charge")?;
    let remaining = memory
        .ancestor_remaining
        .context("finite cgroup headroom required")?;
    for sample in [before, after] {
        ensure!(
            sample.current <= sample.peak && sample.peak < EARLY_STOP,
            "C2048 private charge is invalid or reaches early stop"
        );
    }
    ensure!(
        after.peak >= before.peak,
        "C2048 peak decreased while sampling"
    );
    let total_allowance = remaining
        .checked_add(before.current.min(after.current))
        .context("C2048 private total allowance overflows")?;
    ensure!(
        memory.host_available >= 8 * 1024 * 1024 * 1024
            && required <= PROCESS_LIMIT
            && required <= memory.host_available
            && required <= total_allowance,
        "C2048 total phase forecast exceeds live allowance"
    );
    ensure!(
        additional <= required && additional <= memory.host_available.min(remaining),
        "C2048 additional allocations exceed remaining headroom"
    );
    Ok(())
}

fn admit_phase(stage: &str, checkpoint_bytes: u64, prepared_metadata: u64) -> Result<PhaseCosts> {
    require_scope(stage)?;
    let costs = forecast(checkpoint_bytes, prepared_metadata)?;
    validate_phase_bounds(&costs, PROCESS_LIMIT)?;
    let (required, already_counted) = if stage == "cpu" {
        (costs.cpu_total, checkpoint_bytes)
    } else {
        // Native references have already been read and fully validated. Count
        // their resident payload once; reserve all future native allocations.
        (
            costs.native_total,
            sum(&[
                checkpoint_bytes,
                CHECK_POSITIONS.len() as u64 * VOCAB as u64 * 4,
            ])?,
        )
    };
    let additional = required
        .checked_sub(already_counted)
        .context("C2048 additional forecast underflows")?;
    let sample = || -> Result<Option<ChargeSample>> {
        Ok(cgroup_memory()?.map(|value| ChargeSample {
            current: value.current_bytes(),
            peak: value.peak_bytes(),
        }))
    };
    let before = sample()?;
    let available = available_memory()?;
    let after = sample()?;
    require_scope(stage)?;
    admit_sample(
        required,
        additional,
        AdmissionMemory {
            host_available: available.host_available_bytes(),
            ancestor_remaining: available.cgroup_available_bytes(),
            before,
            after,
        },
    )?;
    Ok(costs)
}

pub(super) fn inspect(options: &Options) -> Result<()> {
    ensure!(
        options.mode == Mode::InspectOnDemand2048 && options.device.is_none(),
        "C2048 inspection cannot execute a model"
    );
    let path = options
        .model
        .as_ref()
        .context("C2048 inspection needs checkpoint")?;
    check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
    let checkpoint = Gguf::open(path)?;
    check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
    let model = QwenModel::load(&checkpoint)?;
    ensure!(
        model.architecture() == QwenArchitecture::Qwen3,
        "dense qwen3 required"
    );
    let shape = Shape::from_model(&model);
    inspect_geometry(shape, CONTEXT)?;
    inspect_checkpoint(&checkpoint, shape)?;
    let plan = model.dense_model_plan_on_demand(CONTEXT, TILE_ROWS)?;
    let memory = plan.memory();
    ensure!(
        plan.requires_on_demand_attention()
            && plan.capacity() == CONTEXT
            && memory.packed_weight_bytes == PACKED
            && memory.resident_tensor_bytes == TENSOR_BYTES
            && memory.largest_staging_bytes == STAGING,
        "C2048 logical geometry/cost differs"
    );
    let config = NativeExecutionConfig {
        weight_capacity_bytes: u32::try_from(PACKED)?,
        tensor_capacity_bytes: u32::try_from(TENSOR_BYTES)?,
        auxiliary_capacity_bytes: PEAK_AUX,
    };
    ensure!(
        config.allocation_bytes()? == HAG_BYTES,
        "C2048 physical charge differs"
    );
    let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |total, path| {
        sum(&[total, std::fs::metadata(path)?.len()])
    })?;
    let retained = plan.on_demand_attention_host_metadata_bytes()?;
    let costs = forecast(checkpoint_bytes, retained)?;
    validate_phase_bounds(&costs, PROCESS_LIMIT)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "execution": "not_requested", "capacity": CONTEXT,
            "full_model_execution_enabled": true, "qualification_modes_only": true,
            "serving_capacity_changed": false,
            "memory_policy": MEMORY_POLICY,
            "cpu_process_retirement_required": true,
            "allocator_release_from_drop_assumed": false,
            "live_headroom_readmission_required": true,
            "checkpoint_allowance_per_phase_bytes": checkpoint_bytes,
            "packed_weight_bytes": PACKED, "resident_tensor_bytes": TENSOR_BYTES,
            "largest_logical_staging_bytes": STAGING,
            "coefficient_pair_and_conversion_peak_bytes": CONTEXT * 128 * 6,
            "bounded_zero_initialization_bytes": 64 * 1024,
            "persistent_auxiliary_bytes": PERSISTENT_AUX, "auxiliary_peak_bytes": PEAK_AUX,
            "expected_hag_allocation_bytes": HAG_BYTES,
            "proposed_hag_budget_bytes": PROPOSED_HAG_BUDGET,
            "native_recording_metadata_bytes": footprint().host_bytes()?,
            "retained_attention_metadata_bytes": retained,
            "cpu_kv_bytes": CPU_KV, "reference_and_working_logits_bytes": REFERENCE_BYTES,
            "cpu_phase_forecast_bytes": costs.cpu_total,
            "native_host_forecast_bytes": costs.native_host,
            "native_phase_combined_forecast_bytes": costs.native_total,
            "all_overlap_forecast_bytes": costs.all_overlap,
            "all_overlap_fits_process_limit": costs.all_overlap <= PROCESS_LIMIT,
            "process_limit_bytes": PROCESS_LIMIT, "early_stop_bytes": EARLY_STOP,
            "scope": "Metadata-only cost proposal; no payload preparation, inference, device open or qualification. Separate process retirement, reference identity, live admission and finite supervision remain required."
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_phases_fit_without_claiming_cpu_allocator_release() {
        let costs = forecast(639_447_744, 64_960).unwrap();
        // NativeTransport now owns ComputeQueue's 32-byte KernelAbi. It is
        // counted once by host_bytes(); CPU KV, GPU charge and margins are unchanged.
        assert_eq!(
            costs,
            PhaseCosts {
                cpu_total: 1_272_561_856,
                native_host: 803_261_010,
                native_total: 1_938_401_874,
                all_overlap: 2_408_163_922
            }
        );
        assert!(costs.all_overlap > PROCESS_LIMIT);
        validate_phase_bounds(&costs, costs.native_total).unwrap();
        assert!(validate_phase_bounds(&costs, costs.native_total - 1).is_err());
        let pressure = forecast(1_000_000_000, 64_960).unwrap();
        assert!(validate_phase_bounds(&pressure, u64::MAX).is_err());
        assert!(forecast(u64::MAX, 0).is_err());
        assert!(forecast(639_447_744, u64::MAX).is_err());
        assert_eq!(CPU_KV, 469_762_048);
        assert!((CONTEXT * 128 * 6) as u64 <= STAGING);
    }

    #[test]
    fn new_mode_is_metadata_only_and_never_accepts_a_render_node() {
        let args = |values: &[&str]| values.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse(args(&["--inspect-on-demand2048", "--model", "m"])).unwrap();
        assert_eq!(options.mode, Mode::InspectOnDemand2048);
        assert!(!options.mode.executes());
        assert_eq!(options.mode.steps(), 0);
        assert_eq!(options.mode.execution(), "not_requested");
        for flags in [
            vec![
                "--inspect-on-demand2048",
                "--model",
                "m",
                "/dev/dri/renderD128",
            ],
            vec!["--inspect-on-demand2048", "--execute", "--model", "m"],
            vec![
                "--on-demand2048-context",
                "--model",
                "m",
                "/dev/dri/renderD128",
            ],
        ] {
            assert!(parse(args(&flags)).is_err());
        }
    }

    #[test]
    fn split_modes_require_their_own_reference_direction_and_identity() {
        let args = |values: &[&str]| values.iter().map(OsString::from).collect::<Vec<_>>();
        let cpu = parse(args(&[
            "--on-demand2048-reference",
            "--model",
            "m",
            "--reference-out",
            "r",
            "--input-identity",
            "i",
        ]))
        .unwrap();
        assert_eq!(cpu.mode, Mode::OnDemand2048Reference);
        assert!(!cpu.mode.executes());
        assert_eq!(cpu.mode.steps(), 2048);
        for flag in ["--on-demand2048-first-token", "--on-demand2048-context"] {
            let native = parse(args(&[
                flag,
                "--model",
                "m",
                "--reference",
                "r",
                "--input-identity",
                "i",
                "/dev/dri/renderD128",
            ]))
            .unwrap();
            assert!(native.mode.executes());
            assert_eq!(
                native.mode.steps(),
                if flag.ends_with("context") { 2049 } else { 1 }
            );
            for invalid in [
                vec![
                    flag,
                    "--model",
                    "m",
                    "--reference",
                    "r",
                    "/dev/dri/renderD128",
                ],
                vec![
                    flag,
                    "--model",
                    "m",
                    "--reference-out",
                    "r",
                    "--input-identity",
                    "i",
                    "/dev/dri/renderD128",
                ],
                vec![
                    flag,
                    "--model",
                    "m",
                    "--reference",
                    "r",
                    "--input-identity",
                    "i",
                ],
                vec![
                    flag,
                    "--model",
                    "m",
                    "--reference",
                    "r",
                    "--reference-out",
                    "s",
                    "--input-identity",
                    "i",
                    "/dev/dri/renderD128",
                ],
            ] {
                assert!(parse(args(&invalid)).is_err());
            }
        }
        for invalid in [
            vec![
                "--on-demand2048-reference",
                "--reference-out",
                "r",
                "--input-identity",
                "i",
            ],
            vec![
                "--on-demand2048-reference",
                "--model",
                "m",
                "--reference",
                "r",
                "--input-identity",
                "i",
            ],
            vec![
                "--on-demand2048-reference",
                "--model",
                "m",
                "--reference-out",
                "r",
                "--input-identity",
                "i",
                "/dev/dri/renderD128",
            ],
            vec![
                "--inspect-on-demand2048",
                "--model",
                "m",
                "--reference",
                "r",
            ],
            vec![
                "--on-demand512-context",
                "--model",
                "m",
                "--reference",
                "r",
                "--input-identity",
                "i",
                "/dev/dri/renderD128",
            ],
        ] {
            assert!(parse(args(&invalid)).is_err());
        }
    }

    #[test]
    fn phase_admission_credits_only_private_charge_and_reserves_future_allocations() {
        let costs = forecast(639_447_744, 64_960).unwrap();
        let native_additional = costs.native_total - 639_447_744 - 12 * VOCAB as u64 * 4;
        assert_eq!(native_additional, 1_291_661_202);
        let cpu_additional = costs.cpu_total - 639_447_744;
        assert_eq!(cpu_additional, 633_114_112);
        let baseline = |current| AdmissionMemory {
            host_available: 8 * 1024 * 1024 * 1024,
            ancestor_remaining: Some(PROCESS_LIMIT - current),
            before: Some(ChargeSample {
                current,
                peak: current,
            }),
            after: Some(ChargeSample {
                current,
                peak: current,
            }),
        };
        for current in [48 * 1024 * 1024, 694_206_464] {
            let memory = baseline(current);
            admit_sample(costs.cpu_total, cpu_additional, memory).unwrap();
            admit_sample(costs.native_total, native_additional, memory).unwrap();
        }
        let memory = baseline(48 * 1024 * 1024);
        for invalid in [
            AdmissionMemory {
                before: None,
                ..memory
            },
            AdmissionMemory {
                after: None,
                ..memory
            },
            AdmissionMemory {
                ancestor_remaining: None,
                ..memory
            },
            AdmissionMemory {
                ancestor_remaining: Some(native_additional - 1),
                ..memory
            },
            AdmissionMemory {
                host_available: 8 * 1024 * 1024 * 1024 - 1,
                ..memory
            },
            AdmissionMemory {
                before: Some(ChargeSample {
                    current: 2,
                    peak: 1,
                }),
                ..memory
            },
            AdmissionMemory {
                after: Some(ChargeSample {
                    current: 1,
                    peak: 1,
                }),
                ..memory
            },
            AdmissionMemory {
                after: Some(ChargeSample {
                    current: 1,
                    peak: EARLY_STOP,
                }),
                ..memory
            },
            AdmissionMemory {
                ancestor_remaining: Some(u64::MAX),
                ..memory
            },
        ] {
            assert!(admit_sample(costs.native_total, native_additional, invalid).is_err());
        }
        assert!(admit_sample(PROCESS_LIMIT + 1, native_additional, memory).is_err());
        assert!(admit_sample(costs.native_total, costs.native_total + 1, memory).is_err());
    }

    #[test]
    fn phase_scope_needs_exact_service_membership_and_no_new_privileges() {
        for stage in ["cpu", "first-token", "context"] {
            let unit = format!("hag-demand2048-{stage}-123");
            let member = format!("0::/user.slice/{unit}.service\n");
            validate_scope(&unit, stage, &member, "NoNewPrivs:\t1\n").unwrap();
            for bad in [
                "",
                "0::/other.service",
                "0::/../bad",
                "0::/x\n1:memory:/x",
                "0::/x\n0::/x",
            ] {
                assert!(validate_scope(&unit, stage, bad, "NoNewPrivs: 1\n").is_err());
            }
            for bad in [
                "",
                "NoNewPrivs: 0",
                "NoNewPrivs: 01",
                "NoNewPrivs: 1\nNoNewPrivs: 1\n",
            ] {
                assert!(validate_scope(&unit, stage, &member, bad).is_err());
            }
            assert!(validate_scope(&unit, "unknown", &member, "NoNewPrivs: 1\n").is_err());
            assert!(
                validate_scope(&(unit.clone() + "x"), stage, &member, "NoNewPrivs: 1\n").is_err()
            );
        }
    }
}
