// Full C16384 allocation with two evaluated positions and one reset/replay.
// This gate does not qualify long-context numerics or serving performance.
use super::*;
use beholdd_backend::hag::native_execution::NativeRecordingFootprint;
use beholdd_memory::{available_memory, cgroup_memory};

const CONTEXT: usize = 16384;
pub(super) const CPU_STEPS: usize = 2;
const CHECK_POSITIONS: [usize; CPU_STEPS] = [0, 1];
const TENSOR_BYTES: u64 = 3_995_575_808;
const PERSISTENT_AUX: u64 = 119_124;
const PEAK_AUX: u32 = 315_748;
const HAG_BYTES: u64 = 4_629_168_128;
const PROPOSED_HAG_BUDGET: u64 = 4608 * 1024 * 1024;
const CPU_PROCESS_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const CPU_EARLY_STOP: u64 = 1536 * 1024 * 1024;
const NATIVE_PROCESS_LIMIT: u64 = 6 * 1024 * 1024 * 1024;
const NATIVE_EARLY_STOP: u64 = 5632 * 1024 * 1024;
const STARTUP_HEADROOM: u64 = 12 * 1024 * 1024 * 1024;
const RETAINED_HOST_HEADROOM: u64 = 4 * 1024 * 1024 * 1024;
#[cfg(test)]
const CPU_KV: u64 = (LAYERS * 2 * CPU_STEPS * 8 * 128 * 4) as u64;
// At C16384 the model-owned coefficient pair exceeds the packed tile maximum.
const MODEL_STAGING: u64 = (CONTEXT * 128 * 4) as u64;
// The coefficient pair and backend byte conversion coexist during preparation.
const PREPARATION_STAGING: u64 = (CONTEXT * (128 / 2) * 3 * 4) as u64;
// Two retained CPU references plus a working/readback vocabulary vector.
const REFERENCE_BYTES: u64 = 3 * VOCAB as u64 * 4;
const MEMORY_POLICY: &str = "separate-cpu-native-capacity16384-two-position-v1";

mod reference {
    include!("capacity16384_reference.rs");
}
mod execution {
    include!("capacity16384_execution.rs");
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
            .context("C16384 phase forecast overflows")
    })
}

#[derive(Debug, PartialEq, Eq)]
struct PhaseCosts {
    cpu_total: u64,
    native_host: u64,
    native_total: u64,
}

#[cfg(test)]
fn forecast(checkpoint: u64, prepared_metadata: u64) -> Result<PhaseCosts> {
    forecast_steps(checkpoint, prepared_metadata, CPU_STEPS)
}

fn forecast_steps(checkpoint: u64, prepared_metadata: u64, cpu_steps: usize) -> Result<PhaseCosts> {
    ensure!(
        cpu_steps == CPU_STEPS || cpu_steps == catalog_prefill32::CPU_STEPS,
        "unsupported CPU prefix length"
    );
    let cpu_kv = (LAYERS * 2 * cpu_steps * 8 * 128 * 4) as u64;
    // The whole checkpoint is counted independently for each process. Streaming
    // reference buffers and small owners fit inside the 16-MiB owner allowance;
    // a separate 128-MiB margin covers other unmodeled process allocations.
    let common = sum(&[checkpoint, REFERENCE_BYTES, 16 << 20, 128 << 20])?;
    let cpu_total = sum(&[common, STAGING, cpu_kv])?;
    let native_host = sum(&[
        common,
        PREPARATION_STAGING.max(STAGING),
        footprint().host_bytes()?,
        prepared_metadata,
    ])?;
    Ok(PhaseCosts {
        cpu_total,
        native_host,
        native_total: sum(&[native_host, HAG_BYTES])?,
    })
}

fn validate_phase_bounds(costs: &PhaseCosts) -> Result<()> {
    ensure!(
        costs.cpu_total < CPU_EARLY_STOP
            && costs.native_total < NATIVE_EARLY_STOP
            && costs.cpu_total <= CPU_PROCESS_LIMIT
            && costs.native_total <= NATIVE_PROCESS_LIMIT,
        "C16384 forecast exceeds phase process or early-stop bounds"
    );
    ensure!(
        HAG_BYTES <= PROPOSED_HAG_BUDGET,
        "C16384 device budget is insufficient"
    );
    Ok(())
}

fn phase_bounds(stage: &str) -> Result<(u64, u64)> {
    match stage {
        "cpu" => Ok((CPU_PROCESS_LIMIT, CPU_EARLY_STOP)),
        "native" => Ok((NATIVE_PROCESS_LIMIT, NATIVE_EARLY_STOP)),
        _ => bail!("unknown C16384 phase"),
    }
}

fn validate_scope(unit: &str, stage: &str, membership: &str, status: &str) -> Result<PathBuf> {
    phase_bounds(stage)?;
    let prefix = format!("hag-capacity16384-{stage}-");
    let suffix = unit.strip_prefix(&prefix).unwrap_or("");
    ensure!(
        !suffix.is_empty() && suffix.bytes().all(|v| v.is_ascii_digit()),
        "C16384 service identity differs"
    );
    let mut memberships = membership.lines();
    let path = memberships
        .next()
        .and_then(|v| v.strip_prefix("0::"))
        .context("C16384 requires unified cgroup membership")?;
    ensure!(
        memberships.next().is_none()
            && path.starts_with('/')
            && path.ends_with(&format!("/{unit}.service"))
            && path
                .split('/')
                .skip(1)
                .all(|p| !p.is_empty() && p != "." && p != ".."),
        "C16384 service does not own the current cgroup"
    );
    let mut nnp = status.lines().filter_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == "NoNewPrivs").then_some(value.trim())
    });
    ensure!(
        nnp.next() == Some("1") && nnp.next().is_none(),
        "C16384 requires NoNewPrivs=1"
    );
    Ok(std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/')))
}

fn require_scope(stage: &str) -> Result<()> {
    ensure!(
        std::env::var_os("HAG_CAPACITY16384_GATE").as_deref() == Some(OsStr::new("1")),
        "C16384 requires its frozen external supervisor"
    );
    let path = validate_scope(
        &std::env::var("HAG_CAPACITY16384_UNIT").context("missing C16384 service identity")?,
        stage,
        &std::fs::read_to_string("/proc/self/cgroup")?,
        &std::fs::read_to_string("/proc/self/status")?,
    )?;
    ensure!(
        path.canonicalize()? == path,
        "C16384 cgroup path is not canonical"
    );
    let (limit, early) = phase_bounds(stage)?;
    for (name, expected) in [
        ("memory.max", limit.to_string()),
        ("memory.high", early.to_string()),
        ("memory.swap.max", "0".to_owned()),
        ("memory.swap.current", "0".to_owned()),
        ("memory.swap.peak", "0".to_owned()),
        ("pids.max", "64".to_owned()),
        ("cpu.max", "25000 100000".to_owned()),
    ] {
        ensure!(
            std::fs::read_to_string(path.join(name))?.trim() == expected,
            "C16384 cgroup {name} differs"
        );
    }
    let events = std::fs::read_to_string(path.join("memory.events"))?;
    ensure!(!events.is_empty(), "C16384 memory events missing");
    for line in events.lines() {
        let mut parts = line.split_whitespace();
        ensure!(
            parts.next().is_some() && parts.next() == Some("0") && parts.next().is_none(),
            "C16384 observed memory pressure"
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

fn admit_sample(
    stage: &str,
    required: u64,
    additional: u64,
    memory: AdmissionMemory,
) -> Result<()> {
    let (limit, early) = phase_bounds(stage)?;
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
            sample.current <= sample.peak && sample.peak < early,
            "C16384 private charge is invalid or reaches early stop"
        );
    }
    ensure!(
        after.peak >= before.peak,
        "C16384 peak decreased while sampling"
    );
    let total_allowance = remaining
        .checked_add(before.current.min(after.current))
        .context("C16384 private total allowance overflows")?;
    ensure!(
        memory.host_available >= STARTUP_HEADROOM
            && required <= limit
            && required <= memory.host_available
            && required <= total_allowance,
        "C16384 total phase forecast exceeds live allowance"
    );
    ensure!(
        additional <= required
            && additional <= remaining
            && additional <= memory.host_available.saturating_sub(RETAINED_HOST_HEADROOM),
        "C16384 additional allocations exceed remaining headroom"
    );
    Ok(())
}

fn admit_phase(
    stage: &str,
    checkpoint_bytes: u64,
    prepared_metadata: u64,
    cpu_steps: usize,
) -> Result<PhaseCosts> {
    require_scope(stage)?;
    let costs = forecast_steps(checkpoint_bytes, prepared_metadata, cpu_steps)?;
    validate_phase_bounds(&costs)?;
    let (required, already_counted) = if stage == "cpu" {
        (costs.cpu_total, checkpoint_bytes)
    } else {
        (
            costs.native_total,
            sum(&[checkpoint_bytes, (CPU_STEPS * VOCAB * 4) as u64])?,
        )
    };
    let additional = required
        .checked_sub(already_counted)
        .context("C16384 additional forecast underflows")?;
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
        stage,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_forecasts_include_large_rope_staging_and_small_cpu_cache() {
        assert_eq!(PREPARATION_STAGING, 12_582_912);
        assert_eq!(MODEL_STAGING, 8_388_608);
        assert_eq!(CPU_KV, 458_752);
        let costs = forecast(637_124_608, 1 << 20).unwrap();
        let catalog = forecast_steps(637_124_608, 1 << 20, 32).unwrap();
        validate_phase_bounds(&catalog).unwrap();
        assert_eq!(catalog.cpu_total - costs.cpu_total, 6_881_280);
        assert_eq!(catalog.native_total, costs.native_total);
        assert!(forecast_steps(0, 0, 8079).is_err());
        validate_phase_bounds(&costs).unwrap();
        assert_eq!(costs.native_total - costs.native_host, HAG_BYTES);
        assert!(costs.cpu_total < CPU_PROCESS_LIMIT);
        assert!(costs.native_total > CPU_PROCESS_LIMIT);
        assert!(forecast(u64::MAX, 0).is_err());
        assert!(validate_phase_bounds(&PhaseCosts {
            native_total: NATIVE_EARLY_STOP,
            ..costs
        })
        .is_err());
    }

    #[test]
    fn explicit_scope_and_phase_specific_limits_are_required() {
        let unit = "hag-capacity16384-native-17";
        let membership = format!("0::/user.slice/{unit}.service\n");
        assert!(validate_scope(unit, "native", &membership, "NoNewPrivs:\t1\n").is_ok());
        for (bad_unit, stage, member, status) in [
            (unit, "cpu", membership.as_str(), "NoNewPrivs:\t1\n"),
            (unit, "native", membership.as_str(), "NoNewPrivs:\t0\n"),
            (unit, "native", "0::/other.service\n", "NoNewPrivs:\t1\n"),
            (
                "hag-capacity16384-native-x",
                "native",
                membership.as_str(),
                "NoNewPrivs:\t1\n",
            ),
        ] {
            assert!(validate_scope(bad_unit, stage, member, status).is_err());
        }
        let memory = AdmissionMemory {
            host_available: STARTUP_HEADROOM,
            ancestor_remaining: Some(NATIVE_PROCESS_LIMIT - (1 << 20)),
            before: Some(ChargeSample {
                current: 1 << 20,
                peak: 1 << 20,
            }),
            after: Some(ChargeSample {
                current: 1 << 20,
                peak: 1 << 20,
            }),
        };
        admit_sample("native", HAG_BYTES, HAG_BYTES, memory).unwrap();
        assert!(admit_sample("cpu", HAG_BYTES, HAG_BYTES, memory).is_err());
        assert!(admit_sample(
            "native",
            HAG_BYTES,
            HAG_BYTES,
            AdmissionMemory {
                host_available: STARTUP_HEADROOM - 1,
                ..memory
            }
        )
        .is_err());
        assert!(admit_sample(
            "native",
            HAG_BYTES,
            HAG_BYTES,
            AdmissionMemory {
                ancestor_remaining: Some(HAG_BYTES - 1),
                ..memory
            }
        )
        .is_err());
        assert!(admit_sample(
            "native",
            HAG_BYTES,
            HAG_BYTES,
            AdmissionMemory {
                after: Some(ChargeSample {
                    current: 1 << 20,
                    peak: NATIVE_EARLY_STOP
                }),
                ..memory
            }
        )
        .is_err());
    }
}
