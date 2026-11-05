//! Turning a requested weight mode and fit estimate into a concrete execution plan, plus
//! the small CLI-argument helpers `decode`'s option parsing shares with its tests.
//!
//! `plan_execution` and `ExecutionPlan` capture a decision `beholdd-server` will need to
//! make too once it exists -- streaming vs. resident, capped to what the fit policy
//! actually authorized -- so this lives in the library rather than the example. That also
//! means the default `cargo test --workspace` run exercises it: it used to live in
//! `examples/decode.rs`, where cargo's default test selection silently skips example
//! targets.

use std::fmt;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use beholdd_memory::{
    CacheConfig, ExactStreamingPlan, FitDecision, FitDegradation, FitEstimate, FitPolicy,
    FitRefusal, FullyResidentPlan, StreamingFitDecision,
};

/// Whether weights are streamed on demand or fully materialized before decode starts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WeightMode {
    /// Weights are read as the forward pass demands them, bounded by a cache.
    #[default]
    Streaming,
    /// Every weight is loaded once, up front.
    Resident,
}

impl fmt::Display for WeightMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Streaming => "streaming",
            Self::Resident => "resident",
        })
    }
}

/// A weight-mode string was neither `streaming` nor `resident`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseWeightModeError {
    value: String,
}

impl fmt::Display for ParseWeightModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown weight mode {:?}; expected streaming or resident",
            self.value
        )
    }
}

impl std::error::Error for ParseWeightModeError {}

impl FromStr for WeightMode {
    type Err = ParseWeightModeError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "streaming" => Ok(Self::Streaming),
            "resident" => Ok(Self::Resident),
            _ => Err(ParseWeightModeError {
                value: value.to_owned(),
            }),
        }
    }
}

/// The concrete weight source a decode run will execute against, and its authorized
/// budget.
#[derive(Debug, Clone, Copy)]
pub enum ExecutionPlan {
    /// Stream weights on demand through a bounded cache.
    Streaming {
        /// The fit-checked forecast and budget behind this plan.
        plan: ExactStreamingPlan,
        /// The cache actually authorized -- may be smaller than what was requested.
        cache: CacheConfig,
    },
    /// Materialize every weight before decode starts.
    Resident(FullyResidentPlan),
}

/// Decide how `weight_mode` executes against `estimate` under `fit_policy`, and cap
/// `requested_cache` to whatever the fit decision authorized.
///
/// Refuses -- rather than silently downgrading -- when resident mode is asked to run
/// under a budget that only fits streaming, or when a streaming cache is requested under
/// resident mode.
pub fn plan_execution(
    weight_mode: WeightMode,
    estimate: FitEstimate,
    fit_policy: FitPolicy,
    memory_budget_bytes: u64,
    requested_cache: CacheConfig,
) -> Result<ExecutionPlan> {
    if matches!(weight_mode, WeightMode::Resident) && requested_cache.total_bytes() != 0 {
        bail!(
            "resident weight mode does not use streaming caches; omit the pinned, demand, and prefetch options or set their byte and queue limits to zero"
        );
    }
    Ok(match weight_mode {
        WeightMode::Streaming => {
            let plan = match estimate.decide_for_streaming(
                fit_policy,
                memory_budget_bytes,
                requested_cache.total_bytes(),
            ) {
                StreamingFitDecision::ExactStreaming(plan) => plan,
                StreamingFitDecision::ResidentExecutorRequired(plan) => bail!(
                    "fit policy refuse requires full residency; its {}-byte forecast fits the {}-byte budget, but --weight-mode streaming cannot satisfy that policy; choose --weight-mode resident or --fit-policy exact",
                    plan.forecast_bytes(),
                    plan.budget_bytes()
                ),
                StreamingFitDecision::Refused(refusal) => refuse_fit(refusal)?,
                StreamingFitDecision::DegradationRequired(required) => {
                    require_degradation(required)?
                }
            };
            ExecutionPlan::Streaming {
                cache: cap_cache(requested_cache, plan.cache_bytes())?,
                plan,
            }
        }
        WeightMode::Resident => match estimate.decide(fit_policy, memory_budget_bytes, 0) {
            FitDecision::FullyResident(plan) => ExecutionPlan::Resident(plan),
            FitDecision::ExactStreaming(plan) => bail!(
                "--weight-mode resident requires the full {}-byte forecast, but the {}-byte budget only authorizes exact streaming at {} bytes; raise --memory-budget-bytes or choose --weight-mode streaming",
                estimate.fully_resident_bytes(),
                plan.budget_bytes(),
                plan.forecast_bytes()
            ),
            FitDecision::Refused(refusal) => refuse_fit(refusal)?,
            FitDecision::DegradationRequired(required) => require_degradation(required)?,
        },
    })
}

fn cap_cache(requested: CacheConfig, authorized_bytes: u64) -> Result<CacheConfig> {
    let pinned = requested.pinned_bytes().min(authorized_bytes);
    let demand = requested
        .demand_bytes()
        .min(authorized_bytes.saturating_sub(pinned));
    let prefetch = requested.prefetch_bytes().min(
        authorized_bytes
            .saturating_sub(pinned)
            .saturating_sub(demand),
    );
    let queue_depth = if prefetch == 0 {
        0
    } else {
        requested.prefetch_queue_depth()
    };
    let workers = if prefetch == 0 {
        0
    } else {
        requested.prefetch_workers()
    };
    CacheConfig::new(pinned, demand)?.with_prefetch(prefetch, queue_depth, workers)
}

fn refuse_fit<T>(refusal: FitRefusal) -> Result<T> {
    bail!(
        "fit policy {} refused execution: {} bytes required, budget is {} bytes (short by {} bytes){}",
        refusal.policy(),
        refusal.required_bytes(),
        refusal.budget_bytes(),
        refusal.shortfall_bytes(),
        if refusal.exact_streaming_would_fit() {
            "; choose --fit-policy exact with --weight-mode streaming to stream unchanged"
        } else {
            ""
        }
    )
}

fn require_degradation<T>(required: FitDegradation) -> Result<T> {
    bail!(
        "fit policy degrade cannot execute implicitly: exact streaming needs {} bytes, budget is {} bytes (short by {} bytes); supply a named degradation before retrying",
        required.exact_required_bytes(),
        required.budget_bytes(),
        required.exact_shortfall_bytes()
    )
}

/// Set `*flag` the first time this option is seen; refuse a repeat.
pub fn set_flag_once(flag: &mut bool, name: &str) -> Result<()> {
    if *flag {
        bail!("{name} was provided more than once");
    }
    *flag = true;
    Ok(())
}

/// Set `*slot` the first time this option is seen; refuse a repeat.
pub fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<()> {
    if slot.is_some() {
        bail!("{flag} was provided more than once");
    }
    *slot = Some(value);
    Ok(())
}

/// Parse a non-negative byte count from a CLI option value.
pub fn parse_byte_count(value: &str, flag: &str) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("{flag} must be a non-negative integer byte count, got {value:?}"))
}

/// Parse a non-negative prefetch queue depth from a CLI option value.
pub fn parse_queue_depth(value: &str) -> Result<usize> {
    value.parse().with_context(|| {
        format!(
            "--prefetch-queue-depth must be a non-negative integer request count, got {value:?}"
        )
    })
}

/// Parse a non-negative prefetch worker-pool size from a CLI option value.
pub fn parse_prefetch_workers(value: &str) -> Result<usize> {
    value.parse().with_context(|| {
        format!("--prefetch-workers must be a non-negative integer thread count, got {value:?}")
    })
}

/// Validate a checksum supplied for a checkpoint file: exactly 64 lowercase hex
/// characters, the `sha256sum` convention.
///
/// Never computed here. Hashing a 30B-400B-class checkpoint would touch every byte and
/// distort the I/O timing this binary exists to measure, and a streaming or
/// demand-driven run does not read every byte in the first place -- a hash taken mid-run
/// would not even be trustworthy. Callers compute it once, outside the timed path
/// (`sha256sum <checkpoint>`), and pass it through so a paired-benchmark analyzer can
/// refuse to compare runs against different checkpoints instead of trusting encoded byte
/// count alone.
pub fn parse_checkpoint_sha256(value: &str) -> Result<String> {
    let is_lowercase_hex64 = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'));
    if !is_lowercase_hex64 {
        bail!("--checkpoint-sha256 must be exactly 64 lowercase hex characters, got {value:?}");
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_modes_are_strict_and_default_to_streaming() {
        assert_eq!(WeightMode::default(), WeightMode::Streaming);
        for (mode, spelling) in [
            (WeightMode::Streaming, "streaming"),
            (WeightMode::Resident, "resident"),
        ] {
            assert_eq!(mode.to_string(), spelling);
            assert_eq!(spelling.parse::<WeightMode>().unwrap(), mode);
        }
        assert_eq!(
            "direct".parse::<WeightMode>().unwrap_err().to_string(),
            "unknown weight mode \"direct\"; expected streaming or resident"
        );
        assert!("Resident".parse::<WeightMode>().is_err());
    }

    #[test]
    fn execution_plans_match_the_selected_weight_executor() {
        let estimate = FitEstimate::new(100, 50).unwrap();
        let cache = CacheConfig::new(20, 20).unwrap();

        match plan_execution(
            WeightMode::Streaming,
            estimate,
            FitPolicy::Exact,
            200,
            cache,
        )
        .unwrap()
        {
            ExecutionPlan::Streaming { plan, cache } => {
                assert_eq!(plan.forecast_bytes(), 90);
                assert_eq!(cache, CacheConfig::new(20, 20).unwrap());
            }
            ExecutionPlan::Resident(_) => panic!("streaming mode selected resident execution"),
        }

        match plan_execution(
            WeightMode::Resident,
            estimate,
            FitPolicy::Refuse,
            100,
            CacheConfig::default(),
        )
        .unwrap()
        {
            ExecutionPlan::Resident(plan) => assert_eq!(plan.forecast_bytes(), 100),
            ExecutionPlan::Streaming { .. } => {
                panic!("resident mode selected streaming execution")
            }
        }

        let error = plan_execution(
            WeightMode::Resident,
            estimate,
            FitPolicy::Exact,
            80,
            CacheConfig::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires the full 100-byte forecast"));
        assert!(error.contains("only authorizes exact streaming at 50 bytes"));

        let error = plan_execution(
            WeightMode::Resident,
            estimate,
            FitPolicy::Exact,
            200,
            CacheConfig::new(1, 0).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("resident weight mode does not use streaming caches"));

        let error = plan_execution(
            WeightMode::Streaming,
            estimate,
            FitPolicy::Refuse,
            100,
            CacheConfig::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--weight-mode streaming cannot satisfy that policy"));
    }

    #[test]
    fn cache_byte_options_are_strict_and_do_not_wrap() {
        assert_eq!(parse_byte_count("0", "--cache").unwrap(), 0);
        assert_eq!(
            parse_byte_count(&u64::MAX.to_string(), "--cache").unwrap(),
            u64::MAX
        );
        assert!(parse_byte_count("-1", "--cache").is_err());
        assert!(parse_byte_count("1GiB", "--cache").is_err());
        assert_eq!(parse_queue_depth("0").unwrap(), 0);
        assert_eq!(parse_queue_depth("17").unwrap(), 17);
        assert!(parse_queue_depth("-1").is_err());
        assert!(parse_queue_depth("many").is_err());

        let mut value = None;
        set_once(&mut value, 1u64, "--cache").unwrap();
        assert_eq!(value, Some(1));
        assert!(set_once(&mut value, 2, "--cache").is_err());
        assert_eq!(value, Some(1));

        let requested = CacheConfig::new(7, 11).unwrap();
        assert_eq!(cap_cache(requested, 100).unwrap(), requested);
        assert_eq!(
            cap_cache(requested, 10).unwrap(),
            CacheConfig::new(7, 3).unwrap()
        );
        assert_eq!(
            cap_cache(requested, 4).unwrap(),
            CacheConfig::new(4, 0).unwrap()
        );

        let requested = CacheConfig::new(7, 11)
            .unwrap()
            .with_prefetch(13, 5, 3)
            .unwrap();
        assert_eq!(cap_cache(requested, 100).unwrap(), requested);
        assert_eq!(
            cap_cache(requested, 24).unwrap(),
            CacheConfig::new(7, 11)
                .unwrap()
                .with_prefetch(6, 5, 3)
                .unwrap()
        );
        assert_eq!(
            cap_cache(requested, 18).unwrap(),
            CacheConfig::new(7, 11).unwrap()
        );

        assert_eq!(parse_prefetch_workers("0").unwrap(), 0);
        assert_eq!(parse_prefetch_workers("8").unwrap(), 8);
        assert!(parse_prefetch_workers("-1").is_err());
        assert!(parse_prefetch_workers("many").is_err());

        let mut flag = false;
        set_flag_once(&mut flag, "--flag").unwrap();
        assert!(flag);
        assert!(set_flag_once(&mut flag, "--flag").is_err());
    }

    #[test]
    fn checkpoint_sha256_is_strict_lowercase_hex() {
        let valid = "0123456789abcdef".repeat(4);
        assert_eq!(valid.len(), 64);
        assert_eq!(parse_checkpoint_sha256(&valid).unwrap(), valid);
        assert!(parse_checkpoint_sha256(&valid[..63]).is_err());
        assert!(parse_checkpoint_sha256(&(valid.clone() + "0")).is_err());
        assert!(parse_checkpoint_sha256(&valid.to_uppercase()).is_err());
        assert!(parse_checkpoint_sha256(&"g".repeat(64)).is_err());
        assert!(parse_checkpoint_sha256("").is_err());
    }
}
