use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use beholdd_memory::{CacheConfig, FitPolicy};
use beholdd_model::execution::WeightMode;
use serde::{Deserialize, Deserializer};

/// Conservative non-model allowance used when a config does not override it.
pub const DEFAULT_RUNTIME_HEADROOM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Strict YAML configuration for the native runtime.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    /// Models exposed by this process.
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    /// Optional hard execution budget shared as the default for every model.
    #[serde(default)]
    pub memory_budget_bytes: Option<u64>,
    /// Allowance for indexes, thread stacks, executable mappings, and allocator overhead.
    #[serde(default = "default_runtime_headroom_bytes")]
    pub runtime_headroom_bytes: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            memory_budget_bytes: None,
            runtime_headroom_bytes: DEFAULT_RUNTIME_HEADROOM_BYTES,
        }
    }
}

impl RuntimeConfig {
    /// Parse a complete YAML document and reject duplicate IDs or inconsistent caches.
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        let config: Self = serde_yaml::from_str(yaml).context("parsing beholdd runtime YAML")?;
        config.validate()?;
        Ok(config)
    }

    /// Load YAML from `path`; relative checkpoint paths resolve beside the config file.
    pub fn from_path(path: &Path) -> Result<Self> {
        let yaml = fs::read_to_string(path)
            .with_context(|| format!("reading runtime config {}", path.display()))?;
        let mut config = Self::from_yaml_str(&yaml)
            .with_context(|| format!("loading runtime config {}", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        for model in &mut config.models {
            if model.path.is_relative() {
                model.path = base.join(&model.path);
            }
        }
        config.validate()?;
        Ok(config)
    }

    /// Load `$BEHOLDD_HOME/config.yaml`, or `$HOME/.beholdd/config.yaml` when unset.
    pub fn from_default_path() -> Result<Self> {
        Self::from_path(&default_config_path()?)
    }

    /// Validate values supplied programmatically as strictly as parsed YAML.
    pub fn validate(&self) -> Result<()> {
        let mut ids = HashSet::new();
        ids.try_reserve(self.models.len())
            .context("reserving model-ID validation set")?;
        for (index, model) in self.models.iter().enumerate() {
            if model.id.is_empty() || model.id.trim() != model.id {
                bail!("models[{index}].id must be non-empty and have no surrounding whitespace");
            }
            if !ids.insert(model.id.as_str()) {
                bail!("duplicate model id {:?}", model.id);
            }
            if model.path.as_os_str().is_empty() {
                bail!("model {:?} has an empty checkpoint path", model.id);
            }
            model.validate_execution_backend().with_context(|| {
                format!("validating execution backend for model {:?}", model.id)
            })?;
            if model.hag_qwen35.is_some()
                && !model
                    .memory_budget_bytes
                    .or(self.memory_budget_bytes)
                    .is_some_and(|v| v > 0)
            {
                bail!("OpenMythos requires an explicit positive complete process memory budget");
            }
            let cache = model
                .cache_config()
                .with_context(|| format!("validating weight cache for model {:?}", model.id))?;
            if matches!(model.weight_mode, WeightMode::Resident) && cache.total_bytes() != 0 {
                bail!(
                    "model {:?} uses resident weights, so all streaming cache limits must be zero",
                    model.id
                );
            }
        }
        Ok(())
    }
}

/// Explicit model execution selection; existing configurations continue to use CPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionBackend {
    /// Production CPU execution.
    #[default]
    Cpu,
    /// Experimental native HAG execution, available only with the `hag-native` feature.
    Hag,
}

/// Explicit device and allocation budget for experimental native HAG execution.
///
/// Existing configurations retain 2048 positions. Larger on-demand contexts are
/// explicit and remain subject to checkpoint, device and combined-memory admission.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HagConfig {
    /// Absolute `/dev/dri/renderD<number>` path; no automatic device selection.
    pub device: PathBuf,
    /// Positive hard budget for native HAG allocations, separate from the process budget.
    pub memory_budget_bytes: u64,
    /// Evaluated prompt/decode positions, 1..=16384; defaults to 2048.
    /// Larger contexts require a correspondingly larger allocation/process budget.
    #[serde(default = "default_hag_context_capacity")]
    pub context_capacity: usize,
}

fn default_hag_context_capacity() -> usize {
    2048
}

/// Explicit pinned OpenMythos profile. This experimental path requires a separate
/// `hag-openmythos` build feature; ordinary `hag-native` builds retain Qwen3 only.
/// Registration contains policy only and never opens a checkpoint or GPU.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HagQwen35Config {
    /// Largest requested row tile, 1..=4096; actual plans can reduce this limit.
    #[serde(default = "default_qwen35_tile_rows")]
    pub tile_rows: usize,
    /// Largest packed positional read, 1..=64 MiB.
    #[serde(default = "default_qwen35_read_bytes")]
    pub encoded_read_bytes: usize,
    /// Decoded tile/result bound, 1..=64 MiB; native payloads stay packed.
    #[serde(default = "default_qwen35_read_bytes")]
    pub decoded_read_bytes: usize,
    /// Independent reset deadline after cancellation, 1..=120000 milliseconds.
    #[serde(default = "default_qwen35_cleanup_timeout_ms")]
    pub cleanup_timeout_ms: u64,
}
fn default_qwen35_tile_rows() -> usize {
    4096
}
fn default_qwen35_read_bytes() -> usize {
    64 * 1024 * 1024
}
fn default_qwen35_cleanup_timeout_ms() -> u64 {
    60_000
}
impl Default for HagQwen35Config {
    fn default() -> Self {
        Self {
            tile_rows: default_qwen35_tile_rows(),
            encoded_read_bytes: default_qwen35_read_bytes(),
            decoded_read_bytes: default_qwen35_read_bytes(),
            cleanup_timeout_ms: default_qwen35_cleanup_timeout_ms(),
        }
    }
}

#[cfg(test)]
mod qwen35_profile_tests {
    use super::*;

    #[test]
    fn profile_defaults_are_explicit_bounded_policy_and_unknown_keys_refuse() {
        let profile: HagQwen35Config = serde_yaml::from_str("{}").unwrap();
        assert_eq!(profile, HagQwen35Config::default());
        assert_eq!(
            (
                profile.tile_rows,
                profile.encoded_read_bytes,
                profile.decoded_read_bytes
            ),
            (4096, 64 * 1024 * 1024, 64 * 1024 * 1024)
        );
        assert_eq!(profile.cleanup_timeout_ms, 60_000);
        profile.validate().unwrap();
        assert!(serde_yaml::from_str::<HagQwen35Config>("allow_cpu_fallback: true").is_err());
        assert!(serde_yaml::from_str::<HagQwen35Config>("checkpoint_sha256: user-claim").is_err());
        assert!(ModelConfig::new("plain", "/missing.gguf")
            .hag_qwen35
            .is_none());
    }

    #[test]
    #[cfg(not(all(target_os = "linux", feature = "hag-openmythos")))]
    fn ordinary_build_cannot_activate_the_staged_profile() {
        let mut model = ModelConfig::new("openmythos", "/missing.gguf");
        model.execution_backend = ExecutionBackend::Hag;
        model.hag = Some(HagConfig {
            device: "/dev/dri/renderD128".into(),
            memory_budget_bytes: 1024,
            context_capacity: 1,
        });
        model.hag_qwen35 = Some(HagQwen35Config::default());
        model.memory_budget_bytes = Some(1024);
        let error = RuntimeConfig {
            models: vec![model],
            ..RuntimeConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(format!("{error:#}").contains("hag-openmythos build feature"));
    }
}
impl HagQwen35Config {
    fn validate(&self) -> Result<()> {
        if !(1..=4096).contains(&self.tile_rows)
            || !(1..=64 * 1024 * 1024).contains(&self.encoded_read_bytes)
            || !(1..=64 * 1024 * 1024).contains(&self.decoded_read_bytes)
            || !(1..=120_000).contains(&self.cleanup_timeout_ms)
        {
            bail!("OpenMythos read, tile or cleanup limits are outside explicit bounds");
        }
        Ok(())
    }
}

/// Native execution policy for one model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Stable API-facing identifier.
    pub id: String,
    /// First GGUF shard, or the only GGUF file.
    pub path: PathBuf,
    /// Explicit execution implementation; omitted entries continue to use CPU.
    #[serde(default)]
    pub execution_backend: ExecutionBackend,
    /// Required explicit HAG settings when `execution_backend` is `hag`.
    #[serde(default)]
    pub hag: Option<HagConfig>,
    /// Opt into the separately built, strictly pinned OpenMythos Q4_K owner.
    /// Omission preserves the existing dense Qwen3 HAG path.
    #[serde(default)]
    pub hag_qwen35: Option<HagQwen35Config>,
    /// Stream weights through bounded caches or preload every tensor payload.
    #[serde(default, deserialize_with = "deserialize_weight_mode")]
    pub weight_mode: WeightMode,
    /// Exact/degrade/refuse fit contract applied before payload reads.
    #[serde(default, deserialize_with = "deserialize_fit_policy")]
    pub fit_policy: FitPolicy,
    /// Model-specific hard budget, overriding the runtime-level value.
    #[serde(default)]
    pub memory_budget_bytes: Option<u64>,
    /// Model-specific runtime allowance, overriding the runtime-level value.
    #[serde(default)]
    pub runtime_headroom_bytes: Option<u64>,
    /// Explicitly allow network, rotational, mixed, or unidentifiable storage.
    #[serde(default)]
    pub allow_unsafe_storage: bool,
    /// Bytes retained for the repeated cyclic weight prefix.
    #[serde(default)]
    pub pinned_cache_bytes: u64,
    /// Bytes retained by the demand-driven LRU.
    #[serde(default)]
    pub demand_cache_bytes: u64,
    /// Bytes reserved for speculative reads.
    #[serde(default)]
    pub prefetch_bytes: u64,
    /// Maximum outstanding speculative requests.
    #[serde(default)]
    pub prefetch_queue_depth: usize,
    /// Number of concurrent speculative-read workers.
    #[serde(default)]
    pub prefetch_workers: usize,
    /// Enable the measured, optional near-miss expert prefetch experiment.
    #[serde(default)]
    pub near_miss_prefetch: bool,
}

impl ModelConfig {
    /// Construct a model entry with exact streaming and conservative storage defaults.
    pub fn new(id: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            execution_backend: ExecutionBackend::default(),
            hag: None,
            hag_qwen35: None,
            weight_mode: WeightMode::default(),
            fit_policy: FitPolicy::default(),
            memory_budget_bytes: None,
            runtime_headroom_bytes: None,
            allow_unsafe_storage: false,
            pinned_cache_bytes: 0,
            demand_cache_bytes: 0,
            prefetch_bytes: 0,
            prefetch_queue_depth: 0,
            prefetch_workers: 0,
            near_miss_prefetch: false,
        }
    }

    pub(crate) fn cache_config(&self) -> Result<CacheConfig> {
        CacheConfig::new(self.pinned_cache_bytes, self.demand_cache_bytes)?.with_prefetch(
            self.prefetch_bytes,
            self.prefetch_queue_depth,
            self.prefetch_workers,
        )
    }

    fn validate_execution_backend(&self) -> Result<()> {
        if let Some(profile) = &self.hag_qwen35 {
            profile.validate()?;
            if self.execution_backend != ExecutionBackend::Hag {
                bail!("OpenMythos requires execution_backend: hag and explicit hag settings");
            }
            if self
                .hag
                .as_ref()
                .is_some_and(|hag| hag.context_capacity > 2048)
            {
                bail!("OpenMythos runtime capacity is bounded to 2048 until further qualification");
            }
            if !cfg!(all(target_os = "linux", feature = "hag-openmythos")) {
                bail!("OpenMythos requires the separately qualified Linux hag-openmythos build feature");
            }
        }
        match self.execution_backend {
            ExecutionBackend::Cpu => {
                if self.hag.is_some() {
                    bail!("CPU execution does not accept hag settings");
                }
            }
            ExecutionBackend::Hag => {
                let hag = self
                    .hag
                    .as_ref()
                    .context("HAG execution requires hag settings")?;
                let valid_device = hag
                    .device
                    .to_str()
                    .and_then(|path| path.strip_prefix("/dev/dri/renderD"))
                    .is_some_and(|number| {
                        !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
                    });
                if !valid_device {
                    bail!("HAG device must be an explicit absolute /dev/dri/renderD<number> path");
                }
                if hag.memory_budget_bytes == 0 {
                    bail!("HAG memory_budget_bytes must be positive");
                }
                if !(1..=beholdd_model::qwen3moe::MAX_ON_DEMAND_CAPACITY)
                    .contains(&hag.context_capacity)
                {
                    bail!("HAG context_capacity must be between 1 and 16384");
                }
                if self.fit_policy != FitPolicy::Exact {
                    bail!("HAG execution requires fit_policy: exact");
                }
                if self.weight_mode != WeightMode::Streaming {
                    bail!("HAG execution requires weight_mode: streaming for bounded preparation staging");
                }
                if self.pinned_cache_bytes != 0
                    || self.demand_cache_bytes != 0
                    || self.prefetch_bytes != 0
                    || self.prefetch_queue_depth != 0
                    || self.prefetch_workers != 0
                    || self.near_miss_prefetch
                {
                    bail!("HAG execution requires all weight caches, prefetch settings, and near_miss_prefetch to be disabled");
                }
                if !cfg!(all(target_os = "linux", feature = "hag-native")) {
                    bail!("HAG execution requires Linux and a build with the hag-native feature");
                }
            }
        }
        Ok(())
    }
}

fn default_runtime_headroom_bytes() -> u64 {
    DEFAULT_RUNTIME_HEADROOM_BYTES
}

fn default_config_path() -> Result<PathBuf> {
    if let Some(home) = env::var_os("BEHOLDD_HOME") {
        if home.is_empty() {
            bail!("BEHOLDD_HOME is set but empty");
        }
        return Ok(PathBuf::from(home).join("config.yaml"));
    }
    let home = env::var_os("HOME").context("HOME is unset; set BEHOLDD_HOME explicitly")?;
    if home.is_empty() {
        bail!("HOME is empty; set BEHOLDD_HOME explicitly");
    }
    Ok(PathBuf::from(home).join(".beholdd/config.yaml"))
}

fn deserialize_weight_mode<'de, D>(deserializer: D) -> Result<WeightMode, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    value.parse().map_err(serde::de::Error::custom)
}

fn deserialize_fit_policy<'de, D>(deserializer: D) -> Result<FitPolicy, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    value.parse().map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_yaml_applies_safe_defaults() {
        let config =
            RuntimeConfig::from_yaml_str("models:\n  - id: coder\n    path: weights/model.gguf\n")
                .unwrap();
        assert_eq!(
            config.runtime_headroom_bytes,
            DEFAULT_RUNTIME_HEADROOM_BYTES
        );
        assert_eq!(
            config.models[0],
            ModelConfig::new("coder", "weights/model.gguf")
        );
    }

    #[test]
    fn duplicate_and_unknown_fields_are_rejected() {
        assert!(RuntimeConfig::from_yaml_str(
            "models: []\nruntime_headroom_bytes: 1\nruntime_headroom_bytes: 2\n"
        )
        .is_err());
        assert!(RuntimeConfig::from_yaml_str("models: []\nserver: {}\n").is_err());
        assert!(RuntimeConfig::from_yaml_str(
            "models:\n  - id: coder\n    path: a.gguf\n    backend: auto\n"
        )
        .is_err());
    }

    #[test]
    fn duplicate_ids_and_partial_prefetch_are_rejected() {
        assert!(RuntimeConfig::from_yaml_str(
            "models:\n  - { id: coder, path: a.gguf }\n  - { id: coder, path: b.gguf }\n"
        )
        .is_err());
        assert!(RuntimeConfig::from_yaml_str(
            "models:\n  - id: coder\n    path: a.gguf\n    prefetch_bytes: 1\n"
        )
        .is_err());
    }

    #[test]
    fn enums_are_case_sensitive_and_strict() {
        assert!(RuntimeConfig::from_yaml_str(
            "models:\n  - { id: coder, path: a.gguf, weight_mode: Resident }\n"
        )
        .is_err());
        assert!(RuntimeConfig::from_yaml_str(
            "models:\n  - { id: coder, path: a.gguf, fit_policy: approximate }\n"
        )
        .is_err());
    }

    fn hag_model() -> ModelConfig {
        let mut model = ModelConfig::new("native", "model.gguf");
        model.execution_backend = ExecutionBackend::Hag;
        model.hag = Some(HagConfig {
            context_capacity: 2048,
            device: PathBuf::from("/dev/dri/renderD128"),
            memory_budget_bytes: 768 * 1024 * 1024,
        });
        model
    }

    #[test]
    fn hag_context_capacity_is_explicit_bounded_and_defaults_to_2048() {
        let settings = "{device: /dev/dri/renderD128, memory_budget_bytes: 1}";
        assert_eq!(
            serde_yaml::from_str::<HagConfig>(settings)
                .unwrap()
                .context_capacity,
            2048
        );
        for capacity in [1, 2048, 8192, 16384] {
            let mut model = hag_model();
            model.hag.as_mut().unwrap().context_capacity = capacity;
            if cfg!(all(target_os = "linux", feature = "hag-native")) {
                model.validate_execution_backend().unwrap();
            }
            let yaml = format!("{{device: /dev/dri/renderD128, memory_budget_bytes: 1, context_capacity: {capacity}}}");
            assert_eq!(
                serde_yaml::from_str::<HagConfig>(&yaml)
                    .unwrap()
                    .context_capacity,
                capacity
            );
        }
        for capacity in [0, 16385, usize::MAX] {
            let mut model = hag_model();
            model.hag.as_mut().unwrap().context_capacity = capacity;
            rejected(model, "context_capacity");
        }
        for value in ["-1", "1.5", "null", "'8192'"] {
            let yaml = format!("{{device: /dev/dri/renderD128, memory_budget_bytes: 1, context_capacity: {value}}}");
            assert!(serde_yaml::from_str::<HagConfig>(&yaml).is_err());
        }
    }

    #[test]
    fn larger_hag_example_has_explicit_context_and_finite_budgets() {
        let config: RuntimeConfig =
            serde_yaml::from_str(include_str!("../../../config.hag-16k.example.yaml")).unwrap();
        assert_eq!(config.memory_budget_bytes, Some(6 * 1024 * 1024 * 1024));
        assert_eq!(config.runtime_headroom_bytes, 128 * 1024 * 1024);
        assert_eq!(config.models.len(), 1);
        let model = &config.models[0];
        assert_eq!(model.execution_backend, ExecutionBackend::Hag);
        let hag = model.hag.as_ref().unwrap();
        assert_eq!(hag.context_capacity, 16_384);
        assert_eq!(hag.memory_budget_bytes, 4608 * 1024 * 1024);
        if cfg!(all(target_os = "linux", feature = "hag-native")) {
            config.validate().unwrap();
        } else {
            assert!(format!("{:#}", config.validate().unwrap_err()).contains("hag-native feature"));
        }
    }

    fn rejected(model: ModelConfig, expected: &str) {
        let config = RuntimeConfig {
            models: vec![model],
            ..RuntimeConfig::default()
        };
        let error = format!("{:#}", config.validate().unwrap_err());
        assert!(error.contains(expected), "{error}");
    }

    #[test]
    fn hag_is_an_explicit_feature_gated_choice_and_cpu_defaults_are_unchanged() {
        let cpu = RuntimeConfig::from_yaml_str(
            "models:\n  - { id: cpu, path: model.gguf, execution_backend: cpu }\n",
        )
        .unwrap();
        assert_eq!(cpu.models[0], ModelConfig::new("cpu", "model.gguf"));
        let yaml = "models:\n  - id: native\n    path: model.gguf\n    execution_backend: hag\n    hag:\n      device: /dev/dri/renderD128\n      memory_budget_bytes: 805306368\n";
        let parsed: RuntimeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(parsed.models, vec![hag_model()]);
        if cfg!(all(target_os = "linux", feature = "hag-native")) {
            parsed.validate().unwrap();
            assert_eq!(RuntimeConfig::from_yaml_str(yaml).unwrap(), parsed);
        } else {
            rejected(hag_model(), "hag-native feature");
            assert!(RuntimeConfig::from_yaml_str(yaml).is_err());
        }
    }

    #[test]
    fn hag_schema_rejects_unknown_fields_and_implicit_values() {
        for backend in ["Hag", "auto", "vulkan", "CPU"] {
            let yaml = format!(
                "models:\n  - {{ id: native, path: model.gguf, execution_backend: {backend} }}\n"
            );
            assert!(serde_yaml::from_str::<RuntimeConfig>(&yaml).is_err());
        }
        for settings in [
            "{}",
            "{ device: /dev/dri/renderD128 }",
            "{ memory_budget_bytes: 1 }",
            "{ device: /dev/dri/renderD128, memory_budget_bytes: -1 }",
            "{ device: /dev/dri/renderD128, memory_budget_bytes: 1, capacity: 8 }",
            "{ device: /dev/dri/renderD128, memory_budget_bytes: 1, memory_budget_bytes: 2 }",
        ] {
            let yaml = format!(
                "models:\n  - id: native\n    path: model.gguf\n    execution_backend: hag\n    hag: {settings}\n"
            );
            assert!(
                serde_yaml::from_str::<RuntimeConfig>(&yaml).is_err(),
                "{yaml}"
            );
        }
        let mut missing = hag_model();
        missing.hag = None;
        rejected(missing, "requires hag settings");
        let mut cpu = hag_model();
        cpu.execution_backend = ExecutionBackend::Cpu;
        rejected(cpu, "CPU execution does not accept hag settings");
    }

    #[test]
    fn hag_requires_explicit_render_node_and_positive_allocation_budget() {
        for path in [
            "",
            "renderD128",
            "dev/dri/renderD128",
            "/dev/dri/card0",
            "/dev/dri/renderD",
            "/dev/dri/renderD-1",
            "/dev/dri/renderD128/child",
            "/dev/dri/renderD128/",
            "/dev/dri/../dri/renderD128",
            "/other/renderD128",
        ] {
            let mut model = hag_model();
            model.hag.as_mut().unwrap().device = path.into();
            rejected(model, "explicit absolute /dev/dri/renderD<number> path");
        }
        let mut zero = hag_model();
        zero.hag.as_mut().unwrap().memory_budget_bytes = 0;
        rejected(zero, "memory_budget_bytes must be positive");
    }

    #[test]
    fn hag_refuses_resident_cpu_weights_degradation_and_all_prefetch_settings() {
        for fit_policy in [FitPolicy::Degrade, FitPolicy::Refuse] {
            let mut model = hag_model();
            model.fit_policy = fit_policy;
            rejected(model, "fit_policy: exact");
        }
        let mut resident = hag_model();
        resident.weight_mode = WeightMode::Resident;
        rejected(resident, "weight_mode: streaming");
        for setting in 0..6 {
            let mut model = hag_model();
            match setting {
                0 => model.pinned_cache_bytes = 1,
                1 => model.demand_cache_bytes = 1,
                2 => model.prefetch_bytes = 1,
                3 => model.prefetch_queue_depth = 1,
                4 => model.prefetch_workers = 1,
                5 => model.near_miss_prefetch = true,
                _ => unreachable!(),
            }
            rejected(model, "requires all weight caches, prefetch settings");
        }
    }
}
