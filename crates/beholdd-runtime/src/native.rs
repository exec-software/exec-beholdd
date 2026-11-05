use std::collections::BTreeMap;
use std::str;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, TryLockError};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use beholdd_backend::cpu::Cpu;
use beholdd_format::Gguf;
use beholdd_memory::{
    available_memory, check_checkpoint_storage, FitEstimate, FullyResidentSource, StorageOverride,
    StreamingSource, WeightSource,
};
use beholdd_model::execution::{plan_execution, ExecutionPlan};
use beholdd_model::{GenerationFinishReason, Qwen2Tokenizer, QwenModel};

use crate::{
    ChatTemplate, ExecutionBackend, ExecutionLimits, FinishReason, GenerateRequest, Generation,
    InferenceRuntime, ModelCapabilities, ModelCompatibility, ModelConfig, ModelInfo, RequestError,
    RuntimeConfig, RuntimeDescriptor, RuntimeKind, RuntimeOperations, UnloadError,
};

#[cfg(all(target_os = "linux", feature = "hag-native"))]
#[path = "native_hag.rs"]
mod native_hag;

#[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
#[path = "native_qwen35.rs"]
mod native_qwen35;

#[path = "native_phase.rs"]
#[cfg_attr(
    not(all(target_os = "linux", feature = "hag-native")),
    allow(dead_code)
)]
mod native_phase;

/// Lazy native runtime with explicit CPU or opt-in resident HAG execution.
pub struct NativeRuntime {
    config: RuntimeConfig,
    registry: BTreeMap<String, RegistryEntry>,
    execution: Mutex<()>,
}

struct RegistryEntry {
    config: ModelConfig,
    loaded: Mutex<Option<Arc<LoadedModel>>>,
}

struct LoadedModel {
    checkpoint: Arc<Gguf>,
    tokenizer: Qwen2Tokenizer,
    model: BoundModel,
    device: Cpu,
    probe: RwLock<LoadedProbe>,
    execution: Mutex<LoadedExecution>,
}

enum BoundModel {
    Qwen3(QwenModel),
    #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
    Qwen35,
}
impl BoundModel {
    fn qwen3(&self) -> Result<&QwenModel> {
        match self {
            Self::Qwen3(model) => Ok(model),
            #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
            Self::Qwen35 => bail!("OpenMythos cannot enter the CPU or dense Qwen3 path"),
        }
    }
    fn probe(&self, probe: &LoadedProbe) -> bool {
        match self {
            Self::Qwen3(model) => model
                .weight_memory()
                .map(|memory| probe.probe(memory.checkpoint_bytes()))
                .unwrap_or(false),
            #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
            Self::Qwen35 => matches!(probe, LoadedProbe::Hag(healthy)
                if healthy.load(std::sync::atomic::Ordering::Acquire)),
        }
    }
}

enum LoadedExecution {
    Cpu {
        weights: LoadedWeights,
        memory_budget_bytes: u64,
    },
    #[cfg(all(target_os = "linux", feature = "hag-native"))]
    Hag(Box<native_hag::HagExecution>),
    #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
    Qwen35(Box<native_qwen35::HagQwen35Execution>),
}

enum LoadedProbe {
    Cpu(LoadedWeights),
    #[cfg(all(target_os = "linux", feature = "hag-native"))]
    Hag(Arc<std::sync::atomic::AtomicBool>),
}

impl LoadedProbe {
    fn probe(&self, expected_checkpoint_bytes: u64) -> bool {
        match self {
            Self::Cpu(weights) => weights.probe(expected_checkpoint_bytes),
            #[cfg(all(target_os = "linux", feature = "hag-native"))]
            Self::Hag(healthy) => healthy.load(std::sync::atomic::Ordering::Acquire),
        }
    }
}

#[derive(Clone)]
enum LoadedWeights {
    Streaming(Arc<StreamingSource>),
    Resident(Arc<FullyResidentSource>),
}

impl LoadedWeights {
    fn source(&self) -> &dyn WeightSource {
        match self {
            Self::Streaming(source) => source.as_ref(),
            Self::Resident(source) => source.as_ref(),
        }
    }

    fn probe(&self, expected_checkpoint_bytes: u64) -> bool {
        match self {
            Self::Streaming(source) => source.probe().is_ok(),
            Self::Resident(source) => source.resident_bytes() == expected_checkpoint_bytes,
        }
    }
}

impl NativeRuntime {
    /// Validate `config` and build a metadata-free, initially unloaded registry.
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        config.validate()?;
        let mut registry = BTreeMap::new();
        for model in &config.models {
            registry.insert(
                model.id.clone(),
                RegistryEntry {
                    config: model.clone(),
                    loaded: Mutex::new(None),
                },
            );
        }
        Ok(Self {
            config,
            registry,
            execution: Mutex::new(()),
        })
    }

    fn entry(&self, id: &str) -> Result<&RegistryEntry> {
        self.registry.get(id).ok_or_else(|| {
            anyhow!(
                "unknown model {id:?}; configured models: {}",
                self.registry
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }

    fn evict_other_models(&self, requested: &str) -> Result<()> {
        let evicted = take_other_loaded(
            self.registry
                .iter()
                .map(|(id, entry)| (id.as_str(), &entry.loaded)),
            requested,
        )?;
        // Streaming-source destruction can stop and join prefetch workers. Do that only
        // after every registry lock has been released, while runtime-wide admission still
        // prevents another generation from observing a half-evicted registry.
        drop(evicted);
        Ok(())
    }

    fn load_model(
        &self,
        config: &ModelConfig,
        request: &GenerateRequest,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        phases: &native_phase::Recorder,
    ) -> Result<LoadedModel> {
        phases.emit(native_phase::Phase::MetadataStart);
        let checkpoint = checkpoint_metadata(config, control)?;
        let tokenizer = Qwen2Tokenizer::from_gguf(&checkpoint)
            .with_context(|| format!("binding tokenizer for model {:?}", config.id))?;
        // This branch is reachable only from an actual generate call. Metadata,
        // model listing, selection and service construction never prepare an owner.
        #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
        if config.hag_qwen35.is_some() {
            let prompt = tokenizer.encode(&request.prompt, true)?;
            let capacity = config
                .hag
                .as_ref()
                .context("missing HAG settings")?
                .context_capacity;
            native_hag::validate_request(capacity, prompt.len(), request.max_new_tokens)?;
            phases.emit(native_phase::Phase::MetadataEnd);
            let process_budget = config
                .memory_budget_bytes
                .or(self.config.memory_budget_bytes)
                .context("OpenMythos requires an explicit complete process memory budget")?;
            let execution = native_qwen35::HagQwen35Execution::prepare(
                &checkpoint,
                &tokenizer,
                config,
                process_budget,
                config
                    .runtime_headroom_bytes
                    .unwrap_or(self.config.runtime_headroom_bytes),
                control,
                phases,
            )?;
            return Ok(LoadedModel {
                checkpoint,
                tokenizer,
                model: BoundModel::Qwen35,
                device: Cpu::new(),
                probe: RwLock::new(LoadedProbe::Hag(execution.health())),
                execution: Mutex::new(LoadedExecution::Qwen35(Box::new(execution))),
            });
        }
        let model = QwenModel::load(&checkpoint)
            .with_context(|| format!("binding native model {:?}", config.id))?;
        let device = Cpu::new();
        if config.execution_backend == ExecutionBackend::Cpu {
            model
                .validate_device(&device)
                .with_context(|| format!("validating CPU support for model {:?}", config.id))?;
        }
        let prompt = tokenizer
            .encode(&request.prompt, true)
            .with_context(|| format!("encoding prompt for model {:?}", config.id))?;
        if prompt.is_empty() {
            return Err(RequestError::EmptyPrompt.into());
        }
        phases.emit(native_phase::Phase::MetadataEnd);

        let availability = available_memory().context("discovering memory available to beholdd")?;
        let available_bytes = availability.effective_available_bytes();
        let memory_budget_bytes = config
            .memory_budget_bytes
            .or(self.config.memory_budget_bytes)
            .unwrap_or(available_bytes / 2);
        if config.execution_backend == ExecutionBackend::Cpu
            && memory_budget_bytes > available_bytes
        {
            bail!(
                "model {:?} requests a {memory_budget_bytes}-byte memory budget, but Linux currently reports only {available_bytes} bytes available",
                config.id
            );
        }
        if config.execution_backend == ExecutionBackend::Hag {
            #[cfg(all(target_os = "linux", feature = "hag-native"))]
            {
                native_hag::validate_request(
                    config
                        .hag
                        .as_ref()
                        .context("missing HAG settings")?
                        .context_capacity,
                    prompt.len(),
                    request.max_new_tokens,
                )?;
                let execution = native_hag::HagExecution::prepare(
                    &model,
                    &checkpoint,
                    config,
                    memory_budget_bytes,
                    config
                        .runtime_headroom_bytes
                        .unwrap_or(self.config.runtime_headroom_bytes),
                    control,
                    phases,
                )?;
                let probe = RwLock::new(LoadedProbe::Hag(execution.health()));
                return Ok(LoadedModel {
                    checkpoint,
                    tokenizer,
                    model: BoundModel::Qwen3(model),
                    device,
                    probe,
                    execution: Mutex::new(LoadedExecution::Hag(Box::new(execution))),
                });
            }
            #[cfg(not(all(target_os = "linux", feature = "hag-native")))]
            bail!("HAG serving requires Linux and the hag-native feature");
        }
        control().context("checking generation control before CPU weight preparation")?;
        let execution_plan = plan_for_request(
            &model,
            config,
            self.config.runtime_headroom_bytes,
            prompt.len(),
            request.max_new_tokens,
            memory_budget_bytes,
        )?;

        // Tensor payloads may be read only after the executable fit plan above succeeds.
        let weights = match execution_plan {
            ExecutionPlan::Streaming { cache, .. } => {
                LoadedWeights::Streaming(Arc::new(StreamingSource::new(checkpoint.clone(), cache)?))
            }
            ExecutionPlan::Resident(_) => {
                let weights =
                    FullyResidentSource::load(checkpoint.as_ref()).with_context(|| {
                        format!("preloading resident weights for model {:?}", config.id)
                    })?;
                let expected = model.weight_memory()?.checkpoint_bytes();
                if weights.resident_bytes() != expected {
                    bail!(
                        "resident source materialized {} bytes, but model {:?} bound {expected} checkpoint bytes",
                        weights.resident_bytes(),
                        config.id
                    );
                }
                LoadedWeights::Resident(Arc::new(weights))
            }
        };
        let probe = RwLock::new(LoadedProbe::Cpu(weights.clone()));
        Ok(LoadedModel {
            checkpoint,
            tokenizer,
            model: BoundModel::Qwen3(model),
            device,
            probe,
            execution: Mutex::new(LoadedExecution::Cpu {
                weights,
                memory_budget_bytes,
            }),
        })
    }
}

impl InferenceRuntime for NativeRuntime {
    fn descriptor(&self) -> RuntimeDescriptor {
        let mut execution_backends = vec!["cpu"];
        if self
            .config
            .models
            .iter()
            .any(|model| model.execution_backend == ExecutionBackend::Hag)
        {
            // RuntimeConfig rejects HAG unless the executable contains its implementation.
            execution_backends.push("hag");
        }
        let mut architectures = vec!["qwen3", "qwen3moe"];
        if self
            .config
            .models
            .iter()
            .any(|model| model.hag_qwen35.is_some())
        {
            architectures.push("qwen35");
        }
        RuntimeDescriptor {
            runtime: RuntimeKind::Native,
            execution_backends,
            architectures,
            model_formats: vec!["gguf"],
            operations: RuntimeOperations { unload: true },
        }
    }

    fn unload(&self, model: &str) -> std::result::Result<(), UnloadError> {
        let entry = self
            .registry
            .get(model)
            .ok_or_else(|| UnloadError::NotFound(model.to_owned()))?;
        unload_registered_model(&self.execution, &entry.loaded)?;
        #[cfg(all(target_os = "linux", feature = "hag-native"))]
        if entry.config.execution_backend == ExecutionBackend::Hag {
            native_hag::require_process_healthy()
                .map_err(|error| UnloadError::Failed(format!("{error:#}")))?;
        }
        Ok(())
    }

    fn models(&self) -> Result<Vec<ModelInfo>> {
        self.registry
            .iter()
            .map(|(id, entry)| {
                let observed = inspect_published_model(&entry.loaded, |model| {
                    observe_liveness(&model.execution, &model.probe, |probe| {
                        model.model.probe(probe)
                    })
                });
                let compatibility = if observed.is_some() {
                    ModelCompatibility::Compatible(ModelCapabilities::CompletionOnly)
                } else {
                    ModelCompatibility::Unverified
                };
                Ok(ModelInfo {
                    id: id.clone(),
                    loaded: observed.unwrap_or(false),
                    compatibility,
                })
            })
            .collect()
    }

    fn execution_limits(&self, model: &str) -> Result<ExecutionLimits> {
        let entry = self.entry(model)?;
        #[cfg(all(target_os = "linux", feature = "hag-native"))]
        if entry.config.execution_backend == ExecutionBackend::Hag {
            return Ok(ExecutionLimits {
                max_evaluated_tokens: Some(
                    entry
                        .config
                        .hag
                        .as_ref()
                        .context("missing HAG settings")?
                        .context_capacity,
                ),
            });
        }
        let _ = entry;
        Ok(ExecutionLimits::default())
    }

    fn chat_template_controlled(
        &self,
        model: &str,
        control: &mut (dyn FnMut() -> Result<()> + Send),
    ) -> Result<ChatTemplate> {
        let entry = self.entry(model)?;
        control().context("checking chat-template control before admission")?;
        let _admission = lock_runtime_execution(&self.execution, control)?;
        let loaded = entry
            .loaded
            .lock()
            .map_err(|_| anyhow!("model {model:?} residency lock is poisoned"))?
            .clone();
        control().context("checking chat-template control before metadata lookup")?;
        let template = if let Some(loaded) = loaded {
            loaded.tokenizer.chat_template().map(str::to_owned)
        } else {
            // Do not bind a tokenizer or model here: template selection only needs
            // the GGUF metadata and must not publish, evict or reset a resident owner.
            let checkpoint = checkpoint_metadata(&entry.config, control)?;
            checkpoint
                .metadata()
                .get("tokenizer.chat_template")
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| anyhow!("tokenizer.chat_template is not a string"))
                })
                .transpose()?
        };
        control().context("checking chat-template control after metadata lookup")?;
        Ok(ChatTemplate::Checkpoint(template))
    }

    fn plan_request(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
    ) -> Result<GenerateRequest> {
        self.plan_request_controlled(
            model,
            prompt,
            max_new_tokens,
            default_max_new_tokens,
            &mut || Ok(()),
        )
    }

    fn plan_request_controlled(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
        control: &mut (dyn FnMut() -> Result<()> + Send),
    ) -> Result<GenerateRequest> {
        let entry = self.entry(model)?;
        control().context("checking request planning control")?;
        if prompt.is_empty() {
            return Err(RequestError::EmptyPrompt.into());
        }
        let limits = self.execution_limits(model)?;
        // Refuse impossible explicit limits before checkpoint I/O. This provisional
        // lower bound never resolves omission: the actual token count below does that.
        if let Some(explicit) = max_new_tokens {
            limits.resolve(1, Some(explicit), default_max_new_tokens)?;
        } else if default_max_new_tokens == 0 {
            return Err(RequestError::InvalidOutputLimit.into());
        }
        let resolved = if limits.max_evaluated_tokens.is_some() {
            // Serialize metadata work with native execution, including embedded callers.
            // Planning never evicts, publishes a model, resets KV or opens a GPU.
            let _admission = lock_runtime_execution(&self.execution, control)?;
            let loaded = entry
                .loaded
                .lock()
                .map_err(|_| anyhow!("model {model:?} residency lock is poisoned"))?
                .clone();
            let token_count = if let Some(loaded) = loaded {
                control().context("checking request planning before resident tokenization")?;
                loaded.tokenizer.encode(prompt, true)?.len()
            } else {
                let checkpoint = checkpoint_metadata(&entry.config, control)?;
                control().context("checking request planning before tokenizer binding")?;
                let tokenizer = Qwen2Tokenizer::from_gguf(&checkpoint)
                    .with_context(|| format!("binding tokenizer for model {model:?}"))?;
                control().context("checking request planning before cold tokenization")?;
                tokenizer.encode(prompt, true)?.len()
            };
            control().context("checking request planning after tokenization")?;
            limits.resolve(token_count, max_new_tokens, default_max_new_tokens)?
        } else {
            // Preserve CPU and alternate-runtime defaults without introducing eager I/O.
            max_new_tokens.unwrap_or(default_max_new_tokens)
        };
        control().context("checking request planning before publication")?;
        Ok(GenerateRequest {
            model: model.to_owned(),
            prompt: prompt.to_owned(),
            max_new_tokens: resolved,
        })
    }

    fn generate(
        &self,
        request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Result<Generation> {
        let mut continue_generation = || Ok(());
        self.generate_controlled(request, &mut continue_generation, on_text)
    }

    fn generate_controlled(
        &self,
        request: GenerateRequest,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Result<Generation> {
        let entry = self.entry(&request.model)?;
        if request.max_new_tokens == 0 {
            return Err(RequestError::InvalidOutputLimit.into());
        }
        let phases =
            native_phase::Recorder::new(entry.config.execution_backend == ExecutionBackend::Hag);
        #[cfg(all(target_os = "linux", feature = "hag-native"))]
        if entry.config.execution_backend == ExecutionBackend::Hag {
            // Any nonempty prompt needs at least one position. Reject impossible
            // output limits before checkpoint metadata or hardware can be touched.
            native_hag::validate_request(
                entry
                    .config
                    .hag
                    .as_ref()
                    .context("missing HAG settings")?
                    .context_capacity,
                1,
                request.max_new_tokens,
            )?;
        }
        // Admission is runtime-wide, not per model. Two independently configured resident
        // models must not both observe the same available-memory snapshot and begin loading
        // against it, and embedded callers must share the same serialization contract as
        // HTTP callers. The per-model execution lock below still owns mutable weight/cache
        // state and remains the liveness probe's indication that that model is active.
        phases.emit(native_phase::Phase::AdmissionStart);
        let _runtime_execution = lock_runtime_execution(&self.execution, control)?;
        phases.emit(native_phase::Phase::AdmissionEnd);
        // M3 intentionally keeps at most one bound model. Per-model forecasts cannot be
        // added safely after the fact, and retaining an inactive model's resident weights
        // or streaming caches would make a later request plan against stale headroom.
        self.evict_other_models(&request.model)?;
        let loaded = {
            let mut state = entry
                .loaded
                .lock()
                .map_err(|_| anyhow!("model {:?} residency lock is poisoned", request.model))?;
            if state.is_none() {
                control().context("checking generation control before loading the model")?;
                let model = self.load_model(&entry.config, &request, control, &phases)?;
                control().context("checking generation control after loading the model")?;
                *state = Some(Arc::new(model));
            }
            state.as_ref().expect("loaded immediately above").clone()
        };
        control().context("checking generation control before prompt encoding")?;
        let prompt = loaded
            .tokenizer
            .encode(&request.prompt, true)
            .with_context(|| format!("encoding prompt for model {:?}", request.model))?;
        if prompt.is_empty() {
            return Err(RequestError::EmptyPrompt.into());
        }

        let mut execution = loaded
            .execution
            .lock()
            .map_err(|_| anyhow!("model {:?} execution lock is poisoned", request.model))?;
        control().context("checking generation control before request fit planning")?;
        let eos = loaded.tokenizer.eos_token_id();
        let mut emitter = Utf8Emitter::new(&loaded.tokenizer, on_text);
        #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
        {
            emitter.openmythos = matches!(&loaded.model, BoundModel::Qwen35);
        }
        let output = match &mut *execution {
            LoadedExecution::Cpu {
                weights,
                memory_budget_bytes,
            } => {
                let model = loaded.model.qwen3()?;
                let execution_plan = plan_for_request(
                    model,
                    &entry.config,
                    self.config.runtime_headroom_bytes,
                    prompt.len(),
                    request.max_new_tokens,
                    *memory_budget_bytes,
                )?;
                if reconcile_streaming_cache(weights, &loaded.checkpoint, execution_plan)? {
                    let mut probe = loaded.probe.write().map_err(|_| {
                        anyhow!("model {:?} liveness probe lock is poisoned", request.model)
                    })?;
                    *probe = LoadedProbe::Cpu(weights.clone());
                }
                model.generate_greedy_controlled(
                    &loaded.device,
                    weights.source(),
                    &prompt,
                    request.max_new_tokens,
                    &[eos],
                    entry.config.near_miss_prefetch,
                    control,
                    |token| emitter.push(token),
                )?
            }
            #[cfg(all(target_os = "linux", feature = "hag-native"))]
            LoadedExecution::Hag(hag) => hag.generate(
                &prompt,
                request.max_new_tokens,
                eos,
                control,
                &phases,
                |token| emitter.push(token),
            )?,
            #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
            LoadedExecution::Qwen35(hag) => {
                hag.generate(&prompt, request.max_new_tokens, control, &phases, |token| {
                    emitter.push(token)
                })?
            }
        };
        let text = emitter.finish()?;
        let completion_tokens = output.generated.len();
        let finish_reason = match output.finish_reason {
            GenerationFinishReason::StopToken => FinishReason::Stop,
            GenerationFinishReason::MaxNewTokens => FinishReason::Length,
        };
        let generation = Generation {
            text,
            token_ids: output.generated,
            prompt_tokens: prompt.len(),
            completion_tokens,
            finish_reason,
        };
        phases.finish();
        Ok(generation)
    }
}

fn checkpoint_metadata(
    config: &ModelConfig,
    control: &mut (dyn FnMut() -> Result<()> + Send),
) -> Result<Arc<Gguf>> {
    control().context("checking generation control before checkpoint metadata")?;
    let storage_override = if config.allow_unsafe_storage {
        StorageOverride::AllowUnsafe
    } else {
        StorageOverride::RefuseUnsafe
    };
    check_checkpoint_storage([&config.path], storage_override).with_context(|| {
        format!("preflighting checkpoint path for model {:?}; set allow_unsafe_storage: true only to explicitly accept network, rotational, mixed, or unknown storage", config.id)
    })?;
    control().context("checking generation control before GGUF metadata")?;
    let checkpoint = Arc::new(
        Gguf::open(&config.path)
            .with_context(|| format!("opening GGUF metadata for model {:?}", config.id))?,
    );
    check_checkpoint_storage(checkpoint.shards(), storage_override).with_context(|| {
        format!("checking every checkpoint shard for model {:?}; set allow_unsafe_storage: true only to explicitly accept unsafe storage", config.id)
    })?;
    control().context("checking generation control after checkpoint metadata")?;
    Ok(checkpoint)
}

fn lock_runtime_execution<'a>(
    execution: &'a Mutex<()>,
    control: &mut (dyn FnMut() -> Result<()> + Send),
) -> Result<MutexGuard<'a, ()>> {
    loop {
        control().context("checking generation control while waiting for runtime admission")?;
        match execution.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::WouldBlock) => std::thread::park_timeout(Duration::from_millis(10)),
            Err(TryLockError::Poisoned(_)) => {
                bail!("native runtime execution lock is poisoned")
            }
        }
    }
}

fn unload_registered_model<T>(
    execution: &Mutex<()>,
    loaded: &Mutex<Option<T>>,
) -> std::result::Result<(), UnloadError> {
    let _execution = match execution.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::WouldBlock) => return Err(UnloadError::Busy),
        Err(TryLockError::Poisoned(_)) => {
            return Err(UnloadError::Failed(
                "runtime execution lock is poisoned".into(),
            ));
        }
    };
    let removed = {
        let mut state = loaded
            .lock()
            .map_err(|_| UnloadError::Failed("model residency lock is poisoned".into()))?;
        state.take()
    };
    // Destruction can stop and join prefetch workers. Release the registry lock before
    // doing that, but retain runtime admission until those resources have been dropped.
    drop(removed);
    Ok(())
}

fn take_other_loaded<'a, T: 'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a Mutex<Option<T>>)>,
    requested: &str,
) -> Result<Vec<T>> {
    let mut evicted = Vec::new();
    for (id, state) in entries {
        if id == requested {
            continue;
        }
        if let Some(value) = state
            .lock()
            .map_err(|_| anyhow!("model {id:?} residency lock is poisoned during eviction"))?
            .take()
        {
            evicted.push(value);
        }
    }
    Ok(evicted)
}

fn inspect_published_model<T, R>(
    state: &Mutex<Option<Arc<T>>>,
    inspect: impl FnOnce(&T) -> R,
) -> Option<R> {
    match state.try_lock() {
        // Do not clone the model Arc into readers: an in-flight model listing must not
        // retain old weights past unload's execution-admission boundary.
        Ok(state) => state.as_deref().map(inspect),
        // Loading has not published an executable model yet.
        Err(TryLockError::WouldBlock) | Err(TryLockError::Poisoned(_)) => None,
    }
}

fn observe_liveness<E, P>(
    execution: &Mutex<E>,
    probe: &RwLock<P>,
    inspect: impl FnOnce(&P) -> bool,
) -> bool {
    match execution.try_lock() {
        // A held execution lock means inference is active, not that the model is dead.
        Ok(execution) => drop(execution),
        Err(TryLockError::WouldBlock) => {}
        Err(TryLockError::Poisoned(_)) => return false,
    }
    match probe.try_read() {
        Ok(probe) => inspect(&probe),
        Err(TryLockError::WouldBlock) | Err(TryLockError::Poisoned(_)) => false,
    }
}

fn plan_for_request(
    model: &QwenModel,
    config: &ModelConfig,
    default_runtime_headroom_bytes: u64,
    prompt_tokens: usize,
    max_new_tokens: usize,
    memory_budget_bytes: u64,
) -> Result<ExecutionPlan> {
    let weight_memory = model.weight_memory()?;
    let generate_buffer_bytes =
        model.greedy_generate_buffer_bytes(prompt_tokens, max_new_tokens)?;
    let cpu_scratch_bytes = weight_memory
        .largest_matvec_elements()
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .context("CPU dequantized-weight scratch byte count overflows")?;
    let runtime_headroom_bytes = config
        .runtime_headroom_bytes
        .unwrap_or(default_runtime_headroom_bytes);
    let always_resident_bytes = runtime_headroom_bytes
        .checked_add(generate_buffer_bytes)
        .and_then(|bytes| bytes.checked_add(cpu_scratch_bytes))
        .context("native generation memory forecast overflows")?;
    let estimate = FitEstimate::from_parts(
        always_resident_bytes,
        weight_memory.checkpoint_bytes(),
        weight_memory.largest_request_bytes(),
    )?;
    plan_execution(
        config.weight_mode,
        estimate,
        config.fit_policy,
        memory_budget_bytes,
        config.cache_config()?,
    )
}

fn reconcile_streaming_cache(
    weights: &mut LoadedWeights,
    checkpoint: &Arc<Gguf>,
    plan: ExecutionPlan,
) -> Result<bool> {
    match (weights, plan) {
        (LoadedWeights::Resident(_), ExecutionPlan::Resident(_)) => Ok(false),
        (LoadedWeights::Streaming(source), ExecutionPlan::Streaming { cache, .. })
            if source.config() == cache =>
        {
            Ok(false)
        }
        (slot @ LoadedWeights::Streaming(_), ExecutionPlan::Streaming { cache, .. }) => {
            *slot = LoadedWeights::Streaming(Arc::new(StreamingSource::new(
                checkpoint.clone(),
                cache,
            )?));
            Ok(true)
        }
        (LoadedWeights::Resident(_), ExecutionPlan::Streaming { .. }) => {
            bail!("resident model produced a streaming execution plan")
        }
        (LoadedWeights::Streaming(_), ExecutionPlan::Resident(_)) => {
            bail!("streaming model produced a resident execution plan")
        }
    }
}

struct Utf8Emitter<'a> {
    tokenizer: &'a Qwen2Tokenizer,
    sink: Utf8Sink<'a>,
    #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
    openmythos: bool,
}

struct Utf8Sink<'a> {
    callback: &'a mut (dyn FnMut(&str) -> Result<()> + Send),
    pending: Vec<u8>,
    text: String,
}

impl<'a> Utf8Emitter<'a> {
    fn new(
        tokenizer: &'a Qwen2Tokenizer,
        callback: &'a mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Self {
        Self {
            tokenizer,
            sink: Utf8Sink::new(callback),
            #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
            openmythos: false,
        }
    }

    fn push(&mut self, token: u32) -> Result<()> {
        #[cfg(all(target_os = "linux", feature = "hag-openmythos"))]
        if self.openmythos {
            let bytes = native_qwen35::output_bytes(token, |id, render_special| {
                self.tokenizer.decode_bytes(&[id], render_special)
            })?;
            return self.sink.push_bytes(&bytes);
        }
        self.sink
            .push_bytes(&self.tokenizer.decode_bytes(&[token], false)?)
    }

    fn finish(self) -> Result<String> {
        self.sink.finish()
    }
}

impl<'a> Utf8Sink<'a> {
    fn new(callback: &'a mut (dyn FnMut(&str) -> Result<()> + Send)) -> Self {
        Self {
            callback,
            pending: Vec::new(),
            text: String::new(),
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.pending.extend(bytes);
        loop {
            match str::from_utf8(&self.pending) {
                Ok(chunk) => {
                    if !chunk.is_empty() {
                        (self.callback)(chunk)?;
                        self.text.push_str(chunk);
                    }
                    self.pending.clear();
                    return Ok(());
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid != 0 {
                        let chunk = str::from_utf8(&self.pending[..valid])
                            .expect("valid_up_to identifies valid UTF-8");
                        (self.callback)(chunk)?;
                        self.text.push_str(chunk);
                        self.pending.drain(..valid);
                        continue;
                    }
                    if let Some(length) = error.error_len() {
                        bail!(
                            "generated token bytes contain invalid UTF-8 at the current boundary ({length}-byte invalid sequence)"
                        );
                    }
                    return Ok(());
                }
            }
        }
    }

    fn finish(self) -> Result<String> {
        if !self.pending.is_empty() {
            bail!(
                "generation ended with an incomplete {}-byte UTF-8 sequence",
                self.pending.len()
            );
        }
        Ok(self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TemplateMetadataFixture {
        path: std::path::PathBuf,
    }

    impl TemplateMetadataFixture {
        fn new(template: Option<&str>) -> Result<Self> {
            let value = template.map(|template| {
                let mut value = (template.len() as u64).to_le_bytes().to_vec();
                value.extend_from_slice(template.as_bytes());
                (8u32, value)
            });
            Self::with_value(value)
        }

        fn with_value(value: Option<(u32, Vec<u8>)>) -> Result<Self> {
            use std::io::Write;
            use std::sync::atomic::{AtomicU64, Ordering};

            static NEXT_ID: AtomicU64 = AtomicU64::new(1);
            let path = std::env::temp_dir().join(format!(
                "beholdd-runtime-template-{}-{}.gguf",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes()); // No tensors or payload.
            bytes.extend_from_slice(&u64::from(value.is_some()).to_le_bytes());
            if let Some((kind, value)) = value {
                let key = "tokenizer.chat_template";
                bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
                bytes.extend_from_slice(key.as_bytes());
                bytes.extend_from_slice(&kind.to_le_bytes());
                bytes.extend_from_slice(&value);
            }
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            let fixture = Self { path };
            file.write_all(&bytes)?;
            Ok(fixture)
        }

        fn runtime(&self) -> NativeRuntime {
            let mut config = ModelConfig::new("template", &self.path);
            config.allow_unsafe_storage = true;
            NativeRuntime::new(RuntimeConfig {
                models: vec![config],
                ..RuntimeConfig::default()
            })
            .unwrap()
        }
    }

    impl Drop for TemplateMetadataFixture {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path).ok();
        }
    }

    #[test]
    fn cold_chat_template_preserves_exact_metadata_without_binding_or_residency() {
        // There is no architecture, tokenizer vocabulary or tensor payload in these
        // fixtures. Successful lookup cannot depend on constructing an executable model.
        for template in [
            Some(" {%- if messages %}\r\nλ\0{{ messages }}\n"),
            Some(""),
            None,
        ] {
            let fixture = TemplateMetadataFixture::new(template).unwrap();
            let runtime = fixture.runtime();
            assert_eq!(
                runtime
                    .chat_template_controlled("template", &mut || Ok(()))
                    .unwrap(),
                ChatTemplate::Checkpoint(template.map(str::to_owned))
            );
            assert_eq!(
                runtime.models().unwrap(),
                vec![ModelInfo {
                    id: "template".into(),
                    loaded: false,
                    compatibility: ModelCompatibility::Unverified,
                }]
            );
            assert!(runtime
                .entry("template")
                .unwrap()
                .loaded
                .lock()
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn chat_template_refuses_non_string_metadata_without_publishing() {
        let fixture =
            TemplateMetadataFixture::with_value(Some((4, 7u32.to_le_bytes().to_vec()))).unwrap();
        let runtime = fixture.runtime();
        let error = runtime
            .chat_template_controlled("template", &mut || Ok(()))
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("tokenizer.chat_template is not a string"));
        assert!(!runtime.models().unwrap()[0].loaded);
    }

    #[test]
    fn chat_template_cancels_before_io_and_while_waiting_for_admission() {
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![ModelConfig::new(
                "template",
                "/does/not/exist-template.gguf",
            )],
            ..RuntimeConfig::default()
        })
        .unwrap();
        let error = runtime
            .chat_template_controlled("template", &mut || bail!("cancel before I/O"))
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancel before I/O"));

        let _held = runtime.execution.lock().unwrap();
        let mut checks = 0;
        let error = runtime
            .chat_template_controlled("template", &mut || {
                checks += 1;
                if checks >= 2 {
                    bail!("cancel while waiting");
                }
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancel while waiting"));
        assert_eq!(checks, 2);
        let error = runtime
            .chat_template_controlled("missing", &mut || {
                panic!("unknown models must not enter admission")
            })
            .unwrap_err();
        assert!(error.to_string().contains("unknown model \"missing\""));
        assert!(!runtime.models().unwrap()[0].loaded);
    }

    #[test]
    fn chat_template_checks_cancellation_after_metadata_without_publication() {
        let fixture = TemplateMetadataFixture::new(Some("exact template")).unwrap();
        let runtime = fixture.runtime();
        let mut total_checks = 0;
        runtime
            .chat_template_controlled("template", &mut || {
                total_checks += 1;
                Ok(())
            })
            .unwrap();
        assert!(total_checks > 1);
        let mut checks = 0;
        let error = runtime
            .chat_template_controlled("template", &mut || {
                checks += 1;
                if checks == total_checks {
                    bail!("cancel after metadata");
                }
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancel after metadata"));
        assert!(!runtime.models().unwrap()[0].loaded);
        assert!(runtime.execution.try_lock().is_ok());
    }

    #[cfg(feature = "test-fault-injection")]
    #[test]
    fn resident_chat_template_matches_cold_metadata_without_reopening_or_replacing() {
        let template = "{{ messages }}\n  exact resident template\r\n";
        let fixture = WorkerFaultFixture::with_template(Some(template)).unwrap();
        let mut config = ModelConfig::new("template", &fixture.path);
        config.allow_unsafe_storage = true;
        config.memory_budget_bytes = Some(16 * 1024 * 1024);
        config.runtime_headroom_bytes = Some(0);
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![config.clone()],
            ..RuntimeConfig::default()
        })
        .unwrap();
        let cold = runtime
            .chat_template_controlled("template", &mut || Ok(()))
            .unwrap();
        assert_eq!(cold, ChatTemplate::Checkpoint(Some(template.to_owned())));
        let loaded = Arc::new(
            runtime
                .load_model(
                    &config,
                    &GenerateRequest {
                        model: "template".into(),
                        prompt: "a".into(),
                        max_new_tokens: 1,
                    },
                    &mut || Ok(()),
                    &native_phase::Recorder::new(false),
                )
                .unwrap(),
        );
        *runtime.entry("template").unwrap().loaded.lock().unwrap() = Some(loaded.clone());
        std::fs::remove_file(&fixture.path).unwrap();
        assert_eq!(
            runtime
                .chat_template_controlled("template", &mut || Ok(()))
                .unwrap(),
            cold
        );
        assert!(Arc::ptr_eq(
            runtime
                .entry("template")
                .unwrap()
                .loaded
                .lock()
                .unwrap()
                .as_ref()
                .unwrap(),
            &loaded
        ));
    }

    #[cfg(all(target_os = "linux", feature = "hag-native"))]
    #[test]
    fn explicit_hag_registry_is_lazy_and_refuses_impossible_requests_before_io() {
        let mut model = ModelConfig::new("hag-test", "/does/not/exist.gguf");
        model.execution_backend = ExecutionBackend::Hag;
        model.hag = Some(crate::HagConfig {
            context_capacity: 2048,
            device: "/dev/dri/renderD999".into(),
            memory_budget_bytes: 768 * 1024 * 1024,
        });
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![model],
            ..RuntimeConfig::default()
        })
        .unwrap();
        assert_eq!(runtime.descriptor().execution_backends, ["cpu", "hag"]);
        assert!(!runtime.models().unwrap()[0].loaded);
        runtime.unload("hag-test").unwrap();
        let error = runtime
            .generate(
                GenerateRequest {
                    model: "hag-test".into(),
                    prompt: "Hello".into(),
                    max_new_tokens: native_hag::CAPACITY + 1,
                },
                &mut |_| Ok(()),
            )
            .unwrap_err();
        assert!(
            matches!(error.downcast_ref::<RequestError>(), Some(RequestError::ContextCapacity { capacity, .. }) if *capacity == native_hag::CAPACITY)
        );
        assert_eq!(
            runtime
                .execution_limits("hag-test")
                .unwrap()
                .max_evaluated_tokens,
            Some(2048)
        );
        assert!(runtime
            .plan_request("hag-test", "a", Some(native_hag::CAPACITY + 1), 512)
            .unwrap_err()
            .downcast_ref::<RequestError>()
            .is_some());
        assert_eq!(
            runtime.models().unwrap()[0].compatibility,
            ModelCompatibility::Unverified
        );
    }

    #[cfg(all(target_os = "linux", feature = "hag-native"))]
    #[test]
    fn hag_registry_reports_each_configured_capacity_before_checkpoint_io() {
        let models = [8192, 16384]
            .into_iter()
            .map(|capacity| {
                let mut model = ModelConfig::new(format!("hag-{capacity}"), "/does/not/exist.gguf");
                model.execution_backend = ExecutionBackend::Hag;
                model.hag = Some(crate::HagConfig {
                    device: "/dev/dri/renderD999".into(),
                    memory_budget_bytes: 4608 << 20,
                    context_capacity: capacity,
                });
                model
            })
            .collect();
        let runtime = NativeRuntime::new(RuntimeConfig {
            models,
            ..RuntimeConfig::default()
        })
        .unwrap();
        for capacity in [8192, 16384] {
            let model = format!("hag-{capacity}");
            assert_eq!(
                runtime
                    .execution_limits(&model)
                    .unwrap()
                    .max_evaluated_tokens,
                Some(capacity)
            );
            for error in [
                runtime
                    .plan_request(&model, "hello", Some(capacity + 1), 512)
                    .unwrap_err(),
                runtime
                    .generate(
                        GenerateRequest {
                            model: model.clone(),
                            prompt: "hello".into(),
                            max_new_tokens: capacity + 1,
                        },
                        &mut |_| Ok(()),
                    )
                    .unwrap_err(),
            ] {
                assert!(
                    matches!(error.downcast_ref::<RequestError>(), Some(RequestError::ContextCapacity { capacity: actual, .. }) if *actual == capacity)
                );
            }
            assert!(runtime.models().unwrap().iter().all(|model| !model.loaded));
        }
    }

    #[test]
    fn cpu_request_planning_preserves_defaults_without_checkpoint_io() {
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![ModelConfig::new("cpu", "/does/not/exist-plan.gguf")],
            ..RuntimeConfig::default()
        })
        .unwrap();
        assert_eq!(
            runtime.execution_limits("cpu").unwrap(),
            ExecutionLimits::default()
        );
        assert_eq!(
            runtime
                .plan_request("cpu", "hello", None, 512)
                .unwrap()
                .max_new_tokens,
            512
        );
        assert_eq!(
            runtime
                .plan_request("cpu", "hello", Some(7), 512)
                .unwrap()
                .max_new_tokens,
            7
        );
        assert!(runtime.plan_request("missing", "hello", None, 512).is_err());
        for (prompt, tokens, default) in [
            ("", Some(1), 512),
            ("hello", Some(0), 512),
            ("hello", None, 0),
        ] {
            assert!(runtime
                .plan_request("cpu", prompt, tokens, default)
                .unwrap_err()
                .downcast_ref::<RequestError>()
                .is_some());
        }
        assert!(!runtime.models().unwrap()[0].loaded);
    }

    #[cfg(all(
        target_os = "linux",
        feature = "hag-native",
        feature = "test-fault-injection"
    ))]
    #[test]
    fn hag_planning_reads_only_tokenizer_metadata_and_never_publishes_or_resets() {
        let fixture = WorkerFaultFixture::new().unwrap();
        let mut config = ModelConfig::new("plan", &fixture.path);
        config.allow_unsafe_storage = true;
        config.execution_backend = ExecutionBackend::Hag;
        config.hag = Some(crate::HagConfig {
            context_capacity: 2048,
            device: "/dev/dri/renderD999".into(),
            memory_budget_bytes: 1,
        });
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![config.clone()],
            ..RuntimeConfig::default()
        })
        .unwrap();
        let prompt = "a".repeat(native_hag::CAPACITY - 1);
        assert_eq!(
            runtime
                .plan_request("plan", &prompt, None, 512)
                .unwrap()
                .max_new_tokens,
            2
        );
        assert_eq!(
            runtime
                .plan_request("plan", &prompt, Some(2), 512)
                .unwrap()
                .max_new_tokens,
            2
        );
        assert!(matches!(
            runtime
                .plan_request("plan", &prompt, Some(3), 512)
                .unwrap_err()
                .downcast_ref::<RequestError>(),
            Some(RequestError::ContextCapacity { .. })
        ));
        assert!(runtime
            .plan_request("plan", &"a".repeat(native_hag::CAPACITY + 1), None, 512)
            .is_err());
        assert_eq!(
            runtime
                .plan_request("plan", &"a".repeat(native_hag::CAPACITY), None, 512)
                .unwrap()
                .max_new_tokens,
            1
        );
        assert!(matches!(
            runtime
                .plan_request("plan", &"a".repeat(native_hag::CAPACITY), Some(2), 512)
                .unwrap_err()
                .downcast_ref::<RequestError>(),
            Some(RequestError::ContextCapacity { capacity, .. }) if *capacity == native_hag::CAPACITY
        ));
        assert!(!runtime.models().unwrap()[0].loaded);

        // Publish a tiny CPU owner solely to test the shared resident tokenizer branch.
        // This fixture cannot execute HAG and its configured render node does not exist.
        config.execution_backend = ExecutionBackend::Cpu;
        config.hag = None;
        config.memory_budget_bytes = Some(16 * 1024 * 1024);
        config.runtime_headroom_bytes = Some(0);
        let loaded = Arc::new(
            runtime
                .load_model(
                    &config,
                    &GenerateRequest {
                        model: "plan".into(),
                        prompt: "a".into(),
                        max_new_tokens: 1,
                    },
                    &mut || Ok(()),
                    &native_phase::Recorder::new(false),
                )
                .unwrap(),
        );
        *runtime.entry("plan").unwrap().loaded.lock().unwrap() = Some(loaded.clone());
        std::fs::remove_file(&fixture.path).unwrap();
        assert_eq!(
            runtime
                .plan_request("plan", &prompt, None, 512)
                .unwrap()
                .max_new_tokens,
            2
        );
        assert!(runtime.plan_request("plan", &prompt, Some(3), 512).is_err());
        assert!(Arc::ptr_eq(
            runtime
                .entry("plan")
                .unwrap()
                .loaded
                .lock()
                .unwrap()
                .as_ref()
                .unwrap(),
            &loaded
        ));
    }

    #[cfg(all(target_os = "linux", feature = "hag-native"))]
    #[test]
    fn controlled_hag_planning_cancels_before_admission_or_metadata() {
        let mut config = ModelConfig::new("plan", "/does/not/exist-plan.gguf");
        config.execution_backend = ExecutionBackend::Hag;
        config.hag = Some(crate::HagConfig {
            context_capacity: 2048,
            device: "/dev/dri/renderD999".into(),
            memory_budget_bytes: 1,
        });
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![config],
            ..RuntimeConfig::default()
        })
        .unwrap();
        let _held = runtime.execution.lock().unwrap();
        let mut checks = 0;
        let error = runtime
            .plan_request_controlled("plan", "a", None, 512, &mut || {
                checks += 1;
                if checks >= 2 {
                    bail!("cancelled before admission");
                }
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancelled before admission"));
        assert_eq!(checks, 2);
        assert!(!runtime.models().unwrap()[0].loaded);
    }

    #[test]
    fn registry_is_lazy_and_sorted_without_touching_model_files() {
        let config = RuntimeConfig {
            models: vec![
                ModelConfig::new("zeta", "/does/not/exist-zeta.gguf"),
                ModelConfig::new("alpha", "/does/not/exist-alpha.gguf"),
            ],
            ..RuntimeConfig::default()
        };
        let runtime = NativeRuntime::new(config).unwrap();
        assert_eq!(
            runtime.models().unwrap(),
            vec![
                ModelInfo {
                    id: "alpha".into(),
                    loaded: false,
                    compatibility: ModelCompatibility::Unverified,
                },
                ModelInfo {
                    id: "zeta".into(),
                    loaded: false,
                    compatibility: ModelCompatibility::Unverified,
                }
            ]
        );
    }

    #[test]
    fn unknown_model_is_reported_before_checkpoint_io() {
        let runtime = NativeRuntime::new(RuntimeConfig::default()).unwrap();
        let error = runtime
            .generate(
                GenerateRequest {
                    model: "missing".into(),
                    prompt: "hello".into(),
                    max_new_tokens: 1,
                },
                &mut |_| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("unknown model \"missing\""));
    }

    #[test]
    fn failed_first_bind_keeps_the_configured_model_unverified() {
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![ModelConfig::new(
                "broken",
                "/does/not/exist-failed-bind.gguf",
            )],
            ..RuntimeConfig::default()
        })
        .unwrap();
        assert!(runtime
            .generate(
                GenerateRequest {
                    model: "broken".into(),
                    prompt: "hello".into(),
                    max_new_tokens: 1,
                },
                &mut |_| Ok(()),
            )
            .is_err());
        assert_eq!(
            runtime.models().unwrap(),
            vec![ModelInfo {
                id: "broken".into(),
                loaded: false,
                compatibility: ModelCompatibility::Unverified,
            }]
        );
    }

    #[test]
    #[ignore = "requires BEHOLDD_QWEN3_DENSE_MODEL pointing to Qwen3-0.6B-Q8_0.gguf"]
    fn real_dense_qwen3_routes_through_the_native_runtime() -> Result<()> {
        let model_path = std::path::PathBuf::from(
            std::env::var_os("BEHOLDD_QWEN3_DENSE_MODEL")
                .context("set BEHOLDD_QWEN3_DENSE_MODEL to the Qwen3-0.6B-Q8_0.gguf checkpoint")?,
        );
        if !model_path.is_file() {
            bail!(
                "BEHOLDD_QWEN3_DENSE_MODEL is not a checkpoint file: {}",
                model_path.display()
            );
        }
        let mut model = ModelConfig::new("qwen3-0.6b", model_path);
        model.weight_mode = beholdd_model::execution::WeightMode::Resident;
        model.memory_budget_bytes = Some(2_000_000_000);
        model.runtime_headroom_bytes = Some(0);
        model.allow_unsafe_storage = true;
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![model],
            memory_budget_bytes: None,
            runtime_headroom_bytes: 0,
        })?;

        let mut chunks = Vec::new();
        let generation = runtime.generate(
            GenerateRequest {
                model: "qwen3-0.6b".into(),
                prompt: "Hello".into(),
                max_new_tokens: 1,
            },
            &mut |text| {
                chunks.push(text.to_owned());
                Ok(())
            },
        )?;
        assert_eq!(generation.token_ids, [21806]);
        assert_eq!(generation.text, " Answer");
        assert_eq!(chunks.concat(), generation.text);
        assert_eq!(
            runtime.models()?,
            vec![ModelInfo {
                id: "qwen3-0.6b".into(),
                loaded: true,
                compatibility: ModelCompatibility::Compatible(ModelCapabilities::CompletionOnly),
            }]
        );
        Ok(())
    }

    #[test]
    fn native_runtime_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NativeRuntime>();
    }

    #[test]
    fn descriptor_and_idle_unload_do_not_open_checkpoint_paths() {
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![ModelConfig::new(
                "owner/model",
                "/does/not/exist-unload.gguf",
            )],
            ..RuntimeConfig::default()
        })
        .unwrap();
        assert_eq!(runtime.descriptor().runtime, RuntimeKind::Native);
        assert_eq!(runtime.descriptor().execution_backends, ["cpu"]);
        assert_eq!(runtime.descriptor().architectures, ["qwen3", "qwen3moe"]);
        assert_eq!(runtime.descriptor().model_formats, ["gguf"]);
        assert!(runtime.descriptor().operations.unload);
        runtime.unload("owner/model").unwrap();
        runtime.unload("owner/model").unwrap();
        assert!(!runtime.models().unwrap()[0].loaded);
        assert!(
            matches!(runtime.unload("missing"), Err(UnloadError::NotFound(id)) if id == "missing")
        );
    }

    #[test]
    fn unload_refuses_runtime_admission_without_changing_residency() {
        let execution = Mutex::new(());
        let loaded = Mutex::new(Some(42));
        let held = execution.lock().unwrap();
        assert!(matches!(
            unload_registered_model(&execution, &loaded),
            Err(UnloadError::Busy)
        ));
        assert_eq!(*loaded.lock().unwrap(), Some(42));
        drop(held);
        // The refusal did not enqueue future work.
        assert_eq!(*loaded.lock().unwrap(), Some(42));
        unload_registered_model(&execution, &loaded).unwrap();
        assert_eq!(*loaded.lock().unwrap(), None);
    }

    #[test]
    fn unload_releases_registry_before_cleanup_but_keeps_runtime_admission() {
        // Arc ownership gives the probe a non-recursive way to inspect the real registry.
        struct OwnedDropProbe {
            execution: Arc<Mutex<()>>,
            registry: std::sync::Weak<Mutex<Option<OwnedDropProbe>>>,
            dropped: Arc<std::sync::atomic::AtomicBool>,
        }

        impl Drop for OwnedDropProbe {
            fn drop(&mut self) {
                assert!(self
                    .registry
                    .upgrade()
                    .unwrap()
                    .try_lock()
                    .unwrap()
                    .is_none());
                assert!(matches!(
                    self.execution.try_lock(),
                    Err(TryLockError::WouldBlock)
                ));
                self.dropped
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }

        let execution = Arc::new(Mutex::new(()));
        let loaded = Arc::new(Mutex::new(None));
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        *loaded.lock().unwrap() = Some(OwnedDropProbe {
            execution: Arc::clone(&execution),
            registry: Arc::downgrade(&loaded),
            dropped: Arc::clone(&dropped),
        });
        unload_registered_model(&execution, &loaded).unwrap();
        assert!(dropped.load(std::sync::atomic::Ordering::Acquire));
        assert!(execution.try_lock().is_ok());
        unload_registered_model(&execution, &loaded).unwrap();
    }

    #[test]
    fn model_listing_borrows_published_state_without_retaining_weights() {
        let model = Arc::new(42);
        let published = Mutex::new(Some(Arc::clone(&model)));
        let observed = inspect_published_model(&published, |value| {
            assert_eq!(Arc::strong_count(&model), 2);
            assert!(matches!(
                published.try_lock(),
                Err(TryLockError::WouldBlock)
            ));
            *value
        });
        assert_eq!(observed, Some(42));
        let held = published.lock().unwrap();
        assert_eq!(
            inspect_published_model(&published, |_| unreachable!()),
            None::<()>
        );
        drop(held);
        let execution = Mutex::new(());
        unload_registered_model(&execution, &published).unwrap();
        assert_eq!(Arc::strong_count(&model), 1);
        assert_eq!(
            inspect_published_model(&published, |_| unreachable!()),
            None::<()>
        );
    }

    #[test]
    fn poisoned_runtime_admission_cannot_authorize_unload() {
        let execution = Mutex::new(());
        let loaded = Mutex::new(Some(42));
        let _ = std::panic::catch_unwind(|| {
            let _held = execution.lock().unwrap();
            panic!("synthetic poisoned admission");
        });
        assert!(matches!(
            unload_registered_model(&execution, &loaded),
            Err(UnloadError::Failed(_))
        ));
        assert_eq!(*loaded.lock().unwrap(), Some(42));
    }

    #[test]
    fn runtime_wide_admission_precedes_checkpoint_io() {
        let config = RuntimeConfig {
            models: vec![ModelConfig::new(
                "alpha",
                "/does/not/exist-runtime-admission.gguf",
            )],
            ..RuntimeConfig::default()
        };
        let runtime = Arc::new(NativeRuntime::new(config).unwrap());
        let held = runtime.execution.lock().unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn({
            let runtime = Arc::clone(&runtime);
            move || {
                started_tx.send(()).unwrap();
                let result = runtime.generate(
                    GenerateRequest {
                        model: "alpha".into(),
                        prompt: "metadata-free".into(),
                        max_new_tokens: 1,
                    },
                    &mut |_| Ok(()),
                );
                done_tx.send(result).unwrap();
            }
        });

        started_rx.recv().unwrap();
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "generation reached checkpoint preflight while runtime admission was held"
        );
        drop(held);
        assert!(done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .is_err());
        worker.join().unwrap();
    }

    #[test]
    fn controlled_generation_can_cancel_while_waiting_for_runtime_admission() {
        let config = RuntimeConfig {
            models: vec![ModelConfig::new(
                "alpha",
                "/does/not/exist-controlled-admission.gguf",
            )],
            ..RuntimeConfig::default()
        };
        let runtime = Arc::new(NativeRuntime::new(config).unwrap());
        let held = runtime.execution.lock().unwrap();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (polled_tx, polled_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn({
            let runtime = Arc::clone(&runtime);
            let cancelled = Arc::clone(&cancelled);
            move || {
                let result = runtime.generate_controlled(
                    GenerateRequest {
                        model: "alpha".into(),
                        prompt: "metadata-free".into(),
                        max_new_tokens: 1,
                    },
                    &mut || {
                        polled_tx.send(()).ok();
                        if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                            bail!("synthetic queued cancellation");
                        }
                        Ok(())
                    },
                    &mut |_| Ok(()),
                );
                done_tx.send(result).unwrap();
            }
        });

        polled_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("controlled request never polled while admission was held");
        cancelled.store(true, std::sync::atomic::Ordering::Release);
        let error = done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("controlled request did not leave the admission queue")
            .unwrap_err();
        assert!(format!("{error:#}").contains("synthetic queued cancellation"));
        drop(held);
        worker.join().unwrap();
    }

    #[test]
    fn eviction_keeps_only_the_requested_entry_without_model_io() {
        let entries = BTreeMap::from([
            ("alpha".to_owned(), Mutex::new(Some(1usize))),
            ("beta".to_owned(), Mutex::new(Some(2usize))),
            ("gamma".to_owned(), Mutex::new(None)),
        ]);
        let evicted = take_other_loaded(
            entries.iter().map(|(id, state)| (id.as_str(), state)),
            "beta",
        )
        .unwrap();

        assert_eq!(evicted, vec![1]);
        assert_eq!(*entries["alpha"].lock().unwrap(), None);
        assert_eq!(*entries["beta"].lock().unwrap(), Some(2));
        assert_eq!(*entries["gamma"].lock().unwrap(), None);
    }

    #[test]
    fn model_listing_observes_a_failed_probe_while_execution_is_busy() {
        struct TestModel {
            execution: Mutex<()>,
            probe: RwLock<bool>,
        }

        let model = Arc::new(TestModel {
            execution: Mutex::new(()),
            probe: RwLock::new(false),
        });
        let published = Mutex::new(Some(Arc::clone(&model)));
        let _busy = model.execution.lock().unwrap();

        assert_eq!(
            inspect_published_model(&published, |observed| {
                observe_liveness(&observed.execution, &observed.probe, |live| *live)
            }),
            Some(false)
        );

        *model.probe.write().unwrap() = true;
        assert_eq!(
            inspect_published_model(&published, |observed| {
                observe_liveness(&observed.execution, &observed.probe, |live| *live)
            }),
            Some(true)
        );
    }

    #[test]
    fn cache_reconciliation_refuses_mode_mismatch_without_io() {
        // The mismatch arms are intentionally explicit rather than silently changing a
        // configured execution mode. Their full construction is covered by model gates.
        let _signature: fn(&mut LoadedWeights, &Arc<Gguf>, ExecutionPlan) -> Result<bool> =
            reconcile_streaming_cache;
    }

    #[test]
    fn utf8_sink_waits_for_complete_scalars_and_preserves_chunks() {
        let mut chunks = Vec::new();
        let text = {
            let mut callback = |chunk: &str| {
                chunks.push(chunk.to_owned());
                Ok(())
            };
            let mut sink = Utf8Sink::new(&mut callback);
            sink.push_bytes(b"hello ").unwrap();
            sink.push_bytes(&[0xe2]).unwrap();
            sink.push_bytes(&[0x82]).unwrap();
            sink.push_bytes(&[0xac, b'!']).unwrap();
            sink.finish().unwrap()
        };
        assert_eq!(chunks, vec!["hello ", "€!"]);
        assert_eq!(text, "hello €!");
    }

    #[test]
    fn utf8_sink_emits_a_valid_prefix_once_before_a_split_scalar() {
        let mut chunks = Vec::new();
        let text = {
            let mut callback = |chunk: &str| {
                chunks.push(chunk.to_owned());
                Ok(())
            };
            let mut sink = Utf8Sink::new(&mut callback);
            sink.push_bytes(&[b'a', 0xe2]).unwrap();
            sink.push_bytes(&[0x82, 0xac]).unwrap();
            sink.finish().unwrap()
        };
        assert_eq!(chunks, vec!["a", "€"]);
        assert_eq!(text, "a€");
    }

    #[test]
    fn utf8_sink_rejects_invalid_and_incomplete_sequences() {
        let mut callback = |_chunk: &str| Ok(());
        let mut invalid = Utf8Sink::new(&mut callback);
        assert!(invalid.push_bytes(&[0xff]).is_err());

        let mut callback = |_chunk: &str| Ok(());
        let mut incomplete = Utf8Sink::new(&mut callback);
        incomplete.push_bytes(&[0xe2]).unwrap();
        assert!(incomplete.finish().is_err());
    }

    #[cfg(feature = "test-fault-injection")]
    #[test]
    fn native_unload_drops_a_published_streaming_source_without_inference() {
        let fixture = WorkerFaultFixture::new().unwrap();
        let mut model = ModelConfig::new("unload-fixture", &fixture.path);
        model.allow_unsafe_storage = true;
        model.memory_budget_bytes = Some(16 * 1024 * 1024);
        model.runtime_headroom_bytes = Some(0);
        model.prefetch_bytes = 4 * 1024;
        model.prefetch_queue_depth = 1;
        model.prefetch_workers = 1;
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![model],
            runtime_headroom_bytes: 0,
            ..RuntimeConfig::default()
        })
        .unwrap();
        let entry = runtime.entry("unload-fixture").unwrap();
        let loaded = runtime
            .load_model(
                &entry.config,
                &GenerateRequest {
                    model: "unload-fixture".into(),
                    prompt: "a".into(),
                    max_new_tokens: 1,
                },
                &mut || Ok(()),
                &native_phase::Recorder::new(false),
            )
            .unwrap();
        let source = {
            let probe = loaded.probe.read().unwrap();
            match &*probe {
                LoadedProbe::Cpu(LoadedWeights::Streaming(source)) => {
                    assert_eq!(source.stats().unwrap().requests(), 0);
                    Arc::downgrade(source)
                }
                _ => panic!("expected streaming source"),
            }
        };
        *entry.loaded.lock().unwrap() = Some(Arc::new(loaded));
        assert!(runtime.models().unwrap()[0].loaded);
        let active = runtime.execution.lock().unwrap();
        assert!(matches!(
            runtime.unload("unload-fixture"),
            Err(UnloadError::Busy)
        ));
        assert!(runtime.models().unwrap()[0].loaded);
        assert!(source.upgrade().is_some());
        drop(active);
        runtime.unload("unload-fixture").unwrap();
        assert!(
            source.upgrade().is_none(),
            "the source survived successful unload"
        );
        let listed = runtime.models().unwrap();
        assert!(!listed[0].loaded);
        assert_eq!(listed[0].compatibility, ModelCompatibility::Unverified);
        runtime.unload("unload-fixture").unwrap();
    }

    #[cfg(feature = "test-fault-injection")]
    #[test]
    fn stopped_real_streaming_worker_is_observed_by_native_registry_within_one_second() {
        let fixture = WorkerFaultFixture::new().unwrap();
        assert!(
            std::fs::metadata(&fixture.path).unwrap().len() < 64 * 1024,
            "the liveness fixture must stay metadata-light"
        );

        let mut model = ModelConfig::new("worker-fixture", &fixture.path);
        model.allow_unsafe_storage = true;
        model.memory_budget_bytes = Some(16 * 1024 * 1024);
        model.runtime_headroom_bytes = Some(0);
        model.prefetch_bytes = 4 * 1024;
        model.prefetch_queue_depth = 1;
        model.prefetch_workers = 1;
        let runtime = NativeRuntime::new(RuntimeConfig {
            models: vec![model],
            memory_budget_bytes: None,
            runtime_headroom_bytes: 0,
        })
        .unwrap();
        assert_eq!(
            runtime.models().unwrap(),
            vec![ModelInfo {
                id: "worker-fixture".into(),
                loaded: false,
                compatibility: ModelCompatibility::Unverified,
            }]
        );

        // Drive the public scheduling path through metadata binding and source
        // publication, then cancel at its first post-publication control boundary. This
        // creates the real StreamingSource worker but cannot enter model computation or
        // request a tensor payload.
        let mut observed_publication = false;
        let error = runtime
            .generate_controlled(
                GenerateRequest {
                    model: "worker-fixture".into(),
                    prompt: "a".into(),
                    max_new_tokens: 1,
                },
                &mut || {
                    if runtime
                        .models()?
                        .iter()
                        .any(|model| model.id == "worker-fixture" && model.loaded)
                    {
                        observed_publication = true;
                        bail!("intentional cancellation after source publication");
                    }
                    Ok(())
                },
                &mut |_| bail!("generation reached a text callback in a metadata-only gate"),
            )
            .unwrap_err();
        assert!(observed_publication);
        assert!(format!("{error:#}").contains("intentional cancellation after source publication"));
        assert!(runtime.models().unwrap()[0].loaded);

        let source = {
            let entry = runtime.entry("worker-fixture").unwrap();
            let loaded = entry
                .loaded
                .lock()
                .unwrap()
                .as_ref()
                .expect("the controlled load published its model")
                .clone();
            let probe = loaded.probe.read().unwrap();
            match &*probe {
                LoadedProbe::Cpu(LoadedWeights::Streaming(source)) => Arc::clone(source),
                _ => panic!("fixture unexpectedly selected resident mode"),
            }
        };
        source.probe().unwrap();
        let stats = source.stats().unwrap();
        assert_eq!(stats.requests(), 0);
        assert_eq!(stats.physical_reads(), 0);
        assert_eq!(stats.read_bytes(), 0);

        let started = std::time::Instant::now();
        let stopped = source
            .terminate_prefetch_workers_for_test(std::time::Duration::from_millis(900))
            .unwrap();
        assert_eq!(stopped, 1);
        loop {
            if !runtime.models().unwrap()[0].loaded {
                break;
            }
            assert!(
                started.elapsed() < std::time::Duration::from_secs(1),
                "NativeRuntime still reported the stopped source as loaded after one second"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "worker termination plus registry observation exceeded one second"
        );
        let failure = source.probe().unwrap_err();
        assert!(format!("{failure:#}").contains("checkpoint-prefetch worker(s) exited"));
        let stats = source.stats().unwrap();
        assert_eq!(stats.requests(), 0);
        assert_eq!(stats.physical_reads(), 0);
        assert_eq!(stats.read_bytes(), 0);
    }

    #[cfg(feature = "test-fault-injection")]
    struct WorkerFaultFixture {
        path: std::path::PathBuf,
    }

    #[cfg(feature = "test-fault-injection")]
    impl WorkerFaultFixture {
        fn new() -> Result<Self> {
            Self::with_template(None)
        }

        fn with_template(template: Option<&str>) -> Result<Self> {
            use std::sync::atomic::{AtomicU64, Ordering};

            static NEXT_ID: AtomicU64 = AtomicU64::new(1);
            let path = std::env::temp_dir().join(format!(
                "beholdd-runtime-worker-fault-{}-{}.gguf",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            write_worker_fault_fixture(&path, template)?;
            Ok(Self { path })
        }
    }

    #[cfg(feature = "test-fault-injection")]
    impl Drop for WorkerFaultFixture {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path).ok();
        }
    }

    #[cfg(feature = "test-fault-injection")]
    fn write_worker_fault_fixture(path: &std::path::Path, template: Option<&str>) -> Result<()> {
        let tensors: Vec<(&str, Vec<u64>)> = vec![
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("output.weight", vec![4, 2]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_q_norm.weight", vec![4]),
            ("blk.0.attn_k_norm.weight", vec![4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
            ("blk.0.ffn_gate_exps.weight", vec![4, 4, 2]),
            ("blk.0.ffn_up_exps.weight", vec![4, 4, 2]),
            ("blk.0.ffn_down_exps.weight", vec![4, 4, 2]),
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(22 + u64::from(template.is_some())).to_le_bytes());

        fixture_string_metadata(&mut bytes, "general.architecture", "qwen3moe");
        for (key, value) in [
            ("qwen3moe.block_count", 1),
            ("qwen3moe.embedding_length", 4),
            ("qwen3moe.attention.head_count", 1),
            ("qwen3moe.attention.head_count_kv", 1),
            ("qwen3moe.attention.key_length", 4),
            ("qwen3moe.attention.value_length", 4),
            ("qwen3moe.expert_count", 2),
            ("qwen3moe.expert_used_count", 1),
            ("qwen3moe.expert_feed_forward_length", 4),
            ("qwen3moe.feed_forward_length", 4),
            ("qwen3moe.context_length", 16),
        ] {
            fixture_u32_metadata(&mut bytes, key, value);
        }
        fixture_f32_metadata(
            &mut bytes,
            "qwen3moe.attention.layer_norm_rms_epsilon",
            1e-6,
        );
        fixture_f32_metadata(&mut bytes, "qwen3moe.rope.freq_base", 10_000.0);
        fixture_string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2");
        fixture_string_metadata(&mut bytes, "tokenizer.ggml.pre", "qwen2");
        fixture_string_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &["a", "<eos>"]);
        fixture_u32_array_metadata(&mut bytes, "tokenizer.ggml.token_type", &[1, 3]);
        fixture_string_array_metadata(&mut bytes, "tokenizer.ggml.merges", &[]);
        fixture_u32_metadata(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        fixture_bool_metadata(&mut bytes, "tokenizer.ggml.add_bos_token", false);
        fixture_bool_metadata(&mut bytes, "tokenizer.ggml.add_eos_token", false);
        if let Some(template) = template {
            fixture_string_metadata(&mut bytes, "tokenizer.chat_template", template);
        }

        let mut payload = Vec::new();
        let mut offsets = Vec::with_capacity(tensors.len());
        for (_, dims) in &tensors {
            offsets.push(payload.len() as u64);
            let elements = dims.iter().try_fold(1usize, |total, dimension| {
                let dimension =
                    usize::try_from(*dimension).context("fixture dimension overflow")?;
                total
                    .checked_mul(dimension)
                    .context("fixture tensor element count overflow")
            })?;
            payload.resize(
                payload
                    .len()
                    .checked_add(
                        elements
                            .checked_mul(std::mem::size_of::<f32>())
                            .context("fixture tensor byte count overflow")?,
                    )
                    .context("fixture payload byte count overflow")?,
                0,
            );
            while !payload.len().is_multiple_of(32) {
                payload.push(0);
            }
        }
        for ((name, dims), offset) in tensors.iter().zip(offsets) {
            fixture_string(&mut bytes, name);
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dimension in dims {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes)
            .with_context(|| format!("writing worker-fault fixture {}", path.display()))
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_key(bytes: &mut Vec<u8>, key: &str, kind: u32) {
        fixture_string(bytes, key);
        bytes.extend_from_slice(&kind.to_le_bytes());
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        fixture_key(bytes, key, 8);
        fixture_string(bytes, value);
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        fixture_key(bytes, key, 4);
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_f32_metadata(bytes: &mut Vec<u8>, key: &str, value: f32) {
        fixture_key(bytes, key, 6);
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_bool_metadata(bytes: &mut Vec<u8>, key: &str, value: bool) {
        fixture_key(bytes, key, 7);
        bytes.push(u8::from(value));
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_string_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        fixture_key(bytes, key, 9);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            fixture_string(bytes, value);
        }
    }

    #[cfg(feature = "test-fault-injection")]
    fn fixture_u32_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[u32]) {
        fixture_key(bytes, key, 9);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}
