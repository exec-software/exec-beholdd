//! Native model configuration, lazy residency, and serialized inference scheduling.
//!
//! The runtime owns model discovery and execution policy. HTTP-specific request shapes,
//! chat-template rendering, and tool-call conventions deliberately live above this crate.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod config;
mod native;

pub use config::{
    ExecutionBackend, HagConfig, HagQwen35Config, ModelConfig, RuntimeConfig,
    DEFAULT_RUNTIME_HEADROOM_BYTES,
};
pub use native::NativeRuntime;

use anyhow::Result;
use serde::Serialize;

/// Execution implementation, reported explicitly rather than inferred from model IDs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeKind {
    /// The runtime has not declared a native execution contract.
    #[default]
    Unknown,
    /// Beholdd's own model execution and arithmetic.
    Native,
}

/// Process-wide implementation facts; these do not certify any configured checkpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RuntimeDescriptor {
    /// Implementation that executes requests.
    pub runtime: RuntimeKind,
    /// Backends actually reachable through this serving runtime.
    pub execution_backends: Vec<&'static str>,
    /// Exact architecture tags accepted by this runtime's model dispatcher.
    pub architectures: Vec<&'static str>,
    /// Checkpoint formats accepted by the serving loader.
    pub model_formats: Vec<&'static str>,
    /// Explicitly implemented model-management operations.
    pub operations: RuntimeOperations,
}

/// Optional operations must be declared, never assumed by a client.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct RuntimeOperations {
    /// Whether the runtime implements explicit model unload.
    pub unload: bool,
}

/// An unload refusal, distinguished from generation failures for management callers.
#[derive(Debug, thiserror::Error)]
pub enum UnloadError {
    /// The model ID was never registered in this runtime.
    #[error("unknown model {0:?}")]
    NotFound(String),
    /// Execution or another management operation currently owns admission.
    #[error("runtime is busy; retry unloading after the active operation finishes")]
    Busy,
    /// Alternate runtimes must opt in to unload support.
    #[error("this runtime does not support model unload")]
    Unsupported,
    /// Internal synchronization failed; residency was not assumed safe to change.
    #[error("model unload failed: {0}")]
    Failed(String),
}

/// Capabilities implemented by one runtime model.
///
/// Every model currently marked [`ModelCompatibility::Compatible`] is a text-completion
/// model. Vision is deliberately not representable until the runtime request boundary can
/// carry structured media, so no runtime can advertise a capability that the server drops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCapabilities {
    /// Text completion through an already-rendered prompt, with no media input.
    CompletionOnly,
}

impl ModelCapabilities {
    /// The capabilities of the current native text-only runtime.
    pub const fn completion_only() -> Self {
        Self::CompletionOnly
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self::completion_only()
    }
}

/// Whether a configured model has been checked against the native runtime.
///
/// Registration is intentionally metadata-free, so a configured path starts unverified.
/// Compatibility becomes [`Self::Compatible`] only after the runtime has successfully
/// bound and published an executable model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelCompatibility {
    /// The checkpoint has not been successfully bound in this process.
    #[default]
    Unverified,
    /// The checkpoint was bound and these capabilities are executable by the runtime.
    Compatible(ModelCapabilities),
}

/// Observable state for one configured model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Stable model identifier accepted by generation requests.
    pub id: String,
    /// Whether this process currently holds a bound model and executable weight source.
    pub loaded: bool,
    /// Binding-derived compatibility and capabilities for this exact model.
    pub compatibility: ModelCompatibility,
}

/// Fixed execution-storage limits that can be reported without opening a checkpoint.
/// These describe admission policy, not checkpoint compatibility or device readiness.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ExecutionLimits {
    /// Maximum evaluated prompt and decode positions. `None` means that the runtime
    /// publishes no fixed cap; it does not promise unlimited checkpoint context.
    pub max_evaluated_tokens: Option<usize>,
}

/// Template selection metadata; rendering and support validation belong to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Preserve the caller's compatibility renderer for alternate runtimes.
    Legacy,
    /// Exact checkpoint template bytes, or explicit absence of that metadata.
    Checkpoint(Option<String>),
}

/// A caller-correctable generation request refusal, before execution state changes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    /// A completion requires at least one encoded prompt token.
    #[error("generation requires a nonempty prompt")]
    EmptyPrompt,
    /// An explicit or default output limit must permit at least one token.
    #[error("max_new_tokens must be greater than zero")]
    InvalidOutputLimit,
    /// The prompt and requested decode steps exceed the backend's storage cap.
    #[error("request with at least {prompt_tokens} prompt tokens and {max_new_tokens} output tokens exceeds the fixed serving capacity of {capacity} evaluated positions")]
    ContextCapacity {
        /// Encoded prompt length including chat framing, or the known minimum of
        /// one token when the output limit alone is refused before metadata access.
        prompt_tokens: usize,
        /// Requested output limit; explicit limits are never silently reduced.
        max_new_tokens: usize,
        /// Fixed evaluated-position capacity of this execution path.
        capacity: usize,
    },
}

impl ExecutionLimits {
    /// Resolve an omitted output budget after tokenization and validate explicit limits.
    ///
    /// The final selected token is emitted without another model evaluation, so the
    /// required positions are `prompt_tokens + max_new_tokens - 1`. Only an omitted
    /// limit may be reduced to fit the remaining positions. Arithmetic cannot wrap.
    pub fn resolve(
        self,
        prompt_tokens: usize,
        requested: Option<usize>,
        default_max_new_tokens: usize,
    ) -> std::result::Result<usize, RequestError> {
        if prompt_tokens == 0 {
            return Err(RequestError::EmptyPrompt);
        }
        let mut max_new_tokens = requested.unwrap_or(default_max_new_tokens);
        if max_new_tokens == 0 {
            return Err(RequestError::InvalidOutputLimit);
        }
        if let Some(capacity) = self.max_evaluated_tokens {
            // Subtraction follows the prompt check; adding one is safe because the
            // prompt is nonempty, including when capacity is usize::MAX.
            if prompt_tokens <= capacity {
                let remaining = capacity - prompt_tokens + 1;
                if requested.is_none() {
                    max_new_tokens = max_new_tokens.min(remaining);
                }
                if max_new_tokens <= remaining {
                    return Ok(max_new_tokens);
                }
            }
            return Err(RequestError::ContextCapacity {
                prompt_tokens,
                max_new_tokens,
                capacity,
            });
        }
        Ok(max_new_tokens)
    }
}

/// One already-rendered completion prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateRequest {
    /// Configured model identifier.
    pub model: String,
    /// Prompt text, including any special tokens rendered by the caller.
    pub prompt: String,
    /// Maximum number of content tokens to emit.
    pub max_new_tokens: usize,
}

/// Why generation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model selected its EOS token.
    Stop,
    /// The caller's maximum token count was reached.
    Length,
}

/// Completed generation and usage accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// Decoded UTF-8 content, excluding the stop token.
    pub text: String,
    /// Emitted content token IDs, excluding the stop token.
    pub token_ids: Vec<u32>,
    /// Number of encoded prompt tokens evaluated.
    pub prompt_tokens: usize,
    /// Number of content tokens emitted.
    pub completion_tokens: usize,
    /// Condition that ended generation.
    pub finish_reason: FinishReason,
}

/// Synchronous, object-safe inference boundary shared by HTTP and embedded callers.
pub trait InferenceRuntime: Send + Sync {
    /// Describe implemented execution paths without opening checkpoints or running inference.
    fn descriptor(&self) -> RuntimeDescriptor {
        RuntimeDescriptor::default()
    }

    /// Release an idle registered model; already-unloaded models succeed idempotently.
    ///
    /// This synchronous operation may join weight-source workers. Callers on an async
    /// executor must move it to blocking work. Busy requests must be refused, not queued
    /// to unexpectedly unload a model after an active generation finishes.
    fn unload(&self, _model: &str) -> std::result::Result<(), UnloadError> {
        Err(UnloadError::Unsupported)
    }

    /// List configured models and their process-observed residency.
    fn models(&self) -> Result<Vec<ModelInfo>>;

    /// Report fixed backend limits without loading or validating model metadata.
    fn execution_limits(&self, _model: &str) -> Result<ExecutionLimits> {
        Ok(ExecutionLimits::default())
    }

    /// Read template metadata without loading weights, allocating execution resources,
    /// or changing residency. Call from blocking work: native lookup may open GGUF
    /// metadata. The caller decides whether a checkpoint template is supported.
    ///
    /// Alternate runtimes preserve their existing renderer unless they opt in.
    fn chat_template_controlled(
        &self,
        _model: &str,
        control: &mut (dyn FnMut() -> Result<()> + Send),
    ) -> Result<ChatTemplate> {
        control()?;
        Ok(ChatTemplate::Legacy)
    }

    /// Plan an already-rendered prompt without device allocation, payload reads or
    /// changes to resident request state. Call from blocking work: native planning
    /// may open tokenizer metadata. Generation revalidates the returned request.
    ///
    /// Alternate runtimes retain the caller's fallback by default. A runtime with
    /// context limits should override this method to resolve omission after encoding.
    fn plan_request(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
    ) -> Result<GenerateRequest> {
        if prompt.is_empty() {
            return Err(RequestError::EmptyPrompt.into());
        }
        let max_new_tokens = max_new_tokens.unwrap_or(default_max_new_tokens);
        if max_new_tokens == 0 {
            return Err(RequestError::InvalidOutputLimit.into());
        }
        Ok(GenerateRequest {
            model: model.to_owned(),
            prompt: prompt.to_owned(),
            max_new_tokens,
        })
    }

    /// Plan while polling a serving-control callback before and after blocking work.
    /// Native planning additionally polls around metadata and tokenizer operations.
    fn plan_request_controlled(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
        control: &mut (dyn FnMut() -> Result<()> + Send),
    ) -> Result<GenerateRequest> {
        control()?;
        let request = self.plan_request(model, prompt, max_new_tokens, default_max_new_tokens)?;
        control()?;
        Ok(request)
    }

    /// Generate from an already-rendered prompt, delivering only valid UTF-8 chunks.
    fn generate(
        &self,
        request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Result<Generation>;

    /// Generate while polling a fallible serving-control callback independently of text.
    ///
    /// Runtimes with long admission, loading, or prefill phases should override this and
    /// invoke `control` at safe progress boundaries. The default preserves compatibility
    /// for alternate runtimes by checking once before their existing generation method.
    fn generate_controlled(
        &self,
        request: GenerateRequest,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Result<Generation> {
        control()?;
        self.generate(request, on_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternate_template_default_only_polls_control() {
        struct AlternateRuntime;

        impl InferenceRuntime for AlternateRuntime {
            fn models(&self) -> Result<Vec<ModelInfo>> {
                panic!("template selection must not list or load models")
            }

            fn generate(
                &self,
                _: GenerateRequest,
                _: &mut (dyn FnMut(&str) -> Result<()> + Send),
            ) -> Result<Generation> {
                panic!("template selection must not generate")
            }
        }

        let runtime: &dyn InferenceRuntime = &AlternateRuntime;
        let mut checks = 0;
        assert_eq!(
            runtime
                .chat_template_controlled("alternate", &mut || {
                    checks += 1;
                    Ok(())
                })
                .unwrap(),
            ChatTemplate::Legacy
        );
        assert_eq!(checks, 1);
        let error = runtime
            .chat_template_controlled("alternate", &mut || anyhow::bail!("cancelled"))
            .unwrap_err();
        assert_eq!(error.to_string(), "cancelled");
    }

    #[test]
    fn omitted_limits_fit_remaining_positions_and_explicit_limits_are_preserved() {
        let limits = ExecutionLimits {
            max_evaluated_tokens: Some(64),
        };
        assert_eq!(limits.resolve(9, None, 512), Ok(56));
        assert_eq!(limits.resolve(9, None, 8), Ok(8));
        assert_eq!(limits.resolve(63, None, 512), Ok(2));
        assert_eq!(limits.resolve(64, None, 512), Ok(1));
        assert_eq!(limits.resolve(9, Some(56), 1), Ok(56));
        assert_eq!(
            limits.resolve(9, Some(57), 512),
            Err(RequestError::ContextCapacity {
                prompt_tokens: 9,
                max_new_tokens: 57,
                capacity: 64,
            })
        );
        assert!(matches!(
            limits.resolve(65, None, 512),
            Err(RequestError::ContextCapacity { .. })
        ));
        assert_eq!(ExecutionLimits::default().resolve(63, None, 512), Ok(512));
    }

    #[test]
    fn request_limits_refuse_empty_zero_and_overflow_without_wrapping() {
        let limits = ExecutionLimits {
            max_evaluated_tokens: Some(64),
        };
        assert_eq!(
            limits.resolve(0, Some(1), 512),
            Err(RequestError::EmptyPrompt)
        );
        assert_eq!(
            limits.resolve(1, Some(0), 512),
            Err(RequestError::InvalidOutputLimit)
        );
        assert_eq!(
            limits.resolve(1, None, 0),
            Err(RequestError::InvalidOutputLimit)
        );
        for (prompt, output) in [(usize::MAX, 2), (2, usize::MAX)] {
            assert!(matches!(
                limits.resolve(prompt, Some(output), 512),
                Err(RequestError::ContextCapacity { .. })
            ));
        }
        let largest = ExecutionLimits {
            max_evaluated_tokens: Some(usize::MAX),
        };
        assert_eq!(largest.resolve(1, None, usize::MAX), Ok(usize::MAX));
        assert_eq!(largest.resolve(usize::MAX, None, usize::MAX), Ok(1));
        assert!(ExecutionLimits {
            max_evaluated_tokens: Some(0)
        }
        .resolve(1, None, 1)
        .is_err());
    }

    #[test]
    fn model_capabilities_are_explicitly_completion_only() {
        assert_eq!(
            ModelCapabilities::default(),
            ModelCapabilities::CompletionOnly
        );
        assert_eq!(
            ModelCompatibility::default(),
            ModelCompatibility::Unverified
        );
    }
}
