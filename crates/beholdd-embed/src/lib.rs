//! In-process inference for applications and native callers.
//!
//! This crate is intentionally a thin handle over `beholdd-runtime`. It performs no
//! prompt rendering, protocol translation, networking, or model computation, so an
//! embedded request follows the same inference path as an HTTP request.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

pub use beholdd_runtime::{
    ExecutionLimits, FinishReason, GenerateRequest, Generation, InferenceRuntime,
    ModelCapabilities, ModelCompatibility, ModelInfo, NativeRuntime, RequestError, RuntimeConfig,
};

/// The usual embedded handle backed by beholdd's native runtime.
pub type Embed = EmbeddedRuntime<NativeRuntime>;

/// A cloneable in-process handle to an inference runtime.
///
/// The generic form is useful for tests and alternate runtimes. Production callers
/// normally use the [`Embed`] alias, whose inner runtime is [`NativeRuntime`].
pub struct EmbeddedRuntime<R: InferenceRuntime + ?Sized = dyn InferenceRuntime> {
    runtime: Arc<R>,
}

impl<R: InferenceRuntime + ?Sized> Clone for EmbeddedRuntime<R> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
        }
    }
}

impl<R: InferenceRuntime + ?Sized> EmbeddedRuntime<R> {
    /// Wrap an existing shared runtime without copying or adapting it.
    pub fn new(runtime: Arc<R>) -> Self {
        Self { runtime }
    }

    /// Return a clone of the shared runtime handle.
    pub fn runtime(&self) -> Arc<R> {
        Arc::clone(&self.runtime)
    }

    /// List configured models and their process-observed residency.
    pub fn models(&self) -> Result<Vec<ModelInfo>> {
        self.runtime.models()
    }

    /// Read one model's execution limits without loading checkpoint metadata or weights.
    pub fn execution_limits(&self, model: &str) -> Result<ExecutionLimits> {
        self.runtime.execution_limits(model)
    }

    /// Resolve an optional output limit through the runtime's metadata-only planner.
    pub fn plan_request(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
    ) -> Result<GenerateRequest> {
        self.runtime
            .plan_request(model, prompt, max_new_tokens, default_max_new_tokens)
    }

    /// Generate a complete response while discarding incremental text callbacks.
    pub fn generate(&self, request: GenerateRequest) -> Result<Generation> {
        fn discard(_: &str) -> Result<()> {
            Ok(())
        }

        self.runtime.generate(request, &mut discard)
    }

    /// Generate a response and deliver each valid UTF-8 text chunk to `on_text`.
    pub fn generate_with<F>(&self, request: GenerateRequest, mut on_text: F) -> Result<Generation>
    where
        F: FnMut(&str) -> Result<()> + Send,
    {
        self.runtime.generate(request, &mut on_text)
    }
}

impl EmbeddedRuntime<NativeRuntime> {
    /// Create an embedded native runtime from an already parsed configuration.
    pub fn from_config(config: RuntimeConfig) -> Result<Self> {
        Ok(Self::new(Arc::new(NativeRuntime::new(config)?)))
    }

    /// Load a runtime configuration file and create an embedded native runtime.
    pub fn from_config_path(path: &Path) -> Result<Self> {
        Self::from_config(RuntimeConfig::from_path(path)?)
    }

    /// Load the standard beholdd configuration and create an embedded native runtime.
    pub fn from_default_config() -> Result<Self> {
        Self::from_config(RuntimeConfig::from_default_path()?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct FakeRuntime {
        requests: Mutex<Vec<GenerateRequest>>,
    }

    impl InferenceRuntime for FakeRuntime {
        fn models(&self) -> Result<Vec<ModelInfo>> {
            Ok(vec![ModelInfo {
                id: "fake".into(),
                loaded: true,
                compatibility: ModelCompatibility::Compatible(ModelCapabilities::default()),
            }])
        }

        fn generate(
            &self,
            request: GenerateRequest,
            on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
        ) -> Result<Generation> {
            self.requests.lock().unwrap().push(request.clone());
            on_text("same ")?;
            on_text("path")?;
            Ok(Generation {
                text: "same path".into(),
                token_ids: vec![10, 11],
                prompt_tokens: 3,
                completion_tokens: 2,
                finish_reason: FinishReason::Stop,
            })
        }
    }

    fn request() -> GenerateRequest {
        GenerateRequest {
            model: "fake".into(),
            prompt: "<|im_start|>user\nhello<|im_end|>".into(),
            max_new_tokens: 8,
        }
    }

    #[test]
    fn embedded_calls_are_runtime_transparent() {
        let runtime = Arc::new(FakeRuntime::default());
        let embedded = EmbeddedRuntime::new(Arc::clone(&runtime));
        assert_eq!(
            embedded.models().unwrap(),
            vec![ModelInfo {
                id: "fake".into(),
                loaded: true,
                compatibility: ModelCompatibility::Compatible(ModelCapabilities::default()),
            }]
        );

        let mut streamed = String::new();
        let generation = embedded
            .generate_with(request(), |chunk| {
                streamed.push_str(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed, generation.text);
        assert_eq!(runtime.requests.lock().unwrap().as_slice(), &[request()]);
    }

    #[test]
    fn complete_generation_uses_the_same_runtime_method() {
        let runtime = Arc::new(FakeRuntime::default());
        let embedded = EmbeddedRuntime::new(Arc::clone(&runtime));
        let generation = embedded.generate(request()).unwrap();
        assert_eq!(generation.text, "same path");
        assert_eq!(runtime.requests.lock().unwrap().as_slice(), &[request()]);
    }

    #[test]
    fn planning_and_execution_limits_forward_without_generating() {
        struct PlanningFake;
        impl InferenceRuntime for PlanningFake {
            fn models(&self) -> Result<Vec<ModelInfo>> {
                Ok(Vec::new())
            }
            fn execution_limits(&self, model: &str) -> Result<ExecutionLimits> {
                assert_eq!(model, "fake");
                Ok(ExecutionLimits {
                    max_evaluated_tokens: Some(16),
                })
            }
            fn plan_request(
                &self,
                model: &str,
                prompt: &str,
                limit: Option<usize>,
                default: usize,
            ) -> Result<GenerateRequest> {
                assert_eq!(default, 128);
                Ok(GenerateRequest {
                    model: model.into(),
                    prompt: prompt.into(),
                    max_new_tokens: limit.unwrap_or(8),
                })
            }
            fn generate(
                &self,
                _: GenerateRequest,
                _: &mut (dyn FnMut(&str) -> Result<()> + Send),
            ) -> Result<Generation> {
                panic!("metadata planning must not generate")
            }
        }
        let embedded = EmbeddedRuntime::new(Arc::new(PlanningFake));
        assert_eq!(
            embedded
                .execution_limits("fake")
                .unwrap()
                .max_evaluated_tokens,
            Some(16)
        );
        assert_eq!(
            embedded
                .plan_request("fake", "prompt", None, 128)
                .unwrap()
                .max_new_tokens,
            8
        );
        assert_eq!(
            embedded
                .plan_request("fake", "prompt", Some(3), 128)
                .unwrap()
                .max_new_tokens,
            3
        );
    }
}
