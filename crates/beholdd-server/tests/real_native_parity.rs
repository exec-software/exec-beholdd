//! Opt-in real-checkpoint parity gate for embedded and HTTP inference.
//!
//! This target is compiled by ordinary tests but does not read checkpoint metadata or
//! tensor payloads unless its ignored test is selected explicitly. Run it only on an
//! idle, cool host, inside the repository's thermal guard and a hard resource cgroup.
//! Prebuild the release test outside the gate, then use this as the guarded inner command
//! (the real config budget is 40,000,000,000 bytes under a 48 GiB cgroup cap):
//! The harness deliberately does not reread and hash a 32 GB checkpoint. Before starting
//! the guarded unit, the operator must independently require SHA-256
//! `a4a0207f4653bfece73d9818c83acf714f5593525fe3aab7026347fd73090fcc`; the template
//! fingerprint below verifies protocol revision, not artifact integrity.
//!
//! ```text
//! cargo test --offline --locked --release -p beholdd-server \
//!   --test real_native_parity --no-run
//!
//! BEHOLDD_REAL_CONFIG=/absolute/path/to/config.yaml \
//! python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 -- \
//! env CARGO_BUILD_JOBS=2 cargo test --offline --locked --release -p beholdd-server \
//!   --test real_native_parity real_qwen_embed_and_http_nonstream_are_identical -- \
//!   --ignored --exact --nocapture --test-threads=1
//! ```

use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Context, Result};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use beholdd_embed::EmbeddedRuntime;
use beholdd_format::Gguf;
use beholdd_runtime::{
    ChatTemplate, ExecutionLimits, FinishReason, GenerateRequest, Generation, InferenceRuntime,
    ModelConfig, ModelInfo, NativeRuntime, RuntimeConfig, RuntimeDescriptor, UnloadError,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const REAL_CONFIG_ENV: &str = "BEHOLDD_REAL_CONFIG";
const USER_CONTENT: &str = "a";
const QWEN_CHAT_PROMPT: &str = "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n";
const ONE_TOKEN_LIMIT: usize = 1;
// Exact tokenizer.chat_template metadata from the selected, externally SHA-256-verified
// Qwen3-Coder-30B-A3B-Instruct-Q8_0 checkpoint. FNV-1a is a dependency-free version
// identifier here, not a checkpoint-integrity substitute; the checkpoint SHA remains the
// artifact-level trust anchor documented by the gate.
const QWEN_TOOL_TEMPLATE_BYTES: usize = 6_722;
const QWEN_TOOL_TEMPLATE_FNV1A64: u64 = 0xa72a_487d_0e22_3214;

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A transparent test seam that records the native result returned to HTTP.
///
/// Embedded inference uses `native` directly. The HTTP router uses this delegate,
/// which owns another `Arc` to that exact same native runtime. Recording the completed
/// generation lets the gate compare token IDs as well as the public HTTP usage fields.
struct RecordingRuntime {
    native: Arc<NativeRuntime>,
    completed: Mutex<Vec<(GenerateRequest, Generation)>>,
    templates: Mutex<Vec<(String, ChatTemplate)>>,
}

impl RecordingRuntime {
    fn new(native: Arc<NativeRuntime>) -> Self {
        Self {
            native,
            completed: Mutex::new(Vec::new()),
            templates: Mutex::new(Vec::new()),
        }
    }
}

impl InferenceRuntime for RecordingRuntime {
    fn descriptor(&self) -> RuntimeDescriptor {
        self.native.descriptor()
    }

    fn unload(&self, model: &str) -> std::result::Result<(), UnloadError> {
        self.native.unload(model)
    }

    fn models(&self) -> Result<Vec<ModelInfo>> {
        self.native.models()
    }

    fn execution_limits(&self, model: &str) -> Result<ExecutionLimits> {
        self.native.execution_limits(model)
    }

    fn chat_template_controlled(
        &self,
        model: &str,
        control: &mut (dyn FnMut() -> Result<()> + Send),
    ) -> Result<ChatTemplate> {
        let template = self.native.chat_template_controlled(model, control)?;
        self.templates
            .lock()
            .map_err(|_| anyhow!("real parity template recorder is poisoned"))?
            .push((model.to_owned(), template.clone()));
        Ok(template)
    }

    fn plan_request(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
    ) -> Result<GenerateRequest> {
        self.native
            .plan_request(model, prompt, max_new_tokens, default_max_new_tokens)
    }

    fn plan_request_controlled(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
        control: &mut (dyn FnMut() -> Result<()> + Send),
    ) -> Result<GenerateRequest> {
        self.native.plan_request_controlled(
            model,
            prompt,
            max_new_tokens,
            default_max_new_tokens,
            control,
        )
    }

    fn generate(
        &self,
        request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Result<Generation> {
        self.generate_controlled(request, &mut || Ok(()), on_text)
    }

    fn generate_controlled(
        &self,
        request: GenerateRequest,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        on_text: &mut (dyn FnMut(&str) -> Result<()> + Send),
    ) -> Result<Generation> {
        let recorded_request = request.clone();
        let generation = self.native.generate_controlled(request, control, on_text)?;
        self.completed
            .lock()
            .map_err(|_| anyhow!("real parity generation recorder is poisoned"))?
            .push((recorded_request, generation.clone()));
        Ok(generation)
    }
}

#[test]
fn native_recorder_forwards_template_refusal_and_cancellation_without_loading() -> Result<()> {
    let native = Arc::new(NativeRuntime::new(RuntimeConfig {
        models: vec![ModelConfig::new(
            "missing",
            "/does/not/exist-real-parity-template.gguf",
        )],
        memory_budget_bytes: None,
        runtime_headroom_bytes: 0,
    })?);
    let recorder = RecordingRuntime::new(Arc::clone(&native));
    let error = recorder
        .chat_template_controlled("unknown", &mut || Ok(()))
        .unwrap_err();
    assert!(format!("{error:#}").contains("unknown model"));
    let error = recorder
        .chat_template_controlled("missing", &mut || {
            Err(anyhow!("cancel selector before metadata"))
        })
        .unwrap_err();
    assert!(format!("{error:#}").contains("cancel selector before metadata"));
    assert!(recorder.templates.lock().unwrap().is_empty());
    assert!(recorder.completed.lock().unwrap().is_empty());
    assert!(!native.models()?[0].loaded);
    Ok(())
}

fn explicit_real_config_path() -> Result<PathBuf> {
    let raw = env::var_os(REAL_CONFIG_ENV).with_context(|| {
        format!(
            "{REAL_CONFIG_ENV} is unset; refusing to touch a real checkpoint. \
             Select this ignored test explicitly and point the variable at a one-model config"
        )
    })?;
    let path = PathBuf::from(raw);
    ensure!(
        path.is_absolute(),
        "{REAL_CONFIG_ENV} must be an absolute path, got {}",
        path.display()
    );
    Ok(path)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real checkpoint: requires BEHOLDD_REAL_CONFIG, thermal_guard.py, and a hard cgroup"]
async fn real_qwen_embed_and_http_nonstream_are_identical() -> Result<()> {
    let config_path = explicit_real_config_path()?;
    let config = RuntimeConfig::from_path(&config_path)
        .with_context(|| format!("loading real-gate config {}", config_path.display()))?;
    ensure!(
        config.models.len() == 1,
        "real parity gate requires exactly one configured model, found {}",
        config.models.len()
    );
    let model = config.models[0].id.clone();
    let checkpoint = Gguf::open(&config.models[0].path).with_context(|| {
        format!(
            "opening real-gate GGUF metadata {}",
            config.models[0].path.display()
        )
    })?;
    let chat_template = checkpoint
        .metadata()
        .get("tokenizer.chat_template")
        .and_then(|value| value.as_str())
        .context("real-gate checkpoint has no string tokenizer.chat_template")?;
    ensure!(
        chat_template.len() == QWEN_TOOL_TEMPLATE_BYTES
            && fnv1a64(chat_template.as_bytes()) == QWEN_TOOL_TEMPLATE_FNV1A64,
        "real-gate checkpoint chat template is not the exact Qwen3-Coder tool template supported by beholdd-server"
    );

    let native = Arc::new(NativeRuntime::new(config)?);
    let embedded = EmbeddedRuntime::new(Arc::clone(&native));
    let before = embedded.models()?;
    ensure!(
        before.len() == 1 && !before[0].loaded,
        "fresh NativeRuntime must remain metadata-free and unloaded before generation: {before:?}"
    );
    let selected_template = native.chat_template_controlled(&model, &mut || Ok(()))?;
    assert_eq!(
        selected_template,
        ChatTemplate::Checkpoint(Some(chat_template.to_owned()))
    );
    ensure!(
        !native.models()?[0].loaded,
        "template metadata lookup published model residency"
    );

    let expected_request = GenerateRequest {
        model: model.clone(),
        prompt: QWEN_CHAT_PROMPT.to_owned(),
        max_new_tokens: ONE_TOKEN_LIMIT,
    };
    let embedded_generation = tokio::task::spawn_blocking({
        let embedded = embedded.clone();
        let request = expected_request.clone();
        move || embedded.generate(request)
    })
    .await
    .context("embedded real-checkpoint worker panicked")??;

    let after_embed = embedded.models()?;
    ensure!(
        after_embed.len() == 1 && after_embed[0].loaded,
        "embedded generation did not publish a live native model: {after_embed:?}"
    );

    let recorder = Arc::new(RecordingRuntime::new(Arc::clone(&native)));
    ensure!(
        Arc::ptr_eq(&native, &recorder.native),
        "HTTP recorder must delegate to the embedded caller's exact NativeRuntime"
    );
    let response = beholdd_server::router(recorder.clone())
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": model,
                        "messages": [{"role": "user", "content": USER_CONTENT}],
                        "max_tokens": ONE_TOKEN_LIMIT,
                        "temperature": 0
                    })
                    .to_string(),
                ))?,
        )
        .await?;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|error| anyhow!("collecting real HTTP response: {error}"))?
        .to_bytes();
    let body: Value = serde_json::from_slice(&bytes).context("decoding real HTTP response")?;
    ensure!(
        status == StatusCode::OK,
        "real HTTP generation returned {status}: {body}"
    );
    assert_eq!(
        recorder.templates.lock().unwrap().as_slice(),
        &[(model.clone(), selected_template)]
    );

    let completed = recorder
        .completed
        .lock()
        .map_err(|_| anyhow!("real parity generation recorder is poisoned"))?;
    ensure!(
        completed.len() == 1,
        "HTTP request completed {} native generations instead of one",
        completed.len()
    );
    let (http_request, http_generation) = &completed[0];
    assert_eq!(http_request, &expected_request);
    assert_eq!(http_generation, &embedded_generation);

    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], model);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        json!(embedded_generation.text)
    );
    assert_eq!(
        body["choices"][0]["finish_reason"],
        match embedded_generation.finish_reason {
            FinishReason::Stop => json!("stop"),
            FinishReason::Length => json!("length"),
        }
    );
    assert_eq!(
        body["usage"]["prompt_tokens"],
        json!(embedded_generation.prompt_tokens)
    );
    assert_eq!(
        body["usage"]["completion_tokens"],
        json!(embedded_generation.completion_tokens)
    );
    assert_eq!(
        body["usage"]["total_tokens"],
        json!(embedded_generation
            .prompt_tokens
            .saturating_add(embedded_generation.completion_tokens))
    );
    assert_eq!(
        embedded_generation.completion_tokens,
        embedded_generation.token_ids.len()
    );
    ensure!(
        embedded_generation.completion_tokens == ONE_TOKEN_LIMIT,
        "one-token gate emitted {} content tokens instead of exactly one",
        embedded_generation.completion_tokens
    );
    ensure!(
        !embedded_generation.text.is_empty(),
        "one-token gate decoded to empty text"
    );
    eprintln!(
        "M3.4 parity passed: model={model:?} prompt_tokens={} completion_tokens={} token_ids={:?} text={:?}",
        embedded_generation.prompt_tokens,
        embedded_generation.completion_tokens,
        embedded_generation.token_ids,
        embedded_generation.text,
    );
    Ok(())
}
