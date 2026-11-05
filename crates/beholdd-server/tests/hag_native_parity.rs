//! Explicit, sequential real-checkpoint HAG serving qualification.
//!
//! Ordinary test runs do not open a checkpoint or GPU. Prebuild this ignored test
//! with `--features hag-native`; run the frozen test binary under the external
//! thermal/crash supervisor, a 2-GiB process cgroup with no swap, and the HAG
//! 1152-MiB budget. Require the Qwen3-0.6B-Q8_0 checkpoint SHA-256
//! e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c
//! before launching; this test intentionally does not reread/hash its 639-MB file.
//!
//! Required environment (no implicit checkpoint or device selection):
//! `BEHOLDD_HAG_MODEL=/path/to/Qwen3-0.6B-Q8_0.gguf`
//! `BEHOLDD_HAG_DEVICE=/dev/dri/renderD128 HAG_TRACE_SUBMISSION=0`
//! `BEHOLDD_HAG_STAGE=runtime` first, then `BEHOLDD_HAG_STAGE=http` with the same
//! frozen binary and armed crash watcher. Select only
//! `real_hag_serving_parity_cancellation_and_unload --ignored --exact --nocapture
//! --test-threads=1`. No cargo/build command belongs inside the live gate.
//!
//! The fixed existing chat framing encodes to nine tokens. Runtime stage executes
//! one two-token HAG request (ten evaluated positions). HTTP stage executes four successful requests
//! and one first-text cancellation, at most 46 evaluated positions in total. CPU
//! reference generation and release finish before HAG construction. The archived
//! live qualification used the frozen C16 binary. This source follows the current
//! C2048 capacity and request-planning contract; that historical result does not
//! qualify these changed bytes. The earlier C8 probe is also separate.

#![cfg(all(target_os = "linux", feature = "hag-native"))]

use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, ensure, Context, Result};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::response::Response;
use axum::Router;
use beholdd_embed::EmbeddedRuntime;
use beholdd_runtime::{
    ChatTemplate, ExecutionBackend, ExecutionLimits, FinishReason, GenerateRequest, Generation,
    HagConfig, InferenceRuntime, ModelCompatibility, ModelConfig, ModelInfo, NativeRuntime,
    RuntimeConfig, RuntimeDescriptor, UnloadError,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio::sync::Notify;
use tower::ServiceExt;

const MODEL_ID: &str = "hag-qwen3-0.6b";
const USER_CONTENT: &str = "Hello";
const RENDERED_PROMPT: &str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
const PROMPT_TOKENS: usize = 9;
const CAPACITY: usize = 2048;
const GIB: u64 = 1024 * 1024 * 1024;
const HAG_BUDGET: u64 = 1152 * 1024 * 1024;
const RUNTIME_ALLOWANCE: u64 = 128 * 1024 * 1024;
const DEADLINE: Duration = Duration::from_secs(45);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Runtime,
    Http,
}

struct Settings {
    model: PathBuf,
    device: PathBuf,
    stage: Stage,
}

impl Settings {
    fn explicit() -> Result<Self> {
        ensure!(
            env::var("HAG_TRACE_SUBMISSION").ok().as_deref() == Some("0"),
            "HAG_TRACE_SUBMISSION must be exactly 0"
        );
        let path = |name: &str| -> Result<PathBuf> {
            let value = env::var_os(name).with_context(|| format!("set explicit {name}"))?;
            let path = PathBuf::from(value);
            ensure!(path.is_absolute(), "{name} must be absolute");
            Ok(path)
        };
        let model = path("BEHOLDD_HAG_MODEL")?;
        let device = path("BEHOLDD_HAG_DEVICE")?;
        ensure!(model.is_file(), "explicit checkpoint is missing");
        ensure!(
            device == std::path::Path::new("/dev/dri/renderD128"),
            "this qualification is fixed to /dev/dri/renderD128"
        );
        let stage = match env::var("BEHOLDD_HAG_STAGE").ok().as_deref() {
            Some("runtime") => Stage::Runtime,
            Some("http") => Stage::Http,
            _ => bail!("BEHOLDD_HAG_STAGE must explicitly select runtime or http"),
        };
        Ok(Self {
            model,
            device,
            stage,
        })
    }

    fn config(&self, backend: ExecutionBackend) -> RuntimeConfig {
        let mut model = ModelConfig::new(MODEL_ID, &self.model);
        model.execution_backend = backend;
        model.memory_budget_bytes = Some(if backend == ExecutionBackend::Hag {
            2 * GIB
        } else {
            GIB
        });
        model.runtime_headroom_bytes = Some(RUNTIME_ALLOWANCE);
        if backend == ExecutionBackend::Hag {
            model.hag = Some(HagConfig {
                context_capacity: 2048,
                device: self.device.clone(),
                memory_budget_bytes: HAG_BUDGET,
            });
        }
        RuntimeConfig {
            models: vec![model],
            memory_budget_bytes: None,
            runtime_headroom_bytes: RUNTIME_ALLOWANCE,
        }
    }
}

fn generation_request(max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        model: MODEL_ID.into(),
        prompt: RENDERED_PROMPT.into(),
        max_new_tokens,
    }
}

fn request(method: Method, uri: &str, body: Value) -> Result<Request<Body>> {
    Ok(Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))?)
}

fn chat(stream: bool, max_tokens: usize) -> Value {
    json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": USER_CONTENT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream
    })
}

async fn json_body(response: Response) -> Result<(StatusCode, Value)> {
    let status = response.status();
    let bytes = tokio::time::timeout(DEADLINE, response.into_body().collect())
        .await
        .context("HTTP response timed out")??
        .to_bytes();
    ensure!(
        bytes.len() < 64 * 1024,
        "unexpectedly large fixture response"
    );
    Ok((status, serde_json::from_slice(&bytes)?))
}

async fn post(app: &Router, uri: &str, body: Value) -> Result<Response> {
    Ok(tokio::time::timeout(
        DEADLINE,
        app.clone().oneshot(request(Method::POST, uri, body)?),
    )
    .await
    .context("HTTP handler timed out")??)
}

fn finish(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn assert_nonstream(body: &Value, expected: &Generation) {
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], MODEL_ID);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], expected.text);
    assert_eq!(
        body["choices"][0]["finish_reason"],
        finish(expected.finish_reason)
    );
    assert_eq!(body["usage"]["prompt_tokens"], expected.prompt_tokens);
    assert_eq!(
        body["usage"]["completion_tokens"],
        expected.completion_tokens
    );
    assert_eq!(
        body["usage"]["total_tokens"],
        expected.prompt_tokens + expected.completion_tokens
    );
}

/// Pause only in the text callback, after a complete native step has returned.
/// This lets the test drop a real SSE body and verify the actual runtime sees its
/// closed sink. No GPU fence or transport fault is manufactured.
struct EmissionPause {
    entered: Notify,
    once: AtomicBool,
    released: Mutex<bool>,
    ready: Condvar,
}

impl EmissionPause {
    fn new() -> Self {
        Self {
            entered: Notify::new(),
            once: AtomicBool::new(false),
            released: Mutex::new(false),
            ready: Condvar::new(),
        }
    }
    fn wait_once(&self) -> Result<()> {
        if self.once.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.entered.notify_one();
        let ready = self
            .released
            .lock()
            .map_err(|_| anyhow!("pause lock poisoned"))?;
        let (ready, timeout) = self
            .ready
            .wait_timeout_while(ready, Duration::from_secs(5), |ready| !*ready)
            .map_err(|_| anyhow!("pause lock poisoned while waiting"))?;
        ensure!(
            *ready && !timeout.timed_out(),
            "SSE test did not release the emission pause"
        );
        Ok(())
    }
    fn release(&self) {
        if let Ok(mut ready) = self.released.lock() {
            *ready = true;
            self.ready.notify_all();
        }
    }
}
struct ReleaseOnDrop(Arc<EmissionPause>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// Preserve the complete runtime boundary; especially do not replace controlled
/// generation with the trait's one-poll compatibility default.
struct RecordingRuntime {
    native: Arc<NativeRuntime>,
    completed: Mutex<Vec<(GenerateRequest, Generation)>>,
    templates: Mutex<Vec<(String, ChatTemplate)>>,
    next_pause: Mutex<Option<Arc<EmissionPause>>>,
    failures: AtomicUsize,
    finished: AtomicUsize,
    changed: Notify,
}

impl RecordingRuntime {
    fn new(native: Arc<NativeRuntime>) -> Self {
        Self {
            native,
            completed: Mutex::new(Vec::new()),
            templates: Mutex::new(Vec::new()),
            next_pause: Mutex::new(None),
            failures: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            changed: Notify::new(),
        }
    }
    fn latest(&self, expected: &Generation) -> Result<()> {
        let completed = self
            .completed
            .lock()
            .map_err(|_| anyhow!("recorder poisoned"))?;
        let (request, actual) = completed.last().context("HTTP runtime did not complete")?;
        ensure!(
            *request == generation_request(1),
            "HTTP prompt differs from embedded prompt"
        );
        ensure!(
            actual == expected,
            "HTTP native token IDs/text/usage differ from CPU embedded reference"
        );
        Ok(())
    }
    async fn wait_finished(&self, count: usize) -> Result<()> {
        tokio::time::timeout(DEADLINE, async {
            while self.finished.load(Ordering::Acquire) < count {
                self.changed.notified().await;
            }
        })
        .await
        .context("native HTTP worker did not finish")?;
        Ok(())
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
            .map_err(|_| anyhow!("template recorder poisoned"))?
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
        let recorded = request.clone();
        let pause = self
            .next_pause
            .lock()
            .map_err(|_| anyhow!("pause selection poisoned"))?
            .take();
        let mut emit = |piece: &str| {
            if let Some(pause) = &pause {
                pause.wait_once()?;
            }
            on_text(piece)
        };
        let outcome = self.native.generate_controlled(request, control, &mut emit);
        match &outcome {
            Ok(value) => self
                .completed
                .lock()
                .map_err(|_| anyhow!("recorder poisoned"))?
                .push((recorded, value.clone())),
            Err(_) => {
                self.failures.fetch_add(1, Ordering::AcqRel);
            }
        }
        self.finished.fetch_add(1, Ordering::Release);
        self.changed.notify_one();
        outcome
    }
}

#[test]
fn hag_recorder_forwards_template_refusal_and_cancellation_without_loading() -> Result<()> {
    // No render node or checkpoint is opened: unknown IDs and an already-cancelled
    // controlled query must fail before native metadata or device access.
    let native = Arc::new(NativeRuntime::new(RuntimeConfig {
        models: vec![ModelConfig::new(
            "missing",
            "/does/not/exist-hag-parity-template.gguf",
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
    assert_eq!(recorder.finished.load(Ordering::Acquire), 0);
    assert!(!native.models()?[0].loaded);
    Ok(())
}

async fn cpu_reference(config: RuntimeConfig, tokens: usize) -> Result<(Generation, ChatTemplate)> {
    tokio::task::spawn_blocking(move || {
        let native = Arc::new(NativeRuntime::new(config)?);
        ensure!(!native.descriptor().execution_backends.contains(&"hag"), "CPU config advertises HAG");
        let template = native.chat_template_controlled(MODEL_ID, &mut || Ok(()))?;
        ensure!(matches!(&template, ChatTemplate::Checkpoint(Some(value)) if !value.is_empty()),
            "CPU reference must select the checkpoint's explicit chat template");
        ensure!(!native.models()?[0].loaded, "template query published CPU model residency");
        let embedded = EmbeddedRuntime::new(Arc::clone(&native));
        let output = embedded.generate(generation_request(tokens))?;
        ensure!(output.prompt_tokens == PROMPT_TOKENS && output.prompt_tokens <= CAPACITY,
            "frozen rendered prompt token count changed: {}", output.prompt_tokens);
        ensure!(output.completion_tokens == tokens && output.token_ids.len() == tokens && !output.text.is_empty(),
            "fixture must emit exactly {tokens} content tokens and nonempty UTF-8 before HAG access");
        native.unload(MODEL_ID)?;
        ensure!(!native.models()?[0].loaded, "CPU model did not unload");
        println!("PASS hag-serving cpu-reference prompt_tokens={} token_ids={:?}", output.prompt_tokens, output.token_ids);
        Ok((output, template))
    }).await.context("CPU reference worker panicked")?
}

async fn assert_unloaded(native: &Arc<NativeRuntime>) -> Result<()> {
    let native = Arc::clone(native);
    tokio::task::spawn_blocking(move || {
        native.unload(MODEL_ID)?;
        native.unload(MODEL_ID)?;
        ensure!(
            !native.models()?[0].loaded,
            "model remained published after unload"
        );
        Ok(())
    })
    .await
    .context("unload worker panicked")?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real HAG: explicit model/device/stage, frozen artifact, root crash capture, thermal guard and no-swap cgroup required"]
async fn real_hag_serving_parity_cancellation_and_unload() -> Result<()> {
    let settings = Settings::explicit()?;
    let selected_tokens = if settings.stage == Stage::Runtime {
        2
    } else {
        1
    };
    let (expected, expected_template) =
        cpu_reference(settings.config(ExecutionBackend::Cpu), selected_tokens).await?;
    if settings.stage == Stage::Runtime {
        let mut refused = settings.config(ExecutionBackend::Hag);
        let hag = refused.models[0]
            .hag
            .as_mut()
            .context("missing explicit HAG settings")?;
        hag.device = PathBuf::from("/dev/dri/renderD999");
        hag.memory_budget_bytes = 1;
        tokio::task::spawn_blocking(move || {
            let runtime = NativeRuntime::new(refused)?;
            let error = runtime
                .generate(generation_request(2), &mut |_| Ok(()))
                .unwrap_err();
            let message = format!("{error:#}");
            ensure!(
                message.contains("exceeding the configured HAG budget 1"),
                "one-byte admission must refuse before opening the missing render node: {message}"
            );
            let models = runtime.models()?;
            ensure!(
                !models[0].loaded && models[0].compatibility == ModelCompatibility::Unverified,
                "failed HAG fit published model compatibility/residency"
            );
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("metadata-only admission worker panicked")??;
        println!("PASS hag-serving metadata-admission-refusal loaded=false device-open=false");
    }
    let native = Arc::new(NativeRuntime::new(settings.config(ExecutionBackend::Hag))?);
    ensure!(
        native.execution_limits(MODEL_ID)?.max_evaluated_tokens == Some(CAPACITY),
        "HAG runtime capacity differs from this qualification"
    );
    ensure!(
        native.descriptor().execution_backends.contains(&"hag"),
        "explicit HAG configuration was not advertised"
    );
    ensure!(
        !native.models()?[0].loaded,
        "fresh HAG runtime eagerly loaded a model"
    );
    ensure!(
        native.chat_template_controlled(MODEL_ID, &mut || Ok(()))? == expected_template
            && !native.models()?[0].loaded,
        "HAG template selection differs from CPU metadata or published residency"
    );

    // Cancellation before admission must not load or allocate a model.
    {
        let native = Arc::clone(&native);
        tokio::task::spawn_blocking(move || {
            let result = native.generate_controlled(
                generation_request(1),
                &mut || bail!("qualification cancellation before admission"),
                &mut |_| Ok(()),
            );
            ensure!(
                result.is_err() && !native.models()?[0].loaded,
                "early cancellation loaded HAG"
            );
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("early cancellation worker panicked")??;
    }
    println!("PASS hag-serving early-cancellation loaded=false");

    if settings.stage == Stage::Runtime {
        let runtime = Arc::clone(&native);
        let actual = tokio::task::spawn_blocking(move || {
            EmbeddedRuntime::new(runtime).generate(generation_request(2))
        })
        .await
        .context("HAG embedded worker panicked")??;
        ensure!(
            actual == expected,
            "embedded HAG differs from embedded CPU tokens/text/usage"
        );
        ensure!(
            native.models()?[0].loaded,
            "successful HAG request did not publish residency"
        );
        assert_unloaded(&native).await?;
        println!("PASS hag-serving runtime-parity prompt_tokens=9 completion_tokens=2 unload=true");
        println!(
            "HAG_SERVING_REPORT {}",
            json!({"execution":"native-hag-serving-runtime", "capacity":CAPACITY,
            "cpu_reference_requests":1, "hag_successful_requests":1, "hag_cancelled_requests":0,
            "max_evaluated_positions":10, "prompt_tokens":9, "completion_tokens":2, "metadata_admission_refusals":1, "greedy_token_ids":expected.token_ids,
            "unloaded":true, "early_cancellation_unloaded":true})
        );
        println!("PASS hag-serving stage=runtime qualified=true");
        return Ok(());
    }

    let recorder = Arc::new(RecordingRuntime::new(Arc::clone(&native)));
    let app = beholdd_server::router(recorder.clone());
    let info = app
        .clone()
        .oneshot(request(Method::GET, "/v1/beholdd/info", json!(null))?)
        .await?;
    let (status, info) = json_body(info).await?;
    ensure!(
        status == StatusCode::OK
            && info["execution_backends"]
                .as_array()
                .is_some_and(|backends| backends.iter().any(|value| value == "hag")),
        "HTTP info omitted configured HAG"
    );

    // This request needs one more evaluated position than the current capacity.
    // Refuse before HAG allocation and before sending successful response headers.
    let (status, body) = json_body(
        post(
            &app,
            "/v1/chat/completions",
            chat(false, CAPACITY - PROMPT_TOKENS + 2),
        )
        .await?,
    )
    .await?;
    ensure!(
        status == StatusCode::BAD_REQUEST,
        "oversized request was not refused before response headers"
    );
    ensure!(
        body["error"]["type"] == "invalid_request_error"
            && body["error"]["code"] == "context_length_exceeded"
            && body["error"]["param"] == "max_tokens"
            && body.get("choices").is_none(),
        "oversized request lacks the typed capacity error envelope"
    );
    ensure!(
        recorder.finished.load(Ordering::Acquire) == 0,
        "oversized request reached generation"
    );
    ensure!(
        !native.models()?[0].loaded,
        "oversized first request published HAG resources"
    );
    println!("PASS hag-serving capacity-refusal loaded=false");

    let (status, body) =
        json_body(post(&app, "/v1/chat/completions", chat(false, 1)).await?).await?;
    ensure!(status == StatusCode::OK, "nonstream request failed: {body}");
    assert_nonstream(&body, &expected);
    recorder.latest(&expected)?;
    println!("PASS hag-serving http-nonstream prompt_tokens=9 completion_tokens=1");

    let response = post(&app, "/v1/chat/completions", chat(true, 1)).await?;
    ensure!(
        response.status() == StatusCode::OK
            && response.headers()[header::CONTENT_TYPE]
                .to_str()?
                .starts_with("text/event-stream"),
        "SSE headers differ"
    );
    let bytes = tokio::time::timeout(DEADLINE, response.into_body().collect())
        .await
        .context("SSE completion timed out")??
        .to_bytes();
    ensure!(bytes.len() < 64 * 1024, "unexpectedly large SSE fixture");
    let text = std::str::from_utf8(&bytes)?;
    ensure!(
        !text.contains("event: error"),
        "SSE returned an error: {text}"
    );
    let mut content = String::new();
    let mut terminal = 0;
    let mut done = 0;
    let mut role = 0;
    for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done += 1;
            continue;
        }
        let chunk: Value = serde_json::from_str(line)?;
        ensure!(
            chunk["object"] == "chat.completion.chunk" && chunk["model"] == MODEL_ID,
            "unexpected SSE envelope"
        );
        if chunk["choices"][0]["delta"]["role"] == "assistant" {
            role += 1;
        }
        if let Some(piece) = chunk["choices"][0]["delta"]["content"].as_str() {
            content.push_str(piece);
        }
        if !chunk["choices"][0]["finish_reason"].is_null() {
            ensure!(
                chunk["choices"][0]["finish_reason"] == finish(expected.finish_reason),
                "SSE finish differs"
            );
            terminal += 1;
        }
    }
    ensure!(
        content == expected.text && role == 1 && terminal == 1 && done == 1,
        "SSE text/role/termination differs"
    );
    recorder.latest(&expected)?;
    println!("PASS hag-serving http-sse text-and-token-parity=true done=1");

    let pause = Arc::new(EmissionPause::new());
    let release = ReleaseOnDrop(Arc::clone(&pause));
    *recorder
        .next_pause
        .lock()
        .map_err(|_| anyhow!("pause selection poisoned"))? = Some(Arc::clone(&pause));
    let finished_before = recorder.finished.load(Ordering::Acquire);
    let failures_before = recorder.failures.load(Ordering::Acquire);
    let response = post(&app, "/v1/chat/completions", chat(true, 2)).await?;
    ensure!(
        response.status() == StatusCode::OK,
        "cancellation stream did not open"
    );
    tokio::time::timeout(DEADLINE, pause.entered.notified())
        .await
        .context("native request did not reach a completed-step text boundary")?;
    let (status, busy) =
        json_body(post(&app, &format!("/v1/models/{MODEL_ID}/unload"), json!(null)).await?).await?;
    ensure!(
        status == StatusCode::CONFLICT && busy["error"]["code"] == "runtime_busy",
        "unload did not refuse while native generation held admission"
    );
    drop(response); // Drop the real SSE receiver before allowing the native sink to run.
    pause.release();
    drop(release);
    recorder.wait_finished(finished_before + 1).await?;
    ensure!(
        recorder.failures.load(Ordering::Acquire) == failures_before + 1,
        "SSE disconnect did not cancel the native runtime"
    );
    println!("PASS hag-serving sse-disconnect cancellation=true busy-unload=409");

    let (status, body) =
        json_body(post(&app, "/v1/chat/completions", chat(false, 1)).await?).await?;
    ensure!(
        status == StatusCode::OK,
        "request after cancellation failed: {body}"
    );
    assert_nonstream(&body, &expected);
    recorder.latest(&expected)?;
    println!("PASS hag-serving cancellation-reuse token-parity=true");

    for _ in 0..2 {
        let (status, body) =
            json_body(post(&app, &format!("/v1/models/{MODEL_ID}/unload"), json!(null)).await?)
                .await?;
        ensure!(
            status == StatusCode::OK && body["loaded"] == false,
            "idle unload was not idempotent"
        );
    }
    ensure!(!native.models()?[0].loaded, "unload left HAG published");
    let (status, body) =
        json_body(post(&app, "/v1/chat/completions", chat(false, 1)).await?).await?;
    ensure!(status == StatusCode::OK, "reload generation failed: {body}");
    assert_nonstream(&body, &expected);
    recorder.latest(&expected)?;
    assert_unloaded(&native).await?;
    let completed = recorder
        .completed
        .lock()
        .map_err(|_| anyhow!("recorder poisoned"))?;
    ensure!(
        completed.len() == 4 && recorder.failures.load(Ordering::Acquire) == 1,
        "unexpected completed/refused/cancelled request counts"
    );
    let templates = recorder
        .templates
        .lock()
        .map_err(|_| anyhow!("template recorder poisoned"))?;
    ensure!(
        templates.len() == 6
            && templates
                .iter()
                .all(|(model, template)| model == MODEL_ID && *template == expected_template),
        "each HTTP generation or capacity refusal must select the same checkpoint template"
    );
    println!("PASS hag-serving unload-reload token-parity=true final-loaded=false");
    println!(
        "HAG_SERVING_REPORT {}",
        json!({"execution":"native-hag-serving-http", "capacity":CAPACITY,
        "cpu_reference_requests":1, "hag_successful_requests":4, "hag_cancelled_requests":1,
        "capacity_refusals":1, "busy_unload_refusals":1, "max_evaluated_positions":46,
        "prompt_tokens":9, "completion_tokens":1, "greedy_token_ids":expected.token_ids, "nonstream_parity":true,
        "sse_parity":true, "disconnect_observed":true, "cancellation_reuse":true,
        "unload_reload":true, "unloaded":true, "early_cancellation_unloaded":true})
    );
    println!("PASS hag-serving stage=http qualified=true");
    Ok(())
}
