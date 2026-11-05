use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::response::Response;
use beholdd_runtime::{
    ChatTemplate, ExecutionLimits, FinishReason, GenerateRequest, Generation, InferenceRuntime,
    ModelCapabilities, ModelCompatibility, ModelConfig, ModelInfo, NativeRuntime, RequestError,
    RuntimeConfig, UnloadError,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio::sync::Notify;
use tower::ServiceExt;

use super::{router, DEFAULT_MAX_TOKENS, MAX_TOKENS};

#[path = "tests/openmythos.rs"]
mod openmythos;

struct FakeRuntime {
    model: ModelInfo,
    generation: Generation,
    chunks: Vec<String>,
    requests: Mutex<Vec<GenerateRequest>>,
    gate: Option<Arc<(Mutex<bool>, Condvar)>>,
}

impl FakeRuntime {
    fn new(text: &str, chunks: &[&str]) -> Self {
        Self {
            model: ModelInfo {
                id: "qwen-test".to_owned(),
                loaded: true,
                compatibility: ModelCompatibility::Compatible(ModelCapabilities::default()),
            },
            generation: Generation {
                text: text.to_owned(),
                token_ids: vec![1, 2],
                prompt_tokens: 7,
                completion_tokens: 2,
                finish_reason: FinishReason::Stop,
            },
            chunks: chunks.iter().map(|chunk| (*chunk).to_owned()).collect(),
            requests: Mutex::new(Vec::new()),
            gate: None,
        }
    }

    fn gated(text: &str, gate: Arc<(Mutex<bool>, Condvar)>) -> Self {
        let mut runtime = Self::new(text, &[text]);
        runtime.gate = Some(gate);
        runtime
    }
}

impl InferenceRuntime for FakeRuntime {
    fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(vec![self.model.clone()])
    }

    fn generate(
        &self,
        request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.requests.lock().unwrap().push(request);
        if let Some(gate) = &self.gate {
            let (lock, ready) = &**gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = ready.wait(open).unwrap();
            }
        }
        for chunk in &self.chunks {
            on_text(chunk)?;
        }
        Ok(self.generation.clone())
    }
}

struct CancellationRuntime {
    entered: Arc<Notify>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    cancelled: Arc<AtomicBool>,
    control_polls: Arc<AtomicUsize>,
    text_emitted: Arc<AtomicBool>,
    finished: Arc<Notify>,
}

struct PlanningRuntime {
    inner: FakeRuntime,
    plans: Mutex<Vec<(Option<usize>, usize)>>,
    entered: Arc<Notify>,
    gate: Option<Arc<(Mutex<bool>, Condvar)>>,
}

struct TemplateRuntime {
    inner: FakeRuntime,
    template: ChatTemplate,
    events: Mutex<Vec<&'static str>>,
    entered: Arc<Notify>,
    gate: Option<Arc<(Mutex<bool>, Condvar)>>,
}

impl TemplateRuntime {
    fn new(template: ChatTemplate) -> Self {
        Self {
            inner: FakeRuntime::new("rendered", &["rendered"]),
            template,
            events: Mutex::new(Vec::new()),
            entered: Arc::new(Notify::new()),
            gate: None,
        }
    }
}

impl InferenceRuntime for TemplateRuntime {
    fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        self.inner.models()
    }

    fn execution_limits(&self, _model: &str) -> anyhow::Result<ExecutionLimits> {
        Ok(ExecutionLimits {
            max_evaluated_tokens: Some(2048),
        })
    }

    fn chat_template_controlled(
        &self,
        model: &str,
        control: &mut (dyn FnMut() -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<ChatTemplate> {
        assert_eq!(model, "qwen-test");
        control()?;
        self.events.lock().unwrap().push("template");
        self.entered.notify_one();
        if let Some(gate) = &self.gate {
            let (lock, ready) = &**gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = ready.wait(open).unwrap();
            }
        }
        control()?;
        Ok(self.template.clone())
    }

    fn plan_request(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
    ) -> anyhow::Result<GenerateRequest> {
        self.events.lock().unwrap().push("plan");
        Ok(GenerateRequest {
            model: model.into(),
            prompt: prompt.into(),
            max_new_tokens: max_new_tokens.unwrap_or(default_max_new_tokens),
        })
    }

    fn generate(
        &self,
        request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.events.lock().unwrap().push("generate");
        self.inner.generate(request, on_text)
    }
}

impl PlanningRuntime {
    fn new() -> Self {
        Self {
            inner: FakeRuntime::new("planned", &["planned"]),
            plans: Mutex::new(Vec::new()),
            entered: Arc::new(Notify::new()),
            gate: None,
        }
    }
}

impl InferenceRuntime for PlanningRuntime {
    fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        self.inner.models()
    }

    fn execution_limits(&self, _model: &str) -> anyhow::Result<ExecutionLimits> {
        Ok(ExecutionLimits {
            max_evaluated_tokens: Some(16),
        })
    }

    fn plan_request(
        &self,
        model: &str,
        prompt: &str,
        max_new_tokens: Option<usize>,
        default_max_new_tokens: usize,
    ) -> anyhow::Result<GenerateRequest> {
        self.plans
            .lock()
            .unwrap()
            .push((max_new_tokens, default_max_new_tokens));
        self.entered.notify_one();
        if let Some(gate) = &self.gate {
            let (lock, ready) = &**gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = ready.wait(open).unwrap();
            }
        }
        let limit = max_new_tokens.unwrap_or(default_max_new_tokens.min(8));
        if limit > 8 {
            return Err(RequestError::ContextCapacity {
                prompt_tokens: 9,
                max_new_tokens: limit,
                capacity: 16,
            }
            .into());
        }
        Ok(GenerateRequest {
            model: model.into(),
            prompt: prompt.into(),
            max_new_tokens: limit,
        })
    }

    fn generate(
        &self,
        request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.inner.generate(request, on_text)
    }
}

impl CancellationRuntime {
    fn run(
        &self,
        control: &mut (dyn FnMut() -> anyhow::Result<()> + Send),
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.entered.notify_one();
        let (lock, ready) = &*self.gate;
        let mut open = lock.lock().unwrap();
        while !*open {
            open = ready.wait(open).unwrap();
        }
        drop(open);

        // Keep inference in a synthetic prefill phase so tests can disconnect either
        // before the first control boundary or after several successful boundaries.
        for _ in 0..200 {
            self.control_polls.fetch_add(1, Ordering::AcqRel);
            if let Err(error) = control() {
                self.cancelled.store(true, Ordering::Release);
                self.finished.notify_one();
                return Err(error);
            }
            thread::sleep(Duration::from_millis(5));
        }
        let mut callback = Ok(());
        for _ in 0..200 {
            self.text_emitted.store(true, Ordering::Release);
            callback = on_text("late token");
            if callback.is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        self.cancelled.store(callback.is_err(), Ordering::Release);
        self.finished.notify_one();
        callback?;
        Ok(Generation {
            text: "late token".to_owned(),
            token_ids: vec![1],
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }
}

impl InferenceRuntime for CancellationRuntime {
    fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(vec![ModelInfo {
            id: "qwen-test".to_owned(),
            loaded: true,
            compatibility: ModelCompatibility::Compatible(ModelCapabilities::default()),
        }])
    }

    fn generate(
        &self,
        _request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.run(&mut || Ok(()), on_text)
    }

    fn generate_controlled(
        &self,
        _request: GenerateRequest,
        control: &mut (dyn FnMut() -> anyhow::Result<()> + Send),
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.run(control, on_text)
    }
}

struct PanicRuntime;

impl InferenceRuntime for PanicRuntime {
    fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(vec![ModelInfo {
            id: "qwen-test".to_owned(),
            loaded: true,
            compatibility: ModelCompatibility::Compatible(ModelCapabilities::default()),
        }])
    }

    fn generate(
        &self,
        _request: GenerateRequest,
        _on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        panic!("synthetic inference panic")
    }
}

struct AdmissionRuntime {
    entered: Arc<Notify>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    calls: AtomicUsize,
    unload_calls: AtomicUsize,
}

impl InferenceRuntime for AdmissionRuntime {
    fn unload(&self, _model: &str) -> Result<(), UnloadError> {
        self.unload_calls.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(vec![ModelInfo {
            id: "qwen-test".to_owned(),
            loaded: false,
            compatibility: ModelCompatibility::Compatible(ModelCapabilities::default()),
        }])
    }

    fn generate(
        &self,
        _request: GenerateRequest,
        on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<Generation> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.entered.notify_one();
        let (lock, ready) = &*self.gate;
        let mut open = lock.lock().unwrap();
        while !*open {
            open = ready.wait(open).unwrap();
        }
        drop(open);
        on_text("ok")?;
        Ok(Generation {
            text: "ok".to_owned(),
            token_ids: vec![1],
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }
}

fn admission_runtime() -> (Arc<AdmissionRuntime>, Arc<(Mutex<bool>, Condvar)>) {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    (
        Arc::new(AdmissionRuntime {
            entered: Arc::new(Notify::new()),
            gate: Arc::clone(&gate),
            calls: AtomicUsize::new(0),
            unload_calls: AtomicUsize::new(0),
        }),
        gate,
    )
}

fn open_gate(gate: &Arc<(Mutex<bool>, Condvar)>) {
    let (lock, ready) = &**gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
}

struct OpenGateOnDrop(Arc<(Mutex<bool>, Condvar)>);

impl Drop for OpenGateOnDrop {
    fn drop(&mut self) {
        open_gate(&self.0);
    }
}

fn request(method: Method, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn completion_request(extra: Value) -> Value {
    let mut base = json!({
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    base.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    base
}

fn write_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }
        }
    })
}

async fn body_json(response: Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_text(response: Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn health_and_models_have_stable_openai_shapes() {
    let runtime = Arc::new(FakeRuntime::new("ok", &["ok"]));
    let app = router(runtime);

    let health_response = app
        .clone()
        .oneshot(request(Method::GET, "/health", json!({})))
        .await
        .unwrap();
    assert_eq!(health_response.status(), StatusCode::OK);
    assert_eq!(body_json(health_response).await, json!({"status":"ok"}));

    let models_response = app
        .oneshot(request(Method::GET, "/v1/models", json!({})))
        .await
        .unwrap();
    assert_eq!(models_response.status(), StatusCode::OK);
    let models = body_json(models_response).await;
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"][0]["id"], "qwen-test");
    assert_eq!(models["data"][0]["loaded"], true);
    assert_eq!(models["data"][0]["compatibility"], "compatible");
    assert_eq!(models["data"][0]["capabilities"], json!(["completion"]));
    assert!(models["data"][0].get("path").is_none());
}

#[tokio::test]
async fn native_info_is_metadata_free_and_does_not_claim_checkpoint_compatibility() {
    let runtime = Arc::new(
        NativeRuntime::new(RuntimeConfig {
            models: vec![ModelConfig::new("owner/model", "/does/not/exist-info.gguf")],
            ..RuntimeConfig::default()
        })
        .unwrap(),
    );
    let response = router(runtime.clone())
        .oneshot(request(Method::GET, "/v1/beholdd/info", json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_json(response).await,
        json!({
            "schema_version": 1,
            "service": "beholdd",
            "generation": 2,
            "runtime": "native",
            "execution_backends": ["cpu"],
            "architectures": ["qwen3", "qwen3moe"],
            "model_formats": ["gguf"],
            "operations": {"unload": true},
        })
    );
    assert_eq!(
        runtime.models().unwrap()[0].compatibility,
        ModelCompatibility::Unverified
    );
    assert!(!runtime.models().unwrap()[0].loaded);
}

#[tokio::test]
async fn alternate_runtime_defaults_to_unknown_identity_and_unsupported_unload() {
    let runtime = Arc::new(FakeRuntime::new("must not run", &[]));
    let app = router(runtime.clone());
    let response = app
        .clone()
        .oneshot(request(Method::GET, "/v1/beholdd/info", json!({})))
        .await
        .unwrap();
    assert_eq!(
        body_json(response).await,
        json!({
            "schema_version": 1,
            "service": "beholdd",
            "generation": 2,
            "runtime": "unknown",
            "execution_backends": [],
            "architectures": [],
            "model_formats": [],
            "operations": {"unload": false},
        })
    );
    let response = app
        .oneshot(request(
            Method::POST,
            "/v1/models/qwen-test/unload",
            json!({}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let error = body_json(response).await;
    assert_eq!(error["error"]["code"], "unsupported_operation");
    assert_eq!(error["error"]["type"], "server_error");
    assert!(runtime.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn native_unload_is_idempotent_and_decodes_model_ids_exactly_once() {
    let ids = ["owner/model", "name with spaces", "模型", "literal%2Fname"];
    let paths = [
        "owner%2Fmodel",
        "name%20with%20spaces",
        "%E6%A8%A1%E5%9E%8B",
        "literal%252Fname",
    ];
    let runtime = Arc::new(
        NativeRuntime::new(RuntimeConfig {
            models: ids
                .iter()
                .map(|id| ModelConfig::new(*id, "/does/not/exist-unload.gguf"))
                .collect(),
            ..RuntimeConfig::default()
        })
        .unwrap(),
    );
    let app = router(runtime.clone());
    for (id, encoded) in ids.iter().zip(paths) {
        for _ in 0..2 {
            let response = app
                .clone()
                .oneshot(request(
                    Method::POST,
                    &format!("/v1/models/{encoded}/unload"),
                    json!({}),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                body_json(response).await,
                json!({"id": id, "loaded": false})
            );
        }
    }
    assert!(runtime.models().unwrap().iter().all(|model| !model.loaded));
    let response = app
        .oneshot(request(Method::POST, "/v1/models/absent/unload", json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let error = body_json(response).await;
    assert_eq!(error["error"]["code"], "model_not_found");
    assert_eq!(error["error"]["param"], "model");
}

#[tokio::test]
async fn unload_errors_from_runtime_keep_the_management_contract() {
    struct RefusingRuntime;

    impl InferenceRuntime for RefusingRuntime {
        fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
            panic!("unload must not infer support from model listing")
        }

        fn generate(
            &self,
            _request: GenerateRequest,
            _on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
        ) -> anyhow::Result<Generation> {
            panic!("management must not generate")
        }

        fn unload(&self, model: &str) -> Result<(), UnloadError> {
            match model {
                "busy" => Err(UnloadError::Busy),
                "failed" => Err(UnloadError::Failed("synthetic lock failure".into())),
                _ => panic!("invalid URL must be refused before runtime access"),
            }
        }
    }

    let app = router(Arc::new(RefusingRuntime));
    for (id, status, code) in [
        ("busy", StatusCode::CONFLICT, "runtime_busy"),
        ("failed", StatusCode::INTERNAL_SERVER_ERROR, "unload_error"),
        ("%FF", StatusCode::BAD_REQUEST, "invalid_model"),
    ] {
        let response = app
            .clone()
            .oneshot(request(
                Method::POST,
                &format!("/v1/models/{id}/unload"),
                json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        assert_eq!(body_json(response).await["error"]["code"], code);
    }
}

#[tokio::test]
async fn active_inference_refuses_unload_without_queueing_it() {
    let (runtime, gate) = admission_runtime();
    let _release_on_failure = OpenGateOnDrop(Arc::clone(&gate));
    let app = router(runtime.clone());
    let generation = tokio::spawn(app.clone().oneshot(request(
        Method::POST,
        "/v1/chat/completions",
        completion_request(json!({})),
    )));
    tokio::time::timeout(Duration::from_secs(2), runtime.entered.notified())
        .await
        .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        app.clone().oneshot(request(
            Method::POST,
            "/v1/models/qwen-test/unload",
            json!({}),
        )),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(body_json(response).await["error"]["code"], "runtime_busy");
    assert_eq!(runtime.unload_calls.load(Ordering::Acquire), 0);
    open_gate(&gate);
    assert_eq!(generation.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(runtime.unload_calls.load(Ordering::Acquire), 0);
    let response = app
        .oneshot(request(
            Method::POST,
            "/v1/models/qwen-test/unload",
            json!({}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(runtime.unload_calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn blocking_unload_keeps_admission_after_http_disconnect_and_leaves_reads_responsive() {
    struct CleanupRuntime {
        entered: Notify,
        gate: Arc<(Mutex<bool>, Condvar)>,
        inference: AdmissionRuntime,
    }

    impl InferenceRuntime for CleanupRuntime {
        fn models(&self) -> anyhow::Result<Vec<ModelInfo>> {
            self.inference.models()
        }

        fn generate(
            &self,
            request: GenerateRequest,
            on_text: &mut (dyn FnMut(&str) -> anyhow::Result<()> + Send),
        ) -> anyhow::Result<Generation> {
            self.inference.generate(request, on_text)
        }

        fn unload(&self, _model: &str) -> Result<(), UnloadError> {
            self.entered.notify_one();
            let (lock, ready) = &*self.gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = ready.wait(open).unwrap();
            }
            Ok(())
        }
    }

    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let _release_on_failure = OpenGateOnDrop(Arc::clone(&gate));
    let runtime = Arc::new(CleanupRuntime {
        entered: Notify::new(),
        gate: Arc::clone(&gate),
        inference: AdmissionRuntime {
            entered: Arc::new(Notify::new()),
            gate: Arc::new((Mutex::new(true), Condvar::new())),
            calls: AtomicUsize::new(0),
            unload_calls: AtomicUsize::new(0),
        },
    });
    let app = router(runtime.clone());
    let unloading = tokio::spawn(app.clone().oneshot(request(
        Method::POST,
        "/v1/models/qwen-test/unload",
        json!({}),
    )));
    tokio::time::timeout(Duration::from_secs(2), runtime.entered.notified())
        .await
        .unwrap();
    unloading.abort();
    assert!(unloading.await.unwrap_err().is_cancelled());

    // The test uses a single-threaded async executor. These reads can complete only if
    // synchronous worker cleanup is not occupying that thread.
    for path in ["/health", "/v1/beholdd/info", "/v1/models"] {
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(request(Method::GET, path, json!({}))),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let response = app
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/models/qwen-test/unload",
            json!({}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);

    let generation = tokio::spawn(app.oneshot(request(
        Method::POST,
        "/v1/chat/completions",
        completion_request(json!({})),
    )));
    assert!(tokio::time::timeout(
        Duration::from_millis(25),
        runtime.inference.entered.notified()
    )
    .await
    .is_err());
    assert_eq!(runtime.inference.calls.load(Ordering::Acquire), 0);
    open_gate(&gate);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), generation)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(runtime.inference.calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn unverified_models_do_not_claim_executable_capabilities() {
    let mut runtime = FakeRuntime::new("unused", &["unused"]);
    runtime.model.loaded = false;
    runtime.model.compatibility = ModelCompatibility::Unverified;
    let response = router(Arc::new(runtime))
        .oneshot(request(Method::GET, "/v1/models", json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let model = &body_json(response).await["data"][0];
    assert_eq!(model["loaded"], false);
    assert_eq!(model["compatibility"], "unverified");
    assert_eq!(model["capabilities"], json!([]));
}

#[tokio::test]
async fn model_names_never_infer_an_unimplemented_vision_capability() {
    let mut runtime = FakeRuntime::new("ok", &["ok"]);
    runtime.model.id = "qwen3.6-vision".to_owned();
    let response = router(Arc::new(runtime))
        .oneshot(request(Method::GET, "/v1/models", json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_json(response).await["data"][0]["capabilities"],
        json!(["completion"])
    );
}

#[tokio::test]
async fn nonstream_completion_has_usage_and_renders_qwen_prompt() {
    let runtime = Arc::new(FakeRuntime::new("Hello back", &["Hello ", "back"]));
    let app = router(runtime.clone());
    let response = app
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello back");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 9);

    let requests = runtime.requests.lock().unwrap();
    assert_eq!(requests[0].max_new_tokens, DEFAULT_MAX_TOKENS);
    assert_eq!(
        requests[0].prompt,
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn model_listing_publishes_execution_limits_without_planning_or_generation() {
    let runtime = Arc::new(PlanningRuntime::new());
    let response = router(runtime.clone())
        .oneshot(request(Method::GET, "/v1/models", json!(null)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let listing = body_json(response).await;
    assert_eq!(
        listing["data"][0]["execution_limits"],
        json!({"max_evaluated_tokens":16})
    );
    assert!(runtime.plans.lock().unwrap().is_empty());
    assert!(runtime.inner.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn both_response_modes_preserve_omission_for_planning_and_explicit_output_limits() {
    for stream in [false, true] {
        for explicit in [None, Some(3)] {
            let runtime = Arc::new(PlanningRuntime::new());
            let mut fields = json!({"stream": stream});
            if let Some(limit) = explicit {
                fields["max_tokens"] = json!(limit);
            }
            let response = router(runtime.clone())
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(fields),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let text = body_text(response).await;
            assert!(text.contains("planned"));
            assert_eq!(
                *runtime.plans.lock().unwrap(),
                vec![(explicit, DEFAULT_MAX_TOKENS)]
            );
            let requests = runtime.inner.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].max_new_tokens, explicit.unwrap_or(8));
            assert!(requests[0].prompt.contains("<|im_start|>"));
        }
    }
}

#[tokio::test]
async fn template_selection_precedes_planning_and_normalizes_only_checkpoint_history() {
    for stream in [false, true] {
        for template in [
            ChatTemplate::Legacy,
            ChatTemplate::Checkpoint(Some(super::protocol::QWEN3_CHAT_TEMPLATE.into())),
        ] {
            let checkpoint = matches!(&template, ChatTemplate::Checkpoint(_));
            let runtime = Arc::new(TemplateRuntime::new(template));
            let response = router(runtime.clone())
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(json!({
                        "stream":stream,
                        "messages":[{"role":"user","content":"First"},
                            {"role":"assistant","content":"<think>\n\n</think>\n\nready."},
                            {"role":"user","content":"Again /no_think"}]
                    })),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(body_text(response).await.contains("rendered"));
            assert_eq!(
                *runtime.events.lock().unwrap(),
                ["template", "plan", "generate"]
            );
            let generated = runtime.inner.requests.lock().unwrap();
            assert_eq!(generated.len(), 1);
            let assistant = if checkpoint {
                "ready."
            } else {
                "<think>\n\n</think>\n\nready."
            };
            assert_eq!(generated[0].prompt, format!(
                "<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n<|im_start|>user\nAgain /no_think<|im_end|>\n<|im_start|>assistant\n"));
        }
    }
}

#[tokio::test]
async fn missing_or_unknown_checkpoint_templates_refuse_before_planning_or_headers() {
    for stream in [false, true] {
        for template in [
            None,
            Some("unknown".into()),
            Some(format!("{}\n", super::protocol::QWEN3_CHAT_TEMPLATE)),
        ] {
            let runtime = Arc::new(TemplateRuntime::new(ChatTemplate::Checkpoint(template)));
            let response = router(runtime.clone())
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(json!({"stream":stream})),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json"));
            let error = body_json(response).await;
            assert_eq!(error["error"]["code"], "unsupported_chat_template");
            assert_eq!(error["error"]["param"], "model");
            assert_eq!(*runtime.events.lock().unwrap(), ["template"]);
            assert!(runtime.inner.requests.lock().unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn qwen3_invalid_tools_and_history_release_admission_for_a_valid_tool_request() {
    for stream in [false, true] {
        let runtime = Arc::new(TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
            super::protocol::QWEN3_CHAT_TEMPLATE.into(),
        ))));
        let app = router(runtime.clone());
        for (extra, param) in [
            (
                json!({"stream":stream,"tools":[write_tool(), write_tool()]}),
                "tools",
            ),
            (
                json!({"stream":stream,"tools":[write_tool()],"messages":[
                    {"role":"assistant","content":null,"tool_calls":[{
                        "id":"pending","type":"function","function":{
                            "name":"write_file","arguments":"{\"path\":\"x\",\"content\":\"y\"}"
                        }
                    }]},
                    {"role":"user","content":"Continue without a result"}
                ]}),
                "messages",
            ),
            (
                json!({"stream":stream,"messages":[{"role":"tool","content":"r","tool_call_id":"id"}]}),
                "messages",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(extra),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let error = body_json(response).await;
            assert_eq!(error["error"]["type"], "invalid_request_error");
            assert_eq!(error["error"]["param"], param);
        }
        assert_eq!(
            *runtime.events.lock().unwrap(),
            ["template", "template", "template"]
        );
        assert!(runtime.inner.requests.lock().unwrap().is_empty());
        let response = app
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({
                    "stream":stream,
                    "tools":[write_tool()],
                    "messages":[
                        {"role":"user","content":"Write the file"},
                        {"role":"assistant","content":null,"tool_calls":[{
                            "id":"completed","type":"function","function":{
                                "name":"write_file","arguments":"{\"path\":\"x\",\"content\":\"y\"}"
                            }
                        }]},
                        {"role":"tool","tool_call_id":"completed","name":"write_file","content":"Written"},
                        {"role":"user","content":"Summarize the result"}
                    ]
                })),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        body_text(response).await;
        let generated = runtime.inner.requests.lock().unwrap();
        assert_eq!(generated.len(), 1);
        assert!(generated[0].prompt.contains("# Tools\n"));
        assert!(generated[0].prompt.contains(
            "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\":\"x\",\"content\":\"y\"}}\n</tool_call>"
        ));
        assert!(generated[0]
            .prompt
            .contains("<|im_start|>user\n<tool_response>\nWritten\n</tool_response><|im_end|>\n"));
        assert_eq!(
            *runtime.events.lock().unwrap(),
            ["template", "template", "template", "template", "plan", "generate"]
        );
    }
}

#[tokio::test]
async fn thinking_options_remain_unsupported_for_qwen3_before_planning() {
    let runtime = Arc::new(TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
        super::protocol::QWEN3_CHAT_TEMPLATE.into(),
    ))));
    for extra in [
        json!({"enable_thinking":false}),
        json!({"messages":[{"role":"assistant","content":"a","reasoning_content":"r"}]}),
    ] {
        let response = router(runtime.clone())
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(extra),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        body_text(response).await;
    }
    assert_eq!(*runtime.events.lock().unwrap(), ["template", "template"]);
}

#[tokio::test]
async fn impossible_explicit_limit_still_refuses_before_template_metadata() {
    let runtime = Arc::new(TemplateRuntime::new(ChatTemplate::Checkpoint(None)));
    for stream in [false, true] {
        let response = router(runtime.clone())
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({"stream":stream,"max_tokens":2049})),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            body_json(response).await["error"]["code"],
            "context_length_exceeded"
        );
    }
    assert!(runtime.events.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_template_lookup_retains_admission_and_never_plans_or_generates() {
    for stream in [false, true] {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let _release = OpenGateOnDrop(Arc::clone(&gate));
        let mut runtime = TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
            super::protocol::QWEN3_CHAT_TEMPLATE.into(),
        )));
        runtime.gate = Some(Arc::clone(&gate));
        let runtime = Arc::new(runtime);
        let app = router(runtime.clone());
        let first = tokio::spawn(app.clone().oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":stream})),
        )));
        tokio::time::timeout(Duration::from_secs(1), runtime.entered.notified())
            .await
            .expect("template lookup did not enter");
        assert!(
            !first.is_finished(),
            "headers escaped before template lookup"
        );
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        let unload = app
            .clone()
            .oneshot(request(
                Method::POST,
                "/v1/models/qwen-test/unload",
                json!(null),
            ))
            .await
            .unwrap();
        assert_eq!(unload.status(), StatusCode::CONFLICT);
        let second = tokio::spawn(app.oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":stream})),
        )));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!second.is_finished());
        assert_eq!(*runtime.events.lock().unwrap(), ["template"]);
        open_gate(&gate);
        let response = tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .expect("replacement request did not retire")
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        body_text(response).await;
        assert_eq!(
            *runtime.events.lock().unwrap(),
            ["template", "template", "plan", "generate"]
        );
        assert_eq!(runtime.inner.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn explicit_capacity_errors_are_http_400_before_headers_cold_and_warm() {
    for stream in [false, true] {
        for warm in [false, true] {
            let runtime = Arc::new(PlanningRuntime::new());
            let app = router(runtime.clone());
            if warm {
                let response = app
                    .clone()
                    .oneshot(request(
                        Method::POST,
                        "/v1/chat/completions",
                        completion_request(json!({"max_tokens":1})),
                    ))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                body_text(response).await;
            }
            let response = app
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(json!({"stream":stream, "max_tokens":9})),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json"));
            let error = body_json(response).await;
            assert_eq!(error["error"]["type"], "invalid_request_error");
            assert_eq!(error["error"]["code"], "context_length_exceeded");
            assert_eq!(error["error"]["param"], "max_tokens");
            assert!(error.get("choices").is_none());
            assert_eq!(
                runtime.inner.requests.lock().unwrap().len(),
                usize::from(warm)
            );
        }
    }
}

#[test]
fn typed_request_errors_preserve_client_status_through_anyhow_context() {
    for (error, parameter, code) in [
        (RequestError::EmptyPrompt, "messages", "invalid_value"),
        (
            RequestError::InvalidOutputLimit,
            "max_tokens",
            "invalid_value",
        ),
        (
            RequestError::ContextCapacity {
                prompt_tokens: 9,
                max_new_tokens: 9,
                capacity: 16,
            },
            "max_tokens",
            "context_length_exceeded",
        ),
    ] {
        let response =
            super::ApiError::runtime(anyhow::Error::new(error).context("planning metadata"));
        assert_eq!(response.status, StatusCode::BAD_REQUEST);
        assert_eq!(response.body.param, Some(parameter));
        assert_eq!(response.body.code, code);
    }
    assert_eq!(
        super::ApiError::runtime(anyhow::anyhow!("storage failure")).status,
        StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_planning_retains_admission_until_worker_exits_and_never_generates() {
    for stream in [false, true] {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let _release = OpenGateOnDrop(Arc::clone(&gate));
        let mut runtime = PlanningRuntime::new();
        runtime.gate = Some(Arc::clone(&gate));
        let runtime = Arc::new(runtime);
        let app = router(runtime.clone());
        let first = tokio::spawn(app.clone().oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":stream})),
        )));
        tokio::time::timeout(Duration::from_secs(1), runtime.entered.notified())
            .await
            .expect("planning did not enter the worker");
        assert!(
            !first.is_finished(),
            "headers escaped before planning finished"
        );
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());

        let management = app
            .clone()
            .oneshot(request(
                Method::POST,
                "/v1/models/qwen-test/unload",
                json!(null),
            ))
            .await
            .unwrap();
        assert_eq!(management.status(), StatusCode::CONFLICT);
        let listing = app
            .clone()
            .oneshot(request(Method::GET, "/v1/models", json!(null)))
            .await
            .unwrap();
        assert_eq!(listing.status(), StatusCode::OK);
        let second = tokio::spawn(app.oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":stream})),
        )));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!second.is_finished());
        assert_eq!(
            runtime.plans.lock().unwrap().len(),
            1,
            "cancelled planning released admission early"
        );
        assert!(runtime.inner.requests.lock().unwrap().is_empty());
        open_gate(&gate);
        let response = tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .expect("replacement request did not finish planning")
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(body_text(response).await.contains("planned"));
        assert_eq!(runtime.plans.lock().unwrap().len(), 2);
        assert_eq!(
            runtime.inner.requests.lock().unwrap().len(),
            1,
            "cancelled plan entered generation"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graceful_shutdown_waits_for_disconnected_blocking_planning_to_retire() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let _release = OpenGateOnDrop(Arc::clone(&gate));
    let mut runtime = PlanningRuntime::new();
    runtime.gate = Some(Arc::clone(&gate));
    let runtime = Arc::new(runtime);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let mut server = tokio::spawn(super::serve_with_shutdown(
        listener,
        runtime.clone(),
        async {
            let _ = stopped.await;
        },
    ));
    let body = completion_request(json!({})).to_string();
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut connection = TcpStream::connect(address).unwrap();
    connection.write_all(wire.as_bytes()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), runtime.entered.notified())
        .await
        .expect("planning did not enter the blocking worker");
    drop(connection);
    tokio::time::sleep(Duration::from_millis(20)).await;
    stop.send(()).unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut server)
            .await
            .is_err(),
        "server reported clean shutdown while detached planning still owned admission"
    );
    assert!(runtime.inner.requests.lock().unwrap().is_empty());
    open_gate(&gate);
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("server did not finish after planning retired")
        .unwrap()
        .unwrap();
    assert!(
        runtime.inner.requests.lock().unwrap().is_empty(),
        "disconnected planning entered generation during shutdown"
    );
}

#[tokio::test]
async fn text_content_parts_are_normalized_before_prompt_rendering() {
    let runtime = Arc::new(FakeRuntime::new("ok", &["ok"]));
    let app = router(runtime.clone());
    let response = app
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"messages":[{
                "role":"user",
                "content":[
                    {"type":"text", "text":"Hello "},
                    {"type":"text", "text":"from parts"}
                ]
            }]})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        runtime.requests.lock().unwrap()[0].prompt,
        "<|im_start|>user\nHello from parts<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn image_content_is_rejected_as_unsupported_before_generation() {
    let runtime = Arc::new(FakeRuntime::new("unused", &["unused"]));
    let response = router(runtime.clone())
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"messages":[{
                "role":"user",
                "content":[
                    {"type":"text", "text":"Describe this."},
                    {"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}
                ]
            }]})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "messages");
    assert_eq!(body["error"]["code"], "unsupported_content_type");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("model \"qwen-test\" does not support image input"));
    assert!(runtime.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn dropping_nonstream_request_cancels_at_a_progress_boundary_without_text() {
    let entered = Arc::new(Notify::new());
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let cancelled = Arc::new(AtomicBool::new(false));
    let control_polls = Arc::new(AtomicUsize::new(0));
    let text_emitted = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(Notify::new());
    let runtime = Arc::new(CancellationRuntime {
        entered: entered.clone(),
        gate: gate.clone(),
        cancelled: cancelled.clone(),
        control_polls,
        text_emitted: text_emitted.clone(),
        finished: finished.clone(),
    });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(super::serve(listener, runtime));
    let body = completion_request(json!({})).to_string();
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut connection = TcpStream::connect(address).unwrap();
    connection.write_all(wire.as_bytes()).unwrap();

    tokio::time::timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("generation did not start");
    drop(connection);
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    tokio::time::timeout(Duration::from_secs(2), finished.notified())
        .await
        .expect("cancelled generation did not stop");
    server.abort();
    let _ = server.await;
    assert!(cancelled.load(Ordering::Acquire));
    assert!(
        !text_emitted.load(Ordering::Acquire),
        "serving control cancellation was converted into a text callback"
    );
}

#[tokio::test]
async fn delayed_nonstream_disconnect_cancels_after_successful_progress_boundaries() {
    let entered = Arc::new(Notify::new());
    let gate = Arc::new((Mutex::new(true), Condvar::new()));
    let cancelled = Arc::new(AtomicBool::new(false));
    let control_polls = Arc::new(AtomicUsize::new(0));
    let text_emitted = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(Notify::new());
    let runtime = Arc::new(CancellationRuntime {
        entered: entered.clone(),
        gate,
        cancelled: cancelled.clone(),
        control_polls: control_polls.clone(),
        text_emitted: text_emitted.clone(),
        finished: finished.clone(),
    });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(super::serve(listener, runtime));
    let body = completion_request(json!({})).to_string();
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut connection = TcpStream::connect(address).unwrap();
    connection.write_all(wire.as_bytes()).unwrap();

    tokio::time::timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("generation did not start");
    tokio::time::timeout(Duration::from_secs(1), async {
        while control_polls.load(Ordering::Acquire) < 10 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("generation did not cross repeated control boundaries");
    assert!(!cancelled.load(Ordering::Acquire));

    // This models a headers-timeout client: the request remains connected while native
    // work is active, then closes before the non-streaming response headers exist.
    drop(connection);
    tokio::time::timeout(Duration::from_secs(2), finished.notified())
        .await
        .expect("delayed disconnect did not stop generation");
    server.abort();
    let _ = server.await;

    assert!(cancelled.load(Ordering::Acquire));
    assert!(control_polls.load(Ordering::Acquire) >= 10);
    assert!(
        !text_emitted.load(Ordering::Acquire),
        "delayed serving-control cancellation was converted into a text callback"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_waits_asynchronously_before_starting_a_second_generation() {
    let (runtime, gate) = admission_runtime();
    let app = router(runtime.clone());
    let first = tokio::spawn(app.clone().oneshot(request(
        Method::POST,
        "/v1/chat/completions",
        completion_request(json!({})),
    )));
    tokio::time::timeout(Duration::from_secs(1), runtime.entered.notified())
        .await
        .expect("first generation did not enter the runtime");

    let second = tokio::spawn(app.oneshot(request(
        Method::POST,
        "/v1/chat/completions",
        completion_request(json!({})),
    )));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        runtime.calls.load(Ordering::Acquire),
        1,
        "a queued request entered the runtime before admission was released"
    );

    open_gate(&gate);
    let first = tokio::time::timeout(Duration::from_secs(1), first)
        .await
        .expect("first response did not finish")
        .unwrap()
        .unwrap();
    let second = tokio::time::timeout(Duration::from_secs(1), second)
        .await
        .expect("second response did not finish")
        .unwrap()
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    assert_eq!(runtime.calls.load(Ordering::Acquire), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnected_queued_sse_never_enters_native_inference() {
    let (runtime, gate) = admission_runtime();
    let _release = OpenGateOnDrop(Arc::clone(&gate));
    let app = router(runtime.clone());
    let first = app
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream": true})),
        ))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), runtime.entered.notified())
        .await
        .expect("first SSE generation did not enter the runtime");

    let queued = tokio::spawn(app.oneshot(request(
        Method::POST,
        "/v1/chat/completions",
        completion_request(json!({"stream": true})),
    )));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !queued.is_finished(),
        "queued stream returned headers before planning"
    );
    assert_eq!(runtime.calls.load(Ordering::Acquire), 1);
    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());

    open_gate(&gate);
    let body = tokio::time::timeout(Duration::from_secs(1), body_text(first))
        .await
        .expect("admitted SSE response did not finish");
    assert!(body.contains("data: [DONE]"));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        runtime.calls.load(Ordering::Acquire),
        1,
        "the disconnected queued stream started native inference"
    );
}

#[tokio::test]
async fn strict_request_errors_use_openai_error_envelopes() {
    let runtime = Arc::new(FakeRuntime::new("unused", &["unused"]));
    let app = router(runtime);
    for (body, status, code) in [
        (
            completion_request(json!({"temperature": 0.2})),
            StatusCode::BAD_REQUEST,
            "unsupported_value",
        ),
        (
            completion_request(json!({"max_tokens": MAX_TOKENS + 1})),
            StatusCode::BAD_REQUEST,
            "invalid_value",
        ),
        (
            completion_request(json!({"unknown_field": true})),
            StatusCode::BAD_REQUEST,
            "invalid_json",
        ),
        (
            json!({"model":"missing", "messages":[{"role":"user","content":"x"}]}),
            StatusCode::NOT_FOUND,
            "model_not_found",
        ),
        (
            json!({"model":"qwen-test", "messages":[{
                "role":"user", "content":[{"type":"text", "text":3}]
            }]}),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
    ] {
        let response = app
            .clone()
            .oneshot(request(Method::POST, "/v1/chat/completions", body))
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], code);
        assert!(body["error"]["message"].is_string());
    }
}

#[tokio::test]
async fn nonstream_tool_call_is_translated_and_malformed_xml_is_an_error() {
    let native = "<tool_call>\n<function=write_file>\n\
        <parameter=path>\nsrc/main.rs\n</parameter>\n\
        <parameter=content>\nfn main() {}\n</parameter>\n\
        <parameter=overwrite>\ntrue\n</parameter>\n\
        </function>\n</tool_call>";
    let runtime = Arc::new(FakeRuntime::new(native, &[native]));
    let app = router(runtime);
    let response = app
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"tools":[write_tool()]})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let call = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "write_file");
    let arguments: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(
        arguments,
        json!({"path":"src/main.rs", "content":"fn main() {}", "overwrite":true})
    );

    let malformed = Arc::new(FakeRuntime::new(
        "<tool_call><function=write_file>",
        &["<tool_call><function=write_file>"],
    ));
    let response = router(malformed)
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"tools":[write_tool()]})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = body_json(response).await;
    assert_eq!(body["error"]["code"], "tool_call_parse_error");
    assert!(body.get("choices").is_none());
}

#[tokio::test]
async fn qwen3_json_tool_calls_use_the_planned_dialect_for_both_response_modes() {
    let native = "Working.\n<tool_call>\n\
        {\"name\":\"write_file\",\"arguments\":{\"path\":\"src/main.rs\",\"content\":\"fn main() {}\",\"overwrite\":true}}\n\
        </tool_call>";
    for stream in [false, true] {
        let mut runtime = TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
            super::protocol::QWEN3_CHAT_TEMPLATE.into(),
        )));
        runtime.inner = FakeRuntime::new(native, &["Working.\n<tool_", "call>", native]);
        let runtime = Arc::new(runtime);
        let response = router(runtime.clone())
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({"stream":stream,"tools":[write_tool()]})),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let call = if stream {
            let body = body_text(response).await;
            assert!(!body.contains("<tool_call>"));
            assert_terminal_usage(&body, "tool_calls");
            let chunks: Vec<Value> = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            let content: String = chunks
                .iter()
                .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
                .collect();
            assert_eq!(content, "Working.");
            let tool_deltas: Vec<_> = chunks
                .iter()
                .filter_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
                .collect();
            assert_eq!(tool_deltas.len(), 1);
            assert_eq!(tool_deltas[0].len(), 1);
            assert_eq!(tool_deltas[0][0]["index"], 0);
            tool_deltas[0][0].clone()
        } else {
            let body = body_json(response).await;
            assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(body["choices"][0]["message"]["content"], "Working.");
            assert_eq!(
                body["choices"][0]["message"]["tool_calls"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            body["choices"][0]["message"]["tool_calls"][0].clone()
        };
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "write_file");
        let arguments: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(
            arguments,
            json!({
                "path":"src/main.rs","content":"fn main() {}","overwrite":true
            })
        );
        assert_eq!(
            *runtime.events.lock().unwrap(),
            ["template", "plan", "generate"]
        );
    }
}

#[tokio::test]
async fn checkpoint_tool_dialects_reject_malformed_or_other_dialect_output() {
    let qwen_template = ChatTemplate::Checkpoint(Some(super::protocol::QWEN3_CHAT_TEMPLATE.into()));
    let json_call = "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"x\",\"content\":\"y\"}}</tool_call>";
    for (template, native) in [
        (qwen_template.clone(), "<tool_call>{invalid json}</tool_call>"),
        (qwen_template.clone(), "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>"),
        (qwen_template.clone(), "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":3,\"content\":\"y\"}}</tool_call>"),
        (qwen_template, "<tool_call><function=write_file><parameter=path>x</parameter><parameter=content>y</parameter></function></tool_call>"),
        (ChatTemplate::Checkpoint(Some(super::protocol::QWEN3_CODER_CHAT_TEMPLATE.into())), json_call),
        (ChatTemplate::Legacy, json_call),
    ] {
        for stream in [false, true] {
            let mut runtime = TemplateRuntime::new(template.clone());
            runtime.inner = FakeRuntime::new(native, &[native]);
            let runtime = Arc::new(runtime);
            let response = router(runtime.clone())
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(json!({"stream":stream,"tools":[write_tool()]})),
                ))
                .await
                .unwrap();
            if stream {
                assert_eq!(response.status(), StatusCode::OK);
                let body = body_text(response).await;
                assert!(body.contains("event: error"));
                assert!(body.contains("\"code\":\"tool_call_parse_error\""));
                assert!(!body.contains("\"tool_calls\""));
                assert!(!body.contains("data: [DONE]"));
                assert!(!body.contains("\"usage\""));
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                let body = body_json(response).await;
                assert_eq!(body["error"]["code"], "tool_call_parse_error");
                assert!(body.get("choices").is_none());
            }
            assert_eq!(
                *runtime.events.lock().unwrap(),
                ["template", "plan", "generate"]
            );
        }
    }
}

#[tokio::test]
async fn qwen3_tool_upper_bounds_are_enforced_before_chat_or_stream_delivery() {
    let tool = json!({"type":"function","function":{
        "name":"select","parameters":{"type":"object","properties":{
            "limit":{"type":"integer","maximum":2},
            "paths":{"type":"array","maxItems":2,"items":{"type":"string"}}
        },"required":["limit","paths"],"additionalProperties":false}
    }});
    for (arguments, valid) in [
        (json!({"limit":2,"paths":["a","b"]}), true),
        (json!({"limit":3,"paths":["a"]}), false),
        (json!({"limit":1,"paths":["a","b","c"]}), false),
    ] {
        for stream in [false, true] {
            let native =
                format!("<tool_call>{{\"name\":\"select\",\"arguments\":{arguments}}}</tool_call>");
            let mut runtime = TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
                super::protocol::QWEN3_CHAT_TEMPLATE.into(),
            )));
            runtime.inner = FakeRuntime::new(&native, &[&native]);
            let runtime = Arc::new(runtime);
            let response = router(runtime.clone())
                .oneshot(request(
                    Method::POST,
                    "/v1/chat/completions",
                    completion_request(json!({"stream":stream,"tools":[tool.clone()]})),
                ))
                .await
                .unwrap();
            if stream {
                assert_eq!(response.status(), StatusCode::OK);
                let body = body_text(response).await;
                if valid {
                    assert_terminal_usage(&body, "tool_calls");
                    assert!(body.contains("\"name\":\"select\""));
                    assert!(!body.contains("event: error"));
                } else {
                    assert!(body.contains("event: error"));
                    assert!(body.contains("\"code\":\"tool_call_parse_error\""));
                    assert!(!body.contains("\"tool_calls\""));
                    assert!(!body.contains("data: [DONE]"));
                    assert!(!body.contains("\"usage\""));
                }
            } else if valid {
                assert_eq!(response.status(), StatusCode::OK);
                let body = body_json(response).await;
                assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
                let parsed: Value = serde_json::from_str(
                    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                        .as_str()
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(parsed, arguments);
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                let body = body_json(response).await;
                assert_eq!(body["error"]["code"], "tool_call_parse_error");
                assert!(body.get("choices").is_none());
            }
            assert_eq!(
                *runtime.events.lock().unwrap(),
                ["template", "plan", "generate"]
            );
            let requests = runtime.inner.requests.lock().unwrap();
            assert!(requests[0].prompt.contains("\"maximum\": 2"));
            assert!(requests[0].prompt.contains("\"maxItems\": 2"));
        }
    }
}

#[tokio::test]
async fn qwen3_text_sse_checks_final_output_when_callback_chunks_are_plain() {
    let mut runtime = TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
        super::protocol::QWEN3_CHAT_TEMPLATE.into(),
    )));
    runtime.inner = FakeRuntime::new(
        "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"x\",\"content\":\"y\"}}</tool_call>",
        &["safe prefix"],
    );
    let runtime = Arc::new(runtime);
    let response = router(runtime.clone())
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":true})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response).await;
    assert!(body.contains("safe prefix"));
    assert!(body.contains("event: error"));
    assert!(body.contains("\"code\":\"tool_call_parse_error\""));
    assert!(!body.contains("data: [DONE]"));
    assert!(!body.contains("\"usage\""));
    assert_eq!(
        *runtime.events.lock().unwrap(),
        ["template", "plan", "generate"]
    );
}

#[tokio::test]
async fn text_sse_returns_headers_before_generation_and_ends_with_done() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let runtime = Arc::new(FakeRuntime::gated("héllo", gate.clone()));
    let app = router(runtime);
    let response_result = tokio::time::timeout(
        Duration::from_millis(100),
        app.oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":true})),
        )),
    )
    .await;
    if response_result.is_err() {
        let (lock, ready) = &*gate;
        *lock.lock().unwrap() = true;
        ready.notify_all();
        panic!("streaming handler waited for generation before returning headers");
    }
    let response = response_result.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let (lock, ready) = &*gate;
    *lock.lock().unwrap() = true;
    ready.notify_all();
    let body = body_text(response).await;
    assert!(body.contains("\"role\":\"assistant\""));
    assert!(body.contains("héllo"));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert!(body.contains("data: [DONE]"));
    assert_terminal_usage(&body, "stop");
}

fn assert_terminal_usage(body: &str, finish: &str) {
    let events: Vec<_> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|value| serde_json::from_str(value).unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .filter(|chunk| chunk.get("usage").is_some())
            .count(),
        1
    );
    let terminal = chunks.last().unwrap();
    assert_eq!(terminal["choices"][0]["delta"], json!({}));
    assert_eq!(terminal["choices"][0]["finish_reason"], finish);
    assert_eq!(
        terminal["usage"],
        json!({"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})
    );
}

#[tokio::test]
async fn tool_sse_buffers_native_xml_and_emits_openai_delta() {
    let native = "<tool_call>\n<function=write_file>\n\
        <parameter=path>\nx\n</parameter>\n\
        <parameter=content>\ny\n</parameter>\n\
        </function>\n</tool_call>";
    let runtime = Arc::new(FakeRuntime::new(native, &["<tool_", "call>", native]));
    let response = router(runtime)
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":true, "tools":[write_tool()]})),
        ))
        .await
        .unwrap();
    let body = body_text(response).await;
    assert!(!body.contains("<tool_call>"));
    assert!(body.contains("\"tool_calls\""));
    assert!(body.contains("\"name\":\"write_file\""));
    assert!(body.contains("\"finish_reason\":\"tool_calls\""));
    assert!(body.contains("data: [DONE]"));
    assert_terminal_usage(&body, "tool_calls");
}

#[tokio::test]
async fn streaming_worker_panic_emits_an_error_without_panicking_the_watcher() {
    let response = router(Arc::new(PanicRuntime))
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"stream":true})),
        ))
        .await
        .unwrap();
    let body = body_text(response).await;
    assert!(body.contains("event: error"));
    assert!(body.contains("\"error\""));
    assert!(!body.contains("data: [DONE]"));
    assert!(!body.contains("\"usage\""));
}
