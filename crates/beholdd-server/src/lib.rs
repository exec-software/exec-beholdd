//! OpenAI-compatible HTTP for the beholdd runtime.
//!
//! This crate owns HTTP and checkpoint-selected Qwen chat/tool wire translation.
//! Inference stays behind [`beholdd_runtime::InferenceRuntime`], which also makes
//! every route testable without loading a model.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod protocol;

#[cfg(test)]
mod tests;

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use beholdd_runtime::{
    FinishReason as RuntimeFinishReason, GenerateRequest, Generation, InferenceRuntime,
    ModelCapabilities, ModelCompatibility, RequestError, UnloadError,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use protocol::{
    normalize_message_content, prepare_chat, translate_output_for_dialect, ChatMessage,
    ChatRequest, OutputDialect, ProtocolError, ToolDefinition, TranslatedOutput,
    DEFAULT_MAX_TOKENS, MAX_TOKENS,
};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);
const STREAM_QUEUE: usize = 32;
const TOOL_MARKERS: &[&str] = &[
    "<tool_call",
    "</tool_call",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
];

#[derive(Clone)]
struct AppState {
    runtime: Arc<dyn InferenceRuntime>,
    admission: Arc<Semaphore>,
}

/// Build the OpenAI-compatible application around an inference runtime.
pub fn router(runtime: Arc<dyn InferenceRuntime>) -> Router {
    router_with_state(AppState {
        runtime,
        admission: Arc::new(Semaphore::new(1)),
    })
}

fn router_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/beholdd/info", get(runtime_info))
        .route("/v1/models", get(models))
        .route("/v1/models/:id/unload", post(unload_model))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

/// Serve the OpenAI-compatible application on an already-bound listener.
pub async fn serve(
    listener: TcpListener,
    runtime: Arc<dyn InferenceRuntime>,
) -> anyhow::Result<()> {
    serve_with_shutdown(listener, runtime, std::future::pending()).await
}

/// Serve until `shutdown` resolves, then finish admitted requests before returning.
pub async fn serve_with_shutdown(
    listener: TcpListener,
    runtime: Arc<dyn InferenceRuntime>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let admission = Arc::new(Semaphore::new(1));
    let app = router_with_state(AppState {
        runtime,
        admission: Arc::clone(&admission),
    });
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("beholdd HTTP server stopped unexpectedly");
    // A disconnected handler can leave detached planning/generation work. Its
    // permit stays in that blocking worker; wait for it even when Axum failed.
    let _finished = admission
        .acquire_owned()
        .await
        .context("waiting for admitted HTTP work during shutdown")?;
    result
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn runtime_info(State(state): State<AppState>) -> Json<Value> {
    let descriptor = state.runtime.descriptor();
    Json(json!({
        "schema_version": 1,
        "service": "beholdd",
        "generation": 2,
        "runtime": descriptor.runtime,
        "execution_backends": descriptor.execution_backends,
        "architectures": descriptor.architectures,
        "model_formats": descriptor.model_formats,
        "operations": descriptor.operations,
    }))
}

async fn unload_model(
    State(state): State<AppState>,
    path: Result<Path<String>, PathRejection>,
) -> Result<Json<Value>, ApiError> {
    let Path(model) =
        path.map_err(|error| ApiError::invalid(error.body_text(), Some("model"), "invalid_model"))?;
    // Unlike inference, unload is never queued behind a request: a delayed management
    // action must not remove the model immediately after a user's active turn completes.
    let permit = state
        .admission
        .try_acquire_owned()
        .map_err(|error| match error {
            tokio::sync::TryAcquireError::NoPermits => ApiError::unload(UnloadError::Busy),
            tokio::sync::TryAcquireError::Closed => ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "model management admission is unavailable",
                "server_error",
                None,
                "admission_closed",
            ),
        })?;
    let runtime = state.runtime;
    let unloaded_id = tokio::task::spawn_blocking(move || {
        // Keep the permit in blocking work if the HTTP caller disconnects during cleanup.
        let _permit = permit;
        runtime.unload(&model)?;
        Ok::<_, UnloadError>(model)
    })
    .await
    .map_err(ApiError::join)?
    .map_err(ApiError::unload)?;
    Ok(Json(json!({"id": unloaded_id, "loaded": false})))
}

async fn models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let models = state.runtime.models().map_err(ApiError::runtime)?;
    let data = models
        .into_iter()
        .map(|model| {
            let execution_limits = state
                .runtime
                .execution_limits(&model.id)
                .map_err(ApiError::runtime)?;
            let (compatibility, capabilities) = match model.compatibility {
                ModelCompatibility::Unverified => ("unverified", Vec::new()),
                ModelCompatibility::Compatible(ModelCapabilities::CompletionOnly) => {
                    ("compatible", vec!["completion"])
                }
            };
            Ok(json!({
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "beholdd",
                "loaded": model.loaded,
                "compatibility": compatibility,
                "capabilities": capabilities,
                "execution_limits": execution_limits,
            }))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    Ok(Json(json!({"object": "list", "data": data})))
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let mut request = payload.map_err(ApiError::json)?.0;
    validate_request(&state, &request)?;
    normalize_message_content(&mut request.messages, &request.model)
        .map_err(ApiError::request_protocol)?;
    let tools = request.tools.unwrap_or_default();
    let (generation_request, permit) = plan_admitted_request(
        Arc::clone(&state.runtime),
        state.admission,
        request.model.clone(),
        request.messages,
        tools.clone(),
        request.max_tokens,
        request.enable_thinking,
        request.preserve_thinking,
    )
    .await?;
    let response_id = next_response_id();
    let created = unix_timestamp();

    if request.stream {
        Ok(streaming_response(
            state.runtime,
            permit,
            generation_request,
            request.model,
            tools,
            response_id,
            created,
        ))
    } else {
        nonstream_response(
            state.runtime,
            permit,
            generation_request,
            request.model,
            tools,
            response_id,
            created,
        )
        .await
    }
}

struct PreparedRequest {
    generation: GenerateRequest,
    // Keep parsing bound to the exact template used for this admitted prompt.
    output_dialect: OutputDialect,
}

async fn plan_admitted_request(
    runtime: Arc<dyn InferenceRuntime>,
    admission: Arc<Semaphore>,
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    max_new_tokens: Option<usize>,
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
) -> Result<(PreparedRequest, OwnedSemaphorePermit), ApiError> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut cancel_on_drop = CancelOnDrop::new(Arc::clone(&cancelled));
    // Queue without occupying a blocking worker. Once admitted, the same permit
    // belongs to planning and generation, even if the HTTP future disappears.
    let permit = admission
        .acquire_owned()
        .await
        .map_err(ApiError::admission)?;
    let worker = tokio::task::spawn_blocking(move || {
        let mut control = || -> anyhow::Result<()> {
            if cancelled.load(Ordering::Acquire) {
                Err(anyhow!(
                    "request was cancelled by the client during planning"
                ))
            } else {
                Ok(())
            }
        };
        control().map_err(ApiError::runtime)?;
        // Preserve refusal of an impossible explicit output limit before any
        // checkpoint metadata. Only the actual rendered prompt can resolve omission.
        if let Some(explicit) = max_new_tokens {
            runtime
                .execution_limits(&model)
                .map_err(ApiError::runtime)?
                .resolve(1, Some(explicit), DEFAULT_MAX_TOKENS)
                .map_err(|error| ApiError::runtime(error.into()))?;
        }
        let template = runtime
            .chat_template_controlled(&model, &mut control)
            .map_err(ApiError::runtime)?;
        control().map_err(ApiError::runtime)?;
        let (prompt, output_dialect) = prepare_chat(
            &messages,
            &tools,
            &template,
            enable_thinking,
            preserve_thinking,
        )
        .map_err(ApiError::request_protocol)?;
        control().map_err(ApiError::runtime)?;
        let request = runtime
            .plan_request_controlled(
                &model,
                &prompt,
                max_new_tokens,
                DEFAULT_MAX_TOKENS,
                &mut control,
            )
            .map_err(ApiError::runtime)?;
        control().map_err(ApiError::runtime)?;
        Ok::<_, ApiError>((
            PreparedRequest {
                generation: request,
                output_dialect,
            },
            permit,
        ))
    })
    .await;
    cancel_on_drop.disarm();
    worker.map_err(ApiError::join)?
}

fn validate_request(state: &AppState, request: &ChatRequest) -> Result<(), ApiError> {
    if request.model.trim().is_empty() {
        return Err(ApiError::invalid(
            "model must not be empty",
            Some("model"),
            "invalid_model",
        ));
    }
    if request
        .temperature
        .is_some_and(|temperature| temperature != 0.0)
    {
        return Err(ApiError::invalid(
            "temperature is unsupported; omit it or set it to 0",
            Some("temperature"),
            "unsupported_value",
        ));
    }
    if let Some(max_tokens) = request.max_tokens {
        if max_tokens == 0 || max_tokens > MAX_TOKENS {
            return Err(ApiError::invalid(
                format!("max_tokens must be between 1 and {MAX_TOKENS}"),
                Some("max_tokens"),
                "invalid_value",
            ));
        }
    }
    let models = state.runtime.models().map_err(ApiError::runtime)?;
    if !models.into_iter().any(|model| model.id == request.model) {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("model {:?} was not found", request.model),
            "invalid_request_error",
            Some("model"),
            "model_not_found",
        ));
    }
    Ok(())
}

async fn nonstream_response(
    runtime: Arc<dyn InferenceRuntime>,
    permit: OwnedSemaphorePermit,
    request: PreparedRequest,
    model: String,
    tools: Vec<ToolDefinition>,
    response_id: String,
    created: u64,
) -> Result<Response, ApiError> {
    let PreparedRequest {
        generation: request,
        output_dialect,
    } = request;
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut cancel_on_drop = CancelOnDrop::new(cancelled.clone());
    let worker = tokio::task::spawn_blocking(move || {
        // Keep admission with the detached blocking work if the HTTP handler is dropped.
        // Releasing it with the handler would allow a replacement request to overlap the
        // still-running native call.
        let _permit = permit;
        let control_cancelled = cancelled.clone();
        let mut control = move || -> anyhow::Result<()> {
            if control_cancelled.load(Ordering::Acquire) {
                Err(anyhow!("request was cancelled by the client"))
            } else {
                Ok(())
            }
        };
        let mut discard = move |_piece: &str| -> anyhow::Result<()> {
            if cancelled.load(Ordering::Acquire) {
                Err(anyhow!("request was cancelled by the client"))
            } else {
                Ok(())
            }
        };
        runtime.generate_controlled(request, &mut control, &mut discard)
    })
    .await;
    cancel_on_drop.disarm();
    let generation = worker.map_err(ApiError::join)?.map_err(ApiError::runtime)?;
    let translated = translate_output_for_dialect(&generation.text, &tools, output_dialect)
        .map_err(ApiError::generated_protocol)?;
    let finish_reason = finish_reason(&generation, &translated);
    let tool_calls = translated
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            })
        })
        .collect::<Vec<_>>();
    let mut message = json!({
        "role": "assistant",
        "content": translated.content,
    });
    if let Some(reasoning) = translated.reasoning_content {
        message
            .as_object_mut()
            .expect("object")
            .insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !tool_calls.is_empty() {
        message
            .as_object_mut()
            .expect("message is an object")
            .insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }

    Ok(Json(json!({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.prompt_tokens.saturating_add(generation.completion_tokens),
        }
    }))
    .into_response())
}

struct CancelOnDrop {
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl CancelOnDrop {
    fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            cancelled,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

fn streaming_response(
    runtime: Arc<dyn InferenceRuntime>,
    permit: OwnedSemaphorePermit,
    request: PreparedRequest,
    model: String,
    tools: Vec<ToolDefinition>,
    response_id: String,
    created: u64,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Event>(STREAM_QUEUE);
    let role = chunk_event(
        &response_id,
        created,
        &model,
        json!({"role": "assistant"}),
        None,
    );
    // A fresh bounded queue always has room. Should that invariant change, the worker
    // still emits a terminal error rather than making the handler wait on generation.
    let _ = sender.try_send(role);

    // Planning has completed before response headers, with admission retained. The
    // receiver now owns cancellation; disconnected streams never enter generation.
    tokio::spawn(async move {
        if sender.is_closed() {
            return;
        }
        let worker_sender = sender.clone();
        let worker = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if matches!(request.output_dialect, OutputDialect::OpenMythos { .. }) {
                stream_openmythos_generation(
                    runtime,
                    request,
                    &tools,
                    &worker_sender,
                    &response_id,
                    created,
                    &model,
                );
            } else if tools.is_empty() {
                stream_text_generation(
                    runtime,
                    request,
                    &worker_sender,
                    &response_id,
                    created,
                    &model,
                );
            } else {
                stream_tool_generation(
                    runtime,
                    request,
                    &tools,
                    &worker_sender,
                    &response_id,
                    created,
                    &model,
                );
            }
        });
        if let Err(error) = worker.await {
            let _ = sender
                .send(
                    Event::default()
                        .event("error")
                        .data(ApiError::join(error).envelope_json()),
                )
                .await;
        }
    });

    let stream = ReceiverStream::new(receiver).map(Ok::<Event, Infallible>);
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(10))
                .text("keep-alive"),
        )
        .into_response()
}

fn stream_openmythos_generation(
    runtime: Arc<dyn InferenceRuntime>,
    request: PreparedRequest,
    tools: &[ToolDefinition],
    sender: &mpsc::Sender<Event>,
    response_id: &str,
    created: u64,
    model: &str,
) {
    let PreparedRequest {
        generation: request,
        output_dialect,
    } = request;
    let OutputDialect::OpenMythos { thinking } = output_dialect else {
        unreachable!("selected dialect")
    };
    let mut parser = protocol::openmythos::Stream::new(thinking);
    let emit_fragments = |fragments: Vec<protocol::openmythos::Fragment>| -> anyhow::Result<()> {
        for fragment in fragments {
            let delta = match fragment {
                protocol::openmythos::Fragment::Reasoning(value) => {
                    json!({"reasoning_content": value})
                }
                protocol::openmythos::Fragment::Content(value) => json!({"content": value}),
            };
            sender
                .blocking_send(chunk_event(response_id, created, model, delta, None))
                .map_err(|_| anyhow!("stream receiver disconnected"))?;
        }
        Ok(())
    };
    let result = {
        let mut control = || -> anyhow::Result<()> {
            if sender.is_closed() {
                Err(anyhow!("stream receiver disconnected"))
            } else {
                Ok(())
            }
        };
        let mut emit = |piece: &str| -> anyhow::Result<()> {
            let fragments = parser
                .push(piece)
                .map_err(|error| anyhow!("{}", error.message))?;
            emit_fragments(fragments)
        };
        runtime.generate_controlled(request, &mut control, &mut emit)
    };
    let generation = match result {
        Ok(value) => value,
        Err(error) => {
            send_stream_error(sender, ApiError::runtime(error));
            return;
        }
    };
    // Do not publish calls from callbacks that disagree with the runtime's final
    // canonical text. Neither malformed nor incomplete payloads execute a tool.
    let canonical = match translate_output_for_dialect(&generation.text, tools, output_dialect) {
        Ok(value) => value,
        Err(error) => {
            send_stream_error(sender, ApiError::generated_protocol(error));
            return;
        }
    };
    let (translated, tail) = match parser.finish(tools) {
        Ok(value) => value,
        Err(error) => {
            send_stream_error(sender, ApiError::generated_protocol(error));
            return;
        }
    };
    if translated != canonical {
        send_stream_error(
            sender,
            ApiError::runtime(anyhow!(
                "OpenMythos callback text disagrees with canonical completion"
            )),
        );
        return;
    }
    if emit_fragments(tail).is_err() {
        return;
    }
    if !translated.tool_calls.is_empty() {
        let calls = translated
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "index": index, "id": call.id, "type": "function",
                    "function": { "name": call.name, "arguments": call.arguments },
                })
            })
            .collect::<Vec<_>>();
        if sender
            .blocking_send(chunk_event(
                response_id,
                created,
                model,
                json!({"tool_calls": calls}),
                None,
            ))
            .is_err()
        {
            return;
        }
    }
    let _ = sender.blocking_send(terminal_chunk_event(
        response_id,
        created,
        model,
        finish_reason(&generation, &translated),
        &generation,
    ));
    send_done(sender);
}

fn stream_text_generation(
    runtime: Arc<dyn InferenceRuntime>,
    request: PreparedRequest,
    sender: &mpsc::Sender<Event>,
    response_id: &str,
    created: u64,
    model: &str,
) {
    let PreparedRequest {
        generation: request,
        output_dialect,
    } = request;
    let callback_sender = sender.clone();
    let control_sender = sender.clone();
    let callback_id = response_id.to_owned();
    let callback_model = model.to_owned();
    let mut guard = MarkupGuard::default();
    let result = {
        let mut control = move || -> anyhow::Result<()> {
            if control_sender.is_closed() {
                Err(anyhow!("stream receiver disconnected"))
            } else {
                Ok(())
            }
        };
        let mut emit = |piece: &str| -> anyhow::Result<()> {
            let Some(safe) = guard.push(piece)? else {
                return Ok(());
            };
            callback_sender
                .blocking_send(chunk_event(
                    &callback_id,
                    created,
                    &callback_model,
                    json!({"content": safe}),
                    None,
                ))
                .map_err(|_| anyhow!("stream receiver disconnected"))
        };
        runtime.generate_controlled(request, &mut control, &mut emit)
    };

    match result {
        Ok(generation) => {
            // Validate the canonical final text too: a runtime is allowed to combine
            // token bytes differently from its callback chunks.
            if let Err(error) = translate_output_for_dialect(&generation.text, &[], output_dialect)
            {
                send_stream_error(sender, ApiError::generated_protocol(error));
                return;
            }
            if let Some(safe) = guard.finish() {
                let _ = sender.blocking_send(chunk_event(
                    response_id,
                    created,
                    model,
                    json!({"content": safe}),
                    None,
                ));
            }
            let finish = runtime_finish_reason(generation.finish_reason);
            let _ = sender.blocking_send(terminal_chunk_event(
                response_id,
                created,
                model,
                finish,
                &generation,
            ));
            send_done(sender);
        }
        Err(error) => {
            send_stream_error(sender, ApiError::runtime(error));
        }
    }
}

fn stream_tool_generation(
    runtime: Arc<dyn InferenceRuntime>,
    request: PreparedRequest,
    tools: &[ToolDefinition],
    sender: &mpsc::Sender<Event>,
    response_id: &str,
    created: u64,
    model: &str,
) {
    let PreparedRequest {
        generation: request,
        output_dialect,
    } = request;
    let cancellation = sender.clone();
    let control_cancellation = cancellation.clone();
    let mut control = move || -> anyhow::Result<()> {
        if control_cancellation.is_closed() {
            Err(anyhow!("stream receiver disconnected"))
        } else {
            Ok(())
        }
    };
    let mut discard = move |_piece: &str| -> anyhow::Result<()> {
        if cancellation.is_closed() {
            Err(anyhow!("stream receiver disconnected"))
        } else {
            Ok(())
        }
    };
    match runtime.generate_controlled(request, &mut control, &mut discard) {
        Ok(generation) => {
            match translate_output_for_dialect(&generation.text, tools, output_dialect) {
                Ok(translated) => {
                    if let Some(content) = translated.content.as_deref() {
                        if !content.is_empty() {
                            let _ = sender.blocking_send(chunk_event(
                                response_id,
                                created,
                                model,
                                json!({"content": content}),
                                None,
                            ));
                        }
                    }
                    if !translated.tool_calls.is_empty() {
                        let calls = translated
                            .tool_calls
                            .iter()
                            .enumerate()
                            .map(|(index, call)| {
                                json!({
                                    "index": index,
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    }
                                })
                            })
                            .collect::<Vec<_>>();
                        let _ = sender.blocking_send(chunk_event(
                            response_id,
                            created,
                            model,
                            json!({"tool_calls": calls}),
                            None,
                        ));
                    }
                    let finish = finish_reason(&generation, &translated);
                    let _ = sender.blocking_send(terminal_chunk_event(
                        response_id,
                        created,
                        model,
                        finish,
                        &generation,
                    ));
                    send_done(sender);
                }
                Err(error) => {
                    send_stream_error(sender, ApiError::generated_protocol(error));
                }
            }
        }
        Err(error) => {
            send_stream_error(sender, ApiError::runtime(error));
        }
    }
}

fn chunk_event(
    response_id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> Event {
    Event::default().data(
        serde_json::to_string(&json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }]
        }))
        .expect("JSON response serialization is infallible"),
    )
}

fn send_stream_error(sender: &mpsc::Sender<Event>, error: ApiError) {
    let _ = sender.blocking_send(Event::default().event("error").data(error.envelope_json()));
}

fn terminal_chunk_event(
    response_id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
    generation: &Generation,
) -> Event {
    // Usage is authoritative only once generation succeeds. Keep it on the
    // terminal choice so consumers can bind the text, finish reason and counts.
    Event::default().data(
        serde_json::to_string(&json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.prompt_tokens.saturating_add(generation.completion_tokens),
            }
        }))
        .expect("JSON response serialization is infallible"),
    )
}

fn send_done(sender: &mpsc::Sender<Event>) {
    let _ = sender.blocking_send(Event::default().data("[DONE]"));
}

fn finish_reason(generation: &Generation, output: &TranslatedOutput) -> &'static str {
    if output.tool_calls.is_empty() {
        runtime_finish_reason(generation.finish_reason)
    } else {
        "tool_calls"
    }
}

fn runtime_finish_reason(reason: RuntimeFinishReason) -> &'static str {
    match reason {
        RuntimeFinishReason::Stop => "stop",
        RuntimeFinishReason::Length => "length",
    }
}

fn next_response_id() -> String {
    format!(
        "chatcmpl-beholdd-{}",
        REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Default)]
struct MarkupGuard {
    pending: String,
}

impl MarkupGuard {
    fn push(&mut self, piece: &str) -> anyhow::Result<Option<String>> {
        self.pending.push_str(piece);
        if TOOL_MARKERS
            .iter()
            .any(|marker| self.pending.contains(marker))
        {
            return Err(anyhow!("model emitted native tool markup without tools"));
        }
        let keep = self
            .pending
            .rfind('<')
            .filter(|at| {
                let suffix = &self.pending[*at..];
                TOOL_MARKERS.iter().any(|marker| marker.starts_with(suffix))
            })
            .map_or(0, |at| self.pending.len() - at);
        let emit_len = self.pending.len() - keep;
        if emit_len == 0 {
            return Ok(None);
        }
        let suffix = self.pending.split_off(emit_len);
        let emit = std::mem::replace(&mut self.pending, suffix);
        Ok(Some(emit))
    }

    fn finish(&mut self) -> Option<String> {
        (!self.pending.is_empty()).then(|| std::mem::take(&mut self.pending))
    }
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    param: Option<&'static str>,
    code: &'static str,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    body: ErrorBody,
}

impl ApiError {
    fn unload(error: UnloadError) -> Self {
        let (status, kind, param, code) = match &error {
            UnloadError::NotFound(_) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                Some("model"),
                "model_not_found",
            ),
            UnloadError::Busy => (StatusCode::CONFLICT, "server_error", None, "runtime_busy"),
            UnloadError::Unsupported => (
                StatusCode::NOT_IMPLEMENTED,
                "server_error",
                None,
                "unsupported_operation",
            ),
            UnloadError::Failed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                None,
                "unload_error",
            ),
        };
        Self::new(status, error.to_string(), kind, param, code)
    }

    fn new(
        status: StatusCode,
        message: impl Into<String>,
        kind: &'static str,
        param: Option<&'static str>,
        code: &'static str,
    ) -> Self {
        Self {
            status,
            body: ErrorBody {
                message: message.into(),
                kind,
                param,
                code,
            },
        }
    }

    fn invalid(
        message: impl Into<String>,
        param: Option<&'static str>,
        code: &'static str,
    ) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            param,
            code,
        )
    }

    fn json(error: JsonRejection) -> Self {
        Self::invalid(
            format!("invalid JSON request: {}", error.body_text()),
            None,
            "invalid_json",
        )
    }

    fn request_protocol(error: ProtocolError) -> Self {
        Self::invalid(error.message, error.param, error.code)
    }

    fn generated_protocol(error: ProtocolError) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            error.message,
            "server_error",
            None,
            error.code,
        )
    }

    fn runtime(error: anyhow::Error) -> Self {
        if let Some(request) = error.downcast_ref::<RequestError>() {
            let (param, code) = match request {
                RequestError::EmptyPrompt => ("messages", "invalid_value"),
                RequestError::InvalidOutputLimit => ("max_tokens", "invalid_value"),
                RequestError::ContextCapacity { .. } => ("max_tokens", "context_length_exceeded"),
            };
            return Self::invalid(request.to_string(), Some(param), code);
        }
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference failed: {error:#}"),
            "server_error",
            None,
            "inference_error",
        )
    }

    fn join(error: tokio::task::JoinError) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference worker failed: {error}"),
            "server_error",
            None,
            "worker_error",
        )
    }

    fn admission(error: tokio::sync::AcquireError) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("inference admission is unavailable: {error}"),
            "server_error",
            None,
            "admission_closed",
        )
    }

    fn envelope_json(&self) -> String {
        serde_json::to_string(&ErrorEnvelope {
            error: ErrorBody {
                message: self.body.message.clone(),
                kind: self.body.kind,
                param: self.body.param,
                code: self.body.code,
            },
        })
        .expect("JSON error serialization is infallible")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(ErrorEnvelope { error: self.body })).into_response()
    }
}
