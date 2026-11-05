//! HTTP seams only: fake runtimes do not qualify any model arithmetic.
use super::*;

fn runtime(text: &str, chunks: &[&str]) -> TemplateRuntime {
    let mut runtime = TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
        crate::protocol::OPENMYTHOS_CHAT_TEMPLATE.into(),
    )));
    runtime.inner = FakeRuntime::new(text, chunks);
    runtime
}

fn xml() -> &'static str {
    "<tool_call>\n<function=write_file>\n<parameter=path>\nx.rs\n</parameter>\n<parameter=content>\nfn main() {}\n</parameter>\n</function>\n</tool_call>"
}

fn deltas(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn openmythos_default_thinking_is_separate_in_both_response_modes() {
    let text = "Reason🙂</think>\n\nAnswer";
    for stream in [false, true] {
        let runtime = Arc::new(runtime(text, &["Rea", "son🙂</thi", "nk>\n\nAns", "wer"]));
        let response = router(runtime.clone())
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({"stream":stream})),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        if stream {
            let body = body_text(response).await;
            assert!(!body.contains("event: error"));
            let chunks = deltas(&body);
            let reasoning: String = chunks
                .iter()
                .filter_map(|value| value["choices"][0]["delta"]["reasoning_content"].as_str())
                .collect();
            let content: String = chunks
                .iter()
                .filter_map(|value| value["choices"][0]["delta"]["content"].as_str())
                .collect();
            assert_eq!(reasoning, "Reason🙂");
            assert_eq!(content, "\n\nAnswer");
            assert_terminal_usage(&body, "stop");
        } else {
            let body = body_json(response).await;
            assert_eq!(
                body["choices"][0]["message"]["reasoning_content"],
                "Reason🙂"
            );
            assert_eq!(body["choices"][0]["message"]["content"], "\n\nAnswer");
            assert_eq!(body["usage"]["completion_tokens"], 2);
        }
        assert!(runtime.inner.requests.lock().unwrap()[0]
            .prompt
            .ends_with("assistant\n<think>\n"));
        assert_eq!(
            *runtime.events.lock().unwrap(),
            ["template", "plan", "generate"]
        );
    }
}

#[tokio::test]
async fn openmythos_reasoning_only_length_stays_reasoning_and_explicit_off_is_scoped() {
    for stream in [false, true] {
        let mut runtime = runtime("still thinking", &["still ", "thinking"]);
        runtime.inner.generation.finish_reason = FinishReason::Length;
        let response = router(Arc::new(runtime))
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({"stream":stream})),
            ))
            .await
            .unwrap();
        if stream {
            let body = body_text(response).await;
            assert_terminal_usage(&body, "length");
            assert!(body.contains("reasoning_content"));
        } else {
            let body = body_json(response).await;
            assert!(body["choices"][0]["message"]["content"].is_null());
            assert_eq!(body["choices"][0]["finish_reason"], "length");
        }
    }
    let runtime = Arc::new(runtime("Answer", &["Answer"]));
    let response = router(runtime.clone())
        .oneshot(request(
            Method::POST,
            "/v1/chat/completions",
            completion_request(json!({"enable_thinking":false})),
        ))
        .await
        .unwrap();
    let body = body_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "Answer");
    assert!(body["choices"][0]["message"]
        .get("reasoning_content")
        .is_none());
    assert!(runtime.inner.requests.lock().unwrap()[0]
        .prompt
        .ends_with("<think>\n\n</think>\n\n"));
}

#[tokio::test]
async fn openmythos_valid_implicit_tool_transition_publishes_once_after_validation() {
    let text = format!("Need a tool.{}", xml());
    for stream in [false, true] {
        let runtime = Arc::new(runtime(&text, &["Need a tool.", xml()]));
        let response = router(runtime)
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({"stream":stream,"tools":[write_tool()]})),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        if stream {
            let body = body_text(response).await;
            assert_terminal_usage(&body, "tool_calls");
            let chunks = deltas(&body);
            assert_eq!(
                chunks
                    .iter()
                    .filter(|value| value["choices"][0]["delta"]["tool_calls"].is_array())
                    .count(),
                1
            );
            assert!(!body.contains("<tool_call>"));
        } else {
            let body = body_json(response).await;
            assert_eq!(
                body["choices"][0]["message"]["reasoning_content"],
                "Need a tool."
            );
            assert_eq!(
                body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "write_file"
            );
            assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        }
    }
}

#[tokio::test]
async fn openmythos_invalid_truncated_or_mismatched_output_never_publishes_a_call() {
    for (canonical, callback) in [
        (format!("{}suffix", xml()), format!("{}suffix", xml())),
        (
            "<tool_call>\n<function=write_file>\n".into(),
            "<tool_call>\n<function=write_file>\n".into(),
        ),
        (xml().into(), "Reason only".into()),
    ] {
        let response = router(Arc::new(runtime(&canonical, &[&callback])))
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(json!({"stream":true,"tools":[write_tool()]})),
            ))
            .await
            .unwrap();
        let body = body_text(response).await;
        assert!(body.contains("event: error"));
        assert!(!body.contains("\"tool_calls\":"));
        assert!(!body.contains("data: [DONE]"));
    }
}

#[tokio::test]
async fn openmythos_unknown_template_and_multimodal_refuse_before_generation() {
    for (template, extra) in [
        (
            format!("{}\n", crate::protocol::OPENMYTHOS_CHAT_TEMPLATE),
            json!({}),
        ),
        (
            crate::protocol::OPENMYTHOS_CHAT_TEMPLATE.into(),
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}),
        ),
    ] {
        let runtime = Arc::new(TemplateRuntime::new(ChatTemplate::Checkpoint(Some(
            template,
        ))));
        let response = router(runtime.clone())
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                completion_request(extra),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!runtime.events.lock().unwrap().contains(&"generate"));
        assert!(!runtime.events.lock().unwrap().contains(&"plan"));
    }
}
