use super::*;
use crate::protocol::{
    normalize_message_content, prepare_chat, OPENMYTHOS_CHAT_TEMPLATE, QWEN3_CHAT_TEMPLATE,
    QWEN3_CODER_CHAT_TEMPLATE,
};
use beholdd_runtime::ChatTemplate;
use serde_json::json;

fn messages(value: Value) -> Vec<ChatMessage> {
    let mut messages: Vec<ChatMessage> = serde_json::from_value(value).unwrap();
    normalize_message_content(&mut messages, "openmythos").unwrap();
    messages
}

fn tool() -> ToolDefinition {
    serde_json::from_value(
        json!({"type":"function","function":{"name":"write","parameters":{
            "type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"}},
            "required":["text"],"additionalProperties":false
        }}}),
    )
    .unwrap()
}

fn call(text: &str) -> String {
    format!("<tool_call>\n<function=write>\n<parameter=text>\n{text}\n</parameter>\n</function>\n</tool_call>")
}

#[test]
fn exact_identity_and_default_thinking_are_preserved() {
    assert_eq!(OPENMYTHOS_CHAT_TEMPLATE.len(), 8057);
    assert!(!OPENMYTHOS_CHAT_TEMPLATE.ends_with('\n'));
    let history = messages(json!([{"role":"user","content":" \tHello\n"}]));
    let template = ChatTemplate::Checkpoint(Some(OPENMYTHOS_CHAT_TEMPLATE.into()));
    let (prompt, dialect) = prepare_chat(&history, &[], &template, None, None).unwrap();
    assert_eq!(
        prompt,
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    assert!(matches!(
        dialect,
        super::super::OutputDialect::OpenMythos { thinking: true }
    ));
    let (disabled, _) = prepare_chat(&history, &[], &template, Some(false), None).unwrap();
    assert!(disabled.ends_with("assistant\n<think>\n\n</think>\n\n"));
    assert!(prepare_chat(
        &history,
        &[],
        &ChatTemplate::Checkpoint(Some(format!("{OPENMYTHOS_CHAT_TEMPLATE}\n"))),
        None,
        None
    )
    .is_err());
    for old in [
        ChatTemplate::Legacy,
        ChatTemplate::Checkpoint(Some(QWEN3_CHAT_TEMPLATE.into())),
        ChatTemplate::Checkpoint(Some(QWEN3_CODER_CHAT_TEMPLATE.into())),
    ] {
        assert!(prepare_chat(&history, &[], &old, Some(true), None).is_err());
        assert!(prepare_chat(
            &messages(json!([{"role":"developer","content":"a"}])),
            &[],
            &old,
            None,
            None
        )
        .is_err());
        assert!(prepare_chat(
            &messages(json!([{"role":"assistant","content":"a","reasoning_content":"r"}])),
            &[],
            &old,
            None,
            None
        )
        .is_err());
    }
}

#[test]
fn leading_instructions_and_ascii_trim_match_pinned_environment() {
    let history = messages(json!([
        {"role":"developer","content":" A "}, {"role":"system","content":" B "},
        {"role":"user","content":[{"type":"text","text":" \n"},{"type":"text","text":"\u{a0}Q\u{1c}"}]}
    ]));
    assert_eq!(render(&history, &[], Options::default()).unwrap(),
        "<|im_start|>system\nA\nB<|im_end|>\n<|im_start|>user\n\u{a0}Q\u{1c}<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert!(render(&messages(json!([{"role":"system","content":" "},{"role":"developer","content":" "},{"role":"user","content":"q"}])), &[], Options::default()).unwrap().starts_with("<|im_start|>system\n\n<|im_end|>\n"));
    for history in [
        json!([]),
        json!([{"role":"user","content":"q"},{"role":"system","content":"late"}]),
        json!([{"role":"system","content":"a"},{"role":"system","content":"b"},{"role":"system","content":"c"}]),
    ] {
        assert!(render(&messages(history), &[], Options::default()).is_err());
    }
}

#[test]
fn historical_reasoning_precedence_and_preservation_are_explicit() {
    let history = messages(json!([
        {"role":"user","content":"q"},
        {"role":"assistant","content":"prefix<think> first </think>discard</think>\nA"},
        {"role":"assistant","content":"B","reasoning_content":" explicit "},
        {"role":"assistant","content":"<think>literal</think>C","reasoning_content":""},
        {"role":"user","content":"next"}
    ]));
    let preserved = render(&history, &[], Options::default()).unwrap();
    assert!(preserved.contains("assistant\n<think>\nfirst\n</think>\n\nA<|im_end|>"));
    assert!(preserved.contains("assistant\n<think>\nexplicit\n</think>\n\nB<|im_end|>"));
    assert!(
        preserved.contains("assistant\n<think>\n\n</think>\n\n<think>literal</think>C<|im_end|>")
    );
    let dropped = render(
        &history,
        &[],
        Options {
            preserve_thinking: false,
            thinking: true,
        },
    )
    .unwrap();
    assert!(dropped.contains("assistant\nA<|im_end|>"));
    assert!(!dropped.contains("explicit"));
    // Later assistants retain even EMPTY reasoning blocks, not only the last one.
    let after_query = messages(
        json!([{"role":"user","content":"q"},{"role":"assistant","content":"A"},{"role":"assistant","content":"B"}]),
    );
    assert_eq!(
        render(
            &after_query,
            &[],
            Options {
                preserve_thinking: false,
                thinking: true
            }
        )
        .unwrap()
        .matches("<think>\n\n</think>")
        .count(),
        2
    );
}

#[test]
fn tools_are_first_and_history_keeps_exact_argument_order_and_scalar_values() {
    let history = messages(json!([
        {"role":"system","content":" system "}, {"role":"user","content":"q"},
        {"role":"assistant","content":" preface ","tool_calls":[{"id":"a","type":"function","function":{"name":"write","arguments":"{\"z\":true,\"a\":null,\"nested\":{\"b\":2,\"a\":1},\"x\":1.0}"}},{"id":"b","type":"function","function":{"name":"write","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"a","content":" one "},{"role":"tool","tool_call_id":"b","content":" two "}
    ]));
    let prompt = render(&history, &[tool()], Options::default()).unwrap();
    assert!(prompt.starts_with("<|im_start|>system\n# Tools\n\nYou have access"));
    assert!(prompt.contains("</IMPORTANT>\n\nsystem<|im_end|>"));
    assert!(prompt.contains("preface\n\n<tool_call>\n<function=write>\n<parameter=z>\nTrue\n</parameter>\n<parameter=a>\nNone\n</parameter>\n<parameter=nested>\n{\"b\": 2, \"a\": 1}\n</parameter>\n<parameter=x>\n1\n</parameter>\n"));
    assert!(prompt.contains(
        "</tool_call>\n<tool_call>\n<function=write>\n</function>\n</tool_call><|im_end|>"
    ));
    assert!(prompt.contains("<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>"));
}

#[test]
fn malformed_history_multimodal_and_ambiguous_number_domains_refuse() {
    for history in [
        json!([{"role":"tool","tool_call_id":"x","content":"result"}]),
        json!([{"role":"assistant","content":"","tool_calls":[{"id":"x","type":"function","function":{"name":"write","arguments":"{}"}}]}]),
        json!([{"role":"user","content":"q","reasoning_content":"r"}]),
    ] {
        assert!(render(&messages(history), &[], Options::default()).is_err());
    }
    for part in [
        json!({"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}),
        json!({"type":"input_audio","input_audio":{}}),
        json!({"type":"video","video":"x"}),
    ] {
        let mut history: Vec<ChatMessage> =
            serde_json::from_value(json!([{"role":"user","content":[part]}])).unwrap();
        assert_eq!(
            normalize_message_content(&mut history, "openmythos")
                .unwrap_err()
                .code,
            "unsupported_content_type"
        );
    }
    for (value, expected) in [
        (1.0, "1"),
        (-0.0, "-0"),
        (1e6, "1e+06"),
        (1e-5, "1e-05"),
        (0.0001, "0.0001"),
        (123456.0, "123456"),
        (1.23456789, "1.23457"),
        (999999.9, "1e+06"),
        (f64::from_bits(1), "4.94066e-324"),
    ] {
        assert_eq!(number_spelling(value).unwrap(), expected);
    }
    for value in [f64::INFINITY, f64::NAN, 9_223_372_036_854_775_808.0] {
        assert!(number_spelling(value).is_err());
    }
    assert!(native_json(&OrderedJson::parse("18446744073709551615").unwrap()).is_err());
    assert!(OrderedJson::parse("{\"a\":1,\"a\":2}").is_err());
}

#[test]
fn completed_tool_rounds_may_reuse_call_ids_but_pending_calls_may_not() {
    let assistant = json!({"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"write","arguments":"{}"}}]});
    let result = json!({"role":"tool","tool_call_id":"call_1","content":"ok"});
    assert!(render(
        &messages(json!([
            assistant.clone(),
            result.clone(),
            assistant.clone(),
            result.clone()
        ])),
        &[],
        Options::default()
    )
    .is_ok());
    let mut duplicate = assistant;
    let call = duplicate["tool_calls"][0].clone();
    duplicate["tool_calls"].as_array_mut().unwrap().push(call);
    assert!(render(
        &messages(json!([duplicate, result])),
        &[],
        Options::default()
    )
    .is_err());
}

#[test]
fn prompt_rounding_does_not_rewrite_request_or_generated_tool_arguments() {
    let raw = "{\"x\":1.23456789}";
    let history = messages(json!([
        {"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"write","arguments":raw}}]},
        {"role":"tool","tool_call_id":"a","content":"ok"}
    ]));
    assert!(render(&history, &[], Options::default())
        .unwrap()
        .contains("\n1.23457\n</parameter>"));
    assert_eq!(
        history[0].tool_calls.as_ref().unwrap()[0]
            .function
            .arguments,
        raw
    );
    let numeric: ToolDefinition = serde_json::from_value(json!({"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"text":{"type":"number"}},"required":["text"]}}})).unwrap();
    let result = translate(&call("1.23456789"), &[numeric], true).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&result.tool_calls[0].arguments).unwrap()["text"],
        json!(1.23456789)
    );
    let structured: ToolDefinition = serde_json::from_value(json!({"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"text":{"type":"object"}},"required":["text"]}}})).unwrap();
    let ordered = translate(
        &call("{\"z\":1.23456789,\"a\":2}"),
        &[structured.clone()],
        true,
    )
    .unwrap();
    assert_eq!(
        ordered.tool_calls[0].arguments,
        "{\"text\":{\"z\": 1.23456789, \"a\": 2}}"
    );
    assert!(translate(&call("{\"x\":1,\"x\":2}"), &[structured], true).is_err());
}

#[test]
fn implicit_reasoning_is_a_separate_channel_and_incomplete_reasoning_is_not_an_answer() {
    let result = translate("reason\n</think>\n\nAnswer", &[], true).unwrap();
    assert_eq!(result.reasoning_content.as_deref(), Some("reason\n"));
    assert_eq!(result.content.as_deref(), Some("\n\nAnswer"));
    let result = translate("unfinished reasoning <tool_ca", &[], true).unwrap();
    assert_eq!(
        result.reasoning_content.as_deref(),
        Some("unfinished reasoning <tool_ca")
    );
    assert!(result.content.is_none());
    assert!(result.tool_calls.is_empty());
    let result = translate("Answer", &[], false).unwrap();
    assert!(result.reasoning_content.is_none());
    assert_eq!(result.content.as_deref(), Some("Answer"));
}

#[test]
fn implicit_tool_transition_validates_before_publication_and_preserves_string_markers() {
    let output = format!(
        "Need a tool.{}\n{}",
        call("literal </tool_call> inside a string"),
        call("second")
    );
    let result = translate(&output, &[tool()], true).unwrap();
    assert_eq!(result.reasoning_content.as_deref(), Some("Need a tool."));
    assert!(result.content.is_none());
    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(
        serde_json::from_str::<Value>(&result.tool_calls[0].arguments).unwrap()["text"],
        "literal </tool_call> inside a string"
    );
    for bad in [
        call("x") + "suffix",
        call("x").replace("\n</parameter>\n", "</parameter>"),
        call("x").replace("write>", "unknown>"),
        call("x").replace("parameter=text", "parameter=unknown"),
        "<tool_call>\n<function=write>\n".into(),
    ] {
        assert!(translate(&bad, &[tool()], true).is_err(), "{bad}");
    }
    assert!(translate(&call("x"), &[], true).is_err());
    assert!(translate(
        &call("x").replace(
            "</function>",
            "<parameter=text>\ny\n</parameter>\n</function>"
        ),
        &[tool()],
        true
    )
    .is_err());
}

#[test]
fn every_utf8_chunk_boundary_preserves_channels_and_never_emits_tool_text() {
    for text in [
        format!("思考🙂</think>\nAnswer {}", call("x")),
        format!("思考{}", call("y")),
        "reason </think> answer".into(),
        "unfinished <tool_ca".into(),
    ] {
        let expected = translate(&text, &[tool()], true).unwrap();
        for at in (0..=text.len()).filter(|at| text.is_char_boundary(*at)) {
            let mut parser = Stream::new(true);
            let mut parts = parser.push(&text[..at]).unwrap();
            parts.extend(parser.push(&text[at..]).unwrap());
            let (actual, tail) = parser.finish(&[tool()]).unwrap();
            parts.extend(tail);
            assert_eq!(actual, expected, "split {at}");
            let reasoning: String = parts
                .iter()
                .filter_map(|part| match part {
                    Fragment::Reasoning(value) => Some(value.as_str()),
                    _ => None,
                })
                .collect();
            let content: String = parts
                .iter()
                .filter_map(|part| match part {
                    Fragment::Content(value) => Some(value.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                reasoning,
                actual.reasoning_content.clone().unwrap_or_default()
            );
            assert_eq!(content, actual.content.clone().unwrap_or_default());
            assert!(!content.contains("<tool_call>"));
        }
    }
}

#[test]
#[ignore = "requires explicit independent model-free Jinja fixture path"]
fn independent_jinja_fixture_prompts_match_exactly() {
    use std::io::Read;
    let path =
        std::env::var_os("OPENMYTHOS_PROTOCOL_ORACLE").expect("set the frozen oracle JSON path");
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .take((1 << 20) + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= 1 << 20);
    let fixture: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["template_sha256"],
        "55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty() && cases.len() <= 64);
    let template = ChatTemplate::Checkpoint(Some(OPENMYTHOS_CHAT_TEMPLATE.into()));
    for case in cases {
        // Direct wire text retains the insertion order of tool definitions.
        let mut request: crate::protocol::ChatRequest =
            serde_json::from_str(case["request_text"].as_str().unwrap()).unwrap();
        normalize_message_content(&mut request.messages, &request.model).unwrap();
        let actual = prepare_chat(
            &request.messages,
            request.tools.as_deref().unwrap_or_default(),
            &template,
            request.enable_thinking,
            request.preserve_thinking,
        )
        .unwrap()
        .0;
        assert_eq!(actual, case["prompt"].as_str().unwrap(), "{}", case["name"]);
    }
}

fn bounded_tool() -> ToolDefinition {
    serde_json::from_value(json!({"type":"function","function":{
        "name":"bounded","parameters":{
            "type":"object","properties":{"value":{
                "type":"object","properties":{
                    "number":{"type":"integer","maximum":9_007_199_254_740_993u64},
                    "scores":{"type":"array","maxItems":2,"items":{
                        "type":["number","null"],"minimum":-0.5,"maximum":0.5
                    }},
                    "rows":{"type":"array","maxItems":1,"items":{
                        "type":"array","maxItems":0
                    }}
                },
                "required":["number","scores","rows"],
                "additionalProperties":{"type":"array","maxItems":1,"items":{
                    "type":"number","maximum":0.5
                }}
            }},
            "required":["value"],"additionalProperties":false
        }
    }}))
    .unwrap()
}

fn bounded_xml(value: &Value) -> String {
    // OpenMythos uses exact function/parameter newline framing. Generated JSON
    // numbers retain their original value; historical prompt rounding is separate.
    format!("<tool_call>\n<function=bounded>\n<parameter=value>\n{value}\n</parameter>\n</function>\n</tool_call>")
}

#[test]
fn generated_openmythos_calls_enforce_nested_and_additional_upper_bounds() {
    let tools = [bounded_tool()];
    super::super::validate_tools(&tools).unwrap();
    let valid = json!({
        "number":9_007_199_254_740_993u64,"scores":[null,0.5],"rows":[[]],"extra":[0.5]
    });
    for thinking in [false, true] {
        let translated = translate(&bounded_xml(&valid), &tools, thinking).unwrap();
        assert_eq!(translated.tool_calls.len(), 1);
        let arguments: Value = serde_json::from_str(&translated.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["value"], valid);
        assert!(translated.tool_calls[0]
            .arguments
            .contains("9007199254740993"));
        for (name, value, expected) in [
            (
                "number",
                json!(9_007_199_254_740_994u64),
                ".number is above its maximum",
            ),
            ("scores", json!([0, 0, 0]), ".scores exceeds its maxItems"),
            (
                "scores",
                json!([0, 0.75]),
                ".scores[1] is above its maximum",
            ),
            (
                "scores",
                json!([0, -0.75]),
                ".scores[1] is below its minimum",
            ),
            ("rows", json!([[], []]), ".rows exceeds its maxItems"),
            ("rows", json!([[0]]), ".rows[0] exceeds its maxItems"),
            ("extra", json!([0, 0]), ".extra exceeds its maxItems"),
            ("extra", json!([0.75]), ".extra[0] is above its maximum"),
        ] {
            let mut invalid = valid.clone();
            invalid[name] = value;
            let error = translate(&bounded_xml(&invalid), &tools, thinking).unwrap_err();
            assert_eq!(error.code, "tool_call_parse_error");
            assert!(
                error.message.contains(expected),
                "thinking={thinking}: {}",
                error.message
            );
        }
    }
}

#[test]
fn bounded_openmythos_batch_never_publishes_earlier_calls_if_a_later_call_fails() {
    let tools = [bounded_tool()];
    let valid = json!({"number":1,"scores":[],"rows":[]});
    let mut invalid = valid.clone();
    invalid["scores"] = json!([0.75]);
    let batch = format!("{}\n{}", bounded_xml(&valid), bounded_xml(&invalid));
    // A complete valid first call must remain buffered while the second call is
    // incomplete or later fails its nested schema. Exercise every UTF-8 split.
    for thinking in [false, true] {
        for at in (0..=batch.len()).filter(|at| batch.is_char_boundary(*at)) {
            let mut parser = Stream::new(thinking);
            assert!(parser.push(&batch[..at]).unwrap().is_empty());
            assert!(parser.push(&batch[at..]).unwrap().is_empty());
            let error = parser.finish(&tools).unwrap_err();
            assert_eq!(error.code, "tool_call_parse_error");
            assert!(error.message.contains(".scores[0] is above its maximum"));
        }
    }
}
