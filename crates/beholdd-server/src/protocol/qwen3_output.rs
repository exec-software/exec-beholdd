//! Decode the JSON tool-call dialect selected by the exact Qwen3 template.

use serde_json::{value::RawValue, Value};

use super::{
    looks_like_tool_markup, ordered_json::OrderedJson, validate_name, validate_value,
    ProtocolError, ToolCall, ToolDefinition, TranslatedOutput,
};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

pub(super) fn translate(
    output: &str,
    tools: &[ToolDefinition],
) -> Result<TranslatedOutput, ProtocolError> {
    // A leading reasoning block is assistant content, including any examples of
    // tool syntax inside it. An unfinished block cannot authorize a tool call.
    let without_space = output.trim_start();
    let search_start = if let Some(reasoning) = without_space.strip_prefix("<think>") {
        let Some(end) = reasoning.find("</think>") else {
            return Ok(prose(output));
        };
        output.len() - without_space.len() + "<think>".len() + end + "</think>".len()
    } else {
        0
    };
    let visible = &output[search_start..];
    let Some(first) = visible.find(OPEN).map(|offset| search_start + offset) else {
        if looks_like_tool_markup(visible) {
            return Err(ProtocolError::generated(
                "model emitted malformed Qwen3 tool-call markup",
            ));
        }
        return Ok(prose(output));
    };
    if looks_like_tool_markup(&output[search_start..first]) {
        return Err(ProtocolError::generated(
            "model emitted malformed markup before a Qwen3 tool call",
        ));
    }

    let mut calls = Vec::new();
    let mut rest = &output[first..];
    while !rest.trim_start().is_empty() {
        rest = rest.trim_start();
        let Some(body) = rest.strip_prefix(OPEN) else {
            return Err(ProtocolError::generated(
                "model emitted text after a Qwen3 tool call",
            ));
        };

        // Consume a JSON value before looking for the closing tag. A literal
        // </tool_call> inside a JSON string is an argument, not a delimiter.
        let mut values = serde_json::Deserializer::from_str(body).into_iter::<Box<RawValue>>();
        let raw = values
            .next()
            .ok_or_else(|| ProtocolError::generated("Qwen3 tool call is missing its JSON object"))?
            .map_err(|error| {
                ProtocolError::generated(format!("Qwen3 tool call is not valid JSON: {error}"))
            })?;
        let value = OrderedJson::parse(raw.get())
            .map_err(|error| {
                ProtocolError::generated(format!("Qwen3 tool call is not valid JSON: {error}"))
            })?
            .into_value();
        let tail = body[values.byte_offset()..].trim_start();
        rest = tail.strip_prefix(CLOSE).ok_or_else(|| {
            ProtocolError::generated(
                "Qwen3 tool call must end after one JSON object with </tool_call>",
            )
        })?;
        let (name, arguments) = parse_call(value, tools)?;
        calls.push(ToolCall {
            id: format!("call_{}", calls.len() + 1),
            name,
            arguments: serde_json::to_string(&arguments).map_err(|error| {
                ProtocolError::generated(format!("cannot encode tool arguments: {error}"))
            })?,
        });
    }

    let prefix = output[..first].trim();
    Ok(TranslatedOutput {
        content: (!prefix.is_empty()).then(|| prefix.to_owned()),
        reasoning_content: None,
        tool_calls: calls,
    })
}

fn prose(output: &str) -> TranslatedOutput {
    TranslatedOutput {
        content: Some(output.to_owned()),
        reasoning_content: None,
        tool_calls: Vec::new(),
    }
}

fn parse_call(value: Value, tools: &[ToolDefinition]) -> Result<(String, Value), ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::generated("Qwen3 tool call must contain a JSON object"))?;
    if object.len() != 2 || !object.contains_key("name") || !object.contains_key("arguments") {
        return Err(ProtocolError::generated(
            "Qwen3 tool-call object must contain exactly name and arguments",
        ));
    }
    let name = object["name"]
        .as_str()
        .ok_or_else(|| ProtocolError::generated("Qwen3 tool-call name must be a string"))?;
    validate_name(name).map_err(ProtocolError::generated)?;
    let tool = tools
        .iter()
        .find(|tool| tool.function.name == name)
        .ok_or_else(|| {
            ProtocolError::generated(format!("model requested unknown function {name:?}"))
        })?;
    let arguments = &object["arguments"];
    if !arguments.is_object() {
        return Err(ProtocolError::generated(
            "Qwen3 tool-call arguments must be a JSON object",
        ));
    }
    // Unlike the Coder XML dialect, JSON already carries its argument types.
    // Validate those exact values; never coerce a string to a number or boolean.
    validate_value(arguments, tool.function.parameters.as_ref(), "arguments")?;
    Ok((name.to_owned(), arguments.clone()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool() -> ToolDefinition {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "count": {"type": "integer", "minimum": 2},
                        "enabled": {"type": "boolean"},
                        "mode": {"type": "string", "enum": ["create", "replace"]},
                        "metadata": {
                            "type": "object",
                            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                            "required": ["tags"],
                            "additionalProperties": false
                        },
                        "optional": {"type": ["string", "null"]}
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }
            }
        }))
        .unwrap()
    }

    fn wrapped(body: &str) -> String {
        format!("<tool_call>\n{body}\n</tool_call>")
    }

    #[test]
    fn ordinary_text_and_reasoning_only_outputs_are_preserved() {
        for text in [
            "",
            "  ordinary 中文🙂 text\n",
            "<think>consider an example: <tool_call>{invalid}</tool_call></think>\nAnswer.",
            " \n<think>unfinished <tool_call>{\"name\":\"write_file\",\"arguments\":{}}",
        ] {
            assert_eq!(translate(text, &[tool()]).unwrap(), prose(text));
        }
    }

    #[test]
    fn json_arguments_preserve_types_unicode_and_literal_markup() {
        let arguments = json!({
            "path": "中文🙂.rs",
            "content": "  </tool_call> <tool_call> <function=x> \\\"\n  ",
            "count": 9_007_199_254_740_993u64,
            "enabled": true,
            "mode": "create",
            "metadata": {"tags": ["a", "b"]},
            "optional": null
        });
        let output = wrapped(&json!({"name":"write_file", "arguments":arguments}).to_string());
        let result = translate(&output, &[tool()]).unwrap();
        assert_eq!(result.content, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_1");
        assert_eq!(result.tool_calls[0].name, "write_file");
        assert_eq!(
            serde_json::from_str::<Value>(&result.tool_calls[0].arguments).unwrap(),
            arguments
        );
    }

    #[test]
    fn complete_reasoning_examples_do_not_become_calls() {
        let call = wrapped(r#"{"name":"write_file","arguments":{"path":"x","content":"y"}}"#);
        let reasoning = format!(
            "<think>Example only: {call}\nAlso <function=example>.</think>\nI will write it."
        );
        let output = format!(" \n{reasoning}\n{call}\n{call}\n");
        let result = translate(&output, &[tool()]).unwrap();
        assert_eq!(result.content.as_deref(), Some(reasoning.as_str()));
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].id, "call_1");
        assert_eq!(result.tool_calls[1].id, "call_2");
    }

    #[test]
    fn incomplete_reasoning_never_yields_calls_even_with_complete_json() {
        let output = format!(
            "<think>I could use {}",
            wrapped(r#"{"name":"write_file","arguments":{"path":"x","content":"y"}}"#)
        );
        assert_eq!(translate(&output, &[tool()]).unwrap(), prose(&output));
    }

    #[test]
    fn malformed_envelopes_and_wrong_dialect_are_refused() {
        for body in [
            "",
            "null",
            "[]",
            r#"{"name":"write_file"}"#,
            r#"{"name":"write_file","arguments":{},"extra":true}"#,
            r#"{"name":3,"arguments":{}}"#,
            r#"{"name":"write_file","arguments":"{}"}"#,
            r#"{"name":"write_file","arguments":[]}"#,
            r#"{"name":"write_file","arguments":null}"#,
            r#"{"name":"write_file","arguments":{},}"#,
            r#"{"name":"write_file","arguments":{}} {}"#,
            "<function=write_file><parameter=path>x</parameter></function>",
        ] {
            let error = translate(&wrapped(body), &[tool()]).unwrap_err();
            assert_eq!(error.code, "tool_call_parse_error", "{body}");
        }
    }

    #[test]
    fn unknown_tools_and_invalid_names_are_refused() {
        for name in ["missing", "", "bad name", "write_file>"] {
            let output =
                wrapped(&json!({"name":name,"arguments":{"path":"x","content":"y"}}).to_string());
            assert!(translate(&output, &[tool()]).is_err(), "{name}");
        }
        let output = wrapped(r#"{"name":"write_file","arguments":{"path":"x","content":"y"}}"#);
        assert!(translate(&output, &[]).is_err());
    }

    #[test]
    fn schema_validation_uses_json_types_without_xml_coercion() {
        for arguments in [
            json!({"path":"x"}),
            json!({"path":1,"content":"y"}),
            json!({"path":"x","content":"y","extra":0}),
            json!({"path":"x","content":"y","count":"2"}),
            json!({"path":"x","content":"y","count":1}),
            json!({"path":"x","content":"y","enabled":"true"}),
            json!({"path":"x","content":"y","mode":"other"}),
            json!({"path":"x","content":"y","metadata":{}}),
            json!({"path":"x","content":"y","metadata":{"tags":[1]}}),
            json!({"path":"x","content":"y","metadata":{"tags":[],"extra":0}}),
        ] {
            let output = wrapped(&json!({"name":"write_file","arguments":arguments}).to_string());
            assert!(translate(&output, &[tool()]).is_err(), "{arguments}");
        }
    }

    #[test]
    fn duplicate_json_keys_are_refused_at_every_depth() {
        for body in [
            r#"{"name":"write_file","name":"write_file","arguments":{"path":"x","content":"y"}}"#,
            r#"{"name":"write_file","arguments":{"path":"x","path":"z","content":"y"}}"#,
            r#"{"name":"write_file","arguments":{"path":"x","content":"y","metadata":{"tags":[],"tags":[]}}}"#,
            r#"{"name":"write_file","arguments":{"path":"x","content":"y","metadata":{"tags":[{"a":1,"a":2}]}}}"#,
            r#"{"name":"write_file","arguments":{"path":"x","\u0070ath":"z","content":"y"}}"#,
        ] {
            let error = translate(&wrapped(body), &[tool()]).unwrap_err();
            assert!(error.message.contains("duplicate JSON"), "{error:?}");
        }
    }

    #[test]
    fn integer_literals_outside_supported_range_are_not_rounded() {
        let tool: ToolDefinition = serde_json::from_str(
            r#"{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}"#,
        )
        .unwrap();
        for arguments in [
            r#"{"value":18446744073709551616}"#,
            r#"{"value":18446744073709551617}"#,
            r#"{"value":-9223372036854775809}"#,
            r#"{"nested":{"value":18446744073709551617}}"#,
            r#"{"nested":[{"value":-9223372036854775809}]}"#,
        ] {
            let output = wrapped(&format!(r#"{{"name":"f","arguments":{arguments}}}"#));
            let error = translate(&output, std::slice::from_ref(&tool)).unwrap_err();
            assert_eq!(error.code, "tool_call_parse_error");
            assert!(error.message.contains("not valid JSON"), "{error:?}");
        }
    }

    #[test]
    fn negative_integer_zero_keeps_integer_type_at_every_depth() {
        let tool: ToolDefinition = serde_json::from_str(
            r#"{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}"#,
        )
        .unwrap();
        let output = wrapped(
            r#"{"name":"f","arguments":{"zero":-0,"nested":[-0,{"zero":-0}],"float":-0.0}}"#,
        );
        let translated = translate(&output, &[tool]).unwrap();
        let arguments: Value = serde_json::from_str(&translated.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["zero"].as_i64(), Some(0));
        assert_eq!(arguments["nested"][0].as_i64(), Some(0));
        assert_eq!(arguments["nested"][1]["zero"].as_i64(), Some(0));
        assert!(arguments["float"].is_f64());
        assert!(arguments["float"].as_f64().unwrap().is_sign_negative());
    }

    #[test]
    fn truncated_markup_suffix_text_and_mixed_dialects_are_refused() {
        let call = wrapped(r#"{"name":"write_file","arguments":{"path":"x","content":"y"}}"#);
        for output in [
            "<tool_call".to_owned(),
            "<tool_call>{\"name\":".to_owned(),
            "<tool_call>{\"name\":\"write_file\",\"arguments\":{}}".to_owned(),
            "</tool_call>".to_owned(),
            "<function=write_file>".to_owned(),
            format!("{call}\nDone."),
            format!("{call}\n<think>more reasoning</think>"),
            format!("<function=wrong>{call}"),
            format!("{call}\n<tool_call><function=write_file></function></tool_call>"),
        ] {
            assert!(translate(&output, &[tool()]).is_err(), "{output}");
        }
    }
}
