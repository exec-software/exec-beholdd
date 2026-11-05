//! Exact OpenMythos text/tool template and prompt-bound reasoning protocol.
//!
//! Pinned llama.cpp 4da6337767f973e2b4d0797e5b323d77d8565e4a compatibility:
//! C-locale ASCII trim, ordered JSON, six-significant-digit numeric spelling,
//! thinking enabled and history preserved. Native numeric rounding changes only
//! the prompt spelling; the request and model-emitted tool values stay intact.
use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::{
    coerce_value, message_text, ordered_json::OrderedJson, reject_message_metadata, validate_name,
    validate_tools, validate_value, ChatMessage, ProtocolError, Role, ToolCall, ToolDefinition,
    TranslatedOutput, TOOL_FORMAT,
};

#[derive(Clone, Copy)]
pub(crate) struct Options {
    pub(crate) thinking: bool,
    pub(crate) preserve_thinking: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            thinking: true,
            preserve_thinking: true,
        }
    }
}

fn trim(value: &str) -> &str {
    value.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'))
}

fn trim_start(value: &str) -> &str {
    value.trim_start_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'))
}

fn number_spelling(value: f64) -> Result<String, ProtocolError> {
    // The pinned C++ value_float constructor also converts to int64. Keep this
    // deliberately bounded domain explicit instead of copying undefined casts.
    if !value.is_finite()
        || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
    {
        return Err(ProtocolError::messages(
            "OpenMythos tool floats must be finite and within the signed 64-bit conversion domain",
        ));
    }
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific format");
    let exponent: i32 = exponent.parse().expect("scientific exponent");
    let spelling = if (-4..6).contains(&exponent) {
        let precision = (5 - exponent) as usize;
        let fixed = format!("{value:.precision$}");
        if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            fixed
        }
    } else {
        format!(
            "{}e{exponent:+03}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        )
    };
    Ok(spelling)
}

fn native_json(value: &OrderedJson) -> Result<String, ProtocolError> {
    Ok(match value {
        OrderedJson::Scalar(Value::Number(number)) => {
            if number.is_f64() {
                number_spelling(number.as_f64().expect("float"))?
            } else {
                number
                    .as_i64()
                    .ok_or_else(|| {
                        ProtocolError::messages("OpenMythos tool integers must fit signed 64-bit")
                    })?
                    .to_string()
            }
        }
        OrderedJson::Scalar(value) => serde_json::to_string(value).expect("finite scalar"),
        OrderedJson::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(native_json)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        OrderedJson::Object(values) => {
            let mut items = Vec::with_capacity(values.len());
            for (key, value) in values {
                items.push(format!(
                    "{}: {}",
                    serde_json::to_string(key).expect("string"),
                    native_json(value)?
                ));
            }
            format!("{{{}}}", items.join(", "))
        }
    })
}

fn argument_spelling(value: &OrderedJson) -> Result<String, ProtocolError> {
    match value {
        OrderedJson::Scalar(Value::String(value)) => Ok(value.clone()),
        OrderedJson::Scalar(Value::Bool(value)) => Ok(if *value { "True" } else { "False" }.into()),
        OrderedJson::Scalar(Value::Null) => Ok("None".into()),
        _ => native_json(value),
    }
}

fn arguments(call: &super::AssistantToolCall) -> Result<Vec<(String, OrderedJson)>, ProtocolError> {
    match OrderedJson::parse(&call.function.arguments).map_err(|error| {
        ProtocolError::messages(format!("tool arguments are not exact JSON: {error}"))
    })? {
        OrderedJson::Object(values) => Ok(values),
        _ => Err(ProtocolError::messages("tool arguments must be an object")),
    }
}

pub(crate) fn render(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: Options,
) -> Result<String, ProtocolError> {
    if messages.is_empty() {
        return Err(ProtocolError::messages("messages must not be empty"));
    }
    validate_tools(tools)?;
    let leading = messages
        .iter()
        .take(2)
        .take_while(|message| matches!(message.role, Role::System | Role::Developer))
        .count();
    let mut pending = BTreeMap::new();
    for (index, message) in messages.iter().enumerate() {
        reject_message_metadata(
            message,
            message.role == Role::Assistant,
            message.role == Role::Tool,
        )?;
        if message.reasoning_content.is_some() && message.role != Role::Assistant {
            return Err(ProtocolError::messages(
                "reasoning_content is only valid on assistant messages",
            ));
        }
        if matches!(message.role, Role::System | Role::Developer) && index >= leading {
            return Err(ProtocolError::messages("OpenMythos accepts at most two leading system/developer messages; later instructions are not discarded"));
        }
        message_text(message, message.role != Role::Assistant)?;
        if message.role != Role::Tool && !pending.is_empty() {
            return Err(ProtocolError::messages(
                "missing tool results before the next turn",
            ));
        }
        if message.role == Role::Assistant {
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                if trim(&call.id).is_empty()
                    || pending
                        .insert(call.id.clone(), call.function.name.clone())
                        .is_some()
                {
                    return Err(ProtocolError::messages(
                        "pending tool call ids must be nonempty and unique",
                    ));
                }
                validate_name(&call.function.name).map_err(ProtocolError::messages)?;
                for (name, value) in arguments(call)? {
                    validate_name(&name).map_err(ProtocolError::messages)?;
                    argument_spelling(&value)?;
                }
            }
        } else if message.role == Role::Tool {
            let id = message
                .tool_call_id
                .as_ref()
                .ok_or_else(|| ProtocolError::messages("tool requires tool_call_id"))?;
            let name = pending
                .remove(id)
                .ok_or_else(|| ProtocolError::messages("tool result has no pending call"))?;
            if message.name.as_ref().is_some_and(|value| value != &name) {
                return Err(ProtocolError::messages(
                    "tool name does not match pending call",
                ));
            }
        }
    }
    if !pending.is_empty() {
        return Err(ProtocolError::messages("missing tool results"));
    }
    let mut system = Vec::new();
    for message in &messages[..leading] {
        system.push(trim(message_text(message, true)?));
    }
    let system = system.join("\n");
    let last_query = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            let content = trim(
                message
                    .content
                    .as_ref()
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            (message.role == Role::User
                && !(content.starts_with("<tool_response>")
                    && content.ends_with("</tool_response>")))
            .then_some(index)
        })
        .unwrap_or(messages.len() - 1);
    let mut prompt = String::new();
    if !tools.is_empty() {
        prompt.push_str(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
        );
        for tool in tools {
            let value = OrderedJson::parse(&tool.qwen_json)
                .map_err(|error| ProtocolError::tools(error.to_string()))?;
            prompt.push('\n');
            prompt.push_str(
                &native_json(&value).map_err(|error| ProtocolError::tools(error.message))?,
            );
        }
        prompt.push_str("\n</tools>");
        prompt.push_str(TOOL_FORMAT);
        if !system.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&system);
        }
        prompt.push_str("<|im_end|>\n");
    } else if !system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&system);
        prompt.push_str("<|im_end|>\n");
    }
    for (index, message) in messages.iter().enumerate().skip(leading) {
        let mut content = trim(message_text(message, message.role != Role::Assistant)?);
        match message.role {
            Role::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(content);
            }
            Role::Assistant => {
                let reasoning = if let Some(reasoning) = message.reasoning_content.as_deref() {
                    reasoning
                } else if let Some((before, _)) = content.split_once("</think>") {
                    let before = before.trim_end_matches('\n');
                    let reasoning = before
                        .rsplit_once("<think>")
                        .map_or(before, |(_, tail)| tail)
                        .trim_start_matches('\n');
                    content = content
                        .rsplit_once("</think>")
                        .expect("close tag")
                        .1
                        .trim_start_matches('\n');
                    reasoning
                } else {
                    ""
                };
                prompt.push_str("<|im_start|>assistant\n");
                if options.preserve_thinking || index > last_query {
                    prompt.push_str("<think>\n");
                    prompt.push_str(trim(reasoning));
                    prompt.push_str("\n</think>\n\n");
                }
                prompt.push_str(content);
                for (call_index, call) in message
                    .tool_calls
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                {
                    if call_index > 0 {
                        prompt.push('\n');
                    } else if !trim(content).is_empty() {
                        prompt.push_str("\n\n");
                    }
                    prompt.push_str("<tool_call>\n<function=");
                    prompt.push_str(&call.function.name);
                    prompt.push_str(">\n");
                    for (name, value) in arguments(call)? {
                        prompt.push_str("<parameter=");
                        prompt.push_str(&name);
                        prompt.push_str(">\n");
                        prompt.push_str(&argument_spelling(&value)?);
                        prompt.push_str("\n</parameter>\n");
                    }
                    prompt.push_str("</function>\n</tool_call>");
                }
            }
            Role::Tool => {
                if index == 0 || messages[index - 1].role != Role::Tool {
                    prompt.push_str("<|im_start|>user");
                }
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(content);
                prompt.push_str("\n</tool_response>");
                if messages
                    .get(index + 1)
                    .is_some_and(|next| next.role == Role::Tool)
                {
                    continue;
                }
            }
            Role::System | Role::Developer => unreachable!("validated leading instructions"),
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n");
    if !options.thinking {
        prompt.push_str("\n</think>\n\n");
    }
    Ok(prompt)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Fragment {
    Reasoning(String),
    Content(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Reasoning,
    Content,
    Tools,
}

/// Delimiter suffixes are retained across callback boundaries. No tool fragment
/// is released: only finish() returns complete, schema-validated calls.
pub(crate) struct Stream {
    phase: Phase,
    pending: String,
    reasoning: String,
    content: String,
}

impl Stream {
    pub(crate) fn new(thinking: bool) -> Self {
        Self {
            phase: if thinking {
                Phase::Reasoning
            } else {
                Phase::Content
            },
            pending: String::new(),
            reasoning: String::new(),
            content: String::new(),
        }
    }

    fn take_fragment(&mut self, count: usize, fragments: &mut Vec<Fragment>) {
        if count == 0 {
            return;
        }
        let value: String = self.pending.drain(..count).collect();
        match self.phase {
            Phase::Reasoning => {
                self.reasoning.push_str(&value);
                fragments.push(Fragment::Reasoning(value));
            }
            Phase::Content => {
                self.content.push_str(&value);
                fragments.push(Fragment::Content(value));
            }
            Phase::Tools => unreachable!("tool payload is buffered"),
        }
    }

    pub(crate) fn push(&mut self, piece: &str) -> Result<Vec<Fragment>, ProtocolError> {
        self.pending.push_str(piece);
        let mut fragments = Vec::new();
        loop {
            if self.phase == Phase::Tools {
                break;
            }
            let close = (self.phase == Phase::Reasoning)
                .then(|| self.pending.find("</think>"))
                .flatten();
            let call = self.pending.find("<tool_call>");
            if close.is_some_and(|end| call.map_or(true, |start| end < start)) {
                let end = close.expect("close exists");
                self.take_fragment(end, &mut fragments);
                self.pending.drain(.."</think>".len());
                self.phase = Phase::Content;
                continue;
            }
            if let Some(start) = call {
                if self.phase == Phase::Content
                    && super::looks_like_tool_markup(&self.pending[..start])
                {
                    return Err(ProtocolError::generated(
                        "malformed markup before OpenMythos tool call",
                    ));
                }
                self.take_fragment(start, &mut fragments);
                self.phase = Phase::Tools;
                break;
            }
            let markers: &[&str] = if self.phase == Phase::Reasoning {
                &["</think>", "<tool_call>"]
            } else {
                &[
                    "<tool_call>",
                    "</tool_call",
                    "<function=",
                    "</function>",
                    "<parameter=",
                    "</parameter>",
                ]
            };
            let keep = self
                .pending
                .rfind('<')
                .filter(|&at| {
                    markers
                        .iter()
                        .any(|marker| marker.starts_with(&self.pending[at..]))
                })
                .map_or(0, |at| self.pending.len() - at);
            let emit = self.pending.len() - keep;
            if self.phase == Phase::Content && super::looks_like_tool_markup(&self.pending[..emit])
            {
                return Err(ProtocolError::generated(
                    "model emitted malformed OpenMythos tool markup",
                ));
            }
            self.take_fragment(emit, &mut fragments);
            break;
        }
        Ok(fragments)
    }

    pub(crate) fn finish(
        mut self,
        tools: &[ToolDefinition],
    ) -> Result<(TranslatedOutput, Vec<Fragment>), ProtocolError> {
        let mut tail = Vec::new();
        let calls = if self.phase == Phase::Tools {
            parse_calls(&self.pending, tools)?
        } else {
            if self.phase == Phase::Content && super::looks_like_tool_markup(&self.pending) {
                return Err(ProtocolError::generated(
                    "unfinished OpenMythos tool markup",
                ));
            }
            self.take_fragment(self.pending.len(), &mut tail);
            Vec::new()
        };
        Ok((
            TranslatedOutput {
                content: (!self.content.is_empty()).then_some(self.content),
                reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
                tool_calls: calls,
            },
            tail,
        ))
    }
}

pub(crate) fn translate(
    output: &str,
    tools: &[ToolDefinition],
    thinking: bool,
) -> Result<TranslatedOutput, ProtocolError> {
    let mut stream = Stream::new(thinking);
    stream.push(output)?;
    Ok(stream.finish(tools)?.0)
}

fn parse_calls(mut rest: &str, tools: &[ToolDefinition]) -> Result<Vec<ToolCall>, ProtocolError> {
    let mut calls = Vec::new();
    while !trim(rest).is_empty() {
        rest = trim_start(rest);
        rest = rest.strip_prefix("<tool_call>\n<function=").ok_or_else(|| ProtocolError::generated("OpenMythos call requires exact <tool_call>/function newline framing and no suffix"))?;
        let (name, tail) = rest
            .split_once(">\n")
            .ok_or_else(|| ProtocolError::generated("unterminated OpenMythos function opener"))?;
        validate_name(name).map_err(ProtocolError::generated)?;
        let tool = tools
            .iter()
            .find(|tool| tool.function.name == name)
            .ok_or_else(|| {
                ProtocolError::generated(format!("model requested undeclared function {name:?}"))
            })?;
        let schema = tool.function.parameters.as_ref();
        let properties = schema
            .and_then(|value| value.get("properties"))
            .and_then(Value::as_object);
        let mut arguments = Map::new();
        let mut encoded = Vec::new();
        rest = tail;
        while let Some(after) = rest.strip_prefix("<parameter=") {
            let (key, after) = after.split_once(">\n").ok_or_else(|| {
                ProtocolError::generated("unterminated OpenMythos parameter opener")
            })?;
            validate_name(key).map_err(ProtocolError::generated)?;
            if arguments.contains_key(key) {
                return Err(ProtocolError::generated("duplicate OpenMythos parameter"));
            }
            let (raw, after) = after.split_once("\n</parameter>\n").ok_or_else(|| {
                ProtocolError::generated("unterminated OpenMythos parameter value")
            })?;
            let parameter_schema = properties.and_then(|values| values.get(key)).or_else(|| {
                schema
                    .and_then(|value| value.get("additionalProperties"))
                    .filter(|value| value.is_object())
            });
            let value = coerce_value(raw, parameter_schema)?;
            // Keep the same admitted numeric domain as historical rendering. The
            // spelling is discarded: actual generated values are NEVER rounded.
            // Parse structured values directly from the generated JSON to retain
            // nested key order and reject duplicate keys before serde's map can
            // hide them. Schema-declared raw strings retain their exact bytes.
            let encoded_value = if value.is_string() {
                serde_json::to_string(&value).expect("string")
            } else {
                raw.to_owned()
            };
            let ordered = OrderedJson::parse(&encoded_value)
                .map_err(|error| ProtocolError::generated(error.to_string()))?;
            native_json(&ordered).map_err(|error| ProtocolError::generated(error.message))?;
            encoded.push(format!(
                "{}:{}",
                serde_json::to_string(key).expect("string"),
                ordered.to_python_json()
            ));
            arguments.insert(key.to_owned(), value);
            rest = after;
        }
        rest = rest
            .strip_prefix("</function>\n</tool_call>")
            .ok_or_else(|| {
                ProtocolError::generated(
                    "OpenMythos call is incomplete or has malformed trailing parameters",
                )
            })?;
        validate_value(&Value::Object(arguments), schema, "arguments")?;
        calls.push(ToolCall {
            id: format!("call_{}", calls.len() + 1),
            name: name.to_owned(),
            arguments: format!("{{{}}}", encoded.join(",")),
        });
    }
    Ok(calls)
}

#[cfg(test)]
#[path = "openmythos_tests.rs"]
mod tests;
