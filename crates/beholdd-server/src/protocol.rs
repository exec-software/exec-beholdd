use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use beholdd_runtime::ChatTemplate;
use serde::Deserialize;
use serde_json::{Map, Value};

#[path = "protocol/openmythos.rs"]
pub(crate) mod openmythos;
#[path = "protocol/ordered_json.rs"]
mod ordered_json;
#[path = "protocol/qwen3_output.rs"]
mod qwen3_output;

pub(crate) const QWEN3_CHAT_TEMPLATE: &str = include_str!("templates/qwen3-0.6b.jinja");
pub(crate) const QWEN3_CODER_CHAT_TEMPLATE: &str = include_str!("templates/qwen3-coder-30b.jinja");
pub(crate) const OPENMYTHOS_CHAT_TEMPLATE: &str = include_str!("templates/openmythos-27b.jinja");

pub(crate) const DEFAULT_MAX_TOKENS: usize = 512;
pub(crate) const MAX_TOKENS: usize = 8_192;

const DEFAULT_TOOL_SYSTEM: &str =
    "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";
const TOOL_INSTRUCTIONS: &str = "\n\nYou have access to the following functions:\n\n<tools>";
const TOOL_FORMAT: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) temperature: Option<f64>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) enable_thinking: Option<bool>,
    #[serde(default)]
    pub(crate) preserve_thinking: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatMessage {
    pub(crate) role: Role,
    #[serde(default)]
    pub(crate) content: Option<Value>,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<AssistantToolCall>>,
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AssistantToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) _kind: FunctionKind,
    pub(crate) function: AssistantFunctionCall,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum FunctionKind {
    Function,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AssistantFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolDefinition {
    pub(crate) _kind: FunctionKind,
    pub(crate) function: FunctionDefinition,
    // Full tool object, including key order and explicit null/omitted fields.
    qwen_json: String,
}

impl<'de> Deserialize<'de> for ToolDefinition {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Definition {
            #[serde(rename = "type")]
            kind: FunctionKind,
            function: FunctionDefinition,
        }
        let raw = Box::<serde_json::value::RawValue>::deserialize(deserializer)?;
        let ordered =
            ordered_json::OrderedJson::parse(raw.get()).map_err(serde::de::Error::custom)?;
        let qwen_json = ordered.to_python_json();
        let definition: Definition =
            serde_json::from_str(&qwen_json).map_err(serde::de::Error::custom)?;
        Ok(Self {
            _kind: definition.kind,
            function: definition.function,
            qwen_json,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: Option<String>,
    #[serde(default)]
    pub(crate) parameters: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranslatedOutput {
    pub(crate) content: Option<String>,
    pub(crate) reasoning_content: Option<String>,
    pub(crate) tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProtocolError {
    pub(crate) message: String,
    pub(crate) param: Option<&'static str>,
    pub(crate) code: &'static str,
}

impl ProtocolError {
    fn messages(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: Some("messages"),
            code: "invalid_request",
        }
    }

    fn unsupported_message_content(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: Some("messages"),
            code: "unsupported_content_type",
        }
    }

    fn tools(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: Some("tools"),
            code: "invalid_request",
        }
    }

    fn generated(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: None,
            code: "tool_call_parse_error",
        }
    }
}

/// Normalize OpenAI text content parts before the text-only prompt renderer runs.
///
/// Image parts are recognized so current completion-only models return a specific
/// capability error rather than the old generic "content must be a string" failure.
pub(crate) fn normalize_message_content(
    messages: &mut [ChatMessage],
    model: &str,
) -> Result<(), ProtocolError> {
    for (message_index, message) in messages.iter_mut().enumerate() {
        let Some(Value::Array(parts)) = message.content.as_ref() else {
            continue;
        };
        let mut text = String::new();
        for (part_index, part) in parts.iter().enumerate() {
            let path = format!("messages[{message_index}].content[{part_index}]");
            let object = part
                .as_object()
                .ok_or_else(|| ProtocolError::messages(format!("{path} must be an object")))?;
            let kind = object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ProtocolError::messages(format!("{path}.type must be a string")))?;
            match kind {
                "text" => {
                    if object.len() != 2 || !object.contains_key("text") {
                        return Err(ProtocolError::messages(format!(
                            "{path} text parts must contain exactly type and text"
                        )));
                    }
                    let part_text =
                        object.get("text").and_then(Value::as_str).ok_or_else(|| {
                            ProtocolError::messages(format!("{path}.text must be a string"))
                        })?;
                    text.push_str(part_text);
                }
                "image_url" => {
                    return Err(ProtocolError::unsupported_message_content(format!(
                        "{path}: model {model:?} does not support image input"
                    )));
                }
                other => {
                    return Err(ProtocolError::unsupported_message_content(format!(
                        "{path}: content part type {other:?} is unsupported"
                    )));
                }
            }
        }
        message.content = Some(Value::String(text));
    }
    Ok(())
}

/// Select only an exact supported checkpoint template. Legacy is an explicit
/// alternate-runtime contract, never a fallback for missing native metadata.
pub(crate) fn prepare_chat(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    template: &ChatTemplate,
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
) -> Result<(String, OutputDialect), ProtocolError> {
    if matches!(template, ChatTemplate::Checkpoint(Some(value)) if value == OPENMYTHOS_CHAT_TEMPLATE)
    {
        let options = openmythos::Options {
            thinking: enable_thinking.unwrap_or(true),
            preserve_thinking: preserve_thinking.unwrap_or(true),
        };
        return Ok((
            openmythos::render(messages, tools, options)?,
            OutputDialect::OpenMythos {
                thinking: options.thinking,
            },
        ));
    }
    if enable_thinking.is_some() || preserve_thinking.is_some() {
        return Err(ProtocolError::messages(
            "thinking options are unsupported for this checkpoint template",
        ));
    }
    Ok((
        render_prompt_for_template(messages, tools, template)?,
        output_dialect_for_template(template)?,
    ))
}

pub(crate) fn render_prompt_for_template(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    template: &ChatTemplate,
) -> Result<String, ProtocolError> {
    if matches!(template, ChatTemplate::Checkpoint(Some(value)) if value == OPENMYTHOS_CHAT_TEMPLATE)
    {
        return openmythos::render(messages, tools, openmythos::Options::default());
    }
    match template {
        ChatTemplate::Legacy => render_prompt(messages, tools),
        ChatTemplate::Checkpoint(Some(template)) if template == QWEN3_CHAT_TEMPLATE => {
            render_qwen3(messages, tools)
        }
        ChatTemplate::Checkpoint(Some(template)) if template == QWEN3_CODER_CHAT_TEMPLATE => {
            render_prompt(messages, tools)
        }
        ChatTemplate::Checkpoint(_) => Err(ProtocolError {
            message: "checkpoint chat template is missing or unsupported".into(),
            param: Some("model"),
            code: "unsupported_chat_template",
        }),
    }
}

/// The exact Qwen3 template's supported text/tool subset, with thinking mode left
/// unspecified. Preserve wire messages; normalization applies only to rendering.
fn render_qwen3(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) -> Result<String, ProtocolError> {
    reject_openmythos_metadata(messages)?;
    if messages.is_empty() {
        return Err(ProtocolError::messages("messages must not be empty"));
    }
    validate_tools(tools)?;
    let mut pending = BTreeMap::new();
    for (index, message) in messages.iter().enumerate() {
        reject_message_metadata(
            message,
            message.role == Role::Assistant,
            message.role == Role::Tool,
        )?;
        if message.role == Role::System && index != 0 {
            return Err(ProtocolError::messages(format!(
                "messages[{index}]: system is only valid as the first message"
            )));
        }
        message_text(message, message.role != Role::Assistant)?;
        if message.role != Role::Tool && !pending.is_empty() {
            return Err(ProtocolError::messages(
                "missing tool results before the next turn",
            ));
        }
        if message.role == Role::Assistant {
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                if call.id.trim().is_empty()
                    || pending
                        .insert(call.id.clone(), call.function.name.clone())
                        .is_some()
                {
                    return Err(ProtocolError::messages(
                        "tool call ids must be non-empty and unique",
                    ));
                }
                validate_name(&call.function.name).map_err(ProtocolError::messages)?;
                let arguments = ordered_json::OrderedJson::parse(&call.function.arguments)
                    .map_err(|error| {
                        ProtocolError::messages(format!("tool arguments are not JSON: {error}"))
                    })?;
                if !matches!(arguments, ordered_json::OrderedJson::Object(_)) {
                    return Err(ProtocolError::messages(
                        "tool arguments must be a JSON object",
                    ));
                }
            }
        } else if message.role == Role::Tool {
            let id = message
                .tool_call_id
                .as_deref()
                .ok_or_else(|| ProtocolError::messages("tool requires tool_call_id"))?;
            let name = pending.remove(id).ok_or_else(|| {
                ProtocolError::messages(format!("tool_call_id {id:?} has no pending call"))
            })?;
            if message
                .name
                .as_ref()
                .is_some_and(|provided| provided != &name)
            {
                return Err(ProtocolError::messages(
                    "tool name does not match pending call",
                ));
            }
        }
    }
    if !pending.is_empty() {
        return Err(ProtocolError::messages("missing tool results"));
    }
    // Match the template's last non-tool-response user query, including its
    // fallback when there is no such user. User content is already validated.
    let last_query = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            let content = message
                .content
                .as_ref()
                .and_then(Value::as_str)
                .unwrap_or("");
            (message.role == Role::User
                && !(content.starts_with("<tool_response>")
                    && content.ends_with("</tool_response>")))
            .then_some(index)
        })
        .unwrap_or(messages.len() - 1);
    let mut prompt = String::new();
    if !tools.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        if messages[0].role == Role::System {
            prompt.push_str(message_text(&messages[0], true)?);
            prompt.push_str("\n\n");
        }
        prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(&tool.qwen_json);
        }
        prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    }
    for (index, message) in messages.iter().enumerate() {
        if index == 0 && message.role == Role::System && !tools.is_empty() {
            continue;
        }
        let raw = message_text(message, message.role != Role::Assistant)?;
        match message.role {
            Role::Developer => unreachable!("unsupported role refused before rendering"),
            Role::System | Role::User => {
                let role = if message.role == Role::System {
                    "system"
                } else {
                    "user"
                };
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(raw);
            }
            Role::Assistant => {
                let (content, reasoning) = if let Some((before, _)) = raw.split_once("</think>") {
                    let content = raw
                        .rsplit_once("</think>")
                        .expect("closing tag exists")
                        .1
                        .trim_start_matches('\n');
                    let before = before.trim_end_matches('\n');
                    let reasoning = before
                        .rsplit_once("<think>")
                        .map_or(before, |(_, tail)| tail)
                        .trim_start_matches('\n');
                    (content, reasoning)
                } else {
                    (raw, "")
                };
                prompt.push_str("<|im_start|>assistant\n");
                if index > last_query
                    && (index + 1 == messages.len()
                        || !reasoning
                            .trim_matches(|c: char| {
                                c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
                            })
                            .is_empty())
                {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning.trim_matches('\n'));
                    prompt.push_str("\n</think>\n\n");
                    prompt.push_str(content.trim_start_matches('\n'));
                } else {
                    prompt.push_str(content);
                }
                for (call_index, call) in message
                    .tool_calls
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                {
                    if call_index > 0 || !content.is_empty() {
                        prompt.push('\n');
                    }
                    prompt.push_str("<tool_call>\n{\"name\": \"");
                    prompt.push_str(&call.function.name);
                    prompt.push_str("\", \"arguments\": ");
                    prompt.push_str(&call.function.arguments);
                    prompt.push_str("}\n</tool_call>");
                }
            }
            Role::Tool => {
                if index == 0 || messages[index - 1].role != Role::Tool {
                    prompt.push_str("<|im_start|>user");
                }
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(raw);
                prompt.push_str("\n</tool_response>");
                if messages
                    .get(index + 1)
                    .is_some_and(|next| next.role == Role::Tool)
                {
                    continue;
                }
            }
        }
        prompt.push_str("<|im_end|>\n");
    }
    // The request API has no enable_thinking option. /no_think remains ordinary
    // user text, so do not inject the template's explicitly-disabled prefix.
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum OutputDialect {
    CoderXml,
    Qwen3Json,
    OpenMythos { thinking: bool },
}

pub(crate) fn output_dialect_for_template(
    template: &ChatTemplate,
) -> Result<OutputDialect, ProtocolError> {
    match template {
        ChatTemplate::Checkpoint(Some(template)) if template == OPENMYTHOS_CHAT_TEMPLATE => {
            Ok(OutputDialect::OpenMythos { thinking: true })
        }
        ChatTemplate::Legacy => Ok(OutputDialect::CoderXml),
        ChatTemplate::Checkpoint(Some(template)) if template == QWEN3_CODER_CHAT_TEMPLATE => {
            Ok(OutputDialect::CoderXml)
        }
        ChatTemplate::Checkpoint(Some(template)) if template == QWEN3_CHAT_TEMPLATE => {
            Ok(OutputDialect::Qwen3Json)
        }
        ChatTemplate::Checkpoint(_) => Err(ProtocolError {
            message: "checkpoint chat template is missing or unsupported".into(),
            param: Some("model"),
            code: "unsupported_chat_template",
        }),
    }
}

pub(crate) fn translate_output_for_dialect(
    output: &str,
    tools: &[ToolDefinition],
    dialect: OutputDialect,
) -> Result<TranslatedOutput, ProtocolError> {
    match dialect {
        OutputDialect::CoderXml => translate_output(output, tools),
        OutputDialect::Qwen3Json => qwen3_output::translate(output, tools),
        OutputDialect::OpenMythos { thinking } => openmythos::translate(output, tools, thinking),
    }
}

pub(crate) fn render_prompt(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) -> Result<String, ProtocolError> {
    reject_openmythos_metadata(messages)?;
    if messages.is_empty() {
        return Err(ProtocolError::messages("messages must not be empty"));
    }
    validate_tools(tools)?;

    let mut prompt = String::new();
    let mut at = 0;
    if tools.is_empty() {
        if messages[0].role == Role::System {
            let content = message_text(&messages[0], true)?;
            reject_message_metadata(&messages[0], false, false)?;
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
            at = 1;
        }
    } else {
        let system = if messages[0].role == Role::System {
            reject_message_metadata(&messages[0], false, false)?;
            at = 1;
            message_text(&messages[0], true)?
        } else {
            DEFAULT_TOOL_SYSTEM
        };
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str(TOOL_INSTRUCTIONS);
        for tool in tools {
            render_tool_definition(&mut prompt, tool)?;
        }
        prompt.push_str("\n</tools>");
        prompt.push_str(TOOL_FORMAT);
        prompt.push_str("<|im_end|>\n");
    }

    let mut pending_calls: BTreeMap<String, String> = BTreeMap::new();
    let mut in_tool_results = false;
    for (index, message) in messages.iter().enumerate().skip(at) {
        if message.role == Role::System {
            return Err(ProtocolError::messages(format!(
                "messages[{index}]: system is only valid as the first message"
            )));
        }

        if message.role != Role::Tool && !pending_calls.is_empty() {
            return Err(ProtocolError::messages(format!(
                "messages[{index}]: missing tool results for {}",
                pending_calls.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }

        match message.role {
            Role::System | Role::Developer => unreachable!(),
            Role::User => {
                reject_message_metadata(message, false, false)?;
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(message_text(message, true)?);
                prompt.push_str("<|im_end|>\n");
                in_tool_results = false;
            }
            Role::Assistant => {
                reject_message_metadata(message, true, false)?;
                prompt.push_str("<|im_start|>assistant");
                let calls = message.tool_calls.as_deref().unwrap_or_default();
                if calls.is_empty() {
                    prompt.push('\n');
                    prompt.push_str(message_text(message, true)?);
                } else {
                    if let Some(content) = optional_message_text(message)? {
                        let content = content.trim();
                        if !content.is_empty() {
                            prompt.push('\n');
                            prompt.push_str(content);
                        }
                    }
                    let mut ids = BTreeSet::new();
                    for call in calls {
                        if call.id.trim().is_empty() || !ids.insert(call.id.clone()) {
                            return Err(ProtocolError::messages(format!(
                                "messages[{index}]: tool call ids must be non-empty and unique"
                            )));
                        }
                        validate_name(&call.function.name).map_err(ProtocolError::messages)?;
                        let arguments: Value = serde_json::from_str(&call.function.arguments)
                            .map_err(|error| {
                                ProtocolError::messages(format!(
                                    "messages[{index}]: tool arguments are not JSON: {error}"
                                ))
                            })?;
                        let arguments = arguments.as_object().ok_or_else(|| {
                            ProtocolError::messages(format!(
                                "messages[{index}]: tool arguments must be a JSON object"
                            ))
                        })?;
                        prompt.push_str("\n<tool_call>\n<function=");
                        prompt.push_str(&call.function.name);
                        prompt.push('>');
                        for (name, value) in arguments {
                            validate_name(name).map_err(ProtocolError::messages)?;
                            prompt.push_str("\n<parameter=");
                            prompt.push_str(name);
                            prompt.push_str(">\n");
                            match value {
                                Value::String(value) => prompt.push_str(value),
                                other => prompt.push_str(&compact_json(other)?),
                            }
                            prompt.push_str("\n</parameter>");
                        }
                        prompt.push_str("\n</function>\n</tool_call>");
                        pending_calls.insert(call.id.clone(), call.function.name.clone());
                    }
                }
                prompt.push_str("<|im_end|>\n");
                in_tool_results = false;
            }
            Role::Tool => {
                reject_message_metadata(message, false, true)?;
                let call_id = message.tool_call_id.as_deref().ok_or_else(|| {
                    ProtocolError::messages(format!(
                        "messages[{index}]: tool requires tool_call_id"
                    ))
                })?;
                let expected_name = pending_calls.remove(call_id).ok_or_else(|| {
                    ProtocolError::messages(format!(
                        "messages[{index}]: tool_call_id {call_id:?} has no pending call"
                    ))
                })?;
                if let Some(name) = message.name.as_deref() {
                    if name != expected_name {
                        return Err(ProtocolError::messages(format!(
                            "messages[{index}]: tool name {name:?} does not match {expected_name:?}"
                        )));
                    }
                }
                if !in_tool_results {
                    prompt.push_str("<|im_start|>user\n");
                }
                prompt.push_str("<tool_response>\n");
                prompt.push_str(message_text(message, true)?);
                prompt.push_str("\n</tool_response>\n");
                in_tool_results = true;

                let next_is_tool = messages
                    .get(index + 1)
                    .is_some_and(|next| next.role == Role::Tool);
                if !next_is_tool {
                    if !pending_calls.is_empty() {
                        return Err(ProtocolError::messages(format!(
                            "messages[{index}]: missing tool results for {}",
                            pending_calls.keys().cloned().collect::<Vec<_>>().join(", ")
                        )));
                    }
                    prompt.push_str("<|im_end|>\n");
                    in_tool_results = false;
                }
            }
        }
    }

    if !pending_calls.is_empty() {
        return Err(ProtocolError::messages(format!(
            "missing tool results for {}",
            pending_calls.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    if in_tool_results {
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

fn reject_openmythos_metadata(messages: &[ChatMessage]) -> Result<(), ProtocolError> {
    if messages
        .iter()
        .any(|message| message.role == Role::Developer || message.reasoning_content.is_some())
    {
        return Err(ProtocolError::messages(
            "developer and reasoning_content are unsupported for this checkpoint template",
        ));
    }
    Ok(())
}

fn reject_message_metadata(
    message: &ChatMessage,
    allow_calls: bool,
    allow_tool_metadata: bool,
) -> Result<(), ProtocolError> {
    if !allow_calls && message.tool_calls.is_some() {
        return Err(ProtocolError::messages(
            "tool_calls is only valid on assistant messages",
        ));
    }
    if !allow_tool_metadata && (message.tool_call_id.is_some() || message.name.is_some()) {
        return Err(ProtocolError::messages(
            "name and tool_call_id are only valid on tool messages",
        ));
    }
    Ok(())
}

fn message_text(message: &ChatMessage, required: bool) -> Result<&str, ProtocolError> {
    match message.content.as_ref() {
        Some(Value::String(content)) => Ok(content),
        None | Some(Value::Null) if !required => Ok(""),
        None | Some(Value::Null) => Err(ProtocolError::messages(format!(
            "{:?} message content must be a string",
            message.role
        ))),
        Some(_) => Err(ProtocolError::messages(
            "only string or null message content is supported",
        )),
    }
}

fn optional_message_text(message: &ChatMessage) -> Result<Option<&str>, ProtocolError> {
    match message.content.as_ref() {
        Some(Value::String(content)) => Ok(Some(content)),
        None | Some(Value::Null) => Ok(None),
        Some(_) => Err(ProtocolError::messages(
            "only string or null message content is supported",
        )),
    }
}

fn validate_tools(tools: &[ToolDefinition]) -> Result<(), ProtocolError> {
    let mut names = BTreeSet::new();
    for tool in tools {
        validate_name(&tool.function.name).map_err(ProtocolError::tools)?;
        if !names.insert(tool.function.name.clone()) {
            return Err(ProtocolError::tools(format!(
                "duplicate function name {:?}",
                tool.function.name
            )));
        }
        if let Some(parameters) = tool.function.parameters.as_ref() {
            validate_schema(parameters, "parameters", true)?;
        }
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(format!(
            "function and parameter names must match [A-Za-z0-9_.-]{{1,128}}: {name:?}"
        ));
    }
    Ok(())
}

fn validate_schema(schema: &Value, path: &str, top_level: bool) -> Result<(), ProtocolError> {
    let object = schema
        .as_object()
        .ok_or_else(|| ProtocolError::tools(format!("{path} must be a JSON Schema object")))?;
    const SUPPORTED: &[&str] = &[
        "type",
        "description",
        "enum",
        "properties",
        "items",
        "additionalProperties",
        "required",
        "minimum",
        "maximum",
        "maxItems",
    ];
    for key in object.keys() {
        if !SUPPORTED.contains(&key.as_str()) {
            return Err(ProtocolError::tools(format!(
                "{path}: unsupported JSON Schema keyword {key:?}"
            )));
        }
    }
    if top_level && object.get("type").and_then(Value::as_str) != Some("object") {
        return Err(ProtocolError::tools(format!(
            "{path}.type must be \"object\""
        )));
    }
    if let Some(kind) = object.get("type") {
        let valid = match kind {
            Value::String(kind) => schema_type_is_supported(kind),
            Value::Array(kinds) => {
                !kinds.is_empty()
                    && kinds
                        .iter()
                        .all(|kind| kind.as_str().is_some_and(schema_type_is_supported))
            }
            _ => false,
        };
        if !valid {
            return Err(ProtocolError::tools(format!(
                "{path}.type is not a supported JSON Schema type"
            )));
        }
    }
    if object
        .get("description")
        .is_some_and(|value| !value.is_string())
    {
        return Err(ProtocolError::tools(format!(
            "{path}.description must be a string"
        )));
    }
    for keyword in ["minimum", "maximum"] {
        if object
            .get(keyword)
            .is_some_and(|bound| !bound.as_f64().is_some_and(f64::is_finite))
        {
            return Err(ProtocolError::tools(format!(
                "{path}.{keyword} must be a finite JSON number"
            )));
        }
    }
    if object.get("maxItems").is_some_and(|bound| {
        // JSON Schema integers describe the mathematical value, so 2.0 and
        // exponent forms with no fractional part are valid count limits too.
        bound.as_u64().is_none()
            && !bound
                .as_f64()
                .is_some_and(|number| number.is_finite() && number >= 0.0 && number.fract() == 0.0)
    }) {
        return Err(ProtocolError::tools(format!(
            "{path}.maxItems must be a nonnegative integer"
        )));
    }
    if let Some(values) = object.get("enum") {
        let values = values.as_array().ok_or_else(|| {
            ProtocolError::tools(format!("{path}.enum must be an array of strings"))
        })?;
        if values.iter().any(|value| !value.is_string()) {
            return Err(ProtocolError::tools(format!(
                "{path}.enum must be an array of strings"
            )));
        }
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| ProtocolError::tools(format!("{path}.properties must be an object")))?;
        for (name, property) in properties {
            validate_name(name).map_err(ProtocolError::tools)?;
            validate_schema(property, &format!("{path}.properties.{name}"), false)?;
        }
    }
    if let Some(items) = object.get("items") {
        validate_schema(items, &format!("{path}.items"), false)?;
    }
    if let Some(additional) = object.get("additionalProperties") {
        match additional {
            Value::Bool(_) => {}
            Value::Object(_) => {
                validate_schema(additional, &format!("{path}.additionalProperties"), false)?;
            }
            _ => {
                return Err(ProtocolError::tools(format!(
                    "{path}.additionalProperties must be a boolean or schema"
                )))
            }
        }
    }
    if let Some(required) = object.get("required") {
        let required = required.as_array().ok_or_else(|| {
            ProtocolError::tools(format!("{path}.required must be an array of strings"))
        })?;
        if required.iter().any(|value| !value.is_string()) {
            return Err(ProtocolError::tools(format!(
                "{path}.required must be an array of strings"
            )));
        }
        let mut names = BTreeSet::new();
        for required in required.iter().filter_map(Value::as_str) {
            if !names.insert(required) {
                return Err(ProtocolError::tools(format!(
                    "{path}.required contains duplicate name {required:?}"
                )));
            }
            if let Some(properties) = object.get("properties").and_then(Value::as_object) {
                if !properties.contains_key(required) {
                    return Err(ProtocolError::tools(format!(
                        "{path}.required names undeclared property {required:?}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn schema_type_is_supported(kind: &str) -> bool {
    matches!(
        kind,
        "string" | "integer" | "number" | "boolean" | "object" | "array" | "null"
    )
}

fn render_tool_definition(prompt: &mut String, tool: &ToolDefinition) -> Result<(), ProtocolError> {
    prompt.push_str("\n<function>\n<name>");
    prompt.push_str(&tool.function.name);
    prompt.push_str("</name>\n<description>");
    prompt.push_str(
        tool.function
            .description
            .as_deref()
            .unwrap_or_default()
            .trim(),
    );
    prompt.push_str("</description>");
    prompt.push_str("\n<parameters>");
    let parameters = tool.function.parameters.as_ref().and_then(Value::as_object);
    if let Some(properties) = parameters
        .and_then(|parameters| parameters.get("properties"))
        .and_then(Value::as_object)
    {
        for (name, schema) in properties {
            prompt.push_str("\n<parameter>\n<name>");
            prompt.push_str(name);
            prompt.push_str("</name>");
            let schema = schema
                .as_object()
                .expect("schema validated before rendering");
            for key in ["type", "description", "enum"] {
                if let Some(value) = schema.get(key) {
                    render_schema_tag(prompt, key, value)?;
                }
            }
            for (key, value) in schema {
                if matches!(key.as_str(), "type" | "description" | "enum" | "required") {
                    continue;
                }
                render_schema_tag(prompt, &normalize_tag(key), value)?;
            }
            if let Some(required) = schema.get("required") {
                render_required_tag(prompt, required)?;
            }
            prompt.push_str("\n</parameter>");
        }
    }
    if let Some(required) = parameters.and_then(|parameters| parameters.get("required")) {
        render_required_tag(prompt, required)?;
    }
    prompt.push_str("\n</parameters>\n</function>");
    Ok(())
}

fn render_schema_tag(prompt: &mut String, tag: &str, value: &Value) -> Result<(), ProtocolError> {
    if tag == "enum" {
        return render_backtick_list(prompt, "enum", value);
    }
    prompt.push_str("\n<");
    prompt.push_str(tag);
    prompt.push('>');
    match value {
        Value::String(value) => prompt.push_str(value),
        other => prompt.push_str(&compact_json(other)?),
    }
    prompt.push_str("</");
    prompt.push_str(tag);
    prompt.push('>');
    Ok(())
}

fn render_required_tag(prompt: &mut String, required: &Value) -> Result<(), ProtocolError> {
    render_backtick_list(prompt, "required", required)
}

fn render_backtick_list(
    prompt: &mut String,
    tag: &str,
    values: &Value,
) -> Result<(), ProtocolError> {
    let values = values.as_array().expect("list validated before rendering");
    prompt.push_str("\n<");
    prompt.push_str(tag);
    prompt.push_str(">[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            prompt.push_str(", ");
        }
        prompt.push('`');
        prompt.push_str(value.as_str().expect("list entries validated"));
        prompt.push('`');
    }
    prompt.push_str("]</");
    prompt.push_str(tag);
    prompt.push('>');
    Ok(())
}

fn normalize_tag(key: &str) -> String {
    key.chars()
        .filter_map(|character| match character {
            '$' => None,
            '-' | ' ' => Some('_'),
            other => Some(other),
        })
        .collect()
}

fn compact_json(value: &Value) -> Result<String, ProtocolError> {
    serde_json::to_string(value)
        .map_err(|error| ProtocolError::tools(format!("cannot encode JSON Schema: {error}")))
}

pub(crate) fn translate_output(
    output: &str,
    tools: &[ToolDefinition],
) -> Result<TranslatedOutput, ProtocolError> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let Some(first) = output.find(OPEN) else {
        if looks_like_tool_markup(output) {
            return Err(ProtocolError::generated(
                "model emitted malformed native tool-call markup",
            ));
        }
        return Ok(TranslatedOutput {
            content: Some(output.to_owned()),
            reasoning_content: None,
            tool_calls: Vec::new(),
        });
    };

    let prefix = output[..first].trim();
    let mut calls = Vec::new();
    let mut rest = &output[first..];
    while !rest.trim_start().is_empty() {
        rest = rest.trim_start();
        if !rest.starts_with(OPEN) {
            return Err(ProtocolError::generated(
                "model emitted text after a native tool call",
            ));
        }
        let body_start = OPEN.len();
        let close_at = rest[body_start..]
            .find(CLOSE)
            .ok_or_else(|| ProtocolError::generated("model emitted an unterminated <tool_call>"))?
            + body_start;
        let body = &rest[body_start..close_at];
        let (name, arguments) = parse_native_function(body, tools)?;
        calls.push(ToolCall {
            id: format!("call_{}", calls.len() + 1),
            name,
            arguments: serde_json::to_string(&Value::Object(arguments)).map_err(|error| {
                ProtocolError::generated(format!("cannot encode tool arguments: {error}"))
            })?,
        });
        rest = &rest[close_at + CLOSE.len()..];
    }
    if calls.is_empty() {
        return Err(ProtocolError::generated(
            "model emitted an empty native tool-call response",
        ));
    }
    Ok(TranslatedOutput {
        content: (!prefix.is_empty()).then(|| prefix.to_owned()),
        reasoning_content: None,
        tool_calls: calls,
    })
}

fn looks_like_tool_markup(output: &str) -> bool {
    [
        "<tool_call",
        "</tool_call",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ]
    .iter()
    .any(|marker| output.contains(marker))
}

fn parse_native_function(
    body: &str,
    tools: &[ToolDefinition],
) -> Result<(String, Map<String, Value>), ProtocolError> {
    const FUNCTION: &str = "<function=";
    const FUNCTION_CLOSE: &str = "</function>";
    const PARAMETER: &str = "<parameter=";
    const PARAMETER_CLOSE: &str = "</parameter>";

    let body = body.trim();
    if !body.starts_with(FUNCTION) {
        return Err(ProtocolError::generated(
            "<tool_call> must contain one <function=name> block",
        ));
    }
    let name_end = body[FUNCTION.len()..]
        .find('>')
        .map(|offset| offset + FUNCTION.len())
        .ok_or_else(|| ProtocolError::generated("unterminated <function=name> tag"))?;
    let name = body[FUNCTION.len()..name_end].trim();
    validate_name(name).map_err(ProtocolError::generated)?;
    let after_name = &body[name_end + 1..];
    let close_at = after_name.rfind(FUNCTION_CLOSE).ok_or_else(|| {
        ProtocolError::generated(format!("function {name:?} is missing </function>"))
    })?;
    if !after_name[close_at + FUNCTION_CLOSE.len()..]
        .trim()
        .is_empty()
    {
        return Err(ProtocolError::generated(
            "unexpected markup after </function>",
        ));
    }
    let mut parameters = &after_name[..close_at];
    let tool = tools
        .iter()
        .find(|tool| tool.function.name == name)
        .ok_or_else(|| {
            ProtocolError::generated(format!("model requested unknown function {name:?}"))
        })?;
    let schema = tool.function.parameters.as_ref().and_then(Value::as_object);
    let properties = schema
        .and_then(|schema| schema.get("properties"))
        .and_then(Value::as_object);
    let mut arguments = Map::new();

    while !parameters.trim().is_empty() {
        parameters = parameters.trim_start();
        if !parameters.starts_with(PARAMETER) {
            return Err(ProtocolError::generated(format!(
                "function {name:?} contains malformed parameter markup"
            )));
        }
        let parameter_end = parameters[PARAMETER.len()..]
            .find('>')
            .map(|offset| offset + PARAMETER.len())
            .ok_or_else(|| ProtocolError::generated("unterminated <parameter=name> tag"))?;
        let parameter_name = parameters[PARAMETER.len()..parameter_end].trim();
        validate_name(parameter_name).map_err(ProtocolError::generated)?;
        let after_parameter = &parameters[parameter_end + 1..];
        let value_end = after_parameter.find(PARAMETER_CLOSE).ok_or_else(|| {
            ProtocolError::generated(format!(
                "parameter {parameter_name:?} is missing </parameter>"
            ))
        })?;
        let raw_value = strip_native_value_framing(&after_parameter[..value_end]);
        if arguments.contains_key(parameter_name) {
            return Err(ProtocolError::generated(format!(
                "function {name:?} repeats parameter {parameter_name:?}"
            )));
        }
        let declared_schema = properties.and_then(|properties| properties.get(parameter_name));
        let parameter_schema = if declared_schema.is_some() {
            declared_schema
        } else {
            match schema.and_then(|schema| schema.get("additionalProperties")) {
                Some(Value::Bool(false)) => {
                    return Err(ProtocolError::generated(format!(
                        "function {name:?} has unknown parameter {parameter_name:?}"
                    )))
                }
                Some(Value::Object(_)) => {
                    schema.and_then(|schema| schema.get("additionalProperties"))
                }
                _ => None,
            }
        };
        let value = coerce_value(raw_value, parameter_schema)?;
        arguments.insert(parameter_name.to_owned(), value);
        parameters = &after_parameter[value_end + PARAMETER_CLOSE.len()..];
    }

    if let Some(required) = schema
        .and_then(|schema| schema.get("required"))
        .and_then(Value::as_array)
    {
        for required in required {
            let required = required.as_str().expect("required entries validated");
            if !arguments.contains_key(required) {
                return Err(ProtocolError::generated(format!(
                    "function {name:?} is missing required parameter {required:?}"
                )));
            }
        }
    }
    Ok((name.to_owned(), arguments))
}

fn coerce_value(raw: &str, schema: Option<&Value>) -> Result<Value, ProtocolError> {
    let allowed_types = schema
        .and_then(Value::as_object)
        .and_then(|schema| schema.get("type"));
    let mut candidates = match allowed_types {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    if candidates.is_empty() {
        candidates.extend([
            "object", "array", "integer", "number", "boolean", "null", "string",
        ]);
    }

    let mut last_error = None;
    for kind in candidates {
        let parsed = match kind {
            // The Qwen native grammar carries string values as text between the
            // parameter tags, not as JSON string literals. Quotes and edge whitespace
            // can be meaningful file content, so preserve them verbatim.
            "string" => Ok(Value::String(raw.to_owned())),
            "integer" => serde_json::from_str::<Value>(raw)
                .ok()
                .filter(|value| value.as_i64().is_some() || value.as_u64().is_some())
                .ok_or_else(|| "expected an integer".to_owned()),
            "number" => serde_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_number)
                .ok_or_else(|| "expected a number".to_owned()),
            "boolean" => serde_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_boolean)
                .ok_or_else(|| "expected true or false".to_owned()),
            "object" => serde_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_object)
                .ok_or_else(|| "expected a JSON object".to_owned()),
            "array" => serde_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_array)
                .ok_or_else(|| "expected a JSON array".to_owned()),
            "null" => (raw == "null")
                .then_some(Value::Null)
                .ok_or_else(|| "expected null".to_owned()),
            _ => continue,
        };
        match parsed {
            Ok(value) => match validate_value(&value, schema, "argument") {
                Ok(()) => return Ok(value),
                Err(error) => last_error = Some(error.message),
            },
            Err(error) => last_error = Some(error),
        }
    }
    Err(ProtocolError::generated(last_error.unwrap_or_else(|| {
        "parameter value does not match its JSON Schema type".to_owned()
    })))
}

fn strip_native_value_framing(raw: &str) -> &str {
    let raw = raw
        .strip_prefix("\r\n")
        .or_else(|| raw.strip_prefix('\n'))
        .unwrap_or(raw);
    raw.strip_suffix("\r\n")
        .or_else(|| raw.strip_suffix('\n'))
        .unwrap_or(raw)
}

fn validate_value(value: &Value, schema: Option<&Value>, path: &str) -> Result<(), ProtocolError> {
    let Some(schema) = schema.and_then(Value::as_object) else {
        return Ok(());
    };
    if let Some(kind) = schema.get("type") {
        let matches = match kind {
            Value::String(kind) => value_matches_type(value, kind),
            Value::Array(kinds) => kinds
                .iter()
                .filter_map(Value::as_str)
                .any(|kind| value_matches_type(value, kind)),
            _ => false,
        };
        if !matches {
            return Err(ProtocolError::generated(format!(
                "{path} does not match its declared JSON Schema type"
            )));
        }
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        if !values.contains(value) {
            return Err(ProtocolError::generated(format!(
                "{path} is not one of the allowed enum values"
            )));
        }
    }
    // JSON Schema numeric bounds apply only to numeric instances; a string/null
    // alternative in a union retains its usual meaning.
    if let (Some(value), Some(minimum)) = (
        value.as_number(),
        schema.get("minimum").and_then(Value::as_number),
    ) {
        if compare_json_numbers(value, minimum) == Ordering::Less {
            return Err(ProtocolError::generated(format!(
                "{path} is below its minimum of {minimum}"
            )));
        }
    }
    if let (Some(value), Some(maximum)) = (
        value.as_number(),
        schema.get("maximum").and_then(Value::as_number),
    ) {
        if compare_json_numbers(value, maximum) == Ordering::Greater {
            return Err(ProtocolError::generated(format!(
                "{path} is above its maximum of {maximum}"
            )));
        }
    }
    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for required in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(required) {
                    return Err(ProtocolError::generated(format!(
                        "{path} is missing required property {required:?}"
                    )));
                }
            }
        }
        for (name, child) in object {
            if let Some(child_schema) = properties.and_then(|properties| properties.get(name)) {
                validate_value(child, Some(child_schema), &format!("{path}.{name}"))?;
            } else {
                match schema.get("additionalProperties") {
                    Some(Value::Bool(false)) => {
                        return Err(ProtocolError::generated(format!(
                            "{path} has unknown property {name:?}"
                        )))
                    }
                    Some(Value::Object(_)) => validate_value(
                        child,
                        schema.get("additionalProperties"),
                        &format!("{path}.{name}"),
                    )?,
                    _ => {}
                }
            }
        }
    }
    if let Some(array) = value.as_array() {
        if let Some(maximum) = schema.get("maxItems").and_then(Value::as_number) {
            // Compare the length exactly instead of narrowing a potentially
            // larger count limit to usize or rounding it through f64.
            let length = serde_json::Number::from(array.len());
            if compare_json_numbers(&length, maximum) == Ordering::Greater {
                return Err(ProtocolError::generated(format!(
                    "{path} exceeds its maxItems limit of {maximum}"
                )));
            }
        }
        if let Some(items) = schema.get("items") {
            for (index, child) in array.iter().enumerate() {
                validate_value(child, Some(items), &format!("{path}[{index}]"))?;
            }
        }
    }
    Ok(())
}

/// Compare stored JSON numbers without losing integer precision above 2^53.
fn compare_json_numbers(left: &serde_json::Number, right: &serde_json::Number) -> Ordering {
    let integer = |number: &serde_json::Number| {
        number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from))
    };
    match (integer(left), integer(right)) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(left), None) => {
            compare_integer_float(left, right.as_f64().expect("finite JSON number"))
        }
        (None, Some(right)) => {
            compare_integer_float(right, left.as_f64().expect("finite JSON number")).reverse()
        }
        (None, None) => left
            .as_f64()
            .expect("finite JSON number")
            .partial_cmp(&right.as_f64().expect("finite JSON number"))
            .expect("JSON numbers are finite"),
    }
}

fn compare_integer_float(integer: i128, float: f64) -> Ordering {
    // Every stored JSON integer is in [i64::MIN, u64::MAX]. These exact power-of-two
    // boundaries avoid casts saturating at the integer limits or rounding u64::MAX.
    if float < -9_223_372_036_854_775_808.0 {
        return Ordering::Greater;
    }
    if float >= 18_446_744_073_709_551_616.0 {
        return Ordering::Less;
    }
    match integer.cmp(&(float.trunc() as i128)) {
        Ordering::Equal if float.fract() > 0.0 => Ordering::Less,
        Ordering::Equal if float.fract() < 0.0 => Ordering::Greater,
        order => order,
    }
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn qwen3_matches_independent_transformers_jinja_goldens() {
        let fixtures: Value =
            serde_json::from_str(include_str!("../tests/fixtures/qwen3-tools/goldens.json"))
                .unwrap();
        let template = ChatTemplate::Checkpoint(Some(QWEN3_CHAT_TEMPLATE.into()));
        for fixture in fixtures["cases"].as_array().unwrap() {
            // Deserialize original wire text, not Value, so tool insertion order survives.
            let request: ChatRequest =
                serde_json::from_str(fixture["request_text"].as_str().unwrap()).unwrap();
            let actual = render_prompt_for_template(
                &request.messages,
                &request.tools.unwrap_or_default(),
                &template,
            )
            .unwrap();
            assert_eq!(
                actual,
                fixture["expected_prompt"].as_str().unwrap(),
                "{}",
                fixture["name"]
            );
        }
        for fixture in fixtures["rejected_cases"].as_array().unwrap() {
            let request =
                serde_json::from_str::<ChatRequest>(fixture["request_text"].as_str().unwrap());
            if fixture["expected_rejection_stage"] == "deserialize" {
                assert!(request.is_err(), "{}", fixture["name"]);
            } else {
                let request = request.unwrap();
                assert!(
                    render_prompt_for_template(
                        &request.messages,
                        &request.tools.unwrap_or_default(),
                        &template
                    )
                    .is_err(),
                    "{}",
                    fixture["name"]
                );
            }
        }
    }

    #[test]
    fn qwen3_requires_complete_unambiguous_tool_history() {
        let template = ChatTemplate::Checkpoint(Some(QWEN3_CHAT_TEMPLATE.into()));
        let call = json!({"role":"assistant","tool_calls":[{"id":"id","type":"function","function":{"name":"f","arguments":"{}"}}]});
        let result = json!({"role":"tool","tool_call_id":"id","content":"ok"});
        for turns in [
            json!([call]),
            json!([result]),
            json!([call, {"role":"user","content":"next"}, result]),
            json!([call, result, result]),
            json!([call, {"role":"tool","tool_call_id":"id","name":"wrong","content":"ok"}]),
            json!([call, {"role":"tool","tool_call_id":"other","content":"ok"}]),
        ] {
            assert!(render_prompt_for_template(&messages(turns), &[], &template).is_err());
        }
        for arguments in ["[]", "null", "{} suffix", r#"{"x":1,"x":2}"#] {
            let mut bad_call = call.clone();
            bad_call["tool_calls"][0]["function"]["arguments"] = arguments.into();
            assert!(render_prompt_for_template(
                &messages(json!([bad_call, result])),
                &[],
                &template
            )
            .is_err());
        }
        let mut duplicate = call.clone();
        duplicate["tool_calls"]
            .as_array_mut()
            .unwrap()
            .push(call["tool_calls"][0].clone());
        assert!(
            render_prompt_for_template(&messages(json!([duplicate, result])), &[], &template)
                .is_err()
        );
    }

    #[test]
    fn tool_definitions_refuse_duplicate_and_unknown_fields() {
        for raw in [
            r#"{"type":"function","type":"function","function":{"name":"f"}}"#,
            r#"{"type":"function","function":{"name":"f","parameters":{"type":"object","type":"object"}}}"#,
            r#"{"type":"function","function":{"name":"f","strict":true}}"#,
        ] {
            assert!(serde_json::from_str::<ToolDefinition>(raw).is_err());
        }
    }

    fn tool() -> ToolDefinition {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": " Write a file. ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Target"},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "enum": ["create", "replace"]},
                        "priority": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "metadata": {
                            "type": "object",
                            "properties": {"executable": {"type": "boolean"}},
                            "additionalProperties": false
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }))
        .unwrap()
    }

    fn messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).unwrap()
    }

    fn qwen3(messages: &[ChatMessage]) -> String {
        render_prompt_for_template(
            messages,
            &[],
            &ChatTemplate::Checkpoint(Some(QWEN3_CHAT_TEMPLATE.into())),
        )
        .unwrap()
    }

    #[test]
    fn qwen3_history_uses_final_content_without_mutating_wire_messages() {
        let messages = messages(json!([
            {"role":"system","content":"Project instructions."},
            {"role":"user","content":"Reply with one word: ready. /no_think"},
            {"role":"assistant","content":"<think>\n\n</think>\n\nready."},
            {"role":"user","content":"Reply with that same one word again. /no_think"}
        ]));
        assert_eq!(
            qwen3(&messages),
            concat!(
                "<|im_start|>system\nProject instructions.<|im_end|>\n",
                "<|im_start|>user\nReply with one word: ready. /no_think<|im_end|>\n",
                "<|im_start|>assistant\nready.<|im_end|>\n",
                "<|im_start|>user\nReply with that same one word again. /no_think<|im_end|>\n",
                "<|im_start|>assistant\n"
            )
        );
        assert_eq!(
            message_text(&messages[2], true).unwrap(),
            "<think>\n\n</think>\n\nready."
        );
        assert!(
            render_prompt_for_template(&messages, &[], &ChatTemplate::Legacy)
                .unwrap()
                .contains("<|im_start|>assistant\n<think>\n\n</think>\n\nready.")
        );
    }

    #[test]
    fn qwen3_later_assistants_follow_exact_reasoning_and_final_turn_rules() {
        let turns = messages(json!([
            {"role":"user","content":"Continue."},
            {"role":"assistant","content":"<think>\nreason\n</think>\n\nA"},
            {"role":"assistant","content":"<think>\n\n</think>\n\nB"},
            {"role":"assistant","content":"\n\nC"}
        ]));
        assert_eq!(
            qwen3(&turns),
            concat!(
                "<|im_start|>user\nContinue.<|im_end|>\n",
                "<|im_start|>assistant\n<think>\nreason\n</think>\n\nA<|im_end|>\n",
                "<|im_start|>assistant\nB<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\nC<|im_end|>\n",
                "<|im_start|>assistant\n"
            )
        );
        let whitespace = messages(json!([
            {"role":"user","content":"q"},
            {"role":"assistant","content":"<think>\u{001c}\n</think>\nA"},
            {"role":"assistant","content":null}
        ]));
        assert!(qwen3(&whitespace).contains("<|im_start|>assistant\nA<|im_end|>\n"));
    }

    #[test]
    fn qwen3_split_boundaries_match_template_without_repairing_malformed_history() {
        let turns = messages(json!([
            {"role":"assistant","content":"prefix<think>first</think>discard</think>\n\n tail "},
            {"role":"assistant","content":"<think>unfinished"},
            {"role":"user","content":"q /no_think"}
        ]));
        assert_eq!(
            qwen3(&turns),
            concat!(
                "<|im_start|>assistant\n tail <|im_end|>\n",
                "<|im_start|>assistant\n<think>unfinished<|im_end|>\n",
                "<|im_start|>user\nq /no_think<|im_end|>\n",
                "<|im_start|>assistant\n"
            )
        );
        // With no ordinary user, last_query_index remains the final index.
        assert_eq!(
            qwen3(&messages(json!([{"role":"assistant","content":"plain"}]))),
            "<|im_start|>assistant\nplain<|im_end|>\n<|im_start|>assistant\n"
        );
        let tool_text = messages(json!([
            {"role":"user","content":"q"},
            {"role":"assistant","content":"<think>reason</think>answer"},
            {"role":"user","content":"<tool_response>quoted text</tool_response>"}
        ]));
        assert!(qwen3(&tool_text).contains("assistant\n<think>\nreason\n</think>\n\nanswer"));
    }

    #[test]
    fn checkpoint_selection_refuses_missing_or_modified_template_without_legacy_fallback() {
        let messages = messages(json!([{"role":"user","content":"q"}]));
        for template in [
            None,
            Some(String::new()),
            Some(format!("{QWEN3_CHAT_TEMPLATE}\n")),
            Some(QWEN3_CHAT_TEMPLATE.replace("enable_thinking", "changed")),
        ] {
            let error =
                render_prompt_for_template(&messages, &[], &ChatTemplate::Checkpoint(template))
                    .unwrap_err();
            assert_eq!(error.code, "unsupported_chat_template");
            assert_eq!(error.param, Some("model"));
        }
    }

    #[test]
    fn exact_coder_template_preserves_legacy_text_and_tool_rendering() {
        assert_eq!(QWEN3_CODER_CHAT_TEMPLATE.len(), 6722);
        let fingerprint = QWEN3_CODER_CHAT_TEMPLATE
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        assert_eq!(fingerprint, 0xa72a_487d_0e22_3214);
        let turns = messages(json!([
            {"role":"user","content":"q"},
            {"role":"assistant","content":"<think>prior</think>answer"},
            {"role":"user","content":"continue"}
        ]));
        let template = ChatTemplate::Checkpoint(Some(QWEN3_CODER_CHAT_TEMPLATE.into()));
        for tools in [vec![], vec![tool()]] {
            assert_eq!(
                render_prompt_for_template(&turns, &tools, &template).unwrap(),
                render_prompt(&turns, &tools).unwrap()
            );
        }
        assert!(render_prompt_for_template(&turns, &[], &template)
            .unwrap()
            .contains("<think>prior</think>answer"));
    }

    #[test]
    fn qwen3_refuses_invalid_metadata_and_extra_request_fields() {
        let template = ChatTemplate::Checkpoint(Some(QWEN3_CHAT_TEMPLATE.into()));
        let plain = messages(json!([{"role":"user","content":"q"}]));
        assert!(render_prompt_for_template(&plain, &[tool()], &template)
            .unwrap()
            .contains("# Tools"));
        for value in [
            json!([{"role":"tool","content":"result","tool_call_id":"id"}]),
            json!([{"role":"assistant","content":"answer","name":"f"}]),
            json!([{"role":"user","content":"q"},{"role":"system","content":"late"}]),
        ] {
            assert!(render_prompt_for_template(&messages(value), &[], &template).is_err());
        }
        assert!(prepare_chat(&plain, &[], &template, Some(false), None).is_err());
        assert!(render_prompt_for_template(
            &messages(json!([
                {"role":"assistant", "content":"a", "reasoning_content":"r"}
            ])),
            &[],
            &template
        )
        .is_err());
        assert_eq!(qwen3(&plain), render_prompt(&plain, &[]).unwrap());
    }

    #[test]
    fn prompt_snapshots_tool_definitions_and_replay() {
        let messages = messages(json!([
            {"role":"system", "content":"Be precise."},
            {"role":"user", "content":"Make it."},
            {"role":"assistant", "content":null, "tool_calls":[{
                "id":"abc", "type":"function",
                "function":{"name":"write_file", "arguments":"{\"path\":\"x.rs\",\"content\":\"fn main() {}\"}"}
            }]},
            {"role":"tool", "tool_call_id":"abc", "name":"write_file", "content":"ok"},
            {"role":"user", "content":"Continue."}
        ]));
        let prompt = render_prompt(&messages, &[tool()]).unwrap();
        assert!(prompt.starts_with("<|im_start|>system\nBe precise.\n\nYou have access"));
        assert!(prompt.contains("<description>Write a file.</description>"));
        assert!(prompt.contains("<enum>[`create`, `replace`]</enum>"));
        assert!(prompt.contains("<required>[`path`, `content`]</required>"));
        assert!(prompt.contains("<function=write_file>\n<parameter=content>\nfn main() {}"));
        assert!(
            prompt.contains("<|im_start|>user\n<tool_response>\nok\n</tool_response>\n<|im_end|>")
        );
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn prompt_rejects_content_parts_and_unpaired_tool_results() {
        let content_parts =
            messages(json!([{"role":"user", "content":[{"type":"text","text":"x"}]}]));
        assert!(render_prompt(&content_parts, &[])
            .unwrap_err()
            .message
            .contains("string or null"));

        let unpaired = messages(json!([{
            "role":"tool", "tool_call_id":"missing", "content":"x"
        }]));
        assert!(render_prompt(&unpaired, &[])
            .unwrap_err()
            .message
            .contains("no pending call"));
    }

    #[test]
    fn native_tool_calls_are_coerced_and_validated() {
        let output = "thinking first\n<tool_call>\n<function=write_file>\n\
            <parameter=path>\nx.rs\n</parameter>\n\
            <parameter=content>\nfn main() {}\n</parameter>\n\
            <parameter=priority>\n3\n</parameter>\n\
            <parameter=tags>\n[\"rust\",\"cli\"]\n</parameter>\n\
            <parameter=metadata>\n{\"executable\":true}\n</parameter>\n\
            </function>\n</tool_call>";
        let translated = translate_output(output, &[tool()]).unwrap();
        assert_eq!(translated.content.as_deref(), Some("thinking first"));
        assert_eq!(translated.tool_calls[0].name, "write_file");
        assert_eq!(
            serde_json::from_str::<Value>(&translated.tool_calls[0].arguments).unwrap(),
            json!({
                "path":"x.rs",
                "content":"fn main() {}",
                "priority":3,
                "tags":["rust", "cli"],
                "metadata":{"executable":true}
            })
        );
    }

    #[test]
    fn native_string_arguments_preserve_quotes_and_meaningful_edge_whitespace() {
        let output = "<tool_call>\n<function=write_file>\n\
            <parameter=path>\nscript.js\n</parameter>\n\
            <parameter=content>\n  \"use strict\";  \n</parameter>\n\
            </function>\n</tool_call>";
        let translated = translate_output(output, &[tool()]).unwrap();
        let arguments = serde_json::from_str::<Value>(&translated.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["content"], json!("  \"use strict\";  "));

        let output = "<tool_call>\n<function=write_file>\n\
            <parameter=path>\nmodule.py\n</parameter>\n\
            <parameter=content>\n\n\"\"\"doc\"\"\"\nbody\n\n</parameter>\n\
            </function>\n</tool_call>";
        let translated = translate_output(output, &[tool()]).unwrap();
        let arguments = serde_json::from_str::<Value>(&translated.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["content"], json!("\n\"\"\"doc\"\"\"\nbody\n"));
    }

    #[test]
    fn union_coercion_tries_later_types_after_schema_validation_fails() {
        let schema = json!({"type":["integer", "string"], "enum":["1"]});
        assert_eq!(coerce_value("1", Some(&schema)).unwrap(), json!("1"));
    }

    fn tool_with_value_schema(schema: Value) -> ToolDefinition {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "bounded",
                "parameters": {
                    "type": "object",
                    "properties": {"value": schema},
                    "required": ["value"]
                }
            }
        }))
        .unwrap()
    }

    fn bounded_call(raw_value: &str) -> String {
        format!(
            "<tool_call><function=bounded><parameter=value>{raw_value}</parameter></function></tool_call>"
        )
    }

    #[test]
    fn minimum_schema_requires_a_finite_json_number() {
        for minimum in [
            json!(0),
            json!(-1.25),
            json!(u64::MAX),
            json!(i64::MIN),
            json!(f64::MAX),
        ] {
            // Numeric keywords may coexist with any type; they constrain only numbers.
            for kind in [json!("number"), json!("string"), json!(["integer", "null"])] {
                validate_tools(&[tool_with_value_schema(json!({
                    "type": kind, "minimum": minimum
                }))])
                .unwrap();
            }
        }
        for minimum in [
            Value::Null,
            json!(true),
            json!("1"),
            json!("NaN"),
            json!([]),
            json!({}),
        ] {
            let error = validate_tools(&[tool_with_value_schema(json!({
                "type": "number", "minimum": minimum
            }))])
            .unwrap_err();
            assert!(error
                .message
                .contains("minimum must be a finite JSON number"));
        }
        // Non-finite numeric spellings are refused during JSON parsing, before schema admission.
        for raw in ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"] {
            assert!(serde_json::from_str::<Value>(raw).is_err(), "{raw}");
        }
        for schema in [
            json!({"type":"array", "items":{"minimum":"0"}}),
            json!({"type":"object", "properties":{"nested":{"minimum":null}}}),
            json!({"type":"object", "additionalProperties":{"minimum":false}}),
        ] {
            assert!(validate_tools(&[tool_with_value_schema(schema)])
                .unwrap_err()
                .message
                .contains("minimum must be a finite JSON number"));
        }
    }

    #[test]
    fn minimum_is_rendered_without_rounding_in_scalar_and_nested_schemas() {
        let tools = [tool_with_value_schema(
            json!({"type":"integer", "minimum":9_007_199_254_740_993u64}),
        )];
        let prompt = render_prompt(
            &messages(json!([{"role":"user", "content":"Use it."}])),
            &tools,
        )
        .unwrap();
        assert!(prompt.contains("<minimum>9007199254740993</minimum>"));
        let tools = [tool_with_value_schema(json!({
            "type":"object",
            "properties":{"numbers":{"type":"array", "items":{"type":"number", "minimum":-0.5}}},
            "additionalProperties":{"type":"integer", "minimum":0}
        }))];
        let prompt = render_prompt(
            &messages(json!([{"role":"user", "content":"Use it."}])),
            &tools,
        )
        .unwrap();
        assert!(prompt.contains("\"minimum\":-0.5"));
        assert!(prompt.contains(
            "<additionalProperties>{\"minimum\":0,\"type\":\"integer\"}</additionalProperties>"
        ));
    }

    #[test]
    fn minimum_comparison_preserves_signed_unsigned_and_float_boundaries() {
        for (value, minimum, allowed) in [
            ("9007199254740992", "9007199254740993", false),
            ("9007199254740993", "9007199254740993", true),
            ("-9007199254740993", "-9007199254740992", false),
            ("-9223372036854775808", "-9223372036854775808", true),
            ("18446744073709551615", "9223372036854775807", true),
            ("-1", "18446744073709551615", false),
            ("9007199254740992.0", "9007199254740993", false),
            ("9007199254740993", "9007199254740992.0", true),
            ("-9007199254740993", "-9007199254740992.0", false),
            ("-9007199254740992.0", "-9007199254740993", true),
            ("9223372036854775807", "9223372036854775808.0", false),
            ("18446744073709551615", "18446744073709551616.0", false),
            ("18446744073709551616.0", "18446744073709551615", true),
            ("-9223372036854775808", "-9223372036854775808.0", true),
            ("-9223372036854775808", "-1e100", true),
            ("-1e100", "-9223372036854775808", false),
            ("18446744073709551615", "1e100", false),
            ("1e100", "18446744073709551615", true),
            ("0", "0.5", false),
            ("1", "0.5", true),
            ("-1", "-0.5", false),
            ("0", "-0.5", true),
            ("0.5", "1", false),
            ("-0.5", "-1", true),
            ("0", "-0.0", true),
            ("-0.0", "0", true),
            ("0.125", "0.25", false),
            ("0.25", "0.25", true),
        ] {
            let minimum = serde_json::from_str::<Value>(minimum).unwrap();
            let schema = json!({"type":"number", "minimum":minimum});
            validate_tools(&[tool_with_value_schema(schema.clone())]).unwrap();
            assert_eq!(
                coerce_value(value, Some(&schema)).is_ok(),
                allowed,
                "value={value}, minimum={minimum}"
            );
        }
    }

    #[test]
    fn maximum_and_max_items_schema_admission_is_recursive_and_type_aware() {
        for maximum in [
            json!(0),
            json!(-1.25),
            json!(u64::MAX),
            json!(i64::MIN),
            json!(f64::MAX),
        ] {
            for kind in [json!("number"), json!("string"), json!(["integer", "null"])] {
                validate_tools(&[tool_with_value_schema(
                    json!({"type":kind, "maximum":maximum}),
                )])
                .unwrap();
            }
        }
        for max_items in [
            json!(0),
            json!(2),
            json!(2.0),
            json!(-0.0),
            json!(u64::MAX),
            json!(1e100),
        ] {
            for kind in [json!("array"), json!("string"), json!(["array", "null"])] {
                validate_tools(&[tool_with_value_schema(
                    json!({"type":kind, "maxItems":max_items}),
                )])
                .unwrap();
            }
        }
        for (keyword, invalid_values, expected) in [
            (
                "maximum",
                vec![
                    Value::Null,
                    json!(true),
                    json!("1"),
                    json!("Infinity"),
                    json!([]),
                    json!({}),
                ],
                "maximum must be a finite JSON number",
            ),
            (
                "maxItems",
                vec![
                    Value::Null,
                    json!(false),
                    json!("2"),
                    json!(-1),
                    json!(-0.5),
                    json!(1.5),
                    json!(1e-100),
                    json!([]),
                    json!({}),
                ],
                "maxItems must be a nonnegative integer",
            ),
        ] {
            for invalid in invalid_values {
                let leaf = json!({(keyword):invalid});
                for (schema, path) in [
                    (leaf.clone(), "parameters.properties.value"),
                    (
                        json!({"type":"array", "items":leaf}),
                        "parameters.properties.value.items",
                    ),
                    (
                        json!({"type":"object", "properties":{"child":leaf}}),
                        "parameters.properties.value.properties.child",
                    ),
                    (
                        json!({"type":"object", "additionalProperties":leaf}),
                        "parameters.properties.value.additionalProperties",
                    ),
                ] {
                    let tools = [tool_with_value_schema(schema)];
                    for template in [QWEN3_CHAT_TEMPLATE, QWEN3_CODER_CHAT_TEMPLATE] {
                        let error = render_prompt_for_template(
                            &messages(json!([{"role":"user", "content":"Use it."}])),
                            &tools,
                            &ChatTemplate::Checkpoint(Some(template.into())),
                        )
                        .unwrap_err();
                        assert_eq!(error.param, Some("tools"));
                        assert!(
                            error.message.contains(&format!("{path}.{expected}")),
                            "{}",
                            error.message
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn maximum_and_max_items_are_preserved_in_both_tool_rendering_dialects() {
        let tools = [tool_with_value_schema(json!({
            "type":["integer","array","object"],
            "maximum":9_007_199_254_740_993u64,
            "maxItems":2,
            "items":{"type":"number", "minimum":-0.5, "maximum":0.5},
            "properties":{"empty":{"type":"array", "maxItems":0}},
            "additionalProperties":{"type":"array", "maxItems":3, "items":{"maximum":1}}
        }))];
        let input = messages(json!([{"role":"user", "content":"Use it."}]));
        let coder = render_prompt_for_template(
            &input,
            &tools,
            &ChatTemplate::Checkpoint(Some(QWEN3_CODER_CHAT_TEMPLATE.into())),
        )
        .unwrap();
        assert!(coder.contains("<maximum>9007199254740993</maximum>"));
        assert!(coder.contains("<maxItems>2</maxItems>"));
        assert!(
            coder.contains("<items>{\"maximum\":0.5,\"minimum\":-0.5,\"type\":\"number\"}</items>")
        );
        assert!(coder
            .contains("<properties>{\"empty\":{\"maxItems\":0,\"type\":\"array\"}}</properties>"));
        assert!(coder.contains("<additionalProperties>{\"items\":{\"maximum\":1},\"maxItems\":3,\"type\":\"array\"}</additionalProperties>"));

        let qwen = render_prompt_for_template(
            &input,
            &tools,
            &ChatTemplate::Checkpoint(Some(QWEN3_CHAT_TEMPLATE.into())),
        )
        .unwrap();
        let definition = qwen
            .split_once("<tools>\n")
            .unwrap()
            .1
            .split_once("\n</tools>")
            .unwrap()
            .0;
        let rendered: Value = serde_json::from_str(definition).unwrap();
        assert_eq!(
            rendered["function"]["parameters"],
            tools[0].function.parameters.as_ref().unwrap().clone()
        );
        assert!(definition.contains("\"maximum\": 9007199254740993"));
        assert!(definition.contains("\"maxItems\": 2"));
    }

    #[test]
    fn maximum_comparison_preserves_signed_unsigned_and_float_boundaries() {
        for (value, maximum, allowed) in [
            ("9007199254740993", "9007199254740992", false),
            ("9007199254740993", "9007199254740993", true),
            ("-9007199254740992", "-9007199254740993", false),
            ("-9223372036854775808", "-9223372036854775808", true),
            ("18446744073709551615", "9223372036854775807", false),
            ("-1", "18446744073709551615", true),
            ("9007199254740992.0", "9007199254740993", true),
            ("9007199254740993", "9007199254740992.0", false),
            ("-9007199254740993", "-9007199254740992.0", true),
            ("-9007199254740992.0", "-9007199254740993", false),
            ("9223372036854775807", "9223372036854775808.0", true),
            ("18446744073709551615", "18446744073709551616.0", true),
            ("18446744073709551616.0", "18446744073709551615", false),
            ("-9223372036854775808", "-9223372036854775808.0", true),
            ("-9223372036854775808", "-1e100", false),
            ("-1e100", "-9223372036854775808", true),
            ("18446744073709551615", "1e100", true),
            ("1e100", "18446744073709551615", false),
            ("0", "0.5", true),
            ("1", "0.5", false),
            ("-1", "-0.5", true),
            ("0", "-0.5", false),
            ("0.5", "1", true),
            ("-0.5", "-1", false),
            ("0", "-0.0", true),
            ("-0.0", "0", true),
            ("0.25", "0.125", false),
            ("0.25", "0.25", true),
        ] {
            let maximum = serde_json::from_str::<Value>(maximum).unwrap();
            let schema = json!({"type":"number", "maximum":maximum});
            validate_tools(&[tool_with_value_schema(schema.clone())]).unwrap();
            assert_eq!(
                coerce_value(value, Some(&schema)).is_ok(),
                allowed,
                "value={value}, maximum={maximum}"
            );
        }
    }

    #[test]
    fn max_items_checks_empty_exact_and_overflow_lengths_without_narrowing_bounds() {
        for (maximum, length, allowed) in [
            ("0", 0, true),
            ("0", 1, false),
            ("-0.0", 0, true),
            ("-0.0", 1, false),
            ("2", 1, true),
            ("2", 2, true),
            ("2", 3, false),
            ("2.0", 2, true),
            ("2.0", 3, false),
            ("2e0", 2, true),
            ("2e0", 3, false),
            ("9007199254740993", 3, true),
            ("18446744073709551615", 3, true),
            ("18446744073709551616.0", 3, true),
            ("1e100", 3, true),
        ] {
            let maximum = serde_json::from_str::<Value>(maximum).unwrap();
            // No items keyword: the length constraint must still apply.
            let schema = json!({"type":"array", "maxItems":maximum});
            validate_tools(&[tool_with_value_schema(schema.clone())]).unwrap();
            let value = Value::Array(vec![Value::Null; length]);
            assert_eq!(
                validate_value(&value, Some(&schema), "argument").is_ok(),
                allowed,
                "length={length}, maxItems={maximum}"
            );
            assert_eq!(
                coerce_value(&value.to_string(), Some(&schema)).is_ok(),
                allowed
            );
        }
    }

    #[test]
    fn upper_bounds_apply_only_to_matching_union_members() {
        let numeric = json!({"type":["integer","number"], "maximum":1.5});
        assert!(coerce_value("2", Some(&numeric)).is_err());
        assert_eq!(coerce_value("1.5", Some(&numeric)).unwrap(), json!(1.5));
        let optional = json!({"type":["array","null"], "maxItems":0});
        assert_eq!(coerce_value("null", Some(&optional)).unwrap(), Value::Null);
        assert!(coerce_value("[0]", Some(&optional)).is_err());
        let text_number = json!({"type":["integer","string"], "maximum":1});
        assert_eq!(coerce_value("2", Some(&text_number)).unwrap(), json!("2"));
        assert!(validate_value(&json!(2), Some(&text_number), "argument").is_err());
        let text_array = json!({"type":["array","string"], "maxItems":0});
        assert_eq!(
            coerce_value("[0]", Some(&text_array)).unwrap(),
            json!("[0]")
        );
        assert!(validate_value(&json!([0]), Some(&text_array), "argument").is_err());
        let untyped = json!({"maximum":1, "maxItems":0});
        for value in [
            json!("2"),
            Value::Null,
            json!(false),
            json!({}),
            json!([]),
            json!(1),
        ] {
            validate_value(&value, Some(&untyped), "argument").unwrap();
        }
        for value in [json!(2), json!([0])] {
            assert!(validate_value(&value, Some(&untyped), "argument").is_err());
        }
        // Contradictory bounds are a valid schema with no numeric instances.
        let empty_numeric = json!({"type":["number","null"], "minimum":2, "maximum":1});
        validate_tools(&[tool_with_value_schema(empty_numeric.clone())]).unwrap();
        for value in [json!(1), json!(1.5), json!(2)] {
            assert!(validate_value(&value, Some(&empty_numeric), "argument").is_err());
        }
        validate_value(&Value::Null, Some(&empty_numeric), "argument").unwrap();
    }

    #[test]
    fn both_generated_call_dialects_enforce_nested_and_additional_upper_bounds() {
        let tools = [tool_with_value_schema(json!({
            "type":"object",
            "properties": {
                "number":{"type":"integer", "maximum":9_007_199_254_740_993u64},
                "scores":{"type":"array", "maxItems":2, "items":{"type":["number","null"], "minimum":-0.5, "maximum":0.5}},
                "rows":{"type":"array", "maxItems":1, "items":{"type":"array", "maxItems":0}}
            },
            "required":["number","scores","rows"],
            "additionalProperties":{"type":"array", "maxItems":1, "items":{"type":"number", "maximum":0.5}}
        }))];
        validate_tools(&tools).unwrap();
        let valid = json!({"number":9_007_199_254_740_993u64, "scores":[null,0.5], "rows":[[]], "extra":[0.5]});
        for dialect in [OutputDialect::CoderXml, OutputDialect::Qwen3Json] {
            let call = |value: &Value| match dialect {
                OutputDialect::CoderXml => bounded_call(&value.to_string()),
                OutputDialect::Qwen3Json => format!(
                    "<tool_call>{}</tool_call>",
                    json!({"name":"bounded", "arguments":{"value":value}})
                ),
                // OpenMythos spells arguments through its own function-tag format,
                // so it needs its own fixture rather than this encoder.
                OutputDialect::OpenMythos { .. } => {
                    unreachable!("this fixture iterates only CoderXml and Qwen3Json")
                }
            };
            let translated = translate_output_for_dialect(&call(&valid), &tools, dialect).unwrap();
            let arguments: Value =
                serde_json::from_str(&translated.tool_calls[0].arguments).unwrap();
            assert_eq!(arguments["value"], valid);
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
                let error =
                    translate_output_for_dialect(&call(&invalid), &tools, dialect).unwrap_err();
                assert_eq!(error.code, "tool_call_parse_error");
                assert!(
                    error.message.contains(expected),
                    "{dialect:?}: {}",
                    error.message
                );
            }
        }
    }

    #[test]
    fn minimum_constrains_numeric_union_members_only() {
        let numeric = json!({"type":["integer", "number"], "minimum":1});
        assert!(coerce_value("0", Some(&numeric)).is_err());
        assert!(coerce_value("0.5", Some(&numeric)).is_err());
        assert_eq!(coerce_value("1.5", Some(&numeric)).unwrap(), json!(1.5));
        let integral = json!({"type":"integer", "minimum":1.5});
        assert!(coerce_value("1", Some(&integral)).is_err());
        assert_eq!(coerce_value("2", Some(&integral)).unwrap(), json!(2));
        let optional = json!({"type":["integer", "null"], "minimum":1});
        assert_eq!(coerce_value("null", Some(&optional)).unwrap(), Value::Null);
        assert!(coerce_value("0", Some(&optional)).is_err());
        let text = json!({"type":["integer", "string"], "minimum":1});
        assert_eq!(coerce_value("0", Some(&text)).unwrap(), json!("0"));
        assert!(validate_value(&json!(0), Some(&text), "argument").is_err());
        let untyped = json!({"minimum":1});
        for nonnumeric in [json!("0"), Value::Null, json!(false), json!([]), json!({})] {
            validate_value(&nonnumeric, Some(&untyped), "argument").unwrap();
        }
        assert!(validate_value(&json!(0.5), Some(&untyped), "argument").is_err());
    }

    #[test]
    fn returned_tool_arguments_enforce_nested_and_additional_minima() {
        let tool = tool_with_value_schema(json!({
            "type":"object",
            "properties": {
                "number":{"type":"integer", "minimum":1},
                "offset":{"type":"integer", "minimum":0},
                "scores":{"type":"array", "items":{"type":["number","null"], "minimum":0.5}}
            },
            "required":["number","offset","scores"],
            "additionalProperties":{"type":"number", "minimum":-1}
        }));
        validate_tools(std::slice::from_ref(&tool)).unwrap();
        let valid = json!({"number":1,"offset":0,"scores":[null,0.5,1],"extra":-1});
        let translated = translate_output(
            &bounded_call(&valid.to_string()),
            std::slice::from_ref(&tool),
        )
        .unwrap();
        let arguments: Value = serde_json::from_str(&translated.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["value"], valid);
        for (name, value, path) in [
            ("number", json!(0), "argument.number"),
            ("offset", json!(-1), "argument.offset"),
            ("scores", json!([1, 0.25]), "argument.scores[1]"),
            ("extra", json!(-1.5), "argument.extra"),
        ] {
            let mut invalid = valid.clone();
            invalid[name] = value;
            let error = translate_output(
                &bounded_call(&invalid.to_string()),
                std::slice::from_ref(&tool),
            )
            .unwrap_err();
            assert!(
                error
                    .message
                    .contains(&format!("{path} is below its minimum")),
                "{}",
                error.message
            );
        }
    }

    #[test]
    fn malformed_or_schema_invalid_tool_calls_never_become_prose() {
        for output in [
            "<tool_call><function=write_file>",
            "<tool_call><function=unknown></function></tool_call>",
            "<tool_call><function=write_file><parameter=path>x</parameter></function></tool_call>",
            "answer <function=write_file></function>",
            "<tool_call><function=write_file><parameter=path>x</parameter><parameter=content>y</parameter><parameter=metadata>{\"executable\":1}</parameter></function></tool_call>",
        ] {
            assert!(translate_output(output, &[tool()]).is_err(), "{output}");
        }
    }

    #[test]
    fn ordinary_output_remains_content() {
        assert_eq!(
            translate_output("ordinary answer", &[tool()]).unwrap(),
            TranslatedOutput {
                content: Some("ordinary answer".to_owned()),
                reasoning_content: None,
                tool_calls: vec![],
            }
        );
    }
}
