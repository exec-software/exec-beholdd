//! Measure the server's exact rendered request using checkpoint tokenizer metadata.
//! No model weights, runtime, server listener, or execution device are created.
//! Optional `--tokens-out PATH` exports one request's exact token IDs to a new file.

#[allow(dead_code)]
#[path = "../src/protocol.rs"]
mod protocol;

use anyhow::{ensure, Context, Result};
use beholdd_format::Gguf;
use beholdd_model::Qwen2Tokenizer;
use beholdd_runtime::ChatTemplate;
use protocol::{normalize_message_content, render_prompt_for_template, ChatRequest};
use serde_json::json;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAX_REQUEST_BYTES: u64 = 8 * 1024 * 1024;
const USAGE: &str =
    "usage: prompt_inspect [--tokens-out PATH] MODEL.gguf REQUEST.json [REQUEST.json ...]";

#[derive(Debug, PartialEq, Eq)]
struct Arguments {
    model: PathBuf,
    requests: Vec<PathBuf>,
    tokens_out: Option<PathBuf>,
}

fn parse_arguments(args: impl IntoIterator<Item = OsString>) -> Result<Arguments> {
    let mut args = args.into_iter();
    let mut positional = Vec::new();
    let mut tokens_out = None;
    let mut options = true;
    while let Some(arg) = args.next() {
        if options && arg == "--" {
            options = false;
        } else if options && arg == "--tokens-out" {
            ensure!(
                tokens_out.is_none(),
                "--tokens-out may only be specified once"
            );
            let path = args
                .next()
                .context("--tokens-out requires an output path")?;
            ensure!(
                !path.is_empty() && path != "--" && path != "--tokens-out",
                "--tokens-out requires an output path"
            );
            tokens_out = Some(PathBuf::from(path));
        } else {
            positional.push(PathBuf::from(arg));
        }
    }
    let mut positional = positional.into_iter();
    let model = positional.next().context(USAGE)?;
    let requests: Vec<_> = positional.collect();
    ensure!(!requests.is_empty(), "at least one request is required");
    ensure!(
        tokens_out.is_none() || requests.len() == 1,
        "--tokens-out requires exactly one request"
    );
    Ok(Arguments {
        model,
        requests,
        tokens_out,
    })
}

struct Inspection {
    message_count: usize,
    tool_count: usize,
    rendered_prompt_bytes: usize,
    token_ids: Vec<u32>,
    output_tokens: usize,
    minimum_evaluated_positions: usize,
    output_budget_source: &'static str,
}

impl Inspection {
    fn report(&self, request: &Path) -> serde_json::Value {
        json!({
            "request": request, "status": "measured", "message_count": self.message_count,
            "tool_count": self.tool_count, "rendered_prompt_bytes": self.rendered_prompt_bytes,
            "prompt_tokens": self.token_ids.len(), "output_tokens": self.output_tokens,
            "minimum_evaluated_positions": self.minimum_evaluated_positions,
            "output_budget_source": self.output_budget_source,
        })
    }

    fn token_artifact(&self, checkpoint: &Path, request: &Path) -> serde_json::Value {
        // Version 1 contains the complete ordered output of encode(prompt, true),
        // including any checkpoint-configured BOS/EOS tokens. The supervisor binds
        // the checkpoint, raw request, and artifact bytes; this is no hash receipt.
        json!({
            "format": "beholdd.prompt-token-ids", "schemaVersion": 1,
            "tokenization": "exact_checkpoint_metadata", "parseSpecial": true,
            "checkpoint": checkpoint, "request": request,
            "promptTokens": self.token_ids, "promptTokenCount": self.token_ids.len(),
            "messageCount": self.message_count, "toolCount": self.tool_count,
            "maxTokens": self.output_tokens,
            "minimumContextPositions": self.minimum_evaluated_positions,
            "renderedPromptBytes": self.rendered_prompt_bytes,
        })
    }
}

fn write_tokens_exclusive(path: &Path, artifact: &serde_json::Value) -> Result<()> {
    // Serialize before opening so serialization failure cannot leave an empty file.
    // On an I/O failure a partial file is retained, never replaced on retry; only a
    // successful exit qualifies an export. Use a fresh path for each attempt.
    let mut bytes = serde_json::to_vec(artifact)?;
    bytes.push(b'\n');
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("exclusively creating token artifact {}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .with_context(|| {
            format!(
                "writing token artifact {}; output may be incomplete",
                path.display()
            )
        })
}

fn inspect(tokenizer: &Qwen2Tokenizer, path: &Path) -> Result<Inspection> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_REQUEST_BYTES,
        "request exceeds 8 MiB"
    );
    let mut request: ChatRequest = serde_json::from_slice(&bytes)?;
    normalize_message_content(&mut request.messages, &request.model)
        .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
    let tools = request.tools.unwrap_or_default();
    let template = ChatTemplate::Checkpoint(tokenizer.chat_template().map(str::to_owned));
    let prompt = render_prompt_for_template(&request.messages, &tools, &template)
        .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
    let output = request.max_tokens.unwrap_or(protocol::DEFAULT_MAX_TOKENS);
    ensure!(
        (1..=protocol::MAX_TOKENS).contains(&output),
        "max_tokens outside server bounds"
    );
    let token_ids = tokenizer.encode(&prompt, true)?;
    let evaluated = token_ids
        .len()
        .checked_add(output - 1)
        .context("position count overflow")?;
    Ok(Inspection {
        message_count: request.messages.len(),
        tool_count: tools.len(),
        rendered_prompt_bytes: prompt.len(),
        token_ids,
        output_tokens: output,
        minimum_evaluated_positions: evaluated,
        output_budget_source: if request.max_tokens.is_some() {
            "explicit"
        } else {
            "uncapped_server_default"
        },
    })
}

fn main() -> Result<()> {
    let args = parse_arguments(std::env::args_os().skip(1))?;
    let checkpoint = Gguf::open(&args.model).context("opening checkpoint metadata")?;
    let tokenizer = Qwen2Tokenizer::from_gguf(&checkpoint)?;
    let mut failed = false;
    let reports: Vec<_> = args
        .requests
        .iter()
        .map(|path| {
            let inspected = inspect(&tokenizer, path).and_then(|inspection| {
                if let Some(output) = &args.tokens_out {
                    write_tokens_exclusive(output, &inspection.token_artifact(&args.model, path))?;
                }
                Ok(inspection.report(path))
            });
            match inspected {
                Ok(report) => report,
                Err(error) => {
                    failed = true;
                    json!({"request": path, "status": "rejected", "error": format!("{error:#}")})
                }
            }
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "execution": "not_requested", "tokenization": "exact_checkpoint_metadata",
            "checkpoint": args.model, "requests": reports,
        }))?
    );
    ensure!(!failed, "one or more requests were rejected; see report");
    Ok(())
}

#[cfg(test)]
mod token_export_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn args(values: &[&str]) -> Result<Arguments> {
        parse_arguments(values.iter().map(|value| OsString::from(*value)))
    }

    #[test]
    fn positional_multiple_requests_remain_supported() {
        assert_eq!(
            args(&["model.gguf", "first.json", "second.json"]).unwrap(),
            Arguments {
                model: "model.gguf".into(),
                requests: vec!["first.json".into(), "second.json".into()],
                tokens_out: None,
            }
        );
    }

    #[test]
    fn export_option_can_precede_or_follow_positionals() {
        let expected = Arguments {
            model: "model.gguf".into(),
            requests: vec!["request.json".into()],
            tokens_out: Some("tokens.json".into()),
        };
        assert_eq!(
            args(&["--tokens-out", "tokens.json", "model.gguf", "request.json"]).unwrap(),
            expected
        );
        assert_eq!(
            args(&["model.gguf", "request.json", "--tokens-out", "tokens.json"]).unwrap(),
            expected
        );
    }

    #[test]
    fn malformed_or_ambiguous_exports_are_rejected() {
        for values in [
            vec![],
            vec!["model.gguf"],
            vec!["model.gguf", "request.json", "--tokens-out"],
            vec!["model.gguf", "request.json", "--tokens-out", ""],
            vec!["model.gguf", "request.json", "--tokens-out", "--"],
            vec!["model.gguf", "request.json", "--tokens-out", "--tokens-out"],
            vec![
                "model.gguf",
                "request.json",
                "--tokens-out",
                "one",
                "--tokens-out",
                "two",
            ],
            vec![
                "model.gguf",
                "one.json",
                "two.json",
                "--tokens-out",
                "tokens.json",
            ],
        ] {
            assert!(
                args(&values).is_err(),
                "accepted invalid arguments: {values:?}"
            );
        }
    }

    #[test]
    fn end_of_options_preserves_literal_request_paths() {
        let parsed = args(&["model.gguf", "--", "--tokens-out", "another.json"]).unwrap();
        assert_eq!(
            parsed.requests,
            vec![PathBuf::from("--tokens-out"), PathBuf::from("another.json")]
        );
        assert!(parsed.tokens_out.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn positional_paths_do_not_require_utf8() {
        use std::os::unix::ffi::OsStringExt;
        let request = OsString::from_vec(vec![b'r', 0xff]);
        let parsed = parse_arguments([OsString::from("model.gguf"), request.clone()]).unwrap();
        assert_eq!(parsed.requests, vec![PathBuf::from(request)]);
    }

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "prompt-inspect-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("create isolated test directory: {error}"),
                }
            }
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn compact_export_preserves_exact_ids_and_refuses_replacement() {
        let directory = TempDirectory::new();
        let path = directory.0.join("tokens.json");
        let inspection = Inspection {
            message_count: 2,
            tool_count: 39,
            rendered_prompt_bytes: 128,
            token_ids: vec![0, u32::MAX, 42, 42],
            output_tokens: 4096,
            minimum_evaluated_positions: 4099,
            output_budget_source: "explicit",
        };
        let artifact =
            inspection.token_artifact(Path::new("model.gguf"), Path::new("request.json"));
        write_tokens_exclusive(&path, &artifact).unwrap();
        let original = std::fs::read(&path).unwrap();
        let mut expected = serde_json::to_vec(&artifact).unwrap();
        expected.push(b'\n');
        assert_eq!(original, expected);
        let parsed: serde_json::Value = serde_json::from_slice(&original).unwrap();
        assert_eq!(parsed["format"], "beholdd.prompt-token-ids");
        assert_eq!(parsed["schemaVersion"], 1);
        assert_eq!(parsed["promptTokens"], json!([0, u32::MAX, 42, 42]));
        let report = inspection.report(Path::new("request.json"));
        for (artifact_key, report_key) in [
            ("promptTokenCount", "prompt_tokens"),
            ("messageCount", "message_count"),
            ("toolCount", "tool_count"),
            ("maxTokens", "output_tokens"),
            ("minimumContextPositions", "minimum_evaluated_positions"),
            ("renderedPromptBytes", "rendered_prompt_bytes"),
        ] {
            assert_eq!(parsed[artifact_key], report[report_key]);
        }
        assert!(write_tokens_exclusive(&path, &json!({"replacement": true})).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }

    #[test]
    fn output_creation_failure_is_reported_without_creating_parent_dirs() {
        let directory = TempDirectory::new();
        let path = directory.0.join("missing").join("tokens.json");
        let error = write_tokens_exclusive(&path, &json!({})).unwrap_err();
        assert!(format!("{error:#}").contains("exclusively creating token artifact"));
        assert!(!directory.0.join("missing").exists());
    }
}
