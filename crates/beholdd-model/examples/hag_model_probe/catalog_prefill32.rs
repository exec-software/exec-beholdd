// The first bounded prefix of the exact production-catalog prompt. Parsing is
// metadata-only and finishes before checkpoint payload access or device open.
use super::*;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

pub(super) const CPU_STEPS: usize = 32;
pub(super) const CHECK_POSITIONS: [usize; 2] = [0, CPU_STEPS - 1];
const PROMPT_TOKENS: usize = 8079;
const MAX_FILE_BYTES: u64 = 256 * 1024;
const KEYS: [&str; 13] = [
    "checkpoint",
    "format",
    "maxTokens",
    "messageCount",
    "minimumContextPositions",
    "parseSpecial",
    "promptTokenCount",
    "promptTokens",
    "renderedPromptBytes",
    "request",
    "schemaVersion",
    "tokenization",
    "toolCount",
];

pub(super) struct Prompt {
    pub(super) tokens: Vec<u32>,
}

fn parse(bytes: &[u8]) -> Result<Prompt> {
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_FILE_BYTES as usize,
        "catalog token artifact exceeds bound"
    );
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let mut canonical = serde_json::to_vec(&value)?;
    canonical.push(b'\n');
    // This also rejects duplicate keys and alternate numeric spellings. The
    // exporter writes the same compact Value representation and one newline.
    ensure!(
        canonical == bytes,
        "catalog token artifact is not canonical compact JSON"
    );
    let object = value
        .as_object()
        .context("catalog token artifact must be an object")?;
    ensure!(
        object.len() == KEYS.len() && object.keys().all(|key| KEYS.contains(&key.as_str())),
        "catalog token artifact field set differs"
    );
    for (key, expected) in [
        ("schemaVersion", 1),
        ("promptTokenCount", PROMPT_TOKENS as u64),
        ("toolCount", 39),
        ("messageCount", 2),
        ("maxTokens", 4096),
        ("minimumContextPositions", 12174),
        ("renderedPromptBytes", 33541),
    ] {
        ensure!(
            value[key].as_u64() == Some(expected),
            "catalog token artifact {key} differs"
        );
    }
    ensure!(
        value["format"].as_str() == Some("beholdd.prompt-token-ids")
            && value["tokenization"].as_str() == Some("exact_checkpoint_metadata")
            && value["parseSpecial"].as_bool() == Some(true),
        "catalog token artifact format/tokenization differs"
    );
    for key in ["checkpoint", "request"] {
        let path = value[key]
            .as_str()
            .context("catalog provenance path is not a string")?;
        ensure!(
            Path::new(path).is_absolute() && !path.contains('\0'),
            "catalog provenance path is not absolute"
        );
    }
    let ids = value["promptTokens"]
        .as_array()
        .context("catalog promptTokens is not an array")?;
    ensure!(ids.len() == PROMPT_TOKENS, "catalog token count differs");
    let tokens = ids
        .iter()
        .map(|value| {
            let token = value
                .as_u64()
                .context("catalog token ID is not an unsigned integer")?;
            ensure!(
                token < VOCAB as u64,
                "catalog token ID exceeds checkpoint vocabulary"
            );
            Ok(token as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Prompt { tokens })
}

fn stamp(metadata: &fs::Metadata) -> Result<(u64, u64, u64, i64, i64, i64, i64)> {
    ensure!(
        metadata.is_file(),
        "catalog token artifact must be a regular file"
    );
    Ok((
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    ))
}

pub(super) fn read(path: &Path) -> Result<Prompt> {
    ensure!(
        path.is_absolute() && fs::canonicalize(path)? == path,
        "catalog token artifact path is not canonical"
    );
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(0x20000 | 0x800)
        .open(path)?;
    let before = stamp(&file.metadata()?)?;
    ensure!(
        before == stamp(&fs::symlink_metadata(path)?)? && (1..=MAX_FILE_BYTES).contains(&before.2),
        "catalog token artifact extent/identity differs"
    );
    let mut bytes = Vec::new();
    (&mut file)
        .take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == before.2
            && before == stamp(&file.metadata()?)?
            && before == stamp(&fs::symlink_metadata(path)?)?
            && fs::canonicalize(path)? == path,
        "catalog token artifact changed while reading"
    );
    parse(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        json!({"format":"beholdd.prompt-token-ids", "schemaVersion":1,
            "tokenization":"exact_checkpoint_metadata", "parseSpecial":true,
            "checkpoint":"/model.gguf", "request":"/request.json",
            "promptTokens":vec![1u32;PROMPT_TOKENS], "promptTokenCount":PROMPT_TOKENS,
            "toolCount":39, "messageCount":2, "maxTokens":4096,
            "minimumContextPositions":12174, "renderedPromptBytes":33541})
    }
    fn encoded(value: &serde_json::Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn exact_catalog_tokens_and_all_metadata_are_required() {
        let value = fixture();
        assert_eq!(parse(&encoded(&value)).unwrap().tokens.len(), PROMPT_TOKENS);
        for (key, bad) in [
            ("format", json!("other")),
            ("schemaVersion", json!(2)),
            ("tokenization", json!("estimate")),
            ("parseSpecial", json!(false)),
            ("promptTokenCount", json!(PROMPT_TOKENS - 1)),
            ("toolCount", json!(38)),
            ("messageCount", json!(3)),
            ("maxTokens", json!(4095)),
            ("minimumContextPositions", json!(12175)),
            ("renderedPromptBytes", json!(33540)),
            ("checkpoint", json!("relative.gguf")),
            ("request", json!(null)),
            ("promptTokens", json!([1, 2])),
        ] {
            let mut changed = value.clone();
            changed[key] = bad;
            assert!(parse(&encoded(&changed)).is_err(), "{key}");
        }
        for bad in [json!(-1), json!(1.0), json!(VOCAB), json!("1"), json!(null)] {
            let mut changed = value.clone();
            changed["promptTokens"][PROMPT_TOKENS - 1] = bad;
            assert!(parse(&encoded(&changed)).is_err());
        }
        let mut changed = value.clone();
        changed["extra"] = json!(true);
        assert!(parse(&encoded(&changed)).is_err());
        let mut changed = value.clone();
        changed.as_object_mut().unwrap().remove("request");
        assert!(parse(&encoded(&changed)).is_err());
        assert!(parse(&serde_json::to_vec_pretty(&value).unwrap()).is_err());
        assert!(parse(&vec![b' '; MAX_FILE_BYTES as usize + 1]).is_err());
        let mut duplicate = encoded(&value);
        duplicate.splice(1..1, b"\"toolCount\":39,".iter().copied());
        assert!(parse(&duplicate).is_err());
    }

    #[test]
    fn catalog_artifact_requires_bounded_canonical_regular_file() {
        use std::os::unix::fs::symlink;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "hag-catalog-tokens-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("tokens.json");
        fs::write(&path, encoded(&fixture())).unwrap();
        assert!(read(&path).is_ok());
        let link = dir.join("link.json");
        symlink(&path, &link).unwrap();
        assert!(read(&link).is_err());
        assert!(read(&dir).is_err());
        fs::write(&path, vec![b' '; MAX_FILE_BYTES as usize + 1]).unwrap();
        assert!(read(&path).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
