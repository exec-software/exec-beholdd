//! The exact-byte logits artifact: a deterministic, little-endian record of one decode's
//! prompt, generated tokens and per-step logits.
//!
//! Byte-identical artifacts across independent runs are the evidence behind this
//! project's `Exact` fit-policy gates (ROADMAP.md M1.3, M2.2, M2.6), so the writer and its
//! validation live here, in the library, rather than in `examples/decode.rs`. Cargo's
//! default test selection silently skips example targets; `cargo test --workspace` alone
//! never ran these tests when they lived there.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::DecodeOutput;

const LOGITS_MAGIC: &[u8; 8] = b"BHDLOGIT";
const LOGITS_FOOTER_MAGIC: &[u8; 8] = b"BHDLEND!";
const LOGITS_VERSION: u16 = 1;
const LOGITS_HEADER_LEN: u16 = 64;
const LOGITS_SCALAR_F32: u32 = 1;

#[derive(Debug, Clone, Copy)]
struct ArtifactLengths {
    body: usize,
    total: usize,
}

/// Write `output` to `path` as the exact little-endian logits artifact.
///
/// Refuses to overwrite an existing file and refuses a self-inconsistent `output` before
/// any byte is written. The write is atomic: bytes land in a sibling temporary file
/// first, are `fsync`ed and length-checked, then published with `hard_link` so a reader
/// never observes a partial artifact at `path`, and so a second writer publishing to the
/// same path fails instead of silently overwriting the first.
pub fn write_logits_artifact(path: &Path, output: &DecodeOutput) -> Result<()> {
    validate_decode_output(output)?;
    refuse_existing_output(path)?;

    let prompt = output
        .tokens
        .get(..output.prompt_len)
        .context("decode output is missing prompt token IDs")?;
    let step_count = output.generated.len();
    let logit_count = checked_product(step_count, output.vocab, "logit count")?;
    let body_values = prompt
        .len()
        .checked_add(step_count)
        .and_then(|count| count.checked_add(logit_count))
        .context("logits artifact body value count overflows")?;
    let body_len = body_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("logits artifact body byte count overflows")?;
    let total_len = usize::from(LOGITS_HEADER_LEN)
        .checked_add(body_len)
        .and_then(|length| length.checked_add(16))
        .context("logits artifact total byte count overflows")?;

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .context("logits artifact path has no filename")?
        .to_string_lossy();
    let mut temp_path = None;
    let mut temp_file = None;
    for nonce in 0..1024u32 {
        let candidate = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temp_path = Some(candidate);
                temp_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("creating temporary logits artifact {}", candidate.display())
                });
            }
        }
    }
    let temp_path =
        temp_path.context("could not choose a unique temporary logits artifact name")?;
    let result = write_and_publish_logits(
        temp_file.expect("temporary file and path are created together"),
        &temp_path,
        path,
        output,
        prompt,
        logit_count,
        ArtifactLengths {
            body: body_len,
            total: total_len,
        },
    );
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// Refuse if `path` already exists (as any kind of filesystem entry, including a broken
/// symlink), rather than silently overwriting it. Callers that want to fail fast before
/// doing expensive work can call this ahead of [`write_logits_artifact`], which also
/// checks it internally as its actual guarantee.
pub fn refuse_existing_output(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => bail!(
            "refusing to overwrite existing logits artifact {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("checking logits artifact {}", path.display()))
        }
    }
}

fn write_and_publish_logits(
    file: File,
    temp_path: &Path,
    path: &Path,
    output: &DecodeOutput,
    prompt: &[u32],
    logit_count: usize,
    lengths: ArtifactLengths,
) -> Result<()> {
    let mut writer = BufWriter::new(file);
    writer.write_all(LOGITS_MAGIC)?;
    writer.write_all(&LOGITS_VERSION.to_le_bytes())?;
    writer.write_all(&LOGITS_HEADER_LEN.to_le_bytes())?;
    writer.write_all(&LOGITS_SCALAR_F32.to_le_bytes())?;
    for value in [
        prompt.len(),
        output.generated.len(),
        output.vocab,
        logit_count,
        lengths.body,
    ] {
        writer.write_all(&u64::try_from(value)?.to_le_bytes())?;
    }
    writer.write_all(&0u64.to_le_bytes())?;
    for &token in prompt.iter().chain(&output.generated) {
        writer.write_all(&token.to_le_bytes())?;
    }
    for &logit in &output.step_logits {
        writer.write_all(&logit.to_bits().to_le_bytes())?;
    }
    writer.write_all(&u64::try_from(lengths.total)?.to_le_bytes())?;
    writer.write_all(LOGITS_FOOTER_MAGIC)?;
    writer.flush()?;
    let file = writer.into_inner().map_err(|error| error.into_error())?;
    file.sync_all()?;
    let actual_len = usize::try_from(file.metadata()?.len())?;
    if actual_len != lengths.total {
        bail!(
            "temporary logits artifact is {actual_len} bytes, expected {}",
            lengths.total
        );
    }
    drop(file);

    std::fs::hard_link(temp_path, path).with_context(|| {
        format!(
            "publishing logits artifact {} without overwriting an existing file",
            path.display()
        )
    })?;
    std::fs::remove_file(temp_path)
        .with_context(|| format!("removing temporary logits artifact {}", temp_path.display()))?;
    Ok(())
}

fn validate_decode_output(output: &DecodeOutput) -> Result<()> {
    if output.prompt_len == 0 {
        bail!("logits artifact requires a non-empty prompt");
    }
    if output.generated.is_empty() {
        bail!("logits artifact requires at least one decode step");
    }
    if output.vocab == 0 {
        bail!("decode output has an empty vocabulary");
    }
    let expected_tokens = output
        .prompt_len
        .checked_add(output.generated.len())
        .context("decode output token count overflows")?;
    if output.tokens.len() != expected_tokens {
        bail!(
            "decode output has {} tokens, expected {expected_tokens}",
            output.tokens.len()
        );
    }
    if output.tokens.get(output.prompt_len..) != Some(output.generated.as_slice()) {
        bail!("decode output token suffix does not match generated IDs");
    }
    let expected_logits = checked_product(output.generated.len(), output.vocab, "logit count")?;
    if output.step_logits.len() != expected_logits {
        bail!(
            "decode output has {} logits, expected {expected_logits}",
            output.step_logits.len()
        );
    }
    for &token in &output.tokens {
        if usize::try_from(token).context("token ID does not fit usize")? >= output.vocab {
            bail!(
                "decode output token ID {token} is outside vocabulary {}",
                output.vocab
            );
        }
    }
    if let Some((index, value)) = output
        .step_logits
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("decode output logit {index} is not finite: {value}");
    }
    for (step, &generated) in output.generated.iter().enumerate() {
        let row_start = checked_product(step, output.vocab, "logit row offset")?;
        let row_end = row_start
            .checked_add(output.vocab)
            .context("logit row end overflows")?;
        let row = output
            .step_logits
            .get(row_start..row_end)
            .context("decode output is missing a complete logit row")?;
        let mut best = 0usize;
        for index in 1..row.len() {
            if row[index] > row[best] {
                best = index;
            }
        }
        if usize::try_from(generated).context("generated token does not fit usize")? != best {
            bail!(
                "decode output generated token {generated} at step {step}, but deterministic argmax is {best}"
            );
        }
    }
    Ok(())
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} overflows: {left} * {right}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn output() -> DecodeOutput {
        DecodeOutput {
            prompt_len: 2,
            vocab: 3,
            tokens: vec![1, 2, 2, 2],
            generated: vec![2, 2],
            step_logits: vec![0.0, -0.0, 1.5, -2.0, 3.25, 4.0],
        }
    }

    fn path(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "beholdd-{label}-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[test]
    fn exact_logits_artifact_has_stable_little_endian_layout() -> Result<()> {
        let path = path("logits-layout");
        write_logits_artifact(&path, &output())?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;

        assert_eq!(&bytes[..8], LOGITS_MAGIC);
        assert_eq!(&bytes[8..10], &1u16.to_le_bytes());
        assert_eq!(&bytes[10..12], &64u16.to_le_bytes());
        assert_eq!(&bytes[16..24], &2u64.to_le_bytes());
        assert_eq!(&bytes[24..32], &2u64.to_le_bytes());
        assert_eq!(&bytes[32..40], &3u64.to_le_bytes());
        assert_eq!(&bytes[40..48], &6u64.to_le_bytes());
        assert_eq!(
            &bytes[64..80],
            &[1u32, 2, 2, 2].map(u32::to_le_bytes).concat()
        );
        assert_eq!(&bytes[80..84], &0.0f32.to_bits().to_le_bytes());
        assert_eq!(&bytes[84..88], &(-0.0f32).to_bits().to_le_bytes());
        assert_ne!(&bytes[80..84], &bytes[84..88]);
        assert_eq!(&bytes[bytes.len() - 8..], LOGITS_FOOTER_MAGIC);
        assert_eq!(bytes.len(), 120);
        Ok(())
    }

    #[test]
    fn exact_logits_artifact_is_deterministic_and_refuses_clobber() -> Result<()> {
        let first = path("logits-first");
        let second = path("logits-second");
        write_logits_artifact(&first, &output())?;
        write_logits_artifact(&second, &output())?;
        assert_eq!(std::fs::read(&first)?, std::fs::read(&second)?);
        let before = std::fs::read(&first)?;
        assert!(write_logits_artifact(&first, &output()).is_err());
        assert_eq!(std::fs::read(&first)?, before);
        std::fs::remove_file(first)?;
        std::fs::remove_file(second)?;
        Ok(())
    }

    #[test]
    fn exact_logits_artifact_refuses_invalid_results() {
        let mut invalid = output();
        invalid.tokens.pop();
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.tokens[3] = 1;
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.step_logits.pop();
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.tokens[0] = 3;
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.step_logits[0] = f32::NAN;
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.prompt_len = 0;
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.tokens.truncate(invalid.prompt_len);
        invalid.generated.clear();
        invalid.step_logits.clear();
        assert!(validate_decode_output(&invalid).is_err());
        let mut invalid = output();
        invalid.generated[0] = 0;
        invalid.tokens[2] = 0;
        assert!(validate_decode_output(&invalid).is_err());
    }
}
