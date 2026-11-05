//! Explicit pinned OpenMythos execution. Staged until whole-graph qualification.
//!
//! Only NativeRuntime's first actual generation constructs this owner. Checkpoint
//! metadata/template lookup, model registration and service startup remain cold.
//! Every activation operation is delegated to the checked native owner. The only
//! host inference operation is greedy selection from a complete finite vocabulary.

use std::ffi::OsStr;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use beholdd_backend::hag::native_execution::{MemoryBudget, NativeExecutionProfile};
use beholdd_format::{Gguf, RangeReader};
use beholdd_model::qwen35::recorded_model::{
    ModelLimits, RecordedQwen35Plan, ResidentQwen35CheckedModel,
};
use beholdd_model::qwen35::{config::Binding, weights::ReadLimits};
use beholdd_model::qwen3moe::{GenerateOutput, GenerationFinishReason};
use beholdd_model::Qwen2Tokenizer;
use sha2::{Digest, Sha256};

use super::{
    native_hag,
    native_phase::{Phase, Recorder},
};
use crate::ModelConfig;

const CHECKPOINT_BYTES: u64 = 16_547_399_872;
const CHECKPOINT_SHA256: [u8; 32] = [
    0xd3, 0xc3, 0x39, 0x8b, 0xa8, 0xb5, 0x12, 0xaa, 0x59, 0x07, 0x4e, 0xfc, 0xaa, 0x34, 0xd7, 0xdc,
    0xab, 0xed, 0x73, 0x81, 0xb6, 0x8a, 0x05, 0xae, 0x6b, 0xcc, 0xe7, 0x7e, 0x14, 0xd7, 0xdb, 0x45,
];
const TEMPLATE_SHA256: [u8; 32] = [
    0x55, 0xd4, 0x93, 0x14, 0x33, 0xfe, 0x50, 0x2b, 0x79, 0x42, 0x26, 0xee, 0x7f, 0x4d, 0x20, 0x6a,
    0x6b, 0xdd, 0x43, 0x6a, 0xc9, 0xf8, 0x0e, 0xb7, 0xd8, 0xeb, 0xb4, 0xc6, 0x39, 0xf9, 0xea, 0x0c,
];
const VOCAB: usize = 248_320;
const EOG: [u32; 5] = [248_044, 248_046, 248_063, 248_064, 248_065];
const EOG_TEXT: [&str; 5] = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
];
// These generated delimiters belong to the admitted text/reasoning/tool
// protocol. They must survive detokenization even when GGUF calls them CONTROL.
// No other control token is made visible by this model-specific policy.
const PROTOCOL_DELIMITERS: [(u32, &str); 4] = [
    (248_058, "<tool_call>"),
    (248_059, "</tool_call>"),
    (248_068, "<think>"),
    (248_069, "</think>"),
];
const HASH_CHUNK: usize = 65_536;
const MIN_HEADROOM: u64 = 128 * 1024 * 1024;

/// Detect ordinary replacement/truncation/in-place updates around verification and
/// each complete token. The deployment must keep the checkpoint immutable while
/// loaded; a metadata stamp is not protection against an adversarial file writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
impl FileStamp {
    fn read(path: &Path) -> Result<Self> {
        let m = std::fs::metadata(path).context("reading pinned checkpoint identity")?;
        ensure!(m.is_file(), "OpenMythos checkpoint is not a regular file");
        Ok(Self {
            device: m.dev(),
            inode: m.ino(),
            bytes: m.len(),
            modified: (m.mtime(), m.mtime_nsec()),
            changed: (m.ctime(), m.ctime_nsec()),
        })
    }
    fn unchanged(self, path: &Path) -> Result<()> {
        ensure!(
            Self::read(path)? == self,
            "pinned OpenMythos checkpoint changed; unload it"
        );
        Ok(())
    }
}

fn verify_tokenizer(tokenizer: &Qwen2Tokenizer) -> Result<()> {
    ensure!(
        tokenizer.pretokenizer() == "qwen35"
            && tokenizer.vocab_len() == VOCAB
            && tokenizer.eos_token_id() == 248_046
            && tokenizer.bos_token_id() == Some(248_044)
            && !tokenizer.adds_bos_token()
            && !tokenizer.adds_eos_token(),
        "OpenMythos tokenizer/automatic special-token policy differs from the pinned checkpoint"
    );
    for (&id, text) in EOG.iter().zip(EOG_TEXT) {
        ensure!(
            tokenizer.decode(&[id], true)? == text && tokenizer.encode(text, true)? == [id],
            "OpenMythos EOG identity differs"
        );
    }
    for (id, text) in PROTOCOL_DELIMITERS {
        ensure!(
            tokenizer.decode(&[id], true)? == text && tokenizer.encode(text, true)? == [id],
            "OpenMythos generated protocol delimiter identity differs"
        );
    }
    let template = tokenizer
        .chat_template()
        .context("OpenMythos template is missing")?;
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    ensure!(
        template.len() == 8057 && digest == TEMPLATE_SHA256,
        "OpenMythos checkpoint chat template differs"
    );
    Ok(())
}

/// Detokenization policy used only after the pinned OpenMythos owner is loaded.
/// The scheduler owns EOG stopping and usage; this second guard also prevents a
/// stop token from escaping if it is accidentally handed to the emitter.
pub(super) fn output_bytes(
    token: u32,
    mut decode: impl FnMut(u32, bool) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    if EOG.contains(&token) {
        return Ok(Vec::new());
    }
    if let Some((_, expected)) = PROTOCOL_DELIMITERS.iter().find(|(id, _)| *id == token) {
        let bytes = decode(token, true)?;
        ensure!(
            bytes == expected.as_bytes(),
            "OpenMythos generated protocol delimiter changed during detokenization"
        );
        return Ok(bytes);
    }
    decode(token, false)
}

// Generic reader seam tests bounded reads, cancellation and digest completeness
// without opening a checkpoint or trusting a filename/sidecar hash assertion.
fn hash_reader(
    bytes: u64,
    mut read: impl FnMut(u64, &mut [u8]) -> Result<()>,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<[u8; 32]> {
    ensure!(bytes > 0, "empty pinned checkpoint");
    let mut buffer = vec![0u8; HASH_CHUNK];
    let mut hasher = Sha256::new();
    let mut offset = 0u64;
    while offset < bytes {
        control()?;
        let n = usize::try_from((bytes - offset).min(HASH_CHUNK as u64))?;
        read(offset, &mut buffer[..n])?;
        control()?;
        hasher.update(&buffer[..n]);
        offset += n as u64;
    }
    control()?;
    Ok(hasher.finalize().into())
}

fn require_pinned_digest(digest: [u8; 32]) -> Result<()> {
    ensure!(
        digest == CHECKPOINT_SHA256,
        "OpenMythos SHA256 differs from the pinned checkpoint"
    );
    Ok(())
}

/// One serialized owner, holding no CPU inference or alternate-backend route.
pub(super) struct HagQwen35Execution {
    resident: Option<ResidentQwen35CheckedModel>,
    budget: MemoryBudget,
    healthy: Arc<AtomicBool>,
    checkpoint_path: PathBuf,
    checkpoint_stamp: FileStamp,
    logits: Vec<f32>,
    capacity: usize,
    cleanup_timeout: Duration,
}

impl HagQwen35Execution {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        checkpoint: &Arc<Gguf>,
        tokenizer: &Qwen2Tokenizer,
        config: &ModelConfig,
        process_budget: u64,
        runtime_headroom: u64,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        phases: &Recorder,
    ) -> Result<Self> {
        phases.emit(Phase::PreparationStart);
        let healthy = native_hag::process_health();
        native_hag::require_process_healthy()?;
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(OsStr::new("0")),
            "OpenMythos serving requires HAG_TRACE_SUBMISSION exactly 0"
        );
        let hag = config
            .hag
            .as_ref()
            .context("missing explicit HAG device/budget")?;
        let profile = config
            .hag_qwen35
            .as_ref()
            .context("missing explicit OpenMythos profile")?;
        control()?;
        verify_tokenizer(tokenizer)?;
        ensure!(
            checkpoint.shard_count() == 1 && checkpoint.shard_len(0) == Some(CHECKPOINT_BYTES),
            "OpenMythos requires the exact single-file Q4_K checkpoint extent"
        );
        let checkpoint_path = checkpoint.shards()[0].clone();
        let checkpoint_stamp = FileStamp::read(&checkpoint_path)?;
        ensure!(
            checkpoint_stamp.bytes == CHECKPOINT_BYTES,
            "OpenMythos path extent changed"
        );
        let binding = Binding::load(checkpoint)?;
        let c = binding.config();
        ensure!(
            [
                c.n_layers,
                c.dim,
                c.feed_forward,
                c.vocab,
                c.n_heads,
                c.n_kv_heads,
                c.key_head_dim,
                c.value_head_dim,
                c.rope_dim,
                c.full_attention_interval,
                c.ssm_conv_kernel,
                c.ssm_group_count,
                c.ssm_inner_size,
                c.ssm_state_size,
                c.ssm_time_step_rank
            ] == [64, 5120, 17408, VOCAB, 24, 4, 256, 256, 64, 4, 4, 16, 6144, 128, 48],
            "OpenMythos shape differs from the pinned native graph"
        );
        let reads = ReadLimits {
            tile_rows: profile.tile_rows,
            encoded_bytes: profile.encoded_read_bytes,
            decoded_bytes: profile.decoded_read_bytes,
        };
        let plan = RecordedQwen35Plan::new(&binding, hag.context_capacity, reads)?;
        let plan = if hag.context_capacity >= 4 {
            plan.with_prefill_chunk(4)?
        } else {
            plan
        };
        let capability = NativeExecutionProfile::Qwen35Probe
            .finite_check_spec()
            .context("native library lacks mandatory finite validation; no fallback")?;
        let requirements = plan.checked_requirements(Some(capability))?;
        let budget = MemoryBudget::new(hag.memory_budget_bytes);
        // Owner forecast already includes the complete caller logits buffer. Add
        // explicit prompt/output token and hash scratch allowances to headroom.
        let headroom = runtime_headroom
            .max(MIN_HEADROOM)
            .checked_add(
                u64::try_from(hag.context_capacity)?
                    .checked_mul(8)
                    .context("OpenMythos token storage overflows")?,
            )
            .and_then(|v| v.checked_add(HASH_CHUNK as u64))
            .context("OpenMythos runtime headroom overflows")?;
        let limits = ModelLimits {
            device_bytes: hag.memory_budget_bytes,
            total_process_bytes: process_budget,
            process_headroom_bytes: headroom,
        };
        let admission = plan.admit(checkpoint, requirements, limits, &budget)?;
        phases.emit(Phase::MemoryAdmissionStart);
        native_hag::admit_memory(
            admission.total_process_bytes,
            process_budget,
            admission.source_file_bytes,
        )?;
        phases.emit(Phase::MemoryAdmissionEnd);
        // Hash the SAME retained GGUF handles later used for all packed reads.
        // No payload bytes are touched before full-file/process/device admission.
        let digest = hash_reader(
            CHECKPOINT_BYTES,
            |offset, buffer| {
                checkpoint.read_range_into(0, offset..offset + buffer.len() as u64, buffer)
            },
            control,
        )?;
        require_pinned_digest(digest)?;
        checkpoint_stamp.unchanged(&checkpoint_path)?;
        control()?;
        // Re-sample live admission after potentially long disk/hash work.
        native_hag::admit_memory(
            admission.total_process_bytes,
            process_budget,
            admission.source_file_bytes,
        )?;
        phases.emit(Phase::DeviceOpenStart);
        let resident = plan.open_checked(
            &hag.device,
            Arc::clone(checkpoint),
            capability,
            limits,
            &budget,
            Arc::clone(&healthy),
            control,
        )?;
        phases.emit(Phase::DeviceOpenEnd);
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(VOCAB)
            .context("allocating complete OpenMythos logits")?;
        logits.resize(VOCAB, 0.0);
        let owner = Self {
            resident: Some(resident),
            budget,
            healthy,
            checkpoint_path,
            checkpoint_stamp,
            logits,
            capacity: hag.context_capacity,
            cleanup_timeout: Duration::from_millis(profile.cleanup_timeout_ms),
        };
        owner.checkpoint_stamp.unchanged(&owner.checkpoint_path)?;
        control()?;
        phases.emit(Phase::PreparationEnd);
        Ok(owner)
    }

    pub(super) fn health(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.healthy)
    }

    pub(super) fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        phases: &Recorder,
        on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenerateOutput> {
        native_hag::validate_request(self.capacity, prompt.len(), max_new_tokens)?;
        ensure!(
            prompt.iter().all(|&token| (token as usize) < VOCAB),
            "OpenMythos prompt token exceeds vocabulary"
        );
        native_hag::require_process_healthy()?;
        self.checkpoint_stamp
            .unchanged(&self.checkpoint_path)
            .inspect_err(|_| {
                self.healthy.store(false, Ordering::Release);
            })?;
        let resident = self
            .resident
            .as_mut()
            .context("OpenMythos owner is retired")?;
        let mut owner = CheckedTokenOwner {
            resident,
            logits: &mut self.logits,
            path: &self.checkpoint_path,
            stamp: self.checkpoint_stamp,
        };
        run_session(
            &mut owner,
            self.capacity,
            prompt,
            max_new_tokens,
            &EOG,
            &self.healthy,
            self.cleanup_timeout,
            control,
            on_token,
            phases,
        )
    }
}

impl Drop for HagQwen35Execution {
    fn drop(&mut self) {
        if let Some(owner) = self.resident.take() {
            let retirement = owner.retire();
            if !retirement.clean {
                self.healthy.store(false, Ordering::Release);
            }
        }
        if self.budget.current_bytes() != 0 {
            self.healthy.store(false, Ordering::Release);
        }
    }
}

trait TokenOwner {
    fn reset(&mut self, control: &mut dyn FnMut() -> Result<()>) -> Result<()>;
    fn prefill_chunk_size(&self) -> Option<usize> {
        None
    }
    fn advance_chunk(
        &mut self,
        _tokens: &[u32],
        _control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        anyhow::bail!("prefill chunk was not configured")
    }
    fn evaluate(
        &mut self,
        token: u32,
        select: bool,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<u32>>;
}
struct CheckedTokenOwner<'a> {
    resident: &'a mut ResidentQwen35CheckedModel,
    logits: &'a mut [f32],
    path: &'a Path,
    stamp: FileStamp,
}
impl TokenOwner for CheckedTokenOwner<'_> {
    fn prefill_chunk_size(&self) -> Option<usize> {
        self.resident.prefill_chunk_size()
    }
    fn advance_chunk(
        &mut self,
        tokens: &[u32],
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.stamp.unchanged(self.path)?;
        let before = self.resident.state()?;
        self.resident.advance_chunk_checked(tokens, control)?;
        let after = self.resident.state()?;
        ensure!(
            before.committed_positions.checked_add(tokens.len()) == Some(after.committed_positions)
                && !after.poisoned
                && after.process_healthy,
            "OpenMythos did not publish the complete prefill chunk"
        );
        self.stamp.unchanged(self.path)?;
        control()
    }
    fn reset(&mut self, control: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        self.resident.reset_checked(control)?;
        let state = self.resident.state()?;
        ensure!(
            state.committed_positions == 0 && !state.poisoned && state.process_healthy,
            "OpenMythos reset did not publish healthy empty state"
        );
        Ok(())
    }
    fn evaluate(
        &mut self,
        token: u32,
        select: bool,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<u32>> {
        self.stamp.unchanged(self.path)?;
        let before = self.resident.state()?;
        let ticket = if select {
            Some(self.resident.step_checked(token, control)?)
        } else {
            self.resident.advance_checked(token, control)?;
            None
        };
        let after = self.resident.state()?;
        ensure!(
            before.committed_positions.checked_add(1) == Some(after.committed_positions)
                && !after.poisoned
                && after.process_healthy,
            "OpenMythos did not publish exactly one complete position"
        );
        // Earlier prompt positions advance only checked recurrent/KV state.
        // The final prompt/decode token retains the complete checked head and
        // full-vocabulary host transfer before selection.
        let selected = if let Some(ticket) = ticket {
            self.resident.read_logits(&ticket, self.logits, control)?;
            Some(greedy_token(self.logits)?)
        } else {
            None
        };
        self.stamp.unchanged(self.path)?;
        control()?;
        Ok(selected)
    }
}

struct Session<'a, O: TokenOwner> {
    owner: &'a mut O,
    healthy: &'a AtomicBool,
    cleanup_timeout: Duration,
    active: bool,
    phases: &'a Recorder,
}
impl<O: TokenOwner> Session<'_, O> {
    fn finish(&mut self) -> Result<()> {
        self.phases.emit(Phase::FinalResetStart);
        // A cancelled request callback is deliberately NOT reused for cleanup.
        let deadline = Instant::now()
            .checked_add(self.cleanup_timeout)
            .context("OpenMythos cleanup deadline overflows")?;
        let healthy = self.healthy;
        let result = self.owner.reset(&mut || {
            ensure!(
                healthy.load(Ordering::Acquire),
                "native health is latched failed"
            );
            ensure!(
                Instant::now() < deadline,
                "OpenMythos cleanup deadline exceeded"
            );
            Ok(())
        });
        self.phases.emit(if result.is_ok() {
            Phase::FinalResetOk
        } else {
            Phase::FinalResetFailed
        });
        if result.is_err() {
            self.healthy.store(false, Ordering::Release);
        }
        // Keep the unwind guard armed until cleanup actually returns. A panic
        // inside reset must lower the process latch just like a token/sink panic.
        self.active = false;
        result
    }
}
impl<O: TokenOwner> Drop for Session<'_, O> {
    fn drop(&mut self) {
        // Unwinding cannot certify safe cancellation or completed reset. Never
        // reopen a poisoned runtime mutex or restore the shared one-way latch.
        if self.active {
            self.healthy.store(false, Ordering::Release);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_session(
    owner: &mut impl TokenOwner,
    capacity: usize,
    prompt: &[u32],
    max_new_tokens: usize,
    stops: &[u32],
    healthy: &AtomicBool,
    cleanup_timeout: Duration,
    control: &mut dyn FnMut() -> Result<()>,
    mut emit: impl FnMut(u32) -> Result<()>,
    phases: &Recorder,
) -> Result<GenerateOutput> {
    native_hag::validate_request(capacity, prompt.len(), max_new_tokens)?;
    ensure!(
        healthy.load(Ordering::Acquire),
        "native health is latched failed"
    );
    ensure!(!stops.is_empty(), "OpenMythos requires an explicit EOG set");
    let callback_failed = std::cell::Cell::new(false);
    let mut checked_control = || {
        let result = control();
        callback_failed.set(callback_failed.get() || result.is_err());
        result
    };
    // Prepare bounded result storage before any model mutation.
    let mut generated = Vec::new();
    generated.try_reserve_exact(max_new_tokens)?;
    let mut session = Session {
        owner,
        healthy,
        cleanup_timeout,
        active: true,
        phases,
    };
    let result = (|| -> Result<GenerateOutput> {
        let chunk_size = session.owner.prefill_chunk_size();
        ensure!(
            chunk_size.is_none_or(|n| matches!(n, 1 | 2 | 4)),
            "OpenMythos owner has an invalid prefill chunk size"
        );
        checked_control()?;
        phases.emit(Phase::InitialResetStart);
        let reset = session.owner.reset(&mut checked_control);
        phases.emit(if reset.is_ok() {
            Phase::InitialResetOk
        } else {
            Phase::InitialResetFailed
        });
        reset?;
        phases.emit(Phase::PrefillStart);
        let mut positions = 0usize;
        // Only the non-final prefix may use state-only chunks. A short tail
        // uses scalar advancement; final prompt and decode always select logits.
        let prefix_len = prompt.len() - 1;
        while positions < prefix_len {
            checked_control()?;
            if let Some(count) = chunk_size.filter(|&count| count <= prefix_len - positions) {
                session
                    .owner
                    .advance_chunk(&prompt[positions..positions + count], &mut checked_control)?;
                positions += count;
            } else {
                ensure!(
                    session
                        .owner
                        .evaluate(prompt[positions], false, &mut checked_control)?
                        .is_none(),
                    "state-only OpenMythos prefill unexpectedly selected a token"
                );
                positions += 1;
            }
            phases.position_fenced(positions);
            checked_control()?;
        }
        checked_control()?;
        let selected = session
            .owner
            .evaluate(prompt[prefix_len], true, &mut checked_control)?;
        positions += 1;
        phases.position_fenced(positions);
        checked_control()?;
        phases.emit(Phase::PrefillEnd);
        let mut next =
            selected.context("complete OpenMythos prefill omitted vocabulary selection")?;
        let mut decode_started = false;
        for step in 0..max_new_tokens {
            checked_control()?;
            if stops.contains(&next) {
                if decode_started {
                    phases.emit(Phase::DecodeEnd);
                }
                return Ok(GenerateOutput {
                    generated,
                    stop_token: Some(next),
                    finish_reason: GenerationFinishReason::StopToken,
                });
            }
            let delivered = emit(next);
            callback_failed.set(callback_failed.get() || delivered.is_err());
            delivered?;
            checked_control()?;
            generated.push(next);
            if step + 1 < max_new_tokens {
                if !decode_started {
                    phases.emit(Phase::DecodeStart);
                    decode_started = true;
                }
                checked_control()?;
                next = session
                    .owner
                    .evaluate(next, true, &mut checked_control)?
                    .context("OpenMythos decode omitted complete vocabulary selection")?;
                positions += 1;
                phases.position_fenced(positions);
                checked_control()?;
            }
        }
        if decode_started {
            phases.emit(Phase::DecodeEnd);
        }
        Ok(GenerateOutput {
            generated,
            stop_token: None,
            finish_reason: GenerationFinishReason::MaxNewTokens,
        })
    })();
    // Preserve the existing serving policy: an unexplained operator/source
    // failure quarantines HAG process-wide. Observed request/sink cancellation
    // may recover only through a complete fresh-deadline healthy reset.
    if result.is_err() && !callback_failed.get() {
        healthy.store(false, Ordering::Release);
    }
    let cleanup = session.finish();
    match (result, cleanup) {
        // Cleanup has its own deadline, but a request that expired during that
        // cleanup must not publish a successful completion afterward.
        (Ok(output), Ok(())) => {
            checked_control()?;
            Ok(output)
        }
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => {
            Err(error.context(format!("OpenMythos cleanup also failed: {cleanup:#}")))
        }
    }
}

fn greedy_token(logits: &[f32]) -> Result<u32> {
    ensure!(
        logits.len() == VOCAB,
        "OpenMythos sampler requires the complete vocabulary"
    );
    let mut best = (0usize, f32::NEG_INFINITY);
    for (index, &value) in logits.iter().enumerate() {
        ensure!(value.is_finite(), "OpenMythos logit {index} is not finite");
        // Strict comparison preserves the existing runtime's lowest-index tie.
        if value > best.1 {
            best = (index, value);
        }
    }
    Ok(u32::try_from(best.0)?)
}

#[cfg(test)]
#[path = "native_qwen35_tests.rs"]
mod tests;
