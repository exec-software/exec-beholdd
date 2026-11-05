//! Source-only lifecycle tests. Fakes certify scheduling, never native arithmetic.

use super::*;
use crate::{
    ExecutionBackend, HagConfig, HagQwen35Config, InferenceRuntime, ModelCompatibility,
    NativeRuntime, RuntimeConfig,
};
use anyhow::bail;
use std::cell::Cell;
use std::collections::VecDeque;

#[derive(Default)]
struct FakeOwner {
    evaluated: Vec<(u32, bool)>,
    chunk_size: Option<usize>,
    chunks: Vec<Vec<u32>>,
    outputs: VecDeque<u32>,
    resets: usize,
    position: usize,
    failed_reset: Option<usize>,
    panic_reset: Option<usize>,
    operator_failure: bool,
    cancel_in_evaluate: bool,
}
impl TokenOwner for FakeOwner {
    fn prefill_chunk_size(&self) -> Option<usize> {
        self.chunk_size
    }
    fn advance_chunk(
        &mut self,
        tokens: &[u32],
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            self.chunk_size == Some(tokens.len()),
            "fake chunk size differs"
        );
        self.chunks.push(tokens.to_vec());
        for _ in tokens {
            self.position += 1; // Deliberate partial internal effect before cancellation.
            ensure!(!self.operator_failure, "injected chunk operator failure");
            if self.cancel_in_evaluate {
                control()?;
            }
        }
        Ok(())
    }
    fn reset(&mut self, control: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        control()?;
        self.resets += 1;
        assert_ne!(self.panic_reset, Some(self.resets), "injected reset panic");
        ensure!(
            self.failed_reset != Some(self.resets),
            "injected reset failure"
        );
        self.position = 0;
        control()?;
        Ok(())
    }
    fn evaluate(
        &mut self,
        token: u32,
        select: bool,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<u32>> {
        self.evaluated.push((token, select));
        self.position += 1; // Deliberate partial effect before a late callback.
        ensure!(!self.operator_failure, "injected operator failure");
        if self.cancel_in_evaluate {
            control()?;
        }
        Ok(if select {
            Some(self.outputs.pop_front().context("no fake output")?)
        } else {
            None
        })
    }
}

fn run(
    owner: &mut FakeOwner,
    prompt: &[u32],
    maximum: usize,
    health: &AtomicBool,
    control: &mut dyn FnMut() -> Result<()>,
    emit: impl FnMut(u32) -> Result<()>,
) -> Result<GenerateOutput> {
    run_session(
        owner,
        32,
        prompt,
        maximum,
        &EOG,
        health,
        Duration::from_secs(1),
        control,
        emit,
        &Recorder::new(false),
    )
}

#[test]
fn full_vocab_sampler_checks_last_coordinate_preserves_ties_and_rejects_nonfinite() {
    let mut logits = vec![-9.0; VOCAB];
    logits[VOCAB - 1] = 4.0;
    assert_eq!(greedy_token(&logits).unwrap(), (VOCAB - 1) as u32);
    logits[7] = 4.0;
    assert_eq!(greedy_token(&logits).unwrap(), 7);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        logits[VOCAB - 1] = value;
        assert!(greedy_token(&logits).is_err());
    }
    assert!(greedy_token(&logits[..VOCAB - 1]).is_err());
    assert!(greedy_token(&[]).is_err());
    logits.fill(-0.0);
    logits[1] = 0.0;
    assert_eq!(greedy_token(&logits).unwrap(), 0);
    logits[2] = f32::from_bits(1);
    assert_eq!(greedy_token(&logits).unwrap(), 2);
}

#[test]
fn all_pinned_eog_tokens_stop_without_delivery_or_extra_evaluation() {
    for stop in EOG {
        let mut owner = FakeOwner {
            outputs: VecDeque::from([stop]),
            ..Default::default()
        };
        let health = AtomicBool::new(true);
        let out = run(&mut owner, &[10, 11], 8, &health, &mut || Ok(()), |_| {
            bail!("EOG must never reach the sink")
        })
        .unwrap();
        assert!(out.generated.is_empty());
        assert_eq!(out.stop_token, Some(stop));
        assert_eq!(out.finish_reason, GenerationFinishReason::StopToken);
        assert_eq!(owner.evaluated, [(10, false), (11, true)]);
        assert_eq!((owner.resets, owner.position), (2, 0));
        assert!(health.load(Ordering::Acquire));
    }
}

#[test]
fn last_delivered_token_is_not_evaluated_and_full_context_one_output_fits() {
    let mut owner = FakeOwner {
        outputs: VecDeque::from([40, 41, 42]),
        ..Default::default()
    };
    let mut delivered = Vec::new();
    let health = AtomicBool::new(true);
    let out = run(&mut owner, &[10, 11], 3, &health, &mut || Ok(()), |t| {
        delivered.push(t);
        Ok(())
    })
    .unwrap();
    assert_eq!(out.generated, [40, 41, 42]);
    assert_eq!(delivered, out.generated);
    assert_eq!(out.finish_reason, GenerationFinishReason::MaxNewTokens);
    assert_eq!(
        owner.evaluated,
        [(10, false), (11, true), (40, true), (41, true)]
    );
    assert_eq!((owner.resets, owner.position), (2, 0));
    let mut boundary = FakeOwner {
        outputs: VecDeque::from([7]),
        ..Default::default()
    };
    run_session(
        &mut boundary,
        2,
        &[1, 2],
        1,
        &EOG,
        &health,
        Duration::from_secs(1),
        &mut || Ok(()),
        |_| Ok(()),
        &Recorder::new(false),
    )
    .unwrap();
    assert_eq!(boundary.evaluated.len(), 2);
    let before = (boundary.resets, boundary.evaluated.len());
    assert!(run_session(
        &mut boundary,
        2,
        &[1, 2],
        2,
        &EOG,
        &health,
        Duration::from_secs(1),
        &mut || panic!("over-capacity control"),
        |_| panic!("over-capacity sink"),
        &Recorder::new(false)
    )
    .is_err());
    assert_eq!(before, (boundary.resets, boundary.evaluated.len()));
}

#[test]
fn prompt_prefix_advances_state_but_final_prompt_and_decode_select_full_logits() {
    let mut owner = FakeOwner {
        outputs: VecDeque::from([40, 41]),
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    let out = run(
        &mut owner,
        &[10, 11, 12],
        2,
        &health,
        &mut || Ok(()),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(out.generated, [40, 41]);
    assert_eq!(
        owner.evaluated,
        [(10, false), (11, false), (12, true), (40, true)]
    );
    assert!(owner.outputs.is_empty());
    assert_eq!((owner.resets, owner.position), (2, 0));
    assert!(health.load(Ordering::Acquire));

    // A single prompt token must still evaluate and select the complete head.
    owner.evaluated.clear();
    owner.outputs.push_back(42);
    let one = run(&mut owner, &[13], 1, &health, &mut || Ok(()), |_| Ok(())).unwrap();
    assert_eq!(one.generated, [42]);
    assert_eq!(owner.evaluated, [(13, true)]);
    assert_eq!((owner.resets, owner.position), (4, 0));
}

#[test]
fn fixed_chunks_cover_only_complete_prefix_groups_and_preserve_tail_head_and_decode() {
    let prompt = [10, 11, 12, 13, 14, 15];
    for count in [1, 2, 4] {
        let mut owner = FakeOwner {
            chunk_size: Some(count),
            outputs: VecDeque::from([40, 41]),
            ..Default::default()
        };
        let health = AtomicBool::new(true);
        let result = run(&mut owner, &prompt, 2, &health, &mut || Ok(()), |_| Ok(())).unwrap();
        let chunked = 5 / count * count;
        assert_eq!(
            owner.chunks,
            prompt[..chunked]
                .chunks_exact(count)
                .map(|v| v.to_vec())
                .collect::<Vec<_>>()
        );
        let mut scalar: Vec<_> = prompt[chunked..5].iter().map(|&t| (t, false)).collect();
        scalar.extend([(15, true), (40, true)]);
        assert_eq!(owner.evaluated, scalar);
        assert_eq!(result.generated, [40, 41]);
        assert!(owner.outputs.is_empty());
        assert_eq!((owner.resets, owner.position), (2, 0));
        assert!(health.load(Ordering::Acquire));
    }
}

#[test]
fn short_prompts_never_fill_a_chunk_with_final_or_unused_tokens() {
    for length in 1..=5 {
        let prompt: Vec<_> = (10..10 + length as u32).collect();
        let mut owner = FakeOwner {
            chunk_size: Some(4),
            outputs: VecDeque::from([40]),
            ..Default::default()
        };
        let health = AtomicBool::new(true);
        let result = run(&mut owner, &prompt, 1, &health, &mut || Ok(()), |_| Ok(())).unwrap();
        if length < 5 {
            assert!(owner.chunks.is_empty());
            assert_eq!(
                owner.evaluated,
                prompt
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| (t, i + 1 == length))
                    .collect::<Vec<_>>()
            );
        } else {
            assert_eq!(owner.chunks, [prompt[..4].to_vec()]);
            assert_eq!(owner.evaluated, [(prompt[4], true)]);
        }
        assert_eq!(result.generated, [40]);
        assert_eq!((owner.resets, owner.position), (2, 0));
    }
}

#[test]
fn partial_chunk_cancellation_resets_before_retry_without_selecting_logits() {
    let mut owner = FakeOwner {
        chunk_size: Some(4),
        outputs: VecDeque::from([7]),
        cancel_in_evaluate: true,
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    let calls = Cell::new(0);
    let error = run(
        &mut owner,
        &[1, 2, 3, 4, 5],
        1,
        &health,
        &mut || {
            calls.set(calls.get() + 1);
            ensure!(calls.get() != 5, "cancel inside partial chunk");
            Ok(())
        },
        |_| panic!("partial chunk cannot publish a token"),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("cancel inside partial chunk"));
    assert_eq!(owner.chunks, [vec![1, 2, 3, 4]]);
    assert!(owner.evaluated.is_empty());
    assert_eq!(owner.outputs, VecDeque::from([7]));
    assert_eq!((owner.resets, owner.position), (2, 0));
    assert!(health.load(Ordering::Acquire));
    owner.chunks.clear();
    owner.cancel_in_evaluate = false;
    let retry = run(
        &mut owner,
        &[1, 2, 3, 4, 5],
        1,
        &health,
        &mut || Ok(()),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(retry.generated, [7]);
    assert_eq!(owner.chunks, [vec![1, 2, 3, 4]]);
    assert_eq!(owner.evaluated, [(5, true)]);
    assert_eq!((owner.resets, owner.position), (4, 0));
}

#[test]
fn unexplained_chunk_failure_latches_health_and_never_selects() {
    let mut owner = FakeOwner {
        chunk_size: Some(4),
        outputs: VecDeque::from([7]),
        operator_failure: true,
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    assert!(run(
        &mut owner,
        &[1, 2, 3, 4, 5],
        1,
        &health,
        &mut || Ok(()),
        |_| { panic!("failed chunk cannot publish") }
    )
    .is_err());
    assert_eq!(owner.chunks, [vec![1, 2, 3, 4]]);
    assert!(owner.evaluated.is_empty());
    assert_eq!(owner.outputs, VecDeque::from([7]));
    assert!(!health.load(Ordering::Acquire));
}

#[test]
fn cancellation_inside_prefix_state_advance_resets_and_retries_before_any_selection() {
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        cancel_in_evaluate: true,
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    let calls = Cell::new(0usize);
    let error = run(
        &mut owner,
        &[1, 2],
        1,
        &health,
        &mut || {
            calls.set(calls.get() + 1);
            ensure!(calls.get() < 5, "cancel inside prefix advance");
            Ok(())
        },
        |_| panic!("prefix cancellation must not publish a token"),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("cancel inside prefix advance"));
    assert_eq!(owner.evaluated, [(1, false)]);
    assert_eq!(owner.outputs, VecDeque::from([7]));
    assert_eq!((owner.resets, owner.position), (2, 0));
    assert!(health.load(Ordering::Acquire));

    owner.evaluated.clear();
    owner.cancel_in_evaluate = false;
    let retry = run(&mut owner, &[1, 2], 1, &health, &mut || Ok(()), |_| Ok(())).unwrap();
    assert_eq!(retry.generated, [7]);
    assert_eq!(owner.evaluated, [(1, false), (2, true)]);
    assert_eq!((owner.resets, owner.position), (4, 0));
}

#[test]
fn unexplained_prefix_state_advance_error_latches_process_health() {
    let mut owner = FakeOwner {
        operator_failure: true,
        outputs: VecDeque::from([7]),
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    assert!(run(&mut owner, &[1, 2], 1, &health, &mut || Ok(()), |_| {
        panic!("failed prefix must not publish")
    })
    .is_err());
    assert_eq!(owner.evaluated, [(1, false)]);
    assert_eq!(owner.outputs, VecDeque::from([7]));
    assert!(!health.load(Ordering::Acquire));
}

#[test]
fn sink_cancellation_uses_fresh_cleanup_control_and_owner_can_retry() {
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    let cancelled = Cell::new(false);
    let error = run(
        &mut owner,
        &[1],
        2,
        &health,
        &mut || {
            ensure!(!cancelled.get(), "request cancelled forever");
            Ok(())
        },
        |_| {
            cancelled.set(true);
            bail!("sink cancelled")
        },
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("sink cancelled"));
    assert_eq!((owner.resets, owner.position), (2, 0));
    assert!(health.load(Ordering::Acquire));
    owner.outputs.push_back(9);
    let retry = run(&mut owner, &[2], 1, &health, &mut || Ok(()), |_| Ok(())).unwrap();
    assert_eq!(retry.generated, [9]);
    assert_eq!((owner.resets, owner.position), (4, 0));
}

#[test]
fn cancellation_inside_partial_evaluation_resets_without_publishing_a_token() {
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        cancel_in_evaluate: true,
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    let calls = Cell::new(0usize);
    // initial boundary, two reset controls, prompt boundary, then inner callback.
    let error = run(
        &mut owner,
        &[1],
        1,
        &health,
        &mut || {
            calls.set(calls.get() + 1);
            ensure!(calls.get() < 5, "late request cancellation");
            Ok(())
        },
        |_| panic!("partial evaluation must not publish"),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("late request cancellation"));
    assert_eq!(owner.evaluated, [(1, true)]);
    assert_eq!((owner.resets, owner.position), (2, 0));
    assert!(health.load(Ordering::Acquire));
}

#[test]
fn cancellation_at_final_publication_keeps_completed_reset_but_refuses_success() {
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    let mut controls = 0;
    let error = run(
        &mut owner,
        &[1],
        1,
        &health,
        &mut || {
            controls += 1;
            ensure!(controls < 8, "deadline during cleanup");
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("deadline during cleanup"));
    assert_eq!(controls, 8);
    assert_eq!((owner.resets, owner.position), (2, 0));
    assert!(health.load(Ordering::Acquire));
}

#[test]
fn unknown_operator_error_and_cleanup_failure_permanently_lower_shared_latch() {
    let mut failed = FakeOwner {
        operator_failure: true,
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    assert!(run(&mut failed, &[1], 1, &health, &mut || Ok(()), |_| Ok(())).is_err());
    assert!(!health.load(Ordering::Acquire));
    let before = failed.evaluated.len();
    assert!(run(
        &mut failed,
        &[1],
        1,
        &health,
        &mut || panic!("latched callback"),
        |_| panic!("latched sink")
    )
    .is_err());
    assert_eq!(failed.evaluated.len(), before);

    let mut cleanup = FakeOwner {
        outputs: VecDeque::from([7]),
        failed_reset: Some(2),
        ..Default::default()
    };
    let health = AtomicBool::new(true);
    assert!(run(&mut cleanup, &[1], 1, &health, &mut || Ok(()), |_| Ok(())).is_err());
    assert!(!health.load(Ordering::Acquire));
    let health = AtomicBool::new(true);
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        ..Default::default()
    };
    assert!(run_session(
        &mut owner,
        1,
        &[1],
        1,
        &EOG,
        &health,
        Duration::ZERO,
        &mut || Ok(()),
        |_| Ok(()),
        &Recorder::new(false)
    )
    .is_err());
    assert!(!health.load(Ordering::Acquire));
}

#[test]
fn unwinding_does_not_restore_health_or_reuse_unknown_state() {
    let health = AtomicBool::new(true);
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        ..Default::default()
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run(&mut owner, &[1], 1, &health, &mut || Ok(()), |_| {
            panic!("sink panic")
        });
    }));
    assert!(outcome.is_err());
    assert!(!health.load(Ordering::Acquire));
}

#[test]
fn cleanup_reset_panic_keeps_unwind_guard_armed_and_lowers_shared_health() {
    let health = AtomicBool::new(true);
    let mut owner = FakeOwner {
        outputs: VecDeque::from([7]),
        panic_reset: Some(2),
        ..Default::default()
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run(&mut owner, &[1], 1, &health, &mut || Ok(()), |_| Ok(()));
    }));
    assert!(outcome.is_err());
    assert_eq!(owner.resets, 2);
    assert!(!health.load(Ordering::Acquire));
}

#[test]
fn digest_covers_every_byte_in_bounded_chunks_and_cancellation_stops_reads() {
    let bytes: Vec<u8> = (0..HASH_CHUNK * 2 + 17).map(|i| (i % 251) as u8).collect();
    let mut ranges = Vec::new();
    let digest = hash_reader(
        bytes.len() as u64,
        |offset, target| {
            ranges.push((offset, target.len()));
            target.copy_from_slice(&bytes[offset as usize..offset as usize + target.len()]);
            Ok(())
        },
        &mut || Ok(()),
    )
    .unwrap();
    let expected: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!(digest, expected);
    assert_eq!(
        ranges,
        [
            (0, HASH_CHUNK),
            (HASH_CHUNK as u64, HASH_CHUNK),
            ((HASH_CHUNK * 2) as u64, 17)
        ]
    );
    // Independent standard SHA256 vector, not just two uses of our wrapper.
    let abc = hash_reader(
        3,
        |offset, target| {
            assert_eq!(offset, 0);
            target.copy_from_slice(b"abc");
            Ok(())
        },
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(
        abc,
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad
        ]
    );
    let mut reads = 0;
    let mut checks = 0;
    assert!(hash_reader(
        bytes.len() as u64,
        |_, target| {
            reads += 1;
            target.fill(0);
            Ok(())
        },
        &mut || {
            checks += 1;
            ensure!(checks < 2, "cancel hash");
            Ok(())
        }
    )
    .is_err());
    assert_eq!(reads, 1);
    assert!(hash_reader(0, |_, _| panic!("empty read"), &mut || Ok(())).is_err());
}

fn model_config() -> ModelConfig {
    let mut model = ModelConfig::new("openmythos-hag", "/does/not/exist-openmythos.gguf");
    model.execution_backend = ExecutionBackend::Hag;
    model.hag = Some(HagConfig {
        device: "/dev/dri/renderD128".into(),
        memory_budget_bytes: 2 * 1024 * 1024 * 1024,
        context_capacity: 2048,
    });
    model.hag_qwen35 = Some(HagQwen35Config::default());
    model.memory_budget_bytes = Some(24 * 1024 * 1024 * 1024);
    model
}

#[test]
fn registration_listing_limits_and_idempotent_unload_never_open_missing_checkpoint() {
    let runtime = NativeRuntime::new(RuntimeConfig {
        models: vec![model_config()],
        ..RuntimeConfig::default()
    })
    .unwrap();
    assert_eq!(
        runtime.descriptor().architectures,
        ["qwen3", "qwen3moe", "qwen35"]
    );
    assert_eq!(runtime.descriptor().execution_backends, ["cpu", "hag"]);
    let models = runtime.models().unwrap();
    assert!(!models[0].loaded);
    assert_eq!(models[0].compatibility, ModelCompatibility::Unverified);
    assert_eq!(
        runtime
            .execution_limits("openmythos-hag")
            .unwrap()
            .max_evaluated_tokens,
        Some(2048)
    );
    runtime.unload("openmythos-hag").unwrap();
    runtime.unload("openmythos-hag").unwrap();
    assert!(!runtime.models().unwrap()[0].loaded);
    let error = runtime
        .plan_request_controlled("openmythos-hag", "x", Some(2049), 64, &mut || Ok(()))
        .unwrap_err();
    assert!(error.downcast_ref::<crate::RequestError>().is_some());
    assert!(!runtime.models().unwrap()[0].loaded);
}

#[test]
fn explicit_profile_requires_hag_no_caches_and_complete_positive_limits() {
    let validate = |model| {
        RuntimeConfig {
            models: vec![model],
            ..RuntimeConfig::default()
        }
        .validate()
    };
    validate(model_config()).unwrap();
    let mut model = model_config();
    model.memory_budget_bytes = Some(0);
    assert!(validate(model).is_err());
    let mut model = model_config();
    model.hag.as_mut().unwrap().context_capacity = 2049;
    assert!(validate(model).is_err());
    let mut model = model_config();
    model.execution_backend = ExecutionBackend::Cpu;
    assert!(validate(model).is_err());
    let mut model = model_config();
    model.memory_budget_bytes = None;
    assert!(validate(model).is_err());
    for index in 0..5 {
        let mut model = model_config();
        match index {
            0 => model.hag_qwen35.as_mut().unwrap().tile_rows = 4097,
            1 => model.hag_qwen35.as_mut().unwrap().encoded_read_bytes = 0,
            2 => model.hag_qwen35.as_mut().unwrap().decoded_read_bytes = 64 * 1024 * 1024 + 1,
            3 => model.hag_qwen35.as_mut().unwrap().cleanup_timeout_ms = 0,
            4 => model.demand_cache_bytes = 1,
            _ => unreachable!(),
        }
        assert!(validate(model).is_err());
    }
}

#[test]
fn exact_pinned_digest_is_required_not_a_caller_claim() {
    require_pinned_digest(CHECKPOINT_SHA256).unwrap();
    for index in 0..32 {
        let mut wrong = CHECKPOINT_SHA256;
        wrong[index] ^= 1;
        assert!(require_pinned_digest(wrong).is_err());
    }
}

#[test]
fn generated_protocol_delimiters_reach_wire_but_eog_and_other_controls_do_not() {
    // Explicit detokenizer seam models both CONTROL (hidden normally) and
    // USER_DEFINED/NORMAL (already visible) layouts. It does not claim that an
    // actual checkpoint's metadata was read; production admission checks it.
    for delimiters_are_control in [false, true] {
        let mut chunks = Vec::new();
        let mut callback = |text: &str| {
            chunks.push(text.to_owned());
            Ok(())
        };
        let mut sink = super::super::Utf8Sink::new(&mut callback);
        let mut requests = Vec::new();
        for token in [
            200, 248_068, 248_069, 248_058, 248_059, 248_045, 248_044, 248_046, 248_063, 248_064,
            248_065,
        ] {
            let bytes = output_bytes(token, |id, render_special| {
                requests.push((id, render_special));
                if id == 200 {
                    return Ok(b"reason".to_vec());
                }
                if let Some((_, text)) = PROTOCOL_DELIMITERS.iter().find(|(token, _)| *token == id)
                {
                    return Ok(if render_special || !delimiters_are_control {
                        text.as_bytes().to_vec()
                    } else {
                        Vec::new()
                    });
                }
                assert_eq!(id, 248_045); // Unrelated im_start CONTROL stays hidden.
                Ok(if render_special {
                    b"<|im_start|>".to_vec()
                } else {
                    Vec::new()
                })
            })
            .unwrap();
            sink.push_bytes(&bytes).unwrap();
        }
        let text = sink.finish().unwrap();
        assert_eq!(text, "reason<think></think><tool_call></tool_call>");
        assert_eq!(chunks.concat(), text);
        assert_eq!(
            requests,
            [
                (200, false),
                (248_068, true),
                (248_069, true),
                (248_058, true),
                (248_059, true),
                (248_045, false)
            ]
        );
    }
}

#[test]
fn delimiter_policy_rejects_changed_mapping_and_preserves_ordinary_decoder_errors() {
    assert_eq!(
        PROTOCOL_DELIMITERS,
        [
            (248_058, "<tool_call>"),
            (248_059, "</tool_call>"),
            (248_068, "<think>"),
            (248_069, "</think>")
        ]
    );
    for (id, _) in PROTOCOL_DELIMITERS {
        assert!(output_bytes(id, |_, special| {
            assert!(special);
            Ok(b"wrong".to_vec())
        })
        .is_err());
    }
    for id in EOG {
        assert!(output_bytes(id, |_, _| panic!("EOG must not be decoded"))
            .unwrap()
            .is_empty());
    }
    let error = output_bytes(7, |id, special| {
        assert_eq!((id, special), (7, false));
        bail!("ordinary decoder failed")
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "ordinary decoder failed");
}
