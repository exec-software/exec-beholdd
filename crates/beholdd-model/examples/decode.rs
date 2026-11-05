//! Correctness-first full-model greedy decode from text.
//!
//! The Qwen2 tokenizer is loaded from the same GGUF as the model. This example accepts a
//! plain completion prompt; it does not interpret the checkpoint's Jinja chat template.
//!
//! usage: cargo run -p beholdd-model --release --example decode -- <model.gguf> <new_tokens> <prompt text> [--weight-mode <streaming|resident>] [--logits-out <path>] [--fit-policy <exact|degrade|refuse>] [--memory-budget-bytes <bytes>] [--runtime-headroom-bytes <bytes>] [--pinned-cache-bytes <bytes>] [--demand-cache-bytes <bytes>] [--prefetch-bytes <bytes>] [--prefetch-queue-depth <requests>] [--prefetch-workers <count>] [--step-timing-out <path>] [--checkpoint-sha256 <hex>] [--near-miss-prefetch] [--allow-unsafe-storage] [--plan-only]

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use beholdd_backend::{cpu::Cpu, Device};
use beholdd_format::Gguf;
use beholdd_memory::{
    available_memory, cgroup_memory, check_checkpoint_storage, process_memory, AvailableMemory,
    CacheConfig, CacheStats, FitEstimate, FitPolicy, FullyResidentSource, StorageClass,
    StorageOverride, StreamingSource,
};
use beholdd_model::artifact::{refuse_existing_output, write_logits_artifact};
use beholdd_model::execution::{
    parse_byte_count, parse_checkpoint_sha256, parse_prefetch_workers, parse_queue_depth,
    plan_execution, set_flag_once, set_once, ExecutionPlan, WeightMode,
};
use beholdd_model::{Qwen2Tokenizer, QwenModel};

const DEFAULT_RUNTIME_HEADROOM_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const USAGE: &str = "usage: decode <model.gguf> <new_tokens> <prompt text> [--weight-mode <streaming|resident>] [--logits-out <path>] [--fit-policy <exact|degrade|refuse>] [--memory-budget-bytes <bytes>] [--runtime-headroom-bytes <bytes>] [--pinned-cache-bytes <bytes>] [--demand-cache-bytes <bytes>] [--prefetch-bytes <bytes>] [--prefetch-queue-depth <requests>] [--prefetch-workers <count>] [--step-timing-out <path>] [--checkpoint-sha256 <hex>] [--near-miss-prefetch] [--allow-unsafe-storage] [--plan-only]";

#[derive(Debug)]
enum ExecutionReport {
    Streaming {
        stats: Box<CacheStats>,
        cache: CacheConfig,
        drain: Duration,
        io_wall: Duration,
    },
    Resident {
        bytes: u64,
        load_seconds: f64,
    },
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().context(USAGE)?;
    let max_new_tokens: usize = args
        .next()
        .context(USAGE)?
        .parse()
        .context("new_tokens must be a non-negative integer")?;
    let prompt_text = args.next().context(USAGE)?;
    let mut logits_path = None;
    let mut step_timing_path = None;
    let mut weight_mode = None;
    let mut pinned_cache_bytes = None;
    let mut demand_cache_bytes = None;
    let mut prefetch_bytes = None;
    let mut prefetch_queue_depth = None;
    let mut prefetch_workers = None;
    let mut fit_policy = None;
    let mut memory_budget_bytes = None;
    let mut runtime_headroom_bytes = None;
    let mut checkpoint_sha256 = None;
    let mut near_miss_prefetch = false;
    let mut allow_unsafe_storage = false;
    let mut plan_only = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--weight-mode" => set_once(
                &mut weight_mode,
                option_value(&mut args, "--weight-mode")?
                    .parse::<WeightMode>()
                    .context("parsing --weight-mode")?,
                "--weight-mode",
            )?,
            "--logits-out" => set_once(
                &mut logits_path,
                option_value(&mut args, "--logits-out")?,
                "--logits-out",
            )?,
            "--step-timing-out" => set_once(
                &mut step_timing_path,
                option_value(&mut args, "--step-timing-out")?,
                "--step-timing-out",
            )?,
            "--pinned-cache-bytes" => set_once(
                &mut pinned_cache_bytes,
                parse_byte_count(
                    &option_value(&mut args, "--pinned-cache-bytes")?,
                    "--pinned-cache-bytes",
                )?,
                "--pinned-cache-bytes",
            )?,
            "--demand-cache-bytes" => set_once(
                &mut demand_cache_bytes,
                parse_byte_count(
                    &option_value(&mut args, "--demand-cache-bytes")?,
                    "--demand-cache-bytes",
                )?,
                "--demand-cache-bytes",
            )?,
            "--prefetch-bytes" => set_once(
                &mut prefetch_bytes,
                parse_byte_count(
                    &option_value(&mut args, "--prefetch-bytes")?,
                    "--prefetch-bytes",
                )?,
                "--prefetch-bytes",
            )?,
            "--prefetch-queue-depth" => set_once(
                &mut prefetch_queue_depth,
                parse_queue_depth(&option_value(&mut args, "--prefetch-queue-depth")?)?,
                "--prefetch-queue-depth",
            )?,
            "--prefetch-workers" => set_once(
                &mut prefetch_workers,
                parse_prefetch_workers(&option_value(&mut args, "--prefetch-workers")?)?,
                "--prefetch-workers",
            )?,
            "--fit-policy" => set_once(
                &mut fit_policy,
                option_value(&mut args, "--fit-policy")?
                    .parse::<FitPolicy>()
                    .context("parsing --fit-policy")?,
                "--fit-policy",
            )?,
            "--memory-budget-bytes" => set_once(
                &mut memory_budget_bytes,
                parse_byte_count(
                    &option_value(&mut args, "--memory-budget-bytes")?,
                    "--memory-budget-bytes",
                )?,
                "--memory-budget-bytes",
            )?,
            "--runtime-headroom-bytes" => set_once(
                &mut runtime_headroom_bytes,
                parse_byte_count(
                    &option_value(&mut args, "--runtime-headroom-bytes")?,
                    "--runtime-headroom-bytes",
                )?,
                "--runtime-headroom-bytes",
            )?,
            "--checkpoint-sha256" => set_once(
                &mut checkpoint_sha256,
                parse_checkpoint_sha256(&option_value(&mut args, "--checkpoint-sha256")?)?,
                "--checkpoint-sha256",
            )?,
            "--near-miss-prefetch" => {
                set_flag_once(&mut near_miss_prefetch, "--near-miss-prefetch")?
            }
            "--allow-unsafe-storage" => {
                set_flag_once(&mut allow_unsafe_storage, "--allow-unsafe-storage")?
            }
            "--plan-only" => set_flag_once(&mut plan_only, "--plan-only")?,
            _ => {
                bail!("unexpected argument {flag:?}; prompt text must be one quoted argument; {USAGE}")
            }
        }
    }
    if let Some(path) = &logits_path {
        refuse_existing_output(Path::new(path))?;
    }
    if let Some(path) = &step_timing_path {
        refuse_existing_output(Path::new(path))?;
    }
    let requested_cache = CacheConfig::new(
        pinned_cache_bytes.unwrap_or(0),
        demand_cache_bytes.unwrap_or(0),
    )?
    .with_prefetch(
        prefetch_bytes.unwrap_or(0),
        prefetch_queue_depth.unwrap_or(0),
        prefetch_workers.unwrap_or(0),
    )?;
    let weight_mode = weight_mode.unwrap_or_default();

    let storage_override = if allow_unsafe_storage {
        StorageOverride::AllowUnsafe
    } else {
        StorageOverride::RefuseUnsafe
    };
    check_checkpoint_storage([Path::new(&model_path)], storage_override).with_context(|| {
        "preflighting the checkpoint path; pass --allow-unsafe-storage to explicitly permit network, rotational, mixed, or unknown storage"
    })?;
    let checkpoint = Arc::new(Gguf::open(&model_path)?);
    let storage = check_checkpoint_storage(checkpoint.shards(), storage_override).with_context(|| {
        "checking every checkpoint shard; pass --allow-unsafe-storage to explicitly permit network, rotational, mixed, or unknown storage"
    })?;
    let tokenizer = Qwen2Tokenizer::from_gguf(&checkpoint)?;
    let prompt = tokenizer.encode(&prompt_text, false)?;
    if prompt.is_empty() {
        anyhow::bail!("prompt tokenized to an empty sequence");
    }
    let model = QwenModel::load(&checkpoint)?;
    let device = Cpu::new();
    model.validate_device(&device)?;
    let weight_memory = model.weight_memory()?;
    let decode_buffer_bytes = model.greedy_decode_buffer_bytes(prompt.len(), max_new_tokens)?;
    let cpu_scratch_bytes = weight_memory
        .largest_matvec_elements()
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .context("CPU dequantized-weight scratch bytes overflow")?;
    let streaming_weight_bytes = weight_memory.largest_request_bytes();
    let runtime_headroom_bytes = runtime_headroom_bytes.unwrap_or(DEFAULT_RUNTIME_HEADROOM_BYTES);
    let always_resident_bytes = runtime_headroom_bytes
        .checked_add(decode_buffer_bytes)
        .and_then(|bytes| bytes.checked_add(cpu_scratch_bytes))
        .context("runtime memory forecast overflows")?;
    let estimate = FitEstimate::from_parts(
        always_resident_bytes,
        weight_memory.checkpoint_bytes(),
        streaming_weight_bytes,
    )?;
    let availability = available_memory()?;
    let available_bytes = availability.effective_available_bytes();
    let memory_budget_bytes = memory_budget_bytes.unwrap_or(available_bytes / 2);
    if memory_budget_bytes > available_bytes {
        bail!(
            "memory budget is {memory_budget_bytes} bytes, but Linux currently reports only {available_bytes} bytes available; lower --memory-budget-bytes"
        );
    }
    let fit_policy = fit_policy.unwrap_or_default();
    let execution_plan = plan_execution(
        weight_mode,
        estimate,
        fit_policy,
        memory_budget_bytes,
        requested_cache,
    )?;
    print_preflight(
        &model_path,
        storage.class(),
        fit_policy,
        execution_plan,
        availability,
        runtime_headroom_bytes,
        decode_buffer_bytes,
        streaming_weight_bytes,
        cpu_scratch_bytes,
        weight_memory.checkpoint_bytes(),
        requested_cache,
        checkpoint_sha256.as_deref(),
    );
    std::io::stdout().flush()?;
    if plan_only {
        println!("  plan only: no tensor payloads read");
        return Ok(());
    }

    let mut step_durations: Option<Vec<Duration>> = None;
    let (output, elapsed, report, execution_process_memory, execution_cgroup_memory) =
        match execution_plan {
            ExecutionPlan::Streaming { cache, .. } => {
                let weights = StreamingSource::new(checkpoint.clone(), cache)?;
                let started = Instant::now();
                let output = if step_timing_path.is_some() {
                    let (output, durations) = model.greedy_decode_with_step_timings(
                        &device,
                        &weights,
                        &prompt,
                        max_new_tokens,
                        near_miss_prefetch,
                    )?;
                    step_durations = Some(durations);
                    output
                } else {
                    model.greedy_decode(
                        &device,
                        &weights,
                        &prompt,
                        max_new_tokens,
                        near_miss_prefetch,
                    )?
                };
                let elapsed = started.elapsed();
                let execution_process_memory = process_memory()?;
                let execution_cgroup_memory = cgroup_memory()?;
                let drain_started = Instant::now();
                let stats = weights.finish()?;
                let drain = drain_started.elapsed();
                let io_wall = started.elapsed();
                (
                    output,
                    elapsed,
                    ExecutionReport::Streaming {
                        stats: Box::new(stats),
                        cache,
                        drain,
                        io_wall,
                    },
                    execution_process_memory,
                    execution_cgroup_memory,
                )
            }
            ExecutionPlan::Resident(_) => {
                let load_started = Instant::now();
                let weights = FullyResidentSource::load(checkpoint.as_ref())?;
                let load_seconds = load_started.elapsed().as_secs_f64();
                if weights.resident_bytes() != weight_memory.checkpoint_bytes() {
                    bail!(
                    "resident source materialized {} bytes, but the bound model forecast {} checkpoint bytes",
                    weights.resident_bytes(),
                    weight_memory.checkpoint_bytes()
                );
                }
                let started = Instant::now();
                let output = if step_timing_path.is_some() {
                    let (output, durations) = model.greedy_decode_with_step_timings(
                        &device,
                        &weights,
                        &prompt,
                        max_new_tokens,
                        near_miss_prefetch,
                    )?;
                    step_durations = Some(durations);
                    output
                } else {
                    model.greedy_decode(
                        &device,
                        &weights,
                        &prompt,
                        max_new_tokens,
                        near_miss_prefetch,
                    )?
                };
                let elapsed = started.elapsed();
                let execution_process_memory = process_memory()?;
                let execution_cgroup_memory = cgroup_memory()?;
                (
                    output,
                    elapsed,
                    ExecutionReport::Resident {
                        bytes: weights.resident_bytes(),
                        load_seconds,
                    },
                    execution_process_memory,
                    execution_cgroup_memory,
                )
            }
        };
    if let Some(path) = &logits_path {
        write_logits_artifact(Path::new(path), &output)?;
    }
    if let Some(path) = &step_timing_path {
        let durations = step_durations
            .as_ref()
            .context("step timing was requested but no timed decode call ran")?;
        let mut out = String::from("step\tduration_ms\n");
        for (step, duration) in durations.iter().enumerate() {
            out.push_str(&format!("{step}\t{:.3}\n", duration.as_secs_f64() * 1e3));
        }
        std::fs::write(path, out).with_context(|| format!("writing step timings to {path}"))?;
    }
    let final_process_memory = process_memory()?;
    let final_cgroup_memory = cgroup_memory()?;

    println!("{model_path}");
    println!("  prompt text: {prompt_text:?}");
    println!("  prompt:    {:?}", &output.tokens[..output.prompt_len]);
    println!("  generated: {:?}", output.generated);
    for (step, &token) in output.generated.iter().enumerate() {
        let logits = output
            .logits_for_step(step)
            .context("decode result is missing a logits row")?;
        println!(
            "    step {step:>3}: token {token:>6}, logit {:.7}",
            logits[token as usize]
        );
    }
    let decoded = tokenizer.decode_bytes(&output.generated, false)?;
    println!("  text:       {:?}", String::from_utf8_lossy(&decoded));
    println!("  device:     {}", device.name());
    if output.generated.is_empty() {
        println!(
            "  decode:     {:.2}s (no generated tokens)",
            elapsed.as_secs_f64()
        );
    } else {
        println!(
            "  decode:     {:.2}s ({:.3}s / generated token)",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / output.generated.len() as f64,
        );
    }
    let generated_tokens = output.generated.len() as f64;
    match report {
        ExecutionReport::Streaming {
            stats,
            cache,
            drain,
            io_wall,
        } => {
            let byte_hit_rate = ratio(stats.hit_bytes(), stats.foreground_demanded_bytes());
            let read_amplification = ratio(stats.read_bytes(), stats.foreground_demanded_bytes());
            let exposed_weight_wait = stats
                .read_duration()
                .saturating_sub(stats.prefetch_physical_read_duration())
                .saturating_add(stats.foreground_wait_duration());
            let wall_read_rate = if io_wall.is_zero() {
                0.0
            } else {
                stats.read_bytes() as f64 / io_wall.as_secs_f64()
            };
            let per_generated_token = |bytes: u64| {
                if generated_tokens == 0.0 {
                    0.0
                } else {
                    bytes as f64 / generated_tokens
                }
            };
            println!(
                "  weight I/O: {} bytes ({:.1} / generated token) in {} physical reads ({} failures) over {:.3}s ({:.3} GB/s read service, {:.3} GB/s over {:.3}s continuous decode-to-settlement wall)",
                stats.read_bytes(),
                per_generated_token(stats.read_bytes()),
                stats.physical_reads(),
                stats.physical_read_failures(),
                stats.read_duration().as_secs_f64(),
                stats.read_bytes_per_second() / 1e9,
                wall_read_rate / 1e9,
                io_wall.as_secs_f64(),
            );
            println!(
                "  demand:     {} bytes ({:.1} / generated token); {} hit bytes ({:.2}%); {:.3}x physical read amplification",
                stats.foreground_demanded_bytes(),
                per_generated_token(stats.foreground_demanded_bytes()),
                stats.hit_bytes(),
                byte_hit_rate * 100.0,
                read_amplification,
            );
            println!(
                "  requests:   {} / {} hit ({:.2}%); {} misses; {:.3}s foreground weight stall",
                stats.hits(),
                stats.requests(),
                stats.hit_rate() * 100.0,
                stats.misses(),
                exposed_weight_wait.as_secs_f64(),
            );
            println!(
                "  prefetch:   {} submitted, {} admitted, {} skipped; {} useful / {} late / {} wasted / {} cancelled",
                stats.prefetch_submitted(),
                stats.prefetch_admitted(),
                stats.prefetch_skipped(),
                stats.prefetch_useful(),
                stats.prefetch_late(),
                stats.prefetch_wasted(),
                stats.prefetch_cancelled(),
            );
            println!(
                "  prefetch I/O: {} useful bytes ({:.1} / generated token), {} wasted bytes; {} bytes in {} reads ({} failures) over {:.3}s, {:.3}s hidden",
                stats.prefetch_useful_bytes(),
                per_generated_token(stats.prefetch_useful_bytes()),
                stats.prefetch_wasted_bytes(),
                stats.prefetch_physical_read_bytes(),
                stats.prefetch_physical_reads(),
                stats.prefetch_physical_read_failures(),
                stats.prefetch_physical_read_duration().as_secs_f64(),
                stats.prefetch_hidden_duration().as_secs_f64(),
            );
            println!(
                "  prefetch peak: {} attached + {} detached, {} queued, {} reading, {} ready; {} / {} reserved bytes",
                stats.prefetch_outstanding_peak(),
                stats.prefetch_detached_peak(),
                stats.prefetch_queued_peak(),
                stats.prefetch_reading_peak(),
                stats.prefetch_ready_peak(),
                stats.prefetch_reserved_bytes_peak(),
                cache.prefetch_bytes(),
            );
            println!(
                "  prefetch drain: {:.3}s after the decode timer",
                drain.as_secs_f64()
            );
            println!(
                "  cache:      {} / {} retained bytes ({} pinned, {} demand, {} evictions); prefetch budget {} bytes / {} attached requests / {} workers",
                stats.cached_bytes(),
                cache.pinned_bytes() + cache.demand_bytes(),
                stats.pinned_bytes(),
                stats.demand_bytes(),
                stats.evictions(),
                cache.prefetch_bytes(),
                cache.prefetch_queue_depth(),
                cache.prefetch_workers(),
            );
        }
        ExecutionReport::Resident {
            bytes,
            load_seconds,
        } => println!(
            "  weight I/O: {bytes} bytes preloaded once over {load_seconds:.3}s ({:.3} GB/s); all decode requests served from resident memory",
            if load_seconds == 0.0 {
                0.0
            } else {
                bytes as f64 / load_seconds / 1e9
            }
        ),
    }
    println!(
        "  process RSS: {} bytes at execution end, {} bytes after finalization, {} bytes lifetime peak at final sample",
        execution_process_memory.resident_bytes(),
        final_process_memory.resident_bytes(),
        final_process_memory.peak_resident_bytes(),
    );
    match (execution_cgroup_memory, final_cgroup_memory) {
        (Some(execution), Some(final_memory)) => println!(
            "  cgroup:     {} bytes at execution end, {} bytes after finalization, {} bytes cgroup lifetime peak at final sample",
            execution.current_bytes(),
            final_memory.current_bytes(),
            final_memory.peak_bytes(),
        ),
        (None, None) => {
            println!("  cgroup:     unavailable (no unified cgroup-v2 membership)")
        }
        _ => println!(
            "  cgroup:     membership changed while sampling; current/peak measurements unavailable"
        ),
    }
    if let Some(path) = logits_path {
        println!("  wrote exact logits: {path}");
    }
    if let Some(path) = step_timing_path {
        println!("  wrote step timings: {path}");
    }
    Ok(())
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn option_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value; {USAGE}"))
}

#[allow(clippy::too_many_arguments)]
fn print_preflight(
    model_path: &str,
    storage: StorageClass,
    policy: FitPolicy,
    execution_plan: ExecutionPlan,
    availability: AvailableMemory,
    runtime_headroom_bytes: u64,
    decode_buffer_bytes: u64,
    streaming_weight_bytes: u64,
    cpu_scratch_bytes: u64,
    checkpoint_bytes: u64,
    requested_cache: CacheConfig,
    checkpoint_sha256: Option<&str>,
) {
    println!("{model_path}");
    println!("  preflight device: cpu");
    println!("  storage:         {storage}");
    println!("  fit policy:      {policy}");
    println!(
        "  weight mode:     {}",
        match execution_plan {
            ExecutionPlan::Streaming { .. } => WeightMode::Streaming,
            ExecutionPlan::Resident(_) => WeightMode::Resident,
        }
    );
    println!(
        "  host available:  {} bytes",
        availability.host_available_bytes()
    );
    if let Some(bytes) = availability.cgroup_available_bytes() {
        println!("  cgroup headroom: {bytes} bytes");
    }
    println!(
        "  usable memory:   {} bytes",
        availability.effective_available_bytes()
    );
    println!("  checkpoint:      {checkpoint_bytes} encoded bytes");
    println!(
        "  checkpoint sha256: {}",
        checkpoint_sha256.unwrap_or("not supplied")
    );
    println!("  runtime reserve: {runtime_headroom_bytes} headroom + {decode_buffer_bytes} decode buffers");
    match execution_plan {
        ExecutionPlan::Streaming { plan, cache } => {
            println!(
                "  max weight live: {streaming_weight_bytes} encoded bytes + {cpu_scratch_bytes} CPU scratch bytes"
            );
            println!(
                "  fit decision:    exact streaming at {} / {} forecast bytes",
                plan.forecast_bytes(),
                plan.budget_bytes()
            );
            println!(
                "  cache plan:      {} requested, {} authorized ({} pinned, {} demand, {} prefetch / {} queued / {} workers)",
                requested_cache.total_bytes(),
                cache.total_bytes(),
                cache.pinned_bytes(),
                cache.demand_bytes(),
                cache.prefetch_bytes(),
                cache.prefetch_queue_depth(),
                cache.prefetch_workers()
            );
        }
        ExecutionPlan::Resident(plan) => {
            println!(
                "  max weight live: {checkpoint_bytes} encoded bytes + {cpu_scratch_bytes} CPU scratch bytes"
            );
            println!(
                "  fit decision:    fully resident at {} / {} forecast bytes",
                plan.forecast_bytes(),
                plan.budget_bytes()
            );
            println!("  cache plan:      not applicable to resident weights");
        }
    }
}
