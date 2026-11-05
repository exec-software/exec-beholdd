//! Build an exact-byte, checkpoint-native Qwen3.5/3.6 MoE FFN layer image.
//!
//! This is a research staging tool for the standalone gfx1151 HIP resident-layer
//! harness. It deliberately is not part of model loading or the production runtime.
//!
//! usage: pack_qwen35moe_bf16_layer <snapshot-dir|first-shard.gguf> <layer> <output.pack>

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Seek, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use beholdd_core::DType;
use beholdd_format::{GgmlType, Gguf, GgufTensor, SafeTensors, TensorInfo};
use serde_json::Value;

const MAGIC: &[u8; 16] = b"BEHOLDDQ35LAYER\0";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 4096;
const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const LAYOUT_CHECKPOINT_COLUMN_MAJOR: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Geometry {
    layer: u32,
    hidden: u32,
    ffn: u32,
    experts: u32,
    top_k: u32,
    shared_ffn: u32,
    eps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Section {
    offset: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    norm: Section,
    router: Section,
    shared_gate: Section,
    gate_up: Section,
    down: Section,
    file_bytes: u64,
}

fn checked_product(values: &[u64], label: &str) -> Result<u64> {
    values.iter().try_fold(1u64, |product, &value| {
        product
            .checked_mul(value)
            .with_context(|| format!("{label} byte geometry overflows"))
    })
}

fn make_layout(geometry: Geometry) -> Result<Layout> {
    let hidden = u64::from(geometry.hidden);
    let ffn = u64::from(geometry.ffn);
    let slots = u64::from(geometry.experts)
        .checked_add(1)
        .context("expert slot count overflows")?;

    let norm_bytes = checked_product(&[hidden, 4], "norm")?;
    let router_bytes = checked_product(&[u64::from(geometry.experts), hidden, 4], "router")?;
    let shared_gate_bytes = checked_product(&[hidden, 4], "shared gate")?;
    let gate_up_bytes = checked_product(&[2, slots, hidden, ffn, 2], "gate/up")?;
    let down_bytes = checked_product(&[slots, ffn, hidden, 2], "down")?;

    let mut cursor = HEADER_BYTES;
    let mut take = |bytes: u64| -> Result<Section> {
        let section = Section {
            offset: cursor,
            bytes,
        };
        cursor = cursor
            .checked_add(bytes)
            .context("layer-pack file length overflows")?;
        Ok(section)
    };

    Ok(Layout {
        norm: take(norm_bytes)?,
        router: take(router_bytes)?,
        shared_gate: take(shared_gate_bytes)?,
        gate_up: take(gate_up_bytes)?,
        down: take(down_bytes)?,
        file_bytes: cursor,
    })
}

fn put_u32(header: &mut [u8], offset: usize, value: u32) {
    header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(header: &mut [u8], offset: usize, value: u64) {
    header[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn encode_header(geometry: Geometry, layout: Layout) -> [u8; HEADER_BYTES as usize] {
    let mut header = [0u8; HEADER_BYTES as usize];
    header[..MAGIC.len()].copy_from_slice(MAGIC);
    put_u32(&mut header, 16, VERSION);
    put_u32(&mut header, 20, HEADER_BYTES as u32);
    put_u32(&mut header, 24, geometry.layer);
    put_u32(&mut header, 28, geometry.hidden);
    put_u32(&mut header, 32, geometry.ffn);
    put_u32(&mut header, 36, geometry.experts);
    put_u32(&mut header, 40, geometry.top_k);
    put_u32(&mut header, 44, geometry.shared_ffn);
    put_u32(&mut header, 48, geometry.eps.to_bits());
    put_u32(&mut header, 52, LAYOUT_CHECKPOINT_COLUMN_MAJOR);

    for (offset, section) in [
        (64, layout.norm),
        (80, layout.router),
        (96, layout.shared_gate),
        (112, layout.gate_up),
        (128, layout.down),
    ] {
        put_u64(&mut header, offset, section.offset);
        put_u64(&mut header, offset + 8, section.bytes);
    }
    put_u64(&mut header, 144, layout.file_bytes);
    header
}

fn metadata_u32(gguf: &Gguf, key: &str) -> Result<u32> {
    let value = gguf
        .kv(key)
        .and_then(|value| value.as_u64())
        .with_context(|| format!("missing unsigned qwen35moe.{key}"))?;
    u32::try_from(value).with_context(|| format!("qwen35moe.{key}={value} does not fit u32"))
}

fn exact_geometry(layer: u32) -> Geometry {
    Geometry {
        layer,
        hidden: 2048,
        ffn: 512,
        experts: 256,
        top_k: 8,
        shared_ffn: 512,
        eps: 1.0e-6,
    }
}

fn validate_exact_geometry(geometry: Geometry) -> Result<Geometry> {
    let expected = exact_geometry(geometry.layer);
    if geometry != expected {
        bail!(
            "checkpoint layer geometry {geometry:?} differs from the exact qualified Qwen3.6 shape {expected:?}"
        );
    }
    Ok(geometry)
}

fn validate_exact_layer_count(layers: u32, source: &str) -> Result<()> {
    if layers != 40 {
        bail!("{source} must declare 40 layers for exact Qwen3.6, found {layers}");
    }
    Ok(())
}

fn geometry_from_gguf(gguf: &Gguf, layer: u32) -> Result<Geometry> {
    if gguf.arch() != Some("qwen35moe") {
        bail!(
            "resident-layer packer requires qwen35moe, checkpoint says {}",
            gguf.arch().unwrap_or("<missing>")
        );
    }
    let layers = metadata_u32(gguf, "block_count")?;
    validate_exact_layer_count(layers, "qwen35moe.block_count")?;
    if layer >= layers {
        bail!("layer {layer} is outside checkpoint block_count {layers}");
    }
    let geometry = Geometry {
        layer,
        hidden: metadata_u32(gguf, "embedding_length")?,
        ffn: metadata_u32(gguf, "expert_feed_forward_length")?,
        experts: metadata_u32(gguf, "expert_count")?,
        top_k: metadata_u32(gguf, "expert_used_count")?,
        shared_ffn: metadata_u32(gguf, "expert_shared_feed_forward_length")?,
        eps: gguf
            .kv("attention.layer_norm_rms_epsilon")
            .and_then(|value| value.as_f64())
            .context("missing qwen35moe.attention.layer_norm_rms_epsilon")? as f32,
    };

    validate_exact_geometry(geometry)
}

fn json_string<'a>(object: &'a Value, key: &str, label: &str) -> Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("{label}.{key} must be a string"))
}

fn json_u32(object: &Value, key: &str, label: &str) -> Result<u32> {
    let value = object
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("{label}.{key} must be an unsigned integer"))?;
    u32::try_from(value).with_context(|| format!("{label}.{key}={value} does not fit u32"))
}

fn json_f32(object: &Value, key: &str, label: &str) -> Result<f32> {
    Ok(object
        .get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("{label}.{key} must be a number"))? as f32)
}

fn require_json_string(object: &Value, key: &str, label: &str, expected: &str) -> Result<()> {
    let actual = json_string(object, key, label)?;
    if actual != expected {
        bail!("{label}.{key} must be {expected:?}, found {actual:?}");
    }
    Ok(())
}

fn geometry_from_config(config: &Value, layer: u32) -> Result<Geometry> {
    require_json_string(config, "model_type", "config", "qwen3_5_moe")?;
    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .context("config.architectures must be an array")?;
    if architectures.len() != 1
        || architectures[0].as_str() != Some("Qwen3_5MoeForConditionalGeneration")
    {
        bail!("config.architectures must be exactly [\"Qwen3_5MoeForConditionalGeneration\"]");
    }

    let text = config
        .get("text_config")
        .context("config is missing text_config")?;
    require_json_string(text, "model_type", "config.text_config", "qwen3_5_moe_text")?;
    require_json_string(text, "dtype", "config.text_config", "bfloat16")?;
    require_json_string(text, "hidden_act", "config.text_config", "silu")?;

    let layers = json_u32(text, "num_hidden_layers", "config.text_config")?;
    validate_exact_layer_count(layers, "config.text_config.num_hidden_layers")?;
    if layer >= layers {
        bail!("layer {layer} is outside config.text_config.num_hidden_layers {layers}");
    }
    let geometry = Geometry {
        layer,
        hidden: json_u32(text, "hidden_size", "config.text_config")?,
        ffn: json_u32(text, "moe_intermediate_size", "config.text_config")?,
        experts: json_u32(text, "num_experts", "config.text_config")?,
        top_k: json_u32(text, "num_experts_per_tok", "config.text_config")?,
        shared_ffn: json_u32(
            text,
            "shared_expert_intermediate_size",
            "config.text_config",
        )?,
        eps: json_f32(text, "rms_norm_eps", "config.text_config")?,
    };
    validate_exact_geometry(geometry)
}

fn read_safetensors_geometry(snapshot: &Path, layer: u32) -> Result<(Geometry, PathBuf)> {
    let config_path = snapshot.join("config.json");
    let bytes = fs::read(&config_path)
        .with_context(|| format!("reading Qwen3.6 config {}", config_path.display()))?;
    let config: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing Qwen3.6 config {}", config_path.display()))?;
    let geometry = geometry_from_config(&config, layer)?;
    Ok((geometry, config_path))
}

fn require_gguf_tensor<'a>(
    gguf: &'a Gguf,
    name: &str,
    kind: GgmlType,
    dims: &[u64],
) -> Result<&'a GgufTensor> {
    let tensor = gguf
        .get(name)
        .with_context(|| format!("checkpoint is missing {name}"))?;
    if tensor.kind() != kind {
        bail!(
            "{name} must remain {}, checkpoint stores {}",
            kind.name(),
            tensor.kind().name()
        );
    }
    if tensor.dims() != dims {
        bail!(
            "{name} must have GGUF dimensions {dims:?}, found {:?}",
            tensor.dims()
        );
    }
    let element_bytes = match kind.0 {
        0 => 4,
        30 => 2,
        _ => bail!("packer has no exact-byte size rule for {}", kind.name()),
    };
    let expected_bytes = checked_product(
        &dims
            .iter()
            .copied()
            .chain(std::iter::once(element_bytes))
            .collect::<Vec<_>>(),
        name,
    )?;
    if tensor.nbytes() != expected_bytes {
        bail!(
            "{name} has {} encoded bytes, expected {expected_bytes}",
            tensor.nbytes()
        );
    }
    Ok(tensor)
}

fn require_safetensor<'a>(
    tensors: &'a SafeTensors,
    name: &str,
    dtype: DType,
    shape: &[u64],
) -> Result<&'a TensorInfo> {
    let tensor = tensors
        .get(name)
        .with_context(|| format!("checkpoint is missing {name}"))?;
    if tensor.dtype() != dtype {
        bail!(
            "{name} must remain {dtype:?}, checkpoint stores {:?}",
            tensor.dtype()
        );
    }
    if tensor.shape() != shape {
        bail!(
            "{name} must have Safetensors shape {shape:?}, found {:?}",
            tensor.shape()
        );
    }
    let element_bytes = dtype
        .size()
        .with_context(|| format!("packer has no exact-byte size rule for {dtype:?}"))?;
    let expected_bytes = checked_product(
        &shape
            .iter()
            .copied()
            .chain(std::iter::once(element_bytes as u64))
            .collect::<Vec<_>>(),
        name,
    )?;
    if tensor.nbytes() != expected_bytes {
        bail!(
            "{name} has {} encoded bytes, expected {expected_bytes}",
            tensor.nbytes()
        );
    }
    Ok(tensor)
}

fn copy_gguf_tensor(
    gguf: &Gguf,
    tensor: &GgufTensor,
    writer: &mut BufWriter<File>,
    scratch: &mut [u8],
) -> Result<()> {
    let mut copied = 0u64;
    while copied < tensor.nbytes() {
        let remaining = tensor.nbytes() - copied;
        let chunk = usize::try_from(remaining.min(scratch.len() as u64))
            .context("copy chunk does not fit usize")?;
        gguf.read_tensor_slice_into(tensor, copied..copied + chunk as u64, &mut scratch[..chunk])?;
        writer.write_all(&scratch[..chunk])?;
        copied += chunk as u64;
    }
    Ok(())
}

fn copy_safetensor_slice(
    tensors: &SafeTensors,
    tensor: &TensorInfo,
    range: Range<u64>,
    writer: &mut BufWriter<File>,
    scratch: &mut [u8],
) -> Result<()> {
    let mut copied = range.start;
    while copied < range.end {
        let remaining = range.end - copied;
        let chunk = usize::try_from(remaining.min(scratch.len() as u64))
            .context("copy chunk does not fit usize")?;
        tensors.read_tensor_slice_into(
            tensor,
            copied..copied + chunk as u64,
            &mut scratch[..chunk],
        )?;
        writer.write_all(&scratch[..chunk])?;
        copied += chunk as u64;
    }
    Ok(())
}

fn copy_safetensor(
    tensors: &SafeTensors,
    tensor: &TensorInfo,
    writer: &mut BufWriter<File>,
    scratch: &mut [u8],
) -> Result<()> {
    copy_safetensor_slice(tensors, tensor, 0..tensor.nbytes(), writer, scratch)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bf16Widen {
    Identity,
    AddOne,
}

fn widen_bf16_chunk(input: &[u8], output: &mut Vec<u8>, transform: Bf16Widen) -> Result<()> {
    if !input.len().is_multiple_of(2) {
        bail!("BF16 byte count {} is not a multiple of two", input.len());
    }
    output.clear();
    let output_bytes = input
        .len()
        .checked_mul(2)
        .context("widened BF16 chunk length overflows")?;
    output
        .try_reserve_exact(output_bytes)
        .map_err(|error| anyhow::anyhow!("cannot reserve widened BF16 buffer: {error}"))?;
    for bytes in input.chunks_exact(2) {
        let bits = u16::from_le_bytes(bytes.try_into().expect("two-byte BF16 chunk"));
        let mut value = beholdd_core::bf16_to_f32(bits);
        if transform == Bf16Widen::AddOne {
            value += 1.0;
        }
        output.extend_from_slice(&value.to_le_bytes());
    }
    debug_assert_eq!(output.len(), output_bytes);
    Ok(())
}

fn copy_safetensor_bf16_as_f32(
    tensors: &SafeTensors,
    tensor: &TensorInfo,
    transform: Bf16Widen,
    writer: &mut BufWriter<File>,
    bf16_scratch: &mut [u8],
    f32_scratch: &mut Vec<u8>,
) -> Result<()> {
    let mut copied = 0u64;
    while copied < tensor.nbytes() {
        let remaining = tensor.nbytes() - copied;
        let mut chunk = usize::try_from(remaining.min(bf16_scratch.len() as u64))
            .context("BF16 widening chunk does not fit usize")?;
        chunk -= chunk % 2;
        if chunk == 0 {
            bail!("BF16 widening buffer is too small");
        }
        tensors.read_tensor_slice_into(
            tensor,
            copied..copied + chunk as u64,
            &mut bf16_scratch[..chunk],
        )?;
        widen_bf16_chunk(&bf16_scratch[..chunk], f32_scratch, transform)?;
        writer.write_all(f32_scratch)?;
        copied += chunk as u64;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedHalf {
    Gate,
    Up,
}

fn fused_gate_up_range(geometry: Geometry, expert: u32, half: FusedHalf) -> Result<Range<u64>> {
    if expert >= geometry.experts {
        bail!(
            "expert {expert} is outside checkpoint expert_count {}",
            geometry.experts
        );
    }
    let matrix_bytes = checked_product(
        &[u64::from(geometry.ffn), u64::from(geometry.hidden), 2],
        "fused gate/up expert matrix",
    )?;
    let stride = matrix_bytes
        .checked_mul(2)
        .context("fused gate/up expert stride overflows")?;
    let mut start = u64::from(expert)
        .checked_mul(stride)
        .context("fused gate/up expert offset overflows")?;
    if half == FusedHalf::Up {
        start = start
            .checked_add(matrix_bytes)
            .context("fused up expert offset overflows")?;
    }
    let end = start
        .checked_add(matrix_bytes)
        .context("fused gate/up expert range overflows")?;
    Ok(start..end)
}

fn gguf_tensor_names(layer: u32) -> [String; 9] {
    [
        format!("blk.{layer}.post_attention_norm.weight"),
        format!("blk.{layer}.ffn_gate_inp.weight"),
        format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        format!("blk.{layer}.ffn_gate_exps.weight"),
        format!("blk.{layer}.ffn_gate_shexp.weight"),
        format!("blk.{layer}.ffn_up_exps.weight"),
        format!("blk.{layer}.ffn_up_shexp.weight"),
        format!("blk.{layer}.ffn_down_exps.weight"),
        format!("blk.{layer}.ffn_down_shexp.weight"),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SafetensorNames {
    norm: String,
    router: String,
    gate_up_experts: String,
    down_experts: String,
    gate_shared: String,
    up_shared: String,
    down_shared: String,
    shared_gate: String,
}

fn safetensor_names(layer: u32) -> SafetensorNames {
    let prefix = format!("model.language_model.layers.{layer}");
    SafetensorNames {
        norm: format!("{prefix}.post_attention_layernorm.weight"),
        router: format!("{prefix}.mlp.gate.weight"),
        gate_up_experts: format!("{prefix}.mlp.experts.gate_up_proj"),
        down_experts: format!("{prefix}.mlp.experts.down_proj"),
        gate_shared: format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        up_shared: format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        down_shared: format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        shared_gate: format!("{prefix}.mlp.shared_expert_gate.weight"),
    }
}

fn write_section<F>(writer: &mut BufWriter<File>, section: Section, write: F) -> Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<()>,
{
    let position = writer.stream_position()?;
    if position != section.offset {
        bail!(
            "internal pack layout mismatch: section starts at {position}, expected {}",
            section.offset
        );
    }
    write(writer)?;
    let expected_end = section
        .offset
        .checked_add(section.bytes)
        .context("section end overflows")?;
    let position = writer.stream_position()?;
    if position != expected_end {
        bail!("internal pack layout mismatch: section ends at {position}, expected {expected_end}");
    }
    Ok(())
}

fn publish_atomic<F>(geometry: Geometry, output: &Path, write_payload: F) -> Result<Layout>
where
    F: FnOnce(&mut BufWriter<File>, Layout) -> Result<()>,
{
    if output.exists() {
        bail!("refusing to overwrite existing output {}", output.display());
    }
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .context("output path needs a UTF-8 file name")?;
    let partial = output.with_file_name(format!(".{file_name}.partial.{}", std::process::id()));
    if partial.exists() {
        bail!("stale partial output already exists: {}", partial.display());
    }

    let result = (|| -> Result<Layout> {
        let layout = make_layout(geometry)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .with_context(|| format!("creating {}", partial.display()))?;
        let mut writer = BufWriter::with_capacity(COPY_BUFFER_BYTES, file);
        writer.write_all(&encode_header(geometry, layout))?;
        write_payload(&mut writer, layout)?;

        writer.flush()?;
        writer.get_ref().sync_all()?;
        if writer.stream_position()? != layout.file_bytes {
            bail!("published layer-pack length differs from declared header length");
        }
        drop(writer);
        fs::hard_link(&partial, output).with_context(|| {
            format!(
                "publishing {} as {} without overwriting",
                partial.display(),
                output.display()
            )
        })?;
        // The output name is now an atomic, no-overwrite link to the fully synced inode.
        // Failure to remove the private link does not invalidate the published output.
        let _ = fs::remove_file(&partial);
        Ok(layout)
    })();

    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn publish_gguf_pack(gguf: &Gguf, geometry: Geometry, output: &Path) -> Result<Layout> {
    let names = gguf_tensor_names(geometry.layer);
    let h = u64::from(geometry.hidden);
    let f = u64::from(geometry.ffn);
    let e = u64::from(geometry.experts);
    let shared = u64::from(geometry.shared_ffn);
    let f32_kind = GgmlType(0);
    let bf16_kind = GgmlType(30);
    let norm = require_gguf_tensor(gguf, &names[0], f32_kind, &[h])?;
    let router = require_gguf_tensor(gguf, &names[1], f32_kind, &[h, e])?;
    let shared_gate = require_gguf_tensor(gguf, &names[2], f32_kind, &[h, 1])?;
    let gate_experts = require_gguf_tensor(gguf, &names[3], bf16_kind, &[h, f, e])?;
    let gate_shared = require_gguf_tensor(gguf, &names[4], bf16_kind, &[h, shared])?;
    let up_experts = require_gguf_tensor(gguf, &names[5], bf16_kind, &[h, f, e])?;
    let up_shared = require_gguf_tensor(gguf, &names[6], bf16_kind, &[h, shared])?;
    let down_experts = require_gguf_tensor(gguf, &names[7], bf16_kind, &[f, h, e])?;
    let down_shared = require_gguf_tensor(gguf, &names[8], bf16_kind, &[shared, h])?;

    publish_atomic(geometry, output, |writer, layout| {
        let mut scratch = vec![0u8; COPY_BUFFER_BYTES];
        write_section(writer, layout.norm, |writer| {
            copy_gguf_tensor(gguf, norm, writer, &mut scratch)
        })?;
        write_section(writer, layout.router, |writer| {
            copy_gguf_tensor(gguf, router, writer, &mut scratch)
        })?;
        write_section(writer, layout.shared_gate, |writer| {
            copy_gguf_tensor(gguf, shared_gate, writer, &mut scratch)
        })?;
        write_section(writer, layout.gate_up, |writer| {
            for tensor in [gate_experts, gate_shared, up_experts, up_shared] {
                copy_gguf_tensor(gguf, tensor, writer, &mut scratch)?;
            }
            Ok(())
        })?;
        write_section(writer, layout.down, |writer| {
            for tensor in [down_experts, down_shared] {
                copy_gguf_tensor(gguf, tensor, writer, &mut scratch)?;
            }
            Ok(())
        })
    })
}

fn publish_safetensors_pack(
    tensors: &SafeTensors,
    geometry: Geometry,
    output: &Path,
) -> Result<Layout> {
    let names = safetensor_names(geometry.layer);
    let h = u64::from(geometry.hidden);
    let f = u64::from(geometry.ffn);
    let e = u64::from(geometry.experts);
    let shared = u64::from(geometry.shared_ffn);
    let two_f = f.checked_mul(2).context("fused gate/up width overflows")?;
    let bf16 = DType::Bf16;

    // Eight physical tensors provide the pack's nine logical tensors: the routed
    // gate and up projections share one expert-major `gate_up_proj` tensor.
    let norm = require_safetensor(tensors, &names.norm, bf16, &[h])?;
    let router = require_safetensor(tensors, &names.router, bf16, &[e, h])?;
    let gate_up_experts =
        require_safetensor(tensors, &names.gate_up_experts, bf16, &[e, two_f, h])?;
    let down_experts = require_safetensor(tensors, &names.down_experts, bf16, &[e, h, f])?;
    let gate_shared = require_safetensor(tensors, &names.gate_shared, bf16, &[shared, h])?;
    let up_shared = require_safetensor(tensors, &names.up_shared, bf16, &[shared, h])?;
    let down_shared = require_safetensor(tensors, &names.down_shared, bf16, &[h, shared])?;
    let shared_gate = require_safetensor(tensors, &names.shared_gate, bf16, &[1, h])?;

    publish_atomic(geometry, output, |writer, layout| {
        let mut bf16_scratch = vec![0u8; COPY_BUFFER_BYTES];
        let mut f32_scratch = Vec::new();
        write_section(writer, layout.norm, |writer| {
            copy_safetensor_bf16_as_f32(
                tensors,
                norm,
                Bf16Widen::AddOne,
                writer,
                &mut bf16_scratch,
                &mut f32_scratch,
            )
        })?;
        write_section(writer, layout.router, |writer| {
            copy_safetensor_bf16_as_f32(
                tensors,
                router,
                Bf16Widen::Identity,
                writer,
                &mut bf16_scratch,
                &mut f32_scratch,
            )
        })?;
        write_section(writer, layout.shared_gate, |writer| {
            copy_safetensor_bf16_as_f32(
                tensors,
                shared_gate,
                Bf16Widen::Identity,
                writer,
                &mut bf16_scratch,
                &mut f32_scratch,
            )
        })?;
        write_section(writer, layout.gate_up, |writer| {
            for expert in 0..geometry.experts {
                copy_safetensor_slice(
                    tensors,
                    gate_up_experts,
                    fused_gate_up_range(geometry, expert, FusedHalf::Gate)?,
                    writer,
                    &mut bf16_scratch,
                )?;
            }
            copy_safetensor(tensors, gate_shared, writer, &mut bf16_scratch)?;
            for expert in 0..geometry.experts {
                copy_safetensor_slice(
                    tensors,
                    gate_up_experts,
                    fused_gate_up_range(geometry, expert, FusedHalf::Up)?,
                    writer,
                    &mut bf16_scratch,
                )?;
            }
            copy_safetensor(tensors, up_shared, writer, &mut bf16_scratch)
        })?;
        write_section(writer, layout.down, |writer| {
            copy_safetensor(tensors, down_experts, writer, &mut bf16_scratch)?;
            copy_safetensor(tensors, down_shared, writer, &mut bf16_scratch)
        })
    })
}

fn parse_args() -> Result<(PathBuf, u32, PathBuf)> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args.next().map(PathBuf::from).context(
        "usage: pack_qwen35moe_bf16_layer <snapshot-dir|first-shard.gguf> <layer> <output.pack>",
    )?;
    let layer_text = args
        .next()
        .context("missing layer index")?
        .into_string()
        .map_err(|_| anyhow::anyhow!("layer index must be UTF-8"))?;
    let layer = layer_text
        .parse::<u32>()
        .with_context(|| format!("invalid layer index {layer_text:?}"))?;
    let output = args
        .next()
        .map(PathBuf::from)
        .context("missing output pack path")?;
    if args.next().is_some() {
        bail!("too many arguments");
    }
    Ok((checkpoint, layer, output))
}

fn main() -> Result<()> {
    let (checkpoint, layer, output) = parse_args()?;
    if checkpoint.is_dir() {
        let (geometry, config_path) = read_safetensors_geometry(&checkpoint, layer)?;
        let tensors = SafeTensors::open(&checkpoint).with_context(|| {
            format!(
                "opening Qwen3.6 Safetensors snapshot {}",
                checkpoint.display()
            )
        })?;
        let layout = publish_safetensors_pack(&tensors, geometry, &output)?;
        println!("checkpoint={}", checkpoint.display());
        println!("source_format=safetensors config={}", config_path.display());
        println!("shards={}", tensors.shards().len());
        println!("layer={} geometry={geometry:?}", geometry.layer);
        println!(
            "provenance=official_hf_bf16 logical_tensors=9 physical_tensors=8 \
             expert_bytes=exact checkpoint_order=true transpose=false"
        );
        println!(
            "scalar_pack=bf16_to_f32 router_transform=identity \
             shared_gate_transform=identity norm_transform=add_one"
        );
        println!("layout=checkpoint_native_column_major version={VERSION}");
        println!("output={} bytes={}", output.display(), layout.file_bytes);
    } else {
        if checkpoint
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        {
            bail!(
                "pass the complete Safetensors snapshot directory, not one shard: {}",
                checkpoint.display()
            );
        }
        let gguf = Gguf::open(&checkpoint)
            .with_context(|| format!("opening Qwen3.6 GGUF checkpoint {}", checkpoint.display()))?;
        let geometry = geometry_from_gguf(&gguf, layer)?;
        let layout = publish_gguf_pack(&gguf, geometry, &output)?;
        println!("checkpoint={}", checkpoint.display());
        println!("source_format=gguf architecture=qwen35moe");
        println!("shards={}", gguf.shards().len());
        println!("layer={} geometry={geometry:?}", geometry.layer);
        println!(
            "provenance=exact_bf16_gguf expert_bytes=exact scalar_pack=f32_exact \
             norm_transform=identity checkpoint_order=true transpose=false"
        );
        println!("layout=checkpoint_native_column_major version={VERSION}");
        println!("output={} bytes={}", output.display(), layout.file_bytes);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn production_geometry() -> Geometry {
        exact_geometry(0)
    }

    fn official_config() -> Value {
        json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "dtype": "bfloat16",
                "hidden_act": "silu",
                "hidden_size": 2048,
                "model_type": "qwen3_5_moe_text",
                "moe_intermediate_size": 512,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 40,
                "rms_norm_eps": 1e-6,
                "shared_expert_intermediate_size": 512
            }
        })
    }

    #[test]
    fn production_layout_is_exact_and_contiguous() {
        let layout = make_layout(production_geometry()).unwrap();
        assert_eq!(
            layout.norm,
            Section {
                offset: 4096,
                bytes: 8192
            }
        );
        assert_eq!(layout.router.offset, layout.norm.offset + layout.norm.bytes);
        assert_eq!(layout.router.bytes, 2_097_152);
        assert_eq!(
            layout.shared_gate.offset,
            layout.router.offset + layout.router.bytes
        );
        assert_eq!(layout.shared_gate.bytes, 8192);
        assert_eq!(layout.gate_up.bytes, 1_077_936_128);
        assert_eq!(layout.down.bytes, 538_968_064);
        assert_eq!(layout.file_bytes, 1_619_021_824);
    }

    #[test]
    fn header_has_stable_little_endian_contract() {
        let geometry = production_geometry();
        let layout = make_layout(geometry).unwrap();
        let header = encode_header(geometry, layout);
        assert_eq!(&header[..16], MAGIC);
        assert_eq!(
            u32::from_le_bytes(header[16..20].try_into().unwrap()),
            VERSION
        );
        assert_eq!(u32::from_le_bytes(header[28..32].try_into().unwrap()), 2048);
        assert_eq!(u32::from_le_bytes(header[36..40].try_into().unwrap()), 256);
        assert_eq!(u32::from_le_bytes(header[52..56].try_into().unwrap()), 1);
        assert_eq!(
            u64::from_le_bytes(header[144..152].try_into().unwrap()),
            layout.file_bytes
        );
        assert!(header[152..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn checked_layout_rejects_overflow() {
        let geometry = Geometry {
            hidden: u32::MAX,
            ffn: u32::MAX,
            experts: u32::MAX,
            shared_ffn: u32::MAX,
            ..production_geometry()
        };
        assert!(make_layout(geometry).is_err());
    }

    #[test]
    fn official_config_geometry_and_layer_bounds_are_exact() {
        assert_eq!(
            geometry_from_config(&official_config(), 39).unwrap(),
            exact_geometry(39)
        );

        let bounds = geometry_from_config(&official_config(), 40)
            .unwrap_err()
            .to_string();
        assert!(bounds.contains("outside config.text_config.num_hidden_layers 40"));

        let mut wrong_dtype = official_config();
        wrong_dtype["text_config"]["dtype"] = json!("float16");
        assert!(geometry_from_config(&wrong_dtype, 0)
            .unwrap_err()
            .to_string()
            .contains("bfloat16"));

        let mut wrong_width = official_config();
        wrong_width["text_config"]["hidden_size"] = json!(4096);
        assert!(geometry_from_config(&wrong_width, 0)
            .unwrap_err()
            .to_string()
            .contains("exact qualified Qwen3.6 shape"));

        let mut wrong_layers = official_config();
        wrong_layers["text_config"]["num_hidden_layers"] = json!(39);
        assert!(geometry_from_config(&wrong_layers, 0)
            .unwrap_err()
            .to_string()
            .contains("must declare 40 layers"));

        assert!(validate_exact_layer_count(39, "qwen35moe.block_count")
            .unwrap_err()
            .to_string()
            .contains("qwen35moe.block_count must declare 40 layers"));
    }

    #[test]
    fn safetensors_names_match_the_official_checkpoint() {
        let names = safetensor_names(3);
        assert_eq!(
            names.norm,
            "model.language_model.layers.3.post_attention_layernorm.weight"
        );
        assert_eq!(
            names.router,
            "model.language_model.layers.3.mlp.gate.weight"
        );
        assert_eq!(
            names.gate_up_experts,
            "model.language_model.layers.3.mlp.experts.gate_up_proj"
        );
        assert_eq!(
            names.down_experts,
            "model.language_model.layers.3.mlp.experts.down_proj"
        );
        assert_eq!(
            names.gate_shared,
            "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight"
        );
        assert_eq!(
            names.up_shared,
            "model.language_model.layers.3.mlp.shared_expert.up_proj.weight"
        );
        assert_eq!(
            names.down_shared,
            "model.language_model.layers.3.mlp.shared_expert.down_proj.weight"
        );
        assert_eq!(
            names.shared_gate,
            "model.language_model.layers.3.mlp.shared_expert_gate.weight"
        );
    }

    #[test]
    fn fused_gate_up_ranges_deinterleave_without_transposing() {
        let geometry = production_geometry();
        let matrix_bytes = 2_097_152;
        assert_eq!(
            fused_gate_up_range(geometry, 0, FusedHalf::Gate).unwrap(),
            0..matrix_bytes
        );
        assert_eq!(
            fused_gate_up_range(geometry, 0, FusedHalf::Up).unwrap(),
            matrix_bytes..2 * matrix_bytes
        );
        assert_eq!(
            fused_gate_up_range(geometry, 1, FusedHalf::Gate).unwrap(),
            2 * matrix_bytes..3 * matrix_bytes
        );
        assert_eq!(
            fused_gate_up_range(geometry, 255, FusedHalf::Up).unwrap(),
            1_071_644_672..1_073_741_824
        );
        assert!(fused_gate_up_range(geometry, 256, FusedHalf::Gate).is_err());
    }

    #[test]
    fn bf16_widening_adds_one_only_when_requested() {
        let input: Vec<u8> = [0x0000u16, 0x3f80, 0xbf00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut output = Vec::new();

        widen_bf16_chunk(&input, &mut output, Bf16Widen::Identity).unwrap();
        let identity: Vec<f32> = output
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(identity, [0.0, 1.0, -0.5]);

        widen_bf16_chunk(&input, &mut output, Bf16Widen::AddOne).unwrap();
        let norm: Vec<f32> = output
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(norm, [1.0, 2.0, 0.5]);
    }
}
