//! Small sparse GGUF fixtures exercise header binding without model payload reads.

use super::*;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone)]
pub(crate) struct Fixture {
    metadata: BTreeMap<String, Value>,
    tensors: BTreeMap<String, (u32, Vec<u64>)>,
}

impl Fixture {
    pub(crate) fn hybrid() -> Self {
        let mut f = Self {
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        f.metadata
            .insert("general.architecture".into(), Value::Str("qwen35".into()));
        f.metadata
            .insert("general.type".into(), Value::Str("model".into()));
        for (suffix, value) in [
            ("block_count", 4),
            ("context_length", 16),
            ("embedding_length", 256),
            ("feed_forward_length", 256),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("full_attention_interval", 4),
            ("rope.dimension_count", 64),
            ("ssm.conv_kernel", 4),
            ("ssm.group_count", 1),
            ("ssm.inner_size", 256),
            ("ssm.state_size", 128),
            ("ssm.time_step_rank", 2),
        ] {
            f.meta(suffix, Value::U(value));
        }
        f.meta("rope.freq_base", Value::F(10_000_000.0));
        f.meta("attention.layer_norm_rms_epsilon", Value::F(1e-6));
        f.meta(
            "rope.dimension_sections",
            Value::Arr([11, 11, 10, 0].map(Value::I).to_vec()),
        );
        f.tensor("token_embd.weight", 12, &[256, 256]);
        f.tensor("output_norm.weight", 0, &[256]);
        f.tensor("output.weight", 14, &[256, 256]);
        for layer in 0..4 {
            let p = |suffix: &str| format!("blk.{layer}.{suffix}");
            for suffix in ["attn_norm.weight", "post_attention_norm.weight"] {
                f.tensor(&p(suffix), 0, &[256]);
            }
            for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                f.tensor(
                    &p(suffix),
                    if suffix == "ffn_down.weight" { 14 } else { 12 },
                    &[256, 256],
                );
            }
            if layer == 3 {
                f.tensor(&p("attn_q.weight"), 12, &[256, 1024]);
                f.tensor(&p("attn_k.weight"), 12, &[256, 256]);
                f.tensor(&p("attn_v.weight"), 14, &[256, 256]);
                f.tensor(&p("attn_output.weight"), 12, &[512, 256]);
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    f.tensor(&p(suffix), 0, &[256]);
                }
            } else {
                f.tensor(&p("attn_qkv.weight"), 14, &[256, 512]);
                f.tensor(&p("attn_gate.weight"), 12, &[256, 256]);
                f.tensor(&p("ssm_alpha.weight"), 12, &[256, 2]);
                f.tensor(&p("ssm_beta.weight"), 12, &[256, 2]);
                f.tensor(&p("ssm_conv1d.weight"), 0, &[4, 512]);
                f.tensor(&p("ssm_a"), 0, &[2]);
                f.tensor(&p("ssm_dt.bias"), 0, &[2]);
                f.tensor(&p("ssm_norm.weight"), 0, &[128]);
                f.tensor(&p("ssm_out.weight"), 12, &[256, 256]);
            }
        }
        f
    }

    fn meta(&mut self, suffix: &str, value: Value) {
        self.metadata.insert(format!("qwen35.{suffix}"), value);
    }
    pub(crate) fn tensor(&mut self, name: &str, kind: u32, dims: &[u64]) {
        self.tensors.insert(name.to_owned(), (kind, dims.to_vec()));
    }

    pub(crate) fn open(&self) -> (Temporary, Gguf) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "beholdd-qwen35-bind-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("header.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            string(&mut bytes, key);
            encode_value(&mut bytes, value, true);
        }
        let mut offset = 0u64;
        for (name, (kind, dims)) in &self.tensors {
            string(&mut bytes, name);
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let (block, block_bytes) = ggml_block_geometry(*kind).unwrap();
            let encoded = dims.iter().product::<u64>() / block as u64 * block_bytes as u64;
            offset = (offset + encoded + 31) & !31;
        }
        bytes.resize((bytes.len() + 31) & !31, 0);
        let mut file = File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        // Sparse length establishes valid indexed spans; no inference reads these zeros.
        file.set_len(bytes.len() as u64 + offset).unwrap();
        drop(file);
        let gguf = Gguf::open(&path).unwrap();
        (Temporary(directory), gguf)
    }
}

pub(crate) struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn encode_value(bytes: &mut Vec<u8>, value: &Value, typed: bool) {
    let tag = match value {
        Value::U(_) => 10u32,
        Value::I(_) => 11,
        Value::F(_) => 12,
        Value::Str(_) => 8,
        Value::Bool(_) => 7,
        Value::Arr(_) => 9,
    };
    if typed {
        bytes.extend_from_slice(&tag.to_le_bytes());
    }
    match value {
        Value::U(n) => bytes.extend_from_slice(&n.to_le_bytes()),
        Value::I(n) => bytes.extend_from_slice(&n.to_le_bytes()),
        Value::F(n) => bytes.extend_from_slice(&n.to_le_bytes()),
        Value::Str(s) => string(bytes, s),
        Value::Bool(v) => bytes.push(u8::from(*v)),
        Value::Arr(values) => {
            // Fixtures use integer sections only, matching the actual checkpoint's I32
            // values after the GGUF reader has widened them to its signed Value form.
            bytes.extend_from_slice(&11u32.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                assert!(matches!(value, Value::I(_)));
                encode_value(bytes, value, false);
            }
        }
    }
}

fn refusal(fixture: &Fixture) -> String {
    let (_file, gguf) = fixture.open();
    Binding::load(&gguf).unwrap_err().to_string()
}

#[test]
fn binds_separate_recurrent_and_gated_full_attention_manifests_without_payload_reads() {
    let (_file, gguf) = Fixture::hybrid().open();
    let binding = Binding::load(&gguf).unwrap();
    assert_eq!(gguf.len(), 56);
    assert_eq!(binding.checkpoint_id(), gguf.checkpoint_id());
    assert_eq!(binding.config().delta_key_width(), 128);
    assert_eq!(binding.config().delta_qkv_width(), 512);
    assert_eq!(binding.config().delta_value_head_dim(), 128);
    assert!(binding.config().is_full_attention(4).is_err());
    assert!(binding.blocks()[..3]
        .iter()
        .all(|b| matches!(&b.attention, Attention::DeltaNet(_))));
    let Attention::Full(full) = &binding.blocks()[3].attention else {
        panic!("fourth block must be full attention");
    };
    assert_eq!(full.query_gate.dims(), [256, 1024]);
    assert_eq!(binding.embedding().kind().0, 12);
    assert_eq!(binding.output().kind().0, 14);
    assert!(binding.checkpoint_bytes() > 0);
}

#[test]
fn refuses_other_architectures_and_unknown_semantic_switches() {
    for architecture in ["qwen3", "qwen35moe", "qwen3next"] {
        let mut f = Fixture::hybrid();
        f.metadata.insert(
            "general.architecture".into(),
            Value::Str(architecture.into()),
        );
        assert!(refusal(&f).contains("general.architecture=qwen35"));
    }
    for suffix in [
        "rope.scaling.factor",
        "attention.sliding_window",
        "expert_count",
        "ssm.activation",
    ] {
        let mut f = Fixture::hybrid();
        f.meta(suffix, Value::U(2));
        assert!(refusal(&f).contains(&format!(
            "unsupported qwen35 architecture metadata: qwen35.{suffix}"
        )));
    }
}

#[test]
fn refuses_missing_malformed_and_inconsistent_hybrid_metadata() {
    let cases = [
        ("full_attention_interval", Value::U(0)),
        ("ssm.conv_kernel", Value::U(3)),
        ("ssm.group_count", Value::U(3)),
        ("ssm.inner_size", Value::U(128)),
        ("ssm.state_size", Value::U(u64::MAX)),
        ("attention.value_length", Value::U(128)),
        ("attention.head_count", Value::F(2.0)),
        ("attention.layer_norm_rms_epsilon", Value::F(f64::NAN)),
        ("rope.freq_base", Value::F(f64::INFINITY)),
        (
            "rope.dimension_sections",
            Value::Arr([11, 11, -1, 0].map(Value::I).to_vec()),
        ),
        (
            "rope.dimension_sections",
            Value::Arr([11, 11, 11, 0].map(Value::I).to_vec()),
        ),
    ];
    for (key, value) in cases {
        let mut f = Fixture::hybrid();
        f.meta(key, value);
        assert!(!refusal(&f).is_empty(), "must refuse {key}");
    }
    let mut f = Fixture::hybrid();
    f.metadata.remove("qwen35.ssm.time_step_rank");
    assert!(refusal(&f).contains("ssm.time_step_rank"));
}

#[test]
fn refuses_incomplete_untied_head_and_layer_pattern_substitution() {
    let mut f = Fixture::hybrid();
    f.tensors.remove("output.weight");
    assert!(refusal(&f).contains("exactly 56 tensors"));
    let mut f = Fixture::hybrid();
    let tensor = f.tensors.remove("blk.0.attn_qkv.weight").unwrap();
    f.tensors.insert("blk.0.attn_q.weight".into(), tensor);
    assert!(refusal(&f).contains("missing qwen35 tensor blk.0.attn_qkv.weight"));
    let mut f = Fixture::hybrid();
    f.tensor("blk.0.unimplemented.bias", 0, &[256]);
    assert!(refusal(&f).contains("exactly 56 tensors"));
}

#[test]
fn refuses_wrong_gate_convolution_and_norm_dimensions() {
    for (name, dims) in [
        ("blk.3.attn_q.weight", vec![256, 512]),
        ("blk.0.ssm_conv1d.weight", vec![512, 4]),
        ("blk.0.ssm_norm.weight", vec![256]),
        ("blk.0.ssm_alpha.weight", vec![256, 1]),
    ] {
        let mut f = Fixture::hybrid();
        f.tensors.get_mut(name).unwrap().1 = dims;
        let error = refusal(&f);
        assert!(error.contains(name) && error.contains("expected GGUF dimensions"));
    }
}

#[test]
fn accepts_both_checkpoint_matrix_quants_but_refuses_unimplemented_storage() {
    for kind in [12, 14] {
        let mut f = Fixture::hybrid();
        f.tensors.get_mut("blk.0.ssm_alpha.weight").unwrap().0 = kind;
        let (_file, gguf) = f.open();
        assert!(Binding::load(&gguf).is_ok());
    }
    for (name, kind, message) in [
        ("blk.0.ssm_alpha.weight", 30, "Q4_K or Q6_K matrix"),
        ("blk.0.ssm_a", 1, "F32 parameter"),
        ("blk.0.ssm_conv1d.weight", 1, "F32 parameter"),
    ] {
        let mut f = Fixture::hybrid();
        f.tensors.get_mut(name).unwrap().0 = kind;
        assert!(refusal(&f).contains(message));
    }
}

#[test]
#[ignore = "requires BEHOLDD_QWEN35_MODEL; reads only its GGUF header and tensor index"]
fn actual_openmythos_header_binds_without_loading_weights() -> Result<()> {
    let path = std::env::var_os("BEHOLDD_QWEN35_MODEL").context("set BEHOLDD_QWEN35_MODEL")?;
    let gguf = Gguf::open(PathBuf::from(path))?;
    let binding = Binding::load(&gguf)?;
    assert_eq!(binding.config().n_layers, 64);
    assert_eq!(binding.config().dim, 5120);
    assert_eq!(binding.config().feed_forward, 17408);
    assert_eq!(binding.config().vocab, 248320);
    assert_eq!(binding.config().delta_key_width(), 2048);
    assert_eq!(binding.config().delta_qkv_width(), 10240);
    assert_eq!(
        binding
            .blocks()
            .iter()
            .filter(|b| matches!(&b.attention, Attention::Full(_)))
            .count(),
        16
    );
    assert_eq!(gguf.len(), 851);
    Ok(())
}
