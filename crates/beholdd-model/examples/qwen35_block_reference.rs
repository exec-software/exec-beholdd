//! Explicit, bounded CPU block probe for comparison with an independent oracle.
//! This example never registers a production backend or discovers a model path.

use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use beholdd_format::{CheckpointId, Gguf, WeightRange};
use beholdd_memory::{ResidentSource, ResidentWeight, WeightAccess, WeightSource};
use beholdd_model::qwen35::config::Binding;
use beholdd_model::qwen35::reference::BlockReference;
use beholdd_model::qwen35::weights::ReadLimits;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Serialize;

const MIB: usize = 1024 * 1024;
const MAX_INPUT_BYTES: usize = MIB;
const MAX_OUTPUT_BYTES: usize = MIB;
const MAX_POSITIONS: usize = 8;
const MAX_TILE_ROWS: usize = 64;

const USAGE: &str = r#"qwen35_block_reference \
  --model MODEL.gguf --layer N --input INPUT.json --output OUTPUT.json \
  --max-positions N --memory-budget-mib N --deadline-secs N [--tile-rows N]

INPUT: {"schema_version":1,"hidden":[[...]]}, with finite f32 values and the
checkpoint's exact hidden width. Positions are bounded by --max-positions (1..8).
Tile rows default to 16 (1..64). Both JSON files are limited to 1 MiB.
The output must not already exist. Memory admission covers CPU payload and JSON
buffers; checkpoint metadata, allocator/thread overhead and OS file cache are
separate. The deadline is cooperative, checked before every bounded weight read
and tile; it cannot interrupt a blocked kernel I/O call or an in-flight CPU tile.
No weight payload is read until arguments, input and memory admission pass.
After the main sequence, capacity refusal is checked without reads, then the first
input is replayed from reset state and must exactly match its original output."#;

struct Args {
    model: PathBuf,
    layer: usize,
    input: PathBuf,
    output: PathBuf,
    max_positions: usize,
    memory_budget: usize,
    deadline_secs: u64,
    tile_rows: usize,
}

fn parse_args() -> Result<Option<Args>> {
    let mut values = std::collections::BTreeMap::<String, OsString>::new();
    let mut args = env::args_os().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("{USAGE}");
            return Ok(None);
        }
        let flag = flag
            .into_string()
            .map_err(|_| anyhow::anyhow!("option is not UTF-8"))?;
        ensure!(
            matches!(
                flag.as_str(),
                "--model"
                    | "--layer"
                    | "--input"
                    | "--output"
                    | "--max-positions"
                    | "--memory-budget-mib"
                    | "--deadline-secs"
                    | "--tile-rows"
            ),
            "unknown option {flag}; use --help"
        );
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        ensure!(
            values.insert(flag.clone(), value).is_none(),
            "duplicate option {flag}"
        );
    }
    let mut take = |flag: &str| {
        values
            .remove(flag)
            .with_context(|| format!("required option {flag}; use --help"))
    };
    let model = PathBuf::from(take("--model")?);
    let layer = number::<usize>(take("--layer")?, "--layer")?;
    let input = PathBuf::from(take("--input")?);
    let output = PathBuf::from(take("--output")?);
    let max_positions = number::<usize>(take("--max-positions")?, "--max-positions")?;
    let memory_mib = number::<usize>(take("--memory-budget-mib")?, "--memory-budget-mib")?;
    let deadline_secs = number::<u64>(take("--deadline-secs")?, "--deadline-secs")?;
    let tile_rows = match values.remove("--tile-rows") {
        Some(value) => number::<usize>(value, "--tile-rows")?,
        None => 16,
    };
    ensure!(
        (1..=MAX_POSITIONS).contains(&max_positions),
        "--max-positions must be 1..={MAX_POSITIONS}"
    );
    ensure!(
        (1..=MAX_TILE_ROWS).contains(&tile_rows),
        "--tile-rows must be 1..={MAX_TILE_ROWS}"
    );
    ensure!(
        memory_mib > 0 && deadline_secs > 0,
        "memory budget and deadline must be positive"
    );
    let memory_budget = memory_mib
        .checked_mul(MIB)
        .context("memory budget overflows host size")?;
    Ok(Some(Args {
        model,
        layer,
        input,
        output,
        max_positions,
        memory_budget,
        deadline_secs,
        tile_rows,
    }))
}

fn number<T: std::str::FromStr>(value: OsString, flag: &str) -> Result<T> {
    value
        .to_str()
        .context("numeric argument is not UTF-8")?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid number for {flag}"))
}

fn check_deadline(deadline: Instant) -> Result<()> {
    ensure!(
        Instant::now() < deadline,
        "qwen35 block reference deadline exceeded"
    );
    Ok(())
}

fn regular_file(path: &Path) -> Result<()> {
    ensure!(
        fs::metadata(path)
            .with_context(|| format!("stat {}", path.display()))?
            .is_file(),
        "{} must be a regular file",
        path.display()
    );
    Ok(())
}

fn read_input(path: &Path) -> Result<Vec<u8>> {
    regular_file(path)?;
    let file = File::open(path).with_context(|| format!("open input {}", path.display()))?;
    ensure!(file.metadata()?.is_file(), "input must be a regular file");
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(MAX_INPUT_BYTES + 1)
        .context("allocate bounded input buffer")?;
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .context("read bounded input JSON")?;
    ensure!(bytes.len() <= MAX_INPUT_BYTES, "input JSON exceeds 1 MiB");
    Ok(bytes)
}

// Seeded visitors bound the number and width of rows *during* deserialization.
// In particular, f32 values are deserialized directly from the JSON decimals,
// without an intermediate serde_json::Value/f64 conversion and reserialization.
struct InputSeed {
    width: usize,
    max_positions: usize,
}
struct HiddenSeed {
    width: usize,
    max_positions: usize,
}
struct RowSeed {
    width: usize,
    permitted: bool,
}

impl<'de> DeserializeSeed<'de> for InputSeed {
    type Value = Vec<Vec<f32>>;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for InputSeed {
    type Value = Vec<Vec<f32>>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("schema_version=1 and bounded hidden rows")
    }
    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut version = None;
        let mut hidden = None;
        while let Some(field) = map.next_key::<String>()? {
            match field.as_str() {
                "schema_version" => {
                    if version.is_some() {
                        return Err(de::Error::duplicate_field("schema_version"));
                    }
                    let value = map.next_value::<u32>()?;
                    if value != 1 {
                        return Err(de::Error::custom("schema_version must be 1"));
                    }
                    version = Some(value);
                }
                "hidden" => {
                    if hidden.is_some() {
                        return Err(de::Error::duplicate_field("hidden"));
                    }
                    hidden = Some(map.next_value_seed(HiddenSeed {
                        width: self.width,
                        max_positions: self.max_positions,
                    })?);
                }
                _ => {
                    return Err(de::Error::unknown_field(
                        &field,
                        &["schema_version", "hidden"],
                    ))
                }
            }
        }
        version.ok_or_else(|| de::Error::missing_field("schema_version"))?;
        hidden.ok_or_else(|| de::Error::missing_field("hidden"))
    }
}

impl<'de> DeserializeSeed<'de> for HiddenSeed {
    type Value = Vec<Vec<f32>>;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for HiddenSeed {
    type Value = Vec<Vec<f32>>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "1..={} hidden rows", self.max_positions)
    }
    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.max_positions)
            .map_err(de::Error::custom)?;
        while let Some(row) = seq.next_element_seed(RowSeed {
            width: self.width,
            permitted: rows.len() < self.max_positions,
        })? {
            rows.push(row);
        }
        if rows.is_empty() {
            return Err(de::Error::custom("hidden must have at least one row"));
        }
        Ok(rows)
    }
}

impl<'de> DeserializeSeed<'de> for RowSeed {
    type Value = Vec<f32>;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        if !self.permitted {
            return Err(de::Error::custom("hidden exceeds --max-positions"));
        }
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for RowSeed {
    type Value = Vec<f32>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "exactly {} finite f32 values", self.width)
    }
    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut row = Vec::new();
        row.try_reserve_exact(self.width)
            .map_err(de::Error::custom)?;
        while let Some(value) = seq.next_element::<f32>()? {
            if row.len() == self.width {
                return Err(de::Error::custom("hidden row exceeds checkpoint width"));
            }
            if !value.is_finite() {
                return Err(de::Error::custom("hidden contains a non-finite f32"));
            }
            row.push(value);
        }
        if row.len() != self.width {
            return Err(de::Error::custom(
                "hidden row does not match checkpoint width",
            ));
        }
        Ok(row)
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Serialize)]
struct ReadStats {
    calls: u64,
    bytes: u64,
    largest_read_bytes: u64,
}

struct MeasuredSource<'a> {
    direct: ResidentSource<'a>,
    stats: Mutex<ReadStats>,
    deadline: Instant,
}

impl WeightSource for MeasuredSource<'_> {
    fn checkpoint_id(&self) -> CheckpointId {
        self.direct.checkpoint_id()
    }
    fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
        check_deadline(self.deadline)?;
        let weight = self.direct.load_range(range, access)?;
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| anyhow::anyhow!("read statistics lock poisoned"))?;
        stats.calls = stats.calls.checked_add(1).context("read count overflow")?;
        stats.bytes = stats
            .bytes
            .checked_add(range.nbytes())
            .context("read byte count overflow")?;
        stats.largest_read_bytes = stats.largest_read_bytes.max(range.nbytes());
        check_deadline(self.deadline)?;
        Ok(weight)
    }
}

#[derive(Serialize)]
struct Output<'a> {
    schema_version: u32,
    layer: usize,
    block_kind: &'static str,
    positions: usize,
    hidden: &'a [Vec<f32>],
    memory: MemoryReport,
    qualification: Qualification,
    /// Successful reads including the reset/replay qualification.
    reads: ReadStats,
    elapsed_seconds: f64,
    tile_rows: usize,
}

#[derive(Serialize)]
struct Qualification {
    capacity: usize,
    capacity_refusal_without_read: bool,
    capacity_refusal_preserved_positions: bool,
    capacity_refusal_not_poisoned: bool,
    reset_replay_exact: bool,
    main_reads: ReadStats,
}

#[derive(Serialize)]
struct MemoryReport {
    state_bytes: usize,
    scratch_bytes: usize,
    block_payload_bytes: usize,
    io_payload_bytes: usize,
    admitted_payload_bytes: usize,
    budget_bytes: usize,
}

struct BoundedOutput(Vec<u8>);
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_OUTPUT_BYTES - self.0.len() {
            return Err(io::Error::other("output JSON exceeds 1 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn write_output(path: &Path, output: &Output<'_>) -> Result<()> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(MAX_OUTPUT_BYTES)
        .context("allocate bounded output buffer")?;
    let mut buffer = BoundedOutput(bytes);
    serde_json::to_writer(&mut buffer, output).context("serialize bounded output JSON")?;
    buffer.write_all(b"\n")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create new output {}", path.display()))?;
    let result = file.write_all(&buffer.0).and_then(|()| file.sync_all());
    if let Err(error) = result {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(error).context("write complete output JSON");
    }
    Ok(())
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let started = Instant::now();
    let deadline = started
        .checked_add(Duration::from_secs(args.deadline_secs))
        .context("deadline is too large")?;
    match fs::symlink_metadata(&args.output) {
        Ok(_) => bail!("output already exists: {}", args.output.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("check output path"),
    }
    let input_json = read_input(&args.input)?;
    check_deadline(deadline)?;
    regular_file(&args.model)?;
    let gguf = Gguf::open(&args.model).context("open model headers")?;
    let binding = Binding::load(&gguf).context("bind qwen35 checkpoint headers")?;
    check_deadline(deadline)?;
    let limits = ReadLimits {
        tile_rows: args.tile_rows,
        ..ReadLimits::default()
    };
    let forecast =
        BlockReference::memory_forecast(&binding, args.layer, args.max_positions, limits)?;
    // Both full input/output matrices can coexist with block scratch. Include the
    // reserved JSON buffers too; metadata and allocator/thread overhead are separate.
    let matrices = binding
        .config()
        .dim
        .checked_mul(args.max_positions)
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>()))
        .context("I/O matrix forecast overflow")?;
    let io_payload = matrices
        .checked_add(MAX_INPUT_BYTES + 1 + MAX_OUTPUT_BYTES)
        .context("I/O payload forecast overflow")?;
    let admitted_payload = forecast
        .payload_bytes
        .checked_add(io_payload)
        .context("total payload forecast overflow")?;
    ensure!(
        admitted_payload <= args.memory_budget,
        "block and I/O payload need {admitted_payload} bytes, budget is {}",
        args.memory_budget
    );
    let mut deserializer = serde_json::Deserializer::from_slice(&input_json);
    let input = InputSeed {
        width: binding.config().dim,
        max_positions: args.max_positions,
    }
    .deserialize(&mut deserializer)
    .context("validate input JSON")?;
    deserializer
        .end()
        .context("trailing content in input JSON")?;
    check_deadline(deadline)?;
    let mut block = BlockReference::new(
        &binding,
        args.layer,
        input.len(),
        forecast.payload_bytes,
        limits,
    )?;
    let source = MeasuredSource {
        direct: ResidentSource::new(&gguf),
        stats: Mutex::default(),
        deadline,
    };
    let mut hidden = Vec::new();
    hidden
        .try_reserve_exact(input.len())
        .context("allocate bounded output rows")?;
    for row in &input {
        hidden.push(block.step(&source, row, &mut || check_deadline(deadline))?);
    }
    check_deadline(deadline)?;
    let main_positions = block.positions();
    let main_reads = *source
        .stats
        .lock()
        .map_err(|_| anyhow::anyhow!("read statistics lock poisoned"))?;
    ensure!(
        main_positions == input.len(),
        "main sequence position count mismatch"
    );
    let refusal = block.step(&source, &input[0], &mut || check_deadline(deadline));
    ensure!(
        refusal.is_err(),
        "block accepted a token beyond its admitted capacity"
    );
    ensure!(
        *source
            .stats
            .lock()
            .map_err(|_| anyhow::anyhow!("read statistics lock poisoned"))?
            == main_reads,
        "capacity refusal performed a weight read"
    );
    ensure!(
        block.positions() == main_positions,
        "capacity refusal changed position count"
    );
    ensure!(!block.is_poisoned(), "capacity refusal poisoned the block");
    check_deadline(deadline)?;
    block.reset();
    ensure!(
        block.positions() == 0 && !block.is_poisoned(),
        "reset did not clear sequence state"
    );
    let replay = block.step(&source, &input[0], &mut || check_deadline(deadline))?;
    ensure!(
        replay == hidden[0],
        "first-position output differs after reset/replay"
    );
    ensure!(
        block.positions() == 1 && !block.is_poisoned(),
        "replay sequence state is invalid"
    );
    drop(replay);
    check_deadline(deadline)?;
    let output = Output {
        schema_version: 1,
        layer: args.layer,
        block_kind: if binding.config().is_full_attention(args.layer)? {
            "full_attention"
        } else {
            "delta_net"
        },
        positions: main_positions,
        hidden: &hidden,
        memory: MemoryReport {
            state_bytes: forecast.state_bytes,
            scratch_bytes: forecast.scratch_bytes,
            block_payload_bytes: forecast.payload_bytes,
            io_payload_bytes: io_payload,
            admitted_payload_bytes: admitted_payload,
            budget_bytes: args.memory_budget,
        },
        qualification: Qualification {
            capacity: input.len(),
            capacity_refusal_without_read: true,
            capacity_refusal_preserved_positions: true,
            capacity_refusal_not_poisoned: true,
            reset_replay_exact: true,
            main_reads,
        },
        reads: *source
            .stats
            .lock()
            .map_err(|_| anyhow::anyhow!("read statistics lock poisoned"))?,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        tile_rows: args.tile_rows,
    };
    write_output(&args.output, &output)?;
    eprintln!("wrote {} positions from layer {} ({}) in {:.3}s; {} reads, {} bytes; payload forecast {} bytes",
        output.positions, output.layer, output.block_kind, output.elapsed_seconds,
        output.reads.calls, output.reads.bytes, output.memory.admitted_payload_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Temporary(PathBuf);

    impl Temporary {
        fn new() -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = env::temp_dir().join(format!(
                "qwen35-block-cli-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn parse(json: &str, width: usize, max_positions: usize) -> Result<Vec<Vec<f32>>> {
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let rows = InputSeed {
            width,
            max_positions,
        }
        .deserialize(&mut deserializer)?;
        deserializer.end()?;
        Ok(rows)
    }

    fn output(hidden: &[Vec<f32>]) -> Output<'_> {
        Output {
            schema_version: 1,
            layer: 0,
            block_kind: "delta_net",
            positions: hidden.len(),
            hidden,
            memory: MemoryReport {
                state_bytes: 4,
                scratch_bytes: 20,
                block_payload_bytes: 24,
                io_payload_bytes: 0,
                admitted_payload_bytes: 24,
                budget_bytes: 24,
            },
            qualification: Qualification {
                capacity: hidden.len(),
                capacity_refusal_without_read: true,
                capacity_refusal_preserved_positions: true,
                capacity_refusal_not_poisoned: true,
                reset_replay_exact: true,
                main_reads: ReadStats::default(),
            },
            reads: ReadStats::default(),
            elapsed_seconds: 0.0,
            tile_rows: 16,
        }
    }

    #[test]
    fn parser_rounds_decimal_values_at_the_f32_boundary() -> Result<()> {
        let rows = parse(
            r#"{"hidden":[[0.1,-0.2,1.00000006,16777217,1e-60,-0.0]],"schema_version":1}"#,
            6,
            1,
        )?;
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0x3dcccccd, 0xbe4ccccd, 0x3f800001, 0x4b800000, 0, 0x80000000]
        );
        assert_eq!(
            parse(
                r#"{"schema_version":1,"hidden":[[1,-2],[0.5,3.25]]} "#,
                2,
                2
            )?,
            [[1.0, -2.0], [0.5, 3.25]]
        );
        Ok(())
    }

    #[test]
    fn parser_refuses_ambiguous_schema_shapes_and_nonfinite_values() {
        for json in [
            r#"{"schema_version":2,"hidden":[[1,2]]}"#,
            r#"{"schema_version":true,"hidden":[[1,2]]}"#,
            r#"{"hidden":[[1,2]]}"#,
            r#"{"schema_version":1}"#,
            r#"{"schema_version":1,"hidden":[[1,2]],"extra":0}"#,
            r#"{"schema_version":1,"schema_version":1,"hidden":[[1,2]]}"#,
            r#"{"schema_version":1,"hidden":[[1,2]],"hidden":[[3,4]]}"#,
            r#"{"schema_version":1,"hidden":[]}"#,
            r#"{"schema_version":1,"hidden":[[1]]}"#,
            r#"{"schema_version":1,"hidden":[[1,2,3]]}"#,
            r#"{"schema_version":1,"hidden":[[1,2],[3,4]]}"#,
            r#"{"schema_version":1,"hidden":[[true,2]]}"#,
            r#"{"schema_version":1,"hidden":[[NaN,2]]}"#,
            r#"{"schema_version":1,"hidden":[[Infinity,2]]}"#,
            r#"{"schema_version":1,"hidden":[[1e39,2]]}"#,
            r#"{"schema_version":1,"hidden":[[-1e39,2]]}"#,
            r#"{"schema_version":1,"hidden":[[1e400,2]]}"#,
            r#"{"schema_version":1,"hidden":[[1,2]]} {}"#,
        ] {
            assert!(parse(json, 2, 1).is_err(), "accepted {json}");
        }
    }

    #[test]
    fn input_read_has_an_exact_byte_limit_and_requires_a_file() -> Result<()> {
        let temporary = Temporary::new()?;
        let path = temporary.0.join("input.json");
        fs::write(&path, vec![b' '; MAX_INPUT_BYTES])?;
        assert_eq!(read_input(&path)?.len(), MAX_INPUT_BYTES);
        OpenOptions::new()
            .append(true)
            .open(&path)?
            .write_all(b" ")?;
        assert!(read_input(&path)
            .unwrap_err()
            .to_string()
            .contains("exceeds 1 MiB"));
        assert!(read_input(&temporary.0).is_err());
        Ok(())
    }

    #[test]
    fn output_writer_enforces_its_byte_limit_without_appending_partial_data() -> Result<()> {
        let mut writer = BoundedOutput(Vec::new());
        writer.write_all(&vec![b'x'; MAX_OUTPUT_BYTES - 1])?;
        writer.write_all(b"y")?;
        assert!(writer.write_all(b"z").is_err());
        assert_eq!(writer.0.len(), MAX_OUTPUT_BYTES);
        assert_eq!(writer.0.last(), Some(&b'y'));
        Ok(())
    }

    #[test]
    fn output_creation_refuses_overwrite_and_serialization_overflow() -> Result<()> {
        let temporary = Temporary::new()?;
        let path = temporary.0.join("output.json");
        let hidden = [vec![0.5, -0.25]];
        write_output(&path, &output(&hidden))?;
        let original = fs::read(&path)?;
        let parsed: serde_json::Value = serde_json::from_slice(&original)?;
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["hidden"], serde_json::json!([[0.5, -0.25]]));
        assert!(write_output(&path, &output(&[vec![9.0, 9.0]])).is_err());
        assert_eq!(fs::read(&path)?, original);
        assert!(write_output(&temporary.0, &output(&hidden)).is_err());
        assert!(temporary.0.is_dir());

        let too_large = temporary.0.join("oversized.json");
        let large_hidden = [vec![f32::MAX; MAX_OUTPUT_BYTES / 8]];
        let error = write_output(&too_large, &output(&large_hidden)).unwrap_err();
        assert!(format!("{error:#}").contains("output JSON exceeds 1 MiB"));
        assert!(
            !too_large.exists(),
            "oversized JSON must refuse before creating a file"
        );
        Ok(())
    }
}
