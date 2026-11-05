//! SOURCE-ONLY DRAFT: bounded synthetic resident Delta qualification.
//! Not compiled, tested, executed, or registered as a repository target.
//! Independent scalar fixtures finish before device open; no GGUF/model discovery.
//! Native requires an external OOM/thermal/crash/fence supervisor.
//!
//! Separate slot16 extension plan (deliberately excluded from these receipts):
//! sigmoid-gated multiply at lengths1/65/4096, two changed-input replays each;
//! independent literal F32 exp->sigmoid->multiply oracle, amplified gates
//! -90/-103/-105, both signed zeros, and +/-MAX; guarded output and exact
//! residency/dispatch/transfer accounting. Do not silently fold that operation
//! into this Delta gate or reuse its expected counts.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{bail, ensure, Context, Result};
    use beholdd_backend::execution::{ExecutionDevice, ExecutionError};
    use beholdd_backend::hag::native_execution::delta::{
        NativeDeltaParameters, NativeDeltaSequence, NativeDeltaShape, NATIVE_DELTA_GUARD,
    };
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionProfile,
        NativeExecutionStats,
    };
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const MIB: u64 = 1024 * 1024;
    const GPU_BUDGET: u64 = 16 * MIB;
    const CPU_PAYLOAD_BOUND: u64 = 64 * MIB;
    const HOST_HEADROOM: usize = 16 * MIB as usize;
    const DEADLINE_SECONDS: u64 = 180;
    // Registered before hardware results. Never widen in response to a failure
    // without investigating and independently reviewing the numerical contract.
    const ABS_TOL: f64 = 2e-5;
    const REL_TOL: f64 = 2e-5;
    const TAIL_EXP_ULPS: u32 = 8;
    // A rounded subnormal exp error is amplified by the subsequent SiLU product
    // (at -90, one exp ULP is about23 output ULPs). Fix this bound before runs.
    const TAIL_PRODUCT_ULPS: u32 = 64;
    // Canonical bit1 exp(-103), signed0 exp(-105), and their small amplified
    // products must be exact; an ULP allowance near zero would hide huge ratios.
    const EXACT_TAIL_MAGNITUDE_BITS: u32 = 256;
    const FIXTURES: usize = 4;
    const REFERENCE_STEPS: usize = 7;
    const MAIN_STEPS: usize = 5;
    const GPU_STEPS_PER_FIXTURE: u64 = 15;
    const STATUS_READS_PER_FIXTURE: u64 = 13;
    const FRAME_READS_PER_FIXTURE: u64 = 18;
    const OUTPUT_READS_PER_FIXTURE: u64 = 9;
    const PROJECTION_WRITES_PER_FIXTURE: u64 = 16;
    const EXPECTED_TAIL_CHECKS: u64 = 378;
    const EXPECTED_RHO_CHECKS: u64 = 108;
    const RHO_MINUS_103_MAX_ULPS: u32 = 1;
    // At -90, an eight-ULP subnormal exp error, amplified by |A|<2^126,
    // changes log_decay by <2^-20. The resulting rho is in [0.5,1), whose
    // F32 spacing is2^-24: at most16 output ULPs before rounding/SFU error.
    // A factor-two margin fixes32ULPs; a flushed softplus yields1 and fails.
    const RHO_MINUS_90_MAX_ULPS: u32 = 32;

    fn config() -> NativeExecutionConfig {
        NativeExecutionConfig {
            weight_capacity_bytes: 4096,
            tensor_capacity_bytes: 14 * MIB as u32,
            auxiliary_capacity_bytes: 4096,
        }
    }

    enum Mode {
        Cpu,
        Native(PathBuf),
        Help,
    }

    fn mode(args: &[OsString]) -> Result<Mode> {
        match args {
            [flag] if flag == "--cpu-reference" => Ok(Mode::Cpu),
            [flag] if flag == "--help" || flag == "-h" => Ok(Mode::Help),
            [flag, path] if flag == "--native" => {
                let path = PathBuf::from(path);
                let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
                let suffix = name.strip_prefix("renderD").unwrap_or("");
                ensure!(
                    path.parent() == Some(Path::new("/dev/dri"))
                        && !suffix.is_empty()
                        && suffix.bytes().all(|b| b.is_ascii_digit()),
                    "--native requires an explicit /dev/dri/renderD... node"
                );
                Ok(Mode::Native(path))
            }
            _ => bail!("expected --cpu-reference or --native /dev/dri/renderD..."),
        }
    }

    fn deadline(until: Instant) -> Result<()> {
        ensure!(
            Instant::now() < until,
            "180-second cooperative deadline exceeded"
        );
        Ok(())
    }

    fn control_deadline(until: Instant) -> std::result::Result<(), ExecutionError> {
        if Instant::now() < until {
            Ok(())
        } else {
            Err(ExecutionError::Shape("probe deadline exceeded".into()))
        }
    }

    fn finite(x: f32) -> Result<f32> {
        ensure!(
            x.is_finite(),
            "independent reference nonfinite intermediate"
        );
        Ok(x)
    }
    fn add(a: f32, b: f32) -> Result<f32> {
        finite(finite(a)? + finite(b)?)
    }
    fn mul(a: f32, b: f32) -> Result<f32> {
        finite(finite(a)? * finite(b)?)
    }
    fn sigmoid(x: f32) -> Result<f32> {
        let t = finite((-finite(x)?.abs()).exp())?;
        finite((if x >= 0.0 { 1.0 } else { t }) / add(1.0, t)?)
    }
    fn silu(x: f32) -> Result<f32> {
        mul(x, sigmoid(x)?)
    }
    fn softplus(x: f32) -> Result<f32> {
        add(finite(x)?.max(0.0), finite((-x.abs()).exp().ln_1p())?)
    }
    fn inverse_root(x: f32) -> Result<f32> {
        finite(finite(x)?.sqrt().recip())
    }
    fn norm_scale(values: &[f32], epsilon: f32) -> Result<f32> {
        let mut total = 0.0f64;
        for &x in values {
            total += f64::from(mul(x, x)?);
        }
        inverse_root(add(finite((total / values.len() as f64) as f32)?, epsilon)?)
    }

    // Independent word geometry. NativeDeltaShape::layout is only cross-checked
    // against this; it does not generate the reference state or fixture layout.
    #[derive(Clone, Copy)]
    struct Geometry {
        hk: usize,
        hv: usize,
        d: usize,
        eps: f32,
    }
    #[derive(Clone, Copy)]
    struct Layout {
        q: usize,
        v: usize,
        c: usize,
        matrix: usize,
        bank: usize,
        conv: usize,
        nq: usize,
        nk: usize,
        rho: usize,
        beta: usize,
        core: usize,
        status: usize,
        frame: usize,
        params: usize,
        input: usize,
    }
    impl Geometry {
        fn layout(self) -> Result<Layout> {
            ensure!(
                (1..=16).contains(&self.hk)
                    && self.hk.is_power_of_two()
                    && (1..=128).contains(&self.d)
                    && self.d.is_power_of_two()
                    && (1..=48).contains(&self.hv)
                    && self.hv % self.hk == 0
                    && self.eps.is_finite()
                    && self.eps > 0.0,
                "fixture geometry outside bound"
            );
            let q = self.hk * self.d;
            let v = self.hv * self.d;
            let c = 2 * q + v;
            let matrix = v * self.d;
            let bank = 3 * c + matrix + v;
            let conv = 2 * bank;
            let nq = conv + c;
            let nk = nq + q;
            let rho = nk + q;
            let beta = rho + self.hv;
            let core = beta + self.hv;
            let status = core + v;
            Ok(Layout {
                q,
                v,
                c,
                matrix,
                bank,
                conv,
                nq,
                nk,
                rho,
                beta,
                core,
                status,
                frame: status + self.hv,
                params: 4 * c + 2 * self.hv + self.d + 1,
                input: c + v + 2 * self.hv,
            })
        }
        fn native(self) -> NativeDeltaShape {
            NativeDeltaShape {
                key_heads: self.hk,
                value_heads: self.hv,
                head_dim: self.d,
                epsilon: self.eps,
            }
        }
    }
    impl Layout {
        fn packet_bytes(self) -> u64 {
            (4 * (self.params + self.input + self.frame) + 24) as u64
        }
        fn output(self) -> usize {
            3 * self.c + self.matrix
        }
    }

    struct Parameters {
        conv: Vec<f32>,
        a: Vec<f32>,
        dt: Vec<f32>,
        norm: Vec<f32>,
    }
    impl Parameters {
        fn native(&self) -> NativeDeltaParameters<'_> {
            NativeDeltaParameters {
                convolution: &self.conv,
                a: &self.a,
                dt_bias: &self.dt,
                norm: &self.norm,
            }
        }
        fn words(&self) -> usize {
            self.conv.len() + self.a.len() + self.dt.len() + self.norm.len()
        }
    }
    struct Snapshot {
        bank: Vec<f32>,
        scratch: Vec<f32>,
    }
    struct Fixture {
        name: &'static str,
        shape: Geometry,
        layout: Layout,
        parameters: Parameters,
        inputs: Vec<Vec<f32>>,
        expected: Vec<Snapshot>,
        tails: bool,
    }
    impl Fixture {
        fn payload_bytes(&self) -> usize {
            4 * (self.parameters.words()
                + self.inputs.iter().map(Vec::len).sum::<usize>()
                + self
                    .expected
                    .iter()
                    .map(|s| s.bank.len() + s.scratch.len())
                    .sum::<usize>())
        }
    }

    // Independent reference has no production DeltaState or native API calls.
    // Raw histories, matrix products, and graph-order boundaries remain explicit.
    struct Reference {
        frame: Vec<f32>,
        active: usize,
    }
    impl Reference {
        fn new(layout: Layout) -> Self {
            Self {
                frame: vec![0.0; layout.frame],
                active: 0,
            }
        }
        fn step(&mut self, case: &Fixture, input: &[f32], until: Instant) -> Result<Snapshot> {
            let g = case.shape;
            let l = case.layout;
            let p = &case.parameters;
            ensure!(
                input.len() == l.input && input.iter().all(|v| v.is_finite()),
                "independent reference invalid projection"
            );
            deadline(until)?;
            let old = self.active * l.bank;
            let next = (1 - self.active) * l.bank;
            for c in 0..l.c {
                let h = old + 3 * c;
                let raw = input[c];
                let a = self.frame[h];
                let b = self.frame[h + 1];
                let c2 = self.frame[h + 2];
                self.frame[next + 3 * c] = b;
                self.frame[next + 3 * c + 1] = c2;
                self.frame[next + 3 * c + 2] = raw;
                let samples = [a, b, c2, raw];
                let mut sum = 0.0f32;
                for t in 0..4 {
                    sum = add(sum, mul(samples[t], p.conv[4 * c + t])?)?;
                }
                self.frame[l.conv + c] = silu(sum)?;
            }
            for branch in 0..2 {
                for h in 0..g.hk {
                    let source = l.conv + branch * l.q + h * g.d;
                    let destination = (if branch == 0 { l.nq } else { l.nk }) + h * g.d;
                    let rms = norm_scale(&self.frame[source..source + g.d], g.eps / g.d as f32)?;
                    let scale = inverse_root(g.d as f32)?;
                    for i in 0..g.d {
                        self.frame[destination + i] =
                            mul(mul(self.frame[source + i], rms)?, scale)?;
                    }
                }
            }
            for h in 0..g.hv {
                let alpha = add(input[l.c + l.v + h], p.dt[h])?;
                let log_decay = mul(p.a[h], softplus(alpha)?)?;
                self.frame[l.rho + h] = finite(log_decay.exp())?;
                self.frame[l.beta + h] = sigmoid(input[l.c + l.v + g.hv + h])?;
            }
            for n in 0..l.v {
                if n % 128 == 0 {
                    deadline(until)?;
                }
                let h = n / g.d;
                let key = h % g.hk;
                let rho = self.frame[l.rho + h];
                let beta = self.frame[l.beta + h];
                let oldrow = old + 3 * l.c + n * g.d;
                let newrow = next + 3 * l.c + n * g.d;
                let mut prediction = 0.0f64;
                for i in 0..g.d {
                    prediction += f64::from(mul(
                        mul(self.frame[oldrow + i], rho)?,
                        self.frame[l.nk + key * g.d + i],
                    )?);
                }
                let delta = mul(
                    add(
                        self.frame[l.conv + 2 * l.q + n],
                        -finite(prediction as f32)?,
                    )?,
                    beta,
                )?;
                let mut output = 0.0f64;
                for i in 0..g.d {
                    let updated = add(
                        mul(self.frame[oldrow + i], rho)?,
                        mul(self.frame[l.nk + key * g.d + i], delta)?,
                    )?;
                    self.frame[newrow + i] = updated;
                    output += f64::from(mul(updated, self.frame[l.nq + key * g.d + i])?);
                }
                self.frame[l.core + n] = mul(finite(output as f32)?, inverse_root(g.d as f32)?)?;
            }
            for h in 0..g.hv {
                let core = l.core + h * g.d;
                let scale = norm_scale(&self.frame[core..core + g.d], g.eps)?;
                for j in 0..g.d {
                    let gate = silu(input[l.c + h * g.d + j])?;
                    self.frame[next + l.output() + h * g.d + j] =
                        mul(mul(mul(self.frame[core + j], scale)?, p.norm[j])?, gate)?;
                }
                self.frame[l.status + h] = f32::from_bits(0);
            }
            self.active = 1 - self.active;
            Ok(Snapshot {
                bank: self.frame[next..next + l.bank].to_vec(),
                scratch: self.frame[l.conv..l.status].to_vec(),
            })
        }
    }

    fn fixture(index: usize) -> Result<Fixture> {
        let (name, shape, tails) = match index {
            0 => (
                "hand-zero",
                Geometry {
                    hk: 1,
                    hv: 1,
                    d: 1,
                    eps: 0.25,
                },
                false,
            ),
            1 => (
                "tiled-small",
                Geometry {
                    hk: 2,
                    hv: 4,
                    d: 2,
                    eps: 1e-6,
                },
                false,
            ),
            2 => (
                "openmythos-shape",
                Geometry {
                    hk: 16,
                    hv: 48,
                    d: 128,
                    eps: 1e-6,
                },
                false,
            ),
            3 => (
                "literal-f32-tails",
                Geometry {
                    hk: 1,
                    hv: 48,
                    d: 1,
                    eps: 1e-6,
                },
                true,
            ),
            _ => bail!("fixture index outside fixed plan"),
        };
        let l = shape.layout()?;
        let mut p = Parameters {
            conv: vec![0.0; 4 * l.c],
            a: vec![0.0; shape.hv],
            dt: vec![0.0; shape.hv],
            norm: vec![0.0; shape.d],
        };
        for c in 0..l.c {
            for (t, value) in [0.125, -0.25, 0.5, 0.75].into_iter().enumerate() {
                p.conv[4 * c + t] = value * if c % 3 == 1 { -1.0 } else { 1.0 };
            }
        }
        for h in 0..shape.hv {
            p.a[h] = if h % 3 == 0 {
                -0.0
            } else {
                -0.125 * (1 + h % 5) as f32
            };
            p.dt[h] = (h as i32 % 7 - 3) as f32 * 0.125;
        }
        for j in 0..shape.d {
            p.norm[j] = 0.5 + (j % 5) as f32 * 0.25;
        }
        let mut inputs = (0..MAIN_STEPS)
            .map(|t| {
                (0..l.input)
                    .map(|i| ((i * 7 + t * 11) % 29) as f32 * 0.0625 - 0.875)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if index == 0 || tails {
            for c in 0..l.c {
                p.conv[4 * c..4 * c + 4].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            }
            p.a.fill(-0.0);
            p.dt.fill(0.0);
            p.norm.fill(1.0);
        }
        if index == 0 {
            for (t, x) in inputs.iter_mut().enumerate() {
                x.copy_from_slice(&[0.0, 0.0, (t + 1) as f32 * 0.25, 1.0, 0.0, 0.0]);
            }
        }
        if tails {
            const VALUES: [f32; 14] = [
                -105.0, -103.0, -100.0, -90.0, -80.0, -48.0, -1.0, -0.0, 0.0, 1.0, 48.0, 80.0,
                103.0, 105.0,
            ];
            for x in &mut inputs {
                x[0] = 0.5;
                x[1] = 0.75;
                for h in 0..shape.hv {
                    let t = VALUES[h % VALUES.len()];
                    x[2 + h] = t;
                    x[l.c + h] = t;
                    x[l.c + l.v + h] = t;
                    x[l.c + l.v + shape.hv + h] = t;
                    p.a[h] = if t < -80.0 { -f32::MAX / 4.0 } else { -0.25 };
                    // Second repetition keeps a normal nonzero core, exposing
                    // final-gate tails rather than masking them with zero state.
                    if (14..28).contains(&h) {
                        x[2 + h] = 1.0;
                        x[l.c + l.v + shape.hv + h] = 0.0;
                    }
                }
                // Finite extremes stay within a valid graph: negative-MAX SiLU
                // is -0; positive-MAX z multiplies an identically zero core.
                x[2 + 46] = -f32::MAX;
                x[l.c + 46] = -f32::MAX;
                x[l.c + 47] = f32::MAX;
                x[l.c + l.v + shape.hv + 47] = -105.0;
            }
        }
        inputs.push(inputs[0].clone());
        inputs.push(inputs[1].clone());
        Ok(Fixture {
            name,
            shape,
            layout: l,
            parameters: p,
            inputs,
            expected: Vec::new(),
            tails,
        })
    }

    #[derive(Default)]
    struct Plan {
        reads: u64,
        writes: u64,
        compared: u64,
        status_bytes: u64,
        status_reads: u64,
        tensor_peak: u64,
        cpu_payload: usize,
    }
    fn plan(cases: &[Fixture]) -> Result<Plan> {
        ensure!(cases.len() == FIXTURES, "fixed fixture count differs");
        let mut out = Plan::default();
        let retained = cases.iter().map(Fixture::payload_bytes).sum::<usize>();
        let mut transient = 0;
        for c in cases {
            let l = c.layout;
            let frame = 4 * l.frame as u64;
            let input = 4 * l.input as u64;
            let params = 4 * l.params as u64;
            let packet = l.packet_bytes();
            let status = STATUS_READS_PER_FIXTURE * 4 * c.shape.hv as u64;
            out.status_bytes += status;
            out.status_reads += STATUS_READS_PER_FIXTURE;
            out.reads += status
                + FRAME_READS_PER_FIXTURE * (frame + 24)
                + OUTPUT_READS_PER_FIXTURE * 4 * l.v as u64;
            out.writes +=
                2 * (packet + params + 24) + PROJECTION_WRITES_PER_FIXTURE * input + frame;
            out.compared += 11 * l.frame as u64 + 7 * l.bank as u64 + 9 * l.v as u64;
            out.tensor_peak = out.tensor_peak.max(2 * packet);
            // One reusable frame, immutable neighbor frame, active-bank snapshot,
            // a temporary projected refusal input, and output readback.
            transient = transient.max(4 * (2 * l.frame + l.bank + l.input + l.v));
        }
        out.cpu_payload = retained + transient + HOST_HEADROOM;
        ensure!(
            out.cpu_payload as u64 <= CPU_PAYLOAD_BOUND,
            "CPU payload forecast exceeds64MiB"
        );
        ensure!(
            out.tensor_peak <= u64::from(config().tensor_capacity_bytes),
            "two packets exceed arena"
        );
        ensure!(
            out.reads == 120_338_528
                && out.writes == 21_886_884
                && out.compared == 24_171_065
                && out.status_bytes == 5_252
                && out.tensor_peak == 13_799_096
                && out.cpu_payload == 57_777_300,
            "independent forecast differs from preregistered fixed fixtures"
        );
        Ok(out)
    }

    fn references(until: Instant) -> Result<(Vec<Fixture>, u64, u64, Plan)> {
        // Forecast before constructing any large reference frame/snapshot vectors.
        let full = Geometry {
            hk: 16,
            hv: 48,
            d: 128,
            eps: 1e-6,
        }
        .layout()?;
        ensure!(
            4 * (REFERENCE_STEPS * (full.bank + full.status - full.conv)
                + 3 * full.frame
                + full.bank
                + REFERENCE_STEPS * full.input)
                + HOST_HEADROOM
                <= CPU_PAYLOAD_BOUND as usize,
            "fixed reference preparation forecast exceeds64MiB"
        );
        let mut cases = Vec::with_capacity(FIXTURES);
        let mut generated = 0;
        let mut checked = 0;
        for index in 0..FIXTURES {
            deadline(until)?;
            let mut case = fixture(index)?;
            let mut state = Reference::new(case.layout);
            let mut snapshots = Vec::with_capacity(REFERENCE_STEPS);
            for input in &case.inputs {
                let snapshot = state.step(&case, input, until)?;
                generated += (snapshot.bank.len() + snapshot.scratch.len()) as u64;
                if index == 0 {
                    for &value in &snapshot.bank[3 * case.layout.c..] {
                        ensure!(value == 0.0, "hand zero-key case changed matrix/output");
                        checked += 1;
                    }
                }
                snapshots.push(snapshot);
            }
            case.expected = snapshots;
            cases.push(case);
        }
        ensure!(sigmoid(-90.0)? > 0.0, "subnormal sigmoid reference lost");
        checked += 1;
        ensure!(silu(-90.0)? < 0.0, "subnormal SiLU reference lost");
        checked += 1;
        ensure!(
            silu(-105.0)?.to_bits() == 0x8000_0000,
            "SiLU exp boundary fused away"
        );
        checked += 1;
        ensure!(
            softplus(-103.0)?.to_bits() == 1,
            "least-subnormal log1p reference lost"
        );
        checked += 1;
        let forecast = plan(&cases)?;
        ensure!(
            generated == 5_911_115 && checked == 18,
            "independent generated/reference-check extents differ from fixed fixtures"
        );
        deadline(until)?;
        Ok((cases, generated, checked, forecast))
    }

    #[derive(Default)]
    struct Receipt {
        physical: u64,
        logical: u64,
        compared: u64,
        generated: u64,
        cpu_checked: u64,
        guards: u64,
        reads: u64,
        writes: u64,
        status_reads: u64,
        status_bytes: u64,
        arithmetic_refusals: u64,
        cancellations: u64,
        resets: u64,
        tail_checks: u64,
        rho_checks: u64,
        max_error: f64,
        allocation: u64,
        expected_allocation: u64,
        budget_current: u64,
        tensor_peak: u64,
        cpu_payload: usize,
    }
    impl Receipt {
        fn print(&self, execution: &str, passed: bool, elapsed: f64, p: &Plan) {
            println!(concat!("{{\"execution\":\"{}\",\"passed\":{},",
                "\"physical_dispatches\":{},\"logical_submissions\":{},",
                "\"compared_values\":{},\"cpu_reference_values\":{},",
                "\"generated_reference_values\":{},\"cpu_checked_values\":{},\"guard_checks\":{},",
                "\"host_read_bytes\":{},\"readback_bytes\":{},\"host_write_bytes\":{},\"status_reads\":{},",
                "\"status_read_bytes\":{},\"arithmetic_refusals\":{},\"cancellations\":{},",
                "\"resets\":{},\"tail_checks\":{},\"rho_checks\":{},\"max_absolute_error\":{},",
                "\"allocation_bytes\":{},\"expected_allocation_bytes\":{},",
                "\"budget_current_bytes\":{},\"budget_after\":{},\"hag_budget_bytes\":{},",
                "\"tensor_peak_bytes\":{},\"cpu_payload_forecast_bytes\":{},",
                "\"cpu_payload_bound_bytes\":{},\"expected_native_read_bytes\":{},",
                "\"expected_native_write_bytes\":{},\"expected_native_compared_values\":{},",
                "\"absolute_tolerance\":{},\"relative_tolerance\":{},",
                "\"tail_exp_max_ulps\":{},\"tail_product_max_ulps\":{},",
                "\"exact_tail_magnitude_bits\":{},",
                "\"rho_minus_103_max_ulps\":{},\"rho_minus_90_max_ulps\":{},",
                "\"rho_minus_103_requires_below_one\":true,\"rho_minus_105_exact_one\":true,",
                "\"elapsed_seconds\":{}}}"),
                // cpu_reference_values retains the historical generated-extent
                // name; generated_reference_values makes that meaning explicit.
                execution,passed,self.physical,self.logical,self.compared,self.generated,
                self.generated,self.cpu_checked,
                self.guards,self.reads,self.reads,self.writes,self.status_reads,self.status_bytes,
                self.arithmetic_refusals,self.cancellations,self.resets,self.tail_checks,self.rho_checks,
                self.max_error,self.allocation,self.expected_allocation,self.budget_current,self.budget_current,
                GPU_BUDGET,self.tensor_peak,self.cpu_payload,CPU_PAYLOAD_BOUND,p.reads,p.writes,
                p.compared,ABS_TOL,REL_TOL,TAIL_EXP_ULPS,TAIL_PRODUCT_ULPS,
                EXACT_TAIL_MAGNITUDE_BITS,RHO_MINUS_103_MAX_ULPS,RHO_MINUS_90_MAX_ULPS,elapsed);
        }
    }

    fn unchanged_residency(a: NativeExecutionStats, b: NativeExecutionStats) -> Result<()> {
        ensure!(
            a.allocation_bytes == b.allocation_bytes
                && a.weight_live_bytes == b.weight_live_bytes
                && a.tensor_live_bytes == b.tensor_live_bytes
                && a.auxiliary_live_bytes == b.auxiliary_live_bytes
                && a.weight_peak_bytes == b.weight_peak_bytes
                && a.tensor_peak_bytes == b.tensor_peak_bytes
                && a.auxiliary_peak_bytes == b.auxiliary_peak_bytes
                && !b.poisoned,
            "step/transfer changed residency or poisoned queue"
        );
        Ok(())
    }

    fn same_bits(actual: &[f32], expected: &[f32], r: &mut Receipt) -> Result<()> {
        ensure!(
            actual.len() == expected.len(),
            "exact snapshot length differs"
        );
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            ensure!(
                a.to_bits() == b.to_bits(),
                "exact snapshot changed at{i}: {a:?}/{b:?}"
            );
        }
        r.compared += actual.len() as u64;
        Ok(())
    }
    fn close(actual: &[f32], expected: &[f32], r: &mut Receipt) -> Result<()> {
        ensure!(actual.len() == expected.len(), "reference length differs");
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            ensure!(
                a.is_finite() && b.is_finite() && error <= ABS_TOL + REL_TOL * f64::from(b).abs(),
                "reference mismatch at{i}: {a:?}/{b:?}, abs error{error}"
            );
            ensure!(
                b == 0.0 || b.abs() >= f32::MIN_POSITIVE || a != 0.0,
                "nonzero subnormal reference flushed at{i}: {a:?}/{b:?}"
            );
            r.max_error = r.max_error.max(error);
        }
        r.compared += actual.len() as u64;
        Ok(())
    }
    fn zeros(actual: &[f32], r: &mut Receipt) -> Result<()> {
        ensure!(
            actual.iter().all(|v| v.to_bits() == 0),
            "reset/unused bank contains nonzero bits"
        );
        r.compared += actual.len() as u64;
        Ok(())
    }
    fn tail(actual: f32, expected: f32, max_ulps: u32, r: &mut Receipt) -> Result<()> {
        ensure!(actual.is_finite() && expected.is_finite(), "nonfinite tail");
        if (expected.to_bits() & 0x7fff_ffff) <= EXACT_TAIL_MAGNITUDE_BITS {
            ensure!(
                actual.to_bits() == expected.to_bits(),
                "canonical small tail bits/sign differ"
            );
        } else {
            ensure!(
                actual != 0.0
                    && actual.is_sign_negative() == expected.is_sign_negative()
                    && actual.to_bits().abs_diff(expected.to_bits()) <= max_ulps,
                "subnormal tail lost or outside{max_ulps}ULPs: {actual:?}/{expected:?}"
            );
        }
        r.tail_checks += 1;
        Ok(())
    }

    fn rho_tail(actual: f32, expected: f32, alpha: f32, r: &mut Receipt) -> Result<()> {
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite strict decay"
        );
        if alpha == -105.0 {
            ensure!(
                expected.to_bits() == 1.0f32.to_bits() && actual.to_bits() == expected.to_bits(),
                "-105 softplus must underflow before A multiplication, giving exact unit decay"
            );
        } else {
            let limit = if alpha == -103.0 {
                ensure!(
                    expected.to_bits() == 0x3f7f_fffe,
                    "-103 independent decay differs from the preregistered two-ULP decrement"
                );
                RHO_MINUS_103_MAX_ULPS
            } else {
                ensure!(alpha == -90.0, "unsupported strict decay fixture");
                RHO_MINUS_90_MAX_ULPS
            };
            ensure!(
                actual > 0.0
                    && actual < 1.0
                    && expected > 0.0
                    && expected < 1.0
                    && actual.to_bits().abs_diff(expected.to_bits()) <= limit,
                "alpha{alpha} decay flushed or outside{limit}ULPs: {actual:?}/{expected:?}"
            );
        }
        r.rho_checks += 1;
        Ok(())
    }

    fn check_layout(case: &Fixture, seq: &NativeDeltaSequence) -> Result<()> {
        let a = seq.layout();
        let l = case.layout;
        let g = case.shape;
        ensure!(
            a.packet_bytes as u64 == l.packet_bytes()
                && a.bank_bytes as usize == 4 * l.bank
                && a.parameter_bytes as usize == 4 * l.params
                && a.projection_bytes as usize == 4 * l.input
                && a.frame_bytes as usize == 4 * l.frame
                && a.recurrent_offset as usize == 12 * l.c
                && a.output_offset as usize == 4 * l.output()
                && a.convolved_offset as usize == 4 * l.conv
                && a.normalized_q_offset as usize == 4 * l.nq
                && a.normalized_k_offset as usize == 4 * l.nk
                && a.decay_offset as usize == 4 * l.rho
                && a.beta_gate_offset as usize == 4 * l.beta
                && a.core_offset as usize == 4 * l.core
                && a.status_offset as usize == 4 * l.status
                && a.epsilon_offset as usize == 4 * (l.params - 1)
                && a.groups == [l.c.div_ceil(64) as u32, 2, l.v.div_ceil(64) as u32, 1]
                && seq.shape() == g.native(),
            "native geometry differs from independent packet"
        );
        Ok(())
    }

    fn upload(
        device: &NativeExecutionDevice,
        seq: &NativeDeltaSequence,
        case: &Fixture,
        input: &[f32],
        r: &mut Receipt,
    ) -> Result<()> {
        let before = device.stats()?;
        let l = case.layout;
        let h = case.shape.hv;
        ensure!(input.len() == l.input, "projection extent differs");
        let views = seq.projection_views()?;
        device.write_f32(&views.qkv, &input[..l.c])?;
        device.write_f32(&views.z, &input[l.c..l.c + l.v])?;
        device.write_f32(&views.alpha, &input[l.c + l.v..l.c + l.v + h])?;
        device.write_f32(&views.beta, &input[l.c + l.v + h..])?;
        drop(views);
        let after = device.stats()?;
        unchanged_residency(before, after)?;
        ensure!(
            after.host_write_bytes == before.host_write_bytes + 4 * l.input as u64
                && after.host_read_bytes == before.host_read_bytes,
            "projection I/O differs"
        );
        r.writes += 4 * l.input as u64;
        Ok(())
    }

    // All control reads happen inside sequence.step but outside device.submit.
    fn step(
        device: &NativeExecutionDevice,
        seq: &mut NativeDeltaSequence,
        until: Instant,
        cancel: Option<usize>,
        refusal: bool,
        r: &mut Receipt,
    ) -> Result<()> {
        deadline(until)?;
        let before = device.stats()?;
        let prior = seq.stats();
        let bank = seq.active_bank();
        let tokens = seq.tokens_seen();
        let mut calls = 0;
        let result = seq.step_controlled(&mut || {
            control_deadline(until)?;
            calls += 1;
            if cancel == Some(calls) {
                Err(ExecutionError::Shape(
                    "intentional probe cancellation".into(),
                ))
            } else {
                Ok(())
            }
        });
        let dispatched = cancel != Some(1);
        let status = dispatched && !matches!(cancel, Some(2) | Some(3));
        if cancel.is_some() {
            let error = result
                .err()
                .context("cancellation unexpectedly committed")?;
            ensure!(
                error.to_string().contains("intentional probe cancellation"),
                "unexpected cancellation error:{error}"
            );
            ensure!(calls == cancel.unwrap(), "control checkpoint count differs");
            r.cancellations += 1;
        } else if refusal {
            let error = result
                .err()
                .context("invalid arithmetic unexpectedly committed")?;
            ensure!(
                error.to_string().contains("refused with status"),
                "unexpected arithmetic error:{error}"
            );
            ensure!(calls == 3, "arithmetic refusal control count differs");
            r.arithmetic_refusals += 1;
        } else {
            result?;
            ensure!(calls == 4, "successful control checkpoint count differs");
        }
        let after = device.stats()?;
        let stats = seq.stats();
        unchanged_residency(before, after)?;
        let dispatches = if dispatched { 4 } else { 0 };
        let logical = u64::from(dispatched);
        let read_bytes = if status {
            4 * seq.shape().value_heads as u64
        } else {
            0
        };
        ensure!(
            after.dispatch_attempts == before.dispatch_attempts + dispatches
                && after.dispatch_count == before.dispatch_count + dispatches
                && after.logical_submissions == before.logical_submissions + logical
                && after.host_read_bytes == before.host_read_bytes + read_bytes
                && after.host_write_bytes == before.host_write_bytes
                && stats.status_reads == prior.status_reads + u64::from(status)
                && stats.status_read_bytes == prior.status_read_bytes + read_bytes,
            "step dispatch/status/I/O accounting differs"
        );
        if cancel.is_some() || refusal {
            ensure!(
                seq.active_bank() == bank && seq.tokens_seen() == tokens,
                "failed step published state"
            );
        } else {
            ensure!(
                seq.active_bank() == 1 - bank && seq.tokens_seen() == tokens + 1,
                "commit metadata differs"
            );
        }
        r.physical += dispatches;
        r.logical += logical;
        r.reads += read_bytes;
        r.status_bytes += read_bytes;
        r.status_reads += u64::from(status);
        deadline(until)
    }

    fn read_frame(
        device: &NativeExecutionDevice,
        seq: &NativeDeltaSequence,
        actual: &mut [f32],
        r: &mut Receipt,
    ) -> Result<()> {
        let before = device.stats()?;
        seq.diagnostic_read_frame(actual)?;
        let guards = seq.diagnostic_read_guards()?;
        ensure!(
            guards
                .iter()
                .all(|v| v.to_bits() == NATIVE_DELTA_GUARD.to_bits()),
            "packet guard changed"
        );
        let after = device.stats()?;
        unchanged_residency(before, after)?;
        let bytes = 4 * actual.len() as u64 + 24;
        ensure!(
            after.host_read_bytes == before.host_read_bytes + bytes
                && after.host_write_bytes == before.host_write_bytes,
            "diagnostic read accounting differs"
        );
        r.reads += bytes;
        r.guards += 6;
        Ok(())
    }

    fn compare_frame(
        device: &NativeExecutionDevice,
        seq: &NativeDeltaSequence,
        case: &Fixture,
        index: usize,
        previous: Option<usize>,
        actual: &mut [f32],
        r: &mut Receipt,
    ) -> Result<()> {
        let l = case.layout;
        let expected = &case.expected[index];
        read_frame(device, seq, actual, r)?;
        let active = seq.active_bank() * l.bank;
        let other = (1 - seq.active_bank()) * l.bank;
        close(&actual[active..active + l.bank], &expected.bank, r)?;
        if let Some(prior) = previous {
            close(
                &actual[other..other + l.bank],
                &case.expected[prior].bank,
                r,
            )?;
        } else {
            zeros(&actual[other..other + l.bank], r)?;
        }
        close(&actual[l.conv..l.status], &expected.scratch, r)?;
        zeros(&actual[l.status..], r)?;
        if case.tails {
            for n in 0..l.c {
                let e = expected.scratch[n];
                if e.abs() < 1e-30 {
                    tail(actual[l.conv + n], e, TAIL_PRODUCT_ULPS, r)?;
                }
            }
            for h in 0..case.shape.hv {
                let offset = l.beta - l.conv + h;
                let e = expected.scratch[offset];
                if e.abs() < 1e-30 {
                    tail(actual[l.beta + h], e, TAIL_EXP_ULPS, r)?;
                }
                let alpha = case.inputs[index][l.c + l.v + h];
                if [-105.0, -103.0, -90.0].contains(&alpha) {
                    ensure!(
                        case.parameters.a[h] == -f32::MAX / 4.0 && case.parameters.dt[h] == 0.0,
                        "strict decay fixture coefficient changed"
                    );
                    rho_tail(
                        actual[l.rho + h],
                        expected.scratch[l.rho - l.conv + h],
                        alpha,
                        r,
                    )?;
                }
            }
            for h in 14..28 {
                let offset = l.output() + h;
                let e = expected.bank[offset];
                if e.abs() < 1e-30 {
                    tail(actual[active + offset], e, TAIL_PRODUCT_ULPS, r)?;
                }
            }
        }
        // Independent published view check, with exact residency and transfer delta.
        let before = device.stats()?;
        let view = seq.output_view()?;
        let mut output = vec![f32::NAN; l.v];
        device.read_f32(&view, &mut output)?;
        drop(view);
        let after = device.stats()?;
        unchanged_residency(before, after)?;
        ensure!(
            after.host_read_bytes == before.host_read_bytes + 4 * l.v as u64
                && after.host_write_bytes == before.host_write_bytes,
            "published output read differs"
        );
        close(&output, &expected.bank[l.output()..], r)?;
        r.reads += 4 * l.v as u64;
        Ok(())
    }

    fn new_sequence(
        device: &NativeExecutionDevice,
        case: &Fixture,
        r: &mut Receipt,
    ) -> Result<NativeDeltaSequence> {
        let before = device.stats()?;
        let seq = NativeDeltaSequence::new(device, case.shape.native(), case.parameters.native())?;
        check_layout(case, &seq)?;
        let after = device.stats()?;
        let bytes = case.layout.packet_bytes() + 4 * case.layout.params as u64 + 24;
        ensure!(
            after.tensor_live_bytes == before.tensor_live_bytes + case.layout.packet_bytes()
                && after.weight_live_bytes == before.weight_live_bytes
                && after.auxiliary_live_bytes == before.auxiliary_live_bytes
                && after.host_write_bytes == before.host_write_bytes + bytes
                && after.host_read_bytes == before.host_read_bytes
                && after.dispatch_count == before.dispatch_count
                && seq.stats().initialization_write_bytes == bytes
                && seq.output_view().is_err()
                && seq.tokens_seen() == 0
                && !seq.reset_required(),
            "sequence initialization changed unexpected resources/publication"
        );
        r.writes += bytes;
        Ok(seq)
    }

    fn native_case(
        device: &NativeExecutionDevice,
        case: &Fixture,
        until: Instant,
        r: &mut Receipt,
    ) -> Result<()> {
        deadline(until)?;
        let l = case.layout;
        let mut neighbor = new_sequence(device, case, r)?;
        let mut main = new_sequence(device, case, r)?;
        ensure!(
            device.stats()?.tensor_live_bytes == 2 * l.packet_bytes(),
            "two-packet residency differs"
        );
        let mut actual = vec![f32::NAN; l.frame];
        upload(device, &neighbor, case, &case.inputs[0], r)?;
        step(device, &mut neighbor, until, None, false, r)?;
        compare_frame(device, &neighbor, case, 0, None, &mut actual, r)?;
        let neighbor_frozen = actual.clone();
        for i in 0..MAIN_STEPS {
            upload(device, &main, case, &case.inputs[i], r)?;
            step(device, &mut main, until, None, false, r)?;
            compare_frame(device, &main, case, i, i.checked_sub(1), &mut actual, r)?;
        }
        let active = main.active_bank() * l.bank;
        let mut preserved = actual[active..active + l.bank].to_vec();
        for kind in 0..3 {
            let mut bad = case.inputs[0].clone();
            match kind {
                0 => bad[l.c] = f32::NAN,
                1 => bad[0] = f32::MAX,
                2 => bad[l.c + l.v + case.shape.hv] = f32::INFINITY,
                _ => unreachable!(),
            }
            upload(device, &main, case, &bad, r)?;
            step(device, &mut main, until, None, true, r)?;
            read_frame(device, &main, &mut actual, r)?;
            same_bits(&actual[active..active + l.bank], &preserved, r)?;
            ensure!(
                actual[l.status..].iter().any(|v| v.to_bits() != 0),
                "refusal lacks failing status"
            );
        }
        upload(device, &main, case, &case.inputs[5], r)?;
        step(device, &mut main, until, None, false, r)?;
        compare_frame(device, &main, case, 5, Some(4), &mut actual, r)?;
        let active = main.active_bank() * l.bank;
        preserved.copy_from_slice(&actual[active..active + l.bank]);
        for checkpoint in 1..=4 {
            upload(device, &main, case, &case.inputs[6], r)?;
            step(device, &mut main, until, Some(checkpoint), false, r)?;
            read_frame(device, &main, &mut actual, r)?;
            same_bits(&actual[active..active + l.bank], &preserved, r)?;
        }
        upload(device, &main, case, &case.inputs[6], r)?;
        step(device, &mut main, until, None, false, r)?;
        compare_frame(device, &main, case, 6, Some(5), &mut actual, r)?;
        let before = device.stats()?;
        let before_seq = main.stats();
        main.reset()?;
        let after = device.stats()?;
        unchanged_residency(before, after)?;
        ensure!(
            main.tokens_seen() == 0
                && main.active_bank() == 0
                && main.output_view().is_err()
                && !main.reset_required()
                && after.host_write_bytes == before.host_write_bytes + 4 * l.frame as u64
                && after.host_read_bytes == before.host_read_bytes
                && after.dispatch_attempts == before.dispatch_attempts
                && after.dispatch_count == before.dispatch_count
                && after.logical_submissions == before.logical_submissions
                && main.stats().reset_write_bytes
                    == before_seq.reset_write_bytes + 4 * l.frame as u64,
            "reset did not clear/block/publicize as specified"
        );
        r.writes += 4 * l.frame as u64;
        r.resets += 1;
        read_frame(device, &main, &mut actual, r)?;
        zeros(&actual, r)?;
        upload(device, &main, case, &case.inputs[0], r)?;
        step(device, &mut main, until, None, false, r)?;
        compare_frame(device, &main, case, 0, None, &mut actual, r)?;
        read_frame(device, &neighbor, &mut actual, r)?;
        same_bits(&actual, &neighbor_frozen, r)?;
        drop(main);
        drop(neighbor);
        let retired = device.stats()?;
        ensure!(
            retired.tensor_live_bytes == 0
                && retired.weight_live_bytes == 0
                && retired.auxiliary_live_bytes == 0
                && !retired.poisoned,
            "sequence resources did not retire"
        );
        deadline(until)
    }

    fn native(
        path: &Path,
        cases: &[Fixture],
        p: &Plan,
        until: Instant,
        r: &mut Receipt,
    ) -> Result<()> {
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must equal0 before device access"
        );
        let allocation =
            config().allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?;
        ensure!(allocation < GPU_BUDGET, "fixed queue charge exceeds16MiB");
        r.expected_allocation = allocation;
        let budget = MemoryBudget::new(GPU_BUDGET);
        let result = (|| -> Result<()> {
            deadline(until)?;
            let device = NativeExecutionDevice::open_with_profile(
                path,
                &budget,
                config(),
                NativeExecutionProfile::Qwen35Probe,
            )?;
            let initial = device.stats()?;
            r.allocation = initial.allocation_bytes;
            r.budget_current = budget.current_bytes();
            ensure!(
                initial
                    == NativeExecutionStats {
                        allocation_bytes: allocation,
                        ..NativeExecutionStats::default()
                    }
                    && budget.current_bytes() == allocation,
                "initial queue differs"
            );
            for case in cases {
                native_case(&device, case, until, r).with_context(|| case.name)?;
            }
            let end = device.stats()?;
            r.tensor_peak = end.tensor_peak_bytes;
            ensure!(
                r.physical == 4 * GPU_STEPS_PER_FIXTURE * FIXTURES as u64
                    && r.logical == GPU_STEPS_PER_FIXTURE * FIXTURES as u64
                    && r.status_reads == p.status_reads
                    && r.status_bytes == p.status_bytes
                    && r.reads == p.reads
                    && r.writes == p.writes
                    && r.compared == p.compared
                    && r.guards == 6 * FRAME_READS_PER_FIXTURE * FIXTURES as u64
                    && r.arithmetic_refusals == 3 * FIXTURES as u64
                    && r.cancellations == 4 * FIXTURES as u64
                    && r.resets == FIXTURES as u64
                    && r.tail_checks == EXPECTED_TAIL_CHECKS
                    && r.rho_checks == EXPECTED_RHO_CHECKS,
                "receipt differs from fixed diagnostic plan"
            );
            ensure!(
                end.dispatch_count == r.physical
                    && end.dispatch_attempts == r.physical
                    && end.logical_submissions == r.logical
                    && end.host_read_bytes == r.reads
                    && end.host_write_bytes == r.writes
                    && end.uploaded_weight_bytes == 0
                    && end.weight_peak_bytes == 0
                    && end.auxiliary_peak_bytes == 0
                    && end.tensor_peak_bytes == p.tensor_peak
                    && end.tensor_live_bytes == 0
                    && end.weight_live_bytes == 0
                    && end.auxiliary_live_bytes == 0
                    && end.allocation_bytes == allocation
                    && !end.poisoned,
                "final adapter accounting differs:{end:?}"
            );
            drop(device);
            Ok(())
        })();
        r.budget_current = budget.current_bytes();
        result?;
        ensure!(budget.current_bytes() == 0, "queue budget did not retire");
        deadline(until)
    }

    pub fn main() -> Result<()> {
        let mode = mode(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
        if matches!(mode, Mode::Help) {
            println!("hag_delta_probe --cpu-reference | --native /dev/dri/renderD...\nSynthetic only;16MiB GPU/64MiB CPU payload bounds;180s cooperative deadline. Native requires external supervision and HAG_TRACE_SUBMISSION=0. No model paths or serving admission.");
            return Ok(());
        }
        let started = Instant::now();
        let until = started + Duration::from_secs(DEADLINE_SECONDS);
        let execution = if matches!(mode, Mode::Cpu) {
            "cpu-delta-independent-reference"
        } else {
            "native-hag-delta-diagnostic"
        };
        let mut r = Receipt::default();
        let mut p = Plan::default();
        let result = (|| -> Result<()> {
            let (cases, generated, checked, forecast) = references(until)?;
            p = forecast;
            r.generated = generated;
            r.cpu_checked = checked;
            r.cpu_payload = p.cpu_payload;
            match mode {
                Mode::Cpu => {
                    r.compared = checked;
                    Ok(())
                }
                Mode::Native(path) => native(&path, &cases, &p, until, &mut r),
                Mode::Help => unreachable!(),
            }
        })();
        r.print(
            execution,
            result.is_ok(),
            started.elapsed().as_secs_f64(),
            &p,
        );
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn explicit_modes_and_fixed_geometry_only() {
            assert!(matches!(
                mode(&["--cpu-reference".into()]).unwrap(),
                Mode::Cpu
            ));
            assert!(mode(&[]).is_err());
            assert!(mode(&["--native".into(), "/tmp/renderD128".into()]).is_err());
            assert!(fixture(FIXTURES).is_err());
            let g = Geometry {
                hk: 16,
                hv: 48,
                d: 128,
                eps: 1e-6,
            };
            let l = g.layout().unwrap();
            assert_eq!(l.packet_bytes(), 6_899_548);
            assert_eq!(l.frame * 4, 6_668_864);
            assert_eq!(g.native().scalars(0).unwrap(), [3048, 1]);
        }
        #[test]
        fn exact_preservation_and_tail_checks_catch_hidden_failure() -> Result<()> {
            let mut r = Receipt::default();
            assert!(same_bits(&[0.0], &[-0.0], &mut r).is_err());
            assert!(tail(0.0, f32::from_bits(1), TAIL_EXP_ULPS, &mut r).is_err());
            assert!(tail(f32::from_bits(2), f32::from_bits(1), TAIL_EXP_ULPS, &mut r).is_err());
            assert!(tail(f32::from_bits(20), f32::from_bits(1), TAIL_EXP_ULPS, &mut r).is_err());
            ensure!(
                silu(-105.0)?.to_bits() == 0x80000000 && softplus(-103.0)?.to_bits() == 1,
                "tail reference boundary differs"
            );
            let below_one = f32::from_bits(0x3f7f_fffe);
            assert!(rho_tail(1.0, below_one, -103.0, &mut r).is_err());
            assert!(rho_tail(f32::from_bits(0x3f7f_ffff), below_one, -103.0, &mut r).is_ok());
            assert!(rho_tail(1.0, 0.9375, -90.0, &mut r).is_err());
            assert!(rho_tail(below_one, 1.0, -105.0, &mut r).is_err());
            rho_tail(1.0, 1.0, -105.0, &mut r)?;
            Ok(())
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::main()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Delta diagnostic requires Linux; no work submitted.");
    std::process::exit(2);
}
