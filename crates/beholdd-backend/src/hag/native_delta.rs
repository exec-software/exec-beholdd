//! Private resident state for the explicit, unqualified Qwen35 Delta diagnostic.
//!
//! This file is a child module of `native_execution`, so it can construct checked
//! native batches without exposing raw dispatches or mutable state to callers.
//! Programs 12..15 must already belong to that queue's immutable probe library.
//! No arithmetic is emulated on the host; only bounded status words are read by
//! an ordinary step. Qualification may explicitly request frame/guard copies.

use super::*;

/// Sentinel surrounding each of the three exact packet descriptor extents.
pub const NATIVE_DELTA_GUARD: f32 = -12345.5;
const CHUNK_FLOATS: usize = 16 * 1024;
static RESET_ZEROES: [f32; CHUNK_FLOATS] = [0.0; CHUNK_FLOATS];

fn shape_error(message: impl Into<String>) -> ExecutionError {
    ExecutionError::Shape(message.into())
}
fn require(condition: bool, message: &str) -> Result<(), ExecutionError> {
    if condition {
        Ok(())
    } else {
        Err(shape_error(message))
    }
}
fn add(a: u32, b: u32) -> Result<u32, ExecutionError> {
    a.checked_add(b)
        .ok_or_else(|| shape_error("native Delta layout addition overflows"))
}
fn mul(a: u32, b: u32) -> Result<u32, ExecutionError> {
    a.checked_mul(b)
        .ok_or_else(|| shape_error("native Delta layout multiplication overflows"))
}

/// Admitted four-tap Delta geometry; parameters are the GGUF stored values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeDeltaShape {
    /// Power of two in 1..=16.
    pub key_heads: usize,
    /// 1..=48 and divisible by `key_heads`.
    pub value_heads: usize,
    /// Equal key/value width, a power of two in 1..=128.
    pub head_dim: usize,
    /// Strictly positive finite F32 normalization epsilon.
    pub epsilon: f32,
}

impl NativeDeltaShape {
    /// Check every dimension before computing any packet extent.
    pub fn layout(self) -> Result<NativeDeltaLayout, ExecutionError> {
        require(
            (1..=16).contains(&self.key_heads) && self.key_heads.is_power_of_two(),
            "native Delta key heads must be a power of two in 1..=16",
        )?;
        require(
            (1..=128).contains(&self.head_dim) && self.head_dim.is_power_of_two(),
            "native Delta head width must be a power of two in 1..=128",
        )?;
        require(
            (1..=48).contains(&self.value_heads) && self.value_heads.is_multiple_of(self.key_heads),
            "native Delta value heads must be in 1..=48 and divisible by key heads",
        )?;
        require(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "native Delta epsilon must be positive and finite",
        )?;
        let h = self.key_heads as u32;
        let v = self.value_heads as u32;
        let d = self.head_dim as u32;
        let q = mul(h, d)?;
        let width = mul(v, d)?;
        let channels = add(mul(2, q)?, width)?;
        let recurrent = mul(width, d)?;
        let bank_bytes = mul(4, add(add(mul(3, channels)?, recurrent)?, width)?)?;
        let a = mul(16, channels)?;
        let dt = add(a, mul(4, v)?)?;
        let norm = add(dt, mul(4, v)?)?;
        let epsilon = add(norm, mul(4, d)?)?;
        let parameter_bytes = add(epsilon, 4)?;
        let z = mul(4, channels)?;
        let alpha = add(z, mul(4, width)?)?;
        let beta_input = add(alpha, mul(4, v)?)?;
        let projection_bytes = add(beta_input, mul(4, v)?)?;
        let convolved = mul(2, bank_bytes)?;
        let normalized_q = add(convolved, mul(4, channels)?)?;
        let normalized_k = add(normalized_q, mul(4, q)?)?;
        let decay = add(normalized_k, mul(4, q)?)?;
        let beta_gate = add(decay, mul(4, v)?)?;
        let core = add(beta_gate, mul(4, v)?)?;
        let status = add(core, mul(4, width)?)?;
        let frame_bytes = add(status, mul(4, v)?)?;
        let parameter_offset = 4;
        let p_end = add(parameter_offset, parameter_bytes)?;
        let projection_offset = add(p_end, 8)?;
        let x_end = add(projection_offset, projection_bytes)?;
        let frame_offset = add(x_end, 8)?;
        let f_end = add(frame_offset, frame_bytes)?;
        Ok(NativeDeltaLayout {
            key_width: q,
            value_width: width,
            channels,
            bank_bytes,
            parameter_bytes,
            projection_bytes,
            frame_bytes,
            parameter_offset,
            projection_offset,
            frame_offset,
            packet_bytes: add(f_end, 4)?,
            guard_offsets: [0, p_end, add(p_end, 4)?, x_end, add(x_end, 4)?, f_end],
            a_offset: a,
            dt_offset: dt,
            norm_offset: norm,
            epsilon_offset: epsilon,
            z_offset: z,
            alpha_offset: alpha,
            beta_input_offset: beta_input,
            recurrent_offset: mul(12, channels)?,
            output_offset: add(mul(12, channels)?, mul(4, recurrent)?)?,
            convolved_offset: convolved,
            normalized_q_offset: normalized_q,
            normalized_k_offset: normalized_k,
            decay_offset: decay,
            beta_gate_offset: beta_gate,
            core_offset: core,
            status_offset: status,
            groups: [channels.div_ceil(64), 2, width.div_ceil(64), 1],
        })
    }

    /// Positive v2 transport scalars; epsilon resides in the parameter trailer.
    pub fn scalars(self, active_bank: usize) -> Result<[u32; 2], ExecutionError> {
        self.layout()?;
        require(
            active_bank < 2,
            "native Delta active bank must be zero or one",
        )?;
        let packed = self.head_dim.trailing_zeros()
            | (self.key_heads.trailing_zeros() << 3)
            | (((self.value_heads - 1) as u32) << 6)
            | ((active_bank as u32) << 12);
        Ok([packed + 1, 1])
    }
}

/// Checked byte geometry for proof/forecasting; all offsets are four-byte aligned.
///
/// `parameter_offset`, `projection_offset`, `frame_offset`, and `guard_offsets`
/// are relative to the one private allocation. Other offsets are relative to
/// their named descriptor, except recurrent/output offsets, which are bank-relative.
/// Callers cannot supply a modified layout to a sequence constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeDeltaLayout {
    /// Hk * D F32 coordinates.
    pub key_width: u32,
    /// Hv * D F32 coordinates.
    pub value_width: u32,
    /// 2 * key_width + value_width F32 coordinates.
    pub channels: u32,
    /// One bank's raw convolution history, recurrent matrix and published output.
    pub bank_bytes: u32,
    /// Exact SRD0 extent.
    pub parameter_bytes: u32,
    /// Exact SRD1 extent.
    pub projection_bytes: u32,
    /// Exact SRD2 extent.
    pub frame_bytes: u32,
    /// Start of SRD0 within the allocation.
    pub parameter_offset: u32,
    /// Start of SRD1 within the allocation.
    pub projection_offset: u32,
    /// Start of SRD2 within the allocation.
    pub frame_offset: u32,
    /// Total allocation, including six guard floats.
    pub packet_bytes: u32,
    /// Prefix/suffix sentinel offsets for parameters, projections and frame.
    pub guard_offsets: [u32; 6],
    /// Stored nonpositive A values in SRD0; convolution starts at zero.
    pub a_offset: u32,
    /// dt bias in SRD0.
    pub dt_offset: u32,
    /// Per-coordinate norm in SRD0.
    pub norm_offset: u32,
    /// Positive finite epsilon in SRD0.
    pub epsilon_offset: u32,
    /// z in SRD1; qkv starts at zero.
    pub z_offset: u32,
    /// alpha in SRD1.
    pub alpha_offset: u32,
    /// beta input in SRD1.
    pub beta_input_offset: u32,
    /// Recurrent matrix within either bank; raw history starts at zero.
    pub recurrent_offset: u32,
    /// Output within either bank.
    pub output_offset: u32,
    /// Convolved QKV in SRD2.
    pub convolved_offset: u32,
    /// Normalized Q in SRD2.
    pub normalized_q_offset: u32,
    /// Normalized K in SRD2.
    pub normalized_k_offset: u32,
    /// Decay in SRD2.
    pub decay_offset: u32,
    /// Sigmoid beta in SRD2.
    pub beta_gate_offset: u32,
    /// Core readout in SRD2.
    pub core_offset: u32,
    /// Hv raw U32 status words in SRD2.
    pub status_offset: u32,
    /// Exact workgroup counts for programs 12..15.
    pub groups: [u32; 4],
}

/// Borrowed immutable parameters, validated before allocation or upload.
pub struct NativeDeltaParameters<'a> {
    /// Four oldest-to-current taps per channel, in [channel][tap] order.
    pub convolution: &'a [f32],
    /// GGUF A already stores -exp(A_log); every value must be <= 0.
    pub a: &'a [f32],
    /// One bias per value head.
    pub dt_bias: &'a [f32],
    /// D stored norm weights; no implicit +1 transform.
    pub norm: &'a [f32],
}

impl NativeDeltaParameters<'_> {
    fn validate(
        &self,
        shape: NativeDeltaShape,
        layout: NativeDeltaLayout,
    ) -> Result<(), ExecutionError> {
        require(
            self.convolution.len() == layout.channels as usize * 4
                && self.a.len() == shape.value_heads
                && self.dt_bias.len() == shape.value_heads
                && self.norm.len() == shape.head_dim,
            "native Delta parameter lengths differ from shape",
        )?;
        require(
            self.convolution
                .iter()
                .chain(self.a)
                .chain(self.dt_bias)
                .chain(self.norm)
                .all(|value| value.is_finite()),
            "native Delta parameters must all be finite",
        )?;
        require(
            self.a.iter().all(|&value| value <= 0.0),
            "native Delta stored A must be nonpositive",
        )
    }
}

/// Writable projection subviews for direct streamed matvec destinations.
/// They retain the private allocation, but cannot be widened into state/parameters.
#[derive(Clone, Debug)]
pub struct NativeDeltaProjectionViews {
    /// Concatenated Q, K and V projection.
    pub qkv: F32TensorView,
    /// Output gate projection.
    pub z: F32TensorView,
    /// Alpha projection.
    pub alpha: F32TensorView,
    /// Beta projection.
    pub beta: F32TensorView,
}

/// Sequence-local transfer accounting; queue statistics also count these transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeDeltaStats {
    /// Allocation zeroing, immutable parameter upload, and six guard writes.
    pub initialization_write_bytes: u64,
    /// Successfully completed status reads, including arithmetic refusals/cancellation.
    pub status_reads: u64,
    /// Successful explicit status bytes read outside submission.
    pub status_read_bytes: u64,
    /// Successful frame-zeroing writes, including a partial failed reset.
    pub reset_write_bytes: u64,
}

/// One private two-bank sequence and two retained parity recordings on one queue.
/// Mutable projection/output views require caller serialization with this owner.
pub struct NativeDeltaSequence {
    device: NativeExecutionDevice,
    packet: F32Tensor,
    batches: [RecordedBatch; 2],
    shape: NativeDeltaShape,
    layout: NativeDeltaLayout,
    active_bank: usize,
    tokens_seen: usize,
    published: bool,
    reset_required: bool,
    stats: NativeDeltaStats,
}

impl NativeDeltaSequence {
    /// Validate shape/parameters/library/arena before any device allocation.
    pub fn new(
        device: &NativeExecutionDevice,
        shape: NativeDeltaShape,
        parameters: NativeDeltaParameters<'_>,
    ) -> Result<Self, ExecutionError> {
        Self::create(device, shape, parameters, || kernels::QWEN35_PROGRAM_COUNT)
    }

    // Only tests replace the library count for a fake transport. The public path
    // always obtains it from the immutable queue profile, never caller metadata.
    fn create(
        device: &NativeExecutionDevice,
        shape: NativeDeltaShape,
        parameters: NativeDeltaParameters<'_>,
        program_count: impl FnOnce() -> usize,
    ) -> Result<Self, ExecutionError> {
        let layout = shape.layout()?;
        parameters.validate(shape, layout)?;
        require(
            device.profile() == NativeExecutionProfile::Qwen35Probe,
            "native Delta requires the explicit Qwen35Probe profile",
        )?;
        require(
            program_count() >= 16,
            "native Delta requires immutable programs 12..15",
        )?;
        require(
            layout.packet_bytes <= device.inner.config.tensor_capacity_bytes,
            "native Delta packet exceeds the tensor arena capacity",
        )?;
        {
            let state = device.state()?;
            require(
                state.arenas[TENSORS]
                    .free
                    .iter()
                    .any(|span| span.bytes >= layout.packet_bytes),
                "native Delta packet needs one contiguous free tensor span",
            )?;
        }
        let packet = device.allocate_f32(layout.packet_bytes as usize / 4)?;
        for offset in layout.guard_offsets {
            device.write_f32(&packet.view(offset as usize / 4, 1)?, &[NATIVE_DELTA_GUARD])?;
        }
        for (offset, values) in [
            (0, parameters.convolution),
            (layout.a_offset, parameters.a),
            (layout.dt_offset, parameters.dt_bias),
            (layout.norm_offset, parameters.norm),
            (layout.epsilon_offset, std::slice::from_ref(&shape.epsilon)),
        ] {
            Self::write_chunks(device, &packet, layout.parameter_offset + offset, values)?;
        }
        let packet_view = packet.full_view();
        let (tensor, region) = device.tensor_region(&packet_view, "native Delta packet")?;
        let bindings = [
            BufferRegion {
                slot: TENSORS,
                offset: region.offset + u64::from(layout.parameter_offset),
                length: layout.parameter_bytes,
            },
            BufferRegion {
                slot: TENSORS,
                offset: region.offset + u64::from(layout.projection_offset),
                length: layout.projection_bytes,
            },
            BufferRegion {
                slot: TENSORS,
                offset: region.offset + u64::from(layout.frame_offset),
                length: layout.frame_bytes,
            },
        ];
        let record = |bank| -> Result<RecordedBatch, ExecutionError> {
            let mut commands = Vec::new();
            commands.try_reserve_exact(4).map_err(backend)?;
            for (index, groups) in layout.groups.into_iter().enumerate() {
                commands.push(NativeDispatch {
                    program: 12 + index,
                    bindings,
                    scalars: shape.scalars(bank)?,
                    groups,
                });
            }
            let mut retained = Vec::new();
            retained.try_reserve_exact(1).map_err(backend)?;
            retained.push(Arc::clone(&tensor.allocation));
            Ok(RecordedBatch::new(
                device.inner.owner,
                4,
                Arc::new(NativeBatch {
                    inner: Arc::clone(&device.inner),
                    semantic_count: 4,
                    commands,
                    _retained: retained,
                }),
            ))
        };
        let batches = [record(0)?, record(1)?];
        let stats = NativeDeltaStats {
            initialization_write_bytes: u64::from(layout.packet_bytes)
                + u64::from(layout.parameter_bytes)
                + 24,
            ..NativeDeltaStats::default()
        };
        Ok(Self {
            device: device.clone(),
            packet,
            batches,
            shape,
            layout,
            active_bank: 0,
            tokens_seen: 0,
            published: false,
            reset_required: false,
            stats,
        })
    }

    fn write_chunks(
        device: &NativeExecutionDevice,
        packet: &F32Tensor,
        byte_offset: u32,
        values: &[f32],
    ) -> Result<(), ExecutionError> {
        for (index, chunk) in values.chunks(CHUNK_FLOATS).enumerate() {
            device.write_f32(
                &packet.view(byte_offset as usize / 4 + index * CHUNK_FLOATS, chunk.len())?,
                chunk,
            )?;
        }
        Ok(())
    }

    /// Validated immutable shape.
    pub fn shape(&self) -> NativeDeltaShape {
        self.shape
    }
    /// Exact checked packet extents/offsets for accounting and diagnostics.
    pub fn layout(&self) -> NativeDeltaLayout {
        self.layout
    }
    /// Successfully published steps only.
    pub fn tokens_seen(&self) -> usize {
        self.tokens_seen
    }
    /// Current committed bank, including bank zero before the first step.
    pub fn active_bank(&self) -> usize {
        self.active_bank
    }
    /// Sequence-local transfer counts; explicit diagnostic reads are queue-only counts.
    pub fn stats(&self) -> NativeDeltaStats {
        self.stats
    }
    /// Whether a partially failed reset requires completion before further use.
    pub fn reset_required(&self) -> bool {
        self.reset_required
    }

    /// Only these four mutable projection ranges escape the sequence owner.
    pub fn projection_views(&self) -> Result<NativeDeltaProjectionViews, ExecutionError> {
        drop(self.device.state()?);
        let base = self.layout.projection_offset as usize / 4;
        Ok(NativeDeltaProjectionViews {
            qkv: self.packet.view(base, self.layout.channels as usize)?,
            z: self.packet.view(
                base + self.layout.z_offset as usize / 4,
                self.layout.value_width as usize,
            )?,
            alpha: self.packet.view(
                base + self.layout.alpha_offset as usize / 4,
                self.shape.value_heads,
            )?,
            beta: self.packet.view(
                base + self.layout.beta_input_offset as usize / 4,
                self.shape.value_heads,
            )?,
        })
    }

    /// Current published output only. Treat it as read-only; it is valid only until
    /// the next step/reset. Retaining a view keeps the packet/queue alive, not its
    /// old semantic contents. No output is published before the first success.
    pub fn output_view(&self) -> Result<F32TensorView, ExecutionError> {
        drop(self.device.state()?);
        require(
            self.published && !self.reset_required,
            "native Delta has no valid published output",
        )?;
        self.packet.view(
            (self.layout.frame_offset
                + self.active_bank as u32 * self.layout.bank_bytes
                + self.layout.output_offset) as usize
                / 4,
            self.layout.value_width as usize,
        )
    }

    /// Run four ordered/fenced stages, then explicitly read at most 192 status bytes.
    pub fn step(&mut self) -> Result<(), ExecutionError> {
        self.step_controlled(&mut || Ok(()))
    }

    /// Cancellation before commit preserves the previous bank, token count and
    /// output. The inactive bank may be partly overwritten and is never published.
    /// Arithmetic refusal leaves a healthy queue retryable; transport uncertainty
    /// remains permanently poisoned through the existing device policy.
    pub fn step_controlled(
        &mut self,
        control: &mut dyn FnMut() -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        drop(self.device.state()?);
        require(
            !self.reset_required,
            "native Delta requires a completed reset",
        )?;
        let next = self
            .tokens_seen
            .checked_add(1)
            .ok_or_else(|| shape_error("native Delta token count overflows"))?;
        let reads = self
            .stats
            .status_reads
            .checked_add(1)
            .ok_or_else(|| shape_error("native Delta status counter overflows"))?;
        let read_bytes = self
            .stats
            .status_read_bytes
            .checked_add((self.shape.value_heads * 4) as u64)
            .ok_or_else(|| shape_error("native Delta status byte counter overflows"))?;
        control()?;
        let submission = self.device.submit(&self.batches[self.active_bank])?;
        self.device.wait(&submission)?;
        drop(submission);
        control()?;
        let status = self.packet.view(
            (self.layout.frame_offset + self.layout.status_offset) as usize / 4,
            self.shape.value_heads,
        )?;
        let mut values = [0.0; 48];
        control()?;
        self.device
            .read_f32(&status, &mut values[..self.shape.value_heads])?;
        self.stats.status_reads = reads;
        self.stats.status_read_bytes = read_bytes;
        for (head, value) in values[..self.shape.value_heads].iter().enumerate() {
            if value.to_bits() != 0 {
                return Err(backend(format!(
                    "native Delta head {head} refused with status 0x{:08x}",
                    value.to_bits()
                )));
            }
        }
        control()?;
        self.active_bank = 1 - self.active_bank;
        self.tokens_seen = next;
        self.published = true;
        Ok(())
    }

    /// Zero both complete state banks and scratch in bounded 64-KiB writes.
    /// Metadata changes only after all writes succeed. A partial reset blocks
    /// publication/stepping until reset completes, and cannot recover a poisoned queue.
    pub fn reset(&mut self) -> Result<(), ExecutionError> {
        drop(self.device.state()?);
        self.reset_required = true;
        let floats = self.layout.frame_bytes as usize / 4;
        for offset in (0..floats).step_by(CHUNK_FLOATS) {
            let count = CHUNK_FLOATS.min(floats - offset);
            let after = self
                .stats
                .reset_write_bytes
                .checked_add((count * 4) as u64)
                .ok_or_else(|| shape_error("native Delta reset byte counter overflows"))?;
            self.device.write_f32(
                &self
                    .packet
                    .view(self.layout.frame_offset as usize / 4 + offset, count)?,
                &RESET_ZEROES[..count],
            )?;
            self.stats.reset_write_bytes = after;
        }
        self.active_bank = 0;
        self.tokens_seen = 0;
        self.published = false;
        self.reset_required = false;
        Ok(())
    }

    /// Explicit qualification-only copy of the entire exact frame. The caller
    /// supplies its forecasted host buffer; adapter scratch stays <=64 KiB/read.
    /// Normal stepping never invokes this method or reads activations/state.
    pub fn diagnostic_read_frame(&self, output: &mut [f32]) -> Result<(), ExecutionError> {
        require(
            output.len() == self.layout.frame_bytes as usize / 4,
            "native Delta diagnostic frame destination length differs",
        )?;
        for (index, chunk) in output.chunks_mut(CHUNK_FLOATS).enumerate() {
            self.device.read_f32(
                &self.packet.view(
                    self.layout.frame_offset as usize / 4 + index * CHUNK_FLOATS,
                    chunk.len(),
                )?,
                chunk,
            )?;
        }
        Ok(())
    }

    /// Explicit qualification-only guard copy; compare every word bitwise with
    /// `NATIVE_DELTA_GUARD`. No mutable guard view is exposed.
    pub fn diagnostic_read_guards(&self) -> Result<[f32; 6], ExecutionError> {
        let mut guards = [0.; 6];
        for (index, offset) in self.layout.guard_offsets.into_iter().enumerate() {
            self.device.read_f32(
                &self.packet.view(offset as usize / 4, 1)?,
                &mut guards[index..index + 1],
            )?;
        }
        Ok(guards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockState {
        bytes: Vec<u8>,
        calls: Vec<(usize, [BufferRegion; 3], [u32; 2], u32)>,
        status: Vec<u32>,
        marker: f32,
        read_calls: usize,
        write_calls: usize,
        fail_dispatch: Option<usize>,
        fail_read: bool,
        fail_write: bool,
    }
    struct Mock {
        state: Arc<Mutex<MockState>>,
        layout: NativeDeltaLayout,
        value_heads: usize,
    }
    impl Transport for Mock {
        fn allocation_bytes(&self) -> u64 {
            123456
        }
        fn is_poisoned(&self) -> bool {
            false
        }
        fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
            assert_eq!(slot, TENSORS);
            let mut state = self.state.lock().unwrap();
            state.write_calls += 1;
            if state.fail_write {
                return Err(backend("injected write failure"));
            }
            let start = offset as usize;
            state.bytes[start..start + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
        fn read(
            &mut self,
            slot: usize,
            offset: u64,
            bytes: &mut [u8],
        ) -> Result<(), ExecutionError> {
            assert_eq!(slot, TENSORS);
            let mut state = self.state.lock().unwrap();
            state.read_calls += 1;
            if state.fail_read {
                return Err(backend("injected read failure"));
            }
            bytes.copy_from_slice(&state.bytes[offset as usize..offset as usize + bytes.len()]);
            Ok(())
        }
        fn dispatch(
            &mut self,
            program: usize,
            bindings: [BufferRegion; 3],
            scalars: [u32; 2],
            groups: u32,
        ) -> Result<u64, ExecutionError> {
            let mut state = self.state.lock().unwrap();
            state.calls.push((program, bindings, scalars, groups));
            if state.fail_dispatch == Some(state.calls.len()) {
                return Err(backend("injected dispatch failure"));
            }
            if program == 15 {
                // Inject explicit completion fixtures, never emulate Delta ISA.
                let inactive = 1 - (((scalars[0] - 1) >> 12) & 1);
                let output = bindings[2].offset as usize
                    + (inactive * self.layout.bank_bytes + self.layout.output_offset) as usize;
                let marker = state.marker.to_le_bytes();
                for i in 0..self.layout.value_width as usize {
                    state.bytes[output + i * 4..output + i * 4 + 4].copy_from_slice(&marker);
                }
                let status = bindings[2].offset as usize + self.layout.status_offset as usize;
                for h in 0..self.value_heads {
                    let word = state.status[h].to_le_bytes();
                    state.bytes[status + h * 4..status + h * 4 + 4].copy_from_slice(&word);
                }
            }
            Ok(2)
        }
    }

    fn shape() -> NativeDeltaShape {
        NativeDeltaShape {
            key_heads: 1,
            value_heads: 2,
            head_dim: 2,
            epsilon: 1e-6,
        }
    }
    fn mock(
        shape: NativeDeltaShape,
        profile: NativeExecutionProfile,
        capacity: u32,
    ) -> (NativeExecutionDevice, Arc<Mutex<MockState>>) {
        let layout = shape.layout().unwrap();
        let state = Arc::new(Mutex::new(MockState {
            bytes: vec![0; capacity as usize],
            status: vec![0; shape.value_heads],
            marker: 7.,
            ..MockState::default()
        }));
        let device = NativeExecutionDevice::from_transport_with_profile(
            NativeExecutionConfig {
                weight_capacity_bytes: 4,
                tensor_capacity_bytes: capacity,
                auxiliary_capacity_bytes: 4,
            },
            profile,
            Box::new(Mock {
                state: state.clone(),
                layout,
                value_heads: shape.value_heads,
            }),
        )
        .unwrap();
        (device, state)
    }
    fn sequence(device: &NativeExecutionDevice, shape: NativeDeltaShape) -> NativeDeltaSequence {
        let conv = vec![0.; shape.layout().unwrap().channels as usize * 4];
        let a = vec![-1.; shape.value_heads];
        let dt = vec![0.; shape.value_heads];
        let norm = vec![1.; shape.head_dim];
        NativeDeltaSequence::create(
            device,
            shape,
            NativeDeltaParameters {
                convolution: &conv,
                a: &a,
                dt_bias: &dt,
                norm: &norm,
            },
            || 16,
        )
        .unwrap()
    }

    #[test]
    fn exact_real_layout_and_positive_scalars_match_v2_contract() {
        let s = NativeDeltaShape {
            key_heads: 16,
            value_heads: 48,
            head_dim: 128,
            epsilon: 1e-6,
        };
        let l = s.layout().unwrap();
        assert_eq!(
            (l.key_width, l.value_width, l.channels, l.bank_bytes),
            (2048, 6144, 10240, 3293184)
        );
        assert_eq!(
            (
                l.a_offset,
                l.dt_offset,
                l.norm_offset,
                l.epsilon_offset,
                l.parameter_bytes
            ),
            (163840, 164032, 164224, 164736, 164740)
        );
        assert_eq!(
            (
                l.z_offset,
                l.alpha_offset,
                l.beta_input_offset,
                l.projection_bytes
            ),
            (40960, 65536, 65728, 65920)
        );
        assert_eq!(
            (
                l.recurrent_offset,
                l.output_offset,
                l.convolved_offset,
                l.normalized_q_offset,
                l.normalized_k_offset
            ),
            (122880, 3268608, 6586368, 6627328, 6635520)
        );
        assert_eq!(
            (
                l.decay_offset,
                l.beta_gate_offset,
                l.core_offset,
                l.status_offset,
                l.frame_bytes
            ),
            (6643712, 6643904, 6644096, 6668672, 6668864)
        );
        assert_eq!(l.packet_bytes, 6899524 + 24);
        assert_eq!(l.groups, [160, 2, 96, 1]);
        assert_eq!(s.scalars(0).unwrap(), [3048, 1]);
        assert_eq!(s.scalars(1).unwrap(), [7144, 1]);
        for bad in [
            NativeDeltaShape { key_heads: 0, ..s },
            NativeDeltaShape { key_heads: 3, ..s },
            NativeDeltaShape {
                value_heads: 47,
                ..s
            },
            NativeDeltaShape { head_dim: 129, ..s },
            NativeDeltaShape { epsilon: -0.0, ..s },
            NativeDeltaShape {
                epsilon: f32::NAN,
                ..s
            },
        ] {
            assert!(bad.layout().is_err());
        }
        assert!(s.scalars(2).is_err());
    }

    #[test]
    fn admission_refuses_before_allocation_and_packet_uses_exact_disjoint_extents() {
        let s = shape();
        let l = s.layout().unwrap();
        let (device, state) = mock(s, NativeExecutionProfile::Qwen35Probe, l.packet_bytes + 16);
        let conv = vec![0.; l.channels as usize * 4];
        let dt = [0.; 2];
        let norm = [1.; 2];
        for a in [[1., -1.], [f32::NAN, -1.]] {
            assert!(NativeDeltaSequence::create(
                &device,
                s,
                NativeDeltaParameters {
                    convolution: &conv,
                    a: &a,
                    dt_bias: &dt,
                    norm: &norm,
                },
                || panic!("invalid parameters must refuse before library query")
            )
            .is_err());
        }
        assert!(NativeDeltaSequence::create(
            &device,
            s,
            NativeDeltaParameters {
                convolution: &conv,
                a: &[-1.; 2],
                dt_bias: &dt,
                norm: &norm,
            },
            || 15
        )
        .is_err());
        assert_eq!(state.lock().unwrap().write_calls, 0);
        let (dense, _) = mock(s, NativeExecutionProfile::DenseQwen3, l.packet_bytes);
        assert!(NativeDeltaSequence::create(
            &dense,
            s,
            NativeDeltaParameters {
                convolution: &conv,
                a: &[-1.; 2],
                dt_bias: &dt,
                norm: &norm,
            },
            || 16
        )
        .is_err());
        let (small, _) = mock(s, NativeExecutionProfile::Qwen35Probe, l.packet_bytes - 4);
        assert!(NativeDeltaSequence::create(
            &small,
            s,
            NativeDeltaParameters {
                convolution: &conv,
                a: &[-1.; 2],
                dt_bias: &dt,
                norm: &norm,
            },
            || 16
        )
        .is_err());
        let prefix = device.allocate_f32(4).unwrap();
        let before = device.stats().unwrap();
        let mut owner = sequence(&device, s);
        assert_eq!(
            owner.diagnostic_read_guards().unwrap().map(f32::to_bits),
            [NATIVE_DELTA_GUARD.to_bits(); 6]
        );
        assert_eq!(
            device.stats().unwrap().tensor_live_bytes,
            u64::from(l.packet_bytes) + 16
        );
        assert_eq!(
            owner.stats().initialization_write_bytes,
            device.stats().unwrap().host_write_bytes - before.host_write_bytes
        );
        assert!(owner.output_view().is_err());
        let projections = owner.projection_views().unwrap();
        assert_eq!(
            (
                projections.qkv.len(),
                projections.z.len(),
                projections.alpha.len(),
                projections.beta.len()
            ),
            (8, 4, 2, 2)
        );
        assert!(!projections.qkv.overlaps(&projections.z));
        let before_step = device.stats().unwrap();
        owner.step().unwrap();
        let after_step = device.stats().unwrap();
        assert_eq!(after_step.host_read_bytes - before_step.host_read_bytes, 8);
        assert_eq!(after_step.host_write_bytes, before_step.host_write_bytes);
        assert_eq!(after_step.dispatch_count - before_step.dispatch_count, 4);
        assert_eq!(
            after_step.logical_submissions - before_step.logical_submissions,
            1
        );
        let state = state.lock().unwrap();
        assert_eq!(
            state.calls.iter().map(|c| c.0).collect::<Vec<_>>(),
            vec![12, 13, 14, 15]
        );
        for (_, bindings, scalars, _) in &state.calls {
            assert_eq!(*scalars, s.scalars(0).unwrap());
            assert_eq!(
                bindings.map(|b| b.length),
                [l.parameter_bytes, l.projection_bytes, l.frame_bytes]
            );
            assert_eq!(
                bindings.map(|b| b.offset),
                [
                    16 + u64::from(l.parameter_offset),
                    16 + u64::from(l.projection_offset),
                    16 + u64::from(l.frame_offset)
                ]
            );
        }
        drop(state);
        drop(owner);
        assert_eq!(
            device.stats().unwrap().tensor_live_bytes,
            u64::from(l.packet_bytes) + 16
        );
        drop(projections);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 16);
        drop(prefix);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    }

    #[test]
    fn status_refusal_and_every_control_boundary_preserve_publication() {
        let s = shape();
        let l = s.layout().unwrap();
        let (device, state) = mock(s, NativeExecutionProfile::Qwen35Probe, l.packet_bytes);
        let mut owner = sequence(&device, s);
        owner.step().unwrap();
        let mut out = [0.; 4];
        device
            .read_f32(&owner.output_view().unwrap(), &mut out)
            .unwrap();
        assert_eq!(out, [7.; 4]);
        state.lock().unwrap().marker = 9.;
        for status in [1, 0xffffffff, 0x80000000, 2] {
            state.lock().unwrap().status[1] = status;
            assert!(owner.step().is_err());
            assert_eq!((owner.active_bank(), owner.tokens_seen()), (1, 1));
            device
                .read_f32(&owner.output_view().unwrap(), &mut out)
                .unwrap();
            assert_eq!(out, [7.; 4]);
            assert!(!device.stats().unwrap().poisoned);
        }
        state.lock().unwrap().status[1] = 0;
        for stop in 1..=4 {
            let mut calls = 0;
            assert!(owner
                .step_controlled(&mut || {
                    calls += 1;
                    if calls == stop {
                        Err(backend("cancelled"))
                    } else {
                        Ok(())
                    }
                })
                .is_err());
            assert_eq!((owner.active_bank(), owner.tokens_seen()), (1, 1));
            device
                .read_f32(&owner.output_view().unwrap(), &mut out)
                .unwrap();
            assert_eq!(out, [7.; 4]);
        }
        owner.step().unwrap();
        assert_eq!((owner.active_bank(), owner.tokens_seen()), (0, 2));
        device
            .read_f32(&owner.output_view().unwrap(), &mut out)
            .unwrap();
        assert_eq!(out, [9.; 4]);
        assert_eq!(owner.stats().status_reads, 7);
        assert_eq!(owner.stats().status_read_bytes, 56);
    }

    #[test]
    fn reset_zeros_both_banks_and_guards_are_explicit_diagnostics() {
        let s = shape();
        let l = s.layout().unwrap();
        let (device, state) = mock(s, NativeExecutionProfile::Qwen35Probe, l.packet_bytes);
        let mut owner = sequence(&device, s);
        owner.step().unwrap();
        owner.reset().unwrap();
        assert_eq!(
            (
                owner.active_bank(),
                owner.tokens_seen(),
                owner.reset_required()
            ),
            (0, 0, false)
        );
        assert!(owner.output_view().is_err());
        let mut frame = vec![1.; l.frame_bytes as usize / 4];
        owner.diagnostic_read_frame(&mut frame).unwrap();
        assert!(frame.iter().all(|&x| x == 0.));
        assert_eq!(owner.stats().reset_write_bytes, u64::from(l.frame_bytes));
        let mut state = state.lock().unwrap();
        state.bytes[l.guard_offsets[3] as usize..l.guard_offsets[3] as usize + 4]
            .copy_from_slice(&0f32.to_le_bytes());
        drop(state);
        let guards = owner.diagnostic_read_guards().unwrap();
        assert_eq!(guards[3], 0.);
        assert_eq!(guards[0], NATIVE_DELTA_GUARD);
    }

    #[test]
    fn transport_failures_poison_permanently_and_never_publish() {
        let s = shape();
        let l = s.layout().unwrap();
        for failure in 0..3 {
            let (device, state) = mock(s, NativeExecutionProfile::Qwen35Probe, l.packet_bytes);
            let mut owner = sequence(&device, s);
            match failure {
                0 => state.lock().unwrap().fail_dispatch = Some(2),
                1 => state.lock().unwrap().fail_read = true,
                _ => state.lock().unwrap().fail_write = true,
            }
            if failure == 2 {
                assert!(owner.reset().is_err());
                assert!(owner.reset_required());
            } else {
                assert!(owner.step().is_err());
            }
            assert_eq!((owner.active_bank(), owner.tokens_seen()), (0, 0));
            assert!(device.stats().unwrap().poisoned);
            let before = device.stats().unwrap();
            assert!(owner.step().is_err());
            assert!(owner.reset().is_err());
            assert!(owner.output_view().is_err());
            assert_eq!(device.stats().unwrap(), before);
            drop(owner);
            assert_eq!(
                device.stats().unwrap().tensor_live_bytes,
                u64::from(l.packet_bytes)
            );
        }
    }
}
