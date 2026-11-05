//! Explicit finite-validation resources and the bounded CPU implementation.
//!
//! Native validation must implement the same capability explicitly using its
//! status-only checker; the generic defaults never read device activations.

use super::*;

/// Fixed payload limits and costs for one backend's finite-validation capability.
///
/// Costs exclude resource/recording metadata, allocator alignment and device-wide
/// program/arena page charges. Cancellation or refusal can incur a prefix of a
/// successful check's transfers. A checker is shared by a serialized model, so its
/// resident storage and initialization are counted once, not once per layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteCheckSpec {
    /// Maximum number of F32 values accepted by one nonempty check.
    pub max_elements: usize,
    /// Additional retained tensor payload owned by one prepared checker.
    pub resident_tensor_bytes: u64,
    /// One-time private packet initialization writes, including zeroing and guards.
    pub initialization_write_bytes: u64,
    /// Maximum temporary host payload used by one check, excluding static tables.
    pub host_scratch_bytes: u64,
    /// Physical GPU dispatches added by one successful check.
    pub physical_dispatches: u64,
    /// Recorded logical submissions added by one successful check.
    pub logical_submissions: u64,
    /// Pending control-status writes required before each successful check.
    pub pending_write_bytes: u64,
    /// Bounded control-status reads required by each successful check.
    pub status_read_bytes: u64,
}

/// An opaque, exclusively used finite validator bound to one execution owner.
///
/// This handle is deliberately not Clone. It exposes no backend status packet or
/// mutable activation view. A successful check applies only to that input snapshot;
/// the handle never stores a reusable assertion that a tensor remains finite.
pub struct FiniteCheck {
    owner: ExecutionOwner,
    payload: Box<dyn Any + Send + Sync>,
}

impl FiniteCheck {
    pub(crate) fn new(owner: ExecutionOwner, payload: impl Any + Send + Sync) -> Self {
        Self {
            owner,
            payload: Box::new(payload),
        }
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.owner
    }

    pub(crate) fn payload_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.payload.as_mut().downcast_mut::<T>()
    }
}

impl fmt::Debug for FiniteCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FiniteCheck").finish_non_exhaustive()
    }
}

pub(super) fn unsupported(device: &'static str) -> ExecutionError {
    DeviceError::Backend {
        device,
        message: "explicit finite validation is unsupported; no activation-read fallback".into(),
    }
    .into()
}

const CPU_MAX_ELEMENTS: usize = 65_536;

pub(super) const CPU_FINITE_SPEC: FiniteCheckSpec = FiniteCheckSpec {
    max_elements: CPU_MAX_ELEMENTS,
    resident_tensor_bytes: 0,
    initialization_write_bytes: 0,
    host_scratch_bytes: (CPU_MAX_ELEMENTS as u64) * 4,
    physical_dispatches: 0,
    logical_submissions: 0,
    pending_write_bytes: 0,
    status_read_bytes: 0,
};

// A zero-sized typed token is enough for CPU ownership and payload validation.
// Native will instead store its private NativeFiniteCheck in the erased handle.
struct CpuFiniteCheck;

impl CpuExecutionDevice {
    pub(super) fn prepare_cpu_finite_check(&self) -> Result<FiniteCheck, ExecutionError> {
        drop(self.queue()?);
        Ok(FiniteCheck::new(self.owner, CpuFiniteCheck))
    }

    // Called with the CPU queue held, but never calls user control. This exact
    // storage check precedes the first callback and is repeated before snapshot
    // allocation. The normal view constructor already proves logical bounds;
    // backend-owned metadata must still agree with the actual resource storage.
    fn validate_cpu_finite_view(&self, input: &F32TensorView) -> Result<(), ExecutionError> {
        let tensor = self.tensor(input.tensor(), "finite-check input")?;
        let values = self.lock_values(tensor, "finite-check input values")?;
        let end = input
            .offset()
            .checked_add(input.len())
            .ok_or_else(|| ExecutionError::Shape("finite-check input extent overflows".into()))?;
        if values.len() != input.tensor().len() || end > values.len() {
            return Err(ExecutionError::Shape(
                "finite-check input metadata differs from its storage".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn check_cpu_finite(
        &self,
        check: &mut FiniteCheck,
        input: &F32TensorView,
        control: &mut dyn FnMut() -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        self.require_owner(check.owner(), "finite checker")?;
        check
            .payload_mut::<CpuFiniteCheck>()
            .ok_or(ExecutionError::IncompatibleResource {
                device: self.name(),
                resource: "finite checker",
            })?;
        self.require_owner(input.owner(), "finite-check input")?;
        if !(1..=CPU_MAX_ELEMENTS).contains(&input.len()) {
            return Err(ExecutionError::Shape(format!(
                "CPU finite check requires 1..={CPU_MAX_ELEMENTS} elements"
            )));
        }
        {
            let _queue = self.queue()?;
            self.validate_cpu_finite_view(input)?;
        }

        control()?;
        let snapshot = {
            let _queue = self.queue()?;
            self.validate_cpu_finite_view(input)?;
            // Exactly one bounded fallible allocation; do not layer generic
            // read_f32 over this helper and accidentally copy the input twice.
            self.snapshot_view(input)?
        };
        control()?;
        if let Some(index) = snapshot.iter().position(|value| !value.is_finite()) {
            return Err(DeviceError::Backend {
                device: self.name(),
                message: format!("finite-check input has a nonfinite value at view index {index}"),
            }
            .into());
        }
        control()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
