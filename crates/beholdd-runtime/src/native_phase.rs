//! Opt-in, content-free phase receipts for one HAG generation call.
//!
//! `HAG_SERVING_PHASES=1` enables best-effort stderr JSON. At most 128 records
//! are emitted per generation and 4096 per process. A missing end record never
//! certifies completion: errors, termination and either bound can truncate a
//! trace. Sequence IDs are independent of `HAG_SERVING_AUDIT` and HTTP planning.
//! Writes are synchronous and may block on stderr in diagnostic mode. Reset
//! receipts describe host/model-session cache reset, never a device reset.

use std::cell::Cell;
use std::ffi::OsStr;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const REQUEST_RECORD_LIMIT: usize = 128;
const PROCESS_RECORD_LIMIT: usize = 4096;
static RECORDS: AtomicUsize = AtomicUsize::new(0);
static SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    AdmissionStart,
    AdmissionEnd,
    MetadataStart,
    MetadataEnd,
    PreparationStart,
    MemoryAdmissionStart,
    MemoryAdmissionEnd,
    DeviceOpenStart,
    DeviceOpenEnd,
    ResidentPreparationStart,
    ResidentPreparationEnd,
    PreparationEnd,
    InitialResetStart,
    InitialResetOk,
    InitialResetFailed,
    PrefillStart,
    PrefillEnd,
    DecodeStart,
    DecodeEnd,
    PositionFenced,
    FinalResetStart,
    FinalResetOk,
    FinalResetFailed,
    RequestComplete,
    RequestIncomplete,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::AdmissionStart => "admission_start",
            Self::AdmissionEnd => "admission_end",
            Self::MetadataStart => "metadata_start",
            Self::MetadataEnd => "metadata_end",
            Self::PreparationStart => "preparation_start",
            Self::MemoryAdmissionStart => "memory_admission_start",
            Self::MemoryAdmissionEnd => "memory_admission_end",
            Self::DeviceOpenStart => "device_open_start",
            Self::DeviceOpenEnd => "device_open_end",
            Self::ResidentPreparationStart => "resident_preparation_start",
            Self::ResidentPreparationEnd => "resident_preparation_end",
            Self::PreparationEnd => "preparation_end",
            Self::InitialResetStart => "initial_session_reset_start",
            Self::InitialResetOk => "initial_session_reset_ok",
            Self::InitialResetFailed => "initial_session_reset_failed",
            Self::PrefillStart => "prefill_start",
            Self::PrefillEnd => "prefill_end",
            Self::DecodeStart => "decode_start",
            Self::DecodeEnd => "decode_end",
            Self::PositionFenced => "position_fenced",
            Self::FinalResetStart => "final_session_reset_start",
            Self::FinalResetOk => "final_session_reset_ok",
            Self::FinalResetFailed => "final_session_reset_failed",
            Self::RequestComplete => "request_complete",
            Self::RequestIncomplete => "request_incomplete",
        }
    }
}

pub(super) struct Recorder {
    active: Option<(u64, Instant)>,
    records: Cell<usize>,
    completed_positions: Cell<usize>,
    complete: Cell<bool>,
}

fn opted_in(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("1"))
}

fn claim_record(request_records: &Cell<usize>, process_records: &AtomicUsize) -> bool {
    if request_records.get() >= REQUEST_RECORD_LIMIT {
        return false;
    }
    if process_records
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
            if count < PROCESS_RECORD_LIMIT {
                Some(count + 1)
            } else {
                None
            }
        })
        .is_err()
    {
        return false;
    }
    request_records.set(request_records.get() + 1);
    true
}

impl Recorder {
    pub(super) fn new(hag: bool) -> Self {
        let active = (hag && opted_in(std::env::var_os("HAG_SERVING_PHASES").as_deref()))
            .then(|| {
                SEQUENCE
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                        value.checked_add(1)
                    })
                    .ok()
                    .map(|sequence| (sequence, Instant::now()))
            })
            .flatten();
        Self {
            active,
            records: Cell::new(0),
            completed_positions: Cell::new(0),
            complete: Cell::new(false),
        }
    }

    pub(super) fn emit(&self, phase: Phase) {
        let Some((request_sequence, started)) = self.active else {
            return;
        };
        if !claim_record(&self.records, &RECORDS) {
            return;
        }
        let event = Event {
            pid: std::process::id(),
            request_sequence,
            phase,
            // A pre-epoch wall clock is explicit null, never a fabricated epoch.
            unix_time_us: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_micros()),
            elapsed_us: started.elapsed().as_micros(),
            completed_positions: self.completed_positions.get(),
        };
        // Diagnostic write errors must not change generation or cleanup.
        // Fixed enum/numeric fields bound every line; no payload is formatted.
        let _ = event.write_to(&mut io::stderr().lock());
    }

    /// Call only after a synchronous resident step/chunk returns successfully. This does
    /// not assert subsequent accounting checks, logits readback or token delivery.
    pub(super) fn position_fenced(&self, completed_positions: usize) {
        let previous = self.completed_positions.replace(completed_positions);
        if progress_due(previous, completed_positions) {
            self.emit(Phase::PositionFenced);
        }
    }

    pub(super) fn finish(&self) {
        self.complete.set(true);
        self.emit(Phase::RequestComplete);
    }
}

fn progress_due(previous: usize, completed: usize) -> bool {
    completed > previous && (previous == 0 || completed / 64 > previous / 64)
}

pub(super) fn session_reset<E>(
    initial: bool,
    reset: impl FnOnce() -> Result<(), E>,
    mut emit: impl FnMut(Phase),
) -> Result<(), E> {
    emit(if initial {
        Phase::InitialResetStart
    } else {
        Phase::FinalResetStart
    });
    let result = reset();
    emit(match (initial, result.is_ok()) {
        (true, true) => Phase::InitialResetOk,
        (true, false) => Phase::InitialResetFailed,
        (false, true) => Phase::FinalResetOk,
        (false, false) => Phase::FinalResetFailed,
    });
    result
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if !self.complete.get() {
            self.emit(Phase::RequestIncomplete);
        }
    }
}

struct Event {
    pid: u32,
    request_sequence: u64,
    phase: Phase,
    unix_time_us: Option<u128>,
    elapsed_us: u128,
    completed_positions: usize,
}

impl Event {
    fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        write!(writer, "HAG_SERVING_PHASE {{\"schema\":1,\"pid\":{},\"request_sequence\":{},\"phase\":\"{}\",\"unix_time_us\":",
               self.pid, self.request_sequence, self.phase.name())?;
        match self.unix_time_us {
            Some(value) => write!(writer, "{value}")?,
            None => write!(writer, "null")?,
        }
        writeln!(
            writer,
            ",\"elapsed_us\":{},\"completed_positions\":{}}}",
            self.elapsed_us, self.completed_positions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_exact_one_enables_phase_receipts() {
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("01"),
            Some("1 "),
        ] {
            assert!(!opted_in(value.map(OsStr::new)));
        }
        assert!(opted_in(Some(OsStr::new("1"))));
        let disabled = Recorder::new(false);
        assert!(disabled.active.is_none());
        disabled.position_fenced(64);
        disabled.finish();
        assert_eq!(disabled.records.get(), 0);
    }

    #[test]
    fn per_request_and_process_bounds_do_not_wrap_or_reset() {
        let process = AtomicUsize::new(0);
        let request = Cell::new(0);
        for _ in 0..REQUEST_RECORD_LIMIT {
            assert!(claim_record(&request, &process));
        }
        assert!(!claim_record(&request, &process));
        assert_eq!(process.load(Ordering::Relaxed), REQUEST_RECORD_LIMIT);
        for _ in REQUEST_RECORD_LIMIT..PROCESS_RECORD_LIMIT {
            assert!(claim_record(&Cell::new(0), &process));
        }
        assert!(!claim_record(&Cell::new(0), &process));
        assert_eq!(process.load(Ordering::Relaxed), PROCESS_RECORD_LIMIT);
        assert!(!claim_record(&Cell::new(0), &AtomicUsize::new(usize::MAX)));
    }

    #[test]
    fn full_capacity_progress_is_bounded_and_never_logs_zero() {
        let points: Vec<_> = (0usize..=2048)
            .filter(|&count| progress_due(count.saturating_sub(1), count))
            .collect();
        assert_eq!(points.len(), 33);
        assert_eq!(&points[..3], &[1, 64, 128]);
        assert_eq!(points.last(), Some(&2048));
    }

    #[test]
    fn chunk_progress_reports_actual_completed_positions_and_crossed_boundaries() {
        assert!(progress_due(0, 4));
        assert!(!progress_due(4, 8));
        assert!(progress_due(60, 64));
        assert!(progress_due(62, 66));
        assert!(!progress_due(64, 64));
        assert!(!progress_due(64, 0));
        assert!(!progress_due(0, 0));
        let points: Vec<_> = (4usize..=2048)
            .step_by(4)
            .filter(|&count| progress_due(count - 4, count))
            .collect();
        assert_eq!(points.len(), 33);
        assert_eq!(&points[..3], &[4, 64, 128]);
        assert_eq!(points.last(), Some(&2048));
    }

    #[test]
    fn fixed_schema_contains_only_phase_identity_times_and_completed_count() {
        let event = Event {
            pid: 12,
            request_sequence: 7,
            phase: Phase::PositionFenced,
            unix_time_us: Some(123456),
            elapsed_us: 99,
            completed_positions: 64,
        };
        let mut output = Vec::new();
        event.write_to(&mut output).unwrap();
        assert_eq!(String::from_utf8(output).unwrap(),
            "HAG_SERVING_PHASE {\"schema\":1,\"pid\":12,\"request_sequence\":7,\"phase\":\"position_fenced\",\"unix_time_us\":123456,\"elapsed_us\":99,\"completed_positions\":64}\n");
        let mut output = Vec::new();
        Event {
            unix_time_us: None,
            ..event
        }
        .write_to(&mut output)
        .unwrap();
        assert!(String::from_utf8(output)
            .unwrap()
            .contains("\"unix_time_us\":null"));
        struct FailedSink;
        impl Write for FailedSink {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert!(event.write_to(&mut FailedSink).is_err());
    }

    #[test]
    fn session_reset_receipts_preserve_error_and_panic_without_false_success() {
        let mut events = Vec::new();
        assert_eq!(
            session_reset(true, || Ok::<_, &str>(()), |phase| events.push(phase)),
            Ok(())
        );
        assert_eq!(events, [Phase::InitialResetStart, Phase::InitialResetOk]);
        events.clear();
        assert_eq!(
            session_reset(false, || Err("reset refused"), |phase| events.push(phase)),
            Err("reset refused")
        );
        assert_eq!(events, [Phase::FinalResetStart, Phase::FinalResetFailed]);
        events.clear();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), ()> = session_reset(
                false,
                || panic!("reset unwound"),
                |phase| events.push(phase),
            );
        }));
        assert!(unwind.is_err());
        assert_eq!(events, [Phase::FinalResetStart]);
    }
}
