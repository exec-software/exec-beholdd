// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! Shared, checked admission accounting for HAG-owned allocations.
//!
//! A reservation represents a charge against a budget, not a physical allocation.
//! Allocation owners must retain it until the charged storage has actually been
//! released, including any deferred retirement after device work completes.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// An allocation could not be admitted without exceeding an accounting boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetError {
    /// The requested charge would exceed the configured hard limit.
    LimitExceeded {
        /// Maximum number of admitted bytes.
        limit_bytes: u64,
        /// Admitted bytes observed when the request was refused.
        current_bytes: u64,
        /// Additional bytes requested by the caller.
        requested_bytes: u64,
    },
    /// The charge cannot be represented by the shared accounting counter.
    AccountingOverflow {
        /// Admitted bytes observed when the request was refused.
        current_bytes: u64,
        /// Additional bytes requested by the caller.
        requested_bytes: u64,
    },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded {
                limit_bytes,
                current_bytes,
                requested_bytes,
            } => write!(
                formatter,
                "HAG memory budget refused {requested_bytes} bytes: {current_bytes} bytes already reserved against a {limit_bytes}-byte limit"
            ),
            Self::AccountingOverflow {
                current_bytes,
                requested_bytes,
            } => write!(
                formatter,
                "HAG memory accounting overflow: {current_bytes} reserved bytes plus {requested_bytes} requested bytes exceeds u64::MAX"
            ),
        }
    }
}

impl std::error::Error for BudgetError {}

#[derive(Debug)]
struct BudgetState {
    limit_bytes: Option<u64>,
    current_bytes: AtomicU64,
}

/// A shared hard limit for outstanding allocation reservations.
///
/// Clones share both the limit and the accounting counter. A zero-byte limit is a
/// real zero budget; use [`Self::unlimited`] to omit the configured limit. Admission
/// remains overflow-checked even for unlimited budgets.
///
/// The counter tracks ownership only. It does not synchronize host or device access
/// to the storage charged against it.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    state: Arc<BudgetState>,
}

impl MemoryBudget {
    /// Create a budget whose outstanding reservations cannot exceed `limit_bytes`.
    pub fn new(limit_bytes: u64) -> Self {
        Self::with_limit(Some(limit_bytes))
    }

    /// Create a budget without a configured limit, but with checked accounting.
    pub fn unlimited() -> Self {
        Self::with_limit(None)
    }

    fn with_limit(limit_bytes: Option<u64>) -> Self {
        Self {
            state: Arc::new(BudgetState {
                limit_bytes,
                current_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// Atomically reserve additional bytes until the returned owner is dropped.
    ///
    /// Failure leaves accounting unchanged. Reserving zero bytes always succeeds,
    /// including against a zero or exhausted budget. Callers must charge the actual
    /// allocation size, including alignment or other allocator overhead, and keep the
    /// reservation alive until that allocation has physically been released.
    pub fn reserve(&self, bytes: u64) -> Result<Reservation, BudgetError> {
        let mut current_bytes = self.state.current_bytes.load(Ordering::Acquire);
        loop {
            let next_bytes =
                current_bytes
                    .checked_add(bytes)
                    .ok_or(BudgetError::AccountingOverflow {
                        current_bytes,
                        requested_bytes: bytes,
                    })?;
            if let Some(limit_bytes) = self.state.limit_bytes {
                if next_bytes > limit_bytes {
                    return Err(BudgetError::LimitExceeded {
                        limit_bytes,
                        current_bytes,
                        requested_bytes: bytes,
                    });
                }
            }
            match self.state.current_bytes.compare_exchange_weak(
                current_bytes,
                next_bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        state: Arc::clone(&self.state),
                        bytes,
                    });
                }
                Err(observed_bytes) => current_bytes = observed_bytes,
            }
        }
    }

    /// Return the current outstanding charge as a point-in-time observation.
    pub fn current_bytes(&self) -> u64 {
        self.state.current_bytes.load(Ordering::Acquire)
    }

    /// Return the maximum additional charge that fits at the observed instant.
    ///
    /// Without a configured limit this is `u64::MAX - current_bytes`, because
    /// accounting must remain representable. Another thread can reserve or release
    /// bytes immediately after this observation; only [`Self::reserve`] admits work.
    pub fn remaining(&self) -> u64 {
        self.state.limit_bytes.unwrap_or(u64::MAX) - self.current_bytes()
    }

    /// Return the configured hard limit, or `None` for an unlimited budget.
    pub fn limit_bytes(&self) -> Option<u64> {
        self.state.limit_bytes
    }
}

/// Unique ownership of a charge against a [`MemoryBudget`].
///
/// Moving this value transfers ownership; it cannot be cloned or manually released.
/// Dropping it returns the charge exactly once. Keep it until physical allocation
/// cleanup completes, rather than merely until a public buffer handle is dropped.
#[must_use = "retain the reservation until the charged allocation has been released"]
#[derive(Debug)]
pub struct Reservation {
    state: Arc<BudgetState>,
    bytes: u64,
}

impl Reservation {
    /// Number of bytes owned by this reservation.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.bytes != 0 {
            let previous_bytes = self
                .state
                .current_bytes
                .fetch_sub(self.bytes, Ordering::AcqRel);
            debug_assert!(previous_bytes >= self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BudgetError, MemoryBudget, Reservation};
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn budget_and_reservations_can_cross_threads() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<MemoryBudget>();
        require_send_sync::<Reservation>();
    }

    #[test]
    fn zero_is_a_real_budget_and_zero_reservations_are_valid() {
        let budget = MemoryBudget::new(0);
        assert_eq!(budget.limit_bytes(), Some(0));
        let reservation = budget.reserve(0).unwrap();
        assert_eq!(reservation.bytes(), 0);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(budget.remaining(), 0);
        assert_eq!(
            budget.reserve(1).unwrap_err(),
            BudgetError::LimitExceeded {
                limit_bytes: 0,
                current_bytes: 0,
                requested_bytes: 1,
            }
        );
        drop(reservation);
        assert_eq!(budget.current_bytes(), 0);
    }

    #[test]
    fn clones_share_charges_and_dropping_returns_each_charge_once() {
        let budget = MemoryBudget::new(10);
        let cloned = budget.clone();
        let first = budget.reserve(4).unwrap();
        let second = cloned.reserve(6).unwrap();
        assert_eq!(first.bytes(), 4);
        assert_eq!(second.bytes(), 6);
        assert_eq!(budget.current_bytes(), 10);
        assert_eq!(cloned.remaining(), 0);
        let zero = cloned.reserve(0).unwrap();
        assert!(budget.reserve(1).is_err());
        assert_eq!(budget.current_bytes(), 10);
        drop(first);
        assert_eq!(cloned.current_bytes(), 6);
        assert_eq!(budget.remaining(), 4);
        drop(zero);
        assert_eq!(budget.current_bytes(), 6);
        drop(second);
        assert_eq!(cloned.current_bytes(), 0);
        assert_eq!(budget.remaining(), 10);
    }

    #[test]
    fn unlimited_accounting_still_refuses_overflow() {
        let budget = MemoryBudget::unlimited();
        assert_eq!(budget.limit_bytes(), None);
        assert_eq!(budget.remaining(), u64::MAX);
        let all = budget.reserve(u64::MAX).unwrap();
        assert_eq!(budget.current_bytes(), u64::MAX);
        assert_eq!(budget.remaining(), 0);
        let zero = budget.reserve(0).unwrap();
        assert_eq!(
            budget.reserve(1).unwrap_err(),
            BudgetError::AccountingOverflow {
                current_bytes: u64::MAX,
                requested_bytes: 1,
            }
        );
        assert_eq!(budget.current_bytes(), u64::MAX);
        drop(zero);
        drop(all);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(budget.remaining(), u64::MAX);
    }

    #[test]
    fn maximum_configured_budget_cannot_wrap_either() {
        let budget = MemoryBudget::new(u64::MAX);
        let reserved = budget.reserve(u64::MAX - 1).unwrap();
        assert!(matches!(
            budget.reserve(2),
            Err(BudgetError::AccountingOverflow { .. })
        ));
        assert_eq!(budget.current_bytes(), u64::MAX - 1);
        assert_eq!(budget.remaining(), 1);
        drop(reserved);
        assert_eq!(budget.current_bytes(), 0);
    }

    #[test]
    fn failed_allocation_releases_its_local_reservation() {
        fn attempt_allocation(budget: &MemoryBudget) -> Result<(), &'static str> {
            let _reservation = budget.reserve(7).map_err(|_| "budget rejected")?;
            Err("allocation failed")
        }

        let budget = MemoryBudget::new(10);
        let existing = budget.reserve(3).unwrap();
        assert_eq!(attempt_allocation(&budget), Err("allocation failed"));
        assert_eq!(budget.current_bytes(), 3);
        assert_eq!(budget.remaining(), 7);
        drop(existing);
        assert_eq!(budget.current_bytes(), 0);
    }

    #[test]
    fn reservation_keeps_accounting_alive_without_original_budget() {
        let budget = MemoryBudget::new(8);
        let observer = budget.clone();
        let reservation = budget.reserve(8).unwrap();
        drop(budget);
        assert_eq!(observer.current_bytes(), 8);
        thread::spawn(move || drop(reservation)).join().unwrap();
        assert_eq!(observer.current_bytes(), 0);
    }

    #[test]
    fn concurrent_admission_does_not_exceed_the_limit() {
        const WORKERS: usize = 8;
        let budget = MemoryBudget::new(3);
        let start = Arc::new(Barrier::new(WORKERS + 1));
        let admitted = Arc::new(Barrier::new(WORKERS + 1));
        let release = Arc::new(Barrier::new(WORKERS + 1));
        let mut workers = Vec::new();
        for _ in 0..WORKERS {
            let worker_budget = budget.clone();
            let worker_start = Arc::clone(&start);
            let worker_admitted = Arc::clone(&admitted);
            let worker_release = Arc::clone(&release);
            workers.push(thread::spawn(move || {
                worker_start.wait();
                let reservation = worker_budget.reserve(1);
                let succeeded = reservation.is_ok();
                worker_admitted.wait();
                worker_release.wait();
                drop(reservation);
                succeeded
            }));
        }
        start.wait();
        admitted.wait();
        let admitted_bytes = budget.current_bytes();
        let remaining_bytes = budget.remaining();
        release.wait();
        let successful = workers
            .into_iter()
            .map(|worker| usize::from(worker.join().unwrap()))
            .sum::<usize>();
        assert_eq!(admitted_bytes, 3);
        assert_eq!(remaining_bytes, 0);
        assert_eq!(successful, 3);
        assert_eq!(budget.current_bytes(), 0);
        let all = budget.reserve(3).unwrap();
        drop(all);
        assert_eq!(budget.remaining(), 3);
    }

    #[test]
    fn refusal_errors_are_actionable() {
        let limited = MemoryBudget::new(2).reserve(3).unwrap_err();
        assert!(limited.to_string().contains("3 bytes"));
        assert!(limited.to_string().contains("2-byte limit"));
        let budget = MemoryBudget::unlimited();
        let _reservation = budget.reserve(u64::MAX).unwrap();
        let overflow = budget.reserve(1).unwrap_err();
        assert!(overflow.to_string().contains("accounting overflow"));
        assert!(overflow.to_string().contains("u64::MAX"));
    }
}
