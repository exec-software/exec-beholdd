//! Pure memory-fit forecasting and policy decisions.
//!
//! A [`FitEstimate`] describes the two exact execution footprints the memory layer can
//! offer: keeping every weight resident, or retaining only the minimum simultaneous
//! working set needed to stream the same checkpoint bytes. [`FitEstimate::decide`] then
//! applies [`FitPolicy`] without probing, allocating, or changing model semantics.

use std::error::Error;
use std::fmt;

use crate::FitPolicy;

/// A checked forecast of the memory needed by the two exact execution paths.
///
/// Both values are totals: callers should include every forecast allocation relevant to
/// the run, such as weights, activations, the KV cache, and backend working memory. The
/// exact-streaming minimum excludes optional weight-cache capacity; the planner adds a
/// bounded cache allowance when it selects streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitEstimate {
    fully_resident_bytes: u64,
    exact_streaming_minimum_bytes: u64,
}

impl FitEstimate {
    /// Build an estimate from its two total footprints.
    ///
    /// Exact streaming must not require more memory than keeping the same checkpoint
    /// fully resident. Treating an estimate that violates that invariant as valid would
    /// make the planner's three fit regimes overlap in surprising ways.
    pub fn new(
        fully_resident_bytes: u64,
        exact_streaming_minimum_bytes: u64,
    ) -> Result<Self, FitEstimateError> {
        if exact_streaming_minimum_bytes > fully_resident_bytes {
            return Err(FitEstimateError::StreamingExceedsResident {
                fully_resident_bytes,
                exact_streaming_minimum_bytes,
            });
        }
        Ok(Self {
            fully_resident_bytes,
            exact_streaming_minimum_bytes,
        })
    }

    /// Build both total footprints from memory common to both paths and their weight
    /// footprints.
    ///
    /// `always_resident_bytes` normally includes activations, KV cache, and backend
    /// working memory. `resident_weight_bytes` is the complete encoded weight payload;
    /// `streaming_working_set_bytes` is the largest exact weight set that must coexist.
    /// Both additions are checked before the ordering invariant is validated.
    pub fn from_parts(
        always_resident_bytes: u64,
        resident_weight_bytes: u64,
        streaming_working_set_bytes: u64,
    ) -> Result<Self, FitEstimateError> {
        let fully_resident_bytes = always_resident_bytes
            .checked_add(resident_weight_bytes)
            .ok_or(FitEstimateError::FullyResidentOverflow {
                always_resident_bytes,
                resident_weight_bytes,
            })?;
        let exact_streaming_minimum_bytes = always_resident_bytes
            .checked_add(streaming_working_set_bytes)
            .ok_or(FitEstimateError::StreamingMinimumOverflow {
                always_resident_bytes,
                streaming_working_set_bytes,
            })?;
        Self::new(fully_resident_bytes, exact_streaming_minimum_bytes)
    }

    /// Forecast bytes needed when all weights remain resident.
    pub const fn fully_resident_bytes(self) -> u64 {
        self.fully_resident_bytes
    }

    /// Minimum forecast bytes needed to execute exactly by streaming weights.
    pub const fn exact_streaming_minimum_bytes(self) -> u64 {
        self.exact_streaming_minimum_bytes
    }

    /// Apply a fit policy to a hard memory budget and an optional streaming-cache
    /// request.
    ///
    /// A resident plan is always preferred when it fits. Otherwise `Exact` streams when
    /// the exact minimum fits, `Refuse` stops, and `Degrade` streams exactly if possible
    /// or returns a non-executable request for an explicit degraded estimate. A selected
    /// streaming plan caps `requested_cache_bytes` to both remaining budget and the
    /// amount of weight memory that streaming displaced.
    pub fn decide(
        self,
        policy: FitPolicy,
        budget_bytes: u64,
        requested_cache_bytes: u64,
    ) -> FitDecision {
        if self.fully_resident_bytes <= budget_bytes {
            return FitDecision::FullyResident(FullyResidentPlan {
                estimate: self,
                budget_bytes,
            });
        }

        if matches!(policy, FitPolicy::Refuse) {
            return FitDecision::Refused(FitRefusal {
                policy,
                reason: RefusalReason::FullResidencyRequired,
                estimate: self,
                budget_bytes,
            });
        }

        if self.exact_streaming_minimum_bytes <= budget_bytes {
            return FitDecision::ExactStreaming(
                self.exact_streaming_plan(budget_bytes, requested_cache_bytes),
            );
        }

        match policy {
            FitPolicy::Exact => FitDecision::Refused(FitRefusal {
                policy,
                reason: RefusalReason::ExactStreamingMinimumDoesNotFit,
                estimate: self,
                budget_bytes,
            }),
            FitPolicy::Degrade => FitDecision::DegradationRequired(FitDegradation {
                estimate: self,
                budget_bytes,
            }),
            FitPolicy::Refuse => unreachable!("refuse policy returned above"),
        }
    }

    /// Apply a fit policy for an executor that can only stream weights.
    ///
    /// Unlike [`FitEstimate::decide`], this method never selects a fully resident plan
    /// that the caller cannot execute. `Exact` and `Degrade` select exact streaming
    /// whenever its minimum fits, even when full residency would also fit. `Refuse`
    /// never authorizes streaming: if full residency fits, the typed result says that a
    /// resident executor is required; otherwise it is an ordinary memory refusal.
    pub fn decide_for_streaming(
        self,
        policy: FitPolicy,
        budget_bytes: u64,
        requested_cache_bytes: u64,
    ) -> StreamingFitDecision {
        if matches!(policy, FitPolicy::Refuse) {
            if self.fully_resident_bytes <= budget_bytes {
                return StreamingFitDecision::ResidentExecutorRequired(FullyResidentPlan {
                    estimate: self,
                    budget_bytes,
                });
            }
            return StreamingFitDecision::Refused(FitRefusal {
                policy,
                reason: RefusalReason::FullResidencyRequired,
                estimate: self,
                budget_bytes,
            });
        }

        if self.exact_streaming_minimum_bytes <= budget_bytes {
            return StreamingFitDecision::ExactStreaming(
                self.exact_streaming_plan(budget_bytes, requested_cache_bytes),
            );
        }

        match policy {
            FitPolicy::Exact => StreamingFitDecision::Refused(FitRefusal {
                policy,
                reason: RefusalReason::ExactStreamingMinimumDoesNotFit,
                estimate: self,
                budget_bytes,
            }),
            FitPolicy::Degrade => StreamingFitDecision::DegradationRequired(FitDegradation {
                estimate: self,
                budget_bytes,
            }),
            FitPolicy::Refuse => unreachable!("refuse policy returned above"),
        }
    }

    fn exact_streaming_plan(
        self,
        budget_bytes: u64,
        requested_cache_bytes: u64,
    ) -> ExactStreamingPlan {
        debug_assert!(self.exact_streaming_minimum_bytes <= budget_bytes);
        let budget_headroom = budget_bytes - self.exact_streaming_minimum_bytes;
        let displaced_weight_bytes = self.fully_resident_bytes - self.exact_streaming_minimum_bytes;
        let cache_bytes = requested_cache_bytes
            .min(budget_headroom)
            .min(displaced_weight_bytes);
        ExactStreamingPlan {
            estimate: self,
            budget_bytes,
            requested_cache_bytes,
            cache_bytes,
        }
    }
}

/// Invalid or unrepresentable inputs to a [`FitEstimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitEstimateError {
    /// The fully resident total cannot be represented as a `u64` byte count.
    FullyResidentOverflow {
        /// Memory that both execution paths retain.
        always_resident_bytes: u64,
        /// Complete encoded weight payload.
        resident_weight_bytes: u64,
    },
    /// The exact-streaming minimum cannot be represented as a `u64` byte count.
    StreamingMinimumOverflow {
        /// Memory that both execution paths retain.
        always_resident_bytes: u64,
        /// Exact weight working set needed concurrently.
        streaming_working_set_bytes: u64,
    },
    /// The claimed streaming minimum is larger than the fully resident footprint.
    StreamingExceedsResident {
        /// Forecast bytes needed for full residency.
        fully_resident_bytes: u64,
        /// Claimed minimum bytes needed for exact streaming.
        exact_streaming_minimum_bytes: u64,
    },
}

impl fmt::Display for FitEstimateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullyResidentOverflow {
                always_resident_bytes,
                resident_weight_bytes,
            } => write!(
                formatter,
                "fully resident estimate overflows u64: {always_resident_bytes} + \
                 {resident_weight_bytes} bytes"
            ),
            Self::StreamingMinimumOverflow {
                always_resident_bytes,
                streaming_working_set_bytes,
            } => write!(
                formatter,
                "exact-streaming minimum overflows u64: {always_resident_bytes} + \
                 {streaming_working_set_bytes} bytes"
            ),
            Self::StreamingExceedsResident {
                fully_resident_bytes,
                exact_streaming_minimum_bytes,
            } => write!(
                formatter,
                "exact-streaming minimum ({exact_streaming_minimum_bytes} bytes) exceeds the \
                 fully resident estimate ({fully_resident_bytes} bytes)"
            ),
        }
    }
}

impl Error for FitEstimateError {}

/// The policy decision for one checked estimate and memory budget.
///
/// Only [`FitDecision::FullyResident`] and [`FitDecision::ExactStreaming`] are executable.
/// Refusal and degradation are first-class results so a caller cannot accidentally start
/// a run before reporting or resolving them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitDecision {
    /// Execute with every weight resident.
    FullyResident(FullyResidentPlan),
    /// Execute with byte-exact weight streaming.
    ExactStreaming(ExactStreamingPlan),
    /// Do not execute under the selected policy.
    Refused(FitRefusal),
    /// Obtain an explicit degraded configuration, estimate it, and plan again.
    DegradationRequired(FitDegradation),
}

impl FitDecision {
    /// Whether this decision authorizes execution.
    pub const fn is_executable(self) -> bool {
        matches!(self, Self::FullyResident(_) | Self::ExactStreaming(_))
    }

    /// Estimate on which this decision was based.
    pub const fn estimate(self) -> FitEstimate {
        match self {
            Self::FullyResident(plan) => plan.estimate,
            Self::ExactStreaming(plan) => plan.estimate,
            Self::Refused(refusal) => refusal.estimate,
            Self::DegradationRequired(degradation) => degradation.estimate,
        }
    }

    /// Hard byte budget against which this decision was made.
    pub const fn budget_bytes(self) -> u64 {
        match self {
            Self::FullyResident(plan) => plan.budget_bytes,
            Self::ExactStreaming(plan) => plan.budget_bytes,
            Self::Refused(refusal) => refusal.budget_bytes,
            Self::DegradationRequired(degradation) => degradation.budget_bytes,
        }
    }

    /// Forecast resident bytes for an executable decision, or `None` when no run is
    /// authorized.
    pub const fn forecast_bytes(self) -> Option<u64> {
        match self {
            Self::FullyResident(plan) => Some(plan.forecast_bytes()),
            Self::ExactStreaming(plan) => Some(plan.forecast_bytes()),
            Self::Refused(_) | Self::DegradationRequired(_) => None,
        }
    }
}

/// The policy decision for an executor whose only weight path is exact streaming.
///
/// Only [`StreamingFitDecision::ExactStreaming`] authorizes that executor to run.
/// [`StreamingFitDecision::ResidentExecutorRequired`] means memory is sufficient but the
/// executor lacks the resident capability demanded by [`FitPolicy::Refuse`]; it is not a
/// memory shortfall. Refusal and degradation retain the same meanings as in
/// [`FitDecision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingFitDecision {
    /// Execute through the exact streaming path.
    ExactStreaming(ExactStreamingPlan),
    /// Full residency fits, but the policy requires an executor that can provide it.
    ResidentExecutorRequired(FullyResidentPlan),
    /// Do not execute under the selected policy and memory budget.
    Refused(FitRefusal),
    /// Obtain an explicit degraded configuration, estimate it, and plan again.
    DegradationRequired(FitDegradation),
}

impl StreamingFitDecision {
    /// Whether this decision authorizes a streaming-only executor to run.
    pub const fn is_executable(self) -> bool {
        matches!(self, Self::ExactStreaming(_))
    }

    /// Estimate on which this decision was based.
    pub const fn estimate(self) -> FitEstimate {
        match self {
            Self::ExactStreaming(plan) => plan.estimate,
            Self::ResidentExecutorRequired(plan) => plan.estimate,
            Self::Refused(refusal) => refusal.estimate,
            Self::DegradationRequired(degradation) => degradation.estimate,
        }
    }

    /// Hard byte budget against which this decision was made.
    pub const fn budget_bytes(self) -> u64 {
        match self {
            Self::ExactStreaming(plan) => plan.budget_bytes,
            Self::ResidentExecutorRequired(plan) => plan.budget_bytes,
            Self::Refused(refusal) => refusal.budget_bytes,
            Self::DegradationRequired(degradation) => degradation.budget_bytes,
        }
    }

    /// Forecast bytes for an authorized streaming run, or `None` when this executor may
    /// not run.
    pub const fn forecast_bytes(self) -> Option<u64> {
        match self {
            Self::ExactStreaming(plan) => Some(plan.forecast_bytes()),
            Self::ResidentExecutorRequired(_) | Self::Refused(_) | Self::DegradationRequired(_) => {
                None
            }
        }
    }
}

/// An executable plan that keeps every checkpoint weight resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullyResidentPlan {
    estimate: FitEstimate,
    budget_bytes: u64,
}

impl FullyResidentPlan {
    /// Estimate from which this plan was produced.
    pub const fn estimate(self) -> FitEstimate {
        self.estimate
    }

    /// Hard byte budget used by the planner.
    pub const fn budget_bytes(self) -> u64 {
        self.budget_bytes
    }

    /// Forecast resident bytes for the run.
    pub const fn forecast_bytes(self) -> u64 {
        self.estimate.fully_resident_bytes
    }

    /// Budget left after the forecast footprint.
    pub const fn spare_bytes(self) -> u64 {
        self.budget_bytes - self.forecast_bytes()
    }
}

/// An executable, semantics-preserving streaming plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactStreamingPlan {
    estimate: FitEstimate,
    budget_bytes: u64,
    requested_cache_bytes: u64,
    cache_bytes: u64,
}

impl ExactStreamingPlan {
    /// Estimate from which this plan was produced.
    pub const fn estimate(self) -> FitEstimate {
        self.estimate
    }

    /// Hard byte budget used by the planner.
    pub const fn budget_bytes(self) -> u64 {
        self.budget_bytes
    }

    /// Minimum bytes that exact execution requires before optional caching.
    pub const fn minimum_bytes(self) -> u64 {
        self.estimate.exact_streaming_minimum_bytes
    }

    /// Cache capacity requested by the caller before fit capping.
    pub const fn requested_cache_bytes(self) -> u64 {
        self.requested_cache_bytes
    }

    /// Cache capacity authorized by the fit budget.
    pub const fn cache_bytes(self) -> u64 {
        self.cache_bytes
    }

    /// Whether the fit budget reduced the requested cache capacity.
    pub const fn cache_was_capped(self) -> bool {
        self.cache_bytes < self.requested_cache_bytes
    }

    /// Forecast resident bytes including the authorized cache capacity.
    pub const fn forecast_bytes(self) -> u64 {
        // Construction caps the cache to `budget - minimum`, so this cannot overflow.
        self.minimum_bytes() + self.cache_bytes
    }

    /// Budget left after the forecast footprint and authorized cache.
    pub const fn spare_bytes(self) -> u64 {
        self.budget_bytes - self.forecast_bytes()
    }
}

/// Why a [`FitDecision::Refused`] result did not authorize a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalReason {
    /// `FitPolicy::Refuse` requires the complete resident footprint to fit.
    FullResidencyRequired,
    /// `FitPolicy::Exact` cannot fit even the minimum exact-streaming working set.
    ExactStreamingMinimumDoesNotFit,
}

/// A non-executable refusal with the byte cost needed to resolve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitRefusal {
    policy: FitPolicy,
    reason: RefusalReason,
    estimate: FitEstimate,
    budget_bytes: u64,
}

impl FitRefusal {
    /// Policy that produced this refusal.
    pub const fn policy(self) -> FitPolicy {
        self.policy
    }

    /// Specific policy condition that could not be satisfied.
    pub const fn reason(self) -> RefusalReason {
        self.reason
    }

    /// Estimate from which this refusal was produced.
    pub const fn estimate(self) -> FitEstimate {
        self.estimate
    }

    /// Hard byte budget used by the planner.
    pub const fn budget_bytes(self) -> u64 {
        self.budget_bytes
    }

    /// Footprint that this policy required.
    pub const fn required_bytes(self) -> u64 {
        match self.reason {
            RefusalReason::FullResidencyRequired => self.estimate.fully_resident_bytes,
            RefusalReason::ExactStreamingMinimumDoesNotFit => {
                self.estimate.exact_streaming_minimum_bytes
            }
        }
    }

    /// Additional budget needed to satisfy the selected policy.
    pub const fn shortfall_bytes(self) -> u64 {
        self.required_bytes() - self.budget_bytes
    }

    /// Whether switching from `Refuse` to an exact-streaming policy would fit the same
    /// estimate in the current budget.
    pub const fn exact_streaming_would_fit(self) -> bool {
        self.estimate.exact_streaming_minimum_bytes <= self.budget_bytes
    }
}

/// A non-executable request to define and disclose a degraded configuration.
///
/// The fit layer cannot invent a smaller context, lower offload, or other semantic
/// change. A caller must construct that configuration, expose its rationale, produce a
/// new [`FitEstimate`], and ask the planner again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitDegradation {
    estimate: FitEstimate,
    budget_bytes: u64,
}

impl FitDegradation {
    /// Exact estimate that could not fit.
    pub const fn estimate(self) -> FitEstimate {
        self.estimate
    }

    /// Hard byte budget used by the planner.
    pub const fn budget_bytes(self) -> u64 {
        self.budget_bytes
    }

    /// Smallest exact footprint considered by the planner.
    pub const fn exact_required_bytes(self) -> u64 {
        self.estimate.exact_streaming_minimum_bytes
    }

    /// Bytes by which the smallest exact footprint exceeds the budget.
    pub const fn exact_shortfall_bytes(self) -> u64 {
        self.exact_required_bytes() - self.budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate() -> FitEstimate {
        // 20 bytes are common, all weights occupy 80 bytes, and the exact streaming
        // path needs 30 weight bytes at once: resident=100, streaming minimum=50.
        FitEstimate::from_parts(20, 80, 30).unwrap()
    }

    #[test]
    fn estimate_accepts_totals_and_checked_parts() {
        let from_totals = FitEstimate::new(100, 50).unwrap();
        assert_eq!(estimate(), from_totals);
        assert_eq!(from_totals.fully_resident_bytes(), 100);
        assert_eq!(from_totals.exact_streaming_minimum_bytes(), 50);

        assert_eq!(FitEstimate::new(0, 0).unwrap().fully_resident_bytes(), 0);
        assert_eq!(
            FitEstimate::new(u64::MAX, u64::MAX)
                .unwrap()
                .exact_streaming_minimum_bytes(),
            u64::MAX
        );
    }

    #[test]
    fn estimate_rejects_inverted_and_overflowing_footprints() {
        let inverted = FitEstimate::new(9, 10).unwrap_err();
        assert_eq!(
            inverted,
            FitEstimateError::StreamingExceedsResident {
                fully_resident_bytes: 9,
                exact_streaming_minimum_bytes: 10,
            }
        );
        assert_eq!(
            inverted.to_string(),
            "exact-streaming minimum (10 bytes) exceeds the fully resident estimate (9 bytes)"
        );

        let resident_overflow = FitEstimate::from_parts(u64::MAX, 1, 0).unwrap_err();
        assert_eq!(
            resident_overflow,
            FitEstimateError::FullyResidentOverflow {
                always_resident_bytes: u64::MAX,
                resident_weight_bytes: 1,
            }
        );
        assert_eq!(
            resident_overflow.to_string(),
            format!(
                "fully resident estimate overflows u64: {} + 1 bytes",
                u64::MAX
            )
        );

        let streaming_overflow = FitEstimate::from_parts(u64::MAX, 0, 1).unwrap_err();
        assert_eq!(
            streaming_overflow,
            FitEstimateError::StreamingMinimumOverflow {
                always_resident_bytes: u64::MAX,
                streaming_working_set_bytes: 1,
            }
        );
        assert_eq!(
            streaming_overflow.to_string(),
            format!(
                "exact-streaming minimum overflows u64: {} + 1 bytes",
                u64::MAX
            )
        );
    }

    #[test]
    fn every_policy_uses_residency_when_the_full_forecast_fits() {
        for policy in [FitPolicy::Exact, FitPolicy::Degrade, FitPolicy::Refuse] {
            for budget_bytes in [100, 101, u64::MAX] {
                let decision = estimate().decide(policy, budget_bytes, u64::MAX);
                let FitDecision::FullyResident(plan) = decision else {
                    panic!("{policy:?} with {budget_bytes} bytes produced {decision:?}");
                };
                assert!(decision.is_executable());
                assert_eq!(decision.forecast_bytes(), Some(100));
                assert_eq!(decision.budget_bytes(), budget_bytes);
                assert_eq!(decision.estimate(), estimate());
                assert_eq!(plan.estimate(), estimate());
                assert_eq!(plan.budget_bytes(), budget_bytes);
                assert_eq!(plan.forecast_bytes(), 100);
                assert_eq!(plan.spare_bytes(), budget_bytes - 100);
            }
        }
    }

    #[test]
    fn streaming_executor_truth_table_is_typed_for_every_policy_and_fit_regime() {
        for policy in [FitPolicy::Exact, FitPolicy::Degrade] {
            for budget_bytes in [100, 101, u64::MAX] {
                let decision = estimate().decide_for_streaming(policy, budget_bytes, 7);
                let StreamingFitDecision::ExactStreaming(plan) = decision else {
                    panic!("{policy:?} with {budget_bytes} bytes produced {decision:?}");
                };
                assert!(decision.is_executable());
                assert_eq!(decision.estimate(), estimate());
                assert_eq!(decision.budget_bytes(), budget_bytes);
                assert_eq!(decision.forecast_bytes(), Some(57));
                assert_eq!(plan.cache_bytes(), 7);
                assert_eq!(plan.forecast_bytes(), 57);
            }
        }

        for budget_bytes in [100, 101, u64::MAX] {
            let decision =
                estimate().decide_for_streaming(FitPolicy::Refuse, budget_bytes, u64::MAX);
            let StreamingFitDecision::ResidentExecutorRequired(plan) = decision else {
                panic!("refuse with {budget_bytes} bytes produced {decision:?}");
            };
            assert!(!decision.is_executable());
            assert_eq!(decision.estimate(), estimate());
            assert_eq!(decision.budget_bytes(), budget_bytes);
            assert_eq!(decision.forecast_bytes(), None);
            assert_eq!(plan.forecast_bytes(), 100);
            assert_eq!(plan.spare_bytes(), budget_bytes - 100);
        }

        for policy in [FitPolicy::Exact, FitPolicy::Degrade] {
            for budget_bytes in [50, 51, 99] {
                let decision = estimate().decide_for_streaming(policy, budget_bytes, 7);
                let StreamingFitDecision::ExactStreaming(plan) = decision else {
                    panic!("{policy:?} with {budget_bytes} bytes produced {decision:?}");
                };
                assert!(decision.is_executable());
                assert_eq!(
                    decision.forecast_bytes(),
                    Some(50 + 7.min(budget_bytes - 50))
                );
                assert_eq!(plan.cache_bytes(), 7.min(budget_bytes - 50));
            }
        }

        for budget_bytes in [50, 51, 99] {
            assert!(matches!(
                estimate().decide_for_streaming(FitPolicy::Refuse, budget_bytes, 7),
                StreamingFitDecision::Refused(FitRefusal {
                    reason: RefusalReason::FullResidencyRequired,
                    ..
                })
            ));
        }

        for budget_bytes in [0, 49] {
            assert!(matches!(
                estimate().decide_for_streaming(FitPolicy::Exact, budget_bytes, 7),
                StreamingFitDecision::Refused(FitRefusal {
                    reason: RefusalReason::ExactStreamingMinimumDoesNotFit,
                    ..
                })
            ));
            assert!(matches!(
                estimate().decide_for_streaming(FitPolicy::Degrade, budget_bytes, 7),
                StreamingFitDecision::DegradationRequired(_)
            ));
            assert!(matches!(
                estimate().decide_for_streaming(FitPolicy::Refuse, budget_bytes, 7),
                StreamingFitDecision::Refused(FitRefusal {
                    reason: RefusalReason::FullResidencyRequired,
                    ..
                })
            ));
        }
    }

    #[test]
    fn streaming_executor_cache_is_bounded_even_when_full_residency_fits() {
        let decision = estimate().decide_for_streaming(FitPolicy::Exact, u64::MAX, u64::MAX);
        let StreamingFitDecision::ExactStreaming(plan) = decision else {
            panic!("expected exact streaming, got {decision:?}");
        };
        // Caching more than the 50 bytes displaced by streaming would cost at least as
        // much as the fully resident path and cannot improve on it.
        assert_eq!(plan.cache_bytes(), 50);
        assert_eq!(plan.forecast_bytes(), 100);
        assert!(plan.cache_was_capped());
    }

    #[test]
    fn exact_and_degrade_stream_exactly_when_only_the_streaming_minimum_fits() {
        for policy in [FitPolicy::Exact, FitPolicy::Degrade] {
            for budget_bytes in [50, 51, 99] {
                let decision = estimate().decide(policy, budget_bytes, 7);
                let FitDecision::ExactStreaming(plan) = decision else {
                    panic!("{policy:?} with {budget_bytes} bytes produced {decision:?}");
                };
                assert!(decision.is_executable());
                assert_eq!(
                    decision.forecast_bytes(),
                    Some(50 + 7.min(budget_bytes - 50))
                );
                assert_eq!(plan.estimate(), estimate());
                assert_eq!(plan.budget_bytes(), budget_bytes);
                assert_eq!(plan.minimum_bytes(), 50);
                assert_eq!(plan.requested_cache_bytes(), 7);
                assert_eq!(plan.cache_bytes(), 7.min(budget_bytes - 50));
                assert_eq!(plan.forecast_bytes(), 50 + plan.cache_bytes());
                assert_eq!(plan.spare_bytes(), budget_bytes - plan.forecast_bytes());
            }
        }
    }

    #[test]
    fn refuse_policy_refuses_when_full_residency_does_not_fit() {
        for budget_bytes in [0, 49, 50, 99] {
            let decision = estimate().decide(FitPolicy::Refuse, budget_bytes, 7);
            let FitDecision::Refused(refusal) = decision else {
                panic!("budget {budget_bytes} produced {decision:?}");
            };
            assert!(!decision.is_executable());
            assert_eq!(decision.forecast_bytes(), None);
            assert_eq!(refusal.policy(), FitPolicy::Refuse);
            assert_eq!(refusal.reason(), RefusalReason::FullResidencyRequired);
            assert_eq!(refusal.estimate(), estimate());
            assert_eq!(refusal.budget_bytes(), budget_bytes);
            assert_eq!(refusal.required_bytes(), 100);
            assert_eq!(refusal.shortfall_bytes(), 100 - budget_bytes);
            assert_eq!(refusal.exact_streaming_would_fit(), budget_bytes >= 50);
        }
    }

    #[test]
    fn exact_refuses_and_degrade_requests_an_explicit_change_below_the_minimum() {
        for budget_bytes in [0, 49] {
            let exact = estimate().decide(FitPolicy::Exact, budget_bytes, u64::MAX);
            let FitDecision::Refused(refusal) = exact else {
                panic!("exact with {budget_bytes} bytes produced {exact:?}");
            };
            assert!(!exact.is_executable());
            assert_eq!(exact.forecast_bytes(), None);
            assert_eq!(refusal.policy(), FitPolicy::Exact);
            assert_eq!(
                refusal.reason(),
                RefusalReason::ExactStreamingMinimumDoesNotFit
            );
            assert_eq!(refusal.required_bytes(), 50);
            assert_eq!(refusal.shortfall_bytes(), 50 - budget_bytes);
            assert!(!refusal.exact_streaming_would_fit());

            let degrade = estimate().decide(FitPolicy::Degrade, budget_bytes, u64::MAX);
            let FitDecision::DegradationRequired(required) = degrade else {
                panic!("degrade with {budget_bytes} bytes produced {degrade:?}");
            };
            assert!(!degrade.is_executable());
            assert_eq!(degrade.forecast_bytes(), None);
            assert_eq!(required.estimate(), estimate());
            assert_eq!(required.budget_bytes(), budget_bytes);
            assert_eq!(required.exact_required_bytes(), 50);
            assert_eq!(required.exact_shortfall_bytes(), 50 - budget_bytes);
        }
    }

    #[test]
    fn streaming_cache_is_capped_without_overflow() {
        let uncapped = estimate().decide(FitPolicy::Exact, 90, 12);
        let FitDecision::ExactStreaming(uncapped) = uncapped else {
            panic!("expected streaming plan");
        };
        assert_eq!(uncapped.cache_bytes(), 12);
        assert!(!uncapped.cache_was_capped());
        assert_eq!(uncapped.forecast_bytes(), 62);
        assert_eq!(uncapped.spare_bytes(), 28);

        let capped = estimate().decide(FitPolicy::Exact, 60, 12);
        let FitDecision::ExactStreaming(capped) = capped else {
            panic!("expected streaming plan");
        };
        assert_eq!(capped.cache_bytes(), 10);
        assert!(capped.cache_was_capped());
        assert_eq!(capped.forecast_bytes(), 60);
        assert_eq!(capped.spare_bytes(), 0);

        let maximum_request = FitEstimate::new(u64::MAX, u64::MAX - 10).unwrap().decide(
            FitPolicy::Exact,
            u64::MAX - 1,
            u64::MAX,
        );
        let FitDecision::ExactStreaming(maximum_request) = maximum_request else {
            panic!("expected streaming plan");
        };
        assert_eq!(maximum_request.cache_bytes(), 9);
        assert_eq!(maximum_request.forecast_bytes(), u64::MAX - 1);
        assert!(maximum_request.cache_was_capped());
    }

    #[test]
    fn equal_exact_and_resident_estimates_have_no_impossible_streaming_regime() {
        let estimate = FitEstimate::new(10, 10).unwrap();
        for policy in [FitPolicy::Exact, FitPolicy::Degrade, FitPolicy::Refuse] {
            assert!(matches!(
                estimate.decide(policy, 10, u64::MAX),
                FitDecision::FullyResident(_)
            ));
        }
        for policy in [FitPolicy::Exact, FitPolicy::Degrade] {
            let decision = estimate.decide_for_streaming(policy, 10, u64::MAX);
            let StreamingFitDecision::ExactStreaming(plan) = decision else {
                panic!("{policy:?} produced {decision:?}");
            };
            assert_eq!(plan.cache_bytes(), 0);
            assert_eq!(plan.forecast_bytes(), 10);
        }
        assert!(matches!(
            estimate.decide_for_streaming(FitPolicy::Refuse, 10, u64::MAX),
            StreamingFitDecision::ResidentExecutorRequired(_)
        ));
        assert!(matches!(
            estimate.decide(FitPolicy::Exact, 9, u64::MAX),
            FitDecision::Refused(FitRefusal {
                reason: RefusalReason::ExactStreamingMinimumDoesNotFit,
                ..
            })
        ));
        assert!(matches!(
            estimate.decide(FitPolicy::Degrade, 9, u64::MAX),
            FitDecision::DegradationRequired(_)
        ));
        assert!(matches!(
            estimate.decide(FitPolicy::Refuse, 9, u64::MAX),
            FitDecision::Refused(FitRefusal {
                reason: RefusalReason::FullResidencyRequired,
                ..
            })
        ));
    }
}
