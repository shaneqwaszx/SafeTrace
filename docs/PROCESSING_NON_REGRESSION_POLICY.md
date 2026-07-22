# Processing non-regression policy

SafeTrace processing contracts protect stable semantics without freezing known
mistakes. Versioned manifests under `evaluation/processing_contracts/` define:

- exact invariants for source/config/model/cache identity where deterministic;
- expected ranges for model-dependent candidate, event, and evidence counts;
- required or forbidden finding types;
- review-only versus authoritative behavior;
- negative controls for scene applicability, ownership, cache, and recovery.

Changes to sampling, thresholds, profile composition, scene gating,
aggregation, or evidence suppression require before/after metrics, a reason,
the expected impact, and regression results. Baselines must never be updated
only to make a failing check pass. The comparison tool requires a non-empty
reason before any explicit baseline-update workflow can proceed.
