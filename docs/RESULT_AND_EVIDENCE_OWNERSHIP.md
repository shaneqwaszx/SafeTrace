# Result and Evidence Ownership

SafeTrace treats a completed analysis as an immutable, single-job snapshot. Shared model weights may be reused, but mutable frames, findings, events, diagnostics, artifacts, cache restores, recovery state, and exports must remain job-scoped.

Every result carries `jobId`, optional `batchId`, `sourceChecksum`, `originalFilename`, `sourceRelativePath`, `sourceGroupPath`, `executionIdentity`, and `resultSchemaVersion`. Every frame/evidence record additionally carries a job-namespaced `evidenceId` and, when present, a job-namespaced `mediaArtifactId`. Findings, events, supporting frames, summaries, and technical ownership metadata repeat the immutable owner identity so mismatches can be detected without relying on filenames or array order.

The API validates the canonical result before returning status-derived results, technical JSON, media, batch child results, or exports. A mismatch fails closed with `cross_job_result_contamination`; foreign evidence is never rendered or exported. Cache reuse validates the origin snapshot, deep-copies it, rewrites every ownership field to the target job, and materializes target-job media paths. Recovery checkpoints are scoped to one job directory and must match that job and its execution identity.

Frontend result state is keyed by job ID. Selection requests use a generation token and cancellation, and evidence is rendered only when `evidence.jobId` matches the selected canonical result.

## Invariants

1. A job result endpoint returns one canonical job owner only.
2. The route job ID and top-level result `jobId` are identical.
3. Every finding, event, supporting frame, evidence record, media artifact, source record, and technical ownership block has the same owner.
4. Batch responses keep child payloads under their child job IDs and never flatten raw evidence into a single-job result.
5. Media URLs and artifact IDs are job-namespaced, and another job's route cannot retrieve them.
6. Exports are generated from a validated deep snapshot of the explicitly selected job.
7. Cache hits validate the origin, deep-copy the result, rewrite ownership, and materialize target-job media before publication.
8. Recovery checkpoints must match both the expected job ID and compatible immutable execution identity.
9. Frontend job selection replaces the visible canonical payload; late requests cannot append to or overwrite another selection.
10. Any mismatch fails closed with `cross_job_result_contamination` and bounded diagnostics.
