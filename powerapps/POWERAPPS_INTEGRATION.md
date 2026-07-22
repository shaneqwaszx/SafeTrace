# SafeTrace and Power Apps integration

The cloud handoff supports Fast Local Analysis only. VLM and chatbot endpoints
may exist in the source OpenAPI but are disabled by the cloud environment and
are not part of the supported Power Apps connector flow.

## Approach A - hosted React application

Host `frontend-react` and the FastAPI backend separately. Power Apps can link
to the hosted application, pass a record URL/identifier, or orchestrate the
surrounding case-management workflow. This is the lowest-risk option when the
React evidence viewer is the primary review UI.

## Approach B - Power Apps custom connector

Import `powerapps/openapi.json` into a custom connector pointing to the hosted
backend base URL. Build Power Apps screens that upload media, store `jobId`,
poll the job endpoint, and open result/evidence URLs.

## Required production controls

- Add platform authentication, an API gateway, or Microsoft Entra ID before
  exposing the API to Power Apps.
- Configure CORS for the actual hosted frontend only.
- Validate Power Apps and gateway file-size limits. Large uploads may require
  object-storage upload first, followed by an analysis request that references
  the stored object through a future authenticated ingestion flow.

Do not expose the unauthenticated analysis API anonymously in production.

## Accuracy and deployment gate

Power Apps rollout must remain a controlled review workflow until the approved
labelled SafeTrace benchmark passes. Runtime success on unlabelled video is not
an accuracy result. Before a production connector is enabled, record the
detector checkpoint SHA256, labelled dataset version, Fast Local benchmark
metrics by use case, false-positive counts for seatbelt and helmet, soak result,
and a named reviewer approval. Keep all findings reviewable and do not automate
disciplinary or clearance decisions from SafeTrace output.

The initial connector should target one backend worker because job state and
media are stored on the local filesystem. Horizontal scaling requires shared
object storage plus an external queue/job store. Configure authentication,
retention, regional storage, upload limits, and deletion policy before real
footage is accepted.

## Overnight batch integration

Power Apps can send repeated `files` parts plus one `relativePaths` value per
file. The relative path is retained in manifests, evidence, hierarchy, and
technical JSON. Supply a stable `importKey` when retrying the same import so a
network retry returns the existing batch instead of creating duplicate jobs.

The connector should poll `/api/batches/{batchId}` and render the returned
`statusCounts`, `groupSummaries`, and `hierarchy`. Durable lifecycle values may
include `waiting_for_capacity`, `retry_wait`, `paused`, `recovering`, and
`completed_with_failures`; these are operational states, not new safety
findings. Operators can pause queued work, resume it, or retry only failed
children through the dedicated endpoints.

## Recovery and storage operations

After reconnect, call `GET /api/recovery`. With the default `prompt` policy,
render the returned source hierarchy, last stage, progress, retry count, stored
bytes, and proposed action. A user can submit selected `jobIds` or `batchIds`
to `POST /api/recovery/resume`, `POST /api/recovery/discard`, or defer the prompt
with `POST /api/recovery/later`. Resume restarts only the interrupted job's
current safe analysis stage; completed sibling jobs remain unchanged.

Use `GET /api/storage/summary` for category and retention metadata. Destructive
retention cleanup is a two-step operation: call
`POST /api/storage/cleanup/preview`, display candidates and estimated bytes,
obtain explicit confirmation, then call `POST /api/storage/cleanup/apply` with
`confirm: true`. Keep browser-cache clearing, job deletion, recovery-data
discard, and reusable-cache cleanup as separate user actions.

## Dashboard and pagination

`GET /api/dashboard/summary` is the canonical source for operational cards and
charts. It intentionally reports `accuracyMetricsAvailable: false` until a
labelled gate exists. Use paginated `GET /api/jobs`, `GET /api/batches`, and
`GET /api/batches/{batchId}/results`; do not download every technical result to
build a dashboard. Load hierarchy from `GET /api/batches/{batchId}/tree` and
load technical JSON/evidence only when the reviewer opens a job.

## Secondary review boundary

`POST /api/jobs/{jobId}/secondary-reviews` records a provider-neutral review
result, latency, agreement, fallback reason, and evidence frame IDs. The review
is always marked `authoritative: false`; external AI output cannot silently
replace the local detector/rule result. Authentication, provider credentials,
quotas, privacy review, and audit retention remain deployment responsibilities.
