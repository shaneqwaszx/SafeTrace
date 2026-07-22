# Endpoint mapping

| Endpoint | Power Apps use |
| --- | --- |
| `GET /api/health` | Connection health indicator |
| `GET /api/system/status` | Capability/status display; cloud profile should show Fast Local only |
| `GET /api/dashboard/summary` | Backend-canonical operational cards and charts; no inferred accuracy |
| `GET /api/recovery` | List interrupted jobs awaiting a recovery decision |
| `POST /api/recovery/resume` | Resume selected interrupted jobs from their current safe stage |
| `POST /api/recovery/later` | Persist a deferred recovery decision |
| `POST /api/recovery/discard` | Delete selected interrupted state and return reclaimed bytes |
| `GET /api/storage/summary` | Storage categories, pressure, protection, and retention metadata |
| `POST /api/storage/cleanup/preview` | Dry-run retention cleanup candidates and estimated bytes |
| `POST /api/storage/cleanup/apply` | Apply a newly calculated cleanup after explicit confirmation |
| `GET /api/jobs` | Paginated/filterable job list |
| `GET /api/batches` | Paginated batch list |
| `POST /api/analyze` | Multipart single image/video upload; returns `jobId` |
| `GET /api/jobs/{job_id}` | Poll queued/running/completed/failed state and elapsed time |
| `GET /api/jobs/{job_id}/result` | Read completed grouped findings and evidence links |
| `GET /api/media/{job_id}/{filename}` | Display protected evidence media after authorization design |
| `GET /api/reports/{job_id}/technical-json` | Retrieve technical JSON for audit or review |
| `DELETE /api/jobs/{job_id}` | Delete a completed/failed job according to retention policy |
| `POST /api/batches/analyze` | Multipart multi-video batch upload; returns `batchId` and child jobs |
| `GET /api/batches/{batch_id}` | Poll batch progress and child status counts |
| `GET /api/batches/{batch_id}/tree` | Read canonical folder/group hierarchy and summaries |
| `GET /api/batches/{batch_id}/results` | Read available child results with source-path metadata |
| `POST /api/batches/{batch_id}/pause` | Pause queued child jobs; running work finishes safely |
| `POST /api/batches/{batch_id}/resume` | Resume paused child jobs |
| `POST /api/batches/{batch_id}/retry-failed` | Retry failed children without duplicating completed work |
| `POST /api/batches/{batch_id}/cancel` | Cancel work that has not started |
| `POST /api/jobs/{job_id}/secondary-reviews` | Attach non-authoritative external review diagnostics |
| `DELETE /api/batches/{batch_id}` | Delete batch record according to retention policy |

`jobId` and `batchId` are different resources. Never poll a batch using
`/api/jobs/{batch_id}`; that endpoint correctly returns 404.

Evidence records include `sourceRelativePath`, `sourceGroup`, `batchId`,
`jobId`, `findingId`, `timestampSeconds`, and `timestampLabel` so a connector
can trace a review card back to its original video and time.
