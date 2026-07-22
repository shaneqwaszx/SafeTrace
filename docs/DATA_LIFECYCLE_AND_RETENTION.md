# SafeTrace data lifecycle and retention

SafeTrace is local-first. Media is not uploaded to a cloud service unless an operator explicitly configures and deploys such storage. The backend filesystem is canonical; browser storage contains IDs, filters, and preferences only.

| Class | Default location | Owner | Restart | Default retention | Deleted with job | Shared/exportable | Sensitive |
|---|---|---|---|---|---|---|---|
| A. Source upload | `data/api_jobs/<job>/uploads` | Job/batch | Yes | Job lifetime | Yes | Not shared; export only by explicit workflow | Yes |
| B. In-progress work | Job media/temp and bounded `data/frames` | Job | Safe stage restart | Recovery policy/72 hours | Yes | No | Yes |
| C. Durable state | Job/batch `manifest.json` | Job/batch | Yes | Status-specific | Yes | Metadata export | May contain names/paths |
| D. Completed metadata | Job `result.json` | Job | Yes | 30 days | Yes | Exportable | Yes |
| E. Evidence derivatives | Job `media` | Job | Yes | Job lifetime | Yes | Exportable | Yes |
| F. Result cache | `data/result_cache/<key>` | Exact content/config key | Yes | 14 days/5 GB | Reference released; shared entry retained until unreferenced/expired | Deduplicated exact results | Yes |
| G. Browser/session cache | Browser storage | Browser user | Session/preferences only | Session/user clear | Separate clear action | No large payloads | IDs/preferences only |
| H. Logs/diagnostics | `logs` or configured runtime path | Backend | Yes | 14 days | No | Support export | May be sensitive |
| I. Pinned/exported results | Configured export storage | User | Yes | Explicit user policy | Protected | Yes | Yes |

## Recovery

`SAFETRACE_RECOVERY_POLICY=prompt` is the default. Interrupted work is marked for a user decision and is not silently resumed. `auto_resume` is intended for supervised deployments; `discard` removes recoverable interrupted work at startup. Mid-inference state cannot be resumed safely, so SafeTrace restarts that job's current analysis stage while preserving completed sibling jobs and batch hierarchy.

## Deletion and cleanup

Deleting a batch deletes its owned child jobs. Deleting a job removes its upload, evidence, result, manifest, and job-owned hard links, then releases its reusable-cache reference. Cleanup preview is mandatory before retention cleanup. Active, resumable, pinned, and exported data is protected. Cleanup produces an audit file with actual reclaimed bytes.

## Configuration

- `SAFETRACE_COMPLETED_RETENTION_DAYS=30`
- `SAFETRACE_FAILED_RETENTION_DAYS=7`
- `SAFETRACE_RECOVERY_RETENTION_HOURS=72`
- `SAFETRACE_CACHE_RETENTION_DAYS=14`
- `SAFETRACE_CACHE_MAX_GB=5`
- `SAFETRACE_LOG_RETENTION_DAYS=14`
- `SAFETRACE_MIN_FREE_DISK_GB=2`

Persistent deployments must put the configured data root on durable storage. A single backend instance remains required until queue, metadata, cache references, and media are moved to shared transactional services.
