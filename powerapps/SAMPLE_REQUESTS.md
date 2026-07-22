# Sample request and polling flow

## Single analysis

`POST /api/analyze` is multipart/form-data:

| Field | Example |
| --- | --- |
| `file` | uploaded image or video binary |
| `query` | `driver missing seatbelt` |
| `fps` | `1.0` |
| `topK` | `5` |
| `enableVlm` | `false` |
| `vlmProfile` | `rule_based` |
| `vlmEnabled` | `false` |
| `device` | `auto` or `cpu` |
| `reviewMode` | `fast_local` or `comprehensive` |

Example accepted response:

```json
{"jobId":"job_20260710_064302_49fa4c77","status":"queued"}
```

Poll `GET /api/jobs/{jobId}` every 2-5 seconds until `status` is `completed`,
`failed`, or `cancelled`. A completed status includes `elapsedSeconds`,
`analysisRuntimeSeconds`, and Fast Local runtime summary fields. Then request
`/api/jobs/{jobId}/result` and `/api/reports/{jobId}/technical-json`.

## Batch analysis

`POST /api/batches/analyze` accepts repeated `files` multipart fields plus the
same analysis settings. Its response includes `batchId` and `jobIds`. Poll
`GET /api/batches/{batchId}`, not the job route.

For folder uploads, repeat `relativePaths` in the same order as `files`:

```text
files=@depot-a.mp4
relativePaths=Depot A/Shift 1/camera.mp4
files=@depot-b.mp4
relativePaths=Depot B/Shift 1/camera.mp4
importKey=overnight-2026-07-15
```

ZIP imports preserve safe nested paths. Unsafe paths, symbolic links,
unsupported entries, duplicates, empty files, excessive expanded size, and
excessive compression ratios are rejected. Valid videos can still be accepted
and listed alongside per-file rejection reasons.

## Errors

- `400`: invalid/missing upload, query, or request settings.
- `404`: unknown job/batch or a batch ID passed to a job endpoint.
- `413`: upload exceeds configured server/platform limit.
- `429`: durable queue capacity reached; retain the import key and retry later.
- `503`/`507`: RAM or disk backpressure guard; retry after capacity recovers.
- `500`: unrecoverable processing or storage failure; show the backend error
  identifier to the operator without treating it as an authoritative safety
  result.
