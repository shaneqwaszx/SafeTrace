# SafeTrace Cloud Fast Local deployment

This profile packages the current source tree for Fast Local Analysis only. It
uses the rule-based detector pipeline and has no VLM or chatbot requirement.

## Prerequisites

- Docker Engine with Docker Compose v2, or equivalent separate frontend and
  backend hosting.
- A persistent volume for `/var/lib/safetrace/data`.
- A protected HTTPS backend URL and frontend URL.

## Configure

1. Copy `deployment/env/backend.cloud.env.example` into the cloud platform's
   backend environment configuration and replace `YOUR_FRONTEND_DOMAIN`.
2. Set `VITE_SAFETRACE_API_BASE_URL` to the public backend URL. The frontend
   build fails when this value is empty.
3. Do not enable VLM or chat variables for this profile.

## Compose check

From the repository root, with the environment variables set:

```powershell
docker compose -f deployment/docker-compose.cloud.yml config
docker compose -f deployment/docker-compose.cloud.yml up --build
```

The backend listens on `0.0.0.0:$PORT`, default `8000`. The frontend listens
on port `8080` in the container. The React API base is a build-time public
configuration and must be the externally reachable backend URL.

## Runtime profile

- `SAFETRACE_VLM_ENABLED=0`, `SAFETRACE_ENABLE_VLM=0`, and both VLM workers
  disabled.
- Chat disabled; no GGUF is included.
- Single analysis worker and single backend replica only.
- MobileSAM checkpoint is included as an optional asset but disabled by
  default. Enable it only after a CPU/cloud worker smoke proves acceptable
  runtime and memory behavior.

## Scaling boundary

The current queue and job store use local process/filesystem state. Run one
backend instance. Horizontal scaling requires a shared object store, database,
and external job queue before adding replicas.

## Security

Put the API behind platform authentication, an API gateway, or Microsoft Entra
ID integration. The current backend does not provide production authentication
by itself. Restrict CORS with `SAFETRACE_ALLOWED_ORIGINS`; do not expose the
analysis API anonymously.
