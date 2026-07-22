# Security and storage notes

- Treat uploads, evidence images, technical JSON, and reports as sensitive
  operational data.
- Use a persistent encrypted volume or managed object storage for
  `SAFETRACE_DATA_DIR`; do not rely on container filesystem lifetime.
- Apply lifecycle/retention policy outside the container. The backend's local
  cleanup behavior is not a replacement for governed retention.
- Use HTTPS, platform authentication, and least-privilege service identities.
- Configure exact frontend origins in `SAFETRACE_ALLOWED_ORIGINS` and set
  `SAFETRACE_INCLUDE_LOCAL_CORS_ORIGINS=false` in cloud environments.
- Do not store secrets in `*.env.example`, Dockerfiles, compose files, or the
  handoff archive.
- For large videos, use direct-to-object-storage uploads or a platform upload
  workflow. Power Apps and gateway request-size limits must be tested by the
  hosting teammate.
