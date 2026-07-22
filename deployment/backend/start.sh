#!/bin/sh
set -eu

: "${PORT:=8000}"
: "${SAFETRACE_DATA_DIR:=/var/lib/safetrace/data}"
mkdir -p "$SAFETRACE_DATA_DIR/api_jobs" "$SAFETRACE_DATA_DIR/api_batches"

exec uvicorn src.api.server:app --host 0.0.0.0 --port "$PORT" --log-level "${SAFETRACE_LOG_LEVEL:-info}"
