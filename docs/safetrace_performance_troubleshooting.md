# SafeTrace Performance Troubleshooting

Use this checklist when local analysis feels slow or appears stuck.

## Start The Backend With The Local Runtime

From the repo root:

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info
```

This keeps the backend on the same Python environment used for local packages
and sets the OpenMP guardrails before optional model runtimes load.

Verify `llama_cpp` with the exact backend Python:

```cmd
.venv\Scripts\python.exe -c "import sys; print(sys.executable); import llama_cpp; print('llama_cpp ok')"
```

If this succeeds but the UI reports `missing_runtime`, restart the backend and
check `/api/chat/status` for `python_executable`, `expected_venv_python`,
`running_in_expected_venv`, and `llama_cpp_import_status`.

## Fast Local Analysis

For fastest local testing:

- keep Visual explanations on only if you want explanation text displayed,
- keep VLM explanation mode set to `Rule-based`,
- keep `Activate VLM` off,
- leave `SAFETRACE_ENABLE_VLM=false`,
- leave `SAFETRACE_VLM_PROFILE=rule_based`.

Rule-based visual explanations do not load a local VLM. In source/local
development, MobileSAM is disabled by default to avoid optional native runtime
startup costs. Packaged launchers can still enable bundled MobileSAM with
`SAFETRACE_MOBILESAM_ENABLED=auto`.

## Lightweight VLM Mode

`Lightweight VLM (256M)` is optional and can be slower than rule-based mode.
Use it only when local image-language explanations are worth the extra runtime
cost. SafeTrace now bounds local VLM generation with
`SAFETRACE_VLM_TIMEOUT_SECONDS`, caps per-job VLM attempts with
`SAFETRACE_VLM_MAX_EVIDENCE_FRAMES`, and caps total VLM worker time with
`SAFETRACE_VLM_JOB_TIMEOUT_SECONDS`; if VLM generation is slow, missing,
times out, or is rejected by quality checks, the job should complete with
rule-based explanations.

Recommended local defaults:

```cmd
set SAFETRACE_VLM_TIMEOUT_SECONDS=10
set SAFETRACE_VLM_MAX_FRAMES=5
set SAFETRACE_VLM_FRAME_LIMIT=5
set SAFETRACE_VLM_MAX_EVIDENCE_FRAMES=5
set SAFETRACE_VLM_JOB_TIMEOUT_SECONDS=0
```

## Progress And Timeouts

During long jobs, the backend updates `heartbeatAt` while analysis is still
running. The React frontend should continue polling beyond the normal wall-clock
timeout while the heartbeat is fresh, and should only fail once backend updates
become stale.

If a job appears stuck:

1. Check the backend terminal for OpenMP, MobileSAM, VLM, or model-loading
   warnings.
2. Open `GET http://127.0.0.1:8000/api/jobs/{job_id}` and inspect
   `currentStep`, `heartbeatAt`, `updatedAt`, and `metrics`.
3. Use Rule-based mode and retry the same clip to isolate VLM overhead.
4. Confirm the backend was started with `.venv\Scripts\python.exe`, not global
   `python`.

Do not commit generated frames, uploaded videos, reports, model files,
checkpoints, GGUF files, cache folders, or packaged outputs.
