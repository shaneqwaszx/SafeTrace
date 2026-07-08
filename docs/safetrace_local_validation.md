# SafeTrace Local Validation Checklist

Use this checklist before building another package.

## Start Local Runtime

For source-development validation of chat and Lightweight VLM worker behavior,
prefer the helper script:

```cmd
scripts\start_safetrace_dev_backend.bat
```

The helper starts from the repository root, prefers `.venv\Scripts\python.exe`,
sets CPU Safe Mode with MobileSAM disabled, enables the local Lightweight VLM
worker, configures packaged llama.cpp chat, checks whether `llama_cpp` imports
in that Python, then starts FastAPI on `127.0.0.1:8000`. If `.venv` is missing,
it warns before falling back to system Python. If `llama_cpp` is missing, it
prints the install command for the selected interpreter, normally:
`.venv\Scripts\python.exe -m pip install llama-cpp-python`.

The helper sets `SAFETRACE_VLM_FRAME_LIMIT=5`, keeps
`SAFETRACE_VLM_MAX_EVIDENCE_FRAMES=5` for compatibility, and sets
`SAFETRACE_VLM_JOB_TIMEOUT_SECONDS=0`. Lightweight VLM remains optional and
watchdog-protected: it is attempted only on eligible selected evidence frames up
to the frame limit. Lower-priority cards should use readable copy such as local
visual review not run for this frame, while raw reasons like
`visual_review_frame_limit_reached`, `no_eligible_violation`, worker timeout, or
quality rejection remain in technical evidence. A mixed result with VLM
assistance on some frames and Fast Local Analysis on others is expected when
only some frames produce accepted, safety-relevant VLM output.

The packaged tester runtime is different: use the generated
`SafeTraceLauncher.bat` inside a release package when validating packaged
Safe Mode, supervisor restart behavior, and packaged asset resolution.

Starting the source backend manually with plain `python -m uvicorn` can make
`/api/system/status` report `runtime_layout=source`, a system Python such as
`C:\Python312\python.exe`, missing `.venv`, missing `llama_cpp`, Lightweight VLM
selected but `lightweightVlmWorkerEnabled=false`, and
`lightweightVlmExplanationSource=rule_based`. That state is useful diagnostics,
but it is not the intended local development profile for validating VLM/chat.
For Lightweight VLM worker validation, `/api/system/status` should instead show
`safeMode=true`, `device=cpu`, and `lightweightVlmWorkerEnabled=true`.

The helper also checks core backend imports before starting uvicorn. If
installing `llama-cpp-python` pulls NumPy 2.x into `.venv`, FAISS may fail with
`numpy.core.multiarray failed to import`. Repair the local dev environment with:

```cmd
.venv\Scripts\python.exe -m pip install "numpy<2"
```

Backend:

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info
```

Safe local validation mode:

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set SAFETRACE_ANALYSIS_SAFE_MODE=true
set SAFETRACE_DEVICE=cpu
set SAFETRACE_MOBILESAM_ENABLED=false
set SAFETRACE_VLM_ENABLED=false
set SAFETRACE_VLM_PROFILE=rule_based
.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info
```

Use safe local validation mode when confirming the stable RC Chat + Rule-Based
path. In this mode SafeTrace uses CPU, rule-based explanations, detector-box
evidence, object/rule-based frame ranking, and skips VLM, MobileSAM, and
SigLIP/FAISS semantic embedding work. Safe Mode samples across long videos and
ranks frames with detector/rule evidence such as people, torso/hand evidence,
driver-cabin cues, query-related objects, violation candidates, and temporal
diversity. The job status diagnostics should show `safeMode=true`,
`embeddingRequested=false`, `vlmLoaded=false`, and `mobileSamLoaded=false`.
Frame technical evidence should include `safeFrameRanking`, `frameRankingScore`,
`rankingReason`, `queryIntent`, and `selectedFor`.

For long driving videos, use query wording that matches the scene, such as
`driver without seatbelt` or `driver using phone while driving`. A generic query
like `worker without seatbelt` can still work when person/torso evidence exists,
but driver-specific wording helps the lightweight ranker prioritize in-car
evidence over road/windshield frames.

MobileSAM can improve mask and evidence quality after relevant frames are found,
but it does not discover frames by itself. Lightweight VLM remains experimental
and should stay explicitly activated with rule-based fallback; Safe Mode still
suppresses VLM execution for stability.

Prepared release-package profiles before packaging:

- `SafeTrace_RC_SafeMode_RuleBased`: main tester release profile with CPU Safe
  Mode, improved object/rule frame ranking, rule-based explanations, MobileSAM
  packaged but disabled by default, VLM disabled, and chatbot enabled.
- `SafeTrace_RC_MobileSAM_RuleBased`: optional future profile for CPU Safe Mode
  with MobileSAM package support when explicitly enabled and VLM disabled.
- `SafeTrace_RC_MobileSAM_LightweightVLM_Experimental`: CPU profile with the
  same improved ranking and MobileSAM package support. Lightweight VLM remains
  optional/experimental only after subprocess-safe preflight; rule-based
  fallback stays active and Enhanced VLM is excluded.

Verify the assistant runtime with the same Python first:

```cmd
.venv\Scripts\python.exe -c "import sys; print(sys.executable); import llama_cpp; print('llama_cpp ok')"
```

Frontend:

```cmd
cd frontend-react
npm.cmd run dev -- --host 127.0.0.1
```

## Checks

1. Confirm the React UI reports the local backend as connected.
2. Open the Visual explanations card.
3. For safe local validation, turn Visual explanations OFF or keep Rule-based
   selected and run a single-video analysis.
4. Select Lightweight VLM (256M), activate it, and confirm the sidebar says the
   backend will attempt VLM while evidence cards still label rule-based fallback
   if generation does not succeed. Lightweight VLM is a slower experimental
   mode; keep Rule-based selected for fastest local testing.
5. Open SafeTrace Assistant.
6. If `llama_cpp` is missing, confirm limited SafeTrace help is available and
   runtime diagnostics show the backend Python executable, import status, setup
   command, and restart requirement. Setup instructions should mention:
   `.venv\Scripts\python.exe -m pip install llama-cpp-python`
7. Ask custom typed SafeTrace questions and the two in-car suggested questions.
8. Run a longer local video and confirm progress shows stage, elapsed time, and
   heartbeat freshness instead of appearing frozen.
9. Open Safety Insights and inspect metrics, hotspots, severity, ranked videos,
   recent analyses, and the per-video table.
10. Download CSV, Markdown, and JSON Safety Insights reports.
11. Stop the backend, confirm the live frontend locks analysis controls, then
   restart and reconnect.
12. Run:

```cmd
python scripts\evaluate_in_car_violations.py --samples-dir data\manual_eval
```

Treat `NOT_TESTED_NO_SAMPLE` as expected when no local manual evaluation outputs
have been placed under `data\manual_eval`.

## Reminder

Do not commit generated media, uploaded videos, reports, cache folders,
packaged outputs, checkpoints, model weights, or GGUF files.
