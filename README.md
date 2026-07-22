# SafeTrace

Modular, **fully-offline**, GPU-ready safety-violation detection system.
Upload videos or images, the pipeline retrieves the most relevant frames
with SigLIP + FAISS, detects objects with YOLOv9-seg + MobileSAM, applies
a deterministic rule engine, and (optionally) generates natural-language
explanations with a local VLM. Results are served through a Streamlit UI.

## Full-local teammate start

1. Checkout or pull the `integrate-frontend-backend` branch.
2. Prepare the supported repository `.venv` and place the external model assets in the documented local paths. Model weights are not stored in Git.
3. Run `START_SAFETRACE_LOCAL.bat` from the repository root, or run `scripts\start_safetrace_full_local.bat` directly.
4. In another terminal, run `cd frontend-react`, `npm install`, and `npm run dev`.
5. Open `http://127.0.0.1:5173`.

The Windows launcher runs the strict `local_full` readiness preflight before starting the backend. It requires the configured CUDA, detector, MobileSAM, local assistant, and visual-explanation assets and does not silently fall back to CPU.

---

## Architecture

```
User upload (video / images)
        │
        ▼
Frame Sampler (configurable FPS)
        │
        ▼
SigLIP image embeddings ──► FAISS IndexFlatIP (data/index.faiss)
        │
        ▼
semantic_search(query, k) ─ top-K candidate frames
        │
        ▼
YOLOv9-seg ── boxes + coarse masks
        │
        ▼
MobileSAM (vit_t) ── refined binary masks per box
        │
        ▼
Rule engine
   • helmet_missing       IoU(head, helmet)        < 0.20
   • hands_off_wheel      IoU(hand, wheel)         < 0.10
   • phone_use            IoU(phone, hand)         > 0.30
   • seatbelt_missing     IoU(seatbelt, torso)     < 0.20
        │
        ▼
Optional local VLM — natural language summary
        │
        ▼
Streamlit UI: overlays, evidence table, JSON download
```

---

## Project layout

```
SafeTrace/
├── src/
│   ├── clip_embedder.py         # SigLIP / CLIP loader + batch embeddings
│   ├── faiss_index.py           # IndexFlatIP build / load / semantic_search
│   ├── yolo_detector.py         # YOLOv9-seg (Ultralytics) wrapper
│   ├── mobile_sam_segmenter.py  # MobileSAM box → refined mask
│   ├── rule_engine.py           # IoU-based safety rules
│   ├── vlm_reasoner.py          # Optional local VLM explanations
│   ├── pipeline.py              # End-to-end orchestrator + analyze_query()
│   ├── schemas.py               # Detection / Violation / FrameAnalysis
│   ├── utils.py                 # Frame extraction, IoU, overlays, IO
│   └── config.py                # Central settings (env-overridable)
├── frontend/
│   └── app.py                   # Streamlit UI
├── data/                        # Frames, embeddings, FAISS index, annotated outputs
├── checkpoints/                 # Local model weights (mounted in Docker)
├── docker/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── main.py                      # CLI: ingest / query / ui
└── README.md
```

---

## Local checkpoints (required for offline mode)

Place each checkpoint under `checkpoints/`:

| Component   | Default path                                            | How to obtain                               |
|-------------|---------------------------------------------------------|---------------------------------------------|
| SigLIP      | `checkpoints/siglip-base-patch16-224/`                  | `huggingface-cli download google/siglip-base-patch16-224 --local-dir checkpoints/siglip-base-patch16-224` |
| YOLOv9-seg  | `checkpoints/yolov9c-seg.pt`                            | Ultralytics release asset                   |
| YOLOv8-seg* | `checkpoints/yolov8s-seg.pt` *(fallback)*               | Ultralytics release asset                   |
| MobileSAM   | `checkpoints/mobile_sam.pt`                             | https://github.com/ChaoningZhang/MobileSAM  |
| VLM (opt.)  | `models/vlm/` or local Ollama at `http://127.0.0.1:11434` | Packaged local VLM assets, or optional Ollama vision model outside Git |

All paths are overridable via environment variables — see `src/config.py`.

---

## Quick start (local Python)

```bash
pip install -r docker/requirements.txt   # or your own pinned env
python main.py ingest path/to/video.mp4 path/to/image.jpg
python main.py query "worker without helmet" --k 5 --output report.json
python main.py ui                        # http://localhost:8501
```

---

## Docker (GPU + offline)

Two Dockerfiles are provided:

| File                                | Base                              | When to use                                                                 |
|-------------------------------------|-----------------------------------|-----------------------------------------------------------------------------|
| `docker/Dockerfile`                 | `pytorch/pytorch:2.2.0-cuda12.1`  | Default. CPU-safe; CUDA works on Ampere/Ada (sm_80–sm_89).                   |
| `docker/Dockerfile.blackwell`       | `nvidia/cuda:12.8.0` + torch 2.7  | Required for **RTX 50-series / Blackwell (sm_120)** — the default image lacks kernels for those GPUs and will raise *"CUDA error: no kernel image is available for execution on the device"*. |

Build the default image:
```bash
docker build -f docker/Dockerfile -t safetrace:latest .
```

Build the Blackwell-capable image (RTX 50-series):
```bash
docker build -f docker/Dockerfile.blackwell -t safetrace:blackwell .
# then change `image:` in docker-compose.yml and set SAFETRACE_DEVICE: "auto"
```

Run with GPU and your local checkpoints / data mounted:
```bash
docker run --gpus all --rm -it \
    -p 8501:8501 \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    -v "$(pwd)/data:/app/data" \
    safetrace:latest
```
Open http://localhost:8501.

CLI variants:
```bash
docker run --gpus all --rm \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    -v "$(pwd)/data:/app/data" \
    safetrace:latest ingest /app/data/in.mp4

docker run --gpus all --rm \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    -v "$(pwd)/data:/app/data" \
    safetrace:latest query "driver using phone" --k 5
```

---

## Configuration

All settings live in `src/config.py` and accept environment overrides:

| Env var                      | Default                                          | Purpose                          |
|------------------------------|--------------------------------------------------|----------------------------------|
| `SAFETRACE_DEVICE`           | `auto`                                           | `auto` / `cuda` / `cpu` (also switchable from the UI sidebar) |
| `SAFETRACE_OFFLINE`          | `1`                                              | Force HF + transformers offline  |
| `SAFETRACE_ENABLE_VLM`       | `0`                                              | Enable optional VLM explanations |
| `SAFETRACE_VLM_ENABLED`      | `auto`                                           | Optional local VLM availability mode |
| `SAFETRACE_VLM_PROVIDER`     | `auto`                                           | Prefer local VLM, then optional Ollama, then rule-based fallback |
| `SAFETRACE_VLM_MODEL_PATH`   | `models/vlm`                                     | Optional packaged local VLM asset dir |
| `SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED` | `0`                              | Run Lightweight VLM in a crash-isolated worker process |
| `SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS` | `60`                    | Per-worker Lightweight VLM subprocess timeout |
| `SAFETRACE_VLM_FRAME_LIMIT` | `5`                                            | Maximum selected evidence frames that may run local visual review |
| `SAFETRACE_VLM_MAX_EVIDENCE_FRAMES` | `5`                                    | Legacy-compatible alias for the local visual review frame limit |
| `SAFETRACE_VLM_JOB_TIMEOUT_SECONDS` | `0`                                    | Shared local VLM job budget; `0` disables it in favor of per-attempt watchdogs |
| `SAFETRACE_VLM_MAX_QUALITY_FAILURES` | `1`                                  | Stop VLM attempts after repeated generic/low-quality outputs |
| `SAFETRACE_VLM_OLLAMA_BASE_URL` | `http://127.0.0.1:11434`                      | Local Ollama vision runtime URL |
| `SAFETRACE_VLM_MODEL`        | `local-vlm`                                      | Packaged VLM label; set to a local Ollama model only when using Ollama |
| `SAFETRACE_FPS`              | `1.0`                                            | Frame sampling FPS               |
| `SAFETRACE_TOPK`             | `5`                                              | Default semantic-search k        |
| `SAFETRACE_SIGLIP_DIR`       | `checkpoints/siglip-base-patch16-224`            | Embedding model dir              |
| `SAFETRACE_YOLO_CKPT`        | `checkpoints/yolov9c-seg.pt`                     | Detector weights                 |
| `SAFETRACE_MOBILESAM_ENABLED` | `auto`                                          | Optional MobileSAM refinement mode |
| `SAFETRACE_MOBILESAM_CHECKPOINT` | `checkpoints/mobile_sam.pt`                   | Optional MobileSAM weights       |
| `SAFETRACE_MSAM_CKPT`        | `checkpoints/mobile_sam.pt`                      | Legacy MobileSAM checkpoint env  |
| `SAFETRACE_VLM_DIR`          | `models/vlm`                                     | Optional VLM dir compatibility alias |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `51200`                                    | Per-file upload limit in **MB** (default 50 GB) |
| `STREAMLIT_SERVER_MAX_MESSAGE_SIZE`| `51200`                                    | Streamlit websocket message cap in **MB**       |

### Frontend controls

The Streamlit sidebar exposes:
- **Frame sampling FPS / Top-K / VLM toggle** — same semantics as the CLI flags.
- **Compute › Device** — switch between `cpu`, `cuda`, and `auto` at runtime.
  Changing the device transparently rebuilds the model cache. If `cuda` is
  selected on an image without GPU kernels for your card, the UI warns and
  falls back to CPU. For RTX 50-series GPUs use the Blackwell image (above).
- **Uploads** — the server cap is lifted to 50 GB; tune via
  `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` (MB).

---

## Programmatic usage

```python
from src.pipeline import SafeTracePipeline

pipe = SafeTracePipeline()
pipe.ingest(["video.mp4", "image.jpg"], fps=1.0)
results = pipe.analyze_query("driver not wearing seatbelt", k=5)
# results: List[Dict] with frame metadata, detections, violations, explanation
```

Or, matching the spec signature:
```python
from src.pipeline import analyze_query
results = analyze_query("worker without helmet")
```

---

## Notes

- The system never reaches the network at runtime: HuggingFace offline flags
  are set in both the container and `src/config.py`.
- MobileSAM and the VLM are **optional**: if checkpoints are missing the
  pipeline degrades gracefully (YOLO masks only, deterministic explanations).
- Packaged releases should include `checkpoints/mobile_sam.pt`,
  `models/chat/`, and `models/vlm/` beside the local runtime, but Git must not
  track checkpoints or model weights. VLM explanations default to `auto`:
  packaged local VLM provider first, optional local Ollama second, then
  rule-based fallback. SafeTrace never uses cloud VLM APIs and Ollama is not
  required for the no-extra-steps release flow.
- YOLO label spaces vary across checkpoints; `src/config.py` contains a
  configurable `CLASS_ALIASES` map normalizing labels for the rule engine.
- The React frontend result cache is local to the user's browser. It can store
  result JSON and backend evidence/report URLs for queue switching, but it does
  not store raw uploaded videos, copied evidence image bytes, model files,
  credentials, or secrets.
