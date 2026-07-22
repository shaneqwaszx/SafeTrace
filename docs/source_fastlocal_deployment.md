# SafeTrace Fast Local Source Deployment

Use `README_SOURCE_HANDOFF.md` as the authoritative setup guide for this
source-only delivery. The backend is FastAPI/Uvicorn and the active frontend is
the React/Vite app in `frontend-react/`.

This profile excludes local VLM models and the chatbot. Start the backend with
`config/safetrace.fastlocal.env.example` copied to `config/safetrace.env`; that
template disables both features and leaves MobileSAM disabled by default.

The supplied checkpoints are local files:

- `checkpoints/yolov8s-seg.pt`: fallback detector.
- `checkpoints/siglip-base-patch16-224`: embedding model.
- `checkpoints/mobile_sam.pt`: optional refinement checkpoint.

To enable MobileSAM later, install its optional package and change the matching
environment settings only after testing the target host. Fast Local Analysis
must remain usable with it disabled.

For production hosting, use persistent storage for `SAFETRACE_DATA_DIR`, set
the React API base before `npm run build`, restrict CORS, and protect the API
with platform authentication. The current filesystem queue supports one backend
instance; do not scale horizontally without external queue and shared storage.
