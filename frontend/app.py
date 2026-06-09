"""SafeTrace Streamlit UI.

Run:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

# Allow running with `streamlit run frontend/app.py` from project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.pipeline import SafeTracePipeline  # noqa: E402
from src.utils import imread_rgb, resolve_device  # noqa: E402
from src.queue_manager import submit_task

st.set_page_config(page_title="SafeTrace", layout="wide", page_icon="🦺")


@st.cache_data(show_spinner=False)
def _cuda_status() -> dict:
    """Probe whether CUDA is *actually usable* on this host.

    `torch.cuda.is_available()` only checks that a driver + device are
    visible; it does NOT check that the installed PyTorch wheel ships
    kernels for the GPU's compute capability. On Blackwell (RTX 50-series,
    sm_120) the default cu12.1 wheels report `is_available() == True` but
    crash on the first conv with:

        CUDA error: no kernel image is available for execution on the device

    We run a tiny conv to detect that case up-front so the UI can disable
    the CUDA toggle and fall back to CPU instead of failing mid-pipeline.
    """
    info = {"available": False, "name": None, "compute_cap": None,
            "usable": False, "error": None}
    try:
        import torch
    except Exception as exc:
        info["error"] = f"torch import failed: {exc}"
        return info

    if not torch.cuda.is_available():
        return info
    info["available"] = True
    try:
        info["name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_cap"] = f"{major}.{minor}"
    except Exception:
        pass

    try:
        # Smallest possible conv that exercises the same kernel path the
        # SigLIP patch embedder uses.
        x = torch.randn(1, 3, 8, 8, device="cuda")
        w = torch.randn(2, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w)
        torch.cuda.synchronize()
        info["usable"] = True
    except Exception as exc:
        info["error"] = str(exc).splitlines()[0]
    return info


# --------------------------------------------------------------------------- #
# Cached pipeline (one instance per device choice)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading SafeTrace models …")
def get_pipeline(device: str) -> SafeTracePipeline:
    # Propagate the chosen device to the global SETTINGS so every component
    # picks it up (ClipEmbedder, YoloDetector, MobileSAM, VLM).
    SETTINGS.device = device
    os.environ["SAFETRACE_DEVICE"] = device
    return SafeTracePipeline()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🦺 SafeTrace")
    st.caption("Offline safety violation detection")

    fps = st.number_input("Frame sampling FPS", 0.1, 30.0, float(SETTINGS.frame_fps), 0.1)
    top_k = st.slider("Top-K frames to analyze", 1, 20, int(SETTINGS.top_k))
    enable_vlm = st.toggle("Enable VLM explanations", value=SETTINGS.enable_vlm,
                           help="Requires a local VLM checkpoint in checkpoints/vlm_model/")

    st.divider()
    st.subheader("Compute")
    cuda = _cuda_status()
    cuda_ok = cuda["usable"]
    device_options = ["cpu", "cuda", "auto"]
    default_choice = SETTINGS.device if SETTINGS.device in device_options else "cpu"
    device_choice = st.selectbox(
        "Device",
        device_options,
        index=device_options.index(default_choice),
        help=(
            "`cpu` is the safest default. `cuda` enables GPU acceleration but "
            "requires a PyTorch build with kernels for your GPU's compute "
            "capability \u2014 see docker/Dockerfile.blackwell for RTX 50-series. "
            "`auto` picks CUDA if usable, else CPU."
        ),
    )

    # Decide what device we will actually run on. If the user asked for cuda
    # (explicitly or via `auto`) but the probe failed, we silently fall back
    # to CPU and tell them why instead of crashing the pipeline later.
    if device_choice == "cpu":
        resolved_device = "cpu"
    elif cuda_ok:
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"
        if device_choice in {"cuda", "auto"}:
            if cuda["available"]:
                st.error(
                    f"GPU detected ({cuda['name']}, sm_{cuda['compute_cap']}) "
                    "but the installed PyTorch build has no kernels for it. "
                    "Falling back to CPU.\n\n"
                    f"Probe error: `{cuda['error']}`\n\n"
                    "Rebuild with `docker/Dockerfile.blackwell` for RTX 50-series "
                    "(sm_120), or use a PyTorch wheel matching your GPU."
                )
            else:
                st.warning(
                    "No CUDA device is visible to this container. "
                    "Falling back to CPU."
                )

    st.divider()
    st.subheader("System")
    st.write(f"Resolved device: `{resolved_device}`")
    if cuda["available"]:
        st.write(f"GPU: `{cuda['name']}` (sm_{cuda['compute_cap']})")
        usable_icon = "\u2705" if cuda_ok else "\u274c"
        st.write(f"GPU usable by PyTorch: {usable_icon}")
    else:
        st.write("GPU: not detected")
    st.write(f"Embedding model: `{SETTINGS.siglip_model_dir.name}`")
    st.write(f"Detector: `{SETTINGS.yolo_checkpoint.name}`")
    st.write("MobileSAM:", "\u2705" if SETTINGS.mobile_sam_checkpoint.exists() else "\u26a0\ufe0f missing")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Safety Violation Detection")
st.write(
    "Upload a **video** or one or more **images**, type a natural-language "
    "query (e.g. *worker without helmet*), and click **Analyze**."
)

uploaded = st.file_uploader(
    "Upload media",
    type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "avi", "mkv", "webm"],
    accept_multiple_files=True,
)

query = st.text_input(
    "Query",
    value="worker without helmet",
    help="Used by FAISS to retrieve the most relevant frames before detection.",
)

col_a, col_b = st.columns([1, 5])
analyze = col_a.button("🚀 Analyze", type="primary", use_container_width=True)
reset = col_b.button("Reset session", use_container_width=False)

if reset:
    st.session_state.clear()
    st.rerun()


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _persist_uploads(files) -> List[Path]:
    out: List[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="safetrace_"))
    for f in files:
        dst = tmp_dir / f.name
        with open(dst, "wb") as fh:
            # Read in chunks to prevent memory spikes on large batch uploads
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                fh.write(chunk)
        out.append(dst)
    return out


if analyze:
    if not uploaded:
        st.warning("Please upload at least one image or video.")
    elif not query.strip():
        st.warning("Please enter a query.")
    else:
        SETTINGS.enable_vlm = enable_vlm  # propagate UI toggle
        SETTINGS.frame_fps = float(fps)
        SETTINGS.top_k = int(top_k)

        pipeline = get_pipeline(resolved_device)
        pipeline.vlm.enabled = enable_vlm and pipeline.vlm._loaded

        with st.status("Running pipeline…", expanded=True) as status:
            st.write("• Saving uploads to disk safely")
            files = _persist_uploads(uploaded)

            st.write("• Task in queue: Extracting frames & building index")
            try:
                # Submit to the queue and block the UI thread until done
                submit_task(pipeline.ingest, files, fps=fps).result()
            except Exception as exc:
                status.update(label="Ingestion failed", state="error")
                st.exception(exc)
                st.stop()

            st.write(f"• Task in queue: Retrieving top-{top_k} frames for query: *{query}*")
            try:
                # Submit to the queue and block the UI thread until done
                results = submit_task(pipeline.analyze_query, query, k=top_k).result()
            except Exception as exc:
                status.update(label="Analysis failed", state="error")
                st.exception(exc)
                st.stop()

            status.update(label=f"Done — {len(results)} frames analyzed", state="complete")

        st.session_state["results"] = results
        st.session_state["query"] = query


# --------------------------------------------------------------------------- #
# Results display
# --------------------------------------------------------------------------- #
results = st.session_state.get("results")
if results:
    st.subheader(f"Results for: *{st.session_state.get('query', '')}*")

    report_bytes = json.dumps(results, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download JSON report",
        data=report_bytes,
        file_name="safetrace_report.json",
        mime="application/json",
    )

    for idx, frame in enumerate(results, start=1):
        st.markdown(f"### Frame {idx} — `{frame['frame_id']}`  (score `{frame['score']:.3f}`)")

        c_img, c_meta = st.columns([2, 1], gap="large")
        with c_img:
            img_path = frame.get("annotated_path") or frame.get("frame_path")
            if img_path and Path(img_path).exists():
                st.image(imread_rgb(img_path), caption=Path(img_path).name,
                         use_column_width=True)

        with c_meta:
            violations = frame.get("violations", [])
            if violations:
                st.error(f"{len(violations)} violation(s) detected")
                for v in violations:
                    st.markdown(
                        f"**{v['name']}** ({v['severity']})  \n"
                        f"{v['description']}  \n"
                        f"Confidence: `{v['confidence']:.2f}`"
                    )
                    st.json(v["evidence"])
            else:
                st.success("No violations detected")

            if frame.get("explanation"):
                st.markdown("**Explanation**")
                st.write(frame["explanation"])

            with st.expander("Detections"):
                st.json(frame.get("detections", []))

        st.divider()
else:
    st.info("Upload media and click **Analyze** to begin.")
