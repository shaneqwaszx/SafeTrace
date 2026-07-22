"""Shared utilities: device selection, frame extraction, IoU, overlay drawing."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .preprocessing import build_processing_metadata

logger = logging.getLogger("safetrace.utils")


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(pref: str = "auto") -> str:
    """Pick 'cuda' if available else 'cpu'. ``pref`` may force a value."""
    pref = (pref or "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch always present in container
        return "cpu"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def imread_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def write_json(path: str | Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    max_frames: int = 600,
    prefix: Optional[str] = None,
) -> List[Path]:
    """Sample frames from ``video_path`` at ``fps`` and write to ``out_dir``.

    Returns the list of written frame paths in capture order.
    """
    frames, _ = extract_frames_with_metadata(
        video_path,
        out_dir,
        fps=fps,
        max_frames=max_frames,
        prefix=prefix,
    )
    return frames


def extract_frames_with_metadata(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    max_frames: int = 600,
    prefix: Optional[str] = None,
    max_duration_seconds: float | None = None,
    uniform_over_video: bool = False,
) -> Tuple[List[Path], dict]:
    """Sample frames and return explicit sampling metadata."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (source_frame_count / src_fps) if source_frame_count and src_fps else None
    if max_duration_seconds and duration_seconds and duration_seconds > max_duration_seconds:
        cap.release()
        raise RuntimeError(
            f"Video duration {duration_seconds:.1f}s exceeds configured limit "
            f"of {max_duration_seconds:.1f}s."
        )

    step = max(int(round(src_fps / max(fps, 1e-6))), 1)
    prefix = prefix or video_path.stem

    saved: List[Path] = []
    sampled_frame_records: List[dict] = []
    sampled = 0
    if uniform_over_video and source_frame_count > 0:
        candidate_indices = list(range(0, source_frame_count, step))
        if len(candidate_indices) > max_frames:
            chosen_positions = np.linspace(0, len(candidate_indices) - 1, max_frames)
            target_indices = [candidate_indices[int(round(pos))] for pos in chosen_positions]
        else:
            target_indices = candidate_indices
        for frame_index in dict.fromkeys(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            timestamp_seconds = float(frame_index) / max(src_fps, 1e-6)
            filename_seconds = int(round(timestamp_seconds))
            out_path = out_dir / f"{prefix}_{filename_seconds:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            sampled_frame_records.append(
                {
                    "framePath": str(out_path),
                    "sourceFrameIndex": int(frame_index),
                    "timestampSeconds": timestamp_seconds,
                }
            )
            sampled += 1
            if sampled >= max_frames:
                break
    else:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                out_path = out_dir / f"{prefix}_{sampled:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved.append(out_path)
                sampled_frame_records.append(
                    {
                        "framePath": str(out_path),
                        "sourceFrameIndex": int(idx),
                        "timestampSeconds": float(idx) / max(src_fps, 1e-6),
                    }
                )
                sampled += 1
                if sampled >= max_frames:
                    break
            idx += 1
    cap.release()
    logger.info("Extracted %d frames from %s", len(saved), video_path.name)
    metadata = build_processing_metadata(
        sampled_frame_count=len(saved),
        sampling_strategy="uniform_fixed_fps" if uniform_over_video else "fixed_fps",
        fps=fps,
        max_frames=max_frames,
        embedding_batch_size=1,
        embedding_window_size=1,
        embedding_window_stride=1,
        embedding_pooling_strategy="mean",
        processing_window_count=len(saved),
        source_video_duration_seconds=duration_seconds,
        source_video_frame_count=source_frame_count,
    )
    metadata.update(
        {
            "sourceVideoPath": str(video_path),
            "sourceVideoFps": src_fps,
            "frameStep": step,
            "uniformOverVideo": bool(uniform_over_video),
            "sampledFrames": sampled_frame_records,
        }
    )
    return saved, metadata


def collect_inputs(paths: Iterable[str | Path]) -> Tuple[List[Path], List[Path]]:
    """Split a list of inputs into (videos, images)."""
    videos, images = [], []
    for p in paths:
        p = Path(p)
        if is_video(p):
            videos.append(p)
        elif is_image(p):
            images.append(p)
    return videos, images


# --------------------------------------------------------------------------- #
# Mask / geometry
# --------------------------------------------------------------------------- #
def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def torso_proxy_from_person(person_mask: np.ndarray) -> np.ndarray:
    """Approximate a torso region as the central vertical band of a person mask."""
    if person_mask is None or not person_mask.any():
        return np.zeros_like(person_mask, dtype=bool) if person_mask is not None else None
    ys, xs = np.where(person_mask)
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min
    # torso ~ top 25% .. 70% of person bbox
    t_top = int(y_min + 0.25 * h)
    t_bot = int(y_min + 0.70 * h)
    out = np.zeros_like(person_mask, dtype=bool)
    out[t_top:t_bot, :] = person_mask[t_top:t_bot, :]
    return out


def head_proxy_from_person(person_mask: np.ndarray) -> np.ndarray:
    if person_mask is None or not person_mask.any():
        return np.zeros_like(person_mask, dtype=bool) if person_mask is not None else None
    ys, _ = np.where(person_mask)
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min
    h_bot = int(y_min + 0.20 * h)
    out = np.zeros_like(person_mask, dtype=bool)
    out[y_min:h_bot, :] = person_mask[y_min:h_bot, :]
    return out


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #
_PALETTE = [
    (255, 56, 56), (255, 159, 56), (255, 218, 56), (138, 255, 56),
    (56, 255, 167), (56, 218, 255), (56, 95, 255), (167, 56, 255),
    (255, 56, 218), (255, 56, 95),
]


def _color_for(label: str) -> Tuple[int, int, int]:
    return _PALETTE[hash(label) % len(_PALETTE)]


def draw_overlays(image_rgb: np.ndarray, detections, alpha: float = 0.4) -> np.ndarray:
    """Render boxes + masks for a list of Detection objects."""
    out = image_rgb.copy()
    overlay = out.copy()
    for det in detections:
        color = _color_for(det.label or det.raw_label)
        mask = det.refined_mask if det.refined_mask is not None else det.coarse_mask
        if mask is not None and mask.any():
            overlay[mask.astype(bool)] = color
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label or det.raw_label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out
