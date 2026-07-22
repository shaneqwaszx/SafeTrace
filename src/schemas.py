"""Shared dataclasses / typed dicts used across SafeTrace modules."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Detection:
    """Single object instance detected in a frame."""

    label: str                       # canonical SafeTrace label (see config.CLASS_ALIASES)
    raw_label: str                   # original model label
    confidence: float
    bbox: List[float]                # [x1, y1, x2, y2] in pixel coords
    coarse_mask: Optional[np.ndarray] = None  # HxW bool, original frame size
    refined_mask: Optional[np.ndarray] = None  # HxW bool, original frame size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "raw_label": self.raw_label,
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
            "has_refined_mask": self.refined_mask is not None,
        }


@dataclass
class Violation:
    name: str                                # e.g. "helmet_missing"
    description: str
    severity: str = "medium"                 # low | medium | high | critical
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "confidence": float(self.confidence),
            "evidence": self.evidence,
        }


@dataclass
class FrameAnalysis:
    frame_id: str
    frame_path: str
    score: float                              # FAISS similarity (or 1.0 for direct)
    detections: List[Detection] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None
    annotated_path: Optional[str] = None
    scene_applicability: Dict[str, Any] = field(default_factory=dict)
    suppressed_findings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_path": self.frame_path,
            "score": float(self.score),
            "detections": [d.to_dict() for d in self.detections],
            "violations": [v.to_dict() for v in self.violations],
            "explanation": self.explanation,
            "explanation_source": self.explanation_source,
            "annotated_path": self.annotated_path,
            "scene_applicability": self.scene_applicability,
            "suppressed_findings": self.suppressed_findings,
        }
