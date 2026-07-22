"""Run the Phase W.1 disposable cleanup provenance certification fixture."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import manage_dev_workspace as workspace


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marked(path: Path, payload: str) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / workspace.MARKER).write_text("Phase W.1 disposable fixture\n", encoding="ascii")
    (path / "payload.txt").write_text(payload, encoding="ascii")
    return path


def _manifest(run: Path, reports: Path, label: str) -> tuple[Path, str, dict[str, Any]]:
    payload = workspace.create_approval_manifest(
        candidate_root=run,
        repository_root=ROOT,
        work_root_path=workspace.work_root(),
        minimum_age_seconds=0,
    )
    path = reports / f"fixture_{label}_approval_manifest.json"
    digest = workspace.write_approval_manifest(path, payload)
    return path, digest, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".ai-pipeline/005_tasks/phase_w1_cleanup_provenance_closeout/disposable_fixture_certification.json",
    )
    args = parser.parse_args()
    reports = args.output.resolve().parent
    reports.mkdir(parents=True, exist_ok=True)
    root = workspace.initialize_work_root()
    fixture_parent = root / "cleanup_certification"
    fixture_parent.mkdir(exist_ok=True)
    run = fixture_parent / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    run.mkdir()
    (run / workspace.MARKER).write_text("Phase W.1 fixture root\n", encoding="ascii")

    approved = _marked(run / ".tmp_phase_w1_a", "approved unchanged")
    modified = _marked(run / ".tmp_phase_w1_b", "approved then modified")
    ambiguous = run / ".tmp_phase_w1_c"
    ambiguous.mkdir()
    (ambiguous / "payload.txt").write_text("initially unmarked ambiguous", encoding="ascii")
    reparse_candidate = _marked(run / ".tmp_phase_w1_e", "approved then reparse")

    first_manifest, first_digest, first_payload = _manifest(run, reports, "first")
    (modified / "payload.txt").write_text("modified after approval", encoding="ascii")
    (ambiguous / workspace.MARKER).write_text("became accessible after approval\n", encoding="ascii")
    later = _marked(run / ".tmp_phase_w1_d", "created after approval")

    reparse_supported = False
    target = _marked(run / "fixture_target", "safe reparse target")
    try:
        for child in reparse_candidate.iterdir():
            child.unlink()
        reparse_candidate.rmdir()
        os.symlink(target, reparse_candidate, target_is_directory=True)
        reparse_supported = True
    except OSError:
        # The unchanged E fixture remains eligible on Windows configurations
        # where unprivileged directory symlinks are unavailable.
        pass

    first_result = workspace.apply_approval_manifest(
        approval_manifest=first_manifest,
        approval_sha256=first_digest,
        certificate_path=reports / "fixture_first_apply_certificate.json",
    )

    second_manifest, second_digest, second_payload = _manifest(run, reports, "second")
    second_result = workspace.apply_approval_manifest(
        approval_manifest=second_manifest,
        approval_sha256=second_digest,
        certificate_path=reports / "fixture_second_apply_certificate.json",
    )

    expected = {
        "approvedDeleted": not approved.exists(),
        "modifiedInitiallySkipped": modified.exists() or any("drift_" in item.get("reason", "") for item in first_result["skipped"]),
        "ambiguousInitiallyUnapproved": all(item.get("canonicalPath") != str(ambiguous.resolve()) for item in first_payload["candidates"]),
        "laterInitiallyUnapproved": all(item.get("canonicalPath") != str(later.resolve()) for item in first_payload["candidates"]),
        "ambiguousSurvivedFirstApply": ambiguous.exists() or not any(path == str(ambiguous.resolve()) for path in first_result["deletedPaths"]),
        "laterSurvivedFirstApply": later.exists() or not any(path == str(later.resolve()) for path in first_result["deletedPaths"]),
        "reparseRejectedWhenSupported": (not reparse_supported) or any(
            item.get("reason") == "reparse_point_after_approval" for item in first_result["rejected"]
        ),
        "noUnapprovedDeletionFirst": not first_result["unapprovedDeletions"],
        "noUnapprovedDeletionSecond": not second_result["unapprovedDeletions"],
    }
    report = {
        "generatedAt": _utc(),
        "fixtureRoot": str(run),
        "reparseSupported": reparse_supported,
        "first": {"manifest": str(first_manifest), "sha256": first_digest, "candidateCount": first_payload["candidateCount"], "result": first_result},
        "second": {"manifest": str(second_manifest), "sha256": second_digest, "candidateCount": second_payload["candidateCount"], "result": second_result},
        "expected": expected,
        "passed": all(expected.values()),
        "retainedFixturePaths": [
            str(path) for path in (reparse_candidate, target) if path.exists() or path.is_symlink()
        ],
    }
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "fixtureRoot": report["fixtureRoot"], "expected": expected}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
