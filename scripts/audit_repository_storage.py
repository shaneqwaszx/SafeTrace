"""Read-only, no-follow SafeTrace repository storage inventory."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPARSE_ATTRIBUTE = 0x400


def _is_reparse(path: Path) -> bool:
    stat = path.lstat()
    return path.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & REPARSE_ATTRIBUTE)


def _measure(path: Path) -> tuple[int, int, int, list[str]]:
    size = 0
    files = 0
    dirs = 0
    errors: list[str] = []
    if path.is_file():
        return path.stat().st_size, 1, 0, []
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            if _is_reparse(current):
                errors.append(f"reparse_not_followed:{current}")
                continue
            dirs += 1
            with os.scandir(current) as entries:
                for entry in entries:
                    child = Path(entry.path)
                    try:
                        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                        if entry.is_symlink() or bool(attributes & REPARSE_ATTRIBUTE):
                            errors.append(f"reparse_not_followed:{child}")
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(child)
                        elif entry.is_file(follow_symlinks=False):
                            files += 1
                            size += entry.stat(follow_symlinks=False).st_size
                    except OSError as exc:
                        errors.append(f"{child}:{type(exc).__name__}:{exc}")
        except OSError as exc:
            errors.append(f"{current}:{type(exc).__name__}:{exc}")
    return size, files, dirs, errors


def _category(path: Path) -> tuple[str, str]:
    name = path.name.casefold()
    if name in {"models", "checkpoints", ".venv"}:
        return "protected", "models/checkpoints/environment are protected"
    if name in {"src", "tests", "scripts", "frontend-react", "docs", "config", "training", "evaluation"}:
        return "source", "source, tests, configuration, or evaluation material"
    if name in {"data"}:
        return "runtime_data", "job, upload, evidence, and report data are protected"
    if name in {"dist", "build", "packaging"}:
        return "release_or_build", "release/package/build output; review before deletion"
    if name.startswith(".tmp_phase_") or name.startswith("tmp_pytest") or name.startswith("_pytest") or name.startswith("tmp_safetrace"):
        return "pytest_or_codex_temporary", "known generated temporary naming pattern"
    if name in {".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis", "htmlcov"}:
        return "reproducible_cache", "reproducible tool cache"
    if name == ".ai-pipeline":
        return "phase_reports", "phase reports are protected; 006_work is separately managed"
    if name == ".git":
        return "protected", "Git history"
    return "unknown", "purpose requires review"


def inventory() -> dict[str, Any]:
    tracked_names = {
        line.split("/", 1)[0].casefold()
        for line in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        if line
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(ROOT.iterdir(), key=lambda item: item.name.casefold()):
        try:
            size, files, dirs, errors = _measure(path)
            modified = datetime.fromtimestamp(path.lstat().st_mtime, timezone.utc).isoformat()
            reparse = _is_reparse(path)
        except OSError as exc:
            size, files, dirs, errors = 0, 0, 0, [f"{type(exc).__name__}:{exc}"]
            modified = None
            reparse = False
        relative = path.name
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
        category, reason = _category(path)
        rows.append({
            "path": relative,
            "type": "directory" if path.is_dir() else "file",
            "sizeBytes": size,
            "fileCount": files,
            "directoryCount": dirs,
            "lastModified": modified,
            "gitTrackedContains": relative.casefold() in tracked_names,
            "gitIgnored": ignored,
            "reparsePoint": reparse,
            "category": "inaccessible" if errors and not files and not dirs else category,
            "classificationReason": reason,
            "errors": errors,
        })
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "totalMeasuredBytes": sum(int(row["sizeBytes"]) for row in rows),
        "entries": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = inventory()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "path", "type", "sizeBytes", "fileCount", "directoryCount", "lastModified",
        "gitTrackedContains", "gitIgnored", "reparsePoint", "category", "classificationReason", "errors",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["entries"]:
            writer.writerow({**row, "errors": " | ".join(row["errors"])})
    print(json.dumps({"totalMeasuredBytes": payload["totalMeasuredBytes"], "entries": len(payload["entries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
