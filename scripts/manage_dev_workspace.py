"""Audit and safely maintain SafeTrace development work directories.

Cleanup is intentionally manifest-driven.  An apply operation only examines
entries named in a SHA-256-approved dry-run manifest; it never discovers new
filesystem candidates while applying that manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
WORK_SUBDIRS = ("pytest", "codex", "smoke", "build", "logs", "quarantine")
MARKER = ".safetrace-dev-work"
ROOT_PATTERNS = (".tmp_phase_*", "tmp_pytest_*", "_pytest*", "tmp_safetrace*", "tmp_pytest_phase*")
PROTECTED_PARTS = {
    ".git", ".venv", "models", "checkpoints", "data", "uploads", "exports",
    "release_archives", "dist", "src", "tests", "frontend-react", "packaging",
}
REPARSE_ATTRIBUTE = 0x400
MANIFEST_SCHEMA_VERSION = "2.0.0"
TOOL_VERSION = "phase-w1-cleanup-provenance"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(path: Path) -> Path:
    return Path(os.path.abspath(path)).resolve(strict=False)


def _path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(_canonical(Path(path)))))


def _inside(path: Path, parent: Path) -> bool:
    try:
        _canonical(path).relative_to(_canonical(parent))
        return True
    except ValueError:
        return False


def _is_reparse(path: Path) -> bool:
    stat = path.lstat()
    return path.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & REPARSE_ATTRIBUTE)


def work_root() -> Path:
    configured = os.getenv("SAFETRACE_DEV_WORK_ROOT", ".ai-pipeline/006_work")
    path = Path(configured)
    return _canonical(path if path.is_absolute() else ROOT / path)


def initialize_work_root(path: Path | None = None, *, repository_root: Path = ROOT) -> Path:
    root = _canonical(path or work_root())
    if not _inside(root, repository_root):
        raise ValueError("Development work root must remain inside the SafeTrace repository.")
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("SafeTrace controlled development work root\n", encoding="ascii")
    for name in WORK_SUBDIRS:
        child = root / name
        child.mkdir(exist_ok=True)
        (child / MARKER).write_text("SafeTrace controlled development work directory\n", encoding="ascii")
    return root


def create_run_directory(category: str, label: str) -> Path:
    if category not in WORK_SUBDIRS[:-1]:
        raise ValueError(f"Unsupported work category: {category}")
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label).strip("_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run = initialize_work_root() / category / f"{safe_label or 'run'}_{stamp}"
    run.mkdir()
    (run / MARKER).write_text("SafeTrace controlled temporary run directory\n", encoding="ascii")
    return run


def finalize_run_directory(path: Path, *, succeeded: bool) -> None:
    candidate = _canonical(path)
    root = work_root()
    if not _inside(candidate, root) or not (candidate / MARKER).is_file() or _is_reparse(candidate):
        raise ValueError("Refusing to finalize an unmarked or unsafe work directory.")
    if succeeded:
        shutil.rmtree(candidate)


def _git_classification(path: Path, repository_root: Path) -> tuple[bool, bool]:
    try:
        relative = str(_canonical(path).relative_to(_canonical(repository_root))).replace("\\", "/")
    except ValueError:
        return False, False
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    return tracked, ignored


def _active_paths() -> set[str]:
    values: set[str] = set()
    try:
        import psutil

        for process in psutil.process_iter(["cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or []).lower()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if command:
                values.add(command)
    except ImportError:
        pass
    return values


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_tree(path: Path) -> dict[str, Any]:
    """No-follow recursive snapshot used by both dry-run and revalidation."""
    size = 0
    files = 0
    directories = 0
    has_reparse = False
    errors: list[str] = []
    fingerprint = hashlib.sha256()
    stack = [path]
    root = _canonical(path)
    while stack:
        current = stack.pop()
        try:
            if _is_reparse(current):
                has_reparse = True
                fingerprint.update(f"R:{current}".encode("utf-8", "surrogateescape"))
                continue
            directories += 1
            relative_directory = "." if current == root else str(current.relative_to(root)).replace("\\", "/")
            fingerprint.update(f"D:{relative_directory}".encode("utf-8", "surrogateescape"))
            with os.scandir(current) as entries:
                for entry in sorted(entries, key=lambda item: item.name.casefold()):
                    child = Path(entry.path)
                    try:
                        stat = entry.stat(follow_symlinks=False)
                        attributes = getattr(stat, "st_file_attributes", 0)
                        relative = str(child.relative_to(root)).replace("\\", "/")
                        if entry.is_symlink() or bool(attributes & REPARSE_ATTRIBUTE):
                            has_reparse = True
                            fingerprint.update(f"R:{relative}".encode("utf-8", "surrogateescape"))
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(child)
                        elif entry.is_file(follow_symlinks=False):
                            digest = _file_digest(child)
                            size += stat.st_size
                            files += 1
                            fingerprint.update(
                                f"F:{relative}:{stat.st_size}:{stat.st_mtime_ns}:{digest}".encode("utf-8", "surrogateescape")
                            )
                    except OSError as exc:
                        errors.append(f"{child}: {type(exc).__name__}: {exc}")
        except OSError as exc:
            errors.append(f"{current}: {type(exc).__name__}: {exc}")
    return {
        "sizeBytes": size,
        "fileCount": files,
        "directoryCount": directories,
        "reparsePoint": has_reparse,
        "errors": errors,
        "treeFingerprint": fingerprint.hexdigest(),
    }


def _marker_identity(path: Path) -> dict[str, Any]:
    marker = path / MARKER
    try:
        if not marker.is_file() or _is_reparse(marker):
            return {"present": False, "sha256": None}
        return {"present": True, "sha256": _file_digest(marker)}
    except OSError:
        return {"present": False, "sha256": None}


def _protected_names(path: Path, repository_root: Path) -> list[str]:
    try:
        parts = _canonical(path).relative_to(_canonical(repository_root)).parts
    except ValueError:
        return ["outside_repository"]
    return sorted({part for part in PROTECTED_PARTS if any(part.casefold() == item.casefold() for item in parts)})


def _audit_path(
    path: Path,
    active_commands: set[str],
    *,
    repository_root: Path = ROOT,
    minimum_age_seconds: float = 600.0,
) -> dict[str, Any]:
    source = Path(path)
    canonical = _canonical(source)
    tracked, ignored = _git_classification(canonical, repository_root)
    errors: list[str] = []
    try:
        stat = source.lstat()
        modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        reparse = _is_reparse(source)
        scan = _scan_tree(source) if not reparse else {
            "sizeBytes": 0, "fileCount": 0, "directoryCount": 0,
            "reparsePoint": True, "errors": [], "treeFingerprint": None,
        }
    except OSError as exc:
        modified = None
        reparse = False
        scan = {
            "sizeBytes": 0, "fileCount": 0, "directoryCount": 0,
            "reparsePoint": False, "errors": [f"{type(exc).__name__}: {exc}"], "treeFingerprint": None,
        }
    errors.extend(scan["errors"])
    marker = _marker_identity(source) if not errors and not reparse else {"present": False, "sha256": None}
    protected_names = _protected_names(canonical, repository_root)
    canonical_key = _path_key(canonical)
    active = any(canonical_key in os.path.normcase(command) for command in active_commands)
    age_seconds = (datetime.now(timezone.utc) - modified).total_seconds() if modified else None
    safe = bool(
        not tracked and ignored and not errors and not reparse and not scan["reparsePoint"]
        and not protected_names and not active and marker["present"]
        and age_seconds is not None and age_seconds >= minimum_age_seconds
    )
    if safe:
        reason = "candidate_verified_for_manifest"
    elif tracked:
        reason = "tracked_path"
    elif errors:
        reason = "inaccessible_or_scan_error"
    elif reparse or scan["reparsePoint"]:
        reason = "reparse_point_present"
    elif protected_names:
        reason = "protected_content_name_present"
    elif active:
        reason = "active_process_reference"
    elif not marker["present"]:
        reason = "missing_or_unsafe_marker"
    elif not ignored:
        reason = "not_ignored_generated_path"
    else:
        reason = "recent_path_may_be_active"
    relative = None
    try:
        relative = str(canonical.relative_to(_canonical(repository_root))).replace("\\", "/")
    except ValueError:
        pass
    return {
        "canonicalPath": str(canonical),
        "relativePath": relative,
        "sizeBytes": int(scan["sizeBytes"]),
        "fileCount": int(scan["fileCount"]),
        "directoryCount": int(scan["directoryCount"]),
        "modifiedAt": modified.isoformat() if modified else None,
        "tracked": tracked,
        "ignored": ignored,
        "reparsePoint": bool(reparse or scan["reparsePoint"]),
        "active": active,
        "accessible": not bool(errors),
        "classification": "safe_generated_cleanup" if safe else "retained",
        "reason": reason,
        "errors": errors,
        "marker": marker,
        "treeFingerprint": scan["treeFingerprint"],
    }


def _root_candidates(candidate_root: Path) -> list[Path]:
    values: dict[str, Path] = {}
    root = _canonical(candidate_root)
    for pattern in ROOT_PATTERNS:
        for path in root.glob(pattern):
            if path.is_dir() or path.is_symlink():
                values[_path_key(path)] = path
    return sorted(values.values(), key=lambda item: item.name.casefold())


def audit_candidates(
    *,
    candidate_root: Path = ROOT,
    repository_root: Path = ROOT,
    minimum_age_seconds: float = 600.0,
) -> list[dict[str, Any]]:
    root = _canonical(candidate_root)
    repo = _canonical(repository_root)
    if not _inside(root, repo):
        raise ValueError("Candidate root must remain inside the repository.")
    active_commands = _active_paths()
    return [
        _audit_path(path, active_commands, repository_root=repo, minimum_age_seconds=minimum_age_seconds)
        for path in _root_candidates(root)
    ]


def _manifest_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["classification"] == "safe_generated_cleanup"]


def create_approval_manifest(
    *,
    candidate_root: Path = ROOT,
    repository_root: Path = ROOT,
    work_root_path: Path | None = None,
    minimum_age_seconds: float = 600.0,
) -> dict[str, Any]:
    repo = _canonical(repository_root)
    work = _canonical(work_root_path or work_root())
    root = _canonical(candidate_root)
    if not _inside(root, repo) or not _inside(work, repo):
        raise ValueError("Approval roots must remain inside the repository.")
    rows = audit_candidates(candidate_root=root, repository_root=repo, minimum_age_seconds=minimum_age_seconds)
    candidates = _manifest_candidates(rows)
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "toolVersion": TOOL_VERSION,
        "repositoryRoot": str(repo),
        "workRoot": str(work),
        "candidateRoot": str(root),
        "generatedAt": _utc(),
        "candidateCount": len(candidates),
        "totalBytes": sum(int(item["sizeBytes"]) for item in candidates),
        "candidates": candidates,
        "retained": [item for item in rows if item["classification"] != "safe_generated_cleanup"],
    }


def _manifest_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_approval_manifest(path: Path, payload: dict[str, Any]) -> str:
    target = _canonical(path)
    if target.exists():
        raise FileExistsError(f"Approval manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _manifest_bytes(payload)
    target.write_bytes(content)
    return _sha256_bytes(content)


def load_approval_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    content = Path(path).read_bytes()
    actual = _sha256_bytes(content)
    if actual.casefold() != expected_sha256.strip().casefold():
        raise ValueError("Approval manifest SHA-256 mismatch.")
    payload = json.loads(content.decode("utf-8"))
    if payload.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported approval manifest schema version.")
    return payload, actual


def _validate_manifest_context(
    payload: dict[str, Any], *, repository_root: Path, work_root_path: Path,
) -> tuple[Path, Path, Path]:
    repo = _canonical(repository_root)
    work = _canonical(work_root_path)
    candidate_root = _canonical(Path(str(payload.get("candidateRoot") or "")))
    if _path_key(payload.get("repositoryRoot", "")) != _path_key(repo):
        raise ValueError("Approval manifest repository root does not match the current repository.")
    if _path_key(payload.get("workRoot", "")) != _path_key(work):
        raise ValueError("Approval manifest work root does not match the current work root.")
    if not _inside(candidate_root, repo):
        raise ValueError("Approval manifest candidate root escaped the repository.")
    return repo, work, candidate_root


def _validate_manifest_candidates(payload: dict[str, Any], candidate_root: Path) -> list[dict[str, Any]]:
    candidates = list(payload.get("candidates") or [])
    if int(payload.get("candidateCount", -1)) != len(candidates):
        raise ValueError("Approval manifest candidate count does not match entries.")
    if int(payload.get("totalBytes", -1)) != sum(int(item.get("sizeBytes") or 0) for item in candidates):
        raise ValueError("Approval manifest total bytes does not match entries.")
    keys: list[str] = []
    for item in candidates:
        path = Path(str(item.get("canonicalPath") or ""))
        if not _inside(path, candidate_root):
            raise ValueError("Approval manifest candidate escaped its approved root.")
        canonical = _canonical(path)
        if _path_key(canonical) != _path_key(item["canonicalPath"]):
            raise ValueError("Approval manifest candidate canonical path is inconsistent.")
        keys.append(_path_key(canonical))
    if len(keys) != len(set(keys)):
        raise ValueError("Approval manifest contains duplicate paths.")
    for left in keys:
        for right in keys:
            if left != right and right.startswith(left + os.sep):
                raise ValueError("Approval manifest contains overlapping paths.")
    return candidates


def _same_snapshot(expected: dict[str, Any], current: dict[str, Any]) -> tuple[bool, str]:
    checks = (
        ("canonicalPath", _path_key(expected["canonicalPath"]), _path_key(current["canonicalPath"])),
        ("marker", expected.get("marker"), current.get("marker")),
        ("treeFingerprint", expected.get("treeFingerprint"), current.get("treeFingerprint")),
        ("sizeBytes", expected.get("sizeBytes"), current.get("sizeBytes")),
        ("fileCount", expected.get("fileCount"), current.get("fileCount")),
        ("directoryCount", expected.get("directoryCount"), current.get("directoryCount")),
        ("modifiedAt", expected.get("modifiedAt"), current.get("modifiedAt")),
        ("reparsePoint", expected.get("reparsePoint"), current.get("reparsePoint")),
        ("active", expected.get("active"), current.get("active")),
        ("accessible", expected.get("accessible"), current.get("accessible")),
        ("classification", expected.get("classification"), current.get("classification")),
    )
    for name, expected_value, current_value in checks:
        if expected_value != current_value:
            return False, f"drift_{name}"
    return True, "unchanged"


def _write_certificate(path: Path, payload: dict[str, Any]) -> None:
    target = _canonical(path)
    if target.exists():
        raise FileExistsError(f"Cleanup certificate already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_approval_manifest(
    *,
    approval_manifest: Path,
    approval_sha256: str,
    certificate_path: Path | None = None,
    repository_root: Path = ROOT,
    work_root_path: Path | None = None,
) -> dict[str, Any]:
    started = _utc()
    payload, actual_sha256 = load_approval_manifest(approval_manifest, approval_sha256)
    repo, work, candidate_root = _validate_manifest_context(
        payload,
        repository_root=repository_root,
        work_root_path=work_root_path or work_root(),
    )
    candidates = _validate_manifest_candidates(payload, candidate_root)
    result: dict[str, Any] = {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "toolVersion": TOOL_VERSION,
        "startedAt": started,
        "endedAt": None,
        "repositoryRoot": str(repo),
        "workRoot": str(work),
        "approvalManifestPath": str(_canonical(approval_manifest)),
        "approvalManifestSha256": actual_sha256,
        "approvedCandidateCount": len(candidates),
        "approvedBytes": sum(int(item["sizeBytes"]) for item in candidates),
        "revalidatedCount": 0,
        "deletedPaths": [],
        "deletedBytes": 0,
        "skipped": [],
        "rejected": [],
        "quarantineRetained": [],
    }
    quarantine: Path | None = None
    for approved in candidates:
        source = Path(str(approved["canonicalPath"]))
        if not source.exists():
            result["skipped"].append({"path": str(source), "reason": "missing_after_approval"})
            continue
        try:
            if _is_reparse(source):
                result["rejected"].append({"path": str(source), "reason": "reparse_point_after_approval"})
                continue
            current = _audit_path(source, _active_paths(), repository_root=repo, minimum_age_seconds=0)
        except OSError as exc:
            result["rejected"].append({"path": str(source), "reason": f"inaccessible_after_approval:{type(exc).__name__}"})
            continue
        result["revalidatedCount"] += 1
        unchanged, reason = _same_snapshot(approved, current)
        if not unchanged:
            result["skipped"].append({"path": str(source), "reason": reason})
            continue
        if current["classification"] != "safe_generated_cleanup":
            result["rejected"].append({"path": str(source), "reason": current["reason"]})
            continue
        if quarantine is None:
            initialized = initialize_work_root(work, repository_root=repo)
            quarantine = initialized / "quarantine" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            quarantine.mkdir(parents=True, exist_ok=False)
            (quarantine / MARKER).write_text("SafeTrace cleanup quarantine\n", encoding="ascii")
        destination = quarantine / source.name
        if destination.exists():
            result["rejected"].append({"path": str(source), "reason": "quarantine_name_collision"})
            continue
        try:
            shutil.move(str(source), str(destination))
            if not (destination / MARKER).is_file() or _is_reparse(destination):
                result["quarantineRetained"].append({"path": str(destination), "reason": "quarantine_verification_failed"})
                continue
            try:
                shutil.rmtree(destination)
            except OSError as exc:
                result["quarantineRetained"].append({"path": str(destination), "reason": f"partial_delete_failure:{type(exc).__name__}"})
                continue
            result["deletedPaths"].append(str(source))
            result["deletedBytes"] += int(approved["sizeBytes"])
        except OSError as exc:
            result["rejected"].append({"path": str(source), "reason": f"quarantine_move_failure:{type(exc).__name__}"})
    approved_keys = {_path_key(item["canonicalPath"]) for item in candidates}
    unapproved = [path for path in result["deletedPaths"] if _path_key(path) not in approved_keys]
    result.update({
        "endedAt": _utc(),
        "unapprovedDeletions": unapproved,
        "protectedMutations": 0,
        "unknownMutations": 0,
        "reparsePointMutations": 0,
        "passed": not unapproved,
    })
    if certificate_path is not None:
        _write_certificate(certificate_path, result)
    return result


def _write_report(rows: Iterable[dict[str, Any]], *, output_json: Path | None, output_csv: Path | None) -> None:
    values = list(rows)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(values, indent=2), encoding="utf-8")
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["canonicalPath", "relativePath", "sizeBytes", "fileCount", "directoryCount", "modifiedAt", "tracked", "ignored", "reparsePoint", "active", "accessible", "classification", "reason", "errors"]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in values:
                writer.writerow({**row, "errors": " | ".join(row.get("errors") or [])})


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    audit = sub.add_parser("audit")
    audit.add_argument("--output-json", type=Path)
    audit.add_argument("--output-csv", type=Path)
    clean = sub.add_parser("cleanup")
    mode = clean.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    clean.add_argument("--approval-manifest", type=Path, required=True)
    clean.add_argument("--approval-sha256")
    clean.add_argument("--certificate", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        root = initialize_work_root()
        print(json.dumps({"workRoot": str(root), "marker": str(root / MARKER)}, indent=2))
        return 0
    if args.command == "audit":
        rows = audit_candidates()
        _write_report(rows, output_json=args.output_json, output_csv=args.output_csv)
        print(json.dumps(rows, indent=2))
        return 0
    if args.dry_run:
        manifest = create_approval_manifest()
        digest = write_approval_manifest(args.approval_manifest, manifest)
        print(json.dumps({"approvalManifest": str(_canonical(args.approval_manifest)), "approvalSha256": digest, "candidateCount": manifest["candidateCount"], "totalBytes": manifest["totalBytes"]}, indent=2))
        return 0
    if not args.approval_sha256:
        parser.error("cleanup --apply requires --approval-sha256")
    result = apply_approval_manifest(
        approval_manifest=args.approval_manifest,
        approval_sha256=args.approval_sha256,
        certificate_path=args.certificate,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
