from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import manage_dev_workspace as workspace


def _mark(path: Path, content: str = "fixture") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / workspace.MARKER).write_text(content, encoding="ascii")
    (path / "payload.txt").write_text("original", encoding="ascii")
    return path


def _approval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, str, Path, Path]:
    monkeypatch.setattr(workspace, "_git_classification", lambda *_args, **_kwargs: (False, True))
    repo = tmp_path / "repo"
    work = repo / "work"
    repo.mkdir()
    candidate = _mark(repo / ".tmp_phase_w1_a")
    manifest = workspace.create_approval_manifest(
        candidate_root=repo,
        repository_root=repo,
        work_root_path=work,
        minimum_age_seconds=0,
    )
    manifest_path = repo / "approval.json"
    digest = workspace.write_approval_manifest(manifest_path, manifest)
    return manifest_path, digest, repo, candidate


def _apply(manifest: Path, digest: str, repo: Path, certificate: Path | None = None):
    return workspace.apply_approval_manifest(
        approval_manifest=manifest,
        approval_sha256=digest,
        certificate_path=certificate,
        repository_root=repo,
        work_root_path=repo / "work",
    )


def _write_payload(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workspace._sha256_bytes(path.read_bytes())


def test_approved_unchanged_fixture_is_deleted_and_certificate_proves_subset(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    certificate = repo / "certificate.json"
    result = _apply(manifest, digest, repo, certificate)
    assert not candidate.exists()
    assert result["deletedPaths"] == [str(candidate.resolve())]
    assert result["unapprovedDeletions"] == []
    assert result["protectedMutations"] == 0
    assert json.loads(certificate.read_text(encoding="utf-8"))["approvedCandidateCount"] == 1


def test_digest_mismatch_and_manifest_edit_fail_closed(monkeypatch, tmp_path):
    manifest, digest, repo, _candidate = _approval(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _apply(manifest, "0" * 64, repo)
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _apply(manifest, digest, repo)


def test_repository_and_work_root_mismatch_fail_closed(monkeypatch, tmp_path):
    manifest, digest, repo, _candidate = _approval(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="repository root"):
        workspace.apply_approval_manifest(
            approval_manifest=manifest,
            approval_sha256=digest,
            repository_root=tmp_path / "different-repo",
            work_root_path=repo / "work",
        )
    with pytest.raises(ValueError, match="work root"):
        workspace.apply_approval_manifest(
            approval_manifest=manifest,
            approval_sha256=digest,
            repository_root=repo,
            work_root_path=repo / "other-work",
        )


def test_new_and_initially_ambiguous_paths_are_never_discovered_during_apply(monkeypatch, tmp_path):
    manifest, digest, repo, approved = _approval(monkeypatch, tmp_path)
    ambiguous = repo / ".tmp_phase_w1_c"
    ambiguous.mkdir()
    (ambiguous / "payload.txt").write_text("ambiguous", encoding="ascii")
    new_path = _mark(repo / ".tmp_phase_w1_d")
    (ambiguous / workspace.MARKER).write_text("later accessible", encoding="ascii")

    result = _apply(manifest, digest, repo)
    assert not approved.exists()
    assert ambiguous.exists()
    assert new_path.exists()
    assert result["deletedPaths"] == [str(approved.resolve())]


def test_modified_approved_candidate_is_skipped(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    (candidate / "payload.txt").write_text("modified", encoding="ascii")
    result = _apply(manifest, digest, repo)
    assert candidate.exists()
    assert result["deletedPaths"] == []
    assert result["skipped"][0]["reason"].startswith("drift_")


def test_reparse_replacement_is_rejected_when_supported(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    target = _mark(repo / ".tmp_phase_w1_target")
    for child in candidate.iterdir():
        child.unlink()
    candidate.rmdir()
    try:
        os.symlink(target, candidate, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink fixture is not available in this environment")
    result = _apply(manifest, digest, repo)
    assert result["deletedPaths"] == []
    assert result["rejected"][0]["reason"] == "reparse_point_after_approval"


def test_reparse_rejection_branch_is_covered_without_symlink_privileges(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    original = workspace._is_reparse

    def fake_reparse(path: Path) -> bool:
        return workspace._path_key(path) == workspace._path_key(candidate) or original(path)

    monkeypatch.setattr(workspace, "_is_reparse", fake_reparse)
    result = _apply(manifest, digest, repo)
    assert candidate.exists()
    assert result["deletedPaths"] == []
    assert result["rejected"][0]["reason"] == "reparse_point_after_approval"


def test_inaccessible_after_approval_is_rejected(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    original = workspace._audit_path

    def inaccessible(path, *args, **kwargs):
        if workspace._path_key(path) == workspace._path_key(candidate):
            raise OSError("fixture access denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(workspace, "_audit_path", inaccessible)
    result = _apply(manifest, digest, repo)
    assert candidate.exists()
    assert result["deletedPaths"] == []
    assert result["rejected"][0]["reason"].startswith("inaccessible_after_approval")


def test_active_candidate_is_skipped(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    monkeypatch.setattr(workspace, "_active_paths", lambda: {workspace._path_key(candidate)})
    result = _apply(manifest, digest, repo)
    assert candidate.exists()
    assert result["deletedPaths"] == []
    assert result["skipped"][0]["reason"] == "drift_active"


def test_duplicate_overlap_and_traversal_manifests_are_rejected(monkeypatch, tmp_path):
    manifest_path, _digest, repo, candidate = _approval(monkeypatch, tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["candidates"].append(dict(payload["candidates"][0]))
    payload["candidateCount"] = 2
    payload["totalBytes"] *= 2
    duplicate_path = repo / "duplicate.json"
    duplicate_digest = _write_payload(duplicate_path, payload)
    with pytest.raises(ValueError, match="duplicate"):
        _apply(duplicate_path, duplicate_digest, repo)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside" / ".tmp_phase_w1_escape"
    payload["candidates"][0]["canonicalPath"] = str(outside)
    traversal_path = repo / "traversal.json"
    traversal_digest = _write_payload(traversal_path, payload)
    with pytest.raises(ValueError, match="escaped"):
        _apply(traversal_path, traversal_digest, repo)
    assert candidate.exists()


def test_windows_path_case_normalization_and_apply_idempotency(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    assert workspace._path_key(candidate) == workspace._path_key(Path(str(candidate).upper()))
    first = _apply(manifest, digest, repo)
    second = _apply(manifest, digest, repo)
    assert first["deletedPaths"] == [str(candidate.resolve())]
    assert second["deletedPaths"] == []
    assert second["skipped"][0]["reason"] == "missing_after_approval"


def test_partial_delete_failure_retains_quarantine_without_reporting_deletion(monkeypatch, tmp_path):
    manifest, digest, repo, candidate = _approval(monkeypatch, tmp_path)
    monkeypatch.setattr(workspace.shutil, "rmtree", lambda _path: (_ for _ in ()).throw(OSError("fixture failure")))
    result = _apply(manifest, digest, repo)
    assert not candidate.exists()
    assert result["deletedPaths"] == []
    assert result["quarantineRetained"][0]["reason"].startswith("partial_delete_failure")
    assert Path(result["quarantineRetained"][0]["path"]).exists()


def test_zero_candidate_manifest_apply_is_zero_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "_git_classification", lambda *_args, **_kwargs: (False, True))
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = workspace.create_approval_manifest(
        candidate_root=repo,
        repository_root=repo,
        work_root_path=repo / "work",
        minimum_age_seconds=0,
    )
    manifest_path = repo / "zero.json"
    digest = workspace.write_approval_manifest(manifest_path, manifest)
    result = _apply(manifest_path, digest, repo)
    assert manifest["candidateCount"] == 0
    assert result["deletedPaths"] == []
    assert result["deletedBytes"] == 0
