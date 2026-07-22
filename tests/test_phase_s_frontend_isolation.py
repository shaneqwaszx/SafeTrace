from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend-react"


def test_delayed_and_out_of_order_frontend_results_remain_selected_job_only(tmp_path):
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    node = shutil.which("node.exe") or shutil.which("node")
    assert npm and node, "Node.js and npm are required for the deterministic frontend isolation test."

    compiled = tmp_path / "compiled"
    subprocess.run(
        [
            npm,
            "exec",
            "tsc",
            "--",
            "src/services/resultIsolation.ts",
            "--target",
            "ES2022",
            "--module",
            "ES2022",
            "--moduleResolution",
            "bundler",
            "--skipLibCheck",
            "--outDir",
            str(compiled),
        ],
        cwd=FRONTEND,
        check=True,
        capture_output=True,
        text=True,
    )
    module = next(compiled.rglob("resultIsolation.js"))
    completed = subprocess.run(
        [node, str(FRONTEND / "scripts" / "verify-result-isolation.mjs"), str(module)],
        cwd=FRONTEND,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "frontend result isolation scenarios passed" in completed.stdout


def test_app_uses_keyed_results_request_generation_and_render_guard():
    app = (FRONTEND / "src" / "App.tsx").read_text(encoding="utf-8")
    evidence = (FRONTEND / "src" / "components" / "EvidenceFrames.tsx").read_text(encoding="utf-8")
    cache = (FRONTEND / "src" / "services" / "resultCache.ts").read_text(encoding="utf-8")

    assert "resultByJobId" in app
    assert "ResultRequestCoordinator" in app
    assert "resultRequestStateByJobId" in app
    assert "commitVisibleResult(result, jobId)" in app
    assert "frame.jobId === jobId" in evidence
    assert "${jobId || 'preview'}:${frame.evidenceId || frame.id}" in evidence
    assert "cacheEntryHasValidOwnership" in cache
