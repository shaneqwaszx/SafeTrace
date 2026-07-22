"""Smoke-check packaged SafeTrace VLM device routing.

The script expects a backend to already be running. It submits the same media
with two device preferences and reports what the backend actually used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request


def _json_request(url: str, *, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    req = request.Request(url, data=data, method=method, headers=headers or {})
    with request.urlopen(req, timeout=10) as response:  # noqa: S310 - local smoke helper
        return json.loads(response.read().decode("utf-8"))


def _multipart_body(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = f"safetrace-{time.time_ns()}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")
    media_type = "image/jpeg" if file_path.suffix.lower() in {".jpg", ".jpeg"} else "video/mp4"
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


def _submit_job(api_base: str, file_path: Path, *, device: str, mode: str) -> str:
    body, boundary = _multipart_body(
        {
            "query": "driver missing seatbelt",
            "fps": "1.0",
            "topK": "1",
            "enableVlm": "true",
            "vlmEnabled": "true",
            "vlmProfile": mode,
            "device": device,
            "useCaseProfile": json.dumps(
                {
                    "profileId": "seatbelt_compliance",
                    "label": "Seatbelt compliance",
                    "category": "Vehicle safety",
                    "description": "Check driver/occupant seatbelt evidence.",
                    "defaultQuery": "driver missing seatbelt",
                    "supportedChecks": ["seatbelt_missing"],
                    "backendSupportLevel": "supported",
                    "requestedQuery": "driver missing seatbelt",
                    "effectiveQuery": "driver missing seatbelt",
                }
            ),
        },
        file_path,
    )
    payload = _json_request(
        f"{api_base.rstrip('/')}/api/analyze",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return str(payload["jobId"])


def _poll_job(api_base: str, job_id: str, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = _json_request(f"{api_base.rstrip('/')}/api/jobs/{parse.quote(job_id)}")
        if status.get("status") in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(1.5)
    raise TimeoutError(f"Timed out waiting for {job_id}")


def run_smoke(api_base: str, file_path: Path, *, mode: str, timeout_seconds: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for device in ("auto", "cpu"):
        started = time.perf_counter()
        job_id = _submit_job(api_base, file_path, device=device, mode=mode)
        status = _poll_job(api_base, job_id, timeout_seconds=timeout_seconds)
        elapsed = time.perf_counter() - started
        summary = status.get("engineRuntimeSummary") or {}
        results.append(
            {
                "deviceRequested": device,
                "jobId": job_id,
                "status": status.get("status"),
                "elapsedSeconds": round(elapsed, 3),
                "actualDeviceLabel": status.get("actualDeviceLabel"),
                "actualDetectorDevice": summary.get("actualDetectorDevice"),
                "actualLightweightVlmDevice": summary.get("actualLightweightVlmDevice"),
                "actualEnhancedVlmDevice": summary.get("actualEnhancedVlmDevice"),
                "cudaAvailableAtJobStart": summary.get("cudaAvailableAtJobStart"),
                "gpuName": summary.get("gpuName"),
                "vlm": summary.get("vlm"),
                "actualReview": summary.get("actualReview"),
            }
        )
    return {"apiBase": api_base, "input": str(file_path), "mode": mode, "runs": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke packaged VLM device routing against a running SafeTrace backend.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8011")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mode", default="enhanced_2b", choices=["lightweight_512m", "enhanced_2b"])
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    try:
        report = run_smoke(args.api_base, args.input, mode=args.mode, timeout_seconds=args.timeout_seconds)
    except error.URLError as exc:
        raise SystemExit(f"Backend request failed: {exc}") from exc
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
