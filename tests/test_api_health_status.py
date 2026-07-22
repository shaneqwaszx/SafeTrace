from fastapi.testclient import TestClient
from types import SimpleNamespace

import src.chat_service as chat_service
import src.device_gateway as device_gateway
from src.api.batches import BatchStore
import src.api.jobs as jobs_module
import src.api.server as server_module
from src.api.jobs import JobStore
from src.api.server import create_app
from src.device_gateway import HardwareGpuStatus, TorchDeviceStatus
import src.vlm_reasoner as vlm_reasoner


def make_client(tmp_path):
    app = create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches"))
    return TestClient(app)


def llama_diagnostics(import_ok: bool):
    return {
        "backendPythonExecutable": r"C:\repo\.venv\Scripts\python.exe" if import_ok else r"C:\Python312\python.exe",
        "expectedVenvPython": r"C:\repo\.venv\Scripts\python.exe",
        "expectedVenvPythonExists": True,
        "runningInExpectedVenv": import_ok,
        "specFound": import_ok,
        "importOk": import_ok,
        "importStatus": "ok" if import_ok else "missing",
        "importErrorType": None,
        "importErrorMessage": None,
        "setupCommand": r".venv\Scripts\python.exe -m pip install llama-cpp-python",
        "restartRequired": "Restart the SafeTrace backend after installing llama-cpp-python.",
    }


def _gateway_settings(**overrides):
    values = {
        "device": "auto",
        "lightweight_vlm_device": "auto",
        "enhanced_vlm_device": "cuda",
        "mobile_sam_device": "auto",
        "enable_gpu_auto": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cpu_device_gateway_payload():
    cpu_decision = {
        "component": "detector",
        "configured": "cpu",
        "selected": "cpu",
        "available": True,
        "reason": "CPU explicitly configured.",
        "requires_gpu": False,
    }
    return {
        "configuredDevice": "cpu",
        "gpuAutoEnabled": True,
        "torch": {
            "installed": True,
            "cuda_available": False,
            "cuda_device_count": 0,
            "gpu_name": None,
            "total_gpu_memory_mb": None,
            "reserved_gpu_memory_mb": None,
            "allocated_gpu_memory_mb": None,
            "error": None,
        },
        "hardware": {
            "nvidia_smi_available": False,
            "nvidia_gpu_detected": False,
            "gpu_name": None,
            "error": "nvidia-smi not found",
        },
        "components": {
            "detector": cpu_decision,
            "mobileSam": {**cpu_decision, "component": "mobileSam"},
            "lightweightVlm": {**cpu_decision, "component": "lightweightVlm"},
            "enhancedVlm": {
                "component": "enhancedVlm",
                "configured": "cuda",
                "selected": "unavailable",
                "available": False,
                "reason": "PyTorch CUDA runtime is unavailable.",
                "requires_gpu": True,
            },
        },
        "gpuUnavailableReason": "No CUDA-capable GPU reported by PyTorch.",
    }


def test_device_gateway_selects_cpu_for_lightweight_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(
        device_gateway,
        "torch_status",
        lambda: TorchDeviceStatus(installed=True, cuda_available=False, cuda_device_count=0),
    )
    monkeypatch.setattr(
        device_gateway,
        "hardware_gpu_status",
        lambda: HardwareGpuStatus(nvidia_smi_available=False, nvidia_gpu_detected=False, error="nvidia-smi not found"),
    )

    payload = device_gateway.device_gateway_payload(_gateway_settings())

    assert payload["components"]["lightweightVlm"]["selected"] == "cpu"
    assert payload["components"]["lightweightVlm"]["available"] is True
    assert payload["components"]["enhancedVlm"]["available"] is False
    assert payload["components"]["enhancedVlm"]["selected"] == "unavailable"
    assert "CUDA" in payload["components"]["enhancedVlm"]["reason"]


def test_cloud_cors_can_exclude_local_origins(monkeypatch):
    monkeypatch.setattr(server_module.SETTINGS, "include_local_cors_origins", False)
    monkeypatch.setattr(server_module.SETTINGS, "allowed_origins", ("https://cloud.example.test",))
    monkeypatch.setenv("SAFETRACE_ALLOWED_ORIGINS", "https://api-client.example.test")

    assert server_module._cors_allowed_origins() == [
        "https://cloud.example.test",
        "https://api-client.example.test",
    ]


def test_device_gateway_selects_cuda_for_auto_components_when_available(monkeypatch):
    monkeypatch.setattr(
        device_gateway,
        "torch_status",
        lambda: TorchDeviceStatus(
            installed=True,
            cuda_available=True,
            cuda_device_count=1,
            gpu_name="RTX Test",
            total_gpu_memory_mb=8192,
        ),
    )
    monkeypatch.setattr(
        device_gateway,
        "hardware_gpu_status",
        lambda: HardwareGpuStatus(nvidia_smi_available=True, nvidia_gpu_detected=True, gpu_name="RTX Test"),
    )

    payload = device_gateway.device_gateway_payload(_gateway_settings())

    assert payload["torch"]["cuda_available"] is True
    assert payload["components"]["detector"]["selected"] == "cuda"
    assert payload["components"]["mobileSam"]["selected"] == "cuda"
    assert payload["components"]["lightweightVlm"]["selected"] == "cuda"
    assert payload["components"]["enhancedVlm"]["available"] is True
    assert payload["components"]["enhancedVlm"]["selected"] == "cuda"


def test_health_does_not_instantiate_pipeline(monkeypatch, tmp_path):
    def fail_if_called(**kwargs):  # noqa: ARG001
        raise AssertionError("pipeline should not run for health checks")

    monkeypatch.setattr(jobs_module, "run_pipeline", fail_if_called)
    client = make_client(tmp_path)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["api"] == "safetrace-local"


def test_cors_allows_local_dev_frontend_origin(tmp_path):
    client = make_client(tmp_path)
    origin = "http://127.0.0.1:5173"

    response = client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_allows_configured_live_origin_with_private_network_header(monkeypatch, tmp_path):
    origin = "https://safetrace-demo.pages.dev"
    monkeypatch.setenv("SAFETRACE_ALLOWED_ORIGINS", origin)
    client = make_client(tmp_path)

    response = client.options(
        "/api/system/status",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-private-network"] == "true"


def test_cors_allows_vercel_analyze_private_network_preflight(monkeypatch, tmp_path):
    origin = "https://safetrace-iota.vercel.app"
    monkeypatch.setenv("SAFETRACE_ALLOWED_ORIGINS", origin)
    client = make_client(tmp_path)

    response = client.options(
        "/api/analyze",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-private-network"] == "true"


def test_cors_blocks_unconfigured_origin_and_private_network_header(monkeypatch, tmp_path):
    monkeypatch.setenv("SAFETRACE_ALLOWED_ORIGINS", "https://safetrace-demo.pages.dev")
    client = make_client(tmp_path)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://example.invalid",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-private-network" not in response.headers


def test_system_status_reports_missing_paths_without_loading_models(monkeypatch, tmp_path):
    monkeypatch.delenv("SAFETRACE_BUILD_MODE", raising=False)
    monkeypatch.delenv("SAFETRACE_RUNTIME_LAYOUT", raising=False)
    monkeypatch.setattr(server_module, "_gpu_available", lambda: False)
    monkeypatch.setattr(server_module, "device_gateway_payload", lambda settings: _cpu_device_gateway_payload())
    monkeypatch.setattr(server_module.SETTINGS, "device", "cpu")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_enabled", "auto")
    monkeypatch.setattr(server_module, "_mobile_sam_runtime_available", lambda: False)
    monkeypatch.setattr(server_module.SETTINGS, "siglip_model_dir", tmp_path / "missing-siglip")
    monkeypatch.setattr(server_module.SETTINGS, "yolo_checkpoint", tmp_path / "missing-yolo.pt")
    monkeypatch.setattr(server_module.SETTINGS, "yolo_fallback_checkpoint", tmp_path / "missing-fallback.pt")
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_checkpoint", tmp_path / "missing-mobile-sam.pt")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_model_dir", tmp_path / "missing-vlm")
    monkeypatch.setattr(
        server_module,
        "vlm_status_payload",
        lambda: {
            "status": "missing_runtime",
            "message": "Local Ollama vision runtime is not reachable.",
            "actionHint": "Start Ollama locally.",
            "details": {"provider": "ollama"},
        },
    )

    client = make_client(tmp_path)
    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["app_version"]
    assert body["backend_version"]
    assert body["build_mode"] == "development"
    assert body["runtime_layout"] == "source"
    assert body["device"] == "cpu"
    assert body["gpuAvailable"] is False
    assert body["models"]["embeddingModel"]["status"] == "missing"
    assert body["models"]["detector"]["status"] == "missing"
    assert body["models"]["mobileSam"]["status"] == "missing_checkpoint"
    assert "detector-box evidence" in body["models"]["mobileSam"]["message"]
    assert body["models"]["vlm"]["status"] == "missing_runtime"
    assert "Ollama" in body["models"]["vlm"]["message"]
    assert body["limits"]["maxUploadMb"] > 0
    assert body["limits"]["maxSampledFrames"] > 0
    assert body["limits"]["maxVideoDurationUnlimited"] is True
    assert "No explicit video duration cap" in body["limits"]["maxVideoDurationMessage"]
    assert body["limits"]["embeddingPoolingStrategy"] in {"mean", "max"}
    assert body["limits"]["analysisConcurrency"] >= 1
    assert body["limits"]["vlmConcurrency"] >= 1
    assert body["queue"]["statusCounts"] == {}


def test_system_status_includes_runtime_preflight_without_loading_chat_model(monkeypatch, tmp_path):
    model_path = tmp_path / "assistant.gguf"
    model_path.write_bytes(b"stub")

    def fail_if_model_loads():
        raise AssertionError("system status should not load the chat model")

    monkeypatch.setattr(server_module, "_gpu_available", lambda: False)
    monkeypatch.setattr(chat_service, "_llama_cpp_runtime_diagnostics", lambda: llama_diagnostics(True))
    monkeypatch.setattr(chat_service, "_get_packaged_model", fail_if_model_loads)
    monkeypatch.setattr(chat_service.SETTINGS, "chat_enabled", "auto")
    monkeypatch.setattr(chat_service.SETTINGS, "chat_provider", "packaged_llamacpp")
    monkeypatch.setattr(chat_service.SETTINGS, "chat_model_path", model_path)
    monkeypatch.setattr(chat_service.SETTINGS, "chat_autoload", True)
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    client = make_client(tmp_path)
    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["runtime"]["backend"]["status"] == "ready"
    assert body["runtime"]["python"]["executable"]
    assert body["runtime"]["workingDirectory"]
    assert body["runtime"]["jobStorePath"]
    assert body["runtime"]["openmp"]["kmpDuplicateLibOk"] is True
    assert body["runtime"]["openmp"]["ompNumThreads"] == "1"
    assert body["runtime"]["chat"]["provider"] == "packaged_llamacpp"
    assert body["runtime"]["chat"]["model_exists"] is True
    assert body["runtime"]["chat"]["runtime_available"] is True
    assert body["runtime"]["chat"]["llama_cpp_import_status"] == "ok"
    assert body["runtime"]["chat"]["running_in_expected_venv"] is True
    assert body["preflight"]["checks"]["assistant"]["status"] == "available"
    assert body["preflight"]["checks"]["assistantModel"]["status"] == "ready"
    assert body["preflight"]["checks"]["assistantRuntime"]["status"] == "ready"
    assert body["preflight"]["checks"]["assistantRuntime"]["details"]["llamaCppImportStatus"] == "ok"
    assert body["preflight"]["checks"]["openmp"]["status"] == "ready"


def test_system_status_reports_chat_unavailable_without_ollama(monkeypatch, tmp_path):
    def fail_ollama_check(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr(chat_service.httpx, "get", fail_ollama_check)
    monkeypatch.setattr(chat_service.SETTINGS, "chat_enabled", True)
    monkeypatch.setattr(chat_service.SETTINGS, "chat_provider", "ollama")
    monkeypatch.setattr(chat_service.SETTINGS, "ollama_base_url", "http://127.0.0.1:9")

    client = make_client(tmp_path)
    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["runtime"]["chat"]["provider"] == "ollama"
    assert body["runtime"]["chat"]["state"] == "unavailable"
    assert body["preflight"]["checks"]["assistant"]["status"] == "unavailable"
    assert "Ollama" in body["preflight"]["checks"]["assistant"]["message"]


def test_system_status_reports_missing_chat_model_structured(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing.gguf"

    monkeypatch.setattr(chat_service.SETTINGS, "chat_enabled", "auto")
    monkeypatch.setattr(chat_service.SETTINGS, "chat_provider", "packaged_llamacpp")
    monkeypatch.setattr(chat_service.SETTINGS, "chat_model_path", missing_model)

    client = make_client(tmp_path)
    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["runtime"]["chat"]["state"] == "missing_model"
    assert body["runtime"]["chat"]["model_exists"] is False
    assert body["preflight"]["checks"]["assistantModel"]["status"] == "missing"
    assert "GGUF" in body["preflight"]["checks"]["assistantModel"]["actionHint"]


def test_system_status_reports_chat_runtime_diagnostics(monkeypatch, tmp_path):
    model_path = tmp_path / "assistant.gguf"
    model_path.write_bytes(b"stub")

    diagnostics = llama_diagnostics(False)
    monkeypatch.setattr(chat_service, "_llama_cpp_runtime_diagnostics", lambda: diagnostics)
    monkeypatch.setattr(chat_service.SETTINGS, "chat_enabled", "auto")
    monkeypatch.setattr(chat_service.SETTINGS, "chat_provider", "packaged_llamacpp")
    monkeypatch.setattr(chat_service.SETTINGS, "chat_model_path", model_path)

    response = make_client(tmp_path).get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    runtime_chat = body["runtime"]["chat"]
    runtime_check = body["preflight"]["checks"]["assistantRuntime"]
    assert runtime_chat["state"] == "missing_runtime"
    assert runtime_chat["python_executable"] == r"C:\Python312\python.exe"
    assert runtime_chat["expected_venv_python"] == r"C:\repo\.venv\Scripts\python.exe"
    assert runtime_chat["running_in_expected_venv"] is False
    assert runtime_chat["llama_cpp_import_status"] == "missing"
    assert runtime_check["status"] == "missing"
    assert runtime_check["details"]["pythonExecutable"] == r"C:\Python312\python.exe"
    assert runtime_check["details"]["setupCommand"] == r".venv\Scripts\python.exe -m pip install llama-cpp-python"
    assert "Restart the SafeTrace backend" in runtime_check["details"]["restartRequired"]


def test_system_status_includes_vlm_profiles_with_installed_assets(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight_512m = tmp_path / "models" / "vlm" / "lightweight-512m"
    enhanced = tmp_path / "models" / "vlm" / "enhanced-2b"
    enhanced_3b = tmp_path / "models" / "vlm" / "enhanced-3b"
    lightweight.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (lightweight / "config.json").write_text("{}", encoding="utf-8")
    (enhanced / "model.safetensors").write_bytes(b"placeholder")

    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_512m_model_path", lightweight_512m)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", enhanced)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_3b_model_path", enhanced_3b)
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)
    monkeypatch.setattr(
        server_module,
        "device_gateway_payload",
        lambda settings: {
            "configuredDevice": "auto",
            "gpuAutoEnabled": True,
            "torch": {"installed": True, "cuda_available": True, "cuda_device_count": 1},
            "hardware": {"nvidia_smi_available": True, "nvidia_gpu_detected": True, "gpu_name": "RTX Test"},
            "components": {
                "detector": {"selected": "cuda", "available": True, "reason": "Auto selected CUDA."},
                "mobileSam": {"selected": "cuda", "available": True, "reason": "Auto selected CUDA."},
                "lightweightVlm": {"selected": "cuda", "available": True, "reason": "Auto selected CUDA."},
                "enhancedVlm": {
                    "selected": "cuda",
                    "available": True,
                    "requires_gpu": True,
                    "reason": "PyTorch CUDA runtime is available.",
                },
            },
            "gpuUnavailableReason": None,
        },
    )

    response = make_client(tmp_path).get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    profiles = {profile["id"]: profile for profile in body["vlm"]["profiles"]}
    assert body["vlm"]["selectedProfile"] == "rule_based"
    assert body["vlm"]["enabled"] is False
    assert body["vlm"]["active"] is False
    assert body["vlm"]["requestedVisualExplanationMode"] == "rule_based"
    assert body["vlm"]["actualExplanationMode"] == "rule_based"
    assert body["vlm"]["ruleBasedFallbackActive"] is True
    assert body["vlm"]["lightweightModelPathChecked"].replace("\\", "/").endswith("models/vlm/lightweight-512m")
    assert body["vlm"]["runtimeAvailable"] is True
    assert profiles["rule_based"]["installed"] is True
    assert profiles["rule_based"]["available"] is True
    assert profiles["rule_based"]["requiresActivation"] is False
    assert profiles["lightweight_256m"]["installed"] is True
    assert profiles["lightweight_256m"]["available"] is True
    assert profiles["lightweight_256m"]["resourceLevel"] == "low"
    assert profiles["lightweight_256m"]["deprecated"] is False
    assert profiles["lightweight_256m"]["notViable"] is False
    assert profiles["lightweight_256m"]["fallback"] is True
    assert "Low-resource 256M verifier fallback layer" in profiles["lightweight_256m"]["message"]
    assert profiles["lightweight_512m"]["installed"] is False
    assert profiles["lightweight_512m"]["available"] is False
    assert profiles["lightweight_512m"]["candidate"] is True
    assert profiles["lightweight_512m"]["resourceLevel"] == "medium"
    assert profiles["enhanced_2b"]["installed"] is True
    assert profiles["enhanced_2b"]["available"] is True
    assert profiles["enhanced_2b"]["resourceLevel"] == "gpu_high"
    assert profiles["enhanced_2b"]["requiresGpu"] is True
    assert profiles["enhanced_3b"]["installed"] is False
    assert profiles["enhanced_3b"]["available"] is False
    assert profiles["enhanced_3b"]["candidate"] is True
    assert profiles["enhanced_3b"]["resourceLevel"] == "very_high"


def test_system_status_reports_candidate_vlm_profiles_when_assets_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", tmp_path / "missing-256")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_512m_model_path", tmp_path / "missing-512")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", tmp_path / "missing-2b")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_3b_model_path", tmp_path / "missing-3b")
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    response = make_client(tmp_path).get("/api/system/status")

    assert response.status_code == 200
    profiles = {profile["id"]: profile for profile in response.json()["vlm"]["profiles"]}
    assert profiles["rule_based"]["available"] is True
    assert profiles["lightweight_256m"]["installed"] is False
    assert profiles["lightweight_256m"]["deprecated"] is False
    assert profiles["lightweight_256m"]["fallback"] is True
    assert profiles["lightweight_512m"]["installed"] is False
    assert profiles["lightweight_512m"]["available"] is False
    assert "512M lightweight verifier layer" in profiles["lightweight_512m"]["message"]
    assert profiles["enhanced_3b"]["installed"] is False
    assert profiles["enhanced_3b"]["available"] is False
    assert "Enhanced 3B VLM candidate" in profiles["enhanced_3b"]["message"]
    assert response.json()["vlm"]["selectedProfile"] == "rule_based"
    assert response.json()["vlm"]["active"] is False


def test_system_status_vlm_profiles_ignore_readme_only_placeholders(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    enhanced = tmp_path / "models" / "vlm" / "enhanced-2b"
    lightweight.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (lightweight / "README.txt").write_text("placeholder", encoding="utf-8")
    (enhanced / "README.md").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_512m_model_path", tmp_path / "missing-512")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", enhanced)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_3b_model_path", tmp_path / "missing-3b")
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    response = make_client(tmp_path).get("/api/system/status")

    assert response.status_code == 200
    profiles = {profile["id"]: profile for profile in response.json()["vlm"]["profiles"]}
    assert profiles["rule_based"]["installed"] is True
    assert profiles["lightweight_256m"]["installed"] is False
    assert profiles["lightweight_256m"]["available"] is False
    assert profiles["lightweight_512m"]["installed"] is False
    assert profiles["lightweight_512m"]["available"] is False
    assert profiles["enhanced_2b"]["installed"] is False
    assert profiles["enhanced_2b"]["available"] is False
    assert profiles["enhanced_3b"]["installed"] is False
    assert profiles["enhanced_3b"]["available"] is False


def test_system_status_does_not_report_vlm_parent_directory_as_loadable(monkeypatch, tmp_path):
    parent = tmp_path / "models" / "vlm"
    lightweight = parent / "lightweight-256m"
    enhanced = parent / "enhanced-2b"
    lightweight.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")

    monkeypatch.setattr(server_module.SETTINGS, "vlm_provider", "local")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_model_dir", parent)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_512m_model_path", tmp_path / "missing-512")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", enhanced)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_3b_model_path", tmp_path / "missing-3b")
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)
    monkeypatch.setattr(vlm_reasoner, "_transformers_runtime_available", lambda: True)

    response = make_client(tmp_path).get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["models"]["vlm"]["status"] == "unavailable"
    assert "profile parent directory" in body["models"]["vlm"]["message"]
    assert body["models"]["vlm"]["path"] == str(parent)
    profiles = {profile["id"]: profile for profile in body["vlm"]["profiles"]}
    assert profiles["lightweight_256m"]["installed"] is True
    assert profiles["lightweight_256m"]["path"].replace("\\", "/").endswith("models/vlm/lightweight-256m")
    assert profiles["lightweight_512m"]["installed"] is False


def test_vlm_settings_endpoint_updates_selection_without_loading_model(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    enhanced = tmp_path / "models" / "vlm" / "enhanced-2b"
    lightweight.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (lightweight / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", enhanced)
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    client = make_client(tmp_path)
    response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedProfile"] == "lightweight_256m"
    assert body["enabled"] is True
    assert body["active"] is True
    assert body["requestedVisualExplanationMode"] == "lightweight_256m"
    assert body["actualExplanationMode"] == "lightweight_256m"
    assert body["ruleBasedFallbackActive"] is False

    status = client.get("/api/system/status").json()
    assert status["vlm"]["selectedProfile"] == "lightweight_256m"
    assert status["vlm"]["enabled"] is True
    assert status["vlm"]["active"] is True
    assert status["vlm"]["actualExplanationMode"] == "lightweight_256m"
    assert status["models"]["vlm"]["status"] == "available"
    assert status["models"]["vlm"]["details"]["actualExplanationMode"] == "lightweight_256m"
    assert status["models"]["vlm"]["path"].replace("\\", "/").endswith("models/vlm/lightweight-256m")


def test_vlm_settings_endpoint_does_not_activate_removed_lightweight_profile(monkeypatch, tmp_path):
    lightweight_512m = tmp_path / "models" / "vlm" / "lightweight-512m"
    lightweight_512m.mkdir(parents=True)
    (lightweight_512m / "model.safetensors").write_bytes(b"placeholder")

    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", tmp_path / "models" / "vlm" / "lightweight-256m")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_512m_model_path", lightweight_512m)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", tmp_path / "models" / "vlm" / "enhanced-2b")
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    client = make_client(tmp_path)
    response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedProfile"] == "lightweight_256m"
    assert body["enabled"] is False
    assert body["active"] is False
    assert body["actualExplanationMode"] == "rule_based"
    assert "unavailable" in body["message"].lower()
    profiles = {profile["id"]: profile for profile in body["profiles"]}
    assert profiles["lightweight_256m"]["installed"] is False
    assert profiles["lightweight_512m"]["installed"] is True
    assert profiles["lightweight_512m"]["available"] is True


def test_safe_mode_lightweight_vlm_worker_status_can_be_active(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    enhanced = tmp_path / "models" / "vlm" / "enhanced-2b"
    lightweight.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")

    monkeypatch.setattr(server_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(server_module.SETTINGS, "safe_mode_allow_mobilesam", True)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(server_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(server_module.SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", enhanced)
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    client = make_client(tmp_path)
    status = client.get("/api/system/status").json()

    assert status["safeMode"] is True
    assert status["vlm"]["selectedProfile"] == "lightweight_256m"
    assert status["vlm"]["enabled"] is True
    assert status["vlm"]["active"] is True
    assert status["vlm"]["actualExplanationMode"] == "lightweight_256m"
    assert status["vlm"]["vlmSuppressedReason"] is None
    assert status["vlm"]["ruleBasedFallbackActive"] is True
    assert status["vlm"]["lightweightVlmWorkerEnabled"] is True
    assert status["runtime"]["analysis"]["lightweightVlmWorkerEnabled"] is True
    assert status["runtime"]["analysis"]["safeModeMessage"] == (
        "MobileSAM worker + Local VLM Assist active. Fast Local Analysis remains available."
    )
    assert status["models"]["vlm"]["status"] == "available"
    assert status["models"]["vlm"]["details"]["lightweightVlmWorkerEnabled"] is True


def test_safe_mode_lightweight_vlm_worker_can_be_active_without_mobilesam(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    enhanced = tmp_path / "models" / "vlm" / "enhanced-2b"
    lightweight.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")

    monkeypatch.setattr(server_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(server_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_enabled", "false")
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(server_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(server_module.SETTINGS, "lightweight_vlm_worker_timeout_seconds", 120)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_max_evidence_frames", 5)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_job_timeout_seconds", 60)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_max_quality_failures", 1)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", enhanced)
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    client = make_client(tmp_path)
    status = client.get("/api/system/status").json()

    assert status["safeMode"] is True
    assert status["runtime"]["analysis"]["safeMode"] is True
    assert status["runtime"]["analysis"]["safeModeMobileSamAllowed"] is False
    assert status["runtime"]["analysis"]["mobileSamWorkerEnabled"] is False
    assert status["runtime"]["analysis"]["lightweightVlmWorkerEnabled"] is True
    assert status["runtime"]["analysis"]["lightweightVlmWorkerTimeoutSeconds"] == 120
    assert status["runtime"]["analysis"]["lightweightVlmEvidenceBudget"] == 5
    assert status["runtime"]["analysis"]["lightweightVlmFrameLimit"] == 5
    assert status["runtime"]["analysis"]["lightweightVlmJobTimeoutSeconds"] == 60
    assert status["runtime"]["analysis"]["lightweightVlmMaxQualityFailures"] == 1
    assert status["runtime"]["analysis"]["safeModeMessage"] == (
        "Local VLM Assist worker enabled. Fast Local Analysis remains available; MobileSAM disabled."
    )
    assert status["vlm"]["selectedProfile"] == "lightweight_256m"
    assert status["vlm"]["active"] is True
    assert status["vlm"]["lightweightVlmWorkerEnabled"] is True
    assert status["vlm"]["lightweightVlmExplanationSource"] == "worker"
    assert status["vlm"]["lightweightVlmEvidenceBudget"] == 5
    assert status["vlm"]["lightweightVlmFrameLimit"] == 5
    assert status["vlm"]["lightweightVlmJobTimeoutSeconds"] == 60
    assert status["vlm"]["lightweightVlmMaxQualityFailures"] == 1
    assert status["runtime"]["visual_explanations"]["lightweightVlmWorkerEnabled"] is True
    assert status["runtime"]["visual_explanations"]["lightweightVlmExplanationSource"] == "worker"
    assert status["runtime"]["visual_explanations"]["lightweightVlmEvidenceBudget"] == 5
    assert status["runtime"]["visual_explanations"]["lightweightVlmFrameLimit"] == 5
    assert status["runtime"]["visual_explanations"]["lightweightVlmJobTimeoutSeconds"] == 60


def test_vlm_settings_endpoint_respects_hard_disabled_configuration(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")

    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "false")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    client = make_client(tmp_path)
    response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedProfile"] == "lightweight_256m"
    assert body["enabled"] is False
    assert body["active"] is False
    assert body["message"] == "Local visual review is disabled by configuration. Fast Local Analysis remains available."

    status = client.get("/api/system/status").json()
    assert status["vlm"]["selectedProfile"] == "lightweight_256m"
    assert status["vlm"]["enabled"] is False
    assert status["vlm"]["active"] is False
    assert status["models"]["vlm"]["status"] == "disabled"
    assert status["preflight"]["checks"]["visualExplanations"]["details"]["explanationSource"] == "rule_based"


def test_vlm_settings_endpoint_preserves_rule_based_fallback_for_missing_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", tmp_path / "missing-lightweight")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enhanced_model_path", tmp_path / "missing-enhanced")
    monkeypatch.setattr(server_module, "_vlm_runtime_available", lambda: True)

    response = make_client(tmp_path).post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "enhanced_2b", "enabled": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedProfile"] == "enhanced_2b"
    assert body["enabled"] is False
    assert body["active"] is False
    assert body["actualExplanationMode"] == "rule_based"


def test_safe_mode_system_status_suppresses_vlm_without_profile_preflight(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")

    def fail_if_vlm_runtime_checked():
        raise AssertionError("safe mode should not run local VLM runtime preflight")

    monkeypatch.setattr(server_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module, "device_gateway_payload", lambda settings: _cpu_device_gateway_payload())
    monkeypatch.setattr(server_module, "_vlm_runtime_available", fail_if_vlm_runtime_checked)

    client = make_client(tmp_path)
    settings_response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )

    assert settings_response.status_code == 200
    settings_body = settings_response.json()
    assert settings_body["selectedProfile"] == "lightweight_256m"
    assert settings_body["enabled"] is False
    assert settings_body["active"] is False
    assert settings_body["actualExplanationMode"] == "rule_based"
    assert settings_body["vlmSuppressedReason"] == "safe_mode"

    status = client.get("/api/system/status")

    assert status.status_code == 200
    body = status.json()
    assert body["safeMode"] is True
    assert body["runtime"]["analysis"]["safeMode"] is True
    assert body["runtime"]["analysis"]["effectiveDevice"] == "cpu"
    assert "deviceGateway" in body["limits"]
    assert "gateway" in body["runtime"]["device"]
    assert "lightweightVlmDevice" in body["runtime"]["analysis"]
    assert "enhancedVlmDevice" in body["runtime"]["analysis"]
    assert "deviceGateway" in body["vlm"]
    assert "lightweightVlmPrimaryPolicy" in body["models"]["vlm"]["details"]
    assert body["models"]["mobileSam"]["status"] == "disabled"
    assert body["models"]["vlm"]["status"] == "disabled"
    assert body["vlm"]["vlmSuppressedReason"] == "safe_mode"
    assert "Low-resource 256M verifier fallback layer" in body["vlm"]["profiles"][1]["message"]
    assert body["vlm"]["ruleBasedFallbackActive"] is True
    assert body["vlm"]["fallbackReason"]
    assert "Fast Local Analysis remains available" in body["vlm"]["message"]


def test_safe_mode_system_status_allows_experimental_mobilesam_without_vlm(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "mobile_sam.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(server_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(server_module.SETTINGS, "safe_mode_allow_mobilesam", True)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_checkpoint", checkpoint)
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "disabled")
    monkeypatch.setattr(server_module, "_mobile_sam_runtime_available", lambda: True)
    monkeypatch.setattr(server_module, "device_gateway_payload", lambda settings: _cpu_device_gateway_payload())
    client = make_client(tmp_path)

    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["safeMode"] is True
    assert body["runtime"]["analysis"]["safeMode"] is True
    assert body["runtime"]["analysis"]["safeModeMobileSamAllowed"] is True
    assert body["runtime"]["analysis"]["effectiveDevice"] == "cpu"
    assert "MobileSAM refinement may run on selected evidence frames" in body["runtime"]["analysis"]["safeModeMessage"]
    assert body["models"]["mobileSam"]["status"] == "available"
    assert body["models"]["mobileSam"]["details"]["safeModeMobileSamAllowed"] is True
    assert body["models"]["mobileSam"]["details"]["mobileSamEnabled"] is True
    assert body["models"]["mobileSam"]["details"]["mobileSamWorkerEnabled"] is False
    assert body["vlm"]["active"] is False
    assert body["vlm"]["enabled"] is False
    assert body["vlm"]["vlmSuppressedReason"] == "hard_disabled"
    assert body["vlm"]["actualExplanationMode"] == "rule_based"


def test_safe_mode_system_status_reports_mobilesam_worker(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "mobile_sam.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(server_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(server_module.SETTINGS, "safe_mode_allow_mobilesam", True)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_worker_timeout_seconds", 60)
    monkeypatch.setattr(server_module.SETTINGS, "mobile_sam_checkpoint", checkpoint)
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "disabled")
    monkeypatch.setattr(server_module, "_mobile_sam_runtime_available", lambda: True)
    client = make_client(tmp_path)

    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["safeMode"] is True
    assert body["runtime"]["analysis"]["safeModeMobileSamAllowed"] is True
    assert body["runtime"]["analysis"]["mobileSamWorkerEnabled"] is True
    assert body["runtime"]["analysis"]["mobileSamWorkerTimeoutSeconds"] == 60
    assert "MobileSAM worker refinement" in body["runtime"]["analysis"]["safeModeMessage"]
    assert body["models"]["mobileSam"]["status"] == "available"
    assert body["models"]["mobileSam"]["details"]["mobileSamWorkerEnabled"] is True
    assert body["models"]["mobileSam"]["details"]["mobileSamWorkerTimeoutSeconds"] == 60
    assert body["models"]["mobileSam"]["details"]["mobileSamRefinementSource"] == "worker"
    assert "MobileSAM worker refinement enabled" in body["models"]["mobileSam"]["message"]
    assert body["vlm"]["active"] is False
    assert body["vlm"]["vlmSuppressedReason"] == "hard_disabled"
