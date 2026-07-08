from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_frontend(path: str) -> str:
    return (ROOT / "frontend-react" / "src" / path).read_text(encoding="utf-8")


def read_all_frontend_sources() -> str:
    src_root = ROOT / "frontend-react" / "src"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in src_root.rglob("*")
        if path.suffix in {".ts", ".tsx"}
    )


def test_local_runtime_docs_use_venv_python_and_openmp_defaults():
    docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/safetrace_local_validation.md",
            "docs/safetrace_assistant.md",
            "docs/safetrace_performance_troubleshooting.md",
            "docs/react_fastapi_backend_integration.md",
        )
    )

    assert ".venv\\Scripts\\python.exe -m uvicorn src.api.server:app" in docs
    assert "set KMP_DUPLICATE_LIB_OK=TRUE" in docs
    assert "set OMP_NUM_THREADS=1" in docs
    assert ".venv\\Scripts\\python.exe -c \"import sys; print(sys.executable); import llama_cpp; print('llama_cpp ok')\"" in docs
    assert "Safe local validation mode" in docs
    assert "set SAFETRACE_ANALYSIS_SAFE_MODE=true" in docs
    assert "set SAFETRACE_DEVICE=cpu" in docs
    assert "set SAFETRACE_MOBILESAM_ENABLED=false" in docs
    assert "set SAFETRACE_VLM_ENABLED=false" in docs
    assert "embeddingRequested=false" in docs
    assert "object/rule-based frame ranking" in docs
    assert "driver without seatbelt" in docs
    assert "MobileSAM can improve mask and evidence quality" in docs
    assert "SafeTrace_RC_MobileSAM_RuleBased" in docs
    assert "SafeTrace_RC_MobileSAM_LightweightVLM_Experimental" in docs
    assert "scripts\\start_safetrace_dev_backend.bat" in docs
    assert "lightweightVlmWorkerEnabled=false" in docs
    assert "lightweightVlmExplanationSource=rule_based" in docs
    assert "python -m uvicorn src.api.server:app" not in docs


def test_dev_backend_startup_helper_sets_vlm_chat_and_llama_cpp_diagnostics():
    script = (ROOT / "scripts" / "start_safetrace_dev_backend.bat").read_text(encoding="utf-8")

    assert "cd /d \"%REPO_ROOT%\"" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "Falling back to system Python from PATH" in script
    assert "SAFETRACE_ANALYSIS_SAFE_MODE=true" in script
    assert "SAFETRACE_DEVICE=auto" in script
    assert "SAFETRACE_ENABLE_GPU_AUTO=true" in script
    assert "SAFETRACE_MOBILESAM_DEVICE=auto" in script
    assert "SAFETRACE_MOBILESAM_ENABLED=auto" in script
    assert "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=true" in script
    assert "SAFETRACE_MOBILESAM_WORKER_ENABLED=true" in script
    assert "SAFETRACE_VLM_ENABLED=true" in script
    assert "SAFETRACE_VLM_PROFILE=lightweight_512m" in script
    assert "SAFETRACE_VLM_MODEL_PATH=models\\vlm\\lightweight-512m" in script
    assert "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH=models\\vlm\\lightweight-256m" in script
    assert "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH=models\\vlm\\lightweight-512m" in script
    assert "SAFETRACE_VLM_ENHANCED_MODEL_PATH=models\\vlm\\enhanced-2b" in script
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=true" in script
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60" in script
    assert "SAFETRACE_LIGHTWEIGHT_VLM_PRIMARY=auto" in script
    assert "SAFETRACE_LIGHTWEIGHT_VLM_FALLBACK=256m" in script
    assert "SAFETRACE_LIGHTWEIGHT_VLM_CPU_PREFER_256M=true" in script
    assert "SAFETRACE_VLM_FRAME_LIMIT=5" in script
    assert "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES=5" in script
    assert "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS=0" in script
    assert "SAFETRACE_VLM_MAX_QUALITY_FAILURES=1" in script
    assert "SAFETRACE_VLM_MAX_TOKENS=40" in script
    assert "SAFETRACE_CHAT_ENABLED=auto" in script
    assert "SAFETRACE_CHAT_PROVIDER=packaged_llamacpp" in script
    assert "SAFETRACE_CHAT_MODEL_PATH=models\\chat\\safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf" in script
    assert "import llama_cpp" in script
    assert "llama_cpp could not be imported" in script
    assert "llama-cpp-python" in script
    assert "Checking backend runtime imports" in script
    assert "Local visual review frame limit" in script
    assert "Local visual review shared time budget: disabled" in script
    assert "import faiss" in script
    assert "import ultralytics" in script
    assert '"numpy<2"' in script
    assert "Backend not started because analysis would fail in this environment." in script
    assert "-m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info" in script


def test_packaged_supervisor_is_preserved_and_dev_helper_is_separate():
    launcher_builder = (ROOT / "scripts" / "build_desktop_prototype.py").read_text(encoding="utf-8")
    entrypoint = (ROOT / "src" / "api" / "__main__.py").read_text(encoding="utf-8")
    spec = (ROOT / "packaging" / "backend" / "safetrace_backend.spec").read_text(encoding="utf-8")
    dev_helper = (ROOT / "scripts" / "start_safetrace_dev_backend.bat").read_text(encoding="utf-8")

    assert "SafeTraceLauncher.bat" in launcher_builder
    assert "backend_supervisor.bat" in launcher_builder
    assert "goto backend_loop" in launcher_builder
    assert "Backend exited with %%EXIT_CODE%%. Restarting in 5 seconds." in launcher_builder
    assert "HEALTH_WAIT_SECONDS=90" in launcher_builder
    assert "Existing backend supervisor detected" in launcher_builder
    assert "--mobile-sam-worker" in entrypoint
    assert "--lightweight-vlm-worker" in entrypoint
    assert "from src.mobile_sam_worker import main as mobile_sam_worker_main" in entrypoint
    assert "from src.lightweight_vlm_worker import main as lightweight_vlm_worker_main" in entrypoint
    assert "src.mobile_sam_worker" in spec
    assert "src.lightweight_vlm_worker" in spec
    assert "SafeTrace Dev" in dev_helper
    assert "SafeTraceLauncher.bat" not in dev_helper
    assert "backend_supervisor.bat" not in dev_helper


def test_sidebar_uses_explicit_vlm_mode_selector():
    source = read_frontend("components/Sidebar.tsx")

    vite_config = (ROOT / "frontend-react" / "vite.config.ts").read_text(encoding="utf-8")

    assert "SafeTrace local runtime frontend" in source
    assert "Build {FRONTEND_BUILD_TIME}" in source
    assert "__SAFETRACE_BUILD_TIME__" in source
    assert "__SAFETRACE_BUILD_TIME__" in vite_config
    assert "new Date().toISOString()" in vite_config
    assert "Visual explanations" in source
    assert "Engine mode" in source
    assert "VLM explanation mode" not in source
    assert "Safe local mode" not in source
    assert "Fast Local Analysis" in source
    assert "Local VLM Assist" in source
    assert "Advanced GPU VLM Assist" in source
    assert "Rule-based + Lightweight VLM" not in source
    assert "Rule-based + Lightweight + Enhanced VLM" not in source
    assert "Activate local visual review" in source
    assert "Fast Local Analysis is active." in source
    assert "Fast Local Analysis remains available." in source
    assert "Visual explanations are hidden. Turn on to choose a local engine mode." in source
    assert "not installed." in source
    assert "unavailable." in source
    assert "SafeTrace adds local visual review on selected evidence frames" in source
    assert "Evidence cards show" in source
    assert "falling back to Fast Local Analysis" in source
    assert "Connect to the local runtime to activate VLM assist. Fast Local Analysis remains available." in source
    assert "Local visual review is disabled by backend configuration. Fast Local Analysis remains available." in source
    assert "safeModeActive" in source
    assert "Runtime guard active" in source
    assert "MobileSAM refinement available" in source
    assert "MobileSAM worker enabled" in source
    assert "Local visual review workers enabled" in source
    assert "budgeted for selected evidence frames" not in source
    assert "Base analysis remains available if a worker fails." in source
    assert "MobileSAM worker refinement enabled. Detector-box fallback is used if the worker fails." in source
    assert "MobileSAM refinement and local VLM assist may run on selected evidence frames." in source
    assert "Optional visual refinement is disabled by the runtime guard." in source
    assert "systemStatus?.vlm?.vlmSuppressedReason === 'safe_mode'" in source
    assert "showVisualExplanations ? (" in source
    assert "selectedProfile !== 'rule_based' ? (" in source
    assert "vlmGloballyDisabled" in source
    assert "|| vlmGloballyDisabled" in source
    assert "Fast Local Analysis: Fastest local review." in source
    assert "Local VLM Assist: Adds local visual explanation on selected evidence frames." in source
    assert "falls back to 256M when needed" in source
    assert "Advanced GPU VLM Assist: Uses the GPU enhanced model for deeper local visual review" in source
    assert "Engine mode help" in source
    assert "Local VLM assist adds visual review on selected evidence frames." in source
    assert "Use Fast Local Analysis when you want the quickest local pass." in source
    assert "absolute left-1/2 top-7" not in source
    assert "w-72 -translate-x-1/2" not in source
    assert "disabled={vlmUnavailable}" not in source
    assert "onClick={() => updateSettings({ visualExplanations: !showVisualExplanations })}" in source
    assert "aria-label=\"Activate selected local visual review\"" in source


def test_frontend_resets_removed_vlm_profiles_from_local_storage():
    source = read_frontend("App.tsx")

    assert "staleRemovedProfile" in source
    assert "storedProfile === 'lightweight_256m'" in source
    assert "vlmProfile = isVlmProfileId(storedProfile) && !staleRemovedProfile ? storedProfile : 'rule_based'" in source


def test_layered_vlm_selector_and_frame_card_copy_are_visible():
    sidebar_source = read_frontend("components/Sidebar.tsx")
    card_source = read_frontend("components/FrameEvidenceCard.tsx")
    details_source = read_frontend("components/TechnicalDetails.tsx")
    smoke_script = (ROOT / "scripts" / "smoke_lightweight_512m_vlm.py").read_text(encoding="utf-8")

    assert "Fast Local Analysis" in sidebar_source
    assert "Local VLM Assist" in sidebar_source
    assert "Advanced GPU VLM Assist" in sidebar_source
    assert "Rule-based + Lightweight VLM" not in sidebar_source
    assert "Rule-based + Lightweight + Enhanced VLM" not in sidebar_source
    assert "local visual explanation on selected evidence frames" in sidebar_source
    assert "profile.id === 'enhanced_2b' && profile.installed && profile.available" in sidebar_source
    assert "rule_template_plus_lightweight_vlm" not in card_source
    assert "rule_template_plus_lightweight_plus_enhanced" not in card_source
    assert "Technical evidence" in details_source
    assert "Safety review" in card_source
    assert "Why this was flagged" in card_source
    assert "Visual review" in card_source
    assert "Review level" in card_source
    assert "What to check in the original footage" in card_source
    assert "Final explanation source:" not in card_source
    assert "--image" in smoke_script
    assert "renderedTemplate" in smoke_script
    assert "parsedFields" in smoke_script
    assert "qualityIssue" in smoke_script


def test_app_uses_visual_toggle_for_display_and_vlm_profile_for_backend_vlm():
    source = read_frontend("App.tsx")

    assert "visualExplanations: true" in source
    assert "vlmProfile: vlmSettings.vlmProfile" in source
    assert "vlmEnabled: vlmSettings.vlmEnabled" in source
    assert "safetrace:vlm:selectedProfile" in source
    assert "safetrace:vlm:enabled" in source
    assert "enableVlm: shouldRequestVlm(settings)" in source
    assert "FRESH_BACKEND_HEARTBEAT_MS" in source
    assert "isBackendHeartbeatFresh(status)" in source
    assert "Analysis timed out because the backend heartbeat became stale" in source
    assert "Batch analysis timed out because the backend status stopped updating" in source
    assert "selectedProfile: settings.vlmProfile" in source
    assert "setSettings(nextSettings)" not in source
    assert "systemStatus?.vlm?.selectedProfile" not in source
    assert "showExplanations={settings.visualExplanations}" in source
    assert "BackendJobFailureError" in source
    assert "metricString(status.metrics, 'errorType')" in source
    assert "jobFailureDebugDetails(status)" in source
    assert "Analysis failed." in source
    assert "checkBackendHealth" in source
    assert "Job status could not be refreshed while the backend was still healthy." in source
    assert "status error" not in source.lower()


def test_phase_a_use_case_profiles_and_queue_safety_are_visible():
    app_source = read_frontend("App.tsx")
    sidebar_source = read_frontend("components/Sidebar.tsx")
    service_source = read_frontend("services/analysisService.ts")
    types_source = read_frontend("types/analysis.ts")
    profiles_source = read_frontend("data/useCaseProfiles.ts")
    queue_source = read_frontend("components/VideoQueue.tsx")
    summary_source = read_frontend("components/AnalysisSummary.tsx")
    media_source = read_frontend("components/SelectedMediaViewer.tsx")
    query_source = read_frontend("components/QueryTabs.tsx")
    evidence_source = read_frontend("components/FrameEvidenceCard.tsx")
    violation_summary_source = read_frontend("components/ViolationSummary.tsx")

    assert "Use-case profile" in sidebar_source
    assert "Seatbelt compliance" in profiles_source
    assert "Phone use / distracted driving" in profiles_source
    assert "Helmet / PPE compliance" in profiles_source
    assert "Uniform compliance" in profiles_source
    assert "General safety review" in profiles_source
    assert "Custom / division-specific policy" in profiles_source
    assert "defaultQuery" in profiles_source
    assert "backendSupportLevel" in profiles_source
    assert "driver or occupant without seatbelt" in profiles_source
    assert "driver using phone while driving" in profiles_source
    assert "metadata_only" in profiles_source
    assert "getProfileQueryConflict" in profiles_source
    assert "buildEffectiveProfileQuery" in profiles_source
    assert "useCaseProfile: resolveUseCaseProfile()" in app_source
    assert "serializeUseCaseProfile" in service_source
    assert "formData.append('useCaseProfile', useCaseProfile)" in service_source
    assert "UseCaseProfileSelection" in types_source
    assert "BackendSupportLevel" in types_source
    assert "withEffectiveProfile" in app_source
    assert "profileForRequest" in app_source
    assert "result.media.useCaseProfile = profileForRequest" in app_source
    assert "effectiveQuery" in app_source
    assert "requestedQuery" in app_source
    assert "Profile:" in queue_source
    assert "Job Queue" in queue_source
    assert "Waiting for worker" in queue_source
    assert "Queue job" in app_source
    assert "activeAnalysisMediaIds" in app_source
    assert "disabled={!canAnalyze}" in query_source
    assert "isLoading || !canAnalyze" not in query_source
    assert "Use-case profile" in summary_source
    assert "Use-case profile" in media_source
    assert "supportLevelLabel" in query_source
    assert "Effective query sent" in query_source
    assert "Resolve the profile/query conflict" in query_source
    assert "Detector classes involved" in evidence_source
    assert "isViolationAlignedWithProfile" in profiles_source
    assert "profileFindingContext" in profiles_source
    assert "de-prioritized from the prominent findings panel" in evidence_source
    assert "Review confidence:" in evidence_source
    assert "Review level:" in evidence_source
    assert "hidden from this prominent overview" in violation_summary_source
    assert "isViolationAlignedWithProfile" in violation_summary_source


def test_phase_a_cache_copy_is_browser_only_and_validated():
    app_source = read_frontend("App.tsx")
    cache_source = read_frontend("services/resultCache.ts")

    assert "hasUsableCachedPayload" in app_source
    assert "stale or incomplete browser cache" in app_source
    assert "Ignored an incomplete browser cache reference" in app_source
    assert "PERSIST_BROWSER_RESULT_CACHE = false" in app_source
    assert "Browser result cache is session-only by default and clears on refresh" in app_source
    assert "clearSafeTraceResultCacheStorageKeys" in app_source
    assert "Clear cache removes SafeTrace browser keys immediately" in app_source
    assert "Clear browser result cache" in app_source
    assert "deleteKnownBackendJobsForCacheEntries" in app_source
    assert "deleteJob(jobId)" in app_source
    assert "Queued/running jobs are never deleted by this cache action." in app_source
    assert "backend job folders may still exist" in app_source
    assert "MAX_CACHE_ENTRY_BYTES" in cache_source
    assert "RESULT_CACHE_STORAGE_PREFIXES" in cache_source


def test_phase_a3_queue_preview_rerun_and_vlm_regressions_are_guarded():
    app_source = read_frontend("App.tsx")
    sidebar_source = read_frontend("components/Sidebar.tsx")
    media_source = read_frontend("components/SelectedMediaViewer.tsx")
    queue_source = read_frontend("components/VideoQueue.tsx")

    assert "objectUrlsRef" in app_source
    assert "trackObjectUrl(previewUrl)" in app_source
    assert "clearLocalPreview" not in app_source
    assert "stillUsedElsewhere" in app_source
    assert "Preview unavailable after refresh; re-upload this media to preview or run analysis again." in media_source
    assert "Run again" in app_source
    assert "Re-upload this media to run analysis again." in app_source
    assert "(rerun)" in app_source
    assert "selectedMediaStatus === 'completed' && !selectedFiles.length && !selectedFile" in app_source
    assert "setSelectedMedia(rerunMedia)" in app_source
    assert "setMediaLibrary((current) => [rerunMedia, ...current])" in app_source
    assert "Job Queue" in queue_source
    assert "Active job" in queue_source
    assert "Waiting for worker" in queue_source
    assert "mainVlmSelectorProfiles" in sidebar_source
    assert "selectorProfiles.map((profile)" in sidebar_source
    assert "disabled={profile.id !== 'rule_based' && vlmGloballyDisabled}" in sidebar_source
    assert "Local VLM assist is locked by backend configuration" in sidebar_source
    assert "backendVlmStatus?.actualExplanationMode || (vlmEnabled ? selectedProfile : '')" in sidebar_source
    assert "vlmEnabled: vlmActivationEnabled" in sidebar_source


def test_analysis_service_supports_optional_vlm_settings_endpoint():
    source = read_frontend("services/analysisService.ts")

    assert "system/vlm/settings" in source
    assert "vlmProfile" in source
    assert "vlmEnabled" in source
    assert "VLM_ARTIFACT_PATTERN" in source
    assert "VLM_ROLE_LABEL_PATTERN" in source
    assert "VLM_PROMPT_ECHO_PATTERN" in source
    assert "async function readErrorMessage(response: Response)" in source
    assert "const raw = await response.text()" in source
    assert "JSON.parse(raw)" in source
    assert "const body = await response.json()" not in source


def test_safety_insights_dashboard_is_standalone_view():
    app_source = read_frontend("App.tsx")
    dashboard_source = read_frontend("components/SafetyInsightsDashboard.tsx")

    assert "SafetyInsightsDashboard" in app_source
    assert "activeView === 'insights'" in app_source
    assert "Safety Insights" in app_source
    assert "Cached analysis overview" in dashboard_source
    assert "Operational hotspots" in dashboard_source
    assert "Review queue" in dashboard_source
    assert "Recent analyses" in dashboard_source
    assert "Violation counts by type" in dashboard_source
    assert "Per-video results" in dashboard_source
    assert "buildSafetyInsightsCsv" in dashboard_source
    assert "buildSafetyInsightsMarkdown" in dashboard_source
    assert "buildSafetyInsightsJson" in dashboard_source
    assert "safetrace-safety-insights.csv" in dashboard_source
    assert "safetrace-safety-insights.md" in dashboard_source
    assert "safetrace-safety-insights.json" in dashboard_source
    assert "Raw uploaded videos and copied evidence images are not stored" in dashboard_source


def test_frontend_error_handling_reads_response_body_once():
    analysis_source = read_frontend("services/analysisService.ts")
    chat_source = read_frontend("services/chatService.ts")

    assert "throw new Error(await readErrorMessage(response))" in analysis_source
    assert "async function readChatErrorMessage(response: Response)" in chat_source
    assert "throw new Error(await readChatErrorMessage(response))" in chat_source
    assert "const raw = await response.text()" in chat_source
    assert "JSON.parse(raw)" in chat_source
    assert "const body = await response.json()" not in chat_source


def test_evidence_cards_use_user_facing_explanation_sections_and_keeps_debug_in_details():
    source = read_frontend("components/FrameEvidenceCard.tsx")
    service_source = read_frontend("services/analysisService.ts")
    details_source = read_frontend("components/TechnicalDetails.tsx")

    assert "Selected because" in source
    assert "Detector-box fallback used" in source
    assert "Local visual review timed out for this frame; Fast Local Analysis remains available." in source
    assert "Local visual review was not run for this lower-priority frame." in source
    assert "Fast Local Analysis explanation" in source
    assert "Local VLM Assist explanation" in source
    assert "Advanced GPU VLM Assist explanation" in source
    assert "SafeTrace used local detector/rule evidence for this frame." in source
    assert "Local visual review did not add a confident result for this frame." in source
    assert "SafeTrace used local visual review to refine the detector/rule evidence." in source
    assert "SafeTrace used the enhanced GPU visual review to refine the detector/rule evidence." in source
    assert "VLM fallback:" not in source
    assert "VLM skip reason:" not in source
    assert "lightweightVlmFallbackReason" in source
    assert "Requested mode:" not in source
    assert "Actual source:" not in source
    assert "Lightweight VLM attempted:" not in source
    assert "Safety review" in source
    assert "Why this was flagged" in source
    assert "Visual review" in source
    assert "Review level" in source
    assert "What to check in the original footage" in source
    assert "Reason:" in source
    assert "Visual review may disagree with the base finding" in source
    assert "Visual review inconclusive" in source
    assert "Reviewer note:" in source
    assert "mobileSamRefinement" in source
    assert "mobileSamRefinement" in details_source
    assert "lightweightVlmExplanation" in source
    assert "lightweightVlmExplanation" in details_source
    assert "searchMetadata" in details_source
    assert "selectionReason" in source
    assert "rankingReason" in service_source
    assert "selectedFor" in service_source
    assert "VLM explanation: local provider" not in source
    assert "VLM explanation: Ollama" not in source


def test_assistant_has_limited_runtime_fallback_and_in_car_prompts():
    source = read_frontend("components/SafeTraceAssistant.tsx")

    assert "Was the driver wearing a seatbelt?" in source
    assert "Is the driver using a phone while driving?" in source
    assert "isLimitedFallbackAvailable" in source
    assert "Limited SafeTrace help is available without llama.cpp" in source
    assert "Runtime diagnostics" in source
    assert "Backend Python" in source
    assert "Expected .venv Python" in source
    assert "llama_cpp import" in source
    assert "<textarea" in source
    assert "Shift+Enter for a new line" in source
    assert "event.key === 'Enter' && !event.shiftKey" in source
    assert "flex-[1_1_65%]" in source
    assert "overflow-y-auto overflow-x-hidden" in source
    assert "[overflow-wrap:anywhere]" in source
    assert "canSubmit" in source
    assert "Selected job" in source
    assert "Copy selected job ID" in source
    assert "No selected result is attached" in source
    assert "because ${reason" in source


def test_frontend_surfaces_copyable_job_identifiers():
    summary_source = read_frontend("components/AnalysisSummary.tsx")
    media_source = read_frontend("components/SelectedMediaViewer.tsx")
    queue_source = read_frontend("components/VideoQueue.tsx")
    details_source = read_frontend("components/TechnicalDetails.tsx")
    app_source = read_frontend("App.tsx")
    helper_source = read_frontend("utils/jobIds.ts")

    assert "Result job" in summary_source
    assert "Copy job ID" in summary_source
    assert "formatShortJobId(jobId)" in summary_source
    assert "Copy selected media job ID" in media_source
    assert "Copy queue job ID" in queue_source
    assert "Copy batch job ID" in app_source
    assert "selectedViewerJobId" in app_source
    assert "Job ID" in details_source
    assert "copyJobIdToClipboard" in helper_source
    assert "job_..." in helper_source


def test_frontend_keeps_batch_ids_out_of_job_status_requests():
    app_source = read_frontend("App.tsx")
    service_source = read_frontend("services/analysisService.ts")

    assert "function isBatchId" in app_source
    assert "function isJobId" in app_source
    assert "getBatchStatus(batchId)" in app_source
    assert "getJobStatus(jobId)" in app_source
    assert "jobId: batchId" not in app_source
    assert "apiFetch<JobStatus>(`jobs/${encodeURIComponent(jobId)}`)" in service_source
    assert "apiFetch<BatchStatus>(`batches/${encodeURIComponent(batchId)}`)" in service_source


def test_frontend_main_vlm_dropdown_hides_removed_unavailable_profiles():
    app_source = read_frontend("App.tsx")
    sidebar_source = read_frontend("components/Sidebar.tsx")

    assert "mainVlmSelectorProfiles" in sidebar_source
    assert "profile.id === 'lightweight_512m' && profile.installed && profile.available" in sidebar_source
    assert "selectorProfiles.map((profile)" in sidebar_source
    assert "vlmProfiles.map((profile)" not in sidebar_source
    assert "Lightweight VLM (256M) - unavailable" not in sidebar_source
    assert "Enhanced VLM (2B) - unavailable" not in sidebar_source
    assert "Enhanced VLM (3B candidate) - unavailable" not in sidebar_source
    assert "storedProfile === 'enhanced_3b'" in app_source
    assert "isMainVlmSelectorProfileAvailable(systemStatus, settings.vlmProfile)" in app_source


def test_progress_card_shows_elapsed_and_heartbeat_copy():
    source = read_frontend("components/AnalysisProgress.tsx")

    assert "Running for" in source
    assert "Queued for" in source
    assert "Completed in" in source
    assert "Failed after" in source
    assert "Cancelled after" in source
    assert "elapsedSeconds" in source
    assert "queueWaitSeconds" in source
    assert "analysisRuntimeSeconds" in source
    assert "formatDurationSeconds" in source
    assert "Backend heartbeat" in source
    assert "Still working locally" in source
    assert "createdAt" in source
    assert "startedAt" in source
    assert "heartbeatAt" in source


def test_frontend_source_does_not_display_raw_vlm_artifacts():
    source = read_all_frontend_sources()

    assert "User:" not in source
    assert "Assistant:" not in source
    assert "<global-img>" not in source
    assert "<row_" not in source
    assert "body stream already read" not in source
