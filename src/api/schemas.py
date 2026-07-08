"""Pydantic schemas for the local SafeTrace API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DeviceMode = Literal["auto", "cpu", "cuda"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
BatchStatus = Literal["queued", "running", "completed", "failed", "partial", "cancelled"]
ChatAvailabilityState = Literal["available", "disabled", "missing_model", "missing_runtime", "loading", "unavailable"]
VlmProfileId = Literal["rule_based", "lightweight_256m", "lightweight_512m", "enhanced_2b", "enhanced_3b"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    api: Literal["safetrace-local"]
    version: str
    offline: bool


class ModelStatus(BaseModel):
    status: Literal["ready", "available", "missing", "missing_checkpoint", "missing_runtime", "disabled", "unavailable"]
    path: Optional[str] = None
    message: Optional[str] = None
    actionHint: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class VlmProfileStatus(BaseModel):
    id: VlmProfileId
    label: str
    installed: bool
    available: bool
    requiresActivation: bool
    resourceLevel: str
    path: Optional[str] = None
    message: Optional[str] = None
    statusCopy: Optional[str] = None
    deprecated: bool = False
    notViable: bool = False
    candidate: bool = False
    legacy: bool = False
    fallback: bool = False
    requiresGpu: bool = False
    deviceDecision: Optional[Dict[str, Any]] = None


class VlmSystemStatus(BaseModel):
    selectedProfile: VlmProfileId
    enabled: bool
    active: bool
    runtimeAvailable: bool
    profiles: List[VlmProfileStatus]
    message: str
    requestedVisualExplanationMode: Optional[str] = None
    actualExplanationMode: Optional[str] = None
    vlmAvailability: Optional[str] = None
    vlmSuppressedReason: Optional[str] = None
    fallbackReason: Optional[str] = None
    lightweightModelPathChecked: Optional[str] = None
    ruleBasedFallbackActive: Optional[bool] = None
    ruleBasedFallbackAvailable: Optional[bool] = True
    lightweightVlmWorkerEnabled: Optional[bool] = None
    lightweightVlmWorkerTimeoutSeconds: Optional[float] = None
    lightweightVlmExplanationSource: Optional[str] = None
    lightweightVlmEvidenceBudget: Optional[int] = None
    lightweightVlmFrameLimit: Optional[int] = None
    lightweightVlmJobTimeoutSeconds: Optional[float] = None
    lightweightVlmMaxQualityFailures: Optional[int] = None
    lightweightVlmPrimaryPolicy: Optional[str] = None
    lightweightVlmFallbackPolicy: Optional[str] = None
    lightweightVlmCpuPrefer256m: Optional[bool] = None
    enhancedVlmRequiresGpu: Optional[bool] = None
    deviceGateway: Optional[Dict[str, Any]] = None


class SystemStatusResponse(BaseModel):
    app_version: Optional[str] = None
    backend_version: Optional[str] = None
    build_mode: Optional[str] = None
    runtime_layout: Optional[str] = None
    safeMode: Optional[bool] = None
    device: str
    gpuAvailable: bool
    models: Dict[str, ModelStatus]
    limits: Optional[Dict[str, Any]] = None
    queue: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None
    preflight: Optional[Dict[str, Any]] = None
    vlm: Optional[VlmSystemStatus] = None


class VlmSettingsRequest(BaseModel):
    selectedProfile: VlmProfileId
    enabled: bool = False


class VlmSettingsResponse(VlmSystemStatus):
    pass


class ChatStatusResponse(BaseModel):
    enabled: bool
    available: bool
    state: ChatAvailabilityState
    status: ChatAvailabilityState
    enabled_mode: str
    provider: str
    model: Optional[str] = None
    model_path: Optional[str] = None
    model_exists: Optional[bool] = None
    runtime_available: Optional[bool] = None
    speed_profile: Optional[str] = None
    warmup_on_open: Optional[bool] = None
    fallback_available: Optional[bool] = None
    fallback_label: Optional[str] = None
    runtime_diagnostics: Optional[Dict[str, Any]] = None
    python_executable: Optional[str] = None
    expected_venv_python: Optional[str] = None
    running_in_expected_venv: Optional[bool] = None
    llama_cpp_import_status: Optional[str] = None
    llama_cpp_spec_found: Optional[bool] = None
    llama_cpp_import_error_type: Optional[str] = None
    llama_cpp_import_error_message: Optional[str] = None
    setup_command: Optional[str] = None
    restart_required: Optional[str] = None
    reason: Optional[str] = None
    action_hint: Optional[str] = None
    message: str


class ChatRequest(BaseModel):
    message: str
    job_id: Optional[str] = None
    batch_id: Optional[str] = None
    include_current_result: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    safeTraceOnly: bool
    modelProvider: str


class AnalyzeResponse(BaseModel):
    jobId: str
    status: JobStatus


class BatchAcceptedFile(BaseModel):
    originalFilename: str
    filename: str
    sizeBytes: int
    mediaType: Literal["video"]
    jobId: str
    status: JobStatus
    error: Optional[str] = None


class BatchRejectedFile(BaseModel):
    filename: str
    reason: str


class BatchResponse(BaseModel):
    batchId: str
    status: BatchStatus
    sourceFilename: str
    acceptedFiles: List[BatchAcceptedFile]
    rejectedFiles: List[BatchRejectedFile]
    jobIds: List[str]
    statusCounts: Dict[str, int]
    createdAt: str
    updatedAt: str
    persistenceWarning: Optional[str] = None


class JobStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    progressPercent: int = Field(ge=0, le=100)
    stage: str
    message: str
    currentStep: str
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    componentDiagnostics: Optional[Dict[str, Any]] = None
    updatedAt: Optional[str] = None
    createdAt: Optional[str] = None
    queuedAt: Optional[str] = None
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    completedAt: Optional[str] = None
    failedAt: Optional[str] = None
    cancelledAt: Optional[str] = None
    elapsedSeconds: Optional[float] = None
    queueWaitSeconds: Optional[float] = None
    analysisRuntimeSeconds: Optional[float] = None
    heartbeatAt: Optional[str] = None
    persistenceWarning: Optional[str] = None
    manifestPersistenceWarning: Optional[str] = None


class MediaSummary(BaseModel):
    id: str
    name: str
    type: Literal["video", "image", "unknown"]
    sizeBytes: int
    durationSeconds: Optional[float] = None


class AnalysisSummary(BaseModel):
    framesAnalyzed: int
    framesWithViolations: int
    uniqueViolationTypes: int
    highestSeverity: Optional[str] = None
    summaryText: str
    potentialEventCount: Optional[int] = None
    eventTypes: Optional[List[str]] = None
    overallConfidence: Optional[float] = None
    keyEvents: Optional[List[Dict[str, Any]]] = None


class AffectedFrame(BaseModel):
    frameId: str
    frameNumber: int
    timestamp: str
    confidence: float


class GroupedViolation(BaseModel):
    id: str
    name: str
    severity: str
    description: str
    affectedFrames: List[AffectedFrame]
    confidenceMin: float
    confidenceMax: float


class FrameViolation(BaseModel):
    id: str
    name: str
    severity: str
    confidence: float
    description: str


class EventSupportingFrame(BaseModel):
    frameId: str
    frameNumber: int
    timestamp: str
    confidence: float
    imageUrl: Optional[str] = None


class ViolationEvent(BaseModel):
    id: str
    type: str
    name: str
    severity: str
    description: str
    startTimestamp: str
    endTimestamp: str
    representativeConfidence: float
    confidenceMin: float
    confidenceMax: float
    supportingFrameCount: int
    supportingFrames: List[EventSupportingFrame]


class FrameResult(BaseModel):
    id: str
    frameNumber: int
    timestamp: str
    queryRelevance: float
    status: Literal["violations_detected", "no_violations"]
    imageUrl: Optional[str] = None
    imageMessage: Optional[str] = None
    explanationSource: Optional[Literal["vlm", "vlm_local", "vlm_ollama", "vlm_lightweight", "vlm_enhanced", "rule_template_plus_vlm", "rule_template_plus_lightweight_vlm", "rule_template_plus_lightweight_plus_enhanced", "rule_based"]] = None
    violations: List[FrameViolation]
    technicalEvidence: Dict[str, Any]


class AnalysisResultResponse(BaseModel):
    jobId: str
    status: Literal["completed"]
    createdAt: Optional[str] = None
    queuedAt: Optional[str] = None
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    completedAt: Optional[str] = None
    elapsedSeconds: Optional[float] = None
    queueWaitSeconds: Optional[float] = None
    analysisRuntimeSeconds: Optional[float] = None
    media: MediaSummary
    query: str
    summary: AnalysisSummary
    violations: List[GroupedViolation]
    events: Optional[List[ViolationEvent]] = None
    frames: List[FrameResult]
    engineMetrics: Optional[Dict[str, Any]] = None
    technicalDetails: Optional[Dict[str, Any]] = None
