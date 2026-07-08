import {
  Bot,
  Copy,
  Loader2,
  MessageCircle,
  Minimize2,
  RefreshCcw,
  Send,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { askSafeTraceAssistant, getChatStatus, warmupSafeTraceAssistant } from '../services/chatService';
import type { AnalysisResult, BatchStatus } from '../types/analysis';
import type { ChatMessage, ChatStatus } from '../types/chat';
import { copyJobIdToClipboard, formatShortJobId } from '../utils/jobIds';

type SafeTraceAssistantProps = {
  backendConnected: boolean;
  result: AnalysisResult | null;
  batch: BatchStatus | null;
  selectedJobId: string | null;
};

const INTRO_MESSAGE: ChatMessage = {
  id: 'assistant-intro',
  role: 'assistant',
  content:
    'Ask about SafeTrace usage, results, evidence frames, batch uploads, exports, troubleshooting, APIs, or limitations.',
};

const QUICK_PROMPTS = [
  'What does overall confidence mean?',
  'Explain this result.',
  'Which frames support the top finding?',
  'Was the driver wearing a seatbelt?',
  'Is the driver using a phone while driving?',
  'How do I upload a ZIP batch?',
  'How do I download the technical JSON?',
  'Why is the assistant unavailable?',
];

function newMessageId() {
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function statusCopy(state: ChatStatus['state']) {
  const labels: Record<ChatStatus['state'], string> = {
    available: 'Available',
    disabled: 'Disabled',
    missing_model: 'Missing model',
    missing_runtime: 'Missing runtime',
    loading: 'Loading',
    unavailable: 'Unavailable',
  };
  const messages: Record<ChatStatus['state'], string> = {
    available: 'SafeTrace Assistant is ready.',
    disabled: 'SafeTrace Assistant is disabled for this backend session.',
    missing_model: 'SafeTrace Assistant model is not installed in this build.',
    missing_runtime: 'SafeTrace Assistant runtime is not installed.',
    loading: 'SafeTrace Assistant is loading the local model. This may take a moment.',
    unavailable: 'SafeTrace Assistant is unavailable, but SafeTrace analysis still works.',
  };
  return { label: labels[state], message: messages[state] };
}

export function SafeTraceAssistant({
  backendConnected,
  result,
  batch,
  selectedJobId,
}: SafeTraceAssistantProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([INTRO_MESSAGE]);
  const [draft, setDraft] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshingStatus, setIsRefreshingStatus] = useState(false);
  const [warmupAttempted, setWarmupAttempted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const jobId = selectedJobId || result?.jobId;
  const batchId = batch?.batchId;
  const shortJobId = formatShortJobId(jobId);
  const contextLabel = useMemo(() => {
    if (batchId && jobId) return `Using selected batch ${batchId} and job ${formatShortJobId(jobId)}`;
    if (jobId) return `Using selected job ${formatShortJobId(jobId)}`;
    if (batchId) return `Using selected batch ${batchId}`;
    return 'General SafeTrace guidance';
  }, [batchId, jobId]);

  const assistantState: ChatStatus['state'] = !backendConnected ? 'unavailable' : status?.state ?? 'loading';
  const isAvailable = backendConnected && status?.state === 'available' && status.available;
  const isLimitedFallbackAvailable = backendConnected
    && status?.state === 'missing_runtime'
    && Boolean(status.fallback_available);
  const canSubmit = isAvailable || isLimitedFallbackAvailable;
  const copy = statusCopy(assistantState);
  const statusDetail = backendConnected ? status?.message : 'Connect the SafeTrace Local Runtime to check assistant status.';
  const badgeClass = isAvailable
    ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
    : isLimitedFallbackAvailable
      ? 'border-blue-200 bg-blue-50 text-blue-700'
    : assistantState === 'loading'
      ? 'border-blue-200 bg-blue-50 text-blue-700'
      : 'border-amber-200 bg-amber-50 text-amber-700';
  const dotClass = isAvailable
    ? 'bg-emerald-400'
    : isLimitedFallbackAvailable
      ? 'bg-blue-400'
    : assistantState === 'loading'
      ? 'bg-blue-400'
      : 'bg-amber-400';
  const actionHint = status?.action_hint;
  const reason = status?.reason || statusDetail;
  const showRestartCommands = assistantState === 'disabled';
  const statusLabel = isLimitedFallbackAvailable ? status?.fallback_label || 'Limited help' : copy.label;
  const statusMessage = isLimitedFallbackAvailable
    ? `SafeTrace Assistant is answering from deterministic local help because ${reason || 'the packaged llama.cpp runtime is unavailable'}.`
    : copy.message;
  const contextStatusMessage = canSubmit && !jobId
    ? 'No selected result is attached. Ask general SafeTrace usage questions, or open a completed job for result-aware answers.'
    : null;
  const runtimeDiagnostics = ([
    ['Backend Python', status?.python_executable],
    ['Expected .venv Python', status?.expected_venv_python],
    ['llama_cpp import', status?.llama_cpp_import_status],
    ['Import error', [status?.llama_cpp_import_error_type, status?.llama_cpp_import_error_message].filter(Boolean).join(': ')],
    ['Setup', status?.setup_command],
    ['Restart', status?.restart_required],
  ] as Array<[string, string | null | undefined]>).filter((row): row is [string, string] => Boolean(row[1]));
  const showRuntimeDiagnostics = assistantState === 'missing_runtime' && runtimeDiagnostics.length > 0;

  async function refreshStatus() {
    if (!backendConnected) {
      setStatus(null);
      return;
    }
    setIsRefreshingStatus(true);
    try {
      const nextStatus = await getChatStatus();
      setStatus(nextStatus);
    } catch {
      setStatus({
        enabled: false,
        available: false,
        state: 'unavailable',
        status: 'unavailable',
        enabled_mode: 'unknown',
        provider: 'unknown',
        model: null,
        model_path: null,
        model_exists: null,
        runtime_available: null,
        speed_profile: null,
        reason: 'SafeTrace Assistant status could not be checked.',
        action_hint: 'Confirm the backend chat API is available, then retry.',
        message: 'SafeTrace Assistant status could not be checked. Confirm the backend chat API is available.',
      });
    } finally {
      setIsRefreshingStatus(false);
    }
  }

  useEffect(() => {
    void refreshStatus();
  }, [backendConnected]);

  useEffect(() => {
    if (!backendConnected) {
      setWarmupAttempted(false);
    }
  }, [backendConnected]);

  useEffect(() => {
    if (!isOpen || !backendConnected || !status?.warmup_on_open || warmupAttempted) return undefined;
    let isMounted = true;
    setWarmupAttempted(true);
    warmupSafeTraceAssistant()
      .then((nextStatus) => {
        if (isMounted) setStatus(nextStatus);
      })
      .catch(() => {
        if (isMounted) void refreshStatus();
      });
    return () => {
      isMounted = false;
    };
  }, [backendConnected, isOpen, status?.warmup_on_open, warmupAttempted]);

  useEffect(() => {
    if (!isOpen) return;
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isOpen, isLoading]);

  function togglePanel() {
    const nextOpen = !isOpen;
    setIsOpen(nextOpen);
    if (nextOpen) {
      void refreshStatus();
    }
  }

  async function submitMessage(nextMessage?: string) {
    const message = (nextMessage ?? draft).trim();
    if (!message || isLoading || !canSubmit) return;

    const userMessage: ChatMessage = {
      id: newMessageId(),
      role: 'user',
      content: message,
    };
    setMessages((current) => [...current, userMessage]);
    setDraft('');
    setError(null);
    setIsLoading(true);
    try {
      const response = await askSafeTraceAssistant({
        message,
        job_id: jobId,
        batch_id: batchId,
        include_current_result: true,
      });
      setMessages((current) => [
        ...current,
        {
          id: newMessageId(),
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch {
      setError('SafeTrace Assistant could not answer right now. Analysis features remain available.');
      void refreshStatus();
    } finally {
      setIsLoading(false);
    }
  }

  function clearMessages() {
    setMessages([INTRO_MESSAGE]);
    setError(null);
  }

  return (
    <div className="fixed bottom-4 right-4 z-40 flex flex-col items-end gap-3 sm:bottom-6 sm:right-6">
      {isOpen ? (
        <section
          className="flex h-[min(840px,calc(100vh-48px))] w-[min(720px,calc(100vw-24px))] flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-2xl"
          aria-label="SafeTrace Assistant chat panel"
        >
          <header className="shrink-0 border-b border-slate-200 bg-slate-950 p-4 text-white">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-safety-teal text-white">
                  <Bot className="h-5 w-5" aria-hidden="true" />
                </div>
                <div className="min-w-0">
                  <h2 className="text-base font-bold">SafeTrace Assistant</h2>
                  <p className="mt-1 truncate text-xs font-medium text-slate-300">{contextLabel}</p>
                  {jobId ? (
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-300">
                      <span className="rounded bg-white/10 px-2 py-1 font-mono" title={jobId}>
                        Selected job {shortJobId}
                      </span>
                      <button
                        type="button"
                        onClick={() => void copyJobIdToClipboard(jobId)}
                        className="focus-ring inline-flex items-center gap-1 rounded border border-white/20 px-2 py-1 font-semibold transition hover:bg-white/10 hover:text-white"
                        aria-label="Copy selected job ID"
                        title={`Copy full job ID ${jobId}`}
                      >
                        <Copy className="h-3 w-3" aria-hidden="true" />
                        Copy job ID
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <button
                  type="button"
                  onClick={() => void refreshStatus()}
                  className="focus-ring inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-300 transition hover:bg-white/10 hover:text-white"
                  aria-label="Refresh assistant status"
                  disabled={isRefreshingStatus}
                >
                  <RefreshCcw className={`h-4 w-4 ${isRefreshingStatus ? 'animate-spin' : ''}`} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  className="focus-ring inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-300 transition hover:bg-white/10 hover:text-white"
                  aria-label="Minimize SafeTrace Assistant"
                >
                  <Minimize2 className="h-4 w-4" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  className="focus-ring inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-300 transition hover:bg-white/10 hover:text-white"
                  aria-label="Close SafeTrace Assistant"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
            </div>
          </header>

          <div className="shrink-0 border-b border-slate-200 bg-white p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-bold uppercase ${badgeClass}`}>
                <span className={`h-2 w-2 rounded-full ${dotClass}`} aria-hidden="true" />
                {statusLabel}
              </span>
              <button
                type="button"
                onClick={clearMessages}
                className="focus-ring inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
              >
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                Clear
              </button>
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-600">{statusMessage}</p>
            {reason && reason !== statusMessage ? (
              <p className="mt-1 text-xs leading-5 text-slate-500">{reason}</p>
            ) : null}
            {contextStatusMessage ? (
              <p className="mt-1 text-xs leading-5 text-slate-500">{contextStatusMessage}</p>
            ) : null}
            {actionHint ? (
              <p className="mt-1 text-xs font-medium leading-5 text-slate-700">{actionHint}</p>
            ) : null}
            {isLimitedFallbackAvailable ? (
              <div className="mt-2 rounded-lg border border-blue-100 bg-blue-50 p-2.5 text-xs leading-5 text-blue-900">
                Limited SafeTrace help is available without llama.cpp. Full model-backed chat starts after installing
                llama-cpp-python and restarting the backend.
              </div>
            ) : null}
            {showRuntimeDiagnostics ? (
              <details className="mt-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs leading-5 text-slate-700">
                <summary className="focus-ring cursor-pointer rounded-md font-semibold text-slate-900">
                  Runtime diagnostics
                </summary>
                <dl className="mt-2 space-y-1">
                  {runtimeDiagnostics.map(([label, value]) => (
                    <div key={label} className="grid gap-1 sm:grid-cols-[9rem_1fr]">
                      <dt className="font-semibold text-slate-500">{label}</dt>
                      <dd className="min-w-0 break-words font-mono text-[11px] text-slate-800">{value}</dd>
                    </div>
                  ))}
                </dl>
              </details>
            ) : null}
            <div className="mt-3 rounded-lg border border-blue-100 bg-blue-50 p-2.5 text-xs leading-5 text-blue-900">
              The VLM explanation describes detected evidence. SafeTrace Assistant is an interactive helper for using
              SafeTrace, interpreting summaries, troubleshooting, and asking questions about the selected result.
            </div>
          </div>

          {canSubmit ? (
            <>
              <div className="min-h-0 flex-[1_1_65%] space-y-3 overflow-y-auto overflow-x-hidden bg-slate-50 p-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`break-words rounded-lg px-3 py-2 text-sm leading-6 [overflow-wrap:anywhere] ${
                      message.role === 'user'
                        ? 'ml-auto max-w-[86%] bg-safety-blue text-white'
                        : 'mr-auto max-w-[95%] border border-slate-200 bg-white text-slate-700'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    {message.sources?.length ? (
                      <p className="mt-2 text-xs font-semibold uppercase text-slate-500">
                        Sources: {message.sources.join(', ')}
                      </p>
                    ) : null}
                  </div>
                ))}
                {isLoading ? (
                  <div className="mr-auto inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Thinking
                  </div>
                ) : null}
                <div ref={messagesEndRef} />
              </div>

              <div className="shrink-0 border-t border-slate-200 bg-white p-3">
                <div className="mb-2 flex max-h-20 flex-wrap gap-2 overflow-y-auto pr-1">
                  {QUICK_PROMPTS.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      onClick={() => void submitMessage(prompt)}
                      disabled={isLoading}
                      className="focus-ring rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold leading-4 text-slate-700 transition hover:border-safety-blue hover:text-safety-blue disabled:opacity-60"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
                {error ? (
                  <p className="mb-3 rounded-lg border border-red-200 bg-red-50 p-2 text-sm text-red-800">{error}</p>
                ) : null}
                <div className="flex gap-2">
                  <textarea
                    className="focus-ring min-h-[92px] min-w-0 flex-1 resize-y rounded-lg border border-slate-300 px-3 py-2 text-sm leading-6"
                    rows={3}
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        void submitMessage();
                      }
                    }}
                    placeholder="Ask about findings, uploads, exports, APIs, or troubleshooting"
                  />
                  <button
                    type="button"
                    onClick={() => void submitMessage()}
                    disabled={isLoading || !draft.trim()}
                    className="focus-ring inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-safety-blue text-white transition hover:bg-blue-700 disabled:opacity-60"
                    aria-label="Send message to SafeTrace Assistant"
                  >
                    <Send className="h-4 w-4" aria-hidden="true" />
                  </button>
                </div>
                <p className="mt-1 text-[11px] text-slate-500">Press Enter to send. Use Shift+Enter for a new line.</p>
              </div>
            </>
          ) : (
            <div className="min-h-0 flex-1 bg-white p-4">
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-950">
                <div className="flex items-start gap-2">
                  <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  <div>
                    <p>{copy.message}</p>
                    {reason && reason !== copy.message ? (
                      <p className="mt-1 text-xs leading-5">{reason}</p>
                    ) : null}
                    {actionHint ? (
                      <p className="mt-1 text-xs font-semibold leading-5">{actionHint}</p>
                    ) : null}
                  </div>
                </div>
              </div>
              {showRestartCommands ? (
                <pre className="mt-3 overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-5 text-slate-100">
{`set SAFETRACE_CHAT_ENABLED=auto
set SAFETRACE_CHAT_PROVIDER=packaged_llamacpp
set SAFETRACE_CHAT_MODEL_PATH=models/chat/safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf`}
                </pre>
              ) : null}
              <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
                {QUICK_PROMPTS.map((prompt) => (
                  <span
                    key={prompt}
                    className="shrink-0 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold text-slate-500"
                  >
                    {prompt}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>
      ) : null}

      <button
        type="button"
        onClick={togglePanel}
        className="focus-ring group relative inline-flex h-14 w-14 items-center justify-center rounded-full bg-safety-blue text-white shadow-2xl transition hover:bg-blue-700"
        aria-label={isOpen ? 'Minimize SafeTrace Assistant' : 'Open SafeTrace Assistant'}
      >
        <span className={`absolute right-1 top-1 h-3.5 w-3.5 rounded-full border-2 border-white ${dotClass}`} />
        {isOpen ? <X className="h-6 w-6" aria-hidden="true" /> : <MessageCircle className="h-6 w-6" aria-hidden="true" />}
      </button>
    </div>
  );
}
