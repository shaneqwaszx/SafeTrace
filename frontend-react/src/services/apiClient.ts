import type { BackendHealth, SystemStatus } from '../types/analysis';

export type SafeTraceApiMode = 'relative_proxy' | 'local_loopback';

type LoopbackRequestInit = RequestInit & {
  targetAddressSpace?: 'loopback';
};

type BridgeBootstrap = {
  api: 'safetrace-local';
  bridgeSupported: boolean;
  connectorProtocolVersion: number;
  runtimeProfile: string;
  sessionId: string;
  sessionToken: string;
  expiresAt: string;
  expiresInSeconds: number;
};

export class SafeTraceApiError extends Error {
  status?: number;
  code?: string;

  constructor(message: string, options: { status?: number; code?: string } = {}) {
    super(message);
    this.name = 'SafeTraceApiError';
    this.status = options.status;
    this.code = options.code;
  }
}

const DEFAULT_RELATIVE_API_BASE = '/api';
const DEFAULT_LOOPBACK_API_BASE = 'http://127.0.0.1:8000/api';
const SESSION_STORAGE_KEY = 'safetrace.loopback.session.v1';
const PROTOCOL_VERSION = 1;
export const SAFETRACE_SESSION_EXPIRED_EVENT = 'safetrace-session-expired';

function queryRequestsLocalBackend(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('backend') === 'local';
}

function configuredMode(): SafeTraceApiMode {
  const raw = String(import.meta.env.VITE_SAFETRACE_API_MODE || '').trim().toLowerCase();
  if (raw === 'local_loopback' || queryRequestsLocalBackend()) return 'local_loopback';
  return 'relative_proxy';
}

function normalizeApiBase(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  if (!trimmed || trimmed === '/') return DEFAULT_RELATIVE_API_BASE;
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
}

function validateLoopbackApiBase(value: string): string {
  const normalized = normalizeApiBase(value);
  let parsed: URL;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new SafeTraceApiError('The Local Backend address is invalid.', { code: 'invalid_api_base' });
  }
  const loopbackHost = parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost';
  if (parsed.protocol !== 'http:' || !loopbackHost || parsed.pathname !== '/api') {
    throw new SafeTraceApiError(
      'Local Backend mode only permits an HTTP loopback API such as http://127.0.0.1:8000/api.',
      { code: 'non_loopback_api_base' },
    );
  }
  return parsed.toString().replace(/\/$/, '');
}

export const SAFETRACE_API_MODE: SafeTraceApiMode = configuredMode();
const configuredLoopbackBase = String(
  import.meta.env.VITE_SAFETRACE_LOCAL_API_BASE || DEFAULT_LOOPBACK_API_BASE,
);
let activeApiBase = SAFETRACE_API_MODE === 'local_loopback'
  ? validateLoopbackApiBase(configuredLoopbackBase)
  : DEFAULT_RELATIVE_API_BASE;

function readSessionToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.sessionStorage.getItem(SESSION_STORAGE_KEY);
}

function storeSessionToken(token: string): void {
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, token);
}

export function clearBridgeSession(): void {
  if (typeof window !== 'undefined') window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

export function supportsLoopbackTargetAddressSpace(): boolean {
  return typeof Request !== 'undefined' && 'targetAddressSpace' in Request.prototype;
}

export async function localNetworkPermissionState(): Promise<PermissionState | 'unsupported' | 'unknown'> {
  if (typeof navigator === 'undefined' || !navigator.permissions?.query) return 'unsupported';
  try {
    const result = await navigator.permissions.query(
      { name: 'local-network-access' } as unknown as PermissionDescriptor,
    );
    return result.state;
  } catch {
    return 'unknown';
  }
}

export function getActiveApiBase(): string {
  return activeApiBase;
}

export function setActiveApiBase(apiBase: string): void {
  activeApiBase = SAFETRACE_API_MODE === 'local_loopback'
    ? validateLoopbackApiBase(apiBase)
    : normalizeApiBase(apiBase);
}

export function buildApiUrl(path: string, apiBase = activeApiBase): string {
  return `${apiBase.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}

async function responseError(response: Response): Promise<SafeTraceApiError> {
  const raw = await response.text();
  let message = `SafeTrace API returned ${response.status}`;
  let code: string | undefined;
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      message = parsed?.detail?.message || parsed?.detail || parsed?.message || message;
      code = parsed?.detail?.code || parsed?.code;
    } catch {
      message = raw;
    }
  }
  return new SafeTraceApiError(String(message), { status: response.status, code });
}

export async function safeTraceFetch(
  path: string,
  options: RequestInit = {},
  apiBase = activeApiBase,
  includeSession = true,
): Promise<Response> {
  const headers = new Headers(options.headers);
  const sessionToken = includeSession ? readSessionToken() : null;
  if (sessionToken) headers.set('X-SafeTrace-Session', sessionToken);
  const request: LoopbackRequestInit = {
    ...options,
    headers,
    cache: options.cache ?? 'no-store',
    credentials: 'omit',
  };
  if (SAFETRACE_API_MODE === 'local_loopback') {
    request.mode = 'cors';
    if (supportsLoopbackTargetAddressSpace()) {
      request.targetAddressSpace = 'loopback';
    }
  }
  const response = await fetch(buildApiUrl(path, apiBase), request);
  if (response.status === 401) {
    clearBridgeSession();
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent(SAFETRACE_SESSION_EXPIRED_EVENT));
    }
  }
  return response;
}

export async function safeTraceApiFetch<T>(
  path: string,
  options: RequestInit = {},
  apiBase = activeApiBase,
): Promise<T> {
  const response = await safeTraceFetch(path, options, apiBase);
  if (!response.ok) throw await responseError(response);
  return response.json() as Promise<T>;
}

export async function connectSafeTraceBackend(): Promise<{
  apiBase: string;
  health: BackendHealth;
  systemStatus: SystemStatus;
}> {
  clearBridgeSession();
  const healthResponse = await safeTraceFetch(
    'health',
    { signal: AbortSignal.timeout(5000) },
    activeApiBase,
    false,
  );
  if (!healthResponse.ok) throw await responseError(healthResponse);
  const health = await healthResponse.json() as BackendHealth;
  if (
    health.status !== 'ok'
    || health.api !== 'safetrace-local'
    || health.bridgeSupported !== true
    || health.connectorProtocolVersion !== PROTOCOL_VERSION
  ) {
    throw new SafeTraceApiError(
      'A service answered on the loopback port, but it is not a compatible SafeTrace Local Backend.',
      { code: 'incompatible_backend' },
    );
  }

  const bootstrapResponse = await safeTraceFetch(
    'bridge/bootstrap',
    { method: 'POST', signal: AbortSignal.timeout(5000) },
    activeApiBase,
    false,
  );
  if (!bootstrapResponse.ok) throw await responseError(bootstrapResponse);
  const bootstrap = await bootstrapResponse.json() as BridgeBootstrap;
  if (
    bootstrap.api !== 'safetrace-local'
    || bootstrap.bridgeSupported !== true
    || bootstrap.connectorProtocolVersion !== PROTOCOL_VERSION
  ) {
    throw new SafeTraceApiError('SafeTrace connector handshake is incompatible.', {
      code: 'protocol_mismatch',
    });
  }
  storeSessionToken(bootstrap.sessionToken);

  const systemStatus = await safeTraceApiFetch<SystemStatus>('system/status', {
    signal: AbortSignal.timeout(5000),
  });
  return { apiBase: activeApiBase, health, systemStatus };
}
