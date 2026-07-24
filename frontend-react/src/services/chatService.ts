import { apiFetch } from './analysisService';
import type { ChatRequest, ChatResponse, ChatStatus } from '../types/chat';

async function chatFetch<T>(path: string, options?: RequestInit): Promise<T> {
  return apiFetch<T>(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
  });
}

export async function getChatStatus(): Promise<ChatStatus> {
  return chatFetch<ChatStatus>('chat/status');
}

export async function warmupSafeTraceAssistant(): Promise<ChatStatus> {
  return chatFetch<ChatStatus>('chat/warmup', {
    method: 'POST',
  });
}

export async function askSafeTraceAssistant(request: ChatRequest): Promise<ChatResponse> {
  return chatFetch<ChatResponse>('chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
