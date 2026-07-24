# SafeTrace Vercel Frontend With Local Backend

The production frontend at `https://safetrace-iota.vercel.app/` is a static
browser application. Analysis requests go directly from that browser tab to
the same computer's SafeTrace backend at `http://127.0.0.1:8000/api`.
Vercel does not proxy, receive, or store analysis uploads.

## Required Vercel variables

Set these on the existing `safetrace` project for Production (and Preview only
when preview deployments should use the same local connector):

```text
VITE_SAFETRACE_API_MODE=local_loopback
VITE_SAFETRACE_LOCAL_API_BASE=http://127.0.0.1:8000/api
VITE_SAFETRACE_REQUIRE_BACKEND=true
VITE_SAFETRACE_ENABLE_PREVIEW_MODE=false
```

Never put a connector token in Vercel or a Vite environment variable. The
backend creates a new signing secret when it starts. The allowed browser
origin obtains a short-lived, origin-bound token during the explicit Connect
Local Backend handshake and stores it in browser `sessionStorage`.

For local Vite development, use `VITE_SAFETRACE_API_MODE=relative_proxy`.
Vite's `/api` proxy remains the development-only path.

## Starting SafeTrace

Run `START_SAFETRACE_VERCEL.bat` from the repository root. It:

1. runs the strict `local_full` preflight;
2. reuses a compatible healthy backend when one is already listening;
3. otherwise starts Uvicorn on `127.0.0.1:8000` only;
4. polls `/api/health`; and
5. opens `https://safetrace-iota.vercel.app/?backend=local`.

In the browser, choose **Connect Local Backend** and grant Local Network Access
to the SafeTrace site when prompted. Denying the browser permission leaves
analysis controls locked and does not switch to a remote service.

## Backend security boundary

The launcher sets this exact allowlist:

```text
SAFETRACE_ALLOWED_FRONTEND_ORIGINS=https://safetrace-iota.vercel.app,http://127.0.0.1:5173,http://localhost:5173
```

The backend:

- binds only to `127.0.0.1`;
- emits `Access-Control-Allow-Private-Network: true` only for an allowlisted
  origin's valid preflight;
- does not use wildcard origins or wildcard request headers;
- requires `X-SafeTrace-Session` for browser API access beyond health,
  bootstrap, and job-owned media rendering;
- rejects tokens from another origin, another backend start, or an expired
  session; and
- preserves no-`Origin` CLI and TestClient operation for local automation.

## Browser compatibility

Chromium's Local Network Access rollout varies by browser version and managed
policy. SafeTrace sets `targetAddressSpace: "loopback"` on connector fetches
where the browser supports it. If no permission prompt appears, check the
site's Local Network Access permission, browser version, enterprise policy,
and whether endpoint security software blocks loopback requests.

Safari and Firefox support may differ. A browser that cannot authorize HTTPS
page access to an HTTP loopback service is not silently redirected elsewhere.

## Troubleshooting

- **Backend not running:** keep the launcher/backend terminal open and retry.
- **Permission denied:** allow Local Network Access for
  `safetrace-iota.vercel.app`, then reconnect.
- **Wrong service:** stop the process occupying port 8000 and run the launcher.
- **Session ended:** reconnect; backend restarts intentionally invalidate old
  browser sessions.
- **CORS error:** verify the page origin exactly matches the backend allowlist.
- **Hosted page calls `/api`:** verify the production Vercel variables above
  and redeploy the frontend. Do not add a Vercel API rewrite.
