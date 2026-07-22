import { LoaderCircle, SendHorizontal, RotateCcw } from 'lucide-react';
import type { FormEvent } from 'react';

type QueryTabsProps = {
  isLoading: boolean;
  hasResult: boolean;
  canAnalyze: boolean;
  disabledReason?: string;
  queryConflict?: string | null;
  buttonLabel?: string;
  previewMode?: boolean;
  onAnalyze: () => void;
  onReset: () => void;
};

export function QueryTabs({
  isLoading,
  hasResult,
  canAnalyze,
  disabledReason,
  queryConflict,
  buttonLabel = 'Send',
  previewMode = false,
  onAnalyze,
  onReset,
}: QueryTabsProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onAnalyze();
  }

  return (
    <form className="rounded-2xl border border-slate-200 bg-white p-4 shadow-lg" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="relative min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <label className="block text-sm font-semibold text-slate-950" htmlFor="tab-query">
                Ready to analyze
              </label>
              {previewMode ? (
                <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold uppercase text-amber-700">
                  Preview Mode
                </span>
              ) : null}
            </div>
            <p className="text-sm text-slate-600">The selected review coverage, profile, and query will be recorded with this job.</p>
          </div>

          <div className="flex gap-2">
            <button
              className="focus-ring inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-safety-blue px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-50"
              type="submit"
              disabled={!canAnalyze}
              title={!canAnalyze ? disabledReason : undefined}
            >
              {isLoading ? (
                <LoaderCircle className="h-4 w-4 animate-spin" />
              ) : (
                <SendHorizontal className="h-4 w-4" />
              )}
              {isLoading ? 'Running' : buttonLabel}
            </button>
            
            <button
              className="focus-ring inline-flex h-12 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
              type="button"
              disabled={isLoading || !hasResult}
              onClick={onReset}
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>

        {!canAnalyze && disabledReason ? (
          <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-800">
            {disabledReason}
          </p>
        ) : null}
        {queryConflict ? (
          <p className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-semibold text-red-800">
            Resolve the profile/query conflict before analysis starts.
          </p>
        ) : null}

    </form>
  );
}
