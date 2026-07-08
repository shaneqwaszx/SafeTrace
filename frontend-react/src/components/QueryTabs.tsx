import { LoaderCircle, SendHorizontal, RotateCcw } from 'lucide-react';
import type { FormEvent } from 'react';
import type { UseCaseProfileSelection } from '../types/analysis';
import { supportLevelLabel } from '../data/useCaseProfiles';

type QueryTabsProps = {
  query: string;
  isLoading: boolean;
  hasResult: boolean;
  canAnalyze: boolean;
  disabledReason?: string;
  useCaseProfile?: UseCaseProfileSelection;
  effectiveQuery?: string;
  queryConflict?: string | null;
  buttonLabel?: string;
  previewMode?: boolean;
  onQueryChange: (query: string) => void;
  onAnalyze: () => void;
  onReset: () => void;
};

const QUERY_EXAMPLES = [
  'driver without seatbelt',
  'driver using phone while driving',
  'worker without helmet',
  'person near machinery',
];

export function QueryTabs({
  query,
  isLoading,
  hasResult,
  canAnalyze,
  disabledReason,
  useCaseProfile,
  effectiveQuery,
  queryConflict,
  buttonLabel = 'Send',
  previewMode = false,
  onQueryChange,
  onAnalyze,
  onReset,
}: QueryTabsProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onAnalyze();
  }

  return (
    <form className="rounded-2xl border border-slate-200 bg-white p-4 shadow-lg" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="relative min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <label className="block text-sm font-semibold text-slate-950" htmlFor="tab-query">
                Profile query refinement
              </label>
              {previewMode ? (
                <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold uppercase text-amber-700">
                  Preview Mode
                </span>
              ) : null}
            </div>
            <input
              id="tab-query"
              className="focus-ring h-12 w-full rounded-lg border border-slate-200 bg-slate-50 px-4 text-base text-slate-950 placeholder:text-slate-400 focus:bg-white"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              placeholder={useCaseProfile?.defaultQuery ?? 'Ask me to analyze the scene...'}
            />
            {useCaseProfile ? (
              <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-600">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-900">{useCaseProfile.label}</span>
                  <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 font-semibold uppercase text-slate-500">
                    {supportLevelLabel(useCaseProfile.backendSupportLevel)}
                  </span>
                </div>
                <p className="mt-1">Default query: <span className="font-semibold text-slate-800">{useCaseProfile.defaultQuery}</span></p>
                {effectiveQuery ? (
                  <p className="mt-1">Effective query sent: <span className="font-semibold text-slate-800">{effectiveQuery}</span></p>
                ) : null}
                {useCaseProfile.limitations ? (
                  <p className="mt-1 text-amber-800">{useCaseProfile.limitations}</p>
                ) : null}
              </div>
            ) : null}
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

        <div className="mt-3 flex flex-wrap gap-2">
          <span className="text-xs font-semibold text-slate-400 mt-1">Suggested:</span>
          {QUERY_EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => onQueryChange(example)}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-safety-blue hover:text-safety-blue"
            >
              {example}
            </button>
          ))}
        </div>
    </form>
  );
}
