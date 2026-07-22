import type { AnalysisSetupSummary as AnalysisSetup } from '../types/analysis';

function labelStrategy(value?: string | null) {
  if (!value) return 'Pending';
  return value.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

function frameCount(setup: AnalysisSetup) {
  const sampling = setup.frameSampling ?? {};
  const sampled = sampling.aggregateSampledFrameCount ?? sampling.sampledFrameCount;
  const source = sampling.aggregateSourceVideoFrameCount ?? sampling.sourceVideoFrameCount;
  if (sampled == null) return 'Pending';
  return source == null ? `${sampled} sampled` : `${sampled} sampled from ${source} source frames`;
}

export function AnalysisSetupSummary({ setup }: { setup: AnalysisSetup }) {
  const sampling = setup.frameSampling ?? {};
  const frequency = sampling.samplingFps != null
    ? `${sampling.samplingFps} FPS`
    : sampling.samplingIntervalSeconds != null
      ? `Every ${sampling.samplingIntervalSeconds}s`
      : `${sampling.requestedFps ?? 'Pending'} FPS requested`;
  return (
    <section aria-labelledby="analysis-setup-summary-title" className="border-y border-slate-200 bg-slate-50 px-4 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="analysis-setup-summary-title" className="text-sm font-bold text-slate-950">Analysis setup</h2>
        {setup.readOnly ? <span className="text-xs font-semibold text-slate-500">Completed setup (read-only)</span> : null}
      </div>
      <dl className="mt-2 grid gap-x-5 gap-y-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        <div><dt className="text-xs font-semibold text-slate-500">Review coverage</dt><dd className="font-semibold text-slate-900">{setup.actualCoverage?.label ?? setup.requestedCoverage?.label ?? 'Pending'}</dd></div>
        <div><dt className="text-xs font-semibold text-slate-500">Profile</dt><dd className="font-semibold text-slate-900">{setup.profile?.label ?? 'General Safety'}</dd></div>
        <div className="sm:col-span-2"><dt className="text-xs font-semibold text-slate-500">Query</dt><dd className="break-words font-medium text-slate-800">{setup.query || 'Not provided'}</dd></div>
        <div><dt className="text-xs font-semibold text-slate-500">Frame sampling</dt><dd className="font-medium text-slate-800">{labelStrategy(sampling.strategy)}</dd></div>
        <div><dt className="text-xs font-semibold text-slate-500">Coverage</dt><dd className="font-medium text-slate-800">{frameCount(setup)}</dd></div>
        <div><dt className="text-xs font-semibold text-slate-500">Rate</dt><dd className="font-medium text-slate-800">{frequency}</dd></div>
        <div><dt className="text-xs font-semibold text-slate-500">Windows / cap</dt><dd className="font-medium text-slate-800">{sampling.processingWindowCount ?? 'Pending'} windows, cap {sampling.maximumSampledFrames ?? 'none'}</dd></div>
      </dl>
      {setup.childSettingsDiffer ? <p className="mt-2 text-xs font-semibold text-amber-800">Some batch children used different analysis settings.</p> : null}
    </section>
  );
}
