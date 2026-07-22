import { RotateCcw } from 'lucide-react';
import type { AnalysisSettings } from '../types/analysis';
import { USE_CASE_PROFILES, resolveUseCaseProfile, supportLevelLabel } from '../data/useCaseProfiles';

export function AnalysisSetupPanel({ settings, query, onSettingsChange, onQueryChange }: {
  settings: AnalysisSettings;
  query: string;
  onSettingsChange: (settings: AnalysisSettings) => void;
  onQueryChange: (query: string) => void;
}) {
  const profile = settings.useCaseProfile;
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft" aria-labelledby="analysis-profile-title">
      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div>
          <h2 id="analysis-profile-title" className="text-sm font-bold text-slate-950">Review coverage</h2>
          <div className="mt-3 inline-flex rounded-lg border border-slate-300 bg-slate-50 p-1" role="group" aria-label="Review coverage">
            {([
              ['fast_local', 'Fast Local'],
              ['comprehensive', 'Comprehensive Review'],
            ] as const).map(([value, label]) => (
              <button
                key={value}
                type="button"
                aria-pressed={settings.reviewMode === value}
                onClick={() => onSettingsChange({ ...settings, reviewMode: value })}
                className={`focus-ring rounded-md px-3 py-2 text-sm font-semibold ${settings.reviewMode === value ? 'bg-safety-blue text-white shadow-sm' : 'text-slate-600 hover:text-slate-950'}`}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            {settings.reviewMode === 'comprehensive'
              ? 'Screens more of the video, then applies heavier refinement only to selected candidates.'
              : 'Reviews the most relevant sampled frames with the stable local pipeline.'}
          </p>
        </div>

        <div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-bold text-slate-950">Analysis profile</h2>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-[10px] font-bold uppercase text-slate-600">
              {supportLevelLabel(profile.backendSupportLevel)}
            </span>
          </div>
          <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(12rem,0.7fr)_minmax(0,1.3fr)_auto] lg:items-end">
            <label className="text-xs font-bold uppercase text-slate-600">
              Use-case profile
              <select
                value={profile.profileId}
                onChange={(event) => onSettingsChange({
                  ...settings,
                  useCaseProfile: resolveUseCaseProfile(
                    event.target.value,
                    event.target.value === 'custom_policy' ? profile.customText ?? '' : '',
                  ),
                })}
                className="focus-ring mt-1 block h-11 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm font-semibold normal-case text-slate-900"
              >
                {USE_CASE_PROFILES.map((item) => <option key={item.profileId} value={item.profileId}>{item.label}</option>)}
              </select>
            </label>
            <label className="text-xs font-bold uppercase text-slate-600">
              Profile-specific query
              <input
                value={query}
                onChange={(event) => onQueryChange(event.target.value)}
                placeholder={profile.defaultQuery}
                className="focus-ring mt-1 block h-11 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm font-medium normal-case text-slate-900"
              />
            </label>
            <button
              type="button"
              title="Reset query to profile default"
              onClick={() => onQueryChange(profile.defaultQuery)}
              className="focus-ring inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-700"
            >
              <RotateCcw className="h-4 w-4" aria-hidden="true" /> Reset
            </button>
          </div>
          {profile.profileId === 'custom_policy' ? (
            <textarea
              value={profile.customText ?? ''}
              onChange={(event) => onSettingsChange({ ...settings, useCaseProfile: resolveUseCaseProfile('custom_policy', event.target.value) })}
              className="focus-ring mt-3 min-h-20 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              placeholder="Add the policy context SafeTrace should carry with this review"
            />
          ) : null}
          <p className="mt-3 text-xs leading-5 text-slate-600">
            <span className="font-semibold text-slate-900">Applicability:</span> {profile.description}
            {profile.limitations ? <span className="ml-1 text-amber-800">{profile.limitations}</span> : null}
          </p>
        </div>
      </div>
    </section>
  );
}
