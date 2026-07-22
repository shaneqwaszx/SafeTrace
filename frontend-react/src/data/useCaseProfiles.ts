import type { BackendSupportLevel, UseCaseProfileId, UseCaseProfileSelection } from '../types/analysis';

export const DEFAULT_USE_CASE_PROFILE_ID: UseCaseProfileId = 'general_safety';

type ProfileDefinition = Omit<UseCaseProfileSelection, 'customText' | 'requestedQuery' | 'effectiveQuery'> & {
  conflictTerms: string[];
};

const PROFILE_DEFINITIONS: ProfileDefinition[] = [
  {
    profileId: 'seatbelt_compliance',
    label: 'Seatbelt compliance',
    category: 'Driving safety',
    description: 'Focus analysis on visible driver or occupant seatbelt compliance.',
    defaultQuery: 'driver or occupant without seatbelt',
    backendSupportLevel: 'partial',
    supportedChecks: ['Missing Seatbelt', 'driver or occupant visible', 'torso/seatbelt overlap evidence'],
    unsupportedChecks: ['definitive legal determination', 'seatbelt hidden by camera angle or clothing'],
    checks: ['Missing Seatbelt', 'Driver or occupant visible', 'Seatbelt visibility caveat'],
    limitations: 'Seatbelt support depends on visible in-cabin evidence. Absence of a visible belt still needs manual confirmation.',
    notes: 'Best for driver-facing or cabin footage. SafeTrace should not treat poor visibility as proof of non-compliance.',
    conflictTerms: ['helmet', 'hardhat', 'ppe', 'uniform', 'phone', 'mobile', 'fall', 'machinery'],
  },
  {
    profileId: 'phone_use',
    label: 'Phone use / distracted driving',
    category: 'Driving safety',
    description: 'Focus analysis on possible handheld phone use or distraction while driving.',
    defaultQuery: 'driver using phone while driving',
    backendSupportLevel: 'metadata_only',
    supportedChecks: ['driver visible', 'phone-use review context', 'manual confirmation prompt'],
    unsupportedChecks: ['reliable phone-use detection', 'driver gaze or distraction classification'],
    checks: ['Phone-like object near hand', 'Driver attention cue', 'Manual confirmation required'],
    limitations: 'Current packaged rules do not reliably detect phone use. Results should be treated as metadata-guided review unless future detector support is added.',
    notes: 'Use only when camera angle can show hands or phone-like objects. Unsupported findings should not be invented.',
    conflictTerms: ['helmet', 'hardhat', 'ppe', 'uniform', 'seatbelt', 'belt', 'fall', 'machinery'],
  },
  {
    profileId: 'helmet_ppe',
    label: 'Helmet / PPE compliance',
    category: 'Worksite safety',
    description: 'Focus analysis on visible helmet, hardhat, vest, and PPE compliance for people in the scene.',
    defaultQuery: 'worker without helmet or missing PPE',
    backendSupportLevel: 'partial',
    supportedChecks: ['Missing Helmet', 'person/worker visible', 'PPE overlap evidence when available'],
    unsupportedChecks: ['all PPE classes at every site', 'policy-specific PPE not represented by detector labels'],
    checks: ['Missing Helmet', 'Missing PPE', 'Person visible in monitored scene'],
    limitations: 'Helmet/PPE support is partial and depends on detector labels and frame quality.',
    conflictTerms: ['seatbelt', 'belt', 'phone', 'mobile', 'uniform'],
  },
  {
    profileId: 'uniform_compliance',
    label: 'Uniform compliance',
    category: 'Operational policy',
    description: 'Attach uniform-policy context to the analysis for manual review.',
    defaultQuery: 'person not wearing required uniform',
    backendSupportLevel: 'metadata_only',
    supportedChecks: ['uniform policy context', 'person visibility review'],
    unsupportedChecks: ['reliable uniform classification', 'division-specific clothing recognition'],
    checks: ['Uniform visibility', 'High-visibility clothing', 'Policy-specific manual review'],
    limitations: 'Uniform compliance is metadata-only in this release unless custom backend detector/rule support is added.',
    notes: 'First version stores policy context only; it does not add a new uniform detector.',
    conflictTerms: ['seatbelt', 'belt', 'phone', 'mobile', 'helmet', 'hardhat'],
  },
  {
    profileId: 'general_safety',
    label: 'General safety review',
    category: 'General',
    description: 'Combine supported safety checks while retaining each check\'s scene and review safeguards.',
    defaultQuery: 'general safety violations',
    backendSupportLevel: 'partial',
    supportedChecks: ['seatbelt review cues', 'supported helmet/PPE checks', 'conditional phone and steering checks', 'query relevance'],
    unsupportedChecks: ['site-specific policy classes without configured detector/rule support'],
    checks: ['Query relevance', 'Visible safety findings', 'Base fallback evidence'],
    limitations: 'General review composes only registered backend checks. Review cues stay non-authoritative, and metadata-only policies are excluded.',
    conflictTerms: [],
  },
  {
    profileId: 'custom_policy',
    label: 'Custom / division-specific policy',
    category: 'Custom',
    description: 'Attach reviewer-provided policy notes and a custom effective query to the analysis.',
    defaultQuery: 'custom division safety policy review',
    backendSupportLevel: 'metadata_only',
    supportedChecks: ['custom policy context', 'manual confirmation required'],
    unsupportedChecks: ['automatic enforcement of arbitrary custom policy text'],
    checks: ['Custom policy note', 'Manual confirmation required'],
    limitations: 'Custom policies are stored as analysis context only unless matching backend detector/rule support exists.',
    conflictTerms: [],
  },
];

export const USE_CASE_PROFILES: UseCaseProfileSelection[] = PROFILE_DEFINITIONS.map(({ conflictTerms: _conflictTerms, ...profile }) => profile);

function profileDefinition(profileId: string | undefined): ProfileDefinition {
  return PROFILE_DEFINITIONS.find((profile) => profile.profileId === profileId) ?? PROFILE_DEFINITIONS[4];
}

export function resolveUseCaseProfile(
  profileId: string | undefined = DEFAULT_USE_CASE_PROFILE_ID,
  customText = '',
  queryText = '',
): UseCaseProfileSelection {
  const base = profileDefinition(profileId);
  const trimmedCustomText = customText.trim();
  const requestedQuery = queryText.trim();
  const effectiveQuery = buildEffectiveProfileQuery(
    { ...base, customText: trimmedCustomText || undefined },
    requestedQuery,
  );
  return {
    ...base,
    customText: base.profileId === 'custom_policy' ? trimmedCustomText : undefined,
    notes: base.profileId === 'custom_policy'
      ? trimmedCustomText || 'Add division-specific policy notes before analysis when needed.'
      : base.notes,
    requestedQuery: requestedQuery || base.defaultQuery,
    effectiveQuery,
  };
}

export function getUseCaseProfileDefaultQuery(profile: UseCaseProfileSelection): string {
  if (profile.profileId === 'custom_policy' && profile.customText) {
    return profile.defaultQuery;
  }
  return profile.defaultQuery;
}

export function buildEffectiveProfileQuery(profile: UseCaseProfileSelection, queryText: string): string {
  const trimmed = queryText.trim();
  const base = profile.defaultQuery.trim();
  if (!trimmed || trimmed.toLowerCase() === base.toLowerCase()) return base;
  if (profile.profileId === 'general_safety' || profile.profileId === 'custom_policy') return trimmed;
  return `${base}. Reviewer refinement: ${trimmed}`;
}

export function getProfileQueryConflict(
  profile: UseCaseProfileSelection,
  queryText: string,
): string | null {
  const trimmed = queryText.trim().toLowerCase();
  if (!trimmed || profile.profileId === 'general_safety' || profile.profileId === 'custom_policy') return null;
  const definition = profileDefinition(profile.profileId);
  const conflictingTerm = definition.conflictTerms.find((term) => trimmed.includes(term));
  if (!conflictingTerm) return null;
  return `The query mentions "${conflictingTerm}", which conflicts with the selected ${profile.label} profile. Choose a matching profile or use the profile default query.`;
}

export function supportLevelLabel(level: BackendSupportLevel): string {
  switch (level) {
    case 'supported':
      return 'Backend supported';
    case 'partial':
      return 'Partially supported';
    case 'metadata_only':
      return 'Metadata-only / manual review';
    case 'unsupported':
      return 'Unsupported';
    default:
      return 'Unknown support';
  }
}

const PROFILE_VIOLATION_ALLOWLIST: Record<UseCaseProfileId, string[]> = {
  seatbelt_compliance: ['seatbelt'],
  phone_use: ['phone', 'distracted'],
  helmet_ppe: ['helmet', 'hardhat', 'ppe', 'vest'],
  uniform_compliance: ['uniform'],
  general_safety: [],
  custom_policy: [],
};

function normalizedViolationText(value: string): string {
  return value.toLowerCase().replace(/[_-]+/g, ' ');
}

export function isViolationAlignedWithProfile(
  profile: UseCaseProfileSelection | undefined,
  violationName: string,
): boolean {
  if (!profile || profile.profileId === 'general_safety' || profile.profileId === 'custom_policy') return true;
  const allowedTerms = PROFILE_VIOLATION_ALLOWLIST[profile.profileId] ?? [];
  if (!allowedTerms.length) return true;
  const normalized = normalizedViolationText(violationName);
  return allowedTerms.some((term) => normalized.includes(term));
}

export function profileFindingContext(
  profile: UseCaseProfileSelection | undefined,
  violationName: string,
): string {
  if (!profile) return 'General SafeTrace review candidate.';
  if (profile.backendSupportLevel === 'metadata_only' || profile.backendSupportLevel === 'unsupported') {
    return `${profile.label} is ${supportLevelLabel(profile.backendSupportLevel).toLowerCase()}; treat findings as manual-review context.`;
  }
  if (!isViolationAlignedWithProfile(profile, violationName)) {
    return `${violationName} is outside the selected ${profile.label} profile and is de-prioritized in this view.`;
  }
  return `${violationName} matches the selected ${profile.label} review profile.`;
}
