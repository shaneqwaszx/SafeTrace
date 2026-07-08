# SafeTrace VLM Profile Strategy

SafeTrace should not force every VLM tier into one tester package. Rule-based
explanations remain the stable default until a local VLM profile produces
measured, safety-relevant output on a labelled validation set.

## Profile Tiers

| Profile | Package target | Status |
| --- | --- | --- |
| `rule_based` | All builds | Stable default. No VLM assets required. |
| `lightweight_256m` | Historical/debug only | Deprecated after Phase B evaluation accepted `0/4` safety outputs and repeatedly returned object-label text such as `Suitcase 0.25 suitcase 0.54`. |
| `lightweight_512m` | Selected tester candidate | Disabled/unavailable unless `models/vlm/lightweight-512m` exists. Requires evaluation before release. |
| `enhanced_3b` | Internal candidate | Disabled/unavailable unless `models/vlm/enhanced-3b` exists. Intended as the enhanced replacement candidate. |

The existing `enhanced_2b` folder remains supported for compatibility but is not
the replacement target. Local inventory showed it is package-heavy as a full
folder, so selected/internal packaging should be deliberate.

## Package Modes

Stable package:
- Rule-based only.
- No VLM model tier by default.
- MobileSAM/chatbot remain controlled by their package profile.

Lightweight experimental:
- Rule-based fallback plus `lightweight_512m`.
- No enhanced 3B.
- Chatbot optional based on size target.
- Selected testers only.

Enhanced experimental:
- Rule-based fallback plus `enhanced_3b`.
- No lightweight 512M unless explicitly needed.
- Chatbot optional or separate.
- Internal testing only.

Full internal lab:
- Rule-based, 512M, 3B, chatbot, and MobileSAM.
- Not for general testers.

## Acceptance Thresholds

For `lightweight_512m`:
- Package size remains meaningfully smaller than enhanced.
- At least 50-70% accepted safety-relevant outputs on the labelled validation set.
- Average explanation time is acceptable for selected/internal use.
- No repeated object-label-only output.
- Can say `cannot determine` with a useful visible-evidence reason.

For `enhanced_3b`:
- Higher quality than 512M.
- At least 70% accepted safety-relevant outputs on the labelled validation set.
- Handles seatbelt, helmet/PPE, phone, and uncertainty better.
- Runtime acceptable in an experimental/internal package, ideally GPU accelerated.
- No high-confidence hallucinations.

If a candidate fails these thresholds, keep it disabled and ship rule-based
mode as the stable path.

## Evaluation Command

Use the profile-aware harness without committing generated reports:

```powershell
python scripts\evaluate_vlm_profile.py --model-profile lightweight_512m --profile seatbelt --input-dir data\eval\vlm_validation\seatbelt --timeout-seconds 120 --generation-timeout-seconds 90
```

Missing model paths should exit clearly with `model path missing`. Dry-run mode
can validate output schema without downloading models:

```powershell
python scripts\evaluate_vlm_profile.py --model-profile lightweight_512m --dry-run --synthetic-if-empty
```
