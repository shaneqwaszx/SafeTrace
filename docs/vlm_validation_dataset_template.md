# SafeTrace VLM Validation Dataset Template

Local VLM candidates must be evaluated on labelled evidence images before they
are enabled for tester packages. Do not commit validation images by default.

Recommended local structure:

```text
data/eval/vlm_validation/
  seatbelt/
    missing/
    worn/
    unclear/
  helmet_ppe/
    missing/
    present/
    unclear/
  phone/
    visible/
    not_visible/
    unclear/
  uniform/
    compliant/
    non_compliant/
    unclear/
  negative_controls/
    irrelevant_objects/
    outdoor_no_subject/
```

Label file example:

```json
{
  "image": "seatbelt/missing/frame_000123.jpg",
  "profile": "seatbelt",
  "expected_status": "not_visible",
  "accepted_explanation_must_mention": ["seatbelt", "torso"],
  "notes": "Driver torso visible; no clear belt path."
}
```

Evaluation report schema:
- `modelProfile`
- `modelPath`
- `imagePath`
- `reviewProfile`
- `prompt`
- `imageRegion`
- `rawOutput`
- `cleanedOutput`
- `accepted`
- `fallbackReason`
- `qualityReason`
- `modelLoadSeconds`
- `generationSeconds`
- `totalSeconds`

Generated evaluation reports should stay under `tmp_vlm_profile_eval/` or
another ignored temporary folder.
