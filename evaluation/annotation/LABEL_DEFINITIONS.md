# Label Definitions

| Label | Definition |
| --- | --- |
| `seatbelt_worn` | A continuous belt path is visibly crossing the relevant occupant's torso. |
| `seatbelt_missing` | The relevant torso is sufficiently visible and no belt path is visible. |
| `helmet_compliant` | The relevant head is visible and a compliant helmet/hardhat is visibly worn. |
| `helmet_missing` | The relevant head is sufficiently visible and no helmet/hardhat is visible where required. |
| `phone_use` | A phone is visible in hand/near the face or driver area with active interaction supported by the footage. |
| `hands_on_wheel` | Required hand contact with the steering wheel is visible. |
| `hands_off_steering_wheel` | The steering wheel and relevant hands are visible and required contact is absent. |
| `no_violation` | The reviewed interval contains no target-profile violation with adequate visibility. |
| `insufficient_visibility` | Blur, glare, darkness, crop, angle, or occlusion prevents a reliable target decision. |

`missing` labels require adequate visibility; absence of a detector output is
not a human-label criterion.
