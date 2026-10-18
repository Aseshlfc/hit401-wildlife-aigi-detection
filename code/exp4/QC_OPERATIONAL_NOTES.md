# Experiment 4 — QC operational definitions (fixed BEFORE any main-set image was reviewed)

Written 2026-09-24 ~16:25 UTC, while main generation was running, after looking only at the 21 pilot images
(which are not part of the dataset). These notes apply the fixed rules R1–R8 in `exp4_qc.py` consistently with
the Dataset v1 SDXL review. They do not add new rules. Detector scores are never available during QC.

- **R2 (text / timestamp overlay):** any rendered text, numbers, date/time strip, camera-brand text or logo
  → REJECT. Dataset v1 rejected invented text, watermarks and UI (and its later negative-prompt variant excluded "timestamp, date,
  camera brand"), so the same applies here even though some real trail cameras print info bars.
- **R7 (style mismatch):**
  - camera_day that clearly reads as a night/infrared frame (fully monochrome grey IR look, dark night scene) → REJECT.
  - camera_night_ir that is a colour daylight scene → REJECT.
  - community that clearly looks like a fixed trail-camera frame (fisheye trail-cam view with IR/timestamp look) → REJECT.
  - Mild desaturation, dim light, blur or noise is NOT a mismatch.
  - cartoon / CGI / painting look → REJECT.
- **R6 (species):** the target species must be recognisable to a reasonable viewer (for night/IR by body shape).
  A small, distant or partly hidden animal is acceptable if it is still recognisable; no animal visible, or
  clearly a different species → REJECT.
- **R4/R5 (anatomy / merged animals):** clearly wrong number of legs, heads or tails, fused bodies, or two animals
  merged → REJECT. Legs hidden by vegetation or pose are fine.
- **Multiple animals** of the target species that are anatomically plausible and separate → ACCEPT (it happens in real photos).
- **Genuinely borderline** cases the reviewer cannot decide by these definitions → NEEDS_USER, to be decided by the user.
- The reviewer records the rule and a short reason for every non-ACCEPT decision.

## USER DECISION 2026-09-24 ~18:20 UTC — R7 for camera_day ("Strict daylight", option 1)
Context: during review of the first complete cells (opossum, raccoon), FLUX rendered most camera_day prompts for
these nocturnal species as flash-lit night scenes. Decision by the user (verbatim intent):
- Keep the existing prompts and protocol unchanged. Do NOT change the prompt; do NOT relax R7 to reduce replacements.
- camera_day: ACCEPT only images clearly lit by natural/ambient daylight (overcast or shaded daytime is fine if it
  is clearly daytime). REJECT flash-lit night scenes, black/dark-background trail-camera scenes, infrared-looking
  or monochrome night scenes, and timestamp/text-overlay failures (existing rules R7/R2).
- Genuinely ambiguous between daylight and night/dusk → NEEDS_USER (user review), never forced acceptance.
- Every rejected image is replaced with the next unused predefined seed, exactly as the protocol specifies.
- Record the number and percentage of camera_day QC failures by species (methodological finding about how
  FLUX interprets the capture-style prompt).
- **Stop rule:** if any species×style cell needs an unusually large number of replacement attempts, stop and report
  the attempts and failure pattern before any methodological change. Operational threshold, set now before any
  replacement round: STOP a cell when its total generated attempts exceed 5 × its quota (camera_day/night_ir: > 60;
  community: > 120) without reaching the quota, or earlier if the observed acceptance rate makes that certain.

## USER DECISIONS 2026-09-25 (round 2 NEEDS_USER review)
- **Bobcat tail rule (R6), applied to ALL bobcat images:** clearly long/ringed tail → REJECT R6; proper short/bobbed
  tail → ACCEPT if no other issue; tail hidden or genuinely unclear → not rejected on tail length alone.
  Do not relax this because FLUX makes the error often. If a bobcat cell reaches the 5× stop rule, stop and report.
- **Squirrel striping rule (R6):** clear chipmunk-like side striping → REJECT R6; faint/non-diagnostic fur striping
  is not a rejection reason by itself.
- Individual cases: dusk/flash-lit camera_day under a deep-blue sky → REJECT R7; unrecognisable night squirrel → R6;
  skunks with unusual transverse banding but clearly skunks → ACCEPT.

## USER DECISION 2026-09-25 — stop rule wording (continue to the cap)
Keep generating next-unused seeds under unchanged rules until the cell quota is filled or the cell reaches its cap
(5 × quota; camera_day 60). Generate up to the cap, never beyond it. A cell that reaches the cap without filling its quota
is STOPPED and reported (attempts, accepted, rejected, rejection reasons); no methodological change until the user reviews it.
The other cells continue normally. (`exp4_generate.py --mode replace` generates min(need, cap − attempts).)

## USER DECISION 2026-09-25 — bobcat camera_day reached the cap ("Decide later")
bobcat camera_day: 60/60 attempts, 5 ACCEPT, 55 REJECT → cell STOPPED at 5/12. User decision: pause this cell,
continue the other cells to quota or cap, then report all capped cells together before any decision.
Also: a blank black strip along an image edge (no text) counts as a border (R1) — user decision on 644038.
