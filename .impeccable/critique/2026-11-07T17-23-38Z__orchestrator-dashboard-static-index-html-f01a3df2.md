---
target: the live dashboard
total_score: 26
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 4
target_identity: "file:/Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html"
target_fingerprint: "sha256:765ab5d61a7efb15aa9f071488c440e2ae285d771503699bf1518052e4a1e0f9"
target_path: /Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html
timestamp: 2026-11-07T17-23-38Z
slug: orchestrator-dashboard-static-index-html-f01a3df2
closed: true
---
Method: dual-agent (A: design-review sub-agent, Opus · B: detector + browser sub-agent, Sonnet). Audit run separately (Opus). Re-review after fix round 1.

## Design Health Score
| # | Heuristic | Score | Key Issue |
|---|---|---|---|
| 1 | Visibility of System Status | 3 | No loop-phase indicator (attempt, diagnose, demo, fine-tune, evaluate) and no run-complete state |
| 2 | Match System / Real World | 3 | "Seen by step 7" ambiguous; episode line dumps seed and id |
| 3 | User Control and Freedom | 2 | Back to live means up to 70 presses of →; no Home/End; Space on a focused button doesn't pause |
| 4 | Consistency and Standards | 3 | Verdict in three places with three wordings; "Configured" chip shaped like provenance chips |
| 5 | Error Prevention | 3 | Key repeat ignored, buttons disable at the ends |
| 6 | Recognition Rather Than Recall | 2 | Presenter must remember which episodes hold the story beats |
| 7 | Flexibility and Efficiency | 2 | Four labelled keys but no jumps to beats or to the latest episode |
| 8 | Aesthetic and Minimalist Design | 2 | "No reason given." on every row, seven identical Placeholder chips, flat type scale (124/132 nodes at 16 px) |
| 9 | Error Recovery | 3 | Hatched missing frames, red error rows; "connection lost" only a muted header note |
| 10 | Help and Documentation | 3 | Empty states explain; no legend for Placeholder / Recorded / Live |
| **Total** | | **26/40** | **Acceptable** (was 21/40) |

## Design Specificity Verdict
LLM assessment: structure is authored for this product (camera as stage, filmstrip, controller-tinted bar, hatched provenance, ink-inverted block move); the skin stays generic (white cards on grey, system-ui, 6 px radii).
Deterministic scan: 1 advisory, repeating-stripes-gradient (style.css:57, :76), a false positive (semantic placeholder hatch). Browser overlay: flat-type-hierarchy (body 18 px, h2 20.3 px, h1 22.5 px; largest step 1.13:1 vs 1.25 target), real but advisory since weight separates headings. Overlay injection ran in the reviewer's own tab, now closed.

## Overall Impression
The fix round cleared every earlier finding (audit 14 → 16/20, heuristics 21 → 26/40) and the dashboard is now trustworthy and readable. What's left is storytelling: the block move isn't visible at projector distance, a teacher demo can't show the diagnosis that caused it, and the presenter has to walk 71 episodes one at a time.

## What's Working
- Provenance is built into the components: one chip shape per meaning, n on every number, configured values separated from measured ones.
- Who is in control never relies on colour.
- Presenter keys are labelled on screen as <kbd>; the block move is a strong, reduced-motion-safe peak.

## Priority Issues
- [P1] Space on a focused presenter button activates the button instead of pausing (audit, app.js:270). After one mouse click the pause key breaks mid-pitch. Fix: Space always pauses; preventDefault the button's own activation. Command: /impeccable harden
- [P1] The block move can't be seen at projector distance: step 1 and step 2 frames look identical, and no re-planning is visible. Adaptation is 25% of the rubric. Fix: P shows the frame before and the first frame after the move side by side at stage size, and marks the first decision after the move. Command: /impeccable bolder
- [P1] One episode per view breaks the diagnosis → demo chain: every teacher demo says "No diagnosis for this episode". Fix: on teacher episodes, show "Acting on diagnosis from episode N" with its hypothesis and cited frame (needs the loop to record which diagnosis a demo acts on). Command: /impeccable shape
- [P1] Linear navigation across 71 episodes (45 presses to reach a diagnosis). Fix: story-beat keys (D diagnosis, T teacher demo, M block move, End live) and a thin episode rail coloured by controller and verdict. Command: /impeccable layout
- [P2] "Where models run" (autonomy, 15%) sits below the fold at 1280x720 with no visible scrollbar; three chrome bars take 21% of the height. Fix: merge header and episode line, fold the Validator panel into the log and control bar, models as a one-line strip. Command: /impeccable distill
- [P2] Releasing P leaves the episode pinned and the log stops following (state.js:205-210, app.js:174-181). Command: /impeccable harden
- [P2] Log noise and flat type: empty reasons halve the log's capacity. Fix: "no reasons yet" once in the header, one line per decision, larger current skill on the stage. Command: /impeccable quieter

## Persona Red Flags
- Robotics judge: can't see what moved; "No diagnosis" during the teacher's demo; "LoRA 0.0 ms n=1" reads as instant; "Seen by step 7" before or after action 7?
- Presenter (Owen): must remember beat episodes; "pinned" is the only sign of leaving live; side column keeps its scroll so the held-out figure may be hidden at "it improved"; focus ring lingers on the Decisions panel after a click.
- Screen-reader user: the aria-live control bar rewrites every step.

## Minor Observations
Present tense on finished episodes; frame shrinks ~75 px when the move caption appears (layout shift at the peak); event rows reuse decision step numbers; "round 0" next to ckpt-r1; 62 px thumbnails at 720p; paused view's chips say Recorded; frames re-fetched with no cache headers; the 30-minute live cap from SPEC §3.5 is not yet in code.

## Questions to Consider
- Should a finished run land on the story (best before/after pair plus the held-out change) instead of the latest, failing episode?
- Could one page-level provenance banner, with panel chips only where a panel differs, do more than seven repeated chips?
- With no reasoning text yet, could each decision show the part of the frame it acted on?

## Audit (separate run): 16/20 Good (was 14/20)
Accessibility 3 · Performance 4 · Theming 3 · Responsive 3 · Implementation integrity 3. All 12 previous findings fixed. New: P1 1, P2 3, P3 3.
