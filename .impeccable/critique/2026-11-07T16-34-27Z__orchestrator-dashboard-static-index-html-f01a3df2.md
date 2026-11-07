---
target: the live dashboard
total_score: 21
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 4
target_identity: "file:/Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html"
target_fingerprint: "sha256:eecbe6e19d1f20faf1796e4057010c7c62ce8079800091a27cff84b2a123e4bf"
target_path: /Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html
timestamp: 2026-11-07T16-34-27Z
slug: orchestrator-dashboard-static-index-html-f01a3df2
closed: true
---
Method: dual-agent (A: design-review sub-agent, Opus · B: detector + browser sub-agent, Sonnet). Audit run separately (Opus).

## Design Health Score
| # | Heuristic | Score | Key Issue |
|---|---|---|---|
| 1 | Visibility of System Status | 3 | Camera overlay credits the teacher with a student frame (uses last decision's controller, not the frame's) |
| 2 | Match System / Real World | 2 | Raw values leak: "(move_block)", "dev", "stopped (no skill)"; fine-tune time listed under "Median latency" |
| 3 | User Control and Freedom | 1 | Log yanked to bottom every 2 s; a new episode wipes the one on screen; no pause or scrub |
| 4 | Consistency and Standards | 3 | Student/teacher colour + shape system holds; overlay contradicts the log |
| 5 | Error Prevention | 2 | FakeSim data labelled "Recorded"; hard-coded model names wear data chips |
| 6 | Recognition Rather Than Recall | 2 | Step N must be matched across three columns; thumbnails not clickable |
| 7 | Flexibility and Efficiency | 1 | No presenter keys (pause, previous/next episode, jump to the perturbation) |
| 8 | Aesthetic and Minimalist Design | 2 | Flat hierarchy: seven identical chips, "0.0 ms" on every row, 155–267 px dead space around the camera |
| 9 | Error Recovery | 3 | Good fallbacks (Frame missing, reconnecting, No reason given); error rows show raw text |
| 10 | Help and Documentation | 2 | Empty states explain; no legend for chips, shapes or step numbers |
| **Total** | | **21/40** | **Acceptable** |

## Design Specificity Verdict
LLM assessment: the information architecture is authored for this product (control bar tinted by who drives, filmstrip with a "block moved" marker, a decision log pairing each skill with its reason, held-out table with n=20). The visual language is category-generic (system-ui, grey field, white 1px-bordered cards, identical chipped headers), and the camera, the product's evidence, sits letterboxed in one of three columns.
Deterministic scan: `impeccable detect` exit 0 with 1 advisory, repeating-stripes-gradient (style.css:41 `.chip.placeholder`, :59 `[data-missing]`), a false positive: the hatching marks placeholder or missing data together with a dashed border and text. Browser overlay at 1440x900 and 1280x720: undersized-ui-text, "block moved" badge at 10.3 px (real; A measured it at 9.3 px at 1280). Overlay injection succeeded in the reviewer's own tab, which is now closed, so no overlay is left visible.

## Overall Impression
Honest, product-specific structure with a generic skin, sized for a laptop, not a projector. Its biggest risk is integrity, not looks: a reconnect doubles every number, and a stale diagnosis sits beside another episode's verdict. The biggest opportunity is making the perturbation and the improvement the two loudest things on screen, since they carry half the rubric.

## What's Working
- Who is in control is shown four ways (tinted bar, colour, circle vs diamond, text), carried into log rows and the overlay; colour-blind safe.
- Data honesty: n on every rate and latency, zeros as zeros, "Frame missing", "No reason given", escaped XSS, no CDN.
- The log reads as a story: skill, plain-English reason, outcome, with the block-moved, takeover and verdict rows between.

## Priority Issues
- [P0] SSE reconnect replays the whole run onto existing state (audit, verified: 2 episodes / 18 events, ckpt-r0 twice, student n=4 vs 2). Why: a server restart mid-pitch puts wrong numbers in front of judges. Fix: reset state, renderedRows and renderedEpisode in onopen (app.js:198-201). Command: /impeccable harden
- [P1] Decision-to-frame trace is broken: overlay credits teacher with a student frame; thumbnail "step N" is the frame after action N; the badge hides the moved block; the diagnosis is run-wide, so a 12.5 cm diagnosis sits beside a 17.3 cm verdict. Why: perception-driven action is 25% of the rubric. Fix: overlay from the frame's controller, label frames "seen by step N+1", corner-flag the moved marker, store diagnosis per episode and show it with a placeholder chip when absent. Command: /impeccable clarify
- [P1] Provenance labels overclaim: "Recorded" on FakeSim data, chips flip to Recorded after 20 s of quiet in a live run (fine-tune, slow Argon), hard-coded model and host names under data chips. Why: a judge who spots one overclaim discounts every number (SPEC §2 rule 6). Fix: sim backend and model/host from events ("Placeholder · FakeSim"), live while a fine-tune is pending, one run-level badge plus per-panel exceptions. Command: /impeccable harden
- [P1] Too small for a projector: root 13.7 px at 1280; chips 10.2, table headers 10.7, latency 11, n= 11.5, badge 9.3. Why: PRODUCT.md's "readable from the back of a room". Fix: html clamp with a 16 px floor, secondary text ≥ 0.85rem, n= at body size. Command: /impeccable typeset
- [P1] The presenter can't hold the key moment, and panels clip: log forced to bottom every 2 s (app.js:128, :214); side column overflow hidden cuts the diagnosis at 1280x720 and the models table at 1024x576 (style.css:28-30); episode line truncated. Fix: autoscroll only when rows were added and the user is at the bottom; side column scrolls; episode line wraps. Command: /impeccable layout
- [P2] Peak and end under-designed: perturbation is a near-white row; improvement is a 12 px note; fine-tune is a dead valley. Fix: ink band plus stage flash on the perturbation; a large "0/20 → x/20 · n=20"; fill the camera's dead space. Command: /impeccable bolder

## Persona Red Flags
- Robotics judge: overlay "Teacher · Step 4" while the log says takeover at step 5; "0.0 ms" on every row; "Recorded" on FakeSim; diagnosis "not diagnosed · 0%" in 12 px, so behaviour when unsure is invisible.
- Presenter (Owen): no keys; the log yanks back while he points; an episode switch erases the perturbation mid-sentence; nothing fills the fine-tune wait.
- Back-row viewer: 9–11 px secondary text, 55x41 px thumbnails, 10 px chips (contrast is fine at ≥ 5.5:1).

## Minor Observations
Clipped focus ring on the log (style.css:10, :28); server hangs on SIGTERM with a client connected (server.py, timeout_graceful_shutdown); raw hex values and inline styles; duplicate li.empty rule in the reduced-motion block; 6 px overflow scrollbar at 1280x720; waiting state shows `uv run mom run` to judges; verdict duplicated in log and panel.

## Questions to Consider
- What if the camera frame were the stage, with decisions as captions and the diagnosis as callouts?
- If a judge looks for ten seconds, which one number should they leave with?
- Should the page follow the latest episode, or the most instructive one (the one with the perturbation and the takeover)?

## Audit (separate run): 14/20 Good
Accessibility 3 · Performance 4 · Theming 3 · Responsive 2 · Implementation integrity 2. P0 1, P1 4, P2 5, P3 2.
