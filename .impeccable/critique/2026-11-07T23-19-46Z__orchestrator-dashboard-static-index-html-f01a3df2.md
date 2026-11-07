---
target: the live dashboard on the MuJoCo fixture
total_score: 30
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 2
target_identity: "file:/Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html"
target_fingerprint: "sha256:91125959b9ce6b0dc69abed656f1094e8a70c22debdefde04cbc32a14a9a91ef"
target_path: /Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html
timestamp: 2026-11-07T23-19-46Z
slug: orchestrator-dashboard-static-index-html-f01a3df2
---
Method: dual-agent (A: Opus design review · B: Sonnet detector + overlay), plus an Opus audit with break-ui and review-animations, live replays, and a heartbeat stall test. Target: the live dashboard on `fixtures/shot-mujoco` (real MuJoCo head-camera frames, fixed_feet task) at cff8641 (fix round 4 + phase-1 #10). First round on real frames.

## Design health score (A)

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 3 | Live/Paused/Recorded/fine-tune/lost/restored all work; under M and P the bar names the episode's final controller, not the stage frame's; between episodes the bar drops kind and seed; a lost stream flips the header to "Recorded" |
| 2 | Match system / real world | 3 | Plain copy, but the header shows `mujoco-4.0.0 task=across_room` and the held-out claim is labelled only `ckpt-r0 → ckpt-r1` |
| 3 | User control and freedom | 3 | P, End, Escape all return; no way to put a chosen step on the stage |
| 4 | Consistency and standards | 3 | Coding consistent; header changes height on Space at 1280; bar vs stage controller contradiction in M |
| 5 | Error prevention | 3 | Keys disable at edges, Space never clicks; a jump silently stops following a live run |
| 6 | Recognition rather than recall | 3 | Keys labelled, move tag carries "· P"; rail code only in Labels; diagnoses cite "episode 22" with a 96 px thumbnail |
| 7 | Flexibility and efficiency | 3 | One key per action, beat jumps; no stepping through frames; URL doesn't hold the episode |
| 8 | Aesthetic and minimalist | 3 | Frame dominates at 720p; six identical provenance chips; 8-control key row; header wraps with real names |
| 9 | Error recovery | 3 | Reconnect rebuilds and announces; while lost the header says Recorded and rows keep "running" |
| 10 | Help and documentation | 3 | Labels, captions, empty states; run phases not shown |
| **Total** | | **30/40** | **Good** |

## Design specificity
Interaction authored for this product (D/T/M/P beats, verdicts by kind, "What the teacher saw at step 7, before choosing", the skill on the frame as the robot's voice, cited diagnosis frame, provenance per panel). The surface is category-generic (grey ground, white 6 px cards, system-ui, outlined chips). Real frames expose its cost: the high-key render (white checker floor) loses its edge on a white panel, so the stage reads as a picture in a card, not a camera feed.

Detector (B): exit 0, one advisory `repeating-stripes-gradient` (style.css:90, :135), a false positive (the hatch is the placeholder/missing-frame sign) and not on screen for the MuJoCo run. Overlay: 0 findings in 5 views (landing 1440/1280, P, T).

Audit: 16/20 (a11y 3, perf 4, theming 3, responsive 3, integrity 3), up from 15. No P0/P1. Holds: 16 px floor in every view; contrast ≥ 5.52:1; Space from a focused button pauses; focus never to body; escaping; `/frames` guard (7 probes 404, real frame 200); provenance; every figure with its n; heartbeat both ways (25 s gap with heartbeats Live; without, Recorded at +20.9 s); lost-connection fine-tune; reconnect rebuild; run-end stay; render 0.7 ms at 1,524 events. Review-animations: approve.

## Priority issues (deduplicated)
- **[P1] Under M and P the control bar names the wrong controller** (A). The bar shows the episode's latest controller; on the judged take (teacher took over at step 8) M shows "Student ckpt-r1" on the frame under an orange "Teacher (Argon)" bar, and in P the bar's teacher is the only controller on screen while both frames are the student's. Fix: while the view holds a step, the bar names that frame's controller and checkpoint; P keeps each cell's controller tag. state.js:87,108,123; app.js:141-143,173. `clarify`.
- **[P1] The header row doesn't fit at 1280 with real simulator names; Space reflows it** (A). `mujoco-4.0.0 task=across_room` wraps the header permanently at 1280 and 1440 (42→78 px, 47→87 px); with the short name, Space wraps it (75→111 px), shrinking the frame 596→575 px while the presenter points at it. Fix: chip says "Live · MuJoCo"; version and task ("walking task (across_room)") go to Labels' run line; `#b-pause` gets a fixed min-width for "Resume · 999 new". app.js:47-52,88,91-93; style.css:27,57-58. `layout`.
- **[P2] The skill chip covers the robot's hands on real frames** (A minor 8 + audit P2). The bottom band holds the evidence (hands, held box, bin at placing distance); the top ~40% is empty floor where the top tags already sit. Fix: move the skill chip into the top band beside the controller tag. style.css:113. `layout`.
- **[P2] P's pair on real frames also shows the robot's own move; the caption overclaims** (A). Frame N is what decision N saw, so the pair spans the executed skill too (on the take, the student's turn_right rotates the whole view; walking adds head bob). "The first to show it" is false for a bin moved out of view. Fix options: (1) SPEC change: the orchestrator observes once more right before `perturb()`, so P's pair differs only by the move; (2) the caption names everything between the frames ("the student's turn_right, then the box was moved"); (3) "the first frame after it". `clarify`.
- **[P2] Above 1280×720 the stage stops being the stage** (A + audit). 1440×900: frame 591×444 vs log 696×448; 1920×1080: frame 738×554 vs log 934×607. The filmstrip and heading take the camera column's height. Fix: show the strip only when the frame already fills the column's width; fold the heading into the caption. style.css:108,125-127,157-163. `adapt`.
- **[P2] The announcer replays the backlog on every reconnect and first load** (audit). ~9 announcements in 160 ms, can settle on a stale count ("Episode 49 of 54" on a 71-episode run); the stage flickers through intermediate episodes. Fix: don't speak or show until the replayed backlog drains (receive already knows replayed events, state.js:227-230); include the count in `lastHeard`. app.js:98-112,457-462. `harden`.
- **[P2] Dead air on the stage** (A). Blank white "No head-camera frame yet" between every episode (0.3–3.8 s here, longer with SmolVLM2) while the bar drops kind and seed; the fine-tune stage is a clock on white. Fix: a frame-shaped slate with "Held-out test · seed 6 · …" and the bar's kind/seed from `episode_started`; during the fine-tune, this round's accepted demonstrations as a contact sheet with "5 accepted · 1 rejected (n=6)". app.js:118-128,149-163. `onboard`.
- **[P2] Single-letter keys act inside editable fields** (audit, WCAG 2.1.4). Harmless today; the typed instruction box (next stage) needs `keydown` to ignore inputs. state.js:331-336. Belongs with the input box change.

## Mechanical batch (P3 / minor, cheap)
- `.sr` spans positioned against the page make it scroll out of reach (`ol.log li`, strip `li` need `position:relative`) (audit P3, real bug).
- "No reasons yet" on a finished episode → "No reasons given" (audit).
- n=0 held-out in words, not "0/0" (audit break-ui).
- Diagnosis evidence clamp (~4 lines with title) (audit break-ui).
- P's pair restacks below 1001 px (audit).
- Held-out panel during a live evaluation says "No held-out evaluation yet" → "Testing ckpt-r0: 6 of 20 run"; claim labels in words "before fine-tuning → after round 1", ids as the note (A minor 4, 5).
- Diagnosis empty-state copy: the teacher first demonstrates on the failed seeds, then new ones (A minor 2).
- A lost stream: header chip "Disconnected" in its own style; rows stop saying "running" (A minor 3).
- The flash's white wash hides the moved frame: ink ring only (audit animation).
- P caption clamp to 3 lines vs SPEC "reason in full" (audit, owner's call).

## For the overhaul (dark glass)
- Dark surround first: the high-key render needs a dark frame edge; chips over frames need a solid backing and 1 px edge.
- Move semantic hues off the scene's colours: teacher orange collides with the orange box, success green with the green bin (pick a teacher hue away from orange; let shape carry more).
- Group the key row [← → End] [D T M] [P Space]; clearer disabled-key state; rail in-progress mark ≥ 3:1; the mono skill caption as the robot's voice; a texture for "no data" the detector won't flag; one provenance chip unless a panel differs from the header.

## Persona red flags
- **Owen presenting:** header jumps on Space; "Teacher (Argon)" in the bar while he says "the student"; white flicker between episodes; a clock to talk over during the fine-tune.
- **Sceptical judge:** `ckpt-r0 → ckpt-r1` instead of words; P's change includes the robot's own turn; 96 px cited frame; `task=across_room` reads as a debug string.
- **Screen-reader user:** stale backlog announcements; rail has no text equivalent; `#side` labelled "Did it work" but holds the diagnosis.

## Fixture notes
Held-out 0/20 → 0/20 and stub timings (student 0.6 ms, fine-tune 0.2 ms) are honest for the fixture but read as claims; canned diagnoses and reasons visibly contradict their frames (don't use it for the video); the take's move at step 1 is subtle (choose a move that changes what the frame shows: the box out of the hands' gap, or the bin in or out of view).
