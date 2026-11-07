---
target: the live dashboard
total_score: 29
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 4
target_identity: "file:/Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html"
target_fingerprint: "sha256:2a0c6b7ad338ea59e2944a5ec2e2566dc8b2a1887d251fcd669a5fa1329793fc"
target_path: /Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html
timestamp: 2026-11-07T19-33-52Z
slug: orchestrator-dashboard-static-index-html-f01a3df2
---
Method: dual-agent (A: Opus design review · B: Sonnet detector + overlay), plus an Opus audit with break-ui and review-animations. Target: the live dashboard, `runs/shot` at e6b7f21 (c738f5e + phase-1 with #8 and #9).

## Design health score (A's scores)

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 3 | During a fine-tune every panel says Live over a finished, already judged frame while the bar says no robot is moving; no elapsed time |
| 2 | Match system / real world | 3 | Plain spoken copy; the judged take is titled "Student trial · development seed 900001" |
| 3 | User control and freedom | 3 | Every key reversible, P returns exactly; nothing returns to the landing story; Escape doesn't close Labels |
| 4 | Consistency and standards | 3 | Controller identity consistent everywhere; P shows student frames under a bar describing the teacher's end |
| 5 | Error prevention | 3 | Keys disable without a target, Space never clicks a focused button; a navigation key quietly stops following a live run |
| 6 | Recognition rather than recall | 3 | Key row is the legend; "acting on episode 29" is a number D then contradicts |
| 7 | Flexibility and efficiency | 3 | Every action has a key and a button; no jump back to the story |
| 8 | Aesthetic and minimalist design | 3 | Restrained; at 1280×720 the frame is 13% of the screen and the largest type is the held-out figure |
| 9 | Error recovery | 2 | Errors and "connection lost" name the problem, never the consequence; Live stays up while disconnected |
| 10 | Help and documentation | 3 | Labels legend and explanatory empty states teach in place |
| **Total** | | **29/40** | **Good** |

## Design specificity
Authored in behaviour, generic in chrome. Controller identity three ways (colour, glyph, rail height), the move in ink in four places at once, a demonstration citing its diagnosis with that episode's frame, provenance on every panel: none of that transfers to another product. The surface (grey ground, white 6 px panels, system-ui, three columns) is stock ops dashboard. Against "camera as the stage": right order, wrong proportion at the projector size (429×322 frame, 13% of 1280×720; 21% at 1440×900).

Detector: `impeccable detect` exit 0, one advisory `repeating-stripes-gradient` (style.css:86, :112). False positive: the hatch is the placeholder and missing-frame signal, explained in the Labels legend. Overlay at 1440×900 also flagged `cramped-padding` on the header Placeholder chip (7.2 px beside 18 px text): real, low impact. Overlays were injected in the reviewer's own tab, which is closed.

Audit: 15/20 (a11y 3, perf 4, theming 3, responsive 3, integrity 2). Integrity verdict pass; the drop from 16 comes from a live replay with a real sim name, which this round ran and round 2 couldn't. All 7 round-2 audit findings fixed (one partly). Escaping holds in every field, `/frames` rejects every traversal probe, reconnect rebuilds exactly, 154 episodes render in 0.48 ms, nothing under 16 px, lowest contrast 5.51:1. review-animations: approve.

## Overall impression
A careful, honest instrument: a sceptical judge can trace frame → skill → verdict → diagnosis → demonstration with an n on every figure, and the live arrival of the move is the best moment on the page. What's weak is the run's end and the presenter's walk: the view leaves the judged take 20 s after it ends, Live is claimed on judged frames and while disconnected, the keys jump by index rather than by cause, and the frame is too small for a room. Biggest opportunity: make the run-end and provenance edges exact, then give the frame the stage at 1280×720.

## What's working
1. The move as the peak, live: ink outline, a one-time flash on live arrival only (reduced-motion safe), the inverted log row, "First decision after the move".
2. The evidence chain on one screen: the landing story shows the accepted demonstration's reasons beside the diagnosis it acted on, with that episode's frame, and the verdict in words.
3. Presenter controls a presenter can trust: the key row is legend and buttons, Space pauses from any focus, P returns exactly to where it was.

## Priority issues
- **[P1] The view leaves the judged demo ~20 s after the run stops.** `episode()` falls back to `story()` once `following` lapses (state.js:219–229), and `mom demo` appends to the run. In the pitch, the screen switches to another episode with its own verdict while Owen talks about the take. Fix: when a followed run stops, pin the episode shown; open on the story only for a page loaded on a finished run; behaviour test. `harden`.
- **[P1] Live is claimed where it isn't true (rule 6).** Found independently by A and the audit: during a fine-tune, panels read Live on an episode already judged (state.js:159–165), and Live stays up for up to 30 min after the stream drops (state.js:153–154 reads only event times). Fix: an episode with a verdict is recorded; provenance reads the connection state; the stage shows a fine-tune state with elapsed time. `harden`.
- **[P1] The camera is not the stage at 1280×720.** Frame 429×322 (13%), bound by height because the filmstrip and caption take 134 of 386 px; in P the pair shrinks to 258×194 each; 295×222 in a windowed 720p browser. The FakeSim drawing flatters this; a real 320×240 MuJoCo frame won't survive it. Fix: at short viewports drop the filmstrip (or fold its marks into the caption) so the frame is width-bound (~530×398); in P let the camera span the camera and log columns. style.css:67, :97–101, :116–124. `layout`.
- **[P1] D and T jump by index, not along the link the page draws.** From the story (episode 49, acting on 29), D lands on 30, a different failure; T from 30 lands on 31, which acts on 21. Fix: from a demonstration D goes to the episode it acts on; from a diagnosed episode T goes to the latest demonstration acting on it; nearest only without a link. Changes SPEC §3.5's wording. state.js:231–247. `shape`.
- **[P2] The story's diagnosis falls below the fold.** On a 3-round run the side column is 901 px in 533 (41% hidden) and the evidence is what's cut; with two rounds the table repeats the figure above it. Fix: collapse the held-out table to the headline (first → last, n) and clamp the hypothesis so the evidence line always shows; a fade cue at the column's foot. app.js:249–257, :286, style.css:69. `distill`.
- **[P2] The log stops short of its verdict and never follows again** when the first paint comes after the replay: `renderLog` scrolls before `renderModels` grows the footer by 29 px (app.js:231, :305). Fix: render the footer first, and decide "following" from the reader's own scroll events. `harden`.
- **[P2] Long reasons are clamped to 3 lines with the rest only in a hover `title`** (style.css:146, app.js:180), unreadable on a projector or by keyboard; real Argon reasons will trip it. Fix: the stage step's reason in full, older rows clamped. `harden`.
- **[P2] M lands on the moved episode's last frame with nothing about the move on screen.** Fix: M opens at the first frame after the move with the log at the move row, so M shows "after" and P adds "before". state.js:249–256. `polish`.

## Persona red flags
- **Owen presenting:** the view jumps off the judged take; D goes to the wrong diagnosis; M needs a second key before anything shows; no key back to the story; Labels ignores Escape and covers the log.
- **Sceptical judge in row six:** the frame is 13% of the screen; the biggest number is "0/20 → 0/20"; the judged take says "development seed 900001"; Live over a frame the bar says isn't moving.
- **Screen-reader user:** live region polite and quiet (11 announcements a run); it never names the episode, and Pause/P say nothing unless their button has focus; the pause toggle flips its label and `aria-pressed` together.

## Minor observations
- Teacher latency median counts the "Stopped, no skill chosen" rows at 0.0 ms: n=177 should be 156 (state.js:100, app.js:179). Rule 6.
- Frame paths resolve against the server's cwd: started from another directory, every frame is null and the page claims "recorded no head-camera frame" (server.py:31).
- During P the control bar describes the episode's end above two student frames (app.js:100–111).
- On FakeSim, Paused never shows as a chip (placeholder wins, state.js:162–164).
- Disabled keys (#566070) are hard to tell from enabled ones at distance.
- Rejected demonstrations don't say they won't train the student.
- Missing-frame thumbnails lose their hatch, and the moved flag replaces "Frame missing" (style.css:119, :122).
- The rail has no phase boundaries; at 600 episodes marks fall to ~1.6 px.
- Rail "here" outline clipped by 1 px at the right end and atop teacher marks (style.css:33–40).
- Header Placeholder chip padding cramped at 1440×900 (detector).
- Stub artefacts, honest but worth knowing: student "0.0 ms n=459", fine-tune "0.1 ms n=1", flat 0/20 → 0/20.

## Questions to consider
- If the real held-out result is flat, should the largest type on the page still be the held-out figure?
- What if P weren't a mode: the first frame after a move always carries the frame before it as an inset?
- Could the fine-tune wait be a beat of the pitch (the training set going in, elapsed time against the last round) rather than a frozen frame?
- Should the rail be 71 equal marks, or the loop's phases with counts that judges can map onto the story?
