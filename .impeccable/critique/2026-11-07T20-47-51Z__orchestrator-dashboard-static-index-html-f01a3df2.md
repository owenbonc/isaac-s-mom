---
target: the live dashboard
total_score: 30
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 2
target_identity: "file:/Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html"
target_fingerprint: "sha256:ed8e9423f61c8e6f346267e370dc5fd7b79eff8a90298e8fb8e1820cec926a06"
target_path: /Users/owenyan/isaac-s-mom/.claude/worktrees/agent-a4b1bd315ceff52fa/isaacs_mom/orchestrator/dashboard/static/index.html
timestamp: 2026-11-07T20-47-51Z
slug: orchestrator-dashboard-static-index-html-f01a3df2
closed: true
---
Method: dual-agent (A: Opus design review · B: Sonnet detector + overlay), plus an Opus audit with break-ui and review-animations, and live replays of the run as a real-sim run. Target: the live dashboard, `runs/shot` at 2a029d5 (fix round 3, layout B, phase-1 #12/#14/#15 merged).

## Design health score (A)

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 3 | Live states honest (Live, Paused with "Resume · 69 new", fine-tune clock, Live→Recorded, lost stream). At 720p the move's cue lasts one step live; a lost stream is one red 16 px phrase |
| 2 | Match system / real world | 3 | Plain language; but "Accepted/Rejected" (training-admission words) on held-out tests, trials and the judged take; teacher "Argon" on stage, "Gemini API" in the footer |
| 3 | User control and freedom | 3 | P returns exactly; D/T/M have no "back", repeated D/T bounce |
| 4 | Consistency and standards | 3 | Controller colour+shape consistent; "Placeholder" means both FakeSim and an empty panel; verdict words change meaning by episode kind |
| 5 | Error prevention | 3 | Space never clicks a focused button; disabled keys; nothing warns that a jump stopped following a live run |
| 6 | Recognition rather than recall | 3 | Keys on screen, Labels explains; one episode carries two ids (episode 49, seed 1508) |
| 7 | Flexibility and efficiency | 3 | Keys + buttons, beat jumps; repeated D never reaches episodes 21–28 |
| 8 | Aesthetic and minimalist | 3 | Restrained, strong 720p stage; six identical provenance chips per view, seven black pills when paused |
| 9 | Error recovery | 3 | Frame missing in place, error rows, auto-retry; the lost-stream message gives nothing to act on |
| 10 | Help and documentation | 3 | Labels defines chips and rail; empty states teach the loop |
| **Total** | | **30/40** | **Good** |

## Design specificity
Authored for this product in structure and interaction: the frame as stage with the chosen skill over it at 31 px and "What the teacher saw at step 6, before choosing", the controller-tinted bar, the rail's height/fill grammar, the ink language for the move, provenance distinguished by shape, P as a full-width pair. The surface (white 6 px cards on grey, system-ui, a Models footer) is stock admin dashboard; the chips are the most generic and most repeated element.

Detector: exit 0, one advisory `repeating-stripes-gradient` (style.css:89, :129). False positive: the hatch is the placeholder and missing-frame signal, defined in Labels. Overlay: same single advisory in all four views (landing 1440×900 and 1280×720, P, T); no per-element rule fired.

Audit: 15/20 (a11y 3, perf 3, theming 3, responsive 3, integrity 3; integrity up from 2, perf down from 4 for the reconnect fast-forward round 3 didn't measure). Every round-3 provenance and run-end defect fixed and re-checked on a live replay. Text ≥ 16 px at 1280×720, 1440×900, 1024×768; contrast ≥ 4.66:1; escaping holds; `/frames` holds against every probe; reconnect rebuilds to exactly 1,093 events; review-animations approves.

## Overall impression
The stage now reads from the back of the room and the provenance system is exact, including the subtle cases (a judged episode Recorded inside a Live run, a lost stream never Live). What's left is mostly what only a real run shows: the judged take's ending, a slow Argon call, long reasons, and a projector that isn't 720p tall.

## What's working
1. The stage at 720p: frame 596×447, the largest element on every view; the skill over the frame; the stage step's reason in full; 16 px floor everywhere.
2. Provenance as a real system: five shapes, one legend, and the page follows the panel rules exactly in live replays; Pause is unmistakable.
3. P as the peak at 1280×720: two 609×456 frames, the caption carries the first decision after the move, P again returns exactly; M opens on the first frame after the move.

## Priority issues
- **[P1] The judged take and held-out tests say "Accepted".** Found independently by A and the audit. The verdict chip, log row, announcement and Labels say Accepted/Rejected on every episode kind. On the judged take the teacher took over at step 8 and the bar ends "✓ Accepted", the student's checkpoint gone from it: judges read the student succeeded. On a held-out success after a real fine-tune, "Accepted" (the training-admission word) suggests a held-out episode entered training, which rule 2 forbids. Fix: words by kind. Teacher demonstration "Accepted for training" / "Rejected"; held-out test and student trial "Succeeded" / "Failed"; judged take "Succeeded" / "Failed", with "after the teacher took over at step N" or "by the student alone". app.js:56, :63, :115-126, :213-217, :237; index.html:34. `clarify`.
- **[P1] P's pair shrinks under a full-width filmstrip on any screen taller than 760 px.** At 1440×900 each P frame is 472×354, smaller than the single frame (591×444), over six 226 px thumbnails; at 1920×1080, 579×434 over 303 px thumbnails. A 1080p projector or a mirrored laptop is common. Fix: hide the strip while P is on (`main:has(#stage .cell + .cell) .strip{display:none}`). style.css:120, :145-152. `adapt`.
- **[P2] A step slower than 20 s flips Live → Recorded → Live mid-episode.** Real Argon calls with retries can take that long (audit fed a 25 s stall). Fix: a heartbeat event while a step is in flight, or a wider window; SPEC-level. state.js:5, :162-163.
- **[P2] At 720p a live move stays on the stage for one step.** The strip (and its moved mark) is hidden at ≤ 760 px; the caption and outline last only while that frame is the latest (~1–2 s on a real run). Fix: a persistent tag on the stage for the rest of the episode ("Box moved at step 2 · P"). style.css:151-152, app.js:160-177. `layout`.
- **[P2] The stage frame's reason can sit half out of view at 720p.** The log is 153–157 px tall on a teacher demonstration; a real-length Argon reason row is ~96 px. Fix: scroll the log to the stage step's row, not the foot, while it is the latest. style.css:136, app.js:265. `harden`.
- **[P2] The models footer clips between 1001 and ~1210 px wide** (1024×768, or 125% zoom on the projector): the fine-tune n and the provenance chip go off screen. style.css:75. `adapt`.
- **[P2] The pause key's accessible name ("Pause") contradicts its visible label ("Resume · N new")**: WCAG 2.5.3. index.html:19, app.js:231-233. `harden`.
- **[P2] Provenance chips repeat at equal weight; "Placeholder" also marks an empty panel on a real run** (app.js:38): a hatched Placeholder on a real run's undiagnosed episode tells a judge "this part is fake". Fix: quieter panel chips; Placeholder for FakeSim only, an empty panel says so in words. `quieter`.
- **[P2] Repeated D or T bounces between two neighbours** (49 → 30 → 29 → 30 …); SPEC §3.5 says "nearest", so an owner call. state.js:272. `harden`.

## Persona red flags
- **Owen presenting:** the cue to press P leaves the 720p stage one step after the move; a dropped stream is a quiet red phrase; after any jump he must remember End.
- **Sceptical judge:** green "Accepted" on a teacher-rescued judged take and on held-out tests; "Argon" on stage vs "Gemini API" in the footer.
- **Screen-reader user:** strong (polite announcements, spoken held-out figure, focus ring); the pause name mismatch; the box move and a lost connection are never announced; the rail has no text equivalent.

## Minor observations
- The fine-tune stage is a bare clock; the side panels keep the last demonstration (often a red rejection); the training-set summary (next round) belongs under the clock.
- The diagnosis's cited frame is ~96 px wide: too small to check "both hands left of the box" from the room.
- After M, the teacher-orange bar ("took over at step 8") sits over a student frame at step 2; same during P.
- "No reasons yet" on student episodes overpromises: the student never gives reasons.
- The side band's height varies with the diagnosis, so the log header moves between episodes; the held-out column width depends on its chip's word.
- A presenter button that disables itself drops focus to `<body>`; P hiding a focused log does the same.
- With the stream down, the fine-tune stage keeps saying the fine-tune is running.
- Model names are fixed in app.js:13, not read from the run (rule 6: configuration, labelled so).
- M is disabled on the only moved episode when it is shown (by design: "nearest other").
- Rail marks fall under 1 px at ~700 episodes; a reconnect visibly fast-forwards.
- Fixture: FakeSim refused the demo's box move (the box was held), but the pre-#12 loop recorded it anyway, so P shows two identical scenes; with #12 a refused move is an `error` row. The demo's `--perturb-at` must fall while the box rests on the table.

## Questions to consider
- What if the judged take had its own end line: "Student ckpt-r1 on an unseen scene: succeeded alone" or "needed the teacher from step 8"?
- On a 320×240 head camera a 10 cm move is a few pixels from the back: could the "After" frame mark what changed (a browser-side difference of the two JPEGs, no ground truth)?
- Does rule 6 need six chips saying the same word, or one loud run label and a quiet label only where a panel differs?
- If the camera is the stage, why is the teacher's diagnosis, a reading of a camera frame, the smallest image on the page?
