# Product

<!-- impeccable:product-schema 1 -->

Inferred from SPEC.md and the workstream brief, not from an interview (no one was available to answer). Treat every line as unconfirmed until Owen reads it.

## Platform

web

## Users
Robotics judges at the AgentVerse 2026 hackathon (Robotics track), watching the dashboard during a 5-minute live pitch on a laptop screen or projector, and later a 1-minute video. They know manipulation and sim-to-real; they are sceptical of staged demos and want evidence. The presenter (Owen) drives the page while talking.

## Product Purpose
Scaffld: an Argon teacher agent watches a weak robot student fail in simulation, diagnoses why from the robot's camera, demonstrates corrected behaviour, keeps only demonstrations an independent validator accepts, and fine-tunes the student. The dashboard shows that loop as it runs: what the robot sees, who is in control, what it decided and why, whether it worked, how much it has improved, and where each model runs. Success is a judge who can state, unprompted, what the robot saw, what changed, and whether the student got better.

## Positioning
The numbers are not asserted, they are derived: every figure comes from saved run files and carries its sample size, and each panel says whether it is live, recorded or placeholder.

## Operating Context
Pitch room: a projector or laptop, viewed at a distance, often in ambient light. One continuous take with a mid-run perturbation (the box or the bin is moved; the robot learns of it only through its camera). Data arrives as server-sent events from `runs/<id>/events.jsonl`.

## Capabilities and Constraints
Event types and fields are fixed by `isaacs_mom/contracts.py`. The head camera is 320x240. Decision events carry a skill call and no free-text reasoning yet; model latencies are not in the events yet (shown as placeholder until they are). Held-out evaluation is always n=20.

## Product Principles
- Evidence over decoration: show the frame, the skill, the verdict, the n.
- A zero is shown as a zero. No panel fills a gap with a plausible number.
- Control is always visible: student checkpoint or teacher, at every step.
- Readable from the back of a room.

## Accessibility & Inclusion
Contrast of at least 4.5:1 for text; status is never carried by colour alone; motion respects `prefers-reduced-motion`.
