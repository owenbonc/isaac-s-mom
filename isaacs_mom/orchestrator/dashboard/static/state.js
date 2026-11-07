// Folds SPEC §4.4 events into the dashboard's state. Pure: no DOM, so node can test it.
(function (root) {
  const CONTROLLERS = ["student", "teacher"];
  const FILMSTRIP = 6;
  const FRESH_SECONDS = 20;
  const FINETUNE_SECONDS = 30 * 60;

  function initial() {
    return {
      runId: null,
      lastT: 0,
      eventCount: 0,
      episodes: [],
      rounds: [],
      latency: { student: [], teacher: [] },
      finetune: { pending: null, durations: [] },
      errorCount: 0,
    };
  }

  // The page's view of the stream. `live` folds every event; `frozen` is what a paused view keeps showing and when
  // it was paused; `selected` pins an episode (null follows the run: its latest episode while live, its story once
  // finished); `step` holds the shown episode at a step (M: the first frame after its move), null at its latest;
  // `compare` is the box or bin move P shows, with the selection to return to when P is pressed again;
  // `followed` is set once an event arrives while the run is live; `lost` while the event stream is down.
  const view = () => ({ live: initial(), frozen: null, selected: null, step: null, compare: null, followed: false, lost: false });
  // Every connection replays events.jsonl from the start, so a reconnect starts over.
  function reconnect(v) {
    v.live = initial();
    v.lost = false;
  }
  function disconnect(v) {
    v.lost = true;
  }

  const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const text = (v) => (typeof v === "string" ? v : "");
  const controllerOf = (v) => (CONTROLLERS.includes(v) ? v : null);
  const current = (s) => s.episodes[s.episodes.length - 1] || null;

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function lastDecision(ep, step) {
    for (let i = ep.rows.length - 1; i >= 0; i--) {
      const row = ep.rows[i];
      if (row.kind === "decision" && row.step === step && row.ok === null && row.skill) return row;
    }
    return null;
  }

  function reduce(s, event) {
    const d = event.data || {};
    s.eventCount += 1;
    s.lastT = event.t;
    s.runId = event.run_id;
    const ep = current(s);
    switch (event.type) {
      case "episode_started":
        s.episodes.push({
          id: text(d.episode_id),
          seed: num(d.seed),
          split: text(d.split),
          round: num(d.round),
          sim: text(d.sim_version) ? { version: d.sim_version, robot: text(d.robot_config) } : null,
          actingOn: text(d.acting_on) || null,
          demo: d.demo === true,
          rows: [],
          frames: [],
          verdict: null,
          perturbedStep: null,
          controller: null,
          checkpoint: null,
          diagnosis: null,
        });
        break;
      case "decision": {
        if (!ep) break;
        if (d.takeover) {
          ep.rows.push({ kind: "takeover", step: num(d.step) });
          ep.controller = "teacher";
          ep.checkpoint = null;
          break;
        }
        const controller = controllerOf(d.controller);
        const latency = num(d.latency_ms);
        const skill = d.skill && typeof d.skill === "object" ? d.skill : null;
        const row = {
          kind: "decision",
          step: num(d.step),
          controller,
          checkpoint: text(d.checkpoint) || null,
          skill: skill ? text(skill.skill) || null : null,
          distance: skill && skill.params ? num(skill.params.distance_cm) : null,
          angle: skill && skill.params ? num(skill.params.angle_deg) : null,
          reason: text(d.reason),
          latency,
          ok: null,
          error: null,
        };
        ep.rows.push(row);
        ep.controller = controller;
        ep.checkpoint = row.checkpoint;
        // Medians time the decisions that chose a skill; a stop carries no measured call.
        if (latency !== null && controller && row.skill) s.latency[controller].push(latency);
        break;
      }
      case "action_executed": {
        if (!ep) break;
        const step = num(d.step);
        const row = lastDecision(ep, step);
        if (row) {
          row.ok = d.ok === true;
          row.error = text(d.error) || null;
        }
        // The recorder stores the observation a decision was made on, so frame N is what decision N saw.
        ep.frames.push({ step, url: text(d.frame) || null, controller: row ? row.controller : null, checkpoint: row ? row.checkpoint : null });
        break;
      }
      case "perturbation":
        if (!ep) break;
        ep.perturbedStep = num(d.step);
        ep.rows.push({ kind: "perturbation", step: num(d.step), what: text(d.kind) });
        break;
      case "validation":
        if (!ep) break;
        ep.verdict = { task: d.task_success === true, physics: d.physics_valid === true, reason: text(d.reason) };
        ep.rows.push({ kind: "verdict" });
        break;
      case "diagnosis": {
        const target = s.episodes.find((e) => e.id === d.episode_id);
        if (target) target.diagnosis = { hypothesis: text(d.hypothesis), evidence: text(d.evidence), lesson: text(d.lesson), confidence: num(d.confidence) };
        break;
      }
      case "evaluation_result":
        if (num(d.successes) !== null && num(d.n) !== null) s.rounds.push({ round: num(d.round), checkpoint: text(d.checkpoint), successes: d.successes, n: d.n });
        break;
      case "finetune_started":
        s.finetune.pending = { round: num(d.round), t: event.t };
        break;
      case "finetune_completed":
        if (s.finetune.pending) s.finetune.durations.push((event.t - s.finetune.pending.t) * 1000);
        s.finetune.pending = null;
        break;
      case "error":
        s.errorCount += 1;
        if (ep) ep.rows.push({ kind: "error", text: text(d.message) || text(d.error) || "error event" });
        break;
      default:
        break;
    }
    return s;
  }

  const isReal = (sim) => sim !== null && !sim.version.startsWith("fake");

  // SPEC §3.5: live while events arrive, or while a fine-tune started under 30 minutes ago has not completed.
  const running = (s, now) =>
    now - s.lastT < FRESH_SECONDS || (s.finetune.pending !== null && now - s.finetune.pending.t < FINETUNE_SECONDS);

  // Rule 6 for what the view shows: placeholder unless every shown episode ran on a real simulator; live while the
  // run is and its stream is connected, paused instead of live while the presenter holds the view, recorded
  // otherwise. Pass an episode for a panel that shows one: only the run's current, unjudged episode can be live.
  function provenance(v, now, ep = null) {
    const s = shown(v);
    const episodes = ep ? [ep] : s.episodes;
    if (!episodes.length || !episodes.every((e) => isReal(e.sim))) return "placeholder";
    if (v.lost || !running(v.live, now) || (ep && (ep !== current(s) || ep.verdict !== null))) return "recorded";
    return v.frozen ? "paused" : "live";
  }

  // Median latency with its sample size; null when there is no data.
  const stat = (values) => (values.length ? { median: median(values), n: values.length } : null);

  function summary(s) {
    return { student: stat(s.latency.student), teacher: stat(s.latency.teacher), finetune: stat(s.finetune.durations) };
  }

  // The head-camera frames to show around the stage frame (`now`): the latest, or the first at or after `focus`.
  // The first frame after the box or bin moved is flagged.
  function filmstrip(ep, focus = null) {
    if (!ep || !ep.frames.length) return [];
    const frames = ep.frames;
    const moved = ep.perturbedStep === null ? null : frames.find((f) => f.step >= ep.perturbedStep);
    const stage = (focus !== null && frames.find((f) => f.step >= focus)) || frames[frames.length - 1];
    const at = frames.indexOf(stage);
    const start = Math.max(0, Math.min(at - (FILMSTRIP >> 1), frames.length - FILMSTRIP));
    return frames.slice(start, start + FILMSTRIP).map((f) => ({ ...f, moved: f === moved, now: f === stage }));
  }

  // The shown episode's box or bin move: the last frame before it, the first frame that shows it, and the first
  // decision made after it. Null for an episode without one.
  function move(ep) {
    if (!ep || ep.perturbedStep === null) return null;
    const at = ep.perturbedStep;
    return {
      step: at,
      before: ep.frames.findLast((f) => f.step < at) || null,
      after: ep.frames.find((f) => f.step >= at) || null,
      decision: ep.rows.find((r) => r.kind === "decision" && r.step >= at) || null,
    };
  }

  // The diagnosis a teacher demonstration acts on (SPEC §3.5): the diagnosed episode's id, its number in the run,
  // its diagnosis and the last frame it saw. Null for an episode that acts on none.
  function actedOn(s, ep) {
    if (!ep || !ep.actingOn) return null;
    const at = s.episodes.findIndex((e) => e.id === ep.actingOn);
    const target = s.episodes[at];
    return {
      id: ep.actingOn,
      number: target ? at + 1 : null,
      diagnosis: target ? target.diagnosis : null,
      frame: target ? target.frames[target.frames.length - 1] || null : null,
    };
  }

  // One event from the stream, arriving at `now`.
  function receive(v, event, now) {
    reduce(v.live, event);
    if (running(v.live, now)) v.followed = true;
  }

  // Presenter controls (SPEC §3.5). Events keep folding into `live` while paused; resuming shows them all.
  const shown = (v) => (v.frozen ? v.frozen.state : v.live);
  function togglePause(v, now) {
    v.frozen = v.frozen ? null : { state: structuredClone(v.live), at: now };
  }
  // Events that arrived since the pause; a reconnect's replay starts from none.
  const buffered = (v) => (v.frozen ? Math.max(0, v.live.eventCount - v.frozen.state.eventCount) : 0);
  // A paused view judges the run as it was when paused, so it stays put if the run ends meanwhile.
  const following = (v, now) => running(v.live, v.frozen ? v.frozen.at : now);

  const accepted = (ep) => ep.verdict !== null && ep.verdict.task && ep.verdict.physics;
  // A finished run's story (SPEC §3.5): its latest accepted teacher demonstration whose diagnosis the run holds.
  const story = (s) => s.episodes.findLast((e) => accepted(e) && actedOn(s, e)?.diagnosis) || current(s);

  // A view that followed the run live stays on its last episode once it stops; a page opened on a finished run
  // lands on the story.
  function episode(v, now) {
    const s = shown(v);
    const pinned = v.selected !== null && s.episodes.find((e) => e.id === v.selected);
    return pinned || (following(v, now) || v.followed ? current(s) : story(s));
  }

  // The fine-tune a following view shows on the stage (SPEC §3.5): the seconds since it started, or null.
  function finetuning(v, now) {
    const pending = shown(v).finetune.pending;
    if (!pending || v.selected !== null || !following(v, now)) return null;
    return (v.frozen ? v.frozen.at : now) - pending.t;
  }

  const teacherDrove = (ep) => ep.rows.some((r) => r.kind === "takeover" || r.controller === "teacher");
  // The story beats D, T and M jump to (SPEC §3.5).
  const BEATS = { d: (e) => e.diagnosis !== null, t: teacherDrove, m: (e) => e.perturbedStep !== null };

  // Where a navigation key leads from the shown episode: an index into the shown episodes, or -1 for nowhere new.
  // A story beat goes to the nearest other episode that has it, forward on a tie.
  function target(v, now, key) {
    const episodes = shown(v).episodes;
    const at = episodes.indexOf(episode(v, now));
    const last = episodes.length - 1;
    if (at < 0) return -1;
    if (key === "ArrowLeft") return at > 0 ? at - 1 : -1;
    if (key === "ArrowRight") return at < last ? at + 1 : -1;
    if (key === "End") return at < last || v.compare || v.step !== null ? last : -1;
    for (let d = 1; d <= last; d++) for (const i of [at + d, at - d]) if (episodes[i] && BEATS[key](episodes[i])) return i;
    return -1;
  }

  // Landing on a live run's last episode follows the run again; M holds its episode at the move instead.
  function go(v, now, key) {
    const i = target(v, now, key);
    if (i < 0) return;
    const episodes = shown(v).episodes;
    v.selected = key !== "m" && i === episodes.length - 1 && following(v, now) ? null : episodes[i].id;
    v.step = key === "m" ? episodes[i].perturbedStep : null;
    v.compare = null;
  }

  // The step the stage holds the shown episode at: its move under P or after M, null for its latest frame.
  const stageStep = (v) => (v.compare ? v.compare.step : v.step);

  // Shows the shown episode's box or bin move; pressing again returns to the selection from before.
  function compareMove(v, now) {
    if (v.compare) {
      v.selected = v.compare.back;
      v.compare = null;
      return;
    }
    const ep = episode(v, now);
    if (!ep || ep.perturbedStep === null) return;
    v.compare = { step: ep.perturbedStep, back: v.selected };
    v.selected = ep.id;
  }

  const KEYS = { " ": togglePause, p: compareMove };
  for (const key of ["ArrowLeft", "ArrowRight", "End", ...Object.keys(BEATS)]) KEYS[key] = (v, now) => go(v, now, key);
  // Presenter keys (SPEC §3.5), whatever has focus. True when the key is the dashboard's, so the page claims it
  // (a focused button must not also take Space as a click); a held key acts once.
  function keydown(v, e, now) {
    const action = KEYS[e.key.length === 1 ? e.key.toLowerCase() : e.key];
    if (!action || e.metaKey || e.ctrlKey || e.altKey) return false;
    if (!e.repeat) action(v, now);
    return true;
  }

  const api = { initial, view, reconnect, disconnect, reduce, receive, current, provenance, summary, filmstrip, move, actedOn, median, shown, buffered, following, accepted, teacherDrove, episode, finetuning, target, stageStep, keydown };
  if (typeof module !== "undefined") module.exports = api;
  else root.MomState = api;
})(this);
