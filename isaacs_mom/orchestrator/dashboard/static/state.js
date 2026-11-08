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
    s.lastT = event.t;
    s.runId = event.run_id;
    // A heartbeat only says the run is alive (SPEC §3.5): no row, not counted, so it never moves the view.
    if (event.type === "heartbeat") return s;
    s.eventCount += 1;
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
          beforeFrame: null,
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
        // The orchestrator's frame from right before the move, at the same pose; a run recorded before it took one has none.
        ep.beforeFrame = { step: ep.perturbedStep, url: text(d.before_frame) || null, controller: ep.controller, checkpoint: ep.checkpoint };
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
        // `after` is where the next evaluation's held-out episodes start.
        if (num(d.round) !== null && num(d.successes) !== null && num(d.n) !== null) s.rounds.push({ round: d.round, checkpoint: text(d.checkpoint), successes: d.successes, n: d.n, after: s.episodes.length });
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

  // The run's simulator in words for the header, and its full version and task (a MuJoCo version's `task=` part) for
  // the legend's run line. A run with any FakeSim or unnamed episode is named by that. Null before any episode.
  function simulator(s) {
    const sims = s.episodes.map((e) => e.sim);
    if (!sims.length) return null;
    if (sims.includes(null)) return { name: "simulator not stated", version: null, task: null };
    const sim = sims.find((x) => !isReal(x)) || sims[sims.length - 1];
    const task = /(?:^|\s)task=(\S+)/.exec(sim.version);
    const name = !isReal(sim) ? "FakeSim" : sim.version.startsWith("mujoco") ? "MuJoCo" : sim.version;
    return { name, version: sim.version, task: task ? task[1] : null };
  }

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
  // The first frame after the box or bin moved is flagged `moved`, it and every later one `after`.
  function filmstrip(ep, focus = null) {
    if (!ep || !ep.frames.length) return [];
    const frames = ep.frames;
    const moved = ep.perturbedStep === null ? null : frames.find((f) => f.step >= ep.perturbedStep);
    const stage = (focus !== null && frames.find((f) => f.step >= focus)) || frames[frames.length - 1];
    const at = frames.indexOf(stage);
    const start = Math.max(0, Math.min(at - (FILMSTRIP >> 1), frames.length - FILMSTRIP));
    const after = (f) => ep.perturbedStep !== null && f.step !== null && f.step >= ep.perturbedStep;
    return frames.slice(start, start + FILMSTRIP).map((f) => ({ ...f, moved: f === moved, after: after(f), now: f === stage }));
  }

  // The shown episode's box or bin move: the frame taken right before it, the first frame after it, and the first
  // decision made after it. A move before any decision is under the controller of the first frame after it. Null for an
  // episode without one.
  function move(ep) {
    if (!ep || ep.perturbedStep === null) return null;
    const at = ep.perturbedStep;
    const after = ep.frames.find((f) => f.step >= at) || null;
    const before = ep.beforeFrame.controller || !after ? ep.beforeFrame : { ...ep.beforeFrame, controller: after.controller, checkpoint: after.checkpoint };
    return { step: at, before, after, decision: ep.rows.find((r) => r.kind === "decision" && r.step >= at) || null };
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

  // One event from the stream, arriving at `now`. An event that arrives fresh means the view saw the run live; a
  // replayed one does not, even inside a replayed fine-tune.
  function receive(v, event, now) {
    reduce(v.live, event);
    if (now - event.t < FRESH_SECONDS) v.followed = true;
  }

  // Presenter controls (SPEC §3.5). Events keep folding into `live` while paused; resuming shows them all.
  const shown = (v) => (v.frozen ? v.frozen.state : v.live);
  function togglePause(v, now) {
    v.frozen = v.frozen ? null : { state: structuredClone(v.live), at: now };
  }
  // Events that arrived since the pause; a reconnect's replay starts from none.
  const buffered = (v) => (v.frozen ? Math.max(0, v.live.eventCount - v.frozen.state.eventCount) : 0);
  // The pause key's label, which is also its accessible name (WCAG 2.5.3): what Space does next.
  const pauseLabel = (v) => (v.frozen ? `Resume · ${buffered(v)} new` : "Pause");
  // A paused view judges the run as it was when paused, so it stays put if the run ends meanwhile.
  const following = (v, now) => running(v.live, v.frozen ? v.frozen.at : now);

  const accepted = (ep) => ep.verdict !== null && ep.verdict.task && ep.verdict.physics;
  // What an episode is: the judged take, a held-out test, a teacher demonstration or a student trial. All but a trial
  // are known from the episode's start; a development episode acting on no diagnosis is null until its first decision.
  function kind(ep) {
    if (ep.demo) return "take";
    if (ep.split === "heldout") return "heldout";
    if (ep.actingOn) return "demonstration";
    const first = ep.rows.find((r) => r.kind === "decision");
    return !first ? null : first.controller === "teacher" ? "demonstration" : "trial";
  }
  // The verdict in the words of the episode's kind (SPEC §3.5): only a demonstration is admitted to training; the
  // rest succeed or fail, and the judged take says whether the student finished it alone. Null before the verdict.
  function verdictWords(ep) {
    if (!ep.verdict) return null;
    const good = accepted(ep);
    const k = kind(ep);
    if (k === "demonstration") return good ? "Accepted for training" : "Rejected";
    const outcome = good ? "Succeeded" : "Failed";
    if (k !== "take") return outcome;
    const took = ep.rows.find((r) => r.kind === "takeover");
    if (!took) return good ? `${outcome}, by the student alone` : outcome;
    return `${outcome} after the teacher took over${took.step === null ? "" : ` at step ${took.step}`}`;
  }
  // The held-out claim (SPEC §3.5): the first and the latest evaluation, each with its n, named in words, with the
  // checkpoint ids as a note. Null before the first evaluation.
  function claim(s) {
    if (!s.rounds.length) return null;
    const first = s.rounds[0];
    const last = s.rounds[s.rounds.length - 1];
    const pair = first === last ? [first] : [first, last];
    const name = (r) => (r.round === 0 ? "before fine-tuning" : `after round ${r.round}`);
    const score = (r, of) => (r.n ? `${r.successes}${of}${r.n}` : "none run");
    const same = pair.every((r) => r.n === first.n) && first.n;
    return {
      figure: pair.map((r) => score(r, "/")).join(" → "),
      spoken: pair.map((r) => `${score(r, " of ")} ${name(r)}`).join(", then "),
      sample: same ? `n=${first.n}${pair.length > 1 ? " each" : ""}` : pair.map((r) => (r.n ? `n=${r.n}` : "no held-out episode ran")).join(" and "),
      names: pair.map(name).join(" → "),
      ids: pair.map((r) => r.checkpoint).join(" → "),
    };
  }

  // The held-out evaluation under way: its checkpoint, how many of its episodes have a verdict, and out of how many,
  // which only an earlier evaluation of the same sealed seeds tells. Null while none runs.
  function testing(s) {
    const last = s.rounds[s.rounds.length - 1];
    const tests = s.episodes.slice(last ? last.after : 0).filter((e) => e.split === "heldout");
    if (!tests.length) return null;
    const decided = tests.flatMap((e) => e.rows).findLast((r) => r.kind === "decision" && r.checkpoint);
    return { checkpoint: decided ? decided.checkpoint : null, run: tests.filter((e) => e.verdict).length, of: last ? last.n : null };
  }

  // What a fine-tune learns from (SPEC §3.5): the last frame of each accepted teacher demonstration so far, with its
  // number in the run, and how many demonstrations were judged and how. One with no frame is counted, not shown.
  function lessons(s) {
    const judged = s.episodes.filter((e) => kind(e) === "demonstration" && e.verdict);
    const kept = judged.filter(accepted);
    return {
      frames: kept.filter((e) => e.frames.length).map((e) => ({ url: e.frames[e.frames.length - 1].url, number: s.episodes.indexOf(e) + 1 })),
      accepted: kept.length,
      rejected: judged.length - kept.length,
      n: judged.length,
    };
  }

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

  // Who the control bar names: the controller of the frame the stage holds after M or P, else the episode's latest.
  function inControl(v, now) {
    const ep = episode(v, now);
    if (!ep) return null;
    const step = stageStep(v);
    const frame = step === null ? null : filmstrip(ep, step).find((f) => f.now);
    return frame ? { controller: frame.controller, checkpoint: frame.checkpoint, step: frame.step } : { controller: ep.controller, checkpoint: ep.checkpoint, step: null };
  }

  // An episode is over once judged, once the run has moved past it, or once the run has stopped.
  const finished = (v, now, ep) => ep.verdict !== null || ep !== current(shown(v)) || !following(v, now);

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

  const api = { initial, view, reconnect, disconnect, reduce, receive, current, provenance, summary, filmstrip, move, actedOn, median, shown, buffered, pauseLabel, following, accepted, kind, verdictWords, simulator, claim, testing, lessons, teacherDrove, episode, finetuning, target, stageStep, inControl, finished, keydown };
  if (typeof module !== "undefined") module.exports = api;
  else root.MomState = api;
})(this);
