// Folds SPEC §4.4 events into the dashboard's state. Pure: no DOM, so node can test it.
(function (root) {
  const CONTROLLERS = ["student", "teacher"];
  const FILMSTRIP = 8;
  const FRESH_SECONDS = 20;

  function initial() {
    return {
      runId: null,
      lastT: 0,
      eventCount: 0,
      episodes: [],
      rounds: [],
      latency: { student: [], teacher: [] },
      argon: [],
      finetune: { pending: null, durations: [] },
      errorCount: 0,
    };
  }

  // The page's view of the stream. Every connection replays events.jsonl from the start, so a reconnect starts over.
  const view = () => ({ live: initial() });
  function reconnect(v) {
    v.live = initial();
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
        ep.rows.push({
          kind: "decision",
          step: num(d.step),
          controller,
          checkpoint: text(d.checkpoint) || null,
          skill: skill ? text(skill.skill) || null : null,
          params: skill && skill.params ? skill.params : {},
          reason: text(d.reason),
          latency,
          ok: null,
          error: null,
        });
        ep.controller = controller;
        ep.checkpoint = text(d.checkpoint) || null;
        if (latency !== null && controller) s.latency[controller].push(latency);
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
      case "argon_call": {
        const latency = num(d.latency_ms);
        if (latency !== null) s.argon.push(latency);
        break;
      }
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

  // Rule 6: placeholder unless every episode ran on a real simulator; then live while events arrive or a
  // fine-tune is pending. Pass an episode for a panel that shows one: only the current episode can be live.
  function provenance(s, now, ep = null) {
    const episodes = ep ? [ep] : s.episodes;
    if (!episodes.length || !episodes.every((e) => isReal(e.sim))) return "placeholder";
    const live = now - s.lastT < FRESH_SECONDS || s.finetune.pending !== null;
    return live && (!ep || ep === current(s)) ? "live" : "recorded";
  }

  // Median latency with its sample size; null when there is no data.
  const stat = (values) => (values.length ? { median: median(values), n: values.length } : null);

  function summary(s) {
    return { student: stat(s.latency.student), teacher: stat(s.latency.teacher), argon: stat(s.argon), finetune: stat(s.finetune.durations) };
  }

  // The head-camera frames to show: the last few of the episode, with the first one after the block moved flagged.
  function filmstrip(ep) {
    if (!ep) return [];
    const shown = ep.frames.slice(-FILMSTRIP);
    const moved = ep.perturbedStep === null ? null : ep.frames.find((f) => f.step >= ep.perturbedStep);
    return shown.map((f) => ({ ...f, moved: f === moved }));
  }

  const api = { initial, view, reconnect, reduce, current, provenance, summary, filmstrip, median };
  if (typeof module !== "undefined") module.exports = api;
  else root.MomState = api;
})(this);
