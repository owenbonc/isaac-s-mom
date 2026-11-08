// Folds SPEC §4.4 events into the dashboard's state. Pure: no DOM, so node can test it.
(function (root) {
  const CONTROLLERS = ["student", "teacher"];
  const FILMSTRIP = 6;
  const FRESH_SECONDS = 20;
  const FINETUNE_SECONDS = 30 * 60;

  function initial() {
    return {
      runId: null,
      // The run's instruction (its first `instruction` event), and why a Teach did not start it, if it did not.
      instruction: null,
      refused: null,
      lastT: 0,
      eventCount: 0,
      episodes: [],
      rounds: [],
      latency: { student: [], teacher: [] },
      finetune: { pending: null, durations: [] },
      // Rounds whose training set was built (`finetune_started`), each with a summary file to read.
      trained: [],
      argon: { used: 0, budget: null, answers: [], failed: 0 },
      errorCount: 0,
      // Held-out episodes that crashed before a verdict, by seed and the evaluation they belong to (its index).
      crashes: [],
      // Set by the error event the dashboard leaves when the `mom run` it launched exits non-zero (SPEC §3.5).
      stopped: false,
    };
  }

  // The page's view of the stream. `live` folds every event; `frozen` is what a paused view keeps showing and when
  // it was paused; `selected` pins Teaching's episode (null follows the run: its latest teaching episode while live, its
  // story once finished); `compare` is the box or bin move that Move shows, with the view to return to when pressed again;
  // `followed` is set once an event arrives while the run is live; `lost` while the event stream is down; `settling`
  // while a connection's replay of the events already written streams in, which the page neither shows nor announces.
  // `tab` is Results, where a run opens, Teaching or Now; in Results, `seed` is the seed open, `opened` its episode
  // open, `focus` the seed focused in the list and `only` the mark it is filtered to.
  const view = () => ({ live: initial(), frozen: null, selected: null, compare: null, followed: false, lost: false, settling: true, tab: "results", seed: null, opened: null, focus: null, only: null });
  // Every connection replays events.jsonl from the start, so a reconnect starts over.
  function reconnect(v) {
    v.live = initial();
    v.lost = false;
    v.settling = true;
  }
  const settle = (v) => {
    v.settling = false;
  };
  // A lost stream shows that it is lost, even before a replay settles.
  function disconnect(v) {
    v.lost = true;
    v.settling = false;
  }

  const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const text = (v) => (typeof v === "string" ? v : "");
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
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

  // Where a run started with `mom run --from-run` began (SPEC §3.6): the earlier run, its checkpoint and the accepted
  // demonstrations carried over; null for anything else, so a malformed one reads as a run from the base model.
  function fromRun(f) {
    if (!f || typeof f !== "object" || !text(f.run) || !text(f.checkpoint) || !Number.isInteger(f.carried) || f.carried < 0) return null;
    return { run: f.run, checkpoint: f.checkpoint, carried: f.carried };
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
        // `after` is where the next evaluation's held-out episodes start; `from` is where a run's round 0 came from.
        if ([d.round, d.successes, d.n, d.errored].every((v) => num(v) !== null))
          s.rounds.push({ round: d.round, checkpoint: text(d.checkpoint), successes: d.successes, n: d.n, errored: d.errored, after: s.episodes.length, from: d.round === 0 ? fromRun(d.from_run) : null });
        break;
      case "finetune_started":
        s.finetune.pending = { round: num(d.round), t: event.t };
        if (num(d.round) !== null && !s.trained.includes(d.round)) s.trained.push(d.round);
        break;
      case "finetune_completed":
        if (s.finetune.pending) s.finetune.durations.push((event.t - s.finetune.pending.t) * 1000);
        s.finetune.pending = null;
        break;
      case "argon_call": {
        // `call` counts on across the run's processes (rule 5), so the highest is how many were used.
        const a = s.argon;
        a.used = Math.max(a.used, num(d.call) ?? 0);
        a.budget = num(d.budget) ?? a.budget;
        if (d.ok !== true) a.failed += 1;
        else if (num(d.latency_ms) !== null) a.answers.push(d.latency_ms);
        break;
      }
      case "instruction":
        if (!s.instruction && text(d.text)) s.instruction = { text: d.text, source: text(d.source) };
        break;
      case "error": {
        s.errorCount += 1;
        const said = text(d.message) || text(d.error) || "error event";
        const refusal = /^not started:\s*/.exec(said);
        if (refusal || said.startsWith("mom run stopped")) s.stopped = true;
        if (refusal && !ep) s.refused = said.slice(refusal[0].length);
        else if (ep) ep.rows.push({ kind: "error", text: said });
        if (d.split === "heldout" && num(d.seed) !== null) s.crashes.push({ seed: d.seed, evaluation: s.rounds.length });
        break;
      }
      default:
        break;
    }
    return s;
  }

  const isReal = (sim) => sim !== null && !sim.version.startsWith("fake");

  // The run's simulator in words for the header, with its full version and task (`task=…`) for the legend; a run with
  // any FakeSim or unnamed episode is named by that, an unnamed one not at all. Null before any episode.
  function simulator(s) {
    const sims = s.episodes.map((e) => e.sim);
    if (!sims.length) return null;
    if (sims.includes(null)) return { name: null, version: null, task: null };
    const sim = sims.find((x) => !isReal(x)) || sims[sims.length - 1];
    const task = /(?:^|\s)task=(\S+)/.exec(sim.version);
    // Any other simulator is named by its version's leading word, so a long version never wraps the header.
    const name = !isReal(sim) ? "FakeSim" : sim.version.startsWith("mujoco") ? "MuJoCo" : sim.version.split(/\s/)[0].replace(/-\d.*$/, "");
    return { name, version: sim.version, task: task ? task[1] : null };
  }

  // SPEC §3.5: live while events arrive, or while a fine-tune started under 30 minutes ago has not completed, and never
  // once the run has stopped, as the runs list (orchestrator/runs.py) reads it.
  const running = (s, now) =>
    !s.stopped && (now - s.lastT < FRESH_SECONDS || (s.finetune.pending !== null && now - s.finetune.pending.t < FINETUNE_SECONDS));

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

  // A panel's provenance (SPEC §3.5): its dot, with the word spelled out only where it differs from the run's label.
  function panelLabel(v, now, ep = null) {
    const kind = provenance(v, now, ep);
    return { kind, spelled: kind !== provenance(v, now) };
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
    if (k === "demonstration") return good ? "Accepted for training" : "Rejected, not used for training";
    const outcome = good ? "Succeeded" : "Failed";
    if (k !== "take") return outcome;
    const took = ep.rows.find((r) => r.kind === "takeover");
    if (!took) return good ? `${outcome}, by the student alone` : outcome;
    return `${outcome} after the teacher took over${took.step === null ? "" : ` at step ${took.step}`}`;
  }
  // A checkpoint by the evaluation that tested it, in words: round 0 from an earlier run names that run and checkpoint;
  // no-break spaces keep each other name on one line when a pair wraps.
  function roundName(r) {
    if (r.from) return `from ${r.from.run}, ${r.from.checkpoint}`;
    return r.round === 0 ? "before\u00a0fine-tuning" : r.round === null ? "the\u00a0evaluation\u00a0under\u00a0way" : `after\u00a0round\u00a0${r.round}`;
  }
  // The base model's checkpoint (student/checkpoints.py): a round 0 tested on another came from an earlier run.
  const BASE_CHECKPOINT = "ckpt-r0";

  // The Results marks (SPEC §3.5) in their sort order: the changes, the unchanged, then errored and pending.
  const MARKS = ["down", "up", "fail", "pass", "error", "pending"];
  const RECORDING_ERROR = "recording error";
  // A held-out episode's outcome: errored on a recording error, a crash, or no verdict once its evaluation finished.
  function outcome(cell, done) {
    if (cell.crashed) return "errored";
    if (!cell.ep || !cell.ep.verdict) return done ? "errored" : "pending";
    if (cell.ep.verdict.reason.startsWith(RECORDING_ERROR)) return "errored";
    return accepted(cell.ep) ? "succeeded" : "failed";
  }
  // The first evaluation against the latest; one that errored in either is counted apart.
  function mark(outcomes) {
    const first = outcomes[0];
    const latest = outcomes[outcomes.length - 1];
    if (first === "errored" || latest === "errored") return "error";
    if (outcomes.length < 2 || latest === "pending") return "pending";
    if (first === "succeeded") return latest === "succeeded" ? "pass" : "down";
    return latest === "succeeded" ? "up" : "fail";
  }
  // Results (SPEC §3.5): each evaluation as a checkpoint, the one under way last; each held-out seed's episode and
  // outcome per checkpoint and its mark, sorted by mark then seed, kept to the marks `only` names; and the count per mark.
  function results(s, only = null) {
    // An evaluation's held-out episodes follow the previous evaluation's result, as testing() reads them.
    const ends = s.rounds.map((r) => r.after);
    const tail = s.episodes.slice(ends.length ? ends[ends.length - 1] : 0).find((e) => e.split === "heldout");
    const checkpoints = s.rounds.map((r) => ({ round: r.round, checkpoint: r.checkpoint, name: roundName(r), done: true }));
    if (tail || s.crashes.some((c) => c.evaluation === ends.length)) {
      const test = testing(s);
      const checkpoint = test ? test.checkpoint : null;
      // Until its result says where it came from, a round 0 on another checkpoint than the base is only under way.
      const round = tail && !(tail.round === 0 && checkpoint !== null && checkpoint !== BASE_CHECKPOINT) ? tail.round : null;
      checkpoints.push({ round, checkpoint, name: roundName({ round, from: null }), done: false });
    }
    const rows = new Map();
    const cell = (seed, k) => {
      if (!rows.has(seed)) rows.set(seed, checkpoints.map(() => ({ ep: null, crashed: false })));
      return rows.get(seed)[k];
    };
    s.episodes.forEach((ep, i) => {
      if (ep.split !== "heldout" || ep.seed === null) return;
      const k = ends.findIndex((end) => i < end);
      const at = cell(ep.seed, k < 0 ? ends.length : k);
      at.ep ??= ep; // a seed's first try in an evaluation is its episode there
    });
    for (const c of s.crashes) cell(c.seed, c.evaluation).crashed = true;
    const seeds = [...rows]
      .map(([seed, cells]) => {
        const episodes = cells.map((c, k) => ({ id: c.ep ? c.ep.id : null, outcome: outcome(c, checkpoints[k].done) }));
        return { seed, mark: mark(episodes.map((e) => e.outcome)), episodes };
      })
      .sort((a, b) => MARKS.indexOf(a.mark) - MARKS.indexOf(b.mark) || a.seed - b.seed);
    const counts = Object.fromEntries(MARKS.map((m) => [m, seeds.filter((r) => r.mark === m).length]));
    return { checkpoints, seeds: only === null ? seeds : seeds.filter((r) => r.mark === only), counts };
  }

  // The held-out claim (SPEC §3.5): the change in percentage points between the first and the latest evaluation, the
  // two counts of successes out of the episodes that didn't error, the errored counts when any errored, and the
  // evaluations named in words with their checkpoint ids. Null before the first evaluation.
  function claim(s) {
    if (!s.rounds.length) return null;
    const first = s.rounds[0];
    const last = s.rounds[s.rounds.length - 1];
    const pair = first === last ? [first] : [first, last];
    const name = roundName;
    const scored = (r) => r.n - r.errored;
    // Whole percentages of the scored episodes; the change is taken between the rates as shown, so the figures agree.
    const rate = (r) => (scored(r) > 0 ? Math.round((100 * r.successes) / scored(r)) : null);
    const rates = pair.map(rate);
    const points = pair.length > 1 && !rates.includes(null) ? rates[1] - rates[0] : null;
    const said = (r) =>
      !r.n
        ? `no held-out episode ran ${name(r)}`
        : scored(r) <= 0
          ? `all ${r.n} held-out episodes errored ${name(r)}`
          : `${rate(r)}%, ${r.successes} of ${scored(r)}, ${name(r)}${r.errored ? `, ${r.errored} errored` : ""}`;
    const change = points === null ? "" : points === 0 ? "No change: " : `${points > 0 ? "Up" : "Down"} ${Math.abs(points)} percentage points: `;
    // One denominator is said once ("4 → 15 of 20"); two that differ go with their counts ("9/18 → 15/20").
    const of = scored(first);
    const counts =
      of > 0 && pair.every((r) => scored(r) === of)
        ? `${pair.map((r) => r.successes).join(" → ")} of ${of}`
        : pair.map((r) => (scored(r) > 0 ? `${r.successes}/${scored(r)}` : "none")).join(" → ");
    return {
      // The change in points, never a relative percentage; with one evaluation, its rate; with nothing scored, none.
      figure: rates.includes(null) ? null : points === null ? `${rates[0]}%` : `${points < 0 ? "\u2212" : "+"}${Math.abs(points)} pts`,
      counts,
      spoken: change + pair.map(said).join(", then "),
      errored: pair.some((r) => r.errored) ? `${pair.map((r) => r.errored).join(" → ")} errored` : null,
      names: pair.map(name).join(" → "),
      ids: pair.map((r) => r.checkpoint).join(" → "),
    };
  }

  // The Argon strip (SPEC §3.5, rule 5): calls used of the run's budget, the median answer with its n, and the calls
  // that failed. Null before the first call.
  function argon(s) {
    const a = s.argon;
    return a.used ? { used: a.used, budget: a.budget, answer: stat(a.answers), failed: a.failed } : null;
  }

  // Saved run files (SPEC §3.5), read only in the shape of their contract models; null for anything else.
  const count = (v) => Number.isInteger(v) && v >= 0;
  const SUMMARY = ["round", "episodes_scanned", "demonstrations", "accepted", "rejected", "unfinished", "empty_dev_episodes", "examples"];
  function trainingSummary(f) {
    if (!f || !SUMMARY.every((k) => count(f[k])) || !f.rejected_by || typeof f.rejected_by !== "object") return null;
    const by = Object.entries(f.rejected_by);
    if (!by.every(([, v]) => count(v))) return null;
    return { ...Object.fromEntries(SUMMARY.map((k) => [k, f[k]])), rejected_by: Object.fromEntries(by) };
  }
  function replayReport(f) {
    if (!f || !count(f.replayed) || !count(f.reproduced) || !Array.isArray(f.failures)) return null;
    const failures = f.failures.map((x) => x && typeof x.episode_id === "string" && typeof x.reason === "string" && { episode_id: x.episode_id, reason: x.reason });
    return failures.every(Boolean) ? { replayed: f.replayed, reproduced: f.reproduced, failures } : null;
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
  // number in the run, how many demonstrations were judged and how, and those carried from an earlier run (null for a
  // run from the base model). One with no frame is counted, not shown.
  function lessons(s) {
    const judged = s.episodes.filter((e) => kind(e) === "demonstration" && e.verdict);
    const kept = judged.filter(accepted);
    const from = s.rounds.length ? s.rounds[0].from : null;
    return {
      frames: kept.filter((e) => e.frames.length).map((e) => ({ url: e.frames[e.frames.length - 1].url, number: s.episodes.indexOf(e) + 1 })),
      accepted: kept.length,
      rejected: judged.length - kept.length,
      n: judged.length,
      carried: from && { count: from.carried, run: from.run },
    };
  }

  // Teaching holds every episode but the held-out tests (SPEC §3.5).
  const teaches = (e) => e.split !== "heldout";
  // A finished run's story (SPEC §3.5): its latest accepted teacher demonstration whose diagnosis the run holds.
  const story = (s) => s.episodes.findLast((e) => accepted(e) && actedOn(s, e)?.diagnosis) || s.episodes.findLast(teaches) || null;

  // Where the view is (SPEC §3.5): in Results, the seed list, a seed's checkpoints or one of its episodes; Teaching;
  // or Now, the episode in progress.
  const where = (v) => (v.tab !== "results" ? v.tab : v.opened !== null ? "episode" : v.seed !== null ? "seed" : "list");

  // The episode the view shows, or null. Teaching follows its latest episode while the view follows the run, and stays
  // there once the run stops; a page opened on a finished run lands on the story.
  function episode(v, now) {
    const s = shown(v);
    if (v.tab === "now") return current(s);
    const find = (id) => (id !== null && s.episodes.find((e) => e.id === id)) || null;
    if (v.tab === "results") return find(v.opened);
    return find(v.selected) || (following(v, now) || v.followed ? s.episodes.findLast(teaches) || null : story(s));
  }

  // The fine-tune a view following the run shows on the stage (SPEC §3.5): the seconds since it started, or null.
  function finetuning(v, now) {
    const pending = shown(v).finetune.pending;
    const follows = v.tab === "now" || (v.tab === "teaching" && v.selected === null);
    if (!pending || !follows || !following(v, now)) return null;
    return (v.frozen ? v.frozen.at : now) - pending.t;
  }

  const teacherDrove = (ep) => ep.rows.some((r) => r.kind === "takeover" || r.controller === "teacher");
  // The story beats D, T and Move jump to (SPEC §3.5): D and T within Teaching, Move to any episode.
  const BEATS = { d: (e) => e.diagnosis !== null && teaches(e), t: (e) => teacherDrove(e) && teaches(e), m: (e) => e.perturbedStep !== null };

  // Where an episode lives: a held-out test in its seed under Results, any other in Teaching.
  const home = (ep) => (ep.split === "heldout" ? { tab: "results", seed: ep.seed, focus: ep.seed, opened: ep.id } : { tab: "teaching", selected: ep.id });
  // Landing on a live run's latest teaching episode follows the run again.
  function land(v, now, ep) {
    const to = home(ep);
    if (to.tab === "teaching" && ep === shown(v).episodes.findLast(teaches) && following(v, now)) to.selected = null;
    return to;
  }
  // The seed `step` places along Results' list under its filter from the open or focused seed (the first from
  // none), or null past either end.
  function seedFrom(v, step) {
    const seeds = results(shown(v), v.only).seeds.map((r) => r.seed);
    const at = seeds.indexOf(v.seed ?? v.focus);
    const to = at < 0 ? 0 : at + step;
    return to >= 0 && to < seeds.length ? seeds[to] : null;
  }

  // Where a key leads from the view, as the view fields it sets, or null for nowhere new (SPEC §3.5): ← and → step
  // through the seeds in Results (↑ and ↓ too in the list), through a seed's episodes, and through Teaching; Escape
  // goes back up from a seed; End opens Now while the run is live; a story beat goes to the nearest other episode
  // with it, forward on a tie, and Move stays on a shown episode that has one.
  function target(v, now, key) {
    const s = shown(v);
    const place = where(v);
    const shownEp = episode(v, now);
    const step = key === "ArrowLeft" || key === "ArrowUp" ? -1 : 1;
    if (key === "Escape") return place === "episode" ? { opened: null } : place === "seed" ? { seed: null, focus: v.seed } : null;
    if (key === "End") return following(v, now) && place !== "now" ? { tab: "now" } : null;
    if (key === "ArrowUp" || key === "ArrowDown" || key === "ArrowLeft" || key === "ArrowRight") {
      const across = key === "ArrowLeft" || key === "ArrowRight";
      if (place === "list" || (place === "seed" && across)) {
        const seed = seedFrom(v, step);
        return seed === null ? null : place === "list" ? { focus: seed } : { seed, focus: seed };
      }
      if (!across) return null;
      if (place === "episode") {
        const row = results(s).seeds.find((r) => r.seed === v.seed);
        const ids = row ? row.episodes.map((e) => e.id) : [];
        for (let i = ids.indexOf(v.opened) + step; i >= 0 && i < ids.length; i += step) if (ids[i] !== null) return { opened: ids[i] };
        return null;
      }
      if (place !== "teaching" || !shownEp) return null;
      for (let i = s.episodes.indexOf(shownEp) + step; i >= 0 && i < s.episodes.length; i += step) if (teaches(s.episodes[i])) return land(v, now, s.episodes[i]);
      return null;
    }
    if (key === "m" && shownEp && BEATS.m(shownEp)) return home(shownEp);
    // With no episode shown, the nearest is the latest.
    const at = shownEp ? s.episodes.indexOf(shownEp) : s.episodes.length;
    for (let d = 1; d <= s.episodes.length; d++)
      for (const i of [at + d, at - d]) if (s.episodes[i] && BEATS[key](s.episodes[i])) return key === "m" ? home(s.episodes[i]) : land(v, now, s.episodes[i]);
    return null;
  }

  // Sets the view's fields: a key's target, a tab, a seed or an episode chosen, a filter. A Move shown is let go.
  function visit(v, fields) {
    Object.assign(v, fields);
    v.compare = null;
  }
  function go(v, now, key) {
    const to = target(v, now, key);
    if (to) visit(v, to);
  }

  // Move (SPEC §3.5): the before and after of the shown episode's move, or of the nearest episode with one, in that
  // episode's tab; pressing it again returns to where the view was.
  function showMove(v, now) {
    if (v.compare) {
      visit(v, v.compare.back);
      return;
    }
    const to = target(v, now, "m");
    if (!to) return;
    const back = { tab: v.tab, selected: v.selected, seed: v.seed, opened: v.opened, focus: v.focus };
    visit(v, to);
    v.compare = { step: episode(v, now).perturbedStep, back };
  }

  // The step the stage holds the shown episode at: its move while Move shows it, null for its latest frame.
  const stageStep = (v) => (v.compare ? v.compare.step : null);

  // Who the control bar names: the controller of the frame the stage holds under Move, else the episode's latest.
  function inControl(v, now) {
    const ep = episode(v, now);
    if (!ep) return null;
    const step = stageStep(v);
    const frame = step === null ? null : filmstrip(ep, step).find((f) => f.now);
    return frame ? { controller: frame.controller, checkpoint: frame.checkpoint, step: frame.step } : { controller: ep.controller, checkpoint: ep.checkpoint, step: null };
  }

  // An episode is over once judged, once the run has moved past it, or once the run has stopped.
  const finished = (v, now, ep) => ep.verdict !== null || ep !== current(shown(v)) || !following(v, now);

  const KEYS = { " ": togglePause, m: showMove };
  for (const key of ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Escape", "End", "d", "t"]) KEYS[key] = (v, now) => go(v, now, key);
  // ↑ and ↓ are Results' only in its list and Escape only inside a seed; elsewhere they keep their usual jobs.
  const OWNED = { ArrowUp: (v) => where(v) === "list", ArrowDown: (v) => where(v) === "list", Escape: (v) => where(v) === "seed" || where(v) === "episode" };
  // Typing in a field is text, never a presenter key (SPEC §3.5).
  const typing = (t) => Boolean(t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable));
  // Presenter keys (SPEC §3.5), whatever else has focus. True when the key is the dashboard's, so the page claims it
  // (a focused button must not also take Space as a click); a held key acts once.
  function keydown(v, e, now) {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const action = KEYS[key];
    if (!action || e.metaKey || e.ctrlKey || e.altKey || typing(e.target) || (OWNED[key] && !OWNED[key](v))) return false;
    if (!e.repeat) action(v, now);
    return true;
  }

  // The dashboard's runs (SPEC §3.5, `GET /runs`), read only in the contract's shape and kept in the server's order.
  const STATUSES = ["live", "recorded", "not_started"];
  const orNull = (v) => v === null || typeof v === "string";
  function runsOf(list) {
    if (!Array.isArray(list)) return [];
    return list.filter((r) => r && typeof r.run_id === "string" && r.run_id && orNull(r.instruction) && STATUSES.includes(r.status) && orNull(r.reason) && num(r.started_t) !== null);
  }
  // The run a page opens: the one its URL names, else the first listed (live first, then newest), else none.
  const runToOpen = (runs, wanted) => wanted || (runs[0] ? runs[0].run_id : null);

  // The Teach box (SPEC §3.5): at most 200 characters once trimmed, as the server takes it.
  const TEACH_MAX = 200;
  function instructionProblem(typed) {
    const length = typed.trim().length;
    return !length ? "Type what the robot should learn." : length > TEACH_MAX ? `At most ${TEACH_MAX} characters.` : null;
  }
  // The count shows as the text nears the limit.
  const teachCount = (typed) => {
    const length = typed.trim().length;
    return length >= TEACH_MAX - 40 ? `${length}/${TEACH_MAX}` : null;
  };
  // The server's answer to a Teach: started (the page opens the new run), not started with the generator's reasons,
  // or a failure in words.
  function teachAnswer(status, body) {
    const id = body && typeof body.run_id === "string" ? body.run_id : null;
    if (status === 201 && id && body.started === true) return { state: "started", runId: id };
    const reasons = body && Array.isArray(body.reasons) && body.reasons.every((r) => typeof r === "string") ? body.reasons : null;
    if (status === 200 && id && body.started === false && reasons) return { state: "refused", runId: id, reasons };
    const message =
      status === 0
        ? "Could not reach the dashboard server."
        : status === 403
          ? "The dashboard takes instructions only from its own page on this computer."
          : status === 400
            ? `An instruction is 1 to ${TEACH_MAX} characters.`
            : `The dashboard server could not take it (HTTP ${status}).`;
    return { state: "failed", message };
  }

  const api = { esc, initial, view, reconnect, settle, disconnect, reduce, receive, current, provenance, panelLabel, summary, filmstrip, move, actedOn, median, shown, buffered, pauseLabel, following, accepted, kind, verdictWords, simulator, claim, testing, results, MARKS, lessons, argon, trainingSummary, replayReport, runsOf, runToOpen, instructionProblem, teachCount, teachAnswer, teacherDrove, episode, where, teaches, finetuning, target, visit, stageStep, inControl, finished, keydown };
  if (typeof module !== "undefined") module.exports = api;
  else root.MomState = api;
})(this);
