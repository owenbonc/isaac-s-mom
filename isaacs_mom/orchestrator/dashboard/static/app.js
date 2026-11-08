// Renders MomState over the /events stream. Every value from an event is escaped; nothing is shown that an event did not carry.
const view = MomState.view();
let renderQueued = false;
let renderedRows = 0;
let renderedEpisode = null;
let renderedFocus = null;
let renderedSide = null;
// The log follows new rows until the reader scrolls up from its bottom, and again once they scroll back down.
let logFollowing = true;
// Where following last parked the log: its foot, or the stage frame's row while a later decision runs.
let logAnchor = 0;
let flashedFrame = null;
let lastHeard = { text: "", episode: null, paused: false, lost: false };
// A one-off line for the announcer: the box or bin move, heard once when it arrives live.
let news = "";
// Run files by name, null once asked for and absent; a written file never changes, so each is read until found.
const runFiles = {};
// Configuration (SPEC §3.2, §3.3), not data: the page shows these labelled as configured.
const CONFIGURED = { student: "SmolVLM2-500M, local", teacher: "Argon (Gemini API)", finetune: "RTX 5070, local" };

const $ = (id) => document.getElementById(id);
let runId = new URLSearchParams(location.search).get("run");
let runs = [];
let teaching = { state: "idle" };
// Assigning identical HTML would reload every image and reset scrolling, so panels skip unchanged writes.
const native = Object.getOwnPropertyDescriptor(Element.prototype, "innerHTML");
for (const el of document.querySelectorAll("[id]")) {
  Object.defineProperty(el, "innerHTML", {
    get: () => native.get.call(el),
    set(html) {
      if (el.dataset.html === html) return;
      el.dataset.html = html;
      native.set.call(el, html);
    },
  });
}
const { esc, shortCheckpoint } = MomState;
const X = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>';
const ARROW = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M1.5 6h8M6.5 2.5L10 6l-3.5 3.5" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const DIAMOND = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1.5L10.5 6 6 10.5 1.5 6z" fill="currentColor"/></svg>';
const BANG = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2v5M6 9.6v.4" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>';
const CHECK = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.6 2.6L10 3" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const MOVE = '<svg class="i" aria-hidden="true"><use href="#i-move"/></svg>';

// Events carry the server's time, so the page's clock is its own plus the skew the /runs answer's Date header shows.
let skew = 0;
const now = () => Date.now() / 1000 + skew;
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const chip = (kind, text = cap(kind)) => `<span class="chip ${kind}">${text}</span>`;
const panelChip = (ep = null) => {
  const { kind, spelled } = MomState.panelLabel(view, now(), ep);
  const word = cap(kind);
  return spelled ? `<span class="chip quiet ${kind}" title="${word}">${word}</span>` : `<span class="chip quiet ${kind}" role="img" aria-label="${word}" title="${word}"></span>`;
};
const WHO = { student: "Student", teacher: "Teacher (Argon)" };
const tag = (controller, checkpoint) => {
  const who = WHO[controller] || "Unknown controller";
  return `<span class="tag ${controller || "unknown"}">${who}${checkpoint ? ` <span class="mono">${esc(shortCheckpoint(checkpoint))}</span>` : ""}</span>`;
};
const amount = (row) => (row.distance !== null ? ` ${+row.distance.toFixed(1)} cm` : row.angle !== null ? ` ${Math.round(row.angle)}°` : "");
const skillText = (row) => (row.skill ? esc(row.skill) + amount(row) : '<span class="stop">Stopped, no skill chosen</span>');
const finished = (ep) => MomState.finished(view, now(), ep);
// What an episode is, named without the loop's round index, which reads like a checkpoint number and is not one.
const KINDS = {
  take: ["Judged take", "unseen scene, seed"],
  heldout: ["Held-out test", "seed"],
  demonstration: ["Teacher demonstration", "development seed"],
  trial: ["Student trial", "development seed"],
  // A development episode before its first decision says whether the student or the teacher drives it.
  null: ["Development episode", "seed"],
};
const kindOf = (ep) => {
  const [name, seed] = KINDS[MomState.kind(ep)];
  return [name, `${seed} ${esc(ep.seed ?? "")}`];
};
// An errored held-out test takes Results' yellow ! wherever it shows, never a failure's red ✗ (rule 6).
const verdictLook = (ep) => (MomState.errored(ep) ? ["warn", markIcon("error")] : MomState.accepted(ep) ? ["good", CHECK] : ["bad", X]);
const verdictChip = (ep) => {
  const [look, icon] = verdictLook(ep);
  return `<span class="verdict ${look}">${icon}${esc(MomState.verdictWords(ep))}</span>`;
};
const MOVED = { move_box: "Box moved", move_bin: "Bin moved" };
// Durations read at a glance from the back of the room: ms under a second, then seconds, then minutes.
const duration = (v) => (v < 1000 ? `${v >= 100 ? Math.round(v) : v.toFixed(1)} ms` : v < 120000 ? `${(v / 1000).toFixed(1)} s` : `${(v / 60000).toFixed(1)} min`);
// A running clock in whole seconds: "45 s", "2 min 05 s".
const elapsed = (seconds) => {
  const whole = Math.max(0, Math.floor(seconds));
  return whole < 60 ? `${whole} s` : `${Math.floor(whole / 60)} min ${String(whole % 60).padStart(2, "0")} s`;
};
const stat = (s) => (s ? `${duration(s.median)} <span class="note">n=${s.n}</span>` : '<span class="ph-text">no data</span>');

function renderHeader() {
  const s = MomState.shown(view);
  const kind = MomState.provenance(view, now());
  const sim = MomState.simulator(s);
  // One line at 1280 px: the simulator in words (its version and task are in the legend), or a lost stream alone.
  const status = view.lost
    ? chip("lost", "Disconnected")
    : s.refused !== null
      ? chip("refused", "Not started")
      : s.eventCount === 0
        ? chip("placeholder", runId ? "Waiting" : "No run")
        : chip(kind, `${cap(kind)}${sim && sim.name ? ` · ${esc(sim.name)}` : ""}`);
  $("status").innerHTML = status;
  const robot = s.episodes.length && s.episodes[s.episodes.length - 1].sim;
  const task = sim && sim.task ? `, the ${sim.task === "fixed_feet" ? "fixed-feet" : "walking"} task (<span class="mono">${esc(sim.task)}</span>)` : "";
  const on = !sim ? "" : sim.version ? ` on <span class="mono">${esc(sim.version)}</span>${task}` : ", which names no simulator";
  $("run-meta").innerHTML = s.eventCount
    ? `Run <b class="mono">${esc(s.runId)}</b>${on}${robot ? `, robot <span class="mono">${esc(robot.robot)}</span>` : ""}: ${s.episodes.length} episode${s.episodes.length === 1 ? "" : "s"}, ${s.eventCount} events.`
    : "";
}

// An episode opened from a seed is placed by its seed and checkpoint instead, so it has no position.
function position(ep) {
  const s = MomState.shown(view);
  return MomState.where(view) === "episode" || !ep ? "" : `Episode ${s.episodes.indexOf(ep) + 1} of ${s.episodes.length}`;
}

// Screen readers hear what changed (episode, control, verdict, the seed open, pause, connection), never every step.
function announce(text) {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const place = MomState.where(view);
  const heard = { text, place, seed: view.seed, episode: ep && ep.id, paused: view.frozen !== null, lost: view.lost };
  const parts = [];
  if (heard.paused !== lastHeard.paused) parts.push(heard.paused ? "Paused." : "Resumed.");
  if (heard.lost !== lastHeard.lost) parts.push(heard.lost ? "Connection to the run lost, retrying." : "Connection restored.");
  const row = place === "seed" && MomState.results(s).seeds.find((x) => x.seed === view.seed);
  if (row && (heard.place !== lastHeard.place || heard.seed !== lastHeard.seed)) parts.push(`Seed ${row.seed}, ${MARK[row.mark].toLowerCase()}: ${outcomes(row)}.`);
  else if (place !== "seed" && ep && (heard.episode !== lastHeard.episode || heard.place !== lastHeard.place)) parts.push(`${position(ep) || `Seed ${view.seed}`}, ${kindOf(ep)[0].toLowerCase()}.`, text);
  else if (text !== lastHeard.text) parts.push(text);
  parts.push(news);
  news = "";
  lastHeard = heard;
  const message = parts.filter(Boolean).join(" ");
  if (message) $("announce").textContent = message;
}

// The control bar; returns its sentence for the announcer.
function renderControl() {
  const ep = MomState.episode(view, now());
  const el = $("control");
  // The bar names the fine-tune; the stage shows what it learns from.
  if (MomState.finetuning(view, now()) !== null) {
    el.className = "control";
    el.innerHTML = view.lost
      ? `<span class="what">Connection lost during the fine-tune</span><span class="note">Whether it is still running shows again when the connection returns.</span>`
      : `<span class="what">Fine-tuning the student</span><span class="note">No robot is moving. The held-out test of the new checkpoint starts when it is ready.</span>`;
    return view.lost ? "" : "Fine-tuning the student.";
  }
  const place = MomState.where(view);
  if (place === "seed") {
    el.className = "control";
    el.innerHTML = `<span class="what">Held-out seed ${esc(view.seed)}</span><span class="note">Choose a checkpoint to see that episode in full.</span>`;
    return "";
  }
  if (!ep) {
    const s = MomState.shown(view);
    el.className = "control";
    // A run that did not start says why on the stage's slate, so the bar names only that.
    el.innerHTML = !runId
      ? '<span class="what">No run yet</span>'
      : s.refused !== null
        ? '<span class="what">Not started</span>'
        : '<span class="what">Waiting for the run to start</span>';
    return "";
  }
  const [kind, where] = kindOf(ep);
  const r = place === "episode" && MomState.results(MomState.shown(view));
  const row = r && r.seeds.find((x) => x.seed === view.seed);
  const k = row ? row.episodes.findIndex((x) => x.id === ep.id) : -1;
  const at = k < 0 ? "" : `, ${esc(r.checkpoints[k].name)}`;
  const among = position(ep);
  const back = place === "episode" ? `<button type="button" class="back" data-back data-f="control-back" aria-label="Back" data-tip="Back · Esc"><svg class="i" aria-hidden="true"><use href="#i-prev"/></svg></button>` : "";
  const head = `${back}<span class="kind"><span class="what">${kind}</span><span class="note">${where}${at}${among ? ` · ${among.toLowerCase()}` : ""}</span></span>`;
  if (!ep.controller) {
    el.className = "control";
    el.innerHTML = `${head}<span class="note">Waiting for the first decision.</span>`;
    return "";
  }
  // Under Move the bar names whoever controlled the frame on the stage, not whoever controls the episode now.
  const { controller, checkpoint, step } = MomState.inControl(view, now());
  const who = WHO[controller] || "An unknown controller";
  const took = ep.rows.find((r) => r.kind === "takeover");
  // While the stream is lost nothing reads as still running.
  const done = finished(ep) || view.lost;
  const words = MomState.verdictWords(ep);
  // The controller tag is the sentence's subject.
  const detail =
    step !== null
      ? `was in control at step ${esc(step)}.`
      : took
        ? `took over when the student stopped${took.step === null ? "" : " at step " + esc(took.step)}.`
        : controller === "student"
          ? `${done ? "chose" : "is choosing"} each skill from the head-camera frame.`
          : `${done ? "demonstrated" : "is demonstrating"} from the head-camera frame.`;
  // A judged take's verdict already says when the teacher took over.
  const sentence = step === null && ep.demo && took && words ? "" : `<span class="note">${detail}</span>`;
  el.className = "control " + (controller || "");
  el.innerHTML = `${head}${tag(controller, checkpoint)}${sentence}${words ? verdictChip(ep) : ""}`;
  const said = step !== null ? `${who} was in control at step ${step}.` : `${who} ${done ? "was" : "is"} in control.`;
  return `${said}${words ? ` Verdict: ${words}.` : ""}`;
}

const noEpisode = () => slate("No episode yet", "The head camera's frames appear here once the run starts.");

// A frame-sized card on the stage where no head-camera frame can be shown yet.
const slate = (title, note) => `<div class="frame"><div class="cell"><div class="shot slate"><p class="what">${title}</p><p class="note">${note}</p></div></div></div>`;
// The contact sheet holds at most 4 × 4 frames, so each stays large enough to read.
const SHEET = 16;

// While the student fine-tunes no robot moves, so the stage shows what it learns from: the last frame of each
// accepted teacher demonstration so far, with the counts behind them and the time since the fine-tune started.
function fineTuneStage(seconds) {
  const l = MomState.lessons(MomState.shown(view));
  const shown = l.frames.slice(-SHEET);
  // A lost stream can't tell whether the fine-tune still runs, so the clock says only when it started.
  const clock = `${view.lost ? "Connection lost · " : ""}${elapsed(seconds)} since the fine-tune started`;
  // A run from an earlier run also learns from that run's accepted demonstrations (SPEC §3.6), counted with these.
  const carried = l.carried ? l.carried.count : 0;
  const from = carried ? `, ${carried} carried from <span class="mono">${esc(l.carried.run)}</span>` : "";
  const total = l.accepted + carried;
  const here = carried ? `this run's ${l.accepted} accepted, ${l.rejected} rejected, n=${l.n}` : `${l.rejected} rejected, n=${l.n}`;
  const sheet = shown.length
    ? `<div class="frame"><div class="cell"><div class="shot sheet"><ol style="--n:${Math.ceil(Math.sqrt(shown.length))}" aria-label="The last frame of each accepted teacher demonstration">${shown
        .map((f) => `<li${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="Episode ${f.number}'s last frame">` : ""}</li>`)
        .join("")}</ol></div></div></div>`
    : slate(
        carried ? `Learning from ${carried} demonstration${carried === 1 ? "" : "s"} carried from <span class="mono">${esc(l.carried.run)}</span>` : "No accepted demonstration to learn from",
        `${clock}. Teacher demonstrations in this run: ${l.accepted} accepted · ${l.rejected} rejected (n=${l.n}).`,
      );
  const latest = l.frames.length > shown.length ? `, the latest ${shown.length} of ${l.frames.length} shown` : "";
  // The counts lead and the line wraps, so no width ever cuts a number off.
  const caption = shown.length
    ? `<p class="caption wrap" title="The last frame of each accepted teacher demonstration in this run${latest}.">Learning from ${total} accepted teacher demonstration${total === 1 ? "" : "s"}${from} (${here}) · ${clock}</p>`
    : "";
  return sheet + foot(caption, panelChip());
}

function renderCamera() {
  const tuning = MomState.finetuning(view, now());
  if (tuning !== null) {
    $("stage").innerHTML = fineTuneStage(tuning);
    $("strip").innerHTML = "";
    return;
  }
  const ep = MomState.episode(view, now());
  const frames = MomState.filmstrip(ep, MomState.stageStep(view));
  if (!frames.length) {
    const [kind, where] = ep ? kindOf(ep) : [];
    const note = ep && finished(ep) ? "This episode recorded no head-camera frame." : "The first frame appears after the first skill.";
    const refused = MomState.shown(view).refused;
    const empty = !runId
      ? slate("No runs yet", "Tell the robot what to learn in the Teach box to start one.")
      : refused !== null
        ? slate("Not started", esc(refused))
        : noEpisode();
    $("stage").innerHTML = (ep ? slate(`${kind} · ${where}`, note) : empty) + foot("", panelChip(ep));
    $("strip").innerHTML = "";
    return;
  }
  const last = frames.find((f) => f.now);
  const move = MomState.move(ep);
  const what = move && esc(MOVED[ep.rows.findLast((r) => r.kind === "perturbation").what] || "Scene changed");
  // The caption row is always there, so the peak's caption never shrinks the frame.
  let frame, caption, movedAt = "";
  if (view.compare && move) {
    // The orchestrator took the left frame at the same pose right before the move, so no skill was chosen from it.
    frame = cell(move.before, null, false, "Before the move") + cell(move.after, skillRow(ep, move.after), true, "After");
    // The pair covers the log, so the caption carries its marked row, clamped so it never shrinks the pair.
    const first = move.decision;
    const next = first ? ` First decision after the move: <span class="mono">${skillText(first)}</span>.${first.reason ? ` ${esc(first.reason)}` : ""}` : "";
    caption = `<p class="caption moved clamped"${first && first.reason ? ` title="${esc(first.reason)}"` : ""}>${what} between these frames, before step ${esc(move.step)}.${next}</p>`;
  } else {
    // After its first frame the move stays marked on the caption line, so it can be shown once the moment has passed.
    movedAt = last.after && !last.moved ? `<span class="moved-at">${MOVE}${what} at step ${esc(move.step)}</span>` : "";
    frame = cell(last, skillRow(ep, last), last.moved);
    caption = last.moved
      ? `<p class="caption moved">${what} before step ${esc(move.step)}. This is the first frame after it.</p>`
      : `<p class="caption">${last.step === null ? "Head-camera frame." : `What the ${last.controller || "robot"} saw at step ${esc(last.step)}, before choosing.`}</p>`;
  }
  $("stage").innerHTML = `<div class="frame">${frame}</div>${foot(movedAt + caption, panelChip(ep))}`;
  $("strip").innerHTML = frames
    .map((f) => `<li class="${[f.now ? "now" : "", f.moved ? "moved" : ""].join(" ").trim()}"><div class="thumb"${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="">` : ""}</div><span class="st">step ${esc(f.step ?? "")}${f.moved ? '<span class="sr"> (the scene moved before this frame)</span>' : ""}</span></li>`)
    .join("");
  // Flash once when the first frame after the move arrives live, never on a key press; added outside the HTML so
  // later identical writes are skipped and the animation runs to its end.
  const key = `${ep.id}@${last.step}`;
  const arriving = view.selected === null && !view.frozen && MomState.following(view, now());
  if (last.moved && arriving && key !== flashedFrame) {
    news = `${what} at step ${move.step}. M shows the frames before and after.`;
    const shot = $("stage").querySelector(".shot");
    shot.classList.add("flash");
    shot.addEventListener("animationend", () => shot.classList.remove("flash"), { once: true });
  }
  flashedFrame = last.moved ? key : flashedFrame;
}

// The stage's caption line, which also carries the camera panel's provenance label.
const foot = (caption, provenance) => `<div class="foot">${caption}${provenance}</div>`;

// The decision that chose a skill from frame `f`, or null.
const skillRow = (ep, f) => (f && ep.rows.find((r) => r.kind === "decision" && r.step === f.step && r.skill)) || null;

function cell(f, row, moved, label = "") {
  const shot = f && f.url ? `<img src="${esc(f.url)}" alt="Head camera frame${f.step === null ? "" : ` at step ${esc(f.step)}`}">` : "";
  const skill = row ? `<span class="said ${row.controller || "unknown"}"><svg class="i" aria-hidden="true"><use href="#s-${esc(row.skill)}"/></svg>${skillText(row)}</span>` : "";
  const notch = label || skill ? `<p class="notch">${label ? `<span>${label}</span>` : ""}${skill}</p>` : "";
  const who = f ? tag(f.controller, f.checkpoint) : "";
  return `<div class="cell"><div class="shot${moved ? " moved" : ""}"${shot ? "" : " data-missing"}>${shot}${notch}${who}</div></div>`;
}

// Event rows carry a glyph where decisions carry their step, so the two never read as the same kind of row.
function logRow(row) {
  const at = row.step === null ? "" : ` before step ${esc(row.step)}`;
  switch (row.kind) {
    case "perturbation":
      return `<li class="event moved"><span class="n">${ARROW}</span><span>${esc(MOVED[row.what] || "Scene changed")}${at} <span class="d">by the orchestrator. The robot sees it only through its camera.</span></span></li>`;
    case "takeover":
      return `<li class="event take"><span class="n">${DIAMOND}</span><span>Teacher took over${row.step === null ? "" : ` at step ${esc(row.step)}`}</span></li>`;
    case "error":
      return `<li class="event bad"><span class="n">${BANG}</span><span>Error <span class="d">${esc(row.text)}</span></span></li>`;
    default:
      return "";
  }
}

// The reason behind the stage's frame, and any decided after it, reads in full; older ones are clamped.
function decisionRow(row, after, done, full) {
  const outcome = !row.skill ? "" : row.ok === null ? (done ? "" : '<span class="lat">running</span>') : row.ok ? `<span class="ok">${CHECK}<span class="sr">executed</span></span>` : '<span class="no">failed</span>';
  const latency = row.latency === null || !row.skill ? "" : `<span class="lat">${duration(row.latency)}</span>`;
  const reason = !row.reason ? "" : full ? `<div class="reason">${esc(row.reason)}</div>` : `<div class="reason clamped" title="${esc(row.reason)}">${esc(row.reason)}</div>`;
  const error = row.error ? `<div class="err">${esc(row.error)}</div>` : "";
  const mark = after ? '<div class="after">First decision after the move</div>' : "";
  return `<li class="${row.controller || "unknown"}${after && view.compare ? " focus" : ""}"><span class="n">${esc(row.step ?? "")}</span><div class="line"><span class="skill">${skillText(row)}</span><span class="right">${latency}${outcome}</span></div>${mark}${reason}${error}</li>`;
}

function verdictRow(ep) {
  const v = ep.verdict;
  const [look, icon] = verdictLook(ep);
  const reason = v.reason ? `${esc(cap(v.reason))}. ` : "";
  return `<li class="event ${look}"><span class="n">${icon}</span><span>${esc(MomState.verdictWords(ep))}. <span class="d">${reason}Task ${v.task ? "succeeded" : "not met"}, physics ${v.physics ? "valid" : "violated"}.</span></span></li>`;
}

const NAVIGATION = { "b-prev": "ArrowLeft", "b-next": "ArrowRight", "b-end": "End", "b-d": "d", "b-t": "t", "b-m": "m" };
const STEPS = { seed: "seed", episode: "checkpoint", run: "episode" };
const PAUSE = $("b-pause");

function renderEpisode() {
  for (const [id, key] of Object.entries(NAVIGATION)) $(id).disabled = MomState.target(view, now(), key) === null;
  const steps = STEPS[MomState.where(view)];
  $("b-prev").dataset.tip = `Previous ${steps} · ←`;
  $("b-next").dataset.tip = `Next ${steps} · →`;
  $("b-m").setAttribute("aria-pressed", String(view.compare !== null));
  PAUSE.classList.toggle("on", view.frozen !== null);
  PAUSE.setAttribute("aria-label", MomState.pauseLabel(view));
  PAUSE.dataset.tip = `${MomState.pauseLabel(view)} · Space`;
  PAUSE.lastElementChild.textContent = view.frozen ? MomState.buffered(view) : "";
}

function renderLog() {
  const ep = MomState.episode(view, now());
  const list = $("log");
  // Said once in the header rather than on every row, until a decision in the episode carries a reason.
  const decisions = ep ? ep.rows.filter((r) => r.kind === "decision") : [];
  const reasons = !decisions.length || decisions.some((r) => r.reason);
  const silent = decisions.every((r) => r.controller === "student") ? "The student gives no reasons" : finished(ep) ? "No reasons given" : "No reasons yet";
  $("chip-log").innerHTML = (reasons ? "" : `<span class="note">${silent}</span>`) + panelChip(ep);
  if (!ep) {
    list.innerHTML = '<li class="empty">No decisions yet. Each decision appears with its reason and latency.</li>';
    renderedEpisode = null;
    return;
  }
  const move = MomState.move(ep);
  const done = finished(ep) || view.lost;
  const step = MomState.stageStep(view);
  const stage = MomState.filmstrip(ep, step).find((f) => f.now);
  const full = (r) => !stage || stage.step === null || (r.step !== null && r.step >= stage.step);
  const html = ep.rows.map((r) => (r.kind === "decision" ? decisionRow(r, move && r === move.decision, done, full(r)) : r.kind === "verdict" ? verdictRow(ep) : logRow(r)));
  // Letting go of Move drops the scroll it parked, so the log follows new rows again.
  const focus = step === null ? null : `${ep.id}@${step}`;
  const fromStart = renderedEpisode !== ep.id || html.length < renderedRows || (renderedFocus !== null && focus === null);
  if (fromStart) logFollowing = true;
  list.innerHTML = html.join("");
  if (!fromStart) [...list.children].slice(renderedRows).forEach((li) => li.classList.add("fresh"));
  if (logFollowing) {
    // Until the verdict, the foot never pushes the stage frame's row (its reason) above the log's top.
    const stageRow = !ep.verdict && stage && list.children[ep.rows.findIndex((r) => r.kind === "decision" && r.step === stage.step && r.skill)];
    list.scrollTop = stageRow ? stageRow.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop : list.scrollHeight;
    logAnchor = list.scrollTop;
  }
  // Move parks the log with the move row a third of the way down, and it stays parked until the reader scrolls.
  const row = list.querySelector("li.event.moved");
  if (focus !== null && focus !== renderedFocus && row) {
    list.scrollTop += row.getBoundingClientRect().top - list.getBoundingClientRect().top - list.clientHeight / 3;
    logFollowing = false;
  }
  renderedFocus = focus;
  renderedRows = html.length;
  renderedEpisode = ep.id;
}

function heldOut() {
  const s = MomState.shown(view);
  const c = MomState.claim(s);
  const test = MomState.testing(s);
  // A run that stopped arriving mid-evaluation can't say the test still runs.
  const live = MomState.following(view, now()) && !view.lost;
  const progress = test && `Testing ${test.checkpoint ? `<span class="mono">${esc(shortCheckpoint(test.checkpoint))}</span>` : "the student"}: ${test.run}${test.of === null ? " run so far" : ` of ${test.of} run`}${live ? "" : " when its events stopped"}.`;
  const errored = c && c.errored ? `<span class="note">${esc(c.errored)}</span>` : "";
  const lead = c && (c.figure
    ? `<span class="fig" role="img" aria-label="Held-out success: ${esc(c.spoken)}">${esc(c.figure)}</span><span class="rates"><span>${esc(c.counts)}</span>${errored}</span>`
    : `<span>${esc(cap(c.spoken))}.</span>${errored}`);
  const gain = c && `<div class="gain">${lead}</div><div class="names" title="${esc(c.ids)}">${esc(c.names)}</div>`;
  // A run that did not start says why where its results would be, so it needs no line here.
  const none = progress || (s.refused === null ? "No held-out evaluation in the events yet." : "");
  const body = gain ? gain + (progress ? `<div class="note">${progress}</div>` : "") : none && `<div class="ph-text">${none}</div>`;
  // A run is taught and tested on its own instruction, so its rates carry it (SPEC §3.5).
  const said = s.instruction && s.instruction.text;
  const taught = said ? `<p class="taught clamped" title="${esc(said)}">${c ? "Tested on" : "Instruction"} <q dir="auto">${esc(said)}</q></p>` : "";
  return `<header><h2>Held-out success</h2>${panelChip()}</header><div class="body stack">${body}${taught}</div>`;
}

const MARK = { down: "Regressed", up: "Improved", fail: "Still fails", pass: "Always passed", error: "Errored", pending: "Pending" };
const markIcon = (m) => `<svg class="i mark ${m}" aria-hidden="true"><use href="#m-${m}"/></svg>`;
const OUTCOME = { succeeded: "Succeeded", failed: "Failed", errored: "Errored", pending: "not run yet" };
const outcomes = (row) => {
  const [first, latest] = [row.episodes[0], row.episodes[row.episodes.length - 1]];
  return row.episodes.length < 2 ? OUTCOME[first.outcome] : `${OUTCOME[first.outcome]} → ${OUTCOME[latest.outcome]}`;
};

// Before and After are uncoloured, so only the Status row's marks carry colour.
const VERDICT = { succeeded: ["pass", "succeeded"], failed: ["fail", "failed"], errored: ["error", "errored"], pending: ["pending", "not run"] };
const verdictIcon = (o) => `<svg class="i plain" aria-hidden="true"><use href="#m-${VERDICT[o][0]}"/></svg>`;
const columnName = (c) => `Seed ${c.seed}: ${MARK[c.mark].toLowerCase()}, ${VERDICT[c.before][1]} before, ${VERDICT[c.after][1]} after`;
// The column that takes the strip's one Tab stop: the seed open, else the one last focused, else the first.
let stripFocus = null;
function renderStrip() {
  const cols = MomState.columns(MomState.shown(view));
  const at = [view.seed, stripFocus].find((seed) => cols.some((c) => c.seed === seed)) ?? (cols.length ? cols[0].seed : null);
  $("columns").innerHTML = cols.length
    ? cols
        .map((c) => {
          const name = esc(columnName(c));
          const open = c.seed === view.seed ? ' aria-current="true"' : "";
          return `<li><button type="button" class="column" data-column="${esc(c.seed)}" data-f="column:${esc(c.seed)}" tabindex="${c.seed === at ? 0 : -1}" aria-label="${name}" title="${name}"${open}>${markIcon(c.mark)}${verdictIcon(c.before)}${verdictIcon(c.after)}</button></li>`;
        })
        .join("")
    : `<li class="none note">${MomState.shown(view).refused !== null ? "Not started." : "Each held-out seed shows here once the student has been tested on it."}</li>`;
  // With no seed open the switch has nothing to compare, so it stays in view, dimmed, and says why.
  const idle = view.seed === null;
  for (const b of $("sides").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.side === view.side));
    b.setAttribute("aria-disabled", String(idle));
    b.dataset.tip = idle ? "Choose a seed to compare" : SIDES[b.dataset.side];
  }
}
const SIDES = { before: "Before: the seed's first checkpoint", both: "Both: every checkpoint side by side", after: "After: the seed's latest checkpoint" };

const FRAME_MIN = 280;
// Until the presenter chooses, the switch shows Both where each card keeps a readable frame in the stage, else After.
let chosen = false;
const stageWidth = () => $("results").clientWidth || document.querySelector(".camera").clientWidth;
function defaultSide() {
  const count = MomState.results(MomState.shown(view)).checkpoints.length || 1;
  if (!chosen) view.side = stageWidth() / count >= FRAME_MIN ? "both" : "after";
}
function cardVerdict(e, ep) {
  if (e.outcome === "pending") return `<span class="note">${ep ? "Running" : "Not run yet"}</span>`;
  if (e.outcome === "errored") {
    const why = !ep ? "This test's events hold no episode of the seed." : ep.verdict ? cap(ep.verdict.reason) : "No verdict before its test ended.";
    return `<span class="verdict warn">${markIcon("error")}Errored</span><span class="note clamped two" title="${esc(why)}">${esc(why)}</span>`;
  }
  return `${verdictChip(ep)}<span class="note clamped two" title="${esc(cap(ep.verdict.reason))}">${esc(cap(ep.verdict.reason))}</span>`;
}
function seedCard(c, e, ep) {
  const short = c.checkpoint && shortCheckpoint(c.checkpoint);
  const id = short && !c.name.endsWith(short) ? `<span class="mono note">${esc(short)}</span>` : "";
  const head = `<span class="card-head"><span class="what">${esc(cap(c.name))}</span>${id}</span>`;
  const f = ep && ep.frames[ep.frames.length - 1];
  const shot = f
    ? `<span class="shot"${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="The episode's last head-camera frame${f.step === null ? "" : `, step ${esc(f.step)}`}">` : ""}${tag(f.controller, f.checkpoint)}</span>`
    : `<span class="shot slate"><span class="note">${!ep ? (e.outcome === "pending" ? "Not run yet." : "No episode.") : e.outcome === "pending" ? "No frame yet." : "No frame recorded."}</span></span>`;
  const body = `${head}<span class="cell">${shot}</span><span class="card-verdict">${cardVerdict(e, ep)}</span>`;
  return ep ? `<button type="button" class="card" data-open="${esc(ep.id)}" data-f="card:${esc(ep.id)}">${body}</button>` : `<div class="card">${body}</div>`;
}
function seedDetail() {
  const s = MomState.shown(view);
  const r = MomState.results(s);
  const row = r.seeds.find((x) => x.seed === view.seed);
  const back = `<button type="button" class="back" data-back data-f="back" aria-label="Back" data-tip="Back · Esc"><svg class="i" aria-hidden="true"><use href="#i-prev"/></svg></button>`;
  const title = `<h2>Seed ${esc(view.seed)}${row ? `${markIcon(row.mark)}<span class="sr">, ${MARK[row.mark].toLowerCase()}</span>` : ""}</h2>`;
  if (!row) return `<header class="seed-head">${back}${title}</header><p class="empty note">This run's events hold no held-out episode on seed ${esc(view.seed)}.</p>`;
  const count = r.checkpoints.length;
  const picks = view.side === "before" ? [0] : view.side === "after" ? [count - 1] : r.checkpoints.map((_, k) => k);
  const cards = picks.map((k) => seedCard(r.checkpoints[k], row.episodes[k], row.episodes[k].id === null ? null : s.episodes.find((x) => x.id === row.episodes[k].id)));
  return `<header class="seed-head">${back}${title}<span class="note">${outcomes(row)}</span></header><div class="cards" style="--n:${picks.length}">${cards.join("")}</div>`;
}

function renderSeed() {
  const place = MomState.where(view);
  $("main").dataset.view = place;
  if (place === "seed") $("results").innerHTML = seedDetail();
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function renderModels() {
  const shown = MomState.shown(view);
  const s = MomState.summary(shown);
  const a = MomState.argon(shown);
  const item = (who, where, measured) => `<span class="item"><b>${who}</b> ${where} <span class="v">${stat(measured)}</span></span>`;
  const calls = a
    ? `${a.used}${a.budget === null ? "" : ` of ${a.budget}`} <span class="v">· ${stat(a.answer)} · ${a.failed} failed</span>`
    : '<span class="ph-text">none in the events</span>';
  $("models").innerHTML =
    // The item names say what the row is; the heading stays for screen readers so the row fits 1280 px.
    `<h2 class="sr">Models</h2><span class="chip quiet config">Configured</span>` +
    item("Student", CONFIGURED.student, s.student) +
    item("Teacher", CONFIGURED.teacher, s.teacher) +
    item("Fine-tune", CONFIGURED.finetune, s.finetune) +
    `<span class="item"><b>Argon calls</b> ${calls}</span>` +
    panelChip();
}

function renderTraining() {
  const s = MomState.shown(view);
  const missing = (name) => `<span class="ph-text">${name in runFiles || !runId ? "Not in the run's files." : "Reading the run's files."}</span>`;
  const round = (r) => {
    const t = runFiles[`training_r${r}.summary.json`];
    if (!t) return `<dt>Round ${esc(r)}</dt><dd>${missing(`training_r${r}.summary.json`)}</dd>`;
    const by = Object.entries(t.rejected_by).map(([why, n]) => `<span class="mono">${esc(why)}</span>&nbsp;${n}`).join(", ");
    const detail = [
      `${t.rejected} rejected${by ? ` (${by})` : ""}`,
      t.unfinished ? `${t.unfinished} unfinished` : "",
      `${t.empty_dev_episodes} empty of ${plural(t.episodes_scanned, "episode")}`,
    ].filter(Boolean);
    return `<dt>Round ${esc(r)}</dt><dd>${plural(t.examples, "example")} from ${t.accepted} accepted of ${plural(t.demonstrations, "demonstration")} so far <span class="note">· ${detail.join(" · ")}</span></dd>`;
  };
  const r = runFiles["replay_report.json"];
  const named = r ? r.failures.slice(0, 2).map((f) => `<span class="mono">${esc(f.episode_id)}</span> (${esc(f.reason)})`) : [];
  const more = r && r.failures.length > 2 ? ` and ${r.failures.length - 2} more` : "";
  const failed = named.length ? ` <span class="note">· not reproduced: ${named.join(", ")}${more}</span>` : "";
  const replay = !r ? missing("replay_report.json") : r.replayed ? `${r.reproduced} of ${plural(r.replayed, "accepted episode")} reproduced${failed}` : "No accepted episode to replay.";
  const rounds = s.trained.length ? s.trained.map(round).join("") : '<dt>Training</dt><dd class="ph-text">No training set built yet.</dd>';
  $("p-training").innerHTML = `<header><h2>Training and replay</h2>${panelChip()}</header><dl class="body files">${rounds}<dt>Replay</dt><dd>${replay}</dd></dl>`;
}

// A run quiet for 30 minutes writes no more files, so its files are read once.
async function readRunFiles() {
  const s = MomState.shown(view);
  if (!runId || (Object.keys(runFiles).length && now() - s.lastT > 30 * 60)) return;
  const wanted = [...s.trained.map((r) => [`training_r${r}.summary.json`, MomState.trainingSummary]), ["replay_report.json", MomState.replayReport]];
  const asked = await Promise.all(
    wanted
      .filter(([name]) => !runFiles[name])
      .map(async ([name, parse]) => {
        const response = await fetch(`/run-files/${encodeURIComponent(runId)}/${name}`).catch(() => null);
        runFiles[name] = response && response.ok ? parse(await response.json().catch(() => null)) : null;
      }),
  );
  if (asked.length) schedule();
}

// Clamped to four lines, so a long one never pushes the side band past its fold; the whole of it is in the title.
const evidenceText = (d) => (d.evidence ? `<div class="note clamped four" title="${esc(d.evidence)}">${esc(d.evidence)}</div>` : "");
const confidenceText = (d) => (d.confidence === null ? "" : `<div class="note">Confidence ${Math.round(d.confidence * 100)}%.</div>`);
// Clamped to three lines, so the evidence under it always shows.
const hypothesisText = (d) => `<div class="clamped" title="${esc(d.hypothesis)}">${esc(d.hypothesis)}</div>`;
const diagnosisText = (d) => `${hypothesisText(d)}${evidenceText(d)}${confidenceText(d)}`;

// A teacher demonstration shows the diagnosis it acts on, named by the diagnosed episode, with that episode's last frame.
function actingOnBody(a) {
  if (a.number === null) return `<div class="body ph-text">Acting on a diagnosis of episode <span class="mono">${esc(a.id)}</span>, which is not in this run's events.</div>`;
  const head = `<p class="acting-on">Acting on diagnosis from episode ${a.number}</p>`;
  if (!a.diagnosis) return `<div class="body stack">${head}<div class="ph-text">Episode ${a.number} has no diagnosis in the events yet.</div></div>`;
  const shot = a.frame && a.frame.url
    ? `<div class="cited"><img src="${esc(a.frame.url)}" alt="Episode ${a.number}'s last head-camera frame"></div>`
    : "";
  const last = shot && a.frame.step !== null ? `<div class="note">Its last frame, step ${esc(a.frame.step)}</div>` : "";
  // The frame and who is acted on lead, so a long hypothesis never pushes the evidence out of view.
  const lead = shot ? `<div class="cited-row">${shot}<div class="stack">${head}${last}${confidenceText(a.diagnosis)}</div></div>` : head + confidenceText(a.diagnosis);
  return `<div class="body stack">${lead}${hypothesisText(a.diagnosis)}${evidenceText(a.diagnosis)}</div>`;
}

function renderDiagnosis() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const acting = MomState.actedOn(s, ep);
  const d = ep && ep.diagnosis;
  const body = acting
    ? actingOnBody(acting)
    : d
      ? `<div class="body stack">${diagnosisText(d)}</div>`
      : `<div class="body ph-text">${ep ? "No diagnosis for this episode. The teacher diagnoses the student's failed trials, then demonstrates on those seeds first and new ones after." : "No episode yet."}</div>`;
  $("p-diagnosis").innerHTML = `<header><h2>Diagnosis</h2>${panelChip(ep)}</header>${body}`;
}

function render() {
  renderQueued = false;
  const focused = document.activeElement;
  renderHeader();
  defaultSide();
  renderStrip();
  renderSeed();
  const said = renderControl();
  renderEpisode();
  renderCamera();
  announce(said);
  $("p-rounds").innerHTML = heldOut();
  renderModels();
  renderDiagnosis();
  renderTraining();
  // Last, so its scroll lands after every bar that sets its height is written.
  renderLog();
  // A new episode starts the side column at its top, where the held-out figure is.
  const ep = MomState.episode(view, now());
  if (ep && ep.id !== renderedSide) $("side").scrollTop = 0;
  renderedSide = ep && ep.id;
  fadeSide();
  keepFocus(focused);
  placeFocus();
  // Every live dot pulses in step, a rewritten one included, from the timeline's start.
  for (const a of document.getAnimations()) if (a.animationName === "pulse") a.startTime = 0;
}

let renderedPlace = null;
let renderedSeed = null;
let seedMoved = false;
let leftEpisode = null;
// Opening a seed, an episode or going back moves the focus with the view; back on the run, to the seed's column.
function placeFocus() {
  const place = MomState.where(view);
  const moved = seedMoved || (renderedPlace !== null && place !== renderedPlace);
  const left = renderedSeed;
  renderedPlace = place;
  renderedSeed = view.seed;
  seedMoved = false;
  if (view.opened !== null) leftEpisode = view.opened;
  // A seed chosen from the strip keeps the focus there, so ← and → go on through its columns.
  if (!moved || (document.activeElement && document.activeElement.closest && document.activeElement.closest("#columns"))) return;
  const at = (selector) => $("results").querySelector(selector);
  const to =
    place === "seed"
      ? (leftEpisode !== null && at(`.card[data-open="${CSS.escape(leftEpisode)}"]`)) || [...$("results").querySelectorAll(".card")].pop() || at(".back")
      : place === "episode"
        ? $("control").querySelector(".back")
        : left !== null && $("columns").querySelector(`[data-column="${CSS.escape(String(left))}"]`);
  if (to) to.focus();
}

// Focus on a control the render disabled, hid or rewrote would fall to <body>, so a nearby control takes it.
function keepFocus(was) {
  if (!was || was === document.body || (was.isConnected && !was.disabled && was.checkVisibility())) return;
  const again = was.dataset.f && document.querySelector(`[data-f="${CSS.escape(was.dataset.f)}"]`);
  if (again && !again.disabled && again.checkVisibility()) return again.focus();
  const keys = [...document.querySelectorAll(".presenter button")];
  const at = keys.indexOf(was);
  if (at < 0) return $("b-m").focus();
  for (let d = 1; d < keys.length; d++) for (const key of [keys[at + d], keys[at - d]]) if (key && !key.disabled) return key.focus();
}

// The side column fades at its foot while more of it lies below.
function fadeSide() {
  const side = $("side");
  side.toggleAttribute("data-more", side.scrollHeight - side.scrollTop - side.clientHeight > 1);
}

// While a connection replays the events already written the page holds still (MomState.settle).
function schedule() {
  if (renderQueued || view.settling) return;
  renderQueued = true;
  requestAnimationFrame(render);
}

const RUN_STATUS = { live: ["live", "Live"], recorded: ["recorded", "Recorded"], not_started: ["refused", "Not started"] };
function renderRuns() {
  $("runs").hidden = false;
  $("run-name").textContent = runId || "No runs yet";
  $("run-name").title = runId || "";
  $("run-list").innerHTML = runs.length
    ? runs
        .map((r) => {
          const [kind, word] = RUN_STATUS[r.status];
          const what = r.instruction ? `<span class="run-what clamped two" dir="auto" title="${esc(r.instruction)}">${esc(r.instruction)}</span>` : '<span class="run-what ph-text">No instruction recorded</span>';
          const why = r.status === "not_started" && r.reason ? `<span class="run-why clamped two" title="${esc(r.reason)}">${esc(r.reason)}</span>` : "";
          const here = r.run_id === runId ? ' aria-current="page"' : "";
          return `<li><a href="?run=${esc(encodeURIComponent(r.run_id))}"${here}><span class="chip ${kind}">${word}</span>${what}<span class="run-id mono">${esc(r.run_id)}</span>${why}</a></li>`;
        })
        .join("")
    : '<li class="ph-text">No runs yet. Tell the robot what to learn in the Teach box to start one.</li>';
}

async function loadRuns() {
  const response = await fetch("/runs").catch(() => null);
  if (!response || !response.ok) return null;
  const date = Date.parse(response.headers.get("date"));
  if (date) skew = date / 1000 - Date.now() / 1000;
  runs = MomState.runsOf(await response.json().catch(() => null));
  if (opened && !runId && runs.length) location.replace(`?run=${encodeURIComponent(MomState.runToOpen(runs))}`);
  renderRuns();
  return runs;
}

const TEACH = $("teach");
const FIELD = $("instruction");
function renderTeach() {
  const checking = teaching.state === "checking";
  // Past the limit the count says so and Teach stays off until the text is shortened, so nothing is cut unseen.
  const over = MomState.tooLong(FIELD.value);
  $("teach-count").textContent = MomState.teachCount(FIELD.value) || "";
  FIELD.readOnly = checking;
  FIELD.setAttribute("aria-invalid", String(over));
  $("teach-go").disabled = checking || over;
  $("teach-go").dataset.tip = checking ? "Checking the instruction…" : over ? "At most 200 characters" : "Teach · Enter";
  TEACH.setAttribute("aria-busy", String(checking));
  const said =
    teaching.state === "refused"
      ? `<p class="what">Not started</p><ul>${teaching.reasons.map((r) => `<li dir="auto">${esc(r)}</li>`).join("")}</ul><p class="note">The run is listed as not started, with why.</p>`
      : teaching.state === "failed"
        ? `<p class="what">Not sent</p><p>${esc(teaching.message)}</p>`
        : teaching.state === "started"
          ? '<p class="what">Started</p><p class="note">Opening the new run.</p>'
          : "";
  $("teach-said").innerHTML = said;
  $("teach-said").hidden = !said;
  const status = checking ? "Checking the instruction." : over ? "At most 200 characters." : $("teach-said").innerText;
  // Said once as the text goes past the limit, not again on every key typed past it.
  if ($("teach-status").textContent !== status) $("teach-status").textContent = status;
}

TEACH.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (teaching.state === "checking") return;
  const problem = MomState.instructionProblem(FIELD.value);
  teaching = problem ? { state: "failed", message: problem } : { state: "checking" };
  renderTeach();
  if (problem) return;
  const response = await fetch("/teach", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ instruction: FIELD.value.trim() }) }).catch(() => null);
  teaching = MomState.teachAnswer(response ? response.status : 0, response && (await response.json().catch(() => null)));
  renderTeach();
  if (teaching.state === "started") location.assign(`?run=${encodeURIComponent(teaching.runId)}`);
  else loadRuns();
});
FIELD.addEventListener("input", () => {
  if (teaching.state !== "checking") teaching = { state: "idle" };
  renderTeach();
});

const POPOVERS = [...document.querySelectorAll("details.legend, details.runs")];
const dismissTeach = () => {
  if (teaching.state === "checking" || teaching.state === "idle") return;
  teaching = { state: "idle" };
  renderTeach();
};
const closePopover = () => {
  const open = POPOVERS.find((d) => d.open);
  if (!open) return false;
  open.open = false;
  open.querySelector("summary").focus();
  return true;
};
document.addEventListener("click", (e) => {
  for (const d of POPOVERS) if (d.open && !d.contains(e.target)) d.open = false;
  if (!TEACH.contains(e.target)) dismissTeach();
});
// A click on a popover's own text moves the focus to no element, so only a move to another element closes it.
for (const d of POPOVERS)
  d.addEventListener("focusout", (e) => {
    if (e.relatedTarget && !d.contains(e.relatedTarget)) d.open = false;
  });

$("side").addEventListener("scroll", fadeSide);
$("log").addEventListener("scroll", (e) => {
  const log = e.currentTarget;
  logFollowing = log.scrollHeight - log.scrollTop - log.clientHeight < 8 || Math.abs(log.scrollTop - logAnchor) < 8;
});

// A frame that fails to load is marked missing instead of showing a broken-image icon.
document.addEventListener(
  "error",
  (e) => {
    if (e.target.tagName === "IMG" && e.target.parentElement) e.target.parentElement.dataset.missing = "";
  },
  true,
);

function press(e) {
  const place = MomState.where(view);
  if (!MomState.keydown(view, e, now())) return false;
  seedMoved = place === "seed" && e.key.startsWith("Arrow");
  schedule();
  return true;
}
// ← and → move between the strip's columns while the focus is in it; Enter on one is its button's click.
function stripKey(e) {
  const column = e.target && e.target.closest && e.target.closest("[data-column]");
  if (!column || (e.key !== "ArrowLeft" && e.key !== "ArrowRight") || e.metaKey || e.ctrlKey || e.altKey) return false;
  const to = MomState.neighbour(MomState.shown(view), Number(column.dataset.column), e.key === "ArrowLeft" ? -1 : 1);
  if (to !== null) {
    stripFocus = to;
    renderStrip();
    $("columns").querySelector(`[data-column="${CSS.escape(String(to))}"]`).focus();
  }
  return true;
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    dismissTeach();
    if (closePopover()) return;
  }
  if (stripKey(e) || press(e)) e.preventDefault(); // also keeps a focused button from taking Space as a click
});
// Some browsers click a focused button on Space's keyup; the keydown above already acted.
document.addEventListener("keyup", (e) => e.key === " " && e.preventDefault());
$("columns").addEventListener("click", (e) => {
  const column = e.target.closest("button[data-column]");
  if (!column) return;
  stripFocus = Number(column.dataset.column);
  MomState.visit(view, MomState.showSeed(view, stripFocus));
  schedule();
});
$("sides").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-side]");
  if (!button || view.seed === null) return;
  chosen = true;
  MomState.choose(view, button.dataset.side);
  schedule();
});
$("results").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  if (button.dataset.open) MomState.visit(view, { opened: button.dataset.open });
  else if ("back" in button.dataset) MomState.keydown(view, { key: "Escape" }, now());
  else return;
  schedule();
});
$("control").addEventListener("click", (e) => {
  if (!e.target.closest("button[data-back]")) return;
  MomState.keydown(view, { key: "Escape" }, now());
  schedule();
});
// Both or After follows the window's width until a choice is made.
addEventListener("resize", schedule);
document.querySelector(".presenter").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-key]");
  if (button) press({ key: button.dataset.key });
});

function follow() {
  const source = new EventSource(`/events?run=${encodeURIComponent(runId)}`);
  // The server replays the whole run on every connection, so the view starts over instead of adding to what is shown.
  source.onopen = () => {
    MomState.reconnect(view);
    renderedRows = 0;
    renderedEpisode = null;
  };
  source.onerror = () => {
    MomState.disconnect(view);
    schedule();
  };
  // The server marks the end of its replay, so the page shows and announces only where the replay settles.
  source.addEventListener("replayed", () => {
    MomState.settle(view);
    schedule();
    readRunFiles();
  });
  source.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data);
      MomState.receive(view, event, now());
      // A round's training-set summary is written just before its fine-tune starts.
      if (event.type === "finetune_started" && !view.settling) readRunFiles();
    } catch (error) {
      console.error("bad event", error);
    }
    schedule();
  };
}

let opened = false;
(async () => {
  const listed = await loadRuns();
  if (!runId) {
    runId = MomState.runToOpen(listed || []);
    if (runId) history.replaceState(null, "", `?run=${encodeURIComponent(runId)}`);
    if (listed) renderRuns();
  }
  opened = true;
  if (runId) return follow();
  MomState.settle(view);
  schedule();
})();
setInterval(loadRuns, 3000);
setInterval(schedule, 1000); // live and recorded change with time, and the fine-tune clock ticks
setInterval(readRunFiles, 10000); // a new round's summary and the replay report are written while the page is open
schedule();
