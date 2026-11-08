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
let lastHeard = { text: "", episode: null, count: 0, paused: false, lost: false };
// A one-off line for the announcer: the box or bin move, heard once when it arrives live.
let news = "";
// Saved run files the events don't carry (SPEC §3.5): each trained round's training-set summary and the replay
// report, by name; null once asked for and absent. A file once written never changes, so each is read until found.
const runFiles = {};
// Configuration (SPEC §3.2, §3.3), not data: the page shows these labelled as configured.
const CONFIGURED = { student: "SmolVLM2-500M, local", teacher: "Argon (Gemini API)", finetune: "RTX 5070, local" };

const $ = (id) => document.getElementById(id);
// The run this page shows (SPEC §3.5): the URL names it; without one, the first in GET /runs' order.
let runId = new URLSearchParams(location.search).get("run");
let runs = [];
let teaching = { state: "idle" };
// Assigning identical HTML would reload every image and reset scrolling, so panels skip unchanged writes.
const nativeSetter = Object.getOwnPropertyDescriptor(Element.prototype, "innerHTML").set;
for (const el of document.querySelectorAll("[id]")) {
  Object.defineProperty(el, "innerHTML", {
    set(html) {
      if (el.dataset.html === html) return;
      el.dataset.html = html;
      nativeSetter.call(el, html);
    },
  });
}
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const X = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>';
const ARROW = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M1.5 6h8M6.5 2.5L10 6l-3.5 3.5" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const DIAMOND = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1.5L10.5 6 6 10.5 1.5 6z" fill="currentColor"/></svg>';
const BANG = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2v5M6 9.6v.4" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>';
const CHECK = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.6 2.6L10 3" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const MOVE = '<svg class="i" aria-hidden="true"><use href="#i-move"/></svg>';

const now = () => Date.now() / 1000;
const label = (kind) => kind[0].toUpperCase() + kind.slice(1);
const chip = (kind, text = label(kind)) => `<span class="chip ${kind}">${text}</span>`;
// A panel's label, quiet beside the header's: what the view shows of the run or of the panel's episode. A panel with
// nothing to show says so in words, so placeholder means only a run that is not on a real simulator.
const panelChip = (ep = null) => {
  const kind = MomState.provenance(view, now(), ep);
  return `<span class="chip quiet ${kind}">${label(kind)}</span>`;
};
const WHO = { student: "Student", teacher: "Teacher (Argon)" };
const tag = (controller, checkpoint) => {
  const who = WHO[controller] || "Unknown controller";
  return `<span class="tag ${controller || "unknown"}">${who}${checkpoint ? ` <span class="mono">${esc(checkpoint)}</span>` : ""}</span>`;
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
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const verdictChip = (ep) => `<span class="verdict ${MomState.accepted(ep) ? "good" : "bad"}">${MomState.accepted(ep) ? CHECK : X}${esc(MomState.verdictWords(ep))}</span>`;
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
        : chip(kind, `${label(kind)}${sim && sim.name ? ` · ${esc(sim.name)}` : ""}`);
  $("status").innerHTML = status;
  const robot = s.episodes.length && s.episodes[s.episodes.length - 1].sim;
  const task = sim && sim.task ? `, the ${sim.task === "fixed_feet" ? "fixed-feet" : "walking"} task (<span class="mono">${esc(sim.task)}</span>)` : "";
  const on = !sim ? "" : sim.version ? ` on <span class="mono">${esc(sim.version)}</span>${task}` : ", which names no simulator";
  $("run-meta").innerHTML = s.eventCount
    ? `Run <b class="mono">${esc(s.runId)}</b>${on}${robot ? `, robot <span class="mono">${esc(robot.robot)}</span>` : ""}: ${s.episodes.length} episode${s.episodes.length === 1 ? "" : "s"}, ${s.eventCount} events.`
    : "";
}

// Screen readers hear the episode when the view moves to another, who is in control and the verdict when either
// changes, and a pause or resume; not every step.
function announce(text) {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const heard = { text, episode: ep && ep.id, count: s.episodes.length, paused: view.frozen !== null, lost: view.lost };
  const parts = [];
  if (heard.paused !== lastHeard.paused) parts.push(heard.paused ? "Paused." : "Resumed.");
  if (heard.lost !== lastHeard.lost) parts.push(heard.lost ? "Connection to the run lost, retrying." : "Connection restored.");
  if (ep && (heard.episode !== lastHeard.episode || heard.count !== lastHeard.count)) parts.push(`Episode ${s.episodes.indexOf(ep) + 1} of ${s.episodes.length}, ${kindOf(ep)[0].toLowerCase()}.`, text);
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
  if (!ep) {
    const s = MomState.shown(view);
    el.className = "control";
    el.innerHTML = !runId
      ? '<span class="what">No run yet</span>'
      : s.refused !== null
        ? `<span class="what">Not started</span><span class="note">${esc(s.refused)}</span>`
        : '<span class="what">Waiting for the run to start</span>';
    return "";
  }
  const [kind, where] = kindOf(ep);
  const head = `<span class="kind"><span class="what">${kind}</span><span class="note">${where}</span></span>`;
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
  const clock = `<span class="tag ink">${view.lost ? "Connection lost · " : ""}${elapsed(seconds)} since the fine-tune started</span>`;
  const sheet = shown.length
    ? `<div class="frame"><div class="cell"><div class="shot sheet"><ol style="--n:${Math.ceil(Math.sqrt(shown.length))}" aria-label="The last frame of each accepted teacher demonstration">${shown
        .map((f) => `<li${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="Episode ${f.number}'s last frame">` : ""}</li>`)
        .join("")}</ol><div class="over">${clock}</div></div></div></div>`
    : slate("No accepted demonstration to learn from", `${clock} Teacher demonstrations: ${l.accepted} accepted · ${l.rejected} rejected (n=${l.n}).`);
  const latest = l.frames.length > shown.length ? `, the latest ${shown.length} of ${l.frames.length} shown` : "";
  // The counts lead and the line wraps, so no width ever cuts a number off.
  const caption = shown.length
    ? `<p class="caption wrap" title="The last frame of each accepted teacher demonstration${latest}.">Learning from ${l.accepted} accepted teacher demonstration${l.accepted === 1 ? "" : "s"} (${l.rejected} rejected, n=${l.n})</p>`
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
        : slate("No episode yet", "The head camera's frames appear here once the run starts.");
    $("stage").innerHTML = (ep ? slate(`${kind} · ${where}`, note) : empty) + foot("", panelChip(ep));
    $("strip").innerHTML = "";
    return;
  }
  const last = frames.find((f) => f.now);
  const move = MomState.move(ep);
  const what = move && esc(MOVED[ep.rows.findLast((r) => r.kind === "perturbation").what] || "Scene changed");
  // The caption row is always there, so the peak's caption never shrinks the frame.
  let frame, caption;
  if (view.compare && move) {
    // The orchestrator took the left frame at the same pose right before the move, so no skill was chosen from it.
    const who = (f) => (f ? tag(f.controller, f.checkpoint) : "");
    frame = cell(move.before, `<span class="tag">Before the move</span>${who(move.before)}`, null, false) + cell(move.after, `<span class="tag">After</span>${who(move.after)}`, skillRow(ep, move.after), true);
    // The pair covers the log, so the caption carries its marked row, clamped so it never shrinks the pair.
    const first = move.decision;
    const next = first ? ` First decision after the move: <span class="mono">${skillText(first)}</span>.${first.reason ? ` ${esc(first.reason)}` : ""}` : "";
    caption = `<p class="caption moved clamped"${first && first.reason ? ` title="${esc(first.reason)}"` : ""}>${what} between these frames, before step ${esc(move.step)}.${next}</p>`;
  } else {
    // The move stays marked for the rest of its episode, with Move's icon, so the presenter can show it after the moment has passed.
    const mark = last.after ? `<span class="tag ink">${what} at step ${esc(move.step)}${MOVE}</span>` : "";
    frame = cell(last, tag(last.controller, last.checkpoint), skillRow(ep, last), last.moved, mark);
    caption = last.moved
      ? `<p class="caption moved">${what} before step ${esc(move.step)}. This is the first frame after it.</p>`
      : `<p class="caption">${last.step === null ? "Head-camera frame." : `What the ${last.controller || "robot"} saw at step ${esc(last.step)}, before choosing.`}</p>`;
  }
  $("stage").innerHTML = `<div class="frame">${frame}</div>${foot(caption, panelChip(ep))}`;
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

// One frame on the stage: its labels, the skill `row` chose from it, then `mark`; a missing frame stays, hatched.
function cell(f, label, row, moved, mark = "") {
  const shot = f && f.url ? `<img src="${esc(f.url)}" alt="Head camera frame${f.step === null ? "" : ` at step ${esc(f.step)}`}">` : "";
  // The skill joins the labels in the frame's top band, which shows floor; the hands, box and bin are in the bottom.
  const said = row ? `<span class="said ${row.controller || "unknown"}">${skillText(row)}</span>` : "";
  return `<div class="cell"><div class="shot${moved ? " moved" : ""}"${shot ? "" : " data-missing"}>${shot}<div class="over">${label}${said}${mark}</div></div></div>`;
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
  const good = MomState.accepted(ep);
  const reason = v.reason ? `${esc(cap(v.reason))}. ` : "";
  return `<li class="event ${good ? "good" : "bad"}"><span class="n">${good ? CHECK : X}</span><span>${esc(MomState.verdictWords(ep))}. <span class="d">${reason}Task ${v.task ? "succeeded" : "not met"}, physics ${v.physics ? "valid" : "violated"}.</span></span></li>`;
}

const NAVIGATION = { "b-prev": "ArrowLeft", "b-next": "ArrowRight", "b-d": "d", "b-t": "t", "b-m": "m", "b-end": "End" };
const PAUSE = $("b-pause");

function renderEpisode() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const at = s.episodes.indexOf(ep);
  // The connection note sits beside the rail, which gives way, so the keys above never wrap.
  const lost = view.lost ? ' <span class="lost">· connection lost, retrying</span>' : "";
  $("episode-line").innerHTML = (ep ? `Episode ${at + 1} of ${s.episodes.length}` : "") + lost;
  for (const [id, key] of Object.entries(NAVIGATION)) $(id).disabled = MomState.target(view, now(), key) < 0;
  $("b-m").setAttribute("aria-pressed", String(view.compare !== null));
  // The visible label is the name; the key stays inverse while paused.
  PAUSE.classList.toggle("on", view.frozen !== null);
  PAUSE.lastElementChild.textContent = MomState.pauseLabel(view);
  PAUSE.dataset.tip = `${view.frozen ? "Resume" : "Pause"} · Space`;
  // Where the view is among the run's episodes: tall for a teacher, short for the student; hollow when it failed or was rejected.
  $("rail").style.setProperty("--n", s.episodes.length);
  $("rail").innerHTML = s.episodes
    .map((e) => `<li class="${MomState.teacherDrove(e) ? "teacher" : "student"} ${e.verdict ? (MomState.accepted(e) ? "accepted" : "rejected") : "open"}${e === ep ? " here" : ""}"></li>`)
    .join("");
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
  // Move hides the log in the two-column layout; letting go of it drops the scroll it set, so the log follows new rows again.
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

function renderRounds() {
  const s = MomState.shown(view);
  const c = MomState.claim(s);
  const test = MomState.testing(s);
  // A run that stopped arriving mid-evaluation can't say the test still runs.
  const live = MomState.following(view, now()) && !view.lost;
  const progress = test && `Testing ${test.checkpoint ? `<span class="mono">${esc(test.checkpoint)}</span>` : "the student"}: ${test.run}${test.of === null ? " run so far" : ` of ${test.of} run`}${live ? "" : " when its events stopped"}.`;
  // First and latest only: the change is the claim.
  const errored = c && c.errored ? ` · ${esc(c.errored)}` : "";
  const gain = c && `<div class="gain"><span class="fig" role="img" aria-label="Held-out successes: ${esc(c.spoken)}">${esc(c.figure)}</span><span>${esc(c.sample)}${errored}</span></div><div>${esc(c.names)}</div><div class="note mono">${esc(c.ids)}</div>`;
  const body = gain ? gain + (progress ? `<div class="note">${progress}</div>` : "") : `<div class="ph-text">${progress || "No held-out evaluation in the events yet."}</div>`;
  // A run is taught and tested on its own instruction, so its rates carry it (SPEC §3.5).
  const said = s.instruction && s.instruction.text;
  const taught = said ? `<p class="taught clamped" title="${esc(said)}">Tested on <q>${esc(said)}</q></p>` : "";
  $("p-rounds").innerHTML = `<header><h2>Held-out success</h2>${panelChip()}</header><div class="body stack">${body}${taught}</div>`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// One line: each model, where it runs (configured), and its measured median latency or fine-tune duration; then the
// Argon strip, the run's calls against its budget (rule 5).
function renderModels() {
  const shown = MomState.shown(view);
  const s = MomState.summary(shown);
  const a = MomState.argon(shown);
  const item = (who, where, measured) => `<span class="item"><b>${who}</b> ${where} <span class="v">${stat(measured)}</span></span>`;
  const calls = a
    ? `${a.used}${a.budget === null ? "" : ` of ${a.budget}`} <span class="v">· answers ${stat(a.answer)} · ${a.failed} failed</span>`
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

// Each round's training set and the replay of the run's accepted episodes, from the run's saved files (SPEC §3.5).
// A file not written yet says so in words.
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
  // The first two that failed, by episode and why; the rest counted.
  const named = r ? r.failures.slice(0, 2).map((f) => `<span class="mono">${esc(f.episode_id)}</span> (${esc(f.reason)})`) : [];
  const more = r && r.failures.length > 2 ? ` and ${r.failures.length - 2} more` : "";
  const failed = named.length ? ` <span class="note">· not reproduced: ${named.join(", ")}${more}</span>` : "";
  const replay = !r ? missing("replay_report.json") : r.replayed ? `${r.reproduced} of ${plural(r.replayed, "accepted episode")} reproduced${failed}` : "No accepted episode to replay.";
  const rounds = s.trained.length ? s.trained.map(round).join("") : '<dt>Training</dt><dd class="ph-text">No training set built yet.</dd>';
  $("p-training").innerHTML = `<header><h2>Training and replay</h2>${panelChip()}</header><dl class="body files">${rounds}<dt>Replay</dt><dd>${replay}</dd></dl>`;
}

// Reads the run's saved files not found yet, while the run may still write them: its last event under 30 minutes ago.
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
  const said = renderControl();
  renderEpisode();
  renderCamera();
  announce(said);
  renderRounds();
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
}

// A key that disabled itself, or Move hiding a focused panel, would drop focus to <body>: the nearest enabled
// presenter key takes it instead, or Move's own key.
function keepFocus(was) {
  if (!was || was === document.body || (!was.disabled && was.checkVisibility())) return;
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
// The run picker: every run in the server's order, live first; a run that did not start says why.
function renderRuns() {
  $("runs").hidden = false;
  $("run-name").textContent = runId || "No runs yet";
  $("run-list").innerHTML = runs.length
    ? runs
        .map((r) => {
          const [kind, word] = RUN_STATUS[r.status];
          const what = r.instruction ? esc(r.instruction) : '<span class="ph-text">No instruction recorded</span>';
          const why = r.status === "not_started" && r.reason ? `<span class="run-why">${esc(r.reason)}</span>` : "";
          const here = r.run_id === runId ? ' aria-current="page"' : "";
          return `<li><a href="?run=${esc(encodeURIComponent(r.run_id))}"${here}><span class="chip ${kind}">${word}</span><span class="run-what">${what}</span><span class="run-id mono">${esc(r.run_id)}</span>${why}</a></li>`;
        })
        .join("")
    : '<li class="ph-text">No runs yet. Tell the robot what to learn in the Teach box to start one.</li>';
}

// The runs list, refreshed every few seconds; null while the server doesn't answer. A page with no run opens the
// first one to appear.
async function loadRuns() {
  const response = await fetch("/runs").catch(() => null);
  if (!response || !response.ok) return null;
  runs = MomState.runsOf(await response.json().catch(() => null));
  if (opened && !runId && runs.length) location.replace(`?run=${encodeURIComponent(runs[0].run_id)}`);
  renderRuns();
  return runs;
}

// The Teach box's count, its state while the generator judges the text, and what came back.
const TEACH = $("teach");
const FIELD = $("instruction");
function renderTeach() {
  const checking = teaching.state === "checking";
  $("teach-count").textContent = MomState.teachCount(FIELD.value) || "";
  FIELD.readOnly = checking;
  $("teach-go").disabled = checking;
  $("teach-go").lastElementChild.textContent = checking ? "Checking…" : "Teach";
  TEACH.setAttribute("aria-busy", String(checking));
  const said =
    teaching.state === "refused"
      ? `<p class="what">Not started</p><ul>${teaching.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul><p class="note">The run is listed as not started, with why.</p>`
      : teaching.state === "failed"
        ? `<p class="what">Not sent</p><p>${esc(teaching.message)}</p>`
        : teaching.state === "started"
          ? '<p class="what">Started</p><p class="note">Opening the new run.</p>'
          : "";
  $("teach-said").innerHTML = said;
  $("teach-said").hidden = !said;
  $("teach-status").textContent = checking ? "Checking the instruction." : $("teach-said").textContent;
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

// Labels and the run picker close on Escape or a click outside them, as popovers do; so does the Teach answer.
const POPOVERS = [...document.querySelectorAll("details.legend, details.runs")];
const dismissTeach = () => {
  if (teaching.state === "checking" || teaching.state === "idle") return;
  teaching = { state: "idle" };
  renderTeach();
};
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  dismissTeach();
  const open = POPOVERS.find((d) => d.open);
  if (!open) return;
  open.open = false;
  open.querySelector("summary").focus();
});
document.addEventListener("click", (e) => {
  for (const d of POPOVERS) if (d.open && !d.contains(e.target)) d.open = false;
  if (!TEACH.contains(e.target)) dismissTeach();
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

// Presenter controls (SPEC §3.5): the same actions from the keyboard, whatever has focus, and from the buttons that label them.
document.addEventListener("keydown", (e) => {
  if (!MomState.keydown(view, e, now())) return;
  e.preventDefault(); // also keeps a focused button from taking Space as a click
  schedule();
});
// Some browsers click a focused button on Space's keyup; the keydown above already acted.
document.addEventListener("keyup", (e) => e.key === " " && e.preventDefault());
document.querySelector(".presenter").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-key]");
  if (!button) return;
  MomState.keydown(view, { key: button.dataset.key }, now());
  schedule();
});

// The shown run's events (SPEC §3.5).
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
      MomState.receive(view, JSON.parse(message.data), now());
    } catch (error) {
      console.error("bad event", error);
    }
    schedule();
  };
}

// A page without a run in its URL opens the first listed and names it in the URL; with none listed, it says so.
let opened = false;
(async () => {
  const listed = await loadRuns();
  if (!runId) {
    runId = MomState.runToOpen(listed || [], null);
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
