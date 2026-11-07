// Renders MomState over the /events stream. Every value from an event is escaped; nothing is shown that an event did not carry.
const view = MomState.view();
let renderQueued = false;
let renderedRows = 0;
let renderedEpisode = null;
let renderedFocus = null;
let renderedSide = null;
// The log follows new rows until the reader scrolls up from its bottom, and again once they scroll back down.
let logFollowing = true;
let flashedFrame = null;
let announced = "";
// Configuration (SPEC §3.2, §3.3), not data: the page shows these labelled as configured.
const CONFIGURED = { student: "SmolVLM2-500M, local", teacher: "Gemini API", finetune: "Modal A10G" };

const $ = (id) => document.getElementById(id);
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

const now = () => Date.now() / 1000;
const label = (kind) => kind[0].toUpperCase() + kind.slice(1);
const chip = (kind, text = label(kind)) => `<span class="chip ${kind}">${text}</span>`;
// The label for a panel: placeholder when it has no data, else what the view shows of the run (or its episode).
const panelChip = (hasData, ep = null) => chip(hasData ? MomState.provenance(view, now(), ep) : "placeholder");
const simName = (s) => {
  const sims = s.episodes.map((e) => e.sim);
  if (sims.some((sim) => sim === null)) return "simulator not stated";
  if (sims.some((sim) => sim.version.startsWith("fake"))) return "FakeSim";
  return esc(sims[sims.length - 1].version);
};
const WHO = { student: "Student", teacher: "Teacher (Argon)" };
const tag = (controller, checkpoint) => {
  const who = WHO[controller] || "Unknown controller";
  return `<span class="tag ${controller || "unknown"}">${who}${checkpoint ? ` <span class="mono">${esc(checkpoint)}</span>` : ""}</span>`;
};
const amount = (row) => (row.distance !== null ? ` ${+row.distance.toFixed(1)} cm` : row.angle !== null ? ` ${Math.round(row.angle)}°` : "");
const skillText = (row) => (row.skill ? esc(row.skill) + amount(row) : '<span class="stop">Stopped, no skill chosen</span>');
// What an episode is, named without the loop's round index, which reads like a checkpoint number and is not one.
// An episode is over once judged, or once the view has moved past it or the run has stopped.
const finished = (ep) => ep.verdict !== null || ep !== MomState.current(MomState.shown(view)) || !MomState.following(view, now());
const kindOf = (ep) => {
  if (ep.split === "heldout") return ["Held-out test", `seed ${esc(ep.seed ?? "")}`];
  const first = ep.rows.find((r) => r.kind === "decision");
  return [first && first.controller === "teacher" ? "Teacher demonstration" : "Student trial", `development seed ${esc(ep.seed ?? "")}`];
};
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const isAccepted = (v) => v.task && v.physics;
const verdictChip = (v) => `<span class="verdict ${isAccepted(v) ? "good" : "bad"}">${isAccepted(v) ? CHECK + "Accepted" : X + "Rejected"}</span>`;
const MOVED = { move_box: "Box moved", move_bin: "Bin moved" };
// Durations read at a glance from the back of the room: ms under a second, then seconds, then minutes.
const duration = (v) => (v < 1000 ? `${v >= 100 ? Math.round(v) : v.toFixed(1)} ms` : v < 120000 ? `${(v / 1000).toFixed(1)} s` : `${(v / 60000).toFixed(1)} min`);
// A running clock in whole seconds: "45 s", "2 min 05 s".
const elapsed = (seconds) => {
  const whole = Math.max(0, Math.floor(seconds));
  return whole < 60 ? `${whole} s` : `${Math.floor(whole / 60)} min ${String(whole % 60).padStart(2, "0")} s`;
};
const pct = (k, n) => (n ? Math.round((100 * k) / n) + "%" : "n/a");
const stat = (s) => (s ? `${duration(s.median)} <span class="note">n=${s.n}</span>` : '<span class="ph-text">no data</span>');

function renderHeader() {
  const s = MomState.shown(view);
  const kind = MomState.provenance(view, now());
  const status = s.eventCount === 0 ? chip("placeholder", "Waiting") : chip(kind, `${label(kind)} · ${simName(s)}`);
  $("status").innerHTML = status;
  const robot = s.episodes.length && s.episodes[s.episodes.length - 1].sim;
  $("run-meta").innerHTML = s.eventCount
    ? `Run <b class="mono">${esc(s.runId)}</b>${robot ? `, robot <span class="mono">${esc(robot.robot)}</span>` : ""}: ${s.episodes.length} episode${s.episodes.length === 1 ? "" : "s"}, ${s.eventCount} events.`
    : "";
}

// Screen readers hear who is in control and the verdict when either changes, not every step.
function announce(text) {
  if (text !== announced) $("announce").textContent = announced = text;
}

function renderControl() {
  const ep = MomState.episode(view, now());
  const el = $("control");
  if (MomState.finetuning(view, now()) !== null) {
    el.className = "control";
    el.innerHTML = `<span class="what">Fine-tuning the student</span><span class="note">No robot is moving. The held-out test of the new checkpoint starts when it is ready.</span>`;
    announce("Fine-tuning the student.");
    return;
  }
  if (!ep || !ep.controller) {
    el.className = "control";
    el.innerHTML = `<span class="what">${ep ? "Waiting for the first decision" : "Waiting for the run to start"}</span>`;
    announce("");
    return;
  }
  const took = ep.rows.find((r) => r.kind === "takeover");
  const done = finished(ep);
  // The controller tag is the sentence's subject.
  const detail = took
    ? `took over when the student stopped${took.step === null ? "" : " at step " + esc(took.step)}.`
    : ep.controller === "student"
      ? `${done ? "chose" : "is choosing"} each skill from the head-camera frame.`
      : `${done ? "demonstrated" : "is demonstrating"} from the head-camera frame.`;
  const [kind, where] = kindOf(ep);
  el.className = "control " + ep.controller;
  el.innerHTML = `<span class="what">${kind}</span><span class="note">${where}</span>${tag(ep.controller, ep.checkpoint)}<span class="note">${detail}</span>${ep.verdict ? verdictChip(ep.verdict) : ""}`;
  announce(`${WHO[ep.controller]} ${done ? "was" : "is"} in control.${ep.verdict ? ` The validator ${isAccepted(ep.verdict) ? "accepted" : "rejected"} this episode.` : ""}`);
}

function renderCamera() {
  // While the student fine-tunes no robot moves, so the stage shows the wait instead of the last episode's frame.
  const tuning = MomState.finetuning(view, now());
  if (tuning !== null) {
    $("chip-camera").innerHTML = panelChip(true);
    $("stage").innerHTML = `<div class="tuning"><p class="what">Fine-tuning the student</p><p class="fig">${elapsed(tuning)}</p><p class="note">since the fine-tune started</p></div>`;
    $("strip").innerHTML = "";
    return;
  }
  const ep = MomState.episode(view, now());
  const frames = MomState.filmstrip(ep, view.compare && view.compare.step);
  $("chip-camera").innerHTML = panelChip(frames.length > 0, ep);
  if (!frames.length) {
    $("stage").innerHTML = `<div class="empty">${ep && finished(ep) ? "This episode recorded no head-camera frame." : "No head-camera frame yet. The latest frame appears here after each executed skill."}</div>`;
    $("strip").innerHTML = "";
    return;
  }
  const last = frames.find((f) => f.now);
  const move = MomState.move(ep);
  const what = move && esc(MOVED[ep.rows.findLast((r) => r.kind === "perturbation").what] || "Scene changed");
  // The caption row is always there, so the peak's caption never shrinks the frame.
  let frame, caption;
  if (view.compare && move) {
    const step = (f) => (f && f.step !== null ? `<span class="step">step ${esc(f.step)}</span>` : "");
    frame = cell(ep, move.before, `<span class="tag">Before the move</span>${step(move.before)}`, false) + cell(ep, move.after, `<span class="tag">After</span>${step(move.after)}`, true);
    caption = `<p class="caption moved">${what} between ${move.before && move.before.step !== null ? `step ${esc(move.before.step)} and ` : "these frames, before "}step ${esc(move.step)}.</p>`;
  } else {
    frame = cell(ep, last, tag(last.controller, last.checkpoint), last.moved);
    caption = last.moved
      ? `<p class="caption moved">${what} just before this frame, the first to show it.</p>`
      : `<p class="caption">${last.step === null ? "Head-camera frame." : `What the ${last.controller || "robot"} saw at step ${esc(last.step)}, before choosing.`}</p>`;
  }
  $("stage").innerHTML = `<div class="frame">${frame}</div>${caption}`;
  $("strip").innerHTML = frames
    .map((f) => `<li class="${[f.now ? "now" : "", f.moved ? "moved" : ""].join(" ").trim()}"><div class="thumb"${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="">` : ""}</div><span class="st">step ${esc(f.step ?? "")}${f.moved ? '<span class="sr"> (the scene moved before this frame)</span>' : ""}</span></li>`)
    .join("");
  // Flash once when the first frame after the move arrives live, never on a key press; added outside the HTML so
  // later identical writes are skipped and the animation runs to its end.
  const key = `${ep.id}@${last.step}`;
  const arriving = view.selected === null && !view.frozen && MomState.following(view, now());
  if (last.moved && arriving && key !== flashedFrame) {
    const shot = $("stage").querySelector(".shot");
    shot.classList.add("flash");
    shot.addEventListener("animationend", () => shot.classList.remove("flash"), { once: true });
  }
  flashedFrame = last.moved ? key : flashedFrame;
}

// One frame on the stage with its label and the skill it led to; a missing frame keeps its place, hatched.
function cell(ep, f, label, moved) {
  const shot = f && f.url ? `<img src="${esc(f.url)}" alt="Head camera frame${f.step === null ? "" : ` at step ${esc(f.step)}`}">` : "";
  const row = f && ep.rows.find((r) => r.kind === "decision" && r.step === f.step && r.skill);
  const said = row ? `<span class="said ${row.controller || "unknown"}">${skillText(row)}</span>` : "";
  return `<div class="cell"><div class="shot${moved ? " moved" : ""}"${shot ? "" : " data-missing"}>${shot}<div class="over">${label}</div>${said}</div></div>`;
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
  const reason = v.reason ? `${esc(cap(v.reason))}. ` : "";
  return `<li class="event ${isAccepted(v) ? "good" : "bad"}"><span class="n">${isAccepted(v) ? CHECK : X}</span><span>The validator ${isAccepted(v) ? "accepted" : "rejected"} this episode. <span class="d">${reason}Task ${v.task ? "succeeded" : "not met"}, physics ${v.physics ? "valid" : "violated"}.</span></span></li>`;
}

const NAVIGATION = { "b-prev": "ArrowLeft", "b-next": "ArrowRight", "b-d": "d", "b-t": "t", "b-m": "m", "b-end": "End" };

function renderEpisode() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const at = s.episodes.indexOf(ep);
  // The connection note sits beside the rail, which gives way, so the keys above never wrap.
  const lost = view.lost ? ' <span class="lost">· connection lost, retrying</span>' : "";
  $("episode-line").innerHTML = (ep ? `Episode ${at + 1} of ${s.episodes.length}` : "") + lost;
  for (const [id, key] of Object.entries(NAVIGATION)) $(id).disabled = MomState.target(view, now(), key) < 0;
  $("b-p").disabled = !view.compare && (!ep || ep.perturbedStep === null);
  $("b-p").setAttribute("aria-pressed", String(view.compare !== null));
  $("b-pause").setAttribute("aria-pressed", String(view.frozen !== null));
  $("b-pause").lastElementChild.textContent = view.frozen ? "Resume" : "Pause";
  // Where the view is among the run's episodes: tall for a teacher, short for the student; hollow when rejected.
  $("rail").style.setProperty("--n", s.episodes.length);
  $("rail").innerHTML = s.episodes
    .map((e) => `<li class="${MomState.teacherDrove(e) ? "teacher" : "student"} ${e.verdict ? (MomState.accepted(e) ? "accepted" : "rejected") : "open"}${e === ep ? " here" : ""}"></li>`)
    .join("");
}

function renderLog() {
  const ep = MomState.episode(view, now());
  const list = $("log");
  // Said once in the header rather than on every row, until a decision in the episode carries a reason.
  const reasons = !ep || ep.rows.some((r) => r.kind === "decision" && r.reason) || !ep.rows.some((r) => r.kind === "decision");
  $("chip-log").innerHTML = (reasons ? "" : '<span class="note">No reasons yet</span>') + panelChip(ep !== null, ep);
  if (!ep) {
    list.innerHTML = '<li class="empty">No decisions yet. Each decision appears with its reason and latency.</li>';
    renderedEpisode = null;
    return;
  }
  const move = MomState.move(ep);
  const done = finished(ep);
  const stage = MomState.filmstrip(ep, view.compare && view.compare.step).find((f) => f.now);
  const full = (r) => !stage || stage.step === null || (r.step !== null && r.step >= stage.step);
  const html = ep.rows.map((r) => (r.kind === "decision" ? decisionRow(r, move && r === move.decision, done, full(r)) : r.kind === "verdict" ? verdictRow(ep) : logRow(r)));
  const focus = view.compare && `${ep.id}@${view.compare.step}`;
  // Letting go of P drops the scroll it set, so the log follows new rows again.
  const fromStart = renderedEpisode !== ep.id || html.length < renderedRows || (renderedFocus !== null && focus === null);
  if (fromStart) logFollowing = true;
  list.innerHTML = html.join("");
  if (!fromStart) [...list.children].slice(renderedRows).forEach((li) => li.classList.add("fresh"));
  if (logFollowing) list.scrollTop = list.scrollHeight;
  const row = list.querySelector(".focus");
  if (focus !== renderedFocus && row) list.scrollTop += row.getBoundingClientRect().top - list.getBoundingClientRect().top - list.clientHeight / 3;
  renderedFocus = focus;
  renderedRows = html.length;
  renderedEpisode = ep.id;
}

function renderRounds() {
  const rounds = MomState.shown(view).rounds;
  let body = '<div class="ph-text">No held-out evaluation in the events yet.</div>';
  if (rounds.length) {
    const first = rounds[0];
    const last = rounds[rounds.length - 1];
    const both = rounds.length > 1;
    const fig = both ? `${first.successes}/${first.n} → ${last.successes}/${last.n}` : `${first.successes}/${first.n}`;
    const spoken = both ? `${first.successes} of ${first.n}, then ${last.successes} of ${last.n}` : `${first.successes} of ${first.n}`;
    const n = first.n === last.n ? `n=${last.n}${both ? " each" : ""}` : `n=${first.n} and n=${last.n}`;
    const span = both ? `${esc(first.checkpoint)} → ${esc(last.checkpoint)}` : esc(first.checkpoint);
    const table = `<table><thead><tr><th>Checkpoint</th><th class="num">Successes</th><th class="num">Rate</th></tr></thead><tbody>${rounds.map((r) => `<tr><td class="mono">${esc(r.checkpoint)}</td><td class="num">${r.successes}/${r.n}</td><td class="num">${pct(r.successes, r.n)}</td></tr>`).join("")}</tbody></table>`;
    body = `<div class="gain"><span class="fig" role="img" aria-label="${spoken} held-out successes">${fig}</span><span>${n}</span><span class="note mono">${span}</span></div>${table}`;
  }
  $("p-rounds").innerHTML = `<header><h2>Held-out success</h2>${panelChip(rounds.length > 0)}</header><div class="body">${body}</div>`;
}

// One line: each model, where it runs (configured), and its measured median latency or fine-tune duration.
function renderModels() {
  const s = MomState.summary(MomState.shown(view));
  const any = s.student || s.teacher || s.finetune;
  const item = (who, where, measured) => `<span class="item"><b>${who}</b> ${where} <span class="v">${stat(measured)}</span></span>`;
  $("models").innerHTML =
    `<h2>Models</h2><span class="chip config">Configured</span>` +
    item("Student", CONFIGURED.student, s.student) +
    item("Teacher", CONFIGURED.teacher, s.teacher) +
    item("Fine-tune", CONFIGURED.finetune, s.finetune) +
    panelChip(Boolean(any));
}

const evidenceText = (d) => (d.evidence ? `<div class="note">${esc(d.evidence)}</div>` : "");
const confidenceText = (d) => (d.confidence === null ? "" : `<div class="note">Confidence ${Math.round(d.confidence * 100)}%.</div>`);
const diagnosisText = (d) => `<div>${esc(d.hypothesis)}</div>${evidenceText(d)}${confidenceText(d)}`;

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
  return `<div class="body stack">${lead}<div>${esc(a.diagnosis.hypothesis)}</div>${evidenceText(a.diagnosis)}</div>`;
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
      : `<div class="body ph-text">${ep ? "No diagnosis for this episode. The teacher diagnoses failed student episodes on development seeds, then demonstrates on new ones." : "No episode yet."}</div>`;
  $("p-diagnosis").innerHTML = `<header><h2>Diagnosis</h2>${panelChip(Boolean(acting ? acting.diagnosis : d), ep)}</header>${body}`;
}

function render() {
  renderQueued = false;
  renderHeader();
  renderControl();
  renderEpisode();
  renderCamera();
  renderRounds();
  renderModels();
  renderDiagnosis();
  // Last, so its scroll lands after every bar that sets its height is written.
  renderLog();
  // A new episode starts the side column at its top, where the held-out figure is.
  const ep = MomState.episode(view, now());
  if (ep && ep.id !== renderedSide) $("side").scrollTop = 0;
  renderedSide = ep && ep.id;
}

function schedule() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(render);
}

$("log").addEventListener("scroll", (e) => {
  const log = e.currentTarget;
  logFollowing = log.scrollHeight - log.scrollTop - log.clientHeight < 8;
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

const source = new EventSource("/events");
// The server replays the whole run on every connection, so the view starts over instead of adding to what is shown.
source.onopen = () => {
  MomState.reconnect(view);
  renderedRows = 0;
  renderedEpisode = null;
  schedule();
};
source.onerror = () => {
  MomState.disconnect(view);
  schedule();
};
source.onmessage = (message) => {
  try {
    MomState.receive(view, JSON.parse(message.data), now());
  } catch (error) {
    console.error("bad event", error);
  }
  schedule();
};
setInterval(schedule, 1000); // live and recorded change with time, and the fine-tune clock ticks
schedule();
