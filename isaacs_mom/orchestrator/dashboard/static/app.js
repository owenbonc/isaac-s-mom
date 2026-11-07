// Renders MomState over the /events stream. Every value from an event is escaped; nothing is shown that an event did not carry.
const view = MomState.view();
let connection = "open";
let renderQueued = false;
let renderedRows = 0;
let renderedEpisode = null;
let renderedFocus = null;
let renderedSide = null;
let flashedFrame = null;
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
const tag = (controller, checkpoint) => {
  const who = { student: "Student", teacher: "Teacher (Argon)" }[controller] || "Unknown controller";
  return `<span class="tag ${controller || "unknown"}">${who}${checkpoint ? ` <span class="mono">${esc(checkpoint)}</span>` : ""}</span>`;
};
const amount = (row) => (row.distance !== null ? ` ${+row.distance.toFixed(1)} cm` : row.angle !== null ? ` ${Math.round(row.angle)}°` : "");
const skillText = (row) => (row.skill ? esc(row.skill) + amount(row) : '<span class="stop">Stopped, no skill chosen</span>');
const SCENES = { dev: "Development scene", heldout: "Held-out test scene" };
const isAccepted = (v) => v.task && v.physics;
const verdictChip = (v) => `<span class="verdict ${isAccepted(v) ? "good" : "bad"}">${isAccepted(v) ? CHECK + "Accepted" : X + "Rejected"}</span>`;
const MOVED = { move_box: "Box moved", move_bin: "Bin moved" };
// Durations read at a glance from the back of the room: ms under a second, then seconds, then minutes.
const duration = (v) => (v < 1000 ? `${v >= 100 ? Math.round(v) : v.toFixed(1)} ms` : v < 120000 ? `${(v / 1000).toFixed(1)} s` : `${(v / 60000).toFixed(1)} min`);
const pct = (k, n) => (n ? Math.round((100 * k) / n) + "%" : "n/a");
const stat = (s) => (s ? `${duration(s.median)} <span class="note">n=${s.n}</span>` : '<span class="ph-text">no data</span>');

function renderHeader() {
  const s = MomState.shown(view);
  const kind = MomState.provenance(view, now());
  const status = s.eventCount === 0 ? chip("placeholder", "Waiting") : chip(kind, `${label(kind)} · ${simName(s)}`);
  $("status").innerHTML = status + (connection === "lost" ? ' <span class="note">connection lost, retrying</span>' : "");
}

function renderControl() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const el = $("control");
  if (s.finetune.pending && view.selected === null && MomState.following(view, now())) {
    el.className = "control";
    el.innerHTML = `<span class="what">Fine-tuning the student, round ${esc(s.finetune.pending.round ?? "")}</span><span class="note">No robot is moving. Held-out evaluation starts when the new checkpoint is ready.</span>`;
    return;
  }
  if (!ep || !ep.controller) {
    el.className = "control";
    el.innerHTML = `<span class="what">${ep ? "Waiting for the first decision" : "Waiting for the run to start"}</span>`;
    return;
  }
  const took = ep.rows.find((r) => r.kind === "takeover");
  const detail = took
    ? `The student stopped without succeeding${took.step === null ? "" : " at step " + took.step}, so the teacher took over.`
    : ep.controller === "student"
      ? "Choosing each skill from the head-camera frame."
      : "Demonstrating from the head-camera frame.";
  el.className = "control " + ep.controller;
  el.innerHTML = `<span class="what">${esc(SCENES[ep.split] || "Scene")}</span><span class="note">seed ${esc(ep.seed ?? "")}, round ${esc(ep.round ?? "")}</span>${tag(ep.controller, ep.checkpoint)}<span class="note">${detail}</span>${ep.verdict ? verdictChip(ep.verdict) : ""}`;
}

function renderCamera() {
  const ep = MomState.episode(view, now());
  const frames = MomState.filmstrip(ep, view.compare && view.compare.step);
  $("chip-camera").innerHTML = panelChip(frames.length > 0, ep);
  if (!frames.length) {
    $("stage").innerHTML = '<div class="empty">No head-camera frame yet. The latest frame appears here after each executed skill.</div>';
    $("strip").innerHTML = "";
    return;
  }
  const last = frames.find((f) => f.now);
  const seen = last.step === null ? "" : `Seen by step ${esc(last.step)}`;
  const shot = last.url ? `<img src="${esc(last.url)}" alt="Head camera frame${seen ? ", " + seen.toLowerCase() : ""}">` : "";
  const moved = last.moved ? `<p class="caption moved">${esc(MOVED[ep.rows.findLast((r) => r.kind === "perturbation").what] || "Scene changed")} just before this frame. This is the first frame that shows it.</p>` : "";
  $("stage").innerHTML = `<div class="frame"><div class="shot${last.moved ? " moved" : ""}"${last.url ? "" : " data-missing"}>${shot}<div class="over">${tag(last.controller, last.checkpoint)}${seen ? `<span class="step">${seen}</span>` : ""}</div></div></div>${moved}`;
  $("strip").innerHTML = frames
    .map((f) => `<li class="${[f.now ? "now" : "", f.moved ? "moved" : ""].join(" ").trim()}"><div class="thumb"${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="">` : ""}</div><span class="st">step ${esc(f.step ?? "")}${f.moved ? '<span class="sr"> (the scene moved before this frame)</span>' : ""}</span></li>`)
    .join("");
  // Flash once when the first frame after the move reaches the stage; added outside the HTML so later
  // identical writes are skipped and the animation runs to its end.
  const key = `${ep.id}@${last.step}`;
  if (last.moved && key !== flashedFrame) {
    const shot = $("stage").querySelector(".shot");
    shot.classList.add("flash");
    shot.addEventListener("animationend", () => shot.classList.remove("flash"), { once: true });
  }
  flashedFrame = last.moved ? key : flashedFrame;
}

function logRow(row) {
  const step = row.step === null ? "" : esc(row.step);
  switch (row.kind) {
    case "perturbation":
      return `<li class="event moved${view.compare && row.step === view.compare.step ? " focus" : ""}"><span class="n">${step}</span><span>${esc(MOVED[row.what] || "Scene changed")} <span class="d">by the orchestrator. The robot learns of it only from its camera${row.step === null ? "" : `, in the frame seen by step ${step}`}.</span></span></li>`;
    case "takeover":
      return `<li class="event take"><span class="n">${step}</span><span>Teacher takes over</span></li>`;
    case "error":
      return `<li class="event bad"><span class="n"></span><span>Error <span class="d">${esc(row.text)}</span></span></li>`;
    default:
      return "";
  }
}

function decisionRow(row) {
  const outcome = !row.skill ? "" : row.ok === null ? '<span class="lat">running</span>' : row.ok ? '<span class="ok">executed</span>' : '<span class="no">failed</span>';
  const latency = row.latency === null ? "" : `<span class="lat">${duration(row.latency)}</span>`;
  const reason = row.reason ? `<div class="reason" title="${esc(row.reason)}">${esc(row.reason)}</div>` : '<div class="reason none">No reason given.</div>';
  const error = row.error ? `<div class="err">${esc(row.error)}</div>` : "";
  return `<li class="${row.controller || "unknown"}"><span class="n">${esc(row.step ?? "")}</span><div class="line"><span class="skill">${skillText(row)}</span><span class="right">${latency}${outcome}</span></div>${reason}${error}</li>`;
}

function verdictRow(ep) {
  const v = ep.verdict;
  return `<li class="event ${isAccepted(v) ? "good" : "bad"}"><span class="n"></span><span>Validator: ${isAccepted(v) ? "accepted" : "rejected"}. <span class="d">${esc(v.reason)}. Task ${v.task ? "succeeded" : "not met"}; physics ${v.physics ? "valid" : "violated"}.</span></span></li>`;
}

const NAVIGATION = { "b-prev": "ArrowLeft", "b-next": "ArrowRight", "b-d": "d", "b-t": "t", "b-m": "m", "b-end": "End" };

function renderEpisode() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view, now());
  const at = s.episodes.indexOf(ep);
  $("episode-line").innerHTML = ep ? `Episode ${at + 1} of ${s.episodes.length}` : "";
  for (const [id, key] of Object.entries(NAVIGATION)) $(id).disabled = MomState.target(view, now(), key) < 0;
  $("b-p").disabled = !view.compare && (!ep || ep.perturbedStep === null);
  $("b-p").setAttribute("aria-pressed", String(view.compare !== null));
  $("b-pause").setAttribute("aria-pressed", String(view.frozen !== null));
  $("b-pause").lastElementChild.textContent = view.frozen ? "Resume" : "Pause";
  // Where the view is among the run's episodes: tall for a teacher, short for the student; hollow when rejected.
  $("rail").innerHTML = s.episodes
    .map((e) => `<li class="${MomState.teacherDrove(e) ? "teacher" : "student"} ${e.verdict ? (MomState.accepted(e) ? "accepted" : "rejected") : "open"}${e === ep ? " here" : ""}"></li>`)
    .join("");
}

function renderLog() {
  const ep = MomState.episode(view, now());
  const list = $("log");
  $("chip-log").innerHTML = panelChip(ep !== null, ep);
  if (!ep) {
    list.innerHTML = '<li class="empty">No decisions yet. Each decision appears with its reason and latency.</li>';
    renderedEpisode = null;
    return;
  }
  const html = ep.rows.map((r) => (r.kind === "decision" ? decisionRow(r) : r.kind === "verdict" ? verdictRow(ep) : logRow(r)));
  const focus = view.compare && `${ep.id}@${view.compare.step}`;
  // Letting go of P drops the scroll it set, so the log follows new rows again.
  const fromStart = renderedEpisode !== ep.id || html.length < renderedRows || (renderedFocus !== null && focus === null);
  // Stay at the bottom only when the reader is already there, so scrolling back to point at a row holds.
  const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 8;
  list.innerHTML = html.join("");
  if (!fromStart) [...list.children].slice(renderedRows).forEach((li) => li.classList.add("fresh"));
  if (fromStart || atBottom) list.scrollTop = list.scrollHeight;
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
    body = `<div class="gain"><span class="fig" role="img" aria-label="${spoken} held-out successes">${fig}</span><span>${n}</span><span class="note">${span}</span></div>${table}`;
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

const evidenceText = (d) => `${d.evidence ? `<div class="note">${esc(d.evidence)}</div>` : ""}${d.confidence === null ? "" : `<div class="note">Confidence ${Math.round(d.confidence * 100)}%.</div>`}`;
const diagnosisText = (d) => `<div>${esc(d.hypothesis)}</div>${evidenceText(d)}`;

// A teacher demonstration shows the diagnosis it acts on, named by the diagnosed episode, with that episode's last frame.
function actingOnBody(a) {
  if (a.number === null) return `<div class="body ph-text">Acting on a diagnosis of episode <span class="mono">${esc(a.id)}</span>, which is not in this run's events.</div>`;
  const head = `<p class="acting-on">Acting on diagnosis from episode ${a.number}</p>`;
  if (!a.diagnosis) return `<div class="body stack">${head}<div class="ph-text">Episode ${a.number} has no diagnosis in the events yet.</div></div>`;
  const shot = a.frame && a.frame.url
    ? `<figure class="cited"><div class="thumb"><img src="${esc(a.frame.url)}" alt="Episode ${a.number}'s last head-camera frame"></div><figcaption class="note">Its last frame${a.frame.step === null ? "" : `, step ${esc(a.frame.step)}`}</figcaption></figure>`
    : "";
  return `<div class="body stack">${head}<div>${esc(a.diagnosis.hypothesis)}</div><div class="cited-row">${shot}<div class="stack">${evidenceText(a.diagnosis)}</div></div></div>`;
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
  const title = acting ? "Diagnosis it acts on" : "Teacher's diagnosis";
  $("p-diagnosis").innerHTML = `<header><h2>${title}</h2>${panelChip(Boolean(acting ? acting.diagnosis : d), ep)}</header>${body}`;
}

function render() {
  renderQueued = false;
  renderHeader();
  renderControl();
  renderEpisode();
  renderCamera();
  renderLog();
  renderRounds();
  renderModels();
  renderDiagnosis();
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
  connection = "open";
  MomState.reconnect(view);
  renderedRows = 0;
  renderedEpisode = null;
  schedule();
};
source.onerror = () => {
  connection = "lost";
  schedule();
};
source.onmessage = (message) => {
  try {
    MomState.reduce(view.live, JSON.parse(message.data));
  } catch (error) {
    console.error("bad event", error);
  }
  schedule();
};
setInterval(schedule, 2000);
schedule();
