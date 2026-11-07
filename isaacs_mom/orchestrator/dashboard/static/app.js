// Renders MomState over the /events stream. Every value from an event is escaped; nothing is shown that an event did not carry.
const view = MomState.view();
let connection = "open";
let renderQueued = false;
let renderedRows = 0;
let renderedEpisode = null;
let renderedFocus = null;
let flashedFrame = null;
// Configuration (SPEC §3.2, §3.3), not data: the page shows these labelled as configured.
const CONFIGURED = { student: "SmolVLM2-500M, local", teacher: "gemini-4-argon, Gemini API", finetune: "LoRA on Modal A10G" };

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
const skillText = (row) => (row.skill ? esc(row.skill) + (row.distance === null ? "" : ` ${+row.distance.toFixed(1)} cm`) : '<span class="stop">Stopped, no skill chosen</span>');
const SCENES = { dev: "Development scene", heldout: "Held-out test scene" };
const MOVED = { move_block: "Block moved", move_target: "Target zone moved" };
// Durations read at a glance from the back of the room: ms under a second, then seconds, then minutes.
const duration = (v) => (v < 1000 ? `${v >= 100 ? Math.round(v) : v.toFixed(1)} ms` : v < 120000 ? `${(v / 1000).toFixed(1)} s` : `${(v / 60000).toFixed(1)} min`);
const pct = (k, n) => (n ? Math.round((100 * k) / n) + "%" : "n/a");
const stat = (s) => (s ? `${duration(s.median)} <span class="note">n=${s.n}</span>` : '<span class="ph-text">no data</span>');

function renderHeader() {
  const s = MomState.shown(view);
  const kind = MomState.provenance(view, now());
  const status = s.eventCount === 0 ? chip("placeholder", "Waiting") : chip(kind, `${label(kind)} · ${simName(s)}`);
  $("status").innerHTML = status + (connection === "lost" ? ' <span class="note">connection lost, retrying</span>' : "");
  const ep = MomState.current(s);
  const robot = ep && ep.sim && ep.sim.robot;
  $("meta").innerHTML = s.eventCount
    ? `<span>Run <b class="mono">${esc(s.runId)}</b></span>${robot ? `<span>Robot <span class="mono">${esc(robot)}</span></span>` : ""}<span>${s.episodes.length} episode${s.episodes.length === 1 ? "" : "s"}</span><span>${s.eventCount} events</span>`
    : "";
}

function renderControl() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view);
  const el = $("control");
  if (s.finetune.pending && view.selected === null) {
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
  const result = ep.verdict ? ` Validator: ${ep.verdict.task && ep.verdict.physics ? "accepted" : "rejected"}.` : "";
  const detail = took
    ? `The student stopped without succeeding${took.step === null ? "" : " at step " + took.step}, so the teacher took over.`
    : ep.controller === "student"
      ? "Choosing each skill from the head-camera frame."
      : "Demonstrating from the head-camera frame.";
  el.className = "control " + ep.controller;
  el.innerHTML = `<span class="what">${ep.verdict ? "Last in control:" : "In control:"}</span>${tag(ep.controller, ep.checkpoint)}<span class="note">${detail}${result}</span>`;
}

function renderCamera() {
  const ep = MomState.episode(view);
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
    .map((f) => `<li class="${[f.now ? "now" : "", f.moved ? "moved" : ""].join(" ").trim()}"><div class="thumb"${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="">` : ""}</div><span class="st">step ${esc(f.step ?? "")}${f.moved ? '<span class="sr"> (block moved before this frame)</span>' : ""}</span></li>`)
    .join("");
  // Flash once when the first frame after the block move reaches the stage; added outside the HTML so later
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
  const ok = ep.verdict.task && ep.verdict.physics;
  return `<li class="event ${ok ? "good" : "bad"}"><span class="n"></span><span>Validator: ${ok ? "accepted" : "rejected"} <span class="d">${esc(ep.verdict.reason)}</span></span></li>`;
}

function renderEpisode() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view);
  const at = s.episodes.indexOf(ep);
  $("episode-line").innerHTML = ep
    ? `<b>Episode ${at + 1} of ${s.episodes.length}${view.selected === null ? "" : ", pinned"}.</b> ${esc(SCENES[ep.split] || "Scene")}, seed ${esc(ep.seed ?? "")}, round ${esc(ep.round ?? "")} <span class="mono">${esc(ep.id)}</span>`
    : "";
  $("b-prev").disabled = at <= 0;
  $("b-next").disabled = at < 0 || at === s.episodes.length - 1;
  $("b-moved").disabled = !ep || ep.perturbedStep === null;
  $("b-moved").setAttribute("aria-pressed", String(view.compare !== null));
  $("b-pause").setAttribute("aria-pressed", String(view.frozen !== null));
  $("b-pause").lastElementChild.textContent = view.frozen ? `Resume · ${MomState.held(view)} new events` : "Pause";
}

function renderLog() {
  const ep = MomState.episode(view);
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

function renderVerdict() {
  const ep = MomState.episode(view);
  const v = ep && ep.verdict;
  const body = v
    ? `<div class="stack"><span class="verdict ${v.task && v.physics ? "good" : "bad"}">${v.task && v.physics ? CHECK + "Accepted" : X + "Rejected"}</span><div>${esc(v.reason)}</div><div class="note">Task ${v.task ? "succeeded" : "not met"}. Physics ${v.physics ? "valid" : "violated"}.</div></div>`
    : `<div class="ph-text">${ep ? "No verdict yet for this episode." : "No episode yet."}</div>`;
  $("p-verdict").innerHTML = `<header><h2>Validator</h2>${panelChip(Boolean(v), ep)}</header><div class="body">${body}</div>`;
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

function renderModels() {
  const s = MomState.summary(MomState.shown(view));
  const any = s.student || s.teacher || s.argon || s.finetune;
  const row = (who, where, measured) => `<tr><td><b>${who}</b> ${where}</td><td class="num">${stat(measured)}</td></tr>`;
  const head = (what, measure) => `<tr><th>${what} <span class="chip config">Configured</span></th><th class="num">${measure}</th></tr>`;
  const inference = row("Student", CONFIGURED.student, s.student) + row("Teacher", CONFIGURED.teacher, s.teacher) + (s.argon ? row("", "Argon API calls", s.argon) : "");
  $("p-models").innerHTML = `<header><h2>Where models run</h2>${panelChip(Boolean(any))}</header><div class="body"><table><thead>${head("Model and host", "Median latency")}</thead><tbody>${inference}</tbody><tbody>${head("Fine-tune", "Median duration")}${row("", CONFIGURED.finetune, s.finetune)}</tbody></table></div>`;
}

const diagnosisText = (d) =>
  `<div>${esc(d.hypothesis)}</div>${d.evidence ? `<div class="note">${esc(d.evidence)}</div>` : ""}${d.confidence === null ? "" : `<div class="note">Confidence ${Math.round(d.confidence * 100)}%.</div>`}`;

// A teacher demonstration shows the diagnosis it acts on, named by the diagnosed episode, with that episode's last frame.
function actingOnBody(a) {
  if (a.number === null) return `<div class="body ph-text">Acting on a diagnosis of episode <span class="mono">${esc(a.id)}</span>, which is not in this run's events.</div>`;
  const head = `<p class="acting-on">Acting on diagnosis from episode ${a.number}</p>`;
  if (!a.diagnosis) return `<div class="body stack">${head}<div class="ph-text">Episode ${a.number} has no diagnosis in the events yet.</div></div>`;
  const shot = a.frame && a.frame.url
    ? `<figure class="cited"><div class="thumb"><img src="${esc(a.frame.url)}" alt="Episode ${a.number}'s last head-camera frame"></div><figcaption class="note">Its last frame${a.frame.step === null ? "" : `, step ${esc(a.frame.step)}`}</figcaption></figure>`
    : "";
  return `<div class="body stack">${head}<div class="cited-row">${shot}<div class="stack">${diagnosisText(a.diagnosis)}</div></div></div>`;
}

function renderDiagnosis() {
  const s = MomState.shown(view);
  const ep = MomState.episode(view);
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
  renderVerdict();
  renderRounds();
  renderModels();
  renderDiagnosis();
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
  if (!MomState.keydown(view, e)) return;
  e.preventDefault(); // also keeps a focused button from taking Space as a click
  schedule();
});
// Some browsers click a focused button on Space's keyup; the keydown above already acted.
document.addEventListener("keyup", (e) => e.key === " " && e.preventDefault());
document.querySelector(".presenter").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-key]");
  if (!button) return;
  MomState.keydown(view, { key: button.dataset.key });
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
