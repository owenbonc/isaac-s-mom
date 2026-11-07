// Renders MomState over the /events stream. Every value from an event is escaped; nothing is shown that an event did not carry.
const view = MomState.view();
let connection = "open";
let renderQueued = false;
let renderedRows = 0;
let renderedEpisode = null;
let renderedFocus = null;
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
// The label for a panel: placeholder when it has no data, else the run's (or its episode's) provenance.
const panelChip = (hasData, ep = null) => {
  if (!hasData) return chip("placeholder");
  const kind = MomState.provenance(MomState.shown(view), now(), ep);
  return chip(view.frozen && kind === "live" ? "recorded" : kind); // a paused view is not live
};
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
const skillText = (row) => (row.skill ? esc(row.skill) + (row.params && row.params.distance_cm != null ? ` ${esc(row.params.distance_cm)} cm` : "") : '<span class="stop">Stopped, no skill chosen</span>');
const SCENES = { dev: "Development scene", heldout: "Held-out test scene" };
const MOVED = { move_block: "Block moved", move_target: "Target zone moved" };
const ms = (v) => (v >= 100 ? Math.round(v) : v.toFixed(1)) + " ms";
const pct = (k, n) => (n ? Math.round((100 * k) / n) + "%" : "n/a");
const stat = (s) => (s ? `${ms(s.median)} <span class="note">n=${s.n}</span>` : '<span class="ph-text">no data</span>');

function renderHeader() {
  const run = view.live;
  const kind = MomState.provenance(run, now());
  const status = run.eventCount === 0 ? chip("placeholder", "Waiting") : chip(kind, `${label(kind)} · ${simName(run)}`);
  const paused = view.frozen ? " " + chip("paused", `Paused · ${MomState.held(view)} new events`) : "";
  $("status").innerHTML = status + paused + (connection === "lost" ? ' <span class="note">connection lost, retrying</span>' : "");
  const s = MomState.shown(view);
  $("meta").innerHTML = s.eventCount
    ? `<span>Run <b class="mono">${esc(s.runId)}</b></span><span>${s.episodes.length} episodes</span><span>${s.eventCount} events</span>`
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
  el.innerHTML = `<span class="what">${ep.verdict ? "Last in control:" : "In control:"}</span>${tag(ep.controller, ep.checkpoint)}<span class="note" style="color:var(--ink)">${detail}${result}</span>`;
}

function renderCamera() {
  const ep = MomState.episode(view);
  const frames = MomState.filmstrip(ep, view.focus);
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
  $("stage").innerHTML = `<div class="frame"><div class="shot"${last.url ? "" : " data-missing"}>${shot}<div class="over">${tag(last.controller, last.checkpoint)}${seen ? `<span class="step">${seen}</span>` : ""}</div></div></div>${moved}`;
  $("strip").innerHTML = frames
    .map((f) => `<li class="${[f.now ? "now" : "", f.moved ? "moved" : ""].join(" ").trim()}"><div class="thumb"${f.url ? "" : " data-missing"}>${f.url ? `<img src="${esc(f.url)}" alt="">` : ""}</div><span class="st">step ${esc(f.step ?? "")}${f.moved ? '<span class="sr"> (block moved before this frame)</span>' : ""}</span></li>`)
    .join("");
}

function logRow(row) {
  const step = row.step === null ? "" : esc(row.step);
  switch (row.kind) {
    case "perturbation":
      return `<li class="event moved${row.step === view.focus ? " focus" : ""}"><span class="n">${step}</span><span>${esc(MOVED[row.what] || "Scene changed")} <span class="d">by the orchestrator. The robot learns of it only from its camera${row.step === null ? "" : `, in the frame seen by step ${step}`}.</span></span></li>`;
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
  const latency = row.latency === null ? "" : `<span class="lat">${ms(row.latency)}</span>`;
  const reason = row.reason ? `<div class="reason" title="${esc(row.reason)}">${esc(row.reason)}</div>` : '<div class="reason none">No reason given.</div>';
  const error = row.error ? `<div class="err">${esc(row.error)}</div>` : "";
  return `<li class="${row.controller || "unknown"}"><span class="n">${esc(row.step ?? "")}</span><span class="skill">${skillText(row)}</span><span class="right">${latency}${outcome}</span>${reason}${error}</li>`;
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
  $("b-moved").setAttribute("aria-pressed", String(view.focus !== null));
  $("b-pause").lastElementChild.textContent = view.frozen ? "Resume" : "Pause";
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
  const fromStart = renderedEpisode !== ep.id || html.length < renderedRows;
  // Stay at the bottom only when the reader is already there, so scrolling back to point at a row holds.
  const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 8;
  list.innerHTML = html.join("");
  if (!fromStart) [...list.children].slice(renderedRows).forEach((li) => li.classList.add("fresh"));
  if (fromStart || atBottom) list.scrollTop = list.scrollHeight;
  const focus = view.focus === null ? null : `${ep.id}@${view.focus}`;
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
    const delta = last.successes - first.successes;
    const trend = rounds.length > 1 ? `<div class="note" style="margin-top:.35rem">${esc(first.checkpoint)} to ${esc(last.checkpoint)}: ${first.successes}/${first.n} to ${last.successes}/${last.n} (${delta >= 0 ? "+" : ""}${delta}${first.n === last.n ? `, n=${last.n} each` : ""}).</div>` : "";
    body = `<table><thead><tr><th>Checkpoint</th><th class="num">Successes</th><th class="num">n</th><th class="num">Rate</th></tr></thead><tbody>${rounds.map((r) => `<tr><td class="mono">${esc(r.checkpoint)}</td><td class="num">${esc(r.successes)}</td><td class="num">${esc(r.n)}</td><td class="num">${pct(r.successes, r.n)}</td></tr>`).join("")}</tbody></table>${trend}`;
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

function renderDiagnosis() {
  const ep = MomState.episode(view);
  const d = ep && ep.diagnosis;
  const conf = d && d.confidence !== null ? `<div class="note">Confidence ${Math.round(d.confidence * 100)}%.</div>` : "";
  const body = d
    ? `<div class="body stack"><div>${esc(d.hypothesis)}</div>${d.evidence ? `<div class="note">${esc(d.evidence)}</div>` : ""}${conf}</div>`
    : `<div class="body ph-text">${ep ? "No diagnosis for this episode. The teacher diagnoses failed student episodes on development seeds." : "No episode yet."}</div>`;
  $("p-diagnosis").innerHTML = `<header><h2>Teacher's diagnosis</h2>${panelChip(Boolean(d), ep)}</header>${body}`;
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

// Presenter controls (SPEC §3.5): the same actions from the keyboard and from the buttons that label them.
const ACTIONS = {
  " ": () => MomState.togglePause(view),
  ArrowLeft: () => MomState.move(view, -1),
  ArrowRight: () => MomState.move(view, 1),
  p: () => MomState.jumpToPerturbation(view),
};
document.addEventListener("keydown", (e) => {
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (!ACTIONS[key] || e.metaKey || e.ctrlKey || e.altKey) return;
  if (key === " " && e.target.closest("button")) return; // a focused button handles its own Space
  e.preventDefault();
  if (e.repeat) return;
  ACTIONS[key]();
  schedule();
});
document.querySelector(".presenter").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-key]");
  if (!button) return;
  ACTIONS[button.dataset.key]();
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
