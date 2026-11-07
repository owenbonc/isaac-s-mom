import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from isaacs_mom.contracts import EpisodeManifest, Perturbation, Split
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.validator import Recorder, validate

MAX_STEPS = 20
PERTURB_SEED_OFFSET = 7919
HEARTBEAT_SECONDS = 10.0


@contextmanager
def heartbeat(events: EventLog, episode_id: str, every: float) -> Iterator[None]:
    """A `heartbeat` event every `every` seconds while the block runs, so a slow step keeps the run live (SPEC §3.5)."""
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(every):
            events.emit("heartbeat", episode_id=episode_id)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def run_episode(
    sim: TruthSim,
    policy: Policy,
    *,
    run_dir: Path,
    events: EventLog,
    seed: int,
    split: Split,
    round_: int,
    instruction: str,
    takeover: Policy | None = None,
    perturb_at: int | None = None,
    max_steps: int = MAX_STEPS,
    acting_on: str | None = None,
    demo: bool = False,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> EpisodeManifest:
    """One recorded episode. If `policy` stops without success, `takeover` (when given) continues from there.

    `acting_on` names the failed episode whose diagnosis a teacher demonstration acts on, and `demo` marks the
    judged take (SPEC §3.5).
    """
    episode_id = sim.reset(seed)
    recorder = Recorder.start(
        run_dir,
        episode_id=episode_id,
        run_id=events.run_id,
        seed=seed,
        split=split,
        round_=round_,
        instruction=instruction,
        sim_version=sim.sim_version,
        robot_config=sim.robot_config,
    )
    events.emit(
        "episode_started",
        episode_id=episode_id,
        seed=seed,
        split=split,
        round=round_,
        sim_version=sim.sim_version,
        robot_config=sim.robot_config,
        acting_on=acting_on,
        demo=demo,
    )
    feedback = "start"
    total_steps = 0
    for phase, active in enumerate([policy] if takeover is None else [policy, takeover]):
        if phase == 1:
            if validate(sim.ground_truth()).accepted:
                break
            recorder.mark_takeover()
            events.emit(
                "decision",
                episode_id=episode_id,
                controller="teacher",
                takeover=True,
                step=total_steps,
                skill=None,
                reason="student did not succeed; teacher takes over",
            )
        for _ in range(max_steps):
            if total_steps == perturb_at:
                perturbation = Perturbation(kind="move_box", seed=seed + PERTURB_SEED_OFFSET)
                if sim.perturb(perturbation):
                    recorder.add_perturbation(perturbation)
                    events.emit("perturbation", episode_id=episode_id, kind=perturbation.kind, step=total_steps)
                else:
                    # Recorded and shown only when it happened, so nothing downstream claims a change that wasn't made.
                    events.emit(
                        "error",
                        episode_id=episode_id,
                        step=total_steps,
                        error=f"{perturbation.kind} not applied: the sim refused it and nothing moved",
                    )
            obs = sim.observe()
            started = time.perf_counter()
            with heartbeat(events, episode_id, heartbeat_seconds):
                call = active.act(obs, instruction, feedback)
            events.emit(
                "decision",
                episode_id=episode_id,
                controller=active.controller,
                checkpoint=active.checkpoint,
                step=obs.step,
                skill=call.model_dump(exclude={"reason"}) if call else None,
                reason=call.reason if call else "",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            if call is None:
                break
            with heartbeat(events, episode_id, heartbeat_seconds):
                result = sim.execute(call)
            recorder.add_step(obs, call, result, active.controller, active.checkpoint)
            events.emit(
                "action_executed",
                episode_id=episode_id,
                step=obs.step,
                ok=result.ok,
                error=result.error,
                # Relative to the run directory, so a reader started from any directory finds it.
                frame=(recorder.dir / recorder.manifest.steps[-1].frame).relative_to(run_dir).as_posix(),
            )
            feedback = f"{call.skill}: {'ok' if result.ok else result.error}"
            total_steps += 1
    validation = validate(sim.ground_truth())
    events.emit("validation", episode_id=episode_id, **validation.model_dump())
    return recorder.finish(validation)
