from pathlib import Path

from isaacs_mom.contracts import EpisodeManifest, Perturbation, Split
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.validator import Recorder, validate

MAX_STEPS = 20
PERTURB_SEED_OFFSET = 7919


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
) -> EpisodeManifest:
    """One recorded episode. If `policy` stops without success, `takeover` (when given) continues from there."""
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
    events.emit("episode_started", episode_id=episode_id, seed=seed, split=split, round=round_)
    feedback = "start"
    total_steps = 0
    for phase, active in enumerate([policy] if takeover is None else [policy, takeover]):
        if phase == 1:
            if validate(sim.ground_truth()).accepted:
                break
            recorder.mark_takeover()
            events.emit("decision", episode_id=episode_id, controller="teacher", takeover=True)
        for _ in range(max_steps):
            if total_steps == perturb_at:
                perturbation = Perturbation(kind="move_block", seed=seed + PERTURB_SEED_OFFSET)
                sim.perturb(perturbation)
                recorder.add_perturbation(perturbation)
                events.emit("perturbation", episode_id=episode_id, kind=perturbation.kind, step=total_steps)
            obs = sim.observe()
            call = active.act(obs, instruction, feedback)
            events.emit(
                "decision",
                episode_id=episode_id,
                controller=active.controller,
                checkpoint=active.checkpoint,
                step=obs.step,
                skill=call.model_dump() if call else None,
            )
            if call is None:
                break
            result = sim.execute(call)
            recorder.add_step(obs, call, result, active.controller, active.checkpoint)
            events.emit(
                "action_executed",
                episode_id=episode_id,
                step=obs.step,
                ok=result.ok,
                error=result.error,
                frame=str(recorder.dir / recorder.manifest.steps[-1].frame),
            )
            feedback = f"{call.skill}: {'ok' if result.ok else result.error}"
            total_steps += 1
    validation = validate(sim.ground_truth())
    events.emit("validation", episode_id=episode_id, **validation.model_dump())
    return recorder.finish(validation)
