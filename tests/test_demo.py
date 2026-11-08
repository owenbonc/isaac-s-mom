import io
from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import pytest
from PIL import Image

from isaacs_mom.contracts import Controller, EpisodeManifest, Event, Observation, SkillCall, Validation
from isaacs_mom.orchestrator.demo import _captions, run_demo
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import StudentFactory
from isaacs_mom.orchestrator.video import record
from isaacs_mom.seeds import DEV_SEED_START, SealedSeedError, heldout_seeds
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.student import StubStudent
from isaacs_mom.teacher.argon import Argon, stub_generate


def jpeg(colour: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="JPEG")
    return out.getvalue()


def frame_count(path: Path) -> int:
    return sum(1 for _ in iio.imiter(path))


def stub_argon(events: EventLog) -> Argon:
    return Argon(events, generate=stub_generate)  # no network in tests


def unbuilt(events: EventLog) -> Argon:
    pytest.fail("a refused take built its teacher")


def take(
    run_dir: Path,
    *,
    seed: int = 900_001,
    perturb_at: int = 3,
    sim: FakeSim | None = None,
    make_student: StudentFactory = StubStudent,
    make_argon: Callable[[EventLog], Argon] = stub_argon,
) -> EpisodeManifest:
    """The judged take on FakeSim with the stand-in student and teacher, its MP4 beside its run directory."""
    return run_demo(
        sim or FakeSim(),
        ckpt="ckpt-r0",
        seed=seed,
        perturb_at=perturb_at,
        out=run_dir.with_suffix(".mp4"),
        run_dir=run_dir,
        run_id=run_dir.name,
        make_argon=make_argon,
        make_student=make_student,
    )


def logged(run_dir: Path) -> list[Event]:
    return [Event.model_validate_json(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


class Grabber:
    """Test-only student: lines its hands up with the box, hugs and lifts it, then stops, using ground truth."""

    controller: Controller = "student"
    checkpoint: str | None = "ckpt-r0"

    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.step == 0:
            y = self.sim.ground_truth().box_xyz[1]
            return SkillCall(skill="reach_left" if y > 0 else "reach_right", params={"distance_cm": abs(y) * 100})
        return {1: SkillCall(skill="hug"), 2: SkillCall(skill="lift")}.get(obs.step)


def test_record_writes_one_video_frame_per_input_frame(tmp_path: Path) -> None:
    frames = [
        (jpeg((i * 20, 0, 0), (320, 240)), jpeg((0, i * 20, 0), (640, 480)), f"student step {i}") for i in range(7)
    ]
    out = tmp_path / "out.mp4"
    assert record(frames, out, fps=4) == 7
    assert frame_count(out) == 7
    width, height = next(iter(iio.imiter(out))).shape[1::-1]
    assert (width % 16, height % 16) == (0, 0)


def test_record_refuses_an_empty_take(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        record([], tmp_path / "out.mp4", fps=2)
    assert not (tmp_path / "out.mp4").exists()


def test_demo_records_every_step_and_the_perturbation(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    manifest = take(run_dir)
    assert frame_count(run_dir.with_suffix(".mp4")) == len(manifest.steps) > 3
    events = logged(run_dir)
    perturbations = [e for e in events if e.type == "perturbation"]
    assert [e.data["step"] for e in perturbations] == [3]
    assert [e.data["demo"] for e in events if e.type == "episode_started"] == [True]


def test_demo_refuses_a_heldout_seed(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    with pytest.raises(ValueError) as refused:
        take(run_dir, seed=next(iter(heldout_seeds())), make_argon=unbuilt)
    assert refused.type is SealedSeedError  # the held-out guard's refusal, though the seed is below 900000 too
    assert not run_dir.exists() and not run_dir.with_suffix(".mp4").exists()


def test_demo_refuses_a_seed_below_the_unseen_range_before_writing_anything(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    for seed in (DEV_SEED_START, 899_999):
        with pytest.raises(ValueError, match="900000"):
            take(run_dir, seed=seed, make_argon=unbuilt)
    assert not run_dir.exists() and not run_dir.with_suffix(".mp4").exists()
    assert take(run_dir, seed=900_000).seed == 900_000


def test_demo_refuses_a_run_directory_that_already_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "loop"
    EventLog(run_dir, "loop").emit("episode_started", seed=DEV_SEED_START)  # a loop run's, whose demos train
    written = (run_dir / "events.jsonl").read_text()
    with pytest.raises(FileExistsError):
        take(run_dir, make_argon=unbuilt)
    assert list(run_dir.iterdir()) == [run_dir / "events.jsonl"] and (run_dir / "events.jsonl").read_text() == written
    assert not run_dir.with_suffix(".mp4").exists()


def test_the_takes_captions_word_its_verdict_as_the_dashboard_does(tmp_path: Path) -> None:
    rescued = take(tmp_path / "runs" / "demo")
    took, failed = rescued.takeover_step, rescued.validation
    assert took is not None and failed is not None and not failed.accepted  # the stand-ins never finish the task
    done = Validation(task_success=True, physics_valid=True, reason="box at rest inside bin")
    alone = rescued.model_copy(update={"validation": done, "takeover_step": None})

    def last(manifest: EpisodeManifest) -> str:
        return _captions(manifest)[-1].split("validator: ")[1]

    assert last(rescued) == f"failed after the teacher took over at step {took}: {failed.reason}"
    assert (
        last(rescued.model_copy(update={"validation": done})) == f"succeeded after the teacher took over at step {took}"
    )
    assert last(alone) == "succeeded, by the student alone"
    assert _captions(rescued)[0].endswith("validator: pending")


def test_the_captions_name_an_applied_move_from_the_step_it_applied(tmp_path: Path) -> None:
    manifest = take(tmp_path / "runs" / "demo", perturb_at=3)
    assert [p.before_step for p in manifest.perturbations] == [3]
    assert ["box moved" in caption for caption in _captions(manifest)] == [i >= 3 for i in range(len(manifest.steps))]


def test_a_take_without_an_applied_move_captions_none(tmp_path: Path) -> None:
    sim = FakeSim()
    held = take(tmp_path / "runs" / "held", perturb_at=3, sim=sim, make_student=lambda _: Grabber(sim))
    refusals = [(e.data["step"], e.data["error"]) for e in logged(tmp_path / "runs" / "held") if e.type == "error"]
    assert refusals == [(3, "move_box not applied: the sim refused it and nothing moved")]  # the box was in hand
    never = take(tmp_path / "runs" / "never", perturb_at=-1)
    for manifest in (held, never):
        assert manifest.perturbations == []
        assert not any("moved" in caption for caption in _captions(manifest))
