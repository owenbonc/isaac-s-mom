import io
from pathlib import Path

import imageio.v3 as iio
import pytest
from PIL import Image

from isaacs_mom.contracts import EpisodeManifest, Event, Validation
from isaacs_mom.orchestrator.demo import _captions, run_demo
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.video import record
from isaacs_mom.seeds import SealedSeedError, heldout_seeds
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.student import StubStudent
from isaacs_mom.teacher.argon import Argon, stub_generate


def jpeg(colour: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="JPEG")
    return out.getvalue()


def frame_count(path: Path) -> int:
    return sum(1 for _ in iio.imiter(path))


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
    out = tmp_path / "demo.mp4"
    manifest = run_demo(
        FakeSim(),
        ckpt="ckpt-r0",
        seed=900_001,
        perturb_at=3,
        out=out,
        run_dir=run_dir,
        run_id="demo",
        argon=Argon(EventLog(run_dir, "demo"), generate=stub_generate),  # no network in tests
        make_student=StubStudent,
    )
    assert frame_count(out) == len(manifest.steps) > 3
    events = [Event.model_validate_json(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    perturbations = [e for e in events if e.type == "perturbation"]
    assert [e.data["step"] for e in perturbations] == [3]
    assert [e.data["demo"] for e in events if e.type == "episode_started"] == [True]


def test_demo_refuses_a_heldout_seed(tmp_path: Path) -> None:
    seed = next(iter(heldout_seeds()))
    with pytest.raises(SealedSeedError):
        run_demo(
            FakeSim(),
            ckpt="ckpt-r0",
            seed=seed,
            perturb_at=3,
            out=tmp_path / "d.mp4",
            run_dir=tmp_path,
            run_id="d",
            argon=Argon(EventLog(tmp_path, "d"), generate=stub_generate),  # no network in tests
            make_student=StubStudent,
        )
    assert not (tmp_path / "d.mp4").exists()


def test_the_takes_captions_word_its_verdict_as_the_dashboard_does(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    rescued = run_demo(
        FakeSim(),
        ckpt="ckpt-r0",
        seed=900_001,
        perturb_at=3,
        out=tmp_path / "demo.mp4",
        run_dir=run_dir,
        run_id="demo",
        argon=Argon(EventLog(run_dir, "demo"), generate=stub_generate),  # no network in tests
        make_student=StubStudent,
    )
    took, failed = rescued.takeover_step, rescued.validation
    assert took is not None and failed is not None and not failed.accepted  # the stand-ins never finish the task
    done = Validation(task_success=True, physics_valid=True, reason="box at rest inside bin")
    alone = rescued.model_copy(update={"validation": done, "takeover_step": None})

    def last(manifest: EpisodeManifest) -> str:
        return _captions(manifest, 3)[-1].split("validator: ")[1]

    assert last(rescued) == f"failed after the teacher took over at step {took}: {failed.reason}"
    assert (
        last(rescued.model_copy(update={"validation": done})) == f"succeeded after the teacher took over at step {took}"
    )
    assert last(alone) == "succeeded, by the student alone"
    assert _captions(rescued, 3)[0].endswith("validator: pending")
