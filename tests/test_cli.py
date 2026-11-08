import sys
from pathlib import Path

import pytest

from isaacs_mom.orchestrator import cli
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.seeds import DEV_SEED_START, heldout_seeds

DEMO_ARGS = ["--ckpt", "ckpt-r0", "--seed", "900001", "--perturb-at", "2"]


@pytest.mark.parametrize("command", [["run"], ["demo", *DEMO_ARGS]])
@pytest.mark.parametrize(
    "argv",
    [
        ["--sim", "mujoco", "--task", "typo"],
        ["--sim", "fake", "--task", "heavy_box"],
        ["--sim", "fake", "--task", "fixed_feet"],
    ],
)
def test_a_task_that_cannot_run_is_refused_before_anything_starts(
    command: list[str], argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["mom", *command, *argv])
    monkeypatch.setattr(cli, "run_loop", lambda *a, **k: pytest.fail("the loop started"))
    monkeypatch.setattr(cli, "Argon", lambda *a, **k: pytest.fail("the run started"))
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 2
    assert "task" in capsys.readouterr().err


class UncheckedArgon:
    """Argon's stand-in: its startup check is a network call, which a refused take must not make."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def check(self) -> None:
        pytest.fail("a refused take checked Argon")


TAKE = ["mom", "demo", "--sim", "fake", "--student", "stub", "--ckpt", "ckpt-r0", "--perturb-at", "1"]


@pytest.mark.parametrize(
    ("seed", "run_id"), [(DEV_SEED_START, "take"), (min(heldout_seeds()), "take"), (900_005, "loop")]
)
def test_a_refused_take_checks_no_teacher_and_writes_nothing(
    seed: int, run_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    loop = EventLog(tmp_path / "runs" / "loop", "loop")
    loop.emit("episode_started", seed=DEV_SEED_START)
    written = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")), loop.path.read_text()
    monkeypatch.setattr(cli, "Argon", UncheckedArgon)
    monkeypatch.setattr(sys, "argv", [*TAKE, "--teacher", "argon", "--seed", str(seed), "--run-id", run_id])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 2
    assert (sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")), loop.path.read_text()) == written


def test_an_unseen_take_records_into_a_run_directory_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [*TAKE, "--seed", "900005", "--run-id", "take", "--out", "take.mp4"])
    cli.main()
    assert len(list((tmp_path / "runs" / "take" / "episodes").iterdir())) == 1
    assert (tmp_path / "take.mp4").stat().st_size > 0
    assert "run saved to runs/take" in capsys.readouterr().out
