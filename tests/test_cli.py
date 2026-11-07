import sys

import pytest

from isaacs_mom.orchestrator import cli

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
