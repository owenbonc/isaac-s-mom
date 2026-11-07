import sys

import pytest

from isaacs_mom.orchestrator import cli


@pytest.mark.parametrize(
    "argv",
    [
        ["--sim", "mujoco", "--task", "typo"],
        ["--sim", "fake", "--task", "heavy_box"],
        ["--sim", "fake", "--task", "fixed_feet"],
    ],
)
def test_a_task_that_cannot_run_is_refused_before_anything_starts(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["mom", "run", *argv])
    monkeypatch.setattr(cli, "run_loop", lambda *a, **k: pytest.fail("the loop started"))
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 2
    assert "task" in capsys.readouterr().err
