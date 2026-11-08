import sys
from pathlib import Path

import pytest

from isaacs_mom.contracts import EpisodeManifest, Event
from isaacs_mom.orchestrator import cli
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import INSTRUCTION

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


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["mom", "run", "--sim", "fake", "--student", "stub", "--rounds", "0", *argv])
    cli.main()


def logged(run_dir: Path) -> list[Event]:
    return [Event.model_validate_json(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


# One argument whatever it holds, even text that reads like an option.
ODD = '--task walk; "quoted" $(touch x)\nput it in the bin'


@pytest.mark.parametrize(("argv", "instruction"), [([], INSTRUCTION), ([f"--instruction={ODD}"], ODD)])
def test_mom_run_records_its_instruction_first_and_tests_on_it(
    argv: list[str], instruction: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_cli(monkeypatch, "--run-id", "r", *argv)
    run_dir = tmp_path / "runs" / "r"
    first = logged(run_dir)[0]
    assert (first.type, first.data) == ("instruction", {"text": instruction, "source": "cli"})
    manifests = [EpisodeManifest.model_validate_json(p.read_text()) for p in run_dir.glob("episodes/*/manifest.json")]
    assert manifests and {m.instruction for m in manifests} == {instruction}


def test_mom_run_on_a_taught_run_keeps_its_instruction_and_counts_its_argon_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARGON_CALL_BUDGET", "2")
    taught = EventLog(tmp_path / "runs" / "r", "r")
    taught.emit("instruction", text="put the box in the bin", source="teach")
    for call in (1, 2):  # the task generator's
        taught.emit("argon_call", purpose="task", call=call, budget=2, ok=True)
    run_cli(monkeypatch, "--run-id", "r", "--instruction", "put the box in the bin")
    instructions = [e.data for e in logged(tmp_path / "runs" / "r") if e.type == "instruction"]
    assert instructions == [{"text": "put the box in the bin", "source": "teach"}]
    assert "argon: 2/2 calls, 0 unavailable or empty, budget used up" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", "   ", "x" * 201, "put\0it in"])
def test_mom_run_refuses_an_instruction_the_teach_box_could_not_send(
    text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_:
        run_cli(monkeypatch, f"--instruction={text}")
    assert exit_.value.code == 2
    assert "argument --instruction" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
