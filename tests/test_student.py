import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from isaacs_mom.contracts import SKILLS, Observation, SkillCall, SkillTarget, TrainingExample
from isaacs_mom.orchestrator import cli
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.student import stub_finetune, tuning
from isaacs_mom.student.checkpoints import ADAPTER_FILES, BASE_CHECKPOINT, adapter_dir, resolve_adapter, valid_run_id
from isaacs_mom.student.policy import StudentPolicy
from isaacs_mom.student.prompt import UnparseableReply, format_target, parse_reply, prompt_text
from isaacs_mom.student.tuning import finetune

REACH_5 = SkillCall(skill="reach_left", params={"distance_cm": 5})


@pytest.mark.parametrize(
    "text",
    [
        '{"skill": "reach_left", "params": {"distance_cm": 5}}',
        '```json\n{"skill": "reach_left", "params": {"distance_cm": 5}}\n```',
        'Plan {not json} then {"skill": "reach_left", "params": {"distance_cm": 5}} since the box is to the left.',
    ],
)
def test_a_reply_containing_a_skill_call_parses(text: str) -> None:
    assert parse_reply(text) == REACH_5


@pytest.mark.parametrize(
    "text",
    [
        "",
        "reach left a bit",
        '{"skill": "fly"}',
        '{"skill": "reach_left", "params": {"torque": 1}}',
        '{"skill": ',
        "[1, 2]",
    ],
)
def test_a_reply_without_a_whitelisted_skill_call_is_unparseable(text: str) -> None:
    with pytest.raises(UnparseableReply):
        parse_reply(text)


@pytest.mark.parametrize("distance", ["NaN", '"nan"', "Infinity", "-Infinity", '"inf"', '"Infinity"', "1e999"])
def test_a_non_finite_distance_is_unparseable(distance: str) -> None:
    with pytest.raises(UnparseableReply):
        parse_reply(f'{{"skill": "reach_left", "params": {{"distance_cm": {distance}}}}}')


def test_a_reason_in_the_reply_is_dropped() -> None:
    assert parse_reply('{"skill": "hug", "params": {}, "reason": "x"}').reason == ""


def test_reply_distance_is_clamped_to_the_skill_limit() -> None:
    assert parse_reply('{"skill": "reach_left", "params": {"distance_cm": 90}}').params == {"distance_cm": 20.0}


def test_a_turn_reply_parses_its_angle() -> None:
    reply = '{"skill":"turn_left","params":{"angle_deg":30}}'
    assert parse_reply(reply) == SkillCall(skill="turn_left", params={"angle_deg": 30})


def test_reply_angle_is_clamped_to_the_skill_limit() -> None:
    assert parse_reply('{"skill": "turn_right", "params": {"angle_deg": 90}}').params == {"angle_deg": 45.0}


def test_training_target_round_trips_through_the_parser() -> None:
    call = SkillCall(skill="reach_right", params={"distance_cm": 7.26})
    assert parse_reply(format_target(call)) == SkillCall(skill="reach_right", params={"distance_cm": 7.3})


def test_training_target_has_no_reason() -> None:
    assert format_target(SkillTarget(skill="hug")) == '{"skill":"hug","params":{}}'


def test_prompt_names_the_task_the_last_result_and_every_skill() -> None:
    text = prompt_text(INSTRUCTION, "reach_left: ok")
    assert INSTRUCTION in text
    assert "reach_left: ok" in text
    assert all(skill in text for skill in SKILLS)


def test_prompt_gives_each_param_its_limit_and_describes_the_humanoid_task() -> None:
    text = prompt_text(INSTRUCTION, "start")
    assert "distance_cm" in text and "0 to 20" in text
    assert "angle_deg" in text and "0 to 45" in text
    assert not {"arm", "block", "zone", "push"} & set(text.lower().replace(".", " ").replace(",", " ").split())


def test_base_checkpoint_is_the_plain_model(tmp_path: Path) -> None:
    assert resolve_adapter(BASE_CHECKPOINT, tmp_path) is None


def test_fine_tuned_checkpoint_needs_a_complete_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "run-a" / "checkpoints" / "ckpt-r1"
    with pytest.raises(FileNotFoundError, match="run-a/ckpt-r1"):
        resolve_adapter("run-a/ckpt-r1", tmp_path)
    adapter.mkdir(parents=True)
    (adapter / ADAPTER_FILES[0]).write_text("{}")
    with pytest.raises(FileNotFoundError, match=ADAPTER_FILES[1]):
        resolve_adapter("run-a/ckpt-r1", tmp_path)
    (adapter / ADAPTER_FILES[1]).write_bytes(b"weights")
    assert resolve_adapter("run-a/ckpt-r1", tmp_path) == adapter


def test_adapter_dir_places_a_checkpoint_inside_its_run_without_touching_disk(tmp_path: Path) -> None:
    assert adapter_dir("run-a/ckpt-r2", tmp_path) == tmp_path / "run-a" / "checkpoints" / "ckpt-r2"
    assert adapter_dir(BASE_CHECKPOINT, tmp_path) is None
    assert adapter_dir("../ckpt-r1", tmp_path) is None


@pytest.mark.parametrize("run_id", ["run-20261107-173845", "a.b_c-1", "x"])
def test_a_plain_ascii_run_id_is_valid(run_id: str) -> None:
    assert valid_run_id(run_id)


@pytest.mark.parametrize("run_id", ["demo 1", "rün", ".hidden", "trailing.", ".", "..", "", "a/b", "a\\b"])
def test_a_run_id_that_a_checkpoint_id_could_not_carry_is_invalid(run_id: str) -> None:
    assert not valid_run_id(run_id)


@pytest.mark.parametrize(
    "checkpoint",
    [
        "ckpt-r1",
        "run-a/ckpt-rx",
        "run-a/ckpt-r1/x",
        "../ckpt-r1",
        "/ckpt-r1",
        "run-a/ckpt-r01",
        "run a/ckpt-r1",
        "rün/ckpt-r1",
        "...../ckpt-r1",
        "run-a\\ckpt-r1",
    ],
)
def test_a_checkpoint_id_without_its_run_is_rejected(tmp_path: Path, checkpoint: str) -> None:
    with pytest.raises(ValueError, match="checkpoint id"):
        resolve_adapter(checkpoint, tmp_path)


def test_the_model_cache_key_changes_when_an_adapters_weights_are_rewritten(tmp_path: Path) -> None:
    from isaacs_mom.student import vlm

    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"old")
    before = vlm.cache_key(tmp_path)
    assert vlm.cache_key(tmp_path) == before
    weights.write_bytes(b"newer")
    assert vlm.cache_key(tmp_path) != before
    assert vlm.cache_key(None) is None


REOBSERVE = '{"skill": "reobserve", "params": {}}'


def frame(step: int = 0, episode_id: str = "ep-1") -> Observation:
    return Observation(episode_id=episode_id, step=step, sim_time=0.0, head_rgb_jpeg=b"jpeg", joint_state=[])


def scripted(*replies: str) -> Callable[[bytes, str, str], str]:
    queue = list(replies)
    return lambda jpeg, instruction, feedback: queue.pop(0)


def test_reply_becomes_the_skill_call_and_its_latency_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted('{"skill": "reach_right", "params": {"distance_cm": 4}}'))
    with caplog.at_level(logging.INFO, logger="isaacs_mom.student"):
        call = policy.act(frame(), "hug", "start")
    assert call == SkillCall(skill="reach_right", params={"distance_cm": 4})
    assert policy.last_latency_ms >= 0
    assert " ms" in caplog.text


def test_unparseable_reply_reobserves_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted("I would hug the box"))
    with caplog.at_level(logging.WARNING, logger="isaacs_mom.student"):
        assert policy.act(frame(), "hug", "start") == SkillCall(skill="reobserve")
    assert "unparseable" in caplog.text


def test_non_finite_reply_reobserves_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted('{"skill": "reach_left", "params": {"distance_cm": NaN}}'))
    with caplog.at_level(logging.WARNING, logger="isaacs_mom.student"):
        assert policy.act(frame(), "hug", "start") == SkillCall(skill="reobserve")
    assert "unparseable" in caplog.text


def test_second_reobserve_in_a_row_means_done() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, REOBSERVE))
    assert policy.act(frame(0), "hug", "start") == SkillCall(skill="reobserve")
    assert policy.act(frame(1), "hug", "reobserve: ok") is None


def test_an_unparseable_reply_breaks_a_reobserve_streak() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, "??", REOBSERVE))
    policy.act(frame(0), "hug", "start")
    policy.act(frame(1), "hug", "reobserve: ok")
    assert policy.act(frame(2), "hug", "reobserve: ok") == SkillCall(skill="reobserve")


def test_unparseable_replies_never_end_the_episode() -> None:
    policy = StudentPolicy(generate=scripted("??", "??", "??"))
    assert all(policy.act(frame(i), "hug", "start") == SkillCall(skill="reobserve") for i in range(3))


def test_reobserve_streak_starts_over_in_a_new_episode() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, REOBSERVE))
    policy.act(frame(0, "ep-1"), "hug", "start")
    assert policy.act(frame(0, "ep-2"), "hug", "start") == SkillCall(skill="reobserve")


def test_missing_checkpoint_fails_before_any_model_call(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run-a/ckpt-r2"):
        StudentPolicy("run-a/ckpt-r2", runs_dir=tmp_path, generate=scripted())


ADAPTER = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"weights", "metrics.json": b"{}"}


def training_set(runs: Path, rows: list[TrainingExample], run_id: str = "run-a") -> Path:
    path = runs / run_id / "training_r1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    return path


def example(tmp_path: Path, target: SkillTarget, feedback: str = "start") -> TrainingExample:
    image = tmp_path / "0000.jpg"
    image.write_bytes(b"frame-bytes")
    return TrainingExample(image=str(image), instruction="hug", feedback=feedback, target=target)


def test_empty_training_set_keeps_the_previous_checkpoint(tmp_path: Path) -> None:
    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        raise AssertionError("nothing to train on")

    assert finetune(training_set(tmp_path, []), "run-a/ckpt-r1", 2, trainer=trainer) == "run-a/ckpt-r1"
    assert not (tmp_path / "run-a" / "checkpoints").exists()


def test_a_run_id_no_checkpoint_id_can_carry_fails_before_training(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="hug"))]

    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        raise AssertionError("trained before checking the run id")

    with pytest.raises(ValueError, match="demo 1"):
        finetune(training_set(tmp_path, rows, "demo 1"), "ckpt-r0", 1, trainer=trainer)


def test_finetune_ships_prompt_ready_examples_and_stores_the_adapter(tmp_path: Path) -> None:
    rows = [
        example(tmp_path, SkillTarget(skill="reach_left", params={"distance_cm": 12.34})),
        example(tmp_path, SkillTarget(skill="reobserve"), feedback="reach_left: ok"),
    ]
    shipped_examples: list[dict[str, str]] = []
    shipped_images: dict[str, bytes] = {}

    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        shipped_examples.extend(examples)
        shipped_images.update(images)
        return ADAPTER

    assert finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=trainer) == "run-a/ckpt-r1"
    assert list(shipped_images.values()) == [b"frame-bytes"]
    assert [e["target"] for e in shipped_examples] == [
        '{"skill":"reach_left","params":{"distance_cm":12.3}}',
        '{"skill":"reobserve","params":{}}',
    ]
    assert {e["image"] for e in shipped_examples} == set(shipped_images)
    adapter = tmp_path / "run-a" / "checkpoints" / "ckpt-r1"
    assert resolve_adapter("run-a/ckpt-r1", tmp_path) == adapter
    meta = json.loads((adapter / "meta.json").read_text())
    assert (meta["examples"], meta["frames"], meta["previous"]) == (2, 1, "ckpt-r0")


def test_two_runs_keep_their_own_round_one_checkpoints(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="hug"))]

    def trainer_for(run_id: str) -> Callable[[list[dict[str, str]], dict[str, bytes]], dict[str, bytes]]:
        return lambda examples, images: {**ADAPTER, "adapter_model.safetensors": run_id.encode()}

    first = finetune(training_set(tmp_path, rows, "run-a"), "ckpt-r0", 1, trainer=trainer_for("run-a"))
    second = finetune(training_set(tmp_path, rows, "run-b"), "ckpt-r0", 1, trainer=trainer_for("run-b"))
    assert (first, second) == ("run-a/ckpt-r1", "run-b/ckpt-r1")
    for checkpoint in (first, second):
        adapter = resolve_adapter(checkpoint, tmp_path)
        assert adapter is not None
        assert (adapter / "adapter_model.safetensors").read_bytes() == checkpoint.split("/")[0].encode()


def test_trainer_without_adapter_weights_leaves_no_checkpoint(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="hug"))]

    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        return {"adapter_config.json": b"{}"}

    with pytest.raises(RuntimeError, match="adapter_model.safetensors"):
        finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=trainer)
    assert not (tmp_path / "run-a" / "checkpoints").exists()


def test_rerunning_a_round_replaces_the_old_checkpoint(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="hug"))]
    checkpoints = tmp_path / "run-a" / "checkpoints"

    def first(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        return {**ADAPTER, "stale.bin": b"old"}

    def second(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        return {"adapter_config.json": b"{}", "adapter_model.safetensors": b"new"}

    finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=first)
    finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=second)
    out = checkpoints / "ckpt-r1"
    assert not (out / "stale.bin").exists()
    assert (out / "adapter_model.safetensors").read_bytes() == b"new"
    assert sorted(p.name for p in checkpoints.iterdir()) == ["ckpt-r1"]


def test_nested_trainer_file_name_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="hug"))]

    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        return {**ADAPTER, "sub/x.bin": b"x"}

    with pytest.raises(RuntimeError, match="not plain"):
        finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=trainer)
    assert not (tmp_path / "run-a" / "checkpoints").exists()


def mom_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *flags: str, cuda: bool) -> dict:
    """Runs `mom run --sim fake` up to the loop, recording what the loop was given instead of running it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(sys, "argv", ["mom", "run", "--sim", "fake", "--run-id", "t", *flags])
    given: dict = {}

    def run_loop(*args: object, **kwargs: object) -> list:
        given.update(kwargs)
        return []

    monkeypatch.setattr(cli, "run_loop", run_loop)
    cli.main()
    return given


def test_without_cuda_the_smolvlm_student_stops_the_run_before_any_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_:
        mom_run(monkeypatch, tmp_path, cuda=False)
    assert exit_.value.code == 2
    assert tuning.NO_CUDA in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_with_cuda_the_smolvlm_student_fine_tunes_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert mom_run(monkeypatch, tmp_path, cuda=True)["train_student"] is finetune


def test_the_stub_student_needs_no_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert mom_run(monkeypatch, tmp_path, "--student", "stub", cuda=False)["train_student"] is stub_finetune
