import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from isaacs_mom.contracts import SKILLS, Observation, SkillCall, SkillTarget, TrainingExample
from isaacs_mom.student.checkpoints import ADAPTER_FILES, BASE_CHECKPOINT, resolve_adapter
from isaacs_mom.student.policy import StudentPolicy
from isaacs_mom.student.prompt import UnparseableReply, format_target, parse_reply, prompt_text
from isaacs_mom.student.tuning import finetune

PUSH_5 = SkillCall(skill="push_forward", params={"distance_cm": 5})


@pytest.mark.parametrize(
    "text",
    [
        '{"skill": "push_forward", "params": {"distance_cm": 5}}',
        '```json\n{"skill": "push_forward", "params": {"distance_cm": 5}}\n```',
        'Plan {not json} then {"skill": "push_forward", "params": {"distance_cm": 5}} since the block is ahead.',
    ],
)
def test_a_reply_containing_a_skill_call_parses(text: str) -> None:
    assert parse_reply(text) == PUSH_5


@pytest.mark.parametrize(
    "text",
    [
        "",
        "push forward a bit",
        '{"skill": "fly"}',
        '{"skill": "push_forward", "params": {"torque": 1}}',
        '{"skill": ',
        "[1, 2]",
    ],
)
def test_a_reply_without_a_whitelisted_skill_call_is_unparseable(text: str) -> None:
    with pytest.raises(UnparseableReply):
        parse_reply(text)


@pytest.mark.parametrize("distance", ["NaN", '"nan"', "Infinity", "-Infinity"])
def test_a_non_finite_distance_is_unparseable(distance: str) -> None:
    with pytest.raises(UnparseableReply):
        parse_reply(f'{{"skill": "push_forward", "params": {{"distance_cm": {distance}}}}}')


def test_a_reason_in_the_reply_is_dropped() -> None:
    assert parse_reply('{"skill": "retract", "params": {}, "reason": "x"}').reason == ""


def test_reply_distance_is_clamped_to_the_skill_limit() -> None:
    assert parse_reply('{"skill": "push_forward", "params": {"distance_cm": 90}}').params == {"distance_cm": 20.0}


def test_training_target_round_trips_through_the_parser() -> None:
    call = SkillCall(skill="approach_left", params={"distance_cm": 7.26})
    assert parse_reply(format_target(call)) == SkillCall(skill="approach_left", params={"distance_cm": 7.3})


def test_training_target_has_no_reason() -> None:
    assert format_target(SkillTarget(skill="retract")) == '{"skill":"retract","params":{}}'


def test_prompt_names_the_task_the_last_result_and_every_skill() -> None:
    text = prompt_text("Push the red block into the green target zone.", "push_forward: ok")
    assert "Push the red block into the green target zone." in text
    assert "push_forward: ok" in text
    assert all(skill in text for skill in SKILLS)


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


@pytest.mark.parametrize("checkpoint", ["ckpt-r1", "run-a/ckpt-rx", "run-a/ckpt-r1/x", "../ckpt-r1", "/ckpt-r1"])
def test_a_checkpoint_id_without_its_run_is_rejected(tmp_path: Path, checkpoint: str) -> None:
    with pytest.raises(ValueError, match="checkpoint id"):
        resolve_adapter(checkpoint, tmp_path)


REOBSERVE = '{"skill": "reobserve", "params": {}}'


def frame(step: int = 0, episode_id: str = "ep-1") -> Observation:
    return Observation(episode_id=episode_id, step=step, sim_time=0.0, head_rgb_jpeg=b"jpeg", joint_state=[])


def scripted(*replies: str) -> Callable[[bytes, str, str], str]:
    queue = list(replies)
    return lambda jpeg, instruction, feedback: queue.pop(0)


def test_reply_becomes_the_skill_call_and_its_latency_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted('{"skill": "approach_left", "params": {"distance_cm": 4}}'))
    with caplog.at_level(logging.INFO, logger="isaacs_mom.student"):
        call = policy.act(frame(), "push", "start")
    assert call == SkillCall(skill="approach_left", params={"distance_cm": 4})
    assert policy.last_latency_ms >= 0
    assert " ms" in caplog.text


def test_unparseable_reply_reobserves_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted("I would push the block"))
    with caplog.at_level(logging.WARNING, logger="isaacs_mom.student"):
        assert policy.act(frame(), "push", "start") == SkillCall(skill="reobserve")
    assert "unparseable" in caplog.text


def test_non_finite_reply_reobserves_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted('{"skill": "push_forward", "params": {"distance_cm": NaN}}'))
    with caplog.at_level(logging.WARNING, logger="isaacs_mom.student"):
        assert policy.act(frame(), "push", "start") == SkillCall(skill="reobserve")
    assert "unparseable" in caplog.text


def test_second_reobserve_in_a_row_means_done() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, REOBSERVE))
    assert policy.act(frame(0), "push", "start") == SkillCall(skill="reobserve")
    assert policy.act(frame(1), "push", "reobserve: ok") is None


def test_an_unparseable_reply_breaks_a_reobserve_streak() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, "??", REOBSERVE))
    policy.act(frame(0), "push", "start")
    policy.act(frame(1), "push", "reobserve: ok")
    assert policy.act(frame(2), "push", "reobserve: ok") == SkillCall(skill="reobserve")


def test_unparseable_replies_never_end_the_episode() -> None:
    policy = StudentPolicy(generate=scripted("??", "??", "??"))
    assert all(policy.act(frame(i), "push", "start") == SkillCall(skill="reobserve") for i in range(3))


def test_reobserve_streak_starts_over_in_a_new_episode() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, REOBSERVE))
    policy.act(frame(0, "ep-1"), "push", "start")
    assert policy.act(frame(0, "ep-2"), "push", "start") == SkillCall(skill="reobserve")


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
    return TrainingExample(image=str(image), instruction="push", feedback=feedback, target=target)


def test_empty_training_set_keeps_the_previous_checkpoint(tmp_path: Path) -> None:
    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        raise AssertionError("nothing to train on")

    assert finetune(training_set(tmp_path, []), "run-a/ckpt-r1", 2, trainer=trainer) == "run-a/ckpt-r1"
    assert not (tmp_path / "run-a" / "checkpoints").exists()


def test_finetune_ships_prompt_ready_examples_and_stores_the_adapter(tmp_path: Path) -> None:
    rows = [
        example(tmp_path, SkillTarget(skill="push_forward", params={"distance_cm": 12.34})),
        example(tmp_path, SkillTarget(skill="reobserve"), feedback="push_forward: ok"),
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
        '{"skill":"push_forward","params":{"distance_cm":12.3}}',
        '{"skill":"reobserve","params":{}}',
    ]
    assert {e["image"] for e in shipped_examples} == set(shipped_images)
    adapter = tmp_path / "run-a" / "checkpoints" / "ckpt-r1"
    assert resolve_adapter("run-a/ckpt-r1", tmp_path) == adapter
    meta = json.loads((adapter / "meta.json").read_text())
    assert (meta["examples"], meta["frames"], meta["previous"]) == (2, 1, "ckpt-r0")


def test_two_runs_keep_their_own_round_one_checkpoints(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="retract"))]

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
    rows = [example(tmp_path, SkillTarget(skill="retract"))]

    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        return {"adapter_config.json": b"{}"}

    with pytest.raises(RuntimeError, match="adapter_model.safetensors"):
        finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=trainer)
    assert not (tmp_path / "run-a" / "checkpoints").exists()


def test_rerunning_a_round_replaces_the_old_checkpoint(tmp_path: Path) -> None:
    rows = [example(tmp_path, SkillTarget(skill="retract"))]
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
    rows = [example(tmp_path, SkillTarget(skill="retract"))]

    def trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
        return {**ADAPTER, "sub/x.bin": b"x"}

    with pytest.raises(RuntimeError, match="not plain"):
        finetune(training_set(tmp_path, rows), "ckpt-r0", 1, trainer=trainer)
    assert not (tmp_path / "run-a" / "checkpoints").exists()
