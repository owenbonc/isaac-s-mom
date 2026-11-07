# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""See the scene: `uv run python -m isaacs_mom.sim.mujoco.preview --seed 3`.

`--gif out.gif` films a scripted episode (third-person view beside the head camera), `--sheet out.png` saves the head
frame after every skill, `--move-bin-at N` moves the bin before step N, `--eval 40` scores the scripted policy over
seeds 0-39 through the validator, and `--viewer` opens MuJoCo's viewer.
"""

import argparse
import io
from collections.abc import Callable

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from isaacs_mom.contracts import Controller, Observation, Perturbation, SkillCall, Validation
from isaacs_mom.orchestrator.episode import MAX_STEPS, PERTURB_SEED_OFFSET
from isaacs_mom.sim.mujoco.scene import TABLE_TOP_Z, table_clearance
from isaacs_mom.sim.mujoco.sim import TWIST_STEP_LIMIT_DEG, MujocoSim
from isaacs_mom.sim.mujoco.task import TASKS
from isaacs_mom.sim.mujoco.walk import wrap
from isaacs_mom.validator import validate

REACH_TOLERANCE = 0.015
BACK_OFF_CM = 30.0
AIM_TOLERANCE_DEG = 4.0
PLACE_STANDOFF = 0.45  # pelvis to bin centre; the box lands ~6 cm short of the centre
STANDOFF_TOLERANCE = 0.03
PLACE_LATERAL_TOLERANCE = 0.05  # the bin is 7 cm wider than the box either side


class OraclePolicy:
    """Previews and tests only: it reads ground truth, which the student and teacher never see (SPEC §2 rule 1).

    Lines the hands up with the box, hugs and lifts it, backs away from the table, turns to the bin, walks up to it
    and places. It re-aims every step, so a bin moved mid-carry is simply found again.
    """

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: MujocoSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        gt, sim = self.sim.ground_truth(), self.sim
        x, y, heading = sim.pose()
        if gt.box_on_bin_floor:
            return None
        if not sim.closed:
            if gt.box_xyz[2] < TABLE_TOP_Z:
                return None  # dropped: nothing left to do
            # The stance the legs hold, not the swaying pelvis, is what the arms' working point moves with.
            box = _local(np.array(gt.box_xyz[:2]) - sim.goal_xy, sim.goal_yaw)
            dy = box[1] - sim.work_y
            if abs(dy) > REACH_TOLERANCE:
                return SkillCall(skill="reach_left" if dy > 0 else "reach_right", params={"distance_cm": abs(dy) * 100})
            return SkillCall(skill="hug")
        if not sim.lifted:
            return SkillCall(skill="lift")
        if not sim.task.walking:
            # Twist the waist until the box faces the bin, aiming from where the feet stand, then place.
            twist = np.degrees(wrap(np.arctan2(gt.bin_xy[1], gt.bin_xy[0]) - sim.waist))
            if abs(twist) > 2:
                angle = min(abs(twist), TWIST_STEP_LIMIT_DEG)
                return SkillCall(skill="turn_left" if twist > 0 else "turn_right", params={"angle_deg": angle})
            return SkillCall(skill="place")
        if table_clearance((gt.box_xyz[0], gt.box_xyz[1])) == 0.0:
            return SkillCall(skill="walk_back", params={"distance_cm": BACK_OFF_CM})
        to_bin = np.array(gt.bin_xy) - (x, y)
        dist = float(np.linalg.norm(to_bin))
        turn = np.degrees(wrap(np.arctan2(to_bin[1], to_bin[0]) - heading))
        ahead = dist - PLACE_STANDOFF
        close = abs(ahead) <= STANDOFF_TOLERANCE or "would walk into the bin" in feedback
        # Up close, a few degrees off is a few centimetres at the hands: re-aim only if the box would miss.
        miss = abs(_local(to_bin, heading)[1])
        if (abs(turn) > AIM_TOLERANCE_DEG and not close) or (close and miss > PLACE_LATERAL_TOLERANCE):
            return SkillCall(skill="turn_left" if turn > 0 else "turn_right", params={"angle_deg": abs(turn)})
        if not close:
            skill = "walk_forward" if ahead > 0 else "walk_back"
            return SkillCall(skill=skill, params={"distance_cm": abs(ahead) * 100})
        return SkillCall(skill="place")


def _local(v: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def film(sim: MujocoSim, frames: list[Image.Image], every: float = 0.08) -> None:
    last = [-1.0]
    head = mujoco.Renderer(sim.model, 240, 320)

    def capture() -> None:
        if sim.data.time - last[0] < every:
            return
        last[0] = sim.data.time
        third = Image.open(io.BytesIO(sim.render_third_person())).resize((480, 360))
        head.update_scene(sim.data, camera="head")
        canvas = Image.new("RGB", (800, 360), "white")
        canvas.paste(third, (0, 0))
        canvas.paste(Image.fromarray(head.render()), (480, 60))
        frames.append(canvas)

    sim.on_tick = capture


def run_scripted(
    sim: MujocoSim, seed: int, move_bin_at: int | None = None, after_step: Callable[[str], None] | None = None
) -> tuple[Validation, int]:
    """One OraclePolicy episode, with the bin moved before step `move_bin_at`; the verdict and the steps taken."""
    sim.reset(seed)
    policy = OraclePolicy(sim)
    steps, feedback = 0, "start"
    for step in range(MAX_STEPS):
        if step == move_bin_at:
            sim.perturb(Perturbation(kind="move_bin", seed=seed + PERTURB_SEED_OFFSET))
            if after_step:
                after_step("the bin moves")
        call = policy.act(sim.observe(), "", feedback)
        if call is None:
            break
        result = sim.execute(call)
        steps += 1
        feedback = f"{call.skill}: {'ok' if result.ok else result.error}"
        label = f"{call.skill} {' '.join(f'{v:.0f}' for v in call.params.values())}".strip()
        print(f"  {label:20s} ok={result.ok} {result.error or ''}")
        if after_step:
            after_step(label if result.ok else f"{label}: refused")
    return validate(sim.ground_truth()), steps


def _head_sheet(frames: list[tuple[str, bytes]], path: str) -> None:
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (320 * cols, 260 * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, jpeg) in enumerate(frames):
        x, y = (i % cols) * 320, (i // cols) * 260
        sheet.paste(Image.open(io.BytesIO(jpeg)), (x, y + 20))
        draw.text((x + 4, y + 4), f"{i}: {label}", fill="black")
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/preview.png")
    parser.add_argument("--gif", help="film a scripted episode to this path")
    parser.add_argument("--sheet", help="save the head frame after every scripted step to this path")
    parser.add_argument("--move-bin-at", type=int, help="move the bin before this step of the scripted episode")
    parser.add_argument("--eval", type=int, help="score the scripted policy over this many seeds")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--task", choices=sorted(TASKS), default="across_room")
    args = parser.parse_args()
    sim = MujocoSim(TASKS[args.task])
    sim.reset(args.seed)
    if args.viewer:
        import mujoco.viewer

        mujoco.viewer.launch(sim.model, sim.data)
        return
    if args.eval:
        accepted, falls, steps = 0, 0, []
        for seed in range(args.eval):
            verdict, n = run_scripted(sim, seed, args.move_bin_at)
            accepted += verdict.accepted
            falls += sim.ground_truth().robot_fallen
            steps.append(n)
            print(f"seed {seed}: {'accepted' if verdict.accepted else 'rejected'} in {n} steps: {verdict.reason}")
        print(f"{accepted}/{args.eval} accepted, {falls} falls, {np.mean(steps):.1f} steps per episode on average")
        return
    if args.gif or args.sheet:
        frames: list[Image.Image] = []
        heads: list[tuple[str, bytes]] = [("start", sim.observe().head_rgb_jpeg)]
        if args.gif:
            film(sim, frames)
        verdict, _ = run_scripted(
            sim, args.seed, args.move_bin_at, lambda label: heads.append((label, sim.observe().head_rgb_jpeg))
        )
        print(verdict.reason)
        if args.gif:
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
            print(f"saved {args.gif}")
        if args.sheet:
            _head_sheet(heads, args.sheet)
            print(f"saved {args.sheet}")
        return
    third = Image.open(io.BytesIO(sim.render_third_person()))
    head = Image.open(io.BytesIO(sim.observe().head_rgb_jpeg)).resize((640, 480))
    canvas = Image.new("RGB", (1280, 480), "white")
    canvas.paste(third, (0, 0))
    canvas.paste(head, (640, 0))
    canvas.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
