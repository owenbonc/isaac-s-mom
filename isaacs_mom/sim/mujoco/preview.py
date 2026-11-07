# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""See the scene: `uv run python -m isaacs_mom.sim.mujoco.preview --seed 3`.

`--gif out.gif` films a scripted episode (head camera beside a third-person view); `--viewer` opens MuJoCo's viewer.
"""

import argparse
import io

import mujoco
import numpy as np
from PIL import Image

from isaacs_mom.contracts import Controller, Observation, SkillCall
from isaacs_mom.sim.mujoco.scene import TABLE_TOP_Z
from isaacs_mom.sim.mujoco.sim import MAX_TURN_STEP_DEG, PLACE_X, MujocoSim


class OraclePolicy:
    """Previews and tests only: it reads ground truth, which the student and teacher never see (SPEC §2 rule 1)."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: MujocoSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        gt, sim = self.sim.ground_truth(), self.sim
        on_table = gt.box_xyz[2] > TABLE_TOP_Z
        if not sim.closed and on_table:
            dy = gt.box_xyz[1] - sim.work_y
            if abs(dy) > 0.01:
                return SkillCall(skill="reach_left" if dy > 0 else "reach_right", params={"distance_cm": abs(dy) * 100})
            return SkillCall(skill="hug")
        if sim.closed and not sim.lifted:
            return SkillCall(skill="lift")
        if sim.lifted:
            aim = np.arctan2(gt.bin_xy[1], gt.bin_xy[0]) - np.arctan2(sim.work_y, PLACE_X)
            turn = np.degrees(aim - sim.waist)
            if abs(turn) > 2:
                return SkillCall(
                    skill="turn_left" if turn > 0 else "turn_right",
                    params={"angle_deg": min(abs(turn), MAX_TURN_STEP_DEG)},
                )
            return SkillCall(skill="place")
        return None


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/preview.png")
    parser.add_argument("--gif", help="film a scripted episode to this path")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    sim = MujocoSim()
    sim.reset(args.seed)
    if args.viewer:
        import mujoco.viewer

        mujoco.viewer.launch(sim.model, sim.data)
        return
    if args.gif:
        frames: list[Image.Image] = []
        film(sim, frames)
        policy = OraclePolicy(sim)
        for _ in range(20):
            call = policy.act(sim.observe(), "", "")
            if call is None:
                break
            result = sim.execute(call)
            print(f"{call.skill:11s} {call.params} ok={result.ok} {result.error or ''}")
        print(sim.ground_truth())
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
        print(f"saved {args.gif}")
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
