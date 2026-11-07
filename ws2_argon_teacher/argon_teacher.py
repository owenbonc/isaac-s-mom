"""
WS2 - Argon AI Teacher (reward design loop for RL)
Loop: failure log (+ optional camera frame) -> Argon diagnoses -> new reward weights
      -> train policy (MOCK for now, replace with Isaac Lab PPO) -> metrics -> Argon again.

Run:  python argon_teacher.py
Needs: pip install google-genai pydantic   and env var GEMINI_API_KEY
"""
import os, json, time, random
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

MODEL = "gemini-4-argon"          # use "gemini-3.8-flash" for cheap testing
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
LOG_FILE = Path("argon_decisions.jsonl")   # WS5 dashboard reads this


# ---------- What Argon must return (structured JSON) ----------
class RewardWeights(BaseModel):
    dist_block_to_target: float = Field(description="weight on -distance(block, target), >=0")
    dist_hand_to_block: float = Field(description="weight on -distance(hand, block), >=0")
    contact_bonus: float = Field(description="bonus while hand touches block, >=0")
    success_bonus: float = Field(description="one-off bonus when block is inside target zone, >=0")
    collision_penalty: float = Field(description="penalty for table/self collisions, >=0")
    action_penalty: float = Field(description="penalty on action magnitude (smoothness), >=0")

class TeacherDecision(BaseModel):
    diagnosis: str = Field(description="most likely cause of the failure, based only on the evidence")
    evidence: list[str] = Field(description="observations that support the diagnosis")
    confidence: float = Field(description="0-1")
    reward: RewardWeights
    expected_effect: str = Field(description="what should change in the next training run")


SYSTEM = """You are the teacher for a fixed-base Unitree G1 humanoid in Isaac Sim.
Task: push a colored block into a target zone using one arm. Sensor: one head RGB camera.
A student policy is trained with PPO using a reward = weighted sum of fixed terms.
Given the failure log (and camera frame if provided), diagnose the failure and output new reward weights.
Change only what the evidence justifies. Be honest about uncertainty."""


def ask_argon(failure_log: dict, history: list, frame_path: str | None = None) -> TeacherDecision:
    parts = [
        f"Previous rounds (weights -> results):\n{json.dumps(history, indent=2)}",
        f"Latest failure log:\n{json.dumps(failure_log, indent=2)}",
    ]
    if frame_path and Path(frame_path).exists():
        parts.append(types.Part.from_bytes(data=Path(frame_path).read_bytes(), mime_type="image/png"))

    t0 = time.time()
    resp = client.models.generate_content(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=TeacherDecision,
        ),
    )
    latency = time.time() - t0
    decision = resp.parsed
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({"t": time.time(), "latency_s": round(latency, 2),
                            "model": MODEL, "decision": decision.model_dump()}) + "\n")
    print(f"[Argon {latency:.1f}s] {decision.diagnosis} (conf {decision.confidence:.2f})")
    return decision


# ---------- MOCK training: replace with Isaac Lab PPO run (WS1 handoff) ----------
def train_policy_mock(w: RewardWeights) -> dict:
    """Fake but reward-sensitive: rewards that guide the hand to the block AND the block to
    the target score higher. Replace body with: write weights -> launch Isaac Lab train -> read metrics."""
    reach = min(w.dist_hand_to_block, 2) / 2
    push = min(w.dist_block_to_target, 2) / 2
    contact = min(w.contact_bonus, 1)
    success = max(0.0, min(1.0, 0.15 + 0.35 * reach * push + 0.25 * contact * push
                           - 0.1 * w.action_penalty + random.uniform(-0.05, 0.05)))
    return {
        "success_rate": round(success, 2),
        "mean_final_block_to_target_m": round(0.30 * (1 - success), 3),
        "hand_reached_block_rate": round(min(1, 0.2 + 0.8 * reach), 2),
        "collisions_per_episode": round(max(0, 0.5 - w.collision_penalty * 0.3), 2),
    }


def main(rounds: int = 3):
    weights = RewardWeights(dist_block_to_target=1.0, dist_hand_to_block=0.0, contact_bonus=0.0,
                            success_bonus=1.0, collision_penalty=0.0, action_penalty=0.01)
    metrics = train_policy_mock(weights)
    history = [{"round": 0, "reward": weights.model_dump(), "metrics": metrics}]
    print("round 0", metrics)

    for r in range(1, rounds + 1):
        failure_log = {"task": "push block to target", "metrics": metrics,
                       "notes": "student often never touches the block"}  # WS4/WS3 fill this for real
        decision = ask_argon(failure_log, history, frame_path="frame.png")
        metrics = train_policy_mock(decision.reward)
        history.append({"round": r, "diagnosis": decision.diagnosis,
                        "reward": decision.reward.model_dump(), "metrics": metrics})
        print(f"round {r}", metrics)

    Path("reward_history.json").write_text(json.dumps(history, indent=2))
    print("Saved reward_history.json and argon_decisions.jsonl")


if __name__ == "__main__":
    main()
