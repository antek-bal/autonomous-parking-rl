from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import SAC

try:
    from src.wrapper import STREET_PARALLEL_PARKING_ID
except ImportError:
    from wrapper import STREET_PARALLEL_PARKING_ID


def build_env(render_mode: str | None):
    return gym.make(STREET_PARALLEL_PARKING_ID, render_mode=render_mode)


def run_trained_agent(model_path: str, render_mode: str | None = None, episodes: int = 5, sleep: float = 0.05):
    env = build_env(render_mode)
    try:
        print(f"Loading model from {model_path}...")
        model = SAC.load(model_path, env=env, custom_objects={"lr_schedule": lambda _: 0.0})

        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            print(f"Parking attempt {ep}")

            terminated = False
            truncated = False
            step = 0
            total_reward = 0.0

            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                print(
                    f"Step {step:03d} | Action: [Acceleration: {action[0]:.2f}, Turn: {action[1]:.2f}] | Success: {info.get('is_success')}")

                if render_mode == "human":
                    time.sleep(sleep)
                step += 1

            print(f"--- Attempt {ep} Finished | Result: {info.get('is_success')} | Reward sum: {total_reward:.2f} ---")

    finally:
        env.close()


def main():
    project_root = Path(__file__).resolve().parent.parent
    model_file = project_root / "models" / "sac_real_parallel"
    run_trained_agent(model_path=str(model_file), render_mode="human")


if __name__ == "__main__":
    main()