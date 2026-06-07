from __future__ import annotations

import time

import gymnasium as gym
import highway_env

try:
    from src.wrapper import STREET_PARALLEL_PARKING_ID
except ImportError:  # pragma: no cover
    from wrapper import STREET_PARALLEL_PARKING_ID


def build_env(render_mode: str | None):
    return gym.make(STREET_PARALLEL_PARKING_ID, render_mode=render_mode)


def run_random_agent(render_mode: str | None = None, max_steps: int = 500, sleep: float = 0.05, seed: int | None = None):
    env = build_env(render_mode)
    try:
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step:03d} reward={reward: .3f} terminated={terminated} truncated={truncated} success={info.get('is_success')}"
            )
            if render_mode == "human":
                time.sleep(sleep)
            step += 1

    finally:
        env.close()


def main():
    run_random_agent(render_mode="human")


if __name__ == "__main__":
    main()
