from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import SAC, HerReplayBuffer
from src.wrapper import STREET_PARALLEL_PARKING_ID


def build_env():
    return gym.make(STREET_PARALLEL_PARKING_ID)


def main(total_timesteps: int = 500000):
    env = build_env()
    try:
        model_dir = Path(__file__).resolve().parent.parent / "models"

        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=str(Path(__file__).resolve().parent.parent / "tensorboard_logs"),
            learning_rate=3e-4,

            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),

            buffer_size=200000,
            batch_size=256,
            gamma=0.995,
            tau=0.02,
            learning_starts=5000,
            ent_coef="auto",
            policy_kwargs={"net_arch": {"pi": [256, 256], "qf": [256, 256]}},
        )

        print(f"Starting training {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        os.makedirs(model_dir, exist_ok=True)
        model.save(str(model_dir / "sac_real_parallel_40k"))
        print("Training finished")
    finally:
        env.close()


if __name__ == "__main__":
    main()