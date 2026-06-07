from __future__ import annotations

import os

import gymnasium as gym
import highway_env  # noqa: F401 - registers the highway-env environments
from stable_baselines3 import SAC

from src.wrapper import STREET_PARALLEL_PARKING_ID


def build_env():
    return gym.make(STREET_PARALLEL_PARKING_ID)


def main(total_timesteps: int = 200000):
    env = build_env()
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=1e-3,
            buffer_size=50000,
            batch_size=256,
            gamma=0.95,
        )

        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        model_dir = "../models"
        os.makedirs(model_dir, exist_ok=True)
        model.save(f"{model_dir}/sac_real_parallel")
        print("Trening zakończony sukcesem! Pora na testy.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
