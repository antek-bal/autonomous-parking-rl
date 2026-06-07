from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register

from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


STREET_PARALLEL_PARKING_ID = "street-parallel-parking-v0"


class StreetParallelParkingEnv(ParkingEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 120,
                "screen_width": 1000,
                "screen_height": 300,
                "centering_position": [0.35, 0.5],
                "scaling": 6,
                "controlled_vehicles": 1,
                "vehicles_count": 0,
                "add_walls": False,
            }
        )
        return config

    def __init__(
        self,
        config: dict | None = None,
        render_mode: str | None = None,
        ego_start: tuple[float, float] = (-18.0, 0.0),
        goal_position: tuple[float, float] = (0.0, -2.8),
        parked_car_offset: float = 6.5,
        parked_car_y: float = -2.8,
        parked_car_heading: float = 0.0,
    ) -> None:
        self.ego_start = np.array(ego_start, dtype=float)
        self.goal_position = np.array(goal_position, dtype=float)
        self.parked_car_offset = float(parked_car_offset)
        self.parked_car_y = float(parked_car_y)
        self.parked_car_heading = float(parked_car_heading)
        super().__init__(config, render_mode)

    def _create_road(self, spots: int = 14) -> None:
        rng = np.random.RandomState(int(self.np_random.integers(0, 2**31 - 1)))
        self.road = Road(
            network=RoadNetwork.straight_road_network(2, length=180, speed_limit=20),
            np_random=rng,
            record_history=self.config["show_trajectories"],
        )

    def _add_curb(self) -> None:
        sidewalk_y = -5.35
        curb_y = -4.45
        for x in np.arange(-35.0, 95.0, 2.5):
            sidewalk = Obstacle(self.road, [x, sidewalk_y], heading=0.0)
            sidewalk.LENGTH = 1.9
            sidewalk.WIDTH = 0.28
            sidewalk.collidable = False
            sidewalk.solid = False
            sidewalk.check_collisions = False
            sidewalk.color = (205, 205, 205)
            self.road.objects.append(sidewalk)

            curb = Obstacle(self.road, [x, curb_y], heading=0.0)
            curb.LENGTH = 1.1
            curb.WIDTH = 0.18
            curb.collidable = False
            curb.solid = False
            curb.check_collisions = False
            curb.color = (120, 120, 120)
            self.road.objects.append(curb)

    def _create_vehicles(self) -> None:
        self.controlled_vehicles = []

        ego = self.action_type.vehicle_class(
            self.road,
            self.ego_start,
            heading=0.0,
            speed=0.0,
        )
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)


        parked_positions = [
            np.array([self.goal_position[0] - self.parked_car_offset, self.parked_car_y]),
            np.array([self.goal_position[0] + self.parked_car_offset, self.parked_car_y]),
        ]
        for position in parked_positions:
            parked = Vehicle(self.road, position, heading=self.parked_car_heading, speed=0.0)
            parked.color = (230, 190, 0)
            self.road.vehicles.append(parked)

        goal = Landmark(self.road, self.goal_position, heading=self.parked_car_heading)
        ego.goal = goal
        self.road.objects.append(goal)

        self._add_curb()


ParallelParkingWrapper = StreetParallelParkingEnv


def register_street_parallel_parking_env() -> None:
    try:
        register(
            id=STREET_PARALLEL_PARKING_ID,
            entry_point="src.wrapper:StreetParallelParkingEnv",
        )
    except Exception:
        pass


register_street_parallel_parking_env()
