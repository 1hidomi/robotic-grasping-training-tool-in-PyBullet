"""Reproducible grasp-pose sampling strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .entities import GraspPose


@dataclass
class SphericalPoseSampler:
    """Sample poses around an object, aimed approximately at its centre."""

    radius: float = 0.12
    position_noise: float = 0.015
    orientation_noise: float = 0.12
    seed: int | None = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(self, count: int) -> list[GraspPose]:
        if count <= 0:
            raise ValueError("count must be positive")

        poses: list[GraspPose] = []
        for _ in range(count):
            direction = self._rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            # Restrict to approaches above the table plane.
            direction[2] = abs(direction[2])
            direction /= np.linalg.norm(direction)
            position = direction * self.radius
            position += self._rng.normal(0.0, self.position_noise, size=3)

            # Point the hand downwards; yaw variations give a useful baseline.
            roll = math.pi + self._rng.normal(0.0, self.orientation_noise)
            pitch = self._rng.normal(0.0, self.orientation_noise)
            yaw = self._rng.uniform(-math.pi, math.pi)
            poses.append(GraspPose(*position.tolist(), roll, pitch, yaw))
        return poses
