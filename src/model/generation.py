"""Batch collection of a balanced, labelled grasp dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import DatasetManager
from .entities import GraspPose
from .simulation import BundledObject, CUBE, SPHERE, GraspSimulator


@dataclass(frozen=True)
class CollectionSummary:
    requested: int
    successful: int
    unsuccessful: int
    attempts: int


class BalancedDatasetGenerator:
    """Generate a balanced dataset through real, independently simulated grasps.

    The two pose bands are intentionally broad enough to include physical
    variation but biased towards a secure, centred pose or a clearly unsuitable
    pose.  The actual PyBullet lift outcome, never the intended band, becomes
    the ground-truth label.
    """

    def __init__(self, simulator: GraspSimulator, seed: int = 42) -> None:
        self._simulator = simulator
        self._rng = np.random.default_rng(seed)

    def collect(self, samples: int = 120) -> tuple[DatasetManager, CollectionSummary]:
        if samples < 2 or samples % 2:
            raise ValueError("samples must be an even number of at least two.")

        target_per_class = samples // 2
        accepted = {True: 0, False: 0}
        attempts = 0
        manager = DatasetManager()
        max_attempts = samples * 4

        while min(accepted.values()) < target_per_class and attempts < max_attempts:
            expected_success = accepted[True] < target_per_class and (
                accepted[False] >= target_per_class or attempts % 2 == 0
            )
            target, gripper = self._configuration(attempts, expected_success)
            pose = self._secure_pose() if expected_success else self._unsuitable_pose()
            result = self._simulator.attempt(pose, target, gripper_kind=gripper)
            attempts += 1
            if accepted[result.success] < target_per_class:
                manager.add(result)
                accepted[result.success] += 1

        if min(accepted.values()) < target_per_class:
            raise RuntimeError(
                f"Could not collect a balanced dataset in {max_attempts} attempts; "
                f"accepted successes={accepted[True]}, failures={accepted[False]}."
            )
        return manager, CollectionSummary(samples, accepted[True], accepted[False], attempts)

    def _secure_pose(self) -> GraspPose:
        return GraspPose(
            x=float(self._rng.uniform(-0.01, 0.01)),
            y=float(self._rng.uniform(-0.01, 0.01)),
            z=float(self._rng.uniform(-0.015, 0.005)),
            roll=float(np.pi + self._rng.normal(0.0, 0.03)),
            pitch=float(self._rng.normal(0.0, 0.03)),
            yaw=float(self._rng.uniform(-0.25, 0.25)),
        )

    def _unsuitable_pose(self) -> GraspPose:
        return GraspPose(
            x=float(self._rng.choice((-1, 1)) * self._rng.uniform(0.10, 0.18)),
            y=float(self._rng.choice((-1, 1)) * self._rng.uniform(0.08, 0.16)),
            z=float(self._rng.uniform(0.08, 0.16)),
            roll=float(np.pi + self._rng.normal(0.0, 0.10)),
            pitch=float(self._rng.normal(0.0, 0.10)),
            yaw=float(self._rng.uniform(-np.pi, np.pi)),
        )

    @staticmethod
    def _configuration(attempt: int, expected_success: bool) -> tuple[BundledObject, str]:
        """Vary both required object and gripper models across collection."""
        if expected_success:
            # These three combinations have passed the reproducible centred
            # grasp check. xArm/sphere remains useful in the failure band.
            return ((CUBE, "panda"), (SPHERE, "panda"), (CUBE, "xarm"))[attempt % 3]
        return ((CUBE, "panda"), (SPHERE, "xarm"), (SPHERE, "panda"))[attempt % 3]
