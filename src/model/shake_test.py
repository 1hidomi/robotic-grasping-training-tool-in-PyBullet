"""A small, deterministic robustness test for an already lifted grasp."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ShakeTestResult:
    """Maximum tested force the grasp survived and its normalised score."""
    max_stable_force: float
    quality_score: float
    dropped: bool

class ShakeTest:
    """Apply alternating horizontal forces and detect excessive object motion.
    The test applies force to the object, not to the gripper.  It is a simple
    repeatable disturbance test rather than a full contact-mechanics metric.
    """
    def __init__(
        self,
        force_limit: float = 1.0,
        force_step: float = 0.1,
        steps_per_direction: int = 30,
        drop_distance: float = 0.06,
    ) -> None:
        if force_limit <= 0 or force_step <= 0 or steps_per_direction < 1:
            raise ValueError("Force limits and simulation steps must be positive.")
        self.force_limit = force_limit
        self.force_step = force_step
        self.steps_per_direction = steps_per_direction
        self.drop_distance = drop_distance

    def run(self, physics, client_id: int, object_id: int) -> ShakeTestResult:
        start = physics.getBasePositionAndOrientation(
            object_id, physicsClientId=client_id
        )[0]
        stable_force = 0.0
        force = self.force_step

        while force <= self.force_limit + 1e-9:
            for direction in (1.0, -1.0):
                for _ in range(self.steps_per_direction):
                    position = physics.getBasePositionAndOrientation(
                        object_id, physicsClientId=client_id
                    )[0]
                    physics.applyExternalForce(
                        object_id,
                        -1,
                        (direction * force, 0.0, 0.0),
                        position,
                        physics.WORLD_FRAME,
                        physicsClientId=client_id,
                    )
                    physics.stepSimulation(physicsClientId=client_id)

                current = physics.getBasePositionAndOrientation(
                    object_id, physicsClientId=client_id
                )[0]
                displacement = sum(
                    (current[index] - start[index]) ** 2 for index in range(3)
                ) ** 0.5
                if displacement > self.drop_distance:
                    score = 100.0 * stable_force / self.force_limit
                    return ShakeTestResult(stable_force, score, True)

            stable_force = force
            force = round(force + self.force_step, 10)

        return ShakeTestResult(
            stable_force,
            min(100.0, 100.0 * stable_force / self.force_limit),
            False,
        )
