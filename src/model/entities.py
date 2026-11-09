"""Value objects and abstract contracts for the grasp-planning domain.

These classes have no PyBullet or UI dependency.  That keeps the domain model
testable and lets the application replace bundled assets with lecturer-supplied
URDF models without changing controllers or views.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GraspPose:
    """Gripper pose expressed relative to the target object's centre.

    Angles are roll, pitch and yaw in radians.  This six-value representation
    is the default classifier feature vector required by the coursework.
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def features(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


@dataclass(frozen=True)
class GraspResult:
    """Outcome of one simulated grasp attempt."""

    pose: GraspPose
    success: bool
    object_name: str
    gripper_name: str
    lift_height: float
    final_object_height: float
    failure_reason: str | None = None


class Gripper(ABC):
    """Contract for a controllable gripper implementation.

    Concrete grippers contain their asset path and joint-index knowledge; the
    simulator only relies on this common interface, demonstrating polymorphism.
    """

    name: str

    @abstractmethod
    def load(self, client_id: int) -> int:
        """Load the gripper and return its PyBullet body id."""

    @abstractmethod
    def move_to(self, world_position: tuple[float, float, float], orientation: tuple[float, float, float]) -> None:
        """Move to a world pose, given Euler orientation in radians."""

    @abstractmethod
    def open(self) -> None:
        """Open fingers before an approach."""

    @abstractmethod
    def close(self) -> None:
        """Close fingers around the object."""

    @abstractmethod
    def step(self, steps: int) -> None:
        """Advance the physics engine while maintaining motor targets."""


class GraspObject(Protocol):
    """Minimal protocol for a physical target object."""

    name: str

    def load(self, client_id: int) -> int: ...
