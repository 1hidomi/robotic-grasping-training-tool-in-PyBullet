"""Controller for validation and coordination of a grasp attempt."""

from __future__ import annotations

from ..model.entities import GraspPose, GraspResult
from ..model.simulation import BundledObject, GraspSimulator, CUBE


class GraspController:
    """Keeps UI event handlers independent of PyBullet details."""

    def __init__(self, simulator: GraspSimulator) -> None:
        self._simulator = simulator

    def run_grasp(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        target: BundledObject = CUBE,
        gripper_kind: str = "panda",
        visual: bool | None = None,
    ) -> GraspResult:
        if z < -0.02:
            raise ValueError("The grasp pose cannot be below the table surface.")
        if gripper_kind not in {"panda", "xarm"}:
            raise ValueError("Choose either the panda or xarm gripper.")
        return self._simulator.attempt(
            GraspPose(x, y, z, roll, pitch, yaw),
            target,
            gripper_kind=gripper_kind,
            visual=visual,
        )
