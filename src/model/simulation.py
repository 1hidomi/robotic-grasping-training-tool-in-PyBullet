"""PyBullet implementation of one approach-close-lift grasp attempt.

The implementation relies only on assets bundled by ``pybullet_data``:
Franka Panda and standard small cube/sphere URDFs.  It is intentionally kept
behind the model layer so a different lecturer-provided gripper can be added
without touching the controller or GUI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .entities import GraspPose, GraspResult


@dataclass(frozen=True)
class BundledObject:
    """A dynamic target object shipped with PyBullet."""

    name: str
    urdf: str
    half_height: float


CUBE = BundledObject("small_cube", "cube_small.urdf", 0.025)
SPHERE = BundledObject("small_sphere", "sphere_small.urdf", 0.03)


class PandaGripper:
    """Franka Panda arm and two-finger gripper adapter.

    Although its URDF includes the arm, this class presents the narrow
    `Gripper` behaviour needed by `GraspSimulator`.  It makes the model
    replaceable with another gripper adapter later.
    """

    name = "franka_panda"
    _ARM_JOINTS = range(7)
    _FINGER_JOINTS = (9, 10)
    _END_EFFECTOR_LINK = 11

    def __init__(self, physics, client_id: int, visual: bool = False, time_step: float = 1 / 240) -> None:
        self._p = physics
        self._client_id = client_id
        self._visual = visual
        self._time_step = time_step
        self._body_id: int | None = None

    def load(self) -> int:
        self._body_id = self._p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=(0.0, 0.0, 0.0),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        # A reproducible, collision-free start configuration.
        for joint, value in enumerate((0.0, -0.65, 0.0, -2.15, 0.0, 1.55, 0.75)):
            self._p.resetJointState(self._body_id, joint, value, physicsClientId=self._client_id)
        self.open()
        return self._body_id

    def move_to(self, world_position: tuple[float, float, float], rpy: tuple[float, float, float]) -> None:
        self._require_loaded()
        orientation = self._p.getQuaternionFromEuler(rpy, physicsClientId=self._client_id)
        joint_targets = self._p.calculateInverseKinematics(
            self._body_id,
            self._END_EFFECTOR_LINK,
            world_position,
            orientation,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self._client_id,
        )
        for joint in self._ARM_JOINTS:
            self._p.setJointMotorControl2(
                self._body_id,
                joint,
                self._p.POSITION_CONTROL,
                targetPosition=joint_targets[joint],
                force=120.0,
                physicsClientId=self._client_id,
            )

    def open(self) -> None:
        self._set_fingers(0.04)

    def close(self) -> None:
        self._set_fingers(0.0)

    def step(self, steps: int) -> None:
        for _ in range(steps):
            self._p.stepSimulation(physicsClientId=self._client_id)
            if self._visual:
                time.sleep(self._time_step)

    def _set_fingers(self, opening: float) -> None:
        self._require_loaded(allow_initialising=True)
        for joint in self._FINGER_JOINTS:
            self._p.setJointMotorControl2(
                self._body_id,
                joint,
                self._p.POSITION_CONTROL,
                targetPosition=opening,
                force=70.0,
                physicsClientId=self._client_id,
            )

    def _require_loaded(self, allow_initialising: bool = False) -> None:
        if self._body_id is None:
            raise RuntimeError("Load the Panda gripper before controlling it.")


class XArmGripper:
    """xArm6 with its built-in two-finger gripper.

    This is a distinct robot/gripper URDF supplied in ``pybullet_data``.  The
    gripper's hand link is above its fingertips, so ``move_to`` converts the
    common grasp-point contract into the arm's end-effector target.
    """

    name = "xarm6_two_finger"
    _ARM_JOINTS = range(1, 7)
    _FINGER_JOINTS = range(8, 14)
    _END_EFFECTOR_LINK = 7
    _HAND_ABOVE_FINGERS = 0.12

    def __init__(self, physics, client_id: int, visual: bool = False, time_step: float = 1 / 240) -> None:
        self._p = physics
        self._client_id = client_id
        self._visual = visual
        self._time_step = time_step
        self._body_id: int | None = None

    def load(self) -> int:
        self._body_id = self._p.loadURDF(
            "xarm/xarm6_with_gripper.urdf",
            basePosition=(0.0, 0.0, 0.0),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        self.open()
        return self._body_id

    def move_to(self, world_position: tuple[float, float, float], rpy: tuple[float, float, float]) -> None:
        self._require_loaded()
        hand_position = (
            world_position[0],
            world_position[1],
            world_position[2] + self._HAND_ABOVE_FINGERS,
        )
        orientation = self._p.getQuaternionFromEuler(rpy, physicsClientId=self._client_id)
        joint_targets = self._p.calculateInverseKinematics(
            self._body_id,
            self._END_EFFECTOR_LINK,
            hand_position,
            orientation,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self._client_id,
        )
        # PyBullet may append fixed/mimic-joint values to this solution; the
        # six leading values correspond to xArm joints 1-6.
        for target, joint in zip(joint_targets[:6], self._ARM_JOINTS, strict=True):
            self._p.setJointMotorControl2(
                self._body_id, joint, self._p.POSITION_CONTROL,
                targetPosition=target, force=120.0, physicsClientId=self._client_id,
            )

    def open(self) -> None:
        self._set_fingers(0.0)

    def close(self) -> None:
        self._set_fingers(0.8)

    def step(self, steps: int) -> None:
        for _ in range(steps):
            self._p.stepSimulation(physicsClientId=self._client_id)
            if self._visual:
                time.sleep(self._time_step)

    def _set_fingers(self, closing_angle: float) -> None:
        self._require_loaded()
        for joint in self._FINGER_JOINTS:
            self._p.setJointMotorControl2(
                self._body_id, joint, self._p.POSITION_CONTROL,
                targetPosition=closing_angle, force=100.0, physicsClientId=self._client_id,
            )

    def _require_loaded(self) -> None:
        if self._body_id is None:
            raise RuntimeError("Load the xArm gripper before controlling it.")


class GraspSimulator:
    """Coordinates PyBullet state and applies the coursework grasp procedure."""

    def __init__(self, visual: bool = False, time_step: float = 1 / 240) -> None:
        self.visual = visual
        self.time_step = time_step

    def attempt(
        self,
        pose: GraspPose,
        target: BundledObject = CUBE,
        gripper_kind: Literal["panda", "xarm"] = "panda",
        hold_seconds: float = 3.1,
        visual: bool | None = None,
    ) -> GraspResult:
        """Execute open -> approach -> close -> lift -> hold and label it."""
        is_visual = self.visual if visual is None else visual
        p, pybullet_data = self._import_pybullet()
        mode = p.GUI if is_visual else p.DIRECT
        client_id = p.connect(mode)
        try:
            p.resetSimulation(physicsClientId=client_id)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
            p.setGravity(0, 0, -9.81, physicsClientId=client_id)
            p.setTimeStep(self.time_step, physicsClientId=client_id)
            if is_visual:
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.1,
                    cameraYaw=45.0,
                    cameraPitch=-25.0,
                    cameraTargetPosition=[0.45, 0.0, 0.15],
                    physicsClientId=client_id,
                )
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)

            p.loadURDF("plane.urdf", physicsClientId=client_id)

            object_position = (0.52, 0.0, target.half_height)
            object_id = p.loadURDF(target.urdf, object_position, physicsClientId=client_id)
            p.changeDynamics(object_id, -1, lateralFriction=1.2, physicsClientId=client_id)
            gripper = (
                PandaGripper(p, client_id, visual=is_visual, time_step=self.time_step)
                if gripper_kind == "panda"
                else XArmGripper(p, client_id, visual=is_visual, time_step=self.time_step)
            )
            gripper.load()

            final_position = tuple(object_position[i] + pose.features()[i] for i in range(3))
            approach_position = (final_position[0], final_position[1], final_position[2] + 0.12)
            rpy = (pose.roll, pose.pitch, pose.yaw)
            gripper.open()
            gripper.move_to(approach_position, rpy)
            gripper.step(300)
            gripper.move_to(final_position, rpy)
            gripper.step(240)
            gripper.close()
            gripper.step(300)

            lift_position = (final_position[0], final_position[1], final_position[2] + 0.18)
            gripper.move_to(lift_position, rpy)
            gripper.step(480)
            gripper.step(max(1, int(hold_seconds / self.time_step)))
            final_height = p.getBasePositionAndOrientation(object_id, physicsClientId=client_id)[0][2]
            success = final_height > target.half_height + 0.08
            if is_visual:
                # Keep the scene visible briefly so the user can see the lifted object
                time.sleep(2.0)
            return GraspResult(
                pose=pose,
                success=success,
                object_name=target.name,
                gripper_name=gripper.name,
                lift_height=0.18,
                final_object_height=final_height,
                failure_reason=None if success else "object_not_held_after_lift",
            )
        finally:
            p.disconnect(physicsClientId=client_id)

    @staticmethod
    def _import_pybullet():
        try:
            import pybullet as p
            import pybullet_data
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyBullet is not installed for this Python interpreter. "
                "Create the Conda environment from environment.yml, then retry."
            ) from exc
        return p, pybullet_data
