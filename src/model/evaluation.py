"""Independent execution-based evaluation for trained grasp classifiers."""

from __future__ import annotations

from dataclasses import dataclass

from .classifier import GraspClassifier
from .generation import BalancedDatasetGenerator
from .simulation import GraspSimulator


@dataclass(frozen=True)
class EvaluationSummary:
    attempts: int
    correct_predictions: int
    prediction_accuracy: float
    actual_successes: int


class GraspEvaluator:
    """Predict unseen poses, execute them, then compare prediction to outcome."""

    def __init__(self, simulator: GraspSimulator, classifier: GraspClassifier, seed: int = 9_001) -> None:
        self._simulator = simulator
        self._classifier = classifier
        self._candidate_source = BalancedDatasetGenerator(simulator, seed=seed)

    def evaluate(self, attempts: int = 10) -> EvaluationSummary:
        if attempts < 1:
            raise ValueError("at least one test attempt is required")
        correct = 0
        actual_successes = 0
        for index in range(attempts):
            expected_success = index % 2 == 0
            target, gripper = self._candidate_source._configuration(index, expected_success)
            pose = (
                self._candidate_source._secure_pose()
                if expected_success
                else self._candidate_source._unsuitable_pose()
            )
            prediction = self._classifier.predict(pose)
            result = self._simulator.attempt(pose, target, gripper_kind=gripper)
            correct += prediction == result.success
            actual_successes += result.success
        return EvaluationSummary(
            attempts=attempts,
            correct_predictions=correct,
            prediction_accuracy=correct / attempts,
            actual_successes=actual_successes,
        )
