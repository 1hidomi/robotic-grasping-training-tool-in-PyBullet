"""Regression tests for the dependency-free and ML model components."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.model.auth import AuthService, AuthenticationError
from src.model.repository import UserDataRepository
from src.model.reporting import ReportGenerator
from src.model.classifier import TrainingResult
from src.model.classifier import GraspClassifier
from src.model.dataset import DatasetManager, FEATURE_COLUMNS, LABEL_COLUMN
from src.model.entities import GraspPose, GraspResult
from src.model.sampling import SphericalPoseSampler


class ModelTests(unittest.TestCase):
    def test_report_generator_creates_three_charts(self) -> None:
        with TemporaryDirectory() as directory:
            frame = DatasetManager()
            for index in range(4):
                frame.add(GraspResult(GraspPose(index * 0.01, 0, 0, 3.14, 0, 0), index % 2 == 0,
                                      "panda", "cube", 0.18, 0.2))
            result = TrainingResult(0.75, [[1, 0], [1, 2]], 3, 1)
            paths = ReportGenerator().generate(frame.to_dataframe(), result, Path(directory))
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths.__dict__.values()))
    def test_repository_limits_grasp_records_to_the_owner(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            auth = AuthService(database)
            repository = UserDataRepository(database)
            alice = auth.register("alice", "secure-password")
            bob = auth.register("bob", "another-password")
            result = GraspResult(GraspPose(0, 0, 0, 3.14, 0, 0), True, "panda", "cube", 0.18, 0.2)
            repository.save_grasp(alice, result)
            self.assertEqual(len(repository.list_grasps(alice)), 1)
            self.assertEqual(repository.list_grasps(bob), [])
    def test_authentication_does_not_store_or_accept_plaintext_passwords(self) -> None:
        with TemporaryDirectory() as directory:
            service = AuthService(Path(directory) / "users.db")
            created = service.register("robot_user", "secure-password")
            logged_in = service.authenticate("robot_user", "secure-password")
            self.assertEqual(created, logged_in)
            with self.assertRaises(AuthenticationError):
                service.authenticate("robot_user", "incorrect-password")
    def test_sampling_is_seeded_and_returns_six_features(self) -> None:
        sampler = SphericalPoseSampler(seed=7)
        poses = sampler.sample(3)
        self.assertEqual(len(poses), 3)
        self.assertEqual(len(poses[0].features()), 6)
        self.assertEqual(poses, SphericalPoseSampler(seed=7).sample(3))

    def test_dataset_keeps_label_as_final_column(self) -> None:
        manager = DatasetManager()
        for index in range(4):
            manager.add(
                GraspResult(
                    GraspPose(index * 0.01, 0, 0.05, 3.14, 0, 0),
                    success=index % 2 == 0,
                    object_name="cube",
                    gripper_name="panda",
                    lift_height=0.18,
                    final_object_height=0.1,
                )
            )
        frame = manager.to_dataframe()
        self.assertEqual(frame.columns.tolist(), [*FEATURE_COLUMNS, LABEL_COLUMN])
        self.assertEqual(frame[LABEL_COLUMN].tolist(), [1, 0, 1, 0])

    def test_classifier_can_fit_balanced_synthetic_data(self) -> None:
        manager = DatasetManager()
        for index in range(30):
            success = index % 2 == 0
            manager.add(
                GraspResult(
                    GraspPose(0.01 if success else 0.2, 0, 0.05, 3.14, 0, 0),
                    success=success,
                    object_name="cube",
                    gripper_name="panda",
                    lift_height=0.18,
                    final_object_height=0.1,
                )
            )
        classifier = GraspClassifier()
        result = classifier.train(manager.to_dataframe())
        self.assertGreaterEqual(result.validation_accuracy, 0.8)
        self.assertTrue(classifier.predict(GraspPose(0.01, 0, 0.05, 3.14, 0, 0)))


if __name__ == "__main__":
    unittest.main()
