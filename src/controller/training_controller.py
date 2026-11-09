"""Coordinates user-owned training, persistence, and visual reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..model.auth import User
from ..model.classifier import GraspClassifier, TrainingResult
from ..model.reporting import ReportGenerator, ReportPaths
from ..model.repository import UserDataRepository


class TrainingController:
    def __init__(self, repository: UserDataRepository, reports: ReportGenerator | None = None) -> None:
        self._repository = repository
        self._reports = reports or ReportGenerator()

    def train_for_user(
        self,
        user: User,
        dataset_path: Path = Path("data") / "grasp_dataset.csv",
        output_root: Path = Path("data") / "user_artifacts",
    ) -> tuple[TrainingResult, Path, ReportPaths]:
        dataset = pd.read_csv(dataset_path)
        classifier = GraspClassifier()
        result = classifier.train(dataset)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        user_dir = output_root / f"user_{user.id}" / timestamp
        model_path = classifier.save(user_dir / "grasp_classifier.joblib")
        report_paths = self._reports.generate(dataset, result, user_dir / "report")
        self._repository.save_model(user, str(model_path), result.validation_accuracy)
        return result, model_path, report_paths
