"""Scikit-learn training and prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .dataset import FEATURE_COLUMNS, LABEL_COLUMN
from .entities import GraspPose


@dataclass(frozen=True)
class TrainingResult:
    validation_accuracy: float
    confusion_matrix: list[list[int]]
    training_samples: int
    validation_samples: int


class GraspClassifier:
    """Train and query a binary grasp-success classifier."""

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=random_state
        )
        self._is_trained = False

    def train(self, dataset: pd.DataFrame) -> TrainingResult:
        if not set([*FEATURE_COLUMNS, LABEL_COLUMN]).issubset(dataset.columns):
            raise ValueError("Dataset is missing one or more required pose/label columns.")
        if dataset[LABEL_COLUMN].nunique() < 2:
            raise ValueError("Training requires both successful and unsuccessful grasps.")

        x_train, x_valid, y_train, y_valid = train_test_split(
            dataset[FEATURE_COLUMNS],
            dataset[LABEL_COLUMN],
            test_size=0.2,
            random_state=self._random_state,
            stratify=dataset[LABEL_COLUMN],
        )
        self._model.fit(x_train, y_train)
        predicted = self._model.predict(x_valid)
        self._is_trained = True
        return TrainingResult(
            validation_accuracy=float(accuracy_score(y_valid, predicted)),
            confusion_matrix=confusion_matrix(y_valid, predicted).tolist(),
            training_samples=len(x_train),
            validation_samples=len(x_valid),
        )

    def predict(self, pose: GraspPose) -> bool:
        if not self._is_trained:
            raise RuntimeError("Train a model before requesting predictions.")
        row = pd.DataFrame([pose.features()], columns=FEATURE_COLUMNS)
        return bool(self._model.predict(row)[0])

    def save(self, destination: Path) -> Path:
        """Persist the fitted classifier with its required feature ordering."""
        if not self._is_trained:
            raise RuntimeError("Train a model before saving it.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "features": FEATURE_COLUMNS}, destination)
        return destination

    @classmethod
    def load(cls, source: Path) -> "GraspClassifier":
        """Restore a saved classifier and validate its feature contract."""
        saved = joblib.load(source)
        if saved.get("features") != FEATURE_COLUMNS:
            raise ValueError("Saved model does not use the expected six pose features.")
        instance = cls()
        instance._model = saved["model"]
        instance._is_trained = True
        return instance
