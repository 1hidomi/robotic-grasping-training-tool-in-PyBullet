"""Generate static visual reports from datasets and training results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .classifier import TrainingResult
from .dataset import LABEL_COLUMN


@dataclass(frozen=True)
class ReportPaths:
    class_balance_chart: Path
    pose_scatter_chart: Path
    confusion_matrix_chart: Path


class ReportGenerator:
    """Create report-ready PNG charts without coupling plotting to the GUI."""

    def generate(self, dataset: pd.DataFrame, training: TrainingResult, output_dir: Path) -> ReportPaths:
        if dataset.empty:
            raise ValueError("Cannot generate a report from an empty dataset.")
        output_dir.mkdir(parents=True, exist_ok=True)
        balance = output_dir / "class_balance.png"
        scatter = output_dir / "pose_distribution.png"
        matrix = output_dir / "confusion_matrix.png"
        self._class_balance(dataset, balance)
        self._pose_scatter(dataset, scatter)
        self._confusion_matrix(training, matrix)
        return ReportPaths(balance, scatter, matrix)

    @staticmethod
    def _class_balance(dataset: pd.DataFrame, destination: Path) -> None:
        counts = dataset[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
        figure, axis = plt.subplots(figsize=(5.5, 3.8))
        bars = axis.bar(["Unsuccessful", "Successful"], counts, color=["#d95f5f", "#3da35d"])
        axis.set_title("Grasp outcome distribution")
        axis.set_ylabel("Number of grasp attempts")
        for bar, count in zip(bars, counts, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    @staticmethod
    def _pose_scatter(dataset: pd.DataFrame, destination: Path) -> None:
        figure, axis = plt.subplots(figsize=(5.5, 4.2))
        for label, colour, name in ((0, "#d95f5f", "Unsuccessful"), (1, "#3da35d", "Successful")):
            subset = dataset[dataset[LABEL_COLUMN] == label]
            axis.scatter(subset["x"], subset["y"], label=name, color=colour, alpha=0.75)
        axis.set_title("Sampled grasp positions")
        axis.set_xlabel("Relative x position (m)")
        axis.set_ylabel("Relative y position (m)")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    @staticmethod
    def _confusion_matrix(training: TrainingResult, destination: Path) -> None:
        matrix = training.confusion_matrix
        figure, axis = plt.subplots(figsize=(4.6, 4.0))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Validation confusion matrix (accuracy {training.validation_accuracy:.1%})")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_xticks([0, 1], ["Failure", "Success"])
        axis.set_yticks([0, 1], ["Failure", "Success"])
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                axis.text(column, row, str(value), ha="center", va="center")
        figure.colorbar(image, ax=axis, label="Count")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
