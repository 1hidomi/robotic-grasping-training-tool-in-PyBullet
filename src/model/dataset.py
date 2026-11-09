"""Dataset creation and persistence, owned by the MVC Model layer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .entities import GraspResult


FEATURE_COLUMNS = ["x", "y", "z", "roll", "pitch", "yaw"]
LABEL_COLUMN = "success"


class DatasetManager:
    """Collect results and export the coursework DataFrame format.

    The six pose features remain the first columns and the ground-truth label
    remains the final column, as specified in the coursework brief.  Metadata
    is available in a separate DataFrame, rather than leaking into features.
    """

    def __init__(self) -> None:
        self._results: list[GraspResult] = []

    def add(self, result: GraspResult) -> None:
        self._results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for result in self._results:
            row = dict(zip(FEATURE_COLUMNS, result.pose.features(), strict=True))
            row[LABEL_COLUMN] = int(result.success)
            rows.append(row)
        return pd.DataFrame(rows, columns=[*FEATURE_COLUMNS, LABEL_COLUMN])

    def metadata_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gripper": item.gripper_name,
                    "object": item.object_name,
                    "lift_height": item.lift_height,
                    "final_object_height": item.final_object_height,
                    "failure_reason": item.failure_reason,
                }
                for item in self._results
            ]
        )

    def save(self, destination: Path) -> tuple[Path, Path]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(destination, index=False)
        metadata_path = destination.with_name(f"{destination.stem}_metadata.csv")
        self.metadata_dataframe().to_csv(metadata_path, index=False)
        return destination, metadata_path

    @property
    def size(self) -> int:
        return len(self._results)
