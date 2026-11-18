"""Persistent, user-owned records for grasps, trained models, and reports."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .auth import User
from .entities import GraspResult


@dataclass(frozen=True)
class StoredGrasp:
    id: int
    gripper_name: str
    object_name: str
    success: bool
    created_at: str
    max_stable_force: float | None = None
    quality_score: float | None = None
    shake_dropped: bool | None = None


@dataclass(frozen=True)
class StoredModel:
    id: int
    model_path: str
    validation_accuracy: float
    created_at: str


class UserDataRepository:
    """Model-layer persistence service, enforcing ownership in every query."""

    def __init__(self, database_path: Path = Path("data") / "grasp_planner.db") -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def save_grasp(self, user: User, result: GraspResult) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO grasp_records
                (user_id, x, y, z, roll, pitch, yaw, success, gripper_name, object_name,
                 lift_height, final_object_height, failure_reason, max_stable_force,
                 quality_score, shake_dropped, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.id, *result.pose.features(), int(result.success), result.gripper_name,
                    result.object_name, result.lift_height, result.final_object_height,
                    result.failure_reason, result.max_stable_force, result.quality_score,
                    (None if result.shake_dropped is None else int(result.shake_dropped)),
                    self._timestamp(),
                ),
            )
            return int(cursor.lastrowid)

    def list_grasps(self, user: User) -> list[StoredGrasp]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, gripper_name, object_name, success, created_at,
                       max_stable_force, quality_score, shake_dropped
                FROM grasp_records WHERE user_id = ? ORDER BY id DESC
                """,
                (user.id,),
            ).fetchall()
        return [
            StoredGrasp(
                row[0], row[1], row[2], bool(row[3]), row[4],
                row[5], row[6], None if row[7] is None else bool(row[7])
            )
            for row in rows
        ]

    def save_model(self, user: User, path: str, validation_accuracy: float) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO trained_models (user_id, model_path, validation_accuracy, created_at) VALUES (?, ?, ?, ?)",
                (user.id, path, validation_accuracy, self._timestamp()),
            )
            return int(cursor.lastrowid)

    def list_models(self, user: User) -> list[StoredModel]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, model_path, validation_accuracy, created_at
                FROM trained_models WHERE user_id = ? ORDER BY id DESC
                """,
                (user.id,),
            ).fetchall()
        return [StoredModel(row[0], row[1], float(row[2]), row[3]) for row in rows]

    def user_report(self, user: User) -> dict[str, int | float]:
        with self._connect() as connection:
            total, successes = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(success), 0) FROM grasp_records WHERE user_id = ?", (user.id,)
            ).fetchone()
            model_count = connection.execute(
                "SELECT COUNT(*) FROM trained_models WHERE user_id = ?", (user.id,)
            ).fetchone()[0]
        return {
            "grasp_count": int(total),
            "successful_grasp_count": int(successes),
            "model_count": int(model_count),
            "success_rate": float(successes / total) if total else 0.0,
        }

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS grasp_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
                    roll REAL NOT NULL, pitch REAL NOT NULL, yaw REAL NOT NULL,
                    success INTEGER NOT NULL CHECK(success IN (0, 1)),
                    gripper_name TEXT NOT NULL, object_name TEXT NOT NULL,
                    lift_height REAL NOT NULL, final_object_height REAL NOT NULL,
                    failure_reason TEXT, max_stable_force REAL, quality_score REAL,
                    shake_dropped INTEGER, created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            # Add the optional robustness columns to databases created by older versions.
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(grasp_records)")
            }
            for name, sql_type in (
                ("max_stable_force", "REAL"),
                ("quality_score", "REAL"),
                ("shake_dropped", "INTEGER"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE grasp_records ADD COLUMN {name} {sql_type}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trained_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    model_path TEXT NOT NULL, validation_accuracy REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._database_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
