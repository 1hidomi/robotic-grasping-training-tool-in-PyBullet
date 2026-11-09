"""SQLite-backed authentication with salted password hashes.

The View never sees password hashes and the Controller never issues raw SQL.
This makes authentication a Model-layer responsibility under MVC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import hmac
import os
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class User:
    id: int
    username: str


class AuthenticationError(ValueError):
    """Raised for invalid credentials without disclosing which part failed."""


class AuthService:
    """Create and authenticate local application users."""

    _ITERATIONS = 310_000

    def __init__(self, database_path: Path = Path("data") / "grasp_planner.db") -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def register(self, username: str, password: str) -> User:
        self._validate_new_credentials(username, password)
        salt = os.urandom(16)
        password_hash = self._hash_password(password, salt)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO users (username, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (username, salt, password_hash, self._timestamp()),
                )
                return User(cursor.lastrowid, username)
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already in use.") from exc

    def authenticate(self, username: str, password: str) -> User:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, password_salt, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            raise AuthenticationError("Invalid username or password.")
        supplied_hash = self._hash_password(password, row[2])
        if not hmac.compare_digest(supplied_hash, row[3]):
            raise AuthenticationError("Invalid username or password.")
        return User(row[0], row[1])

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._database_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def _hash_password(cls, password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, cls._ITERATIONS)

    @staticmethod
    def _validate_new_credentials(username: str, password: str) -> None:
        if not 3 <= len(username) <= 40 or not username.replace("_", "").isalnum():
            raise ValueError("Username must be 3-40 letters, numbers, or underscores.")
        if len(password) < 10:
            raise ValueError("Password must contain at least 10 characters.")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
