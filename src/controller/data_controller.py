"""Controller methods for storing and presenting only the current user's data."""

from __future__ import annotations

from ..model.auth import User
from ..model.entities import GraspResult
from ..model.repository import StoredGrasp, StoredModel, UserDataRepository


class DataController:
    def __init__(self, repository: UserDataRepository) -> None:
        self._repository = repository

    def save_result(self, user: User, result: GraspResult) -> int:
        return self._repository.save_grasp(user, result)

    def records_for(self, user: User) -> list[StoredGrasp]:
        return self._repository.list_grasps(user)

    def report_for(self, user: User) -> dict[str, int | float]:
        return self._repository.user_report(user)

    def models_for(self, user: User) -> list[StoredModel]:
        return self._repository.list_models(user)
