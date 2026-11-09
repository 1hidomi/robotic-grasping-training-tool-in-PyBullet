"""Controller boundary for authentication UI events."""

from __future__ import annotations

from ..model.auth import AuthService, User


class AuthController:
    def __init__(self, service: AuthService) -> None:
        self._service = service
        self.current_user: User | None = None

    def register_and_login(self, username: str, password: str) -> User:
        self.current_user = self._service.register(username, password)
        return self.current_user

    def login(self, username: str, password: str) -> User:
        self.current_user = self._service.authenticate(username, password)
        return self.current_user

    def logout(self) -> None:
        self.current_user = None
