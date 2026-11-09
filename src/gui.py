"""GUI entry point for the MVC application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure Qt platform plugins (e.g. windows) are located under conda environments
_qt_plugin_dir = Path(sys.prefix) / "Library" / "lib" / "qt6" / "plugins"
if _qt_plugin_dir.exists():
    os.environ["QT_PLUGIN_PATH"] = str(_qt_plugin_dir)

from PyQt6.QtWidgets import QApplication

from .controller.auth_controller import AuthController
from .model.auth import AuthService
from .model.repository import UserDataRepository
from .view.windows import LoginWindow


def main() -> None:
    application = QApplication(sys.argv)
    login = LoginWindow(AuthController(AuthService()))
    login.show()
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()
