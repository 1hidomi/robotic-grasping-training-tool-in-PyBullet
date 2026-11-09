"""PyQt6 views. They render state and delegate actions to controllers only."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controller.auth_controller import AuthController
from ..controller.data_controller import DataController
from ..controller.grasp_controller import GraspController
from ..controller.training_controller import TrainingController
from ..model.auth import AuthenticationError
from ..model.simulation import CUBE, SPHERE


class GenerateGraspWindow(QWidget):
    """View for one simulation; all work is delegated to controllers."""

    def __init__(self, auth: AuthController, data: DataController, grasp: GraspController) -> None:
        super().__init__()
        self._auth, self._data, self._grasp = auth, data, grasp
        self.setWindowTitle("Generate grasp record")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.gripper = QComboBox()
        self.gripper.addItems(["panda", "xarm"])
        self.target = QComboBox()
        self.target.addItems(["cube", "sphere"])
        form.addRow("Gripper", self.gripper)
        form.addRow("Object", self.target)
        self.pose_fields: dict[str, QDoubleSpinBox] = {}
        for name, default, low, high in (
            ("x", 0.0, -0.20, 0.20), ("y", 0.0, -0.20, 0.20),
            ("z", 0.0, -0.02, 0.20), ("roll", 3.14159, -3.15, 3.15),
            ("pitch", 0.0, -3.15, 3.15), ("yaw", 0.0, -3.15, 3.15),
        ):
            field = QDoubleSpinBox()
            field.setRange(low, high)
            field.setDecimals(4)
            field.setSingleStep(0.01)
            field.setValue(default)
            self.pose_fields[name] = field
            form.addRow(name, field)
        layout.addLayout(form)
        self.visual_check = QCheckBox("Show 3D simulation window (PyBullet GUI)")
        self.visual_check.setChecked(True)
        layout.addWidget(self.visual_check)
        button = QPushButton("Run simulation and save my record")
        button.clicked.connect(self._run)
        layout.addWidget(button)
        self.status = QLabel("A simulation may take a few seconds.")
        layout.addWidget(self.status)
        self.resize(390, 370)

    def _run(self) -> None:
        user = self._auth.current_user
        if user is None:
            QMessageBox.warning(self, "Not signed in", "Please log in before generating a grasp.")
            return
        try:
            values = {name: field.value() for name, field in self.pose_fields.items()}
            target = CUBE if self.target.currentText() == "cube" else SPHERE
            is_visual = self.visual_check.isChecked()
            result = self._grasp.run_grasp(
                **values,
                target=target,
                gripper_kind=self.gripper.currentText(),
                visual=is_visual,
            )
            record_id = self._data.save_result(user, result)
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Simulation failed", str(error))
            return
        outcome = "successful" if result.success else "unsuccessful"
        self.status.setText(f"Record {record_id} saved: {outcome}; height {result.final_object_height:.3f} m")


class MainMenuWindow(QMainWindow):
    """Post-login menu wired to Controllers, never direct Model/SQL calls."""

    def __init__(
        self, auth: AuthController, data: DataController, grasp: GraspController, training: TrainingController
    ) -> None:
        super().__init__()
        self._auth, self._data, self._grasp, self._training = auth, data, grasp, training
        self._generator: GenerateGraspWindow | None = None
        self.setWindowTitle("Robotic Grasp Planner")
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel(f"Logged in as: {auth.current_user.username if auth.current_user else 'Unknown'}"))
        layout.addWidget(QLabel("Choose an operation:"))
        actions = (
            ("Generate grasp record", self._open_generator),
            ("View my stored records", self._show_records),
            ("Train a machine-learning model", self._train_model),
            ("View my trained models", self._show_models),
            ("View my report", self._show_report),
            ("About / FAQ", self._show_about),
        )
        for text, handler in actions:
            button = QPushButton(text)
            button.clicked.connect(handler)
            layout.addWidget(button)
        self.setCentralWidget(central)
        self.resize(440, 320)

    def _open_generator(self) -> None:
        self._generator = GenerateGraspWindow(self._auth, self._data, self._grasp)
        self._generator.show()

    def _show_records(self) -> None:
        user = self._auth.current_user
        if user is None:
            return
        records = self._data.records_for(user)
        message = "No saved grasp records." if not records else "\n".join(
            f"#{record.id}: {record.gripper_name} / {record.object_name} - "
            f"{'success' if record.success else 'failure'}" for record in records
        )
        QMessageBox.information(self, "My grasp records", message)

    def _show_report(self) -> None:
        user = self._auth.current_user
        if user is None:
            return
        report = self._data.report_for(user)
        QMessageBox.information(
            self, "My report",
            f"Grasp records: {report['grasp_count']}\n"
            f"Successful grasps: {report['successful_grasp_count']}\n"
            f"Success rate: {report['success_rate']:.1%}\n"
            f"Saved models: {report['model_count']}",
        )

    def _train_model(self) -> None:
        user = self._auth.current_user
        if user is None:
            return
        try:
            result, model_path, charts = self._training.train_for_user(user)
        except (FileNotFoundError, ValueError) as error:
            QMessageBox.warning(self, "Training failed", str(error))
            return
        QMessageBox.information(
            self, "Model trained",
            f"Validation accuracy: {result.validation_accuracy:.1%}\n"
            f"Model: {model_path}\n"
            f"Charts: {charts.class_balance_chart.parent}",
        )

    def _show_models(self) -> None:
        user = self._auth.current_user
        if user is None:
            return
        models = self._data.models_for(user)
        message = "No trained models saved." if not models else "\n".join(
            f"#{model.id}: validation accuracy {model.validation_accuracy:.1%}\n{model.model_path}"
            for model in models
        )
        QMessageBox.information(self, "My trained models", message)

    def _show_about(self) -> None:
        QMessageBox.information(self, "About", "PyBullet grasp-planning coursework application.\n"
                                     "Use the simulator to generate records, then train and test a classifier.")


class LoginWindow(QWidget):
    """Authentication view with no database or password-hashing code."""

    def __init__(self, controller: AuthController) -> None:
        super().__init__()
        self._controller = controller
        self._main_menu: MainMenuWindow | None = None
        self.setWindowTitle("Robotic Grasp Planner - Login")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Robotic Grasp Planner"))
        form = QFormLayout()
        self.username = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Username", self.username)
        form.addRow("Password", self.password)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        login_button = QPushButton("Login")
        register_button = QPushButton("Register")
        login_button.clicked.connect(self._login)
        register_button.clicked.connect(self._register)
        buttons.addWidget(login_button)
        buttons.addWidget(register_button)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("New passwords must contain at least 10 characters."))
        self.resize(360, 190)

    def _login(self) -> None:
        try:
            user = self._controller.login(self.username.text().strip(), self.password.text())
        except AuthenticationError as error:
            QMessageBox.warning(self, "Login failed", str(error))
            return
        self._open_main_menu(user.username)

    def _register(self) -> None:
        try:
            user = self._controller.register_and_login(self.username.text().strip(), self.password.text())
        except ValueError as error:
            QMessageBox.warning(self, "Registration failed", str(error))
            return
        self._open_main_menu(user.username)

    def _open_main_menu(self, username: str) -> None:
        from ..model.repository import UserDataRepository
        from ..model.simulation import GraspSimulator
        from ..controller.training_controller import TrainingController

        repository = UserDataRepository()
        self._main_menu = MainMenuWindow(
            self._controller,
            DataController(repository),
            GraspController(GraspSimulator(visual=False)),
            TrainingController(repository),
        )
        self._main_menu.show()
        self.close()
