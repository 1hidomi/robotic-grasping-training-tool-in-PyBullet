# Robotic Grasp Planner

A Python coursework project for **COMP0213 Object-Oriented Programming for Robotics and AI**. It simulates robotic grasping in PyBullet, collects grasp data, trains a classifier to predict grasp success, and provides a PyQt6 GUI for running and reviewing grasp experiments.

## Features

- **Two robot grippers:** Franka Panda and xArm6.
- **Inverse kinematics:** move the robot end-effector to a target pose.
- **Grasp simulation:** test grasp poses against cube and sphere objects in PyBullet.
- **Dataset generation:** collect labelled grasp examples using six pose features and a binary `success` label.
- **Machine learning:** train and evaluate a grasp-success classifier.
- **Shake test:** apply alternating horizontal forces after a successful lift to test grasp robustness.
- **Grip quality score:** report a simple 0–100 score based on the maximum tested stable force. This is an additional robustness measure; it does not replace the binary classifier label.
- **GUI and data storage:** PyQt6 interface with user registration/login, grasp simulation, saved records and report statistics. SQLite is used for local record storage.

## OOP and application structure

The application follows the **Model-View-Controller (MVC)** pattern:

- `src/model/`: simulation, robot/gripper implementations, grasp sampling, dataset generation, classification, evaluation, reporting and persistence.
- `src/controller/`: coordinates application logic between the model and the user interface.
- `src/view/`: PyQt6 windows.
- `src/main.py`: command-line entry point for simulation, data collection, training and evaluation.
- `src/gui.py`: GUI entry point.

The abstract `Gripper` class defines a common interface for robot grippers. The shared PyBullet gripper implementation contains common movement and finger-control logic, while `PandaGripper` and `XArmGripper` provide robot-specific details. This demonstrates abstraction, inheritance, encapsulation and polymorphism.

## Requirements and installation

The recommended environment is the Conda environment defined in `environment.yml`. It includes the dependencies used by the simulation, GUI, machine-learning code and tests, including PyBullet, PyQt6, NumPy, pandas, scikit-learn and matplotlib.

From the project root in PowerShell, create the environment:

```powershell
conda env create --prefix .\.pybullet-env --file environment.yml
```

If Conda is not available as a command, run it using your local Conda installation path. For example:

```powershell
D:\Miniconda\Scripts\conda.exe env create --prefix .\.pybullet-env --file environment.yml
```

All commands below use the same project environment:

```powershell
.\.pybullet-env\python.exe
```

You do not need to activate the environment when using this explicit Python executable.

## Quick start: simulation demo

Run one headless grasp attempt:

```powershell
.\.pybullet-env\python.exe -m src.main --demo
```

To open the PyBullet visualiser during the demo:

```powershell
.\.pybullet-env\python.exe -m src.main --demo --gui
```

## Dataset collection, training and evaluation

The repository includes a sample dataset and a saved classifier under `data/`. To train the classifier using the existing dataset:

```powershell
.\.pybullet-env\python.exe -m src.main --train
```

To generate a new balanced dataset of 120 grasp attempts first:

```powershell
.\.pybullet-env\python.exe -m src.main --collect 120
```

Then train and evaluate the model:

```powershell
.\.pybullet-env\python.exe -m src.main --train
.\.pybullet-env\python.exe -m src.main --evaluate 10
```

Data collection runs physics simulations and may take several minutes. The generated feature CSV contains six grasp-pose features and the final binary `success` label. Additional simulation information is stored in the metadata CSV.

## Launch the GUI

Run the PyQt6 application using the same Conda environment:

```powershell
.\.pybullet-env\python.exe -m src.gui
```

The GUI provides user registration/login, a grasp simulation form, saved grasp records and report statistics. In the simulation form, the shake test can optionally be enabled. It runs after a successful lift, applies alternating horizontal forces to the object and records the maximum tested stable force and the corresponding 0–100 robustness score.

## Run tests

Run the unit tests from the project root:

```powershell
.\.pybullet-env\python.exe -m unittest discover -s tests
```

## Project layout

```text
robotic-grasping-training-tool-in-PyBullet/
├── data/                 # Sample dataset, trained model and local database
├── src/
│   ├── controller/       # Application controllers
│   ├── model/            # Grasping, simulation and ML logic
│   ├── view/             # PyQt6 windows
│   ├── gui.py            # GUI entry point
│   └── main.py           # Command-line entry point
├── tests/                # Unit tests
├── environment.yml       # Conda environment specification
├── requirements.txt      # Python package requirements
└── sync.ps1              # PowerShell helper to commit and push changes
```

## Git workflow

To download the latest changes from GitHub:

```powershell
git pull origin main
```

To upload your local changes, you can use the included PowerShell helper:

```powershell
.\sync.ps1 -Message "Describe your changes"
```

The script stages all non-ignored changes, creates a commit if there are changes, and pushes the current local `main` branch to `origin/main`. Review `git status` before running it.

## Notes

- Run commands from the repository root directory.
- The Conda environment folder (`.pybullet-env/`) is local and is not uploaded to GitHub.
- Do not commit passwords, API keys, personal data or other secrets.
