# Robotic Grasp Planner

Coursework starter implementation for COMP0213 Object-Oriented Programming for Robotics and AI.

It is organised as Model-View-Controller (MVC):

- `src/model`: simulation, grasp sampling, dataset and machine-learning logic.
- `src/controller`: input validation and coordination between the model and UI.
- `src/view`: PyQt6 user-interface windows.

## Setup

PyBullet does not currently provide a usable `pip` wheel for this computer's
default Python installation.  Use Conda Forge's binary package instead:

```powershell
D:\Miniconda\Scripts\conda.exe env create --prefix .\.pybullet-env --file environment.yml
.\.pybullet-env\python.exe -m src.main --demo
```

`--demo` runs one headless grasp attempt. Add `--gui` to watch it in the
PyBullet visualiser. `requirements.txt` remains useful for the non-simulation
data and ML tests in a regular virtual environment.

## Collect and train

After the smoke test, collect the coursework-sized balanced dataset. This
performs every grasp in PyBullet and can take several minutes:

```powershell
.\.pybullet-env\python.exe -m src.main --collect 120
.\.pybullet-env\python.exe -m src.main --train
.\.pybullet-env\python.exe -m src.main --evaluate 10
```

The collection uses two grippers (Franka Panda and xArm6) and two objects
(cube and sphere). The feature CSV has exactly six pose columns followed by
the final `success` label column; extra asset and simulation information is
kept in the accompanying metadata CSV.

## GUI

```powershell
.\.pybullet-env\python.exe -m src.gui
```

The GUI provides registration/login, per-user record browsing, report
statistics, and the single-grasp simulation form (which requires PyBullet).

## Current milestone

The first milestone is a reproducible single-grasp simulation.  It uses only
assets bundled with `pybullet_data`, so no starter code or external URDF files
are required.  The abstract `Gripper` and `GraspObject` contracts deliberately
leave room to substitute teacher's assets later.

## GitHub Synchronization

To push latest local modifications to GitHub at any time:

```powershell
.\sync.ps1 -Message "Your commit message"
```
