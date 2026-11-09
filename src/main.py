"""Command-line entry point for the first simulation milestone."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .controller.grasp_controller import GraspController
from .model.classifier import GraspClassifier
from .model.evaluation import GraspEvaluator
from .model.generation import BalancedDatasetGenerator
from .model.simulation import GraspSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PyBullet grasp attempt.")
    parser.add_argument("--gui", action="store_true", help="Display the PyBullet visualiser.")
    parser.add_argument("--demo", action="store_true", help="Run a fixed, reproducible grasp attempt.")
    parser.add_argument("--collect", type=int, metavar="N", help="Collect N balanced grasp records into data/.")
    parser.add_argument("--train", action="store_true", help="Train and save a classifier from data/grasp_dataset.csv.")
    parser.add_argument("--evaluate", type=int, metavar="N", help="Evaluate the saved model on N newly simulated grasps.")
    args = parser.parse_args()
    if not args.demo and args.collect is None and not args.train and args.evaluate is None:
        parser.error("Use --demo, --collect N, --train, or --evaluate N.")

    simulator = GraspSimulator(visual=args.gui)
    if args.demo:
        controller = GraspController(simulator)
        # The Panda grasp-target link sits below the fingers.  z=0 therefore
        # places the fingers around (rather than above) the small cube.
        result = controller.run_grasp(0.0, 0.0, 0.0, 3.14159, 0.0, 0.0)
        print(f"success={result.success}; final_object_height={result.final_object_height:.3f} m")
    if args.collect is not None:
        manager, summary = BalancedDatasetGenerator(simulator).collect(args.collect)
        dataset_path, metadata_path = manager.save(Path("data") / "grasp_dataset.csv")
        print(f"saved {summary.requested} balanced records after {summary.attempts} attempts")
        print(dataset_path)
        print(metadata_path)
    if args.train:
        dataset = pd.read_csv(Path("data") / "grasp_dataset.csv")
        classifier = GraspClassifier()
        result = classifier.train(dataset)
        model_path = classifier.save(Path("data") / "grasp_classifier.joblib")
        print(f"validation_accuracy={result.validation_accuracy:.3f}")
        print(f"confusion_matrix={result.confusion_matrix}")
        print(model_path)
    if args.evaluate is not None:
        classifier = GraspClassifier.load(Path("data") / "grasp_classifier.joblib")
        summary = GraspEvaluator(simulator, classifier).evaluate(args.evaluate)
        print(f"prediction_accuracy={summary.prediction_accuracy:.3f}")
        print(f"correct_predictions={summary.correct_predictions}/{summary.attempts}")
        print(f"actual_successes={summary.actual_successes}")


if __name__ == "__main__":
    main()
