"""Fit a turboGAP-compatible GAP potential through the autoplex workflow.

Run with::

    python run_training.py [training.yaml]

The fit goes through ``MLIPFitMaker``, so it follows the same protocol as any
other autoplex GAP fit: the only thing this example changes is which descriptors
are switched on, in gap_hyperparameters.yaml.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from jobflow import run_locally
from monty.serialization import loadfn

from autoplex.fitting.common.flows import MLIPFitMaker
from autoplex.settings import MLIPHypers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(settings_file: str | Path = "training.yaml") -> dict:
    """Fit the potential and report the training and test errors."""
    settings_file = Path(settings_file).resolve()
    root = settings_file.parent
    settings = loadfn(settings_file)

    def resolve(key: str) -> Path:
        path = Path(settings[key])
        return path if path.is_absolute() else root / path

    database_dir = resolve("database_dir")
    output_dir = resolve("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = MLIPHypers.from_file(resolve("hyperparameters_file"))

    maker = MLIPFitMaker(
        mlip_type="GAP",
        # the dataset is already split into train.extxyz and test.extxyz, so the
        # preprocessing step that would build the split is not needed
        apply_data_preprocessing=False,
        num_processes_fit=settings["num_processes_fit"],
        auto_delta=settings["auto_delta"],
    )

    flow = maker.make(
        database_dir=database_dir,
        species_list=settings["species_list"],
        hyperparameters=hyperparameters,
    )

    # gap_fit writes the potential into the working directory
    previous_dir = Path.cwd()
    os.chdir(output_dir)
    try:
        responses = run_locally(flow, create_folders=False, ensure_success=True)
    finally:
        os.chdir(previous_dir)

    output = responses[flow.jobs[-1].uuid][1].output
    train_error = output["train_error"]
    test_error = output["test_error"]

    logging.info(f"Training error : {train_error:.6e} eV/atom")
    logging.info(f"Test error     : {test_error:.6e} eV/atom")
    logging.info(f"Potential      : {output_dir / 'gap_file.xml'}")

    return output


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training.yaml")
