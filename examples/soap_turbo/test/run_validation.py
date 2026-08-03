"""Validate the potential fitted by ../train/run_training.py.

Run with::

    python run_validation.py [validation.yaml]

Two things are checked, and they fail for different reasons:

1. Accuracy of the potential against the reference data, on the held-out test
   set. This says whether the fit is any good.
2. Agreement between QUIP and turboGAP on the *same* fitted potential. Both
   codes evaluate identical parameters, so a disagreement here is a conversion
   problem, not a fit-quality one. This is the check that matters when a
   potential is going to be run in turboGAP.

The second check needs a turboGAP installation and is skipped without one.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import ase.io
import numpy as np
import quippy.potential
from monty.serialization import loadfn

from autoplex.fitting.common.turbogap import (
    convert_gap_to_turbogap,
    find_make_gap_files_script,
    validate_turbogap_potential,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def score_test_set(gap_xml: Path, test_set: Path, ref_energy_name: str) -> float:
    """Return the mean absolute energy error per atom, in eV/atom."""
    frames = ase.io.read(test_set, index=":")
    potential = quippy.potential.Potential(param_filename=str(gap_xml))

    errors = []
    for atoms in frames:
        if ref_energy_name not in atoms.info:
            continue
        atoms.calc = potential
        errors.append(
            abs(atoms.get_potential_energy() - atoms.info[ref_energy_name]) / len(atoms)
        )

    return float(np.mean(errors))


def main(settings_file: str | Path = "validation.yaml") -> dict:
    """Score the potential and check it converts to turboGAP faithfully."""
    settings_file = Path(settings_file).resolve()
    root = settings_file.parent
    settings = loadfn(settings_file)

    def resolve(key: str) -> Path:
        path = Path(settings[key])
        return (path if path.is_absolute() else root / path).resolve()

    gap_xml = resolve("mlip_dir") / settings["gap_xml"]
    if not gap_xml.is_file():
        raise SystemExit(
            f"No potential at {gap_xml}. Run ../train/run_training.py first."
        )

    output_dir = resolve("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. how accurate is the potential?
    test_error = score_test_set(
        gap_xml, resolve("test_set"), settings["ref_energy_name"]
    )
    results["test_set_error"] = test_error
    tolerance = settings["test_set_energy_tolerance"]
    logging.info(f"Test set error : {test_error:.6e} eV/atom (tolerance {tolerance:g})")
    if test_error > tolerance:
        raise SystemExit(f"Test set error {test_error:.3e} eV/atom exceeds {tolerance}")

    # 2. does it survive conversion to turboGAP?
    try:
        find_make_gap_files_script(settings.get("turbogap_path"))
    except FileNotFoundError as error:
        logging.warning(f"Skipping the turboGAP comparison: {error}")
        results["turbogap"] = None
        return results

    conversion = convert_gap_to_turbogap(
        gap_xml,
        species_list=settings["species_list"],
        output_dir=output_dir / "potential",
        turbogap_path=settings.get("turbogap_path"),
    )
    logging.info(f"turboGAP potential: {conversion['potential_file']}")
    if conversion["dropped_descriptors"]:
        # descriptors with no training data behind them; turboGAP cannot read a
        # potential that still contains them
        logging.info(
            f"Dropped {len(conversion['dropped_descriptors'])} empty descriptor(s): "
            f"{', '.join(sorted(set(conversion['dropped_descriptors'])))}"
        )

    comparisons = []
    for index, atoms in enumerate(
        ase.io.read(resolve("validation_structures"), index=":")
    ):
        comparison = validate_turbogap_potential(
            gap_xml_path=gap_xml,
            structure=atoms,
            species_list=settings["species_list"],
            workdir=output_dir / f"structure_{index}",
            turbogap_path=settings.get("turbogap_path"),
            turbogap_executable=settings["turbogap_executable"],
            energy_tolerance=settings["energy_tolerance"],
            force_tolerance=settings["force_tolerance"],
        )
        logging.info(
            f"{atoms.get_chemical_formula()}: "
            f"dE = {comparison['energy_difference_per_atom']:.3e} eV/atom, "
            f"max|dF| = {comparison['max_force_difference']:.3e} eV/Angstrom"
        )
        comparisons.append(comparison)

    results["turbogap"] = comparisons

    disagreeing = [c for c in comparisons if not c["agrees"]]
    if disagreeing:
        raise SystemExit(
            f"{len(disagreeing)} of {len(comparisons)} structures disagree between "
            f"QUIP and turboGAP beyond tolerance."
        )

    logging.info("QUIP and turboGAP agree on every validation structure.")

    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "validation.yaml")
