"""Conversion and validation of fitted GAP potentials for turboGAP.

A GAP potential built from the ``distance_2b``, ``angle_3b`` and ``soap_turbo``
descriptors can be evaluated by turboGAP (https://github.com/mcaroba/turbogap)
instead of QUIP. turboGAP ships the converter that rewrites QUIP's XML into its
own potential format, so this module wraps that script rather than
reimplementing it, and adds the post-processing the converter leaves out:

* descriptors that ended up with an empty sparse set are removed. QUIP writes
  them out happily and ignores them at evaluation time, but turboGAP divides by
  the sparse-set size while reading the potential and dies with a
  floating-point exception. Species combinations with no representation in the
  training set produce exactly this situation.
* per-species ``e0`` values are read out of the XML. They live in the QUIP
  potential but not in the converted potential file, and without them turboGAP
  silently returns energies offset by the isolated-atom reference.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import ase.io
import numpy as np
from ase.data import atomic_masses, atomic_numbers

__all__ = [
    "check_turbogap_convertible",
    "convert_gap_to_turbogap",
    "descriptor_types",
    "find_make_gap_files_script",
    "run_turbogap_predict",
    "validate_turbogap_potential",
    "write_turbogap_input",
]

#: Name of the converter shipped with turboGAP.
MAKE_GAP_FILES = "make_gap_files.py"

#: Environment variables searched for a turboGAP checkout, in order.
TURBOGAP_ENV_VARS = ("TURBOGAP_PATH", "TURBOGAP_ROOT", "TURBOGAP_DIR")

#: Descriptors turboGAP can evaluate. Notably this excludes QUIP's ``soap`` and
#: ``distance_Nb``, so a potential meant for turboGAP has to be fitted with the
#: ``distance_2b`` / ``angle_3b`` / ``soap_turbo`` terms instead.
TURBOGAP_SUPPORTED_DESCRIPTORS = frozenset({"distance_2b", "angle_3b", "soap_turbo"})


def find_make_gap_files_script(turbogap_path: str | Path | None = None) -> Path:
    """
    Locate turboGAP's ``make_gap_files.py`` converter.

    Parameters
    ----------
    turbogap_path: str | Path | None
        Path to a turboGAP checkout, or directly to ``make_gap_files.py``. When
        None, the environment variables in :data:`TURBOGAP_ENV_VARS` are tried,
        followed by the directory containing the ``turbogap`` executable on PATH.

    Returns
    -------
        Path
            Path to the converter script.

    Raises
    ------
    FileNotFoundError
        If the converter cannot be located.
    """
    candidate_roots: list[Path] = []

    if turbogap_path is not None:
        turbogap_path = Path(turbogap_path)
        if turbogap_path.is_file():
            return turbogap_path
        candidate_roots.append(turbogap_path)

    candidate_roots.extend(
        Path(os.environ[var]) for var in TURBOGAP_ENV_VARS if os.environ.get(var)
    )

    executable = shutil.which("turbogap")
    if executable:
        # the binary normally lives in <root>/bin
        candidate_roots.append(Path(executable).resolve().parent.parent)

    for root in candidate_roots:
        script = root / "tools" / "quip_xml_to_gap" / MAKE_GAP_FILES
        if script.is_file():
            return script
        if (root / MAKE_GAP_FILES).is_file():
            return root / MAKE_GAP_FILES

    raise FileNotFoundError(
        f"Could not locate turboGAP's {MAKE_GAP_FILES}. Pass turbogap_path explicitly, "
        f"or set one of {', '.join(TURBOGAP_ENV_VARS)} to your turboGAP checkout."
    )


def extract_isolated_atom_energies(gap_xml_path: str | Path) -> dict[int, float]:
    """
    Read the per-species ``e0`` values out of a GAP XML file.

    Parameters
    ----------
    gap_xml_path: str | Path
        Path to the fitted GAP XML file.

    Returns
    -------
        dict[int, float]
            Mapping of atomic number to isolated atom energy, for the species
            that carry a non-zero value.
    """
    root = ET.parse(gap_xml_path).getroot()

    energies = {}
    for element in root.iter("e0"):
        value = float(element.attrib["value"])
        if value != 0.0:
            energies[int(element.attrib["Z"])] = value

    return energies


def descriptor_types(gap_xml_path: str | Path) -> list[str]:
    """
    List the descriptor types present in a GAP XML file, in document order.

    Parameters
    ----------
    gap_xml_path: str | Path
        Path to the fitted GAP XML file.

    Returns
    -------
        list[str]
            One descriptor type per fitted term.
    """
    root = ET.parse(gap_xml_path).getroot()

    types = []
    for element in root.iter("descriptor"):
        text = (element.text or "").split()
        if text:
            types.append(text[0])

    return types


def check_turbogap_convertible(gap_xml_path: str | Path) -> list[str]:
    """
    Check that every descriptor in a potential can be evaluated by turboGAP.

    turboGAP's converter raises a bare ``KeyError`` on descriptors it does not
    know, so this turns that into an actionable message before the converter runs.

    Parameters
    ----------
    gap_xml_path: str | Path
        Path to the fitted GAP XML file.

    Returns
    -------
        list[str]
            The descriptor types found in the potential.

    Raises
    ------
    ValueError
        If the potential contains a descriptor turboGAP cannot evaluate.
    """
    types = descriptor_types(gap_xml_path)
    unsupported = sorted(set(types) - TURBOGAP_SUPPORTED_DESCRIPTORS)

    if unsupported:
        raise ValueError(
            f"{Path(gap_xml_path).name} contains descriptor(s) turboGAP cannot "
            f"evaluate: {', '.join(unsupported)}. turboGAP supports "
            f"{', '.join(sorted(TURBOGAP_SUPPORTED_DESCRIPTORS))}; refit with the "
            f"corresponding autoplex GAP terms (distance_2b, angle_3b, soap_turbo) "
            f"enabled and the others disabled."
        )

    return types


def _count_sparse_points(alphas_path: Path) -> int:
    """Count the sparse points recorded in an alphas file."""
    if not alphas_path.is_file():
        return 0

    with open(alphas_path) as file:
        return sum(1 for line in file if line.strip())


def drop_empty_descriptors(potential_file: str | Path) -> list[str]:
    """
    Remove descriptor blocks whose sparse set is empty, in place.

    turboGAP computes the descriptor dimension as ``n_lines / n_sparse`` while
    reading the potential, so a descriptor with no sparse points raises
    SIGFPE. gap_fit emits such descriptors whenever a species combination has no
    instances in the training set, which is common for the ``angle_3b`` triplets.

    Parameters
    ----------
    potential_file: str | Path
        Path to the converted turboGAP potential file. Rewritten in place.

    Returns
    -------
        list[str]
            Descriptor types of the blocks that were dropped.
    """
    potential_file = Path(potential_file)
    # the paths inside the potential file are relative to its parent directory
    root = potential_file.parent.parent

    blocks = re.findall(
        r"gap_beg .*?gap_end\n", potential_file.read_text(), flags=re.DOTALL
    )

    kept, dropped = [], []
    for block in blocks:
        match = re.search(r'alphas_sparse\s*=\s*"(.*?)"', block)
        n_sparse = _count_sparse_points(root / match.group(1)) if match else 0
        if n_sparse > 0:
            kept.append(block)
        else:
            dropped.append(block.split()[1])

    if dropped:
        potential_file.write_text("\n".join(kept))
        logging.info(
            f"Dropped {len(dropped)} descriptor(s) with an empty sparse set from "
            f"{potential_file.name}: {', '.join(sorted(set(dropped)))}. These have no "
            f"training data and would make turboGAP fail while reading the potential."
        )

    return dropped


def write_turbogap_input(
    directory: str | Path,
    potential_file: str | Path,
    species_list: list[str],
    atoms_file: str = "atoms.xyz",
    isolated_atom_energies: dict[int, float] | None = None,
    filename: str = "input",
    extra_keywords: dict | None = None,
) -> Path:
    """
    Write a turboGAP ``input`` file for a converted potential.

    Parameters
    ----------
    directory: str | Path
        Directory to write the input file into. turboGAP must be run from here,
        because the potential file refers to ``gap_files/`` relatively.
    potential_file: str | Path
        Path to the converted potential, relative to ``directory``.
    species_list: list[str]
        Element names, in the order used when the potential was fitted.
    atoms_file: str
        Name of the structure file turboGAP should read.
    isolated_atom_energies: dict[int, float] | None
        Mapping of atomic number to ``e0``. Species without an entry get 0.0.
        Omitting these shifts every predicted energy by the isolated-atom
        reference, so they should normally be taken from the fitted XML via
        :func:`extract_isolated_atom_energies`.
    filename: str
        Name of the input file to write.
    extra_keywords: dict | None
        Additional ``key = value`` pairs appended verbatim.

    Returns
    -------
        Path
            Path to the written input file.
    """
    directory = Path(directory)
    isolated_atom_energies = isolated_atom_energies or {}

    masses = [atomic_masses[atomic_numbers[species]] for species in species_list]
    e0_values = [
        isolated_atom_energies.get(atomic_numbers[species], 0.0)
        for species in species_list
    ]

    lines = [
        f"atoms_file = '{atoms_file}'",
        f"pot_file = '{Path(potential_file).as_posix()}'",
        f"n_species = {len(species_list)}",
        "species = " + " ".join(species_list),
        "masses = " + " ".join(f"{mass:.6f}" for mass in masses),
        "e0 = " + " ".join(f"{energy:.10f}" for energy in e0_values),
    ]
    lines.extend(f"{key} = {value}" for key, value in (extra_keywords or {}).items())

    input_path = directory / filename
    input_path.write_text("\n".join(lines) + "\n")

    return input_path


def convert_gap_to_turbogap(
    gap_xml_path: str | Path,
    species_list: list[str],
    output_dir: str | Path | None = None,
    potential_name: str = "potential.gap",
    turbogap_path: str | Path | None = None,
    remove_empty_descriptors: bool = True,
) -> dict:
    """
    Convert a fitted GAP potential into turboGAP's potential format.

    Parameters
    ----------
    gap_xml_path: str | Path
        Path to the GAP XML file produced by ``gap_fit``. The ``.sparseX``
        files it references must sit alongside it.
    species_list: list[str]
        Element names present in the potential.
    output_dir: str | Path | None
        Directory to write the converted potential into. Defaults to the
        directory holding the XML file.
    potential_name: str
        Name of the turboGAP potential file to produce.
    turbogap_path: str | Path | None
        Path to a turboGAP checkout, forwarded to
        :func:`find_make_gap_files_script`.
    remove_empty_descriptors: bool
        Whether to strip descriptors with an empty sparse set. Leaving them in
        makes turboGAP abort while reading the potential.

    Returns
    -------
        dict
            ``potential_file``, ``gap_files_dir``, ``isolated_atom_energies``
            and ``dropped_descriptors``.

    Raises
    ------
    RuntimeError
        If the converter fails.
    """
    gap_xml_path = Path(gap_xml_path).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else gap_xml_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    check_turbogap_convertible(gap_xml_path)
    script = find_make_gap_files_script(turbogap_path)

    # the converter resolves the XML, its sparseX files and its gap_files/ output
    # directory relative to the working directory, so stage everything together
    if gap_xml_path.parent != output_dir:
        for source in gap_xml_path.parent.glob(f"{gap_xml_path.name}*"):
            shutil.copy(source, output_dir / source.name)

    # it refuses to overwrite an existing potential of the same name
    existing = output_dir / "gap_files" / potential_name
    if existing.is_file():
        existing.unlink()

    result = subprocess.run(
        [sys.executable, str(script), gap_xml_path.name, potential_name],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    potential_file = output_dir / "gap_files" / potential_name
    if result.returncode != 0 or not potential_file.is_file():
        raise RuntimeError(
            f"turboGAP conversion failed for {gap_xml_path.name}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    dropped = drop_empty_descriptors(potential_file) if remove_empty_descriptors else []
    isolated_atom_energies = extract_isolated_atom_energies(gap_xml_path)

    logging.info(f"Converted {gap_xml_path.name} to turboGAP format: {potential_file}")

    return {
        "potential_file": potential_file,
        "gap_files_dir": potential_file.parent,
        "isolated_atom_energies": isolated_atom_energies,
        "dropped_descriptors": dropped,
        "species_list": list(species_list),
    }


def run_turbogap_predict(
    directory: str | Path,
    turbogap_executable: str = "turbogap",
    timeout: int | None = None,
) -> Path:
    """
    Run ``turbogap predict`` in a prepared directory.

    Parameters
    ----------
    directory: str | Path
        Directory holding the turboGAP ``input`` file, the structure file and
        the ``gap_files/`` directory.
    turbogap_executable: str
        Name or path of the turboGAP executable.
    timeout: int | None
        Optional timeout in seconds.

    Returns
    -------
        Path
            Path to the written ``trajectory_out.xyz``.

    Raises
    ------
    RuntimeError
        If turboGAP fails or writes no output.
    """
    directory = Path(directory)

    result = subprocess.run(
        [turbogap_executable, "predict"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )

    trajectory = directory / "trajectory_out.xyz"
    if result.returncode != 0 or not trajectory.is_file():
        raise RuntimeError(
            f"turbogap predict failed (return code {result.returncode}).\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )

    return trajectory


def _read_turbogap_prediction(trajectory: str | Path) -> tuple[float, np.ndarray]:
    """Read the energy and forces from a turboGAP prediction."""
    lines = Path(trajectory).read_text().splitlines()
    n_atoms = int(lines[0].split()[0])

    energy = float(re.search(r"\benergy=(\S+)", lines[1]).group(1))
    # columns are species, x, y, z, then the forces
    forces = np.array(
        [
            [float(value) for value in line.split()[4:7]]
            for line in lines[2 : 2 + n_atoms]
        ]
    )

    return energy, forces


def validate_turbogap_potential(
    gap_xml_path: str | Path,
    structure,
    species_list: list[str],
    workdir: str | Path,
    turbogap_path: str | Path | None = None,
    turbogap_executable: str = "turbogap",
    energy_tolerance: float = 1e-5,
    force_tolerance: float = 1e-4,
) -> dict:
    """
    Check that a converted potential reproduces the QUIP predictions.

    Fits the two evaluations of the same potential against each other, which is
    the only meaningful correctness test of the conversion: any mismatch points
    at the conversion or at a hyperparameter turboGAP cannot represent, not at
    the quality of the fit.

    Parameters
    ----------
    gap_xml_path: str | Path
        Path to the fitted GAP XML file.
    structure: ase.Atoms
        Structure to evaluate with both codes.
    species_list: list[str]
        Element names present in the potential.
    workdir: str | Path
        Directory to run the comparison in.
    turbogap_path: str | Path | None
        Path to a turboGAP checkout.
    turbogap_executable: str
        Name or path of the turboGAP executable.
    energy_tolerance: float
        Allowed absolute energy difference per atom, in eV.
    force_tolerance: float
        Allowed maximum absolute force component difference, in eV/Angstrom.

    Returns
    -------
        dict
            ``energy_difference_per_atom``, ``max_force_difference``,
            ``quip_energy``, ``turbogap_energy`` and ``agrees``.
    """
    import quippy.potential  # noqa: PLC0415

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    conversion = convert_gap_to_turbogap(
        gap_xml_path,
        species_list=species_list,
        output_dir=workdir,
        turbogap_path=turbogap_path,
    )

    atoms = structure.copy()
    ase.io.write(workdir / "atoms.xyz", atoms, format="extxyz")

    write_turbogap_input(
        workdir,
        potential_file=conversion["potential_file"].relative_to(workdir),
        species_list=species_list,
        isolated_atom_energies=conversion["isolated_atom_energies"],
    )

    turbogap_energy, turbogap_forces = _read_turbogap_prediction(
        run_turbogap_predict(workdir, turbogap_executable=turbogap_executable)
    )

    atoms.calc = quippy.potential.Potential(param_filename=str(gap_xml_path))
    quip_energy = atoms.get_potential_energy()
    quip_forces = atoms.get_forces()

    energy_difference = abs(quip_energy - turbogap_energy) / len(atoms)
    force_difference = float(np.abs(quip_forces - turbogap_forces).max())
    agrees = (
        energy_difference <= energy_tolerance and force_difference <= force_tolerance
    )

    log = logging.info if agrees else logging.warning
    log(
        f"turboGAP vs QUIP: dE = {energy_difference:.3e} eV/atom, "
        f"max|dF| = {force_difference:.3e} eV/Angstrom"
    )

    return {
        "energy_difference_per_atom": energy_difference,
        "max_force_difference": force_difference,
        "quip_energy": quip_energy,
        "turbogap_energy": turbogap_energy,
        "agrees": agrees,
        "dropped_descriptors": conversion["dropped_descriptors"],
    }
