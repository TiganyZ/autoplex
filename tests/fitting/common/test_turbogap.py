"""Tests for the turboGAP conversion and validation layer.

The tests that need a turboGAP installation are skipped when one is not
available; set TURBOGAP_PATH to a turboGAP checkout and put the turbogap
executable on PATH to run them.
"""
import os
import shutil

import pytest
from ase.io import read

from autoplex.fitting.common.turbogap import (
    check_turbogap_convertible,
    convert_gap_to_turbogap,
    descriptor_types,
    drop_empty_descriptors,
    extract_isolated_atom_energies,
    find_make_gap_files_script,
    validate_turbogap_potential,
    write_turbogap_input,
)


def _has_turbogap() -> bool:
    """Whether both the turboGAP converter and executable are available."""
    if shutil.which("turbogap") is None:
        return False
    try:
        find_make_gap_files_script()
    except FileNotFoundError:
        return False
    return True


requires_turbogap = pytest.mark.skipif(
    not _has_turbogap(),
    reason="requires a turboGAP installation (set TURBOGAP_PATH and put turbogap on PATH)",
)


def test_extract_isolated_atom_energies(test_dir):
    energies = extract_isolated_atom_energies(
        test_dir / "fitting" / "GAP" / "gap_file.xml"
    )

    # only species with a non-zero e0 are reported
    assert energies == pytest.approx({14: -0.84696938})


def test_write_turbogap_input(tmp_path):
    input_path = write_turbogap_input(
        tmp_path,
        potential_file="gap_files/potential.gap",
        species_list=["Li", "Cl"],
        isolated_atom_energies={3: -0.28663766, 17: -0.27567309},
    )

    contents = input_path.read_text()
    assert "pot_file = 'gap_files/potential.gap'" in contents
    assert "n_species = 2" in contents
    assert "species = Li Cl" in contents
    # e0 must be carried over, otherwise every energy is offset by the
    # isolated-atom reference
    assert "e0 = -0.2866376600 -0.2756730900" in contents


def test_write_turbogap_input_defaults_missing_e0_to_zero(tmp_path):
    input_path = write_turbogap_input(
        tmp_path,
        potential_file="gap_files/potential.gap",
        species_list=["Li", "Cl"],
        isolated_atom_energies={3: -0.5},
    )

    assert "e0 = -0.5000000000 0.0000000000" in input_path.read_text()


def _write_potential(directory, blocks):
    """Write a minimal turboGAP potential file plus its alphas files."""
    gap_files = directory / "gap_files"
    gap_files.mkdir(parents=True, exist_ok=True)

    text = ""
    for index, (descriptor_type, n_sparse) in enumerate(blocks, start=1):
        alphas = gap_files / f"alphas_{descriptor_type}_{index}.dat"
        alphas.write_text("".join(f"{i} 1.0\n" for i in range(n_sparse)))
        text += (
            f"gap_beg {descriptor_type}\n"
            f'alphas_sparse = "gap_files/{alphas.name}"\n'
            f"gap_end\n"
        )

    potential = gap_files / "potential.gap"
    potential.write_text(text)
    return potential


def test_drop_empty_descriptors(tmp_path):
    potential = _write_potential(
        tmp_path,
        [("distance_2b", 15), ("angle_3b", 0), ("angle_3b", 20), ("soap_turbo", 50)],
    )

    dropped = drop_empty_descriptors(potential)

    assert dropped == ["angle_3b"]
    contents = potential.read_text()
    assert contents.count("gap_beg") == 3
    assert "distance_2b" in contents and "soap_turbo" in contents


def test_drop_empty_descriptors_keeps_full_potential_untouched(tmp_path):
    potential = _write_potential(tmp_path, [("distance_2b", 15), ("soap_turbo", 50)])
    before = potential.read_text()

    assert drop_empty_descriptors(potential) == []
    assert potential.read_text() == before


def test_find_make_gap_files_script_reports_missing(monkeypatch, tmp_path):
    for variable in ("TURBOGAP_PATH", "TURBOGAP_ROOT", "TURBOGAP_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="make_gap_files.py"):
        find_make_gap_files_script()


def test_find_make_gap_files_script_accepts_direct_path(tmp_path):
    script = tmp_path / "make_gap_files.py"
    script.write_text("")

    assert find_make_gap_files_script(script) == script


def test_descriptor_types(test_dir):
    types = descriptor_types(test_dir / "fitting" / "GAP" / "gap_file.xml")

    assert types == ["distance_Nb", "soap"]


def test_conversion_rejects_descriptors_turbogap_cannot_evaluate(test_dir, tmp_path):
    """QUIP's plain `soap` and `distance_Nb` have no turboGAP counterpart. The
    converter would die with a bare KeyError, so this must be caught first with
    an actionable message."""
    gap_xml = test_dir / "fitting" / "GAP" / "gap_file.xml"

    with pytest.raises(ValueError, match="distance_Nb, soap"):
        check_turbogap_convertible(gap_xml)

    with pytest.raises(ValueError, match="distance_Nb, soap"):
        convert_gap_to_turbogap(gap_xml, species_list=["Si"], output_dir=tmp_path)


@requires_turbogap
def test_turbogap_reproduces_quip_predictions(test_dir, tmp_path):
    """The end-to-end check: a potential fitted with the turboGAP descriptor set
    must give the same energies and forces through turboGAP as through QUIP."""
    from autoplex import MLIP_HYPERS
    from autoplex.fitting.common.utils import gap_fitting

    database_dir = tmp_path / "db"
    database_dir.mkdir()
    for name in ("train.extxyz", "test.extxyz"):
        shutil.copy(test_dir / "fitting" / "ref_files" / name, database_dir / name)

    hyperparameters = MLIP_HYPERS.GAP.model_copy(deep=True)
    hyperparameters.update_parameters(
        {
            "general": {
                "soap": False,
                "distance_2b": True,
                "angle_3b": True,
                "soap_turbo": True,
            },
            "angle_3b": {"n_sparse": 20},
            "soap_turbo": {"n_sparse": 50},
        }
    )

    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    cwd = os.getcwd()
    try:
        os.chdir(fit_dir)
        fit = gap_fitting(
            db_dir=database_dir,
            species_list=["Li", "Cl"],
            hyperparameters=hyperparameters,
            num_processes_fit=2,
            auto_delta=True,
        )
    finally:
        os.chdir(cwd)

    assert fit["train_error"] < 1e-3

    structure = read(database_dir / "train.extxyz", index=":")[2]
    result = validate_turbogap_potential(
        gap_xml_path=fit_dir / "gap_file.xml",
        structure=structure,
        species_list=["Li", "Cl"],
        workdir=tmp_path / "validate",
    )

    assert result["agrees"]
    assert result["energy_difference_per_atom"] < 1e-5
    assert result["max_force_difference"] < 1e-4
    # the LiCl training set has no Li-Li-Li or Cl-Cl-Cl triplets inside the
    # angle_3b cutoff, so those descriptors are empty and must be stripped
    assert set(result["dropped_descriptors"]) == {"angle_3b"}
