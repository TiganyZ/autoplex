import os.path
import pytest
from ase.io import read
from autoplex import MLIP_HYPERS
from autoplex.fitting.common.utils import (
    gap_hyperparameter_constructor,
    angle_3b_hyperparameter_constructor,
    check_convergence,
    check_turbogap_compatibility,
    data_distillation,
    prepare_fit_environment,
    calculate_delta_2b,
    soap_turbo_hyperparameter_constructor,
    stratified_dataset_split,
    compute_num_of_descriptor,
    run_gap,
)


def _descriptor_blocks(gap_argument: str) -> list[str]:
    """Split a ``gap={...}`` argument into its individual descriptor strings."""
    assert gap_argument.startswith("gap={") and gap_argument.endswith("}")
    return gap_argument[len("gap={"):-1].split(" :")

def test_stratified_split(test_dir):
    atoms=read(test_dir / "fitting" / "ref_files" / "vasp_ref.extxyz",':')

    train, test=stratified_dataset_split(atoms,0.33, energy_label="REF_energy")

    assert len(train) == 7
    assert len(test) == 3

def test_gap_hyperparameter_constructor():

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=True,
    )
    # test if string for all possible args true
    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap add_species=T "
        "l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=False,
        include_two_body=True,
    )
    # test if string for include_soap==False
    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T}",
    ]

    assert ref_list == gap_input_list

    # test if returned string is changed if passed in dict is updated
    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter.update_parameters({"twob": {"cutoff": 8},
                                                                 "threeb": {"cutoff": 8.0,
                                                                            "n_sparse": 100},
                                                                 "soap": {"delta": 1.5,
                                                                          "zeta": 2}})
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    gap_input_list_updated = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=True,
    )

    ref_list_exp = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=8 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=8.0 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=2 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list_exp == gap_input_list_updated

    # check disable three_body and two_body

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    # three_body_disable

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=False,
        include_soap=True,
        include_two_body=True,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    # disable two_body

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=False,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    # check with only soap enabled

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=False,
        include_soap=True,
        include_two_body=False,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list


def test_check_convergence():
    check_convergence(0.002)

    assert True


def test_data_distillation(test_dir):
    atoms = data_distillation((test_dir / "fitting" / "ref_files" / "vasp_ref.extxyz"), 35.0, force_label="REF_forces")

    for atom in atoms:
        if (atom.symbols == "Li32Cl32").any() or (atom.symbols == "Li").any() or (atom.symbols == "Cl").any():
            assert True


def test_calculate_delta_2b(test_dir):
    atoms=read(test_dir / "fitting" / "rss_training_dataset"/ "train.extxyz",':')
    
    delta = calculate_delta_2b(atoms_db=atoms, cutoff=4.0, e_name='REF_energy')
    assert (delta-0.7810536) < 1e-6
    
    
def test_calculate_num_of_descriptor(test_dir):
    atoms=read(test_dir / "fitting" / "rss_training_dataset"/ "train.extxyz",':')
    num_2b = sum([compute_num_of_descriptor(atom=at, nb=2, cutoff=4) for at in atoms])
    num_3b = sum([compute_num_of_descriptor(atom=at, nb=3, cutoff=3.25) for at in atoms])
    assert num_2b == 2370
    assert num_3b == 2573


def test_prepare_fit_environment(test_dir, clean_dir):
    prepare = prepare_fit_environment(
        database_dir=(test_dir / "fitting" / "ref_files"),
        mlip_path=(test_dir / "fitting" / "test_mlip_path"),
        glue_xml=False,
    )

    assert os.path.isdir(prepare)


def test_soap_turbo_hyperparameter_constructor_single_species():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True).model_dump(by_alias=True)

    descriptors = soap_turbo_hyperparameter_constructor(
        hypers["soap_turbo"], species_list=["Si"]
    )

    # one descriptor per central species
    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.startswith("soap_turbo ")
    assert "n_species=1" in descriptor
    assert "species_Z={{14}}" in descriptor
    assert "central_index=1" in descriptor
    # autoplex expands the species itself, so gap_fit must not repeat it
    assert "add_species=F" in descriptor
    # per-species hyperparameters are broadcast into array literals
    assert "alpha_max={{6}}" in descriptor
    assert "atom_sigma_r={{0.5}}" in descriptor


def test_soap_turbo_hyperparameter_constructor_multi_species():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True).model_dump(by_alias=True)

    descriptors = soap_turbo_hyperparameter_constructor(
        hypers["soap_turbo"], species_list=["Li", "Cl"]
    )

    assert len(descriptors) == 2
    for index, descriptor in enumerate(descriptors, start=1):
        assert "n_species=2" in descriptor
        assert "species_Z={{3 17}}" in descriptor
        assert f"central_index={index}" in descriptor
        # scalars broadcast to one entry per species
        assert "alpha_max={{6 6}}" in descriptor
        assert "central_weight={{1.0 1.0}}" in descriptor

    # compress_mode=None is QUIP's default and must not be emitted, because
    # turboGAP rejects the literal string after conversion
    assert all("compress_mode" not in descriptor for descriptor in descriptors)


def test_soap_turbo_hyperparameter_constructor_per_species_values():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True)
    hypers.update_parameters({"soap_turbo": {"alpha_max": [6, 8], "atom_sigma_r": [0.3, 0.5]}})

    descriptors = soap_turbo_hyperparameter_constructor(
        hypers.model_dump(by_alias=True)["soap_turbo"], species_list=["Li", "Cl"]
    )

    assert "alpha_max={{6 8}}" in descriptors[0]
    assert "atom_sigma_r={{0.3 0.5}}" in descriptors[0]


def test_soap_turbo_hyperparameter_constructor_rejects_wrong_length():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True)
    hypers.update_parameters({"soap_turbo": {"alpha_max": [6, 8, 10]}})

    with pytest.raises(ValueError, match="alpha_max"):
        soap_turbo_hyperparameter_constructor(
            hypers.model_dump(by_alias=True)["soap_turbo"], species_list=["Li", "Cl"]
        )


def test_angle_3b_hyperparameter_constructor():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True).model_dump(by_alias=True)

    descriptors = angle_3b_hyperparameter_constructor(
        hypers["angle_3b"], species_list=["Li", "Cl"]
    )

    # n_species centres times the n_species*(n_species+1)/2 unordered neighbour pairs
    assert len(descriptors) == 6
    triplets = {
        tuple(
            int(field.split("=")[1])
            for field in descriptor.split()
            if field.startswith(("Z_center=", "Z1=", "Z2="))
        )
        for descriptor in descriptors
    }
    assert triplets == {
        (3, 3, 3), (3, 3, 17), (3, 17, 17),
        (17, 3, 3), (17, 3, 17), (17, 17, 17),
    }

    for descriptor in descriptors:
        assert descriptor.startswith("angle_3b ")
        # turboGAP's converter reads the central species from Z_center, whereas
        # gap_fit's own add_species expansion would write it as Z
        assert "Z_center=" in descriptor
        assert "add_species=F" in descriptor


def test_gap_hyperparameter_constructor_turbogap_descriptors():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True)
    hypers.update_parameters(
        {"general": {"soap": False, "distance_2b": True, "angle_3b": True, "soap_turbo": True}}
    )

    arguments = gap_hyperparameter_constructor(
        gap_parameter_dict=hypers.model_dump(by_alias=True),
        include_distance_2b=True,
        include_angle_3b=True,
        include_soap_turbo=True,
        species_list=["Li", "Cl"],
    )

    blocks = _descriptor_blocks(arguments[-1])
    # one distance_2b (gap_fit expands it per pair), six angle_3b, two soap_turbo
    assert len(blocks) == 9
    assert sum(block.startswith("distance_2b ") for block in blocks) == 1
    assert sum(block.startswith("angle_3b ") for block in blocks) == 6
    assert sum(block.startswith("soap_turbo ") for block in blocks) == 2

    # the descriptor toggles are not gap_fit arguments themselves
    general = " ".join(arguments[:-1])
    for toggle in ("distance_2b=", "angle_3b=", "soap_turbo="):
        assert toggle not in general


def test_gap_hyperparameter_constructor_requires_species_list():
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True).model_dump(by_alias=True)

    with pytest.raises(ValueError, match="species_list"):
        gap_hyperparameter_constructor(
            gap_parameter_dict=hypers, include_soap_turbo=True
        )


def test_gap_hyperparameter_constructor_mixes_with_existing_terms():
    """The turboGAP descriptors compose with autoplex's existing SOAP term."""
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True)
    hypers.update_parameters({"general": {"two_body": True, "soap_turbo": True}})

    arguments = gap_hyperparameter_constructor(
        gap_parameter_dict=hypers.model_dump(by_alias=True),
        include_two_body=True,
        include_soap=True,
        include_soap_turbo=True,
        species_list=["Li", "Cl"],
    )

    blocks = _descriptor_blocks(arguments[-1])
    assert blocks[0].startswith("distance_Nb order=2")
    assert blocks[1].startswith("soap ")
    assert sum(block.startswith("soap_turbo ") for block in blocks) == 2


def test_check_turbogap_compatibility_warns(caplog):
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True)
    hypers.update_parameters(
        {
            "distance_2b": {"cutoff_transition_width": 1.0},
            "angle_3b": {"sparse_method": "uniform"},
            "soap_turbo": {"alpha_max": 12},
        }
    )

    with caplog.at_level("WARNING"):
        check_turbogap_compatibility(hypers.model_dump(by_alias=True))

    messages = " ".join(record.message for record in caplog.records)
    assert "cutoff_transition_width" in messages
    assert "uniform" in messages
    assert "alpha_max" in messages


def test_check_turbogap_compatibility_silent_on_defaults(caplog):
    hypers = MLIP_HYPERS.GAP.model_copy(deep=True).model_dump(by_alias=True)

    with caplog.at_level("WARNING"):
        check_turbogap_compatibility(hypers)

    assert not caplog.records


def test_run_gap_raises_on_abort(monkeypatch, tmp_path):
    """gap_fit's Python entry point exits 0 even when the Fortran core aborts, so
    the abort marker in the log is the only reliable failure signal. Missing it
    would leave a stale potential in place and report its errors as the fit's."""
    monkeypatch.chdir(tmp_path)

    def fake_call(_command, stdout, stderr):
        stdout.write("some output\n")
        stderr.write("SYSTEM ABORT: select_uniform: Descriptor is too large\n")
        return 0

    monkeypatch.setattr("autoplex.fitting.common.utils.subprocess.call", fake_call)

    with pytest.raises(RuntimeError, match="SYSTEM ABORT"):
        run_gap(num_processes_fit=1, parameters=["at_file=train.extxyz"])


def test_run_gap_raises_on_nonzero_return_code(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_call(_command, stdout, stderr):
        stderr.write("command not found\n")
        return 127

    monkeypatch.setattr("autoplex.fitting.common.utils.subprocess.call", fake_call)

    with pytest.raises(RuntimeError, match="returned code 127"):
        run_gap(num_processes_fit=1, parameters=["at_file=train.extxyz"])


def test_run_gap_succeeds_quietly(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_call(_command, stdout, stderr):
        stdout.write("libAtoms::Finalise: Bye-Bye!\n")
        return 0

    monkeypatch.setattr("autoplex.fitting.common.utils.subprocess.call", fake_call)

    assert run_gap(num_processes_fit=1, parameters=["at_file=train.extxyz"]) is None
