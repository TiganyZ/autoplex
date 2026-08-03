from typing import Literal

from pydantic import Field

from autoplex._basemodel import AutoplexBaseModel


class GAPGeneralSettings(AutoplexBaseModel):
    """Model describing general hyperparameters for the GAP fits."""

    at_file: str = Field(
        default="train.extxyz", description="Name of the training file"
    )
    default_sigma: str = Field(
        default="{0.0001 0.05 0.05 0}", description="Default sigma values"
    )
    energy_parameter_name: str = Field(
        default="REF_energy", description="Name of the energy parameter"
    )
    force_parameter_name: str = Field(
        default="REF_forces", description="Name of the force parameter"
    )
    virial_parameter_name: str = Field(
        default="REF_virial", description="Name of the virial parameter"
    )
    sparse_jitter: float = Field(default=1.0e-8, description="Sparse jitter")
    do_copy_at_file: str = Field(default="F", description="Copy the training file to")
    openmp_chunk_size: int = Field(default=10000, description="OpenMP chunk size")
    gp_file: str = Field(default="gap_file.xml", description="Name of the GAP file")
    e0_offset: float = Field(default=0.0, description="E0 offset")
    two_body: bool = Field(
        default=False, description="Whether to include two-body terms"
    )
    three_body: bool = Field(
        default=False, description="Whether to include three-body terms"
    )
    soap: bool = Field(default=True, description="Whether to include SOAP terms")
    distance_2b: bool = Field(
        default=False,
        description="Whether to include turboGAP-style two-body (distance_2b) terms",
    )
    angle_3b: bool = Field(
        default=False,
        description="Whether to include turboGAP-style three-body (angle_3b) terms",
    )
    soap_turbo: bool = Field(
        default=False, description="Whether to include soap_turbo terms"
    )


class TwobSettings(AutoplexBaseModel):
    """Model describing two body hyperparameters for the GAP fits."""

    distance_Nb_order: int = Field(
        default=2,
        description="Distance_Nb order for two-body",
        alias="distance_Nb order",
    )
    f0: float = Field(default=0.0, description="F0 value for two-body")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    cutoff: float | int = Field(default=5.0, description="Radial cutoff distance")
    n_sparse: int = Field(default=15, description="Number of sparse points")
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for two-body"
    )
    delta: float = Field(default=2.00, description="Delta value for two-body")
    theta_uniform: float = Field(
        default=0.5, description="Width of the uniform distribution for theta"
    )
    sparse_method: str = Field(
        default="uniform", description="Sparse method for two-body"
    )
    compact_clusters: str = Field(
        default="T", description="Whether to compact clusters"
    )


class ThreebSettings(AutoplexBaseModel):
    """Model describing threebody hyperparameters for the GAP fits."""

    distance_Nb_order: int = Field(
        default=3,
        description="Distance_Nb order for three-body",
        alias="distance_Nb order",
    )
    f0: float = Field(default=0.0, description="F0 value for three-body")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    cutoff: float | int = Field(default=3.25, description="Radial cutoff distance")
    n_sparse: int = Field(default=100, description="Number of sparse points")
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for three-body"
    )
    delta: float = Field(default=2.00, description="Delta value for three-body")
    theta_uniform: float = Field(
        default=1.0, description="Width of the uniform distribution for theta"
    )
    sparse_method: str = Field(
        default="uniform", description="Sparse method for three-body"
    )
    compact_clusters: str = Field(
        default="T", description="Whether to compact clusters"
    )


class SoapSettings(AutoplexBaseModel):
    """Model describing soap hyperparameters for the GAP fits."""

    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    l_max: int = Field(default=10, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=12, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.5, description="Width of Gaussian smearing")
    zeta: int = Field(default=4, description="Exponent for dot-product SOAP kernel")
    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=1.0, description="Width of the transition region for the cutoff"
    )
    central_weight: float = Field(default=1.0, description="Weight for central atom")
    n_sparse: int = Field(default=6000, description="Number of sparse points")
    delta: float = Field(default=1.00, description="Delta value for SOAP")
    f0: float = Field(default=0.0, description="F0 value for SOAP")
    covariance_type: str = Field(
        default="dot_product", description="Covariance type for SOAP"
    )
    sparse_method: str = Field(
        default="cur_points", description="Sparse method for SOAP"
    )


class Distance2bSettings(AutoplexBaseModel):
    """Model describing the turboGAP-style two-body (distance_2b) hyperparameters.

    Unlike :class:`TwobSettings`, which uses QUIP's ``distance_Nb order=2``
    descriptor, this uses the ``distance_2b`` descriptor that turboGAP
    evaluates natively (see ``get_2b_energy_and_forces`` in turboGAP's
    ``src/gap.f90``).

    Notes
    -----
    turboGAP hardcodes the two-body cutoff buffer to 0.5 Angstrom and provides
    no way to override it from the potential file, so ``cutoff_transition_width``
    must stay at 0.5 for a fitted potential to reproduce QUIP energies after
    conversion.
    """

    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=0.5,
        description="Width of the cutoff transition region. Must be 0.5 for "
        "turboGAP compatibility",
    )
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for distance_2b"
    )
    delta: float = Field(default=2.00, description="Delta value for distance_2b")
    theta_uniform: float = Field(
        default=0.5, description="Width of the uniform distribution for theta"
    )
    n_sparse: int = Field(default=15, description="Number of sparse points")
    sparse_method: str = Field(
        default="uniform", description="Sparse method for distance_2b"
    )
    f0: float = Field(default=0.0, description="F0 value for distance_2b")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )


class Angle3bSettings(AutoplexBaseModel):
    """Model describing the turboGAP-style three-body (angle_3b) hyperparameters.

    This is the descriptor turboGAP evaluates as its three-body term, built from
    the coordinates ``q = [r12 + r13, (r12 - r13)^2, r23]`` (see
    ``get_3b_energy_and_forces`` in turboGAP's ``src/gap.f90``). It is *not*
    equivalent to QUIP's ``distance_Nb order=3`` used by :class:`ThreebSettings`.

    Notes
    -----
    ``sparse_method`` cannot be ``uniform``: that method only supports
    one-dimensional descriptors and ``angle_3b`` is three-dimensional.
    As for :class:`Distance2bSettings`, ``cutoff_transition_width`` must stay at
    0.5 for turboGAP compatibility.
    """

    cutoff: float = Field(default=3.25, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=0.5,
        description="Width of the cutoff transition region. Must be 0.5 for "
        "turboGAP compatibility",
    )
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for angle_3b"
    )
    delta: float = Field(default=0.50, description="Delta value for angle_3b")
    theta_fac: float = Field(
        default=0.5, description="Scaling factor applied to the descriptor spread"
    )
    n_sparse: int = Field(default=100, description="Number of sparse points")
    sparse_method: str = Field(
        default="cur_points",
        description="Sparse method for angle_3b. 'uniform' is not supported",
    )
    f0: float = Field(default=0.0, description="F0 value for angle_3b")


class SoapTurboSettings(AutoplexBaseModel):
    """Model describing soap_turbo hyperparameters for the GAP fits.

    The soap_turbo descriptor is provided by the soap_turbo library
    (https://github.com/libAtoms/soap_turbo) and is available through QUIP/GAP
    and turboGAP.

    The fields marked below as per-species accept either a single value, which is
    broadcast to every species, or a list with one entry per species. autoplex
    expands them into the ``{{...}}`` array syntax that ``gap_fit`` expects, and
    emits one descriptor per central species, because QUIP's ``add_species=T``
    machinery cannot be used with soap_turbo.
    """

    rcut_hard: float = Field(default=5.0, description="Hard radial cutoff distance")
    rcut_soft: float = Field(
        default=4.5, description="Soft radial cutoff distance, where smoothing begins"
    )
    l_max: int = Field(default=8, description="Maximum degree of spherical harmonics")
    alpha_max: int | list[int] = Field(
        default=6,
        description="Number of radial basis functions (per species). Values above 7 "
        "are numerically unstable for the poly3 basis",
    )
    atom_sigma_r: float | list[float] = Field(
        default=0.5,
        description="Width of the atomic Gaussians in the radial direction (per species)",
    )
    atom_sigma_t: float | list[float] = Field(
        default=0.5,
        description="Width of the atomic Gaussians in the angular direction (per species)",
    )
    atom_sigma_r_scaling: float | list[float] = Field(
        default=0.0,
        description="Scaling rate of the radial sigma with neighbour distance (per species)",
    )
    atom_sigma_t_scaling: float | list[float] = Field(
        default=0.0,
        description="Scaling rate of the angular sigma with neighbour distance (per species)",
    )
    amplitude_scaling: float | list[float] = Field(
        default=1.0,
        description="Scaling rate of the amplitude with neighbour distance (per species)",
    )
    central_weight: float | list[float] = Field(
        default=1.0, description="Weight of the central atom (per species)"
    )
    basis: Literal["poly3", "poly3gauss"] = Field(
        default="poly3", description="Radial basis set"
    )
    scaling_mode: str = Field(
        default="polynomial", description="Amplitude scaling mode"
    )
    radial_enhancement: int = Field(
        default=1, description="Radial enhancement of the descriptor"
    )
    nf: float = Field(default=4.0, description="Smoothing parameter of the basis")
    compress_mode: str = Field(
        default="None",
        description="Predefined descriptor compression recipe, or 'None' for no compression",
    )
    zeta: int = Field(default=4, description="Exponent for the dot-product kernel")
    delta: float = Field(default=1.00, description="Delta value for soap_turbo")
    f0: float = Field(default=0.0, description="F0 value for soap_turbo")
    covariance_type: str = Field(
        default="dot_product", description="Covariance type for soap_turbo"
    )
    n_sparse: int = Field(default=1000, description="Number of sparse points")
    sparse_method: str = Field(
        default="cur_points", description="Sparse method for soap_turbo"
    )


class GAPSettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the GAP fits for Phonons."""

    general: GAPGeneralSettings = Field(
        default_factory=GAPGeneralSettings,
        description="General hyperparameters for the GAP fits",
    )
    twob: TwobSettings = Field(
        default_factory=TwobSettings,
        description="Two body hyperparameters for the GAP fits",
    )
    threeb: ThreebSettings = Field(
        default_factory=ThreebSettings,
        description="Three body hyperparameters for the GAP fits",
    )
    soap: SoapSettings = Field(
        default_factory=SoapSettings,
        description="Soap hyperparameters for the GAP fits",
    )
    distance_2b: Distance2bSettings = Field(
        default_factory=Distance2bSettings,
        description="turboGAP-style two body hyperparameters for the GAP fits",
    )
    angle_3b: Angle3bSettings = Field(
        default_factory=Angle3bSettings,
        description="turboGAP-style three body hyperparameters for the GAP fits",
    )
    soap_turbo: SoapTurboSettings = Field(
        default_factory=SoapTurboSettings,
        description="soap_turbo hyperparameters for the GAP fits",
    )
