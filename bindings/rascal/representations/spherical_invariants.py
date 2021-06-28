import json
from itertools import product
import ase

from .base import (
    CalculatorFactory,
    cutoff_function_dict_switch,
    check_optimization_for_spherical_representations,
)

from ..neighbourlist import AtomsList
import numpy as np
from copy import deepcopy
from ..utils import BaseIO


def get_power_spectrum_index_mapping(sp_pairs, n_max, l_max):
    feat_idx2coeff_idx = {}
    # i_feat corresponds the global linear index
    i_feat = 0
    for sp_pair, n1, n2, l in product(
        sp_pairs, range(n_max), range(n_max), range(l_max)
    ):
        feat_idx2coeff_idx[i_feat] = dict(a=sp_pair[0], b=sp_pair[1], n1=n1, n2=n2, l=l)
        i_feat += 1
    return feat_idx2coeff_idx


class SphericalInvariants(BaseIO):

    """
    Computes a SphericalInvariants representation, i.e. the SOAP power spectrum.

    Attributes
    ----------
    interaction_cutoff : float
        Maximum pairwise distance for atoms to be considered in
        expansion

    cutoff_smooth_width : float
        The distance over which the the interaction is smoothed to zero

    max_radial : int
        Number of radial basis functions

    max_angular : int
        Highest angular momentum number (l) in the expansion

    gaussian_sigma_type : str
        How the Gaussian atom sigmas (smearing widths) are allowed to
        vary -- fixed ('Constant'), by species ('PerSpecies'), or by
        distance from the central atom ('Radial').

    gaussian_sigma_constant : float
        Specifies the atomic Gaussian widths, in the case where they're
        fixed.

    cutoff_function_type : string
        Choose the type of smooth cutoff function used to define the local
        environment. Can be either 'ShiftedCosine' or 'RadialScaling'.

        If 'ShiftedCosine', the functional form of the switching function is:

        .. math::

            sc(r) = \\begin{cases}
            1 &r < r_c - sw,\\\\
            0.5 + 0.5 \cos(\pi * (r - r_c + sw) / sw) &r_c - sw < r <= r_c, \\\\
            0 &r_c < r,
            \\end{cases}

        where :math:`r_c` is the interaction_cutoff and :math:`sw` is the
        cutoff_smooth_width.

        If 'RadialScaling', the functional form of the switching function is
        as expressed in equation 21 of https://doi.org/10.1039/c8cp05921g:

        .. math::

            rs(r) = sc(r) u(r),

        where

        .. math::

            u(r) = \\begin{cases}
            \\frac{1}{(r/r_0)^m} &\\text{if c=0,}\\\\
            1 &\\text{if m=0,} \\\\
            \\frac{c}{c+(r/r_0)^m} &\\text{else},
            \\end{cases}

        where :math:`c` is the rate, :math:`r_0` is the scale, :math:`m` is the
        exponent.

    soap_type : string
        Specifies the type of representation to be computed
        ("RadialSpectrum", "PowerSpectrum" and "BiSpectrum").

    inversion_symmetry : boolean
        Specifies whether inversion invariance should be enforced.
        (Only relevant for BiSpectrum.)

    radial_basis :  string
        Specifies the type of radial basis R_n to be computed
        ("GTO" for Gaussian typed orbitals and "DVR" discrete variable
        representation using Gaussian-Legendre quadrature rule)

    normalize : boolean
        Whether to normalize so that the kernel between identical environments
        is 1.  Default and highly recommended: True.

    optimization : dict, default None
        Optional arguments for optimization of the computation of spherical
        expansion coefficients. "Spline" and "RadialDimReduction" are available.

        Spline: Enables cubic splining for the radial basis functions.

            accuracy : float
                accuracy of the cubic spline

        RadialDimReduction: Projection matrices to optimize radial basis,
                            requires Spline to be set

            projection_matrices : dict
                Contains or each species a list of projection matrices for each
                angular channel. A projection matrix for an angular channel has
                the shape (max_radial, expanded_max_radial). A number of
                `expanded_max_radial` radial basis are computed
                to be then projected to `max_radial` radial basis. The projected
                radial basis is then splined for each species and angular channel

        Default settings is using spline

        .. code: python

            dict(Spline=dict(accuracy=1e-8))

    expansion_by_species_method : string
        Specifies the how the species key of the invariant are set-up.
        Possible values: 'environment wise', 'user defined', 'structure wise'.
        The descriptor is computed for each atomic enviroment and it is indexed
        using tuples of atomic species that are present within the environment.
        This index is by definition sparse since a species tuple will be non
        zero only if the atomic species are present inside the environment.
        'environment wise' means that each environmental representation
        will only contain the minimal set of species tuples needed by each
        atomic environment.
        'structure wise' means that within a structure the species tuples
        will be the same for each environment coefficients.
        'user defined' uses global_species to set-up the species tuples.

        These different settings correspond to different trade-off between
        the memory efficiency of the invariants and the computational
        efficiency of the kernel computation.
        When computing a kernel using 'environment wise' setting does not allow
        for efficent matrix matrix multiplications which is ensured when
        'user defined' is used. 'structure wise' is a balance between the
        memory footprint and the use of matrix matrix products.

        Note that the sparsity of the gradient coefficients and their use to
        build kernels does not allow for clear efficiency gains so their
        sparsity is kept irrespective of expansion_by_species_method.

    global_species : list of int
        list of species (specified with their atomic number) to use to set-up
        the species key of the invariant. It should contain all the species
        present in the structure for which invariants will be computed

    compute_gradients : bool
        control the computation of the representation's gradients w.r.t. atomic
        positions.

    cutoff_function_parameters : dict
        Additional parameters for the cutoff function.
        if cutoff_function_type == 'RadialScaling' then it should have the form

        .. code:: python

            dict(rate=...,
                 scale=...,
                 exponent=...)

        where :code:`...` should be replaced by the desired positive float.

    coefficient_subselection : list or None
        if None then all the coefficients are computed following max_radial,
        max_angular and the atomic species present.
        if :code:`soap_type == 'PowerSpectrum'` and it has the form
        :code:`{'a': [...], 'b': [...], 'n1': [...], 'n2': [...], 'l': [...]}`
        where 'a' and 'b' are lists of atomic species, 'n1' and 'n2' are lists
        of radial expension indices and 'l' is the list of spherical expansion
        index. Each of these lists have the same size and their ith element
        refers to one PowerSpectrum coefficient that will be computed.
        :class:`..utils.FPSFilter` and :class:`..utils.CURFilter` with
        `act_on` set to `feature` output such dictionary.

    Methods
    -------
    transform(frames)
        Compute the representation for a list of ase.Atoms object.


    .. [soap] Bartók, Kondor, and Csányi, "On representing chemical
        environments", Phys. Rev. B. 87(18), p. 184115
        http://link.aps.org/doi/10.1103/PhysRevB.87.184115

    """

    def __init__(
        self,
        interaction_cutoff,
        cutoff_smooth_width,
        max_radial,
        max_angular,
        gaussian_sigma_type,
        gaussian_sigma_constant=0.3,
        cutoff_function_type="ShiftedCosine",
        soap_type="PowerSpectrum",
        inversion_symmetry=True,
        radial_basis="GTO",
        normalize=True,
        optimization=None,
        optimization_args=None,
        expansion_by_species_method="environment wise",
        global_species=None,
        compute_gradients=False,
        cutoff_function_parameters=dict(),
        coefficient_subselection=None,
    ):
        """Construct a SphericalExpansion representation

        Required arguments are all the hyperparameters named in the
        class documentation
        """
        self.name = "sphericalinvariants"
        self.hypers = dict()
        if global_species is None:
            global_species = []
        elif not isinstance(global_species, list):
            global_species = list(global_species)

        self.update_hyperparameters(
            max_radial=max_radial,
            max_angular=max_angular,
            soap_type=soap_type,
            normalize=normalize,
            inversion_symmetry=inversion_symmetry,
            expansion_by_species_method=expansion_by_species_method,
            global_species=global_species,
            compute_gradients=compute_gradients,
            coefficient_subselection=coefficient_subselection,
        )

        if self.hypers["coefficient_subselection"] is None:
            del self.hypers["coefficient_subselection"]

        self.cutoff_function_parameters = deepcopy(cutoff_function_parameters)
        cutoff_function_parameters.update(
            interaction_cutoff=interaction_cutoff,
            cutoff_smooth_width=cutoff_smooth_width,
        )
        cutoff_function = cutoff_function_dict_switch(
            cutoff_function_type, **cutoff_function_parameters
        )

        gaussian_density = dict(
            type=gaussian_sigma_type,
            gaussian_sigma=dict(value=gaussian_sigma_constant, unit="AA"),
        )
        optimization = check_optimization_for_spherical_representations(
            optimization, optimization_args
        )

        radial_contribution = dict(type=radial_basis, optimization=optimization)

        self.update_hyperparameters(
            cutoff_function=cutoff_function,
            gaussian_density=gaussian_density,
            radial_contribution=radial_contribution,
        )

        if soap_type == "RadialSpectrum":
            self.update_hyperparameters(max_angular=0)

        self.nl_options = [
            dict(name="centers", args=[]),
            dict(name="neighbourlist", args=dict(cutoff=interaction_cutoff)),
            dict(name="centercontribution", args=dict()),
            dict(name="strict", args=dict(cutoff=interaction_cutoff)),
        ]

        self.rep_options = dict(name=self.name, args=[self.hypers])

        self._representation = CalculatorFactory(self.rep_options)

    def update_hyperparameters(self, **hypers):
        """Store the given dict of hyperparameters

        Also updates the internal json-like representation

        """
        allowed_keys = {
            "interaction_cutoff",
            "cutoff_smooth_width",
            "max_radial",
            "max_angular",
            "gaussian_sigma_type",
            "gaussian_sigma_constant",
            "soap_type",
            "inversion_symmetry",
            "cutoff_function",
            "normalize",
            "gaussian_density",
            "radial_contribution",
            "cutoff_function_parameters",
            "expansion_by_species_method",
            "compute_gradients",
            "global_species",
            "coefficient_subselection",
        }
        hypers_clean = {key: hypers[key] for key in hypers if key in allowed_keys}

        self.hypers.update(hypers_clean)
        return

    def transform(self, frames):
        """Compute the representation.

        Parameters
        ----------
        frames : list(ase.Atoms) or AtomsList
            List of atomic structures.

        Returns
        -------
           AtomsList : Object containing the representation

        """
        if not isinstance(frames, AtomsList):
            frames = AtomsList(frames, self.nl_options)

        self._representation.compute(frames.managers)

        return frames

    def get_num_coefficients(self, n_species=1):
        """Return the number of coefficients in the representation

        (this is the descriptor size per atomic centre)

        """
        if self.hypers["soap_type"] == "RadialSpectrum":
            return n_species * self.hypers["max_radial"]
        if self.hypers["soap_type"] == "PowerSpectrum":
            return (
                int((n_species * (n_species + 1)) / 2)
                * self.hypers["max_radial"] ** 2
                * (self.hypers["max_angular"] + 1)
            )
        if self.hypers["soap_type"] == "BiSpectrum":
            if not self.hypers["inversion_symmetry"]:
                return (
                    n_species ** 3
                    * self.hypers["max_radial"] ** 3
                    * int(
                        1
                        + 2 * self.hypers["max_angular"]
                        + 3 * self.hypers["max_angular"] ** 2 / 2
                        + self.hypers["max_angular"] ** 3 / 2
                    )
                )
            else:
                return (
                    n_species ** 3
                    * self.hypers["max_radial"] ** 3
                    * int(
                        np.floor(
                            ((self.hypers["max_angular"] + 1) ** 2 + 1)
                            * (2 * (self.hypers["max_angular"] + 1) + 3)
                            / 8.0
                        )
                    )
                )
        else:
            raise ValueError(
                "Only soap_type = RadialSpectrum || "
                "PowerSpectrum || BiSpectrum "
                "implemented for now"
            )

    def get_keys(self, species):
        """
        return the proper list of keys used to build the representation
        """
        keys = []
        if self.hypers["soap_type"] == "RadialSpectrum":
            for sp in species:
                keys.append([sp])
        elif self.hypers["soap_type"] == "PowerSpectrum":
            for sp1 in species:
                for sp2 in species:
                    if sp1 > sp2:
                        continue
                    keys.append([sp1, sp2])
        elif self.hypers["soap_type"] == "BiSpectrum":
            for sp1 in species:
                for sp2 in species:
                    if sp1 > sp2:
                        continue
                    for sp3 in species:
                        if sp2 > sp3:
                            continue
                        keys.append([sp1, sp2, sp3])
        else:
            raise ValueError(
                "Only soap_type = RadialSpectrum || "
                "PowerSpectrum || BiSpectrum "
                "implemented for now"
            )
        return keys

    def get_feature_index_mapping(self, managers):

        species = []
        for ii in range(len(managers)):
            manager = managers[ii]
            if isinstance(manager, ase.Atoms):
                species.extend(manager.get_atomic_numbers())
            elif isinstance(managers, AtomsList):
                for center in manager:
                    sp = center.atom_type
                    species.append(sp)
        u_species = np.unique(species)
        sp_pairs = self.get_keys(u_species)

        n_max = self.hypers["max_radial"]
        l_max = self.hypers["max_angular"] + 1
        if self.hypers["soap_type"] == "PowerSpectrum":
            feature_index_mapping = get_power_spectrum_index_mapping(
                sp_pairs, n_max, l_max
            )
        else:
            raise NotImplementedError(
                "Only soap_type = " "PowerSpectrum " "implemented for now"
            )

        return feature_index_mapping

    def _get_init_params(self):
        gaussian_density = self.hypers["gaussian_density"]
        cutoff_function = self.hypers["cutoff_function"]
        radial_contribution = self.hypers["radial_contribution"]

        init_params = dict(
            interaction_cutoff=cutoff_function["cutoff"]["value"],
            cutoff_smooth_width=cutoff_function["smooth_width"]["value"],
            max_radial=self.hypers["max_radial"],
            max_angular=self.hypers["max_angular"],
            soap_type=self.hypers["soap_type"],
            inversion_symmetry=self.hypers["inversion_symmetry"],
            normalize=self.hypers["normalize"],
            expansion_by_species_method=self.hypers["expansion_by_species_method"],
            global_species=self.hypers["global_species"],
            compute_gradients=self.hypers["compute_gradients"],
            gaussian_sigma_type=gaussian_density["type"],
            gaussian_sigma_constant=gaussian_density["gaussian_sigma"]["value"],
            cutoff_function_type=cutoff_function["type"],
            radial_basis=radial_contribution["type"],
            optimization=radial_contribution["optimization"],
            cutoff_function_parameters=self.cutoff_function_parameters,
        )
        if "coefficient_subselection" in self.hypers:
            init_params["coefficient_subselection"] = self.hypers[
                "coefficient_subselection"
            ]
        return init_params

    #def _set_data(self, data):
    #    super()._set_data(data)

    #def _get_data(self):
    #    return super()._get_data()

    def _set_data(self, data):
        super()._set_data(data)
        self._representation = self._representation.from_dict(data["representation"])

    def _get_data(self):
        data = super()._get_data()
        data.update(representation=self._representation.to_dict())
        return data

