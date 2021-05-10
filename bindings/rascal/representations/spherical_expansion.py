from collections.abc import Iterable
import json
import logging

from .base import CalculatorFactory, cutoff_function_dict_switch
from ..neighbourlist import AtomsList
import numpy as np
from ..utils import BaseIO
from copy import deepcopy

LOGGER = logging.getLogger(__name__)


def _parse_species_list(species_list):
    """Parse the species_list keyword for initialization

    Return a hypers dictionary that can be used to initialize the
    underlying C++ object, with expansion_by_species_method and
    global_species keys set appropriately
    """
    if isinstance(species_list, str):
        if (species_list == "environment wise") or (
            species_list == "structure wise"
        ):
            expansion_by_species_method = species_list
            global_species = []
        else:
            raise ValueError(
                "'species_list' must be one of: {'environment wise', "
                "'structure_wise'}, or a list of int"
            )
    else:
        if isinstance(species_list, Iterable):
            expansion_by_species_method = "user defined"
            global_species = list(species_list)
        else:
            raise ValueError(
                "'species_list' must be a list of int, if user-defined"
            )
    return {
        "expansion_by_species_method": expansion_by_species_method,
        "global_species": global_species,
    }


class SphericalExpansion(BaseIO):
    """
    Computes the spherical expansion of the neighbour density [soap]

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

    gaussian_sigma_type : str, default="Constant"
        How the Gaussian atom sigmas (smearing widths) are allowed to
        vary -- fixed ('Constant'), by species ('PerSpecies'), or by
        distance from the central atom ('Radial').

    gaussian_sigma_constant : float, default=0.3
        Specifies the atomic Gaussian widths when
        gaussian_sigma_type=="Constant"

    cutoff_function_type : string, default="ShiftedCosine"
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

    radial_basis :  string, default="GTO"
        Specifies the type of radial basis R_n to be computed
        ("GTO" for Gaussian typed orbitals and "DVR" discrete variable
        representation using Gaussian quadrature rule)

    optimization_args : dict, optional
        Additional arguments for optimization.
        Currently spline optimization for the radial basis function is available
        Recommended settings if using: {"type":"Spline", "accuracy": 1e-5}

    species_list : string or list(int), default="environment wise"
        Specifies the how the species key of the invariant are set-up.
        Possible values: 'environment wise', 'structure wise', or a user-defined
        list of species.
        The descriptor is computed for each atomic enviroment and it is indexed
        using tuples of atomic species that are present within the environment.
        This index is by definition sparse since a species tuple will be non
        zero only if the atomic species are present inside the environment.
        'environment wise' means that each environmental representation
        will only contain the minimal set of species tuples needed by each
        atomic environment.
        'structure wise' means that within a structure the species tuples
        will be the same for each environment coefficients.

        The user-defined option means that all the atomic numbers contained
        in the supplied list will be considered.  The user _must_ ensure that
        all species contained in the structure are represented in the list,
        otherwise an error will be raised.  They are free to specify species
        that do not occur in any structure, however; a typical use case of this
        is to compare SOAP vectors between structure sets of possibly different
        species composition.

        These different settings correspond to different trade-offs between
        the memory efficiency of the invariants and the computational
        efficiency of the kernel computation.
        When computing a kernel using 'environment wise' setting does not allow
        for efficent matrix matrix multiplications which is ensured when
        'user defined' is used. 'structure wise' is a balance between the
        memory footprint and the use of matrix matrix products.

        Note that the sparsity of the gradient coefficients and their use to
        build kernels does not allow for clear efficiency gains so their
        sparsity is kept irrespective of the value of this parameter.

    compute_gradients : bool, default False
        control the computation of the representation's gradients w.r.t. atomic
        positions.

    cutoff_function_parameters : dict, optional
        Additional parameters for the cutoff function.
        if cutoff_function_type == 'RadialScaling' then it should have the form

        .. code:: python

            dict(rate=...,
                 scale=...,
                 exponent=...)

        where :code:`...` should be replaced by the desired value.

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
        gaussian_sigma_type="Constant",
        gaussian_sigma_constant=0.3,
        cutoff_function_type="ShiftedCosine",
        radial_basis="GTO",
        optimization_args={},
        species_list="environment wise",
        compute_gradients=False,
        cutoff_function_parameters=dict(),
        expansion_by_species_method=None,
        global_species=None,
    ):
        """Construct a SphericalExpansion representation

        Required arguments are all the hyperparameters named in the
        class documentation
        """

        self.name = "sphericalexpansion"
        self.hypers = dict()

        self.update_hyperparameters(
            max_radial=max_radial,
            max_angular=max_angular,
            expansion_by_species_method=expansion_by_species_method,
            global_species=global_species,
            compute_gradients=compute_gradients,
        )
        self.cutoff_function_parameters = deepcopy(cutoff_function_parameters)
        cutoff_function_parameters.update(
            interaction_cutoff=interaction_cutoff,
            cutoff_smooth_width=cutoff_smooth_width,
        )
        cutoff_function = cutoff_function_dict_switch(
            cutoff_function_type, **cutoff_function_parameters
        )
        # Soft backwards compatibility (remove these two if-clauses after 01.11.2021)
        # Soft backwards compatibility (remove this whole if-statement after 01.11.2021)
        if expansion_by_species_method is not None:
            LOGGER.warning(
                "Warning: The 'expansion_by_species_method' parameter is deprecated "
                "(see 'species_list' parameter instead).\n"
                "This message will become an error after 2021-11-01."
            )
            if expansion_by_species_method != "user defined":
                species_list = expansion_by_species_method
            elif global_species is not None:
                species_list = global_species
            else:
                raise ValueError(
                    "Found deprecated 'expansion_by_species_method' parameter "
                    "set to 'user defined' without 'global_species' set")
        elif global_species is not None:
            LOGGER.warning(
                "Warning: The 'global_species' parameter is deprecated "
                "(see 'species_list' parameter instead).\n"
                "This message will become an error after 2021-11-01."
                "(Also, this needs to be set with "
                "'expansion_by_species_method'=='user defined'; proceeding under "
                "the assumption that this is what you wanted)"
            )
            species_list = global_species
        species_list_hypers = _parse_species_list(species_list)
        self.update_hyperparameters(**species_list_hypers)

        gaussian_density = dict(
            type=gaussian_sigma_type,
            gaussian_sigma=dict(value=gaussian_sigma_constant, unit="A"),
        )
        self.optimization_args = deepcopy(optimization_args)
        if "type" in optimization_args:
            if optimization_args["type"] == "Spline":
                if "accuracy" in optimization_args:
                    accuracy = optimization_args["accuracy"]
                else:
                    accuracy = 1e-5
                    print(
                        "No accuracy for spline optimization was given. "
                        "Switching to default accuracy {:.0e}.".format(
                            accuracy
                        )
                    )
                optimization_args = {"type": "Spline", "accuracy": accuracy}
            elif optimization_args["type"] == "None":
                optimization_args = dict({"type": "None"})
            else:
                print(
                    "Optimization type is not known. Switching to no" " optimization."
                )
                optimization_args = dict({"type": "None"})
        else:
            optimization_args = dict({"type": "None"})
        radial_contribution = dict(type=radial_basis, optimization=optimization_args)

        self.update_hyperparameters(
            cutoff_function=cutoff_function,
            gaussian_density=gaussian_density,
            radial_contribution=radial_contribution,
        )

        self.nl_options = [
            dict(name="centers", args=dict()),
            dict(name="neighbourlist", args=dict(cutoff=interaction_cutoff)),
            dict(name="centercontribution", args=dict()),
            dict(name="strict", args=dict(cutoff=interaction_cutoff)),
        ]

        self.rep_options = dict(name=self.name, args=[self.hypers])

        n_features = self.get_num_coefficients()

        self._representation = CalculatorFactory(self.rep_options)

    def update_hyperparameters(self, **hypers):
        """Store the given dict of hyperparameters

        Also updates the internal json-like _representation

        """
        allowed_keys = {
            "interaction_cutoff",
            "cutoff_smooth_width",
            "max_radial",
            "max_angular",
            "gaussian_sigma_type",
            "gaussian_sigma_constant",
            "gaussian_density",
            "cutoff_function",
            "radial_contribution",
            "compute_gradients",
            "cutoff_function_parameters",
            "expansion_by_species_method",
            "global_species",
        }
        hypers_clean = {key: hypers[key] for key in hypers if key in allowed_keys}
        self.hypers.update(hypers_clean)

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
        """Return the number of coefficients in the spherical expansion

        (this is the descriptor size per atomic centre)

        """
        return (
            n_species
            * self.hypers["max_radial"]
            * (self.hypers["max_angular"] + 1) ** 2
        )

    def get_keys(self, species):
        """
        return the proper list of keys used to build the representation
        """
        keys = []
        for sp in species:
            keys.append([sp])
        return keys

    def _get_init_params(self):
        gaussian_density = self.hypers["gaussian_density"]
        cutoff_function = self.hypers["cutoff_function"]
        radial_contribution = self.hypers["radial_contribution"]
        global_species = self.hypers["global_species"]
        species_list = (
            global_species
            if global_species
            else self.hypers["expansion_by_species_method"]
        )

        init_params = dict(
            interaction_cutoff=cutoff_function["cutoff"]["value"],
            cutoff_smooth_width=cutoff_function["smooth_width"]["value"],
            max_radial=self.hypers["max_radial"],
            max_angular=self.hypers["max_angular"],
            species_list=species_list,
            compute_gradients=self.hypers["compute_gradients"],
            gaussian_sigma_type=gaussian_density["type"],
            gaussian_sigma_constant=gaussian_density["gaussian_sigma"]["value"],
            cutoff_function_type=cutoff_function["type"],
            radial_basis=radial_contribution["type"],
            optimization_args=self.optimization_args,
            cutoff_function_parameters=self.cutoff_function_parameters,
        )
        return init_params

    def _set_data(self, data):
        super()._set_data(data)

    def _get_data(self):
        return super()._get_data()
