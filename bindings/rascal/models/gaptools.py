#!/usr/bin/env python
# coding: utf-8

import os
from collections.abc import Iterable

import ase.io
import numpy as np

from rascal import models, representations, utils
from rascal.models.krr import SparseGPRSolver

WORKDIR = os.getcwd()


def calculate_representation(
    geoms, rep_parameters, rep_class=representations.SphericalInvariants, auto_wrap=True
):
    """Calculate SOAP vectors without sparsification

    Parameters:
        geoms       List of Atoms objects to transform
        rep_parameters
                    Dictionary of parameters used to initialize the
                    representation
    Optional arguments:
        rep_class   Class that specifies which representation to use
                    (default rascal.representations.SphericalInvariants)
        auto_wrap   Automatically wrap all the atoms so they are
                    "inside" the periodic cell by the libRascal
                    definition?  (Default True, recommended for periodic
                    structures).
                    WARNING: Structures intended to be treated as
                    non-periodic require special handling -- do not use
                    this option!  Manually pad the cell and shift the
                    positions instead.

    Returns the Representation object and the SOAP vectors in a tuple.
    """
    rep = rep_class(**rep_parameters)
    if auto_wrap:
        for geom in geoms:
            geom.wrap(eps=1e-10)
    soaps = rep.transform(geoms)
    return rep, soaps


def calculate_features(
    geoms,
    rep_parameters,
    rep_class=representations.SphericalInvariants,
    auto_wrap=True,
):
    """Calculate feature vectors (e.g. SOAPs)

    Parameters:
        geoms       List of Atoms objects to transform
        rep_parameters
                    Dictionary of parameters used to initialize the
                    representation
    Optional arguments:
        rep_class   Class that specifies which representation to use
                    (default rascal.representations.SphericalInvariants)
        auto_wrap   Automatically wrap all the atoms so they are
                    "inside" the periodic cell by the libRascal
                    definition?  (Default True, recommended for periodic
                    structures).
                    WARNING: Structures intended to be treated as
                    non-periodic require special handling -- do not use
                    this option!  Manually pad the cell and shift the
                    positions instead.

    Returns the Representation object and the SOAP vectors (the latter in
    the internal librascal representation)
    """
    rep = rep_class(**rep_parameters)
    if auto_wrap:
        for geom in geoms:
            geom.wrap(eps=1e-10)
    soaps = rep.transform(geoms)
    return rep, soaps


# TODO also support feature sparsification (this just does env. sparsification
#      for now)
def sparsify_environments(
    rep, features, n_sparse, selection_type="CUR", save_sparsepoints=False
):
    """Sparsify the feature matrix along the environments dimension

    Parameters:
        rep         Representation calculator used to compute the features
        features    List of features to use for sparsification, in the
                    internal librascal representation
        n_sparse    Number of sparse points per species, in the form of
                    a dict mapping atomic number to number of requested
                    sparse points
    Optional arguments:
        selection_type
            Which selection algorithm to use. 'CUR' and 'FPS'
            are supported (see the documentation in rascal.utils
            for details on each method), default CUR.
        save_sparsepoints
            Write the list of sparse points to a file for later use?
            (default False)

    Returns the sparse points, again in the internal librascal representation
    """
    if selection_type.upper() == "CUR":
        compressor = utils.CURFilter(rep, n_sparse, act_on="sample per species")
    elif selection_type.upper() == "FPS":
        # TODO random starting index? by default?
        compressor = utils.FPSFilter(rep, n_sparse, act_on="sample per species")
    sparse_points = compressor.select_and_filter(features)
    if save_sparsepoints:
        utils.dump_obj(os.path.join(WORKDIR, "sparsepoints.json"), sparse_points)
    return sparse_points


def build_sparse_list(list_natoms, absolute_index_list):
    """Convert an absolute-indexed selection to structure-indexed format

    Parameters:
        list_natoms     List of the number of atoms in each structure
        absolute_index_list
                        List of sparse points selected, indexed from the
                        beginning of the entire feature matrix
                        (Note: if you need to preserve the order of the sparse
                        points, then you should not use Rascal's SparsePoints)
    """
    index_breaks = np.cumsum(list_natoms)
    find_struct = lambda i: np.searchsorted(index_breaks - 1, i)
    structure_based_indices = [[] for n in list_natoms]
    for abs_idx in absolute_index_list:
        structure_index = find_struct(abs_idx)
        offset = index_breaks[structure_index - 1] if (structure_index > 0) else 0
        structure_based_indices[structure_index].append(abs_idx - offset)
    return structure_based_indices


def compute_kernels(
    rep,
    soaps,
    sparse_points,
    soap_power=2,
    do_gradients=True,
    compute_sparse_kernel=True,
    target_type="Structure",
    save_kernels=False,
):
    """Compute the kernels necessary for a GAP fit

    Parameters:
        rep     Representation object (holds representation
                hyperparameters; necessary for initializing the kernel)
        soaps   SOAP vectors of all training structures
        sparse_points
                Support points for the fit, containing SOAP vectors for
                the selected environments

    Optional arguments:
        soap_power
                Integer power to which to raise the SOAP kernel;
                defaults to 2 (to make the kernel nonlinear)
        do_gradients
                Whether to compute the gradients kernel as well in order
                to fit forces.  Default True; fitting with gradients is
                more expensive but usually much more accurate than
                fitting on energies alone.
        compute_sparse_kernel
                Whether to compute the sparse-sparse kernel (K_MM)
                needed to do a fit.  Note that it is not necessary to
                recompute this kernel when evaluating an existing fit on
                new data; therefore, this option can be set to False for
                that task.  If this option is set to False, an empty
                array will be returned in place of the sparse kernel.
        target_type : string
                Type of target (prediction) properties, must be either 'Atom' (the kernel
                is between atomic environments) or 'Structure' (the kernel is summed over
                atoms in a structure), which is the default

    Returns a tuple of: the kernel object (which contains all the kernel
    parameters), the sparse kernel, and the energy kernel (and force
    kernel, if requested).
    """
    kernel = models.Kernel(
        rep, name="GAP", zeta=soap_power, target_type=target_type, kernel_type="Sparse"
    )
    if compute_sparse_kernel:
        kernel_sparse = kernel(sparse_points)
        if save_kernels:
            np.save(os.path.join(WORKDIR, "K_MM"), kernel_sparse)
    else:
        kernel_sparse = np.array([])
    kernel_sparse_full = kernel(soaps, sparse_points)
    # TODO make kernel name configurable so we don't overwrite training
    #     kernels with possible future test kernels
    if save_kernels:
        np.save(os.path.join(WORKDIR, "K_NM_E"), kernel_sparse_full)
    if do_gradients:
        if target_type == "Atom":
            raise ValueError(
                "Atom-centered properties do not support gradients at present"
            )
        kernel_sparse_full_grads = kernel(soaps, sparse_points, grad=(True, False))
        if save_kernels:
            np.save(os.path.join(WORKDIR, "K_NM_F"), kernel_sparse_full_grads)
        return (kernel, kernel_sparse, kernel_sparse_full, kernel_sparse_full_grads)
    else:
        return kernel, kernel_sparse, kernel_sparse_full


def _get_energy_baseline(geom, atom_contributions, target_type="Structure"):
    """Get the energy baseline for a single structure

    Depends only on the number of atoms of each atomic species present;
    the 'atom_contributions' dictionary says how much energy to assign
    to an atom of each species.
    """

    if target_type == "Structure":
        e0 = 0.0
        for species, e0_value in atom_contributions.items():
            e0 += e0_value * np.sum(geom.get_atomic_numbers() == species)
    elif target_type == "Atom":
        if "center_atoms_mask" in geom.arrays:
            # selects only the masked atoms
            e0 = np.asarray(
                [
                    atom_contributions[species]
                    for species in geom.get_atomic_numbers()[
                        geom.arrays["center_atoms_mask"]
                    ]
                ]
            )
        else:
            # loop over all atoms
            e0 = np.asarray(
                [atom_contributions[species] for species in geom.get_atomic_numbers()]
            )
    else:
        raise ValueError("Invalid baseline target ", target_type)
    return e0


def extract_kernel_indices(
    idces, kernel, kernel_grads=None, natoms=None, target_type="Structure"
):
    """Extract rows from the sparse-full kernel ("K_NM")

    This is useful e.g. to perform a train-test or a CV split

    Parameters
    ----------
    idces: list(int) or 1-D array
        List of indices (structure or atom) to extract
    kernel: 2-D array
        Sparse-full kernel ("K_NM")
    kernel_grads: 2-D array, optional
        Gradient of sparse-full kernel w.r.t. atomic positions
    natoms: int or list(int)
        Number of atoms in each structure.  Can be a single number
        if all structures have the same number of atoms.  Only
        needed if kernel_grads supplied and target_type=="Structure".
    target_type: "Structure" or "Atom"
        Whether the sparse-full kernel and extraction indices are
        for structures or atoms - "Structure" means structure indices
        and structure (summed) kernels, "Atom" means atom indices
        and atomic kernels

    Returns
    -------
    kernel_sub: 2-D array
        Extracted subset of kernel matrix
    kernel_grads_sub: 2-D array
        Corresponding subset of gradient kernel matrix, if supplied
    """
    n_rows = kernel.shape[0]
    n_sub = len(idces)
    kernel_sub = kernel[idces]
    if kernel_grads is None:
        return kernel_sub
    if target_type == "Atom":
        kernel_grads_sub = kernel_grads.reshape((n_rows, 3, -1))[idces].reshape(
            (n_sub * 3, -1)
        )
    elif target_type == "Structure":
        if natoms is None:
            raise ValueError(
                "Must supply number of atoms for each structure "
                "in order to extract subset from kernel_grads"
            )
        elif isinstance(natoms, Iterable):
            kernel_grads_sub = []
            kernel_grads_peratom = kernel_grads.reshape((sum(natoms), 3, -1))
            offsets = np.cumsum(natoms)
            for idx in idces:
                nat = natoms[idx]
                offset = offsets[idx]
                kernel_grads_sub.append(kernel_grads_peratom[offset : offset + nat])
            kernel_grads_sub = np.concatenate(kernel_grads_sub)
            n_grads_sub = kernel_grads_sub.shape[0]
            kernel_grads_sub = kernel_grads_sub.reshape((n_grads_sub * 3, -1))
        else:
            kernel_grads_sub = kernel_grads.reshape((n_rows, natoms, 3, -1))[
                idces
            ].reshape((n_sub * natoms * 3, -1))
    else:
        raise ValueError(f'Unrecognized target_type: "{target_type:s}"')
    return kernel_sub, kernel_grads_sub


# TODO also make energies optional
def fit_gap_simple(
    geoms,
    kernel_sparse,
    energies,
    kernel_energies_sparse,
    energy_regularizer_peratom,
    energy_atom_contributions=None,
    forces=None,
    kernel_gradients_sparse=None,
    force_regularizer=None,
    solver="Normal",
    jitter=1e-10,
    target_type="Structure",
):
    """
    Fit a GAP model to total energies and optionally forces

    Simple version; just takes geometries, kernels, energies, and forces,
    returning only the weights.  No automatic scaling by target property
    variance; the regularizers should therefore be the _ratio_ of the
    expected error in the property to the expected scale (variance) of
    the fitted energy surface.

    In the notation below, M is the number of sparse points, N is the
    number of training structures, and P is the total number of atoms
    in the training structures.

    Parameters:
        geoms           Training structures: List of Atoms objects
        kernel_sparse   Kernel between sparse points and themselves:
                        NumPy array of shape MxM
        energies        Total energies of the structures to fit
        kernel_energies_sparse
                        Kernel between sparse points and training
                        structures: NumPy array of shape NxM
        energy_regularizer_peratom
                        Energy regularizer (actually ratio between
                        expected error and expected target variance;
                        see above).  Expressed in energy units per atom,
                        therefore scaled with the size of the structure.
        energy_atom_contributions
                        Baseline energy contributions per atomic species
                        Dict mapping species to baseline energy value
                        per atom. None to avoid baselining

    Parameters for force fitting:
        forces          Forces of the structures to fit: NumPy array of
                        shape Px3 (or NxQx3 where N*Q=P)
        kernel_gradients_sparse
                        Gradient kernel between sparse points and target
                        structures: NumPy array of shape 3PxM
                        Note that gradients are the negative of the
                        forces, though this is mostly handled
                        transparently by the code (just pass in forces)
        force_regularizer
                        Force regularizer (see above), in units of
                        energy / distance (component-wise)

    Optional arguments:
        solver          Which solver to use for the sparse GPR equations.
                        Options are "Normal", "QR", "RKHS", and "RKHS-QR".
                        See the documentation of
                        rascal.models.krr.SparseGPRSolver for details.
                        Default "Normal".
        jitter          Value to add to the diagonal of the sparse GPR
                        equations to make them stable, see SparseGPRSolver
                        for details. Default 1E-10.

    Returns the weights (1-D array, size M) that define the fit.
    """
    energies_shifted = energies
    if energy_atom_contributions is not None:
        e0_all = np.array(
            [
                _get_energy_baseline(geom, energy_atom_contributions, target_type)
                for geom in geoms
            ]
        ).flatten()
        energies_shifted -= e0_all

    if target_type == "Structure":
        natoms_list = np.array([len(geom) for geom in geoms])
        energy_regularizer = energy_regularizer_peratom * np.sqrt(natoms_list)
        kernel_energies_norm = (
            kernel_energies_sparse / energy_regularizer[:, np.newaxis]
        )
    else:
        energy_regularizer = energy_regularizer_peratom
        kernel_energies_norm = kernel_energies_sparse / energy_regularizer
    if forces is not None:
        gradients = -1 * forces
        kernel_gradients_norm = kernel_gradients_sparse / force_regularizer
        K_NM = np.vstack((kernel_energies_norm, kernel_gradients_norm))
        Y = np.concatenate(
            (
                energies_shifted / energy_regularizer,
                gradients.flatten() / force_regularizer,
            )
        )
    else:
        K_NM = kernel_energies_norm
        Y = energies_shifted / energy_regularizer

    # Old, unstable version - the active code below should be equivalent to
    # solving these equations in a more stable way.
    # K = kernel_sparse + K_NM.T @ K_NM
    # weights, *_ = np.linalg.lstsq(K, K_NM.T @ Y, rcond=rcond)
    #
    # The regularizer is pre-multiplied with the kernel matrices, so we do not
    # include it in the solver below.  This also implies the use of an absolute
    # jitter.
    solver = SparseGPRSolver(
        kernel_sparse,
        regularizer=1,
        jitter=jitter,
        solver=solver,
        relative_jitter=False,
    )
    solver.fit(K_NM, Y)
    return solver._weights


def load_potential(
    model_in, rep_parameters, rep_class=representations.SphericalInvariants
):
    """Load a previously fitted model as an ASE Calculator

    Parameters:
        model_in            KRR model, either a rascal.models.KRR
                            instance or the filename of a model stored
                            in JSON format
        rep_parameters      Dictionary of parameters used to initialize
                            the representation
        rep_class           Class that specifies which representation
                            to use (default
                            rascal.representations.SphericalInvariants)

    Returns an ASE Calculator (rascal.models.ASEMLCalculator instance)
    that computes energies and forces of ASE Atoms objects
    """
    if isinstance(model_in, models.KRR):
        model = model_in
    else:
        model = utils.load_obj(model_in)
    rep = rep_class(**rep_parameters)
    return models.IP_ase_interface.ASEMLCalculator(model, rep)
