import argparse
from collections.abc import Iterable, Mapping
import os
import sys
import time

import ase.io
import numpy as np

import rascal.utils
import rascal.utils.io
import rascal.models
from rascal.models import gaptools


def _get_energy_baseline(energy_baseline_in, source_geoms):
    if isinstance(energy_baseline_in, Mapping):
        energy_baseline = energy_baseline_in
    else:
        all_species = set()
        for geom in source_geoms:
            all_species = all_species.union(geom.get_atomic_numbers())
            # Convert to int because otherwise it's a numpy type
            # (which doesn't play well with json)
            all_species = set(int(sp) for sp in all_species)
        energy_baseline = {species: energy_baseline_in for species in all_species}
    return energy_baseline


def _get_n_train(n_train_in, n_test_in, n_geoms):
    if n_train_in is None:
        n_train = n_geoms
    do_learning_curve = isinstance(n_train_in, Iterable)
    if n_test_in is None:
        if do_learning_curve:
            n_test = n_geoms - max(n_train_in)
        else:
            n_test = n_geoms - n_train
    else:
        n_test = n_test_in
    if do_learning_curve:
        n_train_all = n_train
    else:
        n_train_all = [n_train]
    return n_train_all, n_test, do_learning_curve


def fit_save_model(parameters):
    WORKDIR = parameters.working_directory
    gaptools.WORKDIR = WORKDIR
    os.makedirs(WORKDIR, exist_ok=True)  # Works from Python 3.2 onward

    source_geoms = ase.io.read(parameters.structure_filename, ":")
    # Source geometry transformation
    if parameters.shuffle_file:
        idces = np.loadtxt(parameters.shuffle_file, dtype=int)
        source_geoms = [source_geoms[idx] for idx in idces]
    natoms_source = [len(geom) for geom in source_geoms]

    n_train_all, n_test, do_learning_curve = _get_n_train(
        parameters.n_train, parameters.n_test, len(source_geoms)
    )

    rep, soaps, sparse_points = gaptools.calculate_and_sparsify(
        source_geoms, parameters.soap_hypers, parameters.n_sparse
    )
    (kobj, kernel_sparse, kernel_energies, kernel_forces) = gaptools.compute_kernels(
        rep, soaps, sparse_points, parameters.soap_power
    )
    eparam_name = parameters.energy_parameter_name
    fparam_name = parameters.force_parameter_name
    energies = np.array([geom.info[eparam_name] for geom in source_geoms])
    forces = np.array([geom.arrays[fparam_name] for geom in source_geoms])
    energy_baseline = _get_energy_baseline(parameters.atom_energy_baseline)
    energy_delta = parameters.energy_delta
    # Uncomment the below to get mainline train_gap behaviour
    # energy_delta = np.std(energies)
    energy_regularizer = parameters.energy_regularizer / energy_delta
    force_regularizer = parameters.force_regularizer / energy_delta
    # TODO bundle all the parameters needed below into a "fit object"
    if parameters.print_residuals:
        if parameters.n_test > 0:
            print(f"Residuals on {parameters.n_test:} testing structures:")
        else:
            print("Residuals on training set:")
    for n_train in parameters.n_train:
        energies_train = energies[:n_train]
        kernel_energies_train = kernel_energies[:n_train]
        forces_train = forces[:n_train]
        kernel_forces_train = kernel_forces[: 3 * sum(natoms_source[:n_train])]
        weights = gaptools.fit_gap_simple(
            source_geoms[:n_train],
            kernel_sparse,
            energies_train,
            kernel_energies_train,
            energy_regularizer,
            energy_baseline,
            forces_train,
            kernel_forces_train,
            force_regularizer,
        )
        np.save(os.path.join(WORKDIR, "weights"), weights)
        model_description = parameters.description
        if model_description:
            if model_description.endswith("."):
                model_description += " "
            else:
                model_description += ". "
        model_timestamp = time.strftime("%Y-%m-%d at %H:%M:%S %Z", time.localtime())
        model_description += "Generated by fit_gap.py on {:s}; N_train = {:d}".format(
            model_timestamp, n_train
        )
        model = rascal.models.KRR(
            weights, kobj, sparse_points, energy_baseline, description=model_description
        )
        rascal.utils.dump_obj(parameters.output_filename.format(n_train), model)
        if (parameters.print_residuals) or (parameters.write_residuals):
            if parameters.n_test > 0:
                test_subset = np.arange(len(source_geoms))[-1 * parameters.n_test :]
                natoms_test = natoms_source[-1 * parameters.n_test :]
                test_force_slice = slice(-3 * sum(natoms_test), None)
            else:
                test_subset = np.arange(len(source_geoms))[n_train:]
                natoms_test = natoms_source[n_train:]
                test_force_slice = slice(-3 * sum(natoms_test), None)
            test_soaps = soaps.get_subset(test_subset)
            test_energies = energies[test_subset]
            test_forces = forces[test_subset]
            kernel_energies_test = kernel_energies[test_subset]
            kernel_forces_test = kernel_forces[test_force_slice]
            if parameters.print_residuals:
                print(f"----- {n_train: 4d} train structures -----")
            compute_print_residuals(
                test_soaps,
                model,
                test_energies,
                kernel_energies_test,
                test_forces,
                kernel_forces_test,
                natoms_test,
                parameters.print_residuals,
                parameters.write_residuals.format(n_train),
            )


def compute_print_residuals(
    soaps,
    model,
    energies,
    kernel_energies,
    forces,
    kernel_forces,
    natoms_perstruct,
    do_print=False,
    filename="",
):
    # get predictions
    pred_energies = model.predict(soaps, kernel_energies)
    pred_forces = model.predict_forces(soaps, kernel_forces)
    energy_resids = pred_energies - energies
    force_resids = pred_forces - forces.reshape((-1, 3))
    energy_rmse = np.sqrt(np.mean(energy_resids ** 2))
    energy_rmse_peratom = np.sqrt(
        np.mean((energy_resids / np.array(natoms_perstruct)) ** 2)
    )
    force_rmse = np.sqrt(np.sum(force_resids ** 2) / np.sum(natoms_perstruct))
    if do_print:
        print(
            f"Energy RMSE: {energy_rmse:.06f} eV ({energy_rmse_peratom*1000:.06f} eV/atom)"
        )
        print(f"Force RMSE: {force_rmse:.06f} eV/Å")
    if filename:
        np.savez(
            filename,
            energy_resids=energy_resids,
            force_resids=force_resids,
            energy_rmse=energy_rmse,
            energy_rmse_peratom=energy_rmse_peratom,
            force_rmse=force_rmse,
        )


# TODO add arguments also present in JSON, with appropriate defaults
# (just don't give them short names; reserve those for flags commonly given
# on the command line -- but do include the structure and model filenames)
def parse_command_line():
    parser = argparse.ArgumentParser(description="Fit a GAP model")
    parser.add_argument("parameter_file", help="Name of a JSON parameter file")
    parser.add_argument(
        "-n",
        "--n-train",
        type=int,
        nargs="+",
        help="Number of training structures at the beginning of the file (provide multiple values to do a learning curve)",
    )
    parser.add_argument(
        "-t",
        "--n-test",
        type=int,
        help="Number of testing structures at the end of the file, by default all the remaining structures in the input file",
    )
    parser.add_argument(
        "-s", "--shuffle-file", help="File giving indices for shuffling of input file"
    )
    parser.add_argument(
        "-p",
        "--print-residuals",
        action="store_true",
        help="Print RMSE residuals on the test set (if n_test > 0), or else the training set.",
    )
    parser.add_argument(
        "-w",
        "--write-residuals",
        help="Write residuals to the given filename, formatted with n_train",
        default="",
    )
    parser.add_argument(
        "-o", "--output-filename", help="Name of the file to write the GAP model to"
    )
    parser.add_argument(
        "-z",
        "--soap-power",
        help='Exponent to use for the SOAP kernel ("zeta")',
        type=int,
        default=2,
    )
    parser.add_argument(
        "--energy-parameter-name",
        help="Name of the energy parameter in the input file",
        default="energy",
    )
    parser.add_argument(
        "--force-parameter-name",
        help="Name of the force parameter in the input file",
        default="force",
    )
    parser.add_argument(
        "--description",
        help="Descriptive text to write into the model file",
        default="",
    )
    parser.add_argument(
        "--working-directory",
        help="Directory in which to write kernels and other large, temporary files",
        default="potential_default",
    )
    # Use the rascal loader to be able to read integer keys
    args = parser.parse_args()
    fit_parameters = rascal.utils.io.load_json(args.parameter_file)
    fit_parameters = argparse.Namespace(**fit_parameters)
    # Parameters given on the command line override those in the JSON
    # TODO is there any way to avoid having to do this twice?
    args = parser.parse_args(namespace=fit_parameters)
    # TODO move some validation and as much as possible of the pre-processing here
    return fit_parameters


if __name__ == "__main__":
    fit_save_model(parse_command_line())