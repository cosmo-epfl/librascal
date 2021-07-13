import sys
import os
import json
import argparse
import numpy as np
import ubjson
import json
from scipy.special import sph_harm, lpmn
from mpmath import mp, spherharm
import mpmath
from functools import reduce

# shape (nb_unit_vectors,3)

root = os.path.abspath("../")
rascal_reference_path = os.path.join(root, "reference_data/")
inputs_path = os.path.join(rascal_reference_path, "tests_only")
dump_path = os.path.join("reference_data/", "tests_only")


def load_unit_vectors_from_json():
    fn = os.path.join(inputs_path, "spherical_harmonics_test.json")
    with open(fn, "r") as f:
        data = json.load(f)
    return data["unit_vectors"]


# Sized (l_max+0)**2, contains the l,m components in compressed
# format, i.e. (00)(1-1)(10)(11)(2-2).... as it is in our spherical
# harmonics calculation


def get_ascending_angular_lists(max_angular_l):
    ascending_angular_m_list = []
    ascending_angular_l_list = []
    for angular_l in range(max_angular_l + 1):
        ascending_angular_m_list += list(range(-angular_l, angular_l + 1))
        ascending_angular_l_list += [angular_l] * (angular_l * 2 + 1)
    return ascending_angular_m_list, ascending_angular_l_list


# scipy alreay includes the Condon-Shortley phase, therefore to calculate the
# real form we use
#         ╭ (-1)^|m| √2 Im[Y_l^|m|] for m<0
#         |
# Y_l^m = ┤     Y_l^0      for m==0
#         |
#         ╰ (-1)^m √2 Re[Y_l^m]   for m<0
# where Y_l^m is the output of the spherical harmonics function of scipy
#  following the definition in
# https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form


def dump_reference_json():
    # to produces more readable tests use l_max 2 or 3
    verbose = False
    l_max = 30
    sys.path.insert(0, os.path.join(root, "build/"))
    sys.path.insert(0, os.path.join(root, "tests/"))
    data = []
    mp.dps = 200

    # Calculation of spherical harmonics
    # with mpmath:
    ## spherharm(angular_l,angular_m, theta, phi)
    # with scipy:
    ## sph_harm(angular_m, angular_l, phi, theta)

    unit_vectors = load_unit_vectors_from_json()

    # In arXiv 1410.1748 e.q. (4) the factors in front of the associated
    # polynomial
    alp_normfacts = np.zeros((l_max + 1, l_max + 1))
    for l in range(l_max + 1):
        for m in range(l + 1):
            alp_normfacts[l, m] = mpmath.sqrt(
                (2 * l + 1)
                / (2 * mpmath.pi)
                / reduce(
                    lambda x, y: mpmath.fmul(x, y),
                    mpmath.arange(l - m + 1, l + m + 1),
                    1,
                )
            )
    if verbose:
        print("alp_normfacts")
        print(alp_normfacts)

    for unit_vector in unit_vectors:
        # Calculating the spherical harmonics

        harmonics = []
        # copy of c++ code:
        # double cos_theta = unit_vector[2];
        cos_theta = unit_vector[2]
        theta = np.arccos(cos_theta)
        # copy of c++ code:
        # double phi = std::atan2(unit_vector[1], unit_vector[0]);
        phi = np.arctan2(unit_vector[1], unit_vector[0])
        if verbose:
            print(unit_vector)
        for l in range(l_max + 1):

            # calculation for negative m
            for m in range(-l, 0):
                result = (
                    (-1) ** m * np.sqrt(2) * np.imag(sph_harm(np.abs(m), l, phi, theta))
                )
                harmonics.append(float(result))
                if verbose:
                    print(l, m, result)
            # calculation for m=0
            result = np.real(sph_harm(0, l, phi, theta))
            harmonics.append(float(result))
            if verbose:
                print(l, 0, result)
            # calculation for positive m
            for m in range(1, l + 1):
                result = (-1) ** m * np.sqrt(2) * np.real(sph_harm(m, l, phi, theta))
            print(alps)
        alps *= alp_normfacts
        if verbose:
            print("alps")
            print(alps)
        alps = list(map(float, alps.reshape(-1).tolist()))
        data.append(
            dict(
                max_angular_l=int(l_max),
                unit_vector=unit_vector,
                harmonics=harmonics,
                alps=alps,
            )
        )
    if verbose:
        print(len(data))
    with open(
        os.path.join(root, dump_path, "spherical_harmonics_reference.ubjson"),
        "wb",
    ) as f:
        ubjson.dump(data, f)


###############################################################################
###############################################################################


def main(json_dump):
    if json_dump == True:
        dump_reference_json()


###############################################################################
###############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json_dump", action="store_true", help="Switch for dumping json"
    )

    args = parser.parse_args()
    main(args.json_dump)
