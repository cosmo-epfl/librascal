from rascal.representations import (
    SphericalExpansion,
    SphericalInvariants,
)
from rascal.utils import (
    get_radial_basis_covariance,
    get_radial_basis_pca,
    get_radial_basis_projections,
    get_optimal_radial_basis_hypers,
    json_dumps_frame,
)
from rascal.lib import neighbour_list
from rascal.neighbourlist import base

from test_utils import load_json_frame, BoxList, Box, dot
import tempfile
import unittest
import numpy as np
import sys
import os
import json
import tempfile
from copy import copy, deepcopy
from scipy.stats import ortho_group
import pickle
import ase.io

rascal_reference_path = "reference_data"
inputs_path = os.path.join(rascal_reference_path, "inputs")
dump_path = os.path.join(rascal_reference_path, "tests_only")


class TestOptimalRadialBasis(unittest.TestCase):
    def setUp(self):
        """
        builds the test case.
        """

        fns = [
            os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json"),
            os.path.join(inputs_path, "SiC_moissanite_supercell.json"),
            os.path.join(inputs_path, "methane.json"),
        ]
        self.frames = [load_json_frame(fn) for fn in fns]

        global_species = []
        for frame in self.frames:
            global_species.extend(frame["atom_types"])
        self.global_species = list(np.unique(global_species))

        self.hypers = dict(
            interaction_cutoff=3.5,
            max_radial=6,
            max_angular=6,
            gaussian_sigma_constant=0.4,
            gaussian_sigma_type="Constant",
            cutoff_smooth_width=0.5,
        )

        self.expanded_max_radial = 20

    def test_hypers_construction(self):
        """Checks that manually-constructed and automatic
        framework are consistent."""

        hypers = deepcopy(self.hypers)

        hypers["max_radial"] = self.expanded_max_radial
        spex = SphericalExpansion(**hypers)
        feats = spex.transform(self.frames).get_features_by_species(spex)

        cov = get_radial_basis_covariance(spex, feats)

        p_val, p_vec = get_radial_basis_pca(cov)

        p_mat = get_radial_basis_projections(p_vec, self.hypers["max_radial"])

        # now makes this SOAP
        hypers["max_radial"] = self.hypers["max_radial"]
        hypers["soap_type"] = "PowerSpectrum"
        hypers["optimization"] = {
            "RadialDimReduction": {"projection_matrices": p_mat},
            "Spline": {"accuracy": 1e-8},
        }

        # compute SOAP
        soap_opt = SphericalInvariants(**hypers)
        soap_feats = soap_opt.transform(self.frames).get_features(soap_opt)

        # now we do the same with the compact utils
        hypers = deepcopy(self.hypers)
        hypers["soap_type"] = "PowerSpectrum"
        hypers = get_optimal_radial_basis_hypers(
            hypers, self.frames, expanded_max_radial=self.expanded_max_radial
        )
        soap_opt_2 = SphericalInvariants(**hypers)
        soap_feats_2 = soap_opt_2.transform(self.frames).get_features(soap_opt_2)

        self.assertTrue(np.allclose(soap_feats, soap_feats_2))


class TestIO(unittest.TestCase):
    def setUp(self):
        self.fns = [
            os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json"),
            os.path.join(inputs_path, "SiC_moissanite_supercell.json"),
            os.path.join(inputs_path, "methane.json"),
        ]

    def test_json_dumps_frame(self):
        """
        Checks if json file decoded by RascalEncoder in dumps_frame can be read
        by rascal
        """
        nl_options = [
            dict(name="centers", args=dict()),
            dict(name="neighbourlist", args=dict(cutoff=3)),
            dict(name="centercontribution", args=dict()),
            dict(name="strict", args=dict(cutoff=3)),
        ]
        managers = base.StructureCollectionFactory(nl_options)
        for fn in self.fns:
            frame = ase.io.read(fn)
            dumped_json = json_dumps_frame(frame)
            tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            tmp.write(dumped_json)
            try:
                managers.add_structures(tmp.name)
                tmp.close()
                os.unlink(tmp.name)
            except:
                tmp.close()
                os.unlink(tmp.name)
