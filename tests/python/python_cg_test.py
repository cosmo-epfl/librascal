from rascal.representations import (
    SphericalExpansion,
    SphericalInvariants,
)
from rascal.utils import (
    ClebschGordanReal, WignerDReal, spherical_expansion_reshape, lm_slice
)

from test_utils import load_json_frame, BoxList, Box, dot
import unittest
import numpy as np
import sys
import os
import json
from copy import copy, deepcopy
from scipy.stats import ortho_group
import pickle

rascal_reference_path = "reference_data"
inputs_path = os.path.join(rascal_reference_path, "inputs")
dump_path = os.path.join(rascal_reference_path, "tests_only")


class TestCGUtils(unittest.TestCase):
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
            max_angular=3,
            gaussian_sigma_constant=0.4,
            gaussian_sigma_type="Constant",
            cutoff_smooth_width=0.5,
        )

        self.lmax = self.hypers["max_angular"]
        self.wd = WignerDReal(self.lmax, 0.1, 0.2, 0.3)
        self.cg = ClebschGordanReal(self.lmax)        

        self.spex = SphericalExpansion(**self.hypers)
        self.feats = spherical_expansion_reshape(
            self.spex.transform(self.frames).get_features(self.spex), **self.hypers)


    def test_rotation(self):
        """Checks that spherical expansion features transform as they should."""

        rframes = []
        for f in self.frames:
            rf = f.copy()
            self.wd.rotate_frame(rf)
            rframes.append(rf)
        
        rfeats = spherical_expansion_reshape(
            self.spex.transform(rframes).get_features(self.spex), **self.hypers)
        for L in range(self.lmax+1):
            self.assertTrue(np.allclose(rfeats[...,lm_slice(L)], 
                self.wd.rotate(self.feats[..., lm_slice(L)])))
            
    def test_invariants(self):
        """Checks that spherical invariants computed from CGs are equal to SOAP features."""
        
        hypers = deepcopy(self.hypers)
        hypers["soap_type"] = "PowerSpectrum"
        hypers["normalize"] = False
        
        soap = SphericalInvariants(**hypers)
        sfeats = soap.transform(self.frames).get_features(soap)
        
        pass
        
