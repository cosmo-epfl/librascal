import unittest

import ase.io

from rascal.utils import FPSFilter, CURFilter
from rascal.representations import SphericalInvariants as SOAP

class CURTest(unittest.TestCase):

    def setUp(self):
        example_frames = ase.io.read('reference_data/inputs/small_molecules-20.json', ':')
        global_species = set()
        for frame in example_frames:
            global_species.update([int(sp) for sp in frame.get_atomic_numbers()])
        repr = SOAP(
            max_radial=8,
            max_angular=6,
            interaction_cutoff=4.0,
            cutoff_smooth_width=1.0,
            gaussian_sigma_type="Constant",
            gaussian_sigma_constant=0.3,
            expansion_by_species_method="user defined",
            global_species=sorted(list(global_species))
        )
        managers = repr.transform(example_frames)
        self.example_features = managers.get_features(repr)
        self.repr = repr
        self.example_frames = example_frames

    def test_one(self):
        pass

if __name__ == "__main__":
    unittest.main(verbosity=2)
