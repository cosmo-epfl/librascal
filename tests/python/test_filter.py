import unittest
import numpy as np
import ase.io

from rascal.utils import FPSFilter, CURFilter
from rascal.representations import SphericalInvariants as SOAP


class FilterTest:
    def abstractSetUp(self):
        example_frames = ase.io.read(
            "reference_data/inputs/small_molecules-20.json", ":"
        )
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
            global_species=sorted(list(global_species)),
        )
        self.managers = repr.transform(example_frames)
        self.example_features = self.managers.get_features(repr)
        self.repr = repr
        self.example_frames = example_frames

    def test_sps_mode(self):
        """This test checks that `sample per species` mode returns the correct
        number of selected samples per each species
        """
        n_sparses = {1: 0, 6: 2, 7: 4, 8: 1}
        compressor = self._filter(self.repr, n_sparses, act_on="sample per species")

        x = compressor.select_and_filter(self.managers)

        # Checking that these parameters have not been set
        self.assertTrue(compressor.selected_sample_ids is None)
        self.assertTrue(compressor.selected_feature_ids_global is None)

        for sp in n_sparses:
            sp_idx = np.array(compressor.selected_sample_ids_by_sp[sp])
            self.assertEqual(len(sp_idx), n_sparses[sp])

    def test_sample_mode(self):
        """This test checks that `sample` mode returns the correct
        number of selected samples
        """
        n_sparses = self.example_features.shape[0] // 10
        compressor = self._filter(self.repr, n_sparses, act_on="sample")

        idx = np.concatenate(compressor.select_and_filter(self.managers))

        # Checking that these parameters have not been set
        self.assertTrue(compressor.selected_feature_ids_global is None)
        self.assertTrue(compressor.selected_sample_ids_by_sp is None)

        self.assertTrue(compressor.selected_sample_ids is not None)
        self.assertEqual(idx.shape[0], n_sparses)

    def test_feature_mode(self):
        """This test checks that `feature` mode returns the correct
        number of selected feature
        """
        n_sparses = self.example_features.shape[1] // 100
        compressor = self._filter(self.repr, n_sparses, act_on="feature")
        compressor.select_and_filter(self.managers)

        self.assertTrue(compressor.selected_feature_ids_global is not None)
        idx = np.array(compressor.selected_feature_ids_global)
        self.assertEqual(len(idx), n_sparses)

        # Checking that these parameters have not been set
        self.assertTrue(compressor.selected_sample_ids is None)
        self.assertTrue(compressor.selected_sample_ids_by_sp is None)

    def test_missing_species(self):
        """This test checks that asking for a species which is not present
        raises an error.
        """
        n_sparses = {1: 0, 6: 2, 7: 4, 8: 1, 12: 3}
        compressor = self._filter(self.repr, n_sparses, act_on="sample per species")
        with self.assertRaises(ValueError) as cm:
            x = compressor.select_and_filter(self.managers)
            self.assertEqual(
                cm.message,
                "ValueError: Found array with 0 sample(s) (shape=(0, 4480)) while a minimum of 1 is required.",
            )

    def test_bad_mode(self):
        """This test checks that any mode other than `sample`,
        `sample per species`, and `feature` throws an error
        """
        with self.assertRaises(ValueError) as cm:
            compressor = self._filter(self.repr, 1, act_on="bad mode")
            self.assertEqual(
                str(cm.message),
                '"act_on" should be either of: "sample", "sample per species", "feature"',
            )

    def test_new_n(self):
        """This test checks that `filter` can return an arbitrary number of
        selections strictly less than the number selected.
        """
        n_sparses = self.example_features.shape[1] // 100
        compressor = self._filter(self.repr, n_sparses, act_on="feature")
        compressor.select(self.managers)
        compressor.filter(self.managers, n_select=n_sparses // 10)

        with self.assertRaises(ValueError) as cm:
            compressor.filter(self.managers, n_select=n_sparses * 10)
            self.assertEqual(
                str(cm.message),
                f"It is only possible to filter {n_sparses} features, you have requested {n_sparses * 10}",
            )


class FPSTest(FilterTest, unittest.TestCase):
    def setUp(self):
        self._filter = FPSFilter
        self.abstractSetUp()


class CURTest(FilterTest, unittest.TestCase):
    def setUp(self):
        self._filter = CURFilter
        self.abstractSetUp()


if __name__ == "__main__":
    unittest.main(verbosity=2)
