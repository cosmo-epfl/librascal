from ase.io import read
from ase.build import molecule
from rascal.neighbourlist import get_neighbourlist, AtomsList
from rascal.neighbourlist.structure_manager import (
    mask_center_atoms_by_species,
    mask_center_atoms_by_id,
)
from test_utils import load_json_frame, BoxList, Box
import unittest
import numpy as np
import sys
import os
import faulthandler

rascal_reference_path = "reference_data"
inputs_path = os.path.join(rascal_reference_path, "inputs")
dump_path = os.path.join(rascal_reference_path, "tests_only")


def get_NL_reference(cutoff, cell, pbc, positions, atom_types):
    list_box = BoxList(
        cutoff,
        np.array(cell.T, order="C"),
        pbc.flatten(),
        np.array(positions.T, order="C"),
    )

    neighlist = [[] for it in range(len(atom_types))]
    neighpos = [[] for it in range(len(atom_types))]
    neighshift = [[] for it in range(len(atom_types))]
    neighdist = [[] for it in range(len(atom_types))]
    neightype = [[] for it in range(len(atom_types))]
    dirVec = [[] for it in range(len(atom_types))]
    for box in list_box.iter_box():
        for icenter in box.icenters:
            for jneigh, box_shift in box.iter_neigh_box():

                nnp = positions[:, jneigh] + np.dot(
                    box_shift.reshape((1, 3)), cell
                ).reshape((1, 3))
                rr = nnp - positions[:, icenter].reshape((1, 3))
                dist = np.linalg.norm(rr)

                neighpos[icenter].extend(nnp)
                neighlist[icenter].append(jneigh)
                neightype[icenter].append(atom_types[jneigh])
                neighdist[icenter].append(dist)

    return neighpos, neighlist, neightype, neighdist


def get_NL_strict_reference(cutoff, cell, pbc, positions, atom_types):
    list_box = BoxList(
        cutoff,
        np.array(cell.T, order="C"),
        pbc.flatten(),
        np.array(positions.T, order="C"),
    )

    neighlist = [[] for it in range(len(atom_types))]
    neighpos = [[] for it in range(len(atom_types))]
    neighshift = [[] for it in range(len(atom_types))]
    neighdist = [[] for it in range(len(atom_types))]
    neightype = [[] for it in range(len(atom_types))]
    dirVec = [[] for it in range(len(atom_types))]
    for box in list_box.iter_box():
        for icenter in box.icenters:
            for jneigh, box_shift in box.iter_neigh_box():

                nnp = positions[:, jneigh] + np.dot(
                    box_shift.reshape((1, 3)), cell
                ).reshape((1, 3))
                rr = nnp - positions[:, icenter].reshape((1, 3))
                dist = np.linalg.norm(rr)

                if cutoff > dist and dist > 0:
                    neighpos[icenter].extend(nnp)
                    neighlist[icenter].append(jneigh)
                    neighdist[icenter].append(dist)
                    neightype[icenter].append(atom_types[jneigh])
                    dirVec[icenter].append(rr / dist)
    return neighpos, neighlist, neightype, neighdist, dirVec


class TestStructureManagerCenters(unittest.TestCase):
    def setUp(self):
        """
        builds the test case. Test the order=1 structure manager implementation
        against a triclinic crystal.
        """

        fn = os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json")
        self.frame = load_json_frame(fn)
        self.structure = self.frame
        self.nl_options = [
            dict(name="centers", args={}),
        ]

    def test_manager_iteration(self):
        manager = get_neighbourlist(self.frame, self.nl_options)
        ii = 0
        for center in manager:
            self.assertTrue(ii == center.atom_tag)
            self.assertTrue(self.structure["atom_types"][ii] == center.atom_type)
            self.assertTrue(
                np.allclose(self.structure["positions"][:, ii], center.position)
            )
            ii += 1


class TestNL(unittest.TestCase):
    def setUp(self):
        """
        builds the test case. Test the order=1 structure manager implementation
        against a triclinic crystal.
        """

        fn = os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json")
        self.frame = load_json_frame(fn)
        self.structure = self.frame
        self.cutoff = 3.0
        self.nl_options = [
            dict(name="centers", args=dict()),
            dict(name="neighbourlist", args=dict(cutoff=self.cutoff)),
        ]
        self.pbcs = np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        ).astype(int)

    def test_manager_iteration_1(self):
        manager = get_neighbourlist(self.frame, self.nl_options)
        ii = 0
        for center in manager:
            self.assertTrue(ii == center.atom_tag)
            self.assertTrue(self.structure["atom_types"][ii] == center.atom_type)
            self.assertTrue(
                np.allclose(self.structure["positions"][:, ii], center.position)
            )
            ii += 1

    def test_manager_iteration_2(self):
        frame = self.frame
        structure = self.structure
        for pbc in self.pbcs:
            frame["pbc"] = pbc
            structure["pbc"] = pbc
            manager = get_neighbourlist(frame, self.nl_options)

            neighpos, neighlist, neightype, neighdist = get_NL_reference(
                self.cutoff, **structure
            )

            for center in manager:
                for neigh in center.pairs():
                    dist = np.linalg.norm(neigh.position - center.position)


class TestNLsanitation(unittest.TestCase):
    def setUp(self):
        """
        test that improper structures are sanitized by the
        """

        self.frames = []
        # methane.xyz is not inside the unit cell and is not periodic
        # dummy_structure.json is periodic but not inside the unitcell
        fns = ["methane.xyz", "dummy_structure.json"]
        for fn in fns:
            self.frames += [read(os.path.join(inputs_path, fn))]

        self.cutoff = 3.0
        self.nl_options = [
            dict(name="centers", args=dict()),
            dict(name="neighbourlist", args=dict(cutoff=self.cutoff)),
            dict(name="strict", args=dict(cutoff=self.cutoff)),
        ]

    def test_sanitation(self):
        for frame in self.frames:
            # it should not raise errors
            _ = AtomsList(frame, self.nl_options)


class TestNLStrict(unittest.TestCase):
    def setUp(self):
        """
        builds the test case. Test the order=1 structure manager implementation
        against a triclinic crystal.
        """

        fn = os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json")
        self.frame = load_json_frame(fn)
        self.structure = self.frame
        self.cutoff = 3.0

        self.nl_options = [
            dict(name="centers", args=dict()),
            dict(name="neighbourlist", args=dict(cutoff=self.cutoff)),
            dict(name="strict", args=dict(cutoff=self.cutoff)),
        ]

        self.pbcs = np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        ).astype(int)

    def test_manager_iteration_1(self):
        manager = get_neighbourlist(self.frame, self.nl_options)
        ii = 0
        for center in manager:
            self.assertTrue(ii == center.atom_tag)
            self.assertTrue(self.structure["atom_types"][ii] == center.atom_type)
            self.assertTrue(
                np.allclose(self.structure["positions"][:, ii], center.position)
            )
            ii += 1

    def test_manager_iteration_2(self):
        """
        Compare the distances and direction vector between the reference and
        librascal sctrict neighbourlist
        """
        frame = self.frame
        structure = self.structure
        for pbc in self.pbcs:
            frame["pbc"] = pbc
            structure["pbc"] = pbc
            manager = get_neighbourlist(frame, self.nl_options)

            (
                neighpos,
                neighlist,
                neightype,
                neighdist,
                neighdirVec,
            ) = get_NL_strict_reference(self.cutoff, **structure)
            for ii, center in enumerate(manager):
                dists, dirVecs = [], []
                for neigh in center.pairs():
                    dist = np.linalg.norm(neigh.position - center.position)
                    dists.append(dist)
                    dirVecs.append((neigh.position - center.position) / dist)

                ref_dists, dists = np.array(neighdist[ii]), np.array(dists)
                ref_dirVecs, dirVecs = (
                    np.array(neighdirVec[ii]).reshape((-1, 3)),
                    np.array(dirVecs),
                )
                # sort because the order is not the same
                ref_sort_ids, sort_ids = (
                    np.argsort(ref_dists),
                    np.argsort(dists),
                )


class TestNLExportInfo(unittest.TestCase):
    def setUp(self):
        """Test that informations are consistently exported for a water
        molecule.
        """
        self.neighbors_for_gradient = np.array(
            [
                [0, 0, 0, 8, 8],
                [0, 0, 2, 8, 1],
                [0, 0, 1, 8, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 0, 1, 8],
                [0, 2, 2, 1, 1],
                [0, 2, 0, 1, 8],
            ],
            dtype=np.int32,
        )
        self.atoms_for_predictions = np.array(
            [[0, 0, 8], [0, 1, 1], [0, 2, 1]], dtype=np.int32
        )
        self.distances = np.array(
            [0.0, 0.96856502, 0.96856502, 0.0, 0.96856502, 0.0, 0.96856502]
        )
        self.direction_vectors = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.78801008, -0.61566233],
                [0.0, 0.78801008, -0.61566233],
                [0.0, 0.0, 0.0],
                [0.0, -0.78801008, 0.61566233],
                [0.0, 0.0, 0.0],
                [0.0, 0.78801008, 0.61566233],
            ]
        )
        self.atoms = molecule("H2O")
        self.atoms.center(vacuum=1.0)
        interaction_cutoff = 1.5
        self.nl_options = [
            dict(name="centers", args=dict()),
            dict(name="neighbourlist", args=dict(cutoff=interaction_cutoff)),
            dict(name="centercontribution", args=dict()),
            dict(name="strict", args=dict(cutoff=interaction_cutoff)),
        ]

    def test_get_neighbors_for_gradient(self):
        manager = AtomsList(self.atoms, self.nl_options)
        ij = manager.get_gradients_info()
        self.assertTrue(np.allclose(ij, self.neighbors_for_gradient))

    def test_get_atoms_for_predictions(self):
        manager = AtomsList(self.atoms, self.nl_options)
        ii = manager.get_representation_info()
        self.assertTrue(np.allclose(ii, self.atoms_for_predictions))

    def test_get_direction_vectors(self):
        manager = AtomsList(self.atoms, self.nl_options)
        dir_vecs = manager.get_direction_vectors()
        self.assertTrue(np.allclose(dir_vecs, self.direction_vectors))

    def test_get_distances(self):
        manager = AtomsList(self.atoms, self.nl_options)
        distances = manager.get_distances()
        self.assertTrue(np.allclose(distances, self.distances))


class CenterSelectTest(unittest.TestCase):

    """Test the center-select Python interface

    Make sure it produces the right masks for a variety of inputs
    """

    def setUp(self):
        filename = "reference_data/inputs/small_molecule.json"
        self.frame = read(filename)
        self.natoms = len(self.frame)

    def get_mask(self):
        return self.frame.arrays["center_atoms_mask"]

    def check_mask(self, test_mask):
        self.assertTrue(np.all(self.get_mask() == test_mask))

    def test_mask_id_select(self):
        mask_center_atoms_by_id(self.frame, np.arange(5))
        test_mask = np.zeros((self.natoms,), dtype="bool")
        test_mask[:5] = True
        self.check_mask(test_mask)
        # Now try it with an existing mask
        mask_center_atoms_by_id(self.frame, np.arange(3, 7))
        test_mask[:7] = True
        self.check_mask(test_mask)
        mask_center_atoms_by_id(self.frame, id_blacklist=[0])
        test_mask[0] = False
        self.check_mask(test_mask)

    def test_mask_id_blacklist(self):
        mask_center_atoms_by_id(self.frame, id_blacklist=np.arange(5))
        test_mask = np.ones((self.natoms,), dtype="bool")
        test_mask[:5] = False
        self.check_mask(test_mask)
        mask_center_atoms_by_id(self.frame, id_blacklist=np.arange(3, 7))
        test_mask[:7] = False
        self.check_mask(test_mask)
        mask_center_atoms_by_id(self.frame, id_select=[0])
        test_mask[0] = True
        self.check_mask(test_mask)

    def test_mask_id_both(self):
        mask_center_atoms_by_id(
            self.frame, id_select=np.arange(7), id_blacklist=np.arange(3, 9)
        )
        test_mask = np.zeros((self.natoms,), dtype="bool")
        test_mask[:3] = True
        self.check_mask(test_mask)

    def test_mask_species_select(self):
        mask_center_atoms_by_species(self.frame, ["C", "H"])
        test_mask = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype="bool")
        self.check_mask(test_mask)
        mask_center_atoms_by_species(self.frame, ["N"])
        test_mask[[0, 2, 4, 6]] = True
        self.check_mask(test_mask)
        mask_center_atoms_by_species(self.frame, species_blacklist=["H"])
        test_mask[[9, 10]] = False
        self.check_mask(test_mask)

    def test_mask_species_blacklist(self):
        mask_center_atoms_by_species(self.frame, species_blacklist=["C", "H"])
        test_mask = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], dtype="bool")
        self.check_mask(test_mask)
        mask_center_atoms_by_species(self.frame, species_blacklist=["N"])
        test_mask[[0, 2, 4, 6]] = False
        self.check_mask(test_mask)
        mask_center_atoms_by_species(self.frame, species_select=["H"])
        test_mask[[9, 10]] = True
        self.check_mask(test_mask)

    def test_mask_species_both(self):
        mask_center_atoms_by_species(
            self.frame, species_select=["C", "N"], species_blacklist=["N", "H"]
        )
        test_mask = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="bool")
        self.check_mask(test_mask)

    def test_mask_species_numeric(self):
        # Can also select by atomic number
        mask_center_atoms_by_species(self.frame, species_select=[1, 6])
        test_mask = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype="bool")
        self.check_mask(test_mask)

    def test_mask_species_numeric_combined(self):
        mask_center_atoms_by_species(
            self.frame, species_select=["C", "N"], species_blacklist=[7, 1]
        )
        test_mask = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="bool")
        self.check_mask(test_mask)
        # And check that mixed symbol-numeric lists are disallowed
        with self.assertRaises(ValueError):
            mask_center_atoms_by_species(self.frame, species_select=["C", 1])
        with self.assertRaises(ValueError):
            mask_center_atoms_by_species(self.frame, species_blacklist=["C", 1])

    def test_mask_species_and_id(self):
        mask_center_atoms_by_species(self.frame, species_select=["C"])
        mask_center_atoms_by_id(self.frame, id_blacklist=np.arange(3))
        test_mask = np.zeros((self.natoms,), dtype="bool")
        test_mask[3] = True
        self.check_mask(test_mask)
