#!/usr/bin/env python3

import unittest
import faulthandler

from python_structure_manager_test import (
    TestStructureManagerCenters,TestNL,TestNLStrict
    # TestStructureManagerCenters,TestNL,TestNLStrict
    )
from python_representation_manager_test import (
    TestSortedCoulombRepresentation
    )
from python_math_test import TestMath

class SimpleCheck(unittest.TestCase):
    def setUp(self):
        self.truth = True


    def test_simple_example(self):
        self.assertTrue(self.truth)


if __name__ == '__main__':
    faulthandler.enable()
    
    unittest.main()

