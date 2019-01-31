/**
 * file   test_representation_manager_base.hh
 *
 * @author Musil Felix <musil.felix@epfl.ch>
 * @author Max Veit <max.veit@epfl.ch>
 *
 * @date   14 September 2018
 *
 * @brief  test representation managers
 *
 * Copyright © 2018 Musil Felix, Max Veit, COSMO (EPFL), LAMMM (EPFL)
 *
 * rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef TEST_REPRESENTATION_H
#define TEST_REPRESENTATION_H

#include "tests.hh"
#include "test_structure.hh"
#include "test_adaptor.hh"
#include "representations/representation_manager_base.hh"
#include "representations/representation_manager_sorted_coulomb.hh"
#include "representations/representation_manager_spherical_expansion.hh"

#include "json_io.hh"
#include "rascal_utility.hh"

#include <tuple>

namespace rascal {


  struct MultipleStructureSortedCoulomb {
    MultipleStructureSortedCoulomb() {}
    ~MultipleStructureSortedCoulomb() = default;

    std::vector<std::string> filenames{
      "reference_data/CaCrP2O7_mvc-11955_symmetrized.json",
      "reference_data/simple_cubic_8.json",
      "reference_data/small_molecule.json"
      };

    std::vector<double> cutoffs{{1., 2., 3.}};

    std::list<json> hypers{
      {{"central_decay", 0.5},
      {"interaction_cutoff", 10.},
      {"interaction_decay", 0.5},
      {"size", 120},
      {"sorting_algorithm", "distance"}},
      {{"central_decay", 0.5},
      {"interaction_cutoff", 10.},
      {"interaction_decay", 0.5},
      {"size", 120},
      {"sorting_algorithm", "row_norm"}}
      };
  };

  struct MultipleStructureSphericalExpansion {
    MultipleStructureSphericalExpansion() = default;
    ~MultipleStructureSphericalExpansion() = default;

    std::vector<std::string> filenames {
      "reference_data/simple_cubic_8.json",
      "reference_data/small_molecule.json"
      //"reference_data/methane.json"
    };
    std::vector<double> cutoffs{{1, 2, 3}};

    std::list<json> hypers{{
      {"interaction_cutoff", 6.0},
      {"cutoff_smooth_width", 1.0},
      {"max_radial", 10},
      {"max_angular", 8},
      {"gaussian_sigma_type", "Constant"},
      {"gaussian_sigma_constant", 0.5}
    }};
  };

  struct SortedCoulombTestData {
    SortedCoulombTestData() {
      std::vector<std::uint8_t> ref_data_ubjson;
      internal::read_binary_file(this->ref_filename, ref_data_ubjson);
      ref_data = json::from_ubjson(ref_data_ubjson);
      filenames =
            ref_data.at("filenames").get<std::vector<std::string>>();
      cutoffs = ref_data.at("cutoffs").get<std::vector<double>>();
    }
    ~SortedCoulombTestData() = default;

    // name of the file containing the reference data. it has been generated
    // with the following python code:
    /*
    import ubjson
    from copy import copy
    import numpy as np
    import sys, os
    path = '/local/git/librascal/' # should be changed
    sys.path.insert(0, os.path.join(path, 'build/'))
    sys.path.insert(0, os.path.join(path, 'tests/'))
    from rascal.representation import SortedCoulombMatrix
    from test_utils import load_json_frame
    def json2ase(f):
        from ase import Atoms
        return Atoms(**{k:f[k] for k in ['positions','numbers','pbc','cell'] })

    cutoffs = [2,3,4,5]
    sorts = ['rownorm','distance']

    fns = [
        path+"tests/reference_data/CaCrP2O7_mvc-11955_symmetrized.json",
        path+"tests/reference_data/small_molecule.json"]
    fns_to_write = [
        "reference_data/CaCrP2O7_mvc-11955_symmetrized.json",
        "reference_data/small_molecule.json"
    ]
    data = dict(filenames=fns_to_write,cutoffs=cutoffs)
    hypers = dict(central_decay=-1,interaction_cutoff=-1,interaction_decay=-1,size=10)
    for sort in sorts:
        data[sort] = dict(feature_matrices=[],hypers=[])
        for fn in fns:
            for cutoff in cutoffs:
                rep = SortedCoulombMatrix(cutoff,sort=sort)
                frame = [json2ase(load_json_frame(fn))]
                features = rep.transform(frame)
                test = features.get_feature_matrix()
                data[sort]['feature_matrices'].append(test.tolist())
                hypers['size'] = rep.size
                data[sort]['hypers'].append(copy(hypers))
    with open(path+"tests/reference_data/sorted_coulomb_reference.ubjson",'wb') as f:
        ubjson.dump(data,f)
    */
    std::string ref_filename{"reference_data/sorted_coulomb_reference.ubjson" };
    std::vector<std::string> filenames{};
    std::vector<double> cutoffs{};
    json ref_data{};
  };

  template <class StructureManager,
            template <typename> class RepresentationManager,
            class BaseFixture>
  struct RepresentationFixture
      : MultipleStructureManagerStrictFixture<StructureManager, BaseFixture> {
    using Parent =
        MultipleStructureManagerStrictFixture<StructureManager, BaseFixture>;
    using Manager_t = typename Parent::Manager_t;
    using Representation_t = RepresentationManager<Manager_t>;

    RepresentationFixture() = default;
    ~RepresentationFixture() = default;

    std::list<Representation_t> representations{};
  };

/* ---------------------------------------------------------------------- */

} // RASCAL

#endif /* TEST_REPRESENTATION_H */
