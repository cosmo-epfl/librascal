/**
 * @file   test_adaptor_neighbour_list.cc
 *
 * @author Felix Musil
 *
 * @date   08 Apr 2021
 *
 * @brief tests the implementation of the adaptor kspace
 *
 * Copyright  2021 Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * Rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * Rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "test_adaptor.hh"
#include "test_structure.hh"

#include "rascal/structure_managers/atomic_structure.hh"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

namespace rascal {

  BOOST_AUTO_TEST_SUITE(kspace_adaptor_test);

  /* ---------------------------------------------------------------------- */
  /**
   * Test that the atom index from a neighbour matches the atom tag of the
   * ClusterRefKey returned by get_atom_j for a manager using as root
   * implementation `StructureManagerCenters`.
   */
  BOOST_FIXTURE_TEST_CASE(get_atom_j_test,
                          ManagerFixture<StructureManagerCenters>) {
    auto pair_manager{make_adapted_manager<AdaptorKspace>(manager)};

    constexpr bool verbose{false};

    for (auto atom : pair_manager) {
      for (auto neigh : atom.pairs()) {
        auto atom_j_index = pair_manager->get_atom_index(neigh.back());
        auto atom_j = neigh.get_atom_j();
        auto atom_j_tag = atom_j.get_atom_tag_list();
        if (verbose) {
          std::cout << "neigh: " << atom_j_index << " tag_j: " << atom_j_tag[0]
                    << std::endl;
        }

        BOOST_CHECK_EQUAL(atom_j_index, atom_j_tag[0]);
      }
    }
  }

  /* ---------------------------------------------------------------------- */

  using multiple_fixtures = boost::mpl::list<
      MultipleStructureFixture<MultipleStructureManagerKspaceFixture>>;

  /**
   * Check that it can be built and that the number of neighbors matches
   * the number of get_nb_clusters(2)
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_build_neighbour_multiple, Fix,
                                   multiple_fixtures, Fix) {
    auto & managers = Fix::managers;

    constexpr bool verbose{false};

    if (verbose) {
      std::cout << "Multiple structures " << std::endl;
    }

    for (auto & pair_manager : managers) {
      auto n_pairs{0};
      for (auto atom : pair_manager) {
        if (verbose) {
          std::cout << "atom " << atom.back() << std::endl;
        }
        for (auto neigh : atom.pairs()) {
          n_pairs++;
          if (verbose) {
            std::cout << "   complete neigh " << atom.back() << " "
                      << neigh.back() << " glob " << neigh.get_global_index()
                      << std::endl;
          }
        }
      }
      if (verbose) {
        std::cout << "Number of pairs " << n_pairs << std::endl;
      }
      BOOST_CHECK_EQUAL(n_pairs, pair_manager->get_nb_clusters(2));
    }
  }

  /**
   * Check that atom tags are properly ordered and that atom_j is consistent
   * with the neighbor.
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_idx_consistency, Fix, multiple_fixtures,
                                   Fix) {
    auto & managers = Fix::managers;

    for (auto & pair_manager : managers) {
      auto n_atoms{0};
      for (auto center : pair_manager) {
        int atom_tag = center.get_atom_tag();
        BOOST_CHECK_EQUAL(atom_tag, n_atoms);
        n_atoms++;
      }

      auto n_pairs{n_atoms};
      for (auto center : pair_manager) {
        for (auto neigh : center.pairs()) {
          int atom_tag = neigh.get_atom_tag();
          auto pos = neigh.get_position();
          int neigh_type = neigh.get_atom_type();

          BOOST_CHECK_EQUAL(atom_tag, n_pairs);
          n_pairs++;

          auto atom_j = neigh.get_atom_j();
          int atom_tag_j = atom_j.get_atom_tag();
          auto pos_j = pair_manager->get_position(atom_tag_j);
          int type_j = pair_manager->get_atom_type(atom_tag_j);
          auto diff{math::relative_error(pos, pos_j)};

          BOOST_CHECK(diff.maxCoeff() < math::DBL_FTOL);

          BOOST_CHECK_EQUAL(type_j, neigh_type);
        }
      }
    }
  }

  /**
   * Check that each centers has all the atoms in the unit cell as neighbors
   * except for itself.
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_index_completeness, Fix,
                                   multiple_fixtures, Fix) {
    auto & managers = Fix::managers;

    for (auto & pair_manager : managers) {
      std::set<int> center_ids{};
      for (auto center : pair_manager) {
        int atom_tag = center.get_atom_tag();
        center_ids.insert(atom_tag);
      }

      for (auto center : pair_manager) {
        std::set<int> neigh_ids{};
        int atom_tag = center.get_atom_tag();
        neigh_ids.insert(atom_tag);
        for (auto neigh : center.pairs()) {
          auto atom_j = neigh.get_atom_j();
          int atom_tag_j = atom_j.get_atom_tag();
          neigh_ids.insert(atom_tag_j);
        }
        BOOST_CHECK(neigh_ids == center_ids);
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace rascal
