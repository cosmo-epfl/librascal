/**
 * file   test_properties.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief  tests for cluster-related properties
 *
 * @section LICENSE
 *
 * Copyright  2018 Till Junge, COSMO (EPFL), LAMMM (EPFL)
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

#include "test_properties.hh"

namespace rascal {
  // TODO(felix) test dynamic sized Property
  BOOST_AUTO_TEST_SUITE(Property_tests);
    
  using atom_property_fixtures_with_ghosts = OrderOnePropertyBoostList::type_with_ghosts;
  using atom_property_fixtures_without_ghosts = OrderOnePropertyBoostList::type_without_ghosts;
  using atom_property_fixtures = OrderOnePropertyBoostList::type;
  using pair_property_fixtures = OrderTwoPropertyBoostList::type;
  using triple_property_fixtures = OrderThreePropertyBoostList::type;

  /* ---------------------------------------------------------------------- */

  BOOST_FIXTURE_TEST_CASE(atom_order_one_constructor_tests, AtomPropertyFixture<StructureManagerCentersStackFixture>) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << manager->get_name();
      std::cout << " starts now." << std::endl;
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(atom_constructor_tests, Fix, atom_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(pair_constructor_tests, Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(triple_constructor_tests, Fix, triple_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE(fill_atom_property_order_one_test, AtomPropertyFixture<StructureManagerCentersStackFixture>) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << manager->get_name();
      std::cout << ", manager size " << manager->get_size();
      std::cout << ", manager size with ghosts " << manager->get_size_with_ghosts();
      std::cout << " starts now." << std::endl;
    }

    atom_property.resize(
        manager->get_consider_ghost_neighbours());
    dynamic_property2.resize(
        manager->get_consider_ghost_neighbours());
    if (verbose) {
      std::cout << ">> atom_property size ";
      std::cout << atom_property.size();
      std::cout << std::endl;
    }
    for (auto atom : manager) {
      if (verbose) {
        std::cout << ">> Atom with tag " << atom.get_atom_tag(); 
        std::cout << std::endl;
      }
      atom_property[atom] = atom.get_position();
      dynamic_property2[atom] = atom.get_position();
    }

    for (auto atom : manager) {
      auto error = (atom_property[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error, tol * 100);
      auto error_dynamic = (dynamic_property2[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error_dynamic, tol * 100);
    }

    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fill_atom_property_test, Fix, atom_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << ", manager size " << Fix::manager->get_size();
      std::cout << ", manager size with ghosts " << Fix::manager->get_size_with_ghosts();
      std::cout << " starts now." << std::endl;
    }

    Fix::atom_property.resize(
        Fix::manager->get_consider_ghost_neighbours());
    Fix::dynamic_property2.resize(
        Fix::manager->get_consider_ghost_neighbours());
    if (verbose) {
      std::cout << ">> atom_property size ";
      std::cout << Fix::atom_property.size();
      std::cout << std::endl;
    }
    for (auto atom : Fix::manager) {
      if (verbose) {
        std::cout << ">> Atom with tag " << atom.get_atom_tag(); 
        std::cout << std::endl;
      }
      Fix::atom_property[atom] = atom.get_position();
      Fix::dynamic_property2[atom] = atom.get_position();
    }

    for (auto atom : Fix::manager) {
      auto error = (Fix::atom_property[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error, tol * 100);
      auto error_dynamic = (Fix::dynamic_property2[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error_dynamic, tol * 100);
    }

    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }
  /* ---------------------------------------------------------------------- */
  /*
   * checks if the properties associated with atoms and pairs can be filled
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fill_pair_property_test, Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << ", manager size " << Fix::manager->get_size();
      std::cout << ", manager size with ghosts " << Fix::manager->get_size_with_ghosts();
      std::cout << " starts now." << std::endl;
    }

    Fix::atom_property.resize(
        Fix::manager->get_consider_ghost_neighbours());
    Fix::pair_property.resize();
    if (verbose) {
      std::cout << ">> atom_property size ";
      std::cout << Fix::atom_property.size();
      std::cout << std::endl;
    }
    int pair_property_counter{};
    for (auto atom : Fix::manager->with_ghosts()) {
      Fix::atom_property[atom] = atom.get_position();
      for (auto pair : atom) {
        Fix::pair_property[pair] = ++pair_property_counter;
      }
    }

    pair_property_counter = 0;
    for (auto atom : Fix::manager->with_ghosts()) {
      auto error = (Fix::atom_property[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error, tol * 100);
      for (auto pair : atom) {
        BOOST_CHECK_EQUAL(Fix::pair_property[pair], ++pair_property_counter);
      }
    }

    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fill_triple_property_test, Fix, triple_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " and consider_ghost_neighbours=";
      std::cout << Fix::manager->get_consider_ghost_neighbours();
      std::cout << ", manager size " << Fix::manager->get_size();
      std::cout << ", manager size with ghosts " << Fix::manager->get_size_with_ghosts();
      std::cout << " starts now." << std::endl;
    }

    Fix::atom_property.resize(Fix::manager->get_consider_ghost_neighbours());
    Fix::pair_property.resize();
    Fix::triple_property.resize();
    if (verbose) {
      std::cout << ">> atom_property size ";
      std::cout << Fix::atom_property.size();
      std::cout << ", pair_property size ";
      std::cout << Fix::pair_property.size();
      std::cout << ", triple_property size ";
      std::cout << Fix::triple_property.size();
      std::cout << std::endl;
    }
    int pair_property_counter{};
    int triple_property_counter{};
    for (auto atom : Fix::manager->with_ghosts()) {
      Fix::atom_property[atom] = atom.get_position();
      for (auto pair : atom) {
        Fix::pair_property[pair] = ++pair_property_counter;
        for (auto triple : pair) {
          Fix::triple_property[triple] = ++triple_property_counter;
        }
      }
    }

    pair_property_counter = 0;
    triple_property_counter = 0;
    for (auto atom : Fix::manager->with_ghosts()) {
      auto error = (Fix::atom_property[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error, tol * 100);
      for (auto pair : atom) {
        BOOST_CHECK_EQUAL(Fix::pair_property[pair], ++pair_property_counter);
        for (auto triple : pair) {
          BOOST_CHECK_EQUAL(Fix::triple_property[triple], ++triple_property_counter);
        }
      }
    }

    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }


  /* ---------------------------------------------------------------------- */
  /* If consider_ghost_neighbours is true the atoms index should 
   * correspond to the cluster index of order 1 when StructureManagerCenters is
   * used as  root implementation and no filtering on order 1 has been done.
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(atom_property_fixtures_tests, Fix, atom_property_fixtures_with_ghosts, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
    }
    size_t cluster_index{0};
    for (auto atom : Fix::manager) {
      if (verbose) {
        std::cout << ">> Atom index " << atom.get_atom_tag();
        std::cout << ", ClusterIndex should be " << cluster_index;
        std::cout << " and is ";
        std::cout << Fix::manager->get_cluster_index(atom.get_atom_tag());
        std::cout << "." << std::endl;
      }
      BOOST_CHECK_EQUAL(Fix::manager->get_cluster_index(atom.get_atom_tag()), cluster_index);
      cluster_index++;
    }
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }
  
  /* ---------------------------------------------------------------------- */
  /* Access of atom property with pair. 
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(atom_property_access_with_pair_tests, Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " and consider_ghost_neighbours=";
      std::cout << Fix::manager->get_consider_ghost_neighbours();
      std::cout << ", manager size " << Fix::manager->get_size();
      std::cout << ", manager size with ghosts " << Fix::manager->get_size_with_ghosts();
      std::cout << " starts now." << std::endl;
    }
    // initalize the positions
    Fix::scalar_atom_property.resize(false);
    if (verbose) {
      std::cout << ">> Property for consider_ghost_atoms=false resized to size ";
      std::cout << Fix::scalar_atom_property.size();
      std::cout << std::endl;
    }
    //Fix::atom_dynamic_property.resize();
    for (auto atom : Fix::manager) {
      if (verbose) {
        std::cout << ">> Property for atom with tag ";
        std::cout << atom.get_atom_tag();
        std::cout << " is initialized.";
        std::cout << std::endl;
      }
      Fix::scalar_atom_property[atom] = 0;
      //Fix::atom_dynamic_property[atom] = 0;
    }
    std::vector<size_t> counter{};
    size_t nb_central_atoms = Fix::manager->get_size();
    counter.reserve(nb_central_atoms);
    for (size_t i{0}; i<counter.capacity(); i++) {
      counter.push_back(0);
    }
    if (verbose) {
      std::cout << ">> Counters initialized with size ";
      std::cout <<  counter.size() << std::endl;
    }
    // add the position to the atom and count how often this happens
    for (auto atom : Fix::manager->with_ghosts()) {
      for (auto pair : atom) {
        if (verbose) {
          std::cout << ">> Atom with tag ";
          std::cout << pair.get_internal_neighbour_atom_tag();
          std::cout << " corresponds to central atom in cell with atom index ";
          std::cout << Fix::manager->get_cluster_index(atom.get_atom_tag());
          std::cout << std::endl;
        }
        Fix::scalar_atom_property[pair]++;
        // TODO(alex) does not work, dynamic properties do not give scalar
        // values, I want to move the writing of a
        // test for dynamic properties to a point, when we actual use them
        //Fix::atom_dynamic_property[pair]++;
        counter.at(Fix::manager->get_cluster_index(pair.get_internal_neighbour_atom_tag()))++;
      }
    }
    for (auto atom : Fix::manager) {
      size_t counter_at_cluster_index = counter.at(Fix::manager->get_cluster_index(atom.get_atom_tag()));
      BOOST_CHECK_EQUAL(Fix::scalar_atom_property[atom],
          counter_at_cluster_index);
      //BOOST_CHECK_EQUAL(Fix::atom_dynamic_property[atom],
      //    counter_at_cluster_index);
    }
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }
  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(order_three_constructor_tests, Fix, triple_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
      std::cout << " finished." << std::endl;
    }
  }
  /* Access of atom property with triple. 
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(atom_property_access_with_triple_tests, Fix, triple_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " and consider_ghost_neighbours=";
      std::cout << Fix::manager->get_consider_ghost_neighbours();
      std::cout << ", manager size " << Fix::manager->get_size();
      std::cout << ", manager size with ghosts " << Fix::manager->get_size_with_ghosts();
      std::cout << " starts now." << std::endl;
    }
    Fix::scalar_atom_property.resize(false);
    // initalize the positions
    if (verbose) {
      std::cout << ">> atom_property resized to size ";
      std::cout << Fix::atom_property.size();
      std::cout << std::endl;
    }
    for (auto atom : Fix::manager) {
      if (verbose) {
        std::cout << ">> Atom tag " << atom.get_atom_tag(); 
        std::cout << std::endl;
      }
      Fix::scalar_atom_property[atom] = 0;
    }
    std::vector<size_t> counter{};
    size_t nb_central_atoms = Fix::manager->get_size();
    counter.reserve(nb_central_atoms);
    for (size_t i{0}; i<counter.capacity(); i++) {
      counter.push_back(0);
    }
    
    // add the position to the atom and count how often this happens
    for (auto atom : Fix::manager->with_ghosts()) {
      for (auto pair : atom) {
        for (auto triple : pair) {
          if (verbose) {
            std::cout << ">> Atom with tag " << triple.get_internal_neighbour_atom_tag(); 
            std::cout << " and cluster index " << Fix::manager->get_cluster_index(triple.get_internal_neighbour_atom_tag()); 
            std::cout << std::endl;
          }
          Fix::scalar_atom_property[triple]++;
          counter.at(Fix::manager->get_cluster_index(triple.get_internal_neighbour_atom_tag()))++;
        }
      }
    }
    for (auto atom : Fix::manager) {
      if (verbose) {
        std::cout << ">> atom.get_atom_tag() is "
                  << atom.get_atom_tag() << std::endl;
        std::cout << ">> manager->get_cluster_index(atom.get_atom_tag()) is "
                  << Fix::manager->get_cluster_index(atom.get_atom_tag()) << std::endl;
        std::cout << ">> scalar_atom_property[atom] is "
                  << Fix::scalar_atom_property[atom] << std::endl;
        std::cout << ">> counter.at(manager->get_cluster_index(atom.get_atom_tag())) is "
                  << counter.at(Fix::manager->get_cluster_index(atom.get_atom_tag())) << std::endl;
      }
      BOOST_CHECK_EQUAL(Fix::scalar_atom_property[atom],
          counter.at(Fix::manager->get_cluster_index(atom.get_atom_tag())));
    }
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }
  /* ---------------------------------------------------------------------- */
  /* The access of an order one property with the atom itself
   * and the pair with the atom as neighbour should be the same.
   */ 
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fill_test_simple_order_one_property,
                          Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
    }
    Fix::pair_property.resize();
    Fix::atom_property.resize();
    for (auto atom : Fix::manager) {
      Fix::atom_property[atom] = atom.get_position();
    }

    for (auto atom : Fix::manager) {
      for (auto atom2 : Fix::manager) {
        for (auto pair : atom) {
          if (atom.back() == pair.back()) {
            auto error = (Fix::atom_property[atom] - Fix::atom_property[pair]).norm();
            BOOST_CHECK_LE(error, tol * 100);
          }
        }
      }
    }
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * test, if metadata can be assigned to properties
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(meta_data_test,
      Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
    }
    auto atom_metadata = Fix::atom_property.get_metadata();
    auto dynamic_metadata = Fix::dynamic_property.get_metadata();
    auto dynamic_metadata2 = Fix::dynamic_property2.get_metadata();

    BOOST_CHECK_EQUAL(atom_metadata, Fix::atom_property_metadata);
    BOOST_CHECK_EQUAL(dynamic_metadata, Fix::dynamic_property_metadata);
    BOOST_CHECK_EQUAL(dynamic_metadata2, Fix::dynamic_property2_metadata);
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  /*
   * test filling statically and dynamically sized properties with actual data
   * and comparing if retrieval of those is consistent with the data that was
   * put in
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fill_test_complex,
      Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
    }
    Fix::pair_property.resize();
    Fix::atom_property.resize();
    Fix::dynamic_property.resize();
    Fix::dynamic_property2.resize();

    BOOST_CHECK_THROW(
        Fix::AtomVectorProperty_t ::check_compatibility(Fix::dynamic_property),
        std::runtime_error);

    BOOST_CHECK_NO_THROW(
        Fix::AtomVectorProperty_t ::check_compatibility(Fix::atom_property));

    int pair_property_counter{};
    size_t counter{};
    for (auto atom : Fix::manager) {
      Fix::atom_property[atom] = atom.get_position();
      Fix::dynamic_property2[atom] = atom.get_position();

      Fix::dynamic_property[atom] << counter++, counter, counter;
      for (auto pair : atom) {
        Fix::pair_property[pair] = ++pair_property_counter;
      }
    }

    auto & FakeSizedProperty{
      Fix::AtomVectorProperty_t::check_compatibility(Fix::dynamic_property2)};

    pair_property_counter = 0;
    counter = 0;
    for (auto atom : Fix::manager) {
      auto error = (Fix::atom_property[atom] - atom.get_position()).norm();
      BOOST_CHECK_LE(error, tol * 100);
      Eigen::Matrix<size_t, Fix::DynSize(), Eigen::Dynamic> tmp(Fix::DynSize(), 1);
      tmp << counter++, counter, counter;

      auto ierror{(tmp - Fix::dynamic_property[atom]).norm()};
      BOOST_CHECK_EQUAL(ierror, 0);

      error = (Fix::atom_property[atom] - Fix::dynamic_property2[atom]).norm();
      BOOST_CHECK_LE(error, tol * 100);
      error = (Fix::atom_property[atom] - FakeSizedProperty[atom]).norm();
      BOOST_CHECK_LE(error, tol * 100);
      for (auto pair : atom) {
        BOOST_CHECK_EQUAL(Fix::pair_property[pair], ++pair_property_counter);
      }
    }
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  /*
   * test for retrieval of information from property: is it the same that was
   * put in?
   */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(compute_distances,
      Fix, pair_property_fixtures, Fix) {
    bool verbose{false};
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " starts now." << std::endl;
    }
    Fix::pair_property.resize();

    for (auto atom : Fix::manager) {
      for (auto pair : atom) {
        Fix::pair_property[pair] =
            (atom.get_position() - pair.get_position()).norm();
      }
    }

    for (auto atom : Fix::manager) {
      for (auto pair : atom) {
        auto dist{(atom.get_position() - pair.get_position()).norm()};
        auto error{Fix::pair_property[pair] - dist};
        BOOST_CHECK_LE(error, tol / 100);
      }
    }
    if (verbose) {
      std::cout << ">> Test for manager ";
      std::cout << Fix::manager->get_name();
      std::cout << " finished." << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

  /* ---------------------------------------------------------------------- */

  /*
   * A fixture for testing partially sparse properties.
   * TODO(felix) use MultipleStructureManagerCentersFixture instead of NL
   */
  struct BlockSparsePropertyFixture
      : public MultipleStructureFixture<MultipleStructureManagerNLFixture> {
    using Parent = MultipleStructureFixture<MultipleStructureManagerNLFixture>;
    using ManagerTypeList_t = typename Parent::ManagerTypeHolder_t::type_list;

    using Key_t = std::vector<int>;
    using BlockSparseProperty_t = BlockSparseProperty<double, 1, 0, Manager_t>;
    using Dense_t = typename BlockSparseProperty_t::Dense_t;
    using InputData_t = typename BlockSparseProperty_t::InputData_t;
    using test_data_t = std::vector<InputData_t>;

    constexpr static Dim_t DynSize() { return 3; }

    std::string sparse_features_desc{"some atom centered sparse features"};

    BlockSparsePropertyFixture() : Parent{} {
      std::random_device rd;
      std::mt19937 gen{rd()};
      auto size_dist{std::uniform_int_distribution<size_t>(1, 10)};
      auto key_dist{std::uniform_int_distribution<int>(1, 100)};
      // size_t i_manager{0};
      for (auto & manager : managers) {
        sparse_features.emplace_back(*manager, sparse_features_desc);
        this->keys_list.emplace_back();
        test_data_t test_data{};
        for (auto atom : manager) {
          // set up random unique keys
          auto set_max_size{size_dist(gen)};
          std::set<Key_t> keys{};
          for (size_t ii{0}; ii < set_max_size; ii++) {
            keys.emplace(key_dist(gen));
          }

          // set up the data to fill the property later
          InputData_t datas{};
          for (auto & key : keys) {
            auto data = Dense_t::Random(21, 8);
            datas.emplace(std::make_pair(key, data));
          }
          this->keys_list.back().push_back(keys);
          test_data.push_back(datas);
        }
        this->test_datas.push_back(test_data);
      }
    }

    std::vector<std::vector<std::set<Key_t>>> keys_list{};
    std::vector<test_data_t> test_datas{};
    std::vector<BlockSparseProperty_t> sparse_features{};
  };

  BOOST_AUTO_TEST_SUITE(Property_partially_sparse_tests);

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(constructor_test, BlockSparsePropertyFixture) {}

  /* ---------------------------------------------------------------------- */
  /*
   * checks if the partially sparse properties associated with centers can be
   * filled and that the data can be accessed consistently.
   */
  BOOST_FIXTURE_TEST_CASE(fill_test_simple, BlockSparsePropertyFixture) {
    bool verbose{false};
    // fill the property structures
    auto i_manager{0};
    for (auto & manager : managers) {
      auto i_center{0};
      sparse_features[i_manager].set_shape(21, 8);
      for (auto center : manager) {
        sparse_features[i_manager].push_back(test_datas[i_manager][i_center]);
        i_center++;
      }
      i_manager++;
    }

    i_manager = 0;
    for (auto & manager : managers) {
      if (verbose)
        std::cout << "manager: " << i_manager << std::endl;
      auto i_center{0};
      for (auto center : manager) {
        if (verbose)
          std::cout << "center: " << i_center << std::endl;

        auto data = sparse_features[i_manager].get_dense_row(center);
        size_t key_id{0};
        double error1{0};
        for (auto & key : sparse_features[i_manager].get_keys(center)) {
          auto & value{test_datas[i_manager][i_center][key]};
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> test_data(
              value.data(), value.size());
          error1 += (data.col(key_id) - test_data).squaredNorm();
          key_id += 1;
        }
        if (verbose)
          std::cout << "error1: " << error1 << std::endl;
        BOOST_CHECK_LE(std::sqrt(error1), tol * 100);
        for (auto & key : keys_list[i_manager][i_center]) {
          auto error2 = (sparse_features[i_manager](center, key) -
                         test_datas[i_manager][i_center][key])
                            .norm();
          if (verbose)
            std::cout << "error2: " << error2 << std::endl;
          BOOST_CHECK_LE(error2, tol * 100);
        }
        i_center++;
      }
      i_manager++;
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * test, if metadata can be assigned to properties
   */
  BOOST_FIXTURE_TEST_CASE(meta_data_test, BlockSparsePropertyFixture) {
    for (auto & sparse_feature : sparse_features) {
      auto sparse_feature_metadata = sparse_feature.get_metadata();
      BOOST_CHECK_EQUAL(sparse_feature_metadata, sparse_features_desc);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace rascal
