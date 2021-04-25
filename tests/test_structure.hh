/**
 * @file test_structure.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief  Common headers for tests related to `StructureManager`
 *
 * @section LICENSE
 *
 * Copyright  2018 Till Junge, Markus Stricker, Felix Musil, COSMO (EPFL),
 * LAMMM (EPFL)
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

#ifndef TESTS_TEST_STRUCTURE_HH_
#define TESTS_TEST_STRUCTURE_HH_

#include "rascal/structure_managers/adaptor_center_contribution.hh"
#include "rascal/structure_managers/adaptor_half_neighbour_list.hh"
#include "rascal/structure_managers/adaptor_increase_maxorder.hh"
#include "rascal/structure_managers/adaptor_kspace.hh"
#include "rascal/structure_managers/adaptor_neighbour_list.hh"
#include "rascal/structure_managers/adaptor_strict.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/structure_manager.hh"
#include "rascal/structure_managers/structure_manager_base.hh"
#include "rascal/structure_managers/structure_manager_centers.hh"
#include "rascal/structure_managers/structure_manager_lammps.hh"
#include "rascal/utils/utils.hh"

namespace rascal {

  /**
   * This file generates fixtures for all the classes in structure_managers
   * that need to be tested. Fixtures should be as compatible as possible in
   * terms of interface, so that the tests can be easily templated.  This is a
   * list of conventions that are expected in tests: - If a fixture generates an
   * object that should be accessible with a structure_manager interface, it has
   * to be called "manager"
   */

  /* ---------------------------------------------------------------------- */
  /**
   * Most basic fixture. This is only to guarantee, that the lowest Layer
   * manager, which is initialized with existing data and not `adapted` is
   * always accessible with the variable ``manager``.
   */
  template <class ManagerImplementation>
  struct ManagerFixture {
    using ManagerPtr_t = std::shared_ptr<ManagerImplementation>;
    ManagerFixture() {}   // ctor
    ~ManagerFixture() {}  // dtor

    ManagerPtr_t manager{make_structure_manager<ManagerImplementation>()};
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Basic fixture for using two manager to compare things. This is to
   * guarantee, that the managers, built with existing data and not adapted is
   * always accessible with the variable ``manager_1`` and ``manager_2``. Both
   * managers are of the same class.
   */
  template <class ManagerImplementation>
  struct ManagerFixtureTwo {
    using ManagerPtr_t = std::shared_ptr<ManagerImplementation>;
    ManagerFixtureTwo() {}   // ctor
    ~ManagerFixtureTwo() {}  // dtor

    ManagerPtr_t manager_1{make_structure_manager<ManagerImplementation>()};
    ManagerPtr_t manager_2{make_structure_manager<ManagerImplementation>()};
  };

  /* ---------------------------------------------------------------------- */

  /**
   * general case of a manager fixture, which reads the structure information
   * from a file, atomic structure contains 9 atoms in a very simple cubic unit
   * cell, no periodicity, it inherits publicly from the ManagerFixture, which
   * provides access to the variable ``manager``.
   */
  template <class ManagerImplementation>
  struct ManagerFixtureFile : public ManagerFixture<ManagerImplementation> {
    // initialize manager variable
    ManagerFixtureFile()
        : ManagerFixture<ManagerImplementation>{}, cutoff{3.5},
          filename{"reference_data/inputs/simple_cubic_9.json"}
    // initialize current fixture
    {
      this->manager->update(filename);
    }

    ~ManagerFixtureFile() {}

    // additional variables for current fixture
    double cutoff;
    std::string filename;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Fixture providing two managers ``manager_1`` and ``manager_2``, which both
   * have a hexagonal lattice, but with different unit cells (vectors), each
   * with two atoms. It is used to ensure that the neighbour list algorithm is
   * robust with respect to the unit cell. Same crystal structure and cutoff
   * should yield the same number of neighbours.
   */
  struct ManagerFixtureTwoHcp
      : public ManagerFixtureTwo<StructureManagerCenters> {
    ManagerFixtureTwoHcp()
        : ManagerFixtureTwo<StructureManagerCenters>{}, pbc{{true, true, true}},
          cell_1(3, 3), cell_2(3, 3), positions_1(3, 2), positions_2(3, 2),
          atom_types(2), cutoff{1.01} {
      /**
       * hcp crystal with lattice parameter a = 1, c = sqrt(8/3), defined in two
       * unit cells: basal and prismatic 1. The neighbourlist is built with the
       * same cutoff. The test checks, if all atoms have the same number of
       * neighbours.
       */
      auto a{1.};
      auto c{std::sqrt(8. / 3.)};
      // clang-format off
      // basal cell
      cell_1 << a,                -0.5 * a, 0.,
                0., std::sqrt(3.) / 2. * a, 0.,
                0., 0.,                      c;


      // prism cell
      cell_2 << a,  0.,                0.5 * a,
                0., c ,                     0.,
                0., 0., std::sqrt(3.) / 2. * a;
      // clang-format on
      auto h{std::sqrt(3.) / 2. * a};
      auto r{std::sqrt(3.) / 6. * a};

      Eigen::Vector3d p_1(0, h - r, c / 2.);

      positions_1 << 0.0, p_1[0], 0.0, p_1[1], 0.0, p_1[2];

      Eigen::Vector3d p_2(a / 2., c / 2., r);

      positions_2 << 0.0, p_2[0], 0.0, p_2[1], 0.0, p_2[2];

      atom_types << 1, 1;

      using PBC_t = Eigen::Map<Eigen::Matrix<int, 3, 1>>;

      manager_1->update(positions_1, atom_types, cell_1, PBC_t{pbc.data()});

      manager_2->update(positions_2, atom_types, cell_2, PBC_t{pbc.data()});
    }  // namespace rascal

    ~ManagerFixtureTwoHcp() {}

    std::array<int, 3> pbc;
    Eigen::MatrixXd cell_1;
    Eigen::MatrixXd cell_2;
    Eigen::MatrixXd positions_1;
    Eigen::MatrixXd positions_2;
    // same number of atoms, therefore only one necessary
    Eigen::VectorXi atom_types;

    double cutoff;

    const int natoms{2};
  };  // namespace rascal

  /* ---------------------------------------------------------------------- */
  /**
   * Fixture which provides two managers ``manager_1`` and ``manager_2``, which
   * both have a face centered cubic lattice, but with different unit cells
   * (vectors) with one and four atoms. It is another test to ensure that the
   * neighbour list algorithm is robust with respect to the unit cell.
   */
  struct ManagerFixtureTwoFcc
      : public ManagerFixtureTwo<StructureManagerCenters> {
    ManagerFixtureTwoFcc()
        : ManagerFixtureTwo<StructureManagerCenters>{}, pbc{{true, true, true}},
          cell_1(3, 3), cell_2(3, 3), positions_1(3, 1), positions_2(3, 4),
          atom_types_1(1),
          atom_types_2(4), cutoff{0.7},  // starting with zero neighbours
          natoms_1{1}, natoms_2{4} {
      /*
       * fcc unit cells: first cell consists of only one atom, which is
       * repeated, second cell is the conventional 4 atoms. This test checks, if
       * the found number of neighbours with increasing cutoff is the same for
       * the atom at position (0, 0, 0).
       */
      auto a{1.};
      // clang-format off
      cell_1 <<  a , 0.5 * a, 0.5 * a,
                 0., 0.5 * a, 0.     ,
                 0., 0.,      0.5 * a;

      cell_2 << a,  0., 0.,
                0., a,  0.,
                0., 0., a;
      // clang-format on
      positions_1 << 0., 0., 0.;

      auto p_2 = 0.5 * cell_2.col(0) + 0.5 * cell_2.col(1);
      auto p_3 = 0.5 * cell_2.col(0) + 0.5 * cell_2.col(2);
      auto p_4 = 0.5 * cell_2.col(1) + 0.5 * cell_2.col(2);
      // clang-format off
      positions_2 <<    0.0, p_2[0], p_3[0],
                     p_4[0],    0.0, p_2[1],
                     p_3[1], p_4[1],    0.0,
                     p_2[2], p_3[2], p_4[2];
      // clang-format on
      atom_types_1 << 1;
      atom_types_2 << 1, 1, 1, 1;

      using PBC_t = Eigen::Map<Eigen::Matrix<int, 3, 1>>;

      manager_1->update(positions_1, atom_types_1, cell_1, PBC_t{pbc.data()});

      manager_2->update(positions_2, atom_types_2, cell_2, PBC_t{pbc.data()});
    }

    ~ManagerFixtureTwoFcc() {}

    std::array<int, 3> pbc;
    Eigen::MatrixXd cell_1;
    Eigen::MatrixXd cell_2;
    Eigen::MatrixXd positions_1;
    Eigen::MatrixXd positions_2;
    Eigen::VectorXi atom_types_1;
    Eigen::VectorXi atom_types_2;

    double cutoff;

    const int natoms_1;
    const int natoms_2;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Specialisation for the ManagerFixture for usage with the structure manager
   * with the Lammps interface. Data structures for a 3-atom structure are built
   * here and the manager is initialised.
   */
  // TODO(markus): as of Nov 2018, this interface is outdated with respect to
  // lammps.
  template <>
  struct ManagerFixture<StructureManagerLammps> {
    using Manager_t = StructureManagerLammps;
    constexpr static int nb{3};
    constexpr static int dim{3};
    using ptr_t = double **;

    ManagerFixture()
        : firstneigh{new int *[6]}, x{new double *[nb]}, f{new double *[nb]},
          vatom{new double *[nb]}, manager{
                                       make_structure_manager<Manager_t>()} {
      manager->update(inum, tot_num, ilist, numneigh,
                      static_cast<int **>(firstneigh), ptr_t(x), ptr_t(f), type,
                      eatom, static_cast<double **>(vatom));
      firstneigh[0] = new int[2];
      firstneigh[0][0] = 3;
      firstneigh[0][1] = 5;
      firstneigh[3] = new int;
      firstneigh[3][0] = 0;
      firstneigh[5] = new int;
      firstneigh[5][0] = 0;

      for (int i{0}; i < nb; ++i) {
        x[i] = new double[dim];
        f[i] = new double[dim];
        for (int j{0}; j < dim; ++j) {
          x[i][j] = tx[i][j];
          f[i][j] = tf[i][j];
        }
      }
    }

    ManagerFixture(ManagerFixture &) = delete;
    ManagerFixture & operator=(const ManagerFixture &) = delete;
    ~ManagerFixture() {
      delete[] firstneigh[0];
      delete firstneigh[3];
      delete firstneigh[5];

      delete[] firstneigh;
      delete[] vatom;
      for (int i{0}; i < nb; ++i) {
        delete[] x[i];
        delete[] f[i];
      }
      delete[] x;
      delete[] f;
    }
    double tx[nb][nb] = {{0, 0, 0}, {1, 1, 0}, {1, 0, 0}};
    double tf[nb][nb] = {{1, 1, 0}, {-1, 0, 0}, {0, -1, 0}};

    int inum{nb};
    int tot_num{nb};  // includes ghosts
    // #BUG8486@(all) how does lammps saves the types for non contiguous atom
    // tag lists? how would the type list look like in this case? because you
    // cannot access it simply with the atom tags, e.g. you need to map
    // 0->0,3->1 and 5->2. I am currently doing this, but how lammps actual do
    // it? Also we cannot except negative atom tags at the moment and I would
    // not do it, because lammps does not have negative ones, we do not have the
    // case of using negative ones, why the extra effort?
    int ilist[nb]{0, 3,
                  5};  // TODO(alex): make ilist non-contiguous, eg {1, 28, 12}
    int numneigh[nb]{2, 1, 1};
    int ** firstneigh;
    double ** x;
    double ** f;
    int type[nb]{1, 2, -9};
    double eatom[nb]{2, 1, 1};
    double ** vatom;
    std::shared_ptr<Manager_t> manager;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * fixture for providing a neighbour list from a simple manager, which is read
   * from a JSON file. This fixture provides iteration over the variable
   * ``pair_manager``
   */
  template <class ManagerImplementation>
  struct PairFixtureFile {
    using Manager_t = ManagerImplementation;

    static_assert(Manager_t::traits::MaxOrder == 1,
                  "Lower layer manager has MaxOrder needs MaxOrder = 1");

    using PairManager_t = AdaptorNeighbourList<Manager_t>;

    PairFixtureFile()
        : pair_manager{make_adapted_manager<AdaptorNeighbourList>(
              this->fixture.manager, this->fixture.cutoff)} {
      this->pair_manager->update();
    }

    ~PairFixtureFile() {}

    ManagerFixtureFile<Manager_t> fixture{};
    std::shared_ptr<PairManager_t> pair_manager;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * fixture for providing a neighbour list from a manager which is built with
   * positions in its fixture. this fixture provides iteration over the variable
   * ``pair_manager``.
   */
  template <class ManagerImplementation>
  struct PairFixture {
    using Manager_t = ManagerImplementation;

    static_assert(Manager_t::traits::MaxOrder == 1,
                  "Lower layer manager has MaxOrder needs MaxOrder = 1");

    using PairManager_t = AdaptorNeighbourList<Manager_t>;

    const double cutoff{3.};
    PairFixture()
        : pair_manager{make_adapted_manager<AdaptorNeighbourList>(
              this->fixture.manager, cutoff, false)} {
      this->pair_manager->update();
    }

    ~PairFixture() {}

    ManagerFixture<Manager_t> fixture{};
    std::shared_ptr<PairManager_t> pair_manager;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Specialization of ManagerFixture for StructureManagerCenters with input
   * data directly given.
   */
  template <>
  struct ManagerFixture<StructureManagerCenters> {
    using Manager_t = StructureManagerCenters;
    using ArrayB_t = AtomicStructure<3>::ArrayB_t;

    ManagerFixture()
        : manager{make_structure_manager<Manager_t>()}, cutoff{2.} {
      Eigen::ArrayXi to_change_id(50);
      ArrayB_t center_atoms_mask(50);
      for (auto & filename : this->filenames) {
        for (auto & n_is_not_center : this->n_is_not_centers) {
          structures.emplace_back();
          structures.back().set_structure(filename);

          // don't center on all atoms
          center_atoms_mask =
              ArrayB_t::Ones(structures.back().get_number_of_atoms());
          to_change_id = ((Eigen::ArrayXd::Random(n_is_not_center) +
                           Eigen::ArrayXd::Ones(n_is_not_center)) *
                          ((center_atoms_mask.size() - 1) * 0.5))
                             .cast<int>();
          for (int i_id{0}; i_id < n_is_not_center; ++i_id) {
            center_atoms_mask(to_change_id(i_id)) = false;
          }

          structures.back().center_atoms_mask = center_atoms_mask;
          managers.push_back(make_structure_manager<Manager_t>());
          managers.back()->update(structures.back());
        }
      }
      manager->update(structures[0]);
    }

    ~ManagerFixture() {}

    std::shared_ptr<Manager_t> manager;
    std::vector<AtomicStructure<3>> structures{};
    std::vector<std::shared_ptr<Manager_t>> managers{};
    std::vector<std::string> filenames{
        "reference_data/inputs/CaCrP2O7_mvc-11955_symmetrized.json"};
    std::vector<int> n_is_not_centers{0, 3, 5};
    double cutoff;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * A simple manager using ManagerCenters to check the neighbourlist algorithm
   * with simple positions and a periodicity only in x-direction. This manager
   * is also used to check the species filter.
   */
  struct ManagerFixtureSimple : public ManagerFixture<StructureManagerCenters> {
    ManagerFixtureSimple()
        : ManagerFixture<StructureManagerCenters>{}, pbc{{true, false, false}},
          cell(3, 3), positions(3, 8), atom_types(8), cutoff{2.1}, natoms{8} {
      cell << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0;
      // clang-format off
      positions << 0.4, 1.4, 0.4,
                   1.4, 0.4, 1.4,
                   0.4, 1.4, 0.4,
                   0.4, 1.4, 1.4,
                   0.4, 0.4, 1.4,
                   1.4, 0.4, 0.4,
                   0.4, 0.4, 1.4,
                   1.4, 1.4, 1.4;
      // clang-format on
      atom_types << 1, 3, 2, 1, 1, 2, 2, 3;

      using PBC_t = Eigen::Map<Eigen::Matrix<int, 3, 1>>;
      this->manager->update(positions, atom_types, cell, PBC_t{pbc.data()});
    }

    ~ManagerFixtureSimple() {}

    std::array<int, 3> pbc;
    Eigen::MatrixXd cell;
    Eigen::MatrixXd positions;
    Eigen::VectorXi atom_types;

    double cutoff;

    const int natoms;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * A manager using ManagerCenters to check the neighbourlist algorithm
   * with a very skewed cell and positions right at the edge of the periodicity
   */
  struct ManagerFixtureSkewDeltaRcut
      : public ManagerFixture<StructureManagerCenters> {
    ManagerFixtureSkewDeltaRcut()
        : ManagerFixture<StructureManagerCenters>{}, pbc{{true, true, true}},
          cell(3, 3), positions(12, 3),
          atom_types(12), cutoff{0.997}, natoms{12} {
      cell << 10.0, 0.0, 0.0, 9.396926207859085, 3.420201433256687, 0.0,
          9.396926207859085, 1.6569316261720377, 2.992048701181514;

      cell.transposeInPlace();
      // clang-format off
      positions <<  0.00100000, 0.00000000, 0.00000000,
                    9.99900000, 0.00000000, 0.00000000,
                   10.00000000, 0.00100000, 0.00000000,
                    9.39692621, 3.41920143, 0.00000000,
                   19.39692621, 3.42020143, 0.00100000,
                    9.39692621, 1.65693163, 2.99104870,
                    0.49800000, 0.00000000, 0.00000000,
                    9.50200000, 0.00000000, 0.00000000,
                   10.00000000, 0.49800000, 0.00000000,
                    9.39692621, 2.92220143, 0.00000000,
                   19.39692621, 3.42020143, 0.49800000,
                    9.39692621, 1.65693163, 2.49404870;
      // clang-format on
      positions.transposeInPlace();

      atom_types << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

      using PBC_t = Eigen::Map<Eigen::Matrix<int, 3, 1>>;
      this->manager->update(positions, atom_types, cell, PBC_t{pbc.data()});
    }

    ~ManagerFixtureSkewDeltaRcut() {}

    std::array<int, 3> pbc;
    Eigen::MatrixXd cell;
    Eigen::MatrixXd positions;
    Eigen::VectorXi atom_types;

    double cutoff;

    const int natoms;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * A fixture to check the neighbourlist algorithm with increasing skewedness
   * of the cell as well as a shift of the positions. The manager is built and
   * constructed inside the loop in the test: it skews the cells, therefore it
   * is not templated. The initial cutoff here is chosen to be smaller than the
   * atomic distance in this case to ensure to start with zero neighbours.
   */
  struct ManagerFixtureSkew {
    ManagerFixtureSkew()
        : pbc{{true, true, false}}, cell(3, 3), positions(3, 2),
          atom_types(2), cutoff{0.49}, natoms{2} {
      // clang-format off
      cell << 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 0.5;

      positions << 0.01, 0.01, 0.01,
                   0.51, 0.01, 0.01;
      // clang-format on

      atom_types << 1, 1;
    }

    ~ManagerFixtureSkew() {}

    std::array<int, 3> pbc;
    Eigen::MatrixXd cell;
    Eigen::MatrixXd positions;
    Eigen::VectorXi atom_types;

    double cutoff;

    const int natoms;
  };
}  // namespace rascal

#endif  // TESTS_TEST_STRUCTURE_HH_
