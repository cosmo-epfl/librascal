/**
 * file test_structure.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief  Common headers for tests related to `StructureManager`
 *
 * @section LICENSE
 *
 * Copyright © 2018 Till Junge, COSMO (EPFL), LAMMM (EPFL)
 *
 * rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef TEST_NEIGHBOURHOOD_H
#define TEST_NEIGHBOURHOOD_H

#include "structure_managers/structure_manager_base.hh"
#include "structure_managers/structure_manager_lammps.hh"
#include "structure_managers/structure_manager_chain.hh"
#include "structure_managers/structure_manager_centers.hh"
#include "structure_managers/adaptor_strict.hh"
#include "structure_managers/adaptor_increase_maxorder.hh"
#include "structure_managers/adaptor_neighbour_list.hh"

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
   * Most basic fixture. This is only to guarantee, that the manager, which is
   * built with existing data and not adapted is always accessible with the
   * variable ``manager``.
   */
  template<class ManagerImplementation>
  struct ManagerFixture {
    ManagerFixture():
      pbc{{true, true, true}}, cell(3, 3), positions(22, 3),
      numbers(22), cutoff{2.}
    {
      cell <<
        6.19, 2.41, 0.21,
        0.00, 6.15, 1.02,
        0.00, 0.00, 7.31;
      positions <<
        3.689540159937393, 5.123016813620886, 1.994119731169116,
        6.818437242389163, 2.630056617829216, 6.182500355729062,
        2.114977334498767, 6.697579639059512, 1.392155450018263,
        7.420401523540017, 2.432242071439904, 6.380314902118375,
        1.112656394115962, 7.699900579442317, 3.569715877854675,
        5.242841095703604, 3.122826344932127, 5.689730628626151,
        3.248684682453303, 5.563872291104976, 2.608353462112637,
        6.204203511445642, 5.035681855581504, 2.134827911489532,
        0.946910011088814, 6.223599755982222, 4.168634519120968,
        3.001875247950068, 1.980327734683430, 5.190182032387606,
        2.943861424421339, 4.226648342649697, 5.457161501166098,
        1.713348265904937, 1.501663178733906, 5.668846588337130,
        5.208365510425203, 1.962144256645833, 2.728127406527150,
        4.442382360543885, 2.839975217222644, 4.330534549848392,
        0.744216089807768, 6.426293677263268, 4.643695520786083,
        2.662204050783991, 1.250682335857938, 6.055217235712136,
        0.860905287815103, 6.444994283754972, 4.536108843695142,
        2.769790727874932, 5.609177455068640, 1.696722116501434,
        6.703053268421970, 0.602846303148105, 3.487609972580834,
        3.818289598989240, 1.436734374347541, 5.869165197222533,
        1.054504320562138, 6.251395251007936, 3.998423858825871,
        3.307475712744203, 5.323662899811682, 1.982236671758393;
      positions.transposeInPlace();
      atom_types << 20, 20, 24, 24, 15, 15, 15, 15,  8,  8,  8,
        8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8;

      manager.update(positions, atom_types, cell,
              Eigen::Map<Eigen::Matrix<int, 3, 1>>{pbc.data()});
    }

    ~ManagerFixture() { }

    ManagerImplementation manager{};
    std::array<int, 3> pbc;
    Eigen::MatrixXd cell;
    Eigen::MatrixXd positions; // 3, 22
    Eigen::VectorXi atom_types;
    double cutoff;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Basic fixture for using two manager to compare things. This is to
   * guarantee, that the managers, built with existing data and not adapted is
   * always accessible with the variable ``manager_1`` and ``manager_2``. Both
   * managers are of the same class.
   */
  template<class ManagerImplementation>
  struct ManagerFixtureTwo
  {
    ManagerFixtureTwo() {} // ctor
    ~ManagerFixtureTwo() {} // dtor

    ManagerImplementation manager_1{};
    ManagerImplementation manager_2{};
  };

  /* ---------------------------------------------------------------------- */
  /**
   * general case of a manager fixture, which reads the structure information
   * from a file, atomic structure contains 9 atoms in a very simple cubic unit
   * cell, no periodicity, it inherits publicly from the ManagerFixture, which
   * provides access to the variable ``manager``.
   */
  template<class ManagerImplementation>
  struct ManagerFixtureFile : public ManagerFixture<ManagerImplementation> {
    ManagerFixtureFile() :
      ManagerFixture<ManagerImplementation> {}, // initialize manager variable
      cutoff{1.}, filename{"simple_cubic_9.json"} // initialize current fixture
    {
      this->manager.update(filename);
    }

    ~ManagerFixtureFile()  {}

    // additional variables for present fixture
    double cutoff;
    std::string filename{};
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Fixture which provides two managers ``manager_1`` and ``manager_2``, which
   * both have a hexagonal lattice, but with different unit cells (vectors) with
   * two atoms. It is used to ensure that the neighbour list algorithm is robust
   * with respect to the unit cell.
   */
  struct ManagerFixtureTwoHcp
    : public ManagerFixtureTwo<StructureManagerCenters>
  {
    ManagerFixtureTwoHcp():
      ManagerFixtureTwo<StructureManagerCenters> {},
      pbc{{true,true,true}}, cell_1(3,3), cell_2(3,3),
      positions_1(3, 2), positions_2(3, 2),
      atom_types(2), cutoff{0.7}
    {
      /*
       * hcp crystal with lattice parameter a = 1, c = sqrt(8/3), defined in two
       * unit cells: basal and prismatic 1. The neighbourlist is built with the
       * same cutoff. The test checks, if all atoms have the same number of
       * neighbours.
       */
      auto a{1.};
      auto c{std::sqrt(8./3.)};

      cell_1 <<
        a,  -0.5*a ,            0.,
        0., std::sqrt(3.)/2.*a, 0.,
        0.,  0.,                c;

      cell_2 <<
        a,   0.,         0.5*a,
        0.,  c,             0.,
        0.,  0.,  std::sqrt(3.)/2.*a;

      auto p_1 =
        2./3. * cell_1.col(0)
        + 1./3. * cell_1.col(1)
        + 1./2. * cell_1.col(2);

      positions_1 <<
        0.0, p_1[0],
        0.0, p_1[1],
        0.0, p_1[2];

      auto p_2 =
        -1./3. * cell_2.col(0)
        + 1./2. * cell_2.col(1)
        + 2./3. * cell_2.col(2);

      positions_2 <<
        0.0, p_2[0],
        0.0, p_2[1],
        0.0, p_2[2];

      atom_types << 1, 1;

      manager_1.update(positions_1, atom_types, cell_1,
                       Eigen::Map<Eigen::Matrix<int, 3, 1>>
                       {pbc.data()});

      manager_2.update(positions_2, atom_types, cell_2,
                       Eigen::Map<Eigen::Matrix<int, 3, 1>>
                       {pbc.data()});
    }

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
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Fixture which provides two managers ``manager_1`` and ``manager_2``, which
   * both have a face centered cubic lattice, but with different unit cells
   * (vectors) with one and four atoms. It is another test to ensure that the
   * neighbour list algorithm is robust with respect to the unit cell.
   */
  struct ManagerFixtureTwoFcc
    : public ManagerFixtureTwo<StructureManagerCenters>
  {
    ManagerFixtureTwoFcc():
      ManagerFixtureTwo<StructureManagerCenters> {},
      pbc{{true, true, true}}, cell_1(3, 3), cell_2(3, 3),
      positions_1(3, 1), positions_2(3, 4),
      atom_types_1(1), atom_types_2(4),
      cutoff{0.7}, // starting with zero neighbours
      natoms_1{1}, natoms_2{4}
    {
      /*
       * fcc unit cells: first cell consists of only one atom, which is
       * repeated, second cell is the conventional 4 atoms. This test checks, if
       * the found number of neighbours with increasing cutoff is the same for
       * the atom at position (0, 0, 0).
       */
      auto a{1.};

      cell_1 <<
        a,  0.5*a, 0.5*a,
        0., 0.5*a, 0.   ,
        0., 0.,    0.5*a;

      cell_2 <<
        a,   0., 0.,
        0.,  a,  0.,
        0.,  0., a ;

      positions_1 <<
        0.,
        0.,
        0.;

      auto p_2 = 0.5 * cell_2.col(0) + 0.5 * cell_2.col(1);
      auto p_3 = 0.5 * cell_2.col(0) + 0.5 * cell_2.col(2);
      auto p_4 = 0.5 * cell_2.col(1) + 0.5 * cell_2.col(2);

      positions_2 <<
        0.0, p_2[0], p_3[0], p_4[0],
        0.0, p_2[1], p_3[1], p_4[1],
        0.0, p_2[2], p_3[2], p_4[2];

      atom_types_1 << 1;
      atom_types_2 << 1, 1, 1, 1;

      manager_1.update(positions_1, atom_types_1, cell_1,
                       Eigen::Map<Eigen::Matrix<int, 3, 1>>
                       {pbc.data()});

      manager_2.update(positions_2, atom_types_1, cell_2,
                       Eigen::Map<Eigen::Matrix<int, 3, 1>>
                       {pbc.data()});
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
  template <>
  struct ManagerFixture<StructureManagerLammps> {
    using Manager_t = StructureManagerLammps;
    constexpr static int nb{3};
    constexpr static int dim{3};
    using ptr_t = double**;

    ManagerFixture()
      :firstneigh{new int*[nb]},
       x{new double*[nb]},
       f{new double*[nb]},
       vatom{new double*[nb]},
       manager{} {
         manager.update(inum, tot_num, ilist, numneigh,
                        static_cast<int**>(firstneigh),
                        ptr_t(x),
                        ptr_t(f), type, eatom,
                        static_cast<double**>(vatom));
         firstneigh[0] = new int[2];
         firstneigh[0][0] = 1;
         firstneigh[0][1] = 2;
         firstneigh[1] = new int;
         firstneigh[1][0] = 0;
         firstneigh[2] = new int;
         firstneigh[2][0] = 0;

         for (int i{0} ; i < nb ; ++i) {
           x[i] = new double[dim];
           f[i] = new double[dim];
           for (int j{0} ; j < dim ; ++j) {
             x[i][j] = tx[i][j];
             f[i][j] = tf[i][j];
           }
         }
       }

    ManagerFixture( ManagerFixture &) = delete;
    ManagerFixture & operator=(const ManagerFixture&) = delete;
    ~ManagerFixture() {
      delete[] firstneigh[0];
      delete firstneigh[1];
      delete firstneigh[2];
      delete[] firstneigh;
      delete[] vatom;
      for (int i{0} ; i < nb ; ++i) {
        delete[] x[i];
        delete[] f[i];
      }
      delete[] x;
      delete[] f;
    }
    double tx[nb][nb] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    double tf[nb][nb] = {{1, 1, 0}, {-1, 0, 0}, {0, -1, 0}};

    int inum{nb};
    int tot_num{nb}; //includes ghosts
    int ilist[nb]{0, 1, 2};
    int numneigh[nb]{2, 1, 1};
    int ** firstneigh;
    double **x;
    double **f;
    int type[nb]{1, 1, 1};
    double  eatom[3]{2, 1, 1};
    double ** vatom;
    Manager_t manager;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * fixture for providing a neighbour list from a simple manager, which is read
   * from a JSON file. This fixture provides iteration over the variable
   * ``pair_manager``
   */
  template<class ManagerImplementation>
  struct PairFixtureFile : public ManagerFixtureFile<ManagerImplementation>
  {
    using Manager_t = ManagerImplementation;

    static_assert(ManagerImplementation::traits::MaxOrder == 1,
                  "Lower layer manager has MaxOrder needs MaxOrder = 1");

    using PairManager_t = AdaptorNeighbourList<ManagerImplementation>;

    PairFixtureFile()
      : ManagerFixtureFile<ManagerImplementation> {},
      pair_manager{this->manager, this->cutoff}
    {
      this->pair_manager.update();
    }

    ~PairFixtureFile() {}

    AdaptorNeighbourList<ManagerImplementation> pair_manager;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * fixture for providing a neighbour list from a manager which is built with
   * positions in its fixture. this fixture provides iteration over the variable
   * ``pair_manager``.
   */
  template<class ManagerImplementation>
  struct PairFixture : public ManagerFixture<ManagerImplementation> {
    using Manager_t = ManagerImplementation;

    static_assert(ManagerImplementation::traits::MaxOrder == 1,
                  "Lower layer manager has MaxOrder needs MaxOrder = 1");

    using PairManager_t = AdaptorNeighbourList<ManagerImplementation>;

    PairFixture()
      : ManagerFixture<ManagerImplementation> {},
      pair_manager{this->manager, 3.}
    {
      this->pair_manager.update();
    }

    ~PairFixture() {}

    AdaptorNeighbourList<ManagerImplementation> pair_manager;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * A simple fixture using ManagerCenters to check the neighbourlist algorithm
   * with simple positions and a periodicity only in x-direction.
   *
   */
  struct ManagerFixtureSimple : public ManagerFixture<StructureManagerCenters> {
    ManagerFixtureSimple() :
      ManagerFixture<StructureManagerCenters> {},
      pbc{{true, false, false}}, cell(3, 3), positions(3, 8), atom_types(8),
      cutoff{2.1}, natoms{8}
    {
      cell <<
        2., 0., 0.,
        0., 2., 0.,
        0., 0., 2.;

      positions <<
        0.4, 1.4, 0.4, 1.4, 0.4, 1.4, 0.4, 1.4,
        0.4, 0.4, 1.4, 1.4, 0.4, 0.4, 1.4, 1.4,
        0.4, 0.4, 0.4, 0.4, 1.4, 1.4, 1.4, 1.4;

      atom_types << 1, 1, 1, 1, 1, 1, 1, 1;

      this->manager.update(positions, atom_types, cell,
                           Eigen::Map<Eigen::Matrix<int, 3, 1>>{pbc.data()});
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
   * A fixture to check the neighbourlist algorithm with increasing skewedness
   * of the cell as well as a shift of the positions. The manager is built and
   * constructed inside the loop in the test: it skews the cells, therefore it
   * is not templated. The initial cutoff here is chosen to be smaller than the
   * atomic distance in this case to ensure to start with zero neighbours.
   */
  struct ManagerFixtureSkew {
    ManagerFixtureSkew():
      pbc{{true, true, false}}, cell(3, 3), positions(3, 2), atom_types(2),
      cutoff{0.49}, natoms{2}
    {
      cell <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.5;

      positions <<
        0.01, 0.01,
        0.01, 0.51,
        0.01, 0.01;

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
}  // rascal

#endif /* TEST_NEIGHBOURHOOD_H */
