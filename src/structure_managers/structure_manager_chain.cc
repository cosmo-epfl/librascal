/**
 * file   structure_manager_chain.cc
 *
 * @author Markus Stricker <markus.stricker@epfl.ch>
 *
 * @date   30 May 2018
 *
 * @brief Implementation of the neighbourhood manager for polyalanine
 *        chain from json file
 *
 * Copyright © 2018 Markus Stricker, COSMO (EPFL), LAMMM (EPFL)
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

#include "structure_managers/structure_manager_chain.hh"

#include <numeric>
#include <fstream>
#include <iostream>

namespace rascal {

  /* ---------------------------------------------------------------------- */

  /**
   * After reading the <code>atoms_object</code> from the file, the cell vectors
   * as well as the atomic positions are put into contiguous a
   * <code>std::vector</code> data structure to ensure fast access via the
   * <code>Eigen::Map</code>.  <code>std::vector</code> provide iterator access,
   * which is used here with the <code>auto</code> keyword. Using this, it is
   * unnecessary to know the exact size of the positions/cell array. No
   * distinction between 2 or 3 dimensions. We just put all numbers in a vector
   * and access them with the map. Using the vector type automatically ensures
   * contiguity
   */
  void StructureManagerChain::update(double cutoff) {

    /**
     * Check if a structure has already been read, if not, throw an exception to
     * let the user know, what is wrong.
     */
    try {
      auto pos_size = atoms_object.position.size();
      if (pos_size == 0) {
        throw std::runtime_error("No atomic structure defined. "
                                 "Read structure first!");
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    for (auto vec : atoms_object.cell) {
      for (auto coord : vec) {
        this->cell_data.push_back(coord);
      }
    }

    for (auto pos : atoms_object.position) {
      for (auto coord : pos) {
        this->pos_data.push_back(coord);
      }
    }

    /**
     * Before going further, we check if the number of positions and atom types
     * match, otherwise the data set is incomplete.
     */
    assert(atoms_object.position.size() == atoms_object.type.size());
    /**
     * Set the protected member variable number of atoms, depending on the given
     * number of positions
     */
    this->natoms = atoms_object.position.size();
    /**
     * Invoke neighbourlist builder, because without it, you can not really do
     * anything.
     */
    this->make_neighbourlist(cutoff);
    /**
     * The following two commands build a list of increasing indices. It is
     * assumed that the atoms do not have a unique index, when they are read
     * from file. Therefore a list of increasing integer identifiers is
     * built. Numbers are assigned to the positions in the order in which they
     * appear in the file.
     */
    this->ilist.resize(this->natoms);
    std::iota(ilist.begin(), ilist.end(), 0);
    /**
     * The next commands gather necessary information to iterate over the half
     * neighbourlist.
     */
    this->offsets.reserve(this->natoms);
    this->offsets.resize(1);

    for (size_t i{0} ; i<this->natoms-1 ; ++i) {
      this->offsets.emplace_back(this->offsets[i] + this->numneigh[i]);
    }

    auto & atom_cluster_indices{std::get<0>(this->cluster_indices_container)};
    auto & pair_cluster_indices{std::get<1>(this->cluster_indices_container)};

    atom_cluster_indices.fill_sequence();
    pair_cluster_indices.fill_sequence();

  }

  /* ---------------------------------------------------------------------- */
  /**
   * Helper function to get the number of <code>clusters</code> (atoms, pairs,
   * triplets, quadruplets) with a specific <code>Order</code>. The
   * <code>MaxOrder</code> depends on your implementation or
   * processing. Increasing the order is done with an adaptor.
   */
  size_t StructureManagerChain::get_nb_clusters(size_t cluster_size) const {
    switch (cluster_size) {
    case 1: {
      return this->natoms;
      break;
    }
    case 2: {
      return this->nb_pairs;
    }
    default:
      throw std::runtime_error("Can only handle atoms and pairs,"
                               " use adaptor to increase MaxOrder.");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Function for reading data from a JSON file in the ASE format. See the
   * definition of <code>AtomicStructure</code> and adapt the fields, which
   * should be read to your case. One peculiarity should be mentioned: The type
   * <code>std::fstream</code> does not throw an exception, if the file is not
   * read. That is the reason for the try/catch block -- to make sure, the file
   * is opened.
   */
  void StructureManagerChain::
  read_structure_from_json(const std::string filename) {

    json j;

    try {
      std::ifstream f(filename);
      if (!f.is_open()) throw std::ios::failure("Error opening JSON file!");
      f >> j;
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    /**
     * ASE json format is nested - here, first entry is actual data
     * structure. If in any case you should have multiple
     * <code>atoms_objects</code> in your file, which you want to read, the
     * following line has to be adapted. Nesting on the first level is
     * structure1, structure 2, etc. These could be a time series of a
     * simulation, but also just different structures you want to read in from
     * different simulation runs.  Each structure should contain the necessary
     * fields for the <code>AtomicStructure</code> object defined in the header
     * belonging to this file. Here, just the first one is read.
     */
    this->atoms_object = j.begin().value();
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Helper function for the linked cell algorithm for scaling/rescaling the
   * complete simulation box. It is included, because usually, there is no
   * origin given for the cell of the atomic environment.
   */
  inline double StructureManagerChain::get_box_length(int d) {
    Cell_ref Cell = this->get_cell();
    return Cell.col(d).norm();
  }
  /* ---------------------------------------------------------------------- */
  /**
   * Helper function to get the linear index for the respective cell. The cells
   * are represented as a contiguous 1D vector.
   */
  inline int get_linear_index(std::vector<int> nidx, std::vector<int> nmax) {
    auto dim = nidx.size();
    switch (dim) {
    case 1: {
      return nidx[0];
      break;
    }
    case 2: {
      return nidx[1]*nmax[0] + nidx[0];
      break;
    }
    case 3: {
      return nidx[2] * nmax[0] * nmax[1] +
        nidx[1] * nmax[0] + nidx[0];
      break;
    }
    default:
      throw std::runtime_error("Can only give index for max 3 dimensions");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Helper function for the linked cell algorithm. Given a position, a cutoff,
   * an offset and the maximum number of boxes, this returns the 3D index set of
   * the box.
   */
  inline std::vector<int>
  StructureManagerChain::
  get_box_index(Vector_ref& position,
                std::vector<double>& rc,
                Eigen::Matrix<double, 1, traits::Dim> offset,
                std::vector<int> nmax) {
    std::vector<int> nidx(traits::Dim);
    for(auto dim{0}; dim < traits::Dim; ++dim) {
      nidx[dim] = static_cast<int>(std::floor( (position(dim) - offset(dim))
                                               / rc[dim] ));
      nidx[dim] = std::min(nidx[dim], nmax[dim]-1);
      nidx[dim] = std::max(nidx[dim], 0);
    }
    return nidx;
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Helper function for linked cell algorithm. Given an atomic id, this
   * function collects all neighbours of the box and writes it to the full
   * neighbourlist <code>allneigh</code>.
   */
  inline void StructureManagerChain::
  collect_neighbour_info_of_atom(const int i,
                                 const std::vector<int> boxidx,
                                 const std::vector<int> nmax) {

    auto jcell_index = get_linear_index(boxidx, nmax);
    auto ihead = this->lc[jcell_index]; //ll[jcell_index];
    // Check if any atom is in given cell
    if(ihead != -1) {
      if(ihead != i) this->allneigh[i].push_back(ihead);
      // same cell neighbours?
      auto itail = this->ll[ihead];
      while (itail != -1) {
        if(itail != i) {
          this->allneigh[i].push_back(itail);
        }
        itail = this->ll[itail];
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Function for creating the full and half neighbour list. The result is
   * directly written into protected member variables of the neighbourhood
   * manager.
   */
  void StructureManagerChain::make_neighbourlist(double cutoff) {
    /* Neighbourlist algorithm, non periodic, non triclinic cell
     * for demonstration purposes.
     */
    //! internal variables for linked list/ linked cell
    std::vector<int> nmax(3);
    std::vector<double> rc(3);

    for(auto dim{0}; dim < traits::Dim; ++dim) {
      nmax[dim] = static_cast<int>(std::floor(this->get_box_length(dim)
                                              / cutoff));
      rc[dim] = static_cast<double>(this->get_box_length(dim) / nmax[dim]);
    };

    int nboxes{1};
    nboxes = std::accumulate(nmax.begin(), nmax.end(), 1,
                             std::multiplies<int>());
    //! If the cutoff is smaller than length of the structure.
    nboxes = std::max(nboxes, 1);

    //! Create the data structure for the linked cell algorithm.
    this->ll.resize(this->natoms);
    this->lc.resize(nboxes);
    std::fill(this->ll.begin(), this->ll.end(), -1);
    std::fill(this->lc.begin(), this->lc.end(), -1);

    Positions_ref atom_pos = this->get_positions();

    /**
     * Get an origin. This is needed, because sometimes positions have negative
     * values.
     */
    Eigen::Matrix<double, 1, traits::Dim> offset{};
    for(auto dim{0}; dim < traits::Dim; ++dim) {
      offset(dim) = std::min(0., atom_pos.row(dim).minCoeff());
    }

    //! Make cell lists
    std::vector<int> nidx(traits::Dim);
    for (auto i{0}; i < atom_pos.cols(); ++i) {
      auto * p{atom_pos.col(i).data()};
      Vector_ref pos{p};
      nidx = get_box_index(pos, rc, offset, nmax);
      auto linear_index = get_linear_index(nidx, nmax);
      this->ll[i] = this->lc[linear_index];
      this->lc[linear_index] = i;
    }

    if(this->verbose) {
      std::cout << ">>>> nboxes " << nboxes << std::endl;
      for (auto i{0}; i<nboxes; ++i) {
        auto n = this->lc[i];
        std::cout << "linear index " << i << std::endl;
        while (n != -1) {
          std::cout << "box " << i<< " atom " << n << std::endl;
          n = this->ll[n];
        }
      }
    }
    /**
     * Make verlet table of neighbours (vectorized), not strict Full neighbour
     * list
     */
    this->allneigh.resize(this->natoms);
    if(this->verbose) std::cout << ">>>>>>>>>>>>>>>>start neighbour list" << std::endl;
    for (auto i{0}; i < atom_pos.cols(); ++i) {
      auto * p{atom_pos.col(i).data()};
      Vector_ref pos{p};
      if(this->verbose) std::cout << "---pos \n" << pos << std::endl;

      //! Which is my own cell?
      nidx = get_box_index(pos, rc, offset, nmax);
      auto nidxtmp = nidx;

      switch(traits::Dim) {
      case 1: {
        std::vector<int> nd{-1, 0, 1};
        for (auto dx :nd) {
          nidxtmp[0] = nidx[0] + dx;
          if(nidxtmp[0] < 0 || nidxtmp[0] > nmax[0]-1) continue;
          if(nidxtmp[1] > nmax[1] -1) continue;
          this->collect_neighbour_info_of_atom(i, nidxtmp, nmax);
        }
        break;
      }
      case 2: {
        std::vector<int> nd{-1, 0, 1};
        for (auto dx : nd) {
          for (auto dy : nd) {
            nidxtmp[0] = nidx[0] + dx;
            nidxtmp[1] = nidx[1] + dy;
            if(nidxtmp[0] < 0 || nidxtmp[0] > nmax[0]-1) continue;
            if(nidxtmp[1] < 0 || nidxtmp[1] > nmax[1]-1) continue;
            this->collect_neighbour_info_of_atom(i, nidxtmp, nmax);
          }
        }
        break;
      }
      case 3: {
        std::vector<int> nd{-1, 0, 1};
        for (auto dx : nd) {
          for (auto dy : nd) {
            for (auto dz : nd) {
              nidxtmp[0] = nidx[0] + dx;
              nidxtmp[1] = nidx[1] + dy;
              nidxtmp[2] = nidx[2] + dz;
              if(nidxtmp[0] < 0 || nidxtmp[0] > nmax[0]-1) continue;
              if(nidxtmp[1] < 0 || nidxtmp[1] > nmax[1]-1) continue;
              if(nidxtmp[2] < 0 || nidxtmp[2] > nmax[2]-1) continue;
              this->collect_neighbour_info_of_atom(i, nidxtmp, nmax);
            }
          }
        }
        break;
      }
      default:
        throw std::runtime_error("Neighbourlist only max 3D.");
        break;
      }
    } // atom neighbours

    if (this->verbose) {
      for (size_t i=0; i<this->natoms; ++i) {
        std::cout << "Atom i " << i << std::endl;
        auto neighs = allneigh[i];
        for (auto n : neighs) {
          std::cout << n << " ";
        }
        std::cout << std::endl;
      }
    }

    // Half-neighbourlist
    this->numneigh.resize(this->natoms);
    this->nb_pairs = 0;

    for (size_t i=0; i < this->natoms; ++i) {
      auto neigh = allneigh[i];
      auto i_number_of_neighbours{0};
      for(size_t j : neigh) {
        if (j > i) {
          i_number_of_neighbours += 1;
          this->nb_pairs += 1;
          halfneigh.push_back(j);
        }
      }
      numneigh[i] = i_number_of_neighbours;
    }
    std::cout << std::endl;

    if (this->verbose) {
      std::cout << "Number of pairs: " << this->nb_pairs << std::endl;
      auto ioff{0};
      for (size_t n=0; n < this->natoms; ++n){
        auto a = numneigh[n];
        std::cout << "Next neigh atom---- " << n
                  << " number of neighbours " << a << std::endl;
        for (auto i=ioff; i < ioff+a; i++) {
          std::cout << "Neighbour "
                    << halfneigh[i] << std::endl;
        }
        ioff += a;
      }
    }

    //! Get number of pairs to write into member variable.
    this->nb_pairs = std::accumulate(this->numneigh.begin(),
                                     this->numneigh.end(), 0);
  }
} // rascal
