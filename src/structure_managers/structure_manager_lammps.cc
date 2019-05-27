/**
 * file   structure_manager_lammps.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief Implementation of the neighbourhood manager for lammps
 *        neighbourhood lists
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

#include "structure_managers/structure_manager_lammps.hh"

#include <numeric>

namespace rascal {

  /* ---------------------------------------------------------------------- */
  void StructureManagerLammps::update_self(const int & inum,
                                           const int & tot_num, int * ilist,
                                           int * numneigh, int ** firstneigh,
                                           double ** x, double ** f, int * type,
                                           double * eatom, double ** vatom) {
    // setting the class variables
    this->inum = inum;
    this->tot_num = tot_num;
    this->ilist = ilist;
    this->numneigh = numneigh;
    this->firstneigh = firstneigh;
    this->x = x;
    this->f = f;
    this->type = type;
    this->eatom = eatom;
    this->vatom = vatom;
    this->offsets.reserve(inum);
    this->offsets.resize(1);
    // #BUG8486@(all) it should be this->inum
    for (int i{0}; i < this->inum; ++i) {
      this->offsets.emplace_back(this->offsets[i] + this->numneigh[i]);
    }
    this->nb_pairs = std::accumulate(numneigh, numneigh + this->inum, 0);

    auto & atom_cluster_indices{std::get<0>(this->cluster_indices_container)};
    auto & pair_cluster_indices{std::get<1>(this->cluster_indices_container)};

    // #BUG8486@(all) is this solution efficient, does ilist have huge gaps?
    // e.g. ilist = [1,5000], then the cluster_index_from_atom_tag_list list will
    // have 5000 elements.
    // Also the dummy 0 values which will cause undefined behaviour but not
    // necessary an error worries me
    
    //! Finding maximum atomic index
    int max_atomic_index = 0; 
    for (int i{0}; i<this->inum; ++i) {
      if (ilist[i] > max_atomic_index) {max_atomic_index = ilist[i];}
    }

    //! Filling dummy cluster index 
    this->cluster_index_from_atom_tag_list.reserve(max_atomic_index+1);
    for (int i{0}; i< max_atomic_index+1; ++i) {
      this->cluster_index_from_atom_tag_list.push_back(0);
    }
    //! Replacing dummy values with correct cluster index
    for (int i{0}; i<this->inum; ++i) {
      // ilist does not have negative atom tags therefore the cast is safe
      this->cluster_index_from_atom_tag_list.at(static_cast<size_t>(ilist[i])) = i;
    }
    atom_cluster_indices.fill_sequence();
    pair_cluster_indices.fill_sequence();
  }

  /* ---------------------------------------------------------------------- */
  /*
   * Return the number of clusters of size cluster_size.  Can only handle
   * cluster_size 1 (atoms) and cluster_size 2 (pairs).
   */
  size_t StructureManagerLammps::get_nb_clusters(int order) const {
    switch (order) {
    case 1: {
      return inum;
      break;
    }
    case 2: {
      return nb_pairs;
      break;
    }
    default:
      throw std::runtime_error("Can only handle single atoms and pairs");
      break;
    }
  }
}  // namespace rascal
