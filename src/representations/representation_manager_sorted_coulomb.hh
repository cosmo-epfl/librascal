/**
 * file   representation_manager_sorted_coulomb.hh
 *
 * @author Musil Felix <musil.felix@epfl.ch>
 *
 * @date   14 September 2018
 *
 * @brief  Implements the Sorted Coulomb representation
 *
 * Copyright © 2018 Musil Felix, COSMO (EPFL), LAMMM (EPFL)
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
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef BASIS_REPRESENTATION_MANAGER_SORTED_COULOMB_H
#define BASIS_REPRESENTATION_MANAGER_SORTED_COULOMB_H


#include "representations/representation_manager_base.hh"
#include "structure_managers/structure_manager.hh"
#include "structure_managers/property.hh"
#include "rascal_utility.hh"
#include <vector>
#include <algorithm>

namespace rascal {

  namespace internal {

    typedef std::vector<double>::const_iterator myiter;

    /** Function for the sorting of a container using the order 
     * from another container
     */
    struct ordering {
      bool operator ()(std::pair<size_t, myiter> const& a, 
                              std::pair<size_t, myiter> const& b) {
          return *(a.second) < *(b.second);
      }
    };

    /** Use the ordering from a sorted container to sort another one
     * 
     * @params in Container to sort
     * @params referece Contains the sort order from the other container
     * @returns copy of in that has been sorted
     */
    template <typename T>
    std::vector<T> sort_from_ref(
        std::vector<T> const& in,
        std::vector<std::pair<size_t, myiter> > const& reference) {
        std::vector<T> ret(in.size());

        size_t const size = in.size();
        for (size_t i{0}; i < size; ++i)
          {
            ret[i] = in[reference[i].first];
          }

        return ret;
    }
    /** Sort the coulomb matrix using the distance to the central atom
     * as reference order and linearize it.
     * 
     * @params in Eigen Matrix to sort
     * @params out Eigen Matrix that will contain the result
     * @params reference  Contains the sort order from the other container
     */
    template <typename DerivedA,typename DerivedB>
    void sort_coulomb_matrix(
      const Eigen::DenseBase<DerivedA> & in,
      Eigen::DenseBase<DerivedB> & out,
      std::vector<std::pair<size_t, myiter> > const& reference) {
      
      auto Nneigh{in.cols()};
      size_t lin_id{0};
      // auto nn{ncol*(ncol-1)/2};
      // std::cout <<ncol<<", "<<nn <<std::endl;
      for (int ii{0}; ii < Nneigh; ++ii) {
        size_t iis{reference[ii].first};
        for (int jj{0}; jj < ii+1; ++jj) {
          size_t jjs{reference[jj].first};
          out(lin_id) = in(iis,jjs);
          lin_id += 1;
        } 
      }
    }
  } // internal

  template<class StructureManager>
  class RepresentationManagerSortedCoulomb: public RepresentationManagerBase
  {
  public:

    // TODO make a traits mechanism
    using Manager_t = StructureManager;
    using Parent = RepresentationManagerBase;
    using hypers_t = typename Parent::hypers_t;
    using precision_t = typename Parent::precision_t;
    using Property_t = Property<precision_t, 1, 1, Eigen::Dynamic, 1>;
    
    //! Default constructor 
    RepresentationManagerSortedCoulomb(Manager_t &sm, const hypers_t& hyper)
      :structure_manager{sm},central_decay{},
      interaction_cutoff{},
      interaction_decay{},coulomb_matrices{sm}
      {
        this->set_hyperparameters(hyper);
        this->check_size_compatibility();
      }
    
    RepresentationManagerSortedCoulomb(Manager_t &sm, 
                                      const std::string& hyper_str)
      :structure_manager{sm},central_decay{},
      interaction_cutoff{},
      interaction_decay{},coulomb_matrices{sm}
      {
        this->set_hyperparameters(hyper_str);
        this->check_size_compatibility();
      }

    //! Copy constructor
    RepresentationManagerSortedCoulomb(
      const RepresentationManagerSortedCoulomb &other) = delete;

    //! Move constructor
    RepresentationManagerSortedCoulomb(
      RepresentationManagerSortedCoulomb &&other) = default;

    //! Destructor
    virtual ~RepresentationManagerSortedCoulomb()  = default;

    //! Copy assignment operator
    RepresentationManagerSortedCoulomb& operator=(
      const RepresentationManagerSortedCoulomb &other) = delete;

    //! Move assignment operator
    RepresentationManagerSortedCoulomb& operator=(
      RepresentationManagerSortedCoulomb && other) = default;
    

    //! compute representation
    void compute();

    //! set hypers
    void set_hyperparameters(const hypers_t & );
    void set_hyperparameters(const std::string & );

    //! getter for the representation
    Eigen::Map<Eigen::MatrixXd> get_representation_full() {
      auto Nb_centers{this->structure_manager.nb_clusters(1)};
      auto Nb_features{this->get_n_feature()};
      auto& raw_data{this->coulomb_matrices.get_raw_data()};
      Eigen::Map<Eigen::MatrixXd> representation(raw_data.data(),Nb_features,Nb_centers);
      return representation;
    }

    //! get the raw data of the representation
    std::vector<precision_t>& get_representation_raw_data(){
      return this->coulomb_matrices.get_raw_data();
    }

    //! get the size of a feature vector
    size_t get_feature_size(){
      return this->coulomb_matrices.get_nb_comp();
    }

    //! get the number of centers for the representation
    size_t get_center_size(){
      return this->coulomb_matrices.get_nb_item();
    }

     void check_size_compatibility(){
      for (auto center: this->structure_manager){
        auto Nneighbours{center.size()};
        if (Nneighbours > this->size){
            std::cout << "size is too small for this "
                          "structure and has been reset to: " 
                      << Nneighbours << std::endl;
            this->size = Nneighbours;
        }
      }
    }

    //! get the size of a feature vector from the hyper
    //! parameters
    inline size_t get_n_feature(){
      return this->size*(this->size+1)/2;
    }

    Manager_t& structure_manager;
    //hypers_t hyperparmeters;
    double central_decay{};
    double interaction_cutoff{};
    double interaction_decay{};
    // at least equal to the largest number of neighours
    size_t size{};

    hypers_t hypers{};

    Property_t coulomb_matrices;

  protected:
  private:
  };

  /* ---------------------------------------------------------------------- */

  template<class Mngr>
  void RepresentationManagerSortedCoulomb<Mngr>::set_hyperparameters(
          const RepresentationManagerSortedCoulomb<Mngr>::hypers_t & hyper){

    this->central_decay = hyper["central_decay"];
    this->interaction_cutoff = hyper["interaction_cutoff"];
    this->interaction_decay = hyper["interaction_decay"];
    this->size = hyper["size"];
    this->hypers = hyper; 
  };

  template<class Mngr>
  void RepresentationManagerSortedCoulomb<Mngr>::set_hyperparameters(
          const std::string & hyper_str){
    
    this->hypers = json::parse(hyper_str); 
    this->central_decay = this->hypers["central_decay"];
    this->interaction_cutoff = this->hypers["interaction_cutoff"];
    this->interaction_decay = this->hypers["interaction_decay"];
    this->size = this->hypers["size"];
    
  };

  /* ---------------------------------------------------------------------- */

  template<class Mngr>
  void RepresentationManagerSortedCoulomb<Mngr>::compute(){
    
    // initialise the sorted coulomb_matrices in linear storage
    this->coulomb_matrices.resize_to_zero();
    this->coulomb_matrices.set_nb_row(this->get_n_feature());
    
    // initialize the sorting map
    typedef std::vector<double>::const_iterator distiter;
    
    // initialize the sorted linear coulomb matrix 
    Eigen::MatrixXd lin_sorted_coulomb_mat(this->size*(this->size+1)/2,1);

    // loop over the centers
    for (auto center: this->structure_manager){
            
      // initialize the distances to be sorted. the center is always first
      std::vector<double> distances_to_sort{0};
      // Nneighbour counts the central atom and the neighbours
      size_t Nneighbour{center.size()+1};
      
      // re-use the temporary coulomb mat in linear storage
      lin_sorted_coulomb_mat = Eigen::MatrixXd::Zero(this->size*(this->size+1)/2,1);
      // the local coulomb matrix
      Eigen::MatrixXd coulomb_mat = Eigen::MatrixXd::Zero(Nneighbour,Nneighbour);

      // the coulomb mat first row and col corresponds to central atom to neighbours
      int Zk{center.get_atom_type()};
      coulomb_mat(0,0) = 0.5*std::pow(Zk,2.4);
      for (auto neigh_i: center){
        size_t ii{neigh_i.get_index()+1};
        int Zi{neigh_i.get_atom_type()};
        double dik{this->structure_manager.get_distance(neigh_i)};
        distances_to_sort.push_back(dik);
        coulomb_mat(ii,0) = Zi*Zk/dik;
        coulomb_mat(0,ii) = coulomb_mat(ii,0);
      }

      // find the sorting order
      std::vector<std::pair<size_t, distiter> > order_coulomb(distances_to_sort.size());
      size_t nn{0};
      for (distiter it{distances_to_sort.begin()}; 
                            it != distances_to_sort.end(); ++it, ++nn)
          {order_coulomb[nn] = make_pair(nn, it);}
      std::sort(order_coulomb.begin(), order_coulomb.end(), internal::ordering());

      // compute the neighbour to neighbour part of the coulomb matrix
      for (auto neigh_i: center){
        size_t ii{neigh_i.get_index()+1};
        int Zi{neigh_i.get_atom_type()};
        coulomb_mat(ii,ii) = 0.5*std::pow(Zi,2.4);
        for (auto neigh_j: center){
          size_t jj{neigh_j.get_index()+1};
          // work only on the lower diagonal
          if (ii >= jj) continue;
          int Zj{neigh_j.get_atom_type()};
          double dij{(neigh_i.get_position()-neigh_j.get_position()).norm()};
          coulomb_mat(jj,ii) = Zi*Zj/dij;
          coulomb_mat(ii,jj) = coulomb_mat(jj,ii);
        }
      }
      
      // inject the coulomb matrix into the sorted linear storage
      internal::sort_coulomb_matrix(coulomb_mat,lin_sorted_coulomb_mat,order_coulomb);

      this->coulomb_matrices.push_back(lin_sorted_coulomb_mat);

    }
  }

}

#endif /* BASIS_REPRESENTATION_MANAGER_SORTED_COULOMB_H */

