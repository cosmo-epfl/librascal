/**
 * @file   examples/spherical_invariants_example.cc
 *
 * @author Max Veit <max.veit@epfl.ch>
 *
 * @date   26 June 2019
 *
 * @brief  Example for computing the spherical invariants (SOAP)
 *
 * Copyright © 2018 Max Veit, Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * librascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * librascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "rascal/representations/calculator_spherical_invariants.hh"
#include "rascal/models/sparse_points.hh"
#include "rascal/models/sparse_kernels.hh"
#include "rascal/models/sparse_kernel_predict.hh"
#include "rascal/structure_managers/adaptor_center_contribution.hh"
#include "rascal/structure_managers/adaptor_neighbour_list.hh"
#include "rascal/structure_managers/adaptor_strict.hh"
#include "rascal/structure_managers/atomic_structure.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/structure_manager_centers.hh"
#include "rascal/utils/basic_types.hh"
#include "rascal/utils/utils.hh"

#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <list>
#include <string>

using namespace rascal;  // NOLINT

using Calculator_t = CalculatorSphericalInvariants;

using ManagerCollection_t = ManagerCollection<StructureManagerCenters, AdaptorNeighbourList, AdaptorCenterContribution, AdaptorStrict>;
using Manager_t = AdaptorStrict<
    AdaptorCenterContribution<AdaptorNeighbourList<StructureManagerCenters>>>;
using Prop_t = typename CalculatorSphericalInvariants::Property_t<Manager_t>;
using PropGrad_t =
    typename CalculatorSphericalInvariants::PropertyGradient_t<Manager_t>;

int main(int argc, char * argv[]) {
  if (argc < 3) {
    std::cerr << "Must provide {model argument} {dataset_argument}";
    std::cerr << std::endl;
    return -1;
  }

  std::string model_filename{argv[1]};
  std::string dataset_filename{argv[2]};

  json input = json_io::load(model_filename);
  std::cout << "input loaded" << std::endl;
  json init_params = input.at("init_params").template get<json>();
  std::cout << "init_params" << std::endl;
  json X_train = init_params.at("X_train").template get<json>();
  std::cout << "X_train" << std::endl;

  // sparse points
  json sparse_data = X_train.at("data").template get<json>();
  json sparse_input = sparse_data.at("sparse_points").template get<json>();
  SparsePointsBlockSparse<Calculator_t> sparse_points{};
  from_json(sparse_input, sparse_points);

  // calculator
  json X_train_init_params = X_train.at("init_params").template get<json>();
  json representation = X_train_init_params.at("representation").template get<json>();
  json representation_init_params = representation.at("init_params").template get<json>();
  Calculator_t calculator{representation_init_params};

  // kernel
  json kernel_params = init_params.at("kernel").template get<json>();
  json kernel_init_params = kernel_params.at("init_params").template get<json>();
  SparseKernel kernel{kernel_init_params};

  // weights 
  std::vector<double> weights_vec = init_params.at("weights").template get<json>().at(1).template get<std::vector<double>>();
  if (sparse_points.size() != weights_vec.size()) {
    std::cerr << "weight size and sparse_points size disagree ";
    std::cerr << std::endl;
    return -1;
  }
  math::Vector_t weights = Eigen::Map<math::Vector_t>(weights_vec.data(), static_cast<long int>(weights_vec.size()));

  // manager
  //// cutoff
  double cutoff = representation_init_params.at("cutoff_function").template get<json>().at("cutoff").template get<json>().at("value").template get<double>();
  ////
  json adaptors_input = {
      {
        {"initialization_arguments", {{"cutoff", cutoff}}},
        {"name",   "neighbourlist"}
      },
      {
        {"initialization_arguments", {}},
        {"name", "centercontribution"}
      },
      {
        {"initialization_arguments", {{"cutoff", cutoff}}},
        {"name", "strict"}
      }
  };
  // check for adaptor hypers
  //std::cout << adaptors_input << std::endl;
  //json input_c = json_io::load("../reference_data/tests_only/sparse_kernel_inputs.json").at(0).template get<json>();
  //json adaptors_input_c = input_c.at("adaptors").template get<json>();
  //std::cout << std::endl;
  //std::cout << adaptors_input_c << std::endl;

  ManagerCollection_t managers{adaptors_input};
  managers.add_structures(dataset_filename, 0, -1);

  // compute repr
  calculator.compute(managers);

  for (auto manager : managers) {
    auto && expansions_coefficients{*manager->template get_property<Calculator_t::template Property_t<Manager_t>>(
      calculator.get_name())};
                                                                                                        
    for (auto atom : manager) {                                                                         
      std::cout << expansions_coefficients[atom].get_full_vector().transpose() << std::endl;
    }   
  }


  // predict gradient, stress
  
  // manual
  //const bool compute_stress{false};
  //math::Matrix_t KNM_der{kernel.compute_derivative(calculator, managers,
  //                                       sparse_points, compute_stress)};
  //math::Matrix_t gradients_k = KNM_der * weights.transpose();
  //std::cout << "gradients_k shape: " << gradients_k.rows() << ", " << gradients_k.cols() << std::endl; 
  math::Matrix_t KNM{kernel.compute(calculator, managers, sparse_points)};
  math::Matrix_t energies = KNM * weights.transpose();
  std::cout << energies.transpose() << std::endl;

  //std::string force_name = compute_sparse_kernel_gradients(
  //        calculator, kernel, managers, sparse_points, weights);

  //std::string neg_stress_name = compute_sparse_kernel_neg_stress(
  //    calculator, kernel, managers, sparse_points, weights);

  //size_t i_center{0};
  //for (auto manager : managers) {
  //  math::Matrix_t ee =
  //      energies.block(i_center, 0, 1, 1);
  //  std::cout << "ee shape: " << ee.rows() << ", " << ee.cols() << std::endl; 
 
  //  auto && gradients{*manager->template get_property<
  //      Property<double, 1, Manager_t, 1, ThreeD>>(force_name, true)};
  //  math::Matrix_t ff = Eigen::Map<const math::Matrix_t>(
  //      gradients.view().data(), manager->size() * ThreeD, 1);
  //  std::cout << "ff shape: " << ff.rows() << ", " << ff.cols() << std::endl; 

  //  auto && neg_stress{
  //      *manager->template get_property<Property<double, 0, Manager_t, 6>>(
  //          neg_stress_name, true)};
  //  math::Matrix_t ff_stress =
  //     Eigen::Map<const math::Matrix_t>(neg_stress.view().data(), 6, 1);
  //  std::cout << "ff_stress shape: " << ff_stress.rows() << ", " << ff_stress.cols() << std::endl; 

  //  i_center += manager->size() * ThreeD;
  //}

  //for (auto manager : managers) {
  //  for (auto atom : manager->with_ghosts()) {
  //    std::cout << "atom " << atom.get_atom_tag() << " global index "
  //              << atom.get_global_index() << std::endl;
  //  }
  //}
}
