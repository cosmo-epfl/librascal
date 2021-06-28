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

  // predict gradient, stress
  
  // manual
  //const bool compute_stress{false};
  //math::Matrix_t KNM_der{kernel.compute_derivative(calculator, managers,
  //                                       sparse_points, compute_stress)};
  //math::Matrix_t gradients_k = KNM_der * weights.transpose();
  //std::cout << "gradients_k shape: " << gradients_k.rows() << ", " << gradients_k.cols() << std::endl; 
  math::Matrix_t KNM{kernel.compute(calculator, managers, sparse_points)};
  math::Matrix_t energies = KNM * weights.transpose();

  std::string force_name = compute_sparse_kernel_gradients(
          calculator, kernel, managers, sparse_points, weights);

  std::string neg_stress_name = compute_sparse_kernel_neg_stress(
      calculator, kernel, managers, sparse_points, weights);

  size_t i_center{0};
  for (auto manager : managers) {
    math::Matrix_t ee =
        energies.block(i_center, 0, 1, 1);
    std::cout << "ee shape: " << ee.rows() << ", " << ee.cols() << std::endl; 
 
    auto && gradients{*manager->template get_property<
        Property<double, 1, Manager_t, 1, ThreeD>>(force_name, true)};
    math::Matrix_t ff = Eigen::Map<const math::Matrix_t>(
        gradients.view().data(), manager->size() * ThreeD, 1);
    std::cout << "ff shape: " << ff.rows() << ", " << ff.cols() << std::endl; 

    auto && neg_stress{
        *manager->template get_property<Property<double, 0, Manager_t, 6>>(
            neg_stress_name, true)};
    math::Matrix_t ff_stress =
       Eigen::Map<const math::Matrix_t>(neg_stress.view().data(), 6, 1);
    std::cout << "ff_stress shape: " << ff_stress.rows() << ", " << ff_stress.cols() << std::endl; 

    i_center += manager->size() * ThreeD;
  }

  for (auto manager : managers) {
    for (auto atom : manager->with_ghosts()) {
      std::cout << "atom " << atom.get_atom_tag() << " global index "
                << atom.get_global_index() << std::endl;
    }
  }


  //size_t i_center{0};
  //for (auto manager : managers) {
  //  auto && gradients{*manager->template get_property<
  //      Property<double, 1, Manager_t, 1, ThreeD>>(force_name, true)};
  //  math::Matrix_t ff = Eigen::Map<const math::Matrix_t>(
  //      gradients.view().data(), manager->size() * ThreeD, 1);
  //  math::Matrix_t ff_r =
  //      gradients_k.block(i_center, 0, manager->size() * ThreeD, 1);
  //  std::cout << "ff shape: " << ff.rows() << ", " << ff.cols() << std::endl; 
  //  std::cout << "ff: " << std::endl << ff << std::endl; 
  //  std::cout << "ff_r shape: " << ff_r.rows() << ", " << ff_r.cols() << std::endl; 
  //  std::cout << "ff_r: " << std::endl << ff_r << std::endl; 
  //}



  // TODO vector -> Eigen

  //json adaptors_input = input.at("adaptors").template get<json>();

  //// rascal initalize model
  //Kernel_t kernel{kernel_input};
  //kernel_input.at("target_type") = "Atom";
  //Kernel_t kernel_num{kernel_input};
  //SparsePoints_t sparse_points{};

  //Calculator_t representation{calculator_input};
  //// load structures, compute representation and fill sparse points
  //// This requires an equivalent for StructureManagerLammps input 
  ////managers.add_structures(filename, 0,
  ////                        input.at("n_structures").template get<int>());
  //// requires fnu
  ////sparse_points.push_back(representation, managers, selected_ids);
  //calculator_input["compute_gradients"] = false;
  //Representation_t representation_{calculator_input};

  // ManagerCollection_t managers{adaptors_input};





  //double cutoff{4.};
  //json hypers{{"max_radial", 3},
  //            {"max_angular", 2},
  //            {"compute_gradients", true},
  //            {"soap_type", "PowerSpectrum"},
  //            {"normalize", true}};

  //json fc_hypers{{"type", "ShiftedCosine"},
  //               {"cutoff", {{"value", cutoff}, {"unit", "AA"}}},
  //               {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}}};
  //json sigma_hypers{{"type", "Constant"},
  //                  {"gaussian_sigma", {{"value", 0.4}, {"unit", "AA"}}}};

  //hypers["cutoff_function"] = fc_hypers;
  //hypers["gaussian_density"] = sigma_hypers;
  //hypers["radial_contribution"] = {{"type", "GTO"}};

  //json structure{{"filename", filename}};
  //json adaptors;
  //json ad1{{"name", "AdaptorNeighbourList"},
  //         {"initialization_arguments", {{"cutoff", cutoff}}}};
  //json ad1b{{"name", "AdaptorCenterContribution"},
  //          {"initialization_arguments", {}}};
  //json ad2{{"name", "AdaptorStrict"},
  //         {"initialization_arguments", {{"cutoff", cutoff}}}};
  //adaptors.emplace_back(ad1);
  //adaptors.emplace_back(ad1b);
  //adaptors.emplace_back(ad2);
  //auto manager =
  //    make_structure_manager_stack<StructureManagerCenters,
  //                                 AdaptorNeighbourList,
  //                                 AdaptorCenterContribution, AdaptorStrict>(
  //        structure, adaptors);

  //Representation_t representation{hypers};

  //representation.compute(manager);

  //constexpr size_t n_centers_print{4};
  //constexpr size_t n_neigh_print{1};

  //// Print the first few elements and gradients, so we know we're getting
  //// something
  //std::cout << "Expansion of first " << n_centers_print << " centers:";
  //std::cout << std::endl;
  //std::cout << "Note that the coefficients are printed with species pairs along"
  //             " the columns and n-n'-l along the rows."
  //          << std::endl;
  //std::cout << "Gradients are printed with: First Cartesian component, "
  //             "then species pairs, along the columns; n-n'-l along the rows.";
  //std::cout << std::endl;
  //size_t center_count{0};

  //auto && soap_vectors{
  //    *manager->template get_property<Prop_t>(representation.get_name())};
  //auto && soap_vector_gradients{*manager->template get_property<PropGrad_t>(
  //    representation.get_gradient_name())};

  //for (auto center : manager) {
  //  if (center_count >= n_centers_print) {
  //    break;
  //  }
  //  size_t n_species_center{soap_vectors.get_keys(center).size()};
  //  std::cout << "============================" << std::endl;
  //  std::cout << "Center " << center.get_index();
  //  std::cout << " of type " << center.get_atom_type() << std::endl;
  //  std::cout << soap_vectors.get_dense_row(center);
  //  std::cout << std::endl;
  //  auto keys_center = soap_vectors[center].get_keys();
  //  std::cout << "Center data keys: ";
  //  for (auto key : keys_center) {
  //    std::cout << "(";
  //    for (auto key_sp : key) {
  //      std::cout << key_sp << ", ";
  //    }
  //    std::cout << "\b\b) ";
  //  }
  //  std::cout << std::endl;
  //  auto ii_pair = center.get_atom_ii();
  //  auto keys_grad_center = soap_vector_gradients[ii_pair].get_keys();
  //  std::cout << "Center gradient keys: ";
  //  for (auto key : keys_grad_center) {
  //    std::cout << "(";
  //    for (auto key_sp : key) {
  //      std::cout << key_sp << ", ";
  //    }
  //    std::cout << "\b\b) ";
  //  }
  //  std::cout << std::endl;
  //  std::cout << "Gradient of this expansion wrt center pos: " << std::endl;
  //  // clang-format off
  //  // makes an absolute mess of the below
  //  std::cout << Eigen::Map<Eigen::MatrixXd>(
  //         soap_vector_gradients.get_dense_row(ii_pair).data(),
  //         3 * n_species_center,
  //         soap_vector_gradients.get_nb_comp())
  //    .transpose();
  //  // clang-format on
  //  std::cout << std::endl;
  //  size_t neigh_count{0};
  //  for (auto neigh : center.pairs()) {
  //    if (neigh_count >= n_neigh_print) {
  //      break;
  //    }
  //    auto keys_neigh = soap_vector_gradients[neigh].get_keys();
  //    std::cout << "Neighbour keys: ";
  //    for (auto key : keys_neigh) {
  //      std::cout << "(";
  //      for (auto key_sp : key) {
  //        std::cout << key_sp << ", ";
  //      }
  //      std::cout << "\b\b) ";
  //    }
  //    std::cout << std::endl;
  //    std::cout << "Gradient of the above wrt atom " << neigh.back();
  //    std::cout << " of type " << neigh.get_atom_type() << std::endl;
  //    // clang-format off
  //    std::cout << Eigen::Map<Eigen::MatrixXd>(
  //        soap_vector_gradients.get_dense_row(neigh).data(),
  //        3 * n_species_center,
  //        soap_vector_gradients.get_nb_comp())
  //      .transpose();
  //    // clang-format on
  //    std::cout << std::endl;
  //    ++neigh_count;
  //  }
  //  ++center_count;
  //}
}
