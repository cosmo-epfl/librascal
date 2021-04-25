/**
 * @file   sandbox/playground.cc
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   26 June 2019
 *
 * @brief an executable to test ideas
 *
 * Copyright © 2019 Felix Musil, COSMO (EPFL), LAMMM (EPFL)
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

#include "rascal/models/sparse_kernels.hh"
#include "rascal/models/sparse_points.hh"
#include "rascal/models/numerical_kernel_gradients.hh"
#include "rascal/utils/basic_types.hh"
#include "rascal/models/kernels.hh"
#include "rascal/utils/utils.hh"
#include "rascal/representations/calculator_sorted_coulomb.hh"
#include "rascal/representations/calculator_spherical_expansion.hh"
#include "rascal/representations/calculator_spherical_invariants.hh"
#include "rascal/structure_managers/adaptor_increase_maxorder.hh"
#include "rascal/structure_managers/cluster_ref_key.hh"
#include "rascal/structure_managers/adaptor_center_contribution.hh"
#include "rascal/structure_managers/adaptor_half_neighbour_list.hh"
#include "rascal/structure_managers/adaptor_neighbour_list.hh"
#include "rascal/structure_managers/adaptor_strict.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/structure_manager_centers.hh"
#include "rascal/structure_managers/structure_manager_collection.hh"
#include "rascal/structure_managers/adaptor_kspace.hh"

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <list>
#include <random>
#include <string>
#include <algorithm>
#include <iterator>
#include <chrono>

using namespace rascal;  // NOLINT

using ManagerTypeHolder_t = StructureManagerTypeHolder<
                      StructureManagerCenters, AdaptorKspace>;

using Manager_t = typename ManagerTypeHolder_t::type;
using Representation_t = CalculatorSphericalInvariants;
using ManagerCollection_t =
    typename TypeHolderInjector<ManagerCollection, ManagerTypeHolder_t::type_list>::type;
using Representation_t = CalculatorSphericalInvariants;
using Prop_t = typename Representation_t::template Property_t<Manager_t>;
using PropGrad_t = typename Representation_t::template PropertyGradient_t<Manager_t>;

class Timer {
 private:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_t> m_beg;

 public:
	Timer() : m_beg(clock_t::now()) { }

	void reset() {
		m_beg = clock_t::now();
	}

	double elapsed() const {
		return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
	}
};

inline double std_dev(const rascal::math::Vector_t& vec) {
  double std_dev = std::sqrt((vec.array() - vec.mean()).array().square().sum()/(vec.size()-1));
  return std_dev;
}

constexpr static size_t ClusterLayer_{
          Manager_t::template cluster_layer_from_order<2>()};

int main(int argc, char * argv[]) {
  if (argc < 2) {
    std::cerr << "Must provide setup json filename as argument";
    std::cerr << std::endl;
    return -1;
  }

  json input = json_io::load(argv[1]);

  std::string filename{input["filename"].get<std::string>()};
  json adaptors = input["adaptors"].get<json>();
  json calculator = input["calculator"].get<json>();
  SparseKernel kernel{input["kernel"]};

  ManagerCollection_t managers{adaptors};

  managers.add_structures(filename, 0, input["n_structures"].get<int>());

  for (auto manager : managers) {
    std::cout << "Nb neigh: " << manager->get_nb_clusters(2) << std::endl;
    for (auto center : manager) {
      int atom_tag = center.get_atom_tag();
      int atom_type = center.get_atom_type();
      auto pos = center.get_position();
      std::cout << "tag: " <<atom_tag
                << " type: " << atom_type
                << " pos: " << pos.transpose()
                << std::endl;
    }

    std::cout << std::endl;
    for (auto center : manager) {
      for (auto neigh : center.pairs()) {
        int atom_tag = neigh.get_atom_tag();
        int atom_type = neigh.get_atom_type();
        auto pos = neigh.get_position();
        auto atom_j = neigh.get_atom_j();
        std::cout << "tag: " <<atom_tag
                << " type: " << atom_type
                << " atom_j: " << atom_j.get_atom_tag()
                << " pos: " << pos.transpose()
                << std::endl;
      }
    }

  }
  // Representation_t representation{calculator};
  // auto N_ITERATIONS{input["N_ITERATIONS"].get<int>()};
  // representation.compute(managers);

  // SparsePointsBlockSparse<Representation_t> sparse_points{};

  // auto selected_ids = input["selected_ids"].get<std::vector<std::vector<int>>>();

  // sparse_points.push_back(representation, managers, selected_ids);
  // auto KNM_der{kernel.compute_derivative(representation, managers, sparse_points)};
  // math::Vector_t elapsed{N_ITERATIONS};
  // Timer timer{};

  // for (int looper{0}; looper < N_ITERATIONS; looper++) {
  //   timer.reset();
  //   auto KNM_der_{kernel.compute_derivative(representation, managers, sparse_points)};
  //   elapsed[looper] = timer.elapsed();
  //   KNM_der_(0,0) = 1;
  // }
  // std::cout << elapsed.mean() << ", "<<std_dev(elapsed) << std::endl;

  // json results{};
  // results["elapsed_mean"] = elapsed.mean();
  // results["elapsed_std"] = std_dev(elapsed);
  // results["elapsed"] = elapsed;
  // std::ofstream o(argv[2]);
  // o << std::setw(2) << results << std::endl;

  // make playground -j4 && sandbox/playground ../sandbox/soap_input_molecular_crystal.json ../sandbox/res_ref.json
  // clear && make -j4 test_sparse_kernels  && ctest -R "test_sparse_kernels*" -V

  // auto KNM_num_der{compute_numerical_kernel_gradients(kernel, representation, managers, sparse_points, input["h"].get<double>())};
  // auto diff = math::relative_error(KNM_der, KNM_num_der);

  // std::cout << diff.row(0) << std::endl;
  // std::cout << "============================" << std::endl;
  // std::cout << KNM_der.row(0)<< std::endl;
  // std::cout << "============================" << std::endl;
  // std::cout << KNM_num_der.row(0)<< std::endl;
  // std::cout << "============================" << std::endl;

  // for (auto manager : managers) {
  //   auto && soap_vector_gradients{*manager->template get_property<PropGrad_t>(
  //       representation.get_gradient_name(), true, true)};
  //       std::cout << soap_vector_gradients.sum() << std::endl;
  // }

}
