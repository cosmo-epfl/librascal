/**
 * file   test_math.hh
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 *
 * @date   14 October 2018
 *
 * @brief Test implementation of math functions
 *
 * Copyright © 2018  Felix Musil, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef TEST_MATH_H
#define TEST_MATH_H

#include "tests.hh"
#include "json_io.hh"
#include "math/math_interface.hh"
#include "math/math_utils.hh"

#include <fstream>
#include <Eigen/Dense>

namespace rascal {

  struct ManagerFixture_math
  {
    ManagerFixture_math()
      :numbers(4,3),results_hyp2f1(3),results_airy(3,4)
    {
      numbers <<
        1, 0.1, 2,
        1, 3, 9,
        2, 7, 6,
        0.5, 0.2, 0.3;

      results_hyp2f1 <<
        1.3862943611198901 ,1.0090833356005495 ,3.0875740550280937;
      results_airy <<
        0.13529241631288147, -0.15914744129679328, 1.2074235949528715,
        0.9324359333927756, 0.329203129943538, -0.2571304219075862,
        0.659861690194189, 0.45151263114964657, 0.03492413042327436,
        -0.05309038443365388, 3.2980949999782143, 4.10068204993289;
    }

    ~ManagerFixture_math() {}

    Eigen::Matrix<double, 4, Eigen::Dynamic>  numbers;
    Eigen::Matrix<double, 1, Eigen::Dynamic>  results_hyp2f1;
    Eigen::Matrix<double, 3, Eigen::Dynamic>  results_airy;
    bool vebose{false};

  };

  struct SphericalHarmonicsRefFixture
  {
    SphericalHarmonicsRefFixture() {
      json ref_data;
      std::ifstream ref_file(this->ref_filename);
      ref_file >> ref_data;
      unit_vectors = ref_data.at("unit_vectors").get<StdVector2D_t>();
      harmonics = ref_data.at("harmonics").get<StdVector3D_t>();
      alps = ref_data.at("alps").get<StdVector3D_t>();
    }

    ~SphericalHarmonicsRefFixture() = default;

    std::string ref_filename = "reference_data/spherical_harmonics_test.json";

    using StdVector2D_t = std::vector<std::vector<double>>;
    using StdVector3D_t = std::vector<std::vector<std::vector<double>>>;
    StdVector2D_t unit_vectors{};
    StdVector3D_t harmonics{};
    StdVector3D_t alps{};
    bool verbose{true};
  };
}  // rascal

#endif /* TEST_MATH_H */
