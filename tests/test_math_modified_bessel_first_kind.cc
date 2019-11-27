/**
 * @file   test_math_modified_bessel_first_kind.cc
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 *
 * @date   21 May 2019
 *
 * @brief Test the implementation of modified bessel first kind
 *
 * Copyright © 2019  Felix Musil, COSMO (EPFL), LAMMM (EPFL)
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

#include "test_math.hh"

#include <boost/test/unit_test.hpp>

namespace rascal {

  BOOST_AUTO_TEST_SUITE(MathBesselFirstKindTests);

  /* ----------------------------------------------------------------------
   */
  /**
   * Check the implementation of the modified bessel function of the first
   * kind against mpmath v1.1.0
   */
  BOOST_FIXTURE_TEST_CASE(math_bessel_test, ModifiedBesselFirstKindRefFixture) {
    json & i_complete_square_ref{this->ref_data["i_complete_square"]};
    for (auto & data : i_complete_square_ref) {
      math::ModifiedSphericalBessel j_v_complete_square{};
      auto xs{data["xs"].get<std::vector<double>>()};
      Eigen::Map<Eigen::ArrayXd> xns(&xs[0], xs.size());
      double alpha{data["alpha"]}, rij{data["rij"]};
      auto ref_vals{data["vals"].get<std::vector<std::vector<double>>>()};
      size_t max_order{data["max_order"]};
      j_v_complete_square.precompute(max_order - 1, xns);
      j_v_complete_square.calc(rij, alpha);
      auto vals{j_v_complete_square.get_values()};

      // test that values are either > 1e-100 or 0
      for (size_t i_x{0}; i_x < xs.size(); ++i_x) {
        for (size_t order{0}; order < max_order; ++order) {
          if (vals(i_x, order) > 0) {
            BOOST_TEST(vals(i_x, order) > 9e-101);
          }
        }
      }

      for (size_t i_x{0}; i_x < xs.size(); ++i_x) {
        for (size_t order{0}; order < max_order; ++order) {
          double rel_error{0.};
          if (ref_vals[i_x][order] > 0) {
            rel_error = std::abs(vals(i_x, order) - ref_vals[i_x][order]) /
                        ref_vals[i_x][order];
          } else {
            rel_error = ref_vals[i_x][order];
          }

          // the MBSFs are set to 0 if < 1e-100 so no point checking with
          // the reference then. Note the MBSFs are very accurate when < 1e-200
          if (ref_vals[i_x][order] > 1e-90 and vals(i_x, order) > 1e-90) {
            if ((rel_error > 1e3 * math::DBL_FTOL) and this->verbose) {
              std::cout << " order=" << order << " x=" << xs[i_x]
                        << " alpha=" << alpha << " rij=" << rij
                        << " diff=" << rel_error
                        << " ref=" << ref_vals[i_x][order]
                        << " val=" << vals(i_x, order) << std::endl;
            }

            BOOST_CHECK_LE(rel_error, 1e3 * math::DBL_FTOL);
          }
        }
      }
    }
  }

  BOOST_AUTO_TEST_CASE(MBFs_gradient_test) {
    // use same range as in the reference test
    std::vector<size_t> max_angulars{{0, 20}};
    std::vector<double> alphas{{0.6, 3.5, 8.5, 20, 30, 50}};
    Eigen::ArrayXd xs = Eigen::ArrayXd::LinSpaced(20, 0.005, 10);

    GradientTestFixture fix{"reference_data/inputs/mbfs_derivative_test.json"};

    for (const auto & max_angular : max_angulars) {
      for (const auto & alpha : alphas) {
        ModifiedBesselFirstKindGradientsProvider mbf_calculator{xs, alpha,
                                                                max_angular};
        test_gradients(mbf_calculator, fix);
      }
    }
  }

  /* ----------------------------------------------------------------------
   */
  BOOST_AUTO_TEST_SUITE_END();

}  // namespace rascal
