/**
 * @file   spherical_harmonics.cc
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 * @author  Max Veit <max.veit@epfl.ch>
 * @author  Alex
 *
 * @date   14 October 2018
 *
 * @brief implementation of the spherical harmonics, optimized, with gradients
 *
 * Copyright  2018  Felix Musil, Max Veit, COSMO (EPFL), LAMMM (EPFL)
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

#include "rascal/math/spherical_harmonics.hh"

#include <iostream>

using namespace rascal::math;  // NOLINT

void SphericalHarmonics::precompute(size_t max_angular,
                                    bool calculate_derivatives) {
  this->max_angular = max_angular;
  this->harmonics =
      Vector_t::Zero((this->max_angular + 1) * (this->max_angular + 1));
  // Note the ALPs need an extra zero column to allow natural use of the
  // raising and lowering operators (specifically P_l^(l+1), which should
  // give zero).
  this->assoc_legendre_polynom =
      Matrix_t::Zero(this->max_angular + 1, this->max_angular + 2);
  this->coeff_a =
      Matrix_t::Zero(this->max_angular + 1, 2 * this->max_angular + 1);
  this->coeff_b =
      Matrix_t::Zero(this->max_angular + 1, 2 * this->max_angular + 1);
  this->cos_sin_m_phi = MatrixX2_t::Zero(this->max_angular + 1, 2);

  for (size_t angular_l{0}; angular_l < this->max_angular + 1; angular_l++) {
    double lsq = angular_l * angular_l;
    double lm1sq = (angular_l - 1) * (angular_l - 1);
    // TODO(alex) vectorize
    for (size_t m_count{0}; m_count < angular_l + 1; m_count++) {
      double msq = m_count * m_count;
      this->coeff_a(angular_l, m_count) =
          std::sqrt((4 * lsq - 1.0) / (lsq - msq));
      this->coeff_b(angular_l, m_count) =
          -1.0 * std::sqrt((lm1sq - msq) / (4 * lm1sq - 1.0));
    }
  }
  // this->angular_coeffs1.reserve(this->max_angular);
  this->angular_coeffs1 = Vector_t::Zero(this->max_angular + 1);
  this->angular_coeffs2 = Vector_t::Zero(this->max_angular + 1);
  for (size_t angular_l{0}; angular_l < this->max_angular + 1; angular_l++) {
    angular_coeffs1(angular_l) = std::sqrt(2 * angular_l + 1);
    angular_coeffs2(angular_l) = -std::sqrt(1.0 + 0.5 / angular_l);
  }

  // We want to precompute derivative information in almost any case
  if (calculate_derivatives or this->calculate_derivatives) {
    if (not this->calculate_derivatives) {
      this->calculate_derivatives = true;
    }
    this->harmonics_derivatives =
        Matrix_t::Zero(3, math::pow(this->max_angular + 1, 2));
    this->plm_factors =
        Matrix_t::Zero(this->max_angular + 1, this->max_angular + 1);
    // TODO(alex) can be done with broadcasting and setting half of matrix
    // to zero, but we are actually only calculating the rectangular part,
    // what is faster?
    for (size_t angular_l{0}; angular_l < this->max_angular + 1; angular_l++) {
      Vector_t m_counts =
          Eigen::VectorXd::LinSpaced(angular_l + 1, 0, angular_l);
      this->plm_factors.row(angular_l).head(angular_l + 1) =
          ((angular_l - m_counts.array()) * (angular_l + m_counts.array() + 1))
              .sqrt();
    }
    this->legendre_polynom_differences = Vector_t::Zero(this->max_angular);
    this->phi_derivative_factors = Vector_t::Zero(this->max_angular);
    this->derivatives_precomputed = true;
  }
}

void SphericalHarmonics::compute_assoc_legendre_polynom(double cos_theta) {
  double sin_theta = std::sqrt(1.0 - math::pow(cos_theta, 2));
  const double SQRT_INV_2PI = std::sqrt(0.5 / PI);
  // Compute the associated Legendre polynomials: l < 2 are special cases
  // These include the normalization factors usually needed in the
  // spherical harmonics
  double l_accum{SQRT_INV_2PI};
  this->assoc_legendre_polynom(0, 0) = SQRT_INV_2PI;
  if (this->max_angular > 0) {
    this->assoc_legendre_polynom(1, 0) = cos_theta * SQRT_THREE * SQRT_INV_2PI;
    l_accum = l_accum * -std::sqrt(3.0 / 2.0) * sin_theta;
    this->assoc_legendre_polynom(1, 1) = l_accum;
  }
  for (size_t angular_l{2}; angular_l < this->max_angular + 1; angular_l++) {
    // for l > 1 : Use the recurrence relation
    // TODO(max-veit) don't bother calculating m =/= 0 if sin(theta) == 0
    //                (z-axis)
    // avoid making temp by breaking down the operation in 3 parts
    this->assoc_legendre_polynom.row(angular_l).head(angular_l - 1).array() =
        cos_theta * this->assoc_legendre_polynom.row(angular_l - 1)
                        .head(angular_l - 1)
                        .array();
    this->assoc_legendre_polynom.row(angular_l).head(angular_l - 1).array() +=
        this->coeff_b.row(angular_l).head(angular_l - 1).array() *
        this->assoc_legendre_polynom.row(angular_l - 2)
            .head(angular_l - 1)
            .array();
    this->assoc_legendre_polynom.row(angular_l).head(angular_l - 1).array() *=
        this->coeff_a.row(angular_l).head(angular_l - 1).array();

    this->assoc_legendre_polynom(angular_l, angular_l - 1) =
        // cos_theta * std::sqrt(2 * angular_l + 1) * l_accum;
        l_accum * cos_theta * this->angular_coeffs1(angular_l);
    l_accum = l_accum * sin_theta * this->angular_coeffs2(angular_l);
    this->assoc_legendre_polynom(angular_l, angular_l) = l_accum;
  }
}

void SphericalHarmonics::calc(
    const Eigen::Ref<const Eigen::Vector3d> & direction,
    bool calculate_derivatives) {
  Eigen::Vector3d direction_normed;
  if (std::abs((direction[0] * direction[0] + direction[1] * direction[1] +
                direction[2] * direction[2]) -
               1.0) > math::DBL_FTOL) {
    std::cerr << "Warning: SphericalHarmonics::calc()";
    std::cerr << ": Direction vector unnormalized, normalizing it now";
    std::cerr << std::endl;
    direction_normed = direction / direction.norm();
  } else {
    direction_normed = direction;
  }

  // The cosine against the z-axis is just the z-component of the
  // direction vector
  double cos_theta = direction_normed[2];
  // The less efficient, but more intuitive implementation:
  // double phi = std::atan2(direction[1], direction[0]);
  double sqrt_xy = std::hypot(direction_normed[0], direction_normed[1]);
  // For a vector along the z-axis, define phi=0
  double cos_phi{1.0}, sin_phi{0.0};
  if (sqrt_xy >= math::DBL_FTOL) {
    cos_phi = direction_normed[0] / sqrt_xy;
    sin_phi = direction_normed[1] / sqrt_xy;
  }

  this->compute_assoc_legendre_polynom(cos_theta);
  this->compute_cos_sin_angle_multiples(cos_phi, sin_phi);

  this->compute_spherical_harmonics();
  if (calculate_derivatives) {
    if (this->derivatives_precomputed) {
      // A rose, by any other name, would have the same exact value
      // double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);
      double sin_theta{sqrt_xy};
      this->compute_spherical_harmonics_derivatives(sin_theta, cos_theta,
                                                    sin_phi, cos_phi);
    } else {
      // TODO(max) do we just precompute here instead of throwing a rude
      // error?
      std::stringstream err_str{};
      err_str << "Resources for computation of dervatives have not been "
                 "initialized. Please set calculate_derivatives flag on "
                 "construction of the SphericalHarmonics object or during "
                 "precomputation.";
      throw std::runtime_error(err_str.str());
    }
  }
}

void SphericalHarmonics::compute_cos_sin_angle_multiples(double cos_phi,
                                                         double sin_phi) {
  // computes iteratively a list of (cos(m phi), sin(m phi))
  // uses a modified iteration that yields (-1)^m(cos(m phi), sin(m phi)),
  // that has the right sign to get real-valued sph with the usual
  // phase convention
  for (size_t m_count{0}; m_count < this->max_angular + 1; m_count++) {
    if (m_count == 0) {
      this->cos_sin_m_phi.row(m_count) << 1.0, 0.0;
    } else if (m_count == 1) {
      // standard iter: this->cos_sin_m_phi.row(m_count) << cos_phi, sin_phi;
      this->cos_sin_m_phi.row(m_count) << -cos_phi, -sin_phi;
    } else {
      /* standard iter:
      this->cos_sin_m_phi.row(m_count) =
          2.0 * cos_phi * this->cos_sin_m_phi.row(m_count - 1) -
          this->cos_sin_m_phi.row(m_count - 2);
      */
      this->cos_sin_m_phi.row(m_count) =
          -2.0 * cos_phi * this->cos_sin_m_phi.row(m_count - 1) -
          this->cos_sin_m_phi.row(m_count - 2);
    }
  }
}

void SphericalHarmonics::compute_spherical_harmonics() {
  size_t lm_base{0};  // starting point for storage
  for (size_t angular_l{0}; angular_l < this->max_angular + 1; angular_l++) {
    // uses symmetry of spherical harmonics,
    // careful with the storage order
    // TODO(alex) please clarify -- is it the same storage order we had
    // before (i.e. negative-m components first)?  If so, please refer to
    // the above documentation or delete this comment -- Max
    this->harmonics.segment(lm_base + angular_l, angular_l + 1) =
        this->assoc_legendre_polynom.row(angular_l)
            .head(angular_l + 1)
            .array() *
        cos_sin_m_phi.col(0).head(angular_l + 1).transpose().array();
    this->harmonics.segment(lm_base, angular_l + 1) =
        (this->assoc_legendre_polynom.row(angular_l)
             .head(angular_l + 1)
             .array() *
         cos_sin_m_phi.col(1).head(angular_l + 1).transpose().array())
            .reverse();
    this->harmonics(lm_base + angular_l) =
        this->assoc_legendre_polynom(angular_l, 0) * INV_SQRT_TWO;
    lm_base += 2 * angular_l + 1;
  }  // for (l in [0, lmax])
}

void SphericalHarmonics::compute_spherical_harmonics_derivatives(
    double sin_theta, double cos_theta, double sin_phi, double cos_phi) {
  // angular_l = 0
  // d/dx, d/dy, d/dz
  this->harmonics_derivatives.col(0).setZero();

  // angular_l > 0
  size_t l_block_index{1};
  for (size_t angular_l{1}; angular_l < this->max_angular + 1; angular_l++) {
    // legendre_polynom_difference
    legendre_polynom_differences.head(angular_l) =
        this->plm_factors.row(angular_l).segment(0, angular_l).array() *
            this->assoc_legendre_polynom.row(angular_l)
                .segment(0, angular_l)
                .array() -
        this->plm_factors.row(angular_l).segment(1, angular_l).array() *
            this->assoc_legendre_polynom.row(angular_l)
                .segment(2, angular_l)
                .array();
    // TODO(alex) this if then else could be optimized, but I guess the
    // compiler does it.
    // phi_derivative_factor
    if (sin_theta > 0.1) {
      // TODO(alex) could be precomputed
      Vector_t m_counts = Eigen::VectorXd::LinSpaced(angular_l, 1, angular_l);
      // singularity at the poles
      phi_derivative_factors.head(angular_l) =
          m_counts.head(angular_l).array() / sin_theta *
          this->assoc_legendre_polynom.row(angular_l)
              .segment(1, angular_l)
              .array();
    } else {
      // singularity at the equator
      phi_derivative_factors.head(angular_l) =
          -0.5 / cos_theta *
          (this->plm_factors.row(angular_l).segment(0, angular_l).array() *
               this->assoc_legendre_polynom.row(angular_l)
                   .segment(0, angular_l)
                   .array() +
           this->plm_factors.row(angular_l).segment(1, angular_l).array() *
               this->assoc_legendre_polynom.row(angular_l)
                   .segment(2, angular_l)
                   .array());
    }

    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //                            ‾‾‾‾‾‾‾

    // d/dx
    this->harmonics_derivatives(0, l_block_index + angular_l) =
        cos_theta * cos_phi * this->plm_factors(angular_l, 0) * INV_SQRT_TWO *
        this->assoc_legendre_polynom(angular_l, 1);
    // d/dy
    this->harmonics_derivatives(1, l_block_index + angular_l) =
        cos_theta * sin_phi * this->plm_factors(angular_l, 0) * INV_SQRT_TWO *
        this->assoc_legendre_polynom(angular_l, 1);
    // d/dz
    this->harmonics_derivatives(2, l_block_index + angular_l) =
        -1.0 * sin_theta * this->plm_factors(angular_l, 0) * INV_SQRT_TWO *
        this->assoc_legendre_polynom(angular_l, 1);

    // d/dx
    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //                                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    this->harmonics_derivatives.row(0).segment(l_block_index + angular_l + 1,
                                               angular_l) =
        sin_phi * phi_derivative_factors.head(angular_l).array() *
        this->cos_sin_m_phi.col(1).segment(1, angular_l).transpose().array();

    this->harmonics_derivatives.row(0)
        .segment(l_block_index + angular_l + 1, angular_l)
        .array() +=
        -0.5 * cos_theta * cos_phi *
        this->cos_sin_m_phi.col(0).segment(1, angular_l).transpose().array() *
        legendre_polynom_differences.head(angular_l).array();
    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    this->harmonics_derivatives.row(0)
        .segment(l_block_index, angular_l)
        .reverse() =
        -1.0 * sin_phi * phi_derivative_factors.head(angular_l).array() *
        this->cos_sin_m_phi.col(0).segment(1, angular_l).transpose().array();
    this->harmonics_derivatives.row(0)
        .segment(l_block_index, angular_l)
        .reverse()
        .array() +=
        -0.5 * cos_theta * cos_phi *
        this->cos_sin_m_phi.col(1).segment(1, angular_l).transpose().array() *
        legendre_polynom_differences.head(angular_l).array();

    // d/dy
    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //                                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    this->harmonics_derivatives.row(1).segment(l_block_index + angular_l + 1,
                                               angular_l) =
        -1.0 * cos_phi * phi_derivative_factors.head(angular_l).array() *
        this->cos_sin_m_phi.col(1).segment(1, angular_l).transpose().array();
    this->harmonics_derivatives.row(1)
        .segment(l_block_index + angular_l + 1, angular_l)
        .array() +=
        -0.5 * cos_theta * sin_phi *
        this->cos_sin_m_phi.col(0).segment(1, angular_l).transpose().array() *
        legendre_polynom_differences.head(angular_l).array();
    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    this->harmonics_derivatives.row(1)
        .segment(l_block_index, angular_l)
        .reverse() =
        cos_phi * phi_derivative_factors.head(angular_l).array() *
        this->cos_sin_m_phi.col(0).segment(1, angular_l).transpose().array();
    this->harmonics_derivatives.row(1)
        .segment(l_block_index, angular_l)
        .reverse()
        .array() +=
        -0.5 * cos_theta * sin_phi *
        this->cos_sin_m_phi.col(1).segment(1, angular_l).transpose().array() *
        legendre_polynom_differences.head(angular_l).array();
    // d/dz
    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //                                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    this->harmonics_derivatives.row(2).segment(l_block_index + angular_l + 1,
                                               angular_l) =
        0.5 * sin_theta *
        this->cos_sin_m_phi.col(0).segment(1, angular_l).transpose().array() *
        legendre_polynom_differences.head(angular_l).array();
    // (l_block - angular_l,... , l_block, ..., l_block + angular_l)
    //  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    this->harmonics_derivatives.row(2)
        .segment(l_block_index, angular_l)
        .reverse() =
        0.5 * sin_theta *
        this->cos_sin_m_phi.col(1).segment(1, angular_l).transpose().array() *
        legendre_polynom_differences.head(angular_l).array();

    l_block_index += (2 * angular_l + 1);
  }  // for (l in [0, lmax])
}
