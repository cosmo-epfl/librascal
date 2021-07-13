/**
 * @file   rascal/math/spherical_harmonics.hh
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 * @author  Max Veit <max.veit@epfl.ch>
 * @author  Alexander Goscinski <alexander.goscinski@epfl.ch>
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

#ifndef SRC_RASCAL_MATH_SPHERICAL_HARMONICS_HH_
#define SRC_RASCAL_MATH_SPHERICAL_HARMONICS_HH_

#include "rascal/math/utils.hh"

namespace rascal {
  namespace math {
    /**
     * Compute a full set of spherical harmonics (optimized version)
     *
     * Follows the algorithm described in https://arxiv.org/abs/1410.1748
     * except for an additonal \f$(-1)^m\f$ factor to follow the wikipedia
     * article convention
     * https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form.
     * It effectively cancels out the Condon-Shortley phase in the final
     * real spherical harmonics.
     *
     * In brief, this class computes the real spherical harmonics where the
     * imaginary components of the usualn complex functions are instead stored
     * in the negative-m indices:
     *
     * \f{equation}{
     * Y_\ell^m(\theta, \phi) =
     * \begin{cases}
     * \sqrt{\frac{2\ell + 1}{2\pi} \frac{(\ell + m)!}{(\ell - m)!}}
     *     P_\ell^{-m}(\cos\theta) (-1)^m \sin(-m\phi) & \text{for } m < 0\\
     * \sqrt{\frac{2\ell + 1}{4\pi}} P_\ell(\cos\theta) & \text{for } m = 0\\
     * \sqrt{\frac{2\ell + 1}{2\pi} \frac{(\ell - m)!}{(\ell + m)!}}
     *      P_\ell^{m}(\cos\theta) (-1)^m \cos(m\phi) & \text{for } m > 0
     * \end{cases}
     * \f}
     * Note that \f$P_\ell^{m}\f$ includes the Condon-Shortley phase.
     * In case you're wondering why it's \f$\frac{1}{2π}\f$ on the \f$m \neq
     * 0\f$ components (instead of \f$\frac{1}{4π}\f$), there's an extra factor
     * of 1/2 that comes from integrating cos² or sin² over the full circle of
     * φ, so these are indeed normalized in the same sense as the complex
     * spherical harmonics:
     * \f[
     * \iint_{\mathcal{S}^2} dΩ\; Y_\ell^m(θ, \phi) Y_{\ell'}^{m'}(θ, \phi)
     *      = δ_{\ell\ell'}δ_{mm'}
     * \f]
     *
     * (this extra factor of \f$\frac{1}{\sqrt{2}}\f$ is not included in the
     * normalization of the associated Legendre polynomials defined above;
     * however, all other normalization factors are.)
     *
     * Cartesian gradients can optionally be computed in addition.
     *
     * Part of the efficiency derives from moving the direction-independent
     * calculations into a separate precompute() method; you must therefore call
     * precompute() (which also sets \f$\ell_\text{max}\f$) before any call to
     * calc().
     *
     * Once calc() has been called for a direction vector, the results can be
     * retrieved with get_harmonics() (and the gradients, if computed, with
     * get_harmonics_derivatives())
     *
     * Note the storage order of the harmonics components here follows the
     * "compressed" format, with the \f$\ell\f$ index varying the slowest, from
     * 0 to \f$\ell_\text{max}\f$, and \f$m\f$ varying from \f$-\ell\f$ to
     * \f$\ell\f$ for each value of \f$\ell\f$, for a total of
     * \f$(\ell_\text{max}+1)^2\f$ components.  For example, the first few
     * entries of the array would have the \f$(\ell, m)\f$ numbers: (0,0) (1,-1)
     * (1,0) (1, 1) (2, -2)....
     */
    class SphericalHarmonics {
     public:
      /**
       * Construct a SphericalHarmonics class with a default setting for
       * whether to calculate the gradients
       *
       * Remember to call precompute to finish initialization
       */
      explicit SphericalHarmonics(bool calculate_derivatives = false)
          : calculate_derivatives{calculate_derivatives} {}

      /**
       * Precomputes as many parameters as possible
       *
       * @param max_angular Maximum angular momentum number
       *                    \f$\ell_\text{max}\f$
       *
       * @param calculate_derivatives Whether to precompute for the derivatives
       *     as well.  If the internal default (set at construction) is false,
       *     but the value provided here is true, the internal default will be
       *     set to true.  If the value provided here is false (default
       *     argument) then the internal default is not changed; derivatives
       *     will be precomputed only if the internal default is true.
       */
      void precompute(size_t max_angular, bool calculate_derivatives = false);

      /**
       * Compute a full set of spherical harmonics given a direction vector.
       * If calculate_derivatives flag is on, the derivatives are additionally
       * computed. This function returns void, results have to be retrieved
       * with get functions.
       *
       * @param direction   unit vector defining the angles
       *                    (arguments for the \f$Y_\ell^m\f$)
       *
       * @param calculate_derivatives       Compute the gradients too?
       *
       * @warning Prints warning and normalizes direction if it is not
       *          already normalized.
       */
      void calc(const Eigen::Ref<const Eigen::Vector3d> & direction,
                bool calculate_derivatives);

      /**
       * Same as calc(), but using the internal default to decide whether to
       * compute derivatives.
       */
      void calc(const Eigen::Ref<const Eigen::Vector3d> & direction) {
        this->calc(direction, this->calculate_derivatives);
      }

      const Matrix_Ref get_assoc_legendre_polynom() {
        // Since for calculation purposes assoc_legendre_polynom has one column
        // more than it would have in standard libaries, we return only the
        // segment of size (max_angular+1, max_angular+1) as other standard
        // libaries.
        return Matrix_Ref(this->assoc_legendre_polynom)
            .topLeftCorner(this->max_angular + 1, this->max_angular + 1);
      }

      /// Returns the (max_angular+1, max_angular+2) matrix related to the
      /// associated legendre polynomial with only zeros entries in the last
      /// column. Used for testing purposes.
      const Matrix_Ref get_assoc_legendre_polynom_raw() {
        return Matrix_Ref(this->assoc_legendre_polynom);
      }

      /**
       * Access the computed spherical harmonics.
       *
       * @return  (Eigen)matrix containing the results.
       *          Sized \f$(\ell_\text{max}+1)^2\f$, the index collects
       *          \f$\ell\f$ and \f$m\f$ quantum numbers stored in compact
       *          format (\f$m\f$ varies fastest, from \f$-\ell\f$ to
       *          \f$\ell\f$, and \f$\ell\f$ from 0 to \f$\ell_\text{max}\f$).
       */
      const Vector_Ref get_harmonics() { return this->harmonics; }

      /**
       * Access the Cartesian gradients of the harmonics, if computed
       *
       * @todo (alex) what happens if they haven't been computed and this gets
       *              called anyway?
       *
       * @return  (Eigen)matrix containing the results.
       *          Sized 3 by \f$(\ell_\text{max}+1)^2\f$; the first index runs
       *          over the the x, y, and z gradient components and the second
       *          index collects \f$\ell\f$ and \f$m\f$ quantum numbers, stored
       *          in compact format (\f$m\f$ varies fastest, from \f$-\ell\f$ to
       *          \f$\ell\f$, and \f$\ell\f$ from 0 to \f$\ell_\text{max}\f$).
       */
      const Matrix_Ref get_harmonics_derivatives() {
        return this->harmonics_derivatives;
      }

     private:
      /**
       * Compute a set of normalized associated Legendre polynomials, namely
       * \f$P_\ell^{m}\f$.
       *
       * These are normalized for use in computing the real spherical harmonics;
       * see the class documentation for details.  In particular, the \f$m=0\f$
       * harmonics require an extra factor of \f$\frac{1}{\sqrt{2}}\f$.  The
       * negative-m functions are not computed due to symmetry.
       *
       * Note that \f$P_\ell^{m}\f$ includes the Condon-Shortley phase.
       *
       * @param cos_theta (aka x) Where to evaluate the polynomial
       *
       * Stores the results as an (Eigen)matrix containing the evaluated
       * polynomials, sized \f$\ell_\text{max}\f$ by \f$(2\ell_\text{max} +
       * 1)\f$; the row is indexed by \f$\ell\f$ and the column by \f$m \geq
       * 0\f$.
       */
      void compute_assoc_legendre_polynom(double cos_theta);

      /**
       * Compute \f$(-1)^m\cos(m\phi)\f$ and \f$(-1)^m\sin(m\phi)\f$ from the
       * recurrence relations
       *
       * The relations are (these are the same recurrence relations used to
       * calculate the Chebyshev polynomials):
       *
       * \f{eqnarray}{
       * \cos(m\phi) &=& 2\cos(\phi)\cos((m-1)\phi) - \cos((m-2)\phi)\\
       * \sin(m\phi) &=& 2\cos(\phi)\sin((m-1)\phi) - \sin((m-2)\phi)
       * \f}
       *
       * and they require only \f$\cos(\phi)\f$ and \f$\sin(\phi)\f$ to start.
       *
       * @param cos_phi Value of \f$\cos(\phi)\f$ to start the relation
       *
       * @param sin_phi Value of \f$\sin(\phi)\f$ to start the relation
       *
       * Stores the results as an (Eigen)matrix, sized \f$m_\text{max}\f$ by 2
       * with the \f$(-1)^m\cos(m\phi)\f$ stored in the first column and
       * \f$(-1)^m\sin(m\phi)\f$ in the second column, with \f$m\f$ being the
       * row index
       */
      void compute_cos_sin_angle_multiples(double cos_phi, double sin_phi);

      /**
       * Combine the associated Legendre polynomials with the cos/sin(mφ) to
       * compute the final spherical harmonics
       *
       * Stores the results as an (Eigen)matrix, sized
       * \f$(\ell_\text{max}+1)^2\f$.  The index collects \f$\ell\f$ and \f$m\f$
       * quantum numbers stored in compact format (\f$m\f$ varies fastest, from
       * \f$-\ell\f$ to \f$\ell\f$, and \f$\ell\f$ from 0 to
       * \f$\ell_\text{max}\f$).
       */
      void compute_spherical_harmonics();

      /**
       * Compute a full set of spherical harmonics and their Cartesian gradients
       *
       * Spherical harmonics are defined as described in
       * compute_spherical_harmonics().  Gradients are defined with respect
       * to motion of the neighbor atom, which is in accordance with the usual
       * definition with respect to the _arguments_ of the \f$Y_\ell^m\f$.  The
       * actual Cartesian gradients include an extra factor of \f$\frac{1}{r}\f$
       * that is not included here; the rest is independent of radius.
       *
       * Results are stored as in compute_spherical_harmonics(), but with an
       * extra (row) index for the x, y, and z Cartesian components.
       */
      void compute_spherical_harmonics_derivatives(double sin_theta,
                                                   double cos_theta,
                                                   double sin_phi,
                                                   double cos_phi);

      const MatrixX2_Ref get_cos_sin_m_phi() {
        return MatrixX2_Ref(this->cos_sin_m_phi);
      }

      size_t max_angular{0};
      Vector_t angular_coeffs1{};
      Vector_t angular_coeffs2{};
      Vector_t harmonics{};
      Matrix_t assoc_legendre_polynom{};
      MatrixX2_t cos_sin_m_phi{};
      Matrix_t coeff_a{};
      Matrix_t coeff_b{};
      // derivative related member variables
      bool calculate_derivatives;
      bool derivatives_precomputed{false};
      Matrix_t harmonics_derivatives{};
      Matrix_t plm_factors{};
      Vector_t legendre_polynom_differences{};
      Vector_t phi_derivative_factors{};
    };

  }  // namespace math
}  // namespace rascal

#endif  // SRC_RASCAL_MATH_SPHERICAL_HARMONICS_HH_
