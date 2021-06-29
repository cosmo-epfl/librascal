/**
 * @file   rascal/representations/calculator_spherical_expansion.hh
 *
 * @author Max Veit <max.veit@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 * @author Andrea Grifasi <andrea.grifasi@epfl.ch>
 * @author Alexander Goscinski <alexander.goscinski@epfl.ch>
 *
 * @date   19 October 2018
 *
 * @brief  Compute the spherical harmonics expansion of the local atom density
 *
 * Copyright © 2018 Max Veit, Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; see the file LICENSE. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef SRC_RASCAL_REPRESENTATIONS_CALCULATOR_SPHERICAL_EXPANSION_HH_
#define SRC_RASCAL_REPRESENTATIONS_CALCULATOR_SPHERICAL_EXPANSION_HH_

#include "rascal/math/bessel.hh"
#include "rascal/math/gauss_legendre.hh"
#include "rascal/math/hyp1f1.hh"
#include "rascal/math/interpolator.hh"
#include "rascal/math/spherical_harmonics.hh"
#include "rascal/math/utils.hh"
#include "rascal/representations/calculator_base.hh"
#include "rascal/representations/cutoff_functions.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/property_block_sparse.hh"
#include "rascal/structure_managers/structure_manager.hh"
#include "rascal/utils/utils.hh"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace rascal {

  namespace internal {
    /**
     * List of possible Radial basis that can be used by the spherical
     * expansion.
     */
    enum class RadialBasisType { GTO, DVR, End_ };

    /**
     * List of possible atomic smearing for the definition of the atomic
     * density. If not specified, the gaussian type smearing is implided.
     */
    enum class AtomicSmearingType { Constant, PerSpecies, Radial, End_ };

    /**
     * List of possible usages of optimization types
     */
    enum class OptimizationType {
      None,
      Spline,
      RadialDimReductionSpline,
      End_
    };

    /**
     * Base class for the specification of the atomic smearing.
     */
    struct AtomicSmearingSpecificationBase {
      //! Constructor
      AtomicSmearingSpecificationBase() = default;
      //! Destructor
      virtual ~AtomicSmearingSpecificationBase() = default;
      //! Copy constructor
      AtomicSmearingSpecificationBase(
          const AtomicSmearingSpecificationBase & other) = delete;
      //! Move constructor
      AtomicSmearingSpecificationBase(
          AtomicSmearingSpecificationBase && other) = default;
      //! Copy assignment operator
      AtomicSmearingSpecificationBase &
      operator=(const AtomicSmearingSpecificationBase & other) = delete;
      //! Move assignment operator
      AtomicSmearingSpecificationBase &
      operator=(AtomicSmearingSpecificationBase && other) = default;

      using Hypers_t = CalculatorBase::Hypers_t;
    };

    /**
     * Specification to hold the parameter for the atomic smearing function,
     * currently only Gaussians are supported.
     *
     * This is \f$\sigma\f$ in the definition
     * \f$f(r) = A \exp{\frac{-r^2}{2 \sigma^2}}\f$.
     * The width may depend both on the atomic species of the neighbour as well
     * as the distance.
     *
     * Note that this function is template-specialized by Gaussian sigma type
     * (constant, per-species, or radially dependent).
     *
     * @param pair Atom pair defining the neighbour, as e.g. returned by
     *             iteration over neighbours of a centre
     *
     * @throw logic_error if the requested sigma type has not been implemented
     *
     */
    template <AtomicSmearingType SigmaType>
    struct AtomicSmearingSpecification {};

    template <>
    struct AtomicSmearingSpecification<AtomicSmearingType::Constant>
        : AtomicSmearingSpecificationBase {
      using Hypers_t = typename AtomicSmearingSpecificationBase::Hypers_t;
      explicit AtomicSmearingSpecification(const Hypers_t & hypers) {
        this->constant_gaussian_sigma =
            hypers.at("gaussian_sigma").at("value").get<double>();
        if (this->constant_gaussian_sigma < 5e-2) {
          std::stringstream err_str{};
          err_str << "Constant gaussian sigma is too small: "
                  << this->constant_gaussian_sigma << " < 5e-2";
          throw std::runtime_error(err_str.str());
        }
      }
      template <size_t Order, size_t Layer>
      double
      get_gaussian_sigma(const ClusterRefKey<Order, Layer> & /* pair */) {
        return this->constant_gaussian_sigma;
      }
      double get_gaussian_sigma() { return this->constant_gaussian_sigma; }
      double constant_gaussian_sigma{0.};
    };

    /** Per-species template specialization of the above */

    template <>
    struct AtomicSmearingSpecification<AtomicSmearingType::PerSpecies>
        : AtomicSmearingSpecificationBase {
      using Hypers_t = typename AtomicSmearingSpecificationBase::Hypers_t;
      explicit AtomicSmearingSpecification(const Hypers_t & /* hypers */) {}
      template <size_t Order, size_t Layer>
      double
      get_gaussian_sigma(const ClusterRefKey<Order, Layer> & /* pair */) {
        throw std::logic_error("Requested a sigma type that has not yet "
                               "been implemented");
        return -1;
      }
    };

    /** Radially-dependent template specialization of the above */
    template <>
    struct AtomicSmearingSpecification<AtomicSmearingType::Radial>
        : AtomicSmearingSpecificationBase {
      using Hypers_t = typename AtomicSmearingSpecificationBase::Hypers_t;
      explicit AtomicSmearingSpecification(const Hypers_t & /* hypers */) {}
      template <size_t Order, size_t Layer>
      double
      get_gaussian_sigma(const ClusterRefKey<Order, Layer> & /* pair */) {
        throw std::logic_error("Requested a sigma type that has not yet "
                               "been implemented");
        return -1;
      }
    };

    //! Utility to make shared pointer and cast to base class
    template <AtomicSmearingType Type, class Hypers>
    auto make_atomic_smearing(const Hypers & sigma_hypers) {
      return std::static_pointer_cast<AtomicSmearingSpecificationBase>(
          std::make_shared<AtomicSmearingSpecification<Type>>(sigma_hypers));
    }

    //! Utility to cast base to child class
    template <AtomicSmearingType Type>
    auto downcast_atomic_smearing(
        const std::shared_ptr<AtomicSmearingSpecificationBase> &
            atomic_smearing) {
      return std::static_pointer_cast<AtomicSmearingSpecification<Type>>(
          atomic_smearing);
    }

    /**
     * Base class to define the radial contribution to the spherical expansion
     */
    struct RadialContributionBase {
      //! Constructor
      RadialContributionBase() = default;
      //! Destructor
      virtual ~RadialContributionBase() = default;
      //! Copy constructor
      RadialContributionBase(const RadialContributionBase & other) = delete;
      //! Move constructor
      RadialContributionBase(RadialContributionBase && other) = default;
      //! Copy assignment operator
      RadialContributionBase &
      operator=(const RadialContributionBase & other) = delete;
      //! Move assignment operator
      RadialContributionBase &
      operator=(RadialContributionBase && other) = default;

      using Hypers_t = CalculatorBase::Hypers_t;
      using Matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>;
      using Vector_t = Eigen::VectorXd;
      using Matrix_Ref = typename Eigen::Ref<const Matrix_t>;
      using Vector_Ref = typename Eigen::Ref<const Vector_t>;

      //! Pure virtual function to set hyperparameters of the cutoff function
      virtual void set_hyperparameters(const Hypers_t &) = 0;

      /*
       * Pure virtual function to initialize all Eigen matrices/vectors
       * dependend on max_radial and max_angular.
       */
      virtual void init_matrices() = 0;

      virtual void precompute() = 0;
      //! define the contribution from the central atom to the expansion
      template <AtomicSmearingType AST, size_t Order, size_t Layer>
      Vector_Ref
      compute_center_contribution(ClusterRefKey<Order, Layer> & /*center*/,
                                  int /*center_type*/) {
        throw std::runtime_error("This method is pure virtual and should be "
                                 "implemented in a derived class.");
        return Vector_Ref(Vector_t::Zero());
      }
      //! define the contribution from a neighbour atom to the expansion
      template <AtomicSmearingType AST, size_t Order, size_t Layer>
      Matrix_Ref compute_neighbour_contribution(
          const double /*distance*/,
          const ClusterRefKey<Order, Layer> & /*pair*/,
          int /*neighbour_type*/) {
        throw std::runtime_error("This method is pure virtual and should be "
                                 "implemented in a derived class.");
        return Matrix_Ref(Matrix_t::Zero());
      }

      /**
       * Compute the radial derivative of the neighbour contribution
       *
       * Note that you _must_ call compute_neighbour_contribution() first to
       * populate the relevant arrays!
       *
       * The derivative is taken with respect to the pair distance,
       * \f$r_{ij}\f$.  In order to get the radial component of the gradient,
       * remember to multiply by the direction vector
       * \f$
       *    \renewcommand{\vec}[1]{\mathbf{#1}}
       *    \hat{\vec{r}_{ij}}
       * \f$
       * (and not the vector itself), since
       * \f[
       *    \let\grad\nabla
       *    \grad_{\vec{r}_i} f(r_{ij}) =
       *                    \frac{\dd f}{\dd r_{ij}}
       *                    \frac{- \vec{r}_{ij}}{r_{ij}}
       *                  = \frac{\dd f}{\dd r_{ij}} -\hat{\vec{r}_{ij}}.
       * \f]
       *
       * so multiply by _negative_ \f$\hat{\vec{r}}_{ij}\f$ to get the radial
       * component of the gradient wrt motion of the central atom
       * (\f$\frac{d}{d\vec{r}_i}\f$).
       *
       * And finally, there is no compute_center_derivative() because that's
       * just zero -- the center contribution doesn't vary w.r.t. motion of
       * the central atom
       */
      template <size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_derivative(const double /*distance*/,
                                   const ClusterRefKey<Order, Layer> & /*pair*/,
                                   int /*neighbour_type*/) {
        throw std::runtime_error("This method is pure virtual and should be "
                                 "implemented in a derived class");
        return Matrix_Ref(Matrix_t::Zero());
      }
    };

    template <RadialBasisType RBT>
    struct RadialContribution {};

    /**
     * Implementation of the radial contribution for Gaussian Type Orbitals
     * radial basis functions and gaussian smearing of the atom density.
     *
     * @f[
     *      R^{GTO}_{n}(r) = \mathcal{N}_n\ r^{n} \exp[-br^2]
     * @f]
     *
     * @f{gather*}
     *      \newcommand{\dd}{\mathrm{d}\,}
     *      \mathcal{N}_n^2 = \frac{2}{\sigma_n^{2n + 3}\Gamma(n + 3/2)}\\
     *      \sigma_n = (r_\text{cut}-\delta r_\text{cut})
     *      \max(\sqrt{n},1)/n_\text{max}\\
     *      b=\frac{1}{2\sigma_n}\\
     *      \int_0^\infty R^{GTO}_{n}(r) R^{GTO}_{n^\prime}(r)
     *      \dd{r}= 2 \left(\frac{1}{2 \sigma_{n}^2}+
     *      \frac{1}{2 \sigma_{n^\prime}^2} \right)^{-\frac{1}{2}
     *      (3+n+n^\prime)} \Gamma(\frac{3+n+n^\prime}{2})
     * @f}
     *
     * See [the theory page](../SOAP.html#gto-like-radial-basis) for more
     * details.
     */
    template <>
    struct RadialContribution<RadialBasisType::GTO> : RadialContributionBase {
      // Constructor
      explicit RadialContribution(const Hypers_t & hypers) {
        this->set_hyperparameters(hypers);
      }
      // Destructor
      virtual ~RadialContribution() = default;
      // Copy constructor
      RadialContribution(const RadialContribution & other) = delete;
      // Move constructor
      RadialContribution(RadialContribution && other) = default;
      // Copy assignment operator
      RadialContribution & operator=(const RadialContribution & other) = delete;
      // Move assignment operator
      RadialContribution & operator=(RadialContribution && other) = default;

      using Parent = RadialContributionBase;
      using Hypers_t = typename Parent::Hypers_t;
      using Matrix_t = typename Parent::Matrix_t;
      using Vector_t = typename Parent::Vector_t;
      using Matrix_Ref = typename Parent::Matrix_Ref;
      using Vector_Ref = typename Parent::Vector_Ref;

      /**
       * Set hyperparameters.
       * @param hypers is expected to be the same as the the input of
       *         the spherical expansion
       */
      void set_hyperparameters(const Hypers_t & hypers) override {
        this->hypers = hypers;

        this->max_radial = hypers.at("max_radial");
        this->max_angular = hypers.at("max_angular");

        if (hypers.count("compute_gradients")) {
          this->compute_gradients = hypers.at("compute_gradients").get<bool>();
        } else {  // Default false (don't compute gradients)
          this->compute_gradients = false;
        }

        this->init_matrices();

        // find the cutoff radius of the representation
        auto fc_hypers = hypers.at("cutoff_function").get<json>();
        this->interaction_cutoff =
            fc_hypers.at("cutoff").at("value").get<double>();

        // define the type of smearing to use
        auto smearing_hypers = hypers.at("gaussian_density").get<json>();
        auto smearing_type = smearing_hypers.at("type").get<std::string>();
        if (smearing_type == "Constant") {
          this->atomic_smearing_type = AtomicSmearingType::Constant;
          this->atomic_smearing =
              make_atomic_smearing<AtomicSmearingType::Constant>(
                  smearing_hypers);
        } else {
          throw std::logic_error(
              "Requested Gaussian sigma type \'" + smearing_type +
              "\' has not been implemented.  Must be one of" +
              ": \'Constant\'.");
        }
      }

      // initialize Eigen matrices/vectors, contains both
      // precomputed quantities and actual expansion coefficients
      void init_matrices() override {
        this->radial_ortho_matrix.resize(this->max_radial, this->max_radial);
        this->ortho_norm_matrix.resize(this->max_radial, this->max_radial);
        this->fac_b.resize(this->max_radial, 1);
        this->a_b_l_n.resize(this->max_radial, this->max_angular + 1);
        this->distance_fac_a_l.resize(this->max_angular + 1);
        this->radial_norm_factors.resize(this->max_radial, 1);
        this->radial_n_factors.resize(this->max_radial);
        this->radial_sigmas.resize(this->max_radial, 1);
        this->radial_integral_neighbour.resize(this->max_radial,
                                               this->max_angular + 1);
        this->radial_integral_center.resize(this->max_radial);
        this->radial_neighbour_derivative.resize(this->max_radial,
                                                 this->max_angular + 1);
      }

      void precompute() override {
        this->precompute_radial_sigmas();
        this->precompute_radial_overlap();
        this->ortho_norm_matrix =
            this->radial_norm_factors.asDiagonal() * this->radial_ortho_matrix;

        this->hyp1f1_calculator.precompute(this->max_radial, this->max_angular);
      }

      /**
       * Define the contribution from a neighbour atom to the expansion
       * without requiring a cluster object so it can be used with the
       * spline.
       */
      template <AtomicSmearingType AST>
      Matrix_t compute_contribution(const double distance, const double sigma) {
        using math::PI;
        using math::pow;
        using std::sqrt;

        auto smearing{downcast_atomic_smearing<AST>(this->atomic_smearing)};
        // a = 1 / (2*\sigma^2)
        double fac_a{0.5 * pow(sigma, -2)};

        // computes (r_{ij}*a)^l incrementally
        Vector_t distance_fac_a_l(this->max_angular + 1);
        distance_fac_a_l(0) = 1.;
        double distance_fac_a{distance * fac_a};
        for (size_t angular_l{1}; angular_l < this->max_angular + 1;
             angular_l++) {
          distance_fac_a_l(angular_l) =
              distance_fac_a_l(angular_l - 1) * distance_fac_a;
        }

        // computes (a+b_n)^{-0.5*(3+l+n)}
        Matrix_t a_b_l_n(this->max_radial, this->max_angular + 1);
        for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
          double a_b_l{1. / sqrt(fac_a + this->fac_b[radial_n])};

          a_b_l_n(radial_n, 0) = pow(a_b_l, 3 + radial_n);

          for (size_t angular_l{1}; angular_l < this->max_angular + 1;
               angular_l++) {
            a_b_l_n(radial_n, angular_l) =
                a_b_l_n(radial_n, angular_l - 1) * a_b_l;
          }
        }

        this->hyp1f1_calculator.calc(distance, fac_a, this->fac_b);

        Matrix_t radial_integral_neighbour =
            (a_b_l_n.array() * this->hyp1f1_calculator.get_values().array())
                .matrix() *
            distance_fac_a_l.asDiagonal();
        radial_integral_neighbour.transpose() *=
            this->radial_norm_factors.asDiagonal();
        return radial_integral_neighbour;
      }

      //! define the contribution from the central atom to the expansion
      template <AtomicSmearingType AST, size_t Order, size_t Layer>
      Vector_Ref
      compute_center_contribution(ClusterRefKey<Order, Layer> & center,
                                  int /*center_type*/) {
        using math::pow;

        auto smearing{downcast_atomic_smearing<AST>(this->atomic_smearing)};

        // a = 1 / (2*\sigma^2)
        double fac_a{0.5 * pow(smearing->get_gaussian_sigma(center), -2)};
        return this->compute_center_contribution(fac_a);
      }

      //! define the contribution from the central atom to the expansion
      Vector_Ref compute_center_contribution(double fac_a) {
        using math::pow;
        using std::sqrt;

        // Contribution from the central atom
        // Y_l^m(θ, φ) =  √((2l+1)/(4*π))) \delta_{m0} and
        //  \delta_{l0} (spherical symmetry) and
        // 1F1(a,b,0) = 1
        for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
          double a_b_l_n{0};
          if (radial_n % 2 == 0) {
            a_b_l_n = sqrt(pow(fac_a + this->fac_b[radial_n],
                               -1 * static_cast<int>(3 + radial_n)));
          } else {
            a_b_l_n = pow(fac_a + this->fac_b[radial_n],
                          -1 * static_cast<int>(3 + radial_n) / 2);
          }

          this->radial_integral_center(radial_n) =
              this->radial_n_factors(radial_n) * a_b_l_n;
        }

        return Vector_Ref(this->radial_integral_center);
      }

      //! define the contribution from a neighbour atom to the expansion
      template <AtomicSmearingType AST, size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_contribution(const double distance,
                                     const ClusterRefKey<Order, Layer> & pair) {
        auto smearing{downcast_atomic_smearing<AST>(this->atomic_smearing)};
        double smearing_value{smearing->get_gaussian_sigma(pair)};
        // a = 1 / (2*\sigma^2)
        double fac_a{0.5 * pow(smearing_value, -2)};
        return this->compute_neighbour_contribution(distance, fac_a);
      }

      // Define the contribution from a neighbour atom to the expansion with an
      // already precomputed a-factor

      Matrix_Ref compute_neighbour_contribution(const double distance,
                                                const double fac_a) {
        using math::pow;
        using std::sqrt;

        // computes (r_{ij}*a)^l incrementally
        this->distance_fac_a_l(0) = 1.;

        double distance_fac_a{distance * fac_a};
        for (size_t angular_l{1}; angular_l < this->max_angular + 1;
             angular_l++) {
          this->distance_fac_a_l(angular_l) =
              this->distance_fac_a_l(angular_l - 1) * distance_fac_a;
        }

        // computes (a+b_n)^{-0.5*(3+l+n)}
        Eigen::ArrayXd a_b_l{Eigen::rsqrt(fac_a + this->fac_b.array())};
        for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
          this->a_b_l_n(radial_n, 0) = pow(a_b_l(radial_n), 3 + radial_n);
        }
        // seems like vetorization does not improve things here because it is
        // memory is not contiguous ? or because max_angular is quite small and
        // memory overhead balances out the vector arithmetics gain
        for (size_t angular_l{1}; angular_l < this->max_angular + 1;
             ++angular_l) {
          this->a_b_l_n.col(angular_l) =
              (this->a_b_l_n.col(angular_l - 1).array() * a_b_l).matrix();
        }

        this->hyp1f1_calculator.calc(distance, fac_a, this->fac_b,
                                     this->compute_gradients);

        this->radial_integral_neighbour =
            (this->a_b_l_n.array() *
             this->hyp1f1_calculator.get_values().array())
                .matrix() *
            this->distance_fac_a_l.asDiagonal();

        return Matrix_Ref(this->radial_integral_neighbour);
      }

      //! Compute the radial derivative of the neighbour contribution
      template <size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_derivative(const double distance,
                                   const ClusterRefKey<Order, Layer> & /*pair*/,
                                   int /*neighbour_type*/) {
        using math::PI;
        using math::pow;
        using std::sqrt;

        // start proportional_factors as the list of l from 0 to l_max
        Vector_t proportional_factors =
            Vector_t::LinSpaced(this->max_angular + 1, 0, this->max_angular);
        proportional_factors /= distance;

        this->radial_neighbour_derivative =
            (this->a_b_l_n.array() *
             this->hyp1f1_calculator.get_derivatives().array())
                .matrix() *
            this->distance_fac_a_l.asDiagonal();

        this->radial_neighbour_derivative +=
            this->radial_integral_neighbour * proportional_factors.asDiagonal();

        return Matrix_Ref(this->radial_neighbour_derivative);
      }

      // Can be used after `compute_center_contribution` to finalize the
      // coefficients within the neighbour loop
      void finalize_radial_integral_center() {
        this->radial_integral_center =
            this->ortho_norm_matrix.transpose() * this->radial_integral_center;
      }

      /**
       * Orthonormalize the radial integral computed with
       * `compute_neighbour_contribution` within the neighbour loop
       * It is useful for splining the GTO basis.
       */
      void finalize_radial_integral_neighbour() {
        this->radial_integral_neighbour = this->ortho_norm_matrix.transpose() *
                                          this->radial_integral_neighbour;
      }

      // used within the computation of the spherical expansison to apply global
      // operations after the neighbour loop
      template <typename Coeffs>
      void finalize_coefficients(Coeffs & coefficients) const {
        coefficients.lhs_dot(this->ortho_norm_matrix);
      }

      template <int NDims, typename Coeffs, typename Center>
      void finalize_coefficients_der(Coeffs & coefficients_gradient,
                                     Center & center) const {
        for (auto neigh : center.pairs_with_self_pair()) {
          auto & coefficients_neigh_gradient = coefficients_gradient[neigh];
          coefficients_neigh_gradient.template lhs_dot_der<NDims>(
              this->ortho_norm_matrix);
        }  // for (neigh : center)
      }

      /** Compute common prefactors for the radial Gaussian basis functions */
      void precompute_radial_sigmas() {
        using math::pow;
        // Precompute common prefactors
        for (size_t radial_n{0}; radial_n < this->max_radial; ++radial_n) {
          this->radial_sigmas[radial_n] =
              std::max(std::sqrt(static_cast<double>(radial_n)), 1.0) *
              this->interaction_cutoff / static_cast<double>(this->max_radial);
          this->fac_b[radial_n] = 0.5 * pow(this->radial_sigmas[radial_n], -2);
          this->radial_norm_factors(radial_n) =
              0.25 * std::sqrt(2.0 / (std::tgamma(1.5 + radial_n) *
                                      pow(this->radial_sigmas[radial_n],
                                          3 + 2 * radial_n)));
          this->radial_n_factors(radial_n) =
              std::tgamma(0.5 * (3.0 + radial_n)) / std::tgamma(1.5);
        }
      }

      /**
       * Compute the radial overlap matrix for later orthogonalization.
       *
       * @throw runtime_error if the overlap matrix cannot be diagonalized
       */
      void precompute_radial_overlap() {
        using math::pow;
        using std::sqrt;
        using std::tgamma;

        // TODO(max-veit) see if we can replace the gammas with their natural
        // logs,
        // since it'll overflow for relatively small n (n1 + n2 >~ 300)
        // UPDATE nevermind, the overlap matrix becomes singular well before
        // this point.
        Matrix_t overlap(this->max_radial, this->max_radial);
        for (size_t radial_n1{0}; radial_n1 < this->max_radial; radial_n1++) {
          for (size_t radial_n2{0}; radial_n2 < this->max_radial; radial_n2++) {
            overlap(radial_n1, radial_n2) =
                pow(0.5 / pow(this->radial_sigmas[radial_n1], 2) +
                        0.5 / pow(this->radial_sigmas[radial_n2], 2),
                    -0.5 * (3.0 + radial_n1 + radial_n2)) /
                (pow(this->radial_sigmas[radial_n1], radial_n1) *
                 pow(this->radial_sigmas[radial_n2], radial_n2)) *
                tgamma(0.5 * (3.0 + radial_n1 + radial_n2)) /
                (pow(this->radial_sigmas[radial_n1] *
                         this->radial_sigmas[radial_n2],
                     1.5) *
                 sqrt(tgamma(1.5 + radial_n1) * tgamma(1.5 + radial_n2)));
          }
        }
        // Compute the inverse square root of the overlap matrix
        Eigen::SelfAdjointEigenSolver<Matrix_t> eigensolver(overlap);
        if (eigensolver.info() != Eigen::Success) {
          throw std::runtime_error("Warning: Could not diagonalize "
                                   "radial overlap matrix.");
        }
        Matrix_t eigenvalues = eigensolver.eigenvalues();
        Eigen::ArrayXd eigs_invsqrt = eigenvalues.array().rsqrt();
        Matrix_t unitary = eigensolver.eigenvectors();
        this->radial_ortho_matrix =
            unitary * eigs_invsqrt.matrix().asDiagonal() * unitary.adjoint();
      }

      Matrix_t get_radial_orthonormalization_matrix() const {
        return this->radial_norm_factors.asDiagonal() *
               this->radial_ortho_matrix;
      }

      Matrix_Ref get_radial_integral_neighbour() const {
        return Matrix_Ref(this->radial_integral_neighbour);
      }

      Matrix_Ref get_radial_neighbour_derivative() const {
        return Matrix_Ref(this->radial_neighbour_derivative);
      }

      std::shared_ptr<AtomicSmearingSpecificationBase> atomic_smearing{};
      AtomicSmearingType atomic_smearing_type{};
      math::Hyp1f1SphericalExpansion hyp1f1_calculator{true, 1e-13, 200};
      // data member used to store the contributions to the expansion
      Matrix_t radial_integral_neighbour{};
      Vector_t radial_integral_center{};
      // And derivatives
      Matrix_t radial_neighbour_derivative{};
      // and of course, d/dr of the center contribution is zero

      Hypers_t hypers{};
      // some usefull parameters
      double interaction_cutoff{};
      size_t max_radial{};
      size_t max_angular{};
      bool compute_gradients{};

      // \sigma_n = (r_\text{cut}-\delta r_\text{cut})
      // \max(\sqrt{n},1)/n_\text{max}
      Vector_t radial_sigmas{};
      // b_n = 1 / (2*\sigma_n^2)
      Vector_t fac_b{};
      Matrix_t a_b_l_n{};
      Vector_t distance_fac_a_l{};
      //! constant factors of the GTO basis
      //! N_n=sqrt(2/\Gamma(n+3/2)) b_n^{3+2n}/4
      Vector_t radial_norm_factors{};
      Vector_t radial_n_factors{};
      //! S_{nn'}^{-1/2}, orthonormalization matrix of the GTO basis
      Matrix_t radial_ortho_matrix{};
      //! combination of radial_ortho_matrix and radial_norm_factors
      //! not symmetric
      Matrix_t ortho_norm_matrix{};
    };

    /**
     * Implementation of the radial contribution for DVR basis
     *
     * See the
     * [theory page](../SOAP.html#numerical-integration-of-the-radial-integral)
     * for more details.
     */
    template <>
    struct RadialContribution<RadialBasisType::DVR> : RadialContributionBase {
      //! Constructor
      explicit RadialContribution(const Hypers_t & hypers) {
        this->set_hyperparameters(hypers);
      }
      //! Destructor
      virtual ~RadialContribution() = default;
      //! Copy constructor
      RadialContribution(const RadialContribution & other) = delete;
      //! Move constructor
      RadialContribution(RadialContribution && other) = default;
      //! Copy assignment operator
      RadialContribution & operator=(const RadialContribution & other) = delete;
      //! Move assignment operator
      RadialContribution & operator=(RadialContribution && other) = default;

      using Parent = RadialContributionBase;
      using Hypers_t = typename Parent::Hypers_t;
      using Matrix_t = typename Parent::Matrix_t;
      using Vector_t = typename Parent::Vector_t;
      using Matrix_Ref = typename Parent::Matrix_Ref;
      using Vector_Ref = typename Parent::Vector_Ref;

      /**
       * Set hyperparameters.
       *
       * @param hypers is expected to be the same as the the input of
       *         the spherical expansion
       */
      void set_hyperparameters(const Hypers_t & hypers) override {
        this->hypers = hypers;

        this->max_radial = hypers.at("max_radial");
        this->max_angular = hypers.at("max_angular");

        if (hypers.count("compute_gradients")) {
          this->compute_gradients = hypers.at("compute_gradients").get<bool>();
        } else {  // Default false (don't compute gradients)
          this->compute_gradients = false;
        }

        this->init_matrices();

        // find the cutoff radius of the representation
        auto fc_hypers = hypers.at("cutoff_function").get<json>();
        this->interaction_cutoff =
            fc_hypers.at("cutoff").at("value").get<double>();
        this->smooth_width =
            fc_hypers.at("smooth_width").at("value").get<double>();

        // define the type of smearing to use
        auto smearing_hypers = hypers.at("gaussian_density").get<json>();
        auto smearing_type = smearing_hypers.at("type").get<std::string>();
        if (smearing_type == "Constant") {
          this->atomic_smearing_type = AtomicSmearingType::Constant;
          this->atomic_smearing =
              make_atomic_smearing<AtomicSmearingType::Constant>(
                  smearing_hypers);
          this->smearing =
              smearing_hypers.at("gaussian_sigma").at("value").get<double>();
        } else {
          throw std::logic_error(
              "Requested Gaussian sigma type \'" + smearing_type +
              "\' has not been implemented.  Must be one of" +
              ": \'Constant\'.");
        }
      }

      // initialize Eigen matrices/vectors, contains both
      // precomputed quantities and actual expansion coefficients
      void init_matrices() override {
        this->legendre_radial_factor.resize(this->max_radial);
        this->legendre_points.resize(this->max_radial);

        this->radial_integral_neighbour.resize(this->max_radial,
                                               this->max_angular + 1);
        this->radial_neighbour_derivative.resize(this->max_radial,
                                                 this->max_angular + 1);
        this->radial_integral_center.resize(this->max_radial);
      }

      void precompute() override {
        auto point_weight{math::compute_gauss_legendre_points_weights(
            0., this->interaction_cutoff + 3 * this->smearing,
            this->max_radial)};

        // sqrt(w) * x
        // (if you think it should be x^2 and not x, think again -- the
        // transformation from integrating the overlap in 3-D spherical
        // coordinates to the 1-D radial coordinate absorbs a-factor of r.
        // For more details, see the SOAP theory documentation)
        // TODO(max) link to SOAP theory documentation
        this->legendre_radial_factor =
            point_weight.col(1).array().sqrt() * point_weight.col(0).array();

        this->legendre_points = point_weight.col(0);

        this->bessel.precompute(this->max_angular, this->legendre_points,
                                this->compute_gradients);
      }

      template <AtomicSmearingType AST, size_t Order, size_t Layer>
      Vector_Ref
      compute_center_contribution(ClusterRefKey<Order, Layer> & center,
                                  int /*center_type*/) {
        using math::pow;

        auto smearing{downcast_atomic_smearing<AST>(this->atomic_smearing)};

        // a = 1 / (2*\sigma^2)
        double fac_a{0.5 / pow(smearing->get_gaussian_sigma(center), 2_size_t)};
        return this->compute_center_contribution(fac_a);
      }

      //! define the contribution from the central atom to the expansion
      Vector_Ref compute_center_contribution(const double fac_a) {
        using math::PI;
        using math::pow;
        using std::sqrt;

        this->radial_integral_center =
            this->legendre_radial_factor.array() *
            Eigen::exp((-fac_a * this->legendre_points.array().square()));

        return Matrix_Ref(this->radial_integral_center);
      }

      //! define the contribution from a neighbour atom to the expansion
      template <AtomicSmearingType AST, size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_contribution(const double distance,
                                     const ClusterRefKey<Order, Layer> & pair) {
        auto smearing{downcast_atomic_smearing<AST>(this->atomic_smearing)};
        double smearing_value{smearing->get_gaussian_sigma(pair)};
        // a = 1 / (2*\sigma^2)
        double fac_a{0.5 / pow(smearing_value, 2_size_t)};
        return this->compute_neighbour_contribution(distance, fac_a);
      }

      inline Matrix_Ref compute_neighbour_contribution(const double distance,
                                                       const double fac_a) {
        using math::PI;
        using math::pow;
        using std::sqrt;

        this->bessel.calc(distance, fac_a);

        this->radial_integral_neighbour =
            this->legendre_radial_factor.asDiagonal() *
            this->bessel.get_values().matrix();

        return Matrix_Ref(this->radial_integral_neighbour);
      }

      //! Compute the radial derivative of the neighbour contribution
      template <size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_derivative(const double /*distance*/,
                                   const ClusterRefKey<Order, Layer> & /*pair*/,
                                   int /*neighbour_type*/) {
        this->radial_neighbour_derivative =
            this->legendre_radial_factor.asDiagonal() *
            this->bessel.get_gradients().matrix();

        return Matrix_Ref(this->radial_neighbour_derivative);
      }

      void finalize_radial_integral_center() {}

      void finalize_radial_integral_neighbour() {}

      template <typename Coeffs>
      void finalize_coefficients(Coeffs & /*coefficients*/) const {}

      template <int NDims, typename Coeffs, typename Center>
      void finalize_coefficients_der(Coeffs & /*coefficients_gradient*/,
                                     Center & /*center*/) const {}

      math::ModifiedSphericalBessel bessel{};

      std::shared_ptr<AtomicSmearingSpecificationBase> atomic_smearing{};
      AtomicSmearingType atomic_smearing_type{};

      // data member used to store the contributions to the expansion
      Matrix_t radial_integral_neighbour{};
      Matrix_t radial_neighbour_derivative{};
      Vector_t radial_integral_center{};

      Hypers_t hypers{};
      // some useful parameters
      double interaction_cutoff{};
      double smooth_width{};
      double smearing{};
      size_t max_radial{};
      size_t max_angular{};
      bool compute_gradients{};

      Vector_t legendre_radial_factor{};
      Vector_t legendre_points{};
      Vector_t legendre_points2{};
    };

    /* A RadialContributionHandler handles the different cases of
     * AtomicSmearingType and OptimizationType. Depending on these template
     * parameters different member variables have to be used and different
     * parameters can be precomputed.
     */
    template <RadialBasisType RBT, AtomicSmearingType AST, OptimizationType OT>
    struct RadialContributionHandler {};

    /* For the a constant smearing type the "a" factor can be precomputed
     */
    template <RadialBasisType RBT>
    struct RadialContributionHandler<RBT, AtomicSmearingType::Constant,
                                     OptimizationType::None>
        : public RadialContribution<RBT> {
     public:
      using Parent = RadialContribution<RBT>;
      using Hypers_t = typename Parent::Hypers_t;
      using Matrix_Ref = typename Parent::Matrix_Ref;
      using Vector_Ref = typename Parent::Vector_Ref;

      explicit RadialContributionHandler(const Hypers_t & hypers)
          : Parent(hypers) {
        this->precompute();
      }

      // Returns the precomputed center contribution
      template <size_t Order, size_t Layer>
      Vector_Ref
      compute_center_contribution(ClusterRefKey<Order, Layer> & /*center*/,
                                  int /*center_type*/) {
        return Vector_Ref(this->radial_integral_center);
      }

      template <size_t Order, size_t Layer>
      Matrix_Ref compute_neighbour_contribution(
          const double distance,
          const ClusterRefKey<Order, Layer> & /*neighbour*/,
          int /*neighbour_type*/) {
        return Parent::compute_neighbour_contribution(distance, this->fac_a);
      }

      Matrix_Ref compute_neighbour_contribution(const double) {
        return Parent::compute_neighbour_contribution(this->fac_a);
      }

      template <size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_derivative(const double distance,
                                   const ClusterRefKey<Order, Layer> & pair,
                                   int neighbour_type) {
        return Parent::compute_neighbour_derivative(distance, pair,
                                                    neighbour_type);
      }

     protected:
      void precompute() override {
        Parent::precompute();
        this->precompute_fac_a();
        this->precompute_center_contribution();
      }

      void precompute_fac_a() {
        auto smearing{downcast_atomic_smearing<AtomicSmearingType::Constant>(
            this->atomic_smearing)};
        this->fac_a = 0.5 * pow(smearing->get_gaussian_sigma(), -2);
      }

      // Should be invoked only after the a-factor has been precomputed
      void precompute_center_contribution() {
        Parent::compute_center_contribution(this->fac_a);
      }

      // 1/(2σ^2)
      double fac_a{};
    };

    /* For the a constant smearing type the "a" factor can be precomputed and
     * when using the spline has to be initialized and used.
     */
    template <RadialBasisType RBT>
    struct RadialContributionHandler<RBT, AtomicSmearingType::Constant,
                                     OptimizationType::Spline>
        : public RadialContribution<RBT> {
     public:
      using Parent = RadialContribution<RBT>;
      using Hypers_t = typename Parent::Hypers_t;
      using Matrix_t = typename Parent::Matrix_t;
      using Matrix_Ref = typename Parent::Matrix_Ref;
      using Vector_Ref = typename Parent::Vector_Ref;
      using Spline_t = math::InterpolatorMatrixUniformCubicSpline<
          math::RefinementMethod_t::Exponential>;

      explicit RadialContributionHandler(const Hypers_t & hypers)
          : Parent(hypers) {
        this->precompute();
        this->init_interpolator(hypers);
      }

      // If we find a case where smarter parameters for x1 and x2 can be given
      explicit RadialContributionHandler(const Hypers_t & hypers,
                                         const double range_begin,
                                         const double range_end,
                                         const double accuracy)
          : Parent(hypers) {
        this->precompute();
        this->init_interpolator(range_begin, range_end, accuracy);
      }

      // Returns the precomputed center contribution
      template <size_t Order, size_t Layer>
      Vector_Ref
      compute_center_contribution(ClusterRefKey<Order, Layer> & /*center*/,
                                  int /*center_type*/) {
        return Vector_Ref(this->radial_integral_center);
      }

      template <size_t Order, size_t Layer>
      Matrix_Ref compute_neighbour_contribution(
          const double distance, const ClusterRefKey<Order, Layer> & /*pair*/,
          int /*neighbour_type*/) {
        this->radial_integral_neighbour = this->intp->interpolate(distance);
        return Matrix_Ref(this->radial_integral_neighbour);
      }

      template <size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_derivative(const double distance,
                                   const ClusterRefKey<Order, Layer> & /*pair*/,
                                   int /*neighbour_type*/) {
        this->radial_neighbour_derivative =
            this->intp->interpolate_derivative(distance);
        return Matrix_Ref(this->radial_neighbour_derivative);
      }

      /*
       * Overwriting the finalization function to empty one, since the
       * finalization happens now in the spline
       */
      template <typename Coeffs>
      void finalize_coefficients(Coeffs & /*coefficients*/) {}

      /*
       * Overwriting the finalization function to empty one, since the
       * derivative of the spline is used
       */
      template <int NDims, typename Coeffs, typename Center>
      void finalize_coefficients_der(Coeffs & /*coefficients_gradient*/,
                                     Center & /*center*/) const {}

     protected:
      void precompute() override {
        Parent::precompute();
        this->precompute_fac_a();
        this->precompute_center_contribution();
      }
      void precompute_fac_a() {
        auto smearing{downcast_atomic_smearing<AtomicSmearingType::Constant>(
            this->atomic_smearing)};
        this->fac_a = 0.5 * pow(smearing->get_gaussian_sigma(), -2);
      }

      // Should be invoked only after the a-factor has been precomputed
      void precompute_center_contribution() {
        Parent::compute_center_contribution(this->fac_a);
        Parent::finalize_radial_integral_center();
      }

      void init_interpolator(const Hypers_t & hypers) {
        auto radial_contribution_hypers =
            hypers.at("radial_contribution").template get<json>();
        auto optimization_hypers =
            radial_contribution_hypers.at("optimization").template get<json>();

        double accuracy{this->get_interpolator_accuracy(optimization_hypers)};
        // minimal distance such that it is still stable with the interpolated
        // function
        double range_begin{math::SPHERICAL_BESSEL_FUNCTION_FTOL};
        double range_end{this->interaction_cutoff};
        this->init_interpolator(range_begin, range_end, accuracy);
      }

      void init_interpolator(const double range_begin, const double range_end,
                             const double accuracy) {
        // "this" is passed by reference and is mutable
        std::function<Matrix_t(double)> func{
            [&](const double distance) mutable {
              Parent::compute_neighbour_contribution(distance, this->fac_a);
              Parent::finalize_radial_integral_neighbour();
              return this->radial_integral_neighbour;
            }};
        Matrix_t result = func(range_begin);
        int cols{static_cast<int>(result.cols())};
        int rows{static_cast<int>(result.rows())};
        this->intp = std::make_unique<Spline_t>(func, range_begin, range_end,
                                                accuracy, cols, rows);
      }

      double get_interpolator_accuracy(const Hypers_t & optimization_hypers) {
        auto spline_hypers =
            optimization_hypers.at("Spline").template get<json>();
        if (spline_hypers.count("accuracy")) {
          return spline_hypers.at("accuracy").template get<double>();
        } else {
          std::stringstream err_str{};
          err_str << "No Spline accuracy was given.";
          throw std::logic_error(err_str.str());
        }
      }

      double get_cutoff(const Hypers_t & hypers) {
        auto fc_hypers = hypers.at("cutoff_function").template get<json>();
        return fc_hypers.at("cutoff").at("value").template get<double>();
      }

      double fac_a{};
      std::unique_ptr<Spline_t> intp{};
    };

    /*
     * Handles optimized radial basis function. The radial basis function
     * can be optimized by multiplying projection matrix with the radial basis
     * for each species channel and angular channel.
     * The projection matrices are of shape (max_radial, expanded_max_radial).
     * The spherical expansion coefficients with `expanded_max_radial` as
     * number of radial channels and then projected with the projection
     * matrices to max_radial radial channels.
     */
    template <RadialBasisType RBT>
    struct RadialContributionHandler<RBT, AtomicSmearingType::Constant,
                                     OptimizationType::RadialDimReductionSpline>
        : public RadialContribution<RBT> {
     public:
      using Parent = RadialContribution<RBT>;
      using Hypers_t = typename Parent::Hypers_t;
      using Matrix_Ref = typename Parent::Matrix_Ref;
      using Vector_Ref = typename Parent::Vector_Ref;
      using Matrix_t = math::Matrix_t;
      using Vector_t = math::Vector_t;
      using Spline_t = math::InterpolatorMatrixUniformCubicSpline<
          math::RefinementMethod_t::Exponential>;

      explicit RadialContributionHandler(const Hypers_t & hypers)
          : Parent(hypers) {
        this->set_hyperparameters(hypers);
        this->precompute();
        this->init_interpolator(hypers);
      }

      void set_hyperparameters(const Hypers_t & hypers) override {
        auto radial_contribution_hypers =
            hypers.at("radial_contribution").template get<json>();
        auto optimization_hypers =
            radial_contribution_hypers.at("optimization").template get<json>();
        auto radial_dim_reduction_hypers =
            optimization_hypers.at("RadialDimReduction").template get<json>();
        if (radial_dim_reduction_hypers.count("projection_matrices")) {
          std::map<std::string, std::vector<std::vector<std::vector<double>>>>
              json_projection_matrices{};
          try {
            // hyper keys are converted to string so we also use here strings
            // for species
            json_projection_matrices =
                radial_dim_reduction_hypers.at("projection_matrices")
                    .template get<std::map<
                        std::string,
                        std::vector<std::vector<std::vector<double>>>>>();
          } catch (const std::exception & e) {
            std::stringstream err_str{};
            err_str << "Projection matrices should be of type "
                       "std::map<std::string, "
                       "std::vector<std::vector<std::vector<double>>>>.";
            throw std::logic_error(err_str.str());
          }

          this->n_species = json_projection_matrices.size();
          this->n_components = this->max_radial;

          if (json_projection_matrices.begin()->second.size() == 0) {
            std::stringstream err_str{};
            err_str << "projection_matrices should have the shape "
                       "(max_angular+1, max_radial, expanded_max_radial)";
            throw std::logic_error(err_str.str());
          }
          if (json_projection_matrices.begin()->second.at(0).size() == 0) {
            std::stringstream err_str{};
            err_str << "projection_matrices should have the shape "
                       "(max_angular+1, max_radial, expanded_max_radial)";
            throw std::logic_error(err_str.str());
          }
          if (json_projection_matrices.begin()->second.at(0).at(0).size() ==
              0) {
            std::stringstream err_str{};
            err_str << "projection_matrices should have the shape "
                       "(max_angular+1, max_radial, expanded_max_radial)";
            throw std::logic_error(err_str.str());
          }
          // overwrite max_radial to expanded max_radial
          this->max_radial =
              json_projection_matrices.begin()->second.at(0).at(0).size();

          // check projection matrices sizes
          std::string species;
          for (auto it = json_projection_matrices.begin();
               it != json_projection_matrices.end(); ++it) {
            species = it->first;
            if (json_projection_matrices.at(species).size() !=
                this->max_angular + 1) {
              std::stringstream err_str{};
              err_str << "Projection matrices at species=" << species
                      << " does not match the size max_angular="
                      << this->max_angular << " but is "
                      << json_projection_matrices.at(species).size()
                      << ", should be max_angular+1";
              throw std::logic_error(err_str.str());
            }
            for (size_t angular_l = 0; angular_l < this->max_angular + 1;
                 angular_l++) {
              if (json_projection_matrices.at(species).at(angular_l).size() !=
                  this->n_components) {
                std::stringstream err_str{};
                // be aware that `max_radial` in hyperparameters is
                // this->n_components within in this class
                err_str
                    << "Projection matrices at species=" << species
                    << " and angular_l=" << angular_l
                    << " does not match the max_radial=" << this->n_components
                    << " in hyperparameters but is "
                    << json_projection_matrices.at(species).at(angular_l).size()
                    << ".";
                throw std::logic_error(err_str.str());
              }
              for (size_t radial_c = 0; radial_c < this->n_components;
                   radial_c++) {
                if (json_projection_matrices.at(species)
                        .at(angular_l)
                        .at(radial_c)
                        .size() != this->max_radial) {
                  std::stringstream err_str{};
                  // be aware that `max_radial` in hyperparameters is
                  // this->n_components within in this class
                  err_str << "Projection matrices at species=" << species
                          << ", angular_l=" << angular_l
                          << " and max_radial=" << this->n_components
                          << " is not consistent in expanded_max_radial="
                          << this->max_radial
                          << " with species=0, angular_l=0 and max_radial=0"
                          << json_projection_matrices.at(species)
                                 .at(angular_l)
                                 .at(radial_c)
                                 .size()
                          << ".";
                  throw std::logic_error(err_str.str());
                }
              }
            }
          }

          // init projection matrices
          Matrix_t projection_matrix{};
          std::vector<Matrix_t> angular_projection_matrices(this->max_angular +
                                                            1);
          for (auto it = json_projection_matrices.begin();
               it != json_projection_matrices.end(); ++it) {
            species = it->first;
            for (size_t angular_l = 0; angular_l < this->max_angular + 1;
                 angular_l++) {
              projection_matrix.resize(this->n_components, this->max_radial);
              projection_matrix.setZero();
              for (size_t radial_c = 0; radial_c < this->n_components;
                   radial_c++) {
                for (size_t radial_n = 0; radial_n < this->max_radial;
                     radial_n++) {
                  projection_matrix(radial_c, radial_n) =
                      json_projection_matrices.at(species)
                          .at(angular_l)
                          .at(radial_c)
                          .at(radial_n);
                }
              }
              angular_projection_matrices.at(angular_l) = projection_matrix;
            }
            this->projection_matrices.insert(
                std::pair<int, std::vector<Matrix_t>>(
                    std::stoi(species), angular_projection_matrices));
          }

          this->init_matrices();
        } else {
          std::stringstream err_str{};
          err_str << "No projection matrices were given.";
          throw std::logic_error(err_str.str());
        }
        Parent::init_matrices();
      }

      void precompute() override {
        Parent::precompute();
        this->precompute_fac_a();
        this->precompute_center_contribution();
      }

      void init_matrices() override {
        this->reduced_radial_integral_neighbour.resize(this->n_components,
                                                       this->max_angular + 1);
        this->reduced_radial_integral_neighbour.setZero();
        this->reduced_radial_neighbour_derivative.resize(this->n_components,
                                                         this->max_angular + 1);
        this->reduced_radial_neighbour_derivative.setZero();
      }

      // Returns the precomputed center contribution
      template <size_t Order, size_t Layer>
      Vector_Ref
      compute_center_contribution(ClusterRefKey<Order, Layer> & /*center*/,
                                  int center_type) {
        try {
          return Vector_Ref(this->reduced_radial_integral_centers[center_type]);
        } catch (const std::exception & e) {
          std::stringstream err_str{};
          err_str << "RadialDimReduction is missing projection matrices at "
                     "least for species '"
                  << center_type << "'";
          throw std::runtime_error(err_str.str());
        }
      }

      template <size_t Order, size_t Layer>
      Matrix_Ref compute_neighbour_contribution(
          const double distance, const ClusterRefKey<Order, Layer> & /*pair*/,
          int neighbour_type) {
        try {
          this->radial_integral_neighbour =
              this->intps[neighbour_type]->interpolate(distance);
          return Matrix_Ref(this->radial_integral_neighbour);
        } catch (const std::exception & e) {
          std::stringstream err_str{};
          err_str << "RadialDimReduction is missing projection matrices at "
                     "least for species '"
                  << neighbour_type << "'";
          throw std::runtime_error(err_str.str());
        }
      }

      template <size_t Order, size_t Layer>
      Matrix_Ref
      compute_neighbour_derivative(const double distance,
                                   const ClusterRefKey<Order, Layer> & /*pair*/,
                                   int neighbour_type) {
        try {
          this->radial_neighbour_derivative =
              this->intps[neighbour_type]->interpolate_derivative(distance);
          return Matrix_Ref(this->radial_neighbour_derivative);
        } catch (const std::exception & e) {
          std::stringstream err_str{};
          err_str << "RadialDimReduction is missing projection matrices at "
                     "least for species '"
                  << neighbour_type << "'";
          throw std::runtime_error(err_str.str());
        }
      }

      /*
       * Overwriting the finalization function to empty one, since the
       * finalization happens now in the per neighbour computation
       */
      template <typename Coeffs>
      void finalize_coefficients(Coeffs & /*coefficients*/) {}

      template <int NDims, typename Coeffs>
      void
      finalize_neighbour_derivative(Coeffs & /*coefficients_neigh_gradient*/) {}

     protected:
      Matrix_Ref compute_neighbour_contribution(const double distance,
                                                int neighbour_type) {
        Parent::compute_neighbour_contribution(distance, this->fac_a);
        Parent::finalize_radial_integral_neighbour();
        for (size_t angular_l{0}; angular_l < this->max_angular + 1;
             ++angular_l) {
          this->reduced_radial_integral_neighbour.col(angular_l) =
              this->projection_matrices.at(neighbour_type).at(angular_l) *
              this->radial_integral_neighbour.col(angular_l);
        }
        return Matrix_Ref(this->reduced_radial_integral_neighbour);
      }

      void precompute_fac_a() {
        auto smearing{downcast_atomic_smearing<AtomicSmearingType::Constant>(
            this->atomic_smearing)};
        this->fac_a = 0.5 * pow(smearing->get_gaussian_sigma(), -2);
      }

      // Should be invoked only after the a-factor has been precomputed
      void precompute_center_contribution() {
        Parent::compute_center_contribution(this->fac_a);
        Parent::finalize_radial_integral_center();
        int species;
        // Vector_t reduced_radial_integral_center;
        // for (auto species_projection_matrices : this->projection_matrices) {
        for (auto it = this->projection_matrices.begin();
             it != this->projection_matrices.end(); ++it) {
          species = it->first;
          // use projection matrix at l = 0
          this->reduced_radial_integral_centers.insert(std::pair<int, Vector_t>(
              species, it->second.at(0) * this->radial_integral_center));
        }
      }

      void init_interpolator(const Hypers_t & hypers) {
        auto radial_contribution_hypers =
            hypers.at("radial_contribution").template get<json>();
        auto optimization_hypers =
            radial_contribution_hypers.at("optimization").template get<json>();

        double accuracy{this->get_interpolator_accuracy(optimization_hypers)};
        // minimal distance such that it is still stable with the interpolated
        // function
        double range_begin{math::SPHERICAL_BESSEL_FUNCTION_FTOL};
        double range_end{this->interaction_cutoff};
        this->init_interpolator(range_begin, range_end, accuracy);
      }

      void init_interpolator(const double range_begin, const double range_end,
                             const double accuracy) {
        int species;
        for (auto it = projection_matrices.begin();
             it != projection_matrices.end(); ++it) {
          species = it->first;
          // "this" is passed by reference and is mutable
          std::function<Matrix_t(double)> func{
              [&](const double distance) mutable {
                this->compute_neighbour_contribution(distance, species);
                return this->reduced_radial_integral_neighbour;
              }};
          Matrix_t result = func(range_begin);
          int cols{static_cast<int>(result.cols())};
          int rows{static_cast<int>(result.rows())};
          this->intps.insert(std::pair<int, std::unique_ptr<Spline_t>>(
              species, std::make_unique<Spline_t>(func, range_begin, range_end,
                                                  accuracy, cols, rows)));
        }
      }

      double get_interpolator_accuracy(const Hypers_t & optimization_hypers) {
        auto spline_hypers =
            optimization_hypers.at("Spline").template get<json>();
        if (spline_hypers.count("accuracy")) {
          return spline_hypers.at("accuracy").template get<double>();
        } else {
          std::stringstream err_str{};
          err_str << "No Spline accuracy was given.";
          throw std::logic_error(err_str.str());
        }
      }

      double get_cutoff(const Hypers_t & hypers) {
        auto fc_hypers = hypers.at("cutoff_function").template get<json>();
        return fc_hypers.at("cutoff").at("value").template get<double>();
      }

      Matrix_t reduced_radial_integral_neighbour{};
      Matrix_t reduced_radial_neighbour_derivative{};
      std::map<int, Vector_t> reduced_radial_integral_centers{};
      std::map<int, std::vector<Matrix_t>> projection_matrices{};
      size_t n_components{};
      size_t n_species{};
      // 1/(2σ^2)
      double fac_a{};
      std::map<int, std::unique_ptr<Spline_t>> intps{};
    };

  }  // namespace internal

  template <internal::RadialBasisType Type, class Hypers>
  auto make_radial_integral(const Hypers & basis_hypers) {
    return std::static_pointer_cast<internal::RadialContributionBase>(
        std::make_shared<internal::RadialContribution<Type>>(basis_hypers));
  }

  template <internal::RadialBasisType Type>
  auto downcast_radial_integral(
      const std::shared_ptr<internal::RadialContributionBase> &
          radial_integral) {
    return std::static_pointer_cast<internal::RadialContribution<Type>>(
        radial_integral);
  }

  template <internal::RadialBasisType RBT, internal::AtomicSmearingType AST,
            internal::OptimizationType OT, class Hypers>
  auto make_radial_integral_handler(const Hypers & basis_hypers) {
    return std::static_pointer_cast<internal::RadialContributionBase>(
        std::make_shared<internal::RadialContributionHandler<RBT, AST, OT>>(
            basis_hypers));
  }

  template <internal::RadialBasisType RBT, internal::AtomicSmearingType AST,
            internal::OptimizationType OT>
  auto downcast_radial_integral_handler(
      const std::shared_ptr<internal::RadialContributionBase> &
          radial_integral) {
    return std::static_pointer_cast<
        internal::RadialContributionHandler<RBT, AST, OT>>(radial_integral);
  }

  /**
   * Handles the expansion of an environment in a spherical and radial basis.
   *
   * The local environment of each atom is represented by Gaussians of a
   * certain width (user-defined; can be constant, species-dependent, or
   * radially dependent).  This density field is expanded in an angular basis
   * of spherical harmonics (à la SOAP) and a radial basis of
   * either Gaussians (again, as in SOAP/SphericalInvariants) or one of the more
   * recent bases currently under development.
   */
  class CalculatorSphericalExpansion : public CalculatorBase {
   public:
    using Parent = CalculatorBase;
    using Hypers_t = typename Parent::Hypers_t;
    using ReferenceHypers_t = Parent::ReferenceHypers_t;
    using Key_t = typename Parent::Key_t;

    template <class StructureManager>
    using Property_t = BlockSparseProperty<double, 1, StructureManager, Key_t>;
    template <class StructureManager>
    using PropertyGradient_t =
        BlockSparseProperty<double, 2, StructureManager, Key_t>;

    template <class StructureManager>
    using Dense_t = typename Property_t<StructureManager>::Dense_t;
    template <class StructureManager>
    using Data_t = typename Property_t<StructureManager>::Data_t;

    template <class StructureManager, size_t Order>
    using ClusterRef_t = typename StructureManager::template ClusterRef<Order>;

    using Matrix_t = math::Matrix_t;
    using Vector_t = math::Vector_t;
    using Matrix_Ref = math::Matrix_Ref;
    using Vector_Ref = math::Vector_Ref;

    /**
     * Set the hyperparameters of this descriptor from a json-like container.
     *
     * @param hypers structure (usually parsed from json) containing the
     *               options and hyperparameters
     *
     * @todo (max, felix) document the SOAP/SphExpn-specific hypers here
     *
     * @throw logic_error if an invalid option or combination of options is
     *                    specified in the structure
     */
    void set_hyperparameters(const Hypers_t & hypers) override {
      using internal::AtomicSmearingType;
      using internal::CutoffFunctionType;
      using internal::OptimizationType;
      using internal::RadialBasisType;
      this->hypers = hypers;

      this->max_radial = hypers.at("max_radial").get<size_t>();
      this->max_angular = hypers.at("max_angular").get<size_t>();
      if (hypers.count("compute_gradients")) {
        this->compute_gradients = hypers.at("compute_gradients").get<bool>();
      } else {  // Default false (don't compute gradients)
        this->compute_gradients = false;
      }

      if (hypers.count("expansion_by_species_method")) {
        std::set<std::string> possible_expansion_by_species{
            {"environment wise", "user defined", "structure wise"}};
        auto expansion_by_species_tmp =
            hypers.at("expansion_by_species_method").get<std::string>();
        if (possible_expansion_by_species.count(expansion_by_species_tmp)) {
          this->expansion_by_species = expansion_by_species_tmp;
        } else {
          std::stringstream err_str{};
          err_str << "expansion_by_species_method provided:'"
                  << expansion_by_species_tmp
                  << "' is not part of the implemented methods: '";
          for (const auto & val : possible_expansion_by_species) {
            err_str << val << "', ";
          }
          throw std::logic_error(err_str.str());
        }
      } else {
        // default value for backward compatibility
        this->expansion_by_species = "environment wise";
      }

      if (hypers.count("global_species")) {
        auto species = hypers.at("global_species").get<Key_t>();
        for (const auto & sp : species) {
          this->global_species.insert({sp});
        }
      } else {
        if (this->expansion_by_species == "user defined") {
          std::stringstream err_str{};
          err_str << "expansion_by_species_method is 'user defined'"
                  << " but global_species is not defined.";
          throw std::logic_error(err_str.str());
        }
        this->global_species.clear();
      }

      this->spherical_harmonics.precompute(this->max_angular,
                                           this->compute_gradients);

      // create the class that will compute the radial terms of the
      // expansion. the atomic smearing is an integral part of the
      // radial contribution
      auto smearing_hypers = hypers.at("gaussian_density").get<json>();
      auto smearing_type = smearing_hypers.at("type").get<std::string>();

      if (smearing_type == "Constant") {
        this->atomic_smearing_type = AtomicSmearingType::Constant;
      } else if (smearing_type == "PerSpecies") {
        throw std::logic_error("Requested Smearing type \'PerSpecies\'"
                               "\' has not been implemented.  Must be one of"
                               ": \'Constant\'.");
      } else if (smearing_type == "Radial") {
        throw std::logic_error("Requested Smearing type \'Radial\'"
                               "\' has not been implemented.  Must be one of"
                               ": \'Constant\'.");
      } else {
        throw std::logic_error("Requested Smearing type \'" + smearing_type +
                               "\' is unknown.  Must be one of" +
                               ": \'Constant\'.");
      }

      auto radial_contribution_hypers =
          hypers.at("radial_contribution").get<json>();
      auto radial_contribution_type =
          radial_contribution_hypers.at("type").get<std::string>();

      // create the class that will compute the radial terms of the
      // expansion. the atomic smearing is an integral part of the
      // radial contribution
      if (radial_contribution_type == "GTO") {
        auto rc_shared = std::make_shared<
            internal::RadialContribution<RadialBasisType::GTO>>(hypers);
        this->atomic_smearing_type = rc_shared->atomic_smearing_type;
        this->radial_integral = rc_shared;
        this->radial_integral_type = RadialBasisType::GTO;

      } else if (radial_contribution_type == "DVR") {
        auto rc_shared = std::make_shared<
            internal::RadialContribution<RadialBasisType::DVR>>(hypers);
        this->atomic_smearing_type = rc_shared->atomic_smearing_type;
        this->radial_integral = rc_shared;
        this->radial_integral_type = RadialBasisType::DVR;
      } else {
        throw std::logic_error("Requested Radial contribution type \'" +
                               radial_contribution_type +
                               "\' has not been implemented.  Must be one of" +
                               ": \'GTO\' or \'DVR\'. ");
      }

      if (radial_contribution_hypers.count("optimization")) {
        auto optimization_hypers =
            radial_contribution_hypers.at("optimization").get<json>();
        // Checks for all optimization args used for the computation of the
        // spherical expansion
        if (optimization_hypers.count("Spline") &&
            optimization_hypers.count("RadialDimReduction")) {
          this->optimization_type = OptimizationType::RadialDimReductionSpline;
        } else if (optimization_hypers.count("Spline")) {
          this->optimization_type = OptimizationType::Spline;
        } else if (optimization_hypers.count("RadialDimReduction")) {
          throw std::logic_error("RadialDimReduction requires\'"
                                 "\' Spline to be set.");
        } else {
          this->optimization_type = OptimizationType::None;
        }
      } else {
        this->optimization_type = OptimizationType::None;
      }

      this->set_radial_integral(hypers);

      auto fc_hypers = hypers.at("cutoff_function").get<json>();
      auto fc_type = fc_hypers.at("type").get<std::string>();
      this->interaction_cutoff = fc_hypers.at("cutoff").at("value");
      this->cutoff_smooth_width = fc_hypers.at("smooth_width").at("value");
      if (fc_type == "ShiftedCosine") {
        this->cutoff_function_type = CutoffFunctionType::ShiftedCosine;
        this->cutoff_function =
            make_cutoff_function<CutoffFunctionType::ShiftedCosine>(fc_hypers);
      } else if (fc_type == "RadialScaling") {
        this->cutoff_function_type = CutoffFunctionType::RadialScaling;
        this->cutoff_function =
            make_cutoff_function<CutoffFunctionType::RadialScaling>(fc_hypers);
      } else {
        throw std::logic_error("Requested cutoff function type \'" + fc_type +
                               "\' has not been implemented.  Must be one of" +
                               ": \'ShiftedCosine\' or 'RadialScaling'.");
      }

      this->set_name(hypers);
    }

    bool operator==(const CalculatorSphericalExpansion & other) const {
      bool is_equal{
          (this->does_gradients() == other.does_gradients()) and
          (this->interaction_cutoff == other.interaction_cutoff) and
          (this->cutoff_smooth_width == other.cutoff_smooth_width) and
          (this->interpolator_accuracy == other.interpolator_accuracy) and
          (this->max_radial == other.max_radial) and
          (this->max_angular == other.max_angular) and
          (this->atomic_smearing_type == other.atomic_smearing_type) and
          (this->radial_integral_type == other.radial_integral_type) and
          (this->optimization_type == other.optimization_type) and
          (this->cutoff_function_type == other.cutoff_function_type)};
      return is_equal;
    }

    /**
     * Returns if the calculator is able to compute gradients of the
     * representation w.r.t. atomic positions ?
     */
    bool does_gradients() const override { return this->compute_gradients; }

    /**
     * Construct a new Calculator using a hyperparameters container
     *
     * See set_hyperparameters() for a description of the hypers
     *
     * @todo (max) ffs, why isn't the link above working in Sphinx?
     *             And why aren't the todos showing?!
     *
     * @param hyper container (usually parsed from json) for the options and
     *              hyperparameters
     *
     * @throw logic_error if an invalid option or combination of options is
     *                    specified in the container
     */
    explicit CalculatorSphericalExpansion(const Hypers_t & hyper)
        : CalculatorBase{} {
      this->set_default_prefix("spherical_expansion_");
      this->set_hyperparameters(hyper);
    }

    //! Copy constructor
    CalculatorSphericalExpansion(const CalculatorSphericalExpansion & other) =
        delete;

    //! Move constructor
    CalculatorSphericalExpansion(CalculatorSphericalExpansion && other) noexcept
        : CalculatorBase{std::move(other)}, interaction_cutoff{std::move(
                                                other.interaction_cutoff)},
          cutoff_smooth_width{std::move(other.cutoff_smooth_width)},
          interpolator_accuracy{std::move(other.interpolator_accuracy)},
          max_radial{std::move(other.max_radial)}, max_angular{std::move(
                                                       other.max_angular)},
          compute_gradients{std::move(other.compute_gradients)},
          expansion_by_species{std::move(other.expansion_by_species)},
          global_species{std::move(other.global_species)},
          atomic_smearing_type{std::move(other.atomic_smearing_type)},
          radial_integral{std::move(other.radial_integral)},
          radial_integral_type{std::move(other.radial_integral_type)},
          optimization_type{std::move(other.optimization_type)},
          cutoff_function{std::move(other.cutoff_function)},
          cutoff_function_type{std::move(other.cutoff_function_type)},
          spherical_harmonics{std::move(other.spherical_harmonics)} {}

    //! Destructor
    virtual ~CalculatorSphericalExpansion() = default;

    //! Copy assignment operator
    CalculatorSphericalExpansion &
    operator=(const CalculatorSphericalExpansion & other) = delete;

    //! Move assignment operator
    CalculatorSphericalExpansion &
    operator=(CalculatorSphericalExpansion && other) = default;

    /**
     * Compute representation for a given structure manager.
     *
     * @tparam StructureManager a (single or collection)
     * of structure manager(s) (in an iterator) held in shared_ptr
     */
    template <class StructureManager>
    void compute(StructureManager & managers);

    //! choose the RadialBasisType and AtomicSmearingType from the hypers
    template <internal::CutoffFunctionType FcType, class StructureManager>
    void compute_by_radial_contribution(StructureManager & managers);

    /**
     *  When the RadialBasisType has beeen already selected this function
     *  serves for a distinction of the cases for the optimization type
     */
    template <internal::CutoffFunctionType FcType,
              internal::RadialBasisType RadialType,
              internal::AtomicSmearingType SmearingType, class StructureManager>
    void compute_by_radial_contribution(StructureManager & managers);

    /**
     * loop over a collection of manangers if it is an iterator.
     * Or just call compute_impl() if it's a single manager (see below)
     */
    template <
        internal::CutoffFunctionType FcType,
        internal::RadialBasisType RadialType,
        internal::AtomicSmearingType SmearingType,
        internal::OptimizationType OptType, class StructureManager,
        std::enable_if_t<internal::is_proper_iterator<StructureManager>::value,
                         int> = 0>
    void compute_loop(StructureManager & managers) {
      for (auto & manager : managers) {
        this->compute_impl<FcType, RadialType, SmearingType, OptType>(manager);
      }
    }

    //! single manager case
    template <internal::CutoffFunctionType FcType,
              internal::RadialBasisType RadialType,
              internal::AtomicSmearingType SmearingType,
              internal::OptimizationType OptType, class StructureManager,
              std::enable_if_t<
                  not(internal::is_proper_iterator<StructureManager>::value),
                  int> = 0>
    void compute_loop(StructureManager & manager) {
      this->compute_impl<FcType, RadialType, SmearingType, OptType>(manager);
    }

    //! Compute the spherical expansion given several options
    template <internal::CutoffFunctionType FcType,
              internal::RadialBasisType RadialType,
              internal::AtomicSmearingType SmearingType,
              internal::OptimizationType OptType, class StructureManager>
    void compute_impl(std::shared_ptr<StructureManager> manager);

   protected:
    //! cutoff radius r_c defining the size of the atom centered environment
    double interaction_cutoff{};
    //! size of the transition region r_t spanning [r_c-r_t, r_c] in which the
    //! contributions to the environment expansion go to zero smoothly
    double cutoff_smooth_width{};
    //! defines the maximal mean error allowed to the interpolator when fitting
    //! the reference
    double interpolator_accuracy{};
    //! number of radial basis function to use in the expansion
    size_t max_radial{};
    /**
     * number of angular channels used in the expansion, i.e. all Y_l^m with
     * l < max_angular + 1 are used (the +1 is to follow GAP's convention).
     */
    size_t max_angular{};
    //! controls the computation of the gradients of the expansion wrt. atomic
    //! positions
    bool compute_gradients{};
    /**
     * defines the method to determine the set of species to use in the
     * expansion
     */
    std::string expansion_by_species{};

    //! user defined species appearing in the expansion indexing
    std::set<Key_t> global_species{};

    internal::AtomicSmearingType atomic_smearing_type{};

    std::shared_ptr<internal::RadialContributionBase> radial_integral{};
    internal::RadialBasisType radial_integral_type{};

    internal::OptimizationType optimization_type{};

    std::shared_ptr<internal::CutoffFunctionBase> cutoff_function{};
    internal::CutoffFunctionType cutoff_function_type{};

    math::SphericalHarmonics spherical_harmonics{};

    /**
     * set up chemical keys of the expension so that only species appearing in
     * the environment are present and initialize coeffs to zero.
     *
     * For gradients associated with pair_ii the keys will be the ones in the
     * environment but the ones associated with pair_ij will only contain the
     * non zero keys.
     */
    template <class StructureManager>
    void initialize_expansion_environment_wise(
        std::shared_ptr<StructureManager> & managers,
        Property_t<StructureManager> & expansions_coefficients,
        PropertyGradient_t<StructureManager> &
            expansions_coefficients_gradient);

    /**
     * set up chemical keys of the expension so that all species in the
     * structure will be used in the as keys for the expansion and initialize
     * coeffs to zero.
     * For gradients associated with pair_ii the keys will be the one of the
     * structure but the ones associated with pair_ij will only contain the
     * non zero keys.
     *
     * @throw runtime_error when all the species of the structure are not
     * present in global_species
     */
    template <class StructureManager>
    void initialize_expansion_structure_wise(
        std::shared_ptr<StructureManager> & managers,
        Property_t<StructureManager> & expansions_coefficients,
        PropertyGradient_t<StructureManager> &
            expansions_coefficients_gradient);

    /**
     * Set up chemical keys of the expension using global_species for the keys
     * appearing in the expansion and initialize coeffs to zero.
     * For gradients associated with pair_ii the keys will be the one of
     * global_species but the ones associated with pair_ij will only contain the
     * non zero keys.
     *
     * @throw runtime_error when all the species of the structure are not
     * present in global_species
     */
    template <class StructureManager>
    void initialize_expansion_with_global_species(
        std::shared_ptr<StructureManager> & managers,
        Property_t<StructureManager> & expansions_coefficients,
        PropertyGradient_t<StructureManager> &
            expansions_coefficients_gradient);

   private:
    void set_radial_integral(Hypers_t hypers) {
      using internal::AtomicSmearingType;
      using internal::RadialBasisType;
      switch (internal::combine_enums(this->radial_integral_type,
                                      this->atomic_smearing_type)) {
      case internal::combine_enums(RadialBasisType::GTO,
                                   AtomicSmearingType::Constant): {
        this->set_radial_integral<RadialBasisType::GTO,
                                  AtomicSmearingType::Constant>(hypers);
        break;
      }
      case internal::combine_enums(RadialBasisType::DVR,
                                   AtomicSmearingType::Constant): {
        this->set_radial_integral<RadialBasisType::DVR,
                                  AtomicSmearingType::Constant>(hypers);
        break;
      }
      default:
        std::basic_ostringstream<char> err_message;
        err_message
            << "Invalid combination of atomic smearing and radial basis ";
        err_message << "type encountered (This is a bug.  Debug info for ";
        err_message << "developers: "
                    << "radial_integral_type == ";
        err_message << static_cast<int>(this->radial_integral_type);
        err_message << ", atomic_smearing_type == ";
        err_message << static_cast<int>(this->atomic_smearing_type);
        err_message << ")" << std::endl;
        throw std::logic_error(err_message.str());
      }
    }

    template <internal::RadialBasisType RadialType,
              internal::AtomicSmearingType AST>
    void set_radial_integral(Hypers_t hypers) {
      using internal::OptimizationType;
      switch (this->optimization_type) {
      case (OptimizationType::None): {
        auto rc_shared = std::make_shared<internal::RadialContributionHandler<
            RadialType, AST, OptimizationType::None>>(hypers);
        this->radial_integral = rc_shared;
        break;
      }
      case (OptimizationType::Spline): {
        auto rc_shared = std::make_shared<internal::RadialContributionHandler<
            RadialType, AST, OptimizationType::Spline>>(hypers);
        this->radial_integral = rc_shared;
        break;
      }
      case (OptimizationType::RadialDimReductionSpline): {
        auto rc_shared = std::make_shared<internal::RadialContributionHandler<
            RadialType, AST, OptimizationType::RadialDimReductionSpline>>(
            hypers);
        this->radial_integral = rc_shared;
        break;
      }
      default:
        std::basic_ostringstream<char> err_message;
        err_message << "Invalid optimization type == ";
        err_message << static_cast<int>(this->optimization_type);
        err_message << "(C++ side)" << std::endl;
        throw std::logic_error(err_message.str());
      }
    }
  };

  // compute classes template construction
  template <class StructureManager>
  void CalculatorSphericalExpansion::compute(StructureManager & managers) {
    // specialize based on the cutoff function
    using internal::CutoffFunctionType;

    switch (this->cutoff_function_type) {
    case CutoffFunctionType::ShiftedCosine:
      this->compute_by_radial_contribution<CutoffFunctionType::ShiftedCosine>(
          managers);
      break;
    case CutoffFunctionType::RadialScaling:
      this->compute_by_radial_contribution<CutoffFunctionType::RadialScaling>(
          managers);
      break;
    default:
      // The control flow really should never reach here.  But just in case,
      // provide the necessary information to debug this problem.
      std::basic_ostringstream<char> err_message;
      err_message << "Invalid cutoff function type encountered ";
      err_message << "(This is a bug.  Debug info for developers: ";
      err_message << "cutoff_function_type == ";
      err_message << static_cast<int>(this->cutoff_function_type);
      err_message << ")" << std::endl;
      throw std::logic_error(err_message.str());
      break;
    }
  }

  template <internal::CutoffFunctionType FcType, class StructureManager>
  void CalculatorSphericalExpansion::compute_by_radial_contribution(
      StructureManager & managers) {
    // specialize based on the type of radial contribution
    using internal::AtomicSmearingType;
    using internal::RadialBasisType;

    switch (internal::combine_enums(this->radial_integral_type,
                                    this->atomic_smearing_type)) {
    case internal::combine_enums(RadialBasisType::GTO,
                                 AtomicSmearingType::Constant): {
      this->compute_by_radial_contribution<FcType, RadialBasisType::GTO,
                                           AtomicSmearingType::Constant>(
          managers);
      break;
    }
    case internal::combine_enums(RadialBasisType::DVR,
                                 AtomicSmearingType::Constant): {
      this->compute_by_radial_contribution<FcType, RadialBasisType::DVR,
                                           AtomicSmearingType::Constant>(
          managers);
      break;
    }
    default:
      // The control flow really should never reach here.  In this case, any
      // "invalid combination of parameters" should have already been handled at
      // the parameter processing stage where the user can be notified in a
      // helpful way.  But in case we do get here, provide the necessary
      // information to debug this problem.
      std::basic_ostringstream<char> err_message;
      err_message << "Invalid combination of atomic smearing and radial basis ";
      err_message << "type encountered (This is a bug.  Debug info for ";
      err_message << "developers: "
                  << "radial_integral_type == ";
      err_message << static_cast<int>(this->radial_integral_type);
      err_message << ", atomic_smearing_type == ";
      err_message << static_cast<int>(this->atomic_smearing_type);
      err_message << ")" << std::endl;
      throw std::logic_error(err_message.str());
    }
  }

  template <internal::CutoffFunctionType FcType,
            internal::RadialBasisType RadialType,
            internal::AtomicSmearingType SmearingType, class StructureManager>
  void CalculatorSphericalExpansion::compute_by_radial_contribution(
      StructureManager & managers) {
    using internal::OptimizationType;
    switch (this->optimization_type) {
    case (OptimizationType::None): {
      this->compute_loop<FcType, RadialType, SmearingType,
                         OptimizationType::None>(managers);
      break;
    }
    case (OptimizationType::Spline): {
      this->compute_loop<FcType, RadialType, SmearingType,
                         OptimizationType::Spline>(managers);
      break;
    }
    case (OptimizationType::RadialDimReductionSpline): {
      this->compute_loop<FcType, RadialType, SmearingType,
                         OptimizationType::RadialDimReductionSpline>(managers);
      break;
    }
    default:
      std::basic_ostringstream<char> err_message;
      err_message << "Invalid optimization type == ";
      err_message << static_cast<int>(this->optimization_type);
      err_message << "(C++ side)" << std::endl;
      throw std::logic_error(err_message.str());
    }
  }

  /**
   * Compute the spherical expansion
   */
  template <internal::CutoffFunctionType FcType,
            internal::RadialBasisType RadialType,
            internal::AtomicSmearingType SmearingType,
            internal::OptimizationType OptType, class StructureManager>
  void CalculatorSphericalExpansion::compute_impl(
      std::shared_ptr<StructureManager> manager) {
    using Prop_t = Property_t<StructureManager>;
    using PropGrad_t = PropertyGradient_t<StructureManager>;
    constexpr static bool IsHalfNL{
        StructureManager::traits::NeighbourListType ==
        AdaptorTraits::NeighbourListType::half};
    using math::PI;
    using math::pow;
    constexpr bool ExcludeGhosts{true};
    const bool is_not_masked{manager->is_not_masked()};
    const bool compute_gradients{this->compute_gradients};
    if (not is_not_masked and compute_gradients) {
      throw std::logic_error("Can't compute spherical expansion gradients with "
                             "masked center atoms");
    }
    if (not is_not_masked and IsHalfNL) {
      std::stringstream err_str{};
      err_str << "Half neighbor list should only be used when all the "
              << "atoms inside the unit cell are centers, i.e. "
              << "center_atoms_mask should not mask atoms.";
      throw std::runtime_error(err_str.str());
    }
    //auto manager_root = extract_underlying_manager<0>(manager);
    //auto cell_length = manager_root->get_cell_length();
    //auto pbc = manager_root->get_periodic_boundary_conditions();
    //bool is_cutoff_too_large{false};
    //for (size_t i_dim{0}; i_dim < ThreeD; ++i_dim) {
    //  if (pbc[i_dim]) {
    //    if (cell_length[i_dim] < 2. * this->interaction_cutoff) {
    //      is_cutoff_too_large = true;
    //    }
    //  }
    //}
    //if (IsHalfNL and is_cutoff_too_large) {
    //  std::stringstream err_str{};
    //  err_str << "Half neighbor list should only be used when the diameter of "
    //          << "the spherical expansion is smaller than the unit cell "
    //          << "in periodic directions: "
    //          << "[" << cell_length.transpose() << "] > "
    //          << 2 * this->interaction_cutoff;
    //  throw std::runtime_error(err_str.str());
    //}

    auto && expansions_coefficients{*manager->template get_property<Prop_t>(
        this->get_name(), true, true, ExcludeGhosts)};

    auto && expansions_coefficients_gradient{
        *manager->template get_property<PropGrad_t>(this->get_gradient_name(),
                                                    true, true)};

    // if the representation has already been computed for the current
    // structure then do nothing
    if (expansions_coefficients.is_updated()) {
      return;
    }

    // downcast cutoff and radial contributions so they are functional
    auto cutoff_function{
        downcast_cutoff_function<FcType>(this->cutoff_function)};
    auto radial_integral{
        downcast_radial_integral_handler<RadialType, SmearingType, OptType>(
            this->radial_integral)};
    auto n_row{this->max_radial};
    // to store linearly all l,m components with
    // -l-1<=m<=l+1 needs (l+1)**2 elements
    auto n_col{(this->max_angular + 1) * (this->max_angular + 1)};
    expansions_coefficients.clear();
    expansions_coefficients.set_shape(n_row, n_col);
    if (compute_gradients) {
      expansions_coefficients_gradient.clear();
      // Row-major ordering, so the Cartesian (spatial) index varies slowest
      expansions_coefficients_gradient.set_shape(ThreeD * n_row, n_col);
    }

    if (this->expansion_by_species == "environment wise") {
      this->initialize_expansion_environment_wise(
          manager, expansions_coefficients, expansions_coefficients_gradient);
    } else if (this->expansion_by_species == "user defined") {
      this->initialize_expansion_with_global_species(
          manager, expansions_coefficients, expansions_coefficients_gradient);
    } else if (this->expansion_by_species == "structure wise") {
      this->initialize_expansion_structure_wise(
          manager, expansions_coefficients, expansions_coefficients_gradient);
    } else {
      throw std::runtime_error("should not arrive here");
    }

    // coeff C^{ij}_{nlm}
    auto c_ij_nlm = math::Matrix_t(n_row, n_col);

    for (auto center : manager) {
      // c^{i}
      auto & coefficients_center = expansions_coefficients[center];
      // \grad_i c^{i}
      auto & coefficients_center_gradient =
          expansions_coefficients_gradient[center.get_atom_ii()];
      auto atom_i_tag = center.get_atom_tag();
      Key_t center_type{center.get_atom_type()};

      // Start the accumulation with the central atom contribution
      coefficients_center[center_type].col(0) +=
          radial_integral->template compute_center_contribution(
              center, center.get_atom_type()) /
          sqrt(4.0 * PI);

      for (auto neigh : center.pairs()) {
        auto atom_j = neigh.get_atom_j();
        const int atom_j_tag = atom_j.get_atom_tag();
        const bool is_center_atom{manager->is_center_atom(neigh)};

        const double & dist{manager->get_distance(neigh)};
        const auto direction{manager->get_direction_vector(neigh)};
        Key_t neigh_type{neigh.get_atom_type()};
        this->spherical_harmonics.calc(direction, compute_gradients);
        auto && harmonics{spherical_harmonics.get_harmonics()};
        auto && harmonics_gradients{
            spherical_harmonics.get_harmonics_derivatives()};
        auto && neighbour_contribution =
            radial_integral->template compute_neighbour_contribution(
                dist, neigh, neigh.get_atom_type());
        double f_c{cutoff_function->f_c(dist)};
        auto coefficients_center_by_type{coefficients_center[neigh_type]};

        // compute the coefficients
        size_t l_block_idx{0};
        for (size_t angular_l{0}; angular_l < this->max_angular + 1;
             ++angular_l) {
          size_t l_block_size{2 * angular_l + 1};
          c_ij_nlm.block(0, l_block_idx, this->max_radial, l_block_size) =
              neighbour_contribution.col(angular_l) *
              harmonics.segment(l_block_idx, l_block_size);
          l_block_idx += l_block_size;
        }
        c_ij_nlm *= f_c;
        coefficients_center_by_type += c_ij_nlm;

        // half list branch for c^{ji} terms using
        // c^{ij}_{nlm} = (-1)^l c^{ji}_{nlm}.
        if (IsHalfNL) {
          if (is_center_atom) {
            auto & coefficients_neigh{expansions_coefficients[atom_j]};
            auto coefficients_neigh_by_type{coefficients_neigh[center_type]};
            l_block_idx = 0;
            double parity{1.};
            for (size_t angular_l{0}; angular_l < this->max_angular + 1;
                 ++angular_l) {
              size_t l_block_size{2 * angular_l + 1};
              coefficients_neigh_by_type.block(0, l_block_idx, this->max_radial,
                                               l_block_size) +=
                  parity * c_ij_nlm.block(0, l_block_idx, this->max_radial,
                                          l_block_size);
              l_block_idx += l_block_size;
              parity *= -1.;
            }
          }
        }

        // compute the gradients of the coefficients with respect to
        // atoms positions
        // but only if the neighbour is _not_ an image of the center!
        // (the periodic images move with the center, so their contribution to
        // the center gradient is zero)
        if (compute_gradients) {  // NOLINT
          // \grad_j c^i
          auto & coefficients_neigh_gradient =
              expansions_coefficients_gradient[neigh];

          auto && neighbour_derivative =
              radial_integral->compute_neighbour_derivative(
                  dist, neigh, neigh.get_atom_type());
          double df_c{cutoff_function->df_c(dist)};
          // The type of the contribution c^{ij} to the coefficient c^{i}
          // depends on the type of j (and it is the same for the gradients)
          // In the following atom i is of type a and atom j is of type b

          // grad_i c^{ib}
          auto && gradient_center_by_type{
              coefficients_center_gradient[neigh_type]};
          // grad_j c^{ib}
          auto && gradient_neigh_by_type{
              coefficients_neigh_gradient[neigh_type]};

          // clang-format off
          // d/dr_{ij} (c_{ij} f_c{r_{ij}})
          Matrix_t pair_gradient_contribution_p1 =
                ((neighbour_derivative * f_c)
                 + (neighbour_contribution * df_c));
          // grad_j c^{ij}
          Matrix_t pair_gradient_contribution{this->max_radial,
                                              this->max_angular + 1};
          for (int cartesian_idx{0}; cartesian_idx < ThreeD;
                 ++cartesian_idx) {
            l_block_idx = 0;
            for (size_t angular_l{0}; angular_l < this->max_angular + 1;
                ++angular_l) {
              size_t l_block_size{2 * angular_l + 1};
              pair_gradient_contribution.resize(this->max_radial, l_block_size);
              pair_gradient_contribution =
                pair_gradient_contribution_p1.col(angular_l)
                * harmonics.segment(l_block_idx, l_block_size)
                * direction(cartesian_idx);
              pair_gradient_contribution +=
                  neighbour_contribution.col(angular_l)
                  * harmonics_gradients.block(cartesian_idx, l_block_idx,
                                              1, l_block_size)
                  * f_c / dist;

              // Each Cartesian gradient component occupies a contiguous block
              // (row-major storage)
              // grad_i c^{ib} = - \sum_{j} grad_j c^{ijb}
              if (atom_j_tag != atom_i_tag) {
                gradient_center_by_type.block(
                    cartesian_idx * max_radial, l_block_idx,
                    max_radial, l_block_size) -= pair_gradient_contribution;
              }
              // grad_j c^{ib} =  grad_j c^{ijb}
              gradient_neigh_by_type.block(
                  cartesian_idx * max_radial, l_block_idx,
                  max_radial, l_block_size) = pair_gradient_contribution;
              l_block_idx += l_block_size;
              // clang-format on
            }  // for (angular_l)
          }    // for cartesian_idx

          // half list branch for accumulating parts of grad_j c^{j} using
          // grad_j c^{ji a} = (-1)^l grad_j c^{ij b}
          if (IsHalfNL) {
            if (is_center_atom) {
              // grad_j c^{j}
              auto & coefficients_neigh_center_gradient =
                  expansions_coefficients_gradient[neigh.get_atom_jj()];
              // grad_j c^{j a}
              auto gradient_neigh_center_by_type =
                  coefficients_neigh_center_gradient[center_type];

              for (int cartesian_idx{0}; cartesian_idx < ThreeD;
                   ++cartesian_idx) {
                l_block_idx = 0;
                double parity{1};
                for (size_t angular_l{0}; angular_l < this->max_angular + 1;
                     ++angular_l) {
                  size_t l_block_size{2 * angular_l + 1};
                  // clang-format off
                  gradient_neigh_center_by_type.block(
                      cartesian_idx * max_radial, l_block_idx,
                      max_radial, l_block_size) += parity *
                                  gradient_neigh_by_type.block(
                                    cartesian_idx * max_radial, l_block_idx,
                                    max_radial, l_block_size);

                  l_block_idx += l_block_size;
                  parity *= -1.;
                  // clang-format on
                }  // for (angular_l)
              }    // for cartesian_idx
            }      // if (is_center_atom)
          }        // if (IsHalfNL)
        }          // if (compute_gradients)
      }            // for (neigh : center)

      // Normalize and orthogonalize the radial coefficients
      radial_integral->finalize_coefficients(coefficients_center);
      if (compute_gradients) {
        radial_integral->template finalize_coefficients_der<ThreeD>(
            expansions_coefficients_gradient, center);
      }
    }  // for (center : manager)
  }    // compute()

  template <class StructureManager>
  void CalculatorSphericalExpansion::initialize_expansion_environment_wise(
      std::shared_ptr<StructureManager> & manager,
      Property_t<StructureManager> & expansions_coefficients,
      PropertyGradient_t<StructureManager> & expansions_coefficients_gradient) {
    constexpr static bool IsHalfNL{
        StructureManager::traits::NeighbourListType ==
        AdaptorTraits::NeighbourListType::half};
    std::vector<std::set<Key_t>> keys_list{};
    std::vector<std::set<Key_t>> keys_list_grad{};
    std::map<int, int> center_tag2idx{};
    const bool compute_gradients{this->compute_gradients};
    int i_center{0};
    int j_neigh{0};
    for (auto center : manager) {
      center_tag2idx[center.get_atom_tag()] = i_center;
      i_center++;
      //keys_list.emplace_back();
      if (compute_gradients) {
        for (auto neigh : center.pairs_with_self_pair()) {
          j_neigh++;
          // TODO(alex) why was this used? seems unnessary complicated?
          //(void)neigh;  // to avoid compiler warning
          //keys_list_grad.emplace_back();
        }
      }
    }
    keys_list.resize(i_center);
    keys_list_grad.resize(j_neigh);
    int i_grad{0};
    i_center = 0;
    for (auto center : manager) {
      Key_t center_type{center.get_atom_type()};

      for (auto neigh : center.pairs()) {
        keys_list[i_center].insert({neigh.get_atom_type()});
        // TODO(alex) remove after debug finished
        //std::cout << "strict pair (" << center.get_atom_tag() << ", "
        //  << neigh.get_atom_tag() << ") global index "
        //  << neigh.get_global_index() << std::endl;
        if (IsHalfNL && manager->is_center_atom(neigh)) {
          auto atom_j = neigh.get_atom_j();
          auto j_center = center_tag2idx[atom_j.get_atom_tag()];
          keys_list[j_center].insert(center_type);
        }
      }
      keys_list[i_center].insert({center_type});
      if (compute_gradients) {
        keys_list_grad[i_grad].insert(keys_list[i_center].begin(),
                                      keys_list[i_center].end());
        i_grad++;
        for (auto neigh : center.pairs()) {
          Key_t neigh_type{neigh.get_atom_type()};
          keys_list_grad[i_grad].insert(neigh_type);
          i_grad++;
        }
      }  // if (compute_gradients)
      i_center++;
    }

    expansions_coefficients.resize(keys_list);
    expansions_coefficients.setZero();

    if (compute_gradients) {
      expansions_coefficients_gradient.resize(keys_list_grad);
      expansions_coefficients_gradient.setZero();
    } else {
      expansions_coefficients_gradient.resize();
    }
  }

  template <class StructureManager>
  void CalculatorSphericalExpansion::initialize_expansion_structure_wise(
      std::shared_ptr<StructureManager> & manager,
      Property_t<StructureManager> & expansions_coefficients,
      PropertyGradient_t<StructureManager> & expansions_coefficients_gradient) {
    std::set<Key_t> keys{};
    for (auto center : manager) {
      Key_t center_type{center.get_atom_type()};
      keys.insert({center_type});
      // there might be masked atoms having different types from the centers
      // so also need to loop over the pairs here
      for (auto neigh : center.pairs()) {
        keys.insert({neigh.get_atom_type()});
      }
    }

    std::vector<std::set<Key_t>> keys_list{};
    std::vector<std::set<Key_t>> keys_list_grad{};
    for (auto center : manager) {
      keys_list.emplace_back(keys);
      if (this->compute_gradients) {
        keys_list_grad.emplace_back(keys);
        for (auto neigh : center.pairs()) {
          std::set<Key_t> neigh_types{};
          Key_t neigh_type{neigh.get_atom_type()};
          neigh_types.insert(neigh_type);
          keys_list_grad.emplace_back(neigh_types);
        }
      }
    }

    expansions_coefficients.resize(keys_list);
    expansions_coefficients.setZero();

    if (this->compute_gradients) {
      expansions_coefficients_gradient.resize(keys_list_grad);
      expansions_coefficients_gradient.setZero();
    } else {
      expansions_coefficients_gradient.resize();
    }
  }

  template <class StructureManager>
  void CalculatorSphericalExpansion::initialize_expansion_with_global_species(
      std::shared_ptr<StructureManager> & manager,
      Property_t<StructureManager> & expansions_coefficients,
      PropertyGradient_t<StructureManager> & expansions_coefficients_gradient) {
    std::vector<std::set<Key_t>> keys_list{};
    std::vector<std::set<Key_t>> keys_list_grad{};

    // check that all species in the structure are present in global_species
    Key_t keys{};
    for (auto center : manager) {
      typename Key_t::value_type center_type{center.get_atom_type()};
      keys.push_back(center_type);
    }
    Key_t missing_keys{};
    for (const auto & key : keys) {
      if (not internal::is_element_in(key, this->global_species)) {
        missing_keys.emplace_back(key);
      }
    }
    if (missing_keys.size() > 0) {
      std::stringstream err_str{};
      err_str << "global_species is missing at least these species: '";
      for (const auto & key : missing_keys) {
        err_str << key << ", ";
      }
      err_str << "'.";
      throw std::runtime_error(err_str.str());
    }

    // build the species list
    for (auto center : manager) {
      keys_list.emplace_back(this->global_species);
      if (this->compute_gradients) {
        keys_list_grad.emplace_back(this->global_species);
        for (auto neigh : center.pairs()) {
          std::set<Key_t> neigh_types{};
          Key_t neigh_type{neigh.get_atom_type()};
          neigh_types.insert(neigh_type);
          keys_list_grad.emplace_back(neigh_types);
        }
      }
    }

    expansions_coefficients.resize(keys_list);
    expansions_coefficients.setZero();

    if (this->compute_gradients) {
      expansions_coefficients_gradient.resize(keys_list_grad);
      expansions_coefficients_gradient.setZero();
    } else {
      expansions_coefficients_gradient.resize();
    }
  }

}  // namespace rascal

namespace nlohmann {
  /**
   * Special specialization of the json serialization for non default
   * constructible type.
   */
  template <>
  struct adl_serializer<rascal::CalculatorSphericalExpansion> {
    static rascal::CalculatorSphericalExpansion from_json(const json & j) {
      return rascal::CalculatorSphericalExpansion{j};
    }

    static void to_json(json & j,
                        const rascal::CalculatorSphericalExpansion & t) {
      j = t.hypers;
    }
  };
}  // namespace nlohmann

#endif  // SRC_RASCAL_REPRESENTATIONS_CALCULATOR_SPHERICAL_EXPANSION_HH_
