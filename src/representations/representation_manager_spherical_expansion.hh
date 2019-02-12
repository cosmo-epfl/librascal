/**
 * file   representation_manager_spherical_expansion.hh
 *
 * @author Max Veit <max.veit@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
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

#ifndef BASIS_REPRESENTATION_MANAGER_SPHERICAL_EXPANSION_H
#define BASIS_REPRESENTATION_MANAGER_SPHERICAL_EXPANSION_H


#include "representations/representation_manager_base.hh"
#include "structure_managers/structure_manager.hh"
#include "structure_managers/property.hh"
#include "rascal_utility.hh"
#include "math/math_utils.hh"
#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace rascal {

  namespace internal {

    enum GaussianSigmaType {
      Constant,
      PerSpecies,
      Radial
    };

    /**
     * Specification to hold the parameter for the atomic smearing function,
     * currently only Gaussians are supported.
     *
     * This is `sigma' in the definition `f(r) = A exp(r / (2 sigma^2))'.
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
    template<GaussianSigmaType SigmaType>
    struct AtomicSmearingSpecification {};

    template<>
    struct AtomicSmearingSpecification<GaussianSigmaType::Constant> {
      explicit AtomicSmearingSpecification(json hypers) {
        this->constant_gaussian_sigma =
            hypers.at("gaussian_sigma_constant").get<double>();
      }
      template<size_t Order, size_t Layer>
      double get_gaussian_sigma(
            ClusterRefKey<Order, Layer> & /* pair */) {
        return this->constant_gaussian_sigma;
      }
      double constant_gaussian_sigma{0.};
    };

    /** Per-species template specialization of the above */

    template<>
    struct AtomicSmearingSpecification<GaussianSigmaType::PerSpecies> {
      explicit AtomicSmearingSpecification(json ) {
      }
      template<size_t Order, size_t Layer>
      double get_gaussian_sigma(ClusterRefKey<Order, Layer> & /* pair */) {
        throw std::logic_error("Requested a sigma type that has not yet "
                               "been implemented");
        return -1;
      }
    };

    /** Radially-dependent template specialization of the above */
    template<>
    struct AtomicSmearingSpecification<GaussianSigmaType::Radial> {
      explicit AtomicSmearingSpecification(json ) {}
      template<size_t Order, size_t Layer>
      double get_gaussian_sigma(ClusterRefKey<Order, Layer> & /* pair */) {
        throw std::logic_error("Requested a sigma type that has not yet "
                               "been implemented");
        return -1;
      }
    };



  }

  /**
   * Handles the expansion of an environment in a spherical and radial basis.
   *
   * The local environment of each atom is represented by Gaussians of a
   * certain width (user-defined; can be constant, species-dependent, or
   * radially dependent).  This density field is expanded in an angular basis
   * of spherical harmonics (à la SOAP) and a radial basis of either Gaussians
   * (again, as in SOAP) or one of the more recent bases currently under
   * development.
   */

  template<class StructureManager>
  class RepresentationManagerSphericalExpansion:
    public RepresentationManagerBase {
   public:
    using hypers_t = RepresentationManagerBase::hypers_t;
    using Property_t = Property<double, 1, 1, Eigen::Dynamic, Eigen::Dynamic>;
    using Manager_t = StructureManager;

    /**
     * Set the hyperparameters of this descriptor from a json object.
     *
     * @param hypers structure (usually parsed from json) containing the
     *               options and hyperparameters
     *
     * @throw logic_error if an invalid option or combination of options is
     *                    specified in the structure
     */
    void set_hyperparameters(const hypers_t& hypers) {
      this->max_radial = hypers.at("max_radial");
      this->max_angular = hypers.at("max_angular");
      if (hypers.find("n_species") != hypers.end()) {
        this->n_species = hypers.at("n_species");
      } else {
        this->n_species = 1; // default: no species distinction
      }
      this->interaction_cutoff = hypers.at("interaction_cutoff");
      this->cutoff_smooth_width = hypers.at("cutoff_smooth_width");
      this->gaussian_sigma_str =
                hypers.at("gaussian_sigma_type").get<std::string>();
      this->radial_ortho_matrix.resize(this->max_radial, this->max_radial);
      this->radial_sigmas.resize(this->max_radial, 1);
      this->radial_norm_factors.resize(this->max_radial, 1);
      this->radial_nl_factors.resize(this->max_radial, this->max_angular + 1);
      this->soap_vectors.resize_to_zero();
      //this->soap_vectors.resize(this->n_species * this->max_radial,
                                //pow(this->max_angular + 1, 2));
      this->is_precomputed = false;

      this->hypers = hypers;
      if (this->gaussian_sigma_str.compare("Constant") == 0) {
        this->gaussian_sigma_type = internal::GaussianSigmaType::Constant;
      } else {
        throw std::logic_error("Requested Gaussian sigma type \'"
                               + gaussian_sigma_str
                               + "\' has not been implemented.  Must be one of"
                               + ": \'Constant\'.");
      }
    }

    /**
     * Construct a new RepresentationManager using a hyperparameters container
     *
     * @param hypers container (usually parsed from json) for the options and
     *               hyperparameters
     *
     * @throw logic_error if an invalid option or combination of options is
     *                    specified in the container
     */
    RepresentationManagerSphericalExpansion(Manager_t &sm,
                                            const hypers_t& hyper)
        :structure_manager{sm}, soap_vectors{sm}
    {
      this->set_hyperparameters(hyper);
    }

    //! Copy constructor
    RepresentationManagerSphericalExpansion(
      const RepresentationManagerSphericalExpansion &other) = delete;

    //! Move constructor
    RepresentationManagerSphericalExpansion(
      RepresentationManagerSphericalExpansion &&other) = default;

    //! Destructor
    virtual ~RepresentationManagerSphericalExpansion()  = default;

    //! Copy assignment operator
    RepresentationManagerSphericalExpansion& operator=(
      const RepresentationManagerSphericalExpansion &other) = delete;

    //! Move assignment operator
    RepresentationManagerSphericalExpansion& operator=(
      RepresentationManagerSphericalExpansion && other) = default;

    //! compute representation
    void compute();

    template<internal::GaussianSigmaType SigmaType>
    void compute_by_gaussian_sigma();

    //! Precompute radial Gaussian widths (NOTE specific to basis!)
    void precompute_radial_sigmas();

    //! Precompute radial orthogonalization matrix (NOTE specific to basis!)
    void precompute_radial_overlap();

    //! Precompute everything that doesn't depend on the structure
    void precompute();

    //! getter for the representation
    template <size_t Order, size_t Layer>
    Eigen::Map<Eigen::MatrixXd> get_soap_vector(
        const ClusterRefKey<Order, Layer>& center) {
      return this->soap_vectors[center];
    }

    //TODO(max-veit) overload operator<< instead? But we need the center...
    template <size_t Order, size_t Layer>
    void print_soap_vector(
        const ClusterRefKey<Order, Layer>& center, std::ostream& stream) {
      stream << "Soap vector size " << this->get_feature_size() << std::endl;
      for (size_t radial_n{0}; radial_n < this->max_radial; ++radial_n) {
        stream << "n = " << radial_n << std::endl;
        stream << this->soap_vectors[center].row(radial_n) << std::endl;
      }
    }

    std::vector<precision_t>& get_representation_raw_data() {
      return this->soap_vectors.get_raw_data();
    }

    size_t get_feature_size() {
      // (should be) equivalent:
      //return this->n_species * this->max_radial
      //    * pow(this->max_angular + 1, 2);
      return this->soap_vectors.get_nb_comp();
    }

    size_t get_center_size() {
      return this->soap_vectors.get_nb_item();
    }

    /**
     * Return whether the radial sigmas and overlap matrix have been precomputed
     *
     * This only needs to be done once on initialization, but it is not done in
     * the constructor to avoid spending time and possibly throwing exceptions
     * there.  Instead, by default, it is done once on the first call of the
     * compute() method and the precomputed results are saved for subsequent
     * calls.
     */
    bool get_is_precomputed() {
      return this->is_precomputed;
    }

   protected:
   private:
    double interaction_cutoff{};
    double cutoff_smooth_width{};
    size_t max_radial{};
    size_t max_angular{};
    size_t n_species{};
    std::string gaussian_sigma_str{};
    // TODO(max-veit) these are specific to the radial Gaussian basis
    Eigen::VectorXd radial_sigmas{};
    Eigen::VectorXd radial_norm_factors{};
    Eigen::VectorXd radial_sigma_factors{};
    Eigen::MatrixXd radial_nl_factors{};
    Eigen::MatrixXd radial_ortho_matrix{};
    bool is_precomputed{false};

    Manager_t& structure_manager;
    Property_t soap_vectors;
    internal::GaussianSigmaType gaussian_sigma_type{};

    hypers_t hypers{};
  };


  /** Compute common prefactors for the radial Gaussian basis functions */
  template<class Mngr>
  void RepresentationManagerSphericalExpansion<Mngr>::
      precompute_radial_sigmas() {
    using std::pow;

    for (size_t radial_n{0}; radial_n < this->max_radial; ++radial_n) {
      this->radial_sigmas[radial_n] = std::max(
            std::sqrt(static_cast<double>(radial_n)), 1.0)
          * this->interaction_cutoff / static_cast<double>(this->max_radial);
    }

    // Precompute common prefactors
    for (size_t radial_n{0}; radial_n < this->max_radial; ++radial_n) {
      this->radial_norm_factors(radial_n) = std::sqrt(
          2.0 / std::tgamma(1.5 + radial_n)
          * pow(this->radial_sigmas[radial_n], 3.0 + 2.0*radial_n));
      for (size_t angular_l{0}; angular_l < this->max_angular + 1;
           ++angular_l) {
        this->radial_nl_factors(radial_n, angular_l) =
            std::exp2(-0.5 * (1.0 + angular_l - radial_n))
            * std::tgamma(0.5 * (3.0 + angular_l + radial_n))
            / std::tgamma(1.5 + angular_l);
      }
    }
  }

  /**
   * Compute the radial overlap matrix for later orthogonalization.
   *
   * @throw runtime_error if the overlap matrix cannot be diagonalized
   */
  template<class Mngr>
  void RepresentationManagerSphericalExpansion<Mngr>::
      precompute_radial_overlap() {
    using std::pow;
    using std::sqrt;
    using std::tgamma;

    //TODO(max-veit) see if we can replace the gammas with their natural logs,
    //since it'll overflow for relatively small n (n1 + n2 >~ 300)
    Eigen::MatrixXd overlap(this->max_radial, this->max_radial);
    for (size_t radial_n1{0}; radial_n1 < this->max_radial; radial_n1++) {
      for (size_t radial_n2{0}; radial_n2 < this->max_radial; radial_n2++) {
        overlap(radial_n1, radial_n2) =
            pow(0.5/pow(this->radial_sigmas[radial_n1], 2)
                   + 0.5/pow(this->radial_sigmas[radial_n2], 2),
                -0.5*(3.0 + radial_n1 + radial_n2))
            / (pow(this->radial_sigmas[radial_n1], radial_n1) +
               pow(this->radial_sigmas[radial_n2], radial_n2))
            * tgamma(0.5 * (3.0 + radial_n1 + radial_n2))
            / pow(this->radial_sigmas[radial_n1] *
                       this->radial_sigmas[radial_n2], 1.5)
            * sqrt(tgamma(1.5 + radial_n1)
                 * tgamma(1.5 + radial_n2));
      }
    }

    // Compute the inverse square root of the overlap matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(overlap);
    if (eigensolver.info() != Eigen::Success) {
      throw std::runtime_error("Warning: Could not diagonalize "
                               "radial overlap matrix.");
    }
    Eigen::MatrixXd eigenvalues = eigensolver.eigenvalues();
    Eigen::ArrayXd eigs_invsqrt = eigenvalues.array().sqrt().inverse();
    Eigen::MatrixXd unitary = eigensolver.eigenvectors();
    this->radial_ortho_matrix = unitary.adjoint() *
                                eigs_invsqrt.matrix().asDiagonal() * unitary;
    this->is_precomputed = true;
  }

  /**
   * Precompute everything that doesn't depend on the atomic structure
   * (only on the hyperparameters)
   *
   * Calls the methods precompute_radial_sigmas() and
   * precompute_radial_overlap(); if either of those methods throws exceptions,
   * they will be passed on from here.
   *
   * @throw runtime_error if the overlap matrix cannot be diagonalized
   */
  template<class Mngr>
  void RepresentationManagerSphericalExpansion<Mngr>::precompute() {
    this->precompute_radial_sigmas();
    this->precompute_radial_overlap();
    // Only if none of the above failed (threw exceptions)
    this->is_precomputed = true;
  }


  template<class Mngr>
  void RepresentationManagerSphericalExpansion<Mngr>::compute() {
    using internal::GaussianSigmaType;
    switch (this->gaussian_sigma_type) {
    case GaussianSigmaType::Constant :
      this->compute_by_gaussian_sigma<GaussianSigmaType::Constant>();
      break;
    case GaussianSigmaType::PerSpecies :
      this->compute_by_gaussian_sigma<GaussianSigmaType::PerSpecies>();
      break;
    case GaussianSigmaType::Radial :
      this->compute_by_gaussian_sigma<GaussianSigmaType::Radial>();
      break;
    default:
      // Will never reach here (it's an enum...)
      break;
    }
  }

  /**
   * Compute the spherical expansion
   *
   */
  template<class Mngr>
  template<internal::GaussianSigmaType SigmaType>
  void RepresentationManagerSphericalExpansion<Mngr>::
      compute_by_gaussian_sigma() {
    using math::PI;
    using std::pow;

    internal::AtomicSmearingSpecification<SigmaType> gaussian_spec{
        this->hypers};

    if (not this->is_precomputed) {
      this->precompute();
    }

    this->soap_vectors.set_nb_row(this->n_species * this->max_radial);
    this->soap_vectors.set_nb_col(pow(this->max_angular + 1, 2));

    for (auto center : this->structure_manager) {
      Eigen::MatrixXd soap_vector = Eigen::MatrixXd::Zero(
          this->n_species * this->max_radial,
          pow(this->max_angular + 1, 2));
      Eigen::MatrixXd radial_integral(this->max_radial, this->max_angular + 1);

      // Start the accumulator with the central atom
      // All terms where l =/= 0 cancel
      double sigma2 = pow(gaussian_spec.get_gaussian_sigma(center), 2);
      // TODO(max-veit) this is specific to the Gaussian radial basis
      // (along with the matching computation below)
      // And ditto on the gamma functions (potential overflow)
      for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
        // TODO(max-veit) remember to update when we do multiple n_species!
        radial_integral(radial_n, 0) = radial_norm_factors(radial_n)
            * radial_nl_factors(radial_n, 0)
            * pow(1.0/sigma2 + pow(this->radial_sigmas[radial_n], -2),
                       -0.5 * (3.0 + radial_n));
      }
      soap_vector.col(0) = this->radial_ortho_matrix *
                           radial_integral.col(0) / sqrt(4.0 * PI);

      for (auto neigh : center) {
        auto dist{this->structure_manager.get_distance(neigh)};
        auto direction{this->structure_manager.get_direction_vector(neigh)};
        double exp_factor = std::exp(-0.5 * pow(dist, 2) / sigma2);
        sigma2 = pow(gaussian_spec.get_gaussian_sigma(neigh), 2);

        // Note: the copy _should_ be optimized out (RVO)
        Eigen::MatrixXd harmonics = math::compute_spherical_harmonics(
            direction, this->max_angular);

        // Precompute radial factors that also depend on the Gaussian sigma
        Eigen::VectorXd radial_sigma_factors(this->max_radial);
        for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
          radial_sigma_factors(radial_n) = (pow(sigma2, 2) +
                                 sigma2*pow(this->radial_sigmas(radial_n), 2))
                                / pow(this->radial_sigmas(radial_n), 2);
        }

        for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
          // TODO(max-veit) this is all specific to the Gaussian radial basis
          // (doing just the angular integration would instead spit out
          // spherical Bessel functions below)
          //TODO(max-veit) how does this work with SpeciesFilter?
          for (size_t angular_l{0}; angular_l < this->max_angular + 1;
               angular_l++) {
            radial_integral(radial_n, angular_l) =
                exp_factor * radial_nl_factors(radial_n, angular_l)
                * pow(1.0/sigma2 + 1.0/pow(this->radial_sigmas(radial_n), 2),
                      -0.5 * (3.0 + angular_l + radial_n))
                * pow(dist / sigma2, angular_l)
                * math::hyp1f1(0.5 * (3.0 + angular_l * radial_n),
                               1.5 + angular_l,
                               0.5 * pow(dist, 2)
                                   / radial_sigma_factors(radial_n));
          }
        }
        radial_integral = this->radial_ortho_matrix * radial_integral;

        for (size_t radial_n{0}; radial_n < this->max_radial; radial_n++) {
          size_t lm_collective_idx{0};
          for (size_t angular_l{0}; angular_l < this->max_angular + 1;
               angular_l++) {
            for (size_t m_array_idx{0}; m_array_idx < 2*angular_l + 1;
                m_array_idx++) {
              soap_vector(radial_n, lm_collective_idx) +=
                  radial_integral(radial_n, angular_l)
                  * harmonics(angular_l, m_array_idx);
              ++lm_collective_idx;
            }
          }
        }
      } // for (neigh : center)
      this->soap_vectors.push_back(soap_vector);
    } // for (center : structure_manager)
  } // compute()



} // namespace rascal

#endif /* BASIS_REPRESENTATION_MANAGER_SPHERICAL_EXPANSION_H */

