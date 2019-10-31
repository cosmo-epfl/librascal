/**
 * @file   test_calculator.hh
 *
 * @author Musil Felix <musil.felix@epfl.ch>
 * @author Max Veit <max.veit@epfl.ch>
 *
 * @date   14 September 2018
 *
 * @brief  test representation managers
 *
 * Copyright  2018 Musil Felix, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef TESTS_TEST_CALCULATOR_HH_
#define TESTS_TEST_CALCULATOR_HH_

#include "atomic_structure.hh"
#include "json_io.hh"
#include "rascal_utility.hh"
#include "representations/calculator_base.hh"
#include "representations/calculator_sorted_coulomb.hh"
#include "representations/calculator_spherical_covariants.hh"
#include "representations/calculator_spherical_expansion.hh"
#include "representations/calculator_spherical_invariants.hh"
#include "structure_managers/structure_manager_collection.hh"
#include "structure_managers/cluster_ref_key.hh"
#include "test_adaptor.hh"
#include "test_math.hh"
#include "test_structure.hh"

#include <memory>
#include <tuple>

namespace rascal {

  struct TestData {
    using ManagerTypeHolder_t =
        StructureManagerTypeHolder<StructureManagerCenters,
                                   AdaptorNeighbourList,
                                   AdaptorCenterContribution, AdaptorStrict>;
    TestData() = default;

    void get_ref(const std::string & ref_filename) {
      this->ref_data =
          json::from_ubjson(internal::read_binary_file(ref_filename));
      auto filenames =
          this->ref_data.at("filenames").get<std::vector<std::string>>();
      auto cutoffs = this->ref_data.at("cutoffs").get<std::vector<double>>();

      for (auto && filename : filenames) {
        for (auto && cutoff : cutoffs) {
          // std::cout << filename << " " << cutoff << std::endl;
          json parameters;
          json structure{{"filename", filename}};
          json adaptors;
          json ad1{
              {"name", "AdaptorNeighbourList"},
              {"initialization_arguments",
               {{"cutoff", cutoff},
                {"consider_ghost_neighbours", consider_ghost_neighbours}}}};
          json ad1b{{"name", "AdaptorCenterContribution"},
                    {"initialization_arguments", {}}};
          json ad2{{"name", "AdaptorStrict"},
                   {"initialization_arguments", {{"cutoff", cutoff}}}};
          adaptors.emplace_back(ad1);
          adaptors.emplace_back(ad1b);
          adaptors.emplace_back(ad2);

          parameters["structure"] = structure;
          parameters["adaptors"] = adaptors;

          this->factory_args.emplace_back(parameters);
        }
      }
    }

    ~TestData() = default;

    const bool consider_ghost_neighbours{false};
    json ref_data{};
    json factory_args{};
  };

  template <typename MultipleStructureFixture>
  struct MultipleStructureSphericalInvariants : MultipleStructureFixture {
    using Parent = MultipleStructureFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalInvariants;

    MultipleStructureSphericalInvariants() : Parent{} {
      for (auto & ri_hyp : this->radial_contribution_hypers) {
        for (auto & fc_hyp : this->fc_hypers) {
          for (auto & sig_hyp : this->density_hypers) {
            for (auto & rep_hyp : this->rep_hypers) {
              rep_hyp["cutoff_function"] = fc_hyp;
              rep_hyp["gaussian_density"] = sig_hyp;
              rep_hyp["radial_contribution"] = ri_hyp;
              this->representation_hypers.push_back(rep_hyp);
            }
          }
        }
      }
    };

    ~MultipleStructureSphericalInvariants() = default;

    std::vector<json> representation_hypers{};

    std::vector<json> fc_hypers{
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 3.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}}}};

    std::vector<json> density_hypers{
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.2}, {"unit", "AA"}}}}};
    std::vector<json> radial_contribution_hypers{{{"type", "GTO"}}};

    std::vector<json> rep_hypers{{{"max_radial", 6},
                                  {"max_angular", 0},
                                  {"soap_type", "RadialSpectrum"},
                                  {"normalize", true}},
                                 {{"max_radial", 6},
                                  {"max_angular", 0},
                                  {"soap_type", "RadialSpectrum"},
                                  {"normalize", true}},
                                 {{"max_radial", 3},
                                  {"max_angular", 3},
                                  {"soap_type", "PowerSpectrum"},
                                  {"normalize", true}},
                                 {{"max_radial", 6},
                                  {"max_angular", 4},
                                  {"soap_type", "PowerSpectrum"},
                                  {"normalize", true}},
                                 {{"max_radial", 3},
                                  {"max_angular", 1},
                                  {"soap_type", "BiSpectrum"},
                                  {"inversion_symmetry", true},
                                  {"normalize", true}},
                                 {{"max_radial", 3},
                                  {"max_angular", 1},
                                  {"soap_type", "BiSpectrum"},
                                  {"inversion_symmetry", false},
                                  {"normalize", true}}};
  };

  template <typename MultipleStructureFixture>
  struct MultipleStructureSphericalCovariants : MultipleStructureFixture {
    using Parent = MultipleStructureFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalCovariants;

    MultipleStructureSphericalCovariants() : Parent{} {
      for (auto & ri_hyp : this->radial_contribution_hypers) {
        for (auto & fc_hyp : this->fc_hypers) {
          for (auto & sig_hyp : this->density_hypers) {
            for (auto & rep_hyp : this->rep_hypers) {
              rep_hyp["cutoff_function"] = fc_hyp;
              rep_hyp["gaussian_density"] = sig_hyp;
              rep_hyp["radial_contribution"] = ri_hyp;
              this->representation_hypers.push_back(rep_hyp);
            }
          }
        }
      }
    };

    ~MultipleStructureSphericalCovariants() = default;

    std::vector<json> representation_hypers{};

    std::vector<json> fc_hypers{
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 2.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 1.0}, {"unit", "AA"}}}}};

    std::vector<json> density_hypers{
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.2}, {"unit", "AA"}}}},
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.4}, {"unit", "AA"}}}}};
    std::vector<json> radial_contribution_hypers{{{"type", "GTO"}}};
    std::vector<json> rep_hypers{{{"max_radial", 1},
                                  {"max_angular", 2},
                                  {"soap_type", "LambdaSpectrum"},
                                  {"lam", 2},
                                  {"inversion_symmetry", true},
                                  {"normalize", true}},
                                 {{"max_radial", 2},
                                  {"max_angular", 2},
                                  {"soap_type", "LambdaSpectrum"},
                                  {"lam", 2},
                                  {"inversion_symmetry", false},
                                  {"normalize", true}}};
  };

  struct SphericalInvariantsTestData : TestData {
    using Parent = TestData;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalInvariants;
    SphericalInvariantsTestData() : Parent{} {
      this->get_ref(this->ref_filename);
    }
    ~SphericalInvariantsTestData() = default;
    bool verbose{false};
    std::string ref_filename{
        "reference_data/spherical_invariants_reference.ubjson"};
  };

  struct SphericalCovariantsTestData : TestData {
    using Parent = TestData;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalCovariants;
    SphericalCovariantsTestData() : Parent{} {
      this->get_ref(this->ref_filename);
    }
    ~SphericalCovariantsTestData() = default;
    bool verbose{false};
    std::string ref_filename{
        "reference_data/spherical_covariants_reference.ubjson"};
  };

  template <class MultipleStructureFixture>
  struct MultipleStructureSphericalExpansion : MultipleStructureFixture {
    using Parent = MultipleStructureFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalExpansion;

    MultipleStructureSphericalExpansion() : Parent{} {
      for (auto & ri_hyp : this->radial_contribution_hypers) {
        for (auto & fc_hyp : this->fc_hypers) {
          for (auto & sig_hyp : this->density_hypers) {
            for (auto & rep_hyp : this->rep_hypers) {
              rep_hyp["cutoff_function"] = fc_hyp;
              rep_hyp["gaussian_density"] = sig_hyp;
              rep_hyp["radial_contribution"] = ri_hyp;
              this->representation_hypers.push_back(rep_hyp);
            }
          }
        }
      }
    };
    ~MultipleStructureSphericalExpansion() = default;

    std::vector<json> representation_hypers{};

    std::vector<json> fc_hypers{
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 3.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}}},
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 2.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 1.0}, {"unit", "AA"}}}},
        {{"type", "RadialScaling"},
         {"cutoff", {{"value", 4.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}},
         {"rate", {{"value", .0}, {"unit", "AA"}}},
         {"exponent", {{"value", 4}, {"unit", ""}}},
         {"scale", {{"value", 2.5}, {"unit", "AA"}}}},
        {{"type", "RadialScaling"},
         {"cutoff", {{"value", 4.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}},
         {"rate", {{"value", 1.}, {"unit", "AA"}}},
         {"exponent", {{"value", 3}, {"unit", ""}}},
         {"scale", {{"value", 2.}, {"unit", "AA"}}}}};

    std::vector<json> radial_contribution_hypers{{{"type", "GTO"}},
                                                 {{"type", "DVR"}}};

    std::vector<json> density_hypers{
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.5}, {"unit", "AA"}}}}};

    std::vector<json> rep_hypers{{{"max_radial", 6}, {"max_angular", 4}}};
  };

  /**
   * Simplified version of MultipleStructureManagerNLStrictFixture
   *  that uses only one structure, cutoff, and adaptor set
   *
   *  Useful if we just need a StructureManager to test relatively isolated
   *  functionality on a single structure, but using the rest of the testing
   *  machinery
   */
  struct SimpleStructureManagerNLCCStrictFixture {
    using ManagerTypeHolder_t =
        StructureManagerTypeHolder<StructureManagerCenters,
                                   AdaptorNeighbourList,
                                   AdaptorCenterContribution, AdaptorStrict>;

    SimpleStructureManagerNLCCStrictFixture() {
      json parameters;
      json structure{{"filename", filename}};
      json adaptors;
      json ad1{{"name", "AdaptorNeighbourList"},
               {"initialization_arguments",
                {{"cutoff", cutoff},
                 {"skin", cutoff_skin},
                 {"consider_ghost_neighbours", false}}}};
      json ad1b{{"name", "AdaptorCenterContribution"},
                {"initialization_arguments", {}}};
      json ad2{{"name", "AdaptorStrict"},
               {"initialization_arguments", {{"cutoff", cutoff}}}};
      adaptors.emplace_back(ad1);
      adaptors.push_back(ad1b);
      adaptors.emplace_back(ad2);

      parameters["structure"] = structure;
      parameters["adaptors"] = adaptors;

      this->factory_args.emplace_back(parameters);
    }

    ~SimpleStructureManagerNLCCStrictFixture() = default;

    const std::string filename{
        "reference_data/CaCrP2O7_mvc-11955_symmetrized.json"};
    const double cutoff{3.};
    const double cutoff_skin{0.};

    json factory_args{};
  };

  struct MultipleHypersSphericalExpansion
      : SimpleStructureManagerNLCCStrictFixture {
    using Parent = SimpleStructureManagerNLCCStrictFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalExpansion;

    MultipleHypersSphericalExpansion() : Parent{} {
      for (auto & ri_hyp : this->radial_contribution_hypers) {
        for (auto & fc_hyp : this->fc_hypers) {
          for (auto & sig_hyp : this->density_hypers) {
            for (auto & rep_hyp : this->rep_hypers) {
              rep_hyp["cutoff_function"] = fc_hyp;
              rep_hyp["gaussian_density"] = sig_hyp;
              rep_hyp["radial_contribution"] = ri_hyp;
              this->representation_hypers.push_back(rep_hyp);
            }
          }
        }
      }
    };

    ~MultipleHypersSphericalExpansion() = default;

    std::vector<json> representation_hypers{};
    std::vector<json> fc_hypers{
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 3.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 0.5}, {"unit", "AA"}}}},
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 2.0}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 1.0}, {"unit", "AA"}}}}};

    std::vector<json> density_hypers{
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.2}, {"unit", "AA"}}}},
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.4}, {"unit", "AA"}}}}};
    std::vector<json> radial_contribution_hypers{{{"type", "GTO"}}};
    std::vector<json> rep_hypers{
        {{"max_radial", 4}, {"max_angular", 2}, {"compute_gradients", true}},
        {{"max_radial", 6}, {"max_angular", 4}, {"compute_gradients", true}}};
  };

  /** Contains some simple periodic structures for testing complicated things
   *  like gradients
   */
  struct SimplePeriodicNLCCStrictFixture {
    using ManagerTypeHolder_t =
        StructureManagerTypeHolder<StructureManagerCenters,
                                   AdaptorNeighbourList,
                                   AdaptorCenterContribution, AdaptorStrict>;
    using Structure_t = AtomicStructure<3>;

    SimplePeriodicNLCCStrictFixture() {
      for (auto && filename : filenames) {
        json parameters;
        json structure{{"filename", filename}};
        json adaptors;
        json ad1{{"name", "AdaptorNeighbourList"},
                 {"initialization_arguments",
                  {{"cutoff", cutoff},
                   {"skin", cutoff_skin},
                   {"consider_ghost_neighbours", true}}}};
        json ad1b{{"name", "AdaptorCenterContribution"},
                  {"initialization_arguments", {}}};
        json ad2{{"name", "AdaptorStrict"},
                 {"initialization_arguments", {{"cutoff", cutoff}}}};
        adaptors.emplace_back(ad1);
        adaptors.push_back(ad1b);
        adaptors.emplace_back(ad2);

        parameters["structure"] = structure;
        parameters["adaptors"] = adaptors;

        this->factory_args.emplace_back(parameters);
      }
    }

    ~SimplePeriodicNLCCStrictFixture() = default;

    const std::vector<std::string> filenames{
        "reference_data/diamond_2atom_distorted.json",
        "reference_data/diamond_cubic_distorted.json",
        "reference_data/SiCGe_wurtzite_like.json",
        "reference_data/SiC_moissanite_supercell.json",
        "reference_data/small_molecule.json",
        "reference_data/methane.json"
    };
    // Simpler structures for debugging:
        //"reference_data/diamond_2atom.json",
        //"reference_data/SiC_moissanite.json",
    const double cutoff{2.5};
    const double cutoff_skin{0.};

    json factory_args{};
    std::vector<Structure_t> structures{};
  };

  struct SingleHypersSphericalExpansion : SimplePeriodicNLCCStrictFixture {
    using Parent = SimplePeriodicNLCCStrictFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalExpansion;

    SingleHypersSphericalExpansion() : Parent{} {
      for (auto & ri_hyp : this->radial_contribution_hypers) {
        for (auto & fc_hyp : this->fc_hypers) {
          for (auto & sig_hyp : this->density_hypers) {
            for (auto & rep_hyp : this->rep_hypers) {
              rep_hyp["cutoff_function"] = fc_hyp;
              rep_hyp["gaussian_density"] = sig_hyp;
              rep_hyp["radial_contribution"] = ri_hyp;
              this->representation_hypers.push_back(rep_hyp);
            }
          }
        }
      }
    };

    ~SingleHypersSphericalExpansion() = default;

    std::vector<json> representation_hypers{};
    std::vector<json> fc_hypers{
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 2.5}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 1.0}, {"unit", "AA"}}}}};

    std::vector<json> density_hypers{
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.4}, {"unit", "AA"}}}}};
    std::vector<json> radial_contribution_hypers{{{"type", "GTO"}}};
    std::vector<json> rep_hypers{
        {{"max_radial", 2}, {"max_angular", 2}, {"compute_gradients", true}},
        {{"max_radial", 3}, {"max_angular", 0}, {"compute_gradients", true}}};
  };

  struct SingleHypersSphericalInvariants : SimplePeriodicNLCCStrictFixture {
    using Parent = SimplePeriodicNLCCStrictFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalInvariants;

    SingleHypersSphericalInvariants() : Parent{} {
      for (auto & ri_hyp : this->radial_contribution_hypers) {
        for (auto & fc_hyp : this->fc_hypers) {
          for (auto & sig_hyp : this->density_hypers) {
            for (auto & rep_hyp : this->rep_hypers) {
              rep_hyp["cutoff_function"] = fc_hyp;
              rep_hyp["gaussian_density"] = sig_hyp;
              rep_hyp["radial_contribution"] = ri_hyp;
              this->representation_hypers.push_back(rep_hyp);
            }
          }
        }
      }
    };

    ~SingleHypersSphericalInvariants() = default;

    std::vector<json> representation_hypers{};
    std::vector<json> fc_hypers{
        {{"type", "ShiftedCosine"},
         {"cutoff", {{"value", 2.5}, {"unit", "AA"}}},
         {"smooth_width", {{"value", 1.0}, {"unit", "AA"}}}}};

    std::vector<json> density_hypers{
        {{"type", "Constant"},
         {"gaussian_sigma", {{"value", 0.4}, {"unit", "AA"}}}}};
    std::vector<json> radial_contribution_hypers{{{"type", "GTO"}}};
    std::vector<json> rep_hypers{{{"max_radial", 2},
                                  {"max_angular", 2},
                                  {"normalize", true},
                                  {"soap_type", "PowerSpectrum"},
                                  {"compute_gradients", true}},
                                 {{"max_radial", 3},
                                  {"max_angular", 0},
                                  {"normalize", true},
                                  {"soap_type", "RadialSpectrum"},
                                  {"compute_gradients", true}}};
  };

  struct SphericalExpansionTestData : TestData {
    using Parent = TestData;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSphericalExpansion;

    SphericalExpansionTestData() : Parent{} {
      this->get_ref(this->ref_filename);
    }
    ~SphericalExpansionTestData() = default;
    bool verbose{false};
    std::string ref_filename{
        "reference_data/spherical_expansion_reference.ubjson"};
  };

  /**
   * Calculator specialized to testing the derivative of the RadialIntegral
   * in the definition of the SphericalExpansion representation.
   */
  template <class RadialIntegral, class ClusterRef>
  struct SphericalExpansionRadialDerivative {
    SphericalExpansionRadialDerivative(std::shared_ptr<RadialIntegral> ri,
                                       ClusterRef & pair_in)
        : radial_integral{ri}, pair{pair_in}, max_radial{ri->max_radial},
          max_angular{ri->max_angular} {}

    ~SphericalExpansionRadialDerivative() = default;

    Eigen::Array<double, 1, Eigen::Dynamic>
    f(const Eigen::Matrix<double, 1, 1> & input_v) {
      Eigen::ArrayXXd result(this->max_radial, this->max_angular + 1);
      result = this->radial_integral->template compute_neighbour_contribution<
          internal::AtomicSmearingType::Constant>(input_v(0), this->pair);
      // result.matrix().transpose() *=
      //     this->radial_integral->radial_norm_factors.asDiagonal();
      // result.matrix().transpose() *=
      //     this->radial_integral->radial_ortho_matrix;
      Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> result_flat(
          result.data(), 1, result.size());
      return result_flat;
    }

    Eigen::Array<double, 1, Eigen::Dynamic>
    grad_f(const Eigen::Matrix<double, 1, 1> & input_v) {
      Eigen::ArrayXXd result(this->max_radial, this->max_angular + 1);
      result = this->radial_integral->template compute_neighbour_derivative<
          internal::AtomicSmearingType::Constant>(input_v(0), this->pair);
      Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> result_flat(
          result.data(), 1, result.size());
      return result_flat;
    }

    std::shared_ptr<RadialIntegral> radial_integral;
    ClusterRef & pair;
    size_t max_radial{6};
    size_t max_angular{4};
  };

  /**
   * Gradient provider specialized to testing the gradient of a Calculator
   *
   * The gradient is tested center-by-center, by iterating over each center and
   * doing finite displacements on its position.  This iteration should normally
   * be done by the RepresentationCalculatorGradientFixture class.
   *
   * In the case of periodic structures, the gradient is accumulated only onto
   * _real_ atoms, but the motion of all _images_ of the "moving" atom (the one
   * with respect to which the gradient is being taken) is taken into account.
   *
   * Initialize with a Calculator, a StructureManager, and an
   * AtomicStructure representing the original structure (before modifying with
   * finite-difference displacements).  The gradient of the representation with
   * respect to the center position can then be tested, as usual, with
   * test_gradients() (defined in test_math.hh).
   */
  template <typename RepCalculator, class StructureManager>
  class RepresentationCalculatorGradientProvider {
   public:
    static constexpr size_t Dim = 3;
    static constexpr size_t n_arguments = Dim;
    using Structure_t = AtomicStructure<Dim>;
    using Key_t = typename RepCalculator::Key_t;

    using PairRef_t =
        typename RepCalculator::template ClusterRef_t<StructureManager, 2>;

    using PairRefKey_t = typename PairRef_t::ThisParentClass;

    // type of the data structure holding the representation and its gradients
    using Prop_t =
        typename RepCalculator::template Property_t<StructureManager>;
    using PropGrad_t =
        typename RepCalculator::template PropertyGradient_t<StructureManager>;

    template <typename T, class V>
    friend class RepresentationCalculatorGradientFixture;

    RepresentationCalculatorGradientProvider(
        RepCalculator & representation,
        std::shared_ptr<StructureManager> structure_manager,
        Structure_t atomic_structure)
        : representation{representation}, structure_manager{structure_manager},
          atomic_structure{atomic_structure}, center_it{
                                                  structure_manager->begin()} {}

    ~RepresentationCalculatorGradientProvider() = default;

    Eigen::Array<double, 1, Eigen::Dynamic>
    f(const Eigen::Ref<const Eigen::Vector3d> & center_position) {
      auto center = *center_it;
      Structure_t modified_structure{this->atomic_structure};
      modified_structure.positions.col(center.get_index()) = center_position;
      modified_structure.wrap();
      this->structure_manager->update(modified_structure);
      this->representation.compute(this->structure_manager);

      auto && data_sparse{structure_manager->template get_property_ref<Prop_t>(
          representation.get_name())};
      auto && gradients_sparse{
          structure_manager->template get_property_ref<PropGrad_t>(
              representation.get_gradient_name())};
      auto ii_pair = center.get_atom_ii();
      auto & data_center{data_sparse[ii_pair]};
      auto keys_center = gradients_sparse.get_keys(ii_pair);
      Key_t center_key{center.get_atom_type()};
      size_t n_entries_per_key{static_cast<size_t>(data_sparse.get_nb_comp())};
      size_t n_entries_center{n_entries_per_key * keys_center.size()};
      size_t n_entries_neighbours{0};
      // Count all the keys in the sparse gradient structure where the gradient
      // is nonzero (i.e. where the key has an entry in the structure)
      for (auto neigh : center) {
        // TODO(max,felix) there should be a way to write "if neigh is ghost"
        if (this->structure_manager->is_ghost_atom()) {
          // Don't compute gradient contributions onto ghost atoms
          continue;
        }
        auto swapped_ref{std::move(swap_pair_ref(neigh).front())};
        n_entries_neighbours +=
            (gradients_sparse[swapped_ref].get_keys().size() *
             n_entries_per_key);
      }
      // Packed array containing: The center coefficients (all species) and
      // the neighbour coefficients (only same species as center)
      Eigen::ArrayXd data_pairs(n_entries_center + n_entries_neighbours);

      size_t result_idx{0};
      for (auto & key : keys_center) {
        Eigen::Map<Eigen::RowVectorXd> data_flat(data_center[key].data(),
                                                 n_entries_per_key);
        data_pairs.segment(result_idx, n_entries_per_key) = data_flat;
        result_idx += n_entries_per_key;
      }
      for (auto neigh : center) {
        if (this->structure_manager->is_ghost_atom()) {
          // Don't compute gradient contributions onto ghost atoms
          continue;
        }
        auto & data_neigh{data_sparse[neigh]};
        // The neighbour gradient (i =/= j) only contributes to certain species
        // channels (keys), in the case of SOAP and SphExpn those keys
        // containing the species of the center (the atom wrt the derivative is
        // being taken)
        // The nonzero gradient keys are already indicated in the sparse
        // gradient structure
        auto swapped_ref{std::move(swap_pair_ref(neigh).front())};
        auto keys_neigh{gradients_sparse[swapped_ref].get_keys()};
        for (auto & key : keys_neigh) {
          Eigen::Map<Eigen::ArrayXd> data_flat(data_neigh[key].data(),
                                               n_entries_per_key);
          data_pairs.segment(result_idx, n_entries_per_key) = data_flat;
          result_idx += n_entries_per_key;
        }
      }

      // Reset the atomic structure for the next iteration
      this->structure_manager->update(this->atomic_structure);
      return data_pairs.transpose();
    }

    Eigen::Array<double, Dim, Eigen::Dynamic>
    grad_f(const Eigen::Ref<const Eigen::Vector3d> & /*center_position*/) {
      using Matrix3Xd_RowMaj_t =
          Eigen::Matrix<double, Dim, Eigen::Dynamic, Eigen::RowMajor>;
      // Assume f() was already called and updated the position
      // center_it->position() = center_position;
      // representation.compute();
      auto center = *center_it;

      auto && data_sparse{structure_manager->template get_property_ref<Prop_t>(
          representation.get_name())};
      auto && gradients_sparse{
          structure_manager->template get_property_ref<PropGrad_t>(
              representation.get_gradient_name())};
      auto ii_pair = center.get_atom_ii();
      auto & gradients_center{gradients_sparse[ii_pair]};
      auto keys_center = gradients_center.get_keys();
      size_t n_entries_per_key{static_cast<size_t>(data_sparse.get_nb_comp())};
      size_t n_entries_center{n_entries_per_key * keys_center.size()};
      size_t n_entries_neighbours{0};
      for (auto neigh : center) {
        if (this->structure_manager->is_ghost_atom()) {
          // Don't compute gradient contributions onto ghost atoms
          continue;
        }
        auto swapped_ref{std::move(swap_pair_ref(neigh).front())};
        n_entries_neighbours +=
            (gradients_sparse[swapped_ref].get_keys().size() *
             n_entries_per_key);
      }
      Eigen::Matrix<double, Dim, Eigen::Dynamic, Eigen::RowMajor>
          grad_coeffs_pairs(Dim, n_entries_center + n_entries_neighbours);
      grad_coeffs_pairs.setZero();

      // Use the exact same iteration pattern as in f()  to guarantee that the
      // gradients appear in the same place as their corresponding data
      size_t result_idx{0};
      for (auto & key : keys_center) {
        // Here the 'flattening' retains the three Cartesian dimensions as rows,
        // since they vary the slowest within each key
        Eigen::Map<Matrix3Xd_RowMaj_t> grad_coeffs_flat(
            gradients_center[key].data(), Dim, n_entries_per_key);
        grad_coeffs_pairs.block(0, result_idx, Dim, n_entries_per_key) =
            grad_coeffs_flat;
        result_idx += n_entries_per_key;
      }
      for (auto neigh : center) {
        if (this->structure_manager->is_ghost_atom()) {
          // Don't compute gradient contributions onto ghost atoms
          continue;
        }
        // We need grad_i c^{ji} -- using just 'neigh' would give us
        // grad_j c^{ij}, hence the swap
        auto neigh_swap_images{swap_pair_ref(neigh)};
        auto & gradients_neigh_first{
            gradients_sparse[neigh_swap_images.front()]};
        // The set of species keys should be the same for all images of i
        auto keys_neigh{gradients_neigh_first.get_keys()};
        for (auto & key : keys_neigh) {
          // For each key, accumulate gradients over periodic images of the atom
          // that moves in the finite-difference step
          for (auto & neigh_swap : neigh_swap_images) {
            auto & gradients_neigh{gradients_sparse[neigh_swap]};
            Eigen::Map<Matrix3Xd_RowMaj_t> grad_coeffs_flat(
                gradients_neigh[key].data(), Dim, n_entries_per_key);
            grad_coeffs_pairs.block(0, result_idx, Dim, n_entries_per_key) +=
                grad_coeffs_flat;
          }
          result_idx += n_entries_per_key;
        }
      }
      return grad_coeffs_pairs;
    }

   private:
    RepCalculator & representation;
    std::shared_ptr<StructureManager> structure_manager;
    Structure_t atomic_structure;
    typename StructureManager::iterator center_it;

    inline void advance_center() { ++this->center_it; }

    /**
     * Swap a ClusterRef<order=2> (i, j) so it refers to (j, i) instead
     *
     * @return std::vector of ClusterRefKeys or order 2 (pair keys) of all pairs
     *         (j, i') where i' is either i or any of its periodic images within
     *         the cutoff of j. The atom j, on the other hand, must be a real
     *         atom (not a ghost or periodic image).
     *
     * @todo wouldn't this be better as a member of StructureManager
     *       (viz. AdaptorNeighbourList<whatever>)?
     */
    std::vector<PairRefKey_t> swap_pair_ref(const PairRef_t & pair_ref) {
      const double image_pos_tol = 1E-7;
      auto center_manager{extract_underlying_manager<0>(structure_manager)};
      auto atomic_structure{center_manager->get_atomic_structure()};
      // Get the atom index to the corresponding atom tag
      size_t access_index{structure_manager->get_atom_index(pair_ref.back())};
      auto new_center_it{structure_manager->get_iterator_at(access_index)};
      // Return cluster ref at which the iterator is currently pointing
      auto && new_center{*new_center_it};
      // Get the position of the original central atom (to find all its periodic
      // images)
      size_t i_index{structure_manager->get_atom_index(pair_ref.front())};
      auto i_position{structure_manager->get_position(i_index)};
      // std::cout << "cutoff: " << structure_manager->get_cutoff()<<std::endl;
      // Iterate until (j,i) is found
      std::vector<PairRefKey_t> new_pairs;
      using Positions_t = Eigen::Matrix<double, Dim, Eigen::Dynamic>;
      for (auto new_pair : new_center) {
        Positions_t i_trial_position = new_pair.get_position();
        Positions_t i_wrapped_position =
            atomic_structure.wrap_explicit_positions(i_trial_position);
        if ((i_wrapped_position - i_position).norm() < image_pos_tol) {
          new_pairs.emplace_back(std::move(new_pair));
        }
      }
      if (new_pairs.size() == 0) {
        std::stringstream err_str{};
        err_str << "Didn't find any pairs for pair (i=" << pair_ref.front()
                << ", j=" << pair_ref.back()
                << "); access index for j = " << access_index;
        throw std::range_error(err_str.str());
      }
      return new_pairs;
    }
  };

  /**
   * Test fixture holding the gradient calculator and structure manager
   *
   * Holds data (function values, gradient directions, verbosity) and iterates
   * through the list of centers
   */
  template <typename RepCalculator_t, class StructureManager_t>
  class RepresentationCalculatorGradientFixture : public GradientTestFixture {
   public:
    using StdVector2Dim_t = std::vector<std::vector<double>>;
    using Calculator_t =
        RepresentationCalculatorGradientProvider<RepCalculator_t,
                                                 StructureManager_t>;

    static const size_t n_arguments = 3;

    /**
     * Initialize a gradient test fixture
     *
     * @param filename JSON file holding gradient test parameters, format
     *                 documented in GradientTestFixture
     *
     * @param structure StructureManager on which to test
     *
     * @param calc RepresentationCalculator whose gradient is being tested
     */
    RepresentationCalculatorGradientFixture(
        std::string filename, std::shared_ptr<StructureManager_t> structure,
        Calculator_t & calc)
        : structure{structure}, center_it{structure->begin()}, calculator{
                                                                   calc} {
      json input_data = json_io::load(filename);

      this->function_inputs = this->get_function_inputs();
      this->displacement_directions =
          this->get_displacement_directions(input_data, this->n_arguments);
      this->verbosity = get_verbosity(input_data);
      if (input_data.find("fd_error_tol") != input_data.end()) {
        this->fd_error_tol = input_data["fd_error_tol"].get<double>();
      }
    }

    ~RepresentationCalculatorGradientFixture() = default;

    const Calculator_t & get_calculator() { return calculator; }

    /**
     * Go to the next center in the structure
     *
     * Not (yet) implemented as iterator because that overcomplicates things
     */
    inline void advance_center() {
      ++this->center_it;
      this->calculator.advance_center();
      if (this->has_next()) {
        this->function_inputs = get_function_inputs();
      }
    }

    inline bool has_next() { return (this->center_it != structure->end()); }

   private:
    StdVector2Dim_t get_function_inputs() {
      StdVector2Dim_t inputs_new{};
      auto center_pos = (*center_it).get_position();
      inputs_new.emplace_back(center_pos.data(),
                              center_pos.data() + center_pos.size());
      return inputs_new;
    }

    std::shared_ptr<StructureManager_t> structure;
    typename StructureManager_t::iterator center_it;
    Calculator_t & calculator;
  };

  template <class MultipleStructureFixture>
  struct MultipleStructureSortedCoulomb : MultipleStructureFixture {
    using Parent = MultipleStructureFixture;
    using ManagerTypeHolder_t = typename Parent::ManagerTypeHolder_t;
    using Representation_t = CalculatorSortedCoulomb;
    MultipleStructureSortedCoulomb() : Parent{} {};
    ~MultipleStructureSortedCoulomb() = default;

    std::vector<json> representation_hypers{
        {{"central_cutoff", 3.},
         {"central_decay", 0.5},
         {"interaction_cutoff", 10.},
         {"interaction_decay", 0.5},
         {"size", 120},
         {"sorting_algorithm", "distance"}},
        {{"central_cutoff", 3.},
         {"central_decay", 0.5},
         {"interaction_cutoff", 10.},
         {"interaction_decay", 0.5},
         {"size", 120},
         {"sorting_algorithm", "row_norm"}}};
  };

  struct SortedCoulombTestData {
    using ManagerTypeHolder_t =
        StructureManagerTypeHolder<StructureManagerCenters,
                                   AdaptorNeighbourList, AdaptorStrict>;
    using Representation_t = CalculatorSortedCoulomb;
    SortedCoulombTestData() { this->get_ref(this->ref_filename); }
    ~SortedCoulombTestData() = default;

    void get_ref(const std::string & ref_filename) {
      this->ref_data =
          json::from_ubjson(internal::read_binary_file(ref_filename));
      auto filenames =
          this->ref_data.at("filenames").get<std::vector<std::string>>();
      auto cutoffs = this->ref_data.at("cutoffs").get<std::vector<double>>();

      for (auto && filename : filenames) {
        for (auto && cutoff : cutoffs) {
          // std::cout << filename << " " << cutoff << std::endl;
          json parameters;
          json structure{{"filename", filename}};
          json adaptors;
          json ad1{
              {"name", "AdaptorNeighbourList"},
              {"initialization_arguments",
               {{"cutoff", cutoff},
                {"consider_ghost_neighbours", consider_ghost_neighbours}}}};
          json ad2{{"name", "AdaptorStrict"},
                   {"initialization_arguments", {{"cutoff", cutoff}}}};
          adaptors.emplace_back(ad1);
          adaptors.emplace_back(ad2);

          parameters["structure"] = structure;
          parameters["adaptors"] = adaptors;

          this->factory_args.emplace_back(parameters);
        }
      }
    }

    const bool consider_ghost_neighbours{false};
    json ref_data{};
    json factory_args{};

    std::string ref_filename{"reference_data/sorted_coulomb_reference.ubjson"};
    bool verbose{false};
  };

  template <class BaseFixture>
  struct CalculatorFixture : MultipleStructureFixture<BaseFixture> {
    using Parent = MultipleStructureFixture<BaseFixture>;
    using Manager_t = typename Parent::Manager_t;
    using Representation_t = typename BaseFixture::Representation_t;
    using Property_t =
        typename Representation_t::template Property_t<Manager_t>;

    CalculatorFixture() : Parent{} {}
    ~CalculatorFixture() = default;

    std::vector<Representation_t> representations{};
  };

  /* ---------------------------------------------------------------------- */

}  // namespace rascal

#endif  // TESTS_TEST_CALCULATOR_HH_
