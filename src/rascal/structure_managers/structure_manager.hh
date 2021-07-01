/**
 * @file   rascal/structure_managers/structure_manager.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief  Interface for neighbourhood managers
 *
 * Copyright  2018 Till Junge, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_HH_

/*
 * Each actual implementation of a StructureManager is based on the given
 * interface
 */
#include "rascal/structure_managers/cluster_ref_key.hh"
#include "rascal/structure_managers/property.hh"
#include "rascal/structure_managers/property_block_sparse.hh"
#include "rascal/structure_managers/structure_manager_base.hh"
#include "rascal/utils/json_io.hh"
#include "rascal/utils/utils.hh"

// Some data types and operations are based on the Eigen library
#include <Eigen/Dense>

// And standard header inclusion
#include <array>
#include <cstddef>
#include <limits>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace rascal {

  /* ---------------------------------------------------------------------- */
  namespace AdaptorTraits {
    //! signals if neighbours are sorted by distance
    enum class SortedByDistance : bool { yes = true, no = false };
    //! full neighbourlist or minimal neighbourlist (no permutation of clusters)
    enum class NeighbourListType { full, half };
    //! strictness of a neighbourlist with respect to a given cutoff
    enum class Strict : bool { yes = true, no = false };  //
  }  // namespace AdaptorTraits
  /* ---------------------------------------------------------------------- */

  /**
   * traits structure to avoid incomplete types in crtp Empty because it is not
   * known what it will contain
   */
  template <class ManagerImplementation>
  struct StructureManager_traits {};

  /* ---------------------------------------------------------------------- */
  namespace internal {
    /**
     * Helper function to calculate cluster_indices_container by layer.  An
     * empty template structure is created to manage the clusters and their
     * layer
     */
    template <typename Manager, typename sequence>
    struct ClusterIndexPropertyComputer {};

    /**
     * Empty template helper structure is created to manage the clusters and
     * their layer
     */
    template <typename Manager, size_t Order, typename sequence, typename Tup>
    struct ClusterIndexPropertyComputer_Helper {};

    /**
     * Overloads helper function and is used to cycle on the objects, returning
     * the next object in line
     */
    template <typename Manager, size_t Order, size_t LayersHead,
              size_t... LayersTail, typename... TupComp>
    struct ClusterIndexPropertyComputer_Helper<
        Manager, Order, std::index_sequence<LayersHead, LayersTail...>,
        std::tuple<TupComp...>> {
      using traits = typename Manager::traits;
      using Property_t = Property<size_t, Order, Manager, LayersHead + 1, 1>;
      using type = typename ClusterIndexPropertyComputer_Helper<
          Manager, Order + 1, std::index_sequence<LayersTail...>,
          std::tuple<TupComp..., Property_t>>::type;
    };

    /**
     * Recursion end. Overloads helper function and is used to cycle on the
     * objects, for the last object in the list
     */
    template <typename Manager, size_t Order, size_t LayersHead,
              typename... TupComp>
    struct ClusterIndexPropertyComputer_Helper<Manager, Order,
                                               std::index_sequence<LayersHead>,
                                               std::tuple<TupComp...>> {
      using traits = typename Manager::traits;
      using Property_t = Property<size_t, Order, Manager, LayersHead + 1, 1>;
      using type = std::tuple<TupComp..., Property_t>;
    };

    //! Overloads the base function to call the helper function
    template <typename Manager, size_t... Layers>
    struct ClusterIndexPropertyComputer<Manager,
                                        std::index_sequence<Layers...>> {
      using type = typename ClusterIndexPropertyComputer_Helper<
          Manager, 1, std::index_sequence<Layers...>, std::tuple<>>::type;
    };

    /**
     * Empty template helper structure is created to construct cluster indices
     * tuples
     */
    template <typename Tup, typename Manager>
    struct ClusterIndexConstructor {};

    template <typename PropertyType, typename Manager>
    PropertyType make_individual_property(Manager & manager) {
      return PropertyType{manager};
    }
    //! Overload to build the tuple
    template <typename... PropertyTypes, typename Manager>
    struct ClusterIndexConstructor<std::tuple<PropertyTypes...>, Manager> {
      static std::tuple<PropertyTypes...> make(Manager & manager) {
        return std::tuple<PropertyTypes...>(
            std::move(make_individual_property<PropertyTypes>(manager))...);
      }
    };
  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  /**
   * Base class interface for neighbourhood managers. The actual implementation
   * is written in the class ManagerImplementation, and the base class both
   * inherits from it and is templated by it. This allows for compile-time
   * polymorphism without runtime cost and is called a `CRTP
   * <https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`_
   *
   * It inherits from StructureManagerbase because to provide a common interface
   * to the number of clusters and from `Updateable` to be able to update the
   * structure by using a vector of `Updateables`.
   *
   * @tparam ManagerImplementation class implementation
   */
  template <class ManagerImplementation>
  class StructureManager : public StructureManagerBase {
   public:
    using StructureManager_t = StructureManager<ManagerImplementation>;
    using traits = StructureManager_traits<ManagerImplementation>;
    using PreviousManager_t = typename traits::PreviousManager_t;
    using ImplementationPtr_t = std::shared_ptr<PreviousManager_t>;
    using ConstImplementationPtr_t =
        const std::shared_ptr<const PreviousManager_t>;
    //! type used to represent spatial coordinates, etc
    using Vector_t = Eigen::Matrix<double, traits::Dim, 1>;
    using Vector_ref = Eigen::Map<Vector_t>;
    using ClusterIndex_t = typename internal::ClusterIndexPropertyComputer<
        StructureManager, typename traits::LayerByOrder>::type;
    using ClusterConstructor_t =
        typename internal::ClusterIndexConstructor<ClusterIndex_t,
                                                   StructureManager_t>;

    /**
     * Checks if the current layer is the root of the stack e.g. the root of
     * the stack AdaptorNeighbourList<StructureManagerCenters> is
     * StructureManagerCenters. This boolean used for stopping recursive
     * functions iterating through the whole stack.
     */
    constexpr static bool IsRootImplementation =
        std::is_same<PreviousManager_t, ManagerImplementation>::value;

    //! helper to identify if Manager_t has TargetOrder,
    //! i.e. if  0 <= TargetOrder <= traits::MaxOrder
    template <size_t TargetOrder>
    static constexpr bool has_order() {
      return internal::is_order_available<TargetOrder>(
          std::make_index_sequence<traits::MaxOrder + 1>{});
    }
    //! helper type for Property creation: typed and sized
    template <typename T, size_t Order, Dim_t NbRow = 1, Dim_t NbCol = 1>
    using Property_t = Property<T, Order, StructureManager_t, NbRow, NbCol>;

    //! helper type for Property creation: only typed
    template <typename T, size_t Order>
    using TypedProperty_t = TypedProperty<T, Order, StructureManager_t>;

    //! helper type for BlockSparseProperty creation: typed
    using Key_t = std::vector<int>;
    template <typename T, size_t Order>
    using BlockSparseProperty_t =
        BlockSparseProperty<T, Order, StructureManager_t, Key_t>;

    //! type for the hyper parameter class
    using Hypers_t = json;

    //! Default constructor
    StructureManager()
        : cluster_indices_container{ClusterConstructor_t::make(*this)} {}

    //! Copy constructor
    StructureManager(const StructureManager & other) = delete;

    //! Move constructor
    StructureManager(StructureManager && other) = default;

    //! Destructor
    virtual ~StructureManager() = default;

    //! Copy assignment operator
    StructureManager & operator=(const StructureManager & other) = delete;

    //! Move assignment operator
    StructureManager & operator=(StructureManager && other) = default;

    virtual void update_self() = 0;

    // required for the construction of vectors, etc
    constexpr static int dim() { return traits::Dim; }

    /**
     * iterator over the atoms, pairs, triplets, etc in the manager. Iterators
     * like these can be used as indices for random access in atom-, pair,
     * ... -related properties.
     */
    template <size_t Order>
    class Iterator;
    using Iterator_t = Iterator<1>;
    friend Iterator_t;
    using iterator = Iterator_t;

    /**
     * return type for iterators: a light-weight atom reference, giving access
     * to an atom's position and force
     */
    class AtomRef;

    /**
     * return type for iterators: a light-weight pair, triplet, etc reference,
     * giving access to the AtomRefs of all implicated atoms
     */
    template <size_t Order>
    class ClusterRef;

    //! A proxy class which provides iteration access to atoms and ghost atoms
    class ProxyWithGhosts;

    //! A proxy class which provides iteration acces to only ghost atoms
    class ProxyOnlyGhosts;

    //! Get an iterator for a ClusterRef<1> to access pairs of an atom
    Iterator_t get_iterator_at(const size_t index, const size_t offset = 0) {
      return Iterator_t(*this, index, offset);
    }

    //! start of iterator
    Iterator_t begin() { return Iterator_t(*this, 0, 0); }
    //! end of iterator
    Iterator_t end() {
      return Iterator_t(*this, this->implementation().get_size(),
                        std::numeric_limits<size_t>::max());
    }

    //! Usage of iterator including ghosts; in case no ghost atoms exist, it is
    //! an iteration over all existing center atoms
    ProxyWithGhosts with_ghosts() {
      return ProxyWithGhosts{this->implementation()};
    }

    //! Usage of iterator for only ghosts, in case no ghosts exist, the iterator
    //! is empty
    ProxyOnlyGhosts only_ghosts() {
      return ProxyOnlyGhosts{this->implementation()};
    }

    //! i.e. number of atoms
    size_t size() const { return this->implementation().get_size(); }

    //! Tells if the cluster is a center
    template <size_t Layer>
    bool is_center_atom(const ClusterRefKey<1, Layer> & cluster) const {
      // check if cluster is not a masked atom
      return cluster.get_cluster_index(Layer) < this->size();
    }

    //! Tells if the cluster is a center
    template <size_t Layer>
    bool is_center_atom(const ClusterRefKey<2, Layer> & cluster) {
      // get the corresponding atom_j from pair_ij and check if the tag of j is
      // the same in atom_j and pair_ij. In pair_ij the atom_tag could
      // correspond to a ghost atom while the tag of atom_j always correspond to
      // an atom in the unit cell.
      // the additional check is to make sure atom_j is not a masked atom.

      // TODO(alex) put back after debugging
      auto atom_j_tag = cluster.get_atom_tag();
      auto atom_j_index = this->get_atom_index(atom_j_tag);
      auto atom_j_it = this->get_iterator_at(atom_j_index, 0);
      auto atom_j = *(atom_j_it);
      auto atom_j_tag_r = atom_j.get_atom_tag();
      auto cluster_index = atom_j.get_cluster_index(Layer);
      //std::cout << "atom_j_tag " << atom_j_tag << std::endl;
      //std::cout << "atom_j_tag_r " << atom_j_tag_r << std::endl; // <- valgrind complains, it is uninitialized
      //std::cout << "cluster_index " << cluster_index << std::endl; 
      //std::cout << "this->size() " << this->size() << std::endl; 
      return (atom_j_tag == atom_j_tag_r and
              cluster_index < this->size());
    }

    //! number of atoms including ghosts
    size_t size_with_ghosts() const {
      return this->implementation().get_size_with_ghosts();
    }

    //! number of atoms, pairs, triplets in respective manager
    size_t nb_clusters(size_t order) const final {
      return this->implementation().get_nb_clusters(order);
    }

    //! returns position of an atom with index ``atom_tag``
    Vector_ref position(int atom_tag) {
      return this->implementation().get_position(atom_tag);
    }

    //! returns position of an atom with an AtomRef ``atom``
    Vector_ref position(const AtomRef & atom) {
      return this->implementation().get_position(atom);
    }

    //! returns the atom type (convention is atomic number, but nothing is
    //! imposed apart from being an integer
    int atom_type(int atom_tag) const {
      return this->implementation().get_atom_type(atom_tag);
    }

    /**
     * Helper function to check if a property with the specifier `name` has
     * already been attached in the manager layer which invoked this function.
     */
    inline bool is_property_in_current_level(const std::string & name) const {
      return not(this->properties.find(name) == this->properties.end());
    }

    /**
     * Helper function to check if a property with the specifier `name` has
     * already been attached somewhere in the manager stack. Here an example how
     * it works
     *
     * Property request forwarding in the case property exists
     * AdaptorImpl2 -> has not prop1
     *       | forwards request to lower stack
     *       v
     * AdaptorImpl1 has prop1 -> return true
     * RootImpl
     *
     *
     * Property request forwarding in the case property does not exist
     * AdaptorImpl2
     *       | forwards request to lower stack
     *       v
     * AdaptorImpl1
     *       | forwards request to lower stack
     *       v
     * RootImpl -> return false
     */
    template <bool IsRoot = IsRootImplementation,
              std::enable_if_t<IsRoot, int> = 0>
    inline bool is_property_in_stack(const std::string & name) {
      if (this->is_property_in_current_level(name)) {
        return true;
      }
      return false;
    }

    template <bool IsRoot = IsRootImplementation,
              std::enable_if_t<not(IsRoot), int> = 0>
    inline bool is_property_in_stack(const std::string & name) {
      if (this->is_property_in_current_level(name)) {
        return true;
      }
      return this->get_previous_manager()->is_property_in_stack(name);
    }

    /**
     * Attach a property to a StructureManager. It is also connected with
     * a sanity check so that the naming of attached properties is unique. If a
     * property with the desired `name` already exists, a runtime error is
     * thrown.
     *
     * @tparam UserProperty_t the user property type
     * @param name the name of the property to create.
     *
     * @throw runtime_error if property with name does already exist
     * @return reference of to `UserProperty`
     *
     */
    template <typename UserProperty_t>
    UserProperty_t &
    create_property(const std::string & name, const bool exclude_ghosts = false,
                    const std::string & metadata = "no metadata") {
      if (this->is_property_in_stack(name)) {
        std::stringstream error{};
        error << "A property of name '" << name
              << "' has already been registered"
              << " in manager '" << this->get_name() << "'";
        throw std::runtime_error(error.str());
      } else {
        auto property{std::make_shared<UserProperty_t>(
            this->implementation(), metadata, exclude_ghosts)};
        this->properties[name] = property;
        return *property;
      }
    }

    /**
     * Checks if the user property type matches the  type of the stored
     * propertys
     *
     * @tparam UserProperty_t the user property type
     * @throw runtime_error if property with name does not exist
     * @return true if the type matches otherwise false
     */
    template <typename UserProperty_t>
    bool check_property_t(const std::string & name) const {
      if (not(this->is_property_in_stack(name))) {
        std::stringstream error{};
        error << "A property of name '" << name << "' does not exist"
              << " in manager '" << this->get_name() << "'";
        throw std::runtime_error(error.str());
      } else {
        auto && property = this->forward_get_property_request(name, false);
        try {
          UserProperty_t::check_compatibility(*property);
        } catch (const std::runtime_error & error) {
          return false;
        }
        return true;
      }
    }

    /**
     * Throws an error if property type given from user does not match the type
     * of the property in the argument. It is comparede if template parameter
     * within the UserProperty is in agreement with the stored property of the
     * given name.
     *
     * @throw runtime_error if property with name does not exists
     * @return void
     */
    template <typename UserProperty_t>
    void validate_property_t(std::shared_ptr<PropertyBase> property) const {
      try {
        UserProperty_t::check_compatibility(*property);
      } catch (const std::runtime_error & error) {
        std::stringstream err_str{};
        err_str << "Incompatible UserProperty_t used : " << error.what();
        throw std::runtime_error(err_str.str());
      }
    }

    template <typename UserProperty_t>
    void validate_property_t(const std::string & name) const {
      auto property = this->template get_property<UserProperty_t>(name, false);
      this->template validate_property_t<UserProperty_t>(property);
    }

    /**
     * Returns a typed property of the given name.
     *
     * @tparam UserProperty_t full type of the property to return.
     *
     * @param name the name of the property to get.
     * @param validate_property property is validated if this parameter is true,
     * see validate_property_t
     * @param force_creation if the property does not exist in the manager stack
     * the property is created and returned. The validation step is skipped in
     * this case.
     * @param exclude_ghosts change property sizing behavior when Order == 1
     * and when the property does not already exist.

     *
     * @throw runtime_error If validate_property is true and UserProperty_t is
     * not compatible with the property with the given name.
     * @throw runtime_error If force_creation is false and property has not been
     * found in manager stack.
     */
    template <typename UserProperty_t>
    std::shared_ptr<UserProperty_t>
    get_property(const std::string & name, const bool validate_property = true,
                 const bool force_creation = false,
                 const bool exclude_ghosts = false,
                 const std::string & metadata = "no metadata") {
      bool is_property_in_stack{this->is_property_in_stack(name)};
      if (is_property_in_stack) {
        return this->template forward_get_property_request<UserProperty_t>(
            name, validate_property);
      } else if (not(is_property_in_stack) && force_creation) {
        auto property{std::make_shared<UserProperty_t>(
            this->implementation(), metadata, exclude_ghosts)};
        this->properties[name] = property;
        return property;
      } else {
        std::stringstream error{};
        error << "No property of name '" << name << "' has been registered";
        throw std::runtime_error(error.str());
      }
    }

    /**
     * to keep track if the property is up to date with the structure
     */
    template <bool IsRoot = IsRootImplementation,
              std::enable_if_t<IsRoot, int> = 0>
    inline void set_updated_property_status(const bool is_updated) {
      for (auto & element : this->properties) {
        auto & property{element.second};
        property->set_updated_status(is_updated);
      }
    }

    template <bool IsRoot = IsRootImplementation,
              std::enable_if_t<not(IsRoot), int> = 0>
    inline void set_updated_property_status(const bool is_updated) {
      for (auto & element : this->properties) {
        auto & property{element.second};
        property->set_updated_status(is_updated);
      }
      return this->get_previous_manager()->set_updated_property_status(
          is_updated);
    }

    void set_updated_property_status(const std::string & name,
                                     bool is_updated) {
      if (this->is_property_in_current_level(name)) {
        this->properties[name]->set_updated_status(is_updated);
        return;
      } else {
        std::stringstream error{};
        error << "A property of name '" << name << "' does not exist"
              << " in manager '" << this->get_name() << "' on top layer";
        throw std::runtime_error(error.str());
      }
    }

    /**
     * Forwards property requests to lower layers. Usually to get a property
     * the `get_propertp_ptr` or `get_property_ref` function should be used.
     * This function is however still public because of each structure manager
     * needs to be able to access it from the previous manager.
     */
    template <typename UserProperty_t, bool IsRoot = IsRootImplementation,
              std::enable_if_t<IsRoot, int> = 0>
    std::shared_ptr<UserProperty_t>
    forward_get_property_request(const std::string & name,
                                 const bool validate_property) {
      if (this->is_property_in_current_level(name)) {
        auto property = this->properties.at(name);
        if (validate_property) {
          this->template validate_property_t<UserProperty_t>(property);
        }
        return std::static_pointer_cast<UserProperty_t>(property);
      } else {
        std::stringstream error{};
        error << "A property of name '" << name << "' does not exist"
              << " in manager '" << this->get_name() << "'";
        throw std::runtime_error(error.str());
      }
    }

    /**
     * Returns the property of the given name. Assumes that the property exists
     * somewhere in the stack, therefore applies no checks.
     *
     * @tparam UserProperty_t full type of the property to return.
     *
     * @param name the name of the property.
     * @param validate_property the property is validated if flag is true.
     *
     * It is
     * compared if each template parameter within the UserProperty is in
     * agreement with the stored property of the given name.
     *
     * @throw  runtime error if property name does not exist.
     * @throw  runtime error if property with name does not comply with given
     *         user property type
     */
    template <typename UserProperty_t, bool IsRoot = IsRootImplementation,
              std::enable_if_t<not(IsRoot), int> = 0>
    std::shared_ptr<UserProperty_t>
    forward_get_property_request(const std::string & name,
                                 bool validate_property) {
      if (this->is_property_in_current_level(name)) {
        auto property = this->properties.at(name);
        if (validate_property) {
          this->template validate_property_t<UserProperty_t>(property);
        }
        return std::static_pointer_cast<UserProperty_t>(property);
      } else {
        return this->get_previous_manager()
            ->template forward_get_property_request<UserProperty_t>(
                name, validate_property);
      }
    }

    template <size_t Order, size_t Layer,
              bool HasDistances = traits::HasDistances,
              typename std::enable_if_t<HasDistances, int> = 0>
    const double &
    get_distance(const ClusterRefKey<Order, Layer> & pair) const {
      static_assert(HasDistances == traits::HasDistances,
                    "The manager does not have distances.");
      return this->get_previous_manager()->get_distance(pair);
    }

    template <size_t Order, size_t Layer,
              bool HasDirectionVectors = traits::HasDirectionVectors,
              typename std::enable_if_t<HasDirectionVectors, int> = 0>
    const Vector_ref
    get_direction_vector(const ClusterRefKey<Order, Layer> & pair) const {
      static_assert(HasDirectionVectors == traits::HasDirectionVectors,
                    "The manager does not have direction vectors.");
      return this->get_previous_manager()->get_direction_vector(pair);
    }

    bool is_not_masked() const {
      return this->get_previous_manager()->is_not_masked();
    }

    //! Get the full type of the structure manager
    static std::string get_name() {
      return internal::type_name<ManagerImplementation>();
    }

    //! Create a new shared pointer to the object
    std::shared_ptr<ManagerImplementation> get_shared_ptr() {
      return this->implementation().shared_from_this();
    }

    const std::shared_ptr<const ManagerImplementation> get_shared_ptr() const {
      return this->implementation().shared_from_this();
    }

    //! Create a new weak pointer to the object
    std::weak_ptr<ManagerImplementation> get_weak_ptr() {
      return std::weak_ptr<ManagerImplementation>(this->get_shared_ptr());
    }

    //! return the number of neighbours of a given atom at a given TargetOrder
    template <size_t TargetOrder, size_t Order, size_t Layer>
    size_t get_cluster_size(const ClusterRefKey<Order, Layer> & cluster) const {
      return this->implementation().template get_cluster_size_impl<TargetOrder>(
          cluster);
    }

    //! return if the atom (Order == 1) or neighboring atom (Order == 2)
    //! is a ghost atom
    template <size_t Order>
    bool is_ghost_atom(const ClusterRef<Order> & cluster) {
      static_assert(Order <= 2, R"(Usage of this function for clusters of order
      larger than 3 is ambiguous.)");
      return (not this->is_center_atom(cluster));
    }

    /**
     * Get atom_tag of index-th neighbour of this cluster, e.g. j-th
     * neighbour of atom i or k-th neighbour of pair i-j, etc.
     * Because this function is invoked with with ClusterRefKey<1, Layer> the
     * ParentLayer and NeighbourLayer have to be optional for the case Order
     * = 1.
     */
    template <size_t Order, size_t Layer>
    int get_neighbour_atom_tag(const ClusterRefKey<Order, Layer> & cluster,
                               size_t index) const {
      return this->implementation().get_neighbour_atom_tag(cluster, index);
    }

    //! get atom_tag of the index-th atom in manager
    int get_neighbour_atom_tag(const StructureManager & cluster,
                               size_t & index) const {
      return this->implementation().get_neighbour_atom_tag(cluster, index);
    }

    //! Access to offsets for access of cluster-related properties
    template <size_t Order, size_t Layer>
    size_t get_offset(const ClusterRefKey<Order, Layer> & cluster) const {
      constexpr auto layer{
          StructureManager::template cluster_layer_from_order<Order>()};
      return cluster.get_cluster_index(layer);
    }

    //! Used for building cluster indices
    template <size_t Order>
    size_t get_offset(const std::array<size_t, Order> & counters) const {
      return this->implementation().get_offset_impl(counters);
    }

    /* Returns the neighbour's cluster_index of order 1 from an atomic index.
     */
    size_t get_atom_index(const int atom_tag) const {
      return this->implementation().get_atom_index(atom_tag);
    }

    template <size_t Order>
    constexpr static size_t cluster_layer_from_order() {
      return get_layer<Order>(typename traits::LayerByOrder{});
    }

    /**
     * When the underlying structure changes, all computations are potentially
     * invalid. This function triggers the setting of the statue variable to
     * `false` along the tree to the managers and the properties it holds.
     */
    void send_changed_structure_signal() final {
      this->set_updated_property_status(false);
      this->set_update_status(false);
      for (auto && child : this->children) {
        if (not child.expired()) {
          child.lock()->send_changed_structure_signal();
        }
      }
    }

    size_t get_property_layer(const size_t & order) const {
      return get_dyn_layer<traits::MaxOrder>(order,
                                             typename traits::LayerByOrder{});
    }

    ImplementationPtr_t get_previous_manager() {
      return this->implementation().get_previous_manager_impl();
    }

    ConstImplementationPtr_t get_previous_manager() const {
      return this->implementation().get_previous_manager_impl();
    }

   protected:
    /**
     * Update itself and send update signal to children nodes
     * Should only be used in the StructureManagerRoot
     */
    void update_children() final {
      if (not this->get_update_status()) {
        this->implementation().update_self();
        this->set_update_status(true);
      }
      for (auto && child : this->children) {
        if (not child.expired()) {
          child.lock()->update_children();
        }
      }
    }

    //! returns the current layer
    template <size_t Order>
    constexpr static size_t cluster_layer() {
      return get_layer<Order>(typename traits::LayerByOrder{});
    }

    //! recursion end, not for use
    const std::array<int, 0> get_atom_tag_list() const {
      return std::array<int, 0>{};
    }

    /* Returns the cluster size in given order and layer. Because this
     * function is invoked with with ClusterRefKey<1, Layer> the ParentLayer
     * and NeighbourLayer have to be optional for the case Order = 1.
     */

    //! returns a reference to itself
    StructureManager & get_manager() { return *this; }

    //! necessary casting of the type
    ManagerImplementation & implementation() {
      return static_cast<ManagerImplementation &>(*this);
    }
    //! returns a reference for access of the implementation
    const ManagerImplementation & implementation() const {
      return static_cast<const ManagerImplementation &>(*this);
    }

    //! get an array with all atoms inside
    std::array<AtomRef, 0> get_atoms() const {
      return std::array<AtomRef, 0>{};
    }
    //! Starting array for builing container in iterator
    std::array<int, 0> get_atom_ids() const { return std::array<int, 0>{}; }

    //! recursion end, not for use
    std::array<size_t, 1> get_counters() const {
      return std::array<size_t, 1>{};
    }
    //! access to cluster_indices_container
    ClusterIndex_t & get_cluster_indices_container() {
      return this->cluster_indices_container;
    }

    //! access to cluster_indices_container
    const ClusterIndex_t & get_cluster_indices_container() const {
      return this->cluster_indices_container;
    }

    /**
     * Tuple which contains MaxOrder number of cluster_index lists for
     * reference with increasing layer depth. It is filled upon construction
     * of the neighbourhood manager via a
     * std::get<Order>(this->cluster_indices). Higher order are constructed
     * in adaptors accordingly via the lower level indices and a
     * Order-dependend index is appended to the array.
     */
    ClusterIndex_t cluster_indices_container;

    std::map<std::string, std::shared_ptr<PropertyBase>> properties{};
  };

  /* ---------------------------------------------------------------------- */
  namespace internal {
    //! helper function that allows to append extra elements to an array It
    //! returns the given array, plus one element
    template <typename T, size_t Size, int... Indices>
    std::array<T, Size + 1>
    append_array_helper(const std::array<T, Size> & arr, T && t,
                        std::integer_sequence<int, Indices...>) {
      return std::array<T, Size + 1>{{arr[Indices]..., std::forward<T>(t)}};
    }

    //! template function allows to add an element to an array
    template <typename T, size_t Size>
    std::array<T, Size + 1> append_array(const std::array<T, Size> & arr,
                                         T && t) {
      return append_array_helper(arr, std::forward<T>(t),
                                 std::make_integer_sequence<int, Size>{});
    }

    template <typename T, size_t Size1, int... Indices1, size_t Size2,
              int... Indices2>
    std::array<T, Size1 + Size2>
    concat_array_helper(const std::array<T, Size1> & arr1,
                        const std::array<T, Size2> & arr2,
                        std::integer_sequence<int, Indices1...>,
                        std::integer_sequence<int, Indices2...>) {
      return std::array<T, Size1 + Size2>{
          {arr1[Indices1]..., arr2[Indices2]...}};
    }

    //! concatenate 2 array
    template <typename T, size_t Size1, size_t Size2>
    std::array<T, Size1 + Size2>
    concat_array(const std::array<T, Size1> & arr1,
                 const std::array<T, Size2> & arr2) {
      return concat_array_helper(arr1, arr2,
                                 std::make_integer_sequence<int, Size1>{},
                                 std::make_integer_sequence<int, Size2>{});
    }

    /* ---------------------------------------------------------------------- */
    /**
     * static branching to redirect to the correct function to get sizes,
     * offsets and neighbours. Used later by adaptors which modify or extend
     * the neighbourlist to access the correct offset.
     */
    template <bool AtMaxOrder>
    struct IncreaseHelper {
      template <class Manager_t, class Container_t>
      static int get_neighbour_atom_tag(const Manager_t & /*manager*/,
                                        const Container_t & /*container*/,
                                        size_t /*index*/) {
        throw std::runtime_error("This branch should never exist"
                                 " (cluster neigbour).");
      }
    };

    /**
     * specialization for not at MaxOrder, these refer to the underlying manager
     */
    template <>
    struct IncreaseHelper<false> {
      template <class Manager_t, class Container_t>
      static int get_neighbour_atom_tag(const Manager_t & manager,
                                        const Container_t & container,
                                        size_t index) {
        return manager.get_neighbour_atom_tag(container, index);
      }
    };

    template <class ParentClass, typename ClusterIndicesType, size_t Layer>
    struct ClusterIndicesConstCaster {
      using IndexConstArray_t = typename ParentClass::IndexConstArray;
      using IndexArray_t = typename ParentClass::IndexArray;

      static IndexConstArray_t & cast(IndexConstArray_t & cluster_indices) {
        return cluster_indices;
      }

      static IndexConstArray_t cast(const IndexArray_t & cluster_indices) {
        return IndexConstArray_t(cluster_indices.data());
      }

      static IndexConstArray_t cast(const size_t & cluster_index) {
        return IndexConstArray_t(&cluster_index);
      }
    };
  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  /**
   * Definition of the ``AtomRef`` class. It is the return type when
   * iterating over the first order of a manager.
   */
  template <class ManagerImplementation>
  class StructureManager<ManagerImplementation>::AtomRef {
   public:
    using Vector_t = Eigen::Matrix<double, ManagerImplementation::dim(), 1>;
    using Vector_ref = Eigen::Map<Vector_t>;
    using Manager_t = StructureManager<ManagerImplementation>;

    //! Default constructor
    AtomRef() = delete;

    //! constructor from iterator
    AtomRef(Manager_t & manager, int id) : manager{manager}, index{id} {}

    //! Copy constructor
    AtomRef(const AtomRef & other) = default;

    //! Move constructor
    AtomRef(AtomRef && other) = default;

    //! Destructor
    ~AtomRef() {}

    //! Copy assignment operator
    AtomRef & operator=(const AtomRef & other) = delete;

    //! Move assignment operator
    AtomRef & operator=(AtomRef && other) = default;

    //! return index of the atom
    int get_index() const { return this->index; }

    //! return position vector of the atom
    Vector_ref get_position() { return this->manager.position(this->index); }

    /**
     * return atom type (idea: corresponding atomic number, but is allowed
     * to be arbitrary as long as it is an integer)
     */
    int get_atom_type() const { return this->manager.atom_type(this->index); }

   protected:
    //! reference to the underlying manager
    Manager_t & manager;
    /**
     * The meaning of `index` is manager-dependent. There are no guaranties
     * regarding contiguity. It is used internally to absolutely address
     * atom-related properties.
     */
    int index;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * Class definitionobject when iterating over the manager, then atoms,
   * then pairs, etc. in deeper Orders. This object itself is iterable again
   * up to the corresponding MaxOrder of the manager. I.e. iterating over a
   * manager provides atoms; iterating over atoms gives its pairs, etc.
   */
  template <class ManagerImplementation>
  template <size_t Order>
  class StructureManager<ManagerImplementation>::ClusterRef
      : public ClusterRefKey<
            Order,
            ManagerImplementation::template cluster_layer_from_order<Order>()> {
   public:
    static_assert(Order > 0, "Order < 1 is not allowed.");
    using Manager_t = StructureManager<ManagerImplementation>;
    using traits = StructureManager_traits<ManagerImplementation>;
    constexpr static size_t ClusterLayer{
        ManagerImplementation::template cluster_layer_from_order<Order>()};
    using ThisParentClass = ClusterRefKey<Order, ClusterLayer>;

    using AtomRef_t = typename Manager_t::AtomRef;
    using Iterator_t = typename Manager_t::template Iterator<Order>;
    using Atoms_t = std::array<AtomRef_t, Order>;
    using iterator = typename Manager_t::template Iterator<Order + 1>;
    friend iterator;

    using IndexConstArray_t = typename ThisParentClass::IndexConstArray;
    using IndexArray_t = typename ThisParentClass::IndexArray;

    using AtomIndex_t = typename ThisParentClass::AtomIndex_t;

    static constexpr bool IsOrderOne{Order == 1};

    template <size_t TargetOrder>
    static constexpr bool IsOrderOneAndHasOrder{
        Manager_t::template has_order<TargetOrder>() and IsOrderOne};

    //! true if ClusterRef of Order 1 and the manager has self pairs
    static constexpr bool HasCenterPairAndIsOrderOne{traits::HasCenterPair and
                                                     IsOrderOne};
    //! true if ClusterRef of Order 2 and the manager has self pairs
    static constexpr bool HasCenterPairAndIsOrderTwo{traits::HasCenterPair and
                                                     Order == 2};
    //! Default constructor
    ClusterRef() = delete;

    template <typename ClusterIndicesType>
    ClusterRef(Iterator_t & it, const AtomIndex_t & atom_tag_list,
               ClusterIndicesType & cluster_indices)
        : ThisParentClass{atom_tag_list,
                          internal::ClusterIndicesConstCaster<
                              ThisParentClass, ClusterIndicesType,
                              ClusterLayer>::cast(cluster_indices)},
          it{it} {}

    /**
     * This is a ClusterRef of Order=1, constructed from a higher Order.
     * This function here is self referencing right now. A ClusterRefKey
     * with Order=1 is needed to construct it ?!
     */
    ClusterRef(ClusterRefKey<1, 0> & cluster, Manager_t & manager)
        : ClusterRefKey<1, 0>(cluster.get_atom_tag_list(),
                              cluster.get_cluster_indices()),
          it(manager) {}

    //! Copy constructor
    ClusterRef(const ClusterRef & other) = delete;

    //! Move constructor
    ClusterRef(ClusterRef && other) = default;

    //! Destructor
    virtual ~ClusterRef() = default;

    //! Copy assignment operator
    ClusterRef & operator=(const ClusterRef & other) = default;

    //! Move assignment operator
    ClusterRef & operator=(ClusterRef && other) = default;

    //! return atoms of the current cluster
    const std::array<AtomRef_t, Order> & get_atoms() const {
      return this->atoms;
    }

    /**
     * Getter for a ClusterRefKey refering to the current j-atom of the
     * ij-pair.
     *
     * if you try to use this function and Order != 2 then
     * you will get an error about not finding the function to call
     * because of SFINAE.
     *
     * @return ClusterRefKey of order 1 and proper layer
     */
    template <size_t Order_ = Order, std::enable_if_t<Order_ == 2, int> = 0>
    auto get_atom_j() {
      auto && manager = it.get_manager();
      auto && atom_j_tag = this->get_internal_neighbour_atom_tag();
      auto && atom_j_index = manager.get_atom_index(atom_j_tag);
      auto atom_j_it = manager.get_iterator_at(atom_j_index, 0);
      constexpr static size_t ClusterLayer_{
          ManagerImplementation::template cluster_layer_from_order<1>()};
      auto atom_j = static_cast<ClusterRefKey<1, ClusterLayer_>>(*atom_j_it);
      return atom_j;
    }

    /**
     * Getter for a ClusterRefKey refering to the ii-pair of the current
     * i-atom.
     *
     * if you try to use this function and HasCenterPair == false then
     * you will get an error about not finding the function to call
     * because of SFINAE.
     *
     * @return ClusterRefKey of order 2 and proper layer
     */
    template <bool C = HasCenterPairAndIsOrderOne, std::enable_if_t<C, int> = 0>
    auto get_atom_ii() {
      static_assert(traits::MaxOrder > 1, "Need neighbors to get one");

      auto && atom_ii_it = this->pairs_with_self_pair().begin();
      constexpr static size_t ClusterLayer_{
          ManagerImplementation::template cluster_layer_from_order<2>()};
      auto atom_ii = static_cast<ClusterRefKey<2, ClusterLayer_>>(*atom_ii_it);
      return atom_ii;
    }

    /**
     * Getter for a ClusterRefKey refering to the current jj-pair associated
     * to the current ij-pair.
     *
     * if you try to use this function and HasCenterPair == false then
     * you will get an error about not finding the function to call
     * because of SFINAE.
     *
     * @return ClusterRefKey of order 2 and proper layer
     */
    template <bool C = HasCenterPairAndIsOrderTwo, std::enable_if_t<C, int> = 0>
    auto get_atom_jj() {
      auto && manager = it.get_manager();
      auto && atom_j_tag = this->get_atom_tag();
      auto && atom_j_index = manager.get_atom_index(atom_j_tag);
      auto && atom_j_it = manager.get_iterator_at(atom_j_index, 0);
      auto && atom_j = *atom_j_it;
      auto && atom_jj_it = atom_j.pairs_with_self_pair().begin();
      constexpr static size_t ClusterLayer_{
          ManagerImplementation::template cluster_layer_from_order<2>()};
      auto atom_jj = static_cast<ClusterRefKey<2, ClusterLayer_>>(*atom_jj_it);
      return atom_jj;
    }

    /**
     * Returns the position of the last atom in the cluster, e.g. when
     * cluster order==1 it is the atom position, when cluster order==2 it is
     * the neighbour position, etc.
     */
    auto get_position() {
      return this->get_manager().position(this->get_atom_tag());
    }

    //! returns the type of the last atom in the cluster
    int get_atom_type() const {
      auto && id{this->get_atom_tag()};
      return this->get_manager().atom_type(id);
    }

    /**
     * build a array of atom types from the atoms in this cluster
     */
    std::array<int, Order> get_atom_types() const;

    //! return the index of the atom/pair/etc. it is always the last one,
    //! since the other ones are accessed an Order above.
    int get_atom_tag() const { return this->back(); }
    //! returns a reference to the manager with the maximum layer
    Manager_t & get_manager() { return this->it.get_manager(); }

    //! return a const reference to the manager with maximum layer
    const Manager_t & get_manager() const { return this->it.get_manager(); }

    //! return iterator index - this is used in cluster_indices_container as
    //! well as accessing properties
    size_t get_index() const { return this->it.index; }

    //! returns the clusters index refering to the whole list of current
    //! clusters
    size_t get_global_index() const { return this->it.get_cluster_index(); }
    //! returns the atom tags, which constitute the cluster
    const std::array<int, Order> & get_atom_tag_list() const {
      return this->atom_tag_list;
    }

    Iterator_t & get_iterator() { return this->it; }
    const Iterator_t & get_iterator() const { return this->it; }

   protected:
    //! counters for access
    std::array<size_t, 1> get_counters() const {
      return this->it.get_counters();
    }
    std::array<size_t, 1> get_offsets() const { return this->it.get_offsets(); }
    //! `atom_cluster_indices` is an initially contiguous numbering of atoms
    Iterator_t & it;

    /**
     * Helper struct to iterate in a customised range. Useful to return an
     * iterator over the pairs (TargetOrder == 2),
     * triplets (TargetOrder == 3)...
     */
    template <size_t TargetOrder>
    struct CustomProxy {
      using ClusterRef_t = typename Manager_t::template ClusterRef<1>;
      using iterator = typename Manager_t::template Iterator<TargetOrder>;

      CustomProxy(ClusterRef_t & cluster_ref, const size_t & start,
                  const size_t & offset)
          : cluster_ref{cluster_ref}, start{start}, offset{offset} {}

      //! end of the iterations over the clusters of order TargetOrder
      iterator begin() {
        return iterator(cluster_ref, this->start, this->offset);
      }
      //! end of the iterations over the clusters
      iterator end() {
        return iterator(cluster_ref, this->size(),
                        std::numeric_limits<size_t>::max());
      }
      //! get the number of neighbors of the center at Order == TargetOrder
      size_t size() {
        return cluster_ref.get_manager().template get_cluster_size<TargetOrder>(
            cluster_ref);
      }

      ClusterRef_t & cluster_ref;
      //! starting index of the iteration
      size_t start;
      //! offset with which to start the iteration in the list of all clusters
      //! of Order == TargetOrder
      size_t offset;
    };

   public:
    /**
     * Return an iterable over the clusters of order == Order
     * associated with the current central atom.
     * @param start starting index for the iteration from offset
     */
    template <size_t Order_, bool C = IsOrderOne, std::enable_if_t<C, int> = 0>
    CustomProxy<Order_> get_clusters_of_order(size_t start = 0) {
      static_assert(Order_ > 1,
                    "You should ask at least for pairs, i.e. Order_ >= 2.");
      std::array<size_t, Order_ - 1> counters{};
      counters.back() = this->get_index();
      size_t offset{this->get_manager().get_offset(counters)};
      return CustomProxy<Order_>(*this, start, offset);
    }

    /**
     * Return an iterable for Order == 2 that includes the neighbors (or
     * pairs) associated with the current central atom.
     */
    template <bool C = IsOrderOneAndHasOrder<2>, std::enable_if_t<C, int> = 0>
    CustomProxy<2> pairs() {
      // avoid if statement or sfinae by casting the bool to
      // size_t which turns out to give 0 if false and 1
      // if true, the expected behavior.
      size_t start{static_cast<size_t>(HasCenterPairAndIsOrderOne)};
      return this->get_clusters_of_order<2>(start);
    }

    /**
     * Return an iterable for Order == 2 that includes the neighbors (or
     * pairs) associated with the current central atom and the self pair,
     * i.e. a ClusterRef<2> refering to the central atom if
     * HasCenterPair == true.
     * If HasCenterPair == false then its the regular iteration.
     */
    template <bool C = IsOrderOneAndHasOrder<2>, std::enable_if_t<C, int> = 0>
    CustomProxy<2> pairs_with_self_pair() {
      return this->get_clusters_of_order<2>();
    }

    /**
     * Return an iterable for Order == 3 that includes the triplets associated
     * with the current central atom.
     */
    template <bool C = IsOrderOneAndHasOrder<3>, std::enable_if_t<C, int> = 0>
    CustomProxy<3> triplets() {
      return this->get_clusters_of_order<3>();
    }

    /**
     * Return an iterable for Order == 4 that includes the quadruplets
     * associated with the current central atom.
     */
    template <bool C = IsOrderOneAndHasOrder<4>, std::enable_if_t<C, int> = 0>
    CustomProxy<4> quadruplets() {
      return this->get_clusters_of_order<4>();
    }
  };

  namespace internal {
    template <class Manager, size_t Order, size_t... Indices>
    std::array<int, Order>
    species_aggregator_helper(const std::array<int, Order> & array,
                              const Manager & manager,
                              std::index_sequence<Indices...> /*indices*/) {
      return std::array<int, Order>{{manager.atom_type(array[Indices])...}};
    }

    template <class Manager, size_t Order, size_t... Indices>
    std::array<int, Order>
    species_aggregator(const std::array<int, Order> & index_array,
                       const Manager & manager) {
      return species_aggregator_helper<Manager>(
          index_array, manager, std::make_index_sequence<Order>{});
    }
  };  // namespace internal

  template <class ManagerImplementation>
  template <size_t Order>
  std::array<int, Order>
  StructureManager<ManagerImplementation>::ClusterRef<Order>::get_atom_types()
      const {
    return internal::species_aggregator<StructureManager>(this->atom_tag_list,
                                                          this->get_manager());
  }

  /* ---------------------------------------------------------------- */
  /**
   * Helper functions to avoid needing dereferencing a manager in a
   * shared_ptr to loop over the centers.
   */
  template <typename T>
  auto begin(std::shared_ptr<T> ptr) -> typename T::iterator {
    return ptr->begin();
  }

  template <typename T>
  auto end(std::shared_ptr<T> ptr) -> typename T::iterator {
    return ptr->end();
  }

  /* ---------------------------------------------------------------- */
  /**
   * Class definition of the iterator. This is used by all clusters. It is
   * specialized for the case Order=1, when iterating over a manager and the
   * dereference are atoms, because then the container is a manager. For all
   * other cases, the container is the cluster of the Order below.
   */
  template <class ManagerImplementation>
  template <size_t Order>
  class StructureManager<ManagerImplementation>::Iterator {
   public:
    using Manager_t = StructureManager<ManagerImplementation>;
    friend Manager_t;
    using ClusterRef_t = typename Manager_t::template ClusterRef<Order>;

    friend ClusterRef_t;

    static constexpr bool IsOrderOne{Order == 1};

    static constexpr bool IsOrderOneOrTwo{IsOrderOne or
                                          (Order == 2)};  // NOLINT

    // determine the container type
    using Container_t =
        std::conditional_t<IsOrderOne, Manager_t,
                           typename Manager_t::template ClusterRef<1>>;

    using AtomIndex_t = typename ClusterRef_t::AtomIndex_t;

    static_assert(Order > 0, "Order has to be positive");

    static_assert(Order <= traits::MaxOrder,
                  "Order > MaxOrder, impossible iterator");

    using AtomRef_t = typename Manager_t::AtomRef;

    using value_type = ClusterRef_t;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    // using iterator_category = std::random_access_iterator_tag;

    using reference = value_type;

    //! Default constructor
    Iterator() = delete;

    //! Copy constructor
    Iterator(const Iterator & other) = default;

    //! Move constructor
    Iterator(Iterator && other) = default;

    //! Destructor
    virtual ~Iterator() = default;

    //! Copy assignment operator
    Iterator & operator=(const Iterator & other) = default;

    //! Move assignment operator
    Iterator & operator=(Iterator && other) = default;

    //! pre-increment
    Iterator & operator++() {
      ++this->index;
      return *this;
    }

    //! pre-decrement
    Iterator & operator--() {
      --this->index;
      return *this;
    }

    value_type operator*() {
      auto & cluster_indices_properties = std::get<Order - 1>(
          this->get_manager().get_cluster_indices_container());
      using Ref_t = typename std::remove_reference_t<decltype(
          cluster_indices_properties)>::reference;
      Ref_t cluster_indices{
          cluster_indices_properties[this->get_cluster_index()]};
      return ClusterRef_t(*this, this->get_atom_tag_list(), cluster_indices);
    }

    const value_type operator*() const {
      const auto & cluster_indices_properties = std::get<Order - 1>(
          this->get_manager().get_cluster_indices_container());
      using Ref_t = typename std::remove_reference_t<decltype(
          cluster_indices_properties)>::const_reference;
      Ref_t cluster_indices =
          cluster_indices_properties[this->get_cluster_index()];

      const auto indices{this->get_atom_tag_list()};
      return ClusterRef_t(const_cast<Iterator &>(*this), indices,
                          cluster_indices);
    }

    //! equality
    bool operator==(const Iterator & other) const {
      return this->index == other.index;
    }

    //! inequality
    bool operator!=(const Iterator & other) const {
      return not(*this == other);
    }

    /**
     * const access to container
     */
    const Container_t & get_container() const { return this->container; }

   protected:
    //! constructor with container ref and starting point
    Iterator(Container_t & cont, size_t start, size_t offset)
        : container{cont}, index{start}, offset{offset} {}

    //! add atomic indices in current iteration
    template <bool C = IsOrderOneOrTwo, std::enable_if_t<C, int> = 0>
    AtomIndex_t get_atom_tag_list() {
      return internal::append_array(
          container.get_atom_tag_list(),
          this->get_manager().get_neighbour_atom_tag(container, this->index));
    }
    //! add atomic indices in current iteration
    template <bool C = IsOrderOneOrTwo, std::enable_if_t<C, int> = 0>
    AtomIndex_t get_atom_tag_list() const {
      return internal::append_array(
          container.get_atom_tag_list(),
          this->get_manager().get_neighbour_atom_tag(container, this->index));
    }

    template <bool C = IsOrderOneOrTwo, std::enable_if_t<not(C), int> = 0>
    AtomIndex_t get_atom_tag_list() {
      return internal::concat_array(
          container.get_atom_tag_list(),
          this->get_manager().implementation().get_neighbour_atom_tag_current(
              container, this->index));
    }
    template <bool C = IsOrderOneOrTwo, std::enable_if_t<not(C), int> = 0>
    AtomIndex_t get_atom_tag_list() const {
      return internal::concat_array(
          container.get_atom_tag_list(),
          this->get_manager().implementation().get_neighbour_atom_tag_current(
              container, this->index));
    }

    //! returns the current index of the cluster in iteration
    size_t get_cluster_index() const { return this->index + this->offset; }

    //! returns a reference to the underlying manager at every Order
    Manager_t & get_manager() { return this->container.get_manager(); }
    //! returns a const reference to the underlying manager at every Order
    const Manager_t & get_manager() const {
      return this->container.get_manager();
    }

    //! returns the counters - which is the position in a list at each
    //! Order.
    template <bool C = IsOrderOne, std::enable_if_t<C, int> = 0>
    std::array<size_t, 1> get_counters() {
      std::array<size_t, 1> counters{this->index};
      return counters;
    }

    //! returns the counters - which is the position in a list at each
    //! Order.
    template <bool C = IsOrderOne, std::enable_if_t<not(C), int> = 0>
    std::array<size_t, 2> get_counters() {
      auto && parental_counters = this->container.get_counters();
      std::array<size_t, 2> counters{parental_counters[0], this->index};
      return counters;
    }

    //! get offsets at which the current clusters at several orders should
    //! start
    template <bool C = IsOrderOne, std::enable_if_t<C, int> = 0>
    std::array<size_t, 1> get_offsets() {
      std::array<size_t, 1> offsets{this->offset};
      return offsets;
    }

    //! get offsets at which the current clusters at several orders should
    //! start
    template <bool C = IsOrderOne, std::enable_if_t<not(C), int> = 0>
    std::array<size_t, 2> get_offsets() {
      auto && parental_offsets = this->container.get_offsets();
      std::array<size_t, 2> offsets{parental_offsets[0], this->offset};
      return offsets;
    }

    //! in ascending order, this is: manager, atom, pair, triplet (i.e.
    //! cluster of Order 0, 1, 2, 3, ...
    Container_t & container;
    //! the iterators index (for moving forwards)
    size_t index;
    //! offset for access in a neighbour list during construction of the
    //! begin()
    const size_t offset;
  };

  /* --------------------------------------------------------------------- */
  /**
   * A class which provides the iteration range from start to the end of the
   * atoms including additional ghost atoms.
   */
  template <class ManagerImplementation>
  class StructureManager<ManagerImplementation>::ProxyWithGhosts {
   public:
    using Iterator_t =
        typename StructureManager<ManagerImplementation>::Iterator_t;

    //! Default constructor
    ProxyWithGhosts() = delete;

    //! Constructor
    ProxyWithGhosts(ManagerImplementation & manager) : manager{manager} {}

    //! Copy constructor
    ProxyWithGhosts(const ProxyWithGhosts & other) = delete;

    //! Move constructor
    ProxyWithGhosts(ProxyWithGhosts && other) = default;

    //! Destructor
    virtual ~ProxyWithGhosts() = default;

    //! Copy assignment operator
    ProxyWithGhosts & operator=(const ProxyWithGhosts & other) = delete;

    //! Move assignment operator
    ProxyWithGhosts & operator=(ProxyWithGhosts && other) = default;

    //! Start of atom list
    Iterator_t begin() { return this->manager.begin(); }

    //! End is all atoms including ghosts
    Iterator_t end() {
      return Iterator_t(this->manager, this->manager.size_with_ghosts(),
                        std::numeric_limits<size_t>::max());
    }

   protected:
    ManagerImplementation & manager;

   private:
  };

  /**
   * A class which provides the iteration range from for all ghost atoms in
   * the structure. If no ghost atoms exist, the iterator is of size zero.
   */
  template <class ManagerImplementation>
  class StructureManager<ManagerImplementation>::ProxyOnlyGhosts
      : public StructureManager<ManagerImplementation>::ProxyWithGhosts {
   public:
    using Iterator_t =
        typename StructureManager<ManagerImplementation>::Iterator_t;
    using Parent =
        typename StructureManager<ManagerImplementation>::ProxyWithGhosts;

    //! Default constructor
    ProxyOnlyGhosts() = delete;

    //!
    ProxyOnlyGhosts(ManagerImplementation & manager) : Parent{manager} {}

    //! Copy constructor
    ProxyOnlyGhosts(const ProxyOnlyGhosts & other) = delete;

    //! Move constructor
    ProxyOnlyGhosts(ProxyOnlyGhosts && other) = default;

    //! Destructor
    virtual ~ProxyOnlyGhosts() = default;

    //! Copy assignment operator
    ProxyOnlyGhosts & operator=(const ProxyOnlyGhosts & other) = delete;

    //! Move assignment operator
    ProxyOnlyGhosts & operator=(ProxyOnlyGhosts && other) = default;

    //! Start iteration at first ghost atom
    Iterator_t begin() { return this->manager.end(); }

   protected:
   private:
  };
}  // namespace rascal

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_HH_
