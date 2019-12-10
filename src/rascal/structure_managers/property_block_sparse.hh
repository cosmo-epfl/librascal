/**
 * @file   rascal/structure_managers/property_block_sparse.hh
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 * @author Till Junge <till.junge@epfl.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 *
 * @date   03 April 2019
 *
 * @brief implementation of a property container that has sparse keys
 *
 * Copyright © 2019 Felix Musil, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_BLOCK_SPARSE_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_BLOCK_SPARSE_HH_

#include "rascal/math/utils.hh"
#include "rascal/structure_managers/cluster_ref_key.hh"
#include "rascal/structure_managers/property_base.hh"
#include "rascal/utils.hh"

#include <algorithm>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>

namespace rascal {
  namespace internal {

    /**
     * custom hash function for vector, list...
     * https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
     * // NOLINT
     */
    template <class KeyType>
    struct Hash {
      using result_type = size_t;
      using argument_type = KeyType;
      result_type operator()(argument_type const & vec) const {
        result_type seed{vec.size()};
        for (const auto & i : vec) {
          seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
      }
    };

    template <bool IsSorted>
    struct Sorted {};

    /**
     * Special key container that ensures its content is sorted.
     */
    template <class KeyType>
    struct SortedKey {
      using Key_t = KeyType;
      Key_t data;
      using Value_t = typename Key_t::value_type;

      explicit SortedKey(const Key_t & key) : data{key} {
        if (data.size() > 1) {
          std::sort(data.begin(), data.end());
        }
      }

      SortedKey(const Sorted<false> &, const Key_t & key) : SortedKey{key} {}

      SortedKey(const Sorted<true> &, const Key_t & key) : data{key} {}

      Key_t copy_sort(const Key_t & key) {
        Key_t skey{key};
        if (key.size() > 1) {
          std::sort(skey.begin(), skey.end());
        }
        return skey;
      }

      //! access or insert specified element. use with caution !
      Value_t & operator[](size_t id) { return this->data[id]; }

      const Key_t & get_key() const { return data; }
    };

    template <class K, class V>
    class InternallySortedKeyMap {
     public:
      using MyMap_t = std::map<K, V>;
      // using Map_t = std::unordered_map<K, std::array<int, 3>, Hash<K>>;
      using Map_t = std::map<K, std::tuple<int, int, int>>;
      using Precision_t = typename V::value_type;
      using Array_t = Eigen::Array<Precision_t, Eigen::Dynamic, 1>;
      using Vector_t = Eigen::Matrix<Precision_t, Eigen::Dynamic, 1>;
      using VectorMap_Ref_t = typename Eigen::Map<Vector_t>;
      using VectorMapConst_Ref_t = typename Eigen::Map<const Vector_t>;
      using ArrayMap_Ref_t = typename Eigen::Map<Array_t>;
      using Array_Ref_t = typename Eigen::Ref<Array_t>;
      using Self_t = InternallySortedKeyMap<K, V>;
      //! the data holder.
      Array_t & data;
      Map_t map{};
      size_t global_offset;
      size_t total_length{0};
      bool normalized{false};

      // some member types
      using key_type = typename MyMap_t::key_type;
      using mapped_type = V;
      using value_type = std::pair<const K, mapped_type>;
      using size_type = typename MyMap_t::size_type;
      using reference = typename Eigen::Map<V>;
      using const_reference = typename Eigen::Map<const V>;

      using SortedKey_t = SortedKey<key_type>;

      // member typedefs provided through inheriting from std::iterator
      template <typename Value>
      class Iterator
          : public std::iterator<
                std::bidirectional_iterator_tag,
                std::pair<K, typename std::remove_const<Value>::type>> {
       public:
        using Self_t = Iterator<Value>;

        // to handle properly const and normal cases
        using It_t = typename std::conditional<std::is_const<Value>::value,
                                               typename Map_t::const_iterator,
                                               typename Map_t::iterator>::type;

        using MyData_t =
            typename std::conditional<std::is_const<Value>::value,
                                      const Array_t, Array_t>::type;
        // Map<const Matrix> is already write-only so remove the const
        // which is used to determine the cv of the iterator
        using Value_t = typename std::remove_const<Value>::type;

        Iterator(MyData_t & data, It_t map_iterator)
            : data{data}, map_iterator{map_iterator} {}

        Self_t & operator++() {
          map_iterator++;
          // this->update_current_data();
          return *this;
        }
        Self_t operator++(int) {
          Self_t retval = *this;
          ++(*this);
          return retval;
        }
        std::pair<K, Value_t> operator*() const {
          auto && el{*this->map_iterator};
          auto && key{el.first};
          auto && pos{el.second};
          return std::make_pair(key,
                                Value_t(&data[std::get<0>(pos)],
                                        std::get<1>(pos), std::get<2>(pos)));
        }
        Self_t & operator--() {
          map_iterator--;
          return *this;
        }
        Self_t operator--(int) {
          Self_t retval = *this;
          --(*this);
          return retval;
        }
        bool operator==(const Self_t & rhs) const {
          return map_iterator == rhs.map_iterator;
        }
        bool operator!=(const Self_t & rhs) const {
          return map_iterator != rhs.map_iterator;
        }

       protected:
        MyData_t & data;
        It_t map_iterator;
      };

      using iterator = Iterator<reference>;
      using const_iterator = Iterator<const const_reference>;

      iterator begin() noexcept { return iterator(data, map.begin()); }
      iterator end() noexcept { return iterator(data, map.end()); }

      const_iterator begin() const noexcept {
        return const_iterator(data, map.begin());
      }
      const_iterator end() const noexcept {
        return const_iterator(data, map.end());
      }

      //! Default constructor
      explicit InternallySortedKeyMap(Array_t & data,
                                      const size_t & global_offset = 0)
          : data{data}, global_offset{global_offset} {};

      //! Copy constructor
      InternallySortedKeyMap(const InternallySortedKeyMap & other) = default;

      //! Move constructor
      InternallySortedKeyMap(InternallySortedKeyMap && other) = default;

      //! Destructor
      ~InternallySortedKeyMap() = default;

      //! Copy assignment operator
      Self_t & operator=(const Self_t & other) {
        this->data = other.data;
        this->map = other.map;
        this->global_offset = other.global_offset;
        this->total_length = other.total_length;
        this->normalized = other.normalized;
        return *this;
      }

      //! Move assignment operator
      Self_t & operator=(Self_t && other) {
        this->data = std::move(other.data);
        this->map = std::move(other.map);
        this->global_offset = std::move(other.global_offset);
        this->total_length = std::move(other.total_length);
        this->normalized = std::move(other.normalized);
        return *this;
      }

      /**
       * Returns a reference to the mapped value of the element with key
       * equivalent to key. If no such element exists, an exception of type
       * std::out_of_range is thrown.
       * The elements of the key are sorted in ascending order.
       *
       */
      reference at(const key_type & key) {
        SortedKey_t skey{key};
        return this->at(skey);
      }
      const_reference at(const key_type & key) const {
        SortedKey_t skey{key};
        return this->at(skey);
      }
      //! access or insert specified element
      reference operator[](const key_type & key) {
        SortedKey_t skey{key};
        return this->operator[](skey);
      }
      const_reference operator[](const key_type & key) const {
        SortedKey_t skey{key};
        return this->operator[](skey);
      }

      /**
       * Same as above but does not try to sort since we know it already is.
       */
      reference at(const SortedKey_t & skey) {
        auto & pos{this->map.at(skey.get_key())};
        return reference(&this->data[std::get<0>(pos)], std::get<1>(pos),
                         std::get<2>(pos));
      }

      const_reference at(const SortedKey_t & skey) const {
        auto & pos{this->map.at(skey.get_key())};
        return const_reference(&this->data[std::get<0>(pos)], std::get<1>(pos),
                               std::get<2>(pos));
      }
      //! access or insert specified element
      reference operator[](const SortedKey_t & skey) {
        auto & pos{this->map[skey.get_key()]};
        assert(std::get<1>(pos) * std::get<2>(pos) > 0);
        return reference(&this->data[std::get<0>(pos)], std::get<1>(pos),
                         std::get<2>(pos));
      }
      const_reference operator[](const SortedKey_t & skey) const {
        auto & pos{this->map.at(skey.get_key())};
        assert(std::get<1>(pos) * std::get<2>(pos) > 0);
        return const_reference(&this->data[std::get<0>(pos)], std::get<1>(pos),
                               std::get<2>(pos));
      }

      /**
       * Resize the view of the data to the proper size using the keys, and
       * internal array size n_row and n_col to set up this->map.
       * The view looks at one entry of the property/structure so global_offset
       * tells where this entry starts in the structure wise array of the
       * property.
       *
       * Keys_List is a stl container and it is expected that Keys_List is
       * iterable.
       *
       * Args template parameters is there to accomodate with the
       * fact that most stl container have several optional template parameters
       * that need to be deduced.
       */
      template <template <typename...> class Key_List, typename... Args>
      void resize_view(const Key_List<key_type, Args...> & keys, int n_row,
                       int n_col, const size_t & global_offset) {
        std::vector<SortedKey_t> skeys{};
        for (auto && key : keys) {
          SortedKey_t skey{key};
          skeys.push_back(skey);
        }
        this->resize_view(skeys, n_row, n_col, global_offset);
      }

      template <template <typename...> class Key_List, typename... Args>
      void resize_view(const Key_List<SortedKey_t, Args...> & skeys, int n_row,
                       int n_col, const size_t & global_offset) {
        this->global_offset = global_offset;
        size_t current_position{global_offset};
        size_t block_size{static_cast<size_t>(n_row * n_col)};
        for (auto && skey : skeys) {
          if (this->count(skey) == 0) {
            auto && key{skey.get_key()};
            this->map[key] = std::make_tuple(current_position, n_row, n_col);
            current_position += block_size;
          }
        }
        this->total_length = current_position - global_offset;
      }

      size_t size() const { return this->total_length; }

      //! Returns the number of elements with key that compares equivalent to
      //! the specified argument, which is either 1 or 0 since this container
      //! does not allow duplicates.
      size_t count(const key_type & key) const {
        SortedKey_t skey{key};
        return this->count(skey);
      }

      size_t count(const SortedKey_t & skey) const {
        return this->map.count(skey.get_key());
      }

      //! clear the map but does not change the underlying data
      void clear() noexcept { this->map.clear(); }

      VectorMap_Ref_t get_full_vector() {
        return VectorMap_Ref_t(&this->data[this->global_offset],
                               this->total_length);
      }

      VectorMapConst_Ref_t get_full_vector() const {
        return VectorMapConst_Ref_t(&this->data[this->global_offset],
                                    this->total_length);
      }

      /**
       * returns a vector of the valid keys of the map
       */
      std::vector<key_type> get_keys() const {
        std::vector<key_type> keys{};
        std::transform(this->map.begin(), this->map.end(),
                       std::back_inserter(keys), RetrieveKey());
        return keys;
      }

      void multiply_elements_by(double fac) {
        auto block{this->get_full_vector()};
        block *= fac;
      }

      /**
       * l^2 norm of the entire vector
       */
      Precision_t norm() const {
        auto block{this->get_full_vector()};
        return block.norm();
      }

      Precision_t squaredNorm() const {
        auto block{this->get_full_vector()};
        return block.squaredNorm();
      }

      /**
       * squared l^2 norm of the entire vector (sum of squared elements)
       */
      void normalize() {
        double norm{this->norm()};
        auto block{this->get_full_vector()};
        if (std::abs(norm) > 0.) {
          block /= norm;
        }
      }

      /**
       * Multiply the elements that belong to (key1, key2) entry with
       * key1 =!= key2
       *
       * relevant only when the keys have 2 indices
       */
      void multiply_off_diagonal_elements_by(double fac) {
        for (const auto & el : this->map) {
          auto && pair_type{el.first};
          auto && pos{el.second};
          if (pair_type[0] != pair_type[1]) {
            auto block{reference(&this->data[std::get<0>(pos)],
                                 std::get<1>(pos), std::get<2>(pos))};
            block *= fac;
          }
        }
      }

      /**
       * dot product with another internally sorted map
       */
      Precision_t dot(Self_t & B) {
        Precision_t val{0.};
        auto keys_b{B.get_keys()};
        auto unique_keys{this->intersection(keys_b)};

        for (auto & key : unique_keys) {
          auto && posA{this->map[key]};
          auto vecA{VectorMap_Ref_t(&this->data[std::get<0>(posA)],
                                    std::get<1>(posA) * std::get<2>(posA))};
          auto && posB{B.map[key]};
          auto vecB{VectorMap_Ref_t(&B.data[std::get<0>(posB)],
                                    std::get<1>(posB) * std::get<2>(posB))};
          val += vecA.dot(vecB);
        }
        return val;
      }

      /**
       * dot product from the left side
       * A = left_side_mat*A where A are all the key blocks
       */
      template <typename Derived>
      void lhs_dot(const Eigen::EigenBase<Derived> & left_side_mat) {
        for (const auto & el : this->map) {
          auto && pos{el.second};
          auto block{reference(&this->data[std::get<0>(pos)], std::get<1>(pos),
                               std::get<2>(pos))};
          block.transpose() *= left_side_mat;
        }
      }

      template <int Dim, typename Derived>
      void lhs_dot_der(const Eigen::EigenBase<Derived> & left_side_mat) {
        for (const auto & el : this->map) {
          auto && pos{el.second};
          auto blocks{reference(&this->data[std::get<0>(pos)], std::get<1>(pos),
                                std::get<2>(pos))};
          int n_rows{static_cast<int>(std::get<1>(pos) / Dim)};
          int n_cols{std::get<2>(pos)};
          for (int ii{0}; ii < Dim; ++ii) {
            blocks.block(ii * n_rows, 0, n_rows, n_cols).transpose() *=
                left_side_mat;
          }
        }
      }

     private:
      std::vector<key_type> intersection(std::vector<key_type> & keys) {
        if (keys.empty()) {
          return std::vector<key_type>();
        }
        std::set<key_type> set{keys.cbegin(), keys.cend()};
        std::vector<key_type> intersections;
        for (auto el : this->map) {
          if (set.erase(el.first) >
              0) {  // if n exists in set, then 1 is returned and n is erased;
                    // otherwise, 0.
            intersections.push_back(el.first);
          }
        }
        return intersections;
      }

      /**
       * Functor to get a key from a map
       */
      struct RetrieveKey {
        template <typename T>
        typename T::first_type operator()(T keyValuePair) const {
          return keyValuePair.first;
        }
      };
    };

  }  // namespace internal
  /* ---------------------------------------------------------------------- */
  /**
   * Typed ``property`` class definition, inherits from the base property class
   */
  template <typename Precision_t, size_t Order_, size_t PropertyLayer,
            class Manager, typename Key>
  class BlockSparseProperty : public PropertyBase {
   public:
    using Parent = PropertyBase;
    using Manager_t = Manager;
    using Self_t =
        BlockSparseProperty<Precision_t, Order_, PropertyLayer, Manager, Key>;
    using traits = typename Manager::traits;

    using Matrix_t = math::Matrix_t;
    using DenseRef_t = Eigen::Map<Matrix_t>;
    using Key_t = Key;
    using Keys_t = std::set<Key_t>;
    using InputData_t = internal::InternallySortedKeyMap<Key_t, Matrix_t>;
    using Data_t = Eigen::Array<Precision_t, Eigen::Dynamic, 1>;
    using Maps_t = std::vector<InputData_t>;
    // using Data_t = std::vector<InputData_t>;

    constexpr static size_t Order{Order_};
    constexpr static bool IsOrderOne{Order == 1};

   protected:
    Data_t values{};
    Maps_t maps{};
    std::string type_id;
    /**
     * boolean deciding on including the ghost atoms in the sizing of the
     * property when Order == 1
     */
    const bool exclude_ghosts;

   public:
    //! constructor
    BlockSparseProperty(Manager_t & manager,
                        std::string metadata = "no metadata",
                        bool exclude_ghosts = false)
        : Parent{static_cast<StructureManagerBase &>(manager),
                 0,
                 0,
                 Order,
                 PropertyLayer,
                 metadata},
          type_id{typeid(Self_t).name()}, exclude_ghosts{exclude_ghosts} {}

    //! Default constructor
    BlockSparseProperty() = delete;

    //! Copy constructor
    BlockSparseProperty(const BlockSparseProperty & other) = delete;

    //! Move constructor
    BlockSparseProperty(BlockSparseProperty && other) = default;

    //! Destructor
    ~BlockSparseProperty() = default;

    //! Copy assignment operator
    BlockSparseProperty & operator=(const BlockSparseProperty & other) = delete;

    //! Move assignment operator
    BlockSparseProperty & operator=(BlockSparseProperty && other) = default;

    static void check_compatibility(PropertyBase & other) {
      // check ``type`` compatibility
      auto type_id{typeid(Self_t).name()};
      if (not(other.get_type_info() == type_id)) {
        std::stringstream err_str{};
        err_str << "Incompatible types: '" << other.get_type_info() << "' != '"
                << type_id << "'.";
        throw std::runtime_error(err_str.str());
      }
    }

    /* --------------------------------------------------------------------- */

    //! return info about the type
    const std::string & get_type_info() const final { return this->type_id; }

    /**
     * Adjust size of maps to match the number of entries of the manager
     *
     * Uses SFINAE to differenciate behavior between values of Order.
     * Order > 1 then the size of the property is directly taken
     * from the manager.
     * Order == 0 then the size of the property is 1.
     * Order == 1 then the size of the property depends on the bolean
     * exclude_ghosts. By default it is set to false so that property size is
     * always larger than what could be needed.
     */
    template <size_t Order__ = Order, std::enable_if_t<(Order__ > 1), int> = 0>
    void resize() {
      size_t new_size{this->base_manager.nb_clusters(Order)};
      this->maps.resize(new_size, std::move(InputData_t(this->values)));
    }

    //! Adjust size of maps to match the number of entries of the manager
    template <size_t Order__ = Order, std::enable_if_t<(Order__ == 0), int> = 0>
    void resize() {
      this->maps.resize(1, std::move(InputData_t(this->values)));
    }

    //! Adjust size of maps to match the number of entries of the manager
    template <size_t Order__ = Order, std::enable_if_t<(Order__ == 1), int> = 0>
    void resize() {
      size_t new_size{this->exclude_ghosts
                          ? this->get_manager().size()
                          : this->get_manager().size_with_ghosts()};
      this->maps.resize(new_size, std::move(InputData_t(this->values)));
    }

    /**
     * Adjust size of values (only increases, never frees) and maps with a
     * different set of keys for each entries (order == 1 -> centers,
     * order == 2 -> neighbors ...).
     * Keys_List and Keys are stl container and it is expected that Keys_List
     * allows for random access with index from 0 to size-1 and Keys is
     * iterable.
     *
     * Args1 and Args2 template parameters are there to accomodate with the
     * fact that most stl container that can be indexed have several optional
     * template parameters that need to be deduced.
     * The first template argument of Args2 is expected to be of type Key_t or
     * SortedKey<Key_t>.
     */
    template <template <typename...> class Keys_List,
              template <typename...> class Keys, typename... Args1,
              typename... Args2>
    void resize(const Keys_List<Keys<Args2...>, Args1...> & keys_list) {
      this->resize();
      if (keys_list.size() != this->size()) {
        std::stringstream err_str{};
        err_str << "The number of keys in the list does not match the number"
                << " of entries in the property: '" << keys_list.size()
                << "' != '" << this->size() << "'.";
        throw std::runtime_error(err_str.str());
      }
      int n_row{this->get_nb_row()};
      int n_col{this->get_nb_col()};
      size_t global_offset{0};
      for (size_t i_map{0}; i_map < this->size(); i_map++) {
        this->maps[i_map].resize_view(keys_list[i_map], n_row, n_col,
                                      global_offset);
        global_offset += this->maps[i_map].size();
      }
      this->values.resize(global_offset);
    }

    /**
     * Adjust size of values (only increases, never frees) and maps with the
     * same keys for each entries (order == 1 -> centers,
     * order == 2 -> neighbors ...).
     * Keys is a stl container and iterable.
     *
     * Args template parameters are there to accomodate with the
     * fact that most stl container that can be indexed have several optional
     * template parameters that need to be deduced.
     * The first template argument of Args is expected to be of type Key_t or
     * SortedKey<Key_t>.
     *
     */
    template <template <typename...> class Keys, typename... Args>
    void resize(const Keys<Args...> & keys) {
      this->resize();
      int n_row{this->get_nb_row()};
      int n_col{this->get_nb_col()};
      size_t global_offset{0};
      for (size_t i_map{0}; i_map < this->size(); i_map++) {
        this->maps[i_map].resize_view(keys, n_row, n_col, global_offset);
        global_offset += this->maps[i_map].size();
      }
      this->values.resize(global_offset);
    }

    void setZero() { this->values = 0.; }

    size_t size() const { return this->maps.size(); }

    //! clear all the content of the property
    void clear() {
      this->values.resize(0);
      this->maps.clear();
    }

    Manager_t & get_manager() {
      return static_cast<Manager_t &>(this->base_manager);
    }

    /* -------------------------------------------------------------------- */
    //! Property accessor by cluster ref
    template <size_t CallerLayer>
    InputData_t & operator[](const ClusterRefKey<Order, CallerLayer> & id) {
      static_assert(CallerLayer >= PropertyLayer,
                    "You are trying to access a property that does not exist at"
                    "this depth in the adaptor stack.");

      return this->operator[](id.get_cluster_index(CallerLayer));
    }

    template <size_t CallerLayer>
    const InputData_t &
    operator[](const ClusterRefKey<Order, CallerLayer> & id) const {
      static_assert(CallerLayer >= PropertyLayer,
                    "You are trying to access a property that does not exist at"
                    "this depth in the adaptor stack.");

      return this->operator[](id.get_cluster_index(CallerLayer));
    }

    /**
     * Access a property of order 1 with a clusterRef of order 2
     */
    template <size_t CallerOrder, size_t CallerLayer,
              bool C = (IsOrderOne and (CallerOrder == 2)),  // NOLINT
              std::enable_if_t<C, int> = 0>                  // NOLINT
    InputData_t &
    operator[](const ClusterRefKey<CallerOrder, CallerLayer> & id) {
      return this->operator[](this->get_manager().get_atom_index(
          id.get_internal_neighbour_atom_tag()));
    }

    template <size_t CallerOrder, size_t CallerLayer, size_t Order__ = Order,
              std::enable_if_t<(Order__ == 1) and (CallerOrder == 2),  // NOLINT
                               int> = 0>                               // NOLINT
    const InputData_t &
    operator[](const ClusterRefKey<CallerOrder, CallerLayer> & id) const {
      return this->operator[](this->get_manager().get_atom_index(
          id.get_internal_neighbour_atom_tag()));
    }

    //! Accessor for property by index for dynamically sized properties
    InputData_t & operator[](size_t index) { return this->maps[index]; }

    const InputData_t & operator[](size_t index) const {
      return this->maps[index];
    }

    template <size_t CallerLayer>
    DenseRef_t operator()(const ClusterRefKey<Order, CallerLayer> & id,
                          const Key_t & key) {
      static_assert(CallerLayer >= PropertyLayer,
                    "You are trying to access a property that does not exist at"
                    "this depth in the adaptor stack.");

      return this->operator()(id.get_cluster_index(CallerLayer), key);
    }

    //! Accessor for property by index for dynamically sized properties
    DenseRef_t operator()(size_t index, const Key_t & key) {
      auto && val = this->maps[index].at(key);
      return DenseRef_t(&val(0, 0), val.rows(), val.cols());
    }

    //! Accessor for property by cluster index and return a dense
    //! representation of the property associated to this cluster
    template <size_t CallerLayer>
    Matrix_t get_dense_row(const ClusterRefKey<Order, CallerLayer> & id) {
      static_assert(CallerLayer >= PropertyLayer,
                    "You are trying to access a property that does not exist at"
                    "this depth in the adaptor stack.");

      return this->get_dense_row(id.get_cluster_index(CallerLayer));
    }

    Matrix_t get_dense_row(size_t index) {
      auto keys = this->maps[index].get_keys();
      Matrix_t feature_row = Matrix_t::Zero(this->get_nb_comp(), keys.size());
      size_t i_col{0};
      for (const auto & key : keys) {
        size_t i_row{0};
        for (int i_pos{0}; i_pos < this->get_nb_comp(); i_pos++) {
          feature_row(i_row, i_col) = this->maps[index][key](i_pos);
          i_row++;
        }
        i_col++;
      }
      return feature_row;
    }

    /**
     * Fill a dense feature matrix with layout Ncenter x Nfeatures
     * when Order == 1.
     * It is filled in the lexicografical order provided by all_keys and the
     * missing entries are filled with zeros.
     * The features are flattened out following the underlying storage order.
     *
     * @param features dense Eigen matrix of the proper size
     *
     * @param all_keys set of all the keys that should be considered when
     * building the feature matrix
     *
     */
    void fill_dense_feature_matrix(Eigen::Ref<Matrix_t> features,
                                   const Keys_t & all_keys) const {
      int inner_size{this->get_nb_comp()};
      int i_row{0};
      size_t n_center{this->maps.size()};
      for (size_t i_center{0}; i_center < n_center; i_center++) {
        int i_feat{0};
        const auto & center_val = this->maps[i_center];
        for (const auto & key : all_keys) {
          if (this->maps[i_center].count(key) == 1) {
            const auto & center_key_val = center_val[key];
            for (int i_pos{0}; i_pos < inner_size; i_pos++) {
              features(i_row, i_feat) = center_key_val(i_pos);
              i_feat++;
            }
          } else {
            i_feat += inner_size;
          }
        }  // keys
        i_row++;
      }  // centers
    }

    /**
     * Get a dense feature matrix Ncenter x Nfeatures. The keys to use are
     * deduced from the local storage.
     */
    Matrix_t get_features() {
      auto all_keys = this->get_keys();
      size_t n_elements{this->size()};
      int inner_size{this->get_nb_comp()};
      Matrix_t features =
          Matrix_t::Zero(n_elements, inner_size * all_keys.size());
      this->fill_dense_feature_matrix(features, all_keys);
      return features;
    }

    /**
     * @return set of unique keys at the level of the structure
     */
    Keys_t get_keys() const {
      Keys_t all_keys{};
      size_t n_center{this->maps.size()};
      for (size_t i_center{0}; i_center < n_center; i_center++) {
        auto keys = this->maps[i_center].get_keys();
        for (auto & key : keys) {
          all_keys.insert(key);
        }
      }
      return all_keys;
    }

    //! get number of different distinct element in the property
    //! (typically the number of center)
    size_t get_nb_item() const { return this->size(); }

    template <size_t CallerLayer>
    std::vector<Key_t>
    get_keys(const ClusterRefKey<Order, CallerLayer> & id) const {
      // static_assert(CallerOrder <= Order, "should be CallerOrder <= Order");
      static_assert(CallerLayer >= PropertyLayer,
                    "You are trying to access a property that does not exist at"
                    "this depth in the adaptor stack.");
      return this->maps[id.get_cluster_index(CallerLayer)].get_keys();
    }

    /**
     * dot product between property block sparse A and B
     * assumes order == 1 for the moment should use SFINAE to take care of
     * the case order == 2
     */
    Matrix_t dot(Self_t & B) {
      Matrix_t mat(this->size(), B.size());
      auto && manager_a{this->get_manager()};
      auto && manager_b{B.get_manager()};
      int i_row{0};
      for (auto centerA : manager_a) {
        auto && rowA{this->operator[](centerA)};
        int i_col{0};
        for (auto centerB : manager_b) {
          auto && rowB{B[centerB]};
          mat(i_row, i_col) = rowA.dot(rowB);
          ++i_col;
        }
        ++i_row;
      }
      return mat;
    }
  };

}  // namespace rascal

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_BLOCK_SPARSE_HH_