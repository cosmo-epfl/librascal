/**
 * file   adaptor_neighbour_list.hh
 *
 * @author Markus Stricker <markus.stricker@epfl.ch>
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   04 Oct 2018
 *
 * @brief implements an adaptor for structure_managers, which
 * creates a full or half neighbourlist if there is none
 *
 * Copyright  2018 Markus Stricker, Till Junge, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_STRUCTURE_MANAGERS_ADAPTOR_NEIGHBOUR_LIST_HH_
#define SRC_STRUCTURE_MANAGERS_ADAPTOR_NEIGHBOUR_LIST_HH_

#include "atomic_structure.hh"
#include "basic_types.hh"
#include "lattice.hh"
#include "rascal_utility.hh"
#include "structure_managers/property.hh"
#include "structure_managers/structure_manager.hh"

#include <set>
#include <typeinfo>
#include <vector>

namespace rascal {
  /**
   * Forward declaration for traits
   */
  template <class ManagerImplementation>
  class AdaptorNeighbourList;

  /**
   * Specialisation of traits for increase <code>MaxOrder</code> adaptor
   */
  template <class ManagerImplementation>
  struct StructureManager_traits<AdaptorNeighbourList<ManagerImplementation>> {
    constexpr static AdaptorTraits::Strict Strict{AdaptorTraits::Strict::no};
    constexpr static bool HasDistances{false};
    constexpr static bool HasDirectionVectors{false};
    constexpr static int Dim{ManagerImplementation::traits::Dim};
    // New MaxOrder upon construction, by construction should be 2
    constexpr static size_t MaxOrder{ManagerImplementation::traits::MaxOrder +
                                     1};
    // When using periodic boundary conditions, it is possible that atoms are
    // added upon construction of the neighbour list. Therefore the layering
    // sequence is reset: here is layer 0 again.
    using LayerByOrder = std::index_sequence<0, 0>;
  };

  namespace internal {
    /* ---------------------------------------------------------------------- */
    //! integer base-to-the-power function
    template <typename R, typename I>
    constexpr R ipow(R base, I exponent) {
      static_assert(std::is_integral<I>::value, "Type must be integer");
      R retval{1};
      for (I i = 0; i < exponent; ++i) {
        retval *= base;
      }
      return retval;
    }
    /* ---------------------------------------------------------------------- */
    /**
     * stencil iterator for simple, dimension-dependent stencils to access the
     * neighbouring boxes of the cell algorithm
     */
    template <size_t Dim>
    class Stencil {
     public:
      //! constructor
      explicit Stencil(const std::array<int, Dim> & origin) : origin{origin} {};
      //! copy constructor
      Stencil(const Stencil & other) = default;
      //! assignment operator
      Stencil & operator=(const Stencil & other) = default;
      //! destructor
      ~Stencil() = default;

      //! iterators over `` dereferences to cell coordinates
      class iterator {
       public:
        using value_type = std::array<int, Dim>;    //!< stl conformance
        using const_value_type = const value_type;  //!< stl conformance
        using pointer = value_type *;               //!< stl conformance
        using iterator_category =
            std::forward_iterator_tag;  //!< stl conformance
        //! constructor
        explicit iterator(const Stencil & stencil, bool begin = true)
            : stencil{stencil}, index{begin ? 0 : stencil.size()} {}
        //! destructor
        ~iterator() {}
        //! dereferencing
        value_type operator*() const {
          constexpr int size{3};
          std::array<int, Dim> retval{{0}};
          int factor{1};
          for (int i{Dim - 1}; i >= 0; --i) {
            //! -1 for offset of stencil
            retval[i] =
                this->index / factor % size + this->stencil.origin[i] - 1;
            if (i != 0) {
              factor *= size;
            }
          }
          return retval;
        }
        //! pre-increment
        iterator & operator++() {
          this->index++;
          return *this;
        }
        //! inequality
        inline bool operator!=(const iterator & other) const {
          return this->index != other.index;
        }

       protected:
        //! ref to stencils
        const Stencil & stencil;
        //! index of currect pointed-to voxel
        size_t index;
      };
      //! stl conformance
      inline iterator begin() const { return iterator(*this); }
      //! stl conformance
      inline iterator end() const { return iterator(*this, false); }
      //! stl conformance
      inline size_t size() const { return ipow(3, Dim); }

     protected:
      //! locations of this domain
      const std::array<int, Dim> origin;
    };

    /* ---------------------------------------------------------------------- */
    /**
     * Periodic image iterator for easy access to how many images have to be
     * added for ghost atoms.
     */
    template <size_t Dim>
    class PeriodicImages {
     public:
      //! constructor
      PeriodicImages(const std::array<int, Dim> & origin,
                     const std::array<int, Dim> & nrepetitions,
                     const size_t & ntot)
          : origin{origin}, nrepetitions{nrepetitions}, ntot{ntot} {};
      //! copy constructor
      PeriodicImages(const PeriodicImages & other) = default;
      //! assignment operator
      PeriodicImages & operator=(const PeriodicImages & other) = default;
      ~PeriodicImages() = default;

      //! iterators over `` dereferences to cell coordinates
      class iterator {
       public:
        using value_type = std::array<int, Dim>;    //!< stl conformance
        using const_value_type = const value_type;  //!< stl conformance
        using pointer = value_type *;               //!< stl conformance
        using iterator_category =
            std::forward_iterator_tag;  //!< stl conformance

        //! constructor
        explicit iterator(const PeriodicImages & periodic_images,
                          bool begin = true)
            : periodic_images{periodic_images},
              index{begin ? 0 : periodic_images.size()} {}

        ~iterator() {}
        //! dereferencing
        value_type operator*() const {
          std::array<int, Dim> retval{{0}};
          int factor{1};
          for (int i = Dim - 1; i >= 0; --i) {
            retval[i] =
                this->index / factor % this->periodic_images.nrepetitions[i] +
                this->periodic_images.origin[i];
            if (i != 0) {
              factor *= this->periodic_images.nrepetitions[i];
            }
          }
          return retval;
        }
        //! pre-increment
        iterator & operator++() {
          this->index++;
          return *this;
        }
        //! inequality
        inline bool operator!=(const iterator & other) const {
          return this->index != other.index;
        }

       protected:
        const PeriodicImages & periodic_images;  //!< ref to periodic images
        size_t index;  //!< index of currect pointed-to pixel
      };
      //! stl conformance
      inline iterator begin() const { return iterator(*this); }
      //! stl conformance
      inline iterator end() const { return iterator(*this, false); }
      //! stl conformance
      inline size_t size() const { return this->ntot; }

     protected:
      const std::array<int, Dim> origin;  //!< minimum repetitions
      //! repetitions in each dimension
      const std::array<int, Dim> nrepetitions;
      const size_t ntot;
    };

    /* ---------------------------------------------------------------------- */
    /**
     * Mesh bounding coordinates iterator for easy access to the corners of the
     * mesh for evaluating the multipliers necessary to build necessary periodic
     * images, depending on periodicity.
     */
    template <size_t Dim>
    class MeshBounds {
     public:
      //! constructor
      explicit MeshBounds(const std::array<double, 2 * Dim> & extent)
          : extent{extent} {};
      //! copy constructor
      MeshBounds(const MeshBounds & other) = default;
      //! assignment operator
      MeshBounds & operator=(const MeshBounds & other) = default;
      ~MeshBounds() = default;

      //! iterators over `` dereferences to mesh bound coordinate
      class iterator {
       public:
        using value_type = std::array<double, Dim>;  //!< stl conformance
        using const_value_type = const value_type;   //!< stl conformance
        using pointer = value_type *;                //!< stl conformance
        using iterator_category =
            std::forward_iterator_tag;  //!< stl conformance

        //! constructor
        explicit iterator(const MeshBounds & mesh_bounds, bool begin = true)
            : mesh_bounds{mesh_bounds}, index{begin ? 0 : mesh_bounds.size()} {}
        //! destructor
        ~iterator() {}
        //! dereferencing
        value_type operator*() const {
          std::array<double, Dim> retval{{0}};
          constexpr int size{2};
          for (size_t i{0}; i < Dim; ++i) {
            int idx = (this->index / ipow(size, i)) % size * Dim + i;
            retval[i] = this->mesh_bounds.extent[idx];
          }
          return retval;
        }
        //! pre-increment
        iterator & operator++() {
          this->index++;
          return *this;
        }
        //! inequality
        inline bool operator!=(const iterator & other) const {
          return this->index != other.index;
        }

       protected:
        const MeshBounds & mesh_bounds;  //!< ref to periodic images
        size_t index;                    //!< index of currect pointed-to voxel
      };
      //! stl conformance
      inline iterator begin() const { return iterator(*this); }
      //! stl conformance
      inline iterator end() const { return iterator(*this, false); }
      //! stl conformance
      inline size_t size() const { return ipow(2, Dim); }

     protected:
      const std::array<double, 2 * Dim>
          extent;  //!< repetitions in each dimension
    };

    /* ---------------------------------------------------------------------- */
    //! get dimension dependent neighbour indices (surrounding cell and the cell
    //! itself
    template <size_t Dim, class Container_t>
    std::vector<int> get_neighbours_atom_index(const int current_atom_index,
                                       const std::array<int, Dim> & ccoord,
                                       const Container_t & boxes) {
      std::vector<int> neighbours_atom_index{};
      for (auto && s : Stencil<Dim>{ccoord}) {
        for (const auto & neigh : boxes[s]) {
          // avoid adding the current i atom to the neighbour list
          if (neigh != current_atom_index) {
            neighbours_atom_index.push_back(neigh);
          }
        }
      }
      return neighbours_atom_index;
    }

    /* ---------------------------------------------------------------------- */
    //! get the cell index for a position
    template <class Vector_t>
    decltype(auto) get_box_index(const Vector_t & position, const double & rc) {
      auto constexpr dimension{Vector_t::SizeAtCompileTime};

      std::array<int, dimension> nidx{};
      for (auto dim{0}; dim < dimension; ++dim) {
        auto val = position(dim);
        nidx[dim] = static_cast<int>(std::floor(val / rc));
      }
      return nidx;
    }

    /* ---------------------------------------------------------------------- */
    //! get the linear index of a voxel in a given grid
    template <size_t Dim>
    constexpr Dim_t get_index(const std::array<int, Dim> & sizes,
                              const std::array<int, Dim> & ccoord) {
      Dim_t retval{0};
      Dim_t factor{1};
      for (Dim_t i = Dim - 1; i >= 0; --i) {
        retval += ccoord[i] * factor;
        if (i != 0) {
          factor *= sizes[i];
        }
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! test if position inside
    template <int Dim>
    bool position_in_bounds(const Eigen::Matrix<double, Dim, 1> & min,
                            const Eigen::Matrix<double, Dim, 1> & max,
                            const Eigen::Matrix<double, Dim, 1> & pos) {
      auto pos_lower = pos.array() - min.array();
      auto pos_greater = pos.array() - max.array();

      // check if shifted position inside maximum mesh positions
      auto f_lt = (pos_lower.array() > 0.).all();
      auto f_gt = (pos_greater.array() < 0.).all();

      if (f_lt and f_gt) {
        return true;
      } else {
        return false;
      }
    }

    /* ---------------------------------------------------------------------- */
    /**
     * storage for cell coordinates of atoms depending on the number of
     * dimensions
     */
    template <int Dim>
    class IndexContainer {
     public:
      //! Default constructor
      IndexContainer() = delete;

      //! Constructor with size
      explicit IndexContainer(const std::array<int, Dim> & nboxes)
          : nboxes{nboxes} {
        auto ntot = std::accumulate(nboxes.begin(), nboxes.end(), 1,
                                    std::multiplies<int>());
        data.resize(ntot);
      }

      //! Copy constructor
      IndexContainer(const IndexContainer & other) = delete;
      //! Move constructor
      IndexContainer(IndexContainer && other) = delete;
      //! Destructor
      ~IndexContainer() {}
      //! Copy assignment operator
      IndexContainer & operator=(const IndexContainer & other) = delete;
      //! Move assignment operator
      IndexContainer & operator=(IndexContainer && other) = default;
      //! brackets operator
      std::vector<int> & operator[](const std::array<int, Dim> & ccoord) {
        auto index = get_index(this->nboxes, ccoord);
        return data[index];
      }

      const std::vector<int> &
      operator[](const std::array<int, Dim> & ccoord) const {
        auto index = get_index(this->nboxes, ccoord);
        return this->data[index];
      }

     protected:
      //! a vector of atom indices for every box
      std::vector<std::vector<int>> data{};
      //! number of boxes in each dimension
      std::array<int, Dim> nboxes{};

     private:
    };
  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  /**
   * Adaptor that increases the MaxOrder of an existing StructureManager. This
   * means, if the manager does not have a neighbourlist, it is created, if it
   * exists, triplets, quadruplets, etc. lists are created.
   */
  template <class ManagerImplementation>
  class AdaptorNeighbourList
      : public StructureManager<AdaptorNeighbourList<ManagerImplementation>>,
        public std::enable_shared_from_this<
            AdaptorNeighbourList<ManagerImplementation>> {
   public:
    using Manager_t = AdaptorNeighbourList<ManagerImplementation>;
    using Parent = StructureManager<Manager_t>;
    using ImplementationPtr_t = std::shared_ptr<ManagerImplementation>;
    using traits = StructureManager_traits<AdaptorNeighbourList>;
    using AtomRef_t = typename ManagerImplementation::AtomRef_t;
    using Vector_ref = typename Parent::Vector_ref;
    using Vector_t = typename Parent::Vector_t;
    using Positions_ref =
        Eigen::Map<Eigen::Matrix<double, traits::Dim, Eigen::Dynamic>>;
    using Hypers_t = typename Parent::Hypers_t;
    // using AtomTypes_ref = AtomicStructure<traits::Dim>::AtomTypes_ref;

    static_assert(traits::MaxOrder == 2,
                  "ManagerImplementation needs an atom list "
                  " and can only build a neighbour list (pairs).");

    //! Default constructor
    AdaptorNeighbourList() = delete;

    /**
     * Constructs a full neighbourhood list from a given manager and cut-off
     * radius or extends an existing neighbourlist to the next order
     */
    AdaptorNeighbourList(ImplementationPtr_t manager, double cutoff,
                         bool consider_ghost_neighbours = false);

    AdaptorNeighbourList(ImplementationPtr_t manager, std::tuple<double> tp)
        : AdaptorNeighbourList(manager, std::get<0>(tp)) {}

    AdaptorNeighbourList(ImplementationPtr_t manager,
                         std::tuple<double, bool> tp)
        : AdaptorNeighbourList(manager, std::get<0>(tp), std::get<1>(tp)) {}

    AdaptorNeighbourList(ImplementationPtr_t manager,
                         const Hypers_t & adaptor_hypers)
        : AdaptorNeighbourList(
              manager, adaptor_hypers.at("cutoff").template get<double>(),
              optional_argument(adaptor_hypers)) {}

    //! Copy constructor
    AdaptorNeighbourList(const AdaptorNeighbourList & other) = delete;

    //! Move constructor
    AdaptorNeighbourList(AdaptorNeighbourList && other) = default;

    //! Destructor
    virtual ~AdaptorNeighbourList() = default;

    //! Copy assignment operator
    AdaptorNeighbourList &
    operator=(const AdaptorNeighbourList & other) = delete;

    //! Move assignment operator
    AdaptorNeighbourList & operator=(AdaptorNeighbourList && other) = default;

    bool optional_argument(const Hypers_t & adaptor_hypers) {
      bool consider_ghost_neighbours{false};
      if (adaptor_hypers.find("consider_ghost_neighbours") !=
          adaptor_hypers.end()) {
        consider_ghost_neighbours = adaptor_hypers["consider_ghost_neighbours"];
      }
      return consider_ghost_neighbours;
    }

    /**
     * Updates just the adaptor assuming the underlying manager was
     * updated. this function invokes building either the neighbour list or to
     * make triplets, quadruplets, etc. depending on the MaxOrder
     */
    void update_self();

    //! Updates the underlying manager as well as the adaptor
    template <class... Args>
    void update(Args &&... arguments);


    //! Returns cutoff radius of the neighbourhood manager
    inline double get_cutoff() const { return this->cutoff; }

    //! check whether neighbours of ghosts were considered
    inline bool get_consider_ghost_neighbours() const {
      return this->consider_ghost_neighbours;
    }

    /**
     * Returns the linear indices of the clusters (whose atom indices are stored
     * in counters). For example when counters is just the list of atoms, it
     * returns the index of each atom. If counters is a list of pairs of indices
     * (i.e. specifying pairs), for each pair of indices i,j it returns the
     * number entries in the list of pairs before i,j appears.
     */
    template <size_t Order>
    inline size_t
    get_offset_impl(const std::array<size_t, Order> & counters) const;

    //! Returns the number of clusters of size cluster_size
    inline size_t get_nb_clusters(size_t order) const {
      switch (order) {
      case 1: {
        //#BUG8486@(markus) ghosts only if underlying ghosts is true
        return this->get_size_with_ghosts();
        break;
      }
      case 2: {
        return this->neighbours_atom_index.size();
        break;
      }
      default:
        throw std::runtime_error("Can only handle single atoms and pairs.");
        break;
      }
    }

    //! Returns number of clusters of the original manager
    inline size_t get_size() const { return this->n_centers; }

    //! total number of atoms used for neighbour list, including ghosts
    inline size_t get_size_with_ghosts() const {
      // The return type of this function depends on the construction of the
      // adaptor. If the adaptor is constructed without considering the
      // neighbours of ghosts, only the number of atoms are returned to avoid
      // ambiguity and false access. In case the adaptor is constructed
      // considering the neighbours of ghosts, it is possible to iterate over
      // all atoms including ghosts using the proxy `.with_ghosts`, which takes
      // its `.end()` from here`
      auto nb_atoms{this->consider_ghost_neighbours ? this->n_centers + n_ghosts
                                                    : this->get_size()};
      return nb_atoms;
    }

    //! Returns position of an atom with index atom_index
    inline Vector_ref get_position(const size_t & atom_index) {
      if (atom_index < n_centers) {
        return this->manager->get_position(atom_index);
      } else {
        return this->get_ghost_position(atom_index - this->n_centers);
      }
    }

    //! ghost positions are only available for MaxOrder == 2
    inline Vector_ref get_ghost_position(const size_t & atom_index) {
      auto p = this->get_ghost_positions();
      auto * xval{p.col(atom_index).data()};
      return Vector_ref(xval);
    }

    inline Positions_ref get_ghost_positions() {
      return Positions_ref(this->ghost_positions.data(), traits::Dim,
                           this->ghost_positions.size() / traits::Dim);
    }

    //! ghost types are only available for MaxOrder=2
    inline const int & get_ghost_type(const size_t & atom_index) const {
      auto && p{this->get_ghost_types()};
      return p[atom_index];
    }

    //! ghost types are only available for MaxOrder=2
    inline int & get_ghost_type(const size_t & atom_index) {
      auto && p{this->get_ghost_types()};
      return p[atom_index];
    }

    //! provides access to the atomic types of ghost atoms
    inline std::vector<int> & get_ghost_types() { return this->ghost_types; }

    //! provides access to the atomic types of ghost atoms
    inline const std::vector<int> & get_ghost_types() const {
      return this->ghost_types;
    }

    //! Returns position of the given atom object (useful for users)
    inline Vector_ref get_position(const AtomRef_t & atom) {
      return this->manager->get_position(atom.get_index());
    }

    /**
     * Returns the id of the index-th (neighbour) atom of the cluster that is
     * the full structure/atoms object, i.e. simply the id of the index-th atom
     */
    inline int get_cluster_neighbour_atom_index_impl(const Parent &, size_t index) const {
      return this->manager->get_cluster_neighbour_atom_index_impl(*this->manager, index);
    }

    //! Returns the id of the index-th neighbour atom of a given cluster
    template <size_t Order, size_t Layer>
    inline int
    get_cluster_neighbour_atom_index_impl(
        const ClusterRefKey<Order, Layer> & cluster,
                          size_t index) const {
      static_assert(Order < traits::MaxOrder,
                    "this implementation only handles up to traits::MaxOrder");

      // necessary helper construct for static branching
      using IncreaseHelper_t =
          internal::IncreaseHelper<Order == (traits::MaxOrder - 1)>;

      if (Order < (traits::MaxOrder - 1)) {
        return IncreaseHelper_t::get_cluster_neighbour_atom_index(this->manager, cluster,
                                                       index);
      } else {
        auto && offset = this->offsets[cluster.get_cluster_index(Layer)];
        return this->neighbours_atom_index[offset + index];
      }
    }

    //! Returns atom type given an atom index, also works for ghost atoms
    inline int & get_atom_type(const int & atom_index) {
      return this->atom_types[atom_index];
    }

    /* If consider_ghost_neighbours=true and the atom index corresponds to an
     * ghost atom, then it returns it cluster index of the atom in the original
     * cell.
     */ 
    size_t get_cluster_index_impl(const int atom_index) const {
      return this->cluster_index_from_atom_indices[atom_index];
    }

    //! Returns the type of a given atom, given an AtomRef
    inline const int & get_atom_type(const int & atom_index) const {
      return this->atom_types[atom_index];
    }

    //! Returns the number of neighbors of a given cluster
    template <size_t Order, size_t Layer>
    inline size_t
    get_cluster_size_impl(const ClusterRefKey<Order, Layer> & cluster) const {
      static_assert(Order <= traits::MaxOrder,
                    "this implementation handles only the respective MaxOrder");

      auto && access_index = cluster.get_cluster_index(Layer);
      return nb_neigh[access_index];
    }

    //! Get the manager used to build the instance
    ImplementationPtr_t get_previous_manager() {
      return this->manager->get_shared_ptr();
    }

  protected:
    /* ---------------------------------------------------------------------- */
    /**
     * This function, including the storage of ghost atom positions is
     * necessary, because the underlying manager is not known at this
     * layer. Therefore we can not add positions to the existing array, but have
     * to add positions to a ghost array. This also means, that the get_position
     * function will need to branch, depending on the atom_index > n_centers and
     * offset with n_ghosts to access ghost positions.
     */

    /**
     * Function for adding existing i-atoms and ghost atoms additionally. This
     * is needed, because ghost atoms are also included in the buildup of the
     * pair list.
     */

    inline void add_ghost_atom(const int & atom_index,
                               const Vector_t & position,
                               const int & atom_type) {
      // first add it to the list of atoms
      this->atom_indices.push_back(atom_index);
      this->atom_types.push_back(atom_type);
      // add it to the ghost atom container
      this->ghost_atom_indices.push_back(atom_index);
      this->ghost_types.push_back(atom_type);
      for (auto dim{0}; dim < traits::Dim; ++dim) {
        this->ghost_positions.push_back(position(dim));
      }
      this->n_ghosts++;
    }

    //! Extends the list containing the number of neighbours with a 0
    inline void add_entry_number_of_neighbours() {
      this->nb_neigh.push_back(0);
    }

    //! Sets the correct offsets for accessing neighbours
    inline void set_offsets() {
      auto && n_tuples{nb_neigh.size()};
      if (n_tuples > 0) {
        this->offsets.reserve(n_tuples);
        this->offsets.resize(1);
        for (size_t i{0}; i < n_tuples - 1; ++i) {
          this->offsets.emplace_back(this->offsets[i] + this->nb_neigh[i]);
        }
      }
    }

    /* ---------------------------------------------------------------------- */
    //! full neighbour list with linked cell algorithm
    void make_full_neighbour_list();

    /* ---------------------------------------------------------------------- */
    //! pointer to underlying structure manager
    ImplementationPtr_t manager;

    //! Cutoff radius for neighbour list
    const double cutoff;

    //! stores i-atom and ghost atom indices
    std::vector<int> atom_indices{};

    std::vector<int> atom_types{};

    //! Stores additional atom indices of current Order (only ghost atoms)
    std::vector<int> ghost_atom_indices{};

    //! Stores the number of neighbours for every atom
    std::vector<size_t> nb_neigh{};

    //! Stores neighbour's atom index in a list in sequence of atoms
    std::vector<int> neighbours_atom_index{};
    
    /* Returns the atoms cluster index when accessing it with the atom's atomic 
     * index in a list in sequence of atoms.
     * List of atom indices which have a correpsonding cluster index of order 1.
     * if consider_ghost_neighbours is false ghost atoms will have a unique atom
     * index but no cluster index of order 1. For this case the cluster index
     * the atom in the cell at origin is used. 
     * */
    std::vector<size_t> cluster_index_from_atom_indices{};

    //! Stores the offset for each atom to accessing `neighbours`, this variable
    //! provides the entry point in the neighbour list, `nb_neigh` the number
    //! from the entry point
    std::vector<size_t> offsets{};

    size_t cluster_counter{0};

    //! number of i atoms, i.e. centers from underlying manager
    size_t n_centers;
    /**
     * number of ghost atoms (given by periodicity) filled during full
     * neighbourlist build
     */
    size_t n_ghosts;

    //! ghost atom positions
    std::vector<double> ghost_positions{};

    //! ghost atom type
    std::vector<int> ghost_types{};

    //! whether or not to consider neighbours of ghost atoms
    const bool consider_ghost_neighbours;

   private:
  };

  /* ---------------------------------------------------------------------- */
  //! Constructor of the pair list manager
  template <class ManagerImplementation>
  AdaptorNeighbourList<ManagerImplementation>::AdaptorNeighbourList(
      std::shared_ptr<ManagerImplementation> manager, double cutoff,
      bool consider_ghost_neighbours)
      : manager{std::move(manager)}, cutoff{cutoff}, atom_indices{},
        atom_types{}, ghost_atom_indices{}, nb_neigh{},
        neighbours_atom_index{}, offsets{},
        n_centers{0}, n_ghosts{0}, consider_ghost_neighbours{
                                       consider_ghost_neighbours} {
    static_assert(not(traits::MaxOrder < 1), "No atom list in manager");
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Update function that recursively pass its argument to the base
   * (Centers or Lammps). The base will then update the whole tree from the top.
   */
  template <class ManagerImplementation>
  template <class... Args>
  void
  AdaptorNeighbourList<ManagerImplementation>::update(Args &&... arguments) {
    this->manager->update(std::forward<Args>(arguments)...);
  }
  /* ---------------------------------------------------------------------- */
  /**
   * build a neighbour list based on atomic positions, types and indices, in the
   * following the needed data structures are initialized, after construction,
   * this function must be called to invoke the neighbour list algorithm
   */
  template <class ManagerImplementation>
  void AdaptorNeighbourList<ManagerImplementation>::update_self() {
    // set the number of centers
    this->n_centers = this->manager->get_size();
    this->n_ghosts = 0;
    //! Reset cluster_indices for adaptor to fill with sequence
    internal::for_each(this->cluster_indices_container,
                       internal::ResizePropertyToZero());

    // initialize necessary data structure
    this->atom_indices.clear();
    this->atom_types.clear();
    this->ghost_atom_indices.clear();
    this->nb_neigh.clear();
    this->neighbours_atom_index.clear();
    this->offsets.clear();
    this->ghost_positions.clear();
    this->ghost_types.clear();
    // actual call for building the neighbour list
    this->make_full_neighbour_list();
    this->set_offsets();

    // layering is started from the scratch, therefore all clusters and
    // centers+ghost atoms are in the right order.
    auto & atom_cluster_indices{std::get<0>(this->cluster_indices_container)};
    auto & pair_cluster_indices{std::get<1>(this->cluster_indices_container)};

    atom_cluster_indices.fill_sequence();
    pair_cluster_indices.fill_sequence();
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Builds full neighbour list. Triclinicity is accounted for. The general idea
   * is to anchor a mesh at the origin of the supplied cell (assuming it is at
   * the origin). Then the mesh is extended into space until it is as big as the
   * maximum cell coordinate plus one cutoff in each direction. This mesh has
   * boxes of size ``cutoff``. Depending on the periodicity of the mesh, ghost
   * atoms are added by shifting all i-atoms by the cell vectors corresponding
   * to the desired periodicity. All i-atoms and the ghost atoms are then sorted
   * into the respective boxes of the cartesian mesh and a stencil anchored at
   * the box of the i-atoms is used to build the atom neighbourhoods from the 9
   * (2d) or 27 (3d) boxes, which is checked for the neighbourhood. Correct
   * periodicity is ensured by the placement of the ghost atoms. The resulting
   * neighbourlist is full and not strict.
   */
  template <class ManagerImplementation>
  void AdaptorNeighbourList<ManagerImplementation>::make_full_neighbour_list() {
    using Vector_t = Eigen::Matrix<double, traits::Dim, 1>;

    // short hands for variable
    constexpr auto dim{traits::Dim};

    auto cell{this->manager->get_cell()};
    double cutoff{this->cutoff};

    std::array<int, dim> nboxes_per_dim{};

    // vector for storing the atom indices of each box
    std::vector<std::vector<int>> atoms_in_box{};

    // minimum/maximum coordinate of mesh for neighbour list, it is larger by
    // one cell to be able to provide a neighbour list also over ghost atoms;
    // depends on cell triclinicity and cutoff, coordinates of the mesh are
    // relative to the origin of the given cell. ghost_min is used for placing
    // ghost atoms within the given 'skin'.
    Vector_t mesh_min{Vector_t::Zero()};
    Vector_t mesh_max{Vector_t::Zero()};
    Vector_t ghost_min{Vector_t::Zero()};
    Vector_t ghost_max{Vector_t::Zero()};

    // max and min multipliers for number of cells in mesh per dimension in
    // units of cell vectors to be filled from max/min mesh positions and used
    // to construct ghost positions
    std::array<int, dim> m_min{};
    std::array<int, dim> m_max{};

    // Mesh related stuff for neighbour boxes. Calculate min and max of the mesh
    // in cartesian coordinates and relative to the cell origin.  mesh_min is
    // the origin of the mesh; mesh_max is the maximum coordinate of the mesh;
    // nboxes_per_dim is the number of mesh boxes in each dimension, not to be
    // confused with the number of cells to ensure periodicity
    for (auto i{0}; i < dim; ++i) {
      auto min_coord = std::min(0., cell.row(i).minCoeff());
      auto max_coord = std::max(0., cell.row(i).maxCoeff());

      // minimum is given by -cutoff and a delta to avoid ambiguity during cell
      // sorting of atom position e.g. at x = (0,0,0).
      auto epsilon = 0.25 * cutoff;
      // 2 cutoff for extra layer of emtpy cells (because of stencil iteration)
      mesh_min[i] = min_coord - 2 * cutoff - epsilon;

      // outer mesh, including one layer of emtpy cells
      auto lmesh = std::fabs(mesh_min[i]) + max_coord + 3 * cutoff;
      int n = std::ceil(lmesh / cutoff);
      auto lmax = n * cutoff - std::fabs(mesh_min[i]);
      mesh_max[i] = lmax;
      nboxes_per_dim[i] = n;

      // positions min/max for ghost atoms
      ghost_min[i] = mesh_min[i] + cutoff;
      auto lghost = lmesh - 2 * cutoff;
      int n_ghosts = std::ceil(lghost / cutoff);
      ghost_max[i] = n_ghosts * cutoff - std::fabs(ghost_min[i]);
    }

    // Periodicity related multipliers. Now the mesh coordinates are calculated
    // in units of cell vectors. m_min and m_max give the number of repetitions
    // of the cell in each cell vector direction
    constexpr int ncorners = internal::ipow(2, dim);
    Eigen::Matrix<double, dim, ncorners> xpos{};
    std::array<double, 2 * dim> mesh_bounds{};
    for (auto i{0}; i < dim; ++i) {
      mesh_bounds[i] = mesh_min[i];
      mesh_bounds[i + dim] = mesh_max[i];
    }

    // Get the mesh bounds to solve for the multiplicators
    int n{0};
    for (auto && coord : internal::MeshBounds<dim>{mesh_bounds}) {
      xpos.col(n) = Eigen::Map<Eigen::Matrix<double, dim, 1>>(coord.data());
      n++;
    }

    // solve inverse problem for all multipliers
    auto cell_inv{cell.inverse().eval()};
    auto multiplicator{cell_inv * xpos.eval()};
    auto xmin = multiplicator.rowwise().minCoeff();
    auto xmax = multiplicator.rowwise().maxCoeff();

    // find max and min multipliers for cell vectors
    for (auto i{0}; i < dim; ++i) {
      m_min[i] = std::floor(xmin(i));
      m_max[i] = std::ceil(xmax(i));
    }

    std::array<int, dim> periodic_max{};
    std::array<int, dim> periodic_min{};
    std::array<int, dim> repetitions{};
    auto periodicity = this->manager->get_periodic_boundary_conditions();
    size_t ntot{1};

    // calculate number of actual repetitions of cell, depending on periodicity
    for (auto i{0}; i < dim; ++i) {
      if (periodicity[i]) {
        periodic_max[i] = m_max[i];
        periodic_min[i] = m_min[i];
      } else {
        periodic_max[i] = 0;
        periodic_min[i] = 0;
      }
      auto nrep_in_dim = -periodic_min[i] + periodic_max[i] + 1;
      repetitions[i] = nrep_in_dim;
      ntot *= nrep_in_dim;
    }

    // before generating ghost atoms, all existing atoms are added to the list
    // of current atoms to start the full list of current i-atoms and ghosts
    // This is done before the ghost atom generation, to have them all
    // contiguously at the beginning of the list.
    int ntot_atoms{0};
    for (auto atom : *this->manager) {
      int atom_index = atom.get_atom_index();
      size_t cluster_index = this->manager->get_cluster_index(atom_index);
      auto atom_type = atom.get_atom_type();      
      this->atom_indices.push_back(atom_index);
      this->atom_types.push_back(atom_type);
      ntot_atoms++;
      this->cluster_index_from_atom_indices.push_back(cluster_index);
    }
    // generate ghost atom indices and positions
    for (auto atom : this->get_manager().with_ghosts()) {
      auto pos = atom.get_position();
      auto atom_type = atom.get_atom_type();

      for (auto && p_image :
           internal::PeriodicImages<dim>{periodic_min, repetitions, ntot}) {
        int ncheck{0};
        for (auto i{0}; i < dim; ++i) {
          ncheck += std::abs(p_image[i]);
        }
        // exclude cell itself
        if (ncheck > 0) {
          Vector_t pos_ghost{pos};

          for (auto i{0}; i < dim; ++i) {
            pos_ghost += cell.col(i) * p_image[i];
          }

          auto flag_inside =
              internal::position_in_bounds(ghost_min, ghost_max, pos_ghost);

          if (flag_inside) {
            // next atom index is size, since start is at index = 0
            auto new_atom_index = this->n_centers + this->n_ghosts;
            // get_size_with_ghosts();
            this->add_ghost_atom(new_atom_index, pos_ghost, atom_type);
            ntot_atoms++;

            // adds origin atom cluster_index if true
            // adds ghost atom cluster index if false
            size_t cluster_index = 
                  this->manager->get_cluster_index(atom.get_atom_index());
            this->cluster_index_from_atom_indices.push_back(cluster_index);
          }
        }
      }
    }

    // neighbour boxes
    internal::IndexContainer<dim> atom_id_cell{nboxes_per_dim};

    // sorting i-atoms into boxes
    auto nb_atoms_tot{this->n_centers + this->n_ghosts};
    for (size_t i{0}; i < nb_atoms_tot; ++i) {
      Vector_t pos = this->get_position(i);
      Vector_t dpos = pos - mesh_min;
      auto idx = internal::get_box_index(dpos, cutoff);
      atom_id_cell[idx].push_back(i);
    }

    // go through all atoms and/or ghosts to build neighbour list, depending on
    // the runtime decision flag
    //auto nb_atoms{this->consider_ghost_neighbours
    //                  ? this->n_centers + this->n_ghosts
    //                  : this->get_size()};

    // #BUG8486@(markus) I changed it so it could work with other 
    // ManagerImplementations that does not make the atomic indices
    // as StructurManagerCenters 
    for (auto atom : this->get_manager().with_ghosts()) {
      int atom_index = atom.get_atom_index();
      int nneigh{0};
      Vector_t pos = this->get_position(atom_index);
      Vector_t dpos = pos - mesh_min;
      auto box_index = internal::get_box_index(dpos, cutoff);
      auto && current_j_atoms =
          internal::get_neighbours_atom_index(atom_index, box_index, atom_id_cell);

      for (auto j_atom_index : current_j_atoms) {
        this->neighbours_atom_index.push_back(j_atom_index);
        nneigh++;
      }
      this->nb_neigh.push_back(nneigh);
    }
  }
 

  /* ---------------------------------------------------------------------- */
  /**
   * Returns the linear indices of the clusters (whose atom indices
   * are stored in counters). For example when counters is just the list
   * of atoms, it returns the index of each atom. If counters is a list of pairs
   * of indices (i.e. specifying pairs), for each pair of indices i,j it returns
   * the number entries in the list of pairs before i,j appears.
   */
  template <class ManagerImplementation>
  template <size_t Order>
  inline size_t AdaptorNeighbourList<ManagerImplementation>::get_offset_impl(
      const std::array<size_t, Order> & counters) const {
    // The static assert with <= is necessary, because the template parameter
    // ``Order`` is one Order higher than the MaxOrder at the current
    // level. The return type of this function is used to build the next Order
    // iteration.
    static_assert(Order <= traits::MaxOrder,
                  "this implementation handles only up to the respective"
                  " MaxOrder");
    return this->offsets[counters.front()];
  }

}  // namespace rascal

#endif  // SRC_STRUCTURE_MANAGERS_ADAPTOR_NEIGHBOUR_LIST_HH_
