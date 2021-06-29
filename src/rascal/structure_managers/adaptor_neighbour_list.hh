/**
 * @file   rascal/structure_managers/adaptor_neighbour_list.hh
 *
 * @author Markus Stricker <markus.stricker@epfl.ch>
 * @author Till Junge <till.junge@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   04 Oct 2018
 *
 * @brief implements an adaptor for structure_managers, which
 * creates a full or half neighbourlist if there is none
 *
 * Copyright  2018 Markus Stricker, Till Junge, Felix Musil COSMO (EPFL),
 *  LAMMM (EPFL)
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_ADAPTOR_NEIGHBOUR_LIST_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_ADAPTOR_NEIGHBOUR_LIST_HH_

#include "rascal/structure_managers/atomic_structure.hh"
#include "rascal/structure_managers/lattice.hh"
#include "rascal/structure_managers/property.hh"
#include "rascal/structure_managers/structure_manager.hh"
#include "rascal/utils/basic_types.hh"
#include "rascal/utils/utils.hh"

#include <set>
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
    using parent_traits = StructureManager_traits<ManagerImplementation>;
    constexpr static AdaptorTraits::Strict Strict{AdaptorTraits::Strict::no};
    constexpr static bool HasDistances{false};
    constexpr static bool HasDirectionVectors{false};
    constexpr static int Dim{parent_traits::Dim};
    constexpr static bool HasCenterPair{parent_traits::HasCenterPair};
    constexpr static int StackLevel{parent_traits::StackLevel + 1};
    // New MaxOrder upon construction, by construction should be 2
    constexpr static size_t MaxOrder{parent_traits::MaxOrder + 1};
    // When using periodic boundary conditions, it is possible that atoms are
    // added upon construction of the neighbour list. Therefore the layering
    // sequence is reset: here is layer 0 again.
    using LayerByOrder = std::index_sequence<0, 0>;
    using PreviousManager_t = ManagerImplementation;
    constexpr static AdaptorTraits::NeighbourListType NeighbourListType{
        AdaptorTraits::NeighbourListType::full};
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
      explicit Stencil(const std::array<int, Dim> & origin) : origin{origin} {}
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
        bool operator!=(const iterator & other) const {
          return this->index != other.index;
        }

       protected:
        //! ref to stencils
        const Stencil & stencil;
        //! index of currect pointed-to voxel
        size_t index;
      };
      //! stl conformance
      iterator begin() const { return iterator(*this); }
      //! stl conformance
      iterator end() const { return iterator(*this, false); }
      //! stl conformance
      size_t size() const { return ipow(3, Dim); }

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
                     const std::array<int, Dim> & nrepetitions, size_t ntot)
          : origin{origin}, nrepetitions{nrepetitions}, ntot{ntot} {}
      //! copy constructor
      PeriodicImages(const PeriodicImages & other) = default;
      //! assignment operator
      PeriodicImages & operator=(const PeriodicImages & other) = default;
      ~PeriodicImages() = default;

      //! iterators over `` dereferences to cell coordinates
      class iterator {
       public:
        using value_type = Eigen::Vector3i;
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
          value_type retval{0, 0, 0};
          int factor{1};
          for (int i{Dim - 1}; i >= 0; --i) {
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
        bool operator!=(const iterator & other) const {
          return this->index != other.index;
        }

       protected:
        const PeriodicImages & periodic_images;  //!< ref to periodic images
        size_t index;  //!< index of currect pointed-to pixel
      };
      //! stl conformance
      iterator begin() const { return iterator(*this); }
      //! stl conformance
      iterator end() const { return iterator(*this, false); }
      //! stl conformance
      size_t size() const { return this->ntot; }

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
          : extent{extent} {}
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
        bool operator!=(const iterator & other) const {
          return this->index != other.index;
        }

       protected:
        const MeshBounds & mesh_bounds;  //!< ref to periodic images
        size_t index;                    //!< index of currect pointed-to voxel
      };
      //! stl conformance
      iterator begin() const { return iterator(*this); }
      //! stl conformance
      iterator end() const { return iterator(*this, false); }
      //! stl conformance
      size_t size() const { return ipow(2, Dim); }

     protected:
      const std::array<double, 2 * Dim>
          extent;  //!< repetitions in each dimension
    };

    /* ---------------------------------------------------------------------- */
    //! get dimension dependent neighbour indices (surrounding cell and the cell
    //! itself
    template <size_t Dim, class Container_t>
    void fill_neighbours_atom_tag(const int current_atom_tag,
                                  const std::array<int, Dim> & ccoord,
                                  const Container_t & boxes,
                                  std::vector<int> & neighbours_atom_tag) {
      neighbours_atom_tag.clear();
      for (auto && s : Stencil<Dim>{ccoord}) {
        for (const auto & neigh : boxes[s]) {
          // avoid adding the current i atom to the neighbour list
          if (neigh != current_atom_tag) {
            neighbours_atom_tag.push_back(neigh);
          }
        }
      }
    }

    /* ---------------------------------------------------------------------- */
    //! get the cell index for a position
    template <class Vector_t>
    std::array<int, Vector_t::SizeAtCompileTime>
    get_box_index(const Vector_t & position, double rc) {
      auto constexpr dimension{Vector_t::SizeAtCompileTime};

      std::array<int, dimension> nidx{};
      for (auto dim{0}; dim < dimension; ++dim) {
        auto val = position(dim);
        nidx[dim] = std::max(1, static_cast<int>(std::floor(val / rc)));
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
                            const Eigen::Matrix<double, Dim, 1> & pos,
                            const double epsilon = 1e-4) {
      auto pos_lower = pos.array() - min.array();
      auto pos_greater = pos.array() - max.array();

      // check if shifted position inside maximum mesh positions
      auto f_lt = (pos_lower.array() > epsilon).all();
      auto f_gt = (pos_greater.array() < -epsilon).all();

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
        for (int i = 0; i < static_cast<int>(Dim); ++i) {
          // we make sure not to bin atoms in the outer layer of boxes needed
          // by the stencil
          if (ccoord[i] >= this->nboxes[i] - 1 or (ccoord[i] <= 0)) {  // NOLINT
            std::stringstream error{};
            error << "Error: this atom does not fall in one of the linked cell "
                  << "bin. It might have not been wrapped properly. ccoord = ("
                  << ccoord[0] << ", " << ccoord[1] << ", " << ccoord[2]
                  << "), nboxes = (" << nboxes[0] << ", " << nboxes[1] << ", "
                  << nboxes[2] << ")";
            std::cout << error.str() << std::endl;
            throw std::runtime_error(error.str());
          }
        }
        auto index = get_index(this->nboxes, ccoord);
        return data[index];
      }

      const std::vector<int> &
      operator[](const std::array<int, Dim> & ccoord) const {
        auto index = get_index(this->nboxes, ccoord);
        return this->data[index];
      }

     protected:
      //! a vector of atom tags for every box
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
    using ManagerImplementation_t = ManagerImplementation;
    using ImplementationPtr_t = std::shared_ptr<ManagerImplementation>;
    using ConstImplementationPtr_t =
        const std::shared_ptr<const ManagerImplementation>;
    using traits = StructureManager_traits<AdaptorNeighbourList>;
    using PreviousManager_t = typename traits::PreviousManager_t;
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
                         double skin = 0.);

    AdaptorNeighbourList(ImplementationPtr_t manager,
                         const Hypers_t & adaptor_hypers)
        : AdaptorNeighbourList(
              manager, adaptor_hypers.at("cutoff").template get<double>(),
              optional_argument_skin(adaptor_hypers)) {}

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

    double optional_argument_skin(const Hypers_t & adaptor_hypers) {
      double skin{0.};
      if (adaptor_hypers.find("skin") != adaptor_hypers.end()) {
        skin = adaptor_hypers["skin"];
      }
      return skin;
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
    double get_cutoff() const { return this->cutoff; }

    /**
     * Returns the linear indices of the clusters (whose atom tags are stored
     * in counters). For example when counters is just the list of atoms, it
     * returns the index of each atom. If counters is a list of pairs of indices
     * (i.e. specifying pairs), for each pair of indices i,j it returns the
     * number entries in the list of pairs before i,j appears.
     */
    template <size_t Order>
    size_t get_offset_impl(const std::array<size_t, Order> & counters) const;

    //! Returns the number of clusters of size cluster_size
    size_t get_nb_clusters(size_t order) const {
      switch (order) {
      /**
       * Note: The case for order=1 is abmiguous: one possible answer is the
       * number of centers the other possibility is the number of centers +
       * ghost atoms. Please use the get_size or get_size_with_ghosts member
       * functions
       */
      case 2: {
        return this->neighbours_atom_tag.size();
        break;
      }
      default:
        throw std::runtime_error("Can only handle pairs.");
        break;
      }
    }

    //! Returns number of clusters of the original manager
    size_t get_size() const { return this->n_centers; }

    //! total number of atoms used for neighbour list, including ghosts
    size_t get_size_with_ghosts() const {
      return this->n_centers + this->n_ghosts;
    }

    //! Returns position of an atom with index atom_tag
    Vector_ref get_position(size_t atom_tag) {
      if (atom_tag < this->n_centers) {
        return this->manager->get_position(atom_tag);
      } else {
        return this->get_ghost_position(atom_tag - this->n_centers);
      }
    }

    //! ghost positions are only available for MaxOrder == 2
    Vector_ref get_ghost_position(const size_t ghost_atom_index) {
      auto p = this->get_ghost_positions();
      auto * xval{p.col(ghost_atom_index).data()};
      return Vector_ref(xval);
    }

    Positions_ref get_ghost_positions() {
      return Positions_ref(this->ghost_positions.data(), traits::Dim,
                           this->ghost_positions.size() / traits::Dim);
    }

    //! ghost types are only available for MaxOrder=2
    int get_ghost_type(size_t atom_tag) const {
      auto && p{this->get_ghost_types()};
      return p[atom_tag];
    }

    //! ghost types are only available for MaxOrder=2
    int & get_ghost_type(size_t atom_tag) {
      auto && p{this->get_ghost_types()};
      return p[atom_tag];
    }

    //! provides access to the atomic types of ghost atoms
    std::vector<int> & get_ghost_types() { return this->ghost_types; }

    //! provides access to the atomic types of ghost atoms
    const std::vector<int> & get_ghost_types() const {
      return this->ghost_types;
    }

    //! Returns position of the given atom object (useful for users)
    Vector_ref get_position(const AtomRef_t & atom) {
      return this->manager->get_position(atom.get_index());
    }

    /**
     * Returns the id of the index-th (neighbour) atom of the cluster that is
     * the full structure/atoms object, i.e. simply the id of the index-th atom
     *
     * This is called when ClusterRefKey<1, Layer> so we refer to a center
     * atoms. this function does the same job as get_atom_tag would do.
     */
    int get_neighbour_atom_tag(const Parent &, size_t iteration_index) const {
      return this->atom_tag_list[iteration_index];
    }

    //! Returns the id of the index-th neighbour atom of a given cluster
    template <size_t Order, size_t Layer>
    int get_neighbour_atom_tag(const ClusterRefKey<Order, Layer> & cluster,
                               size_t iteration_index) const {
      static_assert(Order < traits::MaxOrder,
                    "this implementation only handles up to traits::MaxOrder");

      // necessary helper construct for static branching
      using IncreaseHelper_t =
          internal::IncreaseHelper<Order == (traits::MaxOrder - 1)>;

      if (Order < (traits::MaxOrder - 1)) {
        return IncreaseHelper_t::get_neighbour_atom_tag(this->manager, cluster,
                                                        iteration_index);
      } else {
        auto && offset = this->offsets[cluster.get_cluster_index(Layer)];
        return this->neighbours_atom_tag[offset + iteration_index];
      }
    }

    //! Returns atom type given an atom tag, also works for ghost atoms
    int get_atom_type(int atom_tag) const { return this->atom_types[atom_tag]; }

    /** The atom tag corresponds to an ghost atom, then it returns it cluster
     * index of the atom in the original cell.
     */
    size_t get_atom_index(const int atom_tag) const {
      return this->atom_index_from_atom_tag_list[atom_tag];
    }

    //! Returns the number of neighbours of a given atom at a given TargetOrder
    //! Returns the number of pairs of a given center
    template <size_t TargetOrder, size_t Order, size_t Layer>
    typename std::enable_if_t<TargetOrder == 2, size_t>
    get_cluster_size_impl(const ClusterRefKey<Order, Layer> & cluster) const {
      constexpr auto nb_neigh_layer{
          get_layer<TargetOrder>(typename traits::LayerByOrder{})};
      auto && access_index = cluster.get_cluster_index(nb_neigh_layer);
      return nb_neigh[access_index];
    }

    //! Get the manager used to build the instance
    ImplementationPtr_t get_previous_manager_impl() {
      return this->manager->get_shared_ptr();
    }

    //! Get the manager used to build the instance
    ConstImplementationPtr_t get_previous_manager_impl() const {
      return this->manager->get_shared_ptr();
    }

    size_t get_n_update() const { return this->n_update; }

    size_t get_skin2() const { return this->skin2; }

   protected:
    /* ---------------------------------------------------------------------- */
    /**
     * This function, including the storage of ghost atom positions is
     * necessary, because the underlying manager is not known at this
     * layer. Therefore we can not add positions to the existing array, but have
     * to add positions to a ghost array. This also means, that the get_position
     * function will need to branch, depending on the atom_tag > n_centers and
     * offset with n_ghosts to access ghost positions.
     */

    /**
     * Function for adding existing i-atoms and ghost atoms additionally. This
     * is needed, because ghost atoms are also included in the buildup of the
     * pair list.
     */
    void add_ghost_atom(int atom_tag, const Vector_t & position,
                        int atom_type) {
      // first add it to the list of atoms
      this->atom_tag_list.push_back(atom_tag);
      this->atom_types.push_back(atom_type);
      // add it to the ghost atom container
      this->ghost_atom_tag_list.push_back(atom_tag);
      this->ghost_types.push_back(atom_type);
      for (auto dim{0}; dim < traits::Dim; ++dim) {
        this->ghost_positions.push_back(position(dim));
      }
      this->n_ghosts++;
    }

    //! Extends the list containing the number of neighbours with a 0
    void add_entry_number_of_neighbours() { this->nb_neigh.push_back(0); }

    //! Sets the correct offsets for accessing neighbours
    void set_offsets() {
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
    /**
     * If no atom has moved more than the skin-distance since the
     * last call to the update method, then the linked cell neighbor list can
     * be reused. This will save some expensive rebuilds of the list, but
     * extra neighbors outside the cutoff will be considered.
     *
     * use squared skin to avoid computing the sqrt of the squared norm
     * between the two .
     */
    const double skin2;

    //! stores i-atom and ghost atom tags
    std::vector<int> atom_tag_list{};

    std::vector<int> atom_types{};

    //! Stores additional atom tags of current Order (only ghost atoms)
    std::vector<int> ghost_atom_tag_list{};

    //! Stores the number of neighbours for every atom
    std::vector<size_t> nb_neigh{};

    //! Stores neighbour's atom tag in a list in sequence of atoms
    std::vector<int> neighbours_atom_tag{};

    /**
     * Returns the atoms cluster index when accessing it with the atom's atomic
     * index in a list in sequence of atoms.  List of atom tags which have a
     * correpsonding cluster index of order 1.  If ghost atoms have been added
     * they have their own new index.
     */
    std::vector<size_t> atom_index_from_atom_tag_list{};

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

    //! counts the number of time the neighbour list has been updated
    size_t n_update{0};

    /**
     * on top of the main update signal, the skin parameter allow to skip
     * the update. So this variable records this possiblity.
     */
    bool need_update{true};

    //! ghost atom positions
    std::vector<double> ghost_positions{};

    //! ghost atom type
    std::vector<int> ghost_types{};

   private:
  };

  /* ---------------------------------------------------------------------- */
  //! Constructor of the pair list manager
  template <class ManagerImplementation>
  AdaptorNeighbourList<ManagerImplementation>::AdaptorNeighbourList(
      std::shared_ptr<ManagerImplementation> manager, double cutoff,
      double skin)
      : manager{std::move(manager)}, cutoff{cutoff}, skin2{skin * skin},
        atom_tag_list{}, atom_types{}, ghost_atom_tag_list{}, nb_neigh{},
        neighbours_atom_tag{}, offsets{}, n_centers{0}, n_ghosts{0} {
    static_assert(not(traits::MaxOrder < 1), "No atom list in manager");
    if (this->skin2 > 0.) {
      throw std::runtime_error(
          "The verlet list is not functional for the moment, keep skin == 0");
    }
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
    if (sizeof...(arguments) > 0) {
      // TODO(felix) should not have to assume that the underlying manager is
      // manager centers.
      auto && atomic_structure{this->manager->get_atomic_structure()};
      // if the structure has not changed by more than skin**2
      if (not atomic_structure.is_similar(std::forward<Args>(arguments)...,
                                          this->skin2)) {
        this->need_update = true;
      } else {
        this->need_update = false;
      }
    }
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
    if (this->need_update) {
      // set the number of centers
      this->n_centers = this->manager->get_size();
      // this->n_atoms = this->manager->get_n_atoms();
      this->n_ghosts = 0;  // this->manager->get_size_with_ghosts();
      //! Reset cluster_indices for adaptor to fill with sequence
      internal::for_each(this->cluster_indices_container,
                         internal::ResizePropertyToZero());

      // initialize necessary data structure
      this->atom_tag_list.clear();
      this->atom_types.clear();
      this->ghost_atom_tag_list.clear();
      this->nb_neigh.clear();
      this->neighbours_atom_tag.clear();
      this->offsets.clear();
      this->ghost_positions.clear();
      this->ghost_types.clear();
      this->atom_index_from_atom_tag_list.clear();
      // actual call for building the neighbour list
      this->make_full_neighbour_list();
      this->set_offsets();

      // layering is started from the scratch, therefore all clusters and
      // centers+ghost atoms are in the right order.
      auto & atom_cluster_indices{std::get<0>(this->cluster_indices_container)};
      auto & pair_cluster_indices{std::get<1>(this->cluster_indices_container)};

      atom_cluster_indices.fill_sequence();
      pair_cluster_indices.fill_sequence();
      ++this->n_update;
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Build a neighbor list using a linked cell algorithm for finite cutoff
   * interaction computations of lenght \f$r_c\f$. There is no restriction
   * regarding the type of lattice and periodic boundary conditions (triclinic
   * lattices and mixed periodicity are handled by design).
   *
   * To do so we build a cubic box that contains the unit cell and one
   * \f$r_c\f$ in each directions. The length of the box is a multiple of
   * \f$r_c\f$ and it is partitioned into cubic bins of size \f$r_c\f$. The
   * resulting mesh encompass the unit cell and its surrounding up to \f$2
   * r_c\f$ in each directions so that atoms can be binned in it. The
   * additional layer of bins is kept empty so that the connectivity assignment
   * criteria of the linked cell, i.e. atoms belonging to neighbor bins are
   * neighbors, can be applied uniformly (latter referred are stencil).
   * note(felix): the mesh going up to \f$2 r_c\f$ is probably not really
   *  necessary for the connectivity assignment and \f$r_c\f$ would work too.
   *
   * Atoms are assumed to be inside the unit cell (otherwise it will throw an
   * error) and they are binned. Then depending on the periodicity of the
   * system, the periodic images or ghost atoms that fall within the bounds of
   * the mesh are also binned.
   * Then each binned atoms is assigned its neighbor depending on the bin's
   * connectivity criteria (in 3d the 27 nearest bins).
   */
  template <class ManagerImplementation>
  void AdaptorNeighbourList<ManagerImplementation>::make_full_neighbour_list() {
    using Vector_t = Eigen::Matrix<double, traits::Dim, 1>;

    // short hands for parameters and inputs
    constexpr auto dim{traits::Dim};
    const auto & cell{this->manager->get_cell()};
    const double & cutoff{this->cutoff};

    // minimum/maximum coordinate of mesh for neighbour list, it is larger by
    // one cell to be able to provide a neighbour list also over ghost atoms;
    // depends on cell triclinicity and cutoff, coordinates of the mesh are
    // relative to the origin of the given cell.
    // ghost_min/max is used for placing ghost atoms within the given 'skin'.
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
    std::array<int, dim> nboxes_per_dim{};
    // vector for storing the tags of atoms contained in each box
    std::vector<std::vector<int>> atoms_in_box{};
    for (auto i{0}; i < dim; ++i) {
      // min and max coordinates of cell corners in Cartesian space
      double min_coord{0.0};
      double max_coord{0.0};
      for (int col{0}; col < cell.cols(); ++col) {
        min_coord += cell(i, col) * static_cast<double>(cell(i, col) < 0.);
        max_coord += cell(i, col) * static_cast<double>(cell(i, col) > 0.);
      }
      // 2 cutoff for extra layer of emtpy cells (because of stencil iteration)
      mesh_min[i] = min_coord - 2. * cutoff;

      // outer mesh, including one layer of emtpy cells in each direction
      double lmesh{max_coord - min_coord + 4. * cutoff};
      // number of Linked Cell in each directions
      double n{std::ceil(lmesh / cutoff)};
      mesh_max[i] = mesh_min[i] + n * cutoff;
      nboxes_per_dim[i] = static_cast<int>(n);

      // positions min/max for ghost atoms -> this is the actual bounding box
      ghost_min[i] = min_coord - cutoff;
      double lghost{max_coord - min_coord + 2 * cutoff};
      double n_ghost_boxes{std::ceil(lghost / cutoff)};
      ghost_max[i] = n_ghost_boxes * cutoff + ghost_min[i];
    }

    // numerical tolerance when determining if a ghost atom falls into the
    // the range that should be considered for binning
    double max_box_lenght{(ghost_max - ghost_min).maxCoeff()};
    double bound_tol{max_box_lenght * 1e-8};

    // Periodicity related multipliers. Now the mesh coordinates are calculated
    // in units of cell vectors. m_min and m_max give the number of repetitions
    // of the cell in each cell vector direction
    //! we should be probably projecting in scaled coords ghost_min/max
    constexpr int ncorners = internal::ipow(2, dim);
    Eigen::Matrix<double, dim, ncorners> xpos{};
    std::array<double, 2 * dim> mesh_bounds{};
    for (auto i{0}; i < dim; ++i) {
      mesh_bounds[i] = ghost_min[i] - bound_tol;
      mesh_bounds[i + dim] = ghost_max[i] + bound_tol;
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
    auto xmin = multiplicator.rowwise().minCoeff().eval();
    auto xmax = multiplicator.rowwise().maxCoeff().eval();

    // find max and min multipliers for cell vectors
    for (auto i{0}; i < dim; ++i) {
      m_min[i] = std::floor(xmin(i));
      // remove 1 because the original cell is already included
      m_max[i] = std::ceil(xmax(i)) - 1;
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

    // Before generating periodic replicas atoms (also termed ghost atoms), all
    // existing center atoms are added to the list of current atoms to start the
    // full list of current i-atoms to have them all contiguously at the
    // beginning of the list.
    for (size_t atom_tag{0}; atom_tag < this->manager->get_size(); ++atom_tag) {
      auto atom_type = this->manager->get_atom_type(atom_tag);
      auto atom_index = this->manager->get_atom_index(atom_tag);
      this->atom_tag_list.push_back(atom_tag);
      this->atom_types.push_back(atom_type);
      this->atom_index_from_atom_tag_list.push_back(atom_index);
    }

    // And before generating periodic replicas (termed ghost atoms), previous
    // ghost atoms are added to the list of ghost atoms with their associated
    // data.
    for (size_t atom_tag{this->manager->get_size()};
         atom_tag < this->manager->get_size_with_ghosts(); ++atom_tag) {
      auto pos = this->manager->get_position(atom_tag);
      auto atom_type = this->manager->get_atom_type(atom_tag);
      auto new_atom_tag{this->n_centers + this->n_ghosts};
      this->add_ghost_atom(new_atom_tag, pos, atom_type);
      size_t atom_index = this->manager->get_atom_index(atom_tag);
      this->atom_index_from_atom_tag_list.push_back(atom_index);
    }

    // generate ghost atom tags and positions
    for (size_t atom_tag{0}; atom_tag < this->manager->get_size_with_ghosts();
         ++atom_tag) {
      auto pos = this->manager->get_position(atom_tag);
      auto atom_type = this->manager->get_atom_type(atom_tag);

      for (auto && p_image :
           internal::PeriodicImages<dim>{periodic_min, repetitions, ntot}) {
        // exclude the original unit cell
        //! assumption: this assumes atoms were inside the cell initially
        if (not(p_image.array() == 0).all()) {
          Vector_t pos_ghost{pos + cell * p_image.template cast<double>()};
          auto flag_inside = internal::position_in_bounds(ghost_min, ghost_max,
                                                          pos_ghost, bound_tol);

          if (flag_inside) {
            // next atom tag is size, since start is at index = 0
            auto new_atom_tag{this->n_centers + this->n_ghosts};
            this->add_ghost_atom(new_atom_tag, pos_ghost, atom_type);
            // adds origin atom cluster_index if true
            // adds ghost atom cluster index if false
            size_t atom_index = this->manager->get_atom_index(atom_tag);
            this->atom_index_from_atom_tag_list.push_back(atom_index);
          }
        }
      }
    }

    // neighbour boxes
    internal::IndexContainer<dim> atom_id_cell{nboxes_per_dim};

    // sorting the atoms and ghosts inside the cell into boxes
    auto n_potential_neighbours{this->n_centers + this->n_ghosts};
    for (size_t atom_tag{0}; atom_tag < n_potential_neighbours; ++atom_tag) {
      auto pos = this->get_position(atom_tag);
      Vector_t dpos = pos - mesh_min;
      auto idx = internal::get_box_index(dpos, cutoff);
      atom_id_cell[idx].push_back(atom_tag);
    }

    // go through all atoms and/or ghosts to build neighbour list, depending on
    // the runtime decision flag
    std::vector<int> current_j_atoms{};
    for (auto center : this->get_manager()) {
      int atom_tag = center.get_atom_tag();
      int nneigh{0};

      Vector_t pos = center.get_position();
      Vector_t dpos = pos - mesh_min;
      auto box_index = internal::get_box_index(dpos, cutoff);
      internal::fill_neighbours_atom_tag(atom_tag, box_index, atom_id_cell,
                                         current_j_atoms);

      nneigh += current_j_atoms.size();
      for (auto & j_atom_tag : current_j_atoms) {
        this->neighbours_atom_tag.push_back(j_atom_tag);
      }

      this->nb_neigh.push_back(nneigh);
    }

    /**
     * All the ghost atom neighbours have to be added explicitly as zero. This
     * is done after adding the neighbours of centers because ghost atoms are
     * listed after the center atoms in the respective data
     * structures. Technically ghost atoms can not have any neighbour, i.e. not
     * even '0'. It should be _nothing_. But that is not possible with our data
     * structure.
     */
    int nneigh{0};
    for (auto && dummy : this->get_manager().only_ghosts()) {
      std::ignore = dummy;
      this->nb_neigh.push_back(nneigh);
    }
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Returns the linear indices of the clusters (whose atom tags
   * are stored in counters). For example when counters is just the list
   * of atoms, it returns the index of each atom. If counters is a list of pairs
   * of indices (i.e. specifying pairs), for each pair of indices i,j it returns
   * the number entries in the list of pairs before i,j appears.
   */
  template <class ManagerImplementation>
  template <size_t Order>
  size_t AdaptorNeighbourList<ManagerImplementation>::get_offset_impl(
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

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_ADAPTOR_NEIGHBOUR_LIST_HH_
