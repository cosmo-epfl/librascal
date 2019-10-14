/**
 * @file   bind_include.hh
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   31 Oct 2018
 *
 * @brief  File to centralize includes and function declaration
 *
 * Copyright  2018  Felix Musil, COSMO (EPFL), LAMMM (EPFL)
 *
 * Rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * Rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef BINDINGS_BIND_INCLUDE_HH_
#define BINDINGS_BIND_INCLUDE_HH_

#include "atomic_structure.hh"
#include "rascal_utility.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
// for the hasattr function to test the module namespace
#include <pybind11/pytypes.h>

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>

/*
 * Prevent vector of atomic structures from being copied into a Python list,
 * since we already have the AtomsList object.  See also
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
 */
PYBIND11_MAKE_OPAQUE(std::vector<rascal::AtomicStructure<3>>);

namespace py = pybind11;

namespace rascal {

  namespace internal {
    /**
     * Transforms the template type to a string for the python bindings.
     * There are submodules in the python bindings with the class
     * title so to avoid redundancy they are removed from the
     * typename.
     * @tparam T type that should be stringified
     * @returns std::string name of the type
     */
    template <typename T>
    std::string GetBindingTypeName() {
      static std::map<std::string, std::string> replacement_map = {
          {"StructureManager", ""}, {"Adaptor", ""}, {"Calculator", ""}};

      std::string name = type_name<T>();
      for (const auto & map : replacement_map) {
        replace(name, map.first, map.second);
      }

      return name;
    }
  }  // namespace internal
}  // namespace rascal

#endif  // BINDINGS_BIND_INCLUDE_HH_
