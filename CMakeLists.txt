# =============================================================================
# file   CMakeLists.txt
#
# @author Alexander Goscinski <alexander.goscinski@epfl.ch>
# @author Till Junge <till.junge@epfl.ch>
# @author Felix Musil <felix.musil@epfl.ch>
# @author Markus Stricker <markus.stricker@epfl.ch>
#
# @date   17 Jan 2019
#
# @brief  Main configuration file
#
# @section LICENSE
#
# Copyright © 2018 Till Junge, Felix Musil, Markus Stricker, COSMO (EPFL), LAMMM
# (EPFL)
#
# Rascal is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# Rascal is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Rascal; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
# =============================================================================

cmake_minimum_required(VERSION 3.0)

# https://cmake.org/pipermail/cmake/2008-September/023808.html
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their
# definition
# and dump it in the cache along with proper documentation, otherwise set
# CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
ENDIF()
# Set the possible values of build type for cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
"Release" "Debug" "RelWithDebInfo" "MinSizeRel")

message(STATUS "Build type is: ${CMAKE_BUILD_TYPE}")

project(rascal CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(rascalTools)

########## MAIN VARIABLE TO CHANGE THE BUILD AND INSTALL OF THE LIBRARY ########
option(BUILD_BINDINGS "Build the python bindings" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(BUILD_PROFILING "Build the files for profiling" OFF)
option(BUILD_TESTS "Build the unit tests" OFF)
option(BUILD_DOC "Build documentation" OFF)
option(BUILD_SANDBOX "If on, builds the sandbox" OFF)

set(INSTALL_PATH "" CACHE STRING "Path to install the libraries")

#SET(TYPE_ARCHITECTURE "native" CACHE STRING
#    "Choose the type of architecture to compile to in release mode."
#)

########## COMPILATION FLAGS ##########
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++ -Wno-non-virtual-dtor")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")

    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

    # Add -march=${TYPE_ARCHITECTURE} (-march=native by default) to the release
    # build flags
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=${TYPE_ARCHITECTURE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=${TYPE_ARCHITECTURE}")

    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -march=${TYPE_ARCHITECTURE}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=${TYPE_ARCHITECTURE}")
endif()

########## START CONFIGURATION ##########

# set the name of the c++ libraries
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR
    CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(LIBRASCAL_NAME "rascal-dbg")
  set(WIGXJPF_NAME "wigxjpf-dbg")
else()
  set(LIBRASCAL_NAME "rascal")
  set(WIGXJPF_NAME "wigxjpfo")
endif()

add_external_package(wigxjpf VERSION 1.9 CONFIG)
add_external_package(Eigen3 VERSION 3.3.4 CONFIG)

if(BUILD_BENCHMARKS)
  find_package(Threads REQUIRED)
  add_external_package(benchmark VERSION 1.5.0 CONFIG)
endif()

# set up bindings
if(BUILD_BINDINGS)
  find_package(PythonLibsNew 3 MODULE REQUIRED)

  # warning about potential side effects of the installation of the python library
  if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR
      CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(NOT PYTHON_IS_DEBUG)
      message(WARNING "The current configuration will build debug libraries "
              "not linked to a debug python interpreter. "
              "Installation target might overwrite the python bindings "
              "library with a debug build !"
              "see https://wiki.ubuntu.com/PyDbgBuilds or "
              "https://gist.github.com/bombs-kim/c9848c2b09962f2fd753b48b6d2cd87ffor sources and "
              "https://jml.io/2015/08/debugging-python-with-gdb.html for more informations "
              " on how to debug c++ code from python.")
    endif()
  endif()

  add_external_package(pybind11 VERSION 2.3.0 CONFIG)


  execute_process ( COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import \
                    get_python_lib; print(get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process ( COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; \
                    print(sys.prefix)"
    OUTPUT_VARIABLE PACKAGE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

# figure out where to install the library
if(NOT "${INSTALL_PATH}" STREQUAL "")
  set(CMAKE_INSTALL_PREFIX "${INSTALL_PATH}")
elseif(NOT BUILD_BINDINGS)
  set(CMAKE_INSTALL_PREFIX "/usr")
  message(STATUS "INSTALLATION REQUIRES SUDO RIGHTS")
elseif(BUILD_BINDINGS)
  set(CMAKE_INSTALL_PREFIX "${PACKAGE_PREFIX}")
endif()

message(STATUS "Installation ROOT: ${CMAKE_INSTALL_PREFIX}")

if(SKBUILD)
  # make sure we know when installing with setup.py (potentially for using with
  # pip)
  message(STATUS "The project is built using scikit-build")
endif()

add_subdirectory(src)

enable_testing()
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(performance/benchmarks)
endif()

if(BUILD_PROFILING)
    add_subdirectory(performance/profiles)
endif()

if(BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()

if (${BUILD_SANDBOX})
    add_subdirectory(sandbox)
endif (${BUILD_SANDBOX})

if(BUILD_DOC)
    add_subdirectory(docs)
endif()

# target to run linter for all configured sources
set(LINT_TARGET lint)
add_custom_target(${LINT_TARGET})

# Try to find optional cpp linter
include(cpplint)
# Checking hh/cc files for compliance with coding conventions
if(CPPLINT_FOUND)
  # register the folders to check for cpp code conventions compliance
  cpplint_add_subdirectory("${PROJECT_SOURCE_DIR}/bindings" "")
  cpplint_add_subdirectory("${PROJECT_SOURCE_DIR}/examples"
    "--filter=-build/namespaces")
  cpplint_add_subdirectory("${PROJECT_SOURCE_DIR}/performance/benchmarks" "")
  cpplint_add_subdirectory("${PROJECT_SOURCE_DIR}/performance/profiles" "")
  cpplint_add_subdirectory("${PROJECT_SOURCE_DIR}/src" "")
  cpplint_add_subdirectory("${PROJECT_SOURCE_DIR}/tests" "")
  add_dependencies(${LINT_TARGET} ${CPPLINT_TARGET})
endif()

# Check for clang-format
include(clangformat)
# Add all hh/cc files for autoformatting with clang-format
if(CLANG_FORMAT_FOUND)
  # register the folders to apply clang-format with given configuration
  # .clang-format in project root folder
  clang_format_add_subdirectory("${PROJECT_SOURCE_DIR}/bindings" "")
  clang_format_add_subdirectory("${PROJECT_SOURCE_DIR}/examples")
  clang_format_add_subdirectory("${PROJECT_SOURCE_DIR}/performance/benchmarks" "")
  clang_format_add_subdirectory("${PROJECT_SOURCE_DIR}/performance/profiles" "")
  clang_format_add_subdirectory("${PROJECT_SOURCE_DIR}/src" "")
  clang_format_add_subdirectory("${PROJECT_SOURCE_DIR}/tests" "")
endif()

# Check for black
include(black)
# Add python files for autoformatting with black
if(BLACK_FOUND)
  # register the folders to apply black
  black_add_subdirectory("${PROJECT_SOURCE_DIR}/bindings")
  black_add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
  black_add_subdirectory("${PROJECT_SOURCE_DIR}/performance")
  black_add_subdirectory("${PROJECT_SOURCE_DIR}/examples")
  black_add_subdirectory("${PROJECT_SOURCE_DIR}/scripts")
  add_dependencies(${LINT_TARGET} ${BLACK_LINT_TARGET})
  endif()

# Removes every file in the build folder except the external libraries to
# prevent redownloading external libraries
add_custom_target(
    deepclean
    COMMAND "${PYTHON_EXECUTABLE}" ${PROJECT_SOURCE_DIR}/scripts/developer_utils.py deepclean
  )
