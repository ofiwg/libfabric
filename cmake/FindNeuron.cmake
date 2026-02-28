# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  Neuron_INCLUDE_DIR
  NAMES nrt/nrt.h
  HINTS ${Neuron_ROOT} ENV Neuron_ROOT
  PATH_SUFFIXES include
)

find_package_handle_standard_args(Neuron REQUIRED_VARS Neuron_INCLUDE_DIR)

if(Neuron_FOUND)
  set(Neuron_INCLUDE_DIRS "${Neuron_INCLUDE_DIR}")

  if(NOT TARGET Neuron::Neuron)
    add_library(Neuron::Neuron INTERFACE IMPORTED)
    set_target_properties(
      Neuron::Neuron PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${Neuron_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(Neuron_INCLUDE_DIR)
