# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindNeuron
----------

Find AWS Neuron runtime headers for loading the runtime with dlopen.

Imported targets: ``Neuron::Neuron``.

Results: ``Neuron_FOUND``, ``Neuron_INCLUDE_DIRS``.

Hints: ``Neuron_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``Neuron_INCLUDE_DIR``

Unguarded: each find_package call must reevaluate its request and scope.
]]
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
