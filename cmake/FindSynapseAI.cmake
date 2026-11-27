# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindSynapseAI
-------------

Find SynapseAI headers for loading the runtime with dlopen.

Imported targets: ``SynapseAI::SynapseAI``.

Results: ``SynapseAI_FOUND``, ``SynapseAI_INCLUDE_DIRS``.

Hints: ``SynapseAI_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``SynapseAI_INCLUDE_DIR``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_path(
  SynapseAI_INCLUDE_DIR
  NAMES habanalabs/synapse_api.h
  HINTS ${SynapseAI_ROOT} ENV SynapseAI_ROOT
  PATH_SUFFIXES include
)

find_package_handle_standard_args(SynapseAI REQUIRED_VARS SynapseAI_INCLUDE_DIR)

if(SynapseAI_FOUND)
  set(SynapseAI_INCLUDE_DIRS "${SynapseAI_INCLUDE_DIR}")

  if(NOT TARGET SynapseAI::SynapseAI)
    add_library(SynapseAI::SynapseAI INTERFACE IMPORTED)
    set_target_properties(
      SynapseAI::SynapseAI PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                      "${SynapseAI_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(SynapseAI_INCLUDE_DIR)
