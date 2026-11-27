# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricPackageDependencies
----------------------------

Describe installed CMake and pkg-config dependency identities.

Requires the final fabric LINK_LIBRARIES property. Sets LIBFABRIC_DEPENDENCIES,
LIBFABRIC_FIND_DEPENDENCIES, LIBFABRIC_PC_REQUIRES_PRIVATE, and LIBS_PRIVATE.
The local registration helper consumes explicit target/package mappings;
provider-plugin dependencies are not exported as core dependencies.

libfabric_package_dependency(package target [FIND_ARGS ...] [PC ...]
[LIBRARIES ...]) records one linked target. FIND_ARGS are forwarded to the
installed find_dependency call, PC names dependency-owned pkg-config modules,
and LIBRARIES lists explicit link requirements for SDKs without .pc metadata.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()

# Describe package identities explicitly: a target namespace is not a package
# name, and CMake target interfaces cannot be translated into pkg-config flags.
# Only dependencies linked into fabric belong to its installed interfaces;
# dependencies of provider plugins remain private to those plugins.
get_target_property(_libfabric_links fabric LINK_LIBRARIES)
set(LIBFABRIC_DEPENDENCIES "")
set(LIBFABRIC_FIND_DEPENDENCIES "")
set(_libfabric_pc_requires "")
set(_libfabric_pc_libraries "")

function(libfabric_package_dependency package dependency_target)
  if(NOT "${dependency_target}" IN_LIST _libfabric_links)
    return()
  endif()
  cmake_parse_arguments(
    PARSE_ARGV
    2
    P
    ""
    ""
    "FIND_ARGS;PC;LIBRARIES"
  )
  if(P_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Invalid package metadata: ${ARGN}")
  endif()
  list(REMOVE_ITEM _libfabric_links "${dependency_target}")
  set(_libfabric_links
      "${_libfabric_links}"
      PARENT_SCOPE
  )
  if(NOT "${package}" IN_LIST LIBFABRIC_DEPENDENCIES)
    list(APPEND LIBFABRIC_DEPENDENCIES "${package}")
    set(LIBFABRIC_DEPENDENCIES
        "${LIBFABRIC_DEPENDENCIES}"
        PARENT_SCOPE
    )
    list(JOIN P_FIND_ARGS " " _find_args)
    string(APPEND LIBFABRIC_FIND_DEPENDENCIES
           "    find_dependency(${package} ${_find_args})\n"
    )
    set(LIBFABRIC_FIND_DEPENDENCIES
        "${LIBFABRIC_FIND_DEPENDENCIES}"
        PARENT_SCOPE
    )
  endif()
  list(APPEND _libfabric_pc_requires ${P_PC})
  set(_libfabric_pc_requires
      "${_libfabric_pc_requires}"
      PARENT_SCOPE
  )
  list(APPEND _libfabric_pc_libraries ${P_LIBRARIES})
  set(_libfabric_pc_libraries
      "${_libfabric_pc_libraries}"
      PARENT_SCOPE
  )
endfunction()

# Delegate transitive flags to the dependencies' own .pc files. SDKs without
# pkg-config metadata use their explicitly listed runtime libraries instead.
libfabric_package_dependency(
  Threads Threads::Threads LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
)
libfabric_package_dependency(LibUring LibUring::LibUring PC liburing)
libfabric_package_dependency(LTTng LTTng::LTTng PC lttng-ust)
libfabric_package_dependency(XPMEM XPMEM::XPMEM LIBRARIES XPMEM::XPMEM)
libfabric_package_dependency(IBVerbs IBVerbs::IBVerbs PC libibverbs)
libfabric_package_dependency(RdmaCM RdmaCM::RdmaCM PC librdmacm)
libfabric_package_dependency(Efadv Efadv::Efadv PC libefa)
libfabric_package_dependency(Numa Numa::Numa PC numa)
libfabric_package_dependency(UUID UUID::UUID PC uuid)
libfabric_package_dependency(Hwloc Hwloc::Hwloc PC hwloc)
libfabric_package_dependency(PSM2 PSM2::PSM2 LIBRARIES PSM2::PSM2)
libfabric_package_dependency(UCX UCX::UCX PC ucx)
libfabric_package_dependency(LibCxi LibCxi::LibCxi PC libcxi)
libfabric_package_dependency(
  CXIDriverHeaders CXIDriverHeaders::CXIDriverHeaders
)
libfabric_package_dependency(CURL CURL::libcurl PC libcurl)
libfabric_package_dependency(JsonC JsonC::JsonC PC json-c)
if(LibNL_IS_V3)
  libfabric_package_dependency(LibNL LibNL::LibNL PC libnl-3.0 libnl-route-3.0)
else()
  libfabric_package_dependency(LibNL LibNL::LibNL LIBRARIES ${LibNL_LIBRARIES})
endif()
foreach(_cuda cudart cuda_driver nvml)
  libfabric_package_dependency(
    CUDAToolkit CUDA::${_cuda} LIBRARIES CUDA::${_cuda}
  )
endforeach()
libfabric_package_dependency(
  GDRCopy GDRCopy::GDRCopy LIBRARIES GDRCopy::GDRCopy
)
libfabric_package_dependency(LevelZero LevelZero::LevelZero PC libze_loader)
if(TARGET hsa-runtime64::hsa-runtime64)
  libfabric_package_dependency(
    ROCm ROCm::ROCm LIBRARIES hsa-runtime64::hsa-runtime64
  )
else()
  libfabric_package_dependency(ROCm ROCm::ROCm LIBRARIES ROCm::ROCm)
endif()
list(REMOVE_DUPLICATES LIBFABRIC_DEPENDENCIES)
list(REMOVE_DUPLICATES _libfabric_pc_requires)
list(JOIN _libfabric_pc_requires " " LIBFABRIC_PC_REQUIRES_PRIVATE)

# Remaining entries are plain system libraries or private build requirements.
foreach(_link IN LISTS _libfabric_links)
  if(_link MATCHES "^\\$<BUILD_INTERFACE:|^::@")
    continue()
  elseif(TARGET "${_link}" OR _link MATCHES "::|\\$<")
    message(FATAL_ERROR "Missing installed package metadata for ${_link}")
  endif()
  list(APPEND _libfabric_pc_libraries "${_link}")
endforeach()
set(_libfabric_pc_flags "")
foreach(_library IN LISTS _libfabric_pc_libraries)
  if(TARGET "${_library}")
    # Let CMake select the correct imported configuration at generation time.
    list(APPEND _libfabric_pc_flags "\"$<TARGET_LINKER_FILE:${_library}>\"")
  elseif(IS_ABSOLUTE "${_library}")
    list(APPEND _libfabric_pc_flags "\"${_library}\"")
  elseif(_library MATCHES "^-")
    list(APPEND _libfabric_pc_flags "${_library}")
  else()
    list(APPEND _libfabric_pc_flags "-l${_library}")
  endif()
endforeach()
list(JOIN _libfabric_pc_flags " " LIBS_PRIVATE)
