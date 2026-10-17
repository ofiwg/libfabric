# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)
project(find_module_consumer LANGUAGES C)
list(PREPEND CMAKE_MODULE_PATH "${LIBFABRIC_SOURCE}/cmake")
# The fixture SDKs are host-side Unix layouts. A platform SDK's uuid headers or
# framework search must not take precedence over these controlled inputs.
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_FRAMEWORK NEVER)

# These fixtures exercise discovery and generation, without requiring the SDKs
# or linking their placeholder archives.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.c"
     "int main(void) { return 0; }\n"
)
add_executable(consumer "${CMAKE_CURRENT_BINARY_DIR}/main.c")
foreach(
  _package
  Hwloc
  IBVerbs
  RdmaCM
  Efadv
  UUID
  LTTng
  LibUring
  JsonC
)
  set(${_package}_ROOT "${SDK_ROOT}")
  find_package(${_package} ${VERSION_ARGS} QUIET)
  if(EXPECT_MISSING)
    if(${_package}_FOUND)
      message(FATAL_ERROR "${_package} accepted an unavailable version")
    endif()
  else()
    if(NOT ${_package}_FOUND)
      message(FATAL_ERROR "${_package} did not find the fixture SDK")
    endif()
    if(NOT "${SDK_ROOT}/include" IN_LIST ${_package}_INCLUDE_DIRS)
      message(
        FATAL_ERROR
          "${_package} ignored ${SDK_ROOT}: found headers at ${${_package}_INCLUDE_DIRS}"
      )
    endif()
    target_link_libraries(consumer PRIVATE ${_package}::${_package})
    # A find module must not be guarded: later requests in the same configure
    # may ask for a different version, then retry a valid one.
    find_package(${_package} 999 EXACT QUIET)
    if(${_package}_FOUND)
      message(FATAL_ERROR "${_package} retained an earlier discovery result")
    endif()
    find_package(${_package} ${VERSION_ARGS} REQUIRED)
  endif()
endforeach()

# Runtime-loaded CUDA must accept the same environment root as FindCUDAToolkit,
# without needing nvcc or SDK libraries. Other accelerators stay disabled.
set(ENV{CUDAToolkit_ROOT} "${SDK_ROOT}")
set(LIBFABRIC_ENABLE_CUDA ON)
set(LIBFABRIC_CUDA_DLOPEN ON)
include(LibfabricHmemDependencies)
foreach(_header CUDA CUDART NVML)
  if(NOT LIBFABRIC_${_header}_INCLUDE_DIR STREQUAL "${SDK_ROOT}/include")
    message(FATAL_ERROR "${_header} ignored the CUDA environment root")
  endif()
endforeach()

# Header-only discovery must not prevent a later caller from requesting a
# runtime target. Disable the vendor config to exercise ROCr's fallback too.
set(CMAKE_DISABLE_FIND_PACKAGE_hsa-runtime64 TRUE)
foreach(_package GDRCopy LevelZero ROCm)
  set(${_package}_ROOT "${SDK_ROOT}")
  find_package(${_package} REQUIRED COMPONENTS Headers)
  if(TARGET ${_package}::${_package})
    message(FATAL_ERROR "${_package} discovered a runtime for Headers only")
  endif()
  find_package(${_package} REQUIRED COMPONENTS Runtime)
  if(NOT TARGET ${_package}::${_package})
    message(FATAL_ERROR "${_package} skipped the later Runtime request")
  endif()
  target_link_libraries(consumer PRIVATE ${_package}::${_package})
endforeach()
