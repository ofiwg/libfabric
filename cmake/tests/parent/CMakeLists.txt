# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)
project(libfabric_parent LANGUAGES C)
# Simulate temporary state left by an unrelated parent-project compiler check.
# None of these settings belongs to libfabric's feature probes.
set(CMAKE_REQUIRED_LIBRARIES missing_parent_probe_library)
set(CMAKE_REQUIRED_FLAGS "-Dmain=libfabric_parent_probe_main")
set(BUILD_SHARED_LIBS OFF)
set(LIBFABRIC_INSTALL
    OFF
    CACHE BOOL "" FORCE
)
set(LIBFABRIC_BUILD_TESTING
    OFF
    CACHE BOOL "" FORCE
)
set(LIBFABRIC_BUILD_UTILITIES
    OFF
    CACHE BOOL "" FORCE
)
add_subdirectory("${LIBFABRIC_SOURCE}" libfabric)
if(NOT CMAKE_REQUIRED_LIBRARIES STREQUAL "missing_parent_probe_library"
   OR NOT CMAKE_REQUIRED_FLAGS STREQUAL "-Dmain=libfabric_parent_probe_main"
)
  message(FATAL_ERROR "libfabric changed the parent project's probe settings")
endif()
add_executable(parent ../consumer/main.c)
target_link_libraries(parent PRIVATE Libfabric::fabric)
enable_testing()
add_test(NAME parent.consume COMMAND parent)
