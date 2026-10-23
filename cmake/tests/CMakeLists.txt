# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# These integration tests execute built programs and configure host consumers.
if(CMAKE_CROSSCOMPILING)
  return()
endif()
if(TARGET fi_info)
  add_test(NAME cmake.info COMMAND fi_info --version)
  foreach(_hook IN LISTS LIBFABRIC_HOOKS)
    string(TOUPPER "${_hook}" _name)
    if(HAVE_${_name}_DL)
      string(REGEX REPLACE "^hook_" "" _provider "${_hook}")
      add_test(NAME cmake.hook.${_hook} COMMAND fi_info -l)
      set_tests_properties(
        cmake.hook.${_hook}
        PROPERTIES
          ENVIRONMENT
          "FI_PROVIDER_PATH=${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
          PASS_REGULAR_EXPRESSION "ofi_hook_${_provider}:"
      )
    endif()
  endforeach()
  if(HAVE_TCP)
    add_test(NAME cmake.tcp COMMAND fi_info -p tcp)
    set_tests_properties(
      cmake.tcp
      PROPERTIES ENVIRONMENT
                 "FI_PROVIDER_PATH=${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
    )
  endif()
endif()
if(LIBFABRIC_INSTALL)
  # Preserve explicit SDK roots for test consumers without embedding them in the
  # installed package. Some find modules (notably CUDA) need these hints.
  set(_dependency_hints "")
  foreach(_package IN LISTS LIBFABRIC_DEPENDENCIES)
    if(DEFINED ${_package}_ROOT)
      string(
        APPEND _dependency_hints
        "set(${_package}_ROOT [==[${${_package}_ROOT}]==] CACHE PATH \"\")\n"
      )
    endif()
  endforeach()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependency-hints.cmake"
       "${_dependency_hints}"
  )
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    find_program(LIBFABRIC_TEST_CXX_COMPILER NAMES clang++ REQUIRED)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    find_program(LIBFABRIC_TEST_CXX_COMPILER NAMES g++ REQUIRED)
  else()
    set(LIBFABRIC_TEST_CXX_COMPILER "${CMAKE_C_COMPILER}")
  endif()
  mark_as_advanced(LIBFABRIC_TEST_CXX_COMPILER)
  add_test(
    NAME cmake.install
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE_DIR=${libfabric_SOURCE_DIR}"
      "-DBINARY_DIR=${libfabric_BINARY_DIR}" "-DGENERATOR=${CMAKE_GENERATOR}"
      "-DCONFIG=$<CONFIG>" "-DSHARED=${BUILD_SHARED_LIBS}"
      "-DC_COMPILER=${CMAKE_C_COMPILER}"
      "-DCXX_COMPILER=${LIBFABRIC_TEST_CXX_COMPILER}"
      "-DFABTESTS=${LIBFABRIC_BUILD_FABTESTS}"
      "-DBINDIR=${CMAKE_INSTALL_BINDIR}" "-DDATADIR=${CMAKE_INSTALL_DATADIR}"
      "-DMANDIR=${CMAKE_INSTALL_MANDIR}"
      "-DDEPENDENCY_PREFIX_PATH=${CMAKE_PREFIX_PATH}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/install.cmake"
  )
  set_tests_properties(cmake.install PROPERTIES TIMEOUT 120)
endif()
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND AND UNIX)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    add_test(
      NAME cmake.find-modules
      COMMAND
        "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/find-modules.py"
        "${CMAKE_COMMAND}" "${CMAKE_GENERATOR}" "${libfabric_SOURCE_DIR}"
        "${libfabric_BINARY_DIR}"
    )
    set_tests_properties(cmake.find-modules PROPERTIES TIMEOUT 120)
  endif()
endif()
if(Python3_FOUND
   AND TARGET fi_pingpong
   AND HAVE_TCP
)
  add_test(
    NAME cmake.loopback
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/loopback.py"
            "$<TARGET_FILE:fi_pingpong>"
  )
  set_tests_properties(
    cmake.loopback
    PROPERTIES TIMEOUT 45 ENVIRONMENT
               "FI_PROVIDER_PATH=${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
  )
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_test(
    NAME cmake.efadv
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE_DIR=${libfabric_SOURCE_DIR}"
      "-DBINARY_DIR=${libfabric_BINARY_DIR}" "-DGENERATOR=${CMAKE_GENERATOR}"
      "-DC_COMPILER=${CMAKE_C_COMPILER}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/efadv.cmake"
  )
  set_tests_properties(cmake.efadv PROPERTIES TIMEOUT 120)
endif()
if(LIBFABRIC_LINUX AND (LIBFABRIC_X86_64 OR LIBFABRIC_RISCV64))
  add_test(
    NAME cmake.opx
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE_DIR=${libfabric_SOURCE_DIR}"
      "-DBINARY_DIR=${libfabric_BINARY_DIR}" "-DGENERATOR=${CMAKE_GENERATOR}"
      "-DC_COMPILER=${CMAKE_C_COMPILER}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/opx.cmake"
  )
  set_tests_properties(cmake.opx PROPERTIES TIMEOUT 120)
endif()
add_test(
  NAME cmake.configure
  COMMAND
    "${CMAKE_COMMAND}" "-DSOURCE_DIR=${libfabric_SOURCE_DIR}"
    "-DBINARY_DIR=${libfabric_BINARY_DIR}" "-DGENERATOR=${CMAKE_GENERATOR}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/configure.cmake"
)
set_tests_properties(cmake.configure PROPERTIES TIMEOUT 120)
