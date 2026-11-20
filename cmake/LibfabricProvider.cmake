# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricProvider
-----------------

Register transport and hook object libraries or loadable modules.

libfabric_add_provider(NAME name SOURCES ... [HOOK] [LINUX_ONLY]
[X86_64_ONLY] [HEADERS ...] [INCLUDE_DIRS ...] [COMPILE_DEFINITIONS ...]
[COMPILE_OPTIONS ...] [LINK_LIBRARIES ...] [DEPENDS package ...]) creates
a private provider target, adds built-in objects to fabric, or installs a
plugin. Requires fabric, libfabric_common, libfabric_hooks, and
libfabric_configure_target(). Sets HAVE_<NAME> in the caller's scope on success.
HOOK honors LIBFABRIC_HOOK_<NAME>; DEPENDS checks <package>_FOUND.

Globally guarded: declares functions; evaluates no per-directory build state.
]]
include_guard(GLOBAL)

# Each provider owns an object target so its private includes, ISA flags, and
# feature macros do not leak into the core or into other providers.
function(libfabric_add_provider)
  cmake_parse_arguments(
    PARSE_ARGV
    0
    P
    "LINUX_ONLY;X86_64_ONLY;HOOK"
    "NAME"
    "SOURCES;HEADERS;INCLUDE_DIRS;COMPILE_DEFINITIONS;COMPILE_OPTIONS;LINK_LIBRARIES;DEPENDS"
  )
  if(P_UNPARSED_ARGUMENTS
     OR NOT P_NAME
     OR NOT P_SOURCES
  )
    message(FATAL_ERROR "Invalid provider definition: ${ARGN}")
  endif()
  string(TOUPPER "${P_NAME}" _name)
  if(P_HOOK AND NOT LIBFABRIC_HOOK_${_name})
    return()
  endif()
  foreach(_platform LINUX X86_64)
    if(P_${_platform}_ONLY AND NOT LIBFABRIC_${_platform})
      message(STATUS "Provider ${P_NAME}: requires ${_platform}")
      return()
    endif()
  endforeach()
  foreach(_dependency IN LISTS P_DEPENDS)
    if(NOT ${_dependency}_FOUND)
      message(STATUS "Provider ${P_NAME}: missing ${_dependency}")
      return()
    endif()
  endforeach()

  set(_target libfabric_provider_${P_NAME})
  if(LIBFABRIC_${_name}_DLOPEN)
    add_library(
      ${_target} MODULE ${P_SOURCES} ${P_HEADERS}
                        $<TARGET_OBJECTS:libfabric_common>
    )
    if(P_HOOK)
      target_sources(${_target} PRIVATE $<TARGET_OBJECTS:libfabric_hooks>)
    endif()
    target_link_libraries(${_target} PRIVATE fabric ${LIBFABRIC_CORE_LIBRARIES})
    set_target_properties(
      ${_target}
      PROPERTIES
        OUTPUT_NAME "${P_NAME}-fi"
        PREFIX "lib"
        LIBRARY_OUTPUT_DIRECTORY "${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
        LIBFABRIC_PLUGIN TRUE
    )
    if(LIBFABRIC_LINUX AND NOT LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS MATCHES
                           "fsanitize="
    )
      target_link_options(${_target} PRIVATE "LINKER:--no-undefined")
    endif()
    if(LIBFABRIC_INSTALL)
      install(TARGETS ${_target}
              LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/libfabric"
      )
    endif()
  else()
    add_library(${_target} OBJECT ${P_SOURCES} ${P_HEADERS})
    target_sources(fabric PRIVATE $<TARGET_OBJECTS:${_target}>)
    target_link_libraries(fabric PRIVATE ${P_LINK_LIBRARIES})
  endif()
  libfabric_configure_target(${_target})
  target_include_directories(${_target} PRIVATE ${P_INCLUDE_DIRS})
  if(P_HOOK AND LIBFABRIC_HOOK_PERF)
    target_include_directories(
      ${_target} PRIVATE "${libfabric_SOURCE_DIR}/prov/hook/perf/include"
    )
  endif()
  target_compile_definitions(${_target} PRIVATE ${P_COMPILE_DEFINITIONS})
  target_compile_options(${_target} PRIVATE ${P_COMPILE_OPTIONS})
  target_link_libraries(${_target} PRIVATE ${P_LINK_LIBRARIES})
  set(HAVE_${_name}
      TRUE
      PARENT_SCOPE
  )
endfunction()
