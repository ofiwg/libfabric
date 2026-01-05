# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CMakeParseArguments)

# Global tracking lists for provider registration
set_property(GLOBAL PROPERTY LIBFABRIC_PROVIDERS "")
set_property(GLOBAL PROPERTY LIBFABRIC_STATIC_PROVIDERS "")
set_property(GLOBAL PROPERTY LIBFABRIC_DL_PROVIDERS "")
set_property(GLOBAL PROPERTY LIBFABRIC_PROVIDER_LINK_LIBRARIES "")
set_property(GLOBAL PROPERTY LIBFABRIC_PROVIDER_OBJECT_LIBS "")

# -----------------------------------------------------------------------------
# libfabric_add_provider( NAME <name> SOURCES <source files...> [HEADERS <header
# files...>] [INCLUDE_DIRS <directories...>] [COMPILE_DEFINITIONS
# <definitions...>] [LINK_LIBRARIES <libraries...>] [DEPENDS <find package
# dependencies...>] [CONDITION <boolean expression>] [LINUX_ONLY] [FREEBSD_ONLY]
# [MACOS_ONLY] [X86_64_ONLY] [AARCH64_ONLY] )
#
# Registers a provider with the build system. The provider option
# LIBFABRIC_PROVIDER_<NAME> must already exist.
# -----------------------------------------------------------------------------
function(libfabric_add_provider)
  cmake_parse_arguments(
    PROV
    "LINUX_ONLY;FREEBSD_ONLY;MACOS_ONLY;X86_64_ONLY;AARCH64_ONLY"
    "NAME;CONDITION"
    "SOURCES;HEADERS;INCLUDE_DIRS;COMPILE_DEFINITIONS;COMPILE_OPTIONS;LINK_LIBRARIES;DEPENDS"
    ${ARGN}
  )

  if(NOT PROV_NAME)
    message(FATAL_ERROR "libfabric_add_provider: NAME is required")
  endif()
  if(NOT PROV_SOURCES)
    message(
      FATAL_ERROR "libfabric_add_provider(${PROV_NAME}): SOURCES is required"
    )
  endif()

  string(TOUPPER "${PROV_NAME}" PROV_UPPER)

  # Check if provider is enabled
  if(NOT LIBFABRIC_PROVIDER_${PROV_UPPER})
    message(STATUS "Provider ${PROV_NAME}: disabled by option")
    set(HAVE_${PROV_UPPER}
        FALSE
        CACHE INTERNAL ""
    )
    return()
  endif()

  # Check direct mode
  if(LIBFABRIC_DIRECT AND NOT LIBFABRIC_DIRECT STREQUAL PROV_NAME)
    message(
      STATUS "Provider ${PROV_NAME}: skipped (direct mode: ${LIBFABRIC_DIRECT})"
    )
    set(HAVE_${PROV_UPPER}
        FALSE
        CACHE INTERNAL ""
    )
    return()
  endif()

  # Check platform constraints
  set(_can_build TRUE)
  set(_skip_reason "")

  if(PROV_LINUX_ONLY AND NOT LIBFABRIC_LINUX)
    set(_can_build FALSE)
    set(_skip_reason "Linux only")
  endif()

  if(PROV_FREEBSD_ONLY AND NOT LIBFABRIC_FREEBSD)
    set(_can_build FALSE)
    set(_skip_reason "FreeBSD only")
  endif()

  if(PROV_MACOS_ONLY AND NOT LIBFABRIC_MACOS)
    set(_can_build FALSE)
    set(_skip_reason "macOS only")
  endif()

  if(PROV_X86_64_ONLY AND _can_build)
    if(NOT LIBFABRIC_X86_64)
      set(_can_build FALSE)
      set(_skip_reason "x86_64 only")
    endif()
  endif()

  if(PROV_AARCH64_ONLY AND _can_build)
    if(NOT LIBFABRIC_AARCH64)
      set(_can_build FALSE)
      set(_skip_reason "aarch64 only")
    endif()
  endif()

  # Check dependencies
  if(_can_build)
    foreach(_dep IN LISTS PROV_DEPENDS)
      if(NOT ${_dep}_FOUND)
        set(_can_build FALSE)
        set(_skip_reason "missing dependency: ${_dep}")
        break()
      endif()
    endforeach()
  endif()

  # Check custom condition
  if(PROV_CONDITION AND _can_build)
    if(NOT (${PROV_CONDITION}))
      set(_can_build FALSE)
      set(_skip_reason "condition not met")
    endif()
  endif()

  if(NOT _can_build)
    message(STATUS "Provider ${PROV_NAME}: disabled (${_skip_reason})")
    set(HAVE_${PROV_UPPER}
        FALSE
        CACHE INTERNAL ""
    )
    return()
  endif()

  # Provider can be built
  set(HAVE_${PROV_UPPER}
      TRUE
      CACHE INTERNAL "Provider ${PROV_NAME} is available"
  )

  # Prepend source directory to relative paths
  set(_full_sources "")
  foreach(_src IN LISTS PROV_SOURCES PROV_HEADERS)
    if(IS_ABSOLUTE "${_src}")
      list(APPEND _full_sources "${_src}")
    else()
      list(APPEND _full_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
    endif()
  endforeach()

  # Build as DL plugin or static?
  if(LIBFABRIC_${PROV_UPPER}_DLOPEN AND BUILD_SHARED_LIBS)
    set(HAVE_${PROV_UPPER}_DL
        TRUE
        CACHE INTERNAL ""
    )

    # Create module library for DL provider
    add_library(${PROV_NAME}-fi MODULE ${_full_sources})

    target_include_directories(
      ${PROV_NAME}-fi
      PRIVATE ${PROV_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include"
              "${CMAKE_BINARY_DIR}"
    )

    target_compile_definitions(
      ${PROV_NAME}-fi PRIVATE ${PROV_COMPILE_DEFINITIONS}
                              ${LIBFABRIC_PLATFORM_DEFINITIONS}
    )

    if(PROV_COMPILE_OPTIONS)
      target_compile_options(${PROV_NAME}-fi PRIVATE ${PROV_COMPILE_OPTIONS})
    endif()

    target_link_libraries(${PROV_NAME}-fi PRIVATE fabric ${PROV_LINK_LIBRARIES})

    set_target_properties(
      ${PROV_NAME}-fi
      PROPERTIES OUTPUT_NAME "${PROV_NAME}-fi"
                 PREFIX "lib"
                 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/libfabric"
    )

    libfabric_set_visibility(${PROV_NAME}-fi)

    install(TARGETS ${PROV_NAME}-fi
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/libfabric"
                    COMPONENT providers
    )

    set_property(GLOBAL APPEND PROPERTY LIBFABRIC_DL_PROVIDERS ${PROV_NAME})
    message(STATUS "Provider ${PROV_NAME}: enabled (plugin)")
  else()
    set(HAVE_${PROV_UPPER}_DL
        FALSE
        CACHE INTERNAL ""
    )

    # Create an OBJECT library for this provider This isolates each provider's
    # compilation with its own include directories
    set(_objlib_name "_prov_${PROV_NAME}_objects")
    add_library(${_objlib_name} OBJECT ${_full_sources})

    # Enable PIC so objects can be used in shared libraries
    set_target_properties(
      ${_objlib_name} PROPERTIES POSITION_INDEPENDENT_CODE ON
    )

    # Set include directories for this provider only
    target_include_directories(
      ${_objlib_name}
      PRIVATE ${PROV_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include"
              "${CMAKE_BINARY_DIR}" ${LIBFABRIC_PLATFORM_INCLUDE_DIRS}
    )

    # Set compile definitions
    target_compile_definitions(
      ${_objlib_name} PRIVATE ${PROV_COMPILE_DEFINITIONS}
                              ${LIBFABRIC_PLATFORM_DEFINITIONS}
    )

    # Set compile options if specified
    if(PROV_COMPILE_OPTIONS)
      target_compile_options(${_objlib_name} PRIVATE ${PROV_COMPILE_OPTIONS})
    endif()

    # Link libraries (for their include directories and transitive deps)
    if(PROV_LINK_LIBRARIES)
      target_link_libraries(${_objlib_name} PRIVATE ${PROV_LINK_LIBRARIES})
    endif()

    # Apply visibility settings
    libfabric_set_visibility(${_objlib_name})

    # Register the object library for inclusion in the main fabric target
    set_property(
      GLOBAL APPEND PROPERTY LIBFABRIC_PROVIDER_OBJECT_LIBS ${_objlib_name}
    )

    # Also track link libraries for the final fabric target
    if(PROV_LINK_LIBRARIES)
      set_property(
        GLOBAL APPEND PROPERTY LIBFABRIC_PROVIDER_LINK_LIBRARIES
                               ${PROV_LINK_LIBRARIES}
      )
    endif()

    set_property(GLOBAL APPEND PROPERTY LIBFABRIC_STATIC_PROVIDERS ${PROV_NAME})
    message(STATUS "Provider ${PROV_NAME}: enabled (static)")
  endif()

  set_property(GLOBAL APPEND PROPERTY LIBFABRIC_PROVIDERS ${PROV_NAME})
endfunction()

# -----------------------------------------------------------------------------
# libfabric_add_hook_provider( NAME <name> SOURCES <source files...> [HEADERS
# <header files...>] [INCLUDE_DIRS <directories...>] [COMPILE_DEFINITIONS
# <definitions...>] [LINK_LIBRARIES <libraries...>] [DEPENDS <find package
# dependencies...>] [CONDITION <boolean expression>] )
#
# Registers a hook provider with the build system. Hook providers use
# LIBFABRIC_HOOK_<NAME> option instead of LIBFABRIC_PROVIDER_<NAME>.
# -----------------------------------------------------------------------------
function(libfabric_add_hook_provider)
  cmake_parse_arguments(
    HOOK "LINUX_ONLY" "NAME;CONDITION"
    "SOURCES;HEADERS;INCLUDE_DIRS;COMPILE_DEFINITIONS;LINK_LIBRARIES;DEPENDS"
    ${ARGN}
  )

  if(NOT HOOK_NAME)
    message(FATAL_ERROR "libfabric_add_hook_provider: NAME is required")
  endif()
  if(NOT HOOK_SOURCES)
    message(
      FATAL_ERROR
        "libfabric_add_hook_provider(${HOOK_NAME}): SOURCES is required"
    )
  endif()

  string(TOUPPER "${HOOK_NAME}" HOOK_UPPER)

  # Check if hook provider is enabled
  if(NOT LIBFABRIC_HOOK_${HOOK_UPPER})
    message(STATUS "Hook provider ${HOOK_NAME}: disabled by option")
    set(HAVE_${HOOK_UPPER}
        FALSE
        CACHE INTERNAL ""
    )
    return()
  endif()

  # Check platform constraints
  if(HOOK_LINUX_ONLY AND NOT LIBFABRIC_LINUX)
    message(STATUS "Hook provider ${HOOK_NAME}: disabled (Linux only)")
    set(HAVE_${HOOK_UPPER}
        FALSE
        CACHE INTERNAL ""
    )
    return()
  endif()

  # Check dependencies
  set(_can_build TRUE)
  set(_skip_reason "")

  foreach(_dep IN LISTS HOOK_DEPENDS)
    if(NOT ${_dep}_FOUND)
      set(_can_build FALSE)
      set(_skip_reason "missing dependency: ${_dep}")
      break()
    endif()
  endforeach()

  # Check custom condition
  if(HOOK_CONDITION AND _can_build)
    if(NOT (${HOOK_CONDITION}))
      set(_can_build FALSE)
      set(_skip_reason "condition not met")
    endif()
  endif()

  if(NOT _can_build)
    message(STATUS "Hook provider ${HOOK_NAME}: disabled (${_skip_reason})")
    set(HAVE_${HOOK_UPPER}
        FALSE
        CACHE INTERNAL ""
    )
    return()
  endif()

  set(HAVE_${HOOK_UPPER}
      TRUE
      CACHE INTERNAL "Hook provider ${HOOK_NAME} is available"
  )

  # Prepend source directory to relative paths
  set(_full_sources "")
  foreach(_src IN LISTS HOOK_SOURCES HOOK_HEADERS)
    if(IS_ABSOLUTE "${_src}")
      list(APPEND _full_sources "${_src}")
    else()
      list(APPEND _full_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
    endif()
  endforeach()

  # Create an OBJECT library for this hook provider
  set(_objlib_name "_hook_${HOOK_NAME}_objects")
  add_library(${_objlib_name} OBJECT ${_full_sources})

  # Enable PIC so objects can be used in shared libraries
  set_target_properties(${_objlib_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Set include directories for this hook provider only
  target_include_directories(
    ${_objlib_name}
    PRIVATE ${HOOK_INCLUDE_DIRS}
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_SOURCE_DIR}/prov/hook/include"
            "${CMAKE_SOURCE_DIR}/prov/hook/src"
            ${LIBFABRIC_PLATFORM_INCLUDE_DIRS}
  )

  # Set compile definitions
  target_compile_definitions(
    ${_objlib_name} PRIVATE ${HOOK_COMPILE_DEFINITIONS}
                            ${LIBFABRIC_PLATFORM_DEFINITIONS}
  )

  # Link libraries (for their include directories and transitive deps)
  if(HOOK_LINK_LIBRARIES)
    target_link_libraries(${_objlib_name} PRIVATE ${HOOK_LINK_LIBRARIES})
  endif()

  # Link to the hook includes interface library to get access to other hook
  # provider headers (e.g., hook_prov.h includes hook_perf.h when HAVE_PERF is
  # defined)
  if(TARGET libfabric_hook_includes)
    target_link_libraries(${_objlib_name} PRIVATE libfabric_hook_includes)
  endif()

  # Apply visibility settings
  libfabric_set_visibility(${_objlib_name})

  # Register the object library for inclusion in the main fabric target
  set_property(
    GLOBAL APPEND PROPERTY LIBFABRIC_PROVIDER_OBJECT_LIBS ${_objlib_name}
  )

  # Also track link libraries for the final fabric target
  if(HOOK_LINK_LIBRARIES)
    set_property(
      GLOBAL APPEND PROPERTY LIBFABRIC_PROVIDER_LINK_LIBRARIES
                             ${HOOK_LINK_LIBRARIES}
    )
  endif()

  # Add hook provider includes to the interface library so core hook code can
  # find headers (e.g., hook_prov.h includes hook_perf.h when HAVE_PERF is
  # defined) Use BUILD_INTERFACE to ensure these paths are only used during
  # build, not exported
  if(HOOK_INCLUDE_DIRS AND TARGET libfabric_hook_includes)
    foreach(_inc IN LISTS HOOK_INCLUDE_DIRS)
      target_include_directories(
        libfabric_hook_includes INTERFACE $<BUILD_INTERFACE:${_inc}>
      )
    endforeach()
  endif()

  message(STATUS "Hook provider ${HOOK_NAME}: enabled")
endfunction()

# -----------------------------------------------------------------------------
# libfabric_print_provider_summary() Print a summary of enabled providers
# -----------------------------------------------------------------------------
function(libfabric_print_provider_summary)
  get_property(_all GLOBAL PROPERTY LIBFABRIC_PROVIDERS)
  get_property(_static GLOBAL PROPERTY LIBFABRIC_STATIC_PROVIDERS)
  get_property(_dl GLOBAL PROPERTY LIBFABRIC_DL_PROVIDERS)

  message(STATUS "")
  message(STATUS "=== Libfabric Provider Summary ===")
  if(_static)
    string(REPLACE ";" " " _static_str "${_static}")
    message(STATUS "Static providers: ${_static_str}")
  else()
    message(STATUS "Static providers: (none)")
  endif()
  if(_dl)
    string(REPLACE ";" " " _dl_str "${_dl}")
    message(STATUS "Plugin providers: ${_dl_str}")
  else()
    message(STATUS "Plugin providers: (none)")
  endif()
  message(STATUS "")
endfunction()

# -----------------------------------------------------------------------------
# libfabric_get_provider_link_libraries(VAR) Get accumulated provider link
# libraries
# -----------------------------------------------------------------------------
function(libfabric_get_provider_link_libraries VAR)
  get_property(_libs GLOBAL PROPERTY LIBFABRIC_PROVIDER_LINK_LIBRARIES)
  set(${VAR}
      ${_libs}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_get_provider_object_libs(VAR) Get accumulated provider object
# libraries
# -----------------------------------------------------------------------------
function(libfabric_get_provider_object_libs VAR)
  get_property(_libs GLOBAL PROPERTY LIBFABRIC_PROVIDER_OBJECT_LIBS)
  set(${VAR}
      ${_libs}
      PARENT_SCOPE
  )
endfunction()
