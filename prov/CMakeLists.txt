# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
foreach(_provider IN LISTS LIBFABRIC_PROVIDERS)
  string(TOUPPER "${_provider}" _name)
  if(NOT LIBFABRIC_PROVIDER_${_name} STREQUAL "OFF")
    add_subdirectory(${_provider})
  endif()
endforeach()
add_subdirectory(hook)

# Derive availability from actual targets on every configure. These are not
# cached: turning a provider off must also update the generated config header.
foreach(_provider IN LISTS LIBFABRIC_PROVIDERS LIBFABRIC_HOOKS)
  string(TOUPPER "${_provider}" _name)
  set(_target libfabric_provider_${_provider})
  set(HAVE_${_name}
      FALSE
      PARENT_SCOPE
  )
  set(HAVE_${_name}_DL
      FALSE
      PARENT_SCOPE
  )
  if(TARGET ${_target})
    get_target_property(_plugin ${_target} LIBFABRIC_PLUGIN)
    set(HAVE_${_name}
        TRUE
        PARENT_SCOPE
    )
    set(HAVE_${_name}_DL
        ${_plugin}
        PARENT_SCOPE
    )
    if(_plugin)
      message(STATUS "Provider ${_provider}: plugin")
    else()
      message(STATUS "Provider ${_provider}: built-in")
    endif()
    list(APPEND _enabled ${_provider})
  elseif(LIBFABRIC_PROVIDER_${_name} STREQUAL "ON" OR LIBFABRIC_${_name}_DLOPEN)
    message(
      FATAL_ERROR
        "Provider ${_provider} was requested but cannot be built; see dependency checks above"
    )
  endif()
endforeach()
set(LIBFABRIC_ENABLED_PROVIDERS
    "${_enabled}"
    PARENT_SCOPE
)
