# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricTargetSettings
-----------------------

Apply private compilation and instrumentation requirements.

Defines libfabric_configure_target(target), which applies the internal build
settings, PIC, visibility, and C11 requirements to an existing target.
libfabric_configure_instrumentation(target) applies selected sanitizer,
coverage, analysis, and warning options after validating compiler support.
Both functions read the build's LIBFABRIC_* options when called.

Globally guarded: declares functions; evaluates no per-directory build state.
]]
include_guard(GLOBAL)
include(CheckCCompilerFlag)
include(CMakePushCheckState)

# Apply private requirements without exporting internal build targets.
function(libfabric_configure_target target)
  target_link_libraries(
    ${target} PRIVATE $<BUILD_INTERFACE:libfabric_build_settings>
  )
  set_target_properties(
    ${target}
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden
      C_STANDARD 11
      C_STANDARD_REQUIRED ON
  )
endfunction()

function(libfabric_configure_instrumentation target)
  if(LIBFABRIC_ENABLE_ASAN AND LIBFABRIC_ENABLE_TSAN)
    message(
      FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined"
    )
  endif()
  foreach(_kind address thread undefined leak)
    if((_kind STREQUAL "address" AND LIBFABRIC_ENABLE_ASAN)
       OR (_kind STREQUAL "thread" AND LIBFABRIC_ENABLE_TSAN)
       OR (_kind STREQUAL "undefined" AND LIBFABRIC_ENABLE_UBSAN)
       OR (_kind STREQUAL "leak" AND LIBFABRIC_ENABLE_LSAN)
    )
      cmake_push_check_state()
      set(CMAKE_REQUIRED_LIBRARIES "-fsanitize=${_kind}")
      check_c_compiler_flag(
        "-fsanitize=${_kind}" LIBFABRIC_HAS_${_kind}_SANITIZER
      )
      cmake_pop_check_state()
      if(NOT LIBFABRIC_HAS_${_kind}_SANITIZER)
        message(FATAL_ERROR "The compiler does not support -fsanitize=${_kind}")
      endif()
      target_compile_options(
        ${target} INTERFACE "-fsanitize=${_kind}" -fno-omit-frame-pointer
      )
      target_link_options(${target} INTERFACE "-fsanitize=${_kind}")
    endif()
  endforeach()
  if(LIBFABRIC_ENABLE_COVERAGE)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES --coverage)
    check_c_compiler_flag(--coverage LIBFABRIC_HAS_COVERAGE)
    cmake_pop_check_state()
    if(NOT LIBFABRIC_HAS_COVERAGE)
      message(FATAL_ERROR "The compiler/linker does not support --coverage")
    endif()
    target_compile_options(${target} INTERFACE -O0 -g --coverage)
    target_link_options(${target} INTERFACE --coverage)
  endif()
  if(LIBFABRIC_THREAD_SAFETY_ANALYSIS)
    check_c_compiler_flag(
      "-Werror;-Wthread-safety;-Wthread-safety-beta"
      LIBFABRIC_HAS_THREAD_SAFETY_ANALYSIS
    )
    if(LIBFABRIC_HAS_THREAD_SAFETY_ANALYSIS)
      target_compile_options(
        ${target} INTERFACE -Wthread-safety -Wthread-safety-beta
      )
      target_compile_definitions(${target} INTERFACE OFI_THREAD_SAFETY_ANALYSIS)
    else()
      message(
        WARNING "The compiler does not support Clang thread safety analysis"
      )
    endif()
  endif()
  if(LIBFABRIC_PICKY AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      ${target}
      INTERFACE
        -Wall
        -Wextra
        -Wundef
        -Wpointer-arith
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-missing-field-initializers
        -Wno-long-long
        -Wmissing-prototypes
        -Wstrict-prototypes
        -Wcomment
        -pedantic
    )
  endif()
endfunction()
