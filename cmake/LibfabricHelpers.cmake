# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckCSourceCompiles)
include(CheckCSourceRuns)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(CheckCCompilerFlag)
include(CMakePushCheckState)
include(FeatureSummary)
include(GNUInstallDirs)

# -----------------------------------------------------------------------------
# libfabric_check_and_set_flag(<flag> <var>) Check if the compiler supports a
# flag and add it to the variable
# -----------------------------------------------------------------------------
function(libfabric_check_and_set_flag FLAG VAR)
  string(REGEX REPLACE "[^a-zA-Z0-9]" "_" FLAG_VAR "HAVE_FLAG_${FLAG}")
  check_c_compiler_flag("${FLAG}" ${FLAG_VAR})
  if(${FLAG_VAR})
    set(${VAR}
        "${${VAR}} ${FLAG}"
        PARENT_SCOPE
    )
  endif()
endfunction()

# -----------------------------------------------------------------------------
# libfabric_configure_sanitizers() Configure sanitizer flags based on
# LIBFABRIC_ENABLE_* options Must be called after project() and before target
# definitions
# -----------------------------------------------------------------------------
function(libfabric_configure_sanitizers)
  if(LIBFABRIC_ENABLE_ASAN)
    check_c_compiler_flag("-fsanitize=address" _HAVE_ASAN)
    if(_HAVE_ASAN)
      add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
      add_link_options(-fsanitize=address)
      message(STATUS "AddressSanitizer: enabled")
    else()
      message(
        WARNING "AddressSanitizer requested but not supported by compiler"
      )
    endif()
  endif()

  if(LIBFABRIC_ENABLE_TSAN)
    check_c_compiler_flag("-fsanitize=thread" _HAVE_TSAN)
    if(_HAVE_TSAN)
      add_compile_options(-fsanitize=thread)
      add_link_options(-fsanitize=thread)
      message(STATUS "ThreadSanitizer: enabled")
    else()
      message(WARNING "ThreadSanitizer requested but not supported by compiler")
    endif()
  endif()

  if(LIBFABRIC_ENABLE_UBSAN)
    check_c_compiler_flag("-fsanitize=undefined" _HAVE_UBSAN)
    if(_HAVE_UBSAN)
      add_compile_options(-fsanitize=undefined)
      add_link_options(-fsanitize=undefined)
      message(STATUS "UndefinedBehaviorSanitizer: enabled")
    else()
      message(
        WARNING
          "UndefinedBehaviorSanitizer requested but not supported by compiler"
      )
    endif()
  endif()

  if(LIBFABRIC_ENABLE_LSAN)
    # LSAN is included in ASAN on Linux, standalone on some platforms
    check_c_compiler_flag("-fsanitize=leak" _HAVE_LSAN)
    if(_HAVE_LSAN)
      add_compile_options(-fsanitize=leak)
      add_link_options(-fsanitize=leak)
      message(STATUS "LeakSanitizer: enabled")
    else()
      message(WARNING "LeakSanitizer requested but not supported by compiler")
    endif()
  endif()
endfunction()

# -----------------------------------------------------------------------------
# libfabric_configure_picky_warnings() Configure extra compiler warnings when
# LIBFABRIC_PICKY is enabled Mirrors autotools --enable-picky behavior
# -----------------------------------------------------------------------------
function(libfabric_configure_picky_warnings)
  if(NOT LIBFABRIC_PICKY)
    return()
  endif()

  # Base warning flags (always used in debug builds)
  set(_base_warnings -Wall -Wundef -Wpointer-arith)

  # Debug warning flags
  set(_debug_warnings -Wextra -Wno-unused-parameter -Wno-sign-compare
                      -Wno-missing-field-initializers
  )

  # Picky warning flags (pedantic mode)
  set(_picky_warnings -Wno-long-long -Wmissing-prototypes -Wstrict-prototypes
                      -Wcomment -pedantic
  )

  # Check and add flags that the compiler supports
  foreach(_flag IN LISTS _base_warnings _debug_warnings _picky_warnings)
    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" _flag_var
                         "HAVE_WARNING_FLAG${_flag}"
    )
    check_c_compiler_flag("${_flag}" ${_flag_var})
    if(${_flag_var})
      add_compile_options("${_flag}")
    endif()
  endforeach()

  message(STATUS "Picky compiler warnings: enabled")
endfunction()

# -----------------------------------------------------------------------------
# libfabric_set_visibility(TARGET) Configure symbol visibility for a target
# -----------------------------------------------------------------------------
function(libfabric_set_visibility TARGET)
  set_target_properties(
    ${TARGET} PROPERTIES C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN
                                                    ON
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_generate_config() Generate config.h from template
# -----------------------------------------------------------------------------
function(libfabric_generate_config)
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/config.h.in" "${CMAKE_BINARY_DIR}/config.h"
    @ONLY
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_add_pkgconfig() Generate and install pkg-config file
# -----------------------------------------------------------------------------
function(libfabric_add_pkgconfig)
  set(prefix "${CMAKE_INSTALL_PREFIX}")
  set(exec_prefix "\${prefix}")
  # Use GNUInstallDirs FULL variables for absolute paths
  include(GNUInstallDirs)
  set(libdir "${CMAKE_INSTALL_FULL_LIBDIR}")
  set(includedir "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
  set(PACKAGE_VERSION "${PROJECT_VERSION}")

  # Collect required libraries
  set(LIBS_PRIVATE "")
  if(RT_FOUND)
    set(LIBS_PRIVATE "${LIBS_PRIVATE} -lrt")
  endif()
  if(DL_FOUND)
    set(LIBS_PRIVATE "${LIBS_PRIVATE} -ldl")
  endif()
  set(LIBS_PRIVATE "${LIBS_PRIVATE} -lpthread")

  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/libfabric.pc.in"
    "${CMAKE_BINARY_DIR}/libfabric.pc" @ONLY
  )
  install(
    FILES "${CMAKE_BINARY_DIR}/libfabric.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    COMPONENT Development
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_c11_atomics() Check for C11 atomics support
# -----------------------------------------------------------------------------
function(libfabric_check_c11_atomics RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdatomic.h>
    int main() {
      atomic_int a;
      atomic_init(&a, 0);
      #ifdef __STDC_NO_ATOMICS__
        #error c11 atomics are not supported
      #else
        return 0;
      #endif
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_c11_atomic_least_types() Check for C11 atomic least types
# support
# -----------------------------------------------------------------------------
function(libfabric_check_c11_atomic_least_types RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdatomic.h>
    int main() {
      atomic_int_least32_t a;
      atomic_int_least64_t b;
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_builtin_atomics() Check for compiler built-in atomics
# (__sync_* functions)
# -----------------------------------------------------------------------------
function(libfabric_check_builtin_atomics RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdint.h>
    int main() {
      int32_t a;
      __sync_add_and_fetch(&a, 0);
      __sync_sub_and_fetch(&a, 0);
      #if defined(__PPC__) && !defined(__PPC64__)
        #error compiler built-in atomics are not supported on PowerPC 32-bit
      #else
        return 0;
      #endif
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_builtin_mm_atomics() Check for memory model aware built-in
# atomics (__atomic_* functions)
# -----------------------------------------------------------------------------
function(libfabric_check_builtin_mm_atomics RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdint.h>
    int main() {
      uint64_t d;
      uint64_t s;
      uint64_t c;
      uint64_t r;
      r = __atomic_fetch_add(&d, s, __ATOMIC_SEQ_CST);
      r = __atomic_load_8(&d, __ATOMIC_SEQ_CST);
      __atomic_exchange(&d, &s, &r, __ATOMIC_SEQ_CST);
      __atomic_compare_exchange(&d, &c, &s, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
      #if defined(__PPC__) && !defined(__PPC64__)
        #error compiler built-in memory model aware atomics are not supported on PowerPC 32-bit
      #else
        return 0;
      #endif
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_int128_atomics() Check for 128-bit atomic support
# -----------------------------------------------------------------------------
function(libfabric_check_int128_atomics RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdint.h>
    int main() {
      __int128 d;
      __int128 s;
      __int128 c;
      __int128 r;
      r = __atomic_fetch_add(&d, s, __ATOMIC_SEQ_CST);
      __atomic_load(&d, &r, __ATOMIC_SEQ_CST);
      __atomic_exchange(&d, &s, &r, __ATOMIC_SEQ_CST);
      __atomic_compare_exchange(&d, &c, &s, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
      return !__atomic_is_lock_free(sizeof(d), 0);
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_cpuid() Check for CPUID intrinsic support
# -----------------------------------------------------------------------------
function(libfabric_check_cpuid RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stddef.h>
    #include <cpuid.h>
    int main() {
      int a, b, c, d;
      __cpuid_count(0, 0, a, b, c, d);
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_symver_support() Check for .symver assembler support
# -----------------------------------------------------------------------------
function(libfabric_check_symver_support RESULT_VAR)
  check_c_source_compiles(
    "
    __asm__(\".symver main_, main@ABIVER_1.0\");
    int main() { return 0; }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_alias_attribute() Check for __alias__ attribute support
# -----------------------------------------------------------------------------
function(libfabric_check_alias_attribute RESULT_VAR)
  check_c_source_compiles(
    "
    int foo(int arg);
    int foo(int arg) { return arg + 3; }
    int foo2(int arg) __attribute__ (( __alias__(\"foo\")));
    int main() { foo2(1); return 0; }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_ethtool() Check for ethtool support
# -----------------------------------------------------------------------------
function(libfabric_check_ethtool RESULT_VAR)
  check_c_source_compiles(
    "
    #include <net/if.h>
    #include <sys/types.h>
    #include <linux/ethtool.h>
    #include <linux/sockios.h>
    #include <sys/ioctl.h>
    int main() {
      unsigned long ioctl_req = SIOCETHTOOL;
      struct ethtool_cmd cmd = { .cmd = ETHTOOL_GSET };
      long speed = cmd.speed;
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_linux_perf_rdpmc() Check for linux perf rdpmc support
# -----------------------------------------------------------------------------
function(libfabric_check_linux_perf_rdpmc RESULT_VAR)
  check_c_source_compiles(
    "
    #include <linux/perf_event.h>
    int main() {
      __builtin_ia32_rdpmc(0);
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_pthread_spin() Check for pthread_spin_init
# -----------------------------------------------------------------------------
function(libfabric_check_pthread_spin RESULT_VAR)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
  check_symbol_exists(pthread_spin_init "pthread.h" ${RESULT_VAR})
  cmake_pop_check_state()
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_check_getifaddrs() Check for getifaddrs function
# -----------------------------------------------------------------------------
function(libfabric_check_getifaddrs RESULT_VAR)
  check_symbol_exists(getifaddrs "sys/types.h;ifaddrs.h" _have_getifaddrs)
  if(_have_getifaddrs)
    set(${RESULT_VAR}
        1
        PARENT_SCOPE
    )
  else()
    set(${RESULT_VAR}
        0
        PARENT_SCOPE
    )
  endif()
endfunction()

# -----------------------------------------------------------------------------
# libfabric_add_utility( NAME <name> SOURCES <sources...> [INCLUDE_DIRS
# <directories...>] [CONDITION <condition>] ) Add a utility executable that
# links against libfabric
# -----------------------------------------------------------------------------
function(libfabric_add_utility)
  cmake_parse_arguments(UTIL "" "NAME;CONDITION" "SOURCES;INCLUDE_DIRS" ${ARGN})

  if(NOT UTIL_NAME)
    message(FATAL_ERROR "libfabric_add_utility: NAME is required")
  endif()
  if(NOT UTIL_SOURCES)
    message(
      FATAL_ERROR "libfabric_add_utility(${UTIL_NAME}): SOURCES is required"
    )
  endif()

  # Check condition if specified
  if(UTIL_CONDITION)
    if(NOT (${UTIL_CONDITION}))
      return()
    endif()
  endif()

  # On Windows, add getopt implementation
  set(_util_sources ${UTIL_SOURCES})
  set(_util_extra_includes "")
  if(LIBFABRIC_WINDOWS)
    list(APPEND _util_sources
         ${CMAKE_SOURCE_DIR}/util/windows/getopt/getopt.cpp
    )
    list(APPEND _util_extra_includes ${CMAKE_SOURCE_DIR}/util/windows/getopt)
  endif()

  add_executable(${UTIL_NAME} ${_util_sources})
  target_link_libraries(
    ${UTIL_NAME}
    PRIVATE $<IF:$<BOOL:${BUILD_SHARED_LIBS}>,fabric,fabric_static>
            $<$<BOOL:${LIBFABRIC_WINDOWS}>:ws2_32>
            $<$<BOOL:${LIBFABRIC_WINDOWS}>:iphlpapi>
  )
  target_include_directories(
    ${UTIL_NAME}
    PRIVATE ${LIBFABRIC_PLATFORM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${UTIL_INCLUDE_DIRS}
            ${_util_extra_includes}
  )
  target_compile_definitions(
    ${UTIL_NAME} PRIVATE ${LIBFABRIC_PLATFORM_DEFINITIONS}
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_install_man_pages() Install pre-generated man pages from the man/
# directory
# -----------------------------------------------------------------------------
function(libfabric_install_man_pages)
  file(GLOB MAN1_PAGES "${CMAKE_SOURCE_DIR}/man/man1/*.1")
  file(GLOB MAN3_PAGES "${CMAKE_SOURCE_DIR}/man/man3/*.3")
  file(GLOB MAN7_PAGES "${CMAKE_SOURCE_DIR}/man/man7/*.7")

  install(
    FILES ${MAN1_PAGES}
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
    COMPONENT Documentation
  )
  install(
    FILES ${MAN3_PAGES}
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man3
    COMPONENT Documentation
  )
  install(
    FILES ${MAN7_PAGES}
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man7
    COMPONENT Documentation
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_find_pkgconfig(<NAME> <PKGCONFIG_NAME> [IMPORTED_TARGET_NAME])
#
# Helper macro for finding libraries via pkg-config with consistent patterns.
# Creates ${NAME}_FOUND and optionally creates ${NAME}::${NAME} imported target.
#
# Example: libfabric_find_pkgconfig(IBVerbs libibverbs) if(IBVerbs_FOUND)
# target_link_libraries(mylib PRIVATE IBVerbs::IBVerbs) endif()
# -----------------------------------------------------------------------------
macro(libfabric_find_pkgconfig NAME PKGCONFIG_NAME)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    # Use underscore prefix to avoid namespace pollution
    pkg_check_modules(_${NAME} QUIET IMPORTED_TARGET ${PKGCONFIG_NAME})
    if(_${NAME}_FOUND)
      set(${NAME}_FOUND TRUE)
      set(${NAME}_VERSION "${_${NAME}_VERSION}")
      set(${NAME}_INCLUDE_DIRS "${_${NAME}_INCLUDE_DIRS}")
      set(${NAME}_LIBRARIES "${_${NAME}_LIBRARIES}")
      set(${NAME}_LIBRARY_DIRS "${_${NAME}_LIBRARY_DIRS}")
      # Create aliased imported target if it doesn't exist
      if(NOT TARGET ${NAME}::${NAME})
        add_library(${NAME}::${NAME} ALIAS PkgConfig::_${NAME})
      endif()
    else()
      set(${NAME}_FOUND FALSE)
    endif()
  else()
    set(${NAME}_FOUND FALSE)
  endif()
endmacro()
