# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindAtomics
-----------

Probe 32/64-bit C11 atomics, compiler builtins, and Windows Interlocked support.

Results: ``Atomics_FOUND``.

Results ``Atomics_C11_FOUND``, ``Atomics_BUILTIN_FOUND``, and
``Atomics_BUILTIN_MM_FOUND`` distinguish supported atomic implementations.
``Atomics_LIBRARIES`` supplies libatomic when those operations need it.

Requires an enabled C compiler. Probe results use CMake's internal cache.

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(CheckCSourceCompiles)
include(FindPackageHandleStandardArgs)
include(CMakePushCheckState)

# Test the widths used by ofi_atom.h, including targets without native 64-bit
# operations. Keep native and library-backed results in separate cache entries.
set(_Atomics_C11_SOURCE
    "
  #include <stdint.h>
  #include <stdatomic.h>
  int main(void) {
    _Atomic int32_t a = 0;
    _Atomic int64_t b = 0;
    int64_t expected = 0;
    atomic_fetch_add(&a, 1);
    atomic_fetch_sub(&b, 1);
    atomic_compare_exchange_strong(&b, &expected, 2);
    atomic_exchange(&b, 3);
    atomic_store(&b, 1);
    #ifdef __STDC_NO_ATOMICS__
      #error c11 atomics are not supported
    #else
      return (int)atomic_load(&b);
    #endif
  }
"
)

# Check __sync_* builtins (GCC/Clang)
check_c_source_compiles(
  "
  #include <stdint.h>
  int main() {
    int64_t a = 0;
    __sync_add_and_fetch(&a, 1);
    __sync_sub_and_fetch(&a, 1);
    __sync_bool_compare_and_swap(&a, 0, 1);
    #if defined(__PPC__) && !defined(__PPC64__)
      #error compiler builtins are not supported on 32-bit PowerPC
    #endif
    return a;
  }
"
  _Atomics_SYNC_BUILTIN_FOUND
)

# Check __atomic_* builtins (GCC/Clang)
set(_Atomics_BUILTIN_MM_SOURCE
    "
  #include <stdint.h>
  int main() {
    uint64_t d = 0, s = 1, c = 0, r;
    r = __atomic_fetch_add(&d, s, __ATOMIC_SEQ_CST);
    r = __atomic_load_n(&d, __ATOMIC_SEQ_CST);
    __atomic_store_n(&d, s, __ATOMIC_SEQ_CST);
    __atomic_exchange(&d, &s, &r, __ATOMIC_SEQ_CST);
    __atomic_compare_exchange(&d, &c, &s, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    #if defined(__PPC__) && !defined(__PPC64__)
      #error compiler builtins are not supported on 32-bit PowerPC
    #endif
    return (int)r;
  }
"
)
set(Atomics_LIBRARIES "")
foreach(_kind C11 BUILTIN_MM)
  check_c_source_compiles(
    "${_Atomics_${_kind}_SOURCE}" _Atomics_${_kind}_NATIVE
  )
  set(Atomics_${_kind}_FOUND ${_Atomics_${_kind}_NATIVE})
  if(NOT Atomics_${_kind}_FOUND AND NOT WIN32)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_LIBRARIES atomic)
    check_c_source_compiles(
      "${_Atomics_${_kind}_SOURCE}" _Atomics_${_kind}_LIBATOMIC
    )
    cmake_pop_check_state()
    if(_Atomics_${_kind}_LIBATOMIC)
      set(Atomics_${_kind}_FOUND TRUE)
      set(Atomics_LIBRARIES atomic)
    endif()
  endif()
endforeach()

# On Windows/MSVC, libfabric uses Interlocked* intrinsics from windows.h. These
# work through the HAVE_BUILTIN_ATOMICS code path in windows/osd.h, so we report
# them as Atomics_BUILTIN_FOUND.
if(WIN32 AND NOT _Atomics_SYNC_BUILTIN_FOUND)
  check_c_source_compiles(
    "
    #include <windows.h>
    int main() {
      volatile LONG a = 0;
      InterlockedAdd(&a, 1);
      InterlockedCompareExchange(&a, 0, 1);
      InterlockedExchange(&a, 2);
      return (int)a;
    }
  "
    _Atomics_WINDOWS_INTERLOCKED_FOUND
  )

  if(_Atomics_WINDOWS_INTERLOCKED_FOUND)
    set(Atomics_BUILTIN_FOUND TRUE)
  else()
    set(Atomics_BUILTIN_FOUND FALSE)
  endif()
else()
  set(Atomics_BUILTIN_FOUND ${_Atomics_SYNC_BUILTIN_FOUND})
endif()

# Determine overall result
if(Atomics_C11_FOUND
   OR Atomics_BUILTIN_FOUND
   OR Atomics_BUILTIN_MM_FOUND
)
  set(Atomics_FOUND TRUE)
else()
  set(Atomics_FOUND FALSE)
endif()

find_package_handle_standard_args(
  Atomics
  REQUIRED_VARS Atomics_FOUND
  FAIL_MESSAGE
    "No atomics support found (C11, compiler builtins, or Windows Interlocked required)"
)
if(Atomics_FOUND AND CMAKE_C_IMPLICIT_LINK_LIBRARIES MATCHES
                     "(^|;)atomic(_asneeded)?(;|$)"
)
  # GCC 16 can supply libatomic implicitly, including for 128-bit operations.
  # Record it explicitly so static consumers using another compiler still link.
  set(Atomics_LIBRARIES atomic)
endif()
