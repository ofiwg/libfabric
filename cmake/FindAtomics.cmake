# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckCSourceCompiles)
include(FindPackageHandleStandardArgs)

# Check C11 atomics (use atomic_fetch_add which works in both C11 and C17+)
check_c_source_compiles(
  "
  #include <stdatomic.h>
  int main() {
    atomic_int a = ATOMIC_VAR_INIT(0);
    int b = atomic_load(&a);
    atomic_store(&a, 1);
    atomic_fetch_add(&a, 1);
    #ifdef __STDC_NO_ATOMICS__
      #error c11 atomics are not supported
    #else
      return b;
    #endif
  }
"
  Atomics_C11_FOUND
)

# Check __sync_* builtins (GCC/Clang)
check_c_source_compiles(
  "
  #include <stdint.h>
  int main() {
    int32_t a = 0;
    __sync_add_and_fetch(&a, 1);
    __sync_sub_and_fetch(&a, 1);
    __sync_bool_compare_and_swap(&a, 0, 1);
    return a;
  }
"
  _Atomics_SYNC_BUILTIN_FOUND
)

# Check __atomic_* builtins (GCC/Clang)
check_c_source_compiles(
  "
  #include <stdint.h>
  int main() {
    uint64_t d = 0, s = 1, r;
    r = __atomic_fetch_add(&d, s, __ATOMIC_SEQ_CST);
    r = __atomic_load_n(&d, __ATOMIC_SEQ_CST);
    __atomic_store_n(&d, s, __ATOMIC_SEQ_CST);
    return (int)r;
  }
"
  Atomics_BUILTIN_MM_FOUND
)

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
