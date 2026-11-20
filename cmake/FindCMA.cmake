# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindCMA
-------

Probe cross-memory attachment through process_vm_readv and process_vm_writev.

Results: ``CMA_FOUND``.

Requires an enabled C compiler. Probe results use CMake's internal cache.

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)

check_c_source_compiles(
  "
  #define _GNU_SOURCE
  #include <sys/uio.h>
  int main() {
    struct iovec local, remote;
    process_vm_readv(0, &local, 1, &remote, 1, 0);
    process_vm_writev(0, &local, 1, &remote, 1, 0);
    return 0;
  }
"
  CMA_FOUND
)

find_package_handle_standard_args(CMA REQUIRED_VARS CMA_FOUND)
