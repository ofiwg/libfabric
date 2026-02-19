# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
#
# Find the Intel RV (Rendezvous) kernel module headers.
#
# The RV module provides kernel-assisted RDMA operations for PSM3.
#
# Result Variables: RV_FOUND        - True if RV headers were found
# RV_INCLUDE_DIRS - Include directories for RV headers
#
# =============================================================================

include(CheckIncludeFile)

# Check for the RV user ioctls header
check_include_file("rdma/rv_user_ioctls.h" HAVE_RV_USER_IOCTLS_H)

if(HAVE_RV_USER_IOCTLS_H)
  set(RV_FOUND TRUE)
  # Header is in system include path
  set(RV_INCLUDE_DIRS "")
else()
  set(RV_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  RV
  REQUIRED_VARS RV_FOUND
  FAIL_MESSAGE "RV (Rendezvous) kernel module headers not found"
)

mark_as_advanced(HAVE_RV_USER_IOCTLS_H)
