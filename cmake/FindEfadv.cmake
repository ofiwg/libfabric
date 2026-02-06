# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CMakePushCheckState)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_EFADV QUIET IMPORTED_TARGET libefa)
endif()

if(_EFADV_FOUND)
  set(Efadv_FOUND TRUE)
  set(Efadv_VERSION "${_EFADV_VERSION}")
  set(Efadv_INCLUDE_DIRS "${_EFADV_INCLUDE_DIRS}")
  set(Efadv_LIBRARIES "${_EFADV_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET Efadv::Efadv)
    add_library(Efadv::Efadv ALIAS PkgConfig::_EFADV)
  endif()
else()
  find_path(
    Efadv_INCLUDE_DIR
    NAMES infiniband/efadv.h
    HINTS ${Efadv_ROOT} ENV Efadv_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    Efadv_LIBRARY
    NAMES efa
    HINTS ${Efadv_ROOT} ENV Efadv_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    Efadv REQUIRED_VARS Efadv_LIBRARY Efadv_INCLUDE_DIR
  )

  if(Efadv_FOUND)
    set(Efadv_INCLUDE_DIRS "${Efadv_INCLUDE_DIR}")
    set(Efadv_LIBRARIES "${Efadv_LIBRARY}")

    if(NOT TARGET Efadv::Efadv)
      add_library(Efadv::Efadv UNKNOWN IMPORTED)
      set_target_properties(
        Efadv::Efadv
        PROPERTIES IMPORTED_LOCATION "${Efadv_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${Efadv_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

if(Efadv_FOUND)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${Efadv_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${Efadv_LIBRARIES})

  # Check for struct efadv_device_attr.max_rdma_size
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      struct efadv_device_attr attr;
      (void)attr.max_rdma_size;
      return 0;
    }
  "
    Efadv_HAVE_RDMA_SIZE
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_RNR_RETRY
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_RNR_RETRY;
      return x;
    }
  "
    Efadv_HAVE_CAPS_RNR_RETRY
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;
      return x;
    }
  "
    Efadv_HAVE_CAPS_RDMA_WRITE
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV;
      return x;
    }
  "
    Efadv_HAVE_CAPS_UNSOLICITED_WRITE_RECV
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF;
      return x;
    }
  "
    Efadv_HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF
  )

  # Check for extended CQ support (efadv_create_cq, efadv_cq_from_ibv_cq_ex,
  # efadv_wc_read_sgid)
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      (void)efadv_create_cq;
      (void)efadv_cq_from_ibv_cq_ex;
      (void)efadv_wc_read_sgid;
      return 0;
    }
  "
    Efadv_HAVE_CQ_EX
  )

  # Check for efadv_query_mr and related
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      (void)efadv_query_mr;
      struct efadv_mr_attr attr;
      (void)attr.rdma_recv_ic_id;
      int x = EFADV_MR_ATTR_VALIDITY_RDMA_READ_IC_ID;
      return x;
    }
  "
    Efadv_HAVE_QUERY_MR
  )

  # Check for struct efadv_qp_init_attr.sl
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      struct efadv_qp_init_attr attr;
      (void)attr.sl;
      return 0;
    }
  "
    Efadv_HAVE_SL
  )

  # Check for efadv_query_qp_wqs
  check_symbol_exists(
    efadv_query_qp_wqs "infiniband/efadv.h" Efadv_HAVE_QUERY_QP_WQS
  )

  # Check for efadv_query_cq
  check_symbol_exists(efadv_query_cq "infiniband/efadv.h" Efadv_HAVE_QUERY_CQ)

  # Check for struct efadv_cq_attr.doorbell
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      struct efadv_cq_attr attr;
      (void)attr.doorbell;
      return 0;
    }
  "
    Efadv_HAVE_CQ_ATTR_DB
  )

  cmake_pop_check_state()

  # Compute derived features
  if(Efadv_HAVE_QUERY_QP_WQS AND Efadv_HAVE_QUERY_CQ)
    set(Efadv_HAVE_DATA_PATH_DIRECT TRUE)
  else()
    set(Efadv_HAVE_DATA_PATH_DIRECT FALSE)
  endif()
endif()

mark_as_advanced(Efadv_INCLUDE_DIR Efadv_LIBRARY)
