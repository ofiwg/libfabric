# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# POSIX component diagnostics, including Level Zero DMA-BUF tests.
fabtests_add_test(
  NAME sock_test
  CATEGORY component
  SOURCES sock_test.c
)
fabtests_add_test(
  NAME cuda_check_dmabuf
  CATEGORY component
  SOURCES dmabuf-rdma/check_cuda_dmabuf.c
)

if(HAVE_ZE)
  set(_xe dmabuf-rdma)
  fabtests_add_test(
    NAME fi_xe_rdmabw
    CATEGORY component
    SOURCES ${_xe}/fi-rdmabw-xe.c ${_xe}/util.c ${_xe}/xe.c ${_xe}/shared.c
  )
  fabtests_add_test(
    NAME fi_xe_mr_reg
    CATEGORY component
    SOURCES ${_xe}/fi-mr-reg-xe.c ${_xe}/util.c ${_xe}/xe.c ${_xe}/shared.c
  )
  fabtests_add_test(
    NAME xe_memcopy
    CATEGORY component
    SOURCES ${_xe}/memcopy-xe.c ${_xe}/util.c ${_xe}/xe.c
  )
  # These two diagnostics use verbs directly, independently of the provider.
  find_package(IBVerbs QUIET)
  if(IBVerbs_FOUND)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES IBVerbs::IBVerbs)
    check_symbol_exists(
      ibv_reg_dmabuf_mr "infiniband/verbs.h" FABTESTS_HAVE_IBV_REG_DMABUF_MR
    )
    cmake_pop_check_state()
    if(FABTESTS_HAVE_IBV_REG_DMABUF_MR)
      fabtests_add_test(
        NAME xe_rdmabw
        CATEGORY component
        SOURCES ${_xe}/rdmabw-xe.c ${_xe}/util.c ${_xe}/xe.c
        EXTRA_LIBS IBVerbs::IBVerbs
      )
      fabtests_add_test(
        NAME xe_mr_reg
        CATEGORY component
        SOURCES ${_xe}/mr-reg-xe.c ${_xe}/xe.c
        EXTRA_LIBS IBVerbs::IBVerbs
      )
    endif()
  endif()
endif()
