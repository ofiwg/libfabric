# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# PSM2 provider (Intel Performance Scaled Messaging 2)

# Check for PSM2 feature availability Modern PSM2 (11.2.185+) has all these
# features
if(PSM2_FOUND)
  include(CheckSymbolExists)

  # Get include directories from the target
  set(_psm2_includes ${PSM2_INCLUDE_DIRS})
  if(NOT _psm2_includes AND TARGET PSM2::PSM2)
    get_target_property(_psm2_includes PSM2::PSM2 INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _psm2_includes)
      set(_psm2_includes "")
    endif()
  endif()

  set(CMAKE_REQUIRED_INCLUDES ${_psm2_includes})
  set(CMAKE_REQUIRED_LIBRARIES ${PSM2_LIBRARIES})

  # Check for psm2_am_register_handlers_2 (present in modern PSM2)
  check_symbol_exists(
    psm2_am_register_handlers_2 "psm2_am.h" _HAVE_PSM2_AM_REGISTER_HANDLERS_2
  )
  if(_HAVE_PSM2_AM_REGISTER_HANDLERS_2)
    set(HAVE_PSM2_AM_REGISTER_HANDLERS_2 1)
  else()
    set(HAVE_PSM2_AM_REGISTER_HANDLERS_2 0)
  endif()

  # Check for psm2_mq_fp_msg (present in modern PSM2)
  check_symbol_exists(psm2_mq_fp_msg "psm2_mq.h" _HAVE_PSM2_MQ_FP_MSG)
  if(_HAVE_PSM2_MQ_FP_MSG)
    set(HAVE_PSM2_MQ_FP_MSG 1)
  else()
    set(HAVE_PSM2_MQ_FP_MSG 0)
  endif()

  # Check for psm2_mq_ipeek_dequeue_multi (indicates psm2_mq_req_user support)
  check_symbol_exists(
    psm2_mq_ipeek_dequeue_multi "psm2_mq.h" _HAVE_PSM2_MQ_REQ_USER
  )
  if(_HAVE_PSM2_MQ_REQ_USER)
    set(HAVE_PSM2_MQ_REQ_USER 1)
  else()
    set(HAVE_PSM2_MQ_REQ_USER 0)
  endif()

  # Check for psm2_info_query (present in modern PSM2)
  check_symbol_exists(psm2_info_query "psm2.h" _HAVE_PSM2_INFO_QUERY)
  if(_HAVE_PSM2_INFO_QUERY)
    set(HAVE_PSM2_INFO_QUERY 1)
  else()
    set(HAVE_PSM2_INFO_QUERY 0)
  endif()

  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(_psm2_includes)

  # We're not building PSM2 from source
  set(HAVE_PSM2_SRC 0)
endif()

libfabric_add_provider(
  NAME psm2
  SOURCES src/psm2_revision.c
          src/psmx2_am.c
          src/psmx2_atomic.c
          src/psmx2_attr.c
          src/psmx2_av.c
          src/psmx2_cm.c
          src/psmx2_cntr.c
          src/psmx2_cq.c
          src/psmx2_domain.c
          src/psmx2_ep.c
          src/psmx2_fabric.c
          src/psmx2_init.c
          src/psmx2_mr.c
          src/psmx2_msg.c
          src/psmx2_rma.c
          src/psmx2_tagged.c
          src/psmx2_trx_ctxt.c
          src/psmx2_util.c
          src/psmx2_wait.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
  COMPILE_DEFINITIONS
    HAVE_PSM2_SRC=${HAVE_PSM2_SRC}
    HAVE_PSM2_AM_REGISTER_HANDLERS_2=${HAVE_PSM2_AM_REGISTER_HANDLERS_2}
    HAVE_PSM2_MQ_FP_MSG=${HAVE_PSM2_MQ_FP_MSG}
    HAVE_PSM2_MQ_REQ_USER=${HAVE_PSM2_MQ_REQ_USER}
    HAVE_PSM2_INFO_QUERY=${HAVE_PSM2_INFO_QUERY}
  DEPENDS PSM2
  LINK_LIBRARIES PSM2::PSM2
  LINUX_ONLY X86_64_ONLY
)
