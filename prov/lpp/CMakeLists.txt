# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# LPP provider
libfabric_add_provider(
  NAME lpp
  SOURCES src/lpp_atomic.c
          src/lpp_av.c
          src/lpp_cm.c
          src/lpp_cntr.c
          src/lpp_completion.c
          src/lpp_control.c
          src/lpp_cq.c
          src/lpp_ctx.c
          src/lpp_domain.c
          src/lpp_ep.c
          src/lpp_eq.c
          src/lpp_fabric.c
          src/lpp_getinfo.c
          src/lpp_hmem.c
          src/lpp_init.c
          src/lpp_io.c
          src/lpp_klpp.c
          src/lpp_memcpy.c
          src/lpp_mr.c
          src/lpp_msg.c
          src/lpp_rma.c
          src/lpp_rx.c
          src/lpp_stx.c
          src/lpp_tagged.c
          src/lpp_umc.c
          src/lpp_util.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  LINUX_ONLY X86_64_ONLY
)

# Thread-safe option
if(HAVE_LPP AND LIBFABRIC_LPP_THREAD_SAFE)
  # Add thread-safe compile definition
endif()
