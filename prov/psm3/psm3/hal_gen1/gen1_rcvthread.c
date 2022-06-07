#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2021 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2021 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* Copyright (c) 2003-2021 Intel Corporation. All rights reserved. */

#include <sys/poll.h>

#include "psm_user.h"
#include "psm2_hal.h"
#include "psm_mq_internal.h"
#include "ptl_ips.h"
#include "ips_proto.h"
#include "gen1_hal.h"

/*
 * Receiver thread support.
 *
 * By default, polling in the driver asks the chip to generate an interrupt on
 * every packet.  When the driver supports POLLURG we can switch the poll mode
 * to one that requests interrupts only for packets that contain an urgent bit
 * (and optionally enable interrupts for hdrq overflow events).  When poll
 * returns an event, we *try* to make progress on the receive queue but simply
 * go back to sleep if we notice that the main thread is already making
 * progress.
 *
 * returns:
 * 	PSM2_IS_FINALIZED - fd_pipe was closed, caller can exit rcvthread
 * 	PSM2_NO_PROGRESS - got an EINTR, need to be called again with same
 * 			next_timeout value
 * 	PSM2_TIMEOUT - poll waited full timeout, no events
 * 	PSM2_OK - poll found an event and processed it
 * 	PSM2_INTERNAL_ERR - unexpected error attempting poll()
 * updates counters: pollok (poll's which made progress), pollcyc (time spent
 * 	polling without finding any events)
 */
psm2_error_t psm3_gen1_ips_ptl_pollintr(psm2_ep_t ep,
		struct ips_recvhdrq *recvq, int fd_pipe, int next_timeout,
		uint64_t *pollok, uint64_t *pollcyc)
{
	struct pollfd pfd[2];
	int ret;
	uint64_t t_cyc;
	psm2_error_t err;

	// pfd[0] is for urgent inbound packets (NAK, urgent ACK, etc)
	// pfd[1] is for rcvthread termination
	// on timeout (poll() returns 0), we do background process checks
	//		for non urgent inbound packets
	pfd[0].fd = psm3_gen1_get_fd(ep->context.psm_hw_ctxt);
	pfd[0].events = POLLIN;
	pfd[0].revents = 0;
	pfd[1].fd = fd_pipe;
	pfd[1].events = POLLIN;
	pfd[1].revents = 0;
	ret = poll(pfd, 2, next_timeout);
	t_cyc = get_cycles();
	if_pf(ret < 0) {
		if (errno == EINTR) {
			_HFI_DBG("got signal, keep polling\n");
			return PSM2_OK_NO_PROGRESS;
		} else {
			psm3_handle_error(PSMI_EP_NORETURN,
					  PSM2_INTERNAL_ERR,
					  "Receive thread poll() error: %s",
					  strerror(errno));
			return PSM2_INTERNAL_ERR;
		}
	} else if (pfd[1].revents) {
		/* Any type of event on this fd means exit, should be POLLHUP */
		_HFI_DBG("close thread: revents=0x%x\n", pfd[1].revents);
		close(fd_pipe);
		return PSM2_IS_FINALIZED;
	} else {
		if (!PSMI_LOCK_TRY(psm3_creation_lock)) {
			if (ret == 0 || pfd[0].revents & (POLLIN | POLLERR)) {
				if (PSMI_LOCK_DISABLED) {
					// this path is not supported.  having rcvthread
					// and PSMI_PLOCK_IS_NOLOCK define not allowed.
					/* We do this check without acquiring the lock, no sense
					 * adding the overhead and it doesn't matter if we're
					 * wrong. */
					if (psm3_gen1_recvhdrq_isempty(recvq))
						return PSM2_OK;
					if(recvq->proto->flags & IPS_PROTO_FLAG_CCA_PRESCAN) {
						psm3_gen1_recvhdrq_scan_cca(recvq);
					}
					if (!ips_recvhdrq_trylock(recvq))
						return PSM2_OK;
					err = psm3_gen1_recvhdrq_progress(recvq);
					if (err == PSM2_OK)
						(*pollok)++;
					else
						(*pollcyc) += get_cycles() - t_cyc;
					ips_recvhdrq_unlock(recvq);
				} else {

					ep = psm3_opened_endpoint;

					if (!PSMI_LOCK_TRY(ep->mq->progress_lock)) {
						if(recvq->proto->flags & IPS_PROTO_FLAG_CCA_PRESCAN ) {
								psm3_gen1_recvhdrq_scan_cca(recvq);
						}
						PSMI_UNLOCK(ep->mq->progress_lock);
					}
					/* Go through all master endpoints. */
					do{
						if (!PSMI_LOCK_TRY(ep->mq->progress_lock)) {
							/* If we time out, we service shm and NIC.
							* If not, we assume to have received an urgent
							* packet and service only NIC.
							*/
							err = psm3_poll_internal(ep,
										 ret == 0 ? PSMI_TRUE : PSMI_FALSE);
#ifdef PSM_HAVE_REG_MR
#ifdef UMR_CACHE
							if (ep->mr_cache_mode == MR_CACHE_MODE_USER && !ep->verbs_ep.umrc.thread)
								psm3_gen1_poll_uffd_events(ep);
#endif
#endif
							if (err == PSM2_OK)
								(*pollok)++;
							else
								(*pollcyc) += get_cycles() - t_cyc;
							PSMI_UNLOCK(ep->mq->progress_lock);
						}

						/* get next endpoint from multi endpoint list */
						ep = ep->user_ep_next;
					} while(NULL != ep);
				}
			}
			PSMI_UNLOCK(psm3_creation_lock);
		}
		if (ret == 0)
			/* timed out poll */
			return PSM2_TIMEOUT;
		else
			/* found work to do */
			return PSM2_OK;
	}
}
#endif /* PSM_OPA */
