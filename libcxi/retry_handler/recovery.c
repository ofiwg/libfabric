/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */

/* Cassini retry handler
 *
 * Recover from a previous retry handler that left some SPTs and SCTs
 * lying around.
 */

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <err.h>
#include <stdlib.h>
#include <poll.h>
#include <assert.h>
#include <time.h>
#include <search.h>

#include "rh.h"

/* Recover the active SPTs and their attached SCTs */
static void spt_recovery(struct retry_handler *rh)
{
	struct c_event_pct event;
	int idx;

	/* Browse all the SPTs, creating an entry for each that is in
	 * the timeout state or has sw_recycle set.
	 */
	for (idx = 1; idx < C_PCT_CFG_SPT_MISC_INFO_ENTRIES; idx++) {
		struct spt_entry spt = {
			.spt_idx = idx,
		};

		/* Check whether the SPT is valid */
		cxil_read_csr(rh->dev, spt_misc_info_csr(rh, idx),
			      &spt.misc_info, sizeof(spt.misc_info));

		if (!spt.misc_info.vld)
			continue;

		get_spt_info(rh, &spt);

		if (spt.ram0.sw_recycle) {
			;
		} else if ((spt.misc_info.rsp_status == C_RSP_NACK_RCVD) ||
			   (spt.misc_info.rsp_status == C_RSP_PEND &&
			    spt.misc_info.to_flag)) {
			;
		} else {
			continue;
		}

		/* Simulate a TIMEOUT or NACK event */
		event = (struct c_event_pct) {
			.seq_num = spt.misc_info.req_try,
			.event_size = C_EVENT_SIZE_16_BYTE,
			.event_type = C_EVENT_PCT,
			.spt_idx = spt.spt_idx,
		};

		if (spt.misc_info.rsp_status == C_RSP_PEND &&
		    spt.misc_info.to_flag) {
			event.pct_event_type = C_PCT_REQUEST_TIMEOUT;
			request_timeout(rh, &event);
			event.return_code = C_RC_OK;
		}
		else if (spt.misc_info.rsp_status == C_RSP_NACK_RCVD) {
			event.pct_event_type = C_PCT_REQUEST_NACK;
			event.conn_idx.padded_sct_idx.sct_idx =
				spt.ram1.sct_idx;
			 /* Fake code as we don't know why the SPT was
			  * NACK'ed.
			  */
			event.return_code = C_RC_INVALID_AC,
			request_nack(rh, &event);
		}
		else {
			assert(spt.ram0.sw_recycle == 1);

			if (spt.misc_info.rsp_status == C_RSP_OP_COMP) {
				/* Completed. Release it. There may be
				 * a complete event waiting.
				 */
				recycle_spt(rh, &spt);
			}
		}
	}
}

/* Recover active SCTs that were not previously found (ie. they don't
 * have SPTs attached and were closing.
 */
static void sct_recovery(struct retry_handler *rh)
{
	int idx;

	/* Take care of the remaining SCTs. */
	for (idx = 0; idx < C_PCT_CFG_SCT_MISC_INFO_ENTRIES; idx++) {
		struct sct_entry sct = {
			.sct_idx = idx,
		};
		struct sct_entry **sct_ret;
		unsigned int sct_misc_info_offset = sct_misc_info_csr(rh, idx);

		sct_ret = tfind(&sct, &rh->sct_tree, sct_compare);
		if (sct_ret) {
			/* Already known. */
			continue;
		}

		cxil_read_csr(rh->dev, sct_misc_info_offset,
			      &sct.misc_info, sizeof(sct.misc_info));

		if (sct.misc_info.sct_status == C_SCT_CLOSE_TIMEOUT) {
			const struct c_event_pct event = {
				.event_size = C_EVENT_SIZE_16_BYTE,
				.event_type = C_EVENT_PCT,
				.return_code = C_RC_OK,
				.pct_event_type = C_PCT_SCT_TIMEOUT,
				.conn_idx.padded_sct_idx.sct_idx = idx,
				.seq_num = 0, /* TODO?, sct_ram2 */
			};

			sct_timeout(rh, &event);

		} else if (sct.misc_info.sct_status == C_SCT_CLOSE_COMP) {
			const struct c_event_pct event = {
				.event_size = C_EVENT_SIZE_16_BYTE,
				.event_type = C_EVENT_PCT,
				.return_code = C_RC_OK,
				.pct_event_type = C_PCT_ACCEL_CLOSE_COMPLETE,
				.conn_idx.padded_sct_idx.sct_idx = idx,
				.seq_num = 0, /* TODO ? */
			};

			accel_close(rh, &event);
		}
	}
}

/* Recover blocked TCTs */
static void c1_tct_recovery(struct retry_handler *rh)
{
	int idx;
	union c1_pct_cfg_tct_misc_info misc;

	/* Take care of the remaining TCTs. */
	for (idx = 0; idx < C1_PCT_CFG_TCT_MISC_INFO_ENTRIES; idx++) {
		cxil_read_csr(rh->dev, C1_PCT_CFG_TCT_MISC_INFO(idx),
			      &misc, sizeof(misc));

		if (misc.vld && misc.to_flag) {
			const struct c_event_pct event = {
				.event_size = C_EVENT_SIZE_16_BYTE,
				.event_type = C_EVENT_PCT,
				.return_code = C_RC_OK,
				.pct_event_type = C_PCT_TCT_TIMEOUT,
				.conn_idx.tct_idx.tct_idx = idx,
			};

			tct_timeout(rh, &event);
		}
	}
}

static void c2_tct_recovery(struct retry_handler *rh)
{
	int idx;
	union c2_pct_cfg_tct_misc_info misc;

	/* Take care of the remaining TCTs. */
	for (idx = 0; idx < C2_PCT_CFG_TCT_MISC_INFO_ENTRIES; idx++) {
		cxil_read_csr(rh->dev, C2_PCT_CFG_TCT_MISC_INFO(idx),
			      &misc, sizeof(misc));

		if (misc.vld && misc.to_flag) {
			const struct c_event_pct event = {
				.event_size = C_EVENT_SIZE_16_BYTE,
				.event_type = C_EVENT_PCT,
				.return_code = C_RC_OK,
				.pct_event_type = C_PCT_TCT_TIMEOUT,
				.conn_idx.tct_idx.tct_idx = idx,
			};

			tct_timeout(rh, &event);
		}
	}
}

void recovery(struct retry_handler *rh)
{
	union c_pct_prf_sct_status sct_status;
	union c_pct_prf_spt_status spt_status;
	union c_pct_prf_tct_status tct_status;
	union c_pct_prf_smt_status smt_status;
	union c1_pct_prf_trs_status c1_trs_status;
	union c2_pct_prf_trs_status c2_trs_status;
	union c_pct_prf_srb_status srb_status;
	union c_pct_prf_mst_status mst_status;

	rh_printf(rh, LOG_WARNING, "initiating recovery\n");

	cxil_read_csr(rh->dev, C_PCT_PRF_SCT_STATUS,
		      &sct_status, sizeof(sct_status));
	cxil_read_csr(rh->dev, C_PCT_PRF_SPT_STATUS,
		      &spt_status, sizeof(spt_status));
	cxil_read_csr(rh->dev, C_PCT_PRF_TCT_STATUS,
		      &tct_status, sizeof(tct_status));
	cxil_read_csr(rh->dev, C_PCT_PRF_SMT_STATUS,
		      &smt_status, sizeof(smt_status));
	if (rh->is_c1)
		cxil_read_csr(rh->dev, C1_PCT_PRF_TRS_STATUS,
			      &c1_trs_status, sizeof(c1_trs_status));
	else
		cxil_read_csr(rh->dev, C2_PCT_PRF_TRS_STATUS,
			      &c2_trs_status, sizeof(c2_trs_status));
	cxil_read_csr(rh->dev, C_PCT_PRF_SRB_STATUS,
		      &srb_status, sizeof(srb_status));
	cxil_read_csr(rh->dev, C_PCT_PRF_MST_STATUS,
		      &mst_status, sizeof(mst_status));

	rh_printf(rh, LOG_WARNING,
		  " sct_in_use=%d, spt_in_use=%d, tct_in_use=%d, smt_in_use=%d\n",
		  sct_status.sct_in_use, spt_status.spt_in_use,
		  tct_status.tct_in_use, smt_status.smt_in_use);

	rh_printf(rh, LOG_WARNING,
		  " trs_in_use=%d, srb_in_use=%d, mst_in_use=%d\n",
		  rh->is_c1 ? c1_trs_status.trs_in_use : c2_trs_status.trs_in_use,
		  srb_status.srb_in_use, mst_status.mst_in_use);

	if (spt_status.spt_in_use)
		spt_recovery(rh);

	if (sct_status.sct_in_use)
		sct_recovery(rh);

	if (tct_status.tct_in_use) {
		if (rh->is_c1)
			c1_tct_recovery(rh);
		else
			c2_tct_recovery(rh);
	}

	rh_printf(rh, LOG_WARNING, "recovery started\n");
}
