/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */

/* Cassini retry handler
 *
 * Handling of TCT timeout event. Documented in tct-timeout.uml.
 */
#include <stdlib.h>
#include <search.h>
#include <err.h>

#include "rh.h"

static void check_tct_closing(struct retry_handler *rh,
			      struct timer_list *entry);
static void tct_inbound_wait(struct retry_handler *rh,
			     struct tct_entry *tct);

/* Retrieve all the current TRS_CAM entries and store them in the rh
 * structure.
 */
static void get_trs_cam_entries(struct retry_handler *rh)
{
	unsigned int idx;

	/* TODO: use DMAC to read all at once. This likely needs a new
	 * API from libcxi.
	 */
	if (rh->is_c1) {
		for (idx = 0; idx < C1_PCT_CFG_TRS_CAM_ENTRIES; idx++)
			cxil_read_csr(rh->dev, C1_PCT_CFG_TRS_CAM(idx),
				      &rh->trs_cam[idx],
				      sizeof(union c_pct_cfg_trs_cam));
	} else {
		for (idx = 0; idx < C2_PCT_CFG_TRS_CAM_ENTRIES; idx++)
			cxil_read_csr(rh->dev, C2_PCT_CFG_TRS_CAM(idx),
				      &rh->trs_cam[idx],
				      sizeof(union c_pct_cfg_trs_cam));
	}
}

/* Check whether a TCT can be closed */
static void check_tct_closing(struct retry_handler *rh,
			      struct timer_list *entry)
{
	struct tct_entry *tct = container_of(entry, struct tct_entry,
					     timeout_list);
	union c_pct_cfg_sw_sim_tgt_cls_req tgt_cls_req;
	unsigned int idx, csr;
	uint64_t tgt_cls_abort_before, tgt_cls_abort_after;
	union c_pct_cfg_tct_ram tct_ram;
	static const struct timeval wait_time = {
		.tv_sec = 1
	};
	int nb_trs_cam_entries = rh->is_c1 ?
		C1_PCT_CFG_TRS_CAM_ENTRIES : C2_PCT_CFG_TRS_CAM_ENTRIES;

	cxil_read_csr(rh->dev, C_PCT_CFG_TCT_RAM(tct->tct_idx),
		      &tct_ram, sizeof(tct_ram));

	/* Determine if TCT was used for Puts or Gets.
	 * RTL indicates this bit is set even if MST isn't used.
	 */
	if (tct_ram.last_active_mst_is_put) {

		/* Find all the TRS connected to that TCT. If a single
		 * one has the pend_rsp bit set, skip that TCT for
		 * now.
		 */
		get_trs_cam_entries(rh);

		for (idx = 0; idx < nb_trs_cam_entries; idx++) {
			const union c_pct_cfg_trs_cam *trs_cam =
				&rh->trs_cam[idx];
			union c_pct_cfg_trs_ram1 trs_ram1;

			if (trs_cam->tct_idx != tct->tct_idx)
				continue;

			/* If there is a pending TRS CAM belonging to that TCT
			 * in the clr_seqno to exp_seqno interval, exclusive
			 * of these values, then the TCT cannot be closed yet.
			 */

			/* This TCT seqnos are "typical" clr is behind exp */
			if (tct_ram.clr_seqno < tct_ram.exp_seqno) {
				/* TRS is out of the clr->exp range. Ignore */
				if (trs_cam->req_seqno < tct_ram.clr_seqno ||
				    trs_cam->req_seqno > tct_ram.exp_seqno)
					continue;
			/* This TCTs seqnos have rolled over */
			} else if (tct_ram.exp_seqno < tct_ram.clr_seqno) {
				/* TRS is out of the clr->exp range. Ignore */
				if (trs_cam->req_seqno > tct_ram.exp_seqno &&
				    trs_cam->req_seqno < tct_ram.clr_seqno)
					continue;
			/* In TCT clr_seqno can never catchup to exp_seqno */
			} else {
				continue;
			}

			cxil_read_csr(rh->dev, rh->is_c1 ? C1_PCT_CFG_TRS_RAM1(idx) :
				      C2_PCT_CFG_TRS_RAM1(idx),
				      &trs_ram1, sizeof(trs_ram1));

			if (trs_ram1.pend_rsp == 0)
				continue;

			/* This TCT cannot be freed now. Try again in
			 * 1 second.
			 * TODO: should we count the number of these
			 * retries and force the close anyway?
			 */
			rh_printf(rh, LOG_DEBUG, "TRS entry %u for TCT %u is pending\n",
				  idx, tct->tct_idx);

			timer_add(rh, &tct->timeout_list, &wait_time);

			return;
		}

		/* If IOI is enabled for C2, the TCT timeout will cause any
		 * future packets targeting that connection to be dropped.
		 * The inbound wait mechanism is used to ensure that any
		 * past requests sitting in the MST have cleared the IXE.
		 */
		if (!rh->is_c1) {
			rh_printf(rh, LOG_DEBUG, "Issuing inbound wait for PUT tct=%u\n",
				  tct->tct_idx);
			tct_inbound_wait(rh, tct);
		}
	} else {
		rh_printf(rh, LOG_DEBUG, "Issuing inbound wait for GET tct=%u\n",
		       tct->tct_idx);
		tct_inbound_wait(rh, tct);
	}

	/* Check tgt_cls_abort before and after issuing a SW force close.
	 * The value increasing after a SW force close may indicate a TRS Leak
	 */
	csr = C_CQ_STS_EVENT_CNTS(
			offsetof(struct c_cq_cntrs_group,
				 pct.tgt_cls_abort) / 8);
	cxil_read_csr(rh->dev, csr, &tgt_cls_abort_before,
		      sizeof(tgt_cls_abort_before));

	rh_printf(rh, LOG_WARNING, "closing TCT %u\n", tct->tct_idx);
	tgt_cls_req.qw = 0;
	tgt_cls_req.tct_idx = tct->tct_idx;
	tgt_cls_req.loaded = 1;

	wait_loaded_bit(rh, C_PCT_CFG_SW_SIM_TGT_CLS_REQ,
			C_PCT_CFG_SW_SIM_TGT_CLS_REQ__LOADED_MSK);

	cxil_write_csr(rh->dev, C_PCT_CFG_SW_SIM_TGT_CLS_REQ,
		       &tgt_cls_req, sizeof(tgt_cls_req));

	wait_loaded_bit(rh, C_PCT_CFG_SW_SIM_TGT_CLS_REQ,
			C_PCT_CFG_SW_SIM_TGT_CLS_REQ__LOADED_MSK);

	cxil_read_csr(rh->dev, csr, &tgt_cls_abort_after,
		      sizeof(tgt_cls_abort_after));

	if (tgt_cls_abort_after != tgt_cls_abort_before)
		rh_printf(rh, LOG_WARNING, "SW Close of tct=%u, may be responsible for change in tgt_cls_abort value (before=%lu, after=%lu)\n",
			  tct->tct_idx, tgt_cls_abort_before, tgt_cls_abort_after);

	free(tct);
}

/* Developer Note: NETCASSINI-3325, CAS-3283
 *
 * Issue inbound wait for tct
 */
static void tct_inbound_wait(struct retry_handler *rh,
			     struct tct_entry *tct)
{
	unsigned int inbound_wait_rc;

	inbound_wait_rc = cxil_inbound_wait(rh->dev);
	switch (inbound_wait_rc) {
	case 0:
		break;
	case -EHOSTDOWN:
		fatal(rh, "Inbound wait reported hw_failure for tct=%u\n",
			  tct->tct_idx);
		break;
	case -ETIMEDOUT:
		fatal(rh, "Inbound wait timed out for tct=%u\n",
			  tct->tct_idx);
		break;
	case -EALREADY:
		/* This RC will be deprecated. For now, just bail out
		 * here.
		 */
		fatal(rh, "Inbound wait was already in progress for tct=%u\n",
			  tct->tct_idx);
		break;
	default:
		fatal(rh, "Unexpected RC=%d from inbound wait for tct=%u\n",
			  inbound_wait_rc, tct->tct_idx);
		break;
	}
}

/* Process the C_PCT_TCT_TIMEOUT event */
void tct_timeout(struct retry_handler *rh, const struct c_event_pct *event)
{
	unsigned int tct_idx = event->conn_idx.tct_idx.tct_idx;
	union c_pct_cfg_tct_cam tct_cam;
	struct tct_entry *tct;
	struct timeval tv;

	gettimeofday(&tv, NULL);

	cxil_read_csr(rh->dev, C_PCT_CFG_TCT_CAM(tct_idx),
		      &tct_cam, sizeof(tct_cam));

	rh_printf(rh, LOG_WARNING, "PCT TCT timeout (tct=%u, ts=%lu.%06lu, nid=%d, mac=%s, sct=%u)\n",
		  tct_idx, tv.tv_sec, tv.tv_usec, tct_cam.sfa_nid,
		  nid_to_mac(tct_cam.sfa_nid), tct_cam.sct_idx);

	/* Check that idle timeout is not too short
	 *  CSDG 8.2.7 - Closing Target Side Connections
	 */
	if (rh->is_c1) {
		union c1_pct_cfg_tct_misc_info tct_misc;

		cxil_read_csr(rh->dev, C1_PCT_CFG_TCT_MISC_INFO(tct_idx),
			      &tct_misc, sizeof(tct_misc));
		if (!tct_cam.vld) {
			rh_printf(rh, LOG_WARNING, "TCT CAM is not valid, ignoring event\n");
			return;
		}
		if (!tct_misc.vld) {
			rh_printf(rh, LOG_NOTICE, "TCT MISC is not valid, ignoring event\n");
			return;
		}
		if (!tct_misc.to_flag) {
			rh_printf(rh, LOG_NOTICE, "TCT MISC not in timed out state, ignoring event\n");
			return;
		}
	} else {
		union c2_pct_cfg_tct_misc_info tct_misc;

		cxil_read_csr(rh->dev, C2_PCT_CFG_TCT_MISC_INFO(tct_idx),
			      &tct_misc, sizeof(tct_misc));
		if (!tct_cam.vld) {
			rh_printf(rh, LOG_NOTICE, "TCT CAM is not valid, ignoring event\n");
			return;
		}
		if (!tct_misc.vld) {
			rh_printf(rh, LOG_NOTICE, "TCT MISC is not valid, ignoring event\n");
			return;
		}
		if (!tct_misc.to_flag) {
			rh_printf(rh, LOG_NOTICE, "TCT MISC not in timed out state, ignoring event\n");
			return;
		}
	}

	tct = calloc(1, sizeof(*tct));
	if (tct == NULL)
		fatal(rh, "Cannot alloc TCT entry\n");

	tct->tct_idx = tct_idx;
	init_list_head(&tct->timeout_list.list);
	tct->timeout_list.func = check_tct_closing;

	/* Developer Note: CAS-2383
	 * Hang the TCT on a wait list */
	timer_add(rh, &tct->timeout_list, &tct_wait_time);
}
