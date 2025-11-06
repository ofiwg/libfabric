/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */

/* Cassini retry handler
 *
 * Handling of SCT timeout event. Documented in retry-sct-close.uml.
 */

#include <search.h>

#include "rh.h"

static void send_close_request(struct retry_handler *rh, struct sct_entry *sct);
static void schedule_sct_close(struct retry_handler *rh, struct sct_entry *sct);

/* Check whether the SCT is closed after a retry. A successful close
 * retry will not generate an event, so these SCTs have to be checked
 * regularly. If they timeout, a new PCT timeout event will be
 * generated, so the only state to check is whether the SCT is now
 * unused.
 */
static void check_sct_closed(struct retry_handler *rh, struct timer_list *entry)
{
	struct sct_entry *sct = container_of(entry, struct sct_entry,
					     timeout_list);
	unsigned int sct_misc_info_offset = sct_misc_info_csr(rh, sct->sct_idx);
	unsigned int nid = cxi_dfa_nid(sct->sct_cam.dfa);

	/* Is it closed now? */
	cxil_read_csr(rh->dev, sct_misc_info_offset,
		      &sct->misc_info, sizeof(sct->misc_info));

	if (sct->misc_info.sct_status == C_SCT_NOT_USED) {
		rh_printf(rh, LOG_WARNING, "sct=%u (nid=%d, mac=%s, ep=%d, vni=%d) was successfully closed after %d retries\n",
			  sct->sct_idx, nid, nid_to_mac(nid),
			  cxi_dfa_ep(sct->sct_cam.dfa),
			  sct->sct_cam.vni, sct->close_retries);

		release_sct(rh, sct);
	} else {
		const struct timeval wait_time = {
			.tv_usec = rh->base_retry_interval_us,
		};

		/* Put back on the timer list */
		timer_add(rh, &sct->timeout_list, &wait_time);
	}
}

/* Callback function for an SCT whose CLOSE retry timer expired */
static void timeout_retry_sct_close(struct retry_handler *rh,
				    struct timer_list *entry)
{
	struct sct_entry *sct = container_of(entry, struct sct_entry,
					     timeout_list);

	send_close_request(rh, sct);
}

/* Send a close request from an SCT */
static void send_close_request(struct retry_handler *rh,
			       struct sct_entry *sct)
{
	union c_pct_cfg_sct_ram3 ram3;
	union c_pct_cfg_srb_retry_ctrl srb_retry_ctrl;
	unsigned int nid = cxi_dfa_nid(sct->sct_cam.dfa);
	const union c_pct_cfg_sw_retry_src_cls_req src_cls_req = {
		.sct_idx = sct->sct_idx,
		.loaded = 1,
		.close_not_clear = 1,
	};
	const struct timeval wait_time = {
		.tv_usec = rh->base_retry_interval_us,
	};

	cxil_read_csr(rh->dev, C_PCT_CFG_SRB_RETRY_CTRL,
		      &srb_retry_ctrl, sizeof(srb_retry_ctrl));

	/* Defer additional close request if pause is asserted */
	if (srb_retry_ctrl.pcp_pause_st & (1 << sct->pcp)) {
		rh_printf(rh, LOG_WARNING, "PCP %d (%x) is paused. Deferring sct=%u (nid=%d, mac=%s, ep=%d, vni=%d) Close Req\n",
			  sct->pcp, srb_retry_ctrl.pcp_pause_st, sct->sct_idx,
			  nid, nid_to_mac(nid),
			  cxi_dfa_ep(sct->sct_cam.dfa),
			  sct->sct_cam.vni);

		sct->paused = true;
		schedule_sct_close(rh, sct);
		return;
	}

	/* Increment the close_try to prevent a delayed close
	 * response from being used.
	 */
	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM3(sct->sct_idx),
		      &ram3, sizeof(ram3));

	ram3.close_try = (ram3.close_try + 1) % 8;

	cxil_write_csr(rh->dev, C_PCT_CFG_SCT_RAM3(sct->sct_idx),
		       &ram3, sizeof(ram3));

	rh_printf(rh, LOG_WARNING, "retrying close to sct=%u (nid=%d, mac=%s, ep=%d, vni=%d, close_try=%u)\n",
		  sct->sct_idx,
		  nid, nid_to_mac(nid),
		  cxi_dfa_ep(sct->sct_cam.dfa),
		  sct->sct_cam.vni, ram3.close_try);

	/* retry the close operation */
	sct->close_retries++;

	wait_loaded_bit(rh, C_PCT_CFG_SW_RETRY_SRC_CLS_REQ,
			C_PCT_CFG_SW_RETRY_SRC_CLS_REQ__LOADED_MSK);

	cxil_write_csr(rh->dev, C_PCT_CFG_SW_RETRY_SRC_CLS_REQ,
		       &src_cls_req, sizeof(src_cls_req));

	wait_loaded_bit(rh, C_PCT_CFG_SW_RETRY_SRC_CLS_REQ,
			C_PCT_CFG_SW_RETRY_SRC_CLS_REQ__LOADED_MSK);

	sct->timeout_list.func = check_sct_closed;
	timer_add(rh, &sct->timeout_list, &wait_time);
}

/* Schedule an SCT close.
 * Increase timeout between retries.
 */
static void schedule_sct_close(struct retry_handler *rh, struct sct_entry *sct)
{
	unsigned int nid = cxi_dfa_nid(sct->sct_cam.dfa);

	if (sct->close_retries == 0 && !sct->paused) {
		send_close_request(rh, sct);
	} else {
		uint64_t usec = retry_interval_values_us[sct->close_retries];
		const struct timeval wait_time = {
			.tv_sec = usec / 1000000,
			.tv_usec = usec % 1000000,
		};

		if (sct->paused) {
			rh_printf(rh, LOG_WARNING, "sct=%u (nid=%d, mac=%s, ep=%d, vni=%d) was paused. Waiting before issuing close\n",
				  sct->sct_idx,
				  nid, nid_to_mac(nid),
				  cxi_dfa_ep(sct->sct_cam.dfa),
				  sct->sct_cam.vni);

			sct->paused = false;
		}

		rh_printf(rh, LOG_WARNING, "schedule close of sct=%u (nid=%d, mac=%s, ep=%d, vni=%d) in %lu.%06lus\n",
			  sct->sct_idx, nid, nid_to_mac(nid),
			  cxi_dfa_ep(sct->sct_cam.dfa), sct->sct_cam.vni,
			  wait_time.tv_sec, wait_time.tv_usec);

		sct->timeout_list.func = timeout_retry_sct_close;
		timer_add(rh, &sct->timeout_list, &wait_time);
	}
}

/* Process the C_PCT_SCT_TIMEOUT event */
void sct_timeout(struct retry_handler *rh, const struct c_event_pct *event)
{
	unsigned int sct_idx = event->conn_idx.padded_sct_idx.sct_idx;
	union c_pct_cfg_sct_misc_info misc_info;
	const struct sct_entry comp = {
		.sct_idx = sct_idx,
	};
	unsigned int sct_misc_info_offset = sct_misc_info_csr(rh, sct_idx);
	struct sct_entry *sct;
	struct sct_entry **ret;
	struct timeval tv;
	unsigned int nid;

	gettimeofday(&tv, NULL);

	rh_printf(rh, LOG_WARNING, "PCT SCT timeout (sct=%u, ts=%lu.%06lu)\n", sct_idx,
		  tv.tv_sec, tv.tv_usec);

	/* Determine if SCT Timeout is legitimate.
	 * There is a race condition in the PCT which could lead to a
	 * a close response being accepted after an SCT Timeout is
	 * generated. If that is the case, do not redo the close.
	 */
	cxil_read_csr(rh->dev, sct_misc_info_offset,
		      &misc_info, sizeof(misc_info));
	if (misc_info.sct_status != C_SCT_CLOSE_TIMEOUT) {
		rh_printf(rh, LOG_WARNING, "sct=%u not in close_timeout state. Ignoring Event\n",
			  sct_idx);
		rh->stats.ignored_sct_timeouts++;
		return;
	}

	/* Find the existing SCT in the tree, or create a new one */
	ret = tfind(&comp, &rh->sct_tree, sct_compare);
	if (ret) {
		sct = *ret;
		timer_del(&sct->timeout_list);
	} else {
		sct = alloc_sct(rh, sct_idx);

		/* Mark the SCT as unusable, until the retry succeeds
		 * or ultimately fails.
		 */
		set_sct_recycle_bit(rh, sct);
	}

	/* Note that the SCT has timed out */
	sct->has_timed_out = true;
	if (rh->sct_state[sct->sct_idx].pending_timeout) {
		rh_printf(rh, LOG_DEBUG, "Resetting sct=%u pending timeout.\n",
			  sct->sct_idx);
		rh->sct_state[sct->sct_idx].pending_timeout = false;
		timer_del(&rh->sct_state[sct->sct_idx].timeout_list);
	}

	nid = cxi_dfa_nid(sct->sct_cam.dfa);
	/* TODO assert nid==node->nid */

	/* Check if the SCT Timeout is for a parked NID.
	 * Immediately force close this SCT if so.
	 * We only need to do this check if max_sct_close_retries is set,
	 * otherwise we're immediately force closing anyways.
	 */

	if (max_sct_close_retries && nid_parked(rh, nid)) {
		rh_printf(rh, LOG_WARNING, "force close of sct=%u (nid=%d, mac=%s, ep=%d, vni=%d) because it is targetting nid=%d (mac=%s) which was determined to be down\n",
			  sct->sct_idx, nid, nid_to_mac(nid),
			  cxi_dfa_ep(sct->sct_cam.dfa),
			  sct->sct_cam.vni, nid, nid_to_mac(nid));

		sct->do_force_close = true;
		release_sct(rh, sct);
		return;
	}

	/* Too many retries. Force a close and park the nid. */
	if (sct->close_retries >= max_sct_close_retries) {
		rh_printf(rh, LOG_WARNING, "force close of sct=%u (nid=%d, mac=%s, ep=%d, vni=%d) after %d close retries\n",
			  sct->sct_idx, nid, nid_to_mac(nid),
			  cxi_dfa_ep(sct->sct_cam.dfa),
			  sct->sct_cam.vni, sct->close_retries);

		sct->do_force_close = true;
		release_sct(rh, sct);
	} else {
		schedule_sct_close(rh, sct);
	}
}
