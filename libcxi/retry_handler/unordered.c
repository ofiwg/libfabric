/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */

/* Cassini retry handler
 * Handling of unordered SPTs.
 */

#include <search.h>

#include "rh.h"

static void timeout_retry_unordered_pkt(struct retry_handler *rh,
					struct timer_list *timer);

/* Retry an unordered packet */
static void retry_unordered_pkt(struct retry_handler *rh,
				struct spt_entry *spt)
{
	struct c_pct_cfg_srb_retry_ptrs_entry retry_ptrs[4] = {};
	unsigned int rc;

	assert(spt->status == STS_NEED_RETRY);
	assert(spt->ram2_valid);

	/* Store the retry request */
	retry_ptrs[0].spt_idx = spt->spt_idx;
	retry_ptrs[0].ptr = spt->ram2.srb_ptr;
	retry_ptrs[0].vld = 1;
	retry_ptrs[0].try_num = (spt->misc_info.req_try + 1) % SPT_TRY_NUM_SIZE;

	clock_gettime(CLOCK_MONOTONIC_RAW,
			&rh->spt_try_ts[spt->spt_idx][retry_ptrs[0].try_num]);

	spt->to_retries++;

	rh_printf(rh, LOG_NOTICE, "retrying unordered packet (spt=%u, srb=%u, try_num=%u, to_retries=%d)\n",
		  spt->spt_idx, spt->ram2.srb_ptr, retry_ptrs[0].try_num,
		  spt->to_retries);

	rc = retry_pkt(rh, retry_ptrs, spt->ram0.pcp, NULL);
	assert(rc == 0 || rc == 1);

	if (rc == 0) {
		spt->status = STS_RETRIED;
	} else {
		spt->to_retries--;
		timer_add(rh, &spt->timeout_list, &pause_wait_time);
	}
}

/* Schedule an SPT retry.
 * Increase timeout between retries.
 */
static void schedule_retry_unordered_pkt(struct retry_handler *rh,
					 struct spt_entry *spt)
{
	unsigned int usec;

	spt->timeout_list.func = timeout_retry_unordered_pkt;
	usec = (unorder_pkt_min_retry_delay > retry_interval_values_us[spt->to_retries]) ?
		unorder_pkt_min_retry_delay : retry_interval_values_us[spt->to_retries];

	if (usec == 0) {
		retry_unordered_pkt(rh, spt);
	} else {
		const struct timeval wait_time = {
			.tv_sec = usec / 1000000,
			.tv_usec = usec % 1000000,
		};

		rh_printf(rh, LOG_DEBUG, "schedule retry of unordered spt=%u in %lu.%06lus\n",
			  spt->spt_idx, wait_time.tv_sec, wait_time.tv_usec);
		timer_add(rh, &spt->timeout_list, &wait_time);
	}
}

/* Callback function for an SPT whose retry timer expired */
static void timeout_retry_unordered_pkt(struct retry_handler *rh,
					struct timer_list *entry)
{
	struct spt_entry *spt = container_of(entry, struct spt_entry,
					     timeout_list);

	if (is_cq_closed(rh, spt)) {
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u because its CQ %u is disabled\n",
			  spt->spt_idx, spt->ram0.comp_cq);

		schedule_cancel_spt(rh, spt, C_RC_UNDELIVERABLE);
	} else {
		retry_unordered_pkt(rh, spt);
	}
}

/* Find an existing SPT in the SPT tree, or create a new one */
static struct spt_entry *get_unordered_spt(struct retry_handler *rh,
					   const struct spt_entry *spt_in)
{
	struct spt_entry *spt;
	struct spt_entry **ret;

	ret = tfind(spt_in, &rh->spt_tree, spt_compare);
	if (ret) {
		spt = *ret;

		/* Copy the new state */
		spt->ram0 = spt_in->ram0;
		spt->ram1 = spt_in->ram1;
		spt->ram2 = spt_in->ram2;
		spt->ram2_valid = spt_in->ram2_valid;
		spt->misc_info = spt_in->misc_info;
		spt->opcode_valid = spt_in->opcode_valid;
		spt->opcode = spt_in->opcode;
	} else {
		spt = alloc_spt(rh, spt_in);
	}

	return spt;
}

/* TIMEOUT event for a unordered SPT */
void unordered_spt_timeout(struct retry_handler *rh,
			    const struct spt_entry *spt_in,
			    const struct c_event_pct *event)
{
	/* Unordered packet */
	struct spt_entry *spt = get_unordered_spt(rh, spt_in);
	uint32_t nid = cxi_dfa_nid(spt->dfa);

	set_spt_timed_out(rh, spt, spt->misc_info.req_try);

	/* If this was a failed HRP request, always reset the
	 * return code to RC_OK, whether the request is
	 * retried or cancelled.
	 */
	if (event->return_code == C_RC_HRP_RSP_ERROR ||
	    event->return_code == C_RC_HRP_RSP_DISCARD) {
		rh_printf(rh, LOG_DEBUG, "resetting RC to OK for spt=%u because of HRP error %u\n",
			  spt->spt_idx, event->return_code);

		spt->ram2.return_code = C_RC_OK;
		cxil_write_csr(rh->dev,
			       C_PCT_CFG_SPT_RAM2(spt->spt_idx),
			       &spt->ram2, sizeof(spt->ram2));

		if (spt->ram0.req_mp && spt->ram0.eom &&
		    spt->opcode == C_CMD_PUT) {
			/* The HRP return code does not
			 * propagate to the SMT for
			 * multi-packet Puts. Set it back to
			 * RC_OK before cancelling. Not really
			 * needed for retry but still ok.
			 */
			unsigned int smt_idx = spt->ram0.smt_idx;
			union c_pct_cfg_smt_ram0 smt_ram0;

			cxil_read_csr(rh->dev,
				      C_PCT_CFG_SMT_RAM0(smt_idx),
				      &smt_ram0, sizeof(smt_ram0));
			smt_ram0.return_code = C_RC_OK;
			cxil_write_csr(rh->dev,
				       C_PCT_CFG_SMT_RAM0(smt_idx),
				       &smt_ram0, sizeof(smt_ram0));
		}
	}

	if (is_cq_closed(rh, spt)) {
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u because its CQ %u is disabled\n",
			  spt->spt_idx, spt->ram0.comp_cq);
		schedule_cancel_spt(rh, spt, C_RC_UNDELIVERABLE);
	} else if (has_uncor(rh, spt)) {
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u which has an uncorrectable error\n",
			  spt->spt_idx);
		schedule_cancel_spt(rh, spt, C_RC_UNCOR);
	} else if (switch_parked(rh, nid)) {
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u since target nid=%d (mac=%s) switch was determined down\n",
			  spt->spt_idx, nid, nid_to_mac(nid));
		schedule_cancel_spt(rh, spt, C_RC_UNDELIVERABLE);
	} else if (nid_parked(rh, nid)) {
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u since target nid=%d (mac=%s) was determined down\n",
			  spt->spt_idx, nid, nid_to_mac(nid));
		schedule_cancel_spt(rh, spt, C_RC_UNDELIVERABLE);
	} else if ((spt->opcode == C_CMD_ATOMIC ||
		    spt->opcode == C_CMD_FETCHING_ATOMIC) &&
		   event->return_code != C_RC_HRP_RSP_DISCARD) {
		/* Unordered AMOs (excluding HRP_DISCARD) cannot be retried */
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u because unordered AMO op %u can't be retried (rc: %s)\n",
			  spt->spt_idx, spt->opcode, cxi_rc_to_str(event->return_code));
		schedule_cancel_spt(rh, spt, C_RC_CANCELED);
	} else if (spt->to_retries >= max_spt_retries) {
		/* If retried too many times, cancel it. */
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u which was retried %u times\n",
			  spt->spt_idx, spt->to_retries);
		schedule_cancel_spt(rh, spt, C_RC_UNDELIVERABLE);
	} else if (spt->ram0.req_mp && rh->dead_smt[spt->ram0.smt_idx]) {
		rh_printf(rh, LOG_NOTICE, "cancelling spt=%u since smt=%u is dead\n",
			  spt->spt_idx, spt->ram0.smt_idx);
		schedule_cancel_spt(rh, spt, C_RC_CANCELED);
	} else {
		spt->status = STS_NEED_RETRY;
		schedule_retry_unordered_pkt(rh, spt);
	}
}
