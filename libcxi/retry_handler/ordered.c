/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */

/* Cassini retry handler
 * Handling of ordered SPTs and their attached SCT.
 */

#include <unistd.h>
#include <search.h>
#include <err.h>
#include <limits.h>

#include "rh.h"

static void timeout_check_sct(struct retry_handler *rh,
			      struct timer_list *entry);
static void timeout_check_sct_stable(struct retry_handler *rh,
				     struct timer_list *entry);

/* Set try_num in a (SW) SPT based on its response status */
static void set_try_num(struct retry_handler *rh, struct sct_entry *sct,
			struct spt_entry *spt_out,
			const struct spt_entry *spt_in)
{
	if (spt_in->misc_info.rsp_status == C_RSP_NACK_RCVD) {
		spt_out->try_num = spt_in->misc_info.req_try;

		/* If we faked a NACK, we need track that this packet
		 * originally timed out. However there is no need to
		 * increment the TRY in this case since we did it earlier.
		 */
		if (spt_in->spt_idx == sct->faked_spt_idx)
			set_spt_timed_out(rh, spt_out,
					  spt_in->misc_info.req_try);
	} else if (spt_in->misc_info.rsp_status == C_RSP_PEND &&
		   spt_in->misc_info.to_flag) {
		set_spt_timed_out(rh, spt_out, spt_in->misc_info.req_try);
		spt_out->try_num =
			(spt_in->misc_info.req_try + 1) % SPT_TRY_NUM_SIZE;
	} else {
		fatal(rh, "Attempted to set try_num on unstable SPT\n");
	}
}

/* Send partial clear from an SCT to free up resources at its target */
static void send_partial_clear(struct retry_handler *rh, struct sct_entry *sct)
{

	union c_pct_cfg_sw_retry_src_cls_req src_cls_req = {
		.sct_idx = sct->sct_idx,
		.loaded = 1,
		.close_not_clear = 0,
	};

	wait_loaded_bit(rh, C_PCT_CFG_SW_RETRY_SRC_CLS_REQ,
			C_PCT_CFG_SW_RETRY_SRC_CLS_REQ__LOADED_MSK);

	rh_printf(rh, LOG_INFO, "Sending explicit clear from sct=%d\n",
	       sct->sct_idx);

	cxil_write_csr(rh->dev, C_PCT_CFG_SW_RETRY_SRC_CLS_REQ,
		       &src_cls_req, sizeof(src_cls_req));

	wait_loaded_bit(rh, C_PCT_CFG_SW_RETRY_SRC_CLS_REQ,
			C_PCT_CFG_SW_RETRY_SRC_CLS_REQ__LOADED_MSK);

	sct->clr_sent = true;
}


/* Retry the SPTs in an SCT.
 *
 * max_count indicate the maximum number of SPTs to retry, and must be
 * at least 1.
 *
 * Return 0 when every possible SPT was retried, 1 when there was no
 * SPT to retry, and -1 when at least one of the retries failed.
 */
static int retry_spts_from_sct(struct retry_handler *rh,
			       struct sct_entry *sct, unsigned int max_count)
{
	struct c_pct_cfg_srb_retry_ptrs_entry retry_ptrs[4];
	struct spt_entry *spt;
	unsigned int num;
	unsigned int i;
	unsigned int not_retried;
	unsigned int total_retried;
	unsigned int last_timeout_idx;
	struct spt_entry *spts[4];
	struct timeval sct_delta, tv;

	total_retried = 0;
	num = 0;
	last_timeout_idx = 0;
	memset(retry_ptrs, 0, sizeof(retry_ptrs));
	sct->to_pkts_in_batch = false;
	sct->no_tct_in_batch = false;

	list_for_each_entry(spt, &sct->spt_list, list) {
		if (spt->status != STS_NEED_RETRY)
			continue;

		if (spt->current_event_to) {
			spt->to_retries++;
			sct->to_pkts_in_batch = true;
			last_timeout_idx = spt->spt_idx;
		} else {
			spt->nack_retries++;
		}

		assert(spt->ram2_valid);

		spt->status = STS_RETRIED;

		gettimeofday(&tv, NULL);
		timersub(&tv, &sct->alloc_time, &sct_delta);

		rh_printf(rh, LOG_INFO,
			"  retrying spt=%u (srb=%u, sct=%u, try_num=%d, to_retries=%d, nack_retries=%d, sct_delta=%lu.%06lus)\n",
			spt->spt_idx, spt->ram2.srb_ptr, sct->sct_idx,
			spt->try_num, spt->to_retries, spt->nack_retries,
			sct_delta.tv_sec, sct_delta.tv_usec);


		retry_ptrs[num].ptr = spt->ram2.srb_ptr;
		retry_ptrs[num].vld = 1;
		retry_ptrs[num].spt_idx = spt->spt_idx;
		retry_ptrs[num].try_num = spt->try_num;

		clock_gettime(CLOCK_MONOTONIC_RAW,
			      &rh->spt_try_ts[spt->spt_idx][spt->try_num]);

		spts[num] = spt;

		num++;
		total_retried++;
		max_count--;

		if (max_count == 0) {
			sct->batch_last_timeout = last_timeout_idx;
			break;
		}

		if (num == 4) {
			/* Issue retries 4 at a time. */
			not_retried = retry_pkt(rh, retry_ptrs, sct->pcp, sct);
			sct->spt_status_known -= num - not_retried;
			if (not_retried)
				goto retry_fail;

			num = 0;
			memset(retry_ptrs, 0, sizeof(retry_ptrs));
		}
	}

	if (num != 0) {
		not_retried = retry_pkt(rh, retry_ptrs, sct->pcp, sct);
		sct->spt_status_known -= num - not_retried;
		if (not_retried)
			goto retry_fail;
	}

	if (total_retried == 0)
		return 1;
	else
		return 0;

retry_fail:
	/* Some (or all) requests in the last retry attempt were not
	 * retried because the PCP was paused. Change the status of
	 * the SPTs that were not retried.
	 */
	for (i = num - not_retried; i < num; i++) {
		spts[i]->status = STS_NEED_RETRY;
		if (spt->current_event_to)
			spts[i]->to_retries--;
		else
			spts[i]->nack_retries--;
	}

	return -1;
}

/* Retry the SPTs in an SCT */
static void retry_sct(struct retry_handler *rh, struct sct_entry *sct)
{
	int rc;
	unsigned int max_count;
	struct timeval sct_delta, tv;
	unsigned int nid = cxi_dfa_nid(sct->sct_cam.dfa);

	gettimeofday(&tv, NULL);
	timersub(&tv, &sct->alloc_time, &sct_delta);

	rh_printf(rh, LOG_WARNING,
		  "retrying SPT chain for sct=%u (nid=%d, mac=%s, ep=%d, vni=%d sct_delta=%lu.%06lus)\n",
		  sct->sct_idx, nid, nid_to_mac(nid),
		  cxi_dfa_ep(sct->sct_cam.dfa), sct->sct_cam.vni,
		  sct_delta.tv_sec, sct_delta.tv_usec);

	/* retrying only the SPTs that have not completed yet */
	assert(sct->spt_completed < sct->num_entries);
	assert(sct->spt_status_known == sct->num_entries);

	/* Delete any associated timers */
	timer_del(&sct->timeout_list);

	if (list_empty(&sct->spt_list))
		fatal(rh, "list of SPTs is empty for sct=%u\n", sct->sct_idx);

	/* In case we've received more ACKs send an explicit clear */
	send_partial_clear(rh, sct);

	if (sct->batch_size)
		max_count = sct->batch_size;
	else
		max_count = UINT_MAX;

	rc = retry_spts_from_sct(rh, sct, max_count);
	switch (rc) {
	/* All provided packets retried */
	case 0:
		rh_printf(rh, LOG_INFO,
			  "sct=%u all retries issued\n", sct->sct_idx);
		if (!sct->has_retried) {
			sct->has_retried = 1;
			/* After we've retried the SCT no longer track info
			 * about the faked NACK. Subsequent events should
			 * only be generated by HW.
			 */
			sct->faked_spt_idx = 0;
		}
		break;
	/* There were 0 packets to retry in the SCT */
	case 1:
		fatal(rh, "Tried to retry 0 SPTs\n");
	/* Some number of SPTs weren't retried due to pause */
	case -1:
		/* Put back on the timer list */
		rh_printf(rh, LOG_WARNING,
			  "Some retries didn't occur due to pause. Schedule retry of sct=%u in %lu.%06lus\n",
			  sct->sct_idx, pause_wait_time.tv_sec,
			  pause_wait_time.tv_usec);
		timer_add(rh, &sct->timeout_list, &pause_wait_time);
		break;
	}

	return;
}

/* Schedule an SCT retry.
 * Increase timeout between retries.
 */
static void schedule_retry_sct(struct retry_handler *rh,
			       struct sct_entry *sct)
{
	struct timeval wait_time;
	unsigned int usec;
	unsigned int random_usec;
	unsigned int backoff_usec_val;
	unsigned int bit_shift_count;
	unsigned int backoff_to_in_chain = 0;
	struct spt_entry *spt;

	sct->timeout_list.func = timeout_check_sct;

	/* Timeout packets are in chain. Dynamically select timeout retry
	 * internval to be used based on number of times first timed out SPT
	 * has been retried.
	 */
	if (sct->num_to_pkts_in_chain)
		list_for_each_entry(spt, &sct->spt_list, list) {
			if (spt->status != STS_NEED_RETRY)
				continue;

			if (spt->current_event_to) {
				backoff_to_in_chain = spt->to_retries;
				break;
			}
		}
	else {
		/* Only nack in chain */
		sct->backoff_nack_only_in_chain++;
	}

	rh_printf(rh, LOG_DEBUG,
		  "sct=%u, backoff_to_in_chain=%u, backoff_nack_only_in_chain=%u\n",
		  sct->sct_idx, backoff_to_in_chain,
		  sct->backoff_nack_only_in_chain);

	/* If there are only NACKs, retry immediately until we've retried
	 * nack_backoff_start times. At that point delay will be introduced.
	 */
	if (sct->backoff_nack_only_in_chain <= nack_backoff_start &&
		   !sct->num_to_pkts_in_chain) {
		rh_printf(rh, LOG_DEBUG,
			  "sct=%u, backoff_nack_only_in_chain=%d (NACKs only). Immediately issuing retry\n",
			  sct->sct_idx, sct->backoff_nack_only_in_chain);
		retry_sct(rh, sct);
	} else {
		/* Timeouts are in the chain - get retry timing from backoff array */
		if (sct->num_to_pkts_in_chain) {
			if (backoff_to_in_chain >= MAX_SPT_RETRIES_LIMIT)
				fatal(rh, "invalid backoff_to_in_chain value: %i\n",
				      backoff_to_in_chain);

			usec = retry_interval_values_us[backoff_to_in_chain];

		/* Only NACKs in the chain - calculate exponential backoff */
		} else {
			bit_shift_count = sct->backoff_nack_only_in_chain - nack_backoff_start;
			/* Capping bit shift count to prevent overflow by bit shifting.
			 * Assuming 32 bit U.
			 */
			backoff_usec_val = 1U << (bit_shift_count < 32 ? bit_shift_count : 31);

			/* Current calculated backoff for this set of retries */
			usec = rh->base_retry_interval_us * backoff_multiplier *
				(backoff_usec_val < rh->max_backoff_usec_val ?
				backoff_usec_val : rh->max_backoff_usec_val);

			/* Add random delay */
			if (usec > 0) {
				random_usec = rand() % usec;
				if (usec <= UINT_MAX - random_usec)
					usec += random_usec;
			}
			usec = usec < rh->max_retry_interval_us ? usec : rh->max_retry_interval_us;
		}

		if (usec == 0) {
			rh_printf(rh, LOG_DEBUG,
				  "sct=%u retry interval value is 0. Immediately issuing retry\n",
				  sct->sct_idx);
			retry_sct(rh, sct);
		} else {
			wait_time.tv_sec = usec / 1000000;
			wait_time.tv_usec = usec % 1000000;

			rh_printf(rh, LOG_DEBUG,
				  "schedule retry of sct=%u in %lu.%06lus\n",
				  sct->sct_idx, wait_time.tv_sec,
				  wait_time.tv_usec);

			timer_add(rh, &sct->timeout_list, &wait_time);
		}
	}
}

/* Build the SPT chain for an SCT. The SCT should be in the SCT_RETRY
 * state. Should be called after the SCT is in a stable state.
 */
static void build_spt_chain(struct retry_handler *rh, struct sct_entry *sct)
{
	unsigned int spt_idx;
	unsigned int count;
	unsigned int nid;
	union c_pct_cfg_sct_ram2 sct_ram2;

	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		      &sct_ram2, sizeof(sct_ram2));
	spt_idx = sct_ram2.clr_head;

	/* If the SCT was in fact stable, the head shouldn't have changed */
	assert(sct->head == sct_ram2.clr_head);
	nid = cxi_dfa_nid(sct->sct_cam.dfa);
	rh_printf(rh, LOG_INFO,
		  "sct=%d is stable (nid=%d, mac=%s, ep=%d, vni=%d, clr_head=%d). Building SPTs chain.\n",
		  sct->sct_idx, nid, nid_to_mac(nid),
		  cxi_dfa_ep(sct->sct_cam.dfa), sct->sct_cam.vni, sct->head);

	count = 0;
	while (spt_idx != 0) {
		struct spt_entry tmp = {
			.spt_idx = spt_idx,
		};

		get_spt_info(rh, &tmp);

		assert(tmp.ram1.sct_idx == sct->sct_idx);
		assert(tmp.misc_info.rsp_status != C_RSP_GET_ACK_RCVD);

		/* No PCT event will be sent for successful packets  */
		if (tmp.misc_info.rsp_status == C_RSP_OP_COMP) {
			rh_printf(rh, LOG_DEBUG,
				  "  skipping spt=%d (sct=%d, rsp_status=%d)\n",
				  spt_idx, sct->sct_idx,
				  tmp.misc_info.rsp_status);
			sct->conn_established = true;
		/* PEND (and timedout) or NACK */
		} else if ((tmp.misc_info.rsp_status == C_RSP_PEND &&
			     tmp.misc_info.to_flag) ||
			    tmp.misc_info.rsp_status == C_RSP_NACK_RCVD) {
			struct spt_entry *spt;

			/* Allocates memory and copies info from tmp into spt.
			 * Use SPT handle from here on out
			 */
			spt = alloc_spt(rh, &tmp);
			if (spt == NULL)
				fatal(rh, "Cannot allocate a new SPT\n");

			if (spt->misc_info.rsp_status == C_RSP_NACK_RCVD) {
				if (spt->spt_idx != sct->faked_spt_idx) {
					spt->current_event_to = false;
				} else {
					spt->current_event_to = true;
					sct->num_to_pkts_in_chain++;
				}
			} else {
				spt->current_event_to = true;
				sct->num_to_pkts_in_chain++;
				sct->batch_last_timeout = spt_idx;
			}

			set_try_num(rh, sct, spt, spt);
			spt->sct = sct;
			list_add_tail(&spt->list, &sct->spt_list);

			rh_printf(rh, LOG_DEBUG,
				  "  adding spt=%u (sct=%u, rsp_status=%u, pkt_sent=%u, seqno=%u, eom=%u)\n",
				  spt_idx, sct->sct_idx,
				  spt->misc_info.rsp_status,
				  spt->misc_info.pkt_sent, spt->ram2.sct_seqno,
				  spt->ram0.eom);

			count++;
			sct->tail = spt_idx;
		} else {
			fatal(rh, "Tried to build chain with unstable SPT\n");
		}

		spt_idx = tmp.ram1.clr_nxt;
	}

	/* Abundance of caution, ensure clr_head still hasn't moved */
	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		      &sct_ram2, sizeof(sct_ram2));
	assert(sct->head == sct_ram2.clr_head);

	sct->num_entries = count;
	sct->spts_allocated = true;
}

/* Create the chain of SPT for an SCT */
static struct sct_entry *create_sct_entry(struct retry_handler *rh,
					  const struct spt_entry *spt_in)
{
	struct sct_entry *sct;
	union c_pct_cfg_sct_misc_info misc_info;
	union c_pct_cfg_sct_ram2 sct_ram2;
	unsigned int sct_idx = spt_in->ram1.sct_idx;
	unsigned int sct_misc_info_offset = sct_misc_info_csr(rh, sct_idx);

	/* Needed if we need to artificially move the SCT to retry state */
	union c_pct_cfg_sw_sim_src_rsp src_rsp = {
		.spt_sct_idx = spt_in->spt_idx,
		.return_code = C_RC_SEQUENCE_ERROR, /* Any NACK type will do */
		.opcode = spt_in->opcode,
		.rsp_not_gcomp = 1,
		.loaded = 1,
	};

	/* Don't modify original spt passed in from event handler. */
	struct spt_entry spt_tmp = {
		.spt_idx = spt_in->spt_idx
	};

	rh_printf(rh, LOG_WARNING,
		  "Tracking sct=%u, pending_timeout=%u\n", sct_idx,
		  rh->sct_state[sct_idx].pending_timeout);

	sct = alloc_sct(rh, sct_idx);
	gettimeofday(&sct->alloc_time, NULL);

	/* Lets assume there are only NO_MATCHING_CONN_NACKs
	 * on this SCT for now. Change this to false if we see
	 * anything else.
	 */
	sct->only_no_matching_conn_nacks = true;

	/* First, stabilize the SCT. It can be in a few possible
	 * states. Move it to retry.
	 */
	cxil_read_csr(rh->dev, sct_misc_info_offset,
		      &misc_info, sizeof(misc_info));

	/* Developer Note: CAS-3220
	 * Artificially move SCT to retry state */
	if (misc_info.sct_status != C_SCT_RETRY) {
		rh_printf(rh, LOG_DEBUG,
			  "sct=%u is in status %u. Artificially moving to RETRY.\n",
			  sct_idx, misc_info.sct_status);
		rh->stats.rh_sct_status_change++;

		/* Store info in SCT for now since we haven't allocated a
		 * permanent SPT structure for this packet yet so Ignore
		 * subsequent NACK, but still retry this packet
		 */
		sct->faked_spt_idx = spt_in->spt_idx;
		sct->faked_spt_try = spt_in->misc_info.req_try;

		/* Assert event was a timeout */
		assert(spt_in->misc_info.to_flag == 1);

		/* Update SPT Try by 1 */
		increment_rmw_spt_try(rh, &spt_tmp);

		/* Set timeout flag back to 0 */
		update_rmw_spt_to_flag(rh, &spt_tmp, false);

		/* Force response with NACK RC. */
		simulate_rsp(rh, &src_rsp);

		/* We already saw a timeout */
		sct->only_no_matching_conn_nacks = false;
	}

	cxil_read_csr(rh->dev, sct_misc_info_offset,
		      &sct->misc_info, sizeof(sct->misc_info));
	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		      &sct_ram2, sizeof(sct_ram2));

	sct->head = sct->first_pend_spt = sct_ram2.clr_head;
	sct->pcp = -1;

	/* There should be at least one SPT in the chain */
	assert(sct->head != 0);

	return sct;
}

/* Cancel all the SPTs on an SCT chain, and possibly free the SCT. */
static void cancel_spt_entries(struct retry_handler *rh,
			       struct sct_entry *sct)
{
	struct spt_entry *spt;
	bool tail;
	unsigned int nid = cxi_dfa_nid(sct->sct_cam.dfa);

	rh_printf(rh, LOG_WARNING,
		  "now cancelling all %u SPTs on the sct=%u (nid=%d, mac=%s, ep=%d, vni=%d)\n",
		  sct->num_entries, sct->sct_idx,
		  nid, nid_to_mac(nid),
		  cxi_dfa_ep(sct->sct_cam.dfa), sct->sct_cam.vni);

	sct->spt_status_known = 0;

	list_for_each_entry(spt, &sct->spt_list, list) {
		tail = (spt->list.next == &sct->spt_list);
		if (spt->status == STS_COMPLETED) {
			sct->spt_status_known++;
			continue;
		}

		if (!spt->opcode_valid)
			fatal(rh, "Opcode not set in SPT entry");

		schedule_cancel_spt(rh, spt, sct->cancel_rc);
		/* schedule_cancel_spt could potentially free the
		 * associated SCT and SPTs. This could lead to a garbage
		 * next pointer while iterating. Break out if we're on the
		 * last packet.
		 */
		if (tail)
			break;
	}
}

/* Checks if all the SPTs on an SCT chain are in a known state.
 * If not, reschedule SCT to be checked again.
 */
static bool check_sct_stable(struct retry_handler *rh, struct sct_entry *sct)
{
	unsigned int spt_idx;
	union c_pct_cfg_sct_ram2 sct_ram2;

	/* Delete existing timer. Will add another if needed */
	timer_del(&sct->timeout_list);

	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		      &sct_ram2, sizeof(sct_ram2));

again:
	if (sct->head != sct_ram2.clr_head)
		sct->head = sct->first_pend_spt = sct_ram2.clr_head;

	/* Starting from the last known pending SPT, walk until the next pending SPT */
	spt_idx = sct->first_pend_spt;

	assert(spt_idx != 0);

	while (spt_idx != 0) {
		struct spt_entry spt = {
			.spt_idx = spt_idx,
		};

		/* Read each ram/misc of SPT in pieces */
		get_spt_info(rh, &spt);

		/* Reread HW SCT Head again */
		cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
			      &sct_ram2, sizeof(sct_ram2));

		/* If SCT HEAD has moved, we may not be able to trust clr_nxt pointers
		 * Also, ensure this SPT still belongs to the current SCT
		 */
		if (sct->head != sct_ram2.clr_head) {
			rh_printf(rh, LOG_DEBUG,
				  "sct=%d SPT chain was updated, recheck from new clr_head.\n",
				  sct->sct_idx);
			goto again;
		}

		if (spt.ram1.sct_idx != sct->sct_idx) {
			fatal(rh, "sct=%d Head didn't move, but spt=%d does not belong to it.\n",
			      sct->sct_idx, spt.spt_idx);
		}

		/* SPT State is not steady. Come back later */
		if ((spt.misc_info.rsp_status == C_RSP_PEND && !spt.misc_info.to_flag)  ||
		    spt.misc_info.rsp_status == C_RSP_GET_ACK_RCVD) {
			rh_printf(rh, LOG_DEBUG,
				  "Unstable spt=%d (sct=%d, rsp_status=%d, to_flag=%d) Revisit SCT later\n",
				  spt_idx, sct->sct_idx,
				  spt.misc_info.rsp_status,
				  spt.misc_info.to_flag);

			sct->first_pend_spt = spt_idx;

			sct->timeout_list.func = timeout_check_sct_stable;
			timer_add(rh, &sct->timeout_list, &sct_stable_wait_time);
			return false;
		}

		/* RSP_NACK_RCVD || RSP_OP_COMP || (RSP_PEND && TO_FLAG) */
		spt_idx = spt.ram1.clr_nxt;
	}
	return true;
}

/* Called when something changed regarding the status of an SPT. The
 * SCT might be ready to be retried or cancelled. If not, reschedule
 * itself.
 *
 * If timedout is true, the retry was scheduled by a timer. If the SCT
 * is still good to go, the retry will happen now, otherwise it may be
 * scheduled.
 */
void check_sct_status(struct retry_handler *rh, struct sct_entry *sct,
		      bool timedout)
{
	struct spt_entry *spt;
	unsigned int nid;
	bool tct_closed = false;
	struct sct_state *sct_state = &rh->sct_state[sct->sct_idx];

	if (!sct->has_retried)
		goto known;

	/* Go through the SCT chain. It is possible that some SPTs that were
	 * in pending state are now completed. These would not have generated
	 * events.
	 */
	list_for_each_entry(spt, &sct->spt_list, list) {
		if (spt->status != STS_PENDING)
			continue;

		cxil_read_csr(rh->dev,
			      spt_misc_info_csr(rh, spt->spt_idx),
			      &spt->misc_info, sizeof(spt->misc_info));

		/* That SPT was not retried yet. So if it completed,
		 * it means an out of order ack arrived.
		 */
		if (spt->misc_info.rsp_status == C_RSP_OP_COMP) {
			rh_printf(rh, LOG_INFO,
				  "spt=%u has completed out of order (sct=%u)\n",
				  spt->spt_idx, sct->sct_idx);
			spt->status = STS_COMPLETED;
			sct->spt_status_known++;
			sct->spt_completed++;
		} else {
			/* No need to check the rest of the
			 * chain. It's expensive and not helpful.
			 */
			break;
		}
	}

	/* If not all statuses are known come back later. */
	if (sct->spt_status_known != sct->num_entries) {

		/* If there is no existing timer program one in case some acks show up. */
		if (!timer_is_set(&sct->timeout_list)) {
			static const struct timeval wait_time = {
				.tv_usec = 100000, /* TODO: SPT timeout epoch */
			};

			sct->timeout_list.func = timeout_check_sct;
			timer_add(rh, &sct->timeout_list, &wait_time);
		}
		return;
	}

	/* All the statuses are known, act on it */
known:

	/* All Status are completed. We are done with this SCT */
	if (sct->spt_completed == sct->num_entries) {
		unsigned int nid = cxi_dfa_nid(sct->sct_cam.dfa);
		if (sct->cancel_spts)
			rh_printf(rh, LOG_WARNING,
				  "cancel completed for sct=%u (nid=%u, mac=%s)\n",
				  sct->sct_idx, nid, nid_to_mac(nid));
		else
			rh_printf(rh, LOG_WARNING,
				  "retry completed for sct=%u (nid=%u, mac=%s)\n",
				  sct->sct_idx, nid, nid_to_mac(nid));

		/* Start timer for how long this SCT should wait for a timeout
		 * before we automatically clear that bit
		 */
		if (rh->sct_state[sct->sct_idx].pending_timeout)
			timer_add(rh, &rh->sct_state[sct->sct_idx].timeout_list,
				  &peer_tct_free_wait_time);

		release_sct(rh, sct);

		return;
	}

	/* Check additional cancellation conditions */
	list_for_each_entry(spt, &sct->spt_list, list) {

		if (spt->status == STS_COMPLETED)
			continue;

		/* NO_TARGET_CONN is only valid if an SPT is seqno=1 && SOM.
		 * Can't get SOM info.
		 */
		if (spt->nack_rc == C_RC_NO_TARGET_CONN &&
		    spt->ram2.sct_seqno != 1)
			fatal(rh, "sct=%u got NO_TARGET_CONN for spt=%u which wasn't seqno 1\n",
			       sct->sct_idx, spt->spt_idx);

		/* NO_MATCHING_CONN is normal if the initial packet to establish
		 * a connection timed out, etc. However, if we know a connection
		 * was established and then see NO_MATCHING_CONN this means the
		 * TCT disappeared.
		 */
		if (!spt->current_event_to &&
		    spt->nack_rc == C_RC_NO_MATCHING_CONN) {
			if (spt->ram2.sct_seqno == 1) {
				rh_printf(rh, LOG_DEBUG,
					  "spt=%u (seqno=1) of sct=%u, got NO_MATCHING_CONN. Seqno may have wrapped. Continuing\n",
					  spt->spt_idx, sct->sct_idx);
			} else {
				if (sct->conn_established) {
					rh_printf(rh, LOG_DEBUG,
						  "spt=%u (seqno=%u) of sct=%u, got NO_MATCHING_CONN, but this connection was previously established.\n",
						  spt->spt_idx,
						  spt->ram2.sct_seqno,
						  sct->sct_idx);
					tct_closed = true;
					break;
				/* This packet (seqno!=1) was the head of the chain when it was built.
				 * This implies previous packets in the sequence were either OP_COMP or cancelled.
				 * OP_COMP would imply a connection was established - which was handled by the previous condition.
				 * If previous packets were cancelled - a connection may not have been established, but in fact never will be, so lets cancel.
				 */
				} else if (spt->spt_idx == sct->head) {
					rh_printf(rh, LOG_DEBUG,
						  "spt=%u (seqno=%u) of sct=%u, got NO_MATCHING_CONN, but prior packets to establish connection were likely cancelled\n",
						  spt->spt_idx,
						  spt->ram2.sct_seqno,
						  sct->sct_idx);
					sct->cancel_spts = true;
					sct->cancel_rc = C_RC_CANCELED;
					break;
				} else {
					rh_printf(rh, LOG_DEBUG,
						  "spt=%u (seqno=%u) of sct=%u, got NO_MATCHING_CONN. It's unknown if a connection was ever established. Continuing.\n",
						  spt->spt_idx,
						  spt->ram2.sct_seqno,
						  sct->sct_idx);
				}
			}
		}
	}

	/* If the SCT only had NO_MATCHING_CONN_NACKs on the chain cancel */
	if (sct->only_no_matching_conn_nacks) {
		if (sct->only_no_matching_conn_retry_cnt >= MAX_ONLY_NO_MATCHING_CONN_RETRY) {
			sct->cancel_spts = true;
			sct->cancel_rc = C_RC_CANCELED;
			rh->stats.cancel_no_matching_conn++;
			rh_printf(rh, LOG_WARNING, "will close sct=%u because it only saw NO_MATCHING_CONN Nacks(retry cnt %u)\n",
				  sct->sct_idx, sct->only_no_matching_conn_retry_cnt);
		} else {
			/* Retry only_no_matching_conn at least once.
			 * It is possible for packet seqno 2 to arrive at the destination
			 * before seqno 1. The target will respond with a NO_MATCHING_CONN NACK
			 * for seqno 2 and then with an ACK for seqno 1. The seqno 2 still be
			 * retried at least once since the connection was established after the
			 * NACK was generated by the target.
			 */
			sct->only_no_matching_conn_retry_cnt++;
			rh_printf(rh, LOG_DEBUG, "Only saw NO_MATCHING_CONN Nacks for sct=%u (Retrying)\n",
				  sct->sct_idx);
		}
	}

	/* If the TCT is closed there is no need to retry */
	if (tct_closed) {
		sct->cancel_spts = true;
		sct->cancel_rc = C_RC_CANCELED;
		rh->stats.cancel_tct_closed++;
		rh_printf(rh, LOG_WARNING,
			  "will close sct=%u because its TCT has timed out\n",
			  sct->sct_idx);
	}

	/* Check if the destination is on the parked list */
	nid = cxi_dfa_nid(sct->sct_cam.dfa);
	if (switch_parked(rh, nid)) {
		sct->cancel_spts = true;
		sct->cancel_rc = C_RC_UNDELIVERABLE;
		rh_printf(rh, LOG_WARNING,
			  "closing sct=%u since target nid=%d (mac=%s) switch was determined down\n",
			  sct->sct_idx, nid, nid_to_mac(nid));
	} else if (nid_parked(rh, nid)) {
		sct->cancel_spts = true;
		sct->cancel_rc = C_RC_UNDELIVERABLE;
		rh_printf(rh, LOG_WARNING,
			  "closing sct=%u since target nid=%d (mac=%s) was determined down\n",
			  sct->sct_idx, nid, nid_to_mac(nid));
	}

	spt = list_first_entry_or_null(&sct->spt_list,
				       struct spt_entry, list);
	assert(spt != NULL);
	if (is_cq_closed(rh, spt))
		rh_printf(rh, LOG_INFO,
			  "CQ=%u associated with sct=%u is disabled.\n",
			  spt->ram0.comp_cq, sct->sct_idx);

	if (sct_state->pending_timeout) {
		union c_pct_cfg_sct_ram4 sct_ram4;
		unsigned int comp_cnt;

		cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM4(sct->sct_idx),
			      &sct_ram4, sizeof(sct_ram4));
		cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM1(sct->sct_idx),
			      &sct->ram1, sizeof(sct->ram1));

		assert(spt->ram2_valid == true);

		/* It is possible that this SCT was recycled by HW even if the
		 * RH thought a SCT timeout was pending. To determine if an SCT
		 * was recycle, the current SCT head sequence number and Put/Get
		 * count can be compared against the expected values. If there
		 * is a mismatch, the SCT was recycled and pending timeout must
		 * be cleared.
		 */
		comp_cnt = (sct->ram1.put_comp_cnt + sct_ram4.get_comp_cnt) % 2048;

		if (sct_state->dfa == sct->sct_cam.dfa &&
		    sct_state->dscp == sct->sct_cam.dscp &&
		    sct_state->vni == sct->sct_cam.vni &&
		    sct_state->mcu_group == sct->sct_cam.mcu_group &&
		    spt->ram2.sct_seqno == sct_state->seqno &&
		    comp_cnt == sct_state->req_cnt) {
			sct->cancel_spts = true;
			sct->cancel_rc = C_RC_UNDELIVERABLE;
			rh_printf(rh, LOG_WARNING,
				  "will close sct=%u because it is pending a SCT timeout.\n",
				  sct->sct_idx);
		} else {
			sct_state->pending_timeout = false;

			rh_printf(rh, LOG_DEBUG,
				  "Resetting sct=%u (cur_seqno=%u exp_seqno=%u cur_comp_cnt=%u exp_comp_cnt=%u) pending timeout.\n",
				  sct->sct_idx, spt->ram2.sct_seqno,
				  sct_state->seqno, comp_cnt,
				  sct_state->req_cnt);
		}
	}

	/* Cache SCT state used to determine if SCT is recycled without
	 * software's knowledge.
	 */
	sct_state->req_cnt = sct->ram0.req_pend_cnt;
	sct_state->dfa = sct->sct_cam.dfa;
	sct_state->vni = sct->sct_cam.vni;
	sct_state->dscp = sct->sct_cam.dscp;
	sct_state->mcu_group = sct->sct_cam.mcu_group;

	if (sct->cancel_spts) {
		cancel_spt_entries(rh, sct);
	} else if (timedout) {
		rh_printf(rh, LOG_WARNING,
			  "sct=%u scheduled retry ready\n", sct->sct_idx);
		retry_sct(rh, sct);
	} else {
		schedule_retry_sct(rh, sct);
	}
}

/* Callback function for an SCT whose retry timer expired */
static void timeout_check_sct(struct retry_handler *rh,
			      struct timer_list *entry)
{
	struct sct_entry *sct = container_of(entry, struct sct_entry,
					     timeout_list);

	check_sct_status(rh, sct, true);
}

/* Timer expiration callback function to check if SCT has stabilized */
static void timeout_check_sct_stable(struct retry_handler *rh,
				     struct timer_list *entry)
{
	struct sct_entry *sct = container_of(entry, struct sct_entry,
					     timeout_list);
	if (check_sct_stable(rh, sct)) {
		build_spt_chain(rh, sct);
		/* If all statuses are known, call check_sct_status to start retry */
		if (sct->spt_status_known == sct->num_entries)
			check_sct_status(rh, sct, false);
	}

}

/* A timeout or nack event was received for a given SPT. Build a chain
 * for an SCT, or update one.
 */
void new_status_for_spt(struct retry_handler *rh,
			const struct spt_entry *spt_in,
			const struct c_event_pct *event)
{
	const struct sct_entry sct_comp = {
		.sct_idx = spt_in->ram1.sct_idx
	};
	struct sct_entry *sct;
	struct sct_entry **sct_tmp;
	const struct spt_entry spt_comp = {
		.spt_idx = spt_in->spt_idx
	};
	struct spt_entry *spt;
	struct spt_entry **spt_tmp;

	rh_printf(rh, LOG_DEBUG,
		  "new status for SPT (spt=%u, sct=%u)\n",
		  spt_in->spt_idx, spt_in->ram1.sct_idx);

	if (event->return_code == C_RC_NO_MATCHING_CONN) {
		if (rh->sct_state[spt_in->ram1.sct_idx].pending_timeout) {
			rh_printf(rh, LOG_DEBUG,
				  "Resetting sct=%u pending timeout.\n",
				  spt_in->ram1.sct_idx);
			rh->sct_state[spt_in->ram1.sct_idx].pending_timeout = false;
		}
	}

	/* Find the existing SCT in the tree, or create a new one */
	sct_tmp = tfind(&sct_comp, &rh->sct_tree, sct_compare);
	if (sct_tmp)
		sct = *sct_tmp;
	else
		sct = create_sct_entry(rh, spt_in);

	if (!sct)
		fatal(rh, "failed to create sct chain\n");

	/* If it's a new SCT, its PCP isn't set yet */
	if (sct->pcp == -1) {
		sct->pcp = spt_in->ram0.pcp;
		/* If we generated a fake NACK, don't process further. */
		if (sct->faked_spt_idx) {
			rh_printf(rh, LOG_DEBUG,
				  "Stopped processing timeout for SPT (spt=%u, sct=%u), Expecting NACK\n",
				  spt_in->spt_idx, spt_in->ram1.sct_idx);
			return;
		}
	}

	if (sct->pcp != spt_in->ram0.pcp)
		fatal(rh, "Got different PCP: %u and %u\n",
		      sct->pcp, spt_in->ram0.pcp);


	/* Validate event. Might need to mark SCT for cancellation */
	/* If one SPT had an error, cancel the whole SCT.
	 */
	if (!sct->cancel_spts && (has_uncor(rh, spt_in))) {
		rh_printf(rh, LOG_WARNING,
			  "will close sct=%u when stable because spt=%u has uncorrectable error.\n",
			  sct->sct_idx, spt_in->spt_idx);
		sct->cancel_spts = true;
		sct->cancel_rc = C_RC_UNCOR;
	}

	if (!sct->cancel_spts && is_cq_closed(rh, spt_in))
		rh_printf(rh, LOG_DEBUG,
			  "CQ=%u associated with sct=%u is disabled.\n",
			  spt_in->ram0.comp_cq, sct->sct_idx);

	/* Update stats and send partial clear if appropriate */
	switch (event->return_code) {
	case C_RC_NO_MATCHING_CONN:
		rh->stats.nack_no_matching_conn++;
		break;
	case C_RC_NO_TARGET_MST:
		rh->stats.nack_no_target_mst++;
		sct->conn_established = true;
		break;
	case C_RC_NO_TARGET_TRS:
		rh->stats.nack_no_target_trs++;
		sct->conn_established = true;
		if (sct->batch_size > initial_batch_size)
			rh_printf(rh, LOG_DEBUG,
				  "resetting batch_size to %d (was %d)\n",
				  initial_batch_size, sct->batch_size);
		sct->batch_size = initial_batch_size;
		break;
	case C_RC_NO_TARGET_CONN:
		rh->stats.nack_no_target_conn++;
		sct->no_tct_in_batch = true;
		break;
	case C_RC_SEQUENCE_ERROR:
		/* If this was actually a timeout - conn may not have been established */
		if (spt_in->spt_idx != sct->faked_spt_idx) {
			sct->conn_established = true;
			/* This will disagree with HW stats - but more is more
			 * accurate if RH counters are consulted */
			rh->stats.nack_sequence_error++;
		}
		break;
	case C_RC_RESOURCE_BUSY:
		rh->stats.nack_resource_busy++;
		/* Consider edge case where Resource Busy was for TCT that was being cleared */
		sct->conn_established = true;
		break;
	case C_RC_TRS_PEND_RSP:
		rh->stats.nack_trs_pend_rsp++;
		sct->conn_established = true;
		break;
	case C_RC_INVALID_AC:
		/* This is a fake code that only the recovery code
		 * sets. The HW will not issue it.
		 */
		break;
	}

	/* Track whether we've only seen NO_MATCHING_CONN NACKs */
	if (sct->only_no_matching_conn_nacks &&
	    (spt_in->misc_info.to_flag ||
	     event->return_code != C_RC_NO_MATCHING_CONN)) {
		rh_printf(rh, LOG_DEBUG,
			  "clearing sct=%u no_matching_conn flag\n",
			  sct->sct_idx);
		sct->only_no_matching_conn_nacks = false;
	}

	/* If NACK_RCVD this means NID is reachable.
	 * Release NID from parked list if applicable.
	 */
	if (spt_in->misc_info.rsp_status == C_RSP_NACK_RCVD &&
	    (spt_in->spt_idx != sct->faked_spt_idx)) {
		nid_tree_del(rh, cxi_dfa_nid(sct->sct_cam.dfa));
	}

	/* Only these types of packets will be tracked on an SCT.
	 * Keep track of them as they arrive
	 */
	if ((spt_in->misc_info.rsp_status == C_RSP_NACK_RCVD) ||
	    (spt_in->misc_info.rsp_status == C_RSP_PEND &&
	     spt_in->misc_info.to_flag))
		sct->spt_status_known++;

	/* Allocate SPTs to the SCT when chain is stable */
	if (!sct->spts_allocated) {
		if (check_sct_stable(rh, sct))
			build_spt_chain(rh, sct);
		else
			return;
	}

	/* May need to wait for additional events even on a stable SCT */
	if (!sct->has_retried) {
		if (sct->spt_status_known == sct->num_entries)
			check_sct_status(rh, sct, false);
		return;
	}

	/* Find the SPT entry */
	spt_tmp = tfind(&spt_comp, &rh->spt_tree, spt_compare);
	if (spt_tmp == NULL)
		fatal(rh, "Got event for sct=%u, but spt=%u was not found\n",
		      sct->sct_idx, spt_in->spt_idx);

	spt = *spt_tmp;

	/* SPT Sanity Checks */
	if (spt->sct == NULL)
		fatal(rh,
		      "Got event for sct=%u, but spt=%u doesn't belong to an SCT\n",
		      sct->sct_idx, spt_in->spt_idx);

	if (sct != spt->sct)
		fatal(rh,
		      "Got event for sct=%u, but spt=%u belongs to sct=%u\n",
		      sct->sct_idx, spt_in->spt_idx, spt->sct->sct_idx);

	if (spt->status == STS_COMPLETED)
		fatal(rh,
		      "Got event for sct=%u, but spt=%u was completed\n",
		      sct->sct_idx, spt->spt_idx);

	assert(spt->ram2_valid == spt_in->ram2_valid);

	if ((spt_in->misc_info.rsp_status == C_RSP_NACK_RCVD) ||
	    (spt_in->misc_info.rsp_status == C_RSP_PEND &&
	     spt_in->misc_info.to_flag)) {
		spt->status = STS_NEED_RETRY;
	} else {
		fatal(rh,
		      "Got update with unexpected status for spt=%u (rsp_status=%d, to_flag=%d)\n",
		      spt_in->spt_idx, spt_in->misc_info.rsp_status,
		      spt_in->misc_info.to_flag);
	}

	set_try_num(rh, sct, spt, spt_in);

	/* Store information from event */
	if (event->pct_event_type == C_PCT_REQUEST_NACK) {
		/* Normal NACK */
		if (spt->spt_idx != sct->faked_spt_idx) {
			if (spt->current_event_to) {
				assert(sct->num_to_pkts_in_chain > 0);
				sct->num_to_pkts_in_chain--;
			}
			spt->nack_rc = event->return_code;
			spt->current_event_to = false;

			if (event->return_code == C_RC_NO_MATCHING_CONN) {
				spt->no_matching_conn_nacks++;
				if (sct->no_tct_in_batch)
					spt->benign_no_matching_conn++;
			}
			if (event->return_code == C_RC_RESOURCE_BUSY)
				spt->resource_busy_nacks++;
			if (event->return_code == C_RC_TRS_PEND_RSP)
				spt->trs_pend_rsp_nacks++;
		/* Faked NACK */
		} else {
			/* Chain shouldn't have been built yet, so don't
			 * increment sct related info yet. But track this as a
			 * timeout.
			 */
			spt->current_event_to = true;
			rh->stats.event_nack--;
		}
	} else if (event->pct_event_type == C_PCT_REQUEST_TIMEOUT) {
		if (!spt->current_event_to)
			sct->num_to_pkts_in_chain++;
		spt->current_event_to = true;
	} else {
		fatal(rh, "Got bad event type %d\n", event->pct_event_type);
	}

	/* Store information from spt_in TODO do we need to copy RAMs and CAMs? */

	if (sct->cancel_spts == false) {
		/* That SPT has already been retried too many
		 * times due to timeouts. Cancel the whole SCT.
		 */
		if (spt->to_retries >= max_spt_retries) {
			rh_printf(rh, LOG_WARNING,
				  "will close sct=%u. spt=%u was retried due to timeouts %u times and nacks %u times\n",
				  sct->sct_idx, spt->spt_idx,
				  spt->to_retries, spt->nack_retries);
			sct->cancel_spts = true;
			sct->cancel_rc = C_RC_UNDELIVERABLE;
		}
		/* Repeated NO_MATCHING_CONN Nacks
		 * There might be an edge case RH hasn't covered
		 */
		if ((spt->no_matching_conn_nacks - spt->benign_no_matching_conn) >=
		    max_no_matching_conn_retries) {
			rh_printf(rh, LOG_WARNING,
				  "will close sct=%u. spt=%u received NO_MATCHING_CONN nacks %u (or more) times\n",
				  sct->sct_idx, spt->spt_idx,
				  spt->no_matching_conn_nacks);
			sct->cancel_spts = true;
			sct->cancel_rc = C_RC_CANCELED;
			rh->stats.cancel_no_matching_conn++;
		}
		/* Repeated RESOURCE_BUSY Nacks could indicate a HW
		 * resource leak occurred. Don't indefinitely retry
		 */
		if (spt->resource_busy_nacks >=
		    max_resource_busy_retries) {
			rh_printf(rh, LOG_WARNING,
				  "will close sct=%u. spt=%u received RESOURCE_BUSY nacks %u times\n",
				  sct->sct_idx, spt->spt_idx,
				  spt->resource_busy_nacks);
			sct->cancel_spts = true;
			sct->cancel_rc = C_RC_CANCELED;
			rh->stats.cancel_resource_busy++;
		}
		/* Repeated TRS_PEND_RSP Nacks could indicate a HW
		 * resource leak occurred. Don't indefinitely retry
		 */
		if (spt->trs_pend_rsp_nacks >=
			max_trs_pend_rsp_retries) {
			rh_printf(rh, LOG_WARNING,
				  "will close sct=%u. spt=%u received TRS_PEND_RSP nacks %u times\n",
				  sct->sct_idx, spt->spt_idx,
				  spt->trs_pend_rsp_nacks);
			sct->cancel_spts = true;
			sct->cancel_rc = C_RC_CANCELED;
			rh->stats.cancel_trs_pend_rsp++;
		}
	}

	check_sct_status(rh, sct, false);
}
