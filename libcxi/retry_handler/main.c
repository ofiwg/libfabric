/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2020 Hewlett Packard Enterprise Development LP
 */

/* Cassini retry handler
 *
 * This is a new style daemon, intended to be run by systemD as a daemon of type
 * 'notify.' See https://www.freedesktop.org/software/systemd/man/daemon.html
 */
#define _GNU_SOURCE
#include "config.h"

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <err.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <search.h>
#include <getopt.h>
#include <limits.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <linux/mman.h>

#if defined(HAVE_LIBSYSTEMD)
#include <systemd/sd-daemon.h>
#include <systemd/sd-journal.h>
#endif // defined(HAVE_LIBSYSTEMD)

#include "rh.h"

static void timeout_free_spt(struct retry_handler *rh,
			     struct timer_list *entry);

/* Max packet age in usecs a packet can be in the fabric. Default takes into
 * consideration ODP timeout (1 second) plus 1 msec of fabric latency. */
unsigned int max_fabric_packet_age = 1001000;

/* Minimum number of usecs to delay the retrying an unordered packet. This
 * should be greater than or equal to max_fabric_packet_age.
 */
unsigned int unorder_pkt_min_retry_delay = 2000000;

/* If an SPT is retried due to timeouts too many times, cancel it. */
unsigned int max_spt_retries = 4;

/* If an SPT is retried due to NO_MATCHING_CONN too many times, cancel it. */
unsigned int max_no_matching_conn_retries = 12;

/* If an SPT is retried due to RESOURCE_BUSY too many times, cancel it. */
unsigned int max_resource_busy_retries = 10;

/* If an SPT is retried due to TRS_PEND_RSP too many times, cancel it. */
unsigned int max_trs_pend_rsp_retries = 10;

/* If an SCT close is retried too many times, cancel it. */
unsigned int max_sct_close_retries = 4;

/* If NO_TRS Nacks are observed retry packets in batches up to this size. */
unsigned int initial_batch_size = 4;

/* If NO_TRS Nacks are observed retry packets in batches up to this size. */
unsigned int max_batch_size = 4;

/* Exponential backoff is multiplied by this value to increase delay between retries for nacks */
unsigned int backoff_multiplier = 2;

/* Cap backoff factor to avoid overrruns when shifting */
unsigned int max_backoff_factor = 60;

/* Start exponential backoff for NACKs after this many immediate retries */
unsigned int nack_backoff_start = 3;

unsigned int user_spt_timeout_epoch;
unsigned int user_sct_idle_epoch;
unsigned int user_sct_close_epoch;
unsigned int user_tct_timeout_epoch;

/* Retry interval values for the whole retry schedule */
unsigned int retry_interval_values_us[MAX_SPT_RETRIES_LIMIT] =
	{0, 4000000, 8000000, 12000000};

/* Max allowed retry time sum will be x% of tct timeout */
unsigned int allowed_retry_time_percent = 90;

/* Deprecated timeout_backoff_factor is in config file */
bool has_timeout_backoff_factor;

/* Path to config file */
char *config_file;

/* Time to wait before cleaning a TCT. */
struct timeval tct_wait_time = { .tv_sec = 5 };

/* Time to wait before cancelling an SPT */
struct timeval cancel_spt_wait_time = { .tv_sec = 1 };

/* Time to wait for a peer TCT to have timed out and cleaned up */
struct timeval peer_tct_free_wait_time = { .tv_sec = 0 };

/* Time to wait before retrying an SPT/SCT that wasn't retried
 * because of a paused PCP.
 */
struct timeval pause_wait_time = {
	.tv_usec = 100000, /* 100ms */
};

/* Time to wait before taking a NID out of the parked list */
struct timeval down_nid_wait_time = { .tv_sec = 96 };

/* Number of down NIDs on a switch to declare a switch as dead/parked. */
unsigned int down_switch_nid_count = 3;

/* Number of undeliverable packets to a NID to be considered as down. */
unsigned int down_nid_pkt_count = 2;

/* Bitmask identifying switch ID. This may vary based on DFA algorithm
 * being deployed by the network.
 */
unsigned int switch_id_mask = 0xfffc0;

/* Time to wait before cheking SCT stability */
struct timeval sct_stable_wait_time = { .tv_usec = 100000 };

static unsigned int default_spt_timeout_epoch;
unsigned int down_nid_spt_timeout_epoch = 24;

/* Assumes only a single outstanding packet limit is used. */
static unsigned int default_get_packets_inflight;
static unsigned int default_put_limit_inflight;
static unsigned int default_ioi_ord_limit_inflight;
static unsigned int default_ioi_unord_limit_inflight;
unsigned int down_nid_get_packets_inflight = 128;
unsigned int down_nid_put_packets_inflight = 128;

/* RH Printf wrapper to allow log level specification */
void rh_printf(const struct retry_handler *rh,
	       unsigned int base_log_level,
	       const char *fmt, ...)
{
	va_list ap;
	int total_log_level;

	/* Calculate log level to actually use for this message based on the
	 * log level specified - while incorporating any log level modification
	 * made via the RH Fuse FS.
	 */
	assert(base_log_level <= LOG_DEBUG);
	total_log_level = base_log_level - rh->log_increment;
	if (total_log_level < LOG_EMERG)
		total_log_level = LOG_EMERG;
	else if (total_log_level > LOG_DEBUG)
		total_log_level = LOG_DEBUG;

	va_start(ap, fmt);
	if (!rh->log_stdout) {
	#if defined(HAVE_LIBSYSTEMD)
		sd_journal_printv(total_log_level, fmt, ap);
	#endif // defined(HAVE_LIBSYSTEMD)
	} else {
		printf("<%d>cxi_rh: ", total_log_level);
		vprintf(fmt, ap);
	}
	va_end(ap);
}

/* Given a 20 bit NID, convert it to a MAC style format.
 * This can make it easier for admins to identify peers.
 */
const char *nid_to_mac(uint32_t nid)
{
	/* Enough space for DD:EE:FF\0 */
	static char mac[9];

	/* Format in hex with 0 left padding when needed */
	snprintf(mac, sizeof(mac), "%02X:%02X:%02X",
		 (nid >> 16) & 0xFF,
		 (nid >> 8) & 0xFF,
		  nid & 0xFF);

	return mac;
}

/* Poll a CSR until a given bit is 0. Generic code to poll the 4 CSRs
 * that have a "loaded" or "grp_loaded" field.
 */
void wait_loaded_bit(struct retry_handler *rh, unsigned int csr,
		     uint64_t bitmask)
{
	time_t start = time(NULL);
	uint64_t val;

	/* Poll indefinitely */
	while (true) {
		cxil_read_csr(rh->dev, csr, &val, sizeof(val));

		if ((val & bitmask) == 0)
			break;

		usleep(1);

		if (time(NULL) > start + 5) {
			/* This would be a hardware error */
			fatal(rh, "Polled too long for csr %x, bitmask %lx\n",
			      csr, bitmask);
		}
	}
}

/* c1_pct_cfg_srb_retry_grp_ctrl and c2_pct_cfg_srb_retry_grp_ctrl
 * have the same size and same address. The only differences are the
 * CLR_SEQNUM fields. Assume these CSRs are mostly identical to
 * simplify the code here.
 */
static_assert(C1_PCT_CFG_SRB_RETRY_GRP_CTRL(0) ==
	      C2_PCT_CFG_SRB_RETRY_GRP_CTRL(0),
	      "Different SRB_RETRY_GRP_CTRL offsets");
static_assert(C1_PCT_CFG_SRB_RETRY_GRP_CTRL_SIZE ==
	      C2_PCT_CFG_SRB_RETRY_GRP_CTRL_SIZE,
	      "Different SRB_RETRY_GRP_CTRL sizes");

/* Poll until a retry group is done or times out.
 *
 * Return the number of packets that were not retried. This may be
 * non-zero if the pcp is paused, or has been paused in the middle of
 * the retry.
 */
static unsigned int
wait_group_retry(struct retry_handler *rh, unsigned int grp,
		 unsigned int num_ptrs)
{
	unsigned int csr = C1_PCT_CFG_SRB_RETRY_GRP_CTRL(grp);
	union c1_pct_cfg_srb_retry_grp_ctrl grp_ctrl;
	time_t start = time(NULL);
	union c_pct_cfg_srb_retry_ctrl srb_retry_ctrl;

	do {
		cxil_read_csr(rh->dev, csr, &grp_ctrl, sizeof(grp_ctrl));
		if (grp_ctrl.grp_loaded == 0)
			return 0;

		/* TODO: CURRENT_PTR should be queried to see if
		 * progress is made. Otherwise the RH will hang for
		 * too long.
		 */

		usleep(1);
	} while (time(NULL) < start + 5);

	rh_printf(rh, LOG_DEBUG,
		  " polled grp_loaded in group %d for too long\n",
		  grp);

	/* Timed out. Set the SW_ABORT bit, then wait for HW to
	 * stabilize by polling on GRP_ACTIVE.
	 */
	srb_retry_ctrl.qw = 0;
	srb_retry_ctrl.pause_mode = 1;
	srb_retry_ctrl.sw_abort = 1;
	cxil_write_csr(rh->dev, C_PCT_CFG_SRB_RETRY_CTRL,
		       &srb_retry_ctrl, sizeof(srb_retry_ctrl));

	wait_loaded_bit(rh, C1_PCT_CFG_SRB_RETRY_GRP_CTRL(grp),
			C_PCT_CFG_SRB_RETRY_GRP_CTRL__GRP_ACTIVE_MSK);

	/* Read again GRP_LOADED, as the request may have gone
	 * through.
	 */
	cxil_read_csr(rh->dev, csr, &grp_ctrl, sizeof(grp_ctrl));
	if (grp_ctrl.grp_loaded == 0) {
		num_ptrs = 0;
	} else {
		cxil_read_csr(rh->dev, C_PCT_CFG_SRB_RETRY_CTRL,
			      &srb_retry_ctrl, sizeof(srb_retry_ctrl));

		/* Since grp_loaded is 1, there must be at least one packet
		 * that wasn't retried.
		 */
		assert(srb_retry_ctrl.current_ptr - 4 * grp < num_ptrs);
		num_ptrs -= srb_retry_ctrl.current_ptr - 4 * grp;
		assert(num_ptrs > 0);
	}

	srb_retry_ctrl.qw = 0;
	srb_retry_ctrl.pause_mode = 1;
	srb_retry_ctrl.hw_sw_sync = 1;
	cxil_write_csr(rh->dev, C_PCT_CFG_SRB_RETRY_CTRL,
		       &srb_retry_ctrl, sizeof(srb_retry_ctrl));

	return num_ptrs;
}

/* Retry one or more packets. The caller has prepared retry_ptrs with
 * 4 elements, and this function just finds the group to use.
 *
 * Return the number of packets that were not retried. This may be
 * non-zero if the pcp is paused, or has been paused in the middle of
 * the retry.
 */
unsigned int
retry_pkt(struct retry_handler *rh,
	  const struct c_pct_cfg_srb_retry_ptrs_entry retry_ptrs[],
	  unsigned int pcp, struct sct_entry *sct)
{
	union c_pct_cfg_srb_retry_ctrl srb_retry_ctrl;
	union c1_pct_cfg_srb_retry_grp_ctrl srb_retry_grp_ctrl = {};
	union c_pct_cfg_sct_ram2 sct_ram2;
	struct spt_entry *spt;
	struct spt_entry **ret;
	struct spt_entry comp = {};
	unsigned int spt_idx;
	unsigned int grp;
	unsigned int num_ptrs = retry_ptrs[0].vld + retry_ptrs[1].vld +
		retry_ptrs[2].vld + retry_ptrs[3].vld;
	unsigned int pkt_left;

	/* Pick group to use */
	cxil_read_csr(rh->dev, C_PCT_CFG_SRB_RETRY_CTRL,
		      &srb_retry_ctrl, sizeof(srb_retry_ctrl));

	assert(srb_retry_ctrl.pause_mode == 1);

	grp = srb_retry_ctrl.grp_nxt;
	wait_loaded_bit(rh, C1_PCT_CFG_SRB_RETRY_GRP_CTRL(grp),
			C_PCT_CFG_SRB_RETRY_GRP_CTRL__GRP_LOADED_MSK);

	cxil_write_csr(rh->dev, C_PCT_CFG_SRB_RETRY_PTRS(2 * grp), retry_ptrs,
		       4 * sizeof(struct c_pct_cfg_srb_retry_ptrs_entry));

	/* Start the retry */
	srb_retry_grp_ctrl.grp_loaded = 1;
	srb_retry_grp_ctrl.grp_pcp = pcp;

	if (sct) {
		srb_retry_grp_ctrl.grp_sct_idx_vld = 1;
		srb_retry_grp_ctrl.grp_sct_idx = sct->sct_idx;
		if (sct->clr_sent) {
			if (rh->is_c1) {
				srb_retry_grp_ctrl.grp_unset_clr_seqnum = 1;
			} else {
				union c2_pct_cfg_srb_retry_grp_ctrl *c2 =
					(union c2_pct_cfg_srb_retry_grp_ctrl *)
					&srb_retry_grp_ctrl;

				c2->grp_update_clr_seqnum_en = 1;
				c2->grp_update_clr_seqnum = 0;

				cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
					      &sct_ram2, sizeof(sct_ram2));
				spt_idx = sct_ram2.clr_head;
				comp.spt_idx = spt_idx;

				/* Look for the SPT in the tree */
				ret = tfind(&comp, &rh->spt_tree, spt_compare);
				if (ret == NULL)
					fatal(rh, "Update clr_seqnum for unknown spt=%u\n", spt_idx);

				spt = *ret;

				if (spt->ram2.sct_seqno > 1)
					c2->grp_update_clr_seqnum = spt->ram2.sct_seqno - 1;
			}
		}
	}

	rh_printf(rh, LOG_DEBUG, "  using retry group %d\n", grp);

	cxil_write_csr(rh->dev, C1_PCT_CFG_SRB_RETRY_GRP_CTRL(grp),
		       &srb_retry_grp_ctrl, sizeof(srb_retry_grp_ctrl));

	/* Wait for that retry to complete or fail */
	pkt_left = wait_group_retry(rh, grp, num_ptrs);
	if (pkt_left == 0)
		rh_printf(rh, LOG_DEBUG, "  success\n");
	else
		rh_printf(rh, LOG_DEBUG,
			  "  %u packets unsuccessfully retried in group %u\n",
			  pkt_left, grp);

	return pkt_left;
}

/* Compare function for the SPT tree */
int spt_compare(const void *a, const void *b)
{
	const struct spt_entry *x = a;
	const struct spt_entry *y = b;

	if (x->spt_idx < y->spt_idx)
		return -1;
	if (x->spt_idx > y->spt_idx)
		return 1;
	return 0;
}

/* Allocate a new SPT, fill it up, and insert it into the SPT tree. */
struct spt_entry *alloc_spt(struct retry_handler *rh,
			    const struct spt_entry *spt_in)
{
	struct spt_entry *spt;
	struct spt_entry **ret;

	spt = malloc(sizeof(*spt));
	if (!spt)
		fatal(rh, "Cannot alloc SPT\n");
	*spt = *spt_in;
	init_list_head(&spt->list);
	init_list_head(&spt->timeout_list.list);

	if ((spt->misc_info.rsp_status == C_RSP_PEND && spt->misc_info.to_flag == 1) ||
	    spt->misc_info.rsp_status == C_RSP_NACK_RCVD)
		spt->status = STS_NEED_RETRY;
	else
		fatal(rh, "Tried to alloc SPT for packet that might not need retry\n");

	ret = tsearch(spt, &rh->spt_tree, spt_compare);
	if (ret == NULL)
		fatal(rh, "Not enough memory for SPT entry\n");

	if (spt != *ret)
		fatal(rh, "Error inserting spt=%d (duplicate)\n",
		      spt_in->spt_idx);

	/* When doing a recovery, check whether that SPT has been
	 * retried by a previous RH. This is important because an SPT
	 * that wasn't retried will not generate a complete event on
	 * cancellation.
	 */
	if (spt->misc_info.sw_retry)
		spt->to_retries++;

	/* Developer Note: NETCASSINI-2158
	 * Always set the recycle bits to avoid re-use races.
	 * Only write the first 64-bits, since sw_recycle is bit 0.
	 */
	spt->ram0.sw_recycle = 1;
	cxil_write_csr(rh->dev, C_PCT_CFG_SPT_RAM0(spt->spt_idx),
		       &spt->ram0, sizeof(uint64_t));

	rh->stats.spt_alloc++;

	return spt;
}

/* Do a read/modify/write of spt_misc to increment req_try */
void increment_rmw_spt_try(const struct retry_handler *rh,
			   struct spt_entry *spt)
{
	cxil_read_csr(rh->dev, spt_misc_info_csr(rh, spt->spt_idx),
		      &spt->misc_info, sizeof(spt->misc_info));

	spt->misc_info.req_try =
		(spt->misc_info.req_try + 1) % SPT_TRY_NUM_SIZE;

	cxil_write_csr(rh->dev, spt_misc_info_csr(rh, spt->spt_idx),
		       &spt->misc_info, sizeof(spt->misc_info));
}

/* Do a read/modify/write of spt_misc to update to_flag */
void update_rmw_spt_to_flag(const struct retry_handler *rh,
			    struct spt_entry *spt, bool timed_out)
{
	cxil_read_csr(rh->dev, spt_misc_info_csr(rh, spt->spt_idx),
		      &spt->misc_info, sizeof(spt->misc_info));

	spt->misc_info.to_flag = timed_out ? 1 : 0;

	cxil_write_csr(rh->dev, spt_misc_info_csr(rh, spt->spt_idx),
		       &spt->misc_info, sizeof(spt->misc_info));
}

/* Simulate a response to an SPT */
void simulate_rsp(struct retry_handler *rh,
		  union c_pct_cfg_sw_sim_src_rsp *src_rsp)
{
	wait_loaded_bit(rh, C_PCT_CFG_SW_SIM_SRC_RSP,
			C_PCT_CFG_SW_SIM_SRC_RSP__LOADED_MSK);

	cxil_write_csr(rh->dev, C_PCT_CFG_SW_SIM_SRC_RSP,
		       src_rsp, sizeof(*src_rsp));

	wait_loaded_bit(rh, C_PCT_CFG_SW_SIM_SRC_RSP,
			C_PCT_CFG_SW_SIM_SRC_RSP__LOADED_MSK);
}

/* Cancel a request. */
static void cancel_spt(struct retry_handler *rh, struct spt_entry *spt)
{
	unsigned int log_level;
	unsigned int nid;
	union c_pct_cfg_sw_sim_src_rsp src_rsp = {
		.spt_sct_idx = spt->spt_idx,
		.return_code = spt->cancel_return_code,
		.opcode = spt->opcode,
		.rsp_not_gcomp = 1,
		.loaded = 1,
	};

	assert(spt->opcode_valid);
	nid = cxi_dfa_nid(spt->dfa);

	if (spt->ram0.req_order == 1) {
		log_level = spt->has_timed_out ? LOG_NOTICE : LOG_INFO;
		rh_printf(rh, log_level, "force closing (invalidating) spt=%u (sct=%u, op=%u, nid=%u, mac=%s, rc=%s)\n",
			  spt->spt_idx, spt->sct->sct_idx, spt->opcode,
			  nid,
			  nid_to_mac(nid),
			  cxi_rc_to_str(spt->cancel_return_code));

		/* Cache the next SCT sequence number to determine if a
		 * connection is unexpectedly reused.
		 */
		if (spt->spt_idx == spt->sct->tail)
			rh->sct_state[spt->sct->sct_idx].seqno = spt->sct->ram2.req_seqno;
	} else {
		rh_printf(rh, LOG_NOTICE, "force closing (invalidating) spt=%u (op=%u, nid=%u, mac=%s, rc=%s)\n",
			  spt->spt_idx, spt->opcode, nid, nid_to_mac(nid),
			  cxi_rc_to_str(spt->cancel_return_code));
	}

	/* Reset SMT status if needed. */
	if (spt->ram0.req_order && spt->ram0.req_mp && spt->ram0.eom &&
	    rh->dead_smt[spt->ram0.smt_idx]) {
		rh_printf(rh, LOG_DEBUG,
			  "spt=%u clearing dead smt=%u\n", spt->spt_idx,
			  spt->ram0.smt_idx);
		rh->dead_smt[spt->ram0.smt_idx] = false;
	} else if (spt->ram0.req_order && spt->ram0.req_mp &&
		   !rh->dead_smt[spt->ram0.smt_idx]) {
		rh_printf(rh, LOG_DEBUG,
			  "spt=%u marking dead smt=%u\n", spt->spt_idx,
			  spt->ram0.smt_idx);
		rh->dead_smt[spt->ram0.smt_idx] = true;
	}

	/* Increment the try_num to prevent delayed response from
	 * being used.
	 */
	increment_rmw_spt_try(rh, spt);

	/* Simulate the close response */
	simulate_rsp(rh, &src_rsp);

	/* Optionally simulate get response if appropriate */
	if (spt->opcode == C_CMD_GET || spt->opcode == C_CMD_NOMATCH_GET ||
	    spt->opcode == C_CMD_FETCHING_ATOMIC) {
		src_rsp.rsp_not_gcomp = 0;
		simulate_rsp(rh, &src_rsp);
	}

	if (spt->sct)
		rh->stats.pkts_cancelled_o++;
	else
		rh->stats.pkts_cancelled_u++;

	/* If an SPT hasn't been retried at least once, it
	 * won't get a retry complete event, so mark it
	 * completed.
	 */
	if (spt->to_retries + spt->nack_retries == 0) {
		spt->status = STS_COMPLETED;

		if (spt->sct) {
			struct sct_entry *sct = spt->sct;

			sct->spt_completed++;
			sct->spt_status_known++;

			/* If all the SPTs were already completed,
			 * destroy the SCT. Otherwise, the SCT will be
			 * freed when the last SPT event is received.
			 */
			if (sct->spt_completed == sct->num_entries) {
				nid = cxi_dfa_nid(sct->sct_cam.dfa);
				rh_printf(rh, LOG_WARNING, "cancel completed for sct=%u (nid=%u, mac=%s)\n",
					  sct->sct_idx, nid, nid_to_mac(nid));

				if (rh->sct_state[sct->sct_idx].pending_timeout)
					timer_add(rh, &rh->sct_state[sct->sct_idx].timeout_list,
						  &peer_tct_free_wait_time);

				release_sct(rh, sct);
				rh->stats.connections_cancelled++;
			}
		} else {
			release_spt(rh, spt);
		}
	}
}

/* Cancel an SPT whose cancel timer expired */
static void timeout_cancel_spt(struct retry_handler *rh,
			       struct timer_list *entry)
{
	struct spt_entry *spt = container_of(entry, struct spt_entry,
					     timeout_list);

	cancel_spt(rh, spt);
}

/* Reset pending_timeout bit for an SCT that was waiting for a timeout */
static void timeout_reset_sct_pending(struct retry_handler *rh,
				      struct timer_list *entry)
{
	struct sct_state *sct_state = container_of(entry, struct sct_state,
						   timeout_list);
	sct_state->pending_timeout = false;
}

/* This function is used to deliberately cause a sequence error.
 * Incrementing the seqno will cause any future packets using this SCT
 * to generate sequence errors at the target. This can also cause
 * close and clear requests to be rejected by the target.
 */
static void increment_sct_seqno(struct retry_handler *rh, struct sct_entry *sct)
{
	uint32_t new_seqno;

	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		      &sct->ram2, sizeof(sct->ram2));

	new_seqno = sct->ram2.req_seqno + 1;

	/* The expected sequence number after 4095 is 1.
	 * Skip it and go to 2.
	 */
	if (new_seqno > 4095)
		new_seqno = 2;

	rh_printf(rh, LOG_DEBUG, "Changing sct=%u seqno from: %u to: %u\n",
		  sct->sct_idx, sct->ram2.req_seqno, new_seqno);

	sct->ram2.req_seqno = new_seqno;
	cxil_write_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		       &sct->ram2, sizeof(sct->ram2));
}

/* Schedule an SPT cancellation */
void schedule_cancel_spt(struct retry_handler *rh, struct spt_entry *spt,
			 enum c_return_code return_code)
{
	struct timeval sct_delta;
	struct timeval tv;

	spt->cancel_return_code = return_code;

	/* Park the NID if the destination was unreachable */
	if (return_code == C_RC_UNDELIVERABLE)
		nid_tree_inc(rh, cxi_dfa_nid(spt->dfa));

	if (spt->has_timed_out && spt->ram0.req_order == 1) {
		/* Hold onto this packet so an explicit clear isn't
		 * sent out.
		 */
		spt->timeout_list.func = timeout_cancel_spt;
		spt->cancel_return_code = return_code;

		switch (spt->opcode) {
		case C_CMD_GET:
		case C_CMD_NOMATCH_GET:
		case C_CMD_FETCHING_ATOMIC:
			/* By policy, EOM Gets are provided with a larger wait time before being cancelled.
			 * Developer Notes: CAS-3283, NETCASSINI-3345
			 */
			if (spt->ram0.eom &&
			    !rh->sct_state[spt->sct->sct_idx].pending_timeout) {
				rh_printf(rh, LOG_DEBUG, "schedule cancel of EOM Get spt=%u (sct=%u) in %lu.%06lus\n",
					  spt->spt_idx, spt->sct->sct_idx,
					  peer_tct_free_wait_time.tv_sec,
					  peer_tct_free_wait_time.tv_usec);

				/* Add this SCT to a global table to track if cancellation is already scheduled */
				rh->sct_state[spt->sct->sct_idx].pending_timeout = true;

				/* Increment sequence number to cancel EOM GET */
				increment_sct_seqno(rh, spt->sct);

				timer_add(rh, &spt->timeout_list,
					  &peer_tct_free_wait_time);
			} else {
				if (spt->ram0.eom)
					rh_printf(rh, LOG_DEBUG, "skipping EOM Get cancellation for spt=%u (sct=%u)\n",
						  spt->spt_idx, spt->sct->sct_idx);
				cancel_spt(rh, spt);
			}
			break;

		default:
			/* If the SCT is pending timeout, all ODP policy actions
			 * were applied to all packets, and the SCT Seqno was incremented.
			 * No further policy actions required.
			 */
			if (rh->sct_state[spt->sct->sct_idx].pending_timeout) {
				rh_printf(rh, LOG_DEBUG, "skipping remaining ODP policy actions for spt=%u (sct=%u). Immediately canceling.\n",
					  spt->spt_idx, spt->sct->sct_idx);
				cancel_spt(rh, spt);
				break;
			}

			/* After the last packet in our chain
			 * modify the SCT seqnos to cause future pkts
			 * to be NACK'd.
			 */
			if (spt->spt_idx == spt->sct->tail) {
				rh->sct_state[spt->sct->sct_idx].pending_timeout =
					true;
				increment_sct_seqno(rh, spt->sct);
			}

			/* Developer Note: CAS-2953
			 * Must wait for some time for ODP on the remote node
			 * to finish and free its MST/TRS entries
			 */
			gettimeofday(&tv, NULL);
			timersub(&tv, &spt->sct->alloc_time, &sct_delta);

			/* If the SCT delta exceeds cancel_spt_wait_time, there
			 * is no need to apply policy actions since the
			 * cancel_spt_wait_time time period has been exceeded.
			 */
			if (timercmp(&sct_delta, &cancel_spt_wait_time, >)) {
				rh_printf(rh, LOG_DEBUG, "Already waited long enough for ODP policy actions for spt=%u (sct=%u). Immediately canceling.\n",
					  spt->spt_idx, spt->sct->sct_idx);
				cancel_spt(rh, spt);

			/* Schedule a cancel of the packet due to policy reasons */
			} else {
				timersub(&cancel_spt_wait_time, &sct_delta,
					 &tv);
				rh_printf(rh, LOG_DEBUG, "schedule cancel of spt=%u (sct=%u) in %lu.%06lus\n",
					  spt->spt_idx, spt->sct->sct_idx,
					  tv.tv_sec, tv.tv_usec);
				timer_add(rh, &spt->timeout_list, &tv);
			}
		}
	} else {
		cancel_spt(rh, spt);
	}
}

static bool read_srb(struct spt_entry *spt)
{
	/* Don't read the SRB if an SPT is not in VLD state. */
	if (!spt->misc_info.vld)
		return false;

	/* SRB has already been read. No need to read again. */
	if (spt->opcode_valid)
		return false;

	/* SPT has completed successfully. No need to read SRB. */
	if (spt->misc_info.rsp_status == C_RSP_OP_COMP)
		return false;

	/* SPT not in proper retry state. Do not touch SRB until state is known.
	 */
	if (spt->misc_info.rsp_status == C_RSP_PEND && !spt->misc_info.to_flag)
		return false;

	return true;
}

/* Read the SPT CSRs into the SPT structure.
 */
void get_spt_info(struct retry_handler *rh, struct spt_entry *spt)
{
	/* SPT index 0 is not valid */
	assert(spt->spt_idx != 0);

	/* -1 is treated as invalid SCT index. */
	spt->cont_sct = -1;

	cxil_read_csr(rh->dev, C_PCT_CFG_SPT_RAM0(spt->spt_idx),
		      &spt->ram0, sizeof(spt->ram0));

	cxil_read_csr(rh->dev, spt_misc_info_csr(rh, spt->spt_idx),
		      &spt->misc_info, sizeof(spt->misc_info));

	cxil_read_csr(rh->dev, C_PCT_CFG_SPT_RAM1(spt->spt_idx),
		      &spt->ram1, sizeof(spt->ram1));

	/* RAM2 is only valid when the packet has been sent */
	if (spt->misc_info.pkt_sent) {
		cxil_read_csr(rh->dev, C_PCT_CFG_SPT_RAM2(spt->spt_idx),
			      &spt->ram2, sizeof(spt->ram2));
		spt->ram2_valid = true;

		if (read_srb(spt))
			get_srb_info(rh, spt);
	}
}

/* Recycle an SPT. Its sw_recycle must be 1. */
void recycle_spt(struct retry_handler *rh, struct spt_entry *spt)
{
	const union c_pct_cfg_sw_recycle_spt recycle = {
		.spt_idx = spt->spt_idx,
		.bc = spt->ram0.bc,
	};

	assert(spt->ram0.sw_recycle == 1);

	/* Reset the sw_recycle bit. This assumes the recycle
	 * bit is in the first 64-bit word.
	 */
	spt->ram0.sw_recycle = 0;
	cxil_write_csr(rh->dev, C_PCT_CFG_SPT_RAM0(spt->spt_idx),
		       &spt->ram0, sizeof(uint64_t));

	cxil_write_csr(rh->dev, C_PCT_CFG_SW_RECYCLE_SPT,
		       &recycle, sizeof(recycle));
}

/* Free an SPT. If its entry had timed out, it is time to recycle it
 * as well.
 */
static void free_spt(struct retry_handler *rh, struct spt_entry *spt)
{
	if (spt->ram0.sw_recycle) {
		cxil_read_csr(rh->dev, spt_misc_info_csr(rh, spt->spt_idx),
			      &spt->misc_info, sizeof(spt->misc_info));

		/* Defer recycling and freeing if SPT is still valid */
		if (spt->misc_info.vld == 1) {
			const struct timeval wait_time = {
				/* TODO update. Right wait time? */
				.tv_usec = rh->spt_timeout_us,
			};
			spt->timeout_list.func = timeout_free_spt;
			timer_add(rh, &spt->timeout_list, &wait_time);
			return;
		}

		recycle_spt(rh, spt);
	}

	rh_printf(rh, LOG_DEBUG, "now freeing spt=%u\n",
		  spt->spt_idx);
	free(spt);

	rh->stats.spt_freed++;
}

static void timeout_free_spt(struct retry_handler *rh,
			     struct timer_list *entry)
{
	struct spt_entry *spt = container_of(entry, struct spt_entry,
					     timeout_list);

	free_spt(rh, spt);
}

/* After an SPT has completed, release it and possibly free it */
void release_spt(struct retry_handler *rh, struct spt_entry *spt)
{
	tdelete(spt, &rh->spt_tree, spt_compare);

	rh->stats.spt_released++;

	if (spt->has_timed_out) {
		struct timeval wait_time = {};
		uint64_t tv_usec;
		struct timeval now;
		struct timeval end;
		struct timespec nowns;
		struct timespec *spt_try_ts =
			&rh->spt_try_ts[spt->spt_idx][spt->misc_info.req_try];
		struct timeval spt_try_last_used;

		TIMESPEC_TO_TIMEVAL(&spt_try_last_used, spt_try_ts);

		tv_usec = max_fabric_packet_age;
		wait_time.tv_sec = tv_usec / 1000000;
		wait_time.tv_usec = tv_usec % 1000000;
		timeradd(&spt_try_last_used, &wait_time, &end);

		/* Starting from the first timeout for that SPT, add
		 * the timeout. If it's before now, the SPT can be
		 * recycled. Otherwise start a timer with the
		 * remaining time to wait.
		 */
		clock_gettime(CLOCK_MONOTONIC_RAW, &nowns);
		TIMESPEC_TO_TIMEVAL(&now, &nowns);

		if (timercmp(&now, &end, >)) {
			free_spt(rh, spt);
		} else {
			timersub(&end, &now, &wait_time);
			if (spt->ram0.req_order == 1)
				rh_printf(rh, LOG_DEBUG, "waiting %lu.%06lus before freeing spt=%u try=%u (sct=%u)\n",
					  wait_time.tv_sec, wait_time.tv_usec,
					  spt->spt_idx, spt->misc_info.req_try,
					  spt->sct->sct_idx);
			else
				rh_printf(rh, LOG_DEBUG, "waiting %lu.%06lus before freeing spt=%u try=%u\n",
					  wait_time.tv_sec, wait_time.tv_usec,
					  spt->spt_idx,
					  spt->misc_info.req_try);

			spt->timeout_list.func = timeout_free_spt;
			timer_add(rh, &spt->timeout_list, &wait_time);

			rh->stats.spt_free_deferred++;
		}
	} else {
		free_spt(rh, spt);
	}
}

static void modify_spt_timeout(struct retry_handler *rh, int spt_timeout_epoch)
{
	union c_pct_cfg_timing pct_cfg_timing;
	int cur_spt_timeout_epoch;

	cxil_read_csr(rh->dev, C_PCT_CFG_TIMING, &pct_cfg_timing,
		      sizeof(pct_cfg_timing));

	cur_spt_timeout_epoch = pct_cfg_timing.spt_timeout_epoch_sel;
	pct_cfg_timing.spt_timeout_epoch_sel = spt_timeout_epoch;

	cxil_write_csr(rh->dev, C_PCT_CFG_TIMING, &pct_cfg_timing,
		       sizeof(pct_cfg_timing));

	rh_printf(rh, LOG_DEBUG, "SPT timeout change from %d to %d\n",
		  cur_spt_timeout_epoch, spt_timeout_epoch);
}

static void modify_mcu_inflight(struct retry_handler *rh,
				unsigned int get, unsigned int put,
				unsigned int ioi_ord, unsigned int ioi_unord)
{
	union c_oxe_cfg_outstanding_limit limit = {
		.get_limit = get,
		.put_limit = put,
		.ioi_ord_limit = ioi_ord,
		.ioi_unord_limit = ioi_unord,
	};

	cxil_write_csr(rh->dev, C_OXE_CFG_OUTSTANDING_LIMIT(0), &limit,
		       sizeof(limit));
}

#define SWITCH_ID(nid) ((nid) & switch_id_mask)

static int switch_compare(const void *a, const void *b)
{
	const struct switch_entry *x = (const struct switch_entry *)a;
	const struct switch_entry *y = (const struct switch_entry *)b;

	if (x->id < y->id)
		return -1;
	if (x->id > y->id)
		return 1;
	return 0;
}

static void switch_tree_inc(struct retry_handler *rh, uint32_t nid)
{
	struct switch_entry *entry;
	struct switch_entry **ret;
	const struct switch_entry comp = {
		.id = SWITCH_ID(nid),
	};

	if (down_switch_nid_count == 0)
		return;

	ret = tfind(&comp, &rh->switch_tree, switch_compare);
	if (!ret) {
		entry = calloc(1, sizeof(*entry));
		if (!entry)
			fatal(rh, "Cannot allocate switch entry\n");

		entry->id = comp.id;

		ret = tsearch(entry, &rh->switch_tree, switch_compare);
		if (!ret)
			fatal(rh, "Not enough memory for switch tree entry\n");

		if (entry != *ret)
			fatal(rh, "Error inserting switch entry into tree\n");

		rh->switch_tree_count++;
		if (rh->switch_tree_count > rh->stats.max_switch_tree_count)
			rh->stats.max_switch_tree_count = rh->switch_tree_count;

		rh_printf(rh, LOG_DEBUG, "Tracking switch=%d\n", entry->id);
	} else {
		entry = *ret;
	}

	entry->count++;
	if (entry->count == down_switch_nid_count) {
		entry->parked = true;
		rh_printf(rh, LOG_WARNING,
			  "Adding switch=%d to parked switches\n", entry->id);

		rh->parked_switches++;
		if (rh->parked_switches > rh->stats.max_parked_switches)
			rh->stats.max_parked_switches = rh->parked_switches;
	}
}

static void switch_tree_dec(struct retry_handler *rh, uint32_t nid)
{
	struct switch_entry *entry;
	struct switch_entry **ret;
	struct switch_entry comp = {
		.id = SWITCH_ID(nid),
	};

	if (down_switch_nid_count == 0)
		return;

	ret = tfind(&comp, &rh->switch_tree, switch_compare);
	if (!ret)
		fatal(rh, "Cannot find switch=%d in tree\n", comp.id);

	entry = *ret;

	if (entry->count == down_switch_nid_count) {
		entry->parked = false;
		rh_printf(rh, LOG_WARNING,
			  "Removing switch=%d from parked switches\n",
			  entry->id);

		rh->parked_switches--;
	}
	entry->count--;

	if (entry->count == 0) {
		rh->switch_tree_count--;
		tdelete(entry, &rh->switch_tree, switch_compare);
		rh_printf(rh, LOG_DEBUG, "Deleting switch=%d\n", entry->id);
		free(entry);
	}
}

bool switch_parked(struct retry_handler *rh, uint32_t nid)
{
	struct switch_entry *entry;
	struct switch_entry **ret;
	const struct switch_entry comp = {
		.id = SWITCH_ID(nid),
	};

	if (rh->switch_tree_count == 0)
		return false;

	ret = tfind(&comp, &rh->switch_tree, switch_compare);
	if (!ret)
		return false;

	entry = *ret;

	return entry->parked;
}

static int nid_compare(const void *a, const void *b)
{
	const struct nid_entry *x = (const struct nid_entry *)a;
	const struct nid_entry *y = (const struct nid_entry *)b;

	if (x->nid < y->nid)
		return -1;
	if (x->nid > y->nid)
		return 1;
	return 0;
}

static struct nid_entry *nid_find(struct retry_handler *rh, uint32_t nid)
{
	struct nid_entry **ret;
	struct nid_entry comp = {
		.nid = nid,
	};

	ret = tfind(&comp, &rh->nid_tree, nid_compare);
	if (!ret)
		return NULL;

	return *ret;
}

void nid_tree_del(struct retry_handler *rh, uint32_t nid)
{
	struct nid_entry *entry;

	/* Feature is disabled - do nothing */
	if (down_nid_wait_time.tv_sec == 0 &&
	    down_nid_wait_time.tv_usec == 0)
		return;

	entry = nid_find(rh, nid);
	if (!entry)
		return;

	rh_printf(rh, LOG_WARNING,
		  "Removing nid=%d (mac=%s) from parked list\n", entry->nid,
		  nid_to_mac(entry->nid));

	/* This NID is only added to the switch tree if it has met the parked
	 * threshold.
	 */
	if (entry->parked) {
		switch_tree_dec(rh, entry->nid);

		/* Once parked NIDs count reaches zero, reset SPT and MCU
		 * configuration.
		 */
		rh->parked_nids--;
		if (rh->parked_nids == 0) {
			modify_spt_timeout(rh, default_spt_timeout_epoch);
			modify_mcu_inflight(rh, default_get_packets_inflight,
					    default_put_limit_inflight,
					    default_ioi_ord_limit_inflight,
					    default_ioi_unord_limit_inflight);
		}
	}

	timer_del(&entry->timeout_list);
	tdelete(entry, &rh->nid_tree, nid_compare);
	free(entry);

	rh->nid_tree_count--;
}

bool nid_parked(struct retry_handler *rh, uint32_t nid)
{
	struct nid_entry *entry;

	if (rh->parked_nids == 0)
		return false;

	entry = nid_find(rh, nid);
	if (!entry)
		return false;

	return entry->parked;
}

static void timeout_release_nid(struct retry_handler *rh,
				struct timer_list *entry)
{
	struct nid_entry *node =
		container_of(entry, struct nid_entry, timeout_list);
	rh_printf(rh, LOG_DEBUG, "%s called back for nid=%d (mac=%s)\n",
		  __func__, node->nid, nid_to_mac(node->nid));
	nid_tree_del(rh, node->nid);
}

static struct nid_entry *nid_alloc(struct retry_handler *rh, uint32_t nid)
{
	struct nid_entry *entry;
	struct nid_entry **ret;

	entry = calloc(1, sizeof(*entry));
	if (!entry)
		fatal(rh, "Cannot allocate nid entry\n");

	entry->nid = nid;
	entry->parked = false;
	init_list_head(&entry->timeout_list.list);
	entry->timeout_list.func = timeout_release_nid;

	ret = tsearch(entry, &rh->nid_tree, nid_compare);
	if (!ret)
		fatal(rh, "Not enough memory for nid tree entry\n");

	if (entry != *ret)
		fatal(rh, "Error inserting nid entry into tree\n");

	rh_printf(rh, LOG_WARNING, "Adding nid=%d (mac=%s) to parked list\n",
		  entry->nid, nid_to_mac(entry->nid));

	return entry;
}

/* Increment number of undeliverable packet count to a given NID. */
void nid_tree_inc(struct retry_handler *rh, uint32_t nid)
{
	struct nid_entry *entry;

	/* Feature is disabled - do nothing */
	if (down_nid_wait_time.tv_sec == 0 &&
	    down_nid_wait_time.tv_usec == 0)
		return;

	entry = nid_find(rh, nid);
	if (!entry) {
		entry = nid_alloc(rh, nid);
		rh->nid_tree_count++;
		if (rh->nid_tree_count > rh->stats.max_nid_tree_count)
			rh->stats.max_nid_tree_count = rh->nid_tree_count;

	} else {
		timer_del(&entry->timeout_list);
	}

	/* If NID has met the threshold for being parked, add NID to down
	 * switch tree and modify OXE and PCT settings.
	 *
	 * Note: pkt_count is never decrement.
	 */
	entry->pkt_count++;
	if (entry->pkt_count == down_nid_pkt_count) {
		entry->parked = true;
		switch_tree_inc(rh, entry->nid);

		if (rh->parked_nids == 0) {
			modify_spt_timeout(rh, down_nid_spt_timeout_epoch);
			modify_mcu_inflight(rh, down_nid_get_packets_inflight,
					    down_nid_put_packets_inflight,
					    down_nid_put_packets_inflight,
					    down_nid_put_packets_inflight);
		}

		rh->parked_nids++;
		if (rh->parked_nids > rh->stats.max_parked_nids)
			rh->stats.max_parked_nids = rh->parked_nids;
	}

	timer_add(rh, &entry->timeout_list, &down_nid_wait_time);
	rh_printf(rh, LOG_DEBUG, "Re-upping park time for nid=%u (mac=%s)\n",
		  entry->nid, nid_to_mac(entry->nid));
}

/* Compare function for the SCT tree */
int sct_compare(const void *a, const void *b)
{
	const struct sct_entry *x = a;
	const struct sct_entry *y = b;

	if (x->sct_idx < y->sct_idx)
		return -1;
	if (x->sct_idx > y->sct_idx)
		return 1;
	return 0;
}

/* Allocate an SCT entry and insert into the SCT tree */
struct sct_entry *alloc_sct(struct retry_handler *rh, unsigned int sct_idx)
{
	struct sct_entry *sct;
	struct sct_entry **ret;

	sct = calloc(1, sizeof(*sct));
	if (sct == NULL)
		fatal(rh, "Cannot allocate SCT entry\n");

	sct->sct_idx = sct_idx;
	init_list_head(&sct->timeout_list.list);
	init_list_head(&sct->spt_list);

	ret = tsearch(sct, &rh->sct_tree, sct_compare);
	if (ret == NULL)
		fatal(rh, "Not enough memory for SCT entry\n");

	if (sct != *ret)
		fatal(rh, "Error inserting SCT\n");

	rh->stats.sct_alloc++;

	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM0(sct->sct_idx),
		      &sct->ram0, sizeof(sct->ram0));
	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM1(sct->sct_idx),
		      &sct->ram1, sizeof(sct->ram1));
	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM2(sct->sct_idx),
		      &sct->ram2, sizeof(sct->ram2));
	cxil_read_csr(rh->dev, C_PCT_CFG_SCT_CAM(sct->sct_idx),
		      &sct->sct_cam, sizeof(sct->sct_cam));

	return sct;
}

/* Free the SPT chain in an SCT */
void free_spt_chain(struct retry_handler *rh, struct sct_entry *sct)
{
	struct spt_entry *spt;

	while ((spt = list_first_entry_or_null(&sct->spt_list,
					       struct spt_entry, list))) {
		list_del(&spt->list);
		release_spt(rh, spt);
	}
}

static void reset_sct_deny_bit(struct retry_handler *rh, struct sct_entry *sct)
{
	cxil_read_csr(rh->dev, C1_PCT_CFG_SCT_MISC_INFO(sct->sct_idx),
		      &sct->misc_info, sizeof(sct->misc_info));
	sct->misc_info.deny_new_msg = 0;
	cxil_write_csr(rh->dev, C1_PCT_CFG_SCT_MISC_INFO(sct->sct_idx),
		       &sct->misc_info, sizeof(sct->misc_info));
}

/* Free an SCT after recycling it (if applicable) */
static void free_sct(struct retry_handler *rh, struct sct_entry *sct)
{
	if (sct->ram1.sw_recycle) {
		/* The SW_RECYCLE bit was set. Reset the bit and
		 * recycle the entry, which will invalidate it so it
		 * can be re-used.
		 */
		sct->ram1.sw_recycle = 0;
		cxil_write_csr(rh->dev, C_PCT_CFG_SCT_RAM1(sct->sct_idx),
			       &sct->ram1, sizeof(sct->ram1));

		const union c_pct_cfg_sw_recycle_sct recycle = {
			.sct_idx = sct->sct_idx,
			.bc = sct->ram1.bc,
		};

		cxil_write_csr(rh->dev, C_PCT_CFG_SW_RECYCLE_SCT,
			       &recycle, sizeof(recycle));
	}

	timer_del(&sct->timeout_list);
	tdelete(sct, &rh->sct_tree, sct_compare);
	free(sct);

	rh->stats.sct_freed++;
}

static void timeout_free_sct(struct retry_handler *rh,
			     struct timer_list *entry)
{
	struct sct_entry *sct = container_of(entry, struct sct_entry,
					     timeout_list);
	free_sct(rh, sct);
}

/* Cleanup SCT chain and issue a simulated response if applicable */
void release_sct(struct retry_handler *rh, struct sct_entry *sct)
{
	free_spt_chain(rh, sct);

	if (sct->do_force_close) {
		const union c_pct_cfg_sw_sim_src_rsp src_rsp = {
			.spt_sct_idx = sct->sct_idx,
			.return_code = C_RC_CANCELED,
			.opcode = C_CMD_CLOSE,
			.loaded = 1,
			.rsp_not_gcomp = 1,
		};
		union c_pct_cfg_sct_ram3 ram3;

		if (sct->accel_close_event) {
			if (rh->is_c1) {
				/* Developer Note: CAS-2802
				 * Reset deny_new_msg bit.
				 */
				rh_printf(rh, LOG_WARNING, "accel close cleanup for sct=%u\n",
					  sct->sct_idx);

				reset_sct_deny_bit(rh, sct);
			}
		} else {
			/* Increment the close_try to prevent a delayed close
			 * response from being used.
			 */
			rh_printf(rh, LOG_WARNING,
				  "force closing sct=%u\n", sct->sct_idx);

			cxil_read_csr(rh->dev, C_PCT_CFG_SCT_RAM3(sct->sct_idx),
				      &ram3, sizeof(ram3));

			ram3.close_try = (ram3.close_try + 1) % 8;

			cxil_write_csr(rh->dev,
				       C_PCT_CFG_SCT_RAM3(sct->sct_idx),
				       &ram3, sizeof(ram3));
		}

		/* Simulate the close */
		wait_loaded_bit(rh, C_PCT_CFG_SW_SIM_SRC_RSP,
				C_PCT_CFG_SW_SIM_SRC_RSP__LOADED_MSK);

		cxil_write_csr(rh->dev, C_PCT_CFG_SW_SIM_SRC_RSP,
			       &src_rsp, sizeof(src_rsp));

		wait_loaded_bit(rh, C_PCT_CFG_SW_SIM_SRC_RSP,
				C_PCT_CFG_SW_SIM_SRC_RSP__LOADED_MSK);
	}

	/* If max_sct_close_retries is 0, that means SW never reissued a Close
	 * Request to the peer. In this case we need to ensure this SCT is not
	 * recycled until we know the TCT has timed out.
	 */
	if (sct->has_timed_out &&
	    max_sct_close_retries == 0) {
		sct->timeout_list.func = timeout_free_sct;
		timer_add(rh, &sct->timeout_list,
			  &peer_tct_free_wait_time);
	} else {
		free_sct(rh, sct);
	}
}

/* Check whether the CQ that issued that SPT is still active. */
bool is_cq_closed(const struct retry_handler *rh, const struct spt_entry *spt)
{
	unsigned int cq = spt->ram0.comp_cq;
	union c_cq_txq_enable txq_enable;

	cxil_read_csr(rh->dev, C_CQ_TXQ_ENABLE(cq),
		      &txq_enable, sizeof(txq_enable));

	return txq_enable.txq_enable == 0;
}

/* Check whether the SPT has an uncorrectable error. */
bool has_uncor(const struct retry_handler *rh, const struct spt_entry *spt)
{
	union c_pct_cfg_srb_cell_data srb_data;

	/* It is possible that the packet was dropped by the NIC as
	 * the result of a late and uncorrectable error in OXE or
	 * PCT. This rare case will be indicated by the EOPB bit being
	 * set in the SRB cell. The packet will have been dropped by
	 * HNI. It should not be retried.
	 */
	/* Read only the first qword */
	assert(spt->ram2_valid);
	cxil_read_csr(rh->dev, C_PCT_CFG_SRB_CELL_DATA(spt->ram2.srb_ptr << 2),
		      &srb_data, sizeof(uint64_t));
	if (srb_data.eopb)
		rh_printf(rh, LOG_WARNING,
			  "got uncorrectable error for spt=%u\n",
		       spt->spt_idx);

	return srb_data.eopb;
}

/* Called sometime after an SPT has timed out. The RH needs to remember the
 * first time that happened to start tracking the use of try nums.
 */
void set_spt_timed_out(const struct retry_handler *rh, struct spt_entry *spt,
		       const unsigned int to_try_num)
{
	if (spt->has_timed_out)
		return;

	spt->has_timed_out = true;
	spt->timed_out_try_num = to_try_num;
}

/* Get the opcode of a packet from an SRB. This word and bytes swap is
 * really ugly.
 */
void get_srb_info(struct retry_handler *rh, struct spt_entry *spt)
{
	union hni_pkt pkt;
	size_t i;
	union c_pct_cfg_srb_cell_data srb_data;


	/* TODO: read only the first (ie. last) 5 qwords */
	cxil_read_csr(rh->dev, C_PCT_CFG_SRB_CELL_DATA(spt->ram2.srb_ptr << 2),
		      &srb_data, sizeof(srb_data));

	/* Words 0 - 7 are in HW Order / Big Endian. Reverse it.
	 * Read the amount of qwords that fit in the hni_pkt
	 * structure. Currently 5 words, 7-3 (Word 7 is at index 8)
	 */
	for (i = 0; i < ARRAY_SIZE(pkt.buf); i++)
		pkt.buf[i] = be64toh(srb_data.qw[8 - i]);

	switch (pkt.hdr.ver) {
	case 4:
		/* Portals, version 4 */
		spt->opcode = pkt.h4.pkt_type.opcode;
		spt->vni = c_port_fab_hdr_get_vni(&pkt.h4.hdr);
		memcpy(&spt->dfa, &pkt.h4.hdr.dfa, sizeof(spt->dfa));
		spt->dfa = ntohl(spt->dfa);

		if (pkt.h4.pkt_type.ver_pkt_type == C_V4_PKT_CONTINUATION)
			spt->cont_sct = c_port_continuation_hdr_get_sct_idx(&pkt.cont.cont);

		break;
	case 3:			/* C_IXE_CFG_PORT_PKT.VS_VERSION */
		/* Portals, version small
		 *
		 * bit 0 is FETCH_OR_GET, bit 2 is AMO, from
		 * c_port_vs_alt_pkt_type_t.
		 */
		if (pkt.vs.pkt_type.ver_pkt_type & (1 << 2)) {
			/* Convert the atomic opcode */
			if (pkt.vs.pkt_type.ver_pkt_type & (1 << 0))
				spt->opcode = C_CMD_FETCHING_ATOMIC;
			else
				spt->opcode = C_CMD_ATOMIC;
		} else {
			spt->opcode = pkt.vs.pkt_type.opcode;
		}

		spt->vni = c_port_fab_vs_hdr_get_vni(&pkt.vs.hdr);
		memcpy(&spt->dfa, &pkt.vs.hdr.dfa, sizeof(spt->dfa));
		spt->dfa = ntohl(spt->dfa);
		break;
	default:
		fatal(rh,
		     "Retry: Invalid pkt ver %u in srb %u of length %u for spt=%u\n",
		     pkt.hdr.ver, spt->ram2.srb_ptr << 2,
		     srb_data.byte_len, spt->spt_idx);
	}

	spt->opcode_valid = true;
}

/* Set the SW_RECYCLE bit for an SCT */
void set_sct_recycle_bit(const struct retry_handler *rh, struct sct_entry *sct)
{
	sct->ram1.sw_recycle = 1;

	cxil_write_csr(rh->dev, C_PCT_CFG_SCT_RAM1(sct->sct_idx),
		       &sct->ram1, sizeof(sct->ram1));
}

/* Process C_PCT_REQUEST_NACK event */
void request_nack(struct retry_handler *rh, const struct c_event_pct *event)
{
	struct spt_entry spt = {
		.spt_idx = event->spt_idx,
	};
	struct timeval tv;
	unsigned int nid;

	gettimeofday(&tv, NULL);

	get_spt_info(rh, &spt);
	nid = cxi_dfa_nid(spt.dfa);

	rh_printf(rh, LOG_INFO, "PCT NACK event (spt=%u, smt=%u, sct=%u, seqno=%u, exp_seqno=%u, rc=%s, nid=%u, mac=%s, ep=%u, vni=%u, ts=%lu.%06lu)\n",
		  spt.spt_idx, spt.ram0.smt_idx, spt.ram1.sct_idx,
		  spt.ram2.sct_seqno, event->seq_num,
		  cxi_rc_to_str(event->return_code), nid, nid_to_mac(nid),
		  cxi_dfa_ep(spt.dfa), spt.vni,
		  tv.tv_sec, tv.tv_usec);

	if (spt.ram1.sct_idx != event->conn_idx.padded_sct_idx.sct_idx)
		fatal(rh, "Got unexpected SCT IDX in NACK (%d, %d)\n",
		      spt.ram1.sct_idx, event->conn_idx.padded_sct_idx.sct_idx);

	new_status_for_spt(rh, &spt, event);
}

/* Process C_PCT_REQUEST_TIMEOUT event */
void request_timeout(struct retry_handler *rh, const struct c_event_pct *event)
{
	struct spt_entry spt = {
		.spt_idx = event->spt_idx,
	};
	struct timeval tv;
	unsigned int nid;
	int seqno = -1;

	gettimeofday(&tv, NULL);

	get_spt_info(rh, &spt);
	nid = cxi_dfa_nid(spt.dfa);
	if (spt.ram0.req_order)
		seqno = spt.ram2.sct_seqno;

	rh_printf(rh, LOG_NOTICE, "PCT timeout event (spt=%u, smt=%u, seqno=%d, srb=%u, try=%u, ordered=%d, nid=%u, mac=%s, ep=%u, vni=%u, cont_sct=%d, ts=%lu.%06lu)\n",
		  spt.spt_idx, spt.ram0.smt_idx, seqno,
		  spt.ram2.srb_ptr, spt.misc_info.req_try, spt.ram0.req_order,
		  nid, nid_to_mac(nid), cxi_dfa_ep(spt.dfa), spt.vni,
		  spt.cont_sct, tv.tv_sec, tv.tv_usec);

	/* Event might not be valid anymore, as the response might
	 * have arrived in the meantime. See CIS "SPT Timeout".
	 */
	if (spt.misc_info.vld == 0 ||
	    spt.misc_info.rsp_status != C_RSP_PEND ||
	    spt.misc_info.to_flag != 1) {
		rh_printf(rh, LOG_NOTICE, "ignoring event (vld=%u, rsp_status=%u, to_flag=%u)\n",
			  spt.misc_info.vld, spt.misc_info.rsp_status,
			  spt.misc_info.to_flag);
		return;
	}


	if (spt.ram0.req_order == 0) {
		rh->stats.event_spt_timeout_u++;
		unordered_spt_timeout(rh, &spt, event);
	} else {
		/* Ordered packet. SCT index is needed */
		rh->stats.event_spt_timeout_o++;
		new_status_for_spt(rh, &spt, event);
	}
}

/* Free/Recycle SCT after the timer started by the
 * C_PCT_ACCEL_CLOSE_COMPLETE event has expired.
 */
static void complete_accel_close(struct retry_handler *rh,
				 struct timer_list *entry)
{
	struct sct_entry *sct = container_of(entry, struct sct_entry,
					     timeout_list);

	rh_printf(rh, LOG_WARNING,
		  "recycling sct=%u after ACCEL_CLOSE timeout\n",
	       sct->sct_idx);

	/* Developer Note: CAS-2521
	 * There shouldn't be any SPTs on the chain.
	 */
	if (!list_empty(&sct->spt_list))
		fatal(rh,
		      "The SPT list for sct=%u is not empty but got an ACCEL_CLOSE event\n",
		      sct->sct_idx);

	release_sct(rh, sct);
}

/* Process the C_PCT_ACCEL_CLOSE_COMPLETE event */
void accel_close(struct retry_handler *rh, const struct c_event_pct *event)
{
	unsigned int sct_idx = event->conn_idx.padded_sct_idx.sct_idx;
	const struct sct_entry comp = {
		.sct_idx = sct_idx,
	};
	struct sct_entry *sct;
	struct sct_entry **ret;
	static const struct timeval wait_time = {
		.tv_sec = 1,
	};
	struct timeval tv;

	gettimeofday(&tv, NULL);

	rh_printf(rh, LOG_WARNING, "ACCEL_CLOSE_COMPLETE event (sct=%u, seq_num=%u, ts=%lu.%06lu)\n",
		  sct_idx, event->seq_num, tv.tv_sec, tv.tv_usec);

	/* The SCT may be known from a previous NACK or timeout. */
	ret = tfind(&comp, &rh->sct_tree, sct_compare);
	if (ret)
		sct = *ret;
	else
		sct = alloc_sct(rh, sct_idx);

	/* Developer Note: CAS-2802
	 *
	 * Perform force close to free resources.
	 */
	sct->accel_close_event = true;
	sct->do_force_close = true;

	set_sct_recycle_bit(rh, sct);

	/* Wait 1 second before recycling this SCT.
	 * TODO: the time should depend on the load. See CSDG 8.3.
	 */
	sct->timeout_list.func = complete_accel_close;
	timer_add(rh, &sct->timeout_list, &wait_time);
}

static void complete_ordered(struct retry_handler *rh,
			     const struct c_event_pct *event)
{
	struct sct_entry **ret;
	unsigned int sct_idx = event->conn_idx.padded_sct_idx.sct_idx;
	const struct sct_entry comp = {
		.sct_idx = sct_idx,
	};
	struct sct_entry *sct;
	struct spt_entry *spt;

	/* Look for the SCT in the tree */
	ret = tfind(&comp, &rh->sct_tree, sct_compare);
	if (!ret)
		fatal(rh, "cannot find sct=%d in tree\n", sct_idx);

	sct = *ret;

	/* Find the SPT entry in the list */
	list_for_each_entry(spt, &sct->spt_list, list) {
		if (event->spt_idx == spt->spt_idx)
			break;
	}

	if (spt == NULL || event->spt_idx != spt->spt_idx)
		fatal(rh,
		      "Got complete event for spt=%u in sct=%u, but SPT was not found in list\n",
		      event->spt_idx, sct->sct_idx);

	if (spt->status == STS_COMPLETED)
		fatal(rh, "got duplicated complete event for spt=%in sct=%u\n",
		      event->spt_idx, sct_idx);
	if (spt->current_event_to) {
		assert(sct->num_to_pkts_in_chain > 0);
		sct->num_to_pkts_in_chain--;
	}
	spt->status = STS_COMPLETED;
	sct->spt_completed++;
	sct->spt_status_known++;

	/* If we retried a packet and got a retry complete, we know
	 * that a connection must have been established.
	 */
	sct->conn_established = true;

	/* Increase the number of SPTs to be retried after TRS Nacks
	 * were received, but limit that number to up to max_batch_size.
	 */
	if (sct->batch_size && sct->batch_size < max_batch_size)
		sct->batch_size++;

	/* Reset NACK counts when forward progress is made. */
	sct->backoff_nack_only_in_chain = 0;

	check_sct_status(rh, sct, false);
}

/* Process C_PCT_RETRY_COMPLETE event */
static void retry_complete(struct retry_handler *rh,
			   const struct c_event_pct *event)
{
	unsigned int log_level;
	unsigned int spt_idx = event->spt_idx;
	const struct spt_entry comp = {
		.spt_idx = spt_idx,
	};
	struct spt_entry **ret;
	struct spt_entry *spt;
	struct timeval tv;

	gettimeofday(&tv, NULL);

	/* Look for the SPT in the tree */
	ret = tfind(&comp, &rh->spt_tree, spt_compare);
	if (ret == NULL)
		fatal(rh, "Got retry comp event for unknown spt=%u\n",
		      spt_idx);

	spt = *ret;

	/* Validate Event */
	if (event->return_code != C_RC_OK &&
	    event->return_code != C_RC_CANCELED &&
	    event->return_code != C_RC_UNDELIVERABLE &&
	    event->return_code != C_RC_UNCOR)
		fatal(rh, "unexpected completion code for spt=%u (%d)\n",
		      event->spt_idx, event->return_code);

	if (spt->ram0.req_order == 1) {
		/* Put NACK retry completes at a lower level than Timeouts */
		log_level = spt->has_timed_out ? LOG_NOTICE : LOG_INFO;
		rh_printf(rh, log_level, "PCT retry complete event (spt=%u, sct=%u, rc=%s, ts=%lu.%06lu)\n",
			  event->spt_idx,
			  event->conn_idx.padded_sct_idx.sct_idx,
			  cxi_rc_to_str(event->return_code),
			  tv.tv_sec, tv.tv_usec);
		complete_ordered(rh, event);
	} else {
		rh_printf(rh, LOG_NOTICE, "PCT retry complete event (spt=%u, rc=%s, ts=%lu.%06lu)\n",
			  event->spt_idx, cxi_rc_to_str(event->return_code),
			  tv.tv_sec, tv.tv_usec);

		release_spt(rh, spt);
	}
}

/* Event handler. Triage the events and empty the queue. */
static void event_handler(struct retry_handler *rh)
{
	const union c_event *event;
	int event_num = 0;
	int total_events = 0;

	while ((event = cxi_eq_get_event(rh->eq))) {
		if (event->hdr.event_type != C_EVENT_PCT)
			fatal(rh, "Got event %d in retry handler\n",
			      event->hdr.event_type);

		switch (event->pct.pct_event_type) {
		case C_PCT_REQUEST_NACK:
			rh->stats.event_nack++;
			request_nack(rh, &event->pct);
			break;
		case C_PCT_REQUEST_TIMEOUT:
			rh->stats.event_spt_timeout++;
			request_timeout(rh, &event->pct);
			break;
		case C_PCT_SCT_TIMEOUT:
			rh->stats.event_sct_timeout++;
			sct_timeout(rh, &event->pct);
			break;
		case C_PCT_TCT_TIMEOUT:
			rh->stats.event_tct_timeout++;
			tct_timeout(rh, &event->pct);
			break;
		case C_PCT_ACCEL_CLOSE_COMPLETE:
			rh->stats.event_accel_close_complete++;
			accel_close(rh, &event->pct);
			break;
		case C_PCT_RETRY_COMPLETE:
			rh->stats.event_retry_complete++;
			retry_complete(rh, &event->pct);
			break;

		default:
			fatal(rh, "Unknown PCT event %d\n",
			      event->pct.pct_event_type);
		}

		/* Acknowledge EQ every 16 events.
		 * TODO: should this be configurable?
		 */
		event_num++;
		if (event_num == 16) {
			event_num = 0;
			cxi_eq_ack_events(rh->eq);
		}

		total_events++;

		/* For now, stop after every 32 events, so the timer
		 * list may be processed if needed.
		 */
		if (total_events == 32)
			break;
	}

	cxi_eq_ack_events(rh->eq);
}

static void stop_rh(struct retry_handler *rh)
{
	rh_printf(rh, LOG_WARNING, "cleanup\n");

	cxil_destroy_evtq(rh->eq);
	if (rh->eq_pt) {
		munmap(rh->eq_buf, rh->eq_size);
	} else {
		cxil_unmap(rh->eq_buf_md);
		free(rh->eq_buf);
	}
	cxil_destroy_wait_obj(rh->wait);
	cxil_destroy_lni(rh->lni);
	cxil_destroy_svc(rh->dev, rh->svc_id);
	cxil_close_device(rh->dev);
}

/* Return an epoch timeout, given an epoch_sel like
 * sct_close_epoch_sel or spt_timeout_epoch_sel, from
 * C_PCT_CFG_TIMING. Returned timeout is in microseconds, with a
 * minimum of 1.
 */
unsigned int get_epoch_timeout(struct retry_handler *rh,
			       unsigned int epoch_sel)
{
	float timeout;

	timeout = 1ULL << (epoch_sel);
	if (rh->is_c1) {
		/* Cassini 1 clock is 1Ghz. */
		timeout /= 1000;
	} else {
		/* Cassini 2 clock is 1.1Ghz. */
		timeout /= 1100;
	}

	if (rh->mb_sts_rev.platform == 2) {
		/* Z1 is slow */
		timeout *= 1800;
	}

	if (timeout < 1.0)
		timeout = 1;

	return timeout;
}

/* Update C_PCT_CFG_TIMING based on config values and
 * estimate their values in microseconds. Also sets up
 * some internal RH time values.
 */
void setup_timing(struct retry_handler *rh)
{

	union c_pct_cfg_timing pct_cfg_timing;
	unsigned int epoch_us;
	unsigned int max_allowed_retry_time_us;
	unsigned int retry_time_sum;
	unsigned int i;

	cxil_read_csr(rh->dev, C_PCT_CFG_TIMING,
		      &pct_cfg_timing, sizeof(pct_cfg_timing));

	/* Assume we'll write back the driver defaults in case some garbage
	 * values have been left behind on the system
	 */
	default_spt_timeout_epoch = C1_DEFAULT_SPT_TIMEOUT_EPOCH;
	pct_cfg_timing.sct_idle_epoch_sel = C1_DEFAULT_SCT_IDLE_EPOCH;
	pct_cfg_timing.sct_close_epoch_sel = C1_DEFAULT_SCT_CLOSE_EPOCH;
	pct_cfg_timing.tct_timeout_epoch_sel = C1_DEFAULT_TCT_TIMEOUT_EPOCH;

	if (user_spt_timeout_epoch &&
	    user_spt_timeout_epoch != C1_DEFAULT_SPT_TIMEOUT_EPOCH) {
		rh_printf(rh, LOG_WARNING, "CAUTION. Using spt_timeout_epoch: %u instead of default: %u\n",
		       user_spt_timeout_epoch,
		       pct_cfg_timing.spt_timeout_epoch_sel);
		default_spt_timeout_epoch = user_spt_timeout_epoch;
	}
	pct_cfg_timing.spt_timeout_epoch_sel = default_spt_timeout_epoch;

	if (user_sct_idle_epoch &&
	    user_sct_idle_epoch != C1_DEFAULT_SCT_IDLE_EPOCH) {
		rh_printf(rh, LOG_WARNING, "CAUTION. Using sct_idle_epoch: %u instead of default: %u\n",
		       user_sct_idle_epoch,
		       pct_cfg_timing.sct_idle_epoch_sel);
		pct_cfg_timing.sct_idle_epoch_sel = user_sct_idle_epoch;
	}
	if (user_sct_close_epoch &&
	    user_sct_close_epoch != C1_DEFAULT_SCT_CLOSE_EPOCH) {
		rh_printf(rh, LOG_WARNING, "CAUTION. Using sct_close_epoch: %u instead of default: %u\n",
		       user_sct_close_epoch,
		       pct_cfg_timing.sct_close_epoch_sel);
		pct_cfg_timing.sct_close_epoch_sel = user_sct_close_epoch;
	}
	if (user_tct_timeout_epoch &&
	    user_tct_timeout_epoch != C1_DEFAULT_TCT_TIMEOUT_EPOCH) {
		rh_printf(rh, LOG_WARNING, "CAUTION. Using tct_timeout_epoch: %u instead of default: %u\n",
		       user_tct_timeout_epoch,
		       pct_cfg_timing.tct_timeout_epoch_sel);
		pct_cfg_timing.tct_timeout_epoch_sel = user_tct_timeout_epoch;
	}

	cxil_write_csr(rh->dev, C_PCT_CFG_TIMING,
		       &pct_cfg_timing, sizeof(pct_cfg_timing));

	/* Estimate PCT Timeout Epochs */
	rh_printf(rh, LOG_WARNING, "pct_cfg_timing estimates\n");
	epoch_us = get_epoch_timeout(rh, default_spt_timeout_epoch);
	rh_printf(rh, LOG_WARNING, "spt_timeout_epoch_sel(%d) ~ %u us\n",
		  default_spt_timeout_epoch, epoch_us);
	rh->spt_timeout_us = epoch_us;

	epoch_us = get_epoch_timeout(rh, down_nid_spt_timeout_epoch);
	rh_printf(rh, LOG_WARNING, "down_nid_spt_timeout_epoch(%d) ~ %u us\n",
		  down_nid_spt_timeout_epoch, epoch_us);

	epoch_us = get_epoch_timeout(rh,
			pct_cfg_timing.sct_idle_epoch_sel);
	rh_printf(rh, LOG_WARNING, "sct_idle_epoch_sel(%d) ~ %u us\n",
		  pct_cfg_timing.sct_idle_epoch_sel,
		  epoch_us);

	epoch_us = get_epoch_timeout(rh,
			pct_cfg_timing.sct_close_epoch_sel);
	rh_printf(rh, LOG_WARNING, "sct_close_epoch_sel(%d) ~ %u us\n",
		  pct_cfg_timing.sct_close_epoch_sel,
		  epoch_us);

	epoch_us = get_epoch_timeout(rh,
			pct_cfg_timing.tct_timeout_epoch_sel);
	rh_printf(rh, LOG_WARNING, "tct_timeout_epoch_sel(%d) ~ %u us\n",
		  pct_cfg_timing.tct_timeout_epoch_sel,
		  epoch_us);
	rh->base_retry_interval_us = epoch_us / 2000;
	rh->max_retry_interval_us = epoch_us / 2;

	/* Tie peer_tct_timeout to TCT Epoch, unless the user already set it */
	if (!peer_tct_free_wait_time.tv_sec) {
		peer_tct_free_wait_time.tv_sec = (epoch_us * 2) / 1000000;
		peer_tct_free_wait_time.tv_usec = (epoch_us * 2) % 1000000;
	}

	/* Base initial exponential delay value off of SCT TO */
	if (rh->base_retry_interval_us == 0)
		rh->base_retry_interval_us = 1;

	/* Max value of usec that RH can use for a single backoff,
	 * takes into account the seed value and the backoff multiplier.
	 */
	rh->max_backoff_usec_val = UINT_MAX / (rh->base_retry_interval_us * backoff_multiplier);

	rh_printf(rh, LOG_WARNING, "Initial retry backoff set to %uus\n",
		  rh->base_retry_interval_us);
	rh_printf(rh, LOG_WARNING, "Maximum retry backoff set to %uus\n",
		  rh->max_retry_interval_us);

	max_allowed_retry_time_us = epoch_us * (allowed_retry_time_percent / 100.0);

	retry_time_sum = 0;
	for (i = 0; i < max_spt_retries; ++i) {
		rh_printf(rh, LOG_WARNING, "retry_interval[%d]: %f\n",
			  i, (double)retry_interval_values_us[i] / 1000000);
		retry_time_sum += retry_interval_values_us[i];

		/* Each retry time should not be more than max_retry_interval_us */
		if (retry_interval_values_us[i] > rh->max_retry_interval_us)
			fatal(rh, "Backoff interval: %f exceeds allowable maximum:%f\n",
			      (double)retry_interval_values_us[i] / 1000000,
			      (double)rh->max_retry_interval_us / 1000000);
	}

	/* Total retry time sum should be less than max_allowed_retry_time_us */
	if (retry_time_sum > max_allowed_retry_time_us)
		fatal(rh, "Backoff interval totals: %u exceed allowable maximum:%u\n",
		      retry_time_sum, max_allowed_retry_time_us);
}

/* Called after a fatal error */
void fatal(struct retry_handler *rh, const char *fmt, ...)
{
	va_list ap;

	rh_printf(rh, LOG_CRIT, "fatal error\n");

	if (rh && rh->dev) {
		dump_rh_state(rh);
		dump_csrs(rh);
		cxil_destroy_svc(rh->dev, rh->svc_id);
	}

	va_start(ap, fmt);
	verrx(1, fmt, ap);
	va_end(ap);
}


/* Function called when exiting */
/* TODO: this only works when there is one device. It may need a list
 * of devices. This may not be needed if errx() isn't used anymore,
 * but is useful during development.
 */
static struct retry_handler *exit_rh;
static void exit_fn(void)
{
	if (exit_rh) {
		stop_rh(exit_rh);
		exit_rh = NULL;
	}
}

static int start_rh(struct retry_handler *rh, unsigned int dev_id)
{
	int ret;
	struct cxi_eq_attr attr = {};
	union c_pct_cfg_srb_retry_ctrl srb_retry_ctrl = {
		.pause_mode = 1,
	};
	struct cxi_rsrc_limits limits = {
		.type[CXI_RSRC_TYPE_EQ] = { .max = 1, .res = 1},
		.type[CXI_RSRC_TYPE_AC] = { .max = 1, .res = 1},
	};
	struct cxi_svc_desc svc_desc = {
		.enable = true,
		.limits = limits,
		.resource_limits = true,
		.restricted_members = true,
		.restricted_vnis = true,
		.num_vld_vnis = 0,
		.is_system_svc = true,
		.members[0] = {
			.type = CXI_SVC_MEMBER_UID,
			.svc_member.uid = geteuid()
		},
	};
	union c_oxe_cfg_outstanding_limit limit;
	int page_size = sysconf(_SC_PAGESIZE);

	ret = cxil_open_device(dev_id, &rh->dev);
	if (ret != 0)
		fatal(rh, "Failed to open device %u: %d\n", dev_id, ret);

	if (rh->dev->info.is_vf) {
		rh_printf(rh, LOG_ERR, "The retry handler will not start on device %u because it is a virtual function\n", dev_id);
		exit(EXIT_FAILURE);
	}

	rh->is_c1 = rh->dev->info.cassini_version & CASSINI_1;

	ret = cxil_map_csr(rh->dev);
	if (ret != 0)
		fatal(rh, "map csrs failed: %d\n", ret);

	rh->svc_id = cxil_alloc_svc(rh->dev, &svc_desc, NULL);
	if (rh->svc_id < 0)
		fatal(rh, "svc_alloc failed: %d\n", rh->svc_id);
	rh_printf(rh, LOG_DEBUG, "svc_id: %d\n", rh->svc_id);
	rh_printf(rh, LOG_DEBUG, "euid: %d\n", svc_desc.members[0].svc_member.uid);

	ret = cxil_svc_enable(rh->dev, rh->svc_id, true);
	if (ret != 0)
		fatal(rh, "enable svc failed: %d\n", ret);

	ret = cxil_alloc_lni(rh->dev, &rh->lni, rh->svc_id);
	if (ret != 0)
		fatal(rh, "alloc lni failed: %d\n", ret);

	ret = cxil_alloc_wait_obj(rh->lni, &rh->wait);
	if (ret != 0)
		fatal(rh, "alloc wait obj failed: %d\n", ret);

	/* The CSDG recommends 16k entries, plus the 256 bytes for
	 * status, which is just above 256KB.
	 */
	/* TODO: in the future, the kernel will allocate it, and the
	 * rh daemon will simply map it, so the EQ stays alive even if
	 * the daemon dies.
	 */
	rh->eq_size = 2 * 1024 * 1024;
	rh->eq_buf = mmap(NULL, rh->eq_size,
			  PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
			  MAP_HUGE_2MB,
			  -1, 0);
	if (rh->eq_buf == MAP_FAILED) {
		rh_printf(rh, LOG_WARNING,
			  "Failed to mmap hugepage for PCT EQ\n");
		rh->eq_size = 16384 * 16 + page_size;
		rh->eq_buf = aligned_alloc(page_size, rh->eq_size);
		if (rh->eq_buf == NULL)
			fatal(rh, "Failed to allocated the EQ buffer");

		ret = cxil_map(rh->lni, rh->eq_buf, rh->eq_size,
			       CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
			       NULL, &rh->eq_buf_md);
		if (ret != 0)
			fatal(rh, "Failed to map the EQ buffer: %d\n",
			      ret);
	} else {
		rh->eq_pt = true;
		attr.flags |= CXI_EQ_PASSTHROUGH;
	}

	memset(rh->eq_buf, 0, rh->eq_size);

	attr.queue = rh->eq_buf;
	attr.queue_len = rh->eq_size;
	attr.flags |= CXI_EQ_REGISTER_PCT;

	ret = cxil_alloc_evtq(rh->lni, rh->eq_buf_md, &attr,
			      rh->wait, NULL, &rh->eq);
	if (ret != 0)
		fatal(rh, "cxil_alloc_evtq failed: %d\n", ret);

	cxil_write_csr(rh->dev, C_PCT_CFG_SRB_RETRY_CTRL,
		       &srb_retry_ctrl, sizeof(srb_retry_ctrl));

	/* Retrieve some device info */
	cxil_read_csr(rh->dev, C_MB_STS_REV,
		      &rh->mb_sts_rev, sizeof(rh->mb_sts_rev));

	setup_timing(rh);

	cxil_read_csr(rh->dev, C_OXE_CFG_OUTSTANDING_LIMIT(0), &limit,
		      sizeof(limit));
	default_get_packets_inflight = limit.get_limit;
	default_put_limit_inflight = limit.put_limit;
	default_ioi_ord_limit_inflight = limit.ioi_ord_limit;
	default_ioi_unord_limit_inflight = limit.ioi_unord_limit;

	rh->nid_tree = NULL;
	rh->nid_tree_count = 0;
	rh->stats.max_nid_tree_count = 0;
	rh->switch_tree = NULL;
	rh->switch_tree_count = 0;
	rh->stats.max_switch_tree_count = 0;
	rh->parked_nids = 0;
	rh->stats.max_parked_nids = 0;
	rh->parked_switches = 0;
	rh->stats.max_parked_switches = 0;

	/* Print additional information from config */
	rh_printf(rh, LOG_WARNING, "unorder_pkt_min_retry_delay (%u)\n",
		  unorder_pkt_min_retry_delay);
	rh_printf(rh, LOG_WARNING, "down_nid_get_packets_inflight (%u)\n",
		  down_nid_get_packets_inflight);
	rh_printf(rh, LOG_WARNING, "down_nid_put_packets_inflight (%u)\n",
		  down_nid_put_packets_inflight);
	rh_printf(rh, LOG_WARNING, "down_switch_nid_count (%u)\n",
		  down_switch_nid_count);
	rh_printf(rh, LOG_WARNING, "down_nid_pkt_count (%u)\n",
		  down_nid_pkt_count);
	rh_printf(rh, LOG_WARNING, "switch_id_mask (0x%x)\n", switch_id_mask);
	rh_printf(rh, LOG_WARNING, "max_fabric_packet_age (usecs) (%u)\n",
		  max_fabric_packet_age);
	rh_printf(rh, LOG_WARNING, "max_spt_retries (timeouts) (%u)\n",
		  max_spt_retries);
	rh_printf(rh, LOG_WARNING, "max_no_matching_conn_retries (%u)\n",
		  max_no_matching_conn_retries);
	rh_printf(rh, LOG_WARNING, "max_resource_busy_retries (%u)\n",
		  max_resource_busy_retries);
	rh_printf(rh, LOG_WARNING, "max_trs_pend_rsp_retries (%u)\n",
		  max_trs_pend_rsp_retries);
	rh_printf(rh, LOG_WARNING, "max_batch_size (no_trs) (%u)\n", max_batch_size);
	rh_printf(rh, LOG_WARNING, "initial_batch_size (no_trs) (%u)\n",
		  initial_batch_size);
	rh_printf(rh, LOG_WARNING, "backoff_multiplier (%u)\n", backoff_multiplier);
	rh_printf(rh, LOG_WARNING, "nack_backoff_start (%u)\n", nack_backoff_start);
	rh_printf(rh, LOG_WARNING, "tct_wait_time  (%lu.%06lus)\n",
		  tct_wait_time.tv_sec, tct_wait_time.tv_usec);
	rh_printf(rh, LOG_WARNING, "pause_wait_time  (%lu.%06lus)\n",
		  pause_wait_time.tv_sec, pause_wait_time.tv_usec);
	rh_printf(rh, LOG_WARNING, "cancel_spt_wait_time  (%lu.%06lus)\n",
		  cancel_spt_wait_time.tv_sec, cancel_spt_wait_time.tv_usec);
	rh_printf(rh, LOG_WARNING, "peer_tct_free_wait_time  (%lu.%06lus)\n",
		  peer_tct_free_wait_time.tv_sec,
		  peer_tct_free_wait_time.tv_usec);
	rh_printf(rh, LOG_WARNING, "down_nid_wait_time (%lu.%06lus)\n",
		  down_nid_wait_time.tv_sec,
		  down_nid_wait_time.tv_usec);

	exit_rh = rh;
	atexit(exit_fn);

	rh->dev_id = dev_id;
	rh_printf(rh, LOG_WARNING, "device %d (nid=%u) set up\n", dev_id,
		  rh->dev->info.nid);

	return 0;
}

static void usage(const char *name)
{
	printf("Usage: %s [options]...\n"
	       "   -c, --conf=CONFIG       Configuration file\n"
	       "   -d, --device=ID         Device name (e.g. \"cxi0\")\n"
	       "   -h, --help              This help\n"
	       "   -l, --log-stdout        Bypass systemd logging, use stdout)\n",
	       name);
}

static struct option long_opts[] = {
	{"help", no_argument, NULL, 'h' },
	{"conf", required_argument, NULL, 'c'},
	{"device", required_argument, NULL, 'd'},
	{"log-stdout", no_argument, NULL, 'l'},
	{ NULL, 0, NULL, 0 }
};

static uv_poll_t pct_eq_watcher;
uv_timer_t timer_watcher;
static uv_signal_t sigint_watcher;
static uv_signal_t sighup_watcher;
static uv_signal_t sigterm_watcher;
uv_loop_t *loop;

static void pct_eq_cb(uv_poll_t *handle, int status, int revents)
{
	struct retry_handler *rh = handle->data;

	cxil_clear_wait_obj(rh->wait);

	event_handler(rh);
}

/* Handle various signals and break out from uv_run
 * via uv_stop if necessary.
 */
static void signal_cb(uv_signal_t *handle, int signum)
{
	struct retry_handler *rh = handle->data;

	if (signum == SIGINT) {
		modify_spt_timeout(rh, default_spt_timeout_epoch);
		modify_mcu_inflight(rh, default_get_packets_inflight,
				    default_put_limit_inflight,
				    default_ioi_ord_limit_inflight,
				    default_ioi_unord_limit_inflight);
		rh_printf(rh, LOG_ALERT, "got SIGINT\n");
		uv_stop(loop);
	} else if (signum == SIGTERM) {
		modify_spt_timeout(rh, default_spt_timeout_epoch);
		modify_mcu_inflight(rh, default_get_packets_inflight,
				    default_put_limit_inflight,
				    default_ioi_ord_limit_inflight,
				    default_ioi_unord_limit_inflight);
		rh_printf(rh, LOG_ALERT, "got SIGTERM\n");
		uv_stop(loop);
	} else if (signum == SIGHUP) {
		rh_printf(rh, LOG_WARNING, "got SIGHUP\n");
		dump_rh_state(rh);
	}
}

int main(int argc, char *argv[])
{
	struct retry_handler rh = {};
	config_file = SYSCONFDIR "/cxi_rh.conf";
	int option;
	long dev_id = 0;
	char *endptr;
	struct timeval tv;
	int i, stats_rc;

	setbuf(stdout, NULL);

	#ifndef HAVE_LIBSYSTEMD
		rh.log_stdout = true;
	#endif

	while ((option = getopt_long(argc, argv, "c:d:h:l",
				     long_opts, NULL)) != -1) {
		switch (option) {
		case 'c':
			config_file = optarg;
			break;
		case 'd':
			/* Extract device number from, for instance, "cxi2" */
			if (strlen(optarg) < 4 ||
			    strncmp(optarg, "cxi", 3)) {
				rh_printf(&rh, LOG_ERR, "Invalid device name: %s\n",
				       optarg);
				return EXIT_FAILURE;
			}

			optarg += 3; /* skip cxi prefix */

			errno = 0;
			dev_id = strtol(optarg, &endptr, 10);

			if ((errno == ERANGE &&
			     (dev_id == LONG_MAX || dev_id == LONG_MIN)) ||
			    (errno != 0 && dev_id == 0) ||
			    (endptr == optarg) ||
			    (dev_id < 0))  {
				rh_printf(&rh, LOG_ERR, "Invalid device name: %s\n",
				       optarg);
				return EXIT_FAILURE;
			}
			break;
		case 'l':
			rh.log_stdout = true;
			break;
		case 'h':
		default:
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	if (optind < argc) {
		rh_printf(&rh, LOG_ERR, "Extra arguments found on the command line, starting with \"%s\"\n",
		       argv[optind]);
		return EXIT_FAILURE;
	}

	rh_printf(&rh, LOG_WARNING, "CXI retry handler version %s\n", GIT_VERSION);

	/* If started in initrd, set argv[0][0] to '@' to indicate that the
	 * retry handler should not be killed prior to switch_root. In
	 * conjunction with appropriate systemd units, this allows the retry
	 * handler to support a root filesystem mounted over CXI.
	 *
	 * Reference: https://systemd.io/ROOT_STORAGE_DAEMONS/
	 */
	if (access("/etc/initrd-release", F_OK) >= 0) {
		rh_printf(&rh, LOG_WARNING, "Running from initrd\r\n");
		argv[0][0] = '@';
	}

	if (read_config(config_file, &rh))
		return 1;

	gettimeofday(&tv, NULL);
	srand(getpid() * tv.tv_usec);
	init_list_head(&rh.timeout_list.list);
	for (i = 0; i < C_PCT_CFG_SCT_CAM_ENTRIES; i++) {
		init_list_head(&rh.sct_state[i].timeout_list.list);
		rh.sct_state[i].timeout_list.func = timeout_reset_sct_pending;
	}

	start_rh(&rh, dev_id);

	stats_rc = stats_init(&rh);
	if (stats_rc)
		rh_printf(&rh, LOG_ERR, "stats_init failed\n");
	else
		rh_printf(&rh, LOG_WARNING, "statsFS mounted at %s\n",
			  rh_stats_dir);

	loop = uv_default_loop();

	uv_poll_init(loop, &pct_eq_watcher, cxil_get_wait_obj_fd(rh.wait));
	pct_eq_watcher.data = &rh;
	uv_poll_start(&pct_eq_watcher, UV_PRIORITIZED, pct_eq_cb);

	uv_timer_init(loop, &timer_watcher);
	timer_watcher.data = &rh;

	uv_signal_init(loop, &sigint_watcher);
	sigint_watcher.data = &rh;
	uv_signal_start(&sigint_watcher, signal_cb, SIGINT);

	uv_signal_init(loop, &sigterm_watcher);
	sigterm_watcher.data = &rh;
	uv_signal_start(&sigterm_watcher, signal_cb, SIGTERM);

	uv_signal_init(loop, &sighup_watcher);
	sighup_watcher.data = &rh;
	uv_signal_start(&sighup_watcher, signal_cb, SIGHUP);

	/* Developer Note: NETCASSINI-5042
	 *
	 * Stop executing recovery.
	 * RH Crash requires a node reboot as NIC state is unknown after crash occurs
	 */
	// recovery(&rh);

#if defined(HAVE_LIBSYSTEMD)
	sd_notify(0, "READY=1");
	sd_notify(0, "STATUS=Running");
#endif // defined(HAVE_LIBSYSTEMD)
	uv_run(loop, UV_RUN_DEFAULT);

	uv_loop_close(loop);

	if (!stats_rc)
		stats_fini();

	rh_printf(&rh, LOG_WARNING, "exiting normally\n");

	return 0;
}
