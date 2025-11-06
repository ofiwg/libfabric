/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2020 Hewlett Packard Enterprise Development LP
 */

#define _GNU_SOURCE

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <libconfig.h>
#include <uv.h>
#include <sys/time.h>
#include <cassini_user_defs.h>

#include "libcxi.h"
#include "list.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct retry_handler;

struct timer_list {
	struct list_head list;
	unsigned int generation;

	/* Absolute expiration time, in libuv time. */
	uint64_t timeout_ms;

	/* Function to call when the timer expires. */
	void (*func)(struct retry_handler *rh, struct timer_list *entry);
};

struct switch_entry {
	uint32_t id;
	uint16_t count;
	bool parked;
};

struct nid_entry {
	uint32_t nid;
	unsigned int pkt_count;
	bool parked;
	struct timer_list timeout_list;
};

#define SPT_TRY_NUM_SIZE 8U

#define ORDERED_PUT_LIMIT_MAX 2047U
#define MAX_ONLY_NO_MATCHING_CONN_RETRY 1

/* Highest possible value of max_spt_retries */
#define MAX_SPT_RETRIES_LIMIT 7

#define DFA_BITS 20U
#define DFA_MAX ((1 << DFA_BITS) - 1)

struct retry_handler {
	unsigned int dev_id;	/* cxi0, ... */
	int svc_id;
	struct cxil_dev *dev;
	struct cxil_lni *lni;
	struct cxi_eq *eq;
	struct cxil_wait_obj *wait;
	void *eq_buf;
	struct cxi_md *eq_buf_md;
	size_t eq_size;
	bool eq_pt;
	bool cfg_stats;
	bool log_stdout;

	union c_mb_sts_rev mb_sts_rev;
	bool is_c1;		/* Cassini 1 or 2 */

	/* Keep the SCTs and their SPT chains */
	void *sct_tree;

	/* Wait time before retrying an SCT */
	unsigned int base_retry_interval_us;

	/* Max allowed backoff value from shifting to handle overflow */
	uint64_t max_backoff_usec_val;

	/* Max wait time before retrying an SCT. Based on TCT timeout */
	unsigned int max_retry_interval_us;

	unsigned int spt_timeout_us;

	/* List of timers */
	struct timer_list timeout_list;
	unsigned int timer_generation;

	/* Tree of NIDs */
	void *nid_tree;

	/* Count of number of NIDs in parked/down tree. */
	unsigned int nid_tree_count;

	/* Tree of switches */
	void *switch_tree;

	/* Count of number of switches in parked/down tree. */
	unsigned int switch_tree_count;

	/* Signal if OXE and PCT configuration has been modified due to
	 * undeliverable packets.
	 */
	unsigned int parked_nids;

	/* Count for current number of parked switches. */
	unsigned int parked_switches;

	/* Semi permanent storage for the TRS CAM entries. These
	 * should be refreshed on demand.
	 *
	 * Keep only one array; Cassini 1 has a smaller array than
	 * Cassini 2, but elements are otherwise identical.
	 */
	static_assert(C2_PCT_CFG_TRS_CAM_ENTRIES > C1_PCT_CFG_TRS_CAM_ENTRIES,
		      "Bad number of TRS_CAM entries");
	union c_pct_cfg_trs_cam trs_cam[C2_PCT_CFG_TRS_CAM_ENTRIES];

	/* Keep unordered SPTs. These do not appear in
	 * the SCT chains.
	 */
	void *spt_tree;

	/* Track when all SPT try numbers have been used. */
	struct timespec spt_try_ts[C_PCT_CFG_SPT_RAM0_ENTRIES][SPT_TRY_NUM_SIZE];

	/* Track SCT specific information used to determine if connection is
	 * recycled.
	 *
	 * Track if SCTs are expecting a timeout due to the EOM_GET policy actions.
	 */
	struct sct_state {
		unsigned int seqno;
		unsigned int req_cnt;
		bool pending_timeout;
		uint32_t dfa;
		uint16_t vni;
		uint8_t dscp;
		uint8_t mcu_group;
		struct timer_list timeout_list;
	} sct_state[C_PCT_CFG_SCT_CAM_ENTRIES];

	pthread_t stats_thread;
	struct fuse *stats_fuse;
	struct fuse_chan *stats_chan;
	struct {
		unsigned int spt_alloc;
		unsigned int spt_freed;
		unsigned int spt_released;
		unsigned int spt_free_deferred;
		unsigned int sct_alloc;
		unsigned int sct_freed;
		unsigned int connections_cancelled;
		unsigned int pkts_cancelled_u;
		unsigned int pkts_cancelled_o;
		unsigned int cancel_no_matching_conn;
		unsigned int cancel_resource_busy;
		unsigned int cancel_trs_pend_rsp;
		unsigned int cancel_tct_closed;
		unsigned int event_nack;
		unsigned int nack_no_target_trs;
		unsigned int nack_no_target_mst;
		unsigned int nack_no_target_conn;
		unsigned int nack_no_matching_conn;
		unsigned int nack_resource_busy;
		unsigned int nack_trs_pend_rsp;
		unsigned int nack_sequence_error;
		unsigned int event_spt_timeout;
		unsigned int event_spt_timeout_u;
		unsigned int event_spt_timeout_o;
		unsigned int event_sct_timeout;
		unsigned int event_tct_timeout;
		unsigned int event_accel_close_complete;
		unsigned int event_retry_complete;
		unsigned int ignored_sct_timeouts;
		unsigned int rh_sct_status_change;
		unsigned int max_nid_tree_count;
		unsigned int max_switch_tree_count;
		unsigned int max_parked_nids;
		unsigned int max_parked_switches;
	} stats;

	/* Array of SMTs which have previously had messages cancelled. */
	bool dead_smt[C_PCT_CFG_SMT_RAM0_ENTRIES];

	/* Used to increment or decrement log levels of each print statement without
	 * having to change LogLevelMax and restarting the RH.
	 */
	int log_increment;
};

struct sct_entry;

struct spt_entry {
	unsigned int spt_idx;

	struct sct_entry *sct;

	struct list_head list;

	union c_pct_cfg_spt_ram0 ram0;
	union c_pct_cfg_spt_ram1 ram1;
	union c_pct_cfg_spt_misc_info misc_info;

	union c_pct_cfg_spt_ram2 ram2;
	bool ram2_valid;

	/* Whether that SPT has ever timed out, when, and which try_num it
	 * was the first time.
	 */
	bool has_timed_out;
	int timed_out_try_num;

	/* True if the current reason we're working on this SPT is due to a timeout */
	bool current_event_to;

	/* Return code to set when cancelling that SPT */
	enum c_return_code cancel_return_code;

	/* try_num to use in the next retry */
	unsigned int try_num;
	unsigned int nack_rc;

	/* Number of times this SPT has been retried due to NACKs */
	unsigned int nack_retries;

	/* Number of times this SPT has been retried due to Timeouts */
	unsigned int to_retries;

	struct timer_list timeout_list;

	enum {
		/* Should never be seen */
		STS_INVALID,

		/* Packet not sent yet, or sent and never ack'ed or
		 * timedout. It will either nack, time out or just
		 * complete on its own.
		 */
		STS_PENDING,

		/* Packet was nack'ed or timedout. It needs to be
		 * retried.
		 */
		STS_NEED_RETRY,

		/* Has been retried, and is awaiting a
		 * timeout or ACK event.
		 */
		STS_RETRIED,

		/* No work left for this packet. Set after a complete
		 * event or when the packet completed on its own.
		 */
		STS_COMPLETED,
	} status;

	/* Opcode for that SPT, and whether it has been set. */
	bool opcode_valid;
	unsigned int opcode;

	/* Number of NO_MATCHING_CONN Nacks this SPT has seen.
	 * The number might be off by 1 if the SPT wasn't allocated when
	 * we saw the first event.
	 */
	unsigned int no_matching_conn_nacks;

	/* Number of NO_MATCHING_CONN Nacks this SPT has seen
	 * while its parent SCT had a NO_TARGET_CONN NACK. This implies
	 * the no_matching_conn_nack was due to a lack of TCTs, and was benign.
	 */
	unsigned int benign_no_matching_conn;

	/* Number of resource_busy_nacks this SPT has seen. */
	unsigned int resource_busy_nacks;

	/* Number of trs_pend_rsp_nacks this SPT has seen. */
	unsigned int trs_pend_rsp_nacks;

	/* DFA that this SPT was targetting. */
	uint32_t dfa;

	/* VNI used for this SPT */
	unsigned int vni;

	/* Continuation packet SCT index extracted from SRB. */
	unsigned int cont_sct;
};

struct sct_entry {
	unsigned int sct_idx;
	int pcp;

	unsigned int close_retries;
	union c_pct_cfg_sct_ram0 ram0;
	union c_pct_cfg_sct_ram1 ram1;
	union c_pct_cfg_sct_ram2 ram2;
	union c_pct_cfg_sct_cam sct_cam;
	struct timer_list timeout_list;

	/* When was this SCT put into RETRY State */
	struct timeval alloc_time;

	/* Whether that SCT got an ACCEL_CLOSE event. This is needed
	 * for Errata CAS-2802.
	 */
	bool accel_close_event;

	/* Incremented upon a retry getting another nack.
	 * Reset when an op_complete is returned.
	 */
	unsigned int backoff_nack_only_in_chain;

	/* How many packets have timed out on this chain? Decreased when
	 * op_complete is received for a timed out packet.
	 */
	unsigned int num_to_pkts_in_chain;

	/* Whether any portion of the SCT has started retry */
	bool has_retried;

	/* Whether to cancel the SPTs */
	bool cancel_spts;

	/* Return Code to use when cancelling SPTs */
	enum c_return_code cancel_rc;

	/* Whether to do a force close when releasing the SCT */
	bool do_force_close;

	/* Whether an explicit clear has been sent */
	bool clr_sent;

	/* Whether the SCT was paused the last time we checked */
	bool paused;

	/* Number of SPTs belonging to this SCT */
	unsigned int num_entries;

	/* Number of SPT entries whose status is not pending */
	unsigned int spt_status_known;

	/* Number of SPT that were successfully retried */
	unsigned int spt_completed;

	/* SPT that is currently the clr_head */
	unsigned int head;

	/* SPT that is the tail of the batch we will retry */
	unsigned int tail;

	/* Final timed out SPT in the chain/batch that requires retry. */
	unsigned int batch_last_timeout;

	/* First SPT with unknown status */
	unsigned int first_pend_spt;

	/* Whether SPT chain has been allocated for this SCT */
	bool spts_allocated;

	/* Batch size to use when retrying after a TRS Nack was
	 * received for any of the SPTs belonging to that SCT
	 */
	unsigned int batch_size;

	union c_pct_cfg_sct_misc_info misc_info;

	struct list_head spt_list;

	/* Developer Note: CAS-3220
	 *
	 * If we inject a NACK from SW to move an SCT into
	 * RETRY state, store this info about the timed out packet.
	 */
	unsigned int faked_spt_idx;
	unsigned int faked_spt_try;

	/* Do we know that the peer TCT was allocated? If we received any
	 * resource exhaustion NACKs or see OP_COMPs, we know a connection
	 * was established earlier.
	 */
	bool conn_established;

	/* Were there any timed out packets in the current batch of retries */
	bool to_pkts_in_batch;

	/* Were there any NO_TARGET_CONN (No TCT) Nacks in the current batch of retries */
	bool no_tct_in_batch;

	/* Were there only NO_MATCHING_CONN NACKs on this SCT */
	bool only_no_matching_conn_nacks;

	/* How many times the SCT has been retried for only_no_matching_conn_nacks flag */
	unsigned int only_no_matching_conn_retry_cnt;

	/* Did this SCT ever time out */
	bool has_timed_out;
};

struct tct_entry {
	unsigned int tct_idx;

	/* Timed out TCT must wait a bit longer before being acted
	 * upon.
	 */
	struct timer_list timeout_list;
};
struct hni_cont_header {
	struct c_port_fab_hdr hdr;
	struct c_port_continuation_hdr cont;
} __attribute__((packed));

struct hni_header4 {
	struct c_port_fab_hdr hdr;
	struct c_pkt_type pkt_type;
} __attribute__((packed));

struct hni_header_vs {
	struct c_port_fab_vs_hdr hdr;
	struct c_pkt_type pkt_type;
} __attribute__((packed));

union hni_pkt {
	uint64_t buf[5];
	struct c_port_fab_hdr hdr;
	struct hni_header4 h4;
	struct hni_header_vs vs;
	struct hni_cont_header cont;
} __attribute__((packed));

extern unsigned int max_fabric_packet_age;
extern unsigned int unorder_pkt_min_retry_delay;
extern unsigned int max_spt_retries;
extern unsigned int max_no_matching_conn_retries;
extern unsigned int max_resource_busy_retries;
extern unsigned int max_trs_pend_rsp_retries;
extern unsigned int max_sct_close_retries;
extern unsigned int initial_batch_size;
extern unsigned int max_batch_size;
extern unsigned int backoff_multiplier;
extern unsigned int timeout_backoff_multiplier;
extern unsigned int max_backoff_factor;
extern unsigned int nack_backoff_start;
extern unsigned int user_spt_timeout_epoch;
extern unsigned int user_sct_idle_epoch;
extern unsigned int user_sct_close_epoch;
extern unsigned int user_tct_timeout_epoch;
extern struct timeval tct_wait_time;
extern struct timeval pause_wait_time;
extern struct timeval cancel_spt_wait_time;
extern struct timeval peer_tct_free_wait_time;
extern struct timeval down_nid_wait_time;
extern unsigned int down_nid_spt_timeout_epoch;
extern unsigned int down_nid_get_packets_inflight;
extern unsigned int down_nid_put_packets_inflight;
extern unsigned int down_switch_nid_count;
extern unsigned int down_nid_pkt_count;
extern unsigned int switch_id_mask;
extern struct timeval sct_stable_wait_time;
extern char *rh_stats_dir;
extern char *config_file;
extern unsigned int retry_interval_values_us[MAX_SPT_RETRIES_LIMIT];
extern unsigned int allowed_retry_time_percent;
extern bool has_timeout_backoff_factor;

extern uv_loop_t *loop;
extern uv_timer_t timer_watcher;

void fatal(struct retry_handler *rh, const char *fmt, ...);

void dump_rh_state(const struct retry_handler *rh);
void dump_csrs(const struct retry_handler *rh);

int spt_compare(const void *a, const void *b);
void get_spt_info(struct retry_handler *rh, struct spt_entry *spt);
struct spt_entry *alloc_spt(struct retry_handler *rh,
			    const struct spt_entry *spt_in);
void request_timeout(struct retry_handler *rh, const struct c_event_pct *event);
void request_nack(struct retry_handler *rh, const struct c_event_pct *event);

void recovery(struct retry_handler *rh);

void timer_add(struct retry_handler *rh, struct timer_list *new,
	       const struct timeval *timeout);
void timer_del(struct timer_list *timer);
bool timer_is_set(struct timer_list *timer);

int read_config(const char *filename, struct retry_handler *rh);

unsigned int
retry_pkt(struct retry_handler *rh,
	  const struct c_pct_cfg_srb_retry_ptrs_entry retry_ptrs[],
	  unsigned int pcp, struct sct_entry *sct);
bool is_cq_closed(const struct retry_handler *rh, const struct spt_entry *spt);
bool has_uncor(const struct retry_handler *rh, const struct spt_entry *spt);

void increment_rmw_spt_try(const struct retry_handler *rh,
			   struct spt_entry *spt);
void simulate_rsp(struct retry_handler *rh,
		  union c_pct_cfg_sw_sim_src_rsp *src_rsp);
void schedule_cancel_spt(struct retry_handler *rh, struct spt_entry *spt,
			 enum c_return_code return_code);
void update_rmw_spt_to_flag(const struct retry_handler *rh,
			    struct spt_entry *spt, bool timed_out);
void release_spt(struct retry_handler *rh, struct spt_entry *spt);
void recycle_spt(struct retry_handler *rh, struct spt_entry *spt);
int spt_compare(const void *a, const void *b);
struct spt_entry *alloc_spt(struct retry_handler *rh,
			    const struct spt_entry *spt_in);
void set_spt_timed_out(const struct retry_handler *rh, struct spt_entry *spt,
		       const unsigned int to_try_num);
void get_spt_info(struct retry_handler *rh, struct spt_entry *spt);
void free_spt_chain(struct retry_handler *rh, struct sct_entry *sct);
struct sct_entry *alloc_sct(struct retry_handler *rh, unsigned int sct_idx);
int sct_compare(const void *a, const void *b);
void wait_loaded_bit(struct retry_handler *rh, unsigned int csr,
		     uint64_t bitmask);
void get_srb_info(struct retry_handler *rh, struct spt_entry *spt);
void set_sct_recycle_bit(const struct retry_handler *rh, struct sct_entry *sct);

void new_status_for_spt(struct retry_handler *rh,
			const struct spt_entry *spt_in,
			const struct c_event_pct *event);
void release_sct(struct retry_handler *rh, struct sct_entry *sct);

void unordered_spt_timeout(struct retry_handler *rh,
			   const struct spt_entry *spt_in,
			   const struct c_event_pct *event);

void sct_timeout(struct retry_handler *rh, const struct c_event_pct *event);

void tct_timeout(struct retry_handler *rh, const struct c_event_pct *event);

void accel_close(struct retry_handler *rh, const struct c_event_pct *event);
void check_sct_status(struct retry_handler *rh, struct sct_entry *sct,
		      bool timedout);

void nid_tree_del(struct retry_handler *rh, uint32_t nid);
bool nid_parked(struct retry_handler *rh, uint32_t nid);
void nid_tree_inc(struct retry_handler *rh, uint32_t nid);
bool switch_parked(struct retry_handler *rh, uint32_t nid);

int stats_init(struct retry_handler *rh);
void stats_fini(void);

/* SPT_MISC_INFO is not at the same offset in Cassini 1 and 2. */
static inline int spt_misc_info_csr(const struct retry_handler *rh,
				    unsigned int spt_idx)
{
	if (rh->is_c1)
		return C1_PCT_CFG_SPT_MISC_INFO(spt_idx);
	else
		return C2_PCT_CFG_SPT_MISC_INFO(spt_idx);
}

/* SCT_MISC_INFO is not at the same offset in Cassini 1 and 2. */
static inline int sct_misc_info_csr(const struct retry_handler *rh,
				    unsigned int sct_idx)
{
	if (rh->is_c1)
		return C1_PCT_CFG_SCT_MISC_INFO(sct_idx);
	else
		return C2_PCT_CFG_SCT_MISC_INFO(sct_idx);
}

void rh_printf(const struct retry_handler *rh, unsigned int base_log_level,
	       const char *fmt, ...)
__attribute__((format(printf, 3, 4)));

const char *nid_to_mac(uint32_t nid);
