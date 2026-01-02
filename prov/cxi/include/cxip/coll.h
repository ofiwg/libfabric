/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_COLL_H_
#define _CXIP_COLL_H_

#include <ofi_atom.h>
#include <ofi_list.h>
#include <ofi_lock.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct coll_counters;
struct cxip_av_set;
struct cxip_cmdq;
struct cxip_cntr;
struct cxip_ep;
struct cxip_ep_obj;
struct cxip_eq;
struct cxip_evtq;
struct cxip_md;
struct cxip_pte;
struct cxip_req;
struct cxip_zbcoll_obj;

/* Macros */
#define CXIP_COLL_MAX_CONCUR 8

#define CXIP_COLL_MIN_RX_BUFS 8

#define CXIP_COLL_MIN_RX_SIZE 131072

#define CXIP_COLL_MIN_MULTI_RECV 64

#define CXIP_COLL_MAX_DATA_SIZE 32

#define CXIP_COLL_MAX_SEQNO ((1 << 10) - 1)

#define CXIP_COLL_MOD_SEQNO (CXIP_COLL_MAX_SEQNO - 1)

#define CXIP_COLL_MIN_RETRY_USEC 1

#define CXIP_COLL_MAX_RETRY_USEC 32000

#define CXIP_COLL_MAX_LEAF_TIMEOUT_MULT 50

#define CXIP_COLL_MIN_TIMEOUT_USEC 1

#define CXIP_COLL_MAX_TIMEOUT_USEC 20000000

/* Type definitions */
struct cxip_ep_coll_obj {
	struct index_map mcast_map; // mc address -> object
	struct dlist_entry root_retry_list;
	struct dlist_entry mc_list; // list of mcast addresses
	struct cxip_coll_pte *coll_pte; // PTE extensions
	struct dlist_ts sched_list; // scheduled actions
	struct cxip_cmdq *rx_cmdq; // shared with STD EP
	struct cxip_cmdq *tx_cmdq; // shared with STD EP
	struct cxip_cntr *rx_cntr; // shared with STD EP
	struct cxip_cntr *tx_cntr; // shared with STD EP
	struct cxip_evtq *rx_evtq; // shared with STD EP
	struct cxip_evtq *tx_evtq; // shared with STD EP
	struct cxip_eq *eq; // shared with STD EP
	ofi_atomic32_t num_mc; // count of MC objects
	ofi_atomic32_t join_cnt; // advanced on every join
	size_t min_multi_recv; // trigger value to rotate bufs
	size_t buffer_size; // size of receive buffers
	size_t buffer_count; // count of receive buffers
	bool join_busy; // serialize joins on a node
	bool is_hwroot; // set if ep is hw_root
	bool enabled; // enabled
	/* needed for progress after leaf sends its contribution */
	struct dlist_entry leaf_rdma_get_list;
	/* used to change ctrl_msg_type to CXIP_CTRL_MSG_ZB_DATA_RDMA_LAC */
	bool leaf_save_root_lac;
	/* Logical address context for leaf rdma get */
	uint64_t rdma_get_lac_va_tx;
	/* pointer to the source buffer base used in the RDMA */
	uint8_t *root_rdma_get_data_p;
	/* root rdma get memory descriptor, for entire root src buffer */
	struct cxip_md *root_rdma_get_md;
};

struct cxip_intval {
	int64_t ival[4];
};

struct cxip_fltval {
	double fval[4];
};

struct cxip_iminmax {
	int64_t iminval;
	uint64_t iminidx;
	int64_t imaxval;
	uint64_t imaxidx;
};

struct cxip_fltminmax {
	double fminval;
	uint64_t fminidx;
	double fmaxval;
	uint64_t fmaxidx;
};

struct cxip_coll_buf {
	struct dlist_entry buf_entry; // linked list of buffers
	struct cxip_req *req; // associated LINK request
	struct cxip_md *cxi_md; // buffer memory descriptor
	size_t bufsiz; // buffer size in bytes
	uint8_t buffer[]; // buffer space itself
};

struct cxip_coll_pte {
	struct cxip_pte *pte; // Collectives PTE
	struct cxip_ep_obj *ep_obj; // Associated endpoint
	struct cxip_coll_mc *mc_obj; // Associated multicast object
	struct dlist_entry buf_list; // PTE receive buffers
	ofi_atomic32_t buf_cnt; // count of linked buffers
	ofi_atomic32_t buf_swap_cnt; // for diagnostics
	ofi_atomic32_t recv_cnt; // for diagnostics
	int buf_low_water; // for diagnostics
	bool enabled; // enabled
};

struct cxip_coll_data {
	union {
		uint8_t databuf[32]; // raw data buffer
		struct cxip_intval intval; // 4 integer values + flags
		struct cxip_fltval fltval; // 4 double values + flags
		struct cxip_iminmax intminmax; // 1 intminmax structure + flags
		struct cxip_fltminmax
			fltminmax; // 1 fltminmax structure + flags
		struct cxip_repsum repsum; // 1 repsum structure + flags
	};
	cxip_coll_op_t red_op; // reduction opcode
	cxip_coll_rc_t red_rc; // reduction return code
	int red_cnt; // reduction contrib count
	bool initialized;
};

struct cxip_coll_metrics_ep {
	int myrank;
	bool isroot;
};

struct cxip_coll_metrics {
	long red_count_bad;
	long red_count_full;
	long red_count_partial;
	long red_count_unreduced;
	struct cxip_coll_metrics_ep ep_data;
};

struct cxip_coll_reduction {
	struct cxip_coll_mc *mc_obj; // parent mc_obj
	uint32_t red_id; // reduction id
	uint16_t seqno; // reduction sequence number
	uint16_t resno; // reduction result number
	struct cxip_req *op_inject_req; // active operation request
	enum cxip_coll_state coll_state; // reduction state on node
	struct cxip_coll_data accum; // reduction accumulator
	struct cxip_coll_data backup; // copy of above
	void *op_rslt_data; // user recv buffer (or NULL)
	int op_data_bytcnt; // bytes in send/recv buffers
	void *op_context; // caller's context
	bool in_use; // reduction is in-use
	bool pktsent; // reduction packet sent
	bool completed; // reduction is completed
	bool rdma_get_sent; // rdma get from leaf to root
	bool rdma_get_completed; // rdma get completed
	int rdma_get_cb_rc; // rdma get status
	uint64_t leaf_contrib_start_us; // leaf ts after contrib send
	bool drop_send; // drop the next send operation
	bool drop_recv; // drop the next recv operation
	enum cxip_coll_rc red_rc; // set by first error
	struct timespec tv_expires; // need to retry?
	struct timespec arm_expires; // RE expiration time for this red_id
	struct dlist_entry tmout_link; // link to timeout list
	uint8_t tx_msg[64]; // static packet memory
};

struct cxip_coll_mc {
	struct fid_mc mc_fid;
	struct dlist_entry entry; // Link to mc object list
	struct cxip_ep_obj *ep_obj; // Associated endpoint
	struct cxip_av_set *av_set_obj; // associated AV set
	struct cxip_zbcoll_obj *zb; // zb object for zbcol
	struct cxip_coll_pte *coll_pte; // collective PTE
	struct timespec rootexpires; // root wait expiration timeout
	struct timespec leafexpires; // leaf wait expiration timeout
	struct timespec curlexpires; // CURL delete expiration timeout
	fi_addr_t mynode_fiaddr; // fi_addr of this node
	int mynode_idx; // av_set index of this node
	uint32_t hwroot_idx; // av_set index of hwroot node
	uint32_t mcast_addr; // multicast target address
	int tail_red_id; // tail active red_id
	int next_red_id; // next available red_id
	int max_red_id; // limit total concurrency
	int seqno; // rolling seqno for packets
	int close_state; // the state of the close operation
	bool has_closed; // true after a mc close call
	bool has_error; // true if any error
	bool is_multicast; // true if multicast address
	bool arm_disable; // arm-disable for testing
	bool retry_disable; // retry-disable for testing
	bool is_joined; // true if joined
	bool rx_discard; // true to discard RX events
	enum cxi_traffic_class tc; // traffic class
	enum cxi_traffic_class_type tc_type; // traffic class type
	ofi_atomic32_t send_cnt; // for diagnostics
	ofi_atomic32_t recv_cnt; // for diagnostics
	ofi_atomic32_t pkt_cnt; // for diagnostics
	ofi_atomic32_t seq_err_cnt; // for diagnostics
	ofi_atomic32_t tmout_cnt; // for diagnostics
	ofi_spin_t lock;

	struct cxi_md *reduction_md; // memory descriptor for DMA
	struct cxip_coll_reduction reduction[CXIP_COLL_MAX_CONCUR];
	/* Logical address context for leaf rdma get */
	uint64_t rdma_get_lac_va_tx;
	/* Logical address context recieved by the leaf */
	uint64_t rdma_get_lac_va_rx;
	/* pointer to the source buffer base used in the RDMA */
	uint8_t *root_rdma_get_data_p;
	/* pointer to the dest buffer base used in the RDMA */
	uint8_t *leaf_rdma_get_data_p;
	/* root rdma get memory descriptor, for entire root src buffer */
	struct cxip_md *root_rdma_get_md;
	/* leaf rdma get memory descriptor, for entire leaf dest buffer */
	struct cxip_md *leaf_rdma_get_md;
};

/* Function declarations */
void cxip_coll_reset_mc_ctrs(struct fid_mc *mc);

void cxip_coll_get_mc_ctrs(struct fid_mc *mc, struct coll_counters *counters);

void cxip_coll_init_metrics(void);

void cxip_coll_get_metrics(struct cxip_coll_metrics *metrics);

void cxip_coll_init(struct cxip_ep_obj *ep_obj);

int cxip_coll_enable(struct cxip_ep *ep);

int cxip_coll_disable(struct cxip_ep_obj *ep_obj);

void cxip_coll_close(struct cxip_ep_obj *ep_obj);

void cxip_coll_populate_opcodes(void);

int cxip_coll_send(struct cxip_coll_reduction *reduction, int av_set_idx,
		   const void *buffer, size_t buflen, struct cxi_md *md);

int cxip_coll_send_red_pkt(struct cxip_coll_reduction *reduction,
			   const struct cxip_coll_data *coll_data, bool arm,
			   bool retry, bool root_result_pkt);

void cxip_capture_red_id(int *red_id_buf);

ssize_t cxip_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context);

ssize_t cxip_broadcast(struct fid_ep *ep, void *buf, size_t count, void *desc,
		       fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, uint64_t flags,
		       void *context);

ssize_t cxip_reduce(struct fid_ep *ep, const void *buf, size_t count,
		    void *desc, void *result, void *result_desc,
		    fi_addr_t coll_addr, fi_addr_t root_addr,
		    enum fi_datatype datatype, enum fi_op op, uint64_t flags,
		    void *context);

ssize_t cxip_allreduce(struct fid_ep *ep, const void *buf, size_t count,
		       void *desc, void *result, void *result_desc,
		       fi_addr_t coll_addr, enum fi_datatype datatype,
		       enum fi_op op, uint64_t flags, void *context);

int cxip_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			 const struct fid_av_set *coll_av_set, uint64_t flags,
			 struct fid_mc **mc, void *context);

void cxip_coll_progress_join(struct cxip_ep_obj *ep_obj);

void cxip_coll_progress_cq_poll(struct cxip_ep_obj *ep_obj);

int cxip_coll_arm_disable(struct fid_mc *mc, bool disable);

void cxip_coll_limit_red_id(struct fid_mc *mc, int max_red_id);

void cxip_coll_drop_send(struct cxip_coll_reduction *reduction);

void cxip_coll_drop_recv(struct cxip_coll_reduction *reduction);

int cxip_coll_trace_attr cxip_coll_prod_trace(const char *fmt, ...);

void cxip_coll_print_prod_trace(void);

#endif /* _CXIP_COLL_H_ */
