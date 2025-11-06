/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI benchmark common structures and functions */

#ifndef LIBCXI_UTILS_COMMON_H
#define LIBCXI_UTILS_COMMON_H


#include <sys/time.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>

#include "libcxi.h"

// clang-format off
#define DFLT_PORT       49194
#define CTRL_MSG_BUF_SZ 1024

#define PORTALS_MTU               2048
#define MAX_IDC_RESTRICTED_SIZE   224
#define MAX_IDC_UNRESTRICTED_SIZE 192
#define MAX_CQ_DEPTH              65532 /* (4MiB / 64B) - 4 for queue status */
#define MAX_MSG_SIZE              ((1UL << 32) - 1) /* 4GiB - 1 */
#define TWO_MB                    (1UL << 21)
#define ONE_GB                    (1UL << 30)

#define SEC2USEC 1000000
#define SEC2NSEC 1000000000

#define VAL_NO_HRP        -2
#define VAL_NO_IDC        -3
#define VAL_BUF_SZ        -4
#define VAL_BUF_ALIGN     -5
#define VAL_NO_LL         -6
#define VAL_REPORT_ALL    -7
#define VAL_WARMUP        -8
#define VAL_ITER_DELAY    -9
#define VAL_FETCHING      -10
#define VAL_MATCHING      -11
#define VAL_UNRESTRICTED  -12
#define VAL_TX_GPU        -13
#define VAL_RX_GPU        -14
#define VAL_USE_HP        -15
#define VAL_EMU_MODE      -16
#define VAL_IGNORE_CPU_FREQ_MISMATCH -17

#define SIZE_W         12
#define COUNT_W        10
#define BW_W           8
#define BW_FRAC_W      2
#define MRATE_W        15
#define MRATE_FRAC_W   6
#define LAT_DEC_W      7
#define LAT_FRAC_W     2
#define LAT_W          (LAT_DEC_W + 1 + LAT_FRAC_W)
#define LAT_ALL_DEC_W  7
#define LAT_ALL_FRAC_W 3
#define LAT_ALL_W      (LAT_ALL_DEC_W + 1 + LAT_ALL_FRAC_W)
#define NO_TIMEOUT     0
#define POLL_ONCE      1
#define DFLT_HANDSHAKE_TIMEOUT (5 * SEC2USEC)

#define NEXT_MULTIPLE(X, Y) (((X + Y - 1) / Y) * Y)
#define TV_NSEC_DIFF(T0, T1)                                                   \
	(((T1.tv_sec - T0.tv_sec) * SEC2NSEC) + T1.tv_nsec - T0.tv_nsec)
// clang-format on

/* Simple control message header to verify sizes */
struct ctrl_msg {
	uint32_t pyld_len;
	/* payload follows */
} __attribute__((packed));

#define MAX_CTRL_DATA_BYTES (CTRL_MSG_BUF_SZ - sizeof(struct ctrl_msg))

enum clock_type {
	CLOCK_GETTIME,
	CYCLES,
};

/* Control connection state */
struct ctrl_connection {
	const char *dst_addr;
	uint16_t dst_port;
	uint16_t src_port;
	bool is_server;
	bool is_loopback;
	bool connected;

	struct timeval last_rcvtmo;
	int fd;
	char buf[CTRL_MSG_BUF_SZ]; /* unused */
};

/* Combined buffer and associated CXI Memory Descriptor */
struct ctx_buffer {
	void *buf;
	struct cxi_md *md;
	size_t len;
};

enum ctx_buf_pat {
	CTX_BUF_PAT_ZERO,
	CTX_BUF_PAT_URAND, /* Quite slow with larger buf sizes */
	CTX_BUF_PAT_A5,
	CTX_BUF_PAT_NONE
};

enum hugepage_type {
	HP_DISABLED,
	HP_2M,
	HP_1G,
	HP_NOT_APPLIC
};
static const char *const hugepage_names[] = { [HP_2M] = "2M", [HP_1G] = "1G" };

struct ctx_buf_opts {
	size_t length;
	enum ctx_buf_pat pattern;
	enum hugepage_type hp;
};

/* Common CXI context resources */
struct cxi_ep_addr {
	uint32_t nic;
	uint32_t pid;
};

struct cxi_context {
	/* Base Context */
	uint32_t dev_id;
	uint32_t vni;
	struct cxi_ep_addr loc_addr;
	struct cxi_ep_addr rmt_addr;

	struct cxil_dev *dev;
	struct cxil_lni *lni;
	struct cxil_domain *dom;

	/* Initiator Context */
	uint8_t index_ext;
	union c_fab_addr dfa;
	struct cxi_cp *cp;
	struct cxi_cp *hrp_cp;
	struct ctx_buffer *ini_eq_buf;
	struct cxi_eq *ini_eq;
	struct cxi_ct *ini_ct;
	struct ctx_buffer *ini_buf;
	struct cxi_cq *ini_cq;
	struct cxi_cq *ini_trig_cq;

	struct ctx_buffer *ini_rdzv_eq_buf;
	struct cxi_eq *ini_rdzv_eq;
	struct cxi_ct *ini_rdzv_ct;
	struct cxi_cq *ini_rdzv_trig_cq;
	struct cxi_cq *ini_rdzv_pte_cq;
	struct cxil_pte *ini_rdzv_pte;

	/* Target Context */
	struct ctx_buffer *tgt_eq_buf;
	struct cxi_eq *tgt_eq;
	struct ctx_buffer *tgt_buf;
	struct cxi_cq *tgt_cq;
	struct cxil_pte *tgt_pte;
	struct cxil_pte *tgt_final_lat_pte;
	struct cxi_ct *tgt_ct;
	struct cxi_cq *tgt_trig_cq;
};

struct cxi_ctx_ini_opts {
	int pid_offset;
	bool alloc_hrp;
	bool alloc_ct;
	bool use_gpu_buf;
	int gpu_id;
	bool alloc_rdzv;
	struct cxi_eq_attr eq_attr;
	struct ctx_buf_opts buf_opts;
	struct cxi_cq_alloc_opts cq_opts;
};

struct cxi_ctx_tgt_opts {
	int pte_index;
	bool alloc_ct;
	bool use_gpu_buf;
	int gpu_id;
	bool use_final_lat;
	struct cxi_eq_attr eq_attr;
	struct ctx_buf_opts buf_opts;
	struct cxi_cq_alloc_opts cq_opts;
	struct cxi_pt_alloc_opts pt_opts;
};

/* Variable type AMO operand */
struct amo_operand {
	int64_t op_int;
	uint64_t op_uint;
	uint64_t op_uint_w2;
	double op_fp_real;
	double op_fp_imag;
};

/* GPU types */
enum gpu_types {
	AMD,
	NVIDIA,
	INTEL
};
static const char *const gpu_names[] = { [AMD] = "AMD", [NVIDIA] = "NVIDIA", [INTEL] = "INTEL" };

/* This is used to store and share options that may differ between the client
 * and the server. Any changes to the size or ordering here should be
 * accompanied by changes to the major or minor versions of every utility that
 * uses this struct.
 */
struct loc_util_opts {
	uint32_t svc_id;
	uint32_t dev_id;
	uint16_t port;
	int8_t gpu_type;
	uint8_t rsvd1;
	uint32_t rsvd2;
	uint64_t tx_gpu     : 8;
	uint64_t use_tx_gpu : 1;
	uint64_t rx_gpu     : 8;
	uint64_t use_rx_gpu : 1;
	uint64_t rsvd3      : 46;
} __attribute__((packed));

/* This is used to pass the client's command line options to the server.
 * Any changes to the size or ordering here should be accompanied by changes
 * to the major or minor versions of every utility that uses this struct.
 */
struct util_opts {
	enum clock_type clock;
	bool ignore_cpu_freq_mismatch;

	/* Options used by run_bw_active and run_lat_active */
	uint64_t iters;
	uint64_t warmup;
	uint64_t iter_delay;
	uint32_t duration;
	uint16_t list_size;
	uint16_t bidirectional : 1;
	uint16_t report_all    : 1;
	uint16_t print_gbits   : 1;
	uint16_t emu_mode      : 1;

	/* Shared individual options */
	uint16_t hugepages     : 2;
	uint16_t rsvd_0        : 10;
	uint64_t min_size;
	uint64_t max_size;
	uint64_t max_buf_size;
	uint64_t buf_size;
	uint64_t buf_align;
	uint8_t use_hrp        : 1;
	uint8_t use_idc        : 1;
	uint8_t use_ll         : 1;
	uint8_t use_rdzv       : 1;
	uint8_t fetching       : 1;
	uint8_t matching       : 1;
	uint8_t unrestricted   : 1;
	uint8_t rsvd_1         : 1;

	/* Shared AMO options used by amo_* functions */
	uint8_t atomic_op;
	uint8_t cswap_op;
	uint8_t atomic_type;

	/* Options that may vary between client and server */
	struct loc_util_opts loc_opts;
	struct loc_util_opts rmt_opts;
} __attribute__((packed));

enum iter_state { SEND, RECV, RESP, DONE, SINGLE_RECV, RESUME_SEND };

#define MAX_HDR_LEN 100
struct util_context {
	struct ctrl_connection ctrl;
	struct cxi_context cxi;
	struct util_opts opts;

	double clock_ghz;

	/* Common state */
	char header[MAX_HDR_LEN];
	size_t size;
	uint64_t count;
	size_t buf_granularity;
	uint64_t last_lat;

	/* AMO state */
	struct amo_operand op1;
	struct amo_operand op2;
	struct amo_operand tgt_op;
	uint64_t fetch_offset;

	/* send_bw/lat state */
	enum iter_state istate;
	bool final_lat_sent;
	bool final_lat_recv;

	union c_cmdu dma_cmd;
	union c_cmdu idc_cmd;
	union c_cmdu ct_cmd;
	union c_cmdu ini_rdzv_ct_cmd;
	union c_cmdu tgt_ct_cmd;
	union c_cmdu tgt_rdzv_get_cmd;
};

static const char *const amo_op_strs[] = {
	[C_AMO_OP_MIN] = "MIN",	    [C_AMO_OP_MAX] = "MAX",
	[C_AMO_OP_SUM] = "SUM",	    [C_AMO_OP_LOR] = "LOR",
	[C_AMO_OP_LAND] = "LAND",   [C_AMO_OP_BOR] = "BOR",
	[C_AMO_OP_BAND] = "BAND",   [C_AMO_OP_LXOR] = "LXOR",
	[C_AMO_OP_BXOR] = "BXOR",   [C_AMO_OP_SWAP] = "SWAP",
	[C_AMO_OP_CSWAP] = "CSWAP", [C_AMO_OP_AXOR] = "AXOR"
};

static const char *const amo_cswap_op_strs[] = {
	[C_AMO_OP_CSWAP_EQ] = "EQ", [C_AMO_OP_CSWAP_NE] = "NE",
	[C_AMO_OP_CSWAP_LE] = "LE", [C_AMO_OP_CSWAP_LT] = "LT",
	[C_AMO_OP_CSWAP_GE] = "GE", [C_AMO_OP_CSWAP_GT] = "GT"
};

static const char *const amo_type_strs[] = {
	[C_AMO_TYPE_INT8_T] = "INT8",
	[C_AMO_TYPE_UINT8_T] = "UINT8",
	[C_AMO_TYPE_INT16_T] = "INT16",
	[C_AMO_TYPE_UINT16_T] = "UINT16",
	[C_AMO_TYPE_INT32_T] = "INT32",
	[C_AMO_TYPE_UINT32_T] = "UINT32",
	[C_AMO_TYPE_INT64_T] = "INT64",
	[C_AMO_TYPE_UINT64_T] = "UINT64",
	[C_AMO_TYPE_FLOAT_T] = "FLOAT",
	[C_AMO_TYPE_FLOAT_COMPLEX_T] = "FLOAT_COMPLEX",
	[C_AMO_TYPE_DOUBLE_T] = "DOUBLE",
	[C_AMO_TYPE_DOUBLE_COMPLEX_T] = "DOUBLE_COMPLEX",
	[C_AMO_TYPE_UINT128_T] = "UINT128"
};

static const int amo_type_sizes[] = {
	[C_AMO_TYPE_INT8_T] = 1,    [C_AMO_TYPE_UINT8_T] = 1,
	[C_AMO_TYPE_INT16_T] = 2,   [C_AMO_TYPE_UINT16_T] = 2,
	[C_AMO_TYPE_INT32_T] = 4,   [C_AMO_TYPE_UINT32_T] = 4,
	[C_AMO_TYPE_INT64_T] = 8,   [C_AMO_TYPE_UINT64_T] = 8,
	[C_AMO_TYPE_FLOAT_T] = 4,   [C_AMO_TYPE_FLOAT_COMPLEX_T] = 8,
	[C_AMO_TYPE_DOUBLE_T] = 8,  [C_AMO_TYPE_DOUBLE_COMPLEX_T] = 16,
	[C_AMO_TYPE_UINT128_T] = 16
};

/* Control messaging functions */
int ctrl_connect(struct ctrl_connection *ctrl, const char *name,
		 const char *version, struct util_opts *opts,
		 struct cxi_ep_addr *loc_addr, struct cxi_ep_addr *rmt_addr);
int ctrl_close(struct ctrl_connection *ctrl);
int ctrl_exchange_data(struct ctrl_connection *ctrl, const void *client_buf,
		       size_t cbuf_size, void *server_buf, size_t sbuf_size);
int ctrl_barrier(struct ctrl_connection *ctrl, uint64_t tmo_usec, char *label);

/* CXI context functions */
int ctx_alloc(struct cxi_context *ctx, uint32_t dev_id, uint32_t svc_id);
void ctx_destroy(struct cxi_context *ctx);
int ctx_alloc_cp(struct cxi_context *ctx, enum cxi_traffic_class tc,
		 enum cxi_traffic_class_type tc_type, struct cxi_cp **cp);
int ctx_alloc_buf(struct cxi_context *ctx, size_t buf_len,
		  enum ctx_buf_pat pattern, enum hugepage_type hp,
		  struct ctx_buffer **buf);
int ctx_alloc_gpu_buf(struct cxi_context *ctx, size_t buf_len,
		      enum ctx_buf_pat pattern, struct ctx_buffer **buf,
		      int gpu_id);
int ctx_alloc_eq(struct cxi_context *ctx, struct cxi_eq_attr *attr,
		 struct cxi_md *md, struct cxi_eq **eq);
int ctx_alloc_ct(struct cxi_context *ctx, struct cxi_ct **ct);
int ctx_alloc_cq(struct cxi_context *ctx, struct cxi_eq *eq,
		 struct cxi_cq_alloc_opts *opts, struct cxi_cq **cq);
int ctx_alloc_pte(struct cxi_context *ctx, struct cxi_eq *eq,
		  struct cxi_pt_alloc_opts *opts, int pid_offset,
		  struct cxil_pte **pte);

/* Shared benchmark functions */
uint64_t gettimeofday_usec(void);
int init_time_counters(struct cxil_dev *dev);
int clock_gettime_or_counters(clockid_t clock_mode, struct timespec *ts,
							  struct cxil_dev *dev);
void print_separator(size_t len);
int active_sleep(uint64_t usec, struct cxil_dev *dev);
int get_event(struct cxi_eq *eq, enum c_event_type type,
	      const union c_event **ret_event, struct timespec *ts,
	      uint64_t timeout_usec, struct cxil_dev *dev);
int wait_for_ct(struct cxi_eq *eq, uint64_t timeout_usec, char *label);
int inc_ct(struct cxi_cq *cq, struct c_ct_cmd *cmd, size_t inc);
void inc_tx_buf_offsets(struct util_context *util, uint64_t *rmt,
			uint64_t *loc);
int enable_pte(struct cxi_cq *cq, struct cxi_eq *eq, uint16_t ptn);
int append_le(struct cxi_cq *cq, struct cxi_eq *eq, struct ctx_buffer *ctx_buf,
	      size_t offset, uint32_t flags, uint16_t ptlte_index, uint16_t ct,
	      uint16_t buffer_id);
int append_me(struct cxi_cq *cq, struct cxi_eq *eq, struct ctx_buffer *ctx_buf,
	      size_t offset, uint32_t flags, uint16_t ptlte_index, uint16_t ct,
	      uint32_t match_id, uint64_t match_bits, uint64_t ignore_bits,
	      uint16_t buffer_id);
int set_to_dflt_cp(struct util_context *util, struct cxi_cq *cq);
int set_to_hrp_cp(struct util_context *util, struct cxi_cq *cq);
int alloc_ini(struct cxi_context *ctx, struct cxi_ctx_ini_opts *opts);
int alloc_tgt(struct cxi_context *ctx, struct cxi_ctx_tgt_opts *opts);
int sw_rdzv_get(struct util_context *util, struct c_event_target_long ev);
int run_bw_active(struct util_context *util,
		  int (*do_iter)(struct util_context *util));
int run_bw_passive(struct util_context *util);
int run_lat_active(struct util_context *util,
		   int (*do_iter)(struct util_context *util));
int run_lat_passive(struct util_context *util);
void parse_common_opt(char c, struct util_opts *opts, const char *name,
		      const char *version, void (*usage)(void));
void parse_server_addr(int argc, char **argv, struct ctrl_connection *ctrl,
		       uint16_t port);
int get_hugepage_type(char *type);
uint32_t get_free_hugepages(size_t hp_size_in_bytes);
int get_hugepages_needed(struct util_opts *opts, bool ini_buf, bool tgt_buf);
void print_loc_opts(struct util_opts *opts, bool is_server);
void print_hugepage_opts(struct util_opts *opts, int num_hp);

/* Shared AMO benchmark functions */
void amo_validate_op_and_type(int atomic_op, int cswap_op, int atomic_type);
void amo_init_tgt_op(struct util_context *util);
void amo_init_op1(struct util_context *util);
void amo_init_op2(struct util_context *util);
void amo_update_op1(struct util_context *util);
void amo_update_op2(struct util_context *util);

/* Shared GPU functions */
extern int (*g_malloc)(void **devPtr, size_t size);
extern int (*g_free)(void *devPtr);
extern int (*g_memset)(void *devPtr, int value, size_t size);
extern int (*g_memcpy)(void *dst, const void *src, size_t size, int kind);
extern int (*g_set_device)(int deviceId);
extern int (*g_get_device)(int *deviceId);
extern int (*g_device_count)(int *count);
extern int (*g_mem_properties)(const void *addr, void **base, size_t *size, int *dma_buf_fd);
extern int g_memcpy_kind_htod;
int gpu_lib_init(enum gpu_types g_type);
void gpu_lib_fini(enum gpu_types g_type);
int hip_lib_init(void);
void hip_lib_fini(void);
int cuda_lib_init(void);
void cuda_lib_fini(void);
int ze_lib_init(void);
void ze_lib_fini(void);
int gpu_malloc(struct ctx_buffer *win, size_t len);
int gpu_free(void *devPtr);
int gpu_memset(void *devPtr, int value, size_t size);
int gpu_memcpy(void *st, const void *src, size_t size, int kind);
int get_gpu_device_count(void);
int get_gpu_device(void);
int get_gpu_type(char *gpu_name);
int set_gpu_device(int dev_id);

extern int s_page_size;

#endif /* LIBCXI_UTILS_COMMON_H */
