/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_COMMON_H_
#define _CXIP_COMMON_H_


#include <stdint.h>
#include <ofi_list.h>

/* Forward declarations */
struct cxip_domain;
struct cxip_req;
struct cxip_ux_send;

/* Macros */
#define _CXIP_PROV_H_

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

#define FLOOR(a, b) ((long long)(a) - (((long long)(a)) % (b)))

#define CEILING(a, b) ((long long)(a) <= 0LL ? 0 : (FLOOR((a)-1, b) + (b)))

#define CXIP_ALIGN_MASK(x, mask) (((x) + (mask)) & ~(mask))

#define CXIP_ALIGN(x, a) CXIP_ALIGN_MASK(x, (typeof(x))(a) - 1)

#define CXIP_ALIGN_DOWN(x, a) CXIP_ALIGN((x) - ((a) - 1), (a))

#define CXIP_PATH_MAX			256

#define CXIP_BUFFER_ID_MAX		(1 << 16)

#define CXIP_TX_COMP_MODES		(FI_INJECT_COMPLETE | \
					 FI_TRANSMIT_COMPLETE | \
					 FI_DELIVERY_COMPLETE | \
					 FI_MATCH_COMPLETE)

#define CXIP_TX_OP_FLAGS		(FI_INJECT | \
					 FI_COMPLETION | \
					 CXIP_TX_COMP_MODES | \
					 FI_REMOTE_CQ_DATA | \
					 FI_MORE | \
					 FI_FENCE)

#define CXIP_RX_OP_FLAGS		(FI_COMPLETION | \
					 FI_MULTI_RECV | \
					 FI_MORE)

#define CXIP_RX_IGNORE_OP_FLAGS		(FI_REMOTE_CQ_DATA | \
					 FI_INJECT)

#define CXIP_WRITEMSG_ALLOWED_FLAGS	(FI_INJECT | \
					 FI_COMPLETION | \
					 FI_MORE | \
					 FI_FENCE | \
					 CXIP_TX_COMP_MODES)

#define CXIP_READMSG_ALLOWED_FLAGS	(FI_COMPLETION | \
					 FI_MORE | \
					 FI_FENCE | \
					 CXIP_TX_COMP_MODES)

#define CXIP_AMO_MAX_IOV		1

#define CXIP_REMOTE_CQ_DATA_SZ		8

#define CXIP_RDZV_THRESHOLD		16384

#define CXIP_OFLOW_BUF_SIZE		(12*1024*1024)

#define CXIP_OFLOW_BUF_MIN_POSTED	3

#define CXIP_OFLOW_BUF_MAX_CACHED	(CXIP_OFLOW_BUF_MIN_POSTED * 3)

#define CXIP_DEFAULT_MR_CACHE_MAX_CNT	4096

#define CXIP_DEFAULT_MR_CACHE_MAX_SIZE	-1

#define CXIP_SAFE_DEVMEM_COPY_THRESH	4096

#define CXIP_CAPS (CXIP_DOM_CAPS | CXIP_EP_CAPS)

#define CXIP_INJECT_SIZE		C_MAX_IDC_PAYLOAD_UNR

#define CXIP_MAX_TX_SIZE		16384U

#define CXIP_DEFAULT_TX_SIZE		1024U

#define CXI_PROV_LE_PER_EP		1024U

#define LES_PER_EP_MAX			16384U

#define CXIP_MAX_RX_SIZE		(LES_PER_EP_MAX - CXI_PROV_LE_PER_EP)

#define CXIP_DEFAULT_RX_SIZE		1024U

#define CXIP_MAJOR_VERSION		0

#define CXIP_MINOR_VERSION		1

#define CXIP_PROV_VERSION		FI_VERSION(CXIP_MAJOR_VERSION, \
						   CXIP_MINOR_VERSION)

#define CXIP_FI_VERSION			FI_VERSION(2, 4)

#define CXIP_WIRE_PROTO_VERSION		1

#define CXIP_PAUSE()

#define CXIP_PTL_IDX_RXQ				0

#define CXIP_PTL_IDX_RNR_RXQ				1

#define CXIP_PTL_IDX_WRITE_MR_OPT_BASE			17

#define CXIP_PTL_IDX_READ_MR_OPT_BASE			128

#define CXIP_PTL_IDX_MR_OPT_CNT				100

#define CXIP_PTL_IDX_PROV_NUM_CACHE_IDX			8

#define CXIP_PTL_IDX_PROV_MR_OPT_CNT				\
	(CXIP_PTL_IDX_MR_OPT_CNT - CXIP_PTL_IDX_PROV_NUM_CACHE_IDX)

#define CXIP_PTL_IDX_WRITE_MR_OPT(key)		\
	(CXIP_PTL_IDX_WRITE_MR_OPT_BASE +	\
	 CXIP_MR_UNCACHED_KEY_TO_IDX(key))

#define CXIP_PTL_IDX_READ_MR_OPT(key)		\
	(CXIP_PTL_IDX_READ_MR_OPT_BASE +	\
	 CXIP_MR_UNCACHED_KEY_TO_IDX(key))

#define CXIP_PTL_IDX_WRITE_PROV_CACHE_MR_OPT(lac)		\
	(CXIP_PTL_IDX_WRITE_MR_OPT_BASE + (lac))

#define CXIP_PTL_IDX_READ_PROV_CACHE_MR_OPT(lac)		\
	(CXIP_PTL_IDX_READ_MR_OPT_BASE + (lac))

#define CXIP_PTL_IDX_WRITE_MR_STD		117

#define CXIP_PTL_IDX_RDZV_DEST			127

#define CXIP_PTL_IDX_COLL			6

#define CXIP_PTL_IDX_CTRL			CXIP_PTL_IDX_WRITE_MR_STD

#define CXIP_PTL_IDX_READ_MR_STD		228

#define CXIP_PTL_IDX_RDZV_RESTRICTED_BASE	229

#define CXIP_PTL_IDX_RDZV_RESTRICTED(lac)			\
	(CXIP_PTL_IDX_RDZV_RESTRICTED_BASE + (lac))

#define CXIP_PTL_IDX_RDZV_SRC			255

#define CXIP_NUM_CACHED_KEY_LE 8

#define CXIP_TX_ID_WIDTH	11

#define CXIP_RDZV_ID_CMD_WIDTH	8

#define CXIP_RDZV_ID_HIGH_WIDTH 7

#define CXIP_TOTAL_RDZV_ID_WIDTH (CXIP_RDZV_ID_CMD_WIDTH +	\
				  CXIP_RDZV_ID_HIGH_WIDTH)

#define CXIP_CS_TAG_WIDTH	40

#define CXIP_VNI_WIDTH		16

#define CXIP_CS_TAG_MASK	((1UL << CXIP_CS_TAG_WIDTH) - 1)

#define CXIP_IS_PROV_MR_KEY_BIT (1ULL << 63)

#define CXIP_KEY_MATCH_BITS(key) ((key) & ~CXIP_IS_PROV_MR_KEY_BIT)

#define CXI_PLATFORM_ASIC 0

#define CXI_PLATFORM_NETSIM 1

#define CXI_PLATFORM_Z1 2

#define CXI_PLATFORM_FPGA 3

#define MAX_HW_CPS 16

#define TELEMETRY_ENTRY_NAME_SIZE 64U

#define CXIP_DEF_EVENT_HT_BUCKETS	256

#define	ZB_NOSIM	-1

#define	ZB_ALLSIM	-2

#define CXIP_COUNTER_BUCKETS 31U

#define CXIP_BUCKET_MAX (CXIP_COUNTER_BUCKETS - 1)

#define CXIP_LIST_COUNTS 3U

#define CXIP_SW_RX_TX_INIT_MAX_DEFAULT	1024

#define CXIP_SW_RX_TX_INIT_MIN		64

#define CXIP_DONE_NOTIFY_RETRY_DELAY_US 100

#define CXIP_RDZV_IDS	(1 << CXIP_TOTAL_RDZV_ID_WIDTH)

#define CXIP_RDZV_IDS_MULTI_RECV (1 << CXIP_RDZV_ID_CMD_WIDTH)

#define CXIP_TX_IDS	(1 << CXIP_TX_ID_WIDTH)

#define RDZV_SRC_LES 8U

#define RDZV_NO_MATCH_PTES 8U

#define CXIP_RNR_TIMEOUT_US	500000

#define CXIP_NUM_RNR_WAIT_QUEUE	5

#define _CXIP_DBG(subsys, fmt,  ...) \
	FI_DBG(&cxip_prov, subsys, "%s: " fmt "", cxip_env.hostname, \
	       ##__VA_ARGS__)

#define _CXIP_INFO(subsys, fmt, ...) \
	FI_INFO(&cxip_prov, subsys, "%s: " fmt "", cxip_env.hostname, \
		##__VA_ARGS__)

#define _CXIP_WARN(subsys, fmt, ...) \
	FI_WARN(&cxip_prov, subsys, "%s: " fmt "", cxip_env.hostname, \
		##__VA_ARGS__)

#define _CXIP_WARN_ONCE(subsys, fmt, ...) \
	FI_WARN_ONCE(&cxip_prov, subsys, "%s: " fmt "", cxip_env.hostname, \
		     ##__VA_ARGS__)

#define CXIP_UNEXPECTED_EVENT_STS "Unexpected event status, %s rc = %s\n"

#define CXIP_UNEXPECTED_EVENT "Unexpected event %s, rc = %s\n"

#define CXIP_DEFAULT_CACHE_LINE_SIZE 64

#define CXIP_SYSFS_CACHE_LINE_SIZE      \
	"/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"

#define CXIP_HYBRID_RECV_CHECK_INTERVAL (64-1)

#define FC_SW_LE_MSG_FATAL "LE exhaustion during flow control, "\
	"FI_CXI_RX_MATCH_MODE=[hybrid|software] is required\n"

/* Type definitions */
struct cxip_telemetry {
	struct cxip_domain *dom;

	/* List of telemetry entries to being monitored. */
	struct dlist_entry telemetry_list;
};

struct cxip_topo_addr {
	union {
		uint32_t addr;
		struct {
			uint32_t port_num:CXIP_ADDR_PORT_BITS;
			uint32_t switch_num:CXIP_ADDR_SWITCH_BITS;
			uint32_t group_num:CXIP_ADDR_GROUP_BITS;
		} dragonfly;
		struct {
			uint32_t port_num:CXIP_ADDR_FATTREE_PORT_BITS;
			uint32_t switch_num:CXIP_ADDR_FATTREE_SWITCH_BITS;
		} fat_tree;
	};
};

union cxip_def_event_key {
	struct {
		uint64_t initiator	: 32;
		uint64_t rdzv_id	: 15;
		uint64_t pad0		: 16;
		uint64_t rdzv		: 1;
	};
	struct {
		uint64_t start_addr	: 57;
		uint64_t pad1		: 7;
	};
	uint64_t raw;
};

struct cxip_deferred_event {
	struct dlist_entry rxc_entry;
	union cxip_def_event_key key;
	struct cxip_req *req;
	union c_event ev;
	uint64_t mrecv_start;
	uint32_t mrecv_len;

	struct cxip_ux_send *ux_send;
};

struct coll_counters {
	int32_t coll_recv_cnt;
	int32_t send_cnt;
	int32_t recv_cnt;
	int32_t pkt_cnt;
	int32_t seq_err_cnt;
	int32_t tmout_cnt;
};

#endif /* _CXIP_COMMON_H_ */
