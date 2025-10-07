/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2025 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef _FI_PROV_OPX_DOMAIN_H_
#define _FI_PROV_OPX_DOMAIN_H_

#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <uuid/uuid.h>
#include <uthash.h>

#include "rdma/fi_domain.h"

#include "rdma/opx/fi_opx_reliability.h"

#include "rdma/opx/fi_opx_tid_domain.h"

#include "rdma/opx/opx_hmem_domain.h"
#include "rdma/opx/opx_hfisvc_keyset.h"

#ifdef HFISVC
#include "hfisvc_client/hfisvc_client.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct fi_opx_ep; /* forward declaration */

struct opx_tid_fabric;
struct opx_hmem_fabric;

struct fi_opx_fabric {
	struct fid_fabric fabric_fid;

	int64_t		       ref_cnt;
	struct opx_tid_fabric *tid_fabric;
#ifdef OPX_HMEM
	struct opx_hmem_fabric *hmem_fabric;
#endif
};

struct fi_opx_node {
	volatile uint64_t ep_count;
};

#define OPX_JOB_KEY_STR_SIZE	33
#define OPX_DEFAULT_JOB_KEY_STR "00112233445566778899aabbccddeeff"

#define OPX_SDMA_BOUNCE_BUF_MIN	      FI_OPX_SDMA_MIN_PAYLOAD_BYTES_MIN
#define OPX_SDMA_BOUNCE_BUF_THRESHOLD FI_OPX_SDMA_MIN_PAYLOAD_BYTES_DEFAULT
#define OPX_SDMA_BOUNCE_BUF_MAX	      FI_OPX_SDMA_MIN_PAYLOAD_BYTES_MAX

struct fi_opx_domain {
	struct fid_domain     domain_fid;
	struct fi_opx_fabric *fabric;

	enum fi_threading     threading;
	enum fi_resource_mgmt resource_mgmt;
	int		      mr_mode;
	enum fi_progress      data_progress;

	uuid_t unique_job_key;
	char   unique_job_key_str[OPX_JOB_KEY_STR_SIZE];

	char *progress_affinity_str;

	uint32_t rx_count;
	uint32_t tx_count;
	uint8_t	 ep_count;

	uint64_t	  num_mr_keys;
	struct fi_opx_mr *mr_hashmap;

	struct opx_tid_domain *tid_domain;
#ifdef OPX_HMEM
	struct opx_hmem_domain *hmem_domain;
#endif

#ifdef HFISVC
	struct {
		hfisvc_client_t handle;

		/**
		 * @brief Command queue used by the domain for issuing commands to the
		 * hfisvc where we are opening/closing hfisvc memory regions.
		 */
		hfisvc_client_command_queue_t mr_command_queue;

		/**
		 * @brief Completion queue used by the domain for handling completions from the
		 * hfisvc where we are opening/closing hfisvc memory regions.
		 */
		hfisvc_client_completion_queue_t mr_completion_queue;

		opx_hfisvc_keyset_t access_key_set;
		int64_t		    ref_cnt;
		hfisvc_client_key_t client_key;
		uint32_t	    padding;
	} hfisvc;
#endif
	uint8_t use_hfisvc;
	uint8_t padding[7];
	int64_t ref_cnt;
};

struct fi_opx_av {
	/* == CACHE LINE 0 == */

	struct fid_av	      av_fid; /* 32 bytes */
	struct fi_opx_domain *domain;
	void		     *map_addr;
	int64_t		      ref_cnt;
	uint32_t	      addr_count;
	enum fi_av_type	      type;
	unsigned	      ep_tx_count;

	/* == CACHE LINE 1..20 == */

	struct fi_opx_ep *ep_tx[160];

	/* == ALL OTHER CACHE LINES == */

	union fi_opx_addr *table_addr; /* allocated buffer to free */
	uint64_t	   rx_ctx_bits;
	uint32_t	   table_count; /* table, not av, count */
};

enum opx_mr_hfisvc_state {
	OPX_MR_HFISVC_NOT_REGISTERED = 0,
	OPX_MR_HFISVC_PENDING_OPEN,
	OPX_MR_HFISVC_OPENED,
	OPX_MR_HFISVC_PENDING_CLOSE,
	OPX_MR_HFISVC_CLOSED,
};

struct fi_opx_mr {
	/* == CACHE LINE 0-2 == */
	struct fid_mr	      mr_fid; // 40 bytes
	struct fi_mr_attr     attr;   // 112 bytes
	struct fi_opx_domain *domain;
	struct fi_opx_ep     *ep;
	void		     *base_addr;
	struct iovec	      iov;

	/* == CACHE LINE 3 == */
	uint64_t	    flags;
	uint64_t	    cntr_bflags;
	struct fi_opx_cntr *cntr;
	struct {
		union {
			uint32_t reserved;
#ifdef HFISVC
			hfisvc_client_mr_t mr_handle;
#endif
		};
		enum opx_mr_hfisvc_state state;
		uint32_t		 access_key;
		uint32_t		 padding;
	} hfisvc;
	uint64_t hmem_dev_reg_handle;
	uint8_t	 hmem_unified;
	uint8_t	 unused[7];
	uint64_t unused_cacheline3_qw;

	/* == CACHE LINE 4 == */
	UT_hash_handle hh; // 56 bytes
	uint64_t       unused_cacheline4_qw;
} __attribute__((__aligned__(FI_OPX_CACHE_LINE_SIZE))) __attribute__((__packed__));
OPX_COMPILE_TIME_ASSERT(sizeof(struct fi_opx_mr) == (FI_OPX_CACHE_LINE_SIZE * 5),
			"Size of fi_opx_mr should be 5 cachelines!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_mr, flags) == (FI_OPX_CACHE_LINE_SIZE * 3),
			"Offset of fi_opx_mr->flags should start at cacheline 3!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_mr, hh) == (FI_OPX_CACHE_LINE_SIZE * 4),
			"Offset of fi_opx_mr->hh should start at cacheline 4!");

static inline uint32_t fi_opx_domain_get_tx_max(struct fid_domain *domain)
{
	return 160;
}

static inline uint32_t fi_opx_domain_get_rx_max(struct fid_domain *domain)
{
	return 160;
}

#ifdef HFISVC

__OPX_FORCE_INLINE__
void opx_domain_hfisvc_poll(struct fi_opx_domain *opx_domain)
{
	struct hfisvc_client_cq_entry hfisvc_out[64];
	size_t n = hfisvc_client_cq_read(opx_domain->hfisvc.mr_completion_queue, 0ul /* flags */, hfisvc_out, 64);
	while (n > 0) {
		for (size_t i = 0; i < n; ++i) {
			if (hfisvc_out[i].status != HFISVC_CLIENT_CQ_ENTRY_STATUS_SUCCESS) {
				// TODO: FI_WARN, post some kind of error to the error queue
				fprintf(stderr, "Completion error: status was %d\n", hfisvc_out[i].status);
				abort();
			}
			assert(hfisvc_out[i].status == HFISVC_CLIENT_CQ_ENTRY_STATUS_SUCCESS);
			assert(hfisvc_out[i].type == HFISVC_CLIENT_CQ_ENTRY_TYPE_MR);

			struct fi_opx_mr  *opx_mr    = (struct fi_opx_mr *) hfisvc_out[i].app_context;
			hfisvc_client_mr_t mr_handle = hfisvc_out[i].type_mr.mr;

			if (opx_mr->hfisvc.state == OPX_MR_HFISVC_PENDING_OPEN) {
				opx_mr->hfisvc.mr_handle = mr_handle;
				opx_mr->hfisvc.state	 = OPX_MR_HFISVC_OPENED;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_PENDING_CLOSE) {
				assert(opx_mr->hfisvc.mr_handle == mr_handle);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_CLOSED;
			} else {
				// TODO: FI_WARN, post some kind of error to the error queue
				fprintf(stderr, "(%d) %s:%s():%d Got unexpected completion for opx_mr=%p state=%d\n",
					getpid(), __FILE__, __func__, __LINE__, opx_mr, opx_mr->hfisvc.state);
				abort();
			}
		}
		n = hfisvc_client_cq_read(opx_domain->hfisvc.mr_completion_queue, 0ul /* flags */, hfisvc_out, 64);
	}
}

int opx_domain_hfisvc_init(struct fi_opx_domain *domain, const enum hfisvc_client_connect_type type, const int fd);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_PROV_OPX_DOMAIN_H_ */
