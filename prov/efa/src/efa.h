/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_H
#define EFA_H

#include "config.h"

#include <asm/types.h>
#include <errno.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <sys/epoll.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "ofi.h"
#include "ofi_iov.h"
#include "ofi_enosys.h"
#include "ofi_list.h"
#include "ofi_util.h"
#include "ofi_file.h"

#include "efa_base_ep.h"
#include "efa_direct_ope.h"
#include "efa_mr.h"
#include "efa_env.h"
#include "efa_shm.h"
#include "efa_prov.h"
#include "efa_hmem.h"
#include "efa_device.h"
#include "efa_domain.h"
#include "efa_errno.h"
#include "efa_user_info.h"
#include "efa_fork_support.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_util.h"
#include "fi_ext_efa.h"

#define EFA_ABI_VER_MAX_LEN 8

#define SHM_MAX_INJECT_SIZE 4096

#define EFA_FABRIC_NAME 	"efa"
#define EFA_DIRECT_FABRIC_NAME "efa-direct"

#define EFA_EP_TYPE_IS_RDM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM))

#define EFA_EP_TYPE_IS_DGRAM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_DGRAM))

#define EFA_INFO_TYPE_IS_RDM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM) && !strcasecmp(_info->fabric_attr->name, EFA_FABRIC_NAME))

#define EFA_INFO_TYPE_IS_DIRECT(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM) && !strcasecmp(_info->fabric_attr->name, EFA_DIRECT_FABRIC_NAME))

#define EFA_INFO_TYPE_IS_DGRAM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_DGRAM))

#define EFA_DEF_POOL_ALIGNMENT (8)
#define EFA_MEM_ALIGNMENT (64)

/* 4k tx_attr.size + 8k rx_attr.size */
#define EFA_DEF_CQ_SIZE 12288


#define EFA_DEFAULT_RUNT_SIZE (307200)
#define EFA_NEURON_RUNT_SIZE (131072)
#define EFA_DEFAULT_INTER_MAX_MEDIUM_MESSAGE_SIZE (65536)
#define EFA_DEFAULT_INTER_MIN_READ_MESSAGE_SIZE (1048576)
#define EFA_DEFAULT_INTER_MIN_READ_WRITE_SIZE (65536)
#define EFA_DEFAULT_INTRA_MAX_GDRCOPY_FROM_DEV_SIZE (3072)

/*
 * Set alignment to x86 cache line size.
 */
#define EFA_RDM_BUFPOOL_ALIGNMENT	(64)

/*
 * Define bitmask to compare packet generation
 */
#define EFA_RDM_GEN_MASK (EFA_RDM_BUFPOOL_ALIGNMENT - 1)


struct efa_fabric {
	struct util_fabric	util_fabric;
};

struct efa_context {
	uint64_t completion_flags;
	fi_addr_t addr;
};

#if defined(static_assert)
static_assert(sizeof(struct efa_context) <= sizeof(struct fi_context2),
	      "efa_context must not be larger than fi_context2");
#endif

#define EFA_SETUP_IOV(iov, buf, len)           \
	do {                                   \
		iov.iov_base = (void *)buf;    \
		iov.iov_len = (size_t)len;     \
	} while (0)

#define EFA_SETUP_MSG(msg, iov, _desc, count, _addr, _context, _data)    \
	do {                                                             \
		msg.msg_iov = (const struct iovec *)iov;                 \
		msg.desc = (void **)_desc;                               \
		msg.iov_count = (size_t)count;                           \
		msg.addr = (fi_addr_t)_addr;                             \
		msg.context = (void *)_context;                          \
		msg.data = (uint32_t)_data;                              \
	} while (0)

#define EFA_SETUP_RMA_IOV(rma_iov, _addr, _len, _key) \
    do {                                          \
        rma_iov.addr = (uint64_t) _addr;      \
        rma_iov.len = (size_t) _len;          \
        rma_iov.key = (uint64_t) _key;        \
    } while (0)

#define EFA_SETUP_MSG_RMA(msg, iov, _desc, count, _addr, _rma_iov,  \
              _rma_iov_count, _context, _data)          \
    do {                                                        \
        msg.msg_iov = (const struct iovec *) iov;           \
        msg.desc = (void **) _desc;                         \
        msg.iov_count = (size_t) count;                     \
        msg.addr = (fi_addr_t) _addr;                       \
        msg.rma_iov = (const struct fi_rma_iov *) _rma_iov; \
        msg.rma_iov_count = (size_t) _rma_iov_count;        \
        msg.context = (void *) _context;                    \
        msg.data = (uint32_t) _data;                        \
    } while (0)


/**
 * @brief Decide whether a transfer uses the inline data path
 *
 * Shared by the send path (efa_post_send), the RMA write path
 * (efa_rma_post_write), and the work request prepare path (efa_wr_prepare). A
 * transfer uses inline when it fits the given inject threshold and references
 * no HMEM buffer. When it cannot go inline and FI_INJECT was requested, the
 * request cannot be honored: an oversized request is a size error
 * (-FI_EMSGSIZE) and an HMEM buffer is unsupported (-FI_EOPNOTSUPP); the
 * specific reason is logged.
 *
 * @param desc		array of memory descriptors (may be NULL)
 * @param iov_count	number of iov/desc entries
 * @param len		total transfer length (prefix already removed)
 * @param inject_size	inject threshold (inject_msg_size or inject_rma_size)
 * @param flags		operation flags (checked for FI_INJECT)
 * @return 1 to use the inline path, 0 to use the SGL path, or a negative
 *	   libfabric error code when FI_INJECT cannot be honored
 */
static inline int efa_msg_use_inline(void **desc, size_t iov_count,
				     size_t len, size_t inject_size,
				     uint64_t flags)
{
	bool len_fits_inline = len <= inject_size;
	bool is_hmem = false;
	size_t i;

	if (desc) {
		for (i = 0; i < iov_count; i++) {
			if (efa_mr_is_hmem(desc[i])) {
				is_hmem = true;
				break;
			}
		}
	}

	if (len_fits_inline && !is_hmem)
		return 1;

	if (flags & FI_INJECT) {
		if (!len_fits_inline) {
			EFA_WARN(FI_LOG_EP_DATA,
				 "FI_INJECT is requested but message size of "
				 "%zu exceeds inject size of %zu.\n", len,
				 inject_size);
			return -FI_EMSGSIZE;
		}
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_INJECT is not supported for FI_HMEM memory.\n");
		return -FI_EOPNOTSUPP;
	}

	return 0;
}

/**
 * @brief Populate an inline data list from an iov for a transfer
 *
 * Shared by the send path (efa_post_send), the RMA write path
 * (efa_rma_post_write), and the work request prepare path (efa_wr_prepare). For
 * a DGRAM (UD) endpoint the message prefix is stripped from the first entry,
 * since the whole prefix must sit on the first sgl. RMA never runs on a UD
 * endpoint, so the prefix adjustment is a no-op there.
 *
 * @param base_ep		endpoint the transfer is issued on
 * @param msg_iov		local data buffers
 * @param iov_count		number of iov entries to convert
 * @param inline_data_list[out]	array of at least iov_count entries to fill
 */
static inline void efa_msg_setup_inline_data_list(struct efa_base_ep *base_ep,
						  const struct iovec *msg_iov,
						  size_t iov_count,
						  struct ibv_data_buf *inline_data_list)
{
	bool is_ud = base_ep->qp->ibv_qp->qp_type == IBV_QPT_UD;
	size_t i;

	for (i = 0; i < iov_count; i++) {
		inline_data_list[i].addr = msg_iov[i].iov_base;
		inline_data_list[i].length = msg_iov[i].iov_len;

		/* Whole prefix must be on the first sgl for dgram */
		if (!i && is_ud) {
			inline_data_list[i].addr =
				(char *) inline_data_list[i].addr +
				base_ep->info->ep_attr->msg_prefix_size;
			inline_data_list[i].length -=
				base_ep->info->ep_attr->msg_prefix_size;
		}
	}
}

/**
 * @brief Populate a scatter-gather list from an iov for a transfer
 *
 * Shared by the send path (efa_post_send), the RMA read/write paths, and the
 * work request prepare path (efa_wr_prepare). Requires a valid memory
 * descriptor per iov (efa-direct mandates FI_MR_LOCAL). For a DGRAM (UD)
 * endpoint the message prefix is stripped from the first entry; RMA never runs
 * on a UD endpoint, so the prefix adjustment is a no-op there.
 *
 * @param base_ep		endpoint the transfer is issued on
 * @param msg_iov		local data buffers
 * @param desc			array of memory descriptors
 * @param iov_count		number of iov entries to convert
 * @param sg_list[out]		array of at least iov_count entries to fill
 * @return 0 on success, -FI_EINVAL if any descriptor is missing
 */
static inline int efa_msg_setup_sge_list(struct efa_base_ep *base_ep,
					 const struct iovec *msg_iov,
					 void **desc, size_t iov_count,
					 struct ibv_sge *sg_list)
{
	bool is_ud = base_ep->qp->ibv_qp->qp_type == IBV_QPT_UD;
	size_t i;

	for (i = 0; i < iov_count; i++) {
		if (OFI_UNLIKELY(!desc || !desc[i])) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "EFA direct requires FI_MR_LOCAL but "
				 "application does not provide a valid desc\n");
			return -FI_EINVAL;
		}
		sg_list[i].lkey = ((struct efa_mr *) desc[i])->lkey;
		sg_list[i].addr = (uintptr_t) msg_iov[i].iov_base;
		sg_list[i].length = msg_iov[i].iov_len;

		/* Whole prefix must be on the first sgl for dgram */
		if (!i && is_ud) {
			sg_list[i].addr +=
				base_ep->info->ep_attr->msg_prefix_size;
			sg_list[i].length -=
				base_ep->info->ep_attr->msg_prefix_size;
		}
	}

	return 0;
}

/**
 * Prepare and return a pointer to an EFA context structure.
 *
 * @param context           Pointer to the msg context.
 * @param addr              Peer address associated with the operation.
 * @param flags             Operation flags (e.g., FI_COMPLETION).
 * @param completion_flags  Completion flags reported in the cq entry.
 * @return A pointer to an initialized EFA context structure,
 *  or NULL if context is invalid or FI_COMPLETION is not set.
 */
static inline struct efa_context *efa_fill_context(const void *context,
						   fi_addr_t addr,
						   uint64_t flags,
						   uint64_t completion_flags)
{
	if (!context || !(flags & FI_COMPLETION))
		return NULL;

	struct efa_context *efa_context = (struct efa_context *) context;
	efa_context->completion_flags = completion_flags;
	efa_context->addr = addr;

	return efa_context;
}

static inline
int efa_str_to_ep_addr(const char *node, const char *service, struct efa_ep_addr *addr)
{
	int ret;

	if (!node)
		return -FI_EINVAL;

	memset(addr, 0, sizeof(*addr));

	ret = inet_pton(AF_INET6, node, addr->raw);
	if (ret != 1)
		return -FI_EINVAL;
	if (service)
		addr->qpn = atoi(service);

	return 0;
}

#define EFA_HOST_ID_STRING_LENGTH 19
#define EFA_HOST_ID_PREFIX_LENGTH 3 /* host ID prefix is "i-0" */

static inline
uint64_t efa_get_host_id(char *host_id_file)
{
	FILE *fp = NULL;
	char host_id_str[EFA_HOST_ID_STRING_LENGTH - EFA_HOST_ID_PREFIX_LENGTH + 1];
	char *end_ptr = NULL;
	size_t length = 0;
	uint64_t host_id = 0;

	if (!host_id_file) {
		EFA_WARN(FI_LOG_EP_CTRL, "Host id file is not specified\n");
		goto out;
	}

	fp = fopen(host_id_file, "r");
	if (!fp) {
		EFA_WARN(FI_LOG_EP_CTRL, "Cannot open host id file: %s\n", host_id_file);
		goto out;
	}

	if (fseek(fp, EFA_HOST_ID_PREFIX_LENGTH, SEEK_SET) < 0) {
		EFA_WARN(FI_LOG_EP_CTRL, "Cannot locate host id in file\n");
		goto out;
	}

	length = fread(host_id_str, 1, EFA_HOST_ID_STRING_LENGTH - EFA_HOST_ID_PREFIX_LENGTH, fp);
	if (length != EFA_HOST_ID_STRING_LENGTH - EFA_HOST_ID_PREFIX_LENGTH) {
		EFA_WARN(FI_LOG_EP_CTRL, "Failed to read host id. Read length: %lu Expect length: %d\n",
			 length, EFA_HOST_ID_STRING_LENGTH - EFA_HOST_ID_PREFIX_LENGTH);
		goto out;
	}

	host_id_str[EFA_HOST_ID_STRING_LENGTH - EFA_HOST_ID_PREFIX_LENGTH] = '\0';

	host_id = (uint64_t)strtoul(host_id_str, &end_ptr, 16);
	if (*end_ptr != '\0') {
		EFA_WARN(FI_LOG_EP_CTRL, "Host id is not a valid hex string: %s\n", host_id_str);
		host_id = 0;
	}

out:
	if (fp) {
		fclose(fp);
	}
	return host_id;
}

static inline
bool efa_is_same_addr(struct efa_ep_addr *lhs, struct efa_ep_addr *rhs)
{
	return !memcmp(lhs->raw, rhs->raw, sizeof(lhs->raw)) &&
	       lhs->qpn == rhs->qpn && lhs->qkey == rhs->qkey;
}

int efa_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid,
	       void *context);

/* Performance counter declarations */
#ifdef EFA_PERF_ENABLED
#define EFA_PERF_FOREACH(DECL)	\
	DECL(perf_efa_tx),	\
	DECL(perf_efa_recv),	\
	DECL(efa_perf_size)	\

enum efa_perf_counters {
	EFA_PERF_FOREACH(OFI_ENUM_VAL)
};
#endif

static inline
bool efa_use_unsolicited_write_recv()
{
	return efa_env.use_unsolicited_write_recv && efa_device_support_unsolicited_write_recv();
}

/**
 * Convenience macro for setopt with an enforced threshold
 */
#define EFA_EP_SETOPT_THRESHOLD(opt, field, threshold) { \
	size_t _val = *(size_t *) optval; \
	if (optlen != sizeof field) \
		return -FI_EINVAL; \
	if (_val > threshold) { \
		EFA_WARN(FI_LOG_EP_CTRL, \
			"Requested size of %zu for FI_OPT_" #opt " " \
			"exceeds the maximum (%zu)\n", \
			_val, threshold); \
		return -FI_EINVAL; \
	} \
	field = _val; \
}

#endif /* EFA_H */
