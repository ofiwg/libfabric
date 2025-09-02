/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_HMEM_H
#define EFA_HMEM_H

#include "ofi_hmem.h"
#include "efa_mr.h"
#include "efa_tp.h"

#if HAVE_CUDA || HAVE_NEURON || HAVE_SYNAPSEAI
#  define EFA_HAVE_NON_SYSTEM_HMEM 1
#else
#  define EFA_HAVE_NON_SYSTEM_HMEM 0
#endif

#define EFA_HMEM_IFACE_FOREACH_FROM(var, start) \
	for ( \
		const enum fi_hmem_iface *_p = &efa_hmem_ifaces[start]; \
		_p < &efa_hmem_ifaces[sizeof efa_hmem_ifaces / sizeof (enum fi_hmem_iface)] && ((var) = *_p, 1); \
		_p++ \
	)

#define EFA_HMEM_IFACE_FOREACH(var) EFA_HMEM_IFACE_FOREACH_FROM(var, 0)
#define EFA_HMEM_IFACE_FOREACH_NON_SYSTEM(var) EFA_HMEM_IFACE_FOREACH_FROM(var, 1)

/* Order matters */
static const enum fi_hmem_iface efa_hmem_ifaces[] = {
	FI_HMEM_SYSTEM,	/* Must be first here */
	FI_HMEM_CUDA,
	FI_HMEM_NEURON,
	FI_HMEM_SYNAPSEAI
};

struct efa_hmem_info {
	bool initialized; 	/* do we support it at all */
	bool p2p_supported_by_device;	/* do we support p2p with this device */

	size_t max_medium_msg_size;
	size_t runt_size;
	size_t min_read_msg_size;
	size_t min_read_write_size;
};

extern struct efa_hmem_info	g_efa_hmem_info[OFI_HMEM_MAX];

int efa_hmem_validate_p2p_opt(enum fi_hmem_iface iface, int p2p_opt, uint32_t api_version);
int efa_hmem_info_initialize();

/**
 * @brief Copy data from a hmem device to a system buffer
 *
 * @param[in]    desc            Pointer to a memory registration descriptor
 * @param[out]   dest            Destination system memory buffer
 * @param[in]    src             Source hmem device memory
 * @param[in]    size            Data size in bytes to copy
 * @return       FI_SUCCESS status code on success, or an error code.
 */
static inline int efa_copy_from_hmem(void *desc, void *dest, const void *src, size_t size)
{
	struct efa_mr_peer peer = { .iface = FI_HMEM_SYSTEM };

	if (desc)
		peer = ((struct efa_mr *) desc)->peer;

	if (peer.flags & OFI_HMEM_DATA_DEV_REG_HANDLE) {
		assert(peer.hmem_data);
		switch (peer.iface) {
		/* TODO: Fine tune the max data size to switch from gdrcopy to cudaMemcpy */
		case FI_HMEM_CUDA:
			efa_tracepoint(dev_reg_copy_from_hmem, &peer, dest, src, size);
			return ofi_hmem_dev_reg_copy_from_hmem(peer.iface, (uint64_t) peer.hmem_data, dest, src, size);
		default:
			break;
		}
	}

	efa_tracepoint(copy_from_hmem, &peer, dest, src, size);
	return ofi_copy_from_hmem(peer.iface, peer.device, dest, src, size);
};

/**
 * @brief Copy data from a system buffer to a hmem device
 *
 * @param[in]    desc            Pointer to a memory registration descriptor
 * @param[out]   dest            Destination hmem device memory
 * @param[in]    src			 Source system memory buffer
 * @param[in]    size            Data size in bytes to copy
 * @return       FI_SUCCESS status code on success, or an error code.
 */
static inline int efa_copy_to_hmem(void *desc, void *dest, const void *src, size_t size)
{
	struct efa_mr_peer peer = { .iface = FI_HMEM_SYSTEM };

	if (desc)
		peer = ((struct efa_mr *) desc)->peer;

	if (peer.flags & OFI_HMEM_DATA_DEV_REG_HANDLE) {
		assert(peer.hmem_data);
		switch (peer.iface) {
		/* TODO: Fine tune the max data size to switch from gdrcopy to cudaMemcpy */
		case FI_HMEM_CUDA:
			efa_tracepoint(dev_reg_copy_to_hmem, &peer, dest, src, size);
			return ofi_hmem_dev_reg_copy_to_hmem(peer.iface, (uint64_t) peer.hmem_data, dest, src, size);
		default:
			break;
		}
	}

	efa_tracepoint(copy_to_hmem, &peer, dest, src, size);
	return ofi_copy_to_hmem(peer.iface, peer.device, dest, src, size);
};

ssize_t efa_copy_from_hmem_iov(void **desc, char *buff, size_t buff_size, const struct iovec *hmem_iov, size_t iov_count);
ssize_t efa_copy_to_hmem_iov(void **desc, struct iovec *hmem_iov, size_t iov_count, char *buff, size_t buff_size);
#endif
