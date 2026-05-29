/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_HMEM_H
#define EFA_HMEM_H

#include "ofi_hmem.h"

#if HAVE_CUDA || HAVE_NEURON || HAVE_ROCR || HAVE_SYNAPSEAI
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
	FI_HMEM_ROCR,
	FI_HMEM_NEURON,
	FI_HMEM_SYNAPSEAI
};

enum efa_dmabuf_support {
	EFA_DMABUF_NOT_SUPPORTED,
	EFA_DMABUF_SUPPORTED,
	EFA_DMABUF_ASSUMED
};

enum efa_p2p_support {
	EFA_P2P_UNSUPPORTED,
	EFA_P2P_SUPPORTED,
	EFA_P2P_UNDETERMINED,
};

struct efa_hmem_info {
	bool initialized; 	/* do we support it at all */
	enum efa_p2p_support p2p_supported_by_device;	/* do we support p2p with this device */
	bool dmabuf_fallback_enabled;
	enum efa_dmabuf_support dmabuf_supported_by_device;	/* do we support dmabuf with this device */

	size_t max_medium_msg_size;
	size_t runt_size;
	size_t min_read_msg_size;
	size_t min_read_write_size;
};

#define DMABUF_IS_SUPPORTED(info) \
	((info)->dmabuf_supported_by_device == EFA_DMABUF_SUPPORTED || \
	 (info)->dmabuf_supported_by_device == EFA_DMABUF_ASSUMED)

#define DMABUF_IS_NOT_SUPPORTED(info) \
	((info)->dmabuf_supported_by_device == EFA_DMABUF_NOT_SUPPORTED)

#define DMABUF_IS_ASSUMED(info) \
	((info)->dmabuf_supported_by_device == EFA_DMABUF_ASSUMED)

/*
 * NB: p2p_supported_by_device and protocol thresholds may be updated by
 * application code when p2p support is probed at first registration. This could
 * lead to thread-safety questions. These updates are pure writes and idempotent,
 * so ought to be safe. Alternatives are pushing this down to the domain level or
 * similar to take a lock.
 */
extern struct efa_hmem_info	g_efa_hmem_info[OFI_HMEM_MAX];

int efa_hmem_validate_p2p_opt(enum fi_hmem_iface iface, int p2p_opt, uint32_t api_version);
int efa_hmem_info_initialize();
int efa_hmem_info_apply_protocol_thresholds(enum fi_hmem_iface iface);
int efa_copy_from_hmem(void *desc, void *dest, const void *src, size_t size);
int efa_copy_to_hmem(void *desc, void *dest, const void *src, size_t size);
ssize_t efa_copy_from_hmem_iov(void **desc, char *buff, size_t buff_size, const struct iovec *hmem_iov, size_t iov_count);
ssize_t efa_copy_to_hmem_iov(void **desc, struct iovec *hmem_iov, size_t iov_count, char *buff, size_t buff_size);
#endif
