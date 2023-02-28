/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef EFA_HMEM_H
#define EFA_HMEM_H

#define EFA_HMEM_IFACE_FOREACH(var) \
	for ((var) = 0; (var) < ((sizeof efa_hmem_ifaces) / (sizeof (enum fi_hmem_iface))); ++(var))

#define EFA_HMEM_IFACE_FOREACH_NON_SYSTEM(var) \
	for ((var) = 1; (var) < ((sizeof efa_hmem_ifaces) / (sizeof (enum fi_hmem_iface))); ++(var))

/* Order matters */
static const enum fi_hmem_iface efa_hmem_ifaces[] = {
	FI_HMEM_SYSTEM,	/* Must be first here */
	FI_HMEM_CUDA,
	FI_HMEM_NEURON,
	FI_HMEM_SYNAPSEAI
};

struct efa_hmem_info {
	bool initialized; 	/* do we support it at all */
	bool p2p_disabled_by_user;	/* Did the user disable p2p via FI_OPT_FI_HMEM_P2P? */
	bool p2p_required_by_impl;	/* Is p2p required for this interface? */
	bool p2p_supported_by_device;	/* do we support p2p with this device */

	size_t max_intra_eager_size; /* Maximum message size to use eager protocol for intra-node */
	size_t max_medium_msg_size;
	size_t runt_size;
	size_t min_read_msg_size;
	size_t min_read_write_size;
};

int efa_domain_hmem_validate_p2p_opt(struct efa_domain *efa_domain, enum fi_hmem_iface iface, int p2p_opt);
int efa_domain_hmem_info_init_all(struct efa_domain *efa_domain);

ssize_t efa_copy_from_hmem_iov(void **desc, char *buff, int buff_size, const struct iovec *hmem_iov, int iov_count);
ssize_t efa_copy_to_hmem_iov(void **desc, struct iovec *hmem_iov, int iov_count, char *buff, int buff_size);
#endif
