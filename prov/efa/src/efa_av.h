/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef EFA_AV_H
#define EFA_AV_H

#define EFA_SHM_MAX_AV_COUNT       (256)

struct efa_av {
	struct fid_av *shm_rdm_av;
	fi_addr_t shm_rdm_addr_map[EFA_SHM_MAX_AV_COUNT];
	struct efa_domain *domain;
	struct efa_base_ep *base_ep;
	size_t used;
	size_t shm_used;
	enum fi_av_type type;
	/* cur_reverse_av is a map from (ahn + qpn) to current (latest) efa_conn.
	 * prv_reverse_av is a map from (ahn + qpn + connid) to all previous efa_conns.
	 * cur_reverse_av is faster to search because its key size is smaller
	 */
	struct efa_cur_reverse_av *cur_reverse_av;
	struct efa_prv_reverse_av *prv_reverse_av;
	struct efa_ah *ah_map;
	struct util_av util_av;
	enum fi_ep_type ep_type;
};

#endif
