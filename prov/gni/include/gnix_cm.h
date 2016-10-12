/*
 * Copyright (c) 2016 Cray Inc. All rights reserved.
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

#ifndef _GNIX_CM_H_
#define _GNIX_CM_H_

#include "gnix.h"

struct gnix_pep_sock_connreq {
	int type;
	int msg_id;
	struct fi_info info;
	struct gnix_ep_name src_addr;
	struct gnix_ep_name dest_addr;
	struct fi_tx_attr tx_attr;
	struct fi_rx_attr rx_attr;
	struct fi_ep_attr ep_attr;
	struct fi_domain_attr domain_attr;
	struct fi_fabric_attr fabric_attr;
	int vc_id;
	gni_smsg_attr_t vc_mbox_attr;
	gni_mem_handle_t cq_irq_mdh;
	uint64_t peer_caps;
};

enum gnix_pep_sock_resp_cmd {
	GNIX_PEP_SOCK_RESP_ACCEPT,
	GNIX_PEP_SOCK_RESP_REJECT
};

struct gnix_pep_sock_connresp {
	enum gnix_pep_sock_resp_cmd cmd;
	int vc_id;
	gni_smsg_attr_t vc_mbox_attr;
	gni_mem_handle_t cq_irq_mdh;
	uint64_t peer_caps;
};

struct gnix_pep_sock_conn {
	struct fid fid;
	struct dlist_entry list;
	int sock_fd;
	struct gnix_pep_sock_connreq req;
	int bytes_read;
	struct fi_info *info;
};

int _gnix_pep_progress(struct gnix_fid_pep *pep);
int _gnix_ep_progress(struct gnix_fid_ep *ep);

#endif

