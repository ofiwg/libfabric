#!/usr/bin/python3

# Copyright (c) 2018 Intel Corp.  All rights reserved.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import collections
import errno

configure = """\
dnl Configury specific to the libfabric {prov} provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_{PROV}_CONFIGURE],[
       # Determine if we can support the {prov} provider
       {prov}_h_happy=0
       AS_IF([test x"$enable_{prov}" != x"no"], [{prov}_h_happy=1])
       AS_IF([test ${prov}_h_happy -eq 1], [$1], [$2])
])
"""

makefile = """\
if HAVE_{PROV}
_{prov}_files = \\
{files}

if HAVE_{PROV}_DL
pkglib_LTLIBRARIES += lib{prov}-fi.la
lib{prov}_fi_la_SOURCES = $(_{prov}_files) $(common_srcs)
lib{prov}_fi_la_LIBADD = $(linkback) $({prov}_shm_LIBS)
lib{prov}_fi_la_LDFLAGS = -module -avoid-version -shared -export-dynamic
lib{prov}_fi_la_DEPENDENCIES = $(linkback)
else !HAVE_{PROV}_DL
src_libfabric_la_SOURCES += $(_{prov}_files)
src_libfabric_la_LIBADD += $({prov}_shm_LIBS)
endif !HAVE_{PROV}_DL

endif HAVE_{PROV}
"""

header="""\
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include "rdma/providers/fi_log.h"

#include <ofi.h>
#include <ofi_util.h>
#include <ofi_proto.h>
#include <ofi_prov.h>
#include <ofi_enosys.h>

#define {PROV}_MAJOR_VERSION 1
#define {PROV}_MINOR_VERSION 0

// TODO edit me!
#define {PROV}_CAPS FI_MSG
#define {PROV}_IOV_LIMIT 4

extern struct fi_info {prov}_info;
extern struct fi_provider {prov}_prov;
extern struct util_prov {prov}_util_prov;
extern struct fi_fabric_attr {prov}_fabric_attr;

struct {prov}_fabric {{
	struct util_fabric util_fabric;
}};

struct {prov}_domain {{
	struct util_domain util_domain;
}};

struct {prov}_ep {{
	struct util_ep util_ep;
}};

int {prov}_fabric_open(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		       void *context);
int {prov}_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		       struct fid_domain **domain, void *context);
int {prov}_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq_fid, void *context);
int {prov}_ep_open(struct fid_domain *domain, struct fi_info *info,
		   struct fid_ep **ep_fid, void *context);
"""

init = """\
#include "{prov}.h"

static int {prov}_getinfo(uint32_t version, const char *node, const char *service,
			  uint64_t flags, const struct fi_info *hints,
			  struct fi_info **info)
{{

	return 0;
}}

static void {prov}_fini(void)
{{

}}

struct fi_provider {prov}_prov = {{
	.name = "{prov}",
	.version = FI_VERSION({PROV}_MAJOR_VERSION, {PROV}_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = {prov}_getinfo,
	.fabric = {prov}_fabric_open,
	.cleanup = {prov}_fini
}};

struct util_prov {prov}_util_prov = {{
	.prov = &{prov}_prov,
	.info = &{prov}_info,
	.flags = 0,
}};

{PROV}_INI
{{
	return &{prov}_prov;
}}

"""

attr = """\
#include "{prov}.h"

// TODO edit me!
struct fi_tx_attr {prov}_tx_attr = {{
	.caps = {PROV}_CAPS,
	.msg_order = ~0x0ULL,
	.comp_order = ~0x0ULL,
	.size = SIZE_MAX,
	.iov_limit = {PROV}_IOV_LIMIT,
}};

struct fi_rx_attr {prov}_rx_attr = {{
	.caps = {PROV}_CAPS,
	.msg_order = ~0x0ULL,
	.comp_order = ~0x0ULL,
	.size = 1024,
	.iov_limit= {PROV}_IOV_LIMIT,
}};

struct fi_ep_attr {prov}_ep_attr = {{
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_{PROV},
	.protocol_version = 1,
	.max_msg_size = SIZE_MAX,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
	.max_order_raw_size = SIZE_MAX,
	.max_order_war_size = SIZE_MAX,
	.max_order_waw_size = SIZE_MAX,
}};

struct fi_domain_attr {prov}_domain_attr = {{
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_AUTO,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_BASIC | FI_MR_SCALABLE,
	.cq_data_size = sizeof_field(struct ofi_op_hdr, data),
	.cq_cnt = (1 << 16),
	.ep_cnt = (1 << 15),
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
	.max_ep_tx_ctx = 1,
	.max_ep_rx_ctx = 1,
	.mr_iov_limit = 1,
}};

struct fi_fabric_attr {prov}_fabric_attr = {{
	.prov_version = FI_VERSION({PROV}_MAJOR_VERSION, {PROV}_MINOR_VERSION),
}};

struct fi_info {prov}_info = {{
	.caps = {PROV}_CAPS,
	.addr_format = FI_SOCKADDR,
	.tx_attr = &{prov}_tx_attr,
	.rx_attr = &{prov}_rx_attr,
	.ep_attr = &{prov}_ep_attr,
	.domain_attr = &{prov}_domain_attr,
	.fabric_attr = &{prov}_fabric_attr
}};
"""

fabric = """\
#include "{prov}.h"

static int {prov}_fabric_close(fid_t fid)
{{
	struct {prov}_fabric *{prov}_fabric =
		container_of(fid, struct {prov}_fabric, util_fabric.fabric_fid.fid);
	int ret;

	ret = ofi_fabric_close(&{prov}_fabric->util_fabric);
	if (ret)
		return ret;

	free({prov}_fabric);
	//return 0;
	return -FI_ENOSYS;
}}

static struct fi_ops {prov}_fabric_fi_ops = {{
	.size = sizeof(struct fi_ops),
	.close = {prov}_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
}};

static struct fi_ops_fabric {prov}_fabric_ops = {{
	.size = sizeof(struct fi_ops_fabric),
	.domain = {prov}_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = ofi_trywait
}};

int {prov}_fabric_open(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		       void *context)
{{
	struct {prov}_fabric *{prov}_fabric;
	int ret;

	{prov}_fabric = calloc(1, sizeof(*{prov}_fabric));
	if (!{prov}_fabric)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&{prov}_prov, &{prov}_fabric_attr, attr,
			      &{prov}_fabric->util_fabric, context);
	if (ret)
		goto err1;

	*fabric = &{prov}_fabric->util_fabric.fabric_fid;
	(*fabric)->fid.ops = &{prov}_fabric_fi_ops;
	(*fabric)->ops = &{prov}_fabric_ops;

	//return 0;
	return -FI_ENOSYS;
err1:
	free({prov}_fabric);
	return ret;
}}
"""

domain = """\
#include "{prov}.h"

static int {prov}_domain_close(fid_t fid)
{{
	struct {prov}_domain *{prov}_domain =
		container_of(fid, struct {prov}_domain, util_domain.domain_fid.fid);
	int ret;

	ret = ofi_domain_close(&{prov}_domain->util_domain);
	if (ret)
		return ret;

	free({prov}_domain);
	//return 0;
	return -FI_ENOSYS;
}}

static int {prov}_mr_close(fid_t fid)
{{

	return -FI_ENOSYS;
}}

static int {prov}_mr_reg(struct fid *domain_fid, const void *buf, size_t len,
			 uint64_t access, uint64_t offset, uint64_t requested_key,
			 uint64_t flags, struct fid_mr **mr, void *context)
{{

	return -FI_ENOSYS;
}}

static struct fi_ops {prov}_mr_ops = {{
	.size = sizeof(struct fi_ops),
	.close = {prov}_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
}};

static struct fi_ops_mr {prov}_domain_mr_ops = {{
	.size = sizeof(struct fi_ops_mr),
	.reg = {prov}_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
}};

static struct fi_ops {prov}_domain_fi_ops = {{
	.size = sizeof(struct fi_ops),
	.close = {prov}_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
}};

static struct fi_ops_domain {prov}_domain_ops = {{
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_no_av_open,
	.cq_open = {prov}_cq_open,
	.endpoint = {prov}_ep_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
}};

int {prov}_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		       struct fid_domain **domain, void *context)
{{
	struct {prov}_domain *{prov}_domain;
	int ret;

	{prov}_domain = calloc(1, sizeof(*{prov}_domain));
	if (!{prov}_domain)
		return -FI_ENOMEM;

	ret = ofi_domain_init(fabric, info, &{prov}_domain->util_domain, context);
	if (ret)
		goto err1;

	*domain = &{prov}_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &{prov}_domain_fi_ops;
	(*domain)->mr = &{prov}_domain_mr_ops;
	(*domain)->ops = &{prov}_domain_ops;

	//return 0;
	return -FI_ENOSYS;
err1:
	free({prov}_domain);
	return ret;
}}
"""

mr = """\
#include "{prov}.h"

static struct fi_ops_mr {prov}_mr_ops = {{
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_no_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
}};
"""

cq = """\
#include "{prov}.h"

static int {prov}_cq_close(fid_t fid)
{{
	struct util_cq *util_cq = container_of(fid, struct util_cq, cq_fid.fid);
	int ret;

	ret = ofi_cq_cleanup(util_cq);
	if (ret)
		return ret;

	free(util_cq);
	//return 0;
	return -FI_ENOSYS;
}}

static struct fi_ops {prov}_cq_fi_ops = {{
	.size = sizeof(struct fi_ops),
	.close = {prov}_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
}};

static struct fi_ops_cq {prov}_cq_ops = {{
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = ofi_cq_sread,
	.sreadfrom = ofi_cq_sreadfrom,
	.signal = ofi_cq_signal,
	.strerror = fi_no_cq_strerror,
}};

int {prov}_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq_fid, void *context)
{{
	struct util_cq *util_cq;
	int ret;

	util_cq = calloc(1, sizeof(*util_cq));
	if (!util_cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&{prov}_prov, domain, attr, util_cq, &ofi_cq_progress,
			  context);
	if (ret)
		goto err1;

	*cq_fid = &util_cq->cq_fid;
	(*cq_fid)->fid.ops = &{prov}_cq_fi_ops;
	(*cq_fid)->ops = &{prov}_cq_ops;

	//return 0;
	return -FI_ENOSYS;
err1:
	free(util_cq);
	return ret;
}}
"""

av = """\
#include "{prov}.h"

static int {prov}_av_close(struct fid *fid)
{{
	struct {prov}_av *{prov}_av;
	int ret;

	av = container_of(fid, struct {prov}_av, util_av.av_fid);
	ret = ofi_av_close(&{prov}_av->util_av);
	if (ret)
		return ret;

	free({prov}_av);
	return 0;
}}

static int {prov}_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{{
	return ofi_av_bind(fid, bfid, flags);
}}

static const char *{prov}_av_straddr(struct fid_av *av, const void *addr,
				  char *buf, size_t *len)
{{
	return -FI_ENOSYS;
}}

static int {prov}_av_insertsvc(struct fid_av *av, const char *node,
			   const char *service, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{{
	return -FI_ENOSYS;
}}

static int {prov}_av_insertsym(struct fid_av *av_fid, const char *node, size_t nodecnt,
			   const char *service, size_t svccnt, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{{
	return -FI_ENOSYS;
}}

static int {prov}_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			 size_t *addrlen)
{{
	return -FI_ENOSYS;

}}

static int {prov}_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr, size_t count,
			uint64_t flags)
{{
	return -FI_ENOSYS;
}}

static int {prov}_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{{
	return -FI_ENOSYS;
}}

static struct fi_ops_av {prov}_av_ops = {{
	.size = sizeof(struct fi_ops_av),
	.insert = {prov}_av_insert,
	.insertsvc = {prov}_av_insertsvc,
	.insertsym = {prov}_av_insertsym,
	.remove = {prov}_av_remove,
	.lookup = {prov}_av_lookup,
	.straddr = {prov}_av_straddr,
}};

static struct fi_ops {prov}_av_fi_ops = {{
	.size = sizeof(struct fi_ops),
	.close = {prov}_av_close,
	.bind = {prov}_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
}};

int {prov}_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		   struct fid_av **av_fid, void *context)
{{
	struct {prov}_av *{prov}_av;
	struct {prov}_domain *domain;
	struct util_av_attr util_attr;
	int ret;

	domain = container_of(domain_fid, struct {prov}_domain, util_domain.domain_fid);
	{prov}_av = calloc(1, sizeof(*{prov}_av));
	if (!{prov}_av)
		return -FI_ENOMEM;

	util_attr.addrlen = {PROV}_ADDRLEN;
	util_attr.overhead = attr->count;
	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_TABLE;

	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			 &{prov}_av->util_av, context);
	if (ret)
		goto err1;

	{prov}_av->util_av.av_fid.fid.ops = &{prov}_av_fi_ops;
	{prov}_av->util_av.av_fid.ops = &{prov}_av_ops;
	*av_fid = &{prov}_av->util_av.av_fid;
	//return 0;
	return -FI_ENOSYS;
err2:
	ofi_av_close(&{prov}_av->util_av);
err1:
	free({prov}_av);
	return ret;
}}
"""

ep = """\
#include "{prov}.h"

static int {prov}_ep_close(fid_t fid)
{{
	struct {prov}_ep *{prov}_ep =
		container_of(fid, struct {prov}_ep, util_ep.ep_fid.fid);

	ofi_endpoint_close(&{prov}_ep->util_ep);
	free({prov}_ep);
	//return 0;
	return -FI_ENOSYS;
}}

static int {prov}_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{{
	struct {prov}_ep *{prov}_ep =
		container_of(ep_fid, struct {prov}_ep, util_ep.ep_fid.fid);
	struct util_cq *cq;
	struct util_av *av;
	struct util_cntr *cntr;
	int ret = 0;

	switch (bfid->fclass) {{
	case FI_CLASS_AV:
		av = container_of(bfid, struct util_av, av_fid.fid);
		ret = ofi_ep_bind_av(&{prov}_ep->util_ep, av);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);

		ret = ofi_ep_bind_cq(&{prov}_ep->util_ep, cq, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&{prov}_ep->util_ep, cntr, flags);
		if (ret)
			return ret;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&{prov}_prov, FI_LOG_EP_CTRL, "invalid fid class\\n");
		ret = -FI_EINVAL;
		break;
	}}
	//return ret;
	return -FI_ENOSYS;
}}

static int {prov}_ep_ctrl(struct fid *fid, int command, void *arg)
{{
	struct {prov}_ep *{prov}_ep;

	{prov}_ep = container_of(fid, struct {prov}_ep, util_ep.ep_fid.fid);

	switch (command) {{
	case FI_ENABLE:
		if (!{prov}_ep->util_ep.rx_cq || !{prov}_ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!{prov}_ep->util_ep.av)
			return -FI_EOPBADSTATE;
		break;
	default:
		return -FI_ENOSYS;
	}}
	//return 0;
	return -FI_ENOSYS;
}}

static struct fi_ops {prov}_ep_fi_ops = {{
	.size = sizeof(struct fi_ops),
	.close = {prov}_ep_close,
	.bind = {prov}_ep_bind,
	.control = {prov}_ep_ctrl,
	.ops_open = fi_no_ops_open,
}};

static struct fi_ops_ep {prov}_ops_ep = {{
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
}};

static struct fi_ops_cm {prov}_ops_cm = {{
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = fi_no_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
}};

static struct fi_ops_msg {prov}_ops_msg = {{
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
}};

struct fi_ops_tagged {prov}_ops_tagged = {{
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_no_tagged_recv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
}};

struct fi_ops_rma {prov}_ops_rma = {{
	.size = sizeof (struct fi_ops_rma),
	.read = fi_no_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = fi_no_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
}};

void {prov}_ep_progress(struct util_ep *util_ep)
{{
	//struct {prov}_ep *{prov}_ep =
	//	container_of(util_ep, struct {prov}_ep, util_ep);
}}

int {prov}_ep_open(struct fid_domain *domain, struct fi_info *info,
		   struct fid_ep **ep_fid, void *context)
{{
	struct {prov}_ep *{prov}_ep;
	int ret;

	{prov}_ep = calloc(1, sizeof(*{prov}_ep));
	if (!{prov}_ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &{prov}_util_prov, info, &{prov}_ep->util_ep,
				context, &{prov}_ep_progress);
	if (ret)
		goto err1;

	*ep_fid = &{prov}_ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &{prov}_ep_fi_ops;
	(*ep_fid)->ops = &{prov}_ops_ep;
	(*ep_fid)->cm = &{prov}_ops_cm;
	(*ep_fid)->msg = &{prov}_ops_msg;
	(*ep_fid)->tagged = &{prov}_ops_tagged;
	(*ep_fid)->rma = &{prov}_ops_rma;

	//return 0;
	return -FI_ENOSYS;
err1:
	free({prov}_ep);
	return ret;
}}
"""

cntr = """\
#include "{prov}.h"

static struct fi_ops_cntr {prov}_cntr_ops = {{
	.size = sizeof(struct fi_ops_cntr),
	//.read = {prov}_cntr_read,
	//.readerr = {prov}_cntr_readerr,
	.add = fi_no_cntr_add,
	.set = fi_no_cntr_set,
	.wait = fi_no_cntr_wait,
}};
"""

atomic = """\
#include "{prov}.h"

static struct fi_ops_atomic {prov}_atomic_ops = {{
	.size = sizeof(struct fi_ops_atomic),
	.write = fi_no_atomic_write,
	.writev = fi_no_atomic_writev,
	.writemsg = fi_no_atomic_writemsg,
	.inject = fi_no_atomic_inject,
	.readwrite = fi_no_atomic_readwrite,
	.readwritev = fi_no_atomic_readwritev,
	.readwritemsg = fi_no_atomic_readwritemsg,
	.compwrite = fi_no_atomic_compwrite,
	.compwritev = fi_no_atomic_compwritev,
	.compwritemsg = fi_no_atomic_compwritemsg,
	.writevalid = fi_no_atomic_writevalid,
	.readwritevalid = fi_no_atomic_readwritevalid,
	.compwritevalid = fi_no_atomic_compwritevalid,
}};
"""

ofi_prov = """\
/*
// TODO move me to include/ofi_prov.h
#if (HAVE_{PROV}) && (HAVE_{PROV}_DL)
#  define {PROV}_INI FI_EXT_INI
#  define {PROV}_INIT NULL
#elif (HAVE_{PROV})
#  define {PROV}_INI INI_SIG(fi_{prov}_ini)
#  define {PROV}_INIT fi_{prov}_ini()
{PROV}_INI ;
#else
#  define {PROV}_INIT NULL
#endif

// TODO move me to src/fabric.c:fi_ini
ofi_register_provider({PROV}_INIT, NULL);

// TODO move me to configure.ac
FI_PROVIDER_SETUP([{prov}])

// TODO move me to Makefile.am
include prov/{prov}/Makefile.include

// TODO move me to include/rdma/fabric.h:enum (Endpoint protocol)
FI_PROTO_{PROV},
*/
"""

NameVal = collections.namedtuple('NameVal', 'name val')

build_dict = {
	'config'	: NameVal("configure.m4", configure),
	'make'		: NameVal("Makefile.include", makefile),
}

src_dict = {
	'init'		: init,
	'attr'		: attr,
	'fabric'	: fabric,
	'domain'	: domain,
	'cq'		: cq,
	'av'		: av,
	'ep'		: ep,
	'cntr'		: cntr,
	'atomic'	: atomic,
}

def ofi_write_to_file(filepath, content):
	with open(filepath, 'a') as fd:
		fd.write(content)

def ofi_write_to_file_safe(filepath, content):
	try:
		with open(filepath, 'x') as fd:
			print("Writing to %s..." % filepath)
	except OSError as e:
		if e.errno == errno.EEXIST:
			print("File %s already exists! Skipping..." % filepath)
			return
		else:
			raise

	ofi_write_to_file(filepath, content)

def ofi_boilerplate(prov):
	provdir = "prov/%s" % prov
	provsrcdir = "%s/src" % provdir
	os.makedirs(provsrcdir, exist_ok=True)

	print("Writing provider %s to dir: %s" % (prov, provsrcdir))

	PROV=prov.upper()

	ofi_write_to_file_safe("%s/%s.h" % (provsrcdir, prov),
			       header.format(prov=prov, PROV=PROV))

	src_files = []

	for src in src_dict:
		name = "%s/%s_%s.c" % (provsrcdir, prov, src)
		src_files.append(name)
		ofi_write_to_file_safe(name,
				       src_dict[src].format(prov=prov, PROV=PROV))

	# Write code that needs to be copied to other files
	ofi_write_to_file("%s/%s_%s.c" % (provsrcdir, prov, 'init'),
			  ofi_prov.format(prov=prov, PROV=PROV))

	ofi_write_to_file_safe("%s/%s" % (provdir, build_dict['config'].name),
			       build_dict['config'].val.format(prov=prov, PROV=PROV))

	make_src = ["\t%s\t\\" % src for src in src_files[:-1]]
	make_src.append("\t%s" % src_files[-1])

	ofi_write_to_file_safe("%s/%s" % (provdir, build_dict['make'].name),
			       build_dict['make'].val.format(prov=prov, PROV=PROV,
							     files='\n'.join(make_src)))

	print("\nDone. Please review all files!")

if __name__ == "__main__":
	usage = """
	Usage:  {arg0} <provider name>

	- This script generates boilerplate code for a provider that
	  implements RDM/DGRAM EP types.
	- This script must be run from the root directory of libfabric
	  repo.
	- It assumes the new provider would use the libfabric utility
	  code.
	- Existing files won't be overwritten.
	"""
	if len(sys.argv) != 2:
		print(usage.format(arg0=sys.argv[0]))
		sys.exit(1)
	ofi_boilerplate(sys.argv[1])
