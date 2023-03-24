/*
 * Copyright (c) 2013-2021 Intel Corporation. All rights reserved
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <sys/un.h>

#include "ofi_hmem.h"
#include "ofi_iov.h"
#include "ofi_mem.h"
#include "ofi_mr.h"
#include "sm2.h"
#include "sm2_fifo.h"
#include "sm2_signal.h"

extern struct fi_ops_msg sm2_msg_ops, sm2_no_recv_msg_ops;
extern struct fi_ops_tagged sm2_tag_ops, sm2_no_recv_tag_ops;
DEFINE_LIST(sm2_sock_name_list);
pthread_mutex_t sm2_sock_list_lock = PTHREAD_MUTEX_INITIALIZER;
int sm2_global_ep_idx = 0;

int
sm2_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct sm2_ep *ep;
	char *name;

	if (addrlen > SM2_NAME_MAX) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Addrlen exceeds max addrlen (%d)\n", SM2_NAME_MAX);
		return -FI_EINVAL;
	}

	ep = container_of(fid, struct sm2_ep, util_ep.ep_fid.fid);

	name = strdup(addr);
	if (!name)
		return -FI_ENOMEM;

	if (ep->name)
		free((void *) ep->name);
	ep->name = name;
	return 0;
}

int
sm2_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct sm2_ep *ep;
	int ret = 0;

	ep = container_of(fid, struct sm2_ep, util_ep.ep_fid.fid);
	if (!ep->name)
		return -FI_EADDRNOTAVAIL;

	if (!addr || *addrlen == 0 ||
	    snprintf(addr, *addrlen, "%s", ep->name) >= *addrlen)
		ret = -FI_ETOOSMALL;

	*addrlen = strlen(ep->name) + 1;

	if (!ret)
		((char *) addr)[*addrlen - 1] = '\0';

	*addrlen = SM2_NAME_MAX;

	return ret;
}

int
sm2_ep_getopt(fid_t fid, int level, int optname, void *optval, size_t *optlen)
{
	struct sm2_ep *sm2_ep =
		container_of(fid, struct sm2_ep, util_ep.ep_fid);

	return sm2_ep->srx->ops->getopt(&sm2_ep->srx->fid, level, optname,
					optval, optlen);
}

int
sm2_ep_setopt(fid_t fid, int level, int optname, const void *optval,
	      size_t optlen)
{
	struct sm2_ep *sm2_ep =
		container_of(fid, struct sm2_ep, util_ep.ep_fid);
	struct util_srx_ctx *srx =
		util_get_peer_srx(sm2_ep->srx)->ep_fid.fid.context;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	if (optname == FI_OPT_MIN_MULTI_RECV) {
		srx->min_multi_recv_size = *(size_t *) optval;
		return FI_SUCCESS;
	}

	if (optname == FI_OPT_CUDA_API_PERMITTED) {
		if (!hmem_ops[FI_HMEM_CUDA].initialized) {
			FI_WARN(&sm2_prov, FI_LOG_CORE,
				"Cannot set option FI_OPT_CUDA_API_PERMITTED "
				"when cuda "
				"library "
				"or cuda device not available\n");
			return -FI_EINVAL;
		}

		/* our CUDA support relies on the ability to call CUDA API */
		return *(bool *) optval ? FI_SUCCESS : -FI_EOPNOTSUPP;
	}

	return -FI_ENOPROTOOPT;
}

static ssize_t
sm2_ep_cancel(fid_t ep_fid, void *context)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid);

	return ep->srx->ops->cancel(&ep->srx->fid, context);
}

/* Verify peer.
What can we check here?  Ref-count is >0?
*/
int64_t
sm2_verify_peer(struct sm2_ep *ep, fi_addr_t fi_addr)
{
	int32_t id;
	struct sm2_av *sm2_av;
	struct sm2_ep_allocation_entry *entries;

	id = (int32_t) fi_addr;
	if (OFI_UNLIKELY(id > SM2_MAX_PEERS) || OFI_UNLIKELY(id < 0)) {
		fprintf(stderr, "Send to invalid fi_addr.  Aborting\n");
		abort();
	};

	sm2_av = container_of(ep->util_ep.av, struct sm2_av, util_av);
	if (sm2_av->sm2_aux[id].cqfid == FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;

	entries = sm2_mmap_entries(&sm2_av->sm2_mmap);

	// TODO... should this be atomic?
	if (OFI_UNLIKELY(entries[id].startup_ready == 0))
		return -FI_EAGAIN;

	return id;
}

void
sm2_generic_format(struct sm2_free_queue_entry *fqe, int64_t self_id,
		   uint32_t op, uint64_t tag, uint64_t data, uint64_t op_flags)
{
	fqe->protocol_hdr.op = op;
	fqe->protocol_hdr.op_flags = 0;
	fqe->protocol_hdr.tag = tag;
	fqe->protocol_hdr.id = self_id;
	fqe->protocol_hdr.data = data;

	if (op_flags & FI_REMOTE_CQ_DATA)
		fqe->protocol_hdr.op_flags |= SM2_REMOTE_CQ_DATA;
	if (op_flags & FI_COMPLETION)
		fqe->protocol_hdr.op_flags |= SM2_TX_COMPLETION;
}

static void
sm2_format_inject(struct sm2_free_queue_entry *fqe, enum fi_hmem_iface iface,
		  uint64_t device, const struct iovec *iov, size_t count,
		  struct sm2_region *smr)
{
	fqe->protocol_hdr.op_src = sm2_src_inject;
	fqe->protocol_hdr.size = ofi_copy_from_hmem_iov(
		fqe->data, SM2_INJECT_SIZE, iface, device, iov, count, 0);
}

int
sm2_select_proto()
{
	return sm2_src_inject;
}

static ssize_t
sm2_do_inject(struct sm2_ep *ep, struct sm2_region *peer_smr, int64_t peer_id,
	      uint32_t op, uint64_t tag, uint64_t data, uint64_t op_flags,
	      enum fi_hmem_iface iface, uint64_t device,
	      const struct iovec *iov, size_t iov_count, size_t total_len)
{
	struct sm2_free_queue_entry *fqe;
	struct sm2_region *self_region;

	assert(total_len <= 4096);

	self_region = sm2_smr_region(ep, ep->self_fiaddr);

	if (smr_freestack_isempty(sm2_free_stack(self_region))) {
		sm2_progress_recv(ep);
		if (smr_freestack_isempty(sm2_free_stack(self_region))) {
			return -FI_EAGAIN;
		}
	}

	/* Pop FQE from local region for sending */
	fqe = smr_freestack_pop(sm2_free_stack(self_region));

	sm2_generic_format(fqe, ep->self_fiaddr, op, tag, data, op_flags);
	sm2_format_inject(fqe, iface, device, iov, iov_count, peer_smr);

	sm2_fifo_write(ep, peer_id, fqe);
	return FI_SUCCESS;
}

static void
sm2_retreive_fqe_and_close_fifo(struct sm2_ep *ep, int timeout_s)
{
	struct sm2_free_queue_entry *fqe;
	struct sm2_mmap *map = map = ep->mmap_regions;
	struct sm2_region *self_region = sm2_smr_region(ep, ep->self_fiaddr);
	int count = 0;

	/* Return all free queue entries in queue without processing them */
process_incoming_msgs:
	while (NULL != (fqe = sm2_fifo_read(ep))) {
		if (fqe->protocol_hdr.op_src == sm2_buffer_return) {
			smr_freestack_push(sm2_free_stack(sm2_smr_region(
						   ep, ep->self_fiaddr)),
					   fqe);
		} else {
			// TODO Tell other side that we haven't processed their
			// message, just returned FQE
			sm2_fifo_write_back(ep, fqe);
		}
	}

	if (smr_freestack_isfull(sm2_free_stack(self_region))) {
		// TODO Set head/tail of FIFO queue to show peers we aren't
		// accepting new entires
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"All FQE's acquired for map[%d], clean shutdown\n",
			ep->self_fiaddr);
	} else {
		if (count < timeout_s) {
			count++;
			sleep(1);
			goto process_incoming_msgs;
		}

		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Shutting down map[%d] without all FQE's, messy "
			"shutdown, "
			" map[%d] will not be used again (zombie'd) until SHM "
			"file is "
			"cleared\n",
			ep->self_fiaddr, ep->self_fiaddr);
	}
}

static int
sm2_ep_close(struct fid *fid)
{
	struct sm2_ep *ep;
	struct sm2_mmap *map;
	struct sm2_region *self_region;

	ep = container_of(fid, struct sm2_ep, util_ep.ep_fid.fid);
	map = ep->mmap_regions;
	self_region = sm2_smr_region(ep, ep->self_fiaddr);

	/* Make an effort to cleanly shut down... 1 seconds was chosen b/c it
	 * seemed reasonable */
	sm2_retreive_fqe_and_close_fifo(ep, 1);

	ofi_endpoint_close(&ep->util_ep);

	/* Set our PID to 0 in the regions map if our free queue entry stack is
	 * full This will allow other entries re-use us or shrink file.
	 */
	// TODO Do we want to mark our entry as zombie now if we don't have all
	// our FQE?
	if (smr_freestack_isfull(sm2_free_stack(self_region))) {
		sm2_coordinator_lock(map);
		sm2_coordinator_free_entry(map, ep->self_fiaddr);
		sm2_coordinator_unlock(map);
	}

	if (ep->util_ep.ep_fid.msg != &sm2_no_recv_msg_ops)
		(void) util_srx_close(&ep->srx->fid);

	sm2_fqe_ctx_fs_free(ep->fqe_ctx_fs);
	sm2_pend_fs_free(ep->pend_fs);
	ofi_spin_destroy(&ep->tx_lock);

	free((void *) ep->name);
	free(ep);
	return 0;
}

static int
sm2_ep_trywait(void *arg)
{
	struct sm2_ep *ep;

	ep = container_of(arg, struct sm2_ep, util_ep.ep_fid.fid);

	sm2_ep_progress(&ep->util_ep);

	return FI_SUCCESS;
}

static int
sm2_ep_bind_cq(struct sm2_ep *ep, struct util_cq *cq, uint64_t flags)
{
	int ret;

	ret = ofi_ep_bind_cq(&ep->util_ep, cq, flags);
	if (ret)
		return ret;

	if (cq->wait) {
		ret = ofi_wait_add_fid(cq->wait, &ep->util_ep.ep_fid.fid, 0,
				       sm2_ep_trywait);
		if (ret)
			return ret;
	}

	ret = fid_list_insert(&cq->ep_list, &cq->ep_list_lock,
			      &ep->util_ep.ep_fid.fid);

	return ret;
}

static int
sm2_ep_bind_cntr(struct sm2_ep *ep, struct util_cntr *cntr, uint64_t flags)
{
	int ret;

	ret = ofi_ep_bind_cntr(&ep->util_ep, cntr, flags);
	if (ret)
		return ret;

	if (cntr->wait) {
		ret = ofi_wait_add_fid(cntr->wait, &ep->util_ep.ep_fid.fid, 0,
				       sm2_ep_trywait);
		if (ret)
			return ret;
	}

	return FI_SUCCESS;
}

static int
sm2_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct sm2_ep *ep;
	struct util_av *av;
	int ret = 0;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct util_av, av_fid.fid);
		ret = ofi_ep_bind_av(&ep->util_ep, av);
		if (ret) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"duplicate AV binding\n");
			return -FI_EINVAL;
		}
		break;
	case FI_CLASS_CQ:
		ret = sm2_ep_bind_cq(
			ep, container_of(bfid, struct util_cq, cq_fid.fid),
			flags);
		break;
	case FI_CLASS_EQ:
		break;
	case FI_CLASS_CNTR:
		ret = sm2_ep_bind_cntr(
			ep, container_of(bfid, struct util_cntr, cntr_fid.fid),
			flags);
		break;
	case FI_CLASS_SRX_CTX:
		ep->srx = container_of(bfid, struct fid_ep, fid);
		break;
	default:
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int
sm2_discard(struct fi_peer_rx_entry *rx_entry)
{
	struct sm2_fqe_ctx *fqe_ctx = rx_entry->peer_context;

	ofi_freestack_push(fqe_ctx->ep->fqe_ctx_fs, fqe_ctx);
	return FI_SUCCESS;
}

struct fi_ops_srx_peer sm2_srx_peer_ops;

int
sm2_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		struct fid_ep **rx_ep, void *context)
{
	struct sm2_domain *sm2_domain;

	sm2_domain =
		container_of(domain, struct sm2_domain, util_domain.domain_fid);

	if (attr->op_flags & FI_PEER) {
		sm2_domain->srx =
			((struct fi_peer_srx_context *) (context))->srx;
		sm2_domain->srx->peer_ops = &sm2_srx_peer_ops;
		return FI_SUCCESS;
	}
	return util_ep_srx_context(&sm2_domain->util_domain, attr->size,
				   SM2_IOV_LIMIT, SM2_INJECT_SIZE, rx_ep);
}

static int
sm2_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct sm2_attr attr;
	struct sm2_domain *domain;
	struct sm2_ep *ep;
	struct sm2_av *av;
	struct fid_peer_srx *srx;
	int ret, self_id;

	ep = container_of(fid, struct sm2_ep, util_ep.ep_fid.fid);
	av = container_of(ep->util_ep.av, struct sm2_av, util_av);

	switch (command) {
	case FI_ENABLE:
		if ((ofi_needs_rx(ep->util_ep.caps) && !ep->util_ep.rx_cq) ||
		    (ofi_needs_tx(ep->util_ep.caps) && !ep->util_ep.tx_cq))
			return -FI_ENOCQ;
		if (!ep->util_ep.av)
			return -FI_ENOAV;

		attr.name = ep->name;
		// TODO Decide on correct default number of FQE's
		attr.num_fqe = ep->tx_size;
		attr.flags = ep->util_ep.caps & 0;

		ret = sm2_create(&sm2_prov, &attr, &av->sm2_mmap, &self_id);
		ep->mmap_regions = &av->sm2_mmap;
		ep->self_fiaddr = self_id;

		if (ret)
			return ret;

		if (!ep->srx) {
			domain = container_of(ep->util_ep.domain,
					      struct sm2_domain,
					      util_domain.domain_fid);
			// TODO Figure out what size goes here?? ep->tx_size?
			ret = util_ep_srx_context(&domain->util_domain,
						  ep->tx_size, SM2_IOV_LIMIT,
						  SM2_INJECT_SIZE, &ep->srx);
			if (ret)
				return ret;
			util_get_peer_srx(ep->srx)->peer_ops =
				&sm2_srx_peer_ops;
			ret = util_srx_bind(&ep->srx->fid,
					    &ep->util_ep.rx_cq->cq_fid.fid,
					    FI_RECV);
			if (ret)
				return ret;
		} else {
			srx = calloc(1, sizeof(*srx));
			srx->peer_ops = &sm2_srx_peer_ops;
			srx->owner_ops = sm2_get_peer_srx(ep)->owner_ops;
			srx->ep_fid.fid.context =
				sm2_get_peer_srx(ep)->ep_fid.fid.context;
			ep->srx = &srx->ep_fid;
			ep->util_ep.ep_fid.msg = &sm2_no_recv_msg_ops;
			ep->util_ep.ep_fid.tagged = &sm2_no_recv_tag_ops;
		}

		break;
	default:
		return -FI_ENOSYS;
	}
	return ret;
}

static int
sm2_endpoint_name(struct sm2_ep *ep, char *name, char *addr, size_t addrlen)
{
	memset(name, 0, SM2_NAME_MAX);
	if (!addr || addrlen > SM2_NAME_MAX)
		return -FI_EINVAL;

	pthread_mutex_lock(&sm2_ep_list_lock);
	ep->ep_idx = sm2_global_ep_idx++;
	pthread_mutex_unlock(&sm2_ep_list_lock);

	if (strstr(addr, SM2_PREFIX))
		snprintf(name, SM2_NAME_MAX - 1, "%s:%d:%d", addr, getuid(),
			 ep->ep_idx);
	else
		/* this is an fi_ns:// address.*/
		snprintf(name, SM2_NAME_MAX - 1, "%s", addr);

	return 0;
}

static void
sm2_init_sig_handlers(void)
{
	static bool sig_init = false;

	pthread_mutex_lock(&sm2_ep_list_lock);
	if (sig_init)
		goto out;

	/* Signal handlers to cleanup tmpfs files on an unclean shutdown */
	assert(SIGBUS < SIGRTMIN && SIGSEGV < SIGRTMIN && SIGTERM < SIGRTMIN &&
	       SIGINT < SIGRTMIN);
	sm2_reg_sig_handler(SIGBUS);
	sm2_reg_sig_handler(SIGSEGV);
	sm2_reg_sig_handler(SIGTERM);
	sm2_reg_sig_handler(SIGINT);

	sig_init = true;
out:
	pthread_mutex_unlock(&sm2_ep_list_lock);
}

static struct fi_ops sm2_ep_fi_ops;
static struct fi_ops_ep sm2_ep_ops;
static struct fi_ops_cm sm2_cm_ops;

int
sm2_endpoint(struct fid_domain *domain, struct fi_info *info,
	     struct fid_ep **ep_fid, void *context)
{
	struct sm2_ep *ep;
	int ret;
	char name[SM2_NAME_MAX];

	sm2_init_sig_handlers();

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = sm2_endpoint_name(ep, name, info->src_addr, info->src_addrlen);

	if (ret)
		goto ep;
	ret = sm2_setname(&ep->util_ep.ep_fid.fid, name, SM2_NAME_MAX);
	if (ret)
		goto ep;

	ret = ofi_spin_init(&ep->tx_lock);
	if (ret)
		goto name;

	ep->tx_size = info->tx_attr->size;
	ret = ofi_endpoint_init(domain, &sm2_util_prov, info, &ep->util_ep,
				context, sm2_ep_progress);
	if (ret)
		goto lock;

	ep->util_ep.ep_fid.msg = &sm2_msg_ops;
	ep->util_ep.ep_fid.tagged = &sm2_tag_ops;

	ep->fqe_ctx_fs = sm2_fqe_ctx_fs_create(info->rx_attr->size, NULL, NULL);
	ep->pend_fs = sm2_pend_fs_create(info->tx_attr->size, NULL, NULL);

	ep->util_ep.ep_fid.fid.ops = &sm2_ep_fi_ops;
	ep->util_ep.ep_fid.ops = &sm2_ep_ops;
	ep->util_ep.ep_fid.cm = &sm2_cm_ops;
	ep->util_ep.ep_fid.rma = NULL;

	*ep_fid = &ep->util_ep.ep_fid;
	return 0;

lock:
	ofi_spin_destroy(&ep->tx_lock);
name:
	free((void *) ep->name);
ep:
	free(ep);
	return ret;
}

sm2_proto_func sm2_proto_ops[sm2_src_max] = {
	[sm2_src_inject] = &sm2_do_inject,
};

static struct fi_ops_cm sm2_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = sm2_setname,
	.getname = sm2_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};

static struct fi_ops sm2_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sm2_ep_close,
	.bind = sm2_ep_bind,
	.control = sm2_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

struct fi_ops_srx_peer sm2_srx_peer_ops = {
	.size = sizeof(struct fi_ops_srx_peer),
	.start_msg = sm2_unexp_start,
	.start_tag = sm2_unexp_start,
	.discard_msg = sm2_discard,
	.discard_tag = sm2_discard,
};

static struct fi_ops_ep sm2_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = sm2_ep_cancel,
	.getopt = sm2_ep_getopt,
	.setopt = sm2_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};
