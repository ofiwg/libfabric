/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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

#include <sys/uio.h>
#include <sys/param.h>
#include <sys/user.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <fcntl.h>
#include <assert.h>
#include <search.h>
#include <time.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_atomic.h>
#include <rdma/fabric.h>

#include "error.h"
#include "ipc.h"
#include "shared.h"
#include "test_util.h"
#include "test.h"

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

static pthread_mutex_t	fi_tostr_lock;
static const uint64_t max_wait_s = 6;

// fi_tostr is not thread safe, unfortunately.
static char *fi_tostr_safe(const void *buf, enum fi_type datatype)
{
	pthread_mutex_lock(&fi_tostr_lock);
	char *retbuf = strdup(fi_tostr(buf, datatype));
	pthread_mutex_unlock(&fi_tostr_lock);
	return retbuf;
}

static void init_buf(struct rank_info *ri, void *uaddr, size_t length,
			unsigned int seed, enum fi_hmem_iface iface)
{
	struct random_data random_data;
	char statebuf[64];
	uint8_t *buf = malloc(sizeof(*buf) * length);
	if (!buf)
		ERRORX(ri, "Ran out of memory");

	random_data.state = NULL;
	INSIST_EQ(ri,
		  initstate_r(seed, statebuf, sizeof(statebuf), &random_data),
		  0, "%d");
	for (size_t i = 0; i < length; i++) {
		int32_t c;
		INSIST_EQ(ri, random_r(&random_data, &c), 0, "%d");
		// & 0xFF so we take just one byte at a time. That keeps things
		// simple.
		buf[i] = c & 0xFF;
	}

	if (iface == FI_HMEM_SYSTEM) {
		memcpy(uaddr, buf, length);
	} else {
		if (hmem_memcpy_h2d(iface, uaddr, buf, length))
			ERRORX(ri, "Cuda memcpy h2d failed");
	}

	free(buf);
}

void util_global_init()
{
	pthread_mutex_init(&fi_tostr_lock, NULL);
	hmem_init();
}

void util_init(struct rank_info *ri)
{
	int rc;
	struct fi_info hints = { 0 };
	struct fi_ep_attr ep_attr = { 0 };
	struct fi_fabric_attr fabric_attr = { 0 };
	struct fi_domain_attr domain_attr = { 0 };

	hints.fabric_attr = &fabric_attr;
	hints.ep_attr = &ep_attr;
	hints.domain_attr = &domain_attr;

	hints.ep_attr->type = FI_EP_RDM;
	hints.ep_attr->protocol = FI_PROTO_LPP;
	// TODO: Run some tests with more surgical application of caps (e.g.,
	// only FI_MSG and FI_SEND for the sending side endpoint).
	hints.caps = FI_ATOMIC | FI_RMA | FI_MSG | FI_TAGGED | FI_READ |
		 FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_HMEM;

	hints.mode = 0;
	hints.fabric_attr->prov_name = "lpp";
	hints.domain_attr->mr_mode = FI_MR_LOCAL | OFI_MR_BASIC_MAP;

	rc = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			NULL, NULL, 0, &hints, &ri->fi);
	if (rc == -ENODATA) {
		warn("Failed to find provider with FI_HMEM, trying again without\n");
		hints.caps &= ~FI_HMEM;
		INSIST_FI_EQ(ri,
			     fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
				        NULL, NULL, 0, &hints, &ri->fi),
			     0);
	}


	INSIST_FI_EQ(ri,
		     fi_fabric(ri->fi->fabric_attr, &ri->fabric,
			       &ri->sync_op_context.fi_context),
		     0);
	INSIST_FI_EQ(ri,
		     fi_domain(ri->fabric, ri->fi, &ri->domain,
			       &ri->sync_op_context.fi_context),
		     0);
}

static void free_mr_uaddr(struct mr_info *mri)
{
	if (mri->hmem_iface == FI_HMEM_SYSTEM) {
		free(mri->uaddr);
	} else {
		hmem_free(mri->hmem_iface, mri->uaddr);
	}
}

void util_teardown(struct rank_info *ri, struct rank_info *pri)
{
	TRACE(ri, util_barrier(ri));

	for (int i = 0; i < MAX_EP_INFO; i++) {
		struct ep_info *ep_info = &ri->ep_info[i];
		if (ep_info->valid) {
			fi_close(&ep_info->fid->fid);
			fi_close(&ep_info->stx->fid);
			fi_close(&ep_info->tx_cq_fid->fid);
			fi_close(&ep_info->rx_cq_fid->fid);
			fi_close(&ep_info->tx_cntr_fid->fid);
			fi_close(&ep_info->rx_cntr_fid->fid);
			fi_close(&ep_info->av->fid);
		}
		ep_info->valid = 0;
	}

	for (int i = 0; i < MAX_MR_INFO; i++) {
		struct mr_info *mr_info = &ri->mr_info[i];
		if (mr_info->valid) {
			if (!mr_info->skip_reg) {
				fi_close(&mr_info->fid->fid);
			}
			if (mr_info->uaddr != NULL) {
				free_mr_uaddr(mr_info);
			}
		}
		mr_info->valid = 0;
	}
	fi_close(&ri->domain->fid);
	fi_close(&ri->fabric->fid);
	fi_freeinfo(ri->fi);
	if (pri != NULL) {
		put_peer_rank_info(pri);
	}
}

void util_barrier(struct rank_info *ri)
{
	TRACE(ri, peer_barrier(ri));
}

void util_av_insert_all(struct rank_info *ri, struct rank_info *tgt_ri)
{
	for (int i = 0; i < MAX_EP_INFO; i++) {
		struct ep_info *tgt_ep_info = &tgt_ri->ep_info[i];
		if (tgt_ep_info->valid) {
			for (int j = 0; j < MAX_EP_INFO; j++) {
				struct ep_info *ep_info = &ri->ep_info[j];
				if (ep_info->valid) {
					INSIST_FI_EQ(
						ri,
						fi_av_insert(
							ep_info->av,
							tgt_ep_info->name, 1,
							&tgt_ep_info->fi_addr,
							0, NULL),
						1);
				}
			}
		}
	}
}

void util_sync(struct rank_info *ri, struct rank_info **pri)
{
	*pri = exchange_rank_info(ri);

	util_av_insert_all(ri, *pri);
}

void util_create_ep(struct rank_info *ri, struct ep_params *params)
{
	INSIST(ri, params->idx < MAX_EP_INFO);
	struct ep_info *ep_info = &ri->ep_info[params->idx];

	INSIST(ri, ep_info->valid == 0);

	struct fi_cq_attr cq_attr = {
		.size = params->cq_size == 0 ? 256 : params->cq_size,
		.flags = 0,
		.format = FI_CQ_FORMAT_TAGGED,
		.wait_obj = FI_WAIT_NONE,
		.signaling_vector = 0,
		.wait_cond = FI_CQ_COND_NONE,
		.wait_set = NULL
	};

	INSIST_FI_EQ(ri,
		     fi_cq_open(ri->domain, &cq_attr, &ep_info->tx_cq_fid,
				&ri->sync_op_context.fi_context),
		     0);
	INSIST_FI_EQ(ri,
		     fi_cq_open(ri->domain, &cq_attr, &ep_info->rx_cq_fid,
				&ri->sync_op_context.fi_context),
		     0);

	struct fi_cntr_attr cntr_attr = {
		.wait_obj = FI_WAIT_UNSPEC,
		.events = FI_CNTR_EVENTS_COMP
	};
	INSIST_FI_EQ(ri,
		       fi_cntr_open(ri->domain, &cntr_attr,
				    &ep_info->tx_cntr_fid,
				    &ri->sync_op_context.fi_context),
		       0);
	INSIST_FI_EQ(ri,
		       fi_cntr_open(ri->domain, &cntr_attr,
				    &ep_info->rx_cntr_fid,
				    &ri->sync_op_context.fi_context),
		       0);

	// Set any user specified attributes.
	ri->fi->tx_attr->caps |= params->tx_attr.additional_caps;
	ri->fi->tx_attr->op_flags = params->tx_attr.op_flags;
	ri->fi->rx_attr->caps |= params->rx_attr.additional_caps;
	ri->fi->rx_attr->op_flags = params->rx_attr.op_flags;

	INSIST_FI_EQ(ri,
		     fi_endpoint(ri->domain, ri->fi, &ep_info->fid,
				 &ri->sync_op_context.fi_context),
		     0);

	struct fi_av_attr av_attr = {
		.type = FI_AV_MAP,
		.count = 2,
	};
	INSIST_FI_EQ(ri,
		     fi_av_open(ri->domain, &av_attr, &ep_info->av,
				&ri->sync_op_context.fi_context),
		     0);

	INSIST_FI_EQ(ri,
		     fi_stx_context(ri->domain, ri->fi->tx_attr, &ep_info->stx,
				    &ri->sync_op_context.fi_context),
		     0);

	// We don't want the caller setting something like FI_TRANSMIT in the
	// cq_bind_flags; we'll take care of that.
	INSIST(ri, params->cq_bind_flags == 0 ||
			   params->cq_bind_flags == FI_SELECTIVE_COMPLETION);

	INSIST_FI_EQ(ri,
		     fi_ep_bind(ep_info->fid, &ep_info->tx_cq_fid->fid,
				params->cq_bind_flags | FI_TRANSMIT),
		     0);
	INSIST_FI_EQ(ri,
		     fi_ep_bind(ep_info->fid, &ep_info->rx_cq_fid->fid,
				params->cq_bind_flags | FI_RECV),
		     0);
	INSIST_FI_EQ(ri,
		     fi_ep_bind(ep_info->fid, &ep_info->tx_cntr_fid->fid,
				FI_SEND | FI_READ | FI_WRITE),
		     0);
	INSIST_FI_EQ(ri,
		     fi_ep_bind(ep_info->fid, &ep_info->rx_cntr_fid->fid,
				FI_RECV),
		     0);
	INSIST_FI_EQ(ri, fi_ep_bind(ep_info->fid, &ep_info->av->fid, 0), 0);
	INSIST_FI_EQ(ri, fi_ep_bind(ep_info->fid, &ep_info->stx->fid, 0), 0);

	INSIST_FI_EQ(ri, fi_enable(ep_info->fid), 0);

	size_t addrlen = sizeof(ep_info->name);
	INSIST_FI_EQ(
		ri, fi_getname(&ep_info->fid->fid, ep_info->name, &addrlen), 0);

	ep_info->valid = 1;
}

void util_create_mr(struct rank_info *ri, struct mr_params *params)
{
	INSIST(ri, params->idx < MAX_MR_INFO);
	struct mr_info *mr_info = &ri->mr_info[params->idx];

	INSIST_EQ(ri, mr_info->valid, 0, "%d");

	mr_info->orig_seed = params->seed;
	mr_info->length = params->length;
	mr_info->hmem_iface = params->hmem_iface;

	if (mr_info->hmem_iface) {
		hmem_alloc(mr_info->hmem_iface, &mr_info->uaddr, params->length);
	}
	else if (params->page_align) {
		if (posix_memalign(&mr_info->uaddr, PAGE_SIZE,
				   params->length) != 0) {
			ERRORX(ri, "posix_memalign");
		}
	} else {
		mr_info->uaddr = malloc(params->length);
		// NULL is okay if the MR length is 0; malloc may or may not
		// give us an address in that case.
		if (mr_info->uaddr == NULL && params->length != 0) {
			ERRORX(ri, "malloc");
		}
	}

	init_buf(ri, mr_info->uaddr, mr_info->length,
			params->seed, mr_info->hmem_iface);

	if (!params->skip_reg) {
		INSIST_FI_EQ(ri,
			     fi_mr_reg(ri->domain, mr_info->uaddr,
				       mr_info->length, params->access, 0, 0, 0,
				       &mr_info->fid,
				       &ri->sync_op_context.fi_context),
			     0);
		mr_info->key = fi_mr_key(mr_info->fid);
	} else {
		mr_info->key = 0;
	}
	mr_info->skip_reg = params->skip_reg;

	// TODO: bool
	mr_info->valid = 1;
}

void util_validate_cq_entry(struct rank_info *ri,
			    struct fi_cq_tagged_entry *tentry,
			    struct fi_cq_err_entry *errentry,
			    uint64_t flags, uint64_t data,
			    uint64_t context_val, bool multi_recv,
			    uint64_t buf_offset)
{
	INSIST(ri, (tentry == NULL) != (errentry == NULL));

	uint64_t entry_flags = tentry ? tentry->flags : errentry->flags;
	void *entry_context = tentry ? tentry->op_context : errentry->op_context;

	struct context *cp =
		container_of(entry_context, struct context, fi_context);

	if (context_val != 0) {
		INSIST_EQ(ri, cp->context_val, context_val, "%lx");
	}

	char *flags_str = fi_tostr_safe(&flags, FI_TYPE_CAPS);
	char *entry_flags_str = fi_tostr_safe(&entry_flags, FI_TYPE_CAPS);
	if (flags != 0 && entry_flags != flags) {
		ERRORX(ri,
		       "CQ entry requires flags 0x%lx (%s), but instead has 0x%lx (%s)",
		       flags, flags_str, entry_flags, entry_flags_str);
	}
	free(flags_str);
	free(entry_flags_str);

	if (flags & FI_REMOTE_CQ_DATA && tentry->data != data){
		ERRORX(ri,
		       "FI_REMOTE_CQ_DATA not properly set, expected: 0x%lx, got 0x%lx",
		       data, tentry->data);
	}

	if (multi_recv) {
		// If writing a FI_MULTI_RECV test, the MR-aware context create
		// functions should be used so we can look up the MR associated
		// with a given recv here.
		if (cp->mr_idx == CTX_NOMR) {
			ERRORX(ri, "fi_context does not have an associated MR");
		}
		struct mr_info *mr_info = &ri->mr_info[cp->mr_idx];
		INSIST_EQ(ri, mr_info->valid, 1, "%d");

		if ((uint64_t)tentry->buf != (uint64_t)mr_info->uaddr + buf_offset) {
			ERRORX(ri,
			       "MULTI_RECV completion  entry 'buf' is 0x%lx, but should be 0x%lx",
			       (uint64_t)tentry->buf,
			       (uint64_t)mr_info->uaddr + buf_offset);
		}
	}
}

static void wait_cq_common(struct rank_info *ri, struct fid_cq *cq,
			   uint64_t context_val, uint64_t flags, uint64_t data,
			   bool multi_recv, uint64_t buf_offset,
			   bool expect_error, bool expect_empty)
{
	struct fi_cq_tagged_entry tentry;
	struct fi_cq_err_entry eentry;
	uint64_t waited_ms = 0;
	bool error = false;

	// It makes sense to set only one of these flags.
	INSIST(ri, !(expect_error && expect_empty));

	ssize_t nread;
	do {
		nread = fi_cq_read(cq, &tentry, 1);
		usleep(10000);
		if (expect_empty) {
			if (nread != -FI_EAGAIN) {
				ERRORX(ri,
				       "Expected empty CQ, but found an entry (nread == %ld %s)",
				       nread,
				       nread < 0 ? fi_strerror(labs(nread)) :
						   "");
			} else {
				return;
			}
		}
		waited_ms += 10;
		if (waited_ms >= 1000 * max_wait_s) {
			ERRORX(ri, "timed out waiting on CQ after %lus",
			       max_wait_s);
		}
	} while (nread == -FI_EAGAIN);

	if (nread == -FI_EAVAIL) {
		INSIST_FI_EQ(ri, fi_cq_readerr(cq, &eentry, 0), 1);
		if (expect_error) {
			nread = 1;
			error = true;
		} else {
			ERRORX(ri,
			       "CQ error queue contains entry. err: %d (%s) prov_errno: %d",
			       eentry.err, fi_strerror(eentry.err),
			       eentry.prov_errno);
		}
	} else if (nread == 1 && expect_error) {
		ERRORX(ri, "Expected CQ error entry, but got a normal entry");
	}

	INSIST_EQ(ri, nread, 1, "%zd");

	util_validate_cq_entry(ri, error ? NULL : &tentry,
			       error ? &eentry : NULL, flags, data, context_val,
			       multi_recv, buf_offset);
}

void util_wait_tx_cq(struct rank_info *ri, struct wait_tx_cq_params *params)
{
	INSIST(ri, params->ep_idx < MAX_EP_INFO);
	struct ep_info *ep_info = &ri->ep_info[params->ep_idx];
	INSIST(ri, ep_info->valid);

	wait_cq_common(ri, ep_info->tx_cq_fid, params->context_val,
		       params->flags, params->data, 0, 0, params->expect_error,
		       params->expect_empty);
}

void util_wait_rx_cq(struct rank_info *ri, struct wait_rx_cq_params *params)
{
	INSIST(ri, params->ep_idx < MAX_EP_INFO);
	struct ep_info *ep_info = &ri->ep_info[params->ep_idx];
	INSIST(ri, ep_info->valid);

	wait_cq_common(ri, ep_info->rx_cq_fid, params->context_val,
		       params->flags, params->data, params->multi_recv,
		       params->buf_offset, params->expect_error, params->expect_empty);
}

void util_wait_cntr_many(struct rank_info *ri,
			 struct wait_cntr_params *paramslist, size_t n_params)
{
	uint64_t waited_ms = 0;
	uint64_t finished = 0;
	size_t n_finished = 0;

	if (n_params > 64) {
		ERRORX(ri, "n_params is too large\n");
	}

	while (1) {
		for (size_t i = 0; i < n_params; i++) {
			if ((1UL << i) & finished) {
				continue;
			}
			struct wait_cntr_params *params = &paramslist[i];

			INSIST(ri, params->ep_idx < MAX_EP_INFO);
			struct ep_info *ep_info = &ri->ep_info[params->ep_idx];
			INSIST(ri, ep_info->valid);

			struct fid_cntr *cntr;
			if (params->which == WAIT_CNTR_TX) {
				cntr = ep_info->tx_cntr_fid;
			} else if (params->which == WAIT_CNTR_RX) {
				cntr = ep_info->rx_cntr_fid;
			} else {
				ERRORX(ri,
				       "bogus counter which for cntr %zu: %d\n",
				       i, params->which);
			}

			uint64_t desired_val = params->val;
			uint64_t val = fi_cntr_read(cntr);
			uint64_t errval = fi_cntr_readerr(cntr);
			if (errval != 0) {
				ERRORX(ri,
				       "Error counter for cntr %zu is %lu\n", i,
				       errval);
			}
			if (val >= desired_val) {
				finished |= (1UL << i);
				n_finished++;
			}
		}
		if (n_finished < n_params) {
			usleep(10000);
			waited_ms += 10;
			if (waited_ms >= 1000 * max_wait_s) {
				ERRORX(ri,
				       "timed out waiting on cntrs after %lus (n_params: %zu, finished 0x%lx)",
				       max_wait_s, n_params, finished);
			}
		} else {
			break;
		}
	}
}

void util_wait_cntr(struct rank_info *ri, struct wait_cntr_params *params)
{
	TRACE(ri, util_wait_cntr_many(ri, params, 1));
}

uint64_t util_read_tx_cntr(struct rank_info *ri, uint64_t ep_idx)
{
	INSIST(ri, ep_idx < MAX_EP_INFO);
	struct ep_info *ep_info = &ri->ep_info[ep_idx];
	INSIST(ri, ep_info->valid);

	return fi_cntr_read(ep_info->tx_cntr_fid);
}

uint64_t util_read_rx_cntr(struct rank_info *ri, uint64_t ep_idx)
{
	INSIST(ri, ep_idx < MAX_EP_INFO);
	struct ep_info *ep_info = &ri->ep_info[ep_idx];
	INSIST(ri, ep_info->valid);

	return fi_cntr_read(ep_info->rx_cntr_fid);
}

static void verify_buf(struct rank_info *ri, volatile void *uaddr,
		       size_t offset, size_t length, unsigned int orig_seed,
		       unsigned int expected_seed, size_t expected_seed_offset)
{
#define MK_RAND_BUF(name, seed, offset)                                        \
	struct random_data name##_random_data;                                 \
	name##_random_data.state = NULL;                                       \
	char name##_statebuf[64];                                              \
	INSIST_EQ(ri,                                                          \
		  initstate_r(seed, name##_statebuf, sizeof(name##_statebuf),  \
			      &name##_random_data),                            \
		  0, "%d");                                                    \
	int32_t name##_c;                                                      \
	/* Skip into the random sequence according to the offset. */           \
	for (int i = 0; i < offset; i++) {                                     \
		INSIST_EQ(ri, random_r(&name##_random_data, &name##_c), 0,     \
			  "%d");                                               \
	}

	MK_RAND_BUF(orig, orig_seed, offset);
	MK_RAND_BUF(expected, expected_seed, expected_seed_offset);

	// TODO: this is a little more complicated than needed due to old
	// atomic tests. A refactoring might be nice.
	volatile uint8_t *buf = (uint8_t*)uaddr + offset;
	for (int i = 0; i < length; i++) {
		uint64_t orig_buf;
		uint8_t *orig_bufp = (uint8_t*)&orig_buf;
		uint64_t expected_buf;
		uint8_t *expected_bufp = (uint8_t*)&expected_buf;

		// Truncate down to one byte at a time, like the init_buf code.
		INSIST_EQ(ri, random_r(&orig_random_data, &orig_c), 0, "%d");
		INSIST_EQ(ri, random_r(&expected_random_data, &expected_c), 0, "%d");
		*orig_bufp = orig_c & 0xFF;
		*expected_bufp = expected_c & 0xFF;

		orig_c = (uint8_t)orig_buf;
		expected_c = (uint8_t)expected_buf;
		if (buf[i] != expected_c) {
			ERRORX(ri,
			       "buffer invalid at byte %ld (original byte was %x; we found %x, but expected %x)",
			       i + offset, orig_c, buf[i], expected_c);
		}
	}
#undef MK_RAND_BUF
}

void util_verify_buf(struct rank_info *ri, struct verify_buf_params *params)
{

	struct mr_info *mr_info;
	uint8_t *tmpbuf = NULL, *vbuf;

	INSIST(ri, params->mr_idx < MAX_MR_INFO);
	mr_info = &ri->mr_info[params->mr_idx];

	INSIST(ri, mr_info->valid);
	INSIST(ri, params->offset + params->length <
			   (uint64_t)mr_info->uaddr + mr_info->length);

	if(mr_info->hmem_iface == FI_HMEM_SYSTEM){
		vbuf = mr_info->uaddr;
	} else {
		tmpbuf = malloc(params->length);
		if (!tmpbuf)
			ERRORX(ri, "Ran out of memory");
		if(hmem_memcpy_d2h(mr_info->hmem_iface, tmpbuf, mr_info->uaddr,
						params->length))
			ERRORX(ri, "hmem d2h memcpy failed");
		vbuf = tmpbuf;
	}

	verify_buf(ri, vbuf, params->offset, params->length,
		   mr_info->orig_seed, params->expected_seed,
		   params->expected_seed_offset);

	if(tmpbuf)
		free(tmpbuf);
}

void util_simple_setup(struct rank_info *ri, struct rank_info **pri,
		       size_t length, uint64_t lcl_access, uint64_t rem_access)
{
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };

	TRACE(ri, util_init(ri));

	mr_params.idx = 0;
	mr_params.length = length;

	if (my_node == NODE_A) {
		mr_params.access = lcl_access;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = rem_access;
		mr_params.seed = seed_node_b;
	}
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, pri));
}

static int context_compare(const void *a, const void *b)
{
	const struct context *ctx_a = a;
	const struct context *ctx_b = b;

	if (ctx_a->context_val < ctx_b->context_val) {
		return -1;
	} else if (ctx_a->context_val == ctx_b->context_val) {
		return 0;
	} else {
		return 1;
	}
}

// Associating an MR with a ctx allows us to lookup the buffer address when
// validating MULTI_RECV offsets.
#define CTX_NOMR	((uint64_t)(-1))
struct context *get_ctx(struct rank_info *ri, uint64_t context_val,
			uint64_t mr_idx)
{
	void *val;
	struct context *cp;
	struct context search_ctx;

	pthread_mutex_lock(&ri->lock);

	search_ctx.context_val = context_val;

	val = tfind(&search_ctx, &ri->context_tree_root, context_compare);
	if (val != NULL) {
		cp = *(struct context**)val;
		if (mr_idx != CTX_NOMR) {
			if (cp->mr_idx != CTX_NOMR) {
				ERRORX(ri, "mr_idx is already assigned");
			} else {
				cp->mr_idx = mr_idx;
			}
		}
	} else {
		cp = malloc(sizeof(*cp));

		if (cp == NULL) {
			ERRORX(ri, "malloc context");
		}
		cp->context_val = context_val;
		cp->mr_idx = mr_idx;

		val = tsearch((void*)cp, &ri->context_tree_root, context_compare);
		if (val == NULL) {
			ERRORX(ri, "rank %ld: failed to tsearch insert context",
			       ri->rank);
		} else if (*(struct context**)val != cp) {
			ERRORX(ri, "rank %ld: duplicate context in tree", ri->rank);
		}
	}

	pthread_mutex_unlock(&ri->lock);

	return cp;
}

// Must hold ri->lock.
void free_ctx_tree(struct rank_info *ri)
{
	tdestroy(ri->context_tree_root, free);
}

void util_get_time(struct timespec *ts)
{
	clock_gettime(CLOCK_MONOTONIC, ts);
}

uint64_t util_time_delta_ms(struct timespec *start, struct timespec *end)
{
	uint64_t start_ms = start->tv_sec * 1000 + start->tv_nsec / 1000000;
	uint64_t end_ms = end->tv_sec * 1000 + end->tv_nsec / 1000000;

	return end_ms - start_ms;
}
