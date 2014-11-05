/*
 * Copyright (c) 2014, Cisco Systems, Inc. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <asm/types.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "fi_enosys.h"
#include "fi.h"

#include "usnic_direct.h"
#include "usdf.h"
#include "usd_queue.h"

static void
usdf_av_insert_complete(struct usdf_av_insert *insert)
{
	struct fi_eq_entry entry;
	struct usdf_av *av;

	av = insert->avi_av;

	entry.fid = &av->av_fid.fid;
	entry.context = insert->avi_context;
	entry.data = atomic_get(&insert->avi_successful_ops);
	usdf_eq_write_internal(av->av_eq,
		FI_COMPLETE, &entry, sizeof(entry), 0);

	pthread_spin_lock(&av->av_lock);

	atomic_dec(&av->av_active_inserts);
	if (atomic_get(&av->av_active_inserts) == 0 && av->av_closing) {
		free(av);
	} else {
		pthread_spin_unlock(&av->av_lock);
	}

	free(insert);
}

static void
usdf_av_completion(struct usdf_av_req *req, int status, struct usd_dest *dest)
{
	struct fi_eq_err_entry err_entry;
	struct usdf_av_insert *insert;
	struct usdf_av *av;

	insert = req->avr_insert;
	av = insert->avi_av;

	if (status == 0) {
		*(struct usd_dest **)req->avr_fi_addr = dest;
		atomic_inc(&insert->avi_successful_ops);
	} else {
		*req->avr_fi_addr = FI_ADDR_NOTAVAIL;

		err_entry.fid = &av->av_fid.fid;
		err_entry.context = insert->avi_context;
		err_entry.data = req - (struct usdf_av_req *)(insert + 1);
		err_entry.err = -status;

		usdf_eq_write_internal(av->av_eq, FI_COMPLETE,
			&err_entry, sizeof(err_entry),
			USDF_EVENT_FLAG_ERROR);
	}

	/* decrement pending ops and generate event if all done */
	atomic_inc(&insert->avi_completed_ops);
	if (atomic_get(&insert->avi_completed_ops) == insert->avi_pending_ops) {
		usdf_av_insert_complete(insert);
	}
	usdf_progression_item_complete(av->av_domain);
}

/*
 * Called by progression thread to look for AV completions on this domain
 */
void
usdf_av_progress(struct usdf_domain *udp)
{
	int ret;
	int status;
	void *context;
	struct usd_dest *dest;
	struct usdf_av_req *req;

	do {

		/* usd_dest_* not threadsafe... */
		pthread_spin_lock(&udp->dom_usd_lock);
		ret = usd_create_dest_poll(udp->dom_dev, &context, &status,
				&dest);
		pthread_spin_unlock(&udp->dom_usd_lock);

		if (ret == 0) {
			req = context;
			usdf_av_completion(req, status, dest);
		}
	} while (ret == 0);
}

/*
 * The first call to av_insert or the next call after a sync will start
 * a sync_list.  This and each succeeding av_insert will create an insert_block
 * and add it to the current sync_list until fi_sync(av) is called.
 * Once av_sync is called, an event will be generated as soon as the last
 * address in the sync_list is resolved.
 */
static int
usdf_am_insert(struct fid_av *fav, const void *addr, size_t count,
			  fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	const struct sockaddr_in *sin;
	struct usdf_av_insert *insert;
	struct usdf_av_req *req;
	struct usdf_av *av;
	struct usdf_domain *udp;
	struct usd_dest *dest;
	int ret_count;
	int ret;
	int i;

	if ((flags & ~FI_MORE) != 0) {
		return -FI_EBADFLAGS;
	}

	av = av_ftou(fav);

	ret_count = 0;
	sin = addr;

	if (av->av_flags & FI_EVENT) {
		if (av->av_eq == NULL) {
			return -FI_EINVAL;
		}
		udp = av->av_domain;

		/* allocate all we need in one go */
		insert = malloc(sizeof(*insert) + count * sizeof(*req));
		if (insert == NULL) {
			return -errno;
		}
		insert->avi_av = av;
		insert->avi_pending_ops = count;
		atomic_init(&insert->avi_completed_ops, 0);
		atomic_init(&insert->avi_successful_ops, 0);
		insert->avi_context = context;

		/* If no addresses, complete now */
		if (count == 0) {
			usdf_av_insert_complete(insert);
			return 0;
		}

		atomic_inc(&av->av_active_inserts);

		req = (struct usdf_av_req *)(insert + 1);

		for (i = 0; i < count; i++) {
			/* XXX this is temp approach, will ultimately do
			 * address resolution natively instead of thru
			 * usd_create_dest_start()
			 */
			req->avr_insert = insert;
			req->avr_fi_addr = &fi_addr[i];

			pthread_spin_lock(&udp->dom_usd_lock);
			ret = usd_create_dest_start(av->av_domain->dom_dev,
					sin->sin_addr.s_addr, sin->sin_port,
					req);
			pthread_spin_unlock(&udp->dom_usd_lock);
			if (ret != 0) {
				// XXX - no event... this will change
				fi_addr[i] = FI_ADDR_NOTAVAIL;
				atomic_inc(&insert->avi_completed_ops);
			} else {
				usdf_add_progression_item(av->av_domain);
			}

			++sin;
			++req;
		}
		ret_count = count;
	} else {
		/* XXX parallelize */
		for (i = 0; i < count; i++) {
			ret = usd_create_dest(av->av_domain->dom_dev,
					sin->sin_addr.s_addr, sin->sin_port,
					&dest);
			if (ret != 0) {
				fi_addr[i] = FI_ADDR_NOTAVAIL;
			} else {
				fi_addr[i] = (fi_addr_t)dest;
				++ret_count;
			}
			++sin;
		}
	}

	return ret_count;
}

static int
usdf_am_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			  uint64_t flags)
{
	struct usd_dest *dest;

	dest = (struct usd_dest *)(uintptr_t)fi_addr;
	usd_destroy_dest(dest);

	return 0;
}

static int
usdf_am_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			  size_t *addrlen)
{
	struct usd_dest *dest;
	struct sockaddr_in sin;
	size_t copylen;

	dest = (struct usd_dest *)(uintptr_t)fi_addr;

	if (*addrlen < sizeof(sin)) {
		copylen = *addrlen;
	} else {
		copylen = sizeof(sin);
	}

	sin.sin_family = AF_INET;
	usd_expand_dest(dest, &sin.sin_addr.s_addr, &sin.sin_port);
	memcpy(addr, &sin, copylen);

	*addrlen = sizeof(sin);
	return 0;
}

static const char *
usdf_av_straddr(struct fid_av *av, const void *addr,
				    char *buf, size_t *len)
{
	const struct sockaddr_in *sin;
	char straddr[24];
	int size;

	sin = addr;
	size = snprintf(straddr, sizeof straddr, "%s:%d",
			inet_ntoa(sin->sin_addr), sin->sin_port);
	snprintf(buf, *len, "%s", straddr);
	*len = size + 1;
	return buf;
}

static int
usdf_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct usdf_av *av;

	av = av_fidtou(fid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		if (av->av_eq != NULL) {
			return -FI_EINVAL;
		}
		av->av_eq = eq_fidtou(bfid);
		atomic_inc(&av->av_eq->eq_refcnt);
		break;
	default:
		return -FI_EINVAL;
	}

	return 0;
}

static int
usdf_av_close(struct fid *fid)
{
	struct usdf_av *av;

	av = container_of(fid, struct usdf_av, av_fid.fid);
	if (atomic_get(&av->av_refcnt) > 0) {
		return -FI_EBUSY;
	}

	pthread_spin_lock(&av->av_lock);

	if (av->av_eq != NULL) {
		atomic_dec(&av->av_eq->eq_refcnt);
	}
	atomic_dec(&av->av_domain->dom_refcnt);

	if (atomic_get(&av->av_active_inserts) > 0) {
		av->av_closing = 1;
		pthread_spin_unlock(&av->av_lock);
	} else {
		free(av);
	}
	return 0;
}

static struct fi_ops usdf_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = usdf_av_close,
	.bind = usdf_av_bind,
	.control = fi_no_control,
	.sync = fi_no_sync,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_av usdf_am_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = usdf_am_insert,
	.remove = usdf_am_remove,
	.lookup = usdf_am_lookup,
	.straddr = usdf_av_straddr
};

static struct fi_ops_av usdf_am_ops_ro = {
	.size = sizeof(struct fi_ops_av),
	.insert = fi_no_av_insert,
	.remove = fi_no_av_remove,
	.lookup = usdf_am_lookup,
	.straddr = usdf_av_straddr
};

int
usdf_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av_o, void *context)
{
	struct usdf_domain *udp;
	struct usdf_av *av;

	if (attr->name != NULL) {
		return -FI_ENOSYS;
	}
	if ((attr->flags & ~FI_EVENT) != 0) {
		return -FI_ENOSYS;
	}

	udp = dom_ftou(domain);

	av = calloc(1, sizeof(*av));
	if (av == NULL) {
		return -FI_ENOMEM;
	}

	if (attr->flags & FI_READ) {
		av->av_fid.ops = &usdf_am_ops_ro;
	} else {
		av->av_fid.ops = &usdf_am_ops;
	}
	av->av_fid.fid.fclass = FI_CLASS_AV;
	av->av_fid.fid.context = context;
	av->av_fid.fid.ops = &usdf_av_fi_ops;
	av->av_flags = attr->flags;
	pthread_spin_init(&av->av_lock, PTHREAD_PROCESS_PRIVATE);

	atomic_init(&av->av_active_inserts, 0);
	atomic_init(&av->av_refcnt, 0);
	atomic_inc(&udp->dom_refcnt);
	av->av_domain = udp;

	*av_o = av_utof(av);
	return 0;
}
