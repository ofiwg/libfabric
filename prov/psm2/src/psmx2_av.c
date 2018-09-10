/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

/*
 * SEP address query protocol:
 *
 * SEQ Query REQ:
 *	args[0].u32w0	cmd
 *	args[0].u32w1	id
 *	args[1].u64	req
 *	args[2].u64	av_idx
 *
 * SEP Query REP:
 *	args[0].u32w0	cmd
 *	args[0].u32w1	error
 *	args[1].u64	req
 *	args[2].u64	av_idx
 *	args[3].u64	n
 *	data		epaddrs
 */

struct psmx2_sep_query {
	struct psmx2_fid_av	*av;
	void 			*context;
	psm2_error_t		*errors;
	ofi_atomic32_t		error_count;
	ofi_atomic32_t		pending;
};

static int psmx2_am_sep_match(struct dlist_entry *entry, const void *arg)
{
	struct psmx2_fid_sep *sep;

	sep = container_of(entry, struct psmx2_fid_sep, entry);
	return ((uintptr_t)sep->id == (uintptr_t)arg);
}

static void psmx2_am_sep_completion(void *buf)
{
	free(buf);
}

int psmx2_am_sep_handler(psm2_am_token_t token, psm2_amarg_t *args,
			 int nargs, void *src, uint32_t len, void *hctx)
{
	struct psmx2_fid_domain *domain;
	psm2_amarg_t rep_args[8];
	int op_error = 0;
	int err = 0;
	int cmd;
	int n, i, j;
	uint8_t sep_id;
	struct psmx2_fid_sep *sep;
	struct psmx2_sep_query *req;
	struct psmx2_fid_av *av;
	psm2_epid_t *epids;
	psm2_epid_t *buf = NULL;
	int buflen;
	struct dlist_entry *entry;
	struct psmx2_trx_ctxt *trx_ctxt = hctx;

	cmd = PSMX2_AM_GET_OP(args[0].u32w0);
	domain = trx_ctxt->domain;

	switch (cmd) {
	case PSMX2_AM_REQ_SEP_QUERY:
		sep_id = args[0].u32w1;
		psmx2_lock(&domain->sep_lock, 1);
		entry = dlist_find_first_match(&domain->sep_list, psmx2_am_sep_match,
					       (void *)(uintptr_t)sep_id);
		if (!entry) {
			op_error = PSM2_EPID_UNKNOWN;
			n = 0;
			buflen = 0;
		} else {
			sep = container_of(entry, struct psmx2_fid_sep, entry);
			n = sep->ctxt_cnt;
			buflen = n * sizeof(psm2_epid_t);
			if (n) {
				buf = malloc(buflen);
				if (!buf) {
					op_error = -FI_ENOMEM;
					buflen = 0;
					n = 0;
				}
				for (i=0; i< n; i++)
					buf[i] = sep->ctxts[i].trx_ctxt->psm2_epid;
			}
		}
		psmx2_unlock(&domain->sep_lock, 1);

		rep_args[0].u32w0 = PSMX2_AM_REP_SEP_QUERY;
		rep_args[0].u32w1 = op_error;
		rep_args[1].u64 = args[1].u64;
		rep_args[2].u64 = args[2].u64;
		rep_args[3].u64 = n;
		err = psm2_am_reply_short(token, PSMX2_AM_SEP_HANDLER,
					  rep_args, 4, buf, buflen, 0,
					  psmx2_am_sep_completion, buf);
		break;

	case PSMX2_AM_REP_SEP_QUERY:
		op_error = args[0].u32w1;
		req = (void *)(uintptr_t)args[1].u64;
		av = req->av;
		i = args[2].u64;
		if (op_error) {
			ofi_atomic_inc32(&req->error_count);
			req->errors[i] = op_error;
		} else {
			n = args[3].u64;
			epids = malloc(n * sizeof(psm2_epid_t));
			if (!epids) {
				ofi_atomic_inc32(&req->error_count);
				req->errors[i] = PSM2_NO_MEMORY;
			} else {
				for (j=0; j<n; j++)
					epids[j] = ((psm2_epid_t *)src)[j];
				/*
				 * the sender of the SEP query request should
				 * have acquired the lock and is waiting for
				 * the response. see psmx2_av_connect_trx_ctxt.
				 */
				av->peers[i].sep_ctxt_cnt = n;
				av->peers[i].sep_ctxt_epids = epids;
			}
		}
		ofi_atomic_dec32(&req->pending);
		break;

	default:
		err = -FI_EINVAL;
		break;
	}

	return err;
}

#define PSMX2_MIN_CONN_TIMEOUT	5
#define PSMX2_MAX_CONN_TIMEOUT	30

static inline double psmx2_conn_timeout(int sec)
{
	if (sec < PSMX2_MIN_CONN_TIMEOUT)
		return PSMX2_MIN_CONN_TIMEOUT * 1e9;

	if (sec > PSMX2_MAX_CONN_TIMEOUT)
		return PSMX2_MAX_CONN_TIMEOUT * 1e9;

	return sec * 1e9;
}

static void psmx2_set_epaddr_context(struct psmx2_trx_ctxt *trx_ctxt,
				     psm2_epid_t epid, psm2_epaddr_t epaddr)
{
	struct psmx2_epaddr_context *context;

	context = (void *)psm2_epaddr_getctxt(epaddr);
	if (context) {
		if (context->trx_ctxt != trx_ctxt || context->epid != epid) {
			FI_WARN(&psmx2_prov, FI_LOG_AV,
				"trx_ctxt or epid doesn't match\n");
			context = NULL;
		}
	}

	if (context)
		return;

	context = malloc(sizeof *context);
	if (!context) {
		FI_WARN(&psmx2_prov, FI_LOG_AV,
			"cannot allocate context\n");
		return;
	}

	context->trx_ctxt = trx_ctxt;
	context->epid = epid;
	context->epaddr = epaddr;
	psm2_epaddr_setctxt(epaddr, context);

	psmx2_lock(&trx_ctxt->peer_lock, 2);
	dlist_insert_before(&context->entry, &trx_ctxt->peer_list);
	psmx2_unlock(&trx_ctxt->peer_lock, 2);
}

int psmx2_epid_to_epaddr(struct psmx2_trx_ctxt *trx_ctxt,
			 psm2_epid_t epid, psm2_epaddr_t *epaddr)
{
	int err;
	psm2_error_t errors;
	psm2_epconn_t epconn;
	struct psmx2_epaddr_context *context;

	err = psm2_ep_epid_lookup2(trx_ctxt->psm2_ep, epid, &epconn);
	if (err == PSM2_OK) {
		context = psm2_epaddr_getctxt(epconn.addr);
		if (context && context->epid  == epid) {
			*epaddr = epconn.addr;
			return 0;
		}
	}

	err = psm2_ep_connect(trx_ctxt->psm2_ep, 1, &epid, NULL, &errors,
			      epaddr, psmx2_conn_timeout(1));
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_AV,
			"psm2_ep_connect retured error %s, remote epid=%lx.\n",
			psm2_error_get_string(err), epid);
		return psmx2_errno(err);
	}

	psmx2_set_epaddr_context(trx_ctxt,epid,*epaddr);

	return 0;
}

/*
 * Must be called with av->lock held
 */
static int psmx2_av_check_space(struct psmx2_fid_av *av, size_t count)
{
	psm2_epid_t *new_epids;
	psm2_epaddr_t *new_epaddrs;
	psm2_epaddr_t **new_sepaddrs;
	struct psmx2_av_peer *new_peers;
	size_t new_count;
	int i;

	new_count = av->count;
	while (new_count < av->last + count)
		new_count = new_count * 2 + 1;

	if ((new_count <= av->count) && av->epids)
		return 0;

	new_epids = realloc(av->epids, new_count * sizeof(*new_epids));
	if (!new_epids)
		return -FI_ENOMEM;
	av->epids = new_epids;

	new_peers = realloc(av->peers, new_count * sizeof(*new_peers));
	if (!new_peers)
		return -FI_ENOMEM;
	av->peers = new_peers;

	for (i = 0; i < av->max_trx_ctxt; i++) {
		if (!av->tables[i].trx_ctxt)
			continue;

		new_epaddrs = realloc(av->tables[i].epaddrs,
				      new_count * sizeof(*new_epaddrs));
		if (!new_epaddrs)
			return -FI_ENOMEM;
		memset(new_epaddrs + av->last, 0,
		       (new_count - av->last)  * sizeof(*new_epaddrs));
		av->tables[i].epaddrs = new_epaddrs;

		new_sepaddrs = realloc(av->tables[i].sepaddrs,
				       new_count * sizeof(*new_sepaddrs));
		if (!new_sepaddrs)
			return -FI_ENOMEM;
		memset(new_sepaddrs + av->last, 0,
		       (new_count - av->last)  * sizeof(*new_sepaddrs));
		av->tables[i].sepaddrs = new_sepaddrs;
	}

	av->count = new_count;
	return 0;
}

static void psmx2_av_post_completion(struct psmx2_fid_av *av, void *context,
				     uint64_t data, int prov_errno)
{
	if (prov_errno) {
		struct fi_eq_err_entry entry;
		entry.fid = &av->av.fid;
		entry.context = context;
		entry.data = data;
		entry.err = -psmx2_errno(prov_errno);
		entry.prov_errno = prov_errno;
		entry.err_data = NULL;
		entry.err_data_size = 0;
		fi_eq_write(av->eq, FI_AV_COMPLETE, &entry, sizeof(entry),
			    UTIL_FLAG_ERROR);
	} else {
		struct fi_eq_entry entry;
		entry.fid = &av->av.fid;
		entry.context = context;
		entry.data = data;
		fi_eq_write(av->eq, FI_AV_COMPLETE, &entry, sizeof(entry), 0);
	}
}

/*
 * Must be called with av->lock held
 */
static int psmx2_av_connect_trx_ctxt(struct psmx2_fid_av *av,
				     int trx_ctxt_id,
				     size_t av_idx_start,
				     size_t count,
				     psm2_error_t *errors)
{
	struct psmx2_trx_ctxt *trx_ctxt;
	struct psmx2_sep_query *req;
	struct psmx2_av_peer *peers;
	struct psmx2_epaddr_context *epaddr_context;
	psm2_epconn_t epconn;
	psm2_ep_t ep;
	psm2_epid_t *epids;
	psm2_epaddr_t *epaddrs;
	psm2_epaddr_t **sepaddrs;
	psm2_amarg_t args[3];
	int *mask;
	int error_count = 0;
	int to_connect = 0;
	int sep_count = 0;
	int i;

	trx_ctxt = av->tables[trx_ctxt_id].trx_ctxt;
	ep = trx_ctxt->psm2_ep;
	epids = av->epids + av_idx_start;
	epaddrs = av->tables[trx_ctxt_id].epaddrs + av_idx_start;
	sepaddrs = av->tables[trx_ctxt_id].sepaddrs + av_idx_start;
	peers = av->peers + av_idx_start;

	/* set up mask to avoid duplicated connection */

	mask = calloc(count, sizeof(*mask));
	if (!mask) {
		for (i = 0; i < count; i++)
			errors[i] = PSM2_NO_MEMORY;
		error_count += count;
		return error_count;
	}

	for (i = 0; i < count; i++) {
		errors[i] = PSM2_OK;

		if (psm2_ep_epid_lookup2(ep, epids[i], &epconn) == PSM2_OK) {
			epaddr_context = psm2_epaddr_getctxt(epconn.addr);
			if (epaddr_context && epaddr_context->epid == epids[i])
				epaddrs[i] = epconn.addr;
			else
				mask[i] = 1;
		} else {
			mask[i] = 1;
		}

		if (peers[i].type == PSMX2_EP_SCALABLE)
			sep_count++;

		if (mask[i]) {
			if (peers[i].type == PSMX2_EP_SCALABLE) {
				if (peers[i].sep_ctxt_epids)
					mask[i] = 0;
				 else
					to_connect++;
			} else if (psmx2_env.lazy_conn) {
				epaddrs[i] = NULL;
				mask[i] = 0;
			} else {
				to_connect++;
			}
		}
	}

	if (to_connect)
		psm2_ep_connect(ep, count, epids, mask, errors, epaddrs,
				psmx2_conn_timeout(count));

	/* check the connection results */

	for (i = 0; i < count; i++) {
		if (!mask[i]) {
			errors[i] = PSM2_OK;
			continue;
		}

		if (errors[i] == PSM2_OK ||
		    errors[i] == PSM2_EPID_ALREADY_CONNECTED) {
			psmx2_set_epaddr_context(trx_ctxt, epids[i], epaddrs[i]);
			errors[i] = PSM2_OK;
		} else {
			/* If duplicated addrs are passed to psm2_ep_connect(),
			 * all but one will fail with error "Endpoint could not
			 * be reached". This should be treated the same as
			 * "Endpoint already connected".
			 */
			if (psm2_ep_epid_lookup2(ep, epids[i], &epconn) == PSM2_OK) {
				epaddr_context = psm2_epaddr_getctxt(epconn.addr);
				if (epaddr_context &&
				    epaddr_context->epid == epids[i]) {
					epaddrs[i] = epconn.addr;
					errors[i] = PSM2_OK;
					continue;
				}
			}

			FI_WARN(&psmx2_prov, FI_LOG_AV,
				"%d: psm2_ep_connect (%lx --> %lx): %s\n",
				i, trx_ctxt->psm2_epid, epids[i],
				psm2_error_get_string(errors[i]));
			epaddrs[i] = NULL;
			error_count++;
		}
	}

	free(mask);

	if (sep_count) {

		/* query SEP information */

		psmx2_am_init(trx_ctxt); /* check AM handler installation */

		req = malloc(sizeof *req);
		if (req) {
			req->av = av;
			req->errors = errors;
			ofi_atomic_initialize32(&req->error_count, 0);
			ofi_atomic_initialize32(&req->pending, 0);
		}

		for (i = 0; i < count; i++) {
			if (peers[i].type != PSMX2_EP_SCALABLE ||
			    peers[i].sep_ctxt_epids ||
			    errors[i] != PSM2_OK)
				continue;

			if (!req) {
				errors[i] = PSM2_NO_MEMORY;
				error_count++;
				continue;
			}

			ofi_atomic_inc32(&req->pending);
			args[0].u32w0 = PSMX2_AM_REQ_SEP_QUERY;
			args[0].u32w1 = peers[i].sep_id;
			args[1].u64 = (uint64_t)(uintptr_t)req;
			args[2].u64 = av_idx_start + i;
			psm2_am_request_short(epaddrs[i], PSMX2_AM_SEP_HANDLER,
					      args, 3, NULL, 0, 0, NULL, NULL);
		}

		/*
		 * make it synchronous for now to:
		 * (1) ensure the array "req->errors" is valid;
		 * (2) simplify the logic of generating the final completion.
		 */

		if (req) {
			/*
			 * make sure AM is progressed promptly. don't call
			 * psmx2_progress() which may call functions that
			 * need to access the address vector.
			 */
			while (ofi_atomic_get32(&req->pending))
				psm2_poll(trx_ctxt->psm2_ep);

			error_count += ofi_atomic_get32(&req->error_count);
			free(req);
		}
	}

	/* alloate context specific epaddrs for SEP */

	for (i = 0; i < count; i++) {
		if (peers[i].type == PSMX2_EP_SCALABLE &&
		    peers[i].sep_ctxt_epids && !sepaddrs[i])
			sepaddrs[i] = calloc(peers[i].sep_ctxt_cnt,
					     sizeof(*sepaddrs[i]));
	}

	return error_count;
}

int psmx2_av_add_trx_ctxt(struct psmx2_fid_av *av,
			  struct psmx2_trx_ctxt *trx_ctxt,
			  int connect_now)
{
	psm2_error_t *errors;
	int id = trx_ctxt->id;
	int err = 0;

	psmx2_lock(&av->lock, 1);

	if (id >= av->max_trx_ctxt) {
		FI_WARN(&psmx2_prov, FI_LOG_AV,
			"trx_ctxt->id(%d) exceeds av->max_trx_ctxt(%d).\n",
			id, av->max_trx_ctxt);
		err = -FI_EINVAL;
		goto out;
	}

	if (av->tables[id].trx_ctxt) {
		if (av->tables[id].trx_ctxt == trx_ctxt) {
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"trx_ctxt(%p) with id(%d) already added.\n",
				trx_ctxt, id);
			goto out;
		} else {
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"different trx_ctxt(%p) with same id(%d) already added.\n",
				trx_ctxt, id);
			err = -FI_EINVAL;
			goto out;
		}
	}

	av->tables[id].epaddrs = (psm2_epaddr_t *) calloc(av->count,
							  sizeof(psm2_epaddr_t));
	if (!av->tables[id].epaddrs) {
		err = -FI_ENOMEM;
		goto out;
	}

	av->tables[id].sepaddrs = (psm2_epaddr_t **)calloc(av->count,
							   sizeof(psm2_epaddr_t *));
	if (!av->tables[id].sepaddrs) {
		err = -FI_ENOMEM;
		goto out;
	}

	av->tables[id].trx_ctxt = trx_ctxt;

	if (connect_now) {
		errors = calloc(av->count, sizeof(*errors));
		if (errors) {
			psmx2_av_connect_trx_ctxt(av, id, 0, av->last, errors);
			free(errors);
		}
	}

out:
	psmx2_unlock(&av->lock, 1);
	return err;
}

static int psmx2_av_insert(struct fid_av *av, const void *addr,
			   size_t count, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	struct psmx2_fid_av *av_priv;
	struct psmx2_ep_name *ep_name;
	const struct psmx2_ep_name *names = addr;
	const char **string_names = (void *)addr;
	psm2_error_t *errors = NULL;
	int error_count = 0;
	int i, idx, ret;
	int sep_count = 0;

	assert(addr || !count);

	av_priv = container_of(av, struct psmx2_fid_av, av);

	psmx2_lock(&av_priv->lock, 1);

	if ((av_priv->flags & FI_EVENT) && !av_priv->eq) {
		ret = -FI_ENOEQ;
		goto out;
	}

	if (psmx2_av_check_space(av_priv, count)) {
		ret = -FI_ENOMEM;
		goto out;
	}

	errors = calloc(count, sizeof(*errors));
	if (!errors) {
		ret = -FI_ENOMEM;
		goto out;
	}

	/* save the peer address information */
	for (i = 0; i < count; i++) {
		idx = av_priv->last + i;
		if (av_priv->addr_format == FI_ADDR_STR) {
			ep_name = psmx2_string_to_ep_name(string_names[i]);
			if (!ep_name) {
				ret = -FI_EINVAL;
				goto out;
			}
			av_priv->epids[idx] = ep_name->epid;
			av_priv->peers[idx].type = ep_name->type;
			av_priv->peers[idx].sep_id = ep_name->sep_id;
			free(ep_name);
		} else {
			av_priv->epids[idx] = names[i].epid;
			av_priv->peers[idx].type = names[i].type;
			av_priv->peers[idx].sep_id = names[i].sep_id;
		}
		av_priv->peers[idx].sep_ctxt_cnt = 1;
		av_priv->peers[idx].sep_ctxt_epids = NULL;
		if (av_priv->peers[idx].type == PSMX2_EP_SCALABLE)
			sep_count++;
	}

	/*
	 * try to establish connection when:
	 *  (1) there are Tx/Rx context(s) bound to the AV; and
	 *  (2) the connection is desired right now
	 */
	if (sep_count || !psmx2_env.lazy_conn) {
		for (i = 0; i < av_priv->max_trx_ctxt; i++) {
			if (!av_priv->tables[i].trx_ctxt)
				continue;

			error_count = psmx2_av_connect_trx_ctxt(av_priv, i,
								av_priv->last,
								count, errors);

			if (error_count || psmx2_env.lazy_conn)
				break;
		}
	}

	if (fi_addr) {
		for (i = 0; i < count; i++) {
			idx = av_priv->last + i;
			if (errors[i] != PSM2_OK)
				fi_addr[i] = FI_ADDR_NOTAVAIL;
			else if (av_priv->peers[idx].type == PSMX2_EP_SCALABLE)
				fi_addr[i] = idx | PSMX2_SEP_ADDR_FLAG;
			else if (av_priv->type == FI_AV_TABLE)
				fi_addr[i] = idx;
			else
				fi_addr[i] = PSMX2_EP_TO_ADDR(av_priv->tables[0].epaddrs[idx]);
		}
	}

	av_priv->last += count;

	if (av_priv->flags & FI_EVENT) {
		if (error_count) {
			for (i = 0; i < count; i++)
				psmx2_av_post_completion(av_priv, context, i, errors[i]);
		}
		psmx2_av_post_completion(av_priv, context, count - error_count, 0);
		ret = 0;
	} else {
		if (flags & FI_SYNC_ERR) {
			int *fi_errors = context;
			for (i=0; i<count; i++)
				fi_errors[i] = psmx2_errno(errors[i]);
		}
		ret = count - error_count;
	}

out:
	free(errors);
	psmx2_unlock(&av_priv->lock, 1);
	return ret;
}

static int psmx2_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			   uint64_t flags)
{
	return 0;
}

static int psmx2_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			   size_t *addrlen)
{
	struct psmx2_fid_av *av_priv;
	struct psmx2_epaddr_context *context;
	struct psmx2_ep_name name;
	int idx;
	int err = 0;

	assert(addr);
	assert(addrlen);

	av_priv = container_of(av, struct psmx2_fid_av, av);

	memset(&name, 0, sizeof(name));

	psmx2_lock(&av_priv->lock, 1);

	if (PSMX2_SEP_ADDR_TEST(fi_addr)) {
		idx = PSMX2_SEP_ADDR_IDX(fi_addr);
		if (idx >= av_priv->last) {
			err = -FI_EINVAL;
			goto out;
		}
		name.type = PSMX2_EP_SCALABLE;
		name.epid = av_priv->epids[idx];
		name.sep_id = av_priv->peers[idx].sep_id;
	} else if (av_priv->type == FI_AV_TABLE) {
		idx = (int)(int64_t)fi_addr;
		if (idx >= av_priv->last) {
			err = -FI_EINVAL;
			goto out;
		}
		name.type = PSMX2_EP_REGULAR;
		name.epid = av_priv->epids[idx];
	} else {
		context = psm2_epaddr_getctxt(PSMX2_ADDR_TO_EP(fi_addr));
		name.type = PSMX2_EP_REGULAR;
		name.epid = context->epid;
	}

	if (av_priv->addr_format == FI_ADDR_STR) {
		ofi_straddr(addr, addrlen, FI_ADDR_PSMX2, &name);
	} else {
		memcpy(addr, &name, MIN(*addrlen, sizeof(name)));
		*addrlen = sizeof(name);
	}

out:
	psmx2_unlock(&av_priv->lock, 1);
	return err;
}

psm2_epaddr_t psmx2_av_translate_sep(struct psmx2_fid_av *av,
				     struct psmx2_trx_ctxt *trx_ctxt,
				     fi_addr_t addr)
{
	int idx = PSMX2_SEP_ADDR_IDX(addr);
	int ctxt = PSMX2_SEP_ADDR_CTXT(addr, av->rx_ctx_bits);
	psm2_epaddr_t epaddr = NULL;
	psm2_error_t errors;
	int err;

	psmx2_lock(&av->lock, 1);

	if (av->peers[idx].type != PSMX2_EP_SCALABLE ||
	    ctxt >= av->peers[idx].sep_ctxt_cnt)
		goto out;

	/* this can be NULL when lazy connection is enabled */
	if (!av->tables[trx_ctxt->id].sepaddrs[idx]) {
		psmx2_av_connect_trx_ctxt(av, trx_ctxt->id, idx, 1, &errors);
		assert(av->tables[trx_ctxt->id].sepaddrs[idx]);
	}

	if (!av->tables[trx_ctxt->id].sepaddrs[idx][ctxt]) {
		err = psmx2_epid_to_epaddr(trx_ctxt,
					   av->peers[idx].sep_ctxt_epids[ctxt],
					   &epaddr);
		if (err) {
			FI_WARN(&psmx2_prov, FI_LOG_AV,
				"fatal error: unable to translate epid %lx to epaddr.\n",
				av->peers[idx].sep_ctxt_epids[ctxt]);
			goto out;
		}

		av->tables[trx_ctxt->id].sepaddrs[idx][ctxt] = epaddr;
	}

	epaddr = av->tables[trx_ctxt->id].sepaddrs[idx][ctxt];

out:
	psmx2_unlock(&av->lock, 1);
	return epaddr;
}

fi_addr_t psmx2_av_translate_source(struct psmx2_fid_av *av, fi_addr_t source)
{
	psm2_epaddr_t epaddr;
	psm2_epid_t epid;
	fi_addr_t ret = FI_ADDR_NOTAVAIL;
	int i, j, found = 0;

	epaddr = PSMX2_ADDR_TO_EP(source);
	psm2_epaddr_to_epid(epaddr, &epid);

	psmx2_lock(&av->lock, 1);

	for (i = av->last - 1; i >= 0 && !found; i--) {
		if (av->peers[i].type == PSMX2_EP_REGULAR) {
			if (av->epids[i] == epid) {
				ret = (av->type == FI_AV_MAP) ?
				      source : (fi_addr_t)i;
				found = 1;
			}
		} else {
			for (j=0; j<av->peers[i].sep_ctxt_cnt; j++) {
				if (av->peers[i].sep_ctxt_epids[j] == epid) {
					ret = fi_rx_addr((fi_addr_t)i, j,
							 av->rx_ctx_bits);
					found = 1;
					break;
				}
			}
		}
	}

	psmx2_unlock(&av->lock, 1);
	return ret;
}

void psmx2_av_remove_conn(struct psmx2_fid_av *av,
			  struct psmx2_trx_ctxt *trx_ctxt,
			  psm2_epaddr_t epaddr)
{
	psm2_epid_t epid;
	int i, j;

	psm2_epaddr_to_epid(epaddr, &epid);

	psmx2_lock(&av->lock, 1);

	for (i = 0; i < av->last; i++) {
		if (av->peers[i].type == PSMX2_EP_REGULAR) {
			if (av->epids[i] == epid &&
			    av->tables[trx_ctxt->id].epaddrs[i] == epaddr)
				av->tables[trx_ctxt->id].epaddrs[i] = NULL;
		} else {
			for (j=0; j<av->peers[i].sep_ctxt_cnt; j++) {
				if (av->peers[i].sep_ctxt_epids[j] == epid &&
				    av->tables[trx_ctxt->id].sepaddrs[i] &&
				    av->tables[trx_ctxt->id].sepaddrs[i][j] == epaddr)
					    av->tables[trx_ctxt->id].sepaddrs[i][j] = NULL;
			}
		}
	}

	psmx2_unlock(&av->lock, 1);
}

static const char *psmx2_av_straddr(struct fid_av *av, const void *addr,
				    char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_PSMX2, addr);
}

static int psmx2_av_close(fid_t fid)
{
	struct psmx2_fid_av *av;
	int i, j;

	av = container_of(fid, struct psmx2_fid_av, av.fid);
	psmx2_domain_release(av->domain);
	fastlock_destroy(&av->lock);
	for (i = 0; i < av->max_trx_ctxt; i++) {
		if (!av->tables[i].trx_ctxt)
			continue;
		free(av->tables[i].epaddrs);
		if (av->tables[i].sepaddrs) {
			for (j = 0; j < av->last; j++)
				free(av->tables[i].sepaddrs[j]);
		}
		free(av->tables[i].sepaddrs);
	}
	free(av->peers);
	free(av->epids);
	free(av);
	return 0;
}

static int psmx2_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct psmx2_fid_av *av;

	av = container_of(fid, struct psmx2_fid_av, av.fid);

	assert(bfid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		av->eq = (struct fid_eq *)bfid;
		break;

	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static struct fi_ops psmx2_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx2_av_close,
	.bind = psmx2_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_av psmx2_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = psmx2_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = psmx2_av_remove,
	.lookup = psmx2_av_lookup,
	.straddr = psmx2_av_straddr,
};

int psmx2_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		  struct fid_av **av, void *context)
{
	struct psmx2_fid_domain *domain_priv;
	struct psmx2_fid_av *av_priv;
	int type;
	size_t count = 64;
	uint64_t flags = 0;
	int rx_ctx_bits = PSMX2_MAX_RX_CTX_BITS;
	size_t table_size;

	domain_priv = container_of(domain, struct psmx2_fid_domain,
				   util_domain.domain_fid);

	if (psmx2_env.lazy_conn || psmx2_env.max_trx_ctxt > 1)
		type = FI_AV_TABLE;
	else
		type = FI_AV_MAP;

	if (attr) {
		switch (attr->type) {
		case FI_AV_UNSPEC:
			break;

		case FI_AV_MAP:
			if (psmx2_env.lazy_conn) {
				FI_INFO(&psmx2_prov, FI_LOG_AV,
					"Lazy connection is enabled, force FI_AV_TABLE\n");
				break;
			}
			if (psmx2_env.max_trx_ctxt > 1) {
				FI_INFO(&psmx2_prov, FI_LOG_AV,
					"Multi-EP is enabled, force FI_AV_TABLE\n");
				break;
			}
			/* fall through */
		case FI_AV_TABLE:
			type = attr->type;
			break;
		default:
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"attr->type=%d, supported=%d %d\n",
				attr->type, FI_AV_MAP, FI_AV_TABLE);
			return -FI_EINVAL;
		}

		count = attr->count;
		flags = attr->flags;

		if (flags & (FI_READ | FI_SYMMETRIC)) {
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"attr->flags=%"PRIu64", supported=%llu\n",
				attr->flags, FI_EVENT);
			return -FI_ENOSYS;
		}

		if (attr->name) {
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"attr->name=%s, named AV is not supported\n",
				attr->name);
			return -FI_ENOSYS;
		}

		if (attr->rx_ctx_bits > PSMX2_MAX_RX_CTX_BITS) {
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"attr->rx_ctx_bits=%d, maximum allowed is %d\n",
				attr->rx_ctx_bits, PSMX2_MAX_RX_CTX_BITS);
			return -FI_ENOSYS;
		}

		rx_ctx_bits = attr->rx_ctx_bits;
	}

	table_size = psmx2_env.max_trx_ctxt * sizeof(struct psmx2_av_table);
	av_priv = (struct psmx2_fid_av *) calloc(1, sizeof(*av_priv) + table_size);
	if (!av_priv)
		return -FI_ENOMEM;

	fastlock_init(&av_priv->lock);

	psmx2_domain_acquire(domain_priv);

	av_priv->domain = domain_priv;
	av_priv->type = type;
	av_priv->addrlen = sizeof(psm2_epaddr_t);
	av_priv->count = count;
	av_priv->flags = flags;
	av_priv->rx_ctx_bits = rx_ctx_bits;
	av_priv->max_trx_ctxt = psmx2_env.max_trx_ctxt;
	av_priv->addr_format = domain_priv->addr_format;

	av_priv->av.fid.fclass = FI_CLASS_AV;
	av_priv->av.fid.context = context;
	av_priv->av.fid.ops = &psmx2_fi_ops;
	av_priv->av.ops = &psmx2_av_ops;

	*av = &av_priv->av;
	if (attr)
		attr->type = type;

	FI_INFO(&psmx2_prov, FI_LOG_AV,
		"type = %s\n", fi_tostr(&type, FI_TYPE_AV_TYPE));

	return 0;
}

