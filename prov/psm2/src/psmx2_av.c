/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

static void psmx2_av_post_completion(struct psmx2_fid_av *av, void *context,
				     uint64_t data, int prov_errno);

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
			 int nargs, void *src, uint32_t len)
{
	struct psmx2_fid_domain *domain;
	psm2_amarg_t rep_args[8];
	int op_error = 0;
	int err = 0;
	int cmd;
	int n, i, j;
	uint8_t sep_id;
	struct psmx2_fid_sep *sep;
	struct psmx2_sep_addr *p;
	struct psmx2_sep_query *req;
	struct psmx2_fid_av *av;
	psm2_epid_t *buf = NULL;
	int buflen;
	struct dlist_entry *entry;

	cmd = PSMX2_AM_GET_OP(args[0].u32w0);
	domain = psmx2_active_fabric->active_domain;

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
			if (av->flags & FI_EVENT)
				psmx2_av_post_completion(av, req->context, i, op_error);
		} else {
			n = args[3].u64;
			p  = calloc(1, sizeof (struct psmx2_sep_addr) +
				       n * sizeof(struct psmx2_ctxt_addr));
			if (!p) {
				ofi_atomic_inc32(&req->error_count);
				req->errors[i] = PSM2_NO_MEMORY;
			} else {
				p->ctxt_cnt = n;
				for (j=0; j<n; j++) {
					p->ctxt_addrs[j].epid = ((psm2_epid_t *)src)[j];
					p->ctxt_addrs[j].epaddrs =
						calloc(psmx2_env.max_trx_ctxt,
						       sizeof(psm2_epaddr_t));
					if (!p->ctxt_addrs[j].epaddrs) {
						ofi_atomic_inc32(&req->error_count);
						req->errors[i] = PSM2_NO_MEMORY;
						break;
					}
				}
				av->sepaddrs[i] = p;
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
	psm2_epaddr_setctxt(epaddr, context);
}

int psmx2_epid_to_epaddr(struct psmx2_trx_ctxt *trx_ctxt,
			 psm2_epid_t epid, psm2_epaddr_t *epaddr)
{
        int err;
        psm2_error_t errors;
	psm2_epconn_t epconn;
	struct psmx2_epaddr_context *context;

	err = psmx2_ep_epid_lookup(trx_ctxt->psm2_ep, epid, &epconn);
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

psm2_epaddr_t psmx2_av_translate_sep(struct psmx2_fid_av *av,
				     struct psmx2_trx_ctxt *trx_ctxt,
				     fi_addr_t addr)
{
	int idx = PSMX2_SEP_ADDR_IDX(addr);
	int ctxt = PSMX2_SEP_ADDR_CTXT(addr, av->rx_ctx_bits);
	psm2_epaddr_t epaddr;
	int err;

	if (!av->sepaddrs[idx])
		return NULL;

	if (ctxt >= av->sepaddrs[idx]->ctxt_cnt)
		return NULL;

	if (!av->sepaddrs[idx]->ctxt_addrs[ctxt].epaddrs[trx_ctxt->id]) {
		err = psmx2_epid_to_epaddr(trx_ctxt,
					   av->sepaddrs[idx]->ctxt_addrs[ctxt].epid,
					   &epaddr);
		if (err) {
			FI_WARN(&psmx2_prov, FI_LOG_AV,
				"fatal error: unable to translate epid %lx to epaddr.\n",
				av->sepaddrs[idx]->ctxt_addrs[ctxt].epid);
			return NULL;
		}

		av->sepaddrs[idx]->ctxt_addrs[ctxt].epaddrs[trx_ctxt->id] = epaddr;
	}

	return av->sepaddrs[idx]->ctxt_addrs[ctxt].epaddrs[trx_ctxt->id];
}

static int psmx2_av_check_table_size(struct psmx2_fid_av *av, size_t count)
{
	size_t new_count;
	psm2_epid_t *new_epids;
	psm2_epaddr_t *new_epaddrs;
	uint8_t *new_vlanes;
	uint8_t *new_types;
	struct psmx2_sep_addr **new_sepaddrs;

	new_count = av->count;
	while (new_count < av->last + count)
		new_count = new_count * 2 + 1;

	if ((new_count <= av->count) && av->epids)
		return 0;

	new_epids = realloc(av->epids, new_count * sizeof(*new_epids));
	if (!new_epids)
		return -FI_ENOMEM;

	av->epids = new_epids;
	new_epaddrs = realloc(av->epaddrs, new_count * sizeof(*new_epaddrs));
	if (!new_epaddrs)
		return -FI_ENOMEM;

	av->epaddrs = new_epaddrs;
	new_vlanes = realloc(av->vlanes, new_count * sizeof(*new_vlanes));
	if (!new_vlanes)
		return -FI_ENOMEM;

	av->vlanes = new_vlanes;
	new_types = realloc(av->types, new_count * sizeof(*new_types));
	if (!new_types)
		return -FI_ENOMEM;

	av->types = new_types;

	new_sepaddrs = realloc(av->sepaddrs, new_count * sizeof(*new_sepaddrs));
	if (!new_sepaddrs)
		return -FI_ENOMEM;

	av->sepaddrs = new_sepaddrs;

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

static int psmx2_av_connect_eps(struct psmx2_fid_av *av, size_t count,
			        psm2_epid_t *epids, int *mask,
			        psm2_error_t *errors,
			        psm2_epaddr_t *epaddrs,
			        void *context)
{
	int i;
	psm2_epconn_t epconn;
	struct psmx2_epaddr_context *epaddr_context;
	int error_count = 0;
	psm2_ep_t ep = av->domain->base_trx_ctxt->psm2_ep;

	/* set up mask to prevent connecting to an already connected ep */
	for (i=0; i<count; i++) {
		if (psmx2_ep_epid_lookup(ep, epids[i], &epconn) == PSM2_OK) {
			epaddr_context = psm2_epaddr_getctxt(epconn.addr);
			if (epaddr_context && epaddr_context->epid == epids[i])
				epaddrs[i] = epconn.addr;
			else
				mask[i] = 1;
		} else {
			mask[i] = 1;
		}
	}

	psm2_ep_connect(ep, count, epids, mask, errors, epaddrs, psmx2_conn_timeout(count));

	for (i=0; i<count; i++){
		if (!mask[i]) {
			errors[i] = PSM2_OK;
			continue;
		}

		if (errors[i] == PSM2_OK ||
		    errors[i] == PSM2_EPID_ALREADY_CONNECTED) {
			psmx2_set_epaddr_context(av->domain->base_trx_ctxt, epids[i], epaddrs[i]);
			errors[i] = PSM2_OK;
		} else {
			/* If duplicated addrs are passed to psm2_ep_connect(),
			 * all but one will fail with error "Endpoint could not
			 * be reached". This should be treated the same as
			 * "Endpoint already connected".
			 */
			if (psmx2_ep_epid_lookup(ep, epids[i], &epconn) == PSM2_OK) {
				epaddr_context = psm2_epaddr_getctxt(epconn.addr);
				if (epaddr_context &&
				    epaddr_context->epid == epids[i]) {
					epaddrs[i] = epconn.addr;
					errors[i] = PSM2_OK;
					continue;
				}
			}

			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"%d: psm2_ep_connect returned %s. remote epid=%lx.\n",
				i, psm2_error_get_string(errors[i]), epids[i]);
			if (epids[i] == 0)
				FI_INFO(&psmx2_prov, FI_LOG_AV,
					"does the application depend on the provider"
					"to resolve IP address into endpoint id? if so"
					"check if the name server has started correctly"
					"at the other side.\n");
			epaddrs[i] = (void *)FI_ADDR_NOTAVAIL;
			error_count++;

			if (av->flags & FI_EVENT)
				psmx2_av_post_completion(av, context, i, errors[i]);
		}
	}

	return error_count;
}

static int psmx2_av_query_seps(struct psmx2_fid_av *av, size_t count, psm2_epid_t *epids,
			       uint8_t *sep_ids, uint8_t *types, psm2_error_t *errors,
			       psm2_epaddr_t *epaddrs, void *context)
{
	struct psmx2_sep_query *req;
	psm2_amarg_t args[8];
	int error_count = 0;
	int i;

	req = malloc(sizeof *req);

	if (req) {
		req->av = av;
		req->context = context;
		req->errors = errors;
		ofi_atomic_initialize32(&req->error_count, 0);
		ofi_atomic_initialize32(&req->pending, 0);
	}

	for (i=0; i<count; i++) {
		if (types[i] != PSMX2_EP_SCALABLE)
			continue;

		if (errors[i] != PSM2_OK)
			continue;

		if (!req) {
			errors[i] = PSM2_NO_MEMORY;
			error_count++;
			continue;
		}

		ofi_atomic_inc32(&req->pending);
		args[0].u32w0 = PSMX2_AM_REQ_SEP_QUERY;
		args[0].u32w1 = sep_ids[i];
		args[1].u64 = (uint64_t)(uintptr_t)req;
		args[2].u64 = av->last + i;
		psm2_am_request_short(epaddrs[i], PSMX2_AM_SEP_HANDLER,
				      args, 3, NULL, 0, 0, NULL, NULL);
	}

	/*
	 * make it synchronous for now to:
	 * (1) ensure array "req->errors" is valid;
	 * (2) to simplify the logic of generating the final completion event.
	 */

	if (req) {
		while (ofi_atomic_get32(&req->pending))
			psmx2_progress_all(av->domain);
		error_count = ofi_atomic_get32(&req->error_count);
		free(req);
	}

	return error_count;
}

static int psmx2_av_insert(struct fid_av *av, const void *addr,
			   size_t count, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	struct psmx2_fid_av *av_priv;
	psm2_epid_t *epids;
	uint8_t *vlanes;
	uint8_t *types;
	struct psmx2_sep_addr **sepaddrs;
	psm2_epaddr_t *epaddrs;
	psm2_error_t *errors;
	int *mask;
	struct psmx2_ep_name *ep_name;
	const struct psmx2_ep_name *names = addr;
	const char **string_names = (void *)addr;
	int error_count;
	int i, ret;

	if (count && !addr) {
		FI_INFO(&psmx2_prov, FI_LOG_AV,
			"the input address array is NULL.\n");
		return -FI_EINVAL;
	}

	av_priv = container_of(av, struct psmx2_fid_av, av);

	if ((av_priv->flags & FI_EVENT) && !av_priv->eq)
		return -FI_ENOEQ;

	if (psmx2_av_check_table_size(av_priv, count))
		return -FI_ENOMEM;

	epids = av_priv->epids + av_priv->last;
	epaddrs = av_priv->epaddrs + av_priv->last;
	vlanes = av_priv->vlanes + av_priv->last;
	types = av_priv->types + av_priv->last;
	sepaddrs = av_priv->sepaddrs + av_priv->last;

	for (i=0; i<count; i++) {
		if (av_priv->addr_format == FI_ADDR_STR) {
			ep_name = psmx2_string_to_ep_name(string_names[i]);
			if (!ep_name)
				return -FI_EINVAL;
			epids[i] = ep_name->epid;
			vlanes[i] = ep_name->vlane;
			types[i] = ep_name->type;
			free(ep_name);
		} else {
			epids[i] = names[i].epid;
			vlanes[i] = names[i].vlane;
			types[i] = names[i].type;
		}
		sepaddrs[i] = NULL;
	}

	errors = (psm2_error_t *) calloc(count, sizeof *errors);
	mask = (int *) calloc(count, sizeof *mask);
	if (!errors || !mask) {
		free(mask);
		free(errors);
		return -FI_ENOMEM;
	}

	error_count = psmx2_av_connect_eps(av_priv, count, epids, mask,
					   errors, epaddrs, context);

	error_count += psmx2_av_query_seps(av_priv, count, epids, vlanes, types,
					   errors, epaddrs, context);

	if (fi_addr) {
		for (i=0; i<count; i++) {
			if (epaddrs[i] == (void *)FI_ADDR_NOTAVAIL)
				fi_addr[i] = FI_ADDR_NOTAVAIL;
			else if (types[i] == PSMX2_EP_SCALABLE)
				fi_addr[i] = (av_priv->last + i) | PSMX2_SEP_ADDR_FLAG;
			else if (av_priv->type == FI_AV_TABLE)
				fi_addr[i] = av_priv->last + i;
			else
				fi_addr[i] = PSMX2_EP_TO_ADDR(epaddrs[i], vlanes[i]);
		}
	}

	av_priv->last += count;

	if (av_priv->flags & FI_EVENT) {
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

	free(mask);
	free(errors);
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

	if (!addr || !addrlen)
		return -FI_EINVAL;

	av_priv = container_of(av, struct psmx2_fid_av, av);

	memset(&name, 0, sizeof(name));
	if (av_priv->type == FI_AV_TABLE) {
		idx = (int)(int64_t)fi_addr;
		if (idx >= av_priv->last)
			return -FI_EINVAL;

		name.epid = av_priv->epids[idx];
		name.vlane = av_priv->vlanes[idx];
	} else {
		context = psm2_epaddr_getctxt(PSMX2_ADDR_TO_EP(fi_addr));
		name.epid = context->epid;
		name.vlane = PSMX2_ADDR_TO_VL(fi_addr);
	}

	if (av_priv->addr_format == FI_ADDR_STR) {
		ofi_straddr(addr, addrlen, FI_ADDR_PSMX2, &name);
	} else {
		memcpy(addr, &name, MIN(*addrlen, sizeof(name)));
		*addrlen = sizeof(name);
	}

	return 0;
}

fi_addr_t psmx2_av_translate_source(struct psmx2_fid_av *av, fi_addr_t source)
{
	struct psmx2_epaddr_context *context;
	psm2_epaddr_t epaddr;
	int vlane;
	int i;

	epaddr = PSMX2_ADDR_TO_EP(source);
	vlane = PSMX2_ADDR_TO_VL(source);

	context = psm2_epaddr_getctxt(epaddr);
	if (!context)
		return FI_ADDR_NOTAVAIL;

	if (av->type == FI_AV_MAP)
		return source;

	for (i = av->last - 1; i >= 0; i--) {
		if (av->epaddrs[i] == epaddr && av->vlanes[i] == vlane)
			return (fi_addr_t)i;
	}

	return FI_ADDR_NOTAVAIL;
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
	free(av->epids);
	free(av->epaddrs);
	free(av->vlanes);
	free(av->types);
	for (i=0; i<av->last; i++) {
		if (!av->sepaddrs[i])
			continue;
		for (j=0; j<av->sepaddrs[i]->ctxt_cnt; j++)
			free(av->sepaddrs[i]->ctxt_addrs[j].epaddrs);
		free(av->sepaddrs[i]);
	}
	free(av->sepaddrs);
	free(av);
	return 0;
}

static int psmx2_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct psmx2_fid_av *av;

	av = container_of(fid, struct psmx2_fid_av, av.fid);

	if (!bfid)
		return -FI_EINVAL;

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
	int type = FI_AV_MAP;
	size_t count = 64;
	uint64_t flags = 0;
	int rx_ctx_bits = PSMX2_MAX_RX_CTX_BITS;

	domain_priv = container_of(domain, struct psmx2_fid_domain,
				   util_domain.domain_fid);

	if (attr) {
		switch (attr->type) {
		case FI_AV_UNSPEC:
			break;

		case FI_AV_MAP:
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

	av_priv = (struct psmx2_fid_av *) calloc(1, sizeof *av_priv);
	if (!av_priv)
		return -FI_ENOMEM;

	psmx2_domain_acquire(domain_priv);

	av_priv->domain = domain_priv;
	av_priv->type = type;
	av_priv->addrlen = sizeof(psm2_epaddr_t);
	av_priv->count = count;
	av_priv->flags = flags;
	av_priv->rx_ctx_bits = rx_ctx_bits;
	av_priv->addr_format = domain_priv->addr_format;

	av_priv->av.fid.fclass = FI_CLASS_AV;
	av_priv->av.fid.context = context;
	av_priv->av.fid.ops = &psmx2_fi_ops;
	av_priv->av.ops = &psmx2_av_ops;

	*av = &av_priv->av;
	if (attr)
		attr->type = type;

	return 0;
}

