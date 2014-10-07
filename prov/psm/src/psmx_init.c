/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#include "psmx.h"
#include "fi.h"

static int psmx_reserve_tag_bits(int *ep_cap, uint64_t *max_tag_value)
{
	int reserved_bits = 0;

	if (*ep_cap) {
		if (*ep_cap & PSMX_EP_CAP_OPT1)
			reserved_bits++;

		if (*ep_cap & PSMX_EP_CAP_OPT2)
			reserved_bits++;

		if (*max_tag_value > (~0ULL >> reserved_bits)) {
			psmx_debug("%s: unable to reserve %d bits for asked features.\n",
					__func__);
			return -1;
		}

		*max_tag_value = (~0ULL >> reserved_bits);
		return 0;
	}

	*ep_cap = PSMX_EP_CAP_BASE;

	if (*max_tag_value <= (~0ULL >> 1)) {
		*ep_cap |= PSMX_EP_CAP_OPT1;
		reserved_bits++;
	}

	if (*max_tag_value <= (~0ULL >> 2)) {
		*ep_cap |= PSMX_EP_CAP_OPT2;
		reserved_bits++;
	}

	*max_tag_value = (~0ULL >> reserved_bits);
	return 0;
}

static int psmx_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *psmx_info;
	uint32_t cnt = 0;
	void *dest_addr = NULL;
	int type = FI_EP_RDM;
	int ep_cap = 0;
	uint64_t max_tag_value = 0;
	int err = -ENODATA;

	*info = NULL;

	if (psm_ep_num_devunits(&cnt) || !cnt) {
		psmx_debug("%s: no PSM device is found.\n", __func__);
		return -FI_ENODATA;
	}

	if (node)
		dest_addr = psmx_resolve_name(node);

	if (service) {
		/* FIXME: check service */
		/* Can service be used as the port number needed by psmx_resolve_name? */
	}

	if (hints) {
		switch (hints->type) {
		case FI_EP_UNSPEC:
		case FI_EP_RDM:
			break;
		case FI_EP_MSG:
			type = FI_EP_MSG;
			break;
		default:
			psmx_debug("%s: hints->type=%d, supported=%d,%d,%d.\n",
					__func__, hints->type, FI_EP_UNSPEC,
					FI_EP_RDM, FI_EP_MSG);
			goto err_out;
		}

		if (hints->ep_attr) {
			switch (hints->ep_attr->protocol) {
			case FI_PROTO_UNSPEC:
				break;
			default:
				psmx_debug("%s: hints->protocol=%d, supported=%d\n",
						__func__, hints->ep_attr->protocol, FI_PROTO_UNSPEC);
				goto err_out;
			}
		}

		if ((hints->ep_cap & PSMX_EP_CAP) != hints->ep_cap) {
			psmx_debug("%s: hints->ep_cap=0x%llx, supported=0x%llx\n",
					__func__, hints->ep_cap, PSMX_EP_CAP);
			goto err_out;
		}

		if ((hints->op_flags & PSMX_OP_FLAGS) != hints->op_flags) {
			psmx_debug("%s: hints->flags=0x%llx, supported=0x%llx\n",
					__func__, hints->op_flags, PSMX_OP_FLAGS);
			goto err_out;
		}

		if (hints->domain_attr &&
		    ((hints->domain_attr->caps & PSMX_DOMAIN_CAP) !=
		      hints->domain_attr->caps)) {
			psmx_debug("%s: hints->domain_cap=0x%llx, supported=0x%llx\n",
					__func__, hints->domain_attr->caps, PSMX_DOMAIN_CAP);
			goto err_out;
		}

		if (hints->fabric_attr && hints->fabric_attr->name &&
		    strncmp(hints->fabric_attr->name, "psm", 3)) {
			psmx_debug("%s: hints->fabric_name=%s, supported=psm\n",
					__func__, hints->fabric_attr->name);
			goto err_out;
		}

		if (hints->domain_attr && hints->domain_attr->name &&
		    strncmp(hints->domain_attr->name, "psm", 3)) {
			psmx_debug("%s: hints->domain_name=%s, supported=psm\n",
					__func__, hints->domain_attr->name);
			goto err_out;
		}

		if (hints->ep_attr) {
			if (hints->ep_attr->max_msg_size > PSMX_MAX_MSG_SIZE) {
				psmx_debug("%s: hints->ep_attr->max_msg_size=%ld,"
						"supported=%ld.\n", __func__,
						hints->ep_attr->max_msg_size,
						PSMX_MAX_MSG_SIZE);
				goto err_out;
			}
			if (hints->ep_attr->inject_size > PSMX_INJECT_SIZE) {
				psmx_debug("%s: hints->ep_attr->inject_size=%ld,"
						"supported=%ld.\n", __func__,
						hints->ep_attr->inject_size,
						PSMX_INJECT_SIZE);
				goto err_out;
			}
			max_tag_value = fi_tag_bits(hints->ep_attr->mem_tag_format);
		}

		ep_cap = hints->ep_cap;

		/* FIXME: check other fields of hints */
	}

	if (psmx_reserve_tag_bits(&ep_cap, &max_tag_value) < 0)
		goto err_out;

	psmx_info = __fi_allocinfo();
	if (!psmx_info) {
		err = -ENOMEM;
		goto err_out;
	}

	psmx_info->ep_attr->protocol = PSMX_OUI_INTEL << FI_OUI_SHIFT | PSMX_PROTOCOL;
	psmx_info->ep_attr->max_msg_size = PSMX_MAX_MSG_SIZE;
	psmx_info->ep_attr->inject_size = PSMX_INJECT_SIZE;
	psmx_info->ep_attr->total_buffered_recv = ~(0ULL); /* that's how PSM handles it internally! */
	psmx_info->ep_attr->mem_tag_format = fi_tag_format(max_tag_value);
	psmx_info->ep_attr->msg_order = FI_ORDER_SAS;

	psmx_info->domain_attr->threading = FI_THREAD_PROGRESS;
	psmx_info->domain_attr->control_progress = FI_PROGRESS_MANUAL;
	psmx_info->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	psmx_info->domain_attr->caps = (hints && hints->domain_attr &&
					hints->domain_attr->caps) ?
					hints->domain_attr->caps : PSMX_DOMAIN_CAP;
	psmx_info->domain_attr->name = strdup("psm");

	psmx_info->next = NULL;
	psmx_info->type = type;
	psmx_info->ep_cap = (hints && hints->ep_cap) ? hints->ep_cap : ep_cap;
	psmx_info->op_flags = hints ? hints->op_flags : 0;
	psmx_info->addr_format = FI_ADDR_PROTO;
	psmx_info->src_addrlen = 0;
	psmx_info->dest_addrlen = sizeof(psm_epid_t);
	psmx_info->src_addr = NULL;
	psmx_info->dest_addr = dest_addr;
	psmx_info->fabric_attr->name = strdup("psm");

	*info = psmx_info;
	return 0;

err_out:
	return err;
}

static int psmx_fabric_close(fid_t fid)
{
	free(fid);
	return 0;
}

static struct fi_ops psmx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_fabric_close,
};

static struct fi_ops_fabric psmx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = psmx_domain_open,
};

static int psmx_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric, void *context)
{
	struct psmx_fid_fabric *fabric_priv;

	if (strncmp(attr->name, "psm", 3))
		return -FI_ENODATA;

	fabric_priv = calloc(1, sizeof(*fabric_priv));
	if (!fabric_priv)
		return -FI_ENOMEM;

	fabric_priv->fabric.fid.fclass = FI_CLASS_FABRIC;
	fabric_priv->fabric.fid.context = context;
	fabric_priv->fabric.fid.ops = &psmx_fabric_fi_ops;
	fabric_priv->fabric.ops = &psmx_fabric_ops;
	*fabric = &fabric_priv->fabric;
	return 0;
}

static struct fi_provider psmx_prov = {
	.name = "PSM",
	.version = FI_VERSION(0, 9),
	.getinfo = psmx_getinfo,
	.fabric = psmx_fabric,
};

static void __attribute__((constructor)) psmx_ini(void)
{
	int major, minor;
	int err;
	char *s;
	int check_version = 1;

	s = getenv("SFI_PSM_VERSION_CHECK");
	if (s)
		check_version = atoi(s);

        psm_error_register_handler(NULL, PSM_ERRHANDLER_NO_HANDLER);

	major = PSM_VERNO_MAJOR;
	minor = PSM_VERNO_MINOR;

        err = psm_init(&major, &minor);
	if (err != PSM_OK) {
		fprintf(stderr, "%s: psm_init failed: %s\n", __func__,
			psm_error_get_string(err));
		return;
	}

	if (check_version && major != PSM_VERNO_MAJOR) {
		fprintf(stderr, "%s: PSM version mismatch: header %d.%d, library %d.%d.\n",
			__func__, PSM_VERNO_MAJOR, PSM_VERNO_MINOR, major, minor);
		fprintf(stderr, "\tSet envar SFI_PSM_VERSION_CHECK=0 to bypass version check.\n");
		return;
	}

	(void) fi_register(&psmx_prov);
}

static void __attribute__((destructor)) psmx_fini(void)
{
	psm_finalize();
}
