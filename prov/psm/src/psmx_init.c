/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#define PSM_SUPPORTED_FLAGS (FI_NONBLOCK | FI_ACK | FI_EXCL | FI_BUFFERED_SEND | \
			     FI_BUFFERED_RECV | FI_CANCEL)
#define PSM_DEFAULT_FLAGS   (FI_NONBLOCK)

static int psmx_getinfo(char *node, char *service, struct fi_info *hints,
			struct fi_info **info)
{
	struct fi_info *psmx_info;
	uint64_t flags = 0;
	void *dst_addr = NULL;
	void *uuid;
	char *s;

	uuid = calloc(1, sizeof(psm_uuid_t));
	if (!uuid) 
		return -ENOMEM;

	s = getenv("SFI_PSM_UUID");
	if (s)
		psmx_string_to_uuid(s, uuid);

	if (node)
		dst_addr = psmx_resolve_name(node, uuid);

	if (service) {
		/* FIXME: check service */
	}

	if (hints) {
		switch (hints->type) {
		case FID_UNSPEC:
		case FID_RDM:
			break;
		default:
			*info = NULL;
			return -ENODATA;
		}

		switch (hints->protocol) {
		case FI_PROTO_UNSPEC:
			  if (hints->protocol_cap & FI_PROTO_CAP_TAGGED)
				  break;
		/* fall through */
		default:
			*info = NULL;
			return -ENODATA;
		}

		flags = hints->flags;
		if ((flags & PSM_SUPPORTED_FLAGS) != flags) {
			*info = NULL;
			return -ENODATA;
		}

		if (hints->domain_name && strncmp(hints->domain_name, "psm", 3)) {
			*info = NULL;
			return -ENODATA;
		}

		/* FIXME: check other fields of hints */
	}

	psmx_info = calloc(1, sizeof *psmx_info);
	if (!psmx_info) {
		free(uuid);
		return -ENOMEM;
	}

	psmx_info->next = NULL;
	psmx_info->size = sizeof(*psmx_info);
	psmx_info->flags = flags | PSM_DEFAULT_FLAGS;
	psmx_info->type = FID_RDM;
	psmx_info->protocol = PSMX_OUI_INTEL << FI_OUI_SHIFT | PSMX_PROTOCOL;
	psmx_info->protocol_cap = FI_PROTO_CAP_TAGGED;
	psmx_info->iov_format = FI_IOTAGGED; /* FIXME: or FI_IOTAGGEDV? */
	psmx_info->addr_format = FI_ADDR; 
	psmx_info->info_addr_format = FI_ADDR;
	psmx_info->src_addrlen = 0;
	psmx_info->dst_addrlen = sizeof(psm_epid_t);
	psmx_info->src_addr = NULL;
	psmx_info->dst_addr = dst_addr;
	psmx_info->auth_keylen = sizeof(psm_uuid_t);
	psmx_info->auth_key = uuid;
	psmx_info->shared_fd = -1;
	psmx_info->domain_name = strdup("psm");
	psmx_info->datalen = 0;
	psmx_info->data = NULL;

	*info = psmx_info;

	return 0;
}

static struct fi_ops_prov psmx_ops = {
	.size = sizeof(struct fi_ops_prov),
	.getinfo = psmx_getinfo,
	.freeinfo = NULL,
	.socket = psmx_sock_open,
	.open = psmx_domain_open
};

void psmx_ini(void)
{
	int major, minor;
	int err;

        psm_error_register_handler(NULL, PSM_ERRHANDLER_NO_HANDLER);

	major = PSM_VERNO_MAJOR;
	minor = PSM_VERNO_MINOR;

        err = psm_init(&major, &minor);
	if (err != PSM_OK) {
		fprintf(stderr, "%s: psm_init failed: %s\n", __func__,
			psm_error_get_string(err));
		return;
	}

	if (major > PSM_VERNO_MAJOR) {
		fprintf(stderr, "%s: PSM loaded an unexpected/unsupported version %d.%d\n",
			__func__, major, minor);
		return;
	}

	fi_register(&psmx_ops);
}

void psmx_fini(void)
{
	psm_finalize();
}

