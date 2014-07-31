/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

static int psmx_getinfo(const char *node, const char *service, uint64_t flags,
			struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *psmx_info;
	uint32_t cnt = 0;
	void *dest_addr = NULL;
	void *uuid;
	char *s;
	int type = FID_RDM;

	if (psm_ep_num_devunits(&cnt) || !cnt) {
		psmx_debug("%s: no PSM device is found.\n", __func__);
		return -FI_ENODATA;
	}

	uuid = calloc(1, sizeof(psm_uuid_t));
	if (!uuid) 
		return -ENOMEM;

	s = getenv("SFI_PSM_UUID");
	if (s)
		psmx_string_to_uuid(s, uuid);

	if (node)
		dest_addr = psmx_resolve_name(node, uuid);

	if (service) {
		/* FIXME: check service */
	}

	if (hints) {
		switch (hints->type) {
		case FID_UNSPEC:
		case FID_RDM:
			break;
		case FID_MSG:
			type = FID_MSG;
			break;
		default:
			psmx_debug("%s: hints->type=%d, supported=%d,%d,%d.\n",
					__func__, hints->type, FID_UNSPEC, FID_RDM, FID_MSG);
			*info = NULL;
			return -ENODATA;
		}

		switch (hints->protocol) {
		case FI_PROTO_UNSPEC:
			if ((hints->ep_cap & PSMX_EP_CAPS) == hints->ep_cap)
				break;

			psmx_debug("%s: hints->ep_cap=0x%llx, supported=0x%llx\n",
					__func__, hints->ep_cap, PSMX_EP_CAPS);

		/* fall through */
		default:
			psmx_debug("%s: hints->protocol=%d, supported=%d\n",
					__func__, hints->protocol, FI_PROTO_UNSPEC);
			*info = NULL;
			return -ENODATA;
		}

		if ((hints->op_flags & PSMX_SUPPORTED_FLAGS) != hints->op_flags) {
			psmx_debug("%s: hints->flags=0x%llx, supported=0x%llx\n",
					__func__, hints->op_flags, PSMX_SUPPORTED_FLAGS);
			*info = NULL;
			return -ENODATA;
		}

		if (hints->domain_name && strncmp(hints->domain_name, "psm", 3)) {
			psmx_debug("%s: hints->domain_name=%s, supported=psm\n",
					__func__, hints->domain_name);

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
	psmx_info->op_flags = hints ? hints->op_flags : 0;
	psmx_info->type = type;
	psmx_info->protocol = PSMX_OUI_INTEL << FI_OUI_SHIFT | PSMX_PROTOCOL;
	if (hints && hints->ep_cap)
		psmx_info->ep_cap = hints->ep_cap & PSMX_EP_CAPS;
	else
		psmx_info->ep_cap = FI_TAGGED;
	psmx_info->domain_cap = FI_WRITE_COHERENT | FI_CONTEXT | FI_USER_MR_KEY;
	psmx_info->addr_format = FI_ADDR; 
	psmx_info->info_addr_format = FI_ADDR;
	psmx_info->src_addrlen = 0;
	psmx_info->dest_addrlen = sizeof(psm_epid_t);
	psmx_info->src_addr = NULL;
	psmx_info->dest_addr = dest_addr;
	psmx_info->auth_keylen = sizeof(psm_uuid_t);
	psmx_info->auth_key = uuid;
	psmx_info->control_progress = FI_PROGRESS_MANUAL;
	psmx_info->data_progress = FI_PROGRESS_MANUAL;
	psmx_info->fabric_name = strdup("psm");
	psmx_info->domain_name = strdup("psm");
	psmx_info->datalen = 0;
	psmx_info->data = NULL;

	*info = psmx_info;

	return 0;
}

static struct fi_ops_prov psmx_ops = {
	.getinfo = psmx_getinfo,
	.domain = psmx_domain_open,
};

void psmx_ini(void)
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

	(void) fi_register(&psmx_ops);
}

void psmx_fini(void)
{
	psm_finalize();
}

