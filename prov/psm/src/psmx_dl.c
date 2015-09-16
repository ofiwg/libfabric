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

struct psmx_lib psmx_lib;
void *psmx_lib_handle = NULL;

#define PSMX_LOAD_FUNC(func) \
	do { \
		char *err; \
		dlerror(); \
		psmx_lib.func = dlsym(psmx_lib_handle, #func); \
		if ((err = dlerror()) != NULL) { \
			FI_WARN(&psmx_prov, FI_LOG_CORE, "dlsym(%s): %s\n", #func, err); \
			return -1; \
		} \
	} while (0)

int psmx_dl_open(void)
{
	if (psmx_lib_handle)
		return 0;

	psmx_lib_handle = dlopen(PSMX_LIB_NAME, RTLD_LAZY | RTLD_LOCAL);
	if (!psmx_lib_handle) {
		FI_WARN(&psmx_prov, FI_LOG_CORE,"dlopen(%s): %s\n",
			PSMX_LIB_NAME, dlerror());
		return -1;
	}

	PSMX_LOAD_FUNC(psm_init);
	PSMX_LOAD_FUNC(psm_finalize);
	PSMX_LOAD_FUNC(psm_error_register_handler);
	PSMX_LOAD_FUNC(psm_error_defer);
	PSMX_LOAD_FUNC(psm_epid_nid);
	PSMX_LOAD_FUNC(psm_epid_context);
	PSMX_LOAD_FUNC(psm_epid_port);
	PSMX_LOAD_FUNC(psm_ep_num_devunits);
	PSMX_LOAD_FUNC(psm_uuid_generate);
	PSMX_LOAD_FUNC(psm_ep_open);
	PSMX_LOAD_FUNC(psm_ep_open_opts_get_defaults);
	PSMX_LOAD_FUNC(psm_ep_epid_share_memory);
	PSMX_LOAD_FUNC(psm_ep_close);
	PSMX_LOAD_FUNC(psm_map_nid_hostname);
	PSMX_LOAD_FUNC(psm_ep_connect);
	PSMX_LOAD_FUNC(psm_poll);
	PSMX_LOAD_FUNC(psm_epaddr_setlabel);
	PSMX_LOAD_FUNC(psm_epaddr_setctxt);
	PSMX_LOAD_FUNC(psm_epaddr_getctxt);
	PSMX_LOAD_FUNC(psm_setopt);
	PSMX_LOAD_FUNC(psm_getopt);
	PSMX_LOAD_FUNC(psm_ep_query);
	PSMX_LOAD_FUNC(psm_ep_epid_lookup);
	PSMX_LOAD_FUNC(psm_mq_init);
	PSMX_LOAD_FUNC(psm_mq_finalize);
	PSMX_LOAD_FUNC(psm_mq_getopt);
	PSMX_LOAD_FUNC(psm_mq_setopt);
	PSMX_LOAD_FUNC(psm_mq_irecv);
	PSMX_LOAD_FUNC(psm_mq_send);
	PSMX_LOAD_FUNC(psm_mq_isend);
	PSMX_LOAD_FUNC(psm_mq_iprobe);
	PSMX_LOAD_FUNC(psm_mq_ipeek);
	PSMX_LOAD_FUNC(psm_mq_wait);
	PSMX_LOAD_FUNC(psm_mq_test);
	PSMX_LOAD_FUNC(psm_mq_cancel);
	PSMX_LOAD_FUNC(psm_mq_get_stats);
#if (PSM_VERNO_MAJOR >= 2)
	PSMX_LOAD_FUNC(psm_mq_irecv2);
	PSMX_LOAD_FUNC(psm_mq_imrecv);
	PSMX_LOAD_FUNC(psm_mq_send2);
	PSMX_LOAD_FUNC(psm_mq_isend2);
	PSMX_LOAD_FUNC(psm_mq_iprobe2);
	PSMX_LOAD_FUNC(psm_mq_improbe);
	PSMX_LOAD_FUNC(psm_mq_improbe2);
	PSMX_LOAD_FUNC(psm_mq_ipeek2);
	PSMX_LOAD_FUNC(psm_mq_wait2);
	PSMX_LOAD_FUNC(psm_mq_test2);
#endif

	return 0;
}

void psmx_dl_close(void)
{
	if (psmx_lib_handle) {
		dlclose(psmx_lib_handle);
		psmx_lib_handle = NULL;
	}
}

