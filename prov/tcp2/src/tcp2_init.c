/*
 * Copyright (c) 2017-2022 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "tcp2.h"

#include <sys/types.h>
#include <ofi_util.h>
#include <stdlib.h>

static int tcp2_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	return ofi_ip_getinfo(&tcp2_util_prov, version, node, service, flags,
			      hints, info);
}

struct tcp2_port_range tcp2_ports = {
	.low  = 0,
	.high = 0,
};

int tcp2_nodelay = -1;

int tcp2_staging_sbuf_size = 9000;
int tcp2_prefetch_rbuf_size = 9000;
size_t tcp2_default_tx_size = 256;
size_t tcp2_default_rx_size = 256;
size_t tcp2_zerocopy_size = SIZE_MAX;


static void tcp2_init_env(void)
{
	size_t tx_size;
	size_t rx_size;

	/* Checked in util code */
	fi_param_define(&tcp2_prov, "iface", FI_PARAM_STRING,
			"Specify interface name");

	fi_param_define(&tcp2_prov,"port_low_range", FI_PARAM_INT,
			"define port low range");
	fi_param_define(&tcp2_prov,"port_high_range", FI_PARAM_INT,
			"define port high range");
	fi_param_get_int(&tcp2_prov, "port_high_range", &tcp2_ports.high);
	fi_param_get_int(&tcp2_prov, "port_low_range", &tcp2_ports.low);

	if (tcp2_ports.high > TCP2_PORT_MAX_RANGE)
		tcp2_ports.high = TCP2_PORT_MAX_RANGE;

	if (tcp2_ports.low < 0 || tcp2_ports.high < 0 ||
	    tcp2_ports.low > tcp2_ports.high) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,"User provided "
			"port range invalid. Ignoring. \n");
		tcp2_ports.low  = 0;
		tcp2_ports.high = 0;
	}

	fi_param_define(&tcp2_prov,"tx_size", FI_PARAM_SIZE_T,
			"define default tx context size (default: %zu)",
			tcp2_default_tx_size);
	fi_param_define(&tcp2_prov,"rx_size", FI_PARAM_SIZE_T,
			"define default rx context size (default: %zu)",
			tcp2_default_rx_size);
	if (!fi_param_get_size_t(&tcp2_prov, "tx_size", &tx_size)) {
		tcp2_default_tx_size = tx_size;
	}
	if (!fi_param_get_size_t(&tcp2_prov, "rx_size", &rx_size)) {
		tcp2_default_rx_size = rx_size;
	}

	fi_param_define(&tcp2_prov, "nodelay", FI_PARAM_BOOL,
			"overrides default TCP_NODELAY socket setting");
	fi_param_get_bool(&tcp2_prov, "nodelay", &tcp2_nodelay);

	fi_param_define(&tcp2_prov, "staging_sbuf_size", FI_PARAM_INT,
			"size of buffer used to coalesce iovec's or "
			"send requests before posting to the kernel, "
			"set to 0 to disable");
	fi_param_define(&tcp2_prov, "prefetch_rbuf_size", FI_PARAM_INT,
			"size of buffer used to prefetch received data from "
			"the kernel, set to 0 to disable");
	fi_param_define(&tcp2_prov, "zerocopy_size", FI_PARAM_SIZE_T,
			"lower threshold where zero copy transfers will be "
			"used, if supported by the platform, set to -1 to "
			"disable (default: %zu)", tcp2_zerocopy_size);
	fi_param_get_int(&tcp2_prov, "staging_sbuf_size",
			 &tcp2_staging_sbuf_size);
	fi_param_get_int(&tcp2_prov, "prefetch_rbuf_size",
			 &tcp2_prefetch_rbuf_size);
	fi_param_get_size_t(&tcp2_prov, "zerocopy_size", &tcp2_zerocopy_size);
}

static void tcp2_fini(void)
{
	/* empty as of now */
}

struct fi_provider tcp2_prov = {
	.name = "tcp2",
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = tcp2_getinfo,
	.fabric = tcp2_create_fabric,
	.cleanup = tcp2_fini,
};

TCP2_INI
{
#if HAVE_TCP2_DL
	ofi_pmem_init();
#endif
	tcp2_init_env();
	return &tcp2_prov;
}
