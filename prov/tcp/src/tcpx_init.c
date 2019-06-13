/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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
#include "tcpx.h"

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <ofi_util.h>
#include <stdlib.h>

#if HAVE_GETIFADDRS
static void tcpx_getinfo_ifs(struct fi_info **info)
{
	struct fi_info *head = NULL, *tail = NULL, *cur;
	struct slist addr_list;
	size_t addrlen;
	uint32_t addr_format;
	struct slist_entry *entry, *prev;
	struct ofi_addr_list_entry *addr_entry;

	slist_init(&addr_list);

	ofi_get_list_of_addr(&tcpx_prov, "iface", &addr_list);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&addr_list, entry, prev) {
		addr_entry = container_of(entry, struct ofi_addr_list_entry, entry);

		cur = fi_dupinfo(*info);
		if (!cur)
			break;

		if (!head)
			head = cur;
		else
			tail->next = cur;
		tail = cur;

		switch (addr_entry->ipaddr.sin.sin_family) {
		case AF_INET:
			addrlen = sizeof(struct sockaddr_in);
			addr_format = FI_SOCKADDR_IN;
			break;
		case AF_INET6:
			addrlen = sizeof(struct sockaddr_in6);
			addr_format = FI_SOCKADDR_IN6;
			break;
		default:
			continue;
		}

		cur->src_addr = mem_dup(&addr_entry->ipaddr.sa, addrlen);
		if (cur->src_addr) {
			cur->src_addrlen = addrlen;
			cur->addr_format = addr_format;
		}
		/* TODO: rework util code
		util_set_fabric_domain(&tcpx_prov, cur);
		*/
	}

	ofi_free_list_of_addr(&addr_list);
	fi_freeinfo(*info);
	*info = head;
}
#else
#define tcpx_getinfo_ifs(info) do{ } while(0)
#endif

static int tcpx_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	int ret;

	ret = util_getinfo(&tcpx_util_prov, version, node, service, flags,
			   hints, info);
	if (ret)
		return ret;

	if (!(*info)->src_addr && !(*info)->dest_addr)
		tcpx_getinfo_ifs(info);

	return 0;
}

struct tcpx_port_range port_range = {
	.low  = 0,
	.high = 0,
};

static int tcpx_init_env(void)
{
	srand(getpid());

	fi_param_get_int(&tcpx_prov, "port_high_range", &port_range.high);
	fi_param_get_int(&tcpx_prov, "port_low_range", &port_range.low);

	if (port_range.high > TCPX_PORT_MAX_RANGE) {
		port_range.high = TCPX_PORT_MAX_RANGE;
	}
	if (port_range.low < 0 || port_range.high < 0 ||
	    port_range.low > port_range.high) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"User provided "
			"port range invalid. Ignoring. \n");
		port_range.low  = 0;
		port_range.high = 0;
	}
	return 0;
}

static void fi_tcp_fini(void)
{
	/* empty as of now */
}

struct fi_provider tcpx_prov = {
	.name = "tcp",
	.version = FI_VERSION(TCPX_MAJOR_VERSION,TCPX_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 8),
	.getinfo = tcpx_getinfo,
	.fabric = tcpx_create_fabric,
	.cleanup = fi_tcp_fini,
};

TCP_INI
{
#if HAVE_TCP_DL
	ofi_pmem_init();
#endif
	fi_param_define(&tcpx_prov, "iface", FI_PARAM_STRING,
			"Specify interface name");

	fi_param_define(&tcpx_prov,"port_low_range", FI_PARAM_INT,
			"define port low range");

	fi_param_define(&tcpx_prov,"port_high_range", FI_PARAM_INT,
			"define port high range");

	if (tcpx_init_env()) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"Invalid info\n");
		return NULL;
	}
	return &tcpx_prov;
}
