/*
 * Copyright (c) 2015-2016 Intel Corporation. All rights reserved.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "udpx.h"

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>


#if HAVE_GETIFADDRS
static void udpx_getinfo_ifs(struct fi_info **info)
{
	struct ifaddrs *ifaddrs, *ifa;
	struct fi_info *head, *tail, *cur, *loopback;
	size_t addrlen;
	uint32_t addr_format;
	int ret;

	ret = ofi_getifaddrs(&ifaddrs);
	if (ret)
		return;

	head = tail = loopback = NULL;
	for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
		if (ifa->ifa_addr == NULL || !(ifa->ifa_flags & IFF_UP))
			continue;

		switch (ifa->ifa_addr->sa_family) {
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

		cur = fi_dupinfo(*info);
		if (!cur)
			break;

		if(!ofi_is_loopback_addr(ifa->ifa_addr)) {
			if (!head)
				head = cur;
			else
				tail->next = cur;
			tail = cur;
		} else {
			cur->next = loopback;
			loopback = cur;
		}

		cur->src_addr = mem_dup(ifa->ifa_addr, addrlen);
		if (cur->src_addr) {
			cur->src_addrlen = addrlen;
			cur->addr_format = addr_format;
		}
	}
	freeifaddrs(ifaddrs);

	if (head || loopback) {
		if(!head) { /* loopback interface only? */
			head = loopback;
		} else {
			/* append loopback interfaces to tail */
			assert(tail);
			assert(!tail->next);
			tail->next = loopback;
		}

		fi_freeinfo(*info);
		*info = head;
	}
}
#else
#define udpx_getinfo_ifs(info) do{}while(0)
#endif

static int udpx_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	int ret;

	ret = util_getinfo(&udpx_util_prov, version, node, service, flags,
			   hints, info);
	if (ret)
		return ret;

	if (!(*info)->src_addr && !(*info)->dest_addr)
		udpx_getinfo_ifs(info);

	return 0;
}

static void udpx_fini(void)
{
	/* yawn */
}

struct fi_provider udpx_prov = {
	.name = "UDP",
	.version = FI_VERSION(UDPX_MAJOR_VERSION, UDPX_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = udpx_getinfo,
	.fabric = udpx_fabric,
	.cleanup = udpx_fini
};

UDP_INI
{
	return &udpx_prov;
}
