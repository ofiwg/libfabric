/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. Allrights reserved.
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <net/if.h>
#include <poll.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_util.h"

#define BUF_SIZE 256

/*
 * get gni nic addr from AF_INET  ip addr, also return local device id on same
 *subnet
 * as the input ip_addr.
 *
 * returns 0 if ipogif entry found
 * otherwise  -errno
 */
static int gnixu_get_pe_from_ip(const char *ip_addr, uint32_t *gni_nic_addr)
{
	int scount;
	/* return this if no ipgogif for this ip-addr found */
	int ret = -FI_ENODATA;
	FILE *fd = NULL;
	char line[BUF_SIZE], *tmp;
	char dummy[64], iface[64], fnd_ip_addr[64];
	char mac_str[64];
	int w, x, y;

	fd = fopen("/proc/net/arp", "r");
	if (fd == NULL) {
		return -errno;
	}

	if (fd == NULL) {
		return -errno;
	}

	while (1) {
		tmp = fgets(line, BUF_SIZE, fd);
		if (!tmp) {
			break;
		}

		/*
		 * check for a match
		 */
		if ((strstr(line, ip_addr) != NULL) &&
		    (strstr(line, "ipogif") != NULL)) {
			ret = 0;
			scount = sscanf(line, "%s%s%s%s%s%s", fnd_ip_addr,
					dummy, dummy, mac_str, dummy, iface);
			if (scount != 6) {
				ret = -EIO;
				goto err;
			}

			/*
			 * check exact match of ip addr
			 */
			if (!strcmp(fnd_ip_addr, ip_addr)) {
				scount =
				    sscanf(mac_str, "00:01:01:%02x:%02x:%02x",
					   &w, &x, &y);
				if (scount != 3) {
					ret = -EIO;
					goto err;
				}

				/*
				 * mysteries of XE/XC mac to nid mapping, see
				 * nid2mac in xt sysutils
				 */
				*gni_nic_addr = (w << 16) | (x << 8) | y;
				ret = FI_SUCCESS;
				break;
			}
		}
	}

err:
	fclose(fd);
	return ret;
}

/*
 * gnix_resolve_name: given a node hint and a valid pointer to a gnix_ep_name
 * will resolve the gnix specific address of node and fill the provided
 * gnix_ep_name pointer with the information.
 *
 * node (IN) : Node name being resolved to gnix specific address
 * service (IN) : Port number being resolved to gnix specific address
 * resolved_addr (IN/OUT) : Pointer that must be provided to contain the
 *	resolved address.
 */
int gnix_resolve_name(IN const char *node, IN const char *service,
		      IN uint64_t flags, INOUT struct gnix_ep_name
		      *resolved_addr)
{
	int sock = -1;
	uint32_t pe = -1;
	uint32_t cpu_id = -1;
	struct addrinfo *result = NULL;
	struct addrinfo *rp = NULL;

	struct ifreq ifr = {{{0}}};

	struct sockaddr_in *sa = NULL;
	struct sockaddr_in *sin = NULL;

	int ret = FI_SUCCESS;
	gni_return_t status = GNI_RC_SUCCESS;

	struct addrinfo hints = {
		.ai_family = AF_INET,
		.ai_socktype = SOCK_DGRAM,
		.ai_flags = AI_CANONNAME
	};

	if (flags & FI_SOURCE)
		hints.ai_flags |= AI_PASSIVE;

	if (flags & FI_NUMERICHOST)
		hints.ai_flags |= AI_NUMERICHOST;

	if (!resolved_addr) {
		GNIX_WARN(FI_LOG_FABRIC,
			 "Resolved_addr must be a valid pointer.\n");
		ret = -FI_EINVAL;
		goto err;
	}

	sock = socket(AF_INET, SOCK_DGRAM, 0);

	if (sock == -1) {
		GNIX_WARN(FI_LOG_FABRIC, "Socket creation failed: %s\n",
			  strerror(errno));
		ret = -FI_EIO;
		goto err;
	}

	/* Get the address for the ipogif0 interface */
	ifr.ifr_addr.sa_family = AF_INET;
	snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", "ipogif0");

	ret = ioctl(sock, SIOCGIFADDR, &ifr);
	if (ret == -1) {
		GNIX_WARN(FI_LOG_FABRIC,
			  "Failed to get address for ipogif0: %s\n",
			  strerror(errno));
		ret = -FI_EIO;
		goto sock_cleanup;
	}

	sin = (struct sockaddr_in *) &ifr.ifr_addr;

	ret = getaddrinfo(node, service, &hints, &result);
	if (ret != 0) {
		GNIX_WARN(FI_LOG_FABRIC,
			  "Failed to get address for node provided: %s\n",
			  strerror(errno));
		ret = -FI_EINVAL;
		goto sock_cleanup;
	}

	for (rp = result; rp != NULL; rp = rp->ai_next) {
		assert(rp->ai_addr->sa_family == AF_INET);
		sa = (struct sockaddr_in *) rp->ai_addr;

		/*
		 * If we are trying to resolve localhost then use
		 * CdmGetNicAddress.
		 */
		if (sa->sin_addr.s_addr == sin->sin_addr.s_addr) {
			status = GNI_CdmGetNicAddress(0, &pe, &cpu_id);
			if(status == GNI_RC_SUCCESS) {
				break;
			} else {
				GNIX_WARN(FI_LOG_FABRIC,
					  "Unable to get NIC address.");
				ret = gnixu_to_fi_errno(status);
				goto sock_cleanup;
			}
		} else {
			ret =
			    gnixu_get_pe_from_ip(inet_ntoa(sa->sin_addr), &pe);
			if (ret == 0) {
				break;
			}
		}
	}

	/*
	 * Make sure address is valid.
	 */
	if (pe == -1) {
		GNIX_WARN(FI_LOG_FABRIC,
			  "Unable to acquire valid address for node %s\n",
			  node);
		ret = -FI_EADDRNOTAVAIL;
		goto sock_cleanup;
	}

	/*
	 * Fill the INOUT parameter resolved_addr with the address information
	 * acquired for the provided node parameter.
	 */
	memset(resolved_addr, 0, sizeof(struct gnix_ep_name));

	resolved_addr->gnix_addr.device_addr = pe;
	if (service) {
		/* use resolved service/port */
		resolved_addr->gnix_addr.cdm_id = ntohs(sa->sin_port);
		resolved_addr->name_type = GNIX_EPN_TYPE_BOUND;
		resolved_addr->cm_nic_cdm_id = resolved_addr->gnix_addr.cdm_id;
	} else {
		/* generate port internally */
		resolved_addr->name_type = GNIX_EPN_TYPE_UNBOUND;
	}
	GNIX_INFO(FI_LOG_FABRIC, "Resolved: %s:%s to gnix_addr: 0x%lx\n",
		  node ?: "", service ?: "", resolved_addr->gnix_addr);
sock_cleanup:
	if(close(sock) == -1) {
		GNIX_WARN(FI_LOG_FABRIC, "Unable to close socket: %s\n",
			  strerror(errno));
	}
err:
	if (result != NULL) {
		freeaddrinfo(result);
	}
	return ret;
}
