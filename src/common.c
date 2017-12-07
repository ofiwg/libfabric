/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2006-2017 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013 Intel Corp., Inc.  All rights reserved.
 * Copyright (c) 2015 Los Alamos Nat. Security, LLC. All rights reserved.
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

#include "config.h"

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <pthread.h>
#include <sys/time.h>

#include <inttypes.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <fi_signal.h>
#include <rdma/providers/fi_prov.h>
#include <rdma/fi_errno.h>
#include <fi.h>
#include <fi_util.h>

struct fi_provider core_prov = {
	.name = "core",
	.version = 1,
	.fi_version = FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION)
};

struct ofi_common_locks common_locks = {
	.ini_lock = PTHREAD_MUTEX_INITIALIZER,
	.util_fabric_lock = PTHREAD_MUTEX_INITIALIZER,
};

int fi_poll_fd(int fd, int timeout)
{
	struct pollfd fds;
	int ret;

	fds.fd = fd;
	fds.events = POLLIN;
	ret = poll(&fds, 1, timeout);
	return ret == -1 ? -errno : ret;
}

uint64_t fi_tag_bits(uint64_t mem_tag_format)
{
	return UINT64_MAX >> (ffsll(htonll(mem_tag_format)) -1);
}

uint64_t fi_tag_format(uint64_t tag_bits)
{
	return FI_TAG_GENERIC >> (ffsll(htonll(tag_bits)) - 1);
}

int fi_size_bits(uint64_t num)
{
	int size_bits = 0;
	while (num >> ++size_bits);
	return size_bits;
}

int ofi_send_allowed(uint64_t caps)
{
	if (caps & FI_MSG ||
		caps & FI_TAGGED) {
		if (caps & FI_SEND)
			return 1;
		if (caps & FI_RECV)
			return 0;
		return 1;
	}

	return 0;
}

int ofi_recv_allowed(uint64_t caps)
{
	if (caps & FI_MSG ||
		caps & FI_TAGGED) {
		if (caps & FI_RECV)
			return 1;
		if (caps & FI_SEND)
			return 0;
		return 1;
	}

	return 0;
}

int ofi_rma_initiate_allowed(uint64_t caps)
{
	if (caps & FI_RMA ||
		caps & FI_ATOMICS) {
		if (caps & FI_WRITE ||
			caps & FI_READ)
			return 1;
		if (caps & FI_REMOTE_WRITE ||
			caps & FI_REMOTE_READ)
			return 0;
		return 1;
	}

	return 0;
}

int ofi_rma_target_allowed(uint64_t caps)
{
	if (caps & FI_RMA ||
		caps & FI_ATOMICS) {
		if (caps & FI_REMOTE_WRITE ||
			caps & FI_REMOTE_READ)
			return 1;
		if (caps & FI_WRITE ||
			caps & FI_READ)
			return 0;
		return 1;
	}

	return 0;
}

int ofi_ep_bind_valid(const struct fi_provider *prov, struct fid *bfid, uint64_t flags)
{
	if (!bfid) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "NULL bind fid\n");
		return -FI_EINVAL;
	}

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		if (flags & ~(FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION)) {
			FI_WARN(prov, FI_LOG_EP_CTRL, "invalid CQ flags\n");
			return -FI_EBADFLAGS;
		}
		break;
	case FI_CLASS_CNTR:
		if (flags & ~(FI_SEND | FI_RECV | FI_READ | FI_WRITE |
			      FI_REMOTE_READ | FI_REMOTE_WRITE)) {
			FI_WARN(prov, FI_LOG_EP_CTRL, "invalid cntr flags\n");
			return -FI_EBADFLAGS;
		}
		break;
	default:
		if (flags) {
			FI_WARN(prov, FI_LOG_EP_CTRL, "invalid bind flags\n");
			return -FI_EBADFLAGS;
		}
		break;
	}
	return FI_SUCCESS;
}

uint64_t fi_gettime_ms(void)
{
	struct timeval now;

	gettimeofday(&now, NULL);
	return now.tv_sec * 1000 + now.tv_usec / 1000;
}

uint64_t fi_gettime_us(void)
{
	struct timeval now;

	gettimeofday(&now, NULL);
	return now.tv_sec * 1000000 + now.tv_usec;
}

const char *ofi_straddr(char *buf, size_t *len,
			uint32_t addr_format, const void *addr)
{
	const struct sockaddr *sock_addr;
	const struct sockaddr_in6 *sin6;
	const struct sockaddr_in *sin;
	char str[INET6_ADDRSTRLEN + 8];
	size_t size;

	if (!addr || !len)
		return NULL;

	switch (addr_format) {
	case FI_SOCKADDR:
		sock_addr = addr;
		switch (sock_addr->sa_family) {
		case AF_INET:
			goto sa_sin;
		case AF_INET6:
			goto sa_sin6;
		default:
			return NULL;
		}
		break;
	case FI_SOCKADDR_IN:
sa_sin:
		sin = addr;
		if (!inet_ntop(sin->sin_family, &sin->sin_addr, str,
			       sizeof(str)))
			return NULL;

		size = snprintf(buf, MIN(*len, sizeof(str)),
				"fi_sockaddr_in://%s:%" PRIu16, str,
				ntohs(sin->sin_port));
		break;
	case FI_SOCKADDR_IN6:
sa_sin6:
		sin6 = addr;
		if (!inet_ntop(sin6->sin6_family, &sin6->sin6_addr, str,
			       sizeof(str)))
			return NULL;

		size = snprintf(buf, MIN(*len, sizeof(str)),
				"fi_sockaddr_in6://[%s]:%" PRIu16, str,
				ntohs(sin6->sin6_port));
		break;
	case FI_SOCKADDR_IB:
		size = snprintf(buf, *len, "fi_sockaddr_ib://%p", addr);
		break;
	case FI_ADDR_PSMX:
		size = snprintf(buf, *len, "fi_addr_psmx://%" PRIx64,
				*(uint64_t *)addr);
		break;
	case FI_ADDR_PSMX2:
		size =
		    snprintf(buf, *len, "fi_addr_psmx2://%" PRIx64 ":%" PRIx64,
			     *(uint64_t *)addr, *((uint64_t *)addr + 1));
		break;
	case FI_ADDR_GNI:
		size = snprintf(buf, *len, "fi_addr_gni://%" PRIx64,
				*(uint64_t *)addr);
		break;
	case FI_ADDR_BGQ:
		size = snprintf(buf, *len, "fi_addr_bgq://%p", addr);
		break;
	case FI_ADDR_MLX:
		size = snprintf(buf, *len, "fi_addr_mlx://%p", addr);
		break;
	case FI_ADDR_STR:
		size = snprintf(buf, *len, "%s", (const char *) addr);
		break;
	default:
		return NULL;
	}

	/* Make sure that possibly truncated messages have a null terminator. */
	if (buf && *len)
		buf[*len - 1] = '\0';
	*len = size + 1;
	return buf;
}

static uint32_t ofi_addr_format(const char *str)
{
	char fmt[16];
	int ret;

	ret = sscanf(str, "%16[^:]://", fmt);
	if (ret != 1)
		return FI_FORMAT_UNSPEC;

	fmt[sizeof(fmt) - 1] = '\0';
	if (!strcasecmp(fmt, "fi_sockaddr_in"))
		return FI_SOCKADDR_IN;
	else if (!strcasecmp(fmt, "fi_sockaddr_in6"))
		return FI_SOCKADDR_IN6;
	else if (!strcasecmp(fmt, "fi_sockaddr_ib"))
		return FI_SOCKADDR_IB;
	else if (!strcasecmp(fmt, "fi_addr_psmx"))
		return FI_ADDR_PSMX;
	else if (!strcasecmp(fmt, "fi_addr_psmx2"))
		return FI_ADDR_PSMX2;
	else if (!strcasecmp(fmt, "fi_addr_gni"))
		return FI_ADDR_GNI;
	else if (!strcasecmp(fmt, "fi_addr_bgq"))
		return FI_ADDR_BGQ;
	else if (!strcasecmp(fmt, "fi_addr_mlx"))
		return FI_ADDR_MLX;

	return FI_FORMAT_UNSPEC;
}

static int ofi_str_to_psmx(const char *str, void **addr, size_t *len)
{
	int ret;

	*len = sizeof(uint64_t);
	if (!(*addr = calloc(1, *len)))
		return -FI_ENOMEM;

	ret = sscanf(str, "%*[^:]://%" SCNx64, (uint64_t *) *addr);
	if (ret == 1)
		return 0;

	free(*addr);
	return -FI_EINVAL;
}

static int ofi_str_to_psmx2(const char *str, void **addr, size_t *len)
{
	int ret;

	*len = 2 * sizeof(uint64_t);
	if (!(*addr = calloc(1, *len)))
		return -FI_ENOMEM;

	ret = sscanf(str, "%*[^:]://%" SCNx64 ":%" SCNx64,
		     (uint64_t *) *addr, (uint64_t *) *addr + 1);
	if (ret == 2)
		return 0;

	free(*addr);
	return -FI_EINVAL;
}

static int ofi_str_to_sin(const char *str, void **addr, size_t *len)
{
	struct sockaddr_in *sin;
	char ip[64];
	int ret;

	*len = sizeof(*sin);
	if (!(sin = calloc(1, *len)))
		return -FI_ENOMEM;

	sin->sin_family = AF_INET;
	ret = sscanf(str, "%*[^:]://:%" SCNu16, &sin->sin_port);
	if (ret == 1)
		goto match_port;

	ret = sscanf(str, "%*[^:]://%64[^:]:%" SCNu16, ip, &sin->sin_port);
	if (ret == 2)
		goto match_ip;

	ret = sscanf(str, "%*[^:]://%64[^:/]", ip);
	if (ret == 1)
		goto match_ip;

err:
	free(sin);
	return -FI_EINVAL;

match_ip:
	ip[sizeof(ip) - 1] = '\0';
	ret = inet_pton(AF_INET, ip, &sin->sin_addr);
	if (ret != 1)
		goto err;

match_port:
	sin->sin_port = htons(sin->sin_port);
	*addr = sin;
	return 0;
}

static int ofi_str_to_sin6(const char *str, void **addr, size_t *len)
{
	struct sockaddr_in6 *sin6;
	char ip[64];
	int ret;

	*len = sizeof(*sin6);
	if (!(sin6 = calloc(1, *len)))
		return -FI_ENOMEM;

	sin6->sin6_family = AF_INET6;
	ret = sscanf(str, "%*[^:]://:%" SCNu16, &sin6->sin6_port);
	if (ret == 1)
		goto match_port;

	ret = sscanf(str, "%*[^:]://[%64[^]]]:%" SCNu16, ip, &sin6->sin6_port);
	if (ret == 2)
		goto match_ip;

	ret = sscanf(str, "%*[^:]://[%64[^]]", ip);
	if (ret == 1)
		goto match_ip;

err:
	free(sin6);
	return -FI_EINVAL;

match_ip:
	ip[sizeof(ip) - 1] = '\0';
	ret = inet_pton(AF_INET6, ip, &sin6->sin6_addr);
	if (ret != 1)
		goto err;

match_port:
	sin6->sin6_port = htons(sin6->sin6_port);
	*addr = sin6;
	return 0;
}

int ofi_str_toaddr(const char *str, uint32_t *addr_format,
		   void **addr, size_t *len)
{
	*addr_format = ofi_addr_format(str);
	if (*addr_format == FI_FORMAT_UNSPEC)
		return -FI_EINVAL;

	switch (*addr_format) {
	case FI_SOCKADDR_IN:
		return ofi_str_to_sin(str, addr, len);
	case FI_SOCKADDR_IN6:
		return ofi_str_to_sin6(str, addr, len);
	case FI_ADDR_PSMX:
		return ofi_str_to_psmx(str, addr, len);
	case FI_ADDR_PSMX2:
		return ofi_str_to_psmx2(str, addr, len);
	case FI_SOCKADDR_IB:
	case FI_ADDR_GNI:
	case FI_ADDR_BGQ:
	case FI_ADDR_MLX:
	default:
		return -FI_ENOSYS;
	}

	return 0;
}

const char *ofi_hex_str(const uint8_t *data, size_t len)
{
	static char str[64];
	const char hex[] = "0123456789abcdef";
	size_t i, p;

	if (len >= (sizeof(str) >> 1))
		len = (sizeof(str) >> 1) - 1;

	for (p = 0, i = 0; i < len; i++) {
		str[p++] = hex[data[i] >> 4];
		str[p++] = hex[data[i] & 0xF];
	}

	if (len == (sizeof(str) >> 1) - 1)
		str[p++] = '~';

	str[p] = '\0';
	return str;
}

int ofi_addr_cmp(const struct fi_provider *prov, const struct sockaddr *sa1,
		 const struct sockaddr *sa2)
{
	int cmp;

	switch (sa1->sa_family) {
	case AF_INET:
		cmp = memcmp(&ofi_sin_addr(sa1), &ofi_sin_addr(sa2),
			     sizeof(ofi_sin_addr(sa1)));
		return cmp ? cmp : memcmp(&ofi_sin_port(sa1),
					  &ofi_sin_port(sa2),
					  sizeof(ofi_sin_port(sa1)));
	case AF_INET6:
		cmp = memcmp(&ofi_sin6_addr(sa1), &ofi_sin6_addr(sa2),
			     sizeof(ofi_sin6_addr(sa1)));
		return cmp ? cmp : memcmp(&ofi_sin6_port(sa1),
					  &ofi_sin_port(sa2),
					  sizeof(ofi_sin6_port(sa1)));
	default:
		FI_WARN(prov, FI_LOG_FABRIC, "Invalid address format!\n");
		assert(0);
		return 0;
	}
}

#ifndef HAVE_EPOLL

int fi_epoll_create(struct fi_epoll **ep)
{
	*ep = calloc(1, sizeof(struct fi_epoll));
	return *ep ? 0 : -FI_ENOMEM;
}

int fi_epoll_add(struct fi_epoll *ep, int fd, void *context)
{
	struct pollfd *fds;
	void *contexts;

	if (ep->nfds == ep->size) {
		fds = calloc(ep->size + 64,
			     sizeof(*ep->fds) + sizeof(*ep->context));
		if (!fds)
			return -FI_ENOMEM;

		ep->size += 64;
		contexts = fds + ep->size;

		memcpy(fds, ep->fds, ep->nfds * sizeof(*ep->fds));
		memcpy(contexts, ep->context, ep->nfds * sizeof(*ep->context));
		free(ep->fds);
		ep->fds = fds;
		ep->context = contexts;
	}

	ep->fds[ep->nfds].fd = fd;
	ep->fds[ep->nfds].events = POLLIN;
	ep->context[ep->nfds++] = context;
	return 0;
}

int fi_epoll_del(struct fi_epoll *ep, int fd)
{
	int i;

	for (i = 0; i < ep->nfds; i++) {
		if (ep->fds[i].fd == fd) {
			ep->fds[i].fd = ep->fds[ep->nfds - 1].fd;
			ep->context[i] = ep->context[--ep->nfds];
      			return 0;
		}
  	}
	return -FI_EINVAL;
}

void *fi_epoll_wait(struct fi_epoll *ep, int timeout)
{
	int i, ret;

	ret = poll(ep->fds, ep->nfds, timeout);
	if (ret <= 0)
		return NULL;

	for (i = ep->index; i < ep->nfds; i++) {
		if (ep->fds[i].revents)
			goto found;
	}
	for (i = 0; i < ep->index; i++) {
		if (ep->fds[i].revents)
			goto found;
	}
	return NULL;
found:
	ep->index = i;
	return ep->context[i];
}

void fi_epoll_close(struct fi_epoll *ep)
{
	if (ep) {
		free(ep->fds);
		free(ep);
	}
}

#endif

#if HAVE_GETIFADDRS

/* getifaddrs can fail when connecting the netlink socket. Try again
 * as this is a temporary error. After the 2nd retry, sleep a bit as
 * well in case the host is really busy. */
#define MAX_GIA_RETRIES 10
int ofi_getifaddrs(struct ifaddrs **ifaddr)
{
	unsigned int retries;
	int ret;

	for (retries = 0; retries < MAX_GIA_RETRIES; retries++) {
		if (retries > 1) {
			/* Exponentiation sleep after the 2nd try.
			 * 1000 << 9 is 512000, which respects the 1s
			 * constraint for usleep. */
			usleep(1000 << retries);
		}

		ret = getifaddrs(ifaddr);
		if (ret == 0 || errno != ECONNREFUSED)
			break;
	}

	if (ret != 0)
		return -errno;

	return FI_SUCCESS;
}

#endif
