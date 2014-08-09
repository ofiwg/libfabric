/*
 * Copyright (c) 2013,2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the OpenIB.org BSD license
 * below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <errno.h>
#include <netdb.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "shared.h"


int getaddr(char *node, char *service, struct sockaddr **addr, socklen_t *len)
{
	struct addrinfo *ai;
	int ret;

	ret = getaddrinfo(node, service, NULL, &ai);
	if (ret)
		return ret;

	if ((*addr = malloc(ai->ai_addrlen))) {
		memcpy(*addr, ai->ai_addr, ai->ai_addrlen);
		*len = ai->ai_addrlen;
	} else {
		ret = EAI_MEMORY;
	}

	freeaddrinfo(ai);
	return ret;
}

void size_str(char *str, size_t ssize, long long size)
{
	long long base, fraction = 0;
	char mag;

	if (size >= (1 << 30)) {
		base = 1 << 30;
		mag = 'g';
	} else if (size >= (1 << 20)) {
		base = 1 << 20;
		mag = 'm';
	} else if (size >= (1 << 10)) {
		base = 1 << 10;
		mag = 'k';
	} else {
		base = 1;
		mag = '\0';
	}

	if (size / base < 10)
		fraction = (size % base) * 10 / base;
	if (fraction) {
		snprintf(str, ssize, "%lld.%lld%c", size / base, fraction, mag);
	} else {
		snprintf(str, ssize, "%lld%c", size / base, mag);
	}
}

void cnt_str(char *str, size_t ssize, long long cnt)
{
	if (cnt >= 1000000000)
		snprintf(str, ssize, "%lldb", cnt / 1000000000);
	else if (cnt >= 1000000)
		snprintf(str, ssize, "%lldm", cnt / 1000000);
	else if (cnt >= 1000)
		snprintf(str, ssize, "%lldk", cnt / 1000);
	else
		snprintf(str, ssize, "%lld", cnt);
}

int size_to_count(int size)
{
	if (size >= (1 << 20))
		return 100;
	else if (size >= (1 << 16))
		return 1000;
	else if (size >= (1 << 10))
		return 10000;
	else
		return 100000;
}
