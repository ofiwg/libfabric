/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
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
#include <getopt.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include "shared.h"


static struct fi_info hints, *info;
static char *node, *port;


static int run(void)
{
	struct fi_info *cur;
	int ret;

	ret = fi_getinfo(FI_VERSION(1, 0), node, port, 0, &hints, &info);
	if (ret) {
		printf("fi_getinfo %s\n", strerror(-ret));
		return ret;
	}

	for (cur = info; cur; cur = cur->next)
		printf("%s\n", fi_tostr(cur, FI_PP_INFO));

	fi_freeinfo(info);
	return 0;
}

static uint64_t ep_type(char *arg)
{
	if (!strcasecmp(arg, "msg"))
		return FI_EP_MSG;
	else if (!strcasecmp(arg, "rdm"))
		return FI_EP_RDM;
	else if (!strcasecmp(arg, "dgram"))
		return FI_EP_DGRAM;
	else
		return FI_EP_UNSPEC;
}

int main(int argc, char **argv)
{
	int op, ret;

	hints.ep_cap = FI_MSG;

	while ((op = getopt(argc, argv, "e:n:p:")) != -1) {
		switch (op) {
		case 'e':
			hints.type = ep_type(optarg);
			break;
		case 'n':
			node = optarg;
			break;
		case 's':
			port = optarg;
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-e ep_type\n");
			printf("\t    (msg, dgram, rdm)");
			printf("\t[-n node]\n");
			printf("\t[-p service_port]\n");
			exit(1);
		}
	}

	ret = run();
	return ret;
}
