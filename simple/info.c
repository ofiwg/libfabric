/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>

#include "shared.h"

static struct fi_info hints;
static char *node, *port;

/* options and matching help strings need to be kept in sync */

static const struct option longopts[] = {
	{"node", required_argument, NULL, 'n'},
	{"port", required_argument, NULL, 'p'},
	{"caps", required_argument, NULL, 'c'},
	{"mode", required_argument, NULL, 'm'},
	{"ep_type", required_argument, NULL, 'e'},
	{"addr_format", required_argument, NULL, 'a'},
	{"version", no_argument, NULL, 'v'},
	{0,0,0,0}
};

static const char *help_strings[][2] = {
	{"NAME", "\t\tnode name or address"},
	{"PNUM", "\t\tport number"},
	{"CAP1|CAP2..", "\tone or more capabilities: FI_MSG|FI_RMA..."},
	{"MOD1|MOD2..", "\tone or more modes, default all modes"},
	{"EPTYPE", "\t\tspecify single endpoint type: FI_EP_MSG, FI_EP_DGRAM..."},
	{"FMT", "\t\tspecify accepted address format: FI_FORMAT_UNSPEC, FI_SOCKADDR..."},
	{"", "\t\tprint version info and exit"},
	{"", ""}
};

void usage()
{
	int i = 0;
	const struct option *ptr = longopts;

	for (; ptr->name != NULL; ++i, ptr = &longopts[i])
		if (ptr->has_arg)
			printf("  -%c, --%s=%s%s\n", ptr->val, ptr->name,
				help_strings[i][0], help_strings[i][1]);
		else
			printf("  -%c, --%s\t%s\n", ptr->val, ptr->name,
				help_strings[i][1]);
}

#define ORCASE(SYM) \
	do { if (strcmp(#SYM, inputstr) == 0) return SYM; } while (0);

uint64_t str2cap(char *inputstr)
{
	ORCASE(FI_MSG);
	ORCASE(FI_RMA);
	ORCASE(FI_TAGGED);
	ORCASE(FI_ATOMICS);
	ORCASE(FI_DYNAMIC_MR);
	ORCASE(FI_NAMED_RX_CTX);
	ORCASE(FI_BUFFERED_RECV);
	ORCASE(FI_DIRECTED_RECV);
	ORCASE(FI_INJECT);
	ORCASE(FI_MULTI_RECV);
	ORCASE(FI_SOURCE);
	ORCASE(FI_SYMMETRIC);
	ORCASE(FI_READ);
	ORCASE(FI_WRITE);
	ORCASE(FI_RECV);
	ORCASE(FI_SEND);
	ORCASE(FI_REMOTE_READ);
	ORCASE(FI_REMOTE_WRITE);
	ORCASE(FI_REMOTE_CQ_DATA);
	ORCASE(FI_EVENT);
	ORCASE(FI_COMPLETION);
	ORCASE(FI_REMOTE_SIGNAL);
	ORCASE(FI_REMOTE_COMPLETE);
	ORCASE(FI_CANCEL);
	ORCASE(FI_MORE);
	ORCASE(FI_PEEK);
	ORCASE(FI_TRIGGER);
	ORCASE(FI_FENCE);

	return 0;
}

uint64_t str2mode(char *inputstr)
{
	ORCASE(FI_CONTEXT);
	ORCASE(FI_LOCAL_MR);
	ORCASE(FI_PROV_MR_ATTR);
	ORCASE(FI_MSG_PREFIX);
	ORCASE(FI_ASYNC_IOV);

	return 0;
}

enum fi_ep_type str2ep_type(char *inputstr)
{
	ORCASE(FI_EP_UNSPEC);
	ORCASE(FI_EP_MSG);
	ORCASE(FI_EP_DGRAM);
	ORCASE(FI_EP_RDM);

	/* probably not the right thing to do? */
	return FI_EP_UNSPEC;
}

uint32_t str2addr_format(char *inputstr)
{
	ORCASE(FI_FORMAT_UNSPEC);
	ORCASE(FI_SOCKADDR);
	ORCASE(FI_SOCKADDR_IN);
	ORCASE(FI_SOCKADDR_IN6);
	ORCASE(FI_SOCKADDR_IB);
	ORCASE(FI_ADDR_PSMX);

	return FI_FORMAT_UNSPEC;
}

uint64_t tokparse(char *caps, uint64_t (*str2flag) (char *inputstr))
{
	uint64_t flags = 0;
	char *tok;

	for (tok = strtok(caps, "|"); tok != NULL; tok = strtok(NULL, "|"))
		flags |= str2flag(tok);

	return flags;
}

static int run(struct fi_info *hints, char *node, char *port)
{
	struct fi_info *cur;
	struct fi_info *info;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, node, port, 0, hints, &info);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	for (cur = info; cur; cur = cur->next) {
		printf("---\n");
		printf("%s", fi_tostr(cur, FI_TYPE_INFO));
	}

	fi_freeinfo(info);
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int op;

	hints.mode = ~0;

	while ((op = getopt_long(argc, argv, "n:p:c:m:e:a:hv", longopts, NULL)) != -1) {
		switch (op) {
		case 'n':
			node = optarg;
			break;
		case 'p':
			port = optarg;
			break;
		case 'c':
			hints.caps = tokparse(optarg, str2cap);
			break;
		case 'm':
			hints.mode = tokparse(optarg, str2mode);
			break;
		case 'e':
			hints.ep_type = str2ep_type(optarg);
			break;
		case 'a':
			hints.addr_format = str2addr_format(optarg);
			break;
		case 'v':
			printf("%s: %s\n", argv[0], PACKAGE_VERSION);
			printf("libfabric: %s\n", fi_tostr("1", FI_TYPE_VERSION));
			printf("libfabric api: %d.%d\n", FI_MAJOR(FT_FIVERSION), FI_MINOR(FT_FIVERSION));
			return EXIT_SUCCESS;
		case 'h':
		default:
			printf("Usage: %s\n", argv[0]);
			usage();
			return EXIT_FAILURE;
		}
	}

	return run(&hints, node, port);
}
