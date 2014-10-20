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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include <rdma/fabric.h>

static struct fi_info hints;
static char *node, *port;

/* options and matching help strings need to be kept in sync */

static const struct option longopts[] = {
	{"node", required_argument, NULL, 'n'},
	{"port", required_argument, NULL, 'p'},
	{"caps", required_argument, NULL, 'c'},
	{"mode", required_argument, NULL, 'm'},
	{"ep_type", required_argument, NULL, 'e'},
	{0,0,0,0}
};

static const char *help_strings[][2] = {
	{"NAME", "\t\tnode name or address"},
	{"PNUM", "\t\tport number"},
	{"CAP1|CAP2..", "\tone or more capabilities: FI_MSG|FI_RMA..."},
	{"MOD1|MOD2..", "\tone or more modes, default all modes"},
	{"EPTYPE", "\t\tspecify single endpoint type: FI_EP_MSG, FI_EP_DGRAM..."},
	{"", ""}
};

void usage()
{
	int i = 0;
	const struct option *ptr = longopts;

	for (; ptr->name != NULL; ++i, ptr = &longopts[i])
		printf("  -%c, --%s=%s%s\n", ptr->val, ptr->name,
			help_strings[i][0], help_strings[i][1]);
}

#define ORCASE(SYM) \
	do { if (strcmp(#SYM, inputstr) == 0) return SYM; } while (0);

uint64_t str2cap(char *inputstr)
{
	ORCASE(FI_MSG);
	ORCASE(FI_RMA);
	ORCASE(FI_TAGGED);
	ORCASE(FI_ATOMICS);
	ORCASE(FI_MULTICAST);
	ORCASE(FI_DYNAMIC_MR);
	ORCASE(FI_NAMED_RX_CTX);
	ORCASE(FI_BUFFERED_RECV);
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
	ORCASE(FI_REMOTE_SIGNAL);
	ORCASE(FI_REMOTE_COMPLETE);
	ORCASE(FI_CANCEL);
	ORCASE(FI_MORE);
	ORCASE(FI_PEEK);
	ORCASE(FI_TRIGGER);

	return 0;
}

uint64_t str2mode(char *inputstr)
{
	ORCASE(FI_CONTEXT);
	ORCASE(FI_LOCAL_MR);
	ORCASE(FI_WRITE_NONCOHERENT);
	ORCASE(FI_PROV_MR_KEY);
	ORCASE(FI_MSG_PREFIX);

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

	ret = fi_getinfo(FI_VERSION(1, 0), node, port, 0, hints, &info);
	if (ret) {
		printf("fi_getinfo %s\n", strerror(-ret));
		return ret;
	}

	for (cur = info; cur; cur = cur->next)
		printf("%s\n", fi_tostr(cur, FI_TYPE_INFO));

	fi_freeinfo(info);
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int op;

	hints.mode = ~0;

	while ((op = getopt_long(argc, argv, "n:p:c:m:e:h", longopts, NULL)) != -1) {
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
		case 'h':
		default:
			printf("usage: %s\n", argv[0]);
			usage();
			return (EXIT_FAILURE);
		}
	}

	return run(&hints, node, port);
}
