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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>

static struct fi_info *hints;
static char *node, *port;
static int ver = 0;
static int verbose = 0, env = 0;

/* options and matching help strings need to be kept in sync */

static const struct option longopts[] = {
	{"node", required_argument, NULL, 'n'},
	{"port", required_argument, NULL, 'p'},
	{"caps", required_argument, NULL, 'c'},
	{"mode", required_argument, NULL, 'm'},
	{"ep_type", required_argument, NULL, 't'},
	{"addr_format", required_argument, NULL, 'a'},
	{"provider", required_argument, NULL, 'f'},
	{"env", no_argument, NULL, 'e'},
	{"verbose", no_argument, NULL, 'v'},
	{"version", no_argument, &ver, 1},
	{0,0,0,0}
};

static const char *help_strings[][2] = {
	{"NAME", "\t\tnode name or address"},
	{"PNUM", "\t\tport number"},
	{"CAP1|CAP2..", "\tone or more capabilities: FI_MSG|FI_RMA..."},
	{"MOD1|MOD2..", "\tone or more modes, default all modes"},
	{"EPTYPE", "\t\tspecify single endpoint type: FI_EP_MSG, FI_EP_DGRAM..."},
	{"FMT", "\t\tspecify accepted address format: FI_FORMAT_UNSPEC, FI_SOCKADDR..."},
	{"PROV", "\t\tspecify provider explicitly"},
	{"", "\t\tprint libfabric environment variables"},
	{"", "\t\tverbose output"},
	{"", "\t\tprint version info and exit"},
	{"", ""}
};

static void usage(void)
{
	int i = 0;
	const struct option *ptr = longopts;

	for (; ptr->name != NULL; ++i, ptr = &longopts[i])
		if (ptr->has_arg == required_argument)
			printf("  -%c, --%s=%s%s\n", ptr->val, ptr->name,
				help_strings[i][0], help_strings[i][1]);
		else if (ptr->flag != NULL)
			printf("  --%s\t%s\n", ptr->name,
				help_strings[i][1]);
		else
			printf("  -%c, --%s\t%s\n", ptr->val, ptr->name,
				help_strings[i][1]);
}

#define ORCASE(SYM) \
	do { if (strcmp(#SYM, inputstr) == 0) return SYM; } while (0);

static uint64_t str2cap(char *inputstr)
{
	ORCASE(FI_MSG);
	ORCASE(FI_RMA);
	ORCASE(FI_TAGGED);
	ORCASE(FI_ATOMIC);

	ORCASE(FI_READ);
	ORCASE(FI_WRITE);
	ORCASE(FI_RECV);
	ORCASE(FI_SEND);
	ORCASE(FI_REMOTE_READ);
	ORCASE(FI_REMOTE_WRITE);

	ORCASE(FI_MULTI_RECV);
	ORCASE(FI_REMOTE_CQ_DATA);
	ORCASE(FI_MORE);
	ORCASE(FI_PEEK);
	ORCASE(FI_TRIGGER);
	ORCASE(FI_FENCE);

	ORCASE(FI_EVENT);
	ORCASE(FI_INJECT);
	ORCASE(FI_INJECT_COMPLETE);
	ORCASE(FI_TRANSMIT_COMPLETE);
	ORCASE(FI_DELIVERY_COMPLETE);

	ORCASE(FI_RMA_EVENT);
	ORCASE(FI_NAMED_RX_CTX);
	ORCASE(FI_SOURCE);
	ORCASE(FI_DIRECTED_RECV);

	return 0;
}

static uint64_t str2mode(char *inputstr)
{
	ORCASE(FI_CONTEXT);
	ORCASE(FI_LOCAL_MR);
	ORCASE(FI_MSG_PREFIX);
	ORCASE(FI_ASYNC_IOV);
	ORCASE(FI_RX_CQ_DATA);

	return 0;
}

static enum fi_ep_type str2ep_type(char *inputstr)
{
	ORCASE(FI_EP_UNSPEC);
	ORCASE(FI_EP_MSG);
	ORCASE(FI_EP_DGRAM);
	ORCASE(FI_EP_RDM);

	/* probably not the right thing to do? */
	return FI_EP_UNSPEC;
}

static uint32_t str2addr_format(char *inputstr)
{
	ORCASE(FI_FORMAT_UNSPEC);
	ORCASE(FI_SOCKADDR);
	ORCASE(FI_SOCKADDR_IN);
	ORCASE(FI_SOCKADDR_IN6);
	ORCASE(FI_SOCKADDR_IB);
	ORCASE(FI_ADDR_PSMX);

	return FI_FORMAT_UNSPEC;
}

static uint64_t tokparse(char *caps, uint64_t (*str2flag) (char *inputstr))
{
	uint64_t flags = 0;
	char *tok;

	for (tok = strtok(caps, "|"); tok != NULL; tok = strtok(NULL, "|"))
		flags |= str2flag(tok);

	return flags;
}

static const char *param_type(enum fi_param_type type)
{
	switch (type) {
	case FI_PARAM_STRING:
		return "String";
	case FI_PARAM_INT:
		return "Integer";
	case FI_PARAM_BOOL:
		return "Boolean (0/1, on/off, true/false, yes/no)";
	default:
		return "Unknown";
	}
}

static int print_vars(void)
{
	int ret, count;
	struct fi_param *params;
	char delim;

	ret = fi_getparams(&params, &count);
	if (ret)
		return ret;

	for (int i = 0; i < count; ++i) {
		printf("# %s: %s\n", params[i].name, param_type(params[i].type));
		printf("# %s\n", params[i].help_string);

		if (params[i].value) {
			delim = strchr(params[i].value, ' ') ? '"' : '\0';
			printf("%s=%c%s%c\n", params[i].name, delim,
				params[i].value, delim);
		}

		printf("\n");
	}

	fi_freeparams(params);
	return ret;
}

static int print_short_info(struct fi_info *info)
{
	for (struct fi_info *cur = info; cur; cur = cur->next) {
		printf("%s: %s\n", cur->fabric_attr->prov_name, cur->fabric_attr->name);
		printf("    version: %d.%d\n", FI_MAJOR(cur->fabric_attr->prov_version),
			FI_MINOR(cur->fabric_attr->prov_version));
		printf("    type: %s\n", fi_tostr(&cur->ep_attr->type, FI_TYPE_EP_TYPE));
		printf("    protocol: %s\n", fi_tostr(&cur->ep_attr->protocol, FI_TYPE_PROTOCOL));
	}
	return EXIT_SUCCESS;
}

static int print_long_info(struct fi_info *info)
{
	for (struct fi_info *cur = info; cur; cur = cur->next) {
		printf("---\n");
		printf("%s", fi_tostr(cur, FI_TYPE_INFO));
	}
	return EXIT_SUCCESS;
}

static int run(struct fi_info *hints, char *node, char *port)
{
	struct fi_info *info;
	int ret;

	ret = fi_getinfo(FI_VERSION(1, 1), node, port, 0, hints, &info);
	if (ret) {
		fprintf(stderr, "fi_getinfo: %d\n", ret);
		return ret;
	}

	if (verbose)
		ret = print_long_info(info);
	else
		ret = print_short_info(info);

	fi_freeinfo(info);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret, option_index;
	int use_hints = 0;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->mode = ~0;

	while ((op = getopt_long(argc, argv, "n:p:c:m:t:a:f:ehv", longopts, &option_index)) != -1) {
		switch (op) {
		case 0:
			/* If --verbose set a flag, do nothing. */
			if (longopts[option_index].flag != 0)
				break;
		case 'n':
			node = optarg;
			break;
		case 'p':
			port = optarg;
			break;
		case 'c':
			hints->caps = tokparse(optarg, str2cap);
			use_hints = 1;
			break;
		case 'm':
			hints->mode = tokparse(optarg, str2mode);
			use_hints = 1;
			break;
		case 't':
			hints->ep_attr->type = str2ep_type(optarg);
			use_hints = 1;
			break;
		case 'a':
			hints->addr_format = str2addr_format(optarg);
			use_hints = 1;
			break;
		case 'f':
			free(hints->fabric_attr->prov_name);
			hints->fabric_attr->prov_name = strdup(optarg);
			use_hints = 1;
			break;
		case 'e':
			env = 1;
			break;
		case 'v':
			verbose = 1;
			break;
		case 'h':
		default:
			printf("Usage: %s\n", argv[0]);
			usage();
			return EXIT_FAILURE;
		}
	}

	if (ver) {
		printf("%s: %s\n", argv[0], PACKAGE_VERSION);
		printf("libfabric: %s\n", fi_tostr("1", FI_TYPE_VERSION));
		printf("libfabric api: %d.%d\n", FI_MAJOR_VERSION, FI_MINOR_VERSION);
		return EXIT_SUCCESS;
	}

	if (env) {
		ret = print_vars();
		goto out;
	}

	ret = run(use_hints ? hints : NULL, node, port);
out:
	fi_freeinfo(hints);
	return -ret;
}
