/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * fi_directed_recv: matching order of directed and FI_ADDR_UNSPEC receives.
 *
 * With FI_DIRECTED_RECV, the server posts a mix of receives from the client
 * and from FI_ADDR_UNSPEC before the client sends anything. The client then
 * sends one message per receive, in order (FI_ORDER_SAS), with the message
 * index plus one as remote CQ data (ft_post_tx_buf treats 0 as no data).
 * Every receive can match every message, and posted receives are searched
 * in the order in which they were posted (fi_msg(3), fi_tagged(3)), so
 * receive i must get message i.
 *
 * Untagged by default; -o tagged runs the same patterns with tagged
 * receives that all use one tag.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>
#include <inttypes.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_cm.h>

#include "shared.h"

/* Every send and tagged receive uses this one tag, so the tag never decides
 * which receive a message matches; only the posting order does. */
#define DR_TAG 0x42

/* Order in which the server posts its receives, one letter per receive:
 * 'U' = FI_ADDR_UNSPEC, 'D' = directed to the client. A directed receive
 * that is not the oldest one must not win over an older unspec receive,
 * and the other way round. */
static const char *patterns[] = { "UD", "DU", "UDUD", "DUDU" };
/* Longest pattern; also the number of message buffers in each window. */
#define DR_MAX 4

/* ft_alloc_msgs lays out opts.window_size buffers of tx_size/rx_size bytes
 * back to back; slot i holds message i or receive i. */
static char *get_tx_buf(int i)
{
	return tx_buf + tx_size * i;
}

static char *get_rx_buf(int i)
{
	return rx_buf + rx_size * i;
}

/* Read one receive completion. The entry is needed (op_context and CQ
 * data), so this reads the CQ directly instead of ft_get_rx_comp(). */
static int wait_rx_comp(struct fi_cq_tagged_entry *comp)
{
	struct fi_cq_err_entry err = {0};
	int ret;

	do {
		ret = fi_cq_read(rxcq, comp, 1);
	} while (ret == -FI_EAGAIN);
	if (ret == 1)
		return 0;
	if (ret == -FI_EAVAIL) {
		fi_cq_readerr(rxcq, &err, 0);
		FT_ERR("receive completion error: %s (prov_errno %d)",
		       fi_strerror(err.err), err.prov_errno);
		return -err.err;
	}
	FT_PRINTERR("fi_cq_read", ret);
	return ret;
}

/* Receiver side of one pattern: post receive i into slot i as the pattern
 * says, let the client send, then check that receive i got message i. */
static int run_server(const char *pattern)
{
	struct fi_cq_tagged_entry comp;
	uint64_t got[DR_MAX];
	int n = strlen(pattern), i, j, ret, fail = 0;

	for (i = 0; i < n; i++) {
		ret = ft_post_rx_buf(ep, pattern[i] == 'D' ? remote_fi_addr :
							     FI_ADDR_UNSPEC,
				     opts.transfer_size, &rx_ctx_arr[i].context,
				     get_rx_buf(i), mr_desc, DR_TAG);
		if (ret)
			return ret;
	}

	/* All receives are posted before the client sends. */
	ret = ft_sync();
	if (ret)
		return ret;

	/* Completions may be reported in any order. Find the receive each one
	 * belongs to by its context, and record which message it got from
	 * the CQ data the client sent (message index plus one). */
	for (i = 0; i < n; i++) {
		ret = wait_rx_comp(&comp);
		if (ret)
			return ret;
		for (j = 0; j < n; j++)
			if (comp.op_context == &rx_ctx_arr[j].context)
				break;
		if (j == n || !(comp.flags & FI_REMOTE_CQ_DATA)) {
			FT_ERR("unexpected receive completion");
			return -FI_EOTHER;
		}
		got[j] = comp.data - 1;
	}

	/* The ordering check. Every message from the client can match every
	 * posted receive, and posted receives are searched in the order they
	 * were posted, so receive i must hold message i. Log every receive
	 * that does not. */
	for (i = 0; i < n; i++) {
		if (got[i] != (uint64_t) i) {
			FT_ERR("%s %s: receive %d (%c) got message %" PRIu64
			       ", expected %d",
			       hints->caps & FI_TAGGED ? "tagged" : "untagged",
			       pattern, i, pattern[i], got[i], i);
			fail = 1;
		}
	}
	if (fail)
		return -FI_EOTHER;

	/* With -v, also check the payload itself. ft_fill_buf writes the same
	 * stream into every message, so this cannot tell messages apart; the
	 * CQ data check above is what catches a wrong match. */
	if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
		for (i = 0; i < n; i++) {
			ret = ft_check_buf(get_rx_buf(i) + ft_rx_prefix_size(),
					   opts.transfer_size);
			if (ret)
				return ret;
		}
	}

	/* Let the client go on to the next pattern. */
	return ft_sync();
}

/* Sender side of one pattern: once the server has posted all its
 * receives, send one message per receive, in order, tagging message i
 * with CQ data i + 1 (ft_post_tx_buf treats 0 as no data). */
static int run_client(const char *pattern)
{
	int n = strlen(pattern), i, ret;

	/* Wait until the server has posted all its receives. */
	ret = ft_sync();
	if (ret)
		return ret;

	for (i = 0; i < n; i++) {
		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ret = ft_fill_buf(get_tx_buf(i) + ft_tx_prefix_size(),
					  opts.transfer_size);
			if (ret)
				return ret;
		}
		ret = ft_post_tx_buf(ep, remote_fi_addr, opts.transfer_size, i + 1,
				     &tx_ctx_arr[i].context, get_tx_buf(i),
				     mr_desc, DR_TAG);
		if (ret)
			return ret;
	}

	ret = ft_get_tx_comp(tx_seq);
	if (ret)
		return ret;

	/* Wait until the server has checked this pattern. */
	return ft_sync();
}

static int run_test(void)
{
	size_t p;
	int ret;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	/* Without FI_DIRECTED_RECV a directed receive is treated as unspec,
	 * so the test would not cover anything. */
	if (!(fi->caps & FI_DIRECTED_RECV)) {
		FT_ERR("provider did not grant FI_DIRECTED_RECV");
		return -FI_ENODATA;
	}

	/* One round trip completes the receive ft_init_fabric posted on each
	 * side, without reposting it, so only the test's receives are left. */
	if (opts.dst_addr) {
		ret = ft_tx(ep, remote_fi_addr, opts.transfer_size, &tx_ctx);
		if (!ret)
			ret = ft_get_rx_comp(rx_seq);
	} else {
		ret = ft_get_rx_comp(rx_seq);
		if (!ret)
			ret = ft_tx(ep, remote_fi_addr, opts.transfer_size,
				    &tx_ctx);
	}
	if (ret)
		return ret;

	for (p = 0; p < ARRAY_SIZE(patterns); p++) {
		ret = opts.dst_addr ? run_client(patterns[p]) :
				      run_server(patterns[p]);
		if (ret)
			return ret;
	}
	return 0;
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.transfer_size = 64;
	/* One message buffer per receive of the longest pattern. */
	opts.window_size = DR_MAX;
	opts.options |= FT_OPT_OOB_CTRL | FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "vUh" CS_OPTS INFO_OPTS API_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parsecsopts(op, optarg, &opts);
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parse_api_opts(op, optarg, hints, &opts);
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA;
			break;
		case 'U':
			hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "Matching order of directed and FI_ADDR_UNSPEC receives.");
			FT_PRINT_OPTS_USAGE("-o <op>", "msg (default) or tagged");
			FT_PRINT_OPTS_USAGE("-v", "enable data verification");
			FT_PRINT_OPTS_USAGE("-U", "transmit with FI_DELIVERY_COMPLETE");
			ft_longopts_usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG | FI_DIRECTED_RECV;
	/* Messages must arrive in the order sent for "receive i gets message
	 * i" to hold. */
	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
	/* The message index travels as remote CQ data. */
	hints->domain_attr->cq_data_size = 4;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;
	/* A format that carries CQ data for both msg and tagged receives. */
	cq_attr.format = FI_CQ_FORMAT_TAGGED;

	ret = run_test();

	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
