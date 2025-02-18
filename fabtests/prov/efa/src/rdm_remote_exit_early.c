/*
 * Copyright (c) 2021, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This test check the error handling path for receiver when sender exit early
 * in the middle of send.
 */

#include <getopt.h>
#include <shared.h>
#include <stdio.h>
#include <stdlib.h>

static int run(bool server_post_rx)
{
	int ret;
	struct timespec a, b;

	ret = ft_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_init_fabric", -ret);
		return ret;
	}

	/*
	 * sync will use FI_DELIVERY_COMPLETE which will
	 * enforce a handshake between client and server
	 */
	ret = ft_sync_inband(false);
	if (ret) {
		FT_PRINTERR("ft_sync_inband", -ret);
		goto out;
	}

	/* client post a send and then quit */
	if (opts.dst_addr) {
		ret = ft_post_tx(ep, remote_fi_addr, opts.transfer_size,
				 NO_CQ_DATA, &tx_ctx);
	} else {
		/* server post a recv and wait for completion, it should get an
		 * cq error as client exit early in a long protocol
		 */
		if (server_post_rx)
			ret = fi_recv(ep, rx_buf, rx_size, mr_desc,
				      FI_ADDR_UNSPEC, &rx_ctx);

		clock_gettime(CLOCK_MONOTONIC, &a);

		do {
			struct fi_cq_data_entry comp = {0};
			struct fi_cq_err_entry comp_err = {0};

			ret = fi_cq_read(rxcq, &comp, 1);
			clock_gettime(CLOCK_MONOTONIC, &b);
			/* When server posts a recv, we expect to
			 * get a cq entry or cq error.
			 * If no recv is posted, it should just
			 * poll some cq in the timeout range
			 * and exit.
			 */
			if ((b.tv_sec - a.tv_sec) > timeout) {
				if (server_post_rx) {
					fprintf(stderr, "%ds timeout expired\n",
						timeout);
					ret = -FI_ENODATA;
				} else {
					ret = 0;
				}
				goto out;
			}

			if (ret == -FI_EAGAIN) {
				continue;
			} else if (ret == -FI_EAVAIL) {
				ret = fi_cq_readerr(rxcq, &comp_err, FI_SEND);
				if (ret < 0 && ret != -FI_EAGAIN) {
					FT_PRINTERR("fi_cq_readerr", -ret);
					goto out;
				} else if (ret == 1) {
					printf("Got CQ entry as expected: %d, "
					       "%s\n",
					       comp_err.err,
					       fi_strerror(comp_err.err));
					ret = 0;
					goto out;
				}
			} else if (ret < 0 && ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_read", -ret);
				goto out;
			}
		} while (ret < 1);
	}

out:
	ft_free_res();

	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;
	int retv = 0;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	timeout = 5;
	while ((op = getopt(argc, argv, ADDR_OPTS INFO_OPTS CS_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "RDM remote exit early test");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	hints->mode |= FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;

	printf("Test 1: server posted receive\n");
	ret = run(true);
	if (ret) {
		FT_PRINTERR("Test 1 failed", -ret);
		retv = ret;
	} else {
		printf("Test 1 succeeded\n");
	}

	printf("Test 2: server doesn't post receive\n");
	ret = run(true);
	if (ret) {
		FT_PRINTERR("Test 2 failed", -ret);
		retv = ret;
	} else {
		printf("Test 2 succeeded\n");
	}

	return ft_exit_code(retv);
}
