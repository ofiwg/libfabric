/*
 * Copyright (c) 2025, Amazon.com, Inc.  All rights reserved.
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
#include <rdma/fi_tagged.h>

static bool post_rx = false;

enum {
	LONG_OPT_POST_RX,
};

static int run()
{
	int ret;

	ret = ft_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_init_fabric", -ret);
		return ret;
	}

	/*
	 * The handshake procedure between server and client will happen in
	 * either ft_exchange_keys() or ft_sync()
	 */
	if (opts.rma_op == FT_RMA_WRITEDATA) {
		/* ft_exchange_keys finally calls ft_sync(), 
		 * which will call ft_sync_inband(true), so it will cause the 
		 * receiver to always post an outstanding recv buffer. 
		 * This is ok for RMA test because the rx buffer post 
		 * shouldn't matter here, unless we will support FI_RX_CQ_DATA
		 */
		ret = ft_exchange_keys(&remote);
		if (ret) {
			FT_PRINTERR("ft_exchange_keys()", -ret);
			goto out;
		}
	} else {
		ret = ft_sync_inband(false);
		if (ret) {
			FT_PRINTERR("ft_sync_inband", -ret);
			goto out;
		}
	}

	if (hints->caps & FI_TAGGED)
		ft_tag = 0xabcd;

	/* client post a send/writedata and then quit */
	if (opts.dst_addr) {
		if (opts.rma_op == FT_RMA_WRITEDATA)
			ret = ft_post_rma(FT_RMA_WRITEDATA, tx_buf,
					  opts.transfer_size, &remote, &tx_ctx);
		else
			ret = ft_post_tx(ep, remote_fi_addr, opts.transfer_size,
					 NO_CQ_DATA, &tx_ctx);
		printf("client exits early\n");
	} else {
		/* server post a recv and wait for completion, it should get an
		 * cq error as client exit early in a long protocol
		 */
		if (post_rx) {
			if (hints->caps & FI_TAGGED)
				ret = fi_trecv(ep, rx_buf, rx_size, mr_desc,
				      FI_ADDR_UNSPEC, ft_tag, 0x0, &rx_ctx);
			else
				ret = fi_recv(ep, rx_buf, rx_size, mr_desc,
				      FI_ADDR_UNSPEC, &rx_ctx);
			printf("server posts recv\n");
		}

		ft_start();

		do {
			struct fi_cq_data_entry comp = {0};
			struct fi_cq_err_entry comp_err = {0};

			ret = fi_cq_read(rxcq, &comp, 1);
			if (ret == 1) {
				printf("server gets CQ entry successfully\n");
				ret = 0;
				goto out;
			}

			ft_stop();
			/* When server posts a recv, we expect to
			 * get a cq entry or cq error.
			 * If no recv is posted, it should just
			 * poll some cq in the timeout range
			 * and exit.
			 */
			if ((end.tv_sec - start.tv_sec) > timeout) {
				if (post_rx) {
					fprintf(stderr, "%ds timeout expired\n",
						timeout);
					ret = -FI_ENODATA;
				} else {
					printf("server polls cq and exits\n");
					ret = 0;
				}
				goto out;
			}

			if (ret == -FI_EAGAIN) {
				continue;
			} else if (ret == -FI_EAVAIL) {
				ret = fi_cq_readerr(rxcq, &comp_err, 0);
				printf("server posts fi_cq_readerr, ret = %d\n", ret);
				if (ret < 0 && ret != -FI_EAGAIN) {
					FT_PRINTERR("fi_cq_readerr", -ret);
					goto out;
				} else if (ret == 1) {
					printf("server gets CQ err entry as expected: %d, "
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

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	timeout = 5;
	int lopt_idx = 0;
	struct option long_opts[] = {
		{"post-rx", no_argument, NULL, LONG_OPT_POST_RX},
		{0, 0, 0, 0}
	};
	while ((op = getopt_long(argc, argv, ADDR_OPTS INFO_OPTS CS_OPTS API_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ret = ft_parse_api_opts(op, optarg, hints, &opts);
			if (ret)
				return ret;
			break;
		case LONG_OPT_POST_RX:
			post_rx = true;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "RDM remote exit early test");
			FT_PRINT_OPTS_USAGE("-o <op>", "op: tagged|writedata.\n");
			FT_PRINT_OPTS_USAGE( "--post-rx",
					    "Receiver posts fi_recv. "
						"By default receiver does not post receive.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];
	
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG | FI_RMA;
	hints->domain_attr->mr_mode = opts.mr_mode;
	
	ret = run();
	if (ret)
		FT_PRINTERR("Test failed", -ret);

	return ft_exit_code(ret);
}
