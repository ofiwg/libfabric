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
 * This program tests the RNR retry counter reset via fi_setopt.
 * When running the test, use `-R` option to specify RNR retry counter.
 * The valid values are 0 - 7 (7 inidicates infinite retry on firmware).
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <shared.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_ext.h>

static int poll_rnr_cq_error(void)
{
	struct fi_cq_data_entry comp;
	struct fi_cq_err_entry comp_err;
	int total_send, expected_rnr_error;
	int ret, i, cnt, rnr_flag;

	rnr_flag = 0;
	expected_rnr_error = 1;
	/*
	 * In order for the sender to get RNR error, we need to first consume
	 * all pre-posted receive buffer (in efa provider, fi->rx_attr->size
	 * receiving buffer are pre-posted) on the receiver side, the subsequent
	 * sends (expected_rnr_error) will then get RNR errors.
	 */
	total_send = fi->rx_attr->size + expected_rnr_error;

	for (i = 0; i < total_send; i++) {
		do {
			ret = fi_send(ep, tx_buf, 32, mr_desc, remote_fi_addr, &tx_ctx);
			if (ret < 0 && ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_send", -ret);
				return ret;
			}
		} while (ret == -FI_EAGAIN);
	}

	cnt = total_send;
	do {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret == 1) {
			cnt--;
		} else if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(txcq, &comp_err, FI_SEND);
			if (ret < 0 && ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_readerr", -ret);
				return ret;
			} else if (ret == 1) {
				cnt--;
				if (comp_err.err == FI_ENORX) {
					rnr_flag = 1;
					printf("Got RNR error CQ entry as expected: %d, %s\n",
						comp_err.err, fi_strerror(comp_err.err));
				} else {
					printf("Got non-RNR error CQ entry: %d, %s\n",
						comp_err.err, fi_strerror(comp_err.err));
					return comp_err.err;
				}
			}
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", -ret);
			return ret;
		}
	} while (cnt);

	return (rnr_flag) ? 0 : -FI_EINVAL;
}

static int run(size_t rnr_retry)
{
	int ret;

	ret = ft_init();
	if (ret) {
		FT_PRINTERR("ft_init", -ret);
		return ret;
	}

	ret = ft_init_oob();
	if (ret) {
		FT_PRINTERR("ft_init_oob", -ret);
		return ret;
	}

	ret = ft_getinfo(hints, &fi);
	if (ret) {
		FT_PRINTERR("ft_getinfo", -ret);
		return ret;
	}

	ret = ft_open_fabric_res();
	if (ret) {
		FT_PRINTERR("ft_open_fabric_res", -ret);
		return ret;
	}

	ret = ft_alloc_active_res(fi);
	if (ret) {
		FT_PRINTERR("ft_alloc_active_res", -ret);
		return ret;
	}

	fprintf(stdout, "Setting RNR retry count to %zu ...\n", rnr_retry);
	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_RNR_RETRY, &rnr_retry, sizeof(rnr_retry));
	if (ret) {
		FT_PRINTERR("fi_setopt", -ret);
		return ret;
	}
	fprintf(stdout, "RNR retry count has been set to %zu.\n", rnr_retry);

	ret = ft_enable_ep(ep);
	if (ret) {
		FT_PRINTERR("ft_enable_ep_recv", -ret);
		return ret;
	}

	ret = ft_init_av();
	if (ret) {
		FT_PRINTERR("ft_init_av", -ret);
		return ret;
	}
	/* client does fi_send and then poll CQ to get error (FI_ENORX) CQ entry */
	if (opts.dst_addr) {
		ret = poll_rnr_cq_error();
		if (ret) {
			FT_PRINTERR("pingpong", -ret);
			return ret;
		}
	}

	/*
	 * To get RNR error on the client side, the server should not close its
	 * endpoint while the client is still sending.
	 * ft_reset_oob() will re-initialize OOB sync between server and client.
	 * Calling it here to ensure the client has finished the sending.
	 * And both server and client are ready to close endpoint and free resources.
	 */
	ret = ft_reset_oob();
	if (ret) {
		FT_PRINTERR("ft_reset_oob", -ret);
		return ret;
	}
	ret = ft_close_oob();
	if (ret) {
		FT_PRINTERR("ft_close_oob", -ret);
		return ret;
	}

	return 0;
}

static void print_opts_usage(char *name, char *desc)
{
	ft_usage(name, desc);
	/* rdm_ep_rnr_retry test usage */
	FT_PRINT_OPTS_USAGE("-R <number>", "RNR retry count (valid values: 0-7, default: 1)");
}

int main(int argc, char **argv)
{
	int op, ret;
	size_t rnr_retry;

	rnr_retry = 1;
	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "R:h" ADDR_OPTS INFO_OPTS CS_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case 'R':
			rnr_retry = atoi(optarg);
			if (rnr_retry > 7) {
				fprintf(stdout, "RNR retry count invalid, it must be 0-7.\n");
				return EXIT_FAILURE;
			}
			break;
		case '?':
		case 'h':
			print_opts_usage(argv[0], "RDM RNR retry counter reset test");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	/*
	 * RNR error is generated from EFA device, so disable shm transfer by
	 * setting FI_REMOTE_COMM and unsetting FI_LOCAL_COMM in order to ensure
	 * EFA device is being used when running this test on a single node.
	 */
	hints->caps |= FI_REMOTE_COMM;
	hints->caps &= ~FI_LOCAL_COMM;
	hints->mode |= FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;
	/*
	 * FI_RM_DISABLED is required to be set in order for the RNR error CQ entry
	 * to be written to applications. Otherwise, the packet (failed with RNR error)
	 * will be queued and resent.
	 */
	hints->domain_attr->resource_mgmt = FI_RM_DISABLED;

	ret = run(rnr_retry);
	if (ret)
		FT_PRINTERR("run", -ret);

	ft_free_res();
	return ft_exit_code(ret);
}
