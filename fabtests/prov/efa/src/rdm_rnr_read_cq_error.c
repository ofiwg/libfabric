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
 * This test resets RNR retry counter to 0 via fi_setopt, and test if
 * an RNR error CQ entry can be read.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <shared.h>
#include "efa_shared.h"
#include "efa_rnr_shared.h"

/**
 * poll_cq_and_err - Poll completion queue for both successful and error completions
 *
 * @cnt: Pointer to completion counter, decremented for each completion processed
 * @rnr_flag: Pointer to RNR flag, set to 1 when FI_ENORX (RNR) error is detected
 *
 * This function polls the transmit completion queue (txcq) for one completion entry.
 * It handles three types of results:
 * 1. Successful completion (ret == 1): Decrements cnt and continues
 * 2. Error available (ret == -FI_EAVAIL): Reads error entry via fi_cq_readerr
 *    - Sets rnr_flag if error is FI_ENORX (Receiver Not Ready)
 *    - Returns error code for unexpected errors
 * 3. No completions available (ret == -FI_EAGAIN): Normal empty CQ condition
 *
 * Return: 0 on success, negative error code on failure
 */
static int poll_cq_and_err(int *cnt, int *rnr_flag)
{
	int ret;
	struct fi_cq_data_entry comp = {0};
	struct fi_cq_err_entry comp_err = {0};

	/* Attempt to read one completion from the transmit CQ */
	ret = fi_cq_read(txcq, &comp, 1);
	if (ret == 1) {
		/* Successful completion - decrement remaining count */
		(*cnt)--;
	} else if (ret == -FI_EAVAIL) {
		/* Error completion available - read the error details */
		ret = fi_cq_readerr(txcq, &comp_err, FI_SEND);
		if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_readerr", -ret);
			return ret;
		} else if (ret == 1) {
			/* Error entry successfully read - decrement count */
			(*cnt)--;
			if (comp_err.err == FI_ENORX) {
				/* RNR (Receiver Not Ready) error detected - this is expected */
				*rnr_flag = 1;
				printf("Got RNR error CQ entry as expected: "
				       "%d, %s\n",
				       comp_err.err, fi_strerror(comp_err.err));
			} else {
				/* Unexpected error type - test should fail */
				printf("Got non-RNR error CQ entry: %d, %s\n",
				       comp_err.err, fi_strerror(comp_err.err));
				return comp_err.err;
			}
		}
	} else if (ret < 0 && ret != -FI_EAGAIN) {
		/* Unexpected fi_cq_read error */
		FT_PRINTERR("fi_cq_read", -ret);
		return ret;
	}
	/* ret == -FI_EAGAIN (empty CQ) is normal and handled by caller */
	return 0;
}

/**
 * rnr_read_cq_error - Main test function to trigger and detect RNR errors
 *
 * This function implements the core RNR (Receiver Not Ready) test logic:
 * 1. Calculate how many sends are needed to exhaust receive buffers
 * 2. Post all send operations (with retry logic for flow control)
 * 3. Poll completions until all sends complete (some as errors)
 * 4. Verify that at least one RNR error was detected
 *
 * Return: 0 if RNR error detected (test pass), -FI_EINVAL if no RNR error (test fail)
 */
static int rnr_read_cq_error(void)
{
	int total_send, expected_rnr_error;
	int ret, i, cnt, rnr_flag;

	/* Number of RNR errors we expect to see */
	expected_rnr_error = fi->rx_attr->size;

	/*
	 * Calculate total sends needed to trigger RNR errors:
	 *
	 * Strategy: Send more messages than available receive buffers to force RNR.
	 *
	 * For efa-rdm: Provider pre-posts fi->rx_attr->size receive buffers automatically
	 *              Total sends = normal sends + excess sends to trigger RNR
	 *
	 * For efa-direct: Only application-posted receives available (ft_enable_ep_recv posts 1)
	 *                 Total sends = expected RNR count + 1 existing receive buffer
	 */
	if (EFA_INFO_TYPE_IS_RDM(fi)) {
		total_send = fi->rx_attr->size + expected_rnr_error;
	} else {
		assert(EFA_INFO_TYPE_IS_DIRECT(fi));
		total_send = expected_rnr_error + 1;
	}

	/* Initialize completion counter and RNR detection flag */
	cnt = total_send;
	rnr_flag = 0;

	/*
	 * Phase 1: Post all send operations
	 *
	 * This loop ensures exactly 'total_send' operations are successfully posted.
	 * All sends will eventually succeed, but some may initially return -FI_EAGAIN
	 * due to flow control (send queue full).
	 *
	 * When -FI_EAGAIN occurs, we poll completions to make space in the send queue,
	 * then retry the same send operation until it succeeds.
	 */
	for (i = 0; i < total_send; i++) {
		do {
			/* Attempt to post send operation */
			ret = fi_send(ep, tx_buf, 32, mr_desc, remote_fi_addr, &tx_ctx);
			if (ret == -FI_EAGAIN) {
				/* Send queue full - poll completions to make space */
				int poll_ret = poll_cq_and_err(&cnt, &rnr_flag);
				if (poll_ret) {
					FT_PRINTERR("poll_cq_and_err", -poll_ret);
					return poll_ret;
				}
				/* Retry the same send operation (ret still == -FI_EAGAIN) */
			}
		} while (ret == -FI_EAGAIN);

		/* Check for unexpected send errors */
		if (ret < 0) {
			printf("fi_send failed with error %d at iteration %d\n", ret, i+1);
			FT_PRINTERR("fi_send", -ret);
			return ret;
		}
		/* ret == 0: send posted successfully */
	}
	printf("Send loop completed: posted %d sends\n", total_send);

	/*
	 * Phase 2: Collect all remaining completions
	 *
	 * At this point, all send operations have been posted successfully.
	 * Now we need to wait for all their completions (both successful and error).
	 *
	 * The 'cnt' variable tracks remaining completions. It gets decremented
	 * in poll_cq_and_err() for each completion processed.
	 */
	do {
		poll_cq_and_err(&cnt, &rnr_flag);
	} while (cnt > 0);
	printf("Collected all completions and errors\n");

	/*
	 * Test result evaluation:
	 * - Success (return 0): At least one RNR error was detected (rnr_flag == 1)
	 * - Failure (return -FI_EINVAL): No RNR errors detected - test didn't work as expected
	 */
	return (rnr_flag) ? 0 : -FI_EINVAL;
}


static int run()
{
	int ret;

	ret = ft_efa_rnr_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_efa_rnr_init_fabric", -ret);
		return ret;
	}

	/* client does fi_send and then poll CQ to get error (FI_ENORX) CQ entry */
	if (opts.dst_addr) {
		ret = rnr_read_cq_error();
		if (ret) {
			FT_PRINTERR("rnr_poll_cq_error", -ret);
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
	ft_free_res();

	return 0;
}


int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, ADDR_OPTS INFO_OPTS CS_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "RDM RNR poll error CQ entry test");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;

	/* FI_RM_DISABLED is required to get RNR error CQ entry */
	hints->domain_attr->resource_mgmt = FI_RM_DISABLED;
	/*
	 * RNR error is generated from EFA device, so disable shm transfer by
	 * setting FI_REMOTE_COMM and unsetting FI_LOCAL_COMM in order to ensure
	 * EFA device is being used when running this test on a single node.
	 */
	ft_efa_rnr_disable_hints_shm();

	ret = run();
	if (ret)
		FT_PRINTERR("run", -ret);

	return ft_exit_code(ret);
}
