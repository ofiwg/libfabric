/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(rma, .init = cxit_setup_rma, .fini = cxit_teardown_rma);

/* Test basic RMA write */
Test(rma, simple_write, .timeout = 3, .disabled = false)
{
	int i, ret;
	uint8_t *rma_win,  /* Target buffer for RMA */
		*send_buf; /* RMA send buffer */
	int win_len = 0x1000;
	int send_len = 8;
	struct fid_mr *win_mr;
	int key_val = 0;
	struct fi_cq_tagged_entry cqe;

	rma_win = calloc(win_len, 1);
	cr_assert(rma_win);

	send_buf = malloc(win_len);
	cr_assert(send_buf);

	for (i = 0; i < win_len; i++)
		send_buf[i] = i + 0xa0;

	ret = fi_mr_reg(cxit_domain, rma_win, win_len, FI_REMOTE_WRITE, 0,
			key_val, 0, &win_mr, NULL);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_mr_bind(win_mr, &cxit_ep->fid, 0);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_mr_enable(win_mr);
	cr_assert(ret == FI_SUCCESS);

	/* Send 8 bytes from send buffer data to RMA window 0 at FI address 0
	 * (self)
	 */
	ret = fi_write(cxit_ep, send_buf, send_len, 0, 0, 0, key_val, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	/* Validate event fields */
	cr_assert(cqe.op_context == NULL, "CQE Context mismatch");
	cr_assert(cqe.flags == (FI_RMA | FI_WRITE), "CQE flags mismatch");
	cr_assert(cqe.len == 0, "Invalid CQE length");
	cr_assert(cqe.buf == 0, "Invalid CQE address");
	cr_assert(cqe.data == 0, "Invalid CQE data");
	cr_assert(cqe.tag == 0, "Invalid CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_log_info("rma_win[%d]=%u  send_buf[%d]=%u\n",
			    i, rma_win[i], i, send_buf[i]);

		cr_assert(rma_win[i] == send_buf[i],
			  "data mismatch, element: %d\n", i);
	}

	fi_close(&win_mr->fid);
	free(send_buf);
	free(rma_win);
}

