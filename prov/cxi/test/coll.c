/*
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHWARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. const NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER const AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS const THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <time.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

#define	MIN(a,b) (((a)<(b))?(a):(b))

TestSuite(coll_put, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test EP close without enabling collectives.
 */
Test(coll_put, noop)
{
}

/* Test EP close after enabling collectives.
 */
Test(coll_put, enable1)
{
	struct cxip_ep *ep;
	int ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
}

/* Test EP close after enabling collectives twice.
 */
Test(coll_put, enable2)
{
	struct cxip_ep *ep;
	int ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable1 failed: %d\n", ret);

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable2 failed: %d\n", ret);
}

/* 49-byte packet */
struct fakebuf {
	uint64_t count[6];
	uint8_t pad;
} __attribute__((packed));

/* Progression is needed because the test runs in a single execution thread with
 * NETSIM. This waits for completion of 'count' messages on the simulated
 * (loopback) target. It needs to be called periodically during the test run.
 */
#define	PROGRESS_COUNT	10
void _progress(struct cxip_cq *cq, int sendcnt, int *dataval)
{
	struct fi_cq_tagged_entry entry[PROGRESS_COUNT];
	int i, ret;

	while (sendcnt > 0) {
		do {
			int cnt = MIN(PROGRESS_COUNT, sendcnt);
			sched_yield();
			ret = fi_cq_read(&cq->util_cq.cq_fid, entry, cnt);
		} while (ret == -FI_EAGAIN);
		for (i = 0; i < ret; i++) {
			struct fakebuf *fb = entry[i].buf;
			cr_assert(fb->count[0] == *dataval);
			cr_assert(fb->count[5] == *dataval);
			cr_assert(fb->pad == (*dataval & 0xff));
			(*dataval)++;
		}
		sendcnt -= ret;
	}
}

/* Put count packets, and verify them.
 */
void _put_data(int count)
{
	struct cxip_ep *ep;
	fi_addr_t dest;
	struct fakebuf *buf;
	void *buffers;
	int sendcnt, dataval;
	int ret;
	int i;

	cr_assert(sizeof(*buf) <= 49);
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	/* enable the collectives target */
	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);

	/* must persist until _progress, for comparison */
	buffers = calloc(PROGRESS_COUNT, sizeof(*buf));
	dest = 1;

	buf = buffers;
	sendcnt = 0;
	dataval = 0;
	for (i = 0; i < count; i++) {
		buf->count[0] = i;
		buf->count[5] = i;
		buf->pad = (i & 0xff);
		ret = cxip_coll_send(ep->ep_obj, buf, sizeof(*buf), dest);
		cr_assert(ret == 0, "cxip_coll_send failed: %d\n", ret);

		buf++;
		sendcnt++;
		if (sendcnt >= 10) {
			/* _progress() advances dataval by 10 */
			_progress(ep->ep_obj->coll.rx_cq, sendcnt, &dataval);
			buf = buffers;
			sendcnt = 0;
		}
	}
	_progress(ep->ep_obj->coll.rx_cq, sendcnt, &dataval);
	cr_assert(ep->ep_obj->coll.buf_swap_cnt > 3, "Did not recirculate buffers\n");

	free(buffers);
}

Test(coll_put, put_many, .timeout = 30)
{
	_put_data(4000);
}

