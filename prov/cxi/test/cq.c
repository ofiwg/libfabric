/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"
#include "cxi_test_common.h"

TestSuite(cq, .init = cxit_setup_cq, .fini = cxit_teardown_cq);

/* Test basic CQ creation */
Test(cq, simple)
{
	cxit_create_cqs();
	cr_assert(cxit_tx_cq != NULL);
	cr_assert(cxit_rx_cq != NULL);

	cxit_destroy_cqs();
}

static void req_populate(struct cxi_req *req, fi_addr_t *addr)
{
	*addr = 0xabcd0;
	req->flags = FI_SEND;
	req->context = 0xabcd2;
	req->addr = 0xabcd3;
	req->data = 0xabcd4;
	req->tag = 0xabcd5;
	req->buf = 0xabcd6;
	req->data_len = 0xabcd7;
}

Test(cq, read_fmt_context)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_entry entry;
	fi_addr_t req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_read(cxit_tx_cq, &entry, 1);
	cr_assert(ret == 1);
	cr_assert((uint64_t)entry.op_context == req.context);

	cxit_destroy_cqs();
}

Test(cq, read_fmt_msg)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_msg_entry entry;
	fi_addr_t req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_MSG;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_read(cxit_tx_cq, &entry, 1);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(entry.flags == req.flags);
	cr_assert(entry.len == req.data_len);

	cxit_destroy_cqs();
}

Test(cq, read_fmt_data)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_data_entry entry;
	fi_addr_t req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_DATA;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_read(cxit_tx_cq, &entry, 1);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(entry.flags == req.flags);
	cr_assert(entry.len == req.data_len);
	cr_assert((uint64_t)entry.buf == req.buf);
	cr_assert(entry.data == req.data);

	cxit_destroy_cqs();
}

Test(cq, read_fmt_tagged)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_tagged_entry entry;
	fi_addr_t req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_read(cxit_tx_cq, &entry, 1);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(entry.flags == req.flags);
	cr_assert(entry.len == req.data_len);
	cr_assert((uint64_t)entry.buf == req.buf);
	cr_assert(entry.data == req.data);
	cr_assert(entry.tag == req.tag);

	cxit_destroy_cqs();
}

Test(cq, readfrom_fmt_context)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_entry entry;
	fi_addr_t addr = 0, req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_readfrom(cxit_tx_cq, &entry, 1, &addr);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(addr == req_addr);

	cxit_destroy_cqs();
}

Test(cq, readfrom_fmt_msg)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_msg_entry entry;
	fi_addr_t addr = 0, req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_MSG;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_readfrom(cxit_tx_cq, &entry, 1, &addr);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(entry.flags == req.flags);
	cr_assert(entry.len == req.data_len);
	cr_assert(addr == req_addr);

	cxit_destroy_cqs();
}

Test(cq, readfrom_fmt_data)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_data_entry entry;
	fi_addr_t addr = 0, req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_DATA;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_readfrom(cxit_tx_cq, &entry, 1, &addr);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(entry.flags == req.flags);
	cr_assert(entry.len == req.data_len);
	cr_assert((uint64_t)entry.buf == req.buf);
	cr_assert(entry.data == req.data);
	cr_assert(addr == req_addr);

	cxit_destroy_cqs();
}

Test(cq, readfrom_fmt_tagged)
{
	int ret;
	struct cxi_cq *cxi_cq;
	struct cxi_req req;
	struct fi_cq_tagged_entry entry;
	fi_addr_t addr = 0, req_addr;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_create_cqs();
	cxi_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid);

	req_populate(&req, &req_addr);

	cxi_cq->report_completion(cxi_cq, req_addr, &req);
	ret = fi_cq_readfrom(cxit_tx_cq, &entry, 1, &addr);
	cr_assert(ret == 1);

	cr_assert((uint64_t)entry.op_context == req.context);
	cr_assert(entry.flags == req.flags);
	cr_assert(entry.len == req.data_len);
	cr_assert((uint64_t)entry.buf == req.buf);
	cr_assert(entry.data == req.data);
	cr_assert(entry.tag == req.tag);
	cr_assert(addr == req_addr);

	cxit_destroy_cqs();
}
