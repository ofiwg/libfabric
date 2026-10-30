/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2025 Hewlett Packard Enterprise Development LP
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

void cxit_setup_rma_writedata(void)
{
	cxip_env.enable_writedata = 1;
	cxit_setup_getinfo();
	cxit_fi_hints->caps |= FI_RMA | FI_RMA_EVENT | FI_MSG | FI_SOURCE;
	cxit_fi_hints->domain_attr->mr_mode |= FI_MR_PROV_KEY | FI_MR_ENDPOINT;
	cxit_setup_rma();
}

TestSuite(rma_writedata, .init = cxit_setup_rma_writedata, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

static void cxit_setup_writedata_source(void)
{
	struct cxip_cq *cq;
	struct cxip_addr fake_addr = {.nic = 0xad, .pid = 0xbc};
	uint64_t caps, rx_caps, domain_caps;
	size_t addrlen = sizeof(cxit_ep_addr);
	size_t i;
	int ret;

	cxip_env.enable_writedata = 1;
	cxit_setup_getinfo();
	cxit_fi_hints->caps |= FI_RMA | FI_RMA_EVENT | FI_MSG | FI_SOURCE;
	cxit_fi_hints->domain_attr->mr_mode |= FI_MR_PROV_KEY | FI_MR_ENDPOINT;
	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	cxit_fi_hints->tx_attr->size = 512;
	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_rx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_av_attr.type = FI_AV_TABLE;
	cxit_setup_domain();

	/* Open the domain and CQs before requesting endpoint source tracking. */
	caps = cxit_fi->caps;
	rx_caps = cxit_fi->rx_attr->caps;
	domain_caps = cxit_fi->domain_attr->caps;
	cxit_fi->caps &= ~(FI_SOURCE | FI_SOURCE_ERR);
	cxit_fi->rx_attr->caps &= ~(FI_SOURCE | FI_SOURCE_ERR);
	cxit_fi->domain_attr->caps &= ~(FI_SOURCE | FI_SOURCE_ERR);
	cxit_create_domain();
	cxit_create_cqs();
	cxit_fi->caps = caps;
	cxit_fi->rx_attr->caps = rx_caps;
	cxit_fi->domain_attr->caps = domain_caps;

	cq = container_of(cxit_rx_cq, struct cxip_cq, util_cq.cq_fid);
	cr_assert_null(cq->util_cq.src);
	cxit_create_ep();
	cxit_create_eq();
	cxit_bind_eq();
	cxit_bind_cqs();
	cr_assert_not_null(cq->util_cq.src);
	for (i = 0; i < cq->util_cq.cirq->size; i++)
		cr_assert_eq(cq->util_cq.src[i], FI_ADDR_NOTAVAIL);
	cxit_create_cntrs();
	cxit_bind_cntrs();
	cxit_create_av();
	cxit_bind_av();

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable failed %d", ret);
	ret = fi_getname(&cxit_ep->fid, &cxit_ep_addr, &addrlen);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getname failed %d", ret);
	cr_assert_eq(addrlen, sizeof(cxit_ep_addr));
	ret = fi_av_insert(cxit_av, &fake_addr, 1, NULL, 0, NULL);
	cr_assert_eq(ret, 1);
	ret = fi_av_insert(cxit_av, &cxit_ep_addr, 1, &cxit_ep_fi_addr,
			   0, NULL);
	cr_assert_eq(ret, 1);
}

TestSuite(rma_writedata_source,
	  .init = cxit_setup_writedata_source,
	  .fini = cxit_teardown_rma, .timeout = CXIT_DEFAULT_TIMEOUT);

Test(rma_writedata, simple)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 0x1234;
	struct fi_cq_tagged_entry cqe;
	uint64_t immediate_data = 0xDEADBEEF;
	size_t len = 1024;
	uint8_t *send_buf;
	struct cxip_ep *ep = container_of(cxit_ep, struct cxip_ep, ep);
	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");
	memset(send_buf, 0xAA, len);

	/* Manual MR creation to include FI_RMA_EVENT flag */
	mem_window.mem = calloc(1, len);
	cr_assert_not_null(mem_window.mem, "mem_window alloc failed");

	ret = fi_mr_reg(cxit_domain, mem_window.mem, len,
			FI_REMOTE_WRITE | FI_REMOTE_READ | FI_SEND | FI_RECV,
			0, key_val, FI_RMA_EVENT, &mem_window.mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mem_window.mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed %d", ret);

	if (cxit_rem_cntr) {
		ret = fi_mr_bind(mem_window.mr, &cxit_rem_cntr->fid, FI_REMOTE_WRITE);
		cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind(cntr) failed %d", ret);
	}

	ret = fi_mr_enable(mem_window.mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed %d", ret);

	key_val = fi_mr_key(mem_window.mr);

	/* Perform writedata */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, immediate_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	if (ep->ep_obj->domain->rma_cq_data_size) {
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed %d", ret);
	} else {
		cr_assert_eq(ret, -FI_ENOSYS, "fi_writedata bad return %d", ret);
		goto done;
	}

	/* Wait for local completion (send) */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read (tx) failed %d", ret);
	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Wait for remote completion (recv) */
	ret = cxit_await_completion(cxit_rx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read (rx) failed %d", ret);

	/* Validate remote completion */
	cr_assert(cqe.flags & FI_REMOTE_WRITE, "Missing FI_REMOTE_WRITE flag");
	cr_assert_eq(cqe.data, immediate_data, "Data mismatch: 0x%lx != 0x%lx",
		     cqe.data, immediate_data);

	/* Verify data */
	for (size_t i = 0; i < len; i++) {
		cr_assert_eq(mem_window.mem[i], send_buf[i],
 			     "Memory mismatch at index %zu: 0x%02x != 0x%02x",
 			     i, mem_window.mem[i], send_buf[i]);
	}
done:
	free(send_buf);
	mr_destroy(&mem_window);
}

static void writedata_with_source(struct fid_ep *tx_ep, fi_addr_t dest_addr,
				  fi_addr_t expected_source)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 0x1234;
	struct fi_cq_tagged_entry cqe;
	uint64_t immediate_data = 0xDEADBEEF;
	size_t len = 1024;
	uint8_t *send_buf;
	struct cxip_ep *ep = container_of(tx_ep, struct cxip_ep, ep);
	fi_addr_t src_addr;
	int poll_count = 0;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");
	memset(send_buf, 0xAA, len);

	/* Manual MR creation to include FI_RMA_EVENT flag */
	mem_window.mem = calloc(1, len);
	cr_assert_not_null(mem_window.mem, "mem_window alloc failed");

	ret = fi_mr_reg(cxit_domain, mem_window.mem, len,
			FI_REMOTE_WRITE | FI_REMOTE_READ | FI_SEND | FI_RECV,
			0, key_val, FI_RMA_EVENT, &mem_window.mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mem_window.mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed %d", ret);

	if (cxit_rem_cntr) {
		ret = fi_mr_bind(mem_window.mr, &cxit_rem_cntr->fid, FI_REMOTE_WRITE);
		cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind(cntr) failed %d", ret);
	}

	ret = fi_mr_enable(mem_window.mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed %d", ret);

	key_val = fi_mr_key(mem_window.mr);

	/* Perform writedata */
	ret = fi_writedata(tx_ep, send_buf, len, NULL, immediate_data,
			   dest_addr, 0, key_val, NULL);
	if (ep->ep_obj->domain->rma_cq_data_size) {
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed %d", ret);
	} else {
		cr_assert_eq(ret, -FI_ENOSYS, "fi_writedata bad return %d", ret);
		goto done;
	}

	/* Wait for local completion (send) */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read (tx) failed %d", ret);
	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Wait for remote completion (recv) with source address */
	while (poll_count < CXIT_DEFAULT_TIMEOUT * 1000) {
		ret = fi_cq_readfrom(cxit_rx_cq, &cqe, 1, &src_addr);
		if (ret == 1)
			break;
		if (ret != -FI_EAGAIN)
			break;
		poll_count++;
		usleep(1000);
	}
	cr_assert_eq(ret, 1, "fi_cq_readfrom (rx) failed %d", ret);

	/* Validate remote completion */
	cr_assert(cqe.flags & FI_REMOTE_WRITE, "Missing FI_REMOTE_WRITE flag");
	cr_assert_eq(cqe.data, immediate_data, "Data mismatch: 0x%lx != 0x%lx",
		     cqe.data, immediate_data);

	/* Validate source address */
	cr_assert_eq(src_addr, expected_source, "Source address mismatch");

	/* Verify data */
	for (size_t i = 0; i < len; i++) {
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "Memory mismatch at index %zu: 0x%02x != 0x%02x",
			     i, mem_window.mem[i], send_buf[i]);
	}
done:
	free(send_buf);
	mr_destroy(&mem_window);
}

Test(rma_writedata, with_source)
{
	writedata_with_source(cxit_ep, cxit_ep_fi_addr, cxit_ep_fi_addr);
}

Test(rma_writedata_source, shared_cq)
{
	struct cxip_cq *cq;
	struct fid_ep *shared_ep;
	struct cxip_addr shared_addr;
	size_t addrlen = sizeof(shared_addr);
	fi_addr_t shared_fi_addr;
	fi_addr_t *source;
	int ret;

	cq = container_of(cxit_rx_cq, struct cxip_cq, util_cq.cq_fid);
	source = cq->util_cq.src;
	ret = fi_endpoint(cxit_domain, cxit_fi, &shared_ep, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_endpoint failed %d", ret);
	ret = fi_ep_bind(shared_ep, &cxit_eq->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "shared EQ bind failed %d", ret);
	ret = fi_ep_bind(shared_ep, &cxit_tx_cq->fid, cxit_tx_cq_bind_flags);
	cr_assert_eq(ret, FI_SUCCESS, "shared TX CQ bind failed %d", ret);
	ret = fi_ep_bind(shared_ep, &cxit_rx_cq->fid, cxit_rx_cq_bind_flags);
	cr_assert_eq(ret, FI_SUCCESS, "shared RX CQ bind failed %d", ret);
	ret = fi_ep_bind(shared_ep, &cxit_av->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "shared AV bind failed %d", ret);
	cr_assert_eq(cq->util_cq.src, source);

	ret = fi_enable(shared_ep);
	cr_assert_eq(ret, FI_SUCCESS, "shared endpoint enable failed %d", ret);
	ret = fi_getname(&shared_ep->fid, &shared_addr, &addrlen);
	cr_assert_eq(ret, FI_SUCCESS, "shared endpoint getname failed %d", ret);
	cr_assert_eq(addrlen, sizeof(shared_addr));
	ret = fi_av_insert(cxit_av, &shared_addr, 1, &shared_fi_addr, 0, NULL);
	cr_assert_eq(ret, 1, "shared endpoint AV insert failed %d", ret);

	writedata_with_source(shared_ep, cxit_ep_fi_addr, shared_fi_addr);

	ret = fi_close(&shared_ep->fid);
	cr_assert_eq(ret, FI_SUCCESS, "shared endpoint close failed %d", ret);
}
