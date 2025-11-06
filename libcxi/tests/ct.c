/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <time.h>

#include "libcxi_test_common.h"
#include "uapi/misc/cxi.h"

TestSuite(counting_event,
	  .init = counting_event_setup,
	  .fini = counting_event_teardown);

Test(counting_event, doorbell_inc)
{
	uint64_t success = 0xFFF;
	uint64_t failure = 0x3F;

	cxi_ct_inc_success(ct, success);
	cxi_ct_inc_failure(ct, failure);

	expect_ct_values(ct, success, failure);
}

void cntr_read(struct cxi_cq *cmdq)
{
	int rc;
	struct c_ct_cmd cmd = {
		.ct = (uint16_t)ct->ctn,
	};

	rc = cxi_cq_emit_ct(cmdq, C_CMD_CT_GET, &cmd);
	cr_assert_eq(rc, 0, "Failed cxi_cq_emit_ct() %d", rc);

	cxi_cq_ring(cmdq);
}

Test(counting_event, doorbell_inc_2)
{
	uint64_t success = 10;

	cxi_ct_inc_success(ct, success);
	sfence();
	cxi_ct_inc_success(ct, success);

	cntr_read(trig_cmdq);
	expect_ct_values(ct, success * 2, 0);
}

void get_mmio_addr(struct cxi_ct *ct, void **addr, size_t *len)
{
	*addr = ct->doorbell;
	*len = sizeof(ct->doorbell);
}

void wait_wb(struct c_ct_writeback *wb, uint64_t success, uint8_t failure)
{
	time_t timeout;

	// Wait for valid CT writeback
	timeout = time(NULL) + 5;
	while (wb->ct_writeback == 0) {
		cr_assert_leq(time(NULL), timeout, "Timeout waiting for CT WB");
		sched_yield();
	}

	cr_assert_eq(wb->ct_success, success, "Bad CT WB success value");
	cr_assert_eq(wb->ct_failure, failure, "Bad CT WB failure value");

	// Reset the writeback bit
	wb->ct_writeback = 0;
}

Test(counting_event, doorbell_wb_update)
{
	int rc;
	uint64_t success = 10;
	struct c_ct_writeback *wb;

	cxi_ct_inc_success(ct, success);

	cntr_read(trig_cmdq);
	expect_ct_values(ct, success, 0);

	wb = aligned_alloc(8, sizeof(*wb));
	cr_assert_neq(wb, NULL, "Failed to allocated memory");

	rc = cxil_ct_wb_update(ct, wb);
	cr_assert_eq(rc, 0, "cxil_ct_wb_update() failed %d", rc);

	cxi_ct_inc_success(ct, success);
	cntr_read(trig_cmdq);
	wait_wb(wb, success * 2, 0);
}

Test(counting_event, invalid_alignment)
{
	int rc;
	void *wb;
	void *bad_wb_addr;
	struct cxi_ct *tmp_ct;

	wb = malloc(20);
	bad_wb_addr = (void *)((uint64_t)wb | (uint64_t)1);

	cr_assert_neq(wb, NULL, "Failed to allocated memory");

	rc = cxil_alloc_ct(lni, (struct c_ct_writeback *)bad_wb_addr, &tmp_ct);
	cr_assert_eq(rc, -EINVAL, "Unexpected cxil_alloc_ct() returned");

	free(wb);
}

#if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT)
void device_ct_setup(void)
{
	int ret;
	struct cxi_cq_alloc_opts cq_opts = {
		.count = 256,
		.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS,
	};

	data_xfer_setup();

	/* Use the transmit EQ for trigger CMDQ. */
	ret = cxil_alloc_cmdq(lni, transmit_evtq, &cq_opts, &trig_cmdq);
	cr_assert_eq(ret, 0, "Triggered cxil_alloc_cmdq() failed %d", ret);

}

void device_ct_teardown(void)
{
	int ret;

	ret = cxil_destroy_cmdq(trig_cmdq);
	cr_assert_eq(ret, 0, "Destroy Trigger CQ Failed %d", ret);

	data_xfer_teardown();
}

TestSuite(device_ct,
	  .init = device_ct_setup,
	  .fini = device_ct_teardown);

Test(device_ct, no_device)
{
	int rc;
	uint64_t success = 0xFFF;
	uint64_t failure = 0x3F;

	wb = aligned_alloc(8, sizeof(*wb));
	memset(wb, 0x5a, sizeof(*wb));

	rc = cxil_alloc_ct(lni, wb, &ct);
	cr_assert_eq(rc, 0, "Failed cxil_alloc_ct() failed %d", rc);

	cxi_ct_inc_success(ct, success);
	cxi_ct_inc_failure(ct, failure);
	expect_ct_values(ct, success, failure);

	rc = cxil_destroy_ct(ct);
	cr_assert_eq(rc, 0, "Failed cxil_destroy_ct() failed %d", rc);

	free(wb);
}

int wait_ct_values_device(struct cxi_ct *ct, uint64_t *success,
			     uint64_t *failure)
{
	time_t timeout;
	struct c_ct_writeback lwb;
	struct c_ct_writeback *wb = ct->wb;
	struct mem_window dev_buf;

	dev_buf.buffer = (void *)wb;
	dev_buf.length = sizeof(lwb);

	memcpy_device_to_host(&lwb, &dev_buf);

	timeout = time(NULL) + 2;
	while (lwb.ct_writeback == 0) {
		if (time(NULL) > timeout) {
			printf("Timeout waiting for CT WB\n");
			return -1;
		}
		memcpy_device_to_host(&lwb, &dev_buf);
		sched_yield();
	}

	*success = lwb.ct_success;
	*failure = lwb.ct_failure;

	return 0;
}

#define SUCCESS 0xFFF
#define FAILURE 0x3F

Test(device_ct, device)
{
	int rc;
	struct mem_window buf;
	uint64_t success;
	uint64_t failure;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	buf.length = sizeof(*wb);
	rc = gpu_malloc(&buf);
	cr_assert_eq(rc, 0, "gpu_malloc() failed %d", rc);

	wb = (void *)buf.buffer;
	gpu_memset(wb, 0, sizeof(*wb));

	rc = cxil_alloc_ct(lni, wb, &ct);
	cr_assert_eq(rc, 0, "Failed cxil_alloc_ct() failed %d", rc);

	cxi_ct_inc_success(ct, SUCCESS);
	cxi_ct_inc_failure(ct, FAILURE);

	rc = wait_ct_values_device(ct, &success, &failure);
	cr_assert_eq(rc, 0, "Timed out");

	cr_assert_eq(SUCCESS, success, "Bad CT WB success value");
	cr_assert_eq(FAILURE, failure, "Bad CT WB failure value");

	rc = cxil_destroy_ct(ct);
	cr_assert_eq(rc, 0, "Failed cxil_destroy_ct() failed %d", rc);

	gpu_free(buf.buffer);
	gpu_lib_fini();
}

/* test a bad address */
Test(device_ct, device_bad_addr)
{
	int rc;

	wb = (void *)0x100000000UL;
	rc = cxil_alloc_ct(lni, wb, &ct);
	cr_assert_neq(rc, 0, "cxil_alloc_ct() should failed %d", rc);
}
#endif
