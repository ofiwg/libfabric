/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"
#include "cxip_ki_test_common.h"
#include "fi_cxi_gpu_ext.h"

/* KI (GPU kernel interface) hints tests. The fixture leaves the endpoint
 * un-enabled so each test controls which objects exist when fi_enable()
 * runs.
 */

static void ki_hints_setup(void)
{
	cxit_ki_setup_ep(false);
}

static void ki_hints_teardown(void)
{
	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
	cxit_teardown_ep();
}

TestSuite(ki_hints, .init = ki_hints_setup, .fini = ki_hints_teardown,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test NULL hints pointer */
Test(ki_hints, null_arg)
{
	int ret;

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control(NULL) returned %d", ret);
}

/* Test that command queue buffers must be page aligned */
Test(ki_hints, misaligned_buf)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	uint8_t *tx_buf;
	uint8_t *trig_buf;

	tx_buf = cxit_ki_page_alloc(s_page_size);
	trig_buf = cxit_ki_page_alloc(s_page_size);

	cxit_ki_init_hints(&hints, tx_buf + 1, s_page_size, trig_buf, s_page_size);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "misaligned TX buffer returned %d", ret);

	cxit_ki_init_hints(&hints, tx_buf, s_page_size, trig_buf + 1, s_page_size);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "misaligned triggered buffer returned %d", ret);

	cxit_ki_page_free(tx_buf);
	cxit_ki_page_free(trig_buf);
}

/* Test that the reserved flags field must be zero */
Test(ki_hints, reserved_flags)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	uint8_t *tx_buf;

	tx_buf = cxit_ki_page_alloc(s_page_size);

	cxit_ki_init_hints(&hints, tx_buf, s_page_size, NULL, 0);
	hints.flags = 1;
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL, "non-zero flags returned %d", ret);

	cxit_ki_page_free(tx_buf);
}

/* Test that the dmabuf fields must agree with is_host_mem */
Test(ki_hints, dmabuf_mismatch)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	uint8_t *tx_buf;
	uint8_t *trig_buf;

	tx_buf = cxit_ki_page_alloc(s_page_size);
	trig_buf = cxit_ki_page_alloc(s_page_size);

	/* Host memory with a dmabuf fd */
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, trig_buf, s_page_size);
	hints.tx_cmdq.dmabuf_fd = 0;
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL, "host TX with dmabuf fd returned %d",
		     ret);

	/* Host memory with a dmabuf offset */
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, trig_buf, s_page_size);
	hints.trig_cmdq.dmabuf_offset = s_page_size;
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "host triggered with dmabuf offset returned %d", ret);

	/* Device memory, not a dmabuf, with a dmabuf offset */
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, trig_buf, s_page_size);
	hints.tx_cmdq.is_host_mem = false;
	hints.tx_cmdq.dmabuf_offset = s_page_size;
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "device TX non-dmabuf with offset returned %d", ret);

	/* Device memory with an invalid dmabuf fd */
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, trig_buf, s_page_size);
	hints.trig_cmdq.is_host_mem = false;
	hints.trig_cmdq.dmabuf_fd = -2;
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "device triggered with fd -2 returned %d", ret);

	cxit_ki_page_free(tx_buf);
	cxit_ki_page_free(trig_buf);
}

/* Test that buffer and length must both be set or both be unset */
Test(ki_hints, buf_len_mismatch)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	void *buf;

	buf = cxit_ki_page_alloc(s_page_size);

	cxit_ki_init_hints(&hints, buf, 0, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL, "TX buf/len mismatch returned %d", ret);

	cxit_ki_init_hints(&hints, NULL, s_page_size, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL, "TX len without buf returned %d", ret);

	cxit_ki_init_hints(&hints, NULL, 0, buf, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "triggered buf/len mismatch returned %d", ret);

	cxit_ki_init_hints(&hints, NULL, 0, NULL, s_page_size);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "triggered len without buf returned %d", ret);

	cxit_ki_page_free(buf);
}

/* Test that a buffer length must be a power-of-two number of pages, no
 * larger than the hardware maximum.  The kernel rounds the ring up to
 * a power-of-two number of pages and maps that length from the buffer,
 * so any other length would be overrun.  fi_control() only validates,
 * so the buffer need not be as large as the length claims.
 */
Test(ki_hints, buf_len_rounding)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	void *buf;
	size_t max_len = (size_t)CXI_MAX_CQ_COUNT * C_CQ_CMD_SIZE;

	buf = cxit_ki_page_alloc(s_page_size);

	cxit_ki_init_hints(&hints, buf, s_page_size / 2, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL, "TX sub-page length returned %d", ret);

	cxit_ki_init_hints(&hints, buf, s_page_size + C_CQ_CMD_SIZE, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "TX non-page-multiple length returned %d", ret);

	cxit_ki_init_hints(&hints, buf, 3 * s_page_size, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "TX non-power-of-two length returned %d", ret);

	cxit_ki_init_hints(&hints, buf, 2 * max_len, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL, "TX over-maximum length returned %d",
		     ret);

	cxit_ki_init_hints(&hints, NULL, 0, buf, 3 * s_page_size);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "triggered non-power-of-two length returned %d", ret);

	cxit_ki_init_hints(&hints, buf, max_len, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "TX maximum length returned %d", ret);

	cxit_ki_page_free(buf);
}

/* Test that leaving both command queues unset is valid */
Test(ki_hints, both_unset_ok)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints = {};

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "unset hints returned %d", ret);
}

/* Test that a TX buffer is rejected on an endpoint with FI_MSG, FI_TAGGED
 * or FI_COLLECTIVE, since those let the provider emit on the shared TX
 * ring. A triggered-only buffer is still accepted. The fixture endpoint
 * lacks those capabilities, so replace it with a default-capability one.
 */
Test(ki_hints, msg_caps)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	uint8_t *tx_buf;
	uint8_t *trig_buf;

	cxit_destroy_ep();
	cxit_teardown_ep();
	cxit_setup_getinfo();
	cxit_setup_ep();
	cxit_create_ep();

	tx_buf = cxit_ki_page_alloc(s_page_size);
	trig_buf = cxit_ki_page_alloc(s_page_size);

	cxit_ki_init_hints(&hints, tx_buf, s_page_size, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EINVAL,
		     "TX buffer with FI_MSG caps returned %d", ret);

	cxit_ki_init_hints(&hints, NULL, 0, trig_buf, s_page_size);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS,
		     "triggered buffer with FI_MSG caps returned %d", ret);

	cxit_ki_page_free(tx_buf);
	cxit_ki_page_free(trig_buf);
}

/* Test that KI hints and get_gpu_target_info() are rejected on an
 * endpoint with per-peer VNIs (FI_AV_AUTH_KEY), since the KI command
 * queues serve one VNI only.
 */
Test(ki_hints, av_auth_key)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_target_info target;
	fi_addr_t addr = 0;
	uint8_t *tx_buf;

	cxit_destroy_ep();
	cxit_teardown_ep();
	cxit_setup_getinfo();
	cxit_fi_hints->caps = CXIT_KI_EP_CAPS;
	cxit_fi_hints->domain_attr->auth_key_size = FI_AV_AUTH_KEY;
	cxit_fi_hints->domain_attr->max_ep_auth_key = 2;
	cxit_setup_ep();
	cxit_create_ep();

	tx_buf = cxit_ki_page_alloc(s_page_size);

	cxit_ki_init_hints(&hints, tx_buf, s_page_size, NULL, 0);
	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EOPNOTSUPP,
		     "hints on FI_AV_AUTH_KEY EP returned %d", ret);

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);

	ret = ops->get_gpu_target_info(&cxit_ep->fid, &addr, 1, &target);
	cr_assert_eq(ret, -FI_EOPNOTSUPP,
		     "get_gpu_target_info on FI_AV_AUTH_KEY EP returned %d",
		     ret);

	cxit_ki_page_free(tx_buf);
}

/* Test that get_gpu_target_info() fails on an endpoint with no bound AV.
 * The fixture endpoint has none.
 */
Test(ki_hints, target_info_no_av)
{
	int ret;
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_target_info target;
	fi_addr_t addr = 0;

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);

	ret = ops->get_gpu_target_info(&cxit_ep->fid, &addr, 1, &target);
	cr_assert_eq(ret, -FI_EOPBADSTATE,
		     "get_gpu_target_info with no AV returned %d", ret);
}

/* Test that host data-transfer operations are disabled on an endpoint
 * given a TX buffer. The fixture capabilities exclude FI_MSG and
 * FI_TAGGED, so only the RMA and atomic tables change at enable.
 */
Test(ki_hints, no_host_ops)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	uint8_t *tx_buf;
	uint64_t val = 0;
	uint64_t res = 0;

	tx_buf = cxit_ki_page_alloc(s_page_size);
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, NULL, 0);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control returned %d", ret);

	cxit_ki_create_enabled_objects(true);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	ret = fi_write(cxit_ep, &val, sizeof(val), NULL, 0, 0, 0, NULL);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_write returned %d", ret);

	ret = fi_read(cxit_ep, &val, sizeof(val), NULL, 0, 0, 0, NULL);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_read returned %d", ret);

	ret = fi_inject_write(cxit_ep, &val, sizeof(val), 0, 0, 0);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_inject_write returned %d", ret);

	ret = fi_atomic(cxit_ep, &val, 1, NULL, 0, 0, 0, FI_UINT64, FI_SUM,
			NULL);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_atomic returned %d", ret);

	ret = fi_fetch_atomic(cxit_ep, &val, 1, NULL, &res, NULL, 0, 0, 0,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_fetch_atomic returned %d", ret);

	ret = fi_inject_atomic(cxit_ep, &val, 1, 0, 0, 0, FI_UINT64, FI_SUM);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_inject_atomic returned %d", ret);

	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
	cxit_ki_page_free(tx_buf);
}

/* Test that hints are accepted before fi_enable() and rejected after */
Test(ki_hints, after_enable)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	uint8_t *tx_buf;
	uint8_t *trig_buf;

	tx_buf = cxit_ki_page_alloc(s_page_size);
	trig_buf = cxit_ki_page_alloc(s_page_size);
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, trig_buf, s_page_size);

	cxit_ki_create_enabled_objects(true);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control returned %d", ret);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, -FI_EOPBADSTATE,
		     "post-enable fi_control returned %d", ret);

	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
	cxit_ki_page_free(tx_buf);
	cxit_ki_page_free(trig_buf);
}

/* Test enable with no hinted buffers. The TX command queue always comes
 * from the provider-managed queue; the triggered queue is left zeroed.
 */
Test(ki_hints, zero_len_both)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints = {};
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_ep_info gpu_ep = {};

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "unset hints returned %d", ret);

	cxit_ki_create_enabled_objects(true);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);

	ret = ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_gpu_ep_info returned %d", ret);

	cr_assert_not_null(gpu_ep.tx_cmdq.hptr_buff,
			   "tx_cmdq.hptr_buff is NULL");
	cr_assert_not_null(gpu_ep.tx_cmdq.hptr_mmio_db,
			   "tx_cmdq.hptr_mmio_db is NULL");

	cr_assert_null(gpu_ep.trig_cmdq.hptr_buff,
		       "trig_cmdq.hptr_buff is non-NULL");
	cr_assert_null(gpu_ep.trig_cmdq.hptr_mmio_db,
		       "trig_cmdq.hptr_mmio_db is non-NULL");
	cr_assert_null(gpu_ep.trig_cmdq.cq.cmds32,
		       "trig_cmdq.cq.cmds32 is non-NULL");
	cr_assert_null(gpu_ep.trig_cmdq.cq.wp_addr,
		       "trig_cmdq.cq.wp_addr is non-NULL");

	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
}

/* Test enable with only a triggered command queue buffer */
Test(ki_hints, trig_only)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	uint8_t *trig_buf;

	trig_buf = cxit_ki_page_alloc(s_page_size);
	cxit_ki_init_hints(&hints, NULL, 0, trig_buf, s_page_size);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "trig-only hints returned %d", ret);

	cxit_ki_create_enabled_objects(true);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);

	ret = ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info returned %d", ret);

	cr_assert_not_null(gpu_ep.tx_cmdq.hptr_buff,
			   "tx_cmdq.hptr_buff is NULL");
	cr_assert_neq(gpu_ep.tx_cmdq.hptr_buff, trig_buf,
		      "tx_cmdq.hptr_buff aliases the triggered buffer");
	cr_assert_eq(gpu_ep.trig_cmdq.hptr_buff, trig_buf,
		     "trig_cmdq.hptr_buff mismatch");
	cr_assert_not_null(gpu_ep.trig_cmdq.hptr_mmio_db,
			   "trig_cmdq.hptr_mmio_db is NULL");
	cr_assert_eq(gpu_ep.trig_cmdq.cq.size32, s_page_size / 32,
		     "trig_cmdq.cq.size32 is %u", gpu_ep.trig_cmdq.cq.size32);

	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
	cxit_ki_page_free(trig_buf);
}

/* Test enable with a multi-page TX ring. The kernel requires a host
 * ring to be physically contiguous, which only a huge page guarantees,
 * so the test is skipped when none is available.
 */
Test(ki_hints, huge_page_ring)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	size_t len = 2 * 1024 * 1024;
	void *tx_buf;

	tx_buf = mmap(NULL, len, PROT_READ | PROT_WRITE,
		      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
	if (tx_buf == MAP_FAILED)
		cr_skip_test("no huge page available");
	memset(tx_buf, 0, len);

	cxit_ki_init_hints(&hints, tx_buf, len, NULL, 0);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control returned %d", ret);

	cxit_ki_create_enabled_objects(true);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);

	ret = ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info returned %d", ret);

	cr_assert_eq(gpu_ep.tx_cmdq.hptr_buff, tx_buf,
		     "tx_cmdq.hptr_buff mismatch");
	cr_assert_eq(gpu_ep.tx_cmdq.cq.size32, len / 32,
		     "tx_cmdq.cq.size32 is %u", gpu_ep.tx_cmdq.cq.size32);

	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
	munmap(tx_buf, len);
}

/* Test get_gpu_ep_info() with no bound send counter */
Test(ki_hints, no_send_cntr)
{
	int ret;
	struct fi_cxi_ki_ep_hints hints;
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	uint8_t *tx_buf;

	tx_buf = cxit_ki_page_alloc(s_page_size);
	cxit_ki_init_hints(&hints, tx_buf, s_page_size, NULL, 0);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control returned %d", ret);

	cxit_ki_create_enabled_objects(false);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);

	ret = ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info returned %d", ret);

	cr_assert_eq(gpu_ep.ctn, C_CT_NONE, "ctn is %u with no send counter",
		     gpu_ep.ctn);
	cr_assert_eq(gpu_ep.tx_cmdq.hptr_buff, tx_buf,
		     "tx_cmdq.hptr_buff mismatch");
	cr_assert_neq(gpu_ep.eqn, C_EQ_NONE, "eqn is C_EQ_NONE");
	cr_assert_neq(gpu_ep.user_ptr, 0, "user_ptr is zero");

	cxit_destroy_ep();
	cxit_ki_destroy_enabled_objects();
	cxit_ki_page_free(tx_buf);
}

/* Test that ops open on an un-enabled endpoint but get_gpu_ep_info()
 * fails until it is enabled.
 */
Test(ki_hints, ops_before_enable)
{
	int ret;
	struct fi_cxi_ki_ops *ops;
	struct fi_cxi_gpu_ep_info gpu_ep = {};

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);
	cr_assert_not_null(ops);

	ret = ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, -FI_EOPBADSTATE,
		     "pre-enable get_gpu_ep_info returned %d", ret);
}

/* Test fi_open_ops() failure cases */
Test(ki_hints, open_ops_bad_name)
{
	int ret;
	struct fi_cxi_ki_ops *ops = NULL;

	ret = fi_open_ops(&cxit_ep->fid, "not_a_cxi_ops_name", 0,
			  (void **)&ops, NULL);
	cr_assert_eq(ret, -FI_ENOSYS, "fi_open_ops(bad name) returned %d",
		     ret);
	cr_assert_null(ops, "ops set on failure");

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 1, (void **)&ops,
			  NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_open_ops(flags) returned %d", ret);
	cr_assert_null(ops, "ops set on failure");

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_open_ops(NULL) returned %d", ret);
}
