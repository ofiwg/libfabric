/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

#include "libcxi_test_common.h"
#include "uapi/misc/cxi.h"

#define WIN_LENGTH (0x1000)
#define PUT_BUFFER_ID 0xb0f
#define GET_BUFFER_ID 0xa0e
#define PUT_DISABLED (false)
#define GET_DISABLED (false)
#define VALID_VNI 31
#define INVALID_VNI 32

const char *script_path = "profile.sh";

TestSuite(rma, .init = data_xfer_setup, .fini = data_xfer_teardown);

void excp_profile_teardown(void)
{
	char script_cmd[256];
	int return_code;

	snprintf(script_cmd, sizeof(script_cmd), "bash %s -c", script_path);
	return_code = system(script_cmd);

	cr_assert_eq(return_code, 0,
		     "excp_profile_teardown() failed %d", return_code);
}

void excp_profile_setup(void)
{
	char script_cmd[256];
	int return_code;

	snprintf(script_cmd, sizeof(script_cmd), "bash %s -x", script_path);
	return_code = system(script_cmd);

	cr_assert_eq(return_code, 0,
		     "excp_profile_setup() failed %d", return_code);
}

/* Test basic RMA Put command */
void rma_simple_put(void)
{
	struct mem_window src_mem;
	struct mem_window dst_mem;
	int pid_idx = 0;

	/* Allocate buffers */
	alloc_iobuf(WIN_LENGTH, &dst_mem, CXI_MAP_WRITE);
	alloc_iobuf(WIN_LENGTH, &src_mem, CXI_MAP_READ);

	/* Initialize Send Memory */
	for (int i = 0; i < src_mem.length; i++)
		src_mem.buffer[i] = i;

	/* Initialize RMA PtlTE and Post RMA Buffer */
	ptlte_setup(pid_idx, false, false);

	append_le_sync(rx_pte, &dst_mem, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, true, false, NULL);

	/* Do a set of xfers in increments until the whole source mem is xfered
	 * to the destination mem
	 */
	for (int xfer_len = 1; xfer_len <= src_mem.length; xfer_len <<= 1) {
		/* Initiate Put Operation */
		memset(dst_mem.buffer, 0, dst_mem.length);
		do_put_sync(src_mem, xfer_len, 0, 0, pid_idx, true, 0, 0, 0,
			    false);

		/* Validate Source and Destination Data Match */
		for (int i = 0; i < xfer_len; i++)
			cr_expect_eq(src_mem.buffer[i],
				     dst_mem.buffer[i],
				     "Data mismatch: idx %4d - %02x != %02x",
				     i, src_mem.buffer[i], dst_mem.buffer[i]);
	}

	/* Clean up PTE and RMA buffer */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID);
	ptlte_teardown();

	/* Deallocate buffers */
	free_iobuf(&src_mem);
	free_iobuf(&dst_mem);
}

/* Test to validate RMA Put command after an exclusive cp modify call */
void rma_put_with_cp_modify(void)
{
	struct mem_window src_mem;
	struct mem_window dst_mem;
	int pid_idx = 0;
	int ret = -1;
	struct cxi_cq_alloc_opts cq_opts;

	excp_profile_setup();

	/* Set up Exclusive Communication Profile*/
	ret = cxil_alloc_cp(lni, vni_excp, CXI_TC_BEST_EFFORT,
			    CXI_TC_TYPE_DEFAULT, &excp);
	cr_assert_eq(ret, 0, "cxil_alloc_cp() failed %d", ret);
	cr_assert_neq(excp, NULL);
	cr_log_info("assigned LCID: %u\n", excp->lcid);

	/* Set up Domain to send data from Exclusive Communication Profile*/
	ret = cxil_alloc_domain(lni, vni_excp, domain_pid, &domain_excp);
	cr_log_info("assigned PID: %u\n", domain_excp->pid);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(domain_excp, NULL);

	/* Allocate CMDQs for exclusive cp */
	transmit_eq_buf_len = s_page_size;
	target_eq_buf_len = s_page_size * 2;
	memset(&cq_opts, 0, sizeof(cq_opts));
	cq_opts.count = 1024;
	cq_opts.flags = CXI_CQ_IS_TX;
	cq_opts.lcid = excp->lcid;
	ret = cxil_alloc_cmdq(lni, NULL, &cq_opts, &transmit_cmdq_excp);
	cr_assert_eq(ret, 0, "TX cxil_alloc_cmdq() failed %d", ret);

	/* Allocate buffers */
	alloc_iobuf(WIN_LENGTH, &dst_mem, CXI_MAP_WRITE);
	alloc_iobuf(WIN_LENGTH, &src_mem, CXI_MAP_READ);

	/* Initialize Send Memory */
	for (int i = 0; i < src_mem.length; i++)
		src_mem.buffer[i] = i;

	/* Initialize RMA PtlTE and Post RMA Buffer */
	ptlte_setup(pid_idx, false, true);

	append_le_sync(rx_pte, &dst_mem, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, true, false, NULL);

	/* Do a set of xfers in increments until the whole source mem is xfered
	 * to the destination mem
	 */
	for (int xfer_len = 1; xfer_len <= src_mem.length; xfer_len <<= 1) {
		/* Initiate Put Operation */
		memset(dst_mem.buffer, 0, dst_mem.length);
		do_put_sync(src_mem, xfer_len, 0, 0, pid_idx, true, 0, 0, 0,
			    true);

		/* Validate Source and Destination Data Match */
		for (int i = 0; i < xfer_len; i++)
			cr_expect_eq(src_mem.buffer[i],
				     dst_mem.buffer[i],
				     "Data mismatch: idx %4d - %02x != %02x",
				     i, src_mem.buffer[i], dst_mem.buffer[i]);
	}

	ret = cxil_modify_cp(lni, excp, INVALID_VNI);
	cr_assert_neq(ret, 0,
		      "cxil_modify_cp() was successful for invalid VNI %u",
		      vni_excp);

	ret = cxil_modify_cp(lni, excp, VALID_VNI);
	cr_assert_eq(ret, 0, "cxil_modify_cp() failed %d", ret);

	/* Do a set of xfers in increments until the whole source mem is xfered
	 * to the destination mem
	 */
	for (int xfer_len = 1; xfer_len <= src_mem.length; xfer_len <<= 1) {
		/* Initiate Put Operation */
		memset(dst_mem.buffer, 0, dst_mem.length);
		do_put_sync(src_mem, xfer_len, 0, 0, pid_idx, true, 0, 0, 0,
			    true);

		/* Validate Source and Destination Data Match */
		for (int i = 0; i < xfer_len; i++)
			cr_expect_eq(src_mem.buffer[i],
				     dst_mem.buffer[i],
				     "Data mismatch: idx %4d - %02x != %02x",
				     i, src_mem.buffer[i], dst_mem.buffer[i]);
	}

	/* Clean up PTE and RMA buffer */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID);
	ptlte_teardown();

	/* Deallocate buffers */
	free_iobuf(&src_mem);
	free_iobuf(&dst_mem);

	ret = cxil_destroy_cmdq(transmit_cmdq_excp);
	cr_assert_eq(ret, 0, "Destroy TX CQ Failed %d", ret);

	ret = cxil_destroy_domain(domain_excp);
	cr_expect_eq(ret, 0,
		     "%s: cxil_destroy_domain() returns (%d) %s",
		     __func__, ret, strerror(-ret));
	domain_excp = NULL;

	ret = cxil_destroy_cp(excp);
	cr_assert_eq(ret, 0, "Destroy CP failed %d", ret);
	excp = NULL;

	excp_profile_teardown();
}

Test(rma, simple_put, .timeout = 15, .disabled = PUT_DISABLED)
{
	rma_simple_put();
}

Test(rma, put_with_cp_modify, .timeout = 15, .disabled = PUT_DISABLED)
{
	rma_put_with_cp_modify();
}

/* Test basic RMA Get command */
Test(rma, simple_get, .timeout = 15, .disabled = GET_DISABLED)
{
	struct mem_window src_mem;
	struct mem_window dst_mem;
	int pid_idx = 0;

	/* Allocate buffers */
	alloc_iobuf(WIN_LENGTH, &src_mem, CXI_MAP_READ);
	alloc_iobuf(WIN_LENGTH, &dst_mem, CXI_MAP_WRITE);

	/* Initialize Source Memory */
	for (int i = 0; i < src_mem.length; i++)
		src_mem.buffer[i] = i;

	/* Set up PTE and RMA buffer */
	ptlte_setup(pid_idx, false, false);

	append_le_sync(rx_pte, &src_mem, C_PTL_LIST_PRIORITY, GET_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, false, true, NULL);

	/* Do a set of xfers in increments until the whole source mem is xfered
	 * to the destination mem
	 */
	for (int xfer_len = 1; xfer_len <= src_mem.length; xfer_len <<= 1) {
		/* Initiate Get Operation */
		memset(dst_mem.buffer, 0, dst_mem.length);
		do_get_sync(dst_mem, xfer_len, 0, pid_idx, true, 0, 0, 0,
			     transmit_evtq);

		/* Validate Source and Destination Data Match */
		for (int i = 0; i < xfer_len; i++)
			cr_expect_eq(src_mem.buffer[i],
				     dst_mem.buffer[i],
				     "Data mismatch: idx %4d - %02x != %02x",
				     i, src_mem.buffer[i], dst_mem.buffer[i]);
	}

	/* Clean up PTE and RMA buffer */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, GET_BUFFER_ID);
	ptlte_teardown();

	/* Deallocate buffers */
	free_iobuf(&dst_mem);
	free_iobuf(&src_mem);
}

void data_xfer_setup_eqpt(void)
{
	transmit_eq_attr.flags = target_eq_attr.flags = CXI_EQ_PASSTHROUGH;
	data_xfer_setup();
}

TestSuite(rma_eqpt, .init = data_xfer_setup_eqpt, .fini = data_xfer_teardown);

Test(rma_eqpt, simple_put, .timeout = 5, .disabled = PUT_DISABLED)
{
	rma_simple_put();
}
