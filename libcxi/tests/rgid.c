/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Copyright 2018 Cray Inc. All rights reserved */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <sys/queue.h>

#include "libcxi_test_common.h"
#include "uapi/misc/cxi.h"

#define WIN_LENGTH (0x1000)
#define PUT_BUFFER_ID 0xb0f
#define LNIS_PER_RGID 2
#define LNIS_MAX (C_NUM_RGIDS * LNIS_PER_RGID)

struct rgid_objs {
	struct cxil_lni *lni;
	struct cxi_cp *cp;
	struct cxil_domain *domain;
	struct cxi_cq *transmit_cmdq;
	struct cxi_cq *target_cmdq;
	struct cxil_wait_obj *wait_obj;
	struct cxi_eq_attr transmit_eq_attr;
	size_t transmit_eq_buf_len;
	void *transmit_eq_buf;
	struct cxi_md *transmit_eq_md;
	struct cxi_eq *transmit_evtq;
	struct cxi_eq_attr target_eq_attr;
	size_t target_eq_buf_len;
	void *target_eq_buf;
	struct cxi_md *target_eq_md;
	struct cxi_eq *target_evtq;
	struct cxil_pte *rx_pte;
	struct cxil_pte_map *rx_pte_map;
};

struct rgid_objs obj[2] = {};
struct cxi_svc_desc svc_desc = {
	.restricted_vnis = 1,
	.num_vld_vnis = 1,
	.vnis[0] = 8,
};
struct cxi_rsrc_use rsrc_use = {};

int get_rgids_avail(void)
{
	int rc;
	FILE *fp;
	int rgids = -1;

	fp = fopen("/sys/class/cxi/cxi0/device/properties/rgids_avail", "r");
	if (fp) {
		rc = fscanf(fp, "%d", &rgids);
		cr_assert_eq(rc, 1, "rc=%d\n", rc);
		fclose(fp);
	}


	return rgids;
}

void rgid_teardown(struct rgid_objs *obj)
{
	int ret;

	ret = cxil_get_svc_rsrc_use(dev, svc_desc.svc_id, &rsrc_use);
	cr_assert_eq(ret, 0, "Failed to get resource usage for service_id %d: %s",
		     svc_desc.svc_id, strerror(-ret));

	/* Destroy EQs */
	ret = cxil_destroy_evtq(obj->transmit_evtq);
	cr_assert_eq(ret, 0, "Destroy TX EQ Failed %d", ret);

	ret = cxil_unmap(obj->transmit_eq_md);
	cr_assert(!ret);
	free(obj->transmit_eq_buf);

	ret = cxil_destroy_evtq(obj->target_evtq);
	cr_assert_eq(ret, 0, "Destroy RX EQ Failed %d", ret);

	ret = cxil_unmap(obj->target_eq_md);
	cr_assert(!ret);
	free(obj->target_eq_buf);

	/* Destroy wait object */
	ret = cxil_destroy_wait_obj(obj->wait_obj);
	cr_assert_eq(ret, 0, "Destroy wait obj Failed %d", ret);

	/* Destroy CQs */
	ret = cxil_destroy_cmdq(obj->target_cmdq);
	cr_assert_eq(ret, 0, "Destroy RX CQ Failed %d", ret);
	ret = cxil_destroy_cmdq(obj->transmit_cmdq);
	cr_assert_eq(ret, 0, "Destroy TX CQ Failed %d", ret);

	ret = cxil_destroy_domain(obj->domain);
	cr_expect_eq(ret, 0, "cxil_destroy_domain() returns (%d) %s",
		     ret, strerror(-ret));
	obj->domain = NULL;

	/* Destroy CP */
	ret = cxil_destroy_cp(obj->cp);
	cr_assert_eq(ret, 0, "Destroy CP failed %d", ret);

	obj->cp = NULL;

	ret = cxil_destroy_lni(obj->lni);
	cr_expect_eq(ret, 0, "cxil_destroy_lni() returns (%d) %s",
		     ret, strerror(-ret));
	obj->lni = NULL;
}

void rgid_setup(struct rgid_objs *obj)
{
	int ret;
	struct cxi_cq_alloc_opts cq_opts;

	ret = cxil_alloc_lni(dev, &obj->lni, svc_desc.svc_id);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(obj->lni, NULL);

	/* Set up Communication Profile */
	ret = cxil_alloc_cp(obj->lni, vni, CXI_TC_BEST_EFFORT,
			    CXI_TC_TYPE_DEFAULT, &obj->cp);
	cr_assert_eq(ret, 0, "cxil_alloc_cp() failed %d", ret);
	cr_assert_neq(obj->cp, NULL);
	cr_log_info("assigned LCID: %u\n", obj->cp->lcid);

	ret = cxil_alloc_domain(obj->lni, vni, domain_pid, &obj->domain);
	cr_log_info("assigned PID: %u\n", obj->domain->pid);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(obj->domain, NULL);

	obj->transmit_eq_buf_len = s_page_size;
	obj->target_eq_buf_len = s_page_size * 2;

	/* Allocate CMDQs */
	memset(&cq_opts, 0, sizeof(cq_opts));
	cq_opts.count = 1024;
	cq_opts.flags = CXI_CQ_IS_TX;
	cq_opts.lcid = obj->cp->lcid;
	ret = cxil_alloc_cmdq(obj->lni, NULL, &cq_opts, &obj->transmit_cmdq);
	cr_assert_eq(ret, 0, "TX cxil_alloc_cmdq() failed %d", ret);
	cq_opts.flags = 0;
	ret = cxil_alloc_cmdq(obj->lni, NULL, &cq_opts, &obj->target_cmdq);
	cr_assert_eq(ret, 0, "RX cxil_alloc_cmdq() failed %d", ret);

	/* Allocate wait object */
	ret = cxil_alloc_wait_obj(obj->lni, &obj->wait_obj);
	cr_assert_eq(ret, 0, "cxil_alloc_wait_obj() failed %d", ret);

	/* Allocate EQs */
	obj->transmit_eq_buf = aligned_alloc(s_page_size,
					     obj->transmit_eq_buf_len);
	cr_assert(obj->transmit_eq_buf);
	memset(obj->transmit_eq_buf, 0, obj->transmit_eq_buf_len);

	ret = cxil_map(obj->lni, obj->transmit_eq_buf, obj->transmit_eq_buf_len,
		       CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		       NULL, &obj->transmit_eq_md);
	cr_assert_eq(ret, 0, "cxil_map() failed %d", ret);

	obj->transmit_eq_attr.queue = obj->transmit_eq_buf;
	obj->transmit_eq_attr.queue_len = obj->transmit_eq_buf_len;
	obj->transmit_eq_attr.flags = 0;

	ret = cxil_alloc_evtq(obj->lni, obj->transmit_eq_md,
			      &obj->transmit_eq_attr, obj->wait_obj,
			      obj->wait_obj, &obj->transmit_evtq);
	cr_assert_eq(ret, 0, "Allocate TX EQ Failed %d", ret);

	obj->target_eq_buf = aligned_alloc(s_page_size, obj->target_eq_buf_len);
	cr_assert(obj->target_eq_buf);
	memset(obj->target_eq_buf, 0, obj->target_eq_buf_len);

	ret = cxil_map(obj->lni, obj->target_eq_buf, obj->target_eq_buf_len,
		       CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		       NULL, &obj->target_eq_md);
	cr_assert_eq(ret, 0, "cxil_map() failed %d", ret);

	obj->target_eq_attr.queue = obj->target_eq_buf;
	obj->target_eq_attr.queue_len = obj->target_eq_buf_len;
	obj->target_eq_attr.flags = 0;

	ret = cxil_alloc_evtq(obj->lni, obj->target_eq_md, &obj->target_eq_attr,
			      obj->wait_obj, obj->wait_obj, &obj->target_evtq);
	cr_assert_eq(ret, 0, "Allocate RX EQ Failed %d", ret);
}

static void service_setup(struct cxi_svc_desc *svc_desc)
{
	int ret;
	int svc_id;

	svc_id = cxil_alloc_svc(dev, svc_desc, NULL);
	cr_assert_gt(svc_id, 0, "cxil_alloc_svc(): Failed. ret:%d", svc_id);

	svc_desc->svc_id = svc_id;
	svc_desc->enable = true;

	ret = cxil_update_svc(dev, svc_desc, NULL);
	cr_assert_eq(ret, 0, "cxil_update_svc(): Failed ret:%d", ret);

	ret = cxil_set_svc_lpr(dev, svc_id, LNIS_PER_RGID);
	cr_assert_eq(ret, 0, "cxil_set_svc_lpr(): Failed ret:%d", ret);

	ret = cxil_get_svc_lpr(dev, svc_id);
	cr_assert_eq(ret, LNIS_PER_RGID,
		     "cxil_get_svc_lpr() did not return correct LNIs per RGID (%d):%d",
		     LNIS_PER_RGID, ret);
}

static void setup(void)
{
	dev_setup();
	service_setup(&svc_desc);
	rgid_setup(&obj[0]);
	rgid_setup(&obj[1]);
}

static void rgid_ptlte_tear_down(struct rgid_objs *obj)
{
	int rc;

	rc = cxil_unmap_pte(obj->rx_pte_map);
	cr_assert_eq(rc, 0, "RX cxil_unmap_pte failed %d", rc);

	rc = cxil_destroy_pte(obj->rx_pte);
	cr_assert_eq(rc, 0, "RX cxil_destroy_pte failed %d", rc);
}

static void service_teardown(struct cxi_svc_desc *svc_desc)
{
	int ret = cxil_destroy_svc(dev, svc_desc->svc_id);

	cr_assert_eq(ret, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d, ret: %d",
		     svc_desc->svc_id, ret);
}

static void teardown(void)
{
	rgid_teardown(&obj[1]);
	rgid_teardown(&obj[0]);
	service_teardown(&svc_desc);
	dev_teardown();
}

static void rgid_ptlte_setup(struct rgid_objs *obj, uint32_t pid_idx, bool matching)
{
	int rc;
	union c_cmdu cmd = {};
	const union c_event *event;
	unsigned int ptn;
	enum c_ptlte_state state;
	struct cxi_pt_alloc_opts pt_opts = {};

	pt_opts.is_matching = matching;

	/* Allocate */
	rc = cxil_alloc_pte(obj->lni, obj->target_evtq, &pt_opts, &obj->rx_pte);
	cr_assert_eq(rc, 0, "RX cxil_alloc_pte failed %d", rc);

	/* Map */
	rc = cxil_map_pte(obj->rx_pte, obj->domain, pid_idx, false,
			  &obj->rx_pte_map);
	cr_assert_eq(rc, 0, "RX cxil_map_pte failed %d", rc);

	/* Enable */
	cmd.set_state.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = obj->rx_pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	rc = cxi_cq_emit_target(obj->target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "cxi_cq_emit_target failed %d", rc);

	cxi_cq_ring(obj->target_cmdq);

	/* Wait for enable response EQ event */
	while (!(event = cxi_eq_get_event(obj->target_evtq)))
		sched_yield();

	state = event->tgt_long.initiator.state_change.ptlte_state;
	ptn = event->tgt_long.ptlte_index;
	cr_assert_eq(event->hdr.event_type, C_EVENT_STATE_CHANGE,
		     "Invalid event_type, expected: %d got %d",
		     C_EVENT_STATE_CHANGE, event->hdr.event_type);
	cr_assert_eq(state, C_PTLTE_ENABLED,
		     "Invalid state, expected: %d got %d", C_PTLTE_ENABLED,
		     state);
	cr_assert_eq(ptn, obj->rx_pte->ptn, "Invalid ptn, %d != %d",
		     ptn, obj->rx_pte->ptn);

	cxi_eq_ack_events(obj->target_evtq);
}

static void alloc_map(struct rgid_objs *obj, size_t len, struct mem_window *win,
		      uint32_t prot)
{
	int rc;

	prot &= (CXI_MAP_WRITE | CXI_MAP_READ);

	memset(&win->md, 0, sizeof(win->md));
	win->length = len;
	win->buffer = aligned_alloc(s_page_size, win->length);
	win->loc = on_host;

	cr_assert_not_null(win->buffer, "Failed to allocate iobuf");
	memset(win->buffer, 0, win->length);

	rc = cxil_map(obj->lni, win->buffer, win->length,
		      CXI_MAP_PIN | prot, NULL, &win->md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
}

static void rgid_append_le(struct rgid_objs *obj,
			   const struct cxil_pte *pte,
			   struct mem_window *mem_win,
			   enum c_ptl_list list,
			   uint32_t buffer_id,
			   uint64_t match_bits,
			   uint64_t ignore_bits,
			   uint32_t match_id,
			   uint64_t min_free,
			   bool event_success_disable,
			   bool event_unlink_disable,
			   bool use_once,
			   bool manage_local,
			   bool no_truncate,
			   bool op_put,
			   bool op_get)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode     = C_CMD_TGT_APPEND;
	cmd.target.ptl_list    = list;
	cmd.target.ptlte_index = pte->ptn;
	cmd.target.op_put      = op_put ? 1 : 0;
	cmd.target.op_get      = op_get ? 1 : 0;
	cmd.target.manage_local = manage_local ? 1 : 0;
	cmd.target.no_truncate = no_truncate ? 1 : 0;
	cmd.target.unexpected_hdr_disable = 0;
	cmd.target.buffer_id   = buffer_id;
	cmd.target.lac         = mem_win->md->lac;
	cmd.target.start       = CXI_VA_TO_IOVA(mem_win->md, mem_win->buffer);
	cmd.target.length      = mem_win->length;
	cmd.target.event_success_disable = event_success_disable ? 1 : 0;
	cmd.target.event_unlink_disable = event_unlink_disable ? 1 : 0;
	cmd.target.use_once    = use_once ? 1 : 0;
	cmd.target.match_bits  = match_bits;
	cmd.target.ignore_bits = ignore_bits;
	cmd.target.match_id    = match_id;
	cmd.target.min_free    = min_free >> dev->info.min_free_shift;

	rc = cxi_cq_emit_target(obj->target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(obj->target_cmdq);
}

void rgid_append_le_sync(struct rgid_objs *obj,
			 const struct cxil_pte *pte,
			 struct mem_window *mem_win,
			 enum c_ptl_list list,
			 uint32_t buffer_id,
			 uint64_t match_bits,
			 uint64_t ignore_bits,
			 uint32_t match_id,
			 uint64_t min_free,
			 bool event_success_disable,
			 bool event_unlink_disable,
			 bool use_once,
			 bool manage_local,
			 bool no_truncate,
			 bool op_put,
			 bool op_get,
			 union c_event *event)
{
	rgid_append_le(obj, pte, mem_win, list, buffer_id,
		  match_bits, ignore_bits, match_id, min_free,
		  event_success_disable, event_success_disable, use_once,
		  manage_local, no_truncate, op_put, op_get);

	process_eqe(obj->target_evtq, EQE_TGT_LONG, C_EVENT_LINK, buffer_id,
		    event);
}

void rgid_do_put_sync(struct rgid_objs *obj, struct mem_window mem_win,
		      size_t len, uint64_t r_off, uint64_t l_off,
		      unsigned int pid, uint32_t pid_idx, bool restricted,
		      uint64_t match_bits, uint64_t user_ptr,
		      uint32_t initiator)
{
	union c_cmdu cmd = {};
	union c_fab_addr dfa;
	uint8_t idx_ext;
	int rc;

	cxi_build_dfa(dev->info.nid, pid, dev->info.pid_bits,
		      pid_idx, &dfa, &idx_ext);

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_PUT;
	cmd.full_dma.index_ext = idx_ext;
	cmd.full_dma.lac = mem_win.md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.remote_offset = r_off;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(mem_win.md,
						 mem_win.buffer + l_off);
	cmd.full_dma.eq = obj->transmit_evtq->eqn;
	cmd.full_dma.user_ptr = user_ptr;
	cmd.full_dma.request_len = len;
	cmd.full_dma.restricted = restricted ? 1 : 0;
	cmd.full_dma.match_bits = match_bits;
	cmd.full_dma.initiator = initiator;

	rc = cxi_cq_emit_dma(obj->transmit_cmdq, &cmd.full_dma);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(obj->transmit_cmdq);
	process_eqe(obj->transmit_evtq, EQE_INIT_SHORT, C_EVENT_ACK, 0, NULL);
}

void rgid_unlink_le(struct rgid_objs *obj, const struct cxil_pte *pte,
		  enum c_ptl_list list, uint32_t buffer_id)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = list;
	cmd.target.buffer_id = buffer_id;
	cmd.target.ptlte_index = pte->ptn;

	rc = cxi_cq_emit_target(obj->target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(obj->target_cmdq);
}

void rgid_unlink_le_sync(struct rgid_objs *obj, const struct cxil_pte *pte,
		       enum c_ptl_list list, uint32_t buffer_id)
{
	rgid_unlink_le(obj, pte, list, buffer_id);

	/* Wait for an unlink EQ event */
	process_eqe(obj->target_evtq, EQE_TGT_LONG, C_EVENT_UNLINK, buffer_id,
		    NULL);
}

void simple_put(void)
{
	int i;
	int pid_idx = 0;
	struct mem_window src_mem;
	struct mem_window dst_mem;

	/* Allocate buffers */
	alloc_map(&obj[0], WIN_LENGTH, &src_mem, CXI_MAP_READ);
	alloc_map(&obj[1], WIN_LENGTH, &dst_mem, CXI_MAP_WRITE);

	/* Initialize source buffer */
	for (int i = 0; i < src_mem.length; i++)
		src_mem.buffer[i] = i;

	/* Initialize RMA PtlTE and Post RMA Buffer */
	rgid_ptlte_setup(&obj[1], pid_idx, false);
	rgid_append_le_sync(&obj[1], obj[1].rx_pte, &dst_mem,
			    C_PTL_LIST_PRIORITY, PUT_BUFFER_ID, 0, 0,
			    CXI_MATCH_ID_ANY, 0, true, false, false, false,
			    false, true, false, NULL);

	memset(dst_mem.buffer, 0, dst_mem.length);
	rgid_do_put_sync(&obj[0], src_mem, src_mem.length, 0, 0,
			 obj[1].domain->pid, pid_idx, true, 0, 0, 0);

	/* Validate Source and Destination Data Match */
	for (i = 0; i < dst_mem.length; i++)
		cr_assert_eq(src_mem.buffer[i], dst_mem.buffer[i],
			     "Data mismatch: i:%4d - %02x != %02x",
			     i, src_mem.buffer[i], dst_mem.buffer[i]);

	/* Clean up PTE and RMA buffer */
	rgid_unlink_le_sync(&obj[1], obj[1].rx_pte, C_PTL_LIST_PRIORITY,
			    PUT_BUFFER_ID);
	rgid_ptlte_tear_down(&obj[1]);

	free_iobuf(&dst_mem);
	free_iobuf(&src_mem);
}

TestSuite(rgid, .init = setup, .fini = teardown);

Test(rgid, simple_put)
{
	simple_put();
}

Test(rgid_share, alloc_lnis)
{
	int i;
	int ret;
	int rgids_avail;
	struct cxil_lni **lnis;

	dev_setup();
	service_setup(&svc_desc);

	lnis = calloc(LNIS_MAX, sizeof(*lnis));
	cr_assert_neq(lnis, NULL, "Could not alloc lni array");

	rgids_avail = get_rgids_avail();

	for (i = 0; i < rgids_avail * LNIS_PER_RGID + 1; i++) {
		ret = cxil_alloc_lni(dev, &lnis[i], svc_desc.svc_id);
		if (ret) {
			/* should fail at rgids_avail * LNIS_PER_RGID + 1 */
			cr_expect_neq(i, rgids_avail * LNIS_PER_RGID + 1,
				     "cxil_lni_alloc failed (%d) at %d LNIs\n",
				     ret, i);
			break;
		}
	}

	cr_log_info("Have %d LNIs\n", i);
	cr_assert_eq(get_rgids_avail(), 0, "Available rgids (%d) should be 0\n",
		     get_rgids_avail());

	for (i--; i >= 0; i--) {
		ret = cxil_destroy_lni(lnis[i]);
		cr_expect_eq(ret, 0, "cxil_destroy_lni(%d) returns (%d) %s",
			     i, ret, strerror(-ret));
	}

	cr_assert_eq(get_rgids_avail(), rgids_avail,
		     "Available rgids (%d) should be %d\n", get_rgids_avail(),
		     rgids_avail);

	free(lnis);
	service_teardown(&svc_desc);
	dev_teardown();
}
