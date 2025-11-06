/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018, 2020 Hewlett Packard Enterprise Development LP */

#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <poll.h>
#include <sched.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "libcxi_priv.h"
#include "libcxi_test_common.h"

uint32_t dev_id; /* Dev 0 */
uint32_t vni = 1;
uint32_t vni_excp = 16U;
uint32_t domain_pid = C_PID_ANY;
struct cxil_dev *dev;
struct cxil_lni *lni;
struct cxi_cp *cp;
struct cxi_cp *excp;
struct cxil_domain *domain;
struct cxil_domain *domain_excp;
struct cxi_cq *transmit_cmdq;
struct cxi_cq *transmit_cmdq_excp;
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
struct cxil_test_data *test_data;
int test_data_len;

/* Counting event and trigger ops stuff. */
struct cxi_ct *ct;
struct cxi_cq *trig_cmdq;
struct c_ct_writeback *wb;

int (*gpu_malloc)(struct mem_window *win);
int (*gpu_host_alloc)(struct mem_window *win);
int (*gpu_free)(void *devPtr);
int (*gpu_host_free)(void *p);
int (*gpu_memset)(void *devPtr, int value, size_t count);
int (*gpu_memcpy)(void *dst, const void *src, size_t count,
		  enum gpu_copy_dir dir);
int (*gpu_props)(struct mem_window *win, void **base, size_t *size);
int (*gpu_close_fd)(int dma_buf_fd);

int s_page_size;

/* Get _SC_PAGESIZE */
static void set_system_page_size(void)
{
	if (!s_page_size)
		s_page_size = sysconf(_SC_PAGESIZE);
}

bool is_netsim(void)
{
	return dev->info.device_platform == C_PLATFORM_NETSIM;
}

void dev_setup(void)
{
	int ret;

	set_system_page_size();
	ret = cxil_open_device(0, &dev);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(dev, NULL);
}

void dev_teardown(void)
{
	cxil_close_device(dev);
	dev = NULL;
}

void lni_setup(void)
{
	int ret;

	dev_setup();

	ret = cxil_alloc_lni(dev, &lni, CXI_DEFAULT_SVC_ID);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(lni, NULL);
}

void lni_teardown(void)
{
	int ret;

	ret = cxil_destroy_lni(lni);
	cr_expect_eq(ret, 0, "%s: cxil_destroy_lni() returns (%d) %s",
		     __func__, ret, strerror(-ret));
	lni = NULL;

	dev_teardown();
}


void cp_setup(void)
{
	int ret;

	lni_setup();

	/* Set up Communication Profile */
	ret = cxil_alloc_cp(lni, vni, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_DEFAULT,
			    &cp);
	cr_assert_eq(ret, 0, "cxil_alloc_cp() failed %d", ret);
	cr_assert_neq(cp, NULL);
	cr_log_info("assigned LCID: %u\n", cp->lcid);
}

void cp_teardown(void)
{
	int ret;

	/* Destroy CP */
	ret = cxil_destroy_cp(cp);
	cr_assert_eq(ret, 0, "Destroy CP failed %d", ret);

	cp = NULL;

	lni_teardown();
}

void domain_setup(void)
{
	int ret;

	cp_setup();

	ret = cxil_alloc_domain(lni, vni, domain_pid, &domain);
	cr_log_info("assigned PID: %u\n", domain->pid);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(domain, NULL);
}

void domain_teardown(void)
{
	int ret;

	ret = cxil_destroy_domain(domain);
	cr_expect_eq(ret, 0, "%s: cxil_destroy_domain() returns (%d) %s",
		     __func__, ret, strerror(-ret));
	domain = NULL;

	cp_teardown();
}

void data_xfer_setup(void)
{
	int ret;
	struct cxi_cq_alloc_opts cq_opts;

	domain_setup();

	transmit_eq_buf_len = s_page_size;
	target_eq_buf_len = s_page_size * 2;

	/* Allocate CMDQs */
	memset(&cq_opts, 0, sizeof(cq_opts));
	cq_opts.count = 1024;
	cq_opts.flags = CXI_CQ_IS_TX;
	cq_opts.lcid = cp->lcid;
	ret = cxil_alloc_cmdq(lni, NULL, &cq_opts, &transmit_cmdq);
	cr_assert_eq(ret, 0, "TX cxil_alloc_cmdq() failed %d", ret);

	cq_opts.flags = 0;
	ret = cxil_alloc_cmdq(lni, NULL, &cq_opts, &target_cmdq);
	cr_assert_eq(ret, 0, "RX cxil_alloc_cmdq() failed %d", ret);

	/* Allocate wait object */
	ret = cxil_alloc_wait_obj(lni, &wait_obj);
	cr_assert_eq(ret, 0, "cxil_alloc_wait_obj() failed %d", ret);

	/* Allocate EQs */
	transmit_eq_buf = aligned_alloc(s_page_size, transmit_eq_buf_len);
	cr_assert(transmit_eq_buf);
	memset(transmit_eq_buf, 0, transmit_eq_buf_len);

	ret = cxil_map(lni, transmit_eq_buf, transmit_eq_buf_len,
		       CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		       NULL, &transmit_eq_md);
	cr_assert_eq(ret, 0, "cxil_map() failed %d", ret);

	transmit_eq_attr.queue = transmit_eq_buf;
	transmit_eq_attr.queue_len = transmit_eq_buf_len;
	transmit_eq_attr.flags = 0;

	ret = cxil_alloc_evtq(lni, transmit_eq_md, &transmit_eq_attr,
			      wait_obj, wait_obj, &transmit_evtq);
	cr_assert_eq(ret, 0, "Allocate TX EQ Failed %d", ret);

	target_eq_buf = aligned_alloc(s_page_size, target_eq_buf_len);
	cr_assert(target_eq_buf);
	memset(target_eq_buf, 0, target_eq_buf_len);

	ret = cxil_map(lni, target_eq_buf, target_eq_buf_len,
		       CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		       NULL, &target_eq_md);
	cr_assert_eq(ret, 0, "cxil_map() failed %d", ret);

	target_eq_attr.queue = target_eq_buf;
	target_eq_attr.queue_len = target_eq_buf_len;
	target_eq_attr.flags = 0;

	ret = cxil_alloc_evtq(lni, target_eq_md, &target_eq_attr, wait_obj,
			      wait_obj, &target_evtq);
	cr_assert_eq(ret, 0, "Allocate RX EQ Failed %d", ret);
}

void data_xfer_teardown(void)
{
	int ret;

	/* Destroy EQs */
	ret = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(ret, 0, "Destroy TX EQ Failed %d", ret);

	ret = cxil_unmap(transmit_eq_md);
	cr_assert(!ret);
	free(transmit_eq_buf);

	ret = cxil_destroy_evtq(target_evtq);
	cr_assert_eq(ret, 0, "Destroy RX EQ Failed %d", ret);

	ret = cxil_unmap(target_eq_md);
	cr_assert(!ret);
	free(target_eq_buf);

	/* Destroy wait object */
	ret = cxil_destroy_wait_obj(wait_obj);
	cr_assert_eq(ret, 0, "Destroy wait obj Failed %d", ret);

	/* Destroy CQs */
	ret = cxil_destroy_cmdq(target_cmdq);
	cr_assert_eq(ret, 0, "Destroy RX CQ Failed %d", ret);
	ret = cxil_destroy_cmdq(transmit_cmdq);
	cr_assert_eq(ret, 0, "Destroy TX CQ Failed %d", ret);

	domain_teardown();
}

void counting_event_setup(void)
{
	int ret;
	struct cxi_cq_alloc_opts cq_opts = {
		.count = 256,
		.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS,
	};

	data_xfer_setup();

	wb = aligned_alloc(8, sizeof(*wb));
	cr_assert_neq(wb, NULL, "Failed to allocated memory");

	/* Use the transmit EQ for trigger CMDQ. */
	ret = cxil_alloc_cmdq(lni, transmit_evtq, &cq_opts, &trig_cmdq);
	cr_assert_eq(ret, 0, "Triggered cxil_alloc_cmdq() failed %d", ret);

	ret = cxil_alloc_ct(lni, wb, &ct);
	cr_assert_eq(ret, 0, "Failed cxil_alloc_ct() failed %d", ret);
}

void counting_event_teardown(void)
{
	int ret;

	ret = cxil_destroy_ct(ct);
	cr_assert_eq(ret, 0, "Failed cxil_destroy_ct() failed %d", ret);

	ret = cxil_destroy_cmdq(trig_cmdq);
	cr_assert_eq(ret, 0, "Destroy Trigger CQ Failed %d", ret);

	free(wb);

	data_xfer_teardown();
}

void expect_ct_values(struct cxi_ct *ct, uint64_t success, uint8_t failure)
{
	time_t timeout;
	struct c_ct_writeback *wb = ct->wb;

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


/* Allocate and enable a PtlTE. */
void ptlte_setup(uint32_t pid_idx, bool matching, bool exclusive_cp)
{
	int rc;
	union c_cmdu cmd = {};
	const union c_event *event;
	unsigned int ptn;
	enum c_ptlte_state state;
	struct cxi_pt_alloc_opts pt_opts = {};
	struct cxil_domain *ptlte_domain = exclusive_cp ? domain_excp : domain;

	pt_opts.is_matching = matching;

	/* Allocate */
	rc = cxil_alloc_pte(lni, target_evtq, &pt_opts, &rx_pte);
	cr_assert_eq(rc, 0, "RX cxil_alloc_pte failed %d", rc);

	/* Map */
	rc = cxil_map_pte(rx_pte, ptlte_domain, pid_idx, false, &rx_pte_map);
	cr_assert_eq(rc, 0, "RX cxil_map_pte failed %d", rc);

	/* Enable */
	cmd.set_state.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = rx_pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	rc = cxi_cq_emit_target(target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "cxi_cq_emit_target failed %d", rc);

	cxi_cq_ring(target_cmdq);

	/* Wait for enable response EQ event */
	while (!(event = cxi_eq_get_event(target_evtq)))
		sched_yield();

	state = event->tgt_long.initiator.state_change.ptlte_state;
	ptn = event->tgt_long.ptlte_index;
	cr_assert_eq(event->hdr.event_type, C_EVENT_STATE_CHANGE,
		     "Invalid event_type, expected: %d got %d",
		     C_EVENT_STATE_CHANGE, event->hdr.event_type);
	cr_assert_eq(state, C_PTLTE_ENABLED,
		     "Invalid state, expected: %d got %d", C_PTLTE_ENABLED,
		     state);
	cr_assert_eq(ptn, rx_pte->ptn, "Invalid ptn, %d != %d",
		     ptn, rx_pte->ptn);

	cxi_eq_ack_events(target_evtq);
}

/* Undo ptlte_setup */
void ptlte_teardown(void)
{
	int rc;

	rc = cxil_unmap_pte(rx_pte_map);
	cr_assert_eq(rc, 0, "RX cxil_unmap_pte failed %d", rc);

	rc = cxil_destroy_pte(rx_pte);
	cr_assert_eq(rc, 0, "RX cxil_destroy_pte failed %d", rc);
}

void process_eqe(struct cxi_eq *evtq, enum eqe_fmt fmt, uint32_t type,
		 uint64_t id, union c_event *event_out)
{
	const union c_event *event;

	while (!(event = cxi_eq_get_event(evtq)))
		sched_yield();

	cr_assert_eq(event->hdr.event_type, type,
		     "Invalid event_type, expected: %d got %d", type,
		     event->hdr.event_type);

	if (event->hdr.return_code > C_RC_OK)
		cr_assert(event->hdr.return_code,
			  "Return code not OK, got %d",
			  event->hdr.return_code);

	switch (fmt) {
	case EQE_TGT_SHORT:
		cr_assert_eq(event->tgt_short.event_size, C_EVENT_SIZE_32_BYTE,
			     "Event is %d", event->tgt_short.event_size);
		if (id != (uint64_t)-1)
			cr_assert_eq(event->tgt_short.buffer_id, id,
				"Invalid buffer_id, expected: %lx, got %x", id,
				event->tgt_short.buffer_id);
		break;
	case EQE_TGT_LONG:
		cr_assert_eq(event->tgt_long.event_size, C_EVENT_SIZE_64_BYTE);
		if (id != (uint64_t)-1)
			cr_assert_eq(event->tgt_long.buffer_id, id,
				"Invalid buffer_id, expected: %lx, got %x", id,
				event->tgt_long.buffer_id);
		break;
	case EQE_INIT_SHORT:
		cr_assert_eq(event->init_short.event_size, C_EVENT_SIZE_16_BYTE,
			     "Invalid event size expected: %u, got: %u\n",
			     C_EVENT_SIZE_16_BYTE,
			     event->init_short.event_size);
		if (id != (uint64_t)-1)
			cr_assert_eq(event->init_short.user_ptr, id,
				"Invalid user_ptr, expected: %lx, got %lx", id,
				event->init_short.user_ptr);
		break;
	case EQE_EQ_SWITCH:
		cr_assert_eq(event->init_short.event_size, C_EVENT_SIZE_16_BYTE,
			     "Invalid event size expected: %u, got: %u\n",
			     C_EVENT_SIZE_16_BYTE,
			     event->init_short.event_size);
		cr_assert_eq(event->eq_switch.return_code, C_RC_OK);
		break;

	default:
		cr_assert_fail("Unsupported event type %d", fmt);
	}

	if (event_out)
		memcpy(event_out, event, sizeof(*event));

	cxi_eq_ack_events(evtq);
}

/* Append a buffer to a PtlTE. */
void append_le(const struct cxil_pte *pte,
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

	rc = cxi_cq_emit_target(target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(target_cmdq);
}

void append_le_sync(const struct cxil_pte *pte,
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
	append_le(pte, mem_win, list, buffer_id,
		  match_bits, ignore_bits, match_id, min_free,
		  event_success_disable, event_success_disable, use_once,
		  manage_local, no_truncate, op_put, op_get);

	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_LINK, buffer_id, event);
}

void unlink_le(const struct cxil_pte *pte, enum c_ptl_list list,
	       uint32_t buffer_id)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = list;
	cmd.target.buffer_id = buffer_id;
	cmd.target.ptlte_index = pte->ptn;

	rc = cxi_cq_emit_target(target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(target_cmdq);
}

void unlink_le_sync(const struct cxil_pte *pte, enum c_ptl_list list,
		    uint32_t buffer_id)
{
	unlink_le(pte, list, buffer_id);

	/* Wait for an unlink EQ event */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_UNLINK, buffer_id, NULL);
}

/* Block until an event may have occurred on an EQ */
int wait_for_event(struct cxil_wait_obj *wait)
{
	int rc;
	struct pollfd fds = {
		.fd = cxil_get_wait_obj_fd(wait),
		.events = POLLPRI | POLLERR,
	};

	/* timeout in msec, ret == 0 timeout, 1 event occurred */
	rc = poll(&fds, 1, 3000);

	if (rc > 0)
		cxil_clear_wait_obj(wait);

	return rc;
}

/* Do a DMA Put transaction. */
void do_put(struct mem_window mem_win, size_t len, uint64_t r_off,
	    uint64_t l_off, uint32_t pid_idx, bool restricted,
	    uint64_t match_bits, uint64_t user_ptr, uint32_t initiator,
	    bool exclusive_cp)
{
	union c_cmdu cmd = {};
	uint32_t dfa_domain_pid = exclusive_cp ? domain_excp->pid : domain->pid;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	int rc;
	struct cxi_cq *cmdq = exclusive_cp ? transmit_cmdq_excp : transmit_cmdq;

	cxi_build_dfa(dev->info.nid, dfa_domain_pid, dev->info.pid_bits,
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
	cmd.full_dma.eq = transmit_evtq->eqn;
	cmd.full_dma.user_ptr = user_ptr;
	cmd.full_dma.request_len = len;
	cmd.full_dma.restricted = restricted ? 1 : 0;
	cmd.full_dma.match_bits = match_bits;
	cmd.full_dma.initiator = initiator;

	rc = cxi_cq_emit_dma(cmdq, &cmd.full_dma);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(cmdq);
}

void do_put_sync(struct mem_window mem_win, size_t len, uint64_t r_off,
		 uint64_t l_off, uint32_t pid_idx, bool restricted,
		 uint64_t match_bits, uint64_t user_ptr, uint32_t initiator,
		 bool exclusive_cp)
{
	do_put(mem_win, len, r_off, l_off, pid_idx, restricted, match_bits,
	       user_ptr, initiator, exclusive_cp);

	process_eqe(transmit_evtq, EQE_INIT_SHORT, C_EVENT_ACK, user_ptr, NULL);
}

/* Do a Get transaction. */
void do_get(struct mem_window mem_win, size_t len, uint64_t r_off,
	     uint32_t pid_idx, bool restricted, uint64_t match_bits,
	     uint64_t user_ptr, uint32_t initiator,
	     struct cxi_eq *evtq)
{
	union c_cmdu cmd = {};
	union c_fab_addr dfa;
	uint8_t idx_ext;
	int rc;

	cxi_build_dfa(dev->info.nid, domain->pid, dev->info.pid_bits,
		      pid_idx, &dfa, &idx_ext);

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_GET;
	cmd.full_dma.index_ext = idx_ext;
	cmd.full_dma.lac = mem_win.md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.remote_offset = r_off;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(mem_win.md,
						 &mem_win.buffer[r_off]);
	cmd.full_dma.eq = evtq->eqn;
	cmd.full_dma.user_ptr = user_ptr;
	cmd.full_dma.request_len = len;
	cmd.full_dma.restricted = restricted ? 1 : 0;
	cmd.full_dma.match_bits = match_bits;
	cmd.full_dma.initiator = initiator;

	rc = cxi_cq_emit_dma(transmit_cmdq, &cmd.full_dma);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(transmit_cmdq);
}

void do_get_sync(struct mem_window mem_win, size_t len, uint64_t r_off,
		 uint32_t pid_idx, bool restricted, uint64_t match_bits,
		 uint64_t user_ptr, uint32_t initiator,
		 struct cxi_eq *evtq)
{

	do_get(mem_win, len, r_off, pid_idx, restricted, match_bits, user_ptr,
	       initiator, evtq);

	process_eqe(evtq, EQE_INIT_SHORT, C_EVENT_REPLY, user_ptr, NULL);
}

void alloc_iobuf(size_t len, struct mem_window *win, uint32_t prot)
{
	int rc;

	prot &= (CXI_MAP_WRITE | CXI_MAP_READ);

	memset(&win->md, 0, sizeof(win->md));
	win->length = len;
	win->buffer = aligned_alloc(s_page_size, win->length);
	win->loc = on_host;

	cr_assert_not_null(win->buffer, "Failed to allocate iobuf");
	memset(win->buffer, 0, win->length);

	rc = cxil_map(lni, win->buffer, win->length,
		      CXI_MAP_PIN | prot, NULL, &win->md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
}

void free_iobuf(struct mem_window *win)
{
	int rc;

	rc = cxil_unmap(win->md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	free(win->buffer);
}

void memcpy_device_to_host(void *dest, struct mem_window *win)
{
	int rc;

	rc = gpu_memcpy(dest, win->buffer, win->length, to_host);
	cr_assert_eq(rc, 0, "gpu_memcpy() failed %d", rc);
}

void memcpy_host_to_device(struct mem_window *win, void *src)
{
	int rc;

	rc = gpu_memcpy(win->buffer, src, win->length, to_device);
	cr_assert_eq(rc, 0, "gpu_memcpy() failed %d", rc);
}

void memset_device(struct mem_window *win, int value, size_t count)
{
	int rc;

	rc = gpu_memset(win->buffer, value, count);
	cr_assert_eq(rc, 0, "gpu_memset() failed %d", rc);
}

static void map_devicebuf(struct mem_window *win, uint32_t prot)
{
	int rc;
	void *base_addr;
	size_t size;

	prot &= (CXI_MAP_WRITE | CXI_MAP_READ);

	memset(&win->md, 0, sizeof(win->md));
	base_addr = win->buffer;
	size = win->length;

	rc = gpu_props(win, &base_addr, &size);
	cr_assert_eq(rc, 0, "gpu_props failed\n");

	if (win->is_device)
		prot |= CXI_MAP_DEVICE;

	rc = cxil_map(lni, base_addr, size,
		      CXI_MAP_PIN | prot,
		      &win->hints, &win->md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
}

void alloc_map_devicebuf(size_t len, struct mem_window *win, uint32_t prot)
{
	int rc;

	win->length = len;
	rc = gpu_malloc(win);
	cr_assert_eq(rc, 0, "gpu_malloc() failed %d", rc);

	map_devicebuf(win, prot);
}

void alloc_map_hostbuf(size_t len, struct mem_window *win, uint32_t prot)
{
	int rc;

	win->length = len;
	rc = gpu_host_alloc(win);
	cr_assert_eq(rc, 0, "gpu_malloc_host() failed %d", rc);

	map_devicebuf(win, prot);
}

void free_unmap_devicebuf(struct mem_window *win)
{
	int rc = cxil_unmap(win->md);

	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	if (win->hints.dmabuf_valid)
		gpu_close_fd(win->hints.dmabuf_fd);

	if (win->loc == on_device)
		gpu_free(win->buffer);
	else
		gpu_host_free(win->buffer);
}

void pte_setup(void)
{
	struct cxi_pt_alloc_opts pte_opts = {};
	int ret;

	domain_setup();

	ret = cxil_alloc_pte(lni, NULL, &pte_opts, &rx_pte);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
}

void pte_teardown(void)
{
	int ret;

	ret = cxil_destroy_pte(rx_pte);
	cr_expect_eq(ret, 0, "%s:cxil_pte_free() returns (%d) %s",
		     __func__, ret, strerror(-ret));
	rx_pte = NULL;

	domain_teardown();
}

void test_data_setup(void)
{
	uint64_t len;
	int i;

	lni_setup();

	test_data = calloc(16, sizeof(*test_data));
	cr_assert_neq(test_data, NULL);

	for (i = 0, len = 16*1024L;
	     i < 16 && len >= 4*1024L;
	     i++, len /= 2) {
		test_data[i].len = len;
		test_data[i].addr = aligned_alloc(s_page_size, len);
		cr_assert_neq(test_data[i].addr, NULL);

		memset(test_data[i].addr, i + 1, len);
	}
	test_data_len = i;

}

void test_data_teardown(void)
{
	int i;

	for (i = 0; i < test_data_len; i++)
		free(test_data[i].addr);
	free(test_data);
	test_data = NULL;
	test_data_len = 0;

	lni_teardown();
}

char *hp_nr_fmt = "/sys/kernel/mm/hugepages/hugepages-%dkB/nr_hugepages";
char *hp_free_fmt = "/sys/kernel/mm/hugepages/hugepages-%dkB/free_hugepages";

int check_huge_pg_free(size_t hp_size, uint32_t npg_needed)
{
	uint32_t total;
	char buf[128];
	int fd, ret = 0;
	char *fmt;

	ret = asprintf(&fmt, hp_free_fmt, hp_size / 1024);
	cr_assert(ret > 0);

	fd = open(fmt, O_RDONLY);
	if (fd < 0)
		return 0;

	ret = read(fd, buf, 8);
	cr_assert_geq(ret, 0, "read failed %d", ret);

	ret = sscanf(buf, "%u", &total);
	cr_assert_eq(ret, 1, "failed to get total. ret %d", ret);
	if (npg_needed)
		cr_assert_geq(total, npg_needed,
			     "Not enough huge pages available %u, %u needed",
			     total,
		     npg_needed);

	cr_log_info("%u %lu MB hugepages free\n", total, hp_size / 1024 / 1024);

	close(fd);

	free(fmt);

	return total;
}

int huge_pg_setup(size_t hp_size, uint32_t npg)
{
	uint32_t total;
	uint32_t orig;
	char buf[128];
	int fd, ret = 0;
	char *fmt;
	int new;
	int free_hps;

	free_hps = check_huge_pg_free(hp_size, 0);
	if (free_hps >= npg)
		return npg;

	ret = asprintf(&fmt, hp_nr_fmt, hp_size / 1024);
	cr_assert(ret > 0);

	fd = open(fmt, O_RDWR);
	if (fd < 0) {
		cr_skip_test("No %ld MB hugepages available to run test\n",
			     hp_size / 1024 / 1024);
		return -1;
	}
	cr_assert_geq(fd, 0, "open failed %d", fd);

	/* Read original value */
	ret = read(fd, buf, 8);
	cr_assert_geq(ret, 0, "read failed %d", ret);

	ret = sscanf(buf, "%u", &orig);
	cr_assert_eq(ret, 1);

	/* Write new value */
	ret = lseek(fd, 0, SEEK_SET);
	cr_assert_geq(ret, 0, "lseek failed %d", ret);

	new = orig + npg - free_hps;

	sprintf(buf, "%u\n", new);
	ret = write(fd, buf, strlen(buf));
	cr_assert_geq(ret, 0, "write failed %d", ret);

	/* Read back new value */
	ret = lseek(fd, 0, SEEK_SET);
	cr_assert_geq(ret, 0, "lseek failed %d", ret);

	ret = read(fd, buf, 8);
	cr_assert_geq(ret, 0, "read failed %d", ret);

	sscanf(buf, "%u", &total);
	if (total < new)
		cr_skip_test("Not enough hugepages available to run test");

	/* it's ok to not set nr_hugepage to 0 */
	if (new)
		cr_assert_leq(new, total,
			     "Failed to set huge pages to %u, %u available",
			     npg, total);

	close(fd);

	free(fmt);

	return orig;
}

bool is_vm(void) {
	int hypervisor_count;
	FILE *fp = popen("lscpu | grep -c Hypervisor", "r");
	if (fp) {
		int rc;

		rc = fscanf(fp, "%d", &hypervisor_count);
		cr_assert_eq(rc, 1, "ret=%d\n", rc);
		pclose(fp);
	}
	return hypervisor_count > 0;
}

static int no_gpu_props(struct mem_window *win, void **base, size_t *size)
{
	win->hints.dmabuf_valid = false;

	return 0;
}

static int no_gpu_close_fd(int fd)
{
	return 0;
}

int gpu_lib_init(void)
{
	int rc = -1;

#ifdef HAVE_HIP_SUPPORT
	rc = hip_lib_init();
	if (!rc)
		return rc;
#endif /* HAVE_HIP_SUPPORT */

#ifdef HAVE_CUDA_SUPPORT
	rc = cuda_lib_init();
	if (!rc)
		return rc;
#endif /* HAVE_CUDA_SUPPORT */

#ifdef HAVE_ZE_SUPPORT
	rc = ze_init();
	if (!rc)
		return rc;
#endif /* HAVE_ZE_SUPPORT */

	printf("No GPU found.\n");

	gpu_malloc = NULL;
	gpu_host_alloc = NULL;
	gpu_free = NULL;
	gpu_host_free = NULL;
	gpu_memset = NULL;
	gpu_memcpy = NULL;
	gpu_props = no_gpu_props;
	gpu_close_fd = no_gpu_close_fd;

	return rc;
}

void gpu_lib_fini(void)
{
#ifdef HAVE_HIP_SUPPORT
	hip_lib_fini();
#endif /* HAVE_HIP_SUPPORT */

#ifdef HAVE_CUDA_SUPPORT
	cuda_lib_fini();
#endif /* HAVE_CUDA_SUPPORT */

#ifdef HAVE_ZE_SUPPORT
	ze_fini();
#endif /* HAVE_CUDA_SUPPORT */
}
