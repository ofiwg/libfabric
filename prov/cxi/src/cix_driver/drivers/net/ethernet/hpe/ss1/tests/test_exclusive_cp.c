// SPDX-License-Identifier: GPL-2.0
/* Copyright 2025 Hewlett Packard Enterprise Development LP */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/hpe/cxi/cxi.h>
#include "cxi_core.h"
#include "cass_core.h"

#define VNI 16U
#define VNI_IGNORE 15U
#define VNI_INVALID 32U
#define TLE_COUNT 10U
#define TIMEOUT 2U

static bool test_pass = true;

#ifdef pr_fmt
#undef pr_fmt
#endif
#define pr_fmt(fmt) KBUILD_MODNAME ":%s:%d " fmt, __func__, __LINE__

static struct cxi_dev *dev;

static unsigned int vni;
static unsigned int pid;

static struct cxi_rgroup *rgroup;
static struct cxi_rx_profile *rx_profile;
static struct cxi_tx_profile *tx_profile;
static struct cxi_lni *lni;
static struct cxi_domain *domain;
static void *eq_buf;
static size_t eq_buf_len;
static struct cxi_md *eq_buf_md;
static u64 eq_flags;
static struct cxi_eq *eq;
static struct cxi_cp *cp;
static struct cxi_cq *cq_transmit;
static struct cxi_cq *cq_target;
static struct cxi_pte *pt;

static struct cxi_pt_alloc_opts pt_alloc_opts;
static struct cxi_cq_alloc_opts cq_alloc_opts;
static unsigned int pid_offset;
static unsigned int pt_index;
static union c_fab_addr dfa;
static u8 index_ext;

struct mem_window {
	size_t length;
	u8 *buffer;
	struct cxi_md *md;
};

static struct cxi_client cxiu_client;

#define LIMIT_LE_MAX 22
#define LIMIT_PTLTE_MAX 22
#define LIMIT_TXQ_MAX 22
#define LIMIT_TGQ_MAX 22
#define LIMIT_EQ_MAX 22
#define LIMIT_CT_MAX 22
#define LIMIT_AC_MAX 22
#define LIMIT_LE_RES 20
#define LIMIT_PTLTE_RES 20
#define LIMIT_TXQ_RES 20
#define LIMIT_TGQ_RES 20
#define LIMIT_EQ_RES 20
#define LIMIT_CT_RES 20
#define LIMIT_TLE_RES 20
#define LIMIT_AC_RES 20

static void dump_resources(struct cxi_rgroup *rgroup)
{
	int i;
	struct cxi_resource_limits limits = {};

	for (i = CXI_RESOURCE_PTLTE; i < CXI_RESOURCE_MAX; i++) {
		cxi_rgroup_get_resource(rgroup, i, &limits);
		pr_info("%s max:%ld res:%ld in_use:%ld\n",
			cxi_resource_type_to_str(i),
			limits.max, limits.reserved, limits.in_use);
	}
}

static int alloc_rgroup_rsrcs(struct cxi_dev *dev)
{
	int i;
	int rc;
	unsigned int ac_entry_id;
	struct cxi_resource_limits limits[CXI_RESOURCE_MAX];
	const union cxi_ac_data ac_data = {
		.uid = __kuid_val(current_euid()),
	};

	limits[CXI_RESOURCE_PTLTE].max = LIMIT_PTLTE_MAX;
	limits[CXI_RESOURCE_PTLTE].reserved = LIMIT_PTLTE_RES;
	limits[CXI_RESOURCE_TXQ].max = LIMIT_TXQ_MAX;
	limits[CXI_RESOURCE_TXQ].reserved = LIMIT_TXQ_RES;
	limits[CXI_RESOURCE_TGQ].max = LIMIT_TGQ_MAX;
	limits[CXI_RESOURCE_TGQ].reserved = LIMIT_TGQ_RES;
	limits[CXI_RESOURCE_EQ].max = LIMIT_EQ_MAX;
	limits[CXI_RESOURCE_EQ].reserved = LIMIT_EQ_RES;
	limits[CXI_RESOURCE_CT].max = LIMIT_CT_MAX;
	limits[CXI_RESOURCE_CT].reserved = LIMIT_CT_RES;
	limits[CXI_RESOURCE_PE0_LE].max = LIMIT_LE_MAX;
	limits[CXI_RESOURCE_PE0_LE].reserved = LIMIT_LE_RES;
	limits[CXI_RESOURCE_PE1_LE].max = LIMIT_LE_MAX;
	limits[CXI_RESOURCE_PE1_LE].reserved = LIMIT_LE_RES;
	limits[CXI_RESOURCE_PE2_LE].max = LIMIT_LE_MAX;
	limits[CXI_RESOURCE_PE2_LE].reserved = LIMIT_LE_RES;
	limits[CXI_RESOURCE_PE3_LE].max = LIMIT_LE_MAX;
	limits[CXI_RESOURCE_PE3_LE].reserved = LIMIT_LE_RES;
	limits[CXI_RESOURCE_TLE].max = LIMIT_TLE_RES;
	limits[CXI_RESOURCE_TLE].reserved = LIMIT_TLE_RES;
	limits[CXI_RESOURCE_AC].max = LIMIT_AC_MAX;
	limits[CXI_RESOURCE_AC].reserved = LIMIT_AC_RES;

	rgroup = cxi_dev_alloc_rgroup(dev, NULL);
	if (IS_ERR(rgroup)) {
		rc = PTR_ERR(rgroup);
		pr_err("Failed to allocate rgroup:%d\n", rc);
		return rc;
	}

	cxi_rgroup_set_system_service(rgroup, true);
	cxi_rgroup_set_name(rgroup, "test-exclusive-cp-rgroup");

	for (i = CXI_RESOURCE_PTLTE; i < CXI_RESOURCE_MAX; i++) {
		if (!limits[i].reserved && !limits[i].max)
			continue;

		rc = cxi_rgroup_add_resource(rgroup, i, &limits[i]);
		if (rc) {
			pr_err("Add %s resource failed:%d\n",
			       cxi_resource_type_to_str(i), rc);
			dump_resources(rgroup);
			goto err;
		}
	}

	rc = cxi_rgroup_add_ac_entry(rgroup, CXI_AC_UID, &ac_data,
				     &ac_entry_id);
	if (rc)
		goto err;

	cxi_rgroup_enable(rgroup);

	pr_info("Allocated %s id:%d\n", cxi_rgroup_name(rgroup),
		cxi_rgroup_id(rgroup));

	return 0;

err:
	cxi_rgroup_dec_refcount(rgroup);

	return rc;
}

static void free_profiles(void)
{
	cxi_tx_profile_dec_refcount(dev, tx_profile, true);
	cxi_rx_profile_dec_refcount(dev, rx_profile);
}

static int alloc_profiles(void)
{
	int rc;
	int ac_entry_id;
	struct cxi_rx_attr rx_attr = {
		.vni_attr.match = VNI,
		.vni_attr.ignore = VNI_IGNORE,
	};
	struct cxi_tx_attr tx_attr = {
		.vni_attr.match = VNI,
		.vni_attr.ignore = VNI_IGNORE,
	};

	rx_profile = cxi_dev_alloc_rx_profile(dev, &rx_attr);
	if (IS_ERR(rx_profile)) {
		rc = PTR_ERR(rx_profile);
		pr_info("Allocate RX profile failed vni:%d rc:%d", VNI, rc);
		return rc;
	}

	rc = cxi_rx_profile_add_ac_entry(rx_profile, CXI_AC_UID,
					 __kuid_val(current_euid()), 0,
					 &ac_entry_id);
	if (rc)
		goto remove_rx_profile;

	rc = cxi_rx_profile_enable(dev, rx_profile);
	if (rc)
		goto remove_rx_profile;

	tx_profile = cxi_dev_alloc_tx_profile(dev, &tx_attr);
	if (IS_ERR(tx_profile)) {
		rc = PTR_ERR(tx_profile);
		pr_info("Allocate TX profile failed vni:%d rc:%d", VNI, rc);
		goto remove_rx_profile;
	}

	rc = cxi_tx_profile_add_ac_entry(tx_profile, CXI_AC_UID,
					 __kuid_val(current_euid()), 0,
					 &ac_entry_id);
	if (rc)
		goto remove_tx_profile;

	cxi_tx_profile_set_tc(tx_profile, CXI_TC_BEST_EFFORT, true);

	rc = cxi_tx_profile_set_exclusive_cp(tx_profile, true);
	if (rc)
		goto remove_tx_profile;

	rc = cxi_tx_profile_enable(dev, tx_profile);
	if (rc)
		goto remove_tx_profile;

	return 0;

remove_tx_profile:
	cxi_tx_profile_dec_refcount(dev, tx_profile, true);
remove_rx_profile:
	cxi_rx_profile_dec_refcount(dev, rx_profile);

	return rc;
}

static int test_map(struct mem_window *mem, int lac)
{
	mem->md = cxi_map(lni, (uintptr_t)mem->buffer, mem->length,
			  CXI_MAP_WRITE | CXI_MAP_READ, NULL);

	return PTR_ERR_OR_ZERO(mem->md);
}

static int test_unmap(struct mem_window *mem)
{
	return cxi_unmap(mem->md);
}

static atomic_t eq_cb_called = ATOMIC_INIT(0);

static void eq_cb(void *context)
{
	atomic_inc(&eq_cb_called);
}

static int set_up_eq(void)
{
	int rc;
	struct cxi_eq_attr eq_attr = {};

	eq_buf_len = PAGE_SIZE * 4;
	eq_buf = kzalloc(eq_buf_len, GFP_KERNEL);
	if (!eq_buf)
		return -ENOMEM;

	eq_buf_md = cxi_map(lni, (uintptr_t)eq_buf,
			    eq_buf_len,
			    CXI_MAP_WRITE | CXI_MAP_READ,
			    NULL);
	if (IS_ERR(eq_buf_md)) {
		rc = PTR_ERR(eq_buf_md);
		goto free_eq_buf;
	}

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = eq_buf_len;
	eq_attr.flags = eq_flags;

	eq = cxi_eq_alloc(lni, eq_buf_md, &eq_attr,
			  eq_cb, NULL, NULL, NULL);
	if (IS_ERR(eq)) {
		rc = PTR_ERR(eq);
		goto unmap_eq_buf;
	}

	return 0;

unmap_eq_buf:
	cxi_unmap(eq_buf_md);
free_eq_buf:
	kfree(eq_buf);

	return rc;
}

static void tear_down_eq(void)
{
	cxi_eq_free(eq);
	cxi_unmap(eq_buf_md);
	kfree(eq_buf);
}

static int test_setup(struct cxi_dev *dev)
{
	int rc;
	int retry;

	rc = alloc_rgroup_rsrcs(dev);
	if (rc) {
		pr_err("Can't allocate rgroup:%d\n", rc);
		return rc;
	}

	rc = alloc_profiles();
	if (rc)
		goto out;

	lni = cxi_lni_alloc(dev, cxi_rgroup_id(rgroup));
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		goto free_the_profiles;
	}

	/* Choose a random pid */
	vni = VNI;
	pid = 30;
	pid_offset = 3;
	domain = cxi_domain_alloc(lni, vni, pid);
	if (IS_ERR(domain)) {
		rc = PTR_ERR(domain);
		goto free_ni;
	}

	cp = cxi_cp_alloc(lni, vni, CXI_TC_BEST_EFFORT,
			  CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(cp)) {
		rc = PTR_ERR(cp);
		goto free_dom;
	}

	rc = set_up_eq();
	if (rc)
		goto free_cp;

	cq_alloc_opts.count = 50;
	cq_alloc_opts.flags = CXI_CQ_IS_TX;
	cq_alloc_opts.lcid = cp->lcid;
	cq_transmit = cxi_cq_alloc(lni, NULL, &cq_alloc_opts,
				   0);
	if (IS_ERR(cq_transmit)) {
		rc = PTR_ERR(cq_transmit);
		goto free_eq;
	}

	memset(&cq_alloc_opts, 0, sizeof(cq_alloc_opts));
	cq_alloc_opts.count = 50;
	cq_target = cxi_cq_alloc(lni, NULL, &cq_alloc_opts,
				 0);
	if (IS_ERR(cq_target)) {
		rc = PTR_ERR(cq_target);
		goto free_cq_transmit;
	}

	pt_alloc_opts = (struct cxi_pt_alloc_opts) {
		.is_matching = 0,
	};

	pt = cxi_pte_alloc(lni, eq, &pt_alloc_opts);
	if (IS_ERR(pt)) {
		rc = PTR_ERR(pt);
		goto free_cq_target;
	}

	rc = cxi_pte_map(pt, domain, pid_offset,
			 false, &pt_index);
	if (rc)
		goto free_pt;

	WARN_ON(cq_transmit->status->rd_ptr != 0 + C_CQ_FIRST_WR_PTR);

	/* Setup DFA. */
	cxi_build_dfa(dev->prop.nid, pid,
		      dev->prop.pid_bits, pid_offset,
		      &dfa, &index_ext);

	/* Set LCID to index 0 */
	rc = cxi_cq_emit_cq_lcid(cq_transmit, 0);
	if (rc)
		pr_err("Emit LCID failed: %d\n", rc);

	cxi_cq_ring(cq_transmit);

	/* LCID is 64 bytes plus 64 bytes NOP */
	retry = 100;
	while (retry-- &&
	       cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR)
		mdelay(10);
	WARN_ON(cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	return 0;

free_pt:
	cxi_pte_free(pt);
free_cq_target:
	cxi_cq_free(cq_target);
free_cq_transmit:
	cxi_cq_free(cq_transmit);
free_eq:
	tear_down_eq();
free_cp:
	cxi_cp_free(cp);
free_dom:
	cxi_domain_free(domain);
free_ni:
	cxi_lni_free(lni);
free_the_profiles:
	free_profiles();
out:
	cxi_rgroup_dec_refcount(rgroup);
	return rc;
}

static void test_teardown(void)
{
	cxi_pte_unmap(pt, domain, pt_index);
	cxi_pte_free(pt);
	cxi_cq_free(cq_target);
	cxi_cq_free(cq_transmit);
	tear_down_eq();
	cxi_cp_free(cp);
	cxi_domain_free(domain);
	cxi_lni_free(lni);
	free_profiles();
	cxi_rgroup_dec_refcount(rgroup);
}

static void wait_for_event(enum c_event_type type)
{
	int retry;
	const union c_event *event;

	for (retry = 100; retry > 0; retry--) {
		event = cxi_eq_get_event(eq);
		if (event && event->hdr.event_type == type)
			break;

		mdelay(10);
	}

	if (retry > 0)
		pr_debug("received event %d\n", type);
}

/* Do a DMA Put transaction. */
static int test_do_put(struct mem_window *mem_win,
		       size_t len, u64 r_off, u64 l_off, u32 index_ext)
{
	int rc;
	union c_cmdu cmd = {};

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_PUT;
	cmd.full_dma.index_ext = index_ext;
	cmd.full_dma.lac = mem_win->md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.remote_offset = r_off;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(mem_win->md,
						 mem_win->buffer + l_off);
	cmd.full_dma.eq = eq->eqn;
	cmd.full_dma.user_ptr = 0;
	cmd.full_dma.request_len = len;
	cmd.full_dma.restricted = 1;
	cmd.full_dma.match_bits = 0;
	cmd.full_dma.initiator = 0;

	rc = cxi_cq_emit_dma(cq_transmit, &cmd.full_dma);
	if (rc)
		return rc;

	cxi_cq_ring(cq_transmit);

	wait_for_event(C_EVENT_ACK);

	return 0;
}

/* Append a list entry to the priority list for the non-matching Portals table.
 * Enable the non-matching Portals table.
 */
static void test_append_le(u64 len, struct cxi_md *md, size_t offset)
{
	union c_cmdu cq_cmd;

	/* Append a list entry to the priority list for the
	 * non-matching Portals table.
	 */
	memset(&cq_cmd, 0, sizeof(struct c_target_cmd));
	cq_cmd.command.opcode = C_CMD_TGT_APPEND;
	cq_cmd.target.ptl_list = C_PTL_LIST_PRIORITY;
	cq_cmd.target.ptlte_index = pt->id;
	cq_cmd.target.op_put = 1;
	cq_cmd.target.op_get = 1;
	cq_cmd.target.event_ct_comm = 1;
	cq_cmd.target.event_ct_overflow = 1;
	cq_cmd.target.start = md->iova + offset;
	cq_cmd.target.length = len;
	cq_cmd.target.lac = md->lac;

	cxi_cq_emit_target(cq_target, &cq_cmd);

	/* Enable the non-matching Portals table. */
	memset(&cq_cmd, 0, sizeof(struct c_target_cmd));
	cq_cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cq_cmd.set_state.ptlte_index = pt->id;
	cq_cmd.set_state.current_addr = md->iova;
	cq_cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	cxi_cq_emit_target(cq_target, &cq_cmd);

	WARN_ON(cq_target->status->rd_ptr != 0 + C_CQ_FIRST_WR_PTR);

	cxi_cq_ring(cq_target);

	wait_for_event(C_EVENT_LINK);
}

static int test_rma(struct cxi_dev *dev)
{
	int i;
	int rc;
	int ret;
	int lac = 0;
	int errors = 0;
	struct cxi_md snd_md = {};
	struct cxi_md rma_md = {};
	struct mem_window snd_mem;
	struct mem_window rma_mem;
	size_t len = 256 * 1024;

	rma_mem.length = len;
	snd_mem.length = len;
	rma_mem.md = &rma_md;
	snd_mem.md = &snd_md;

	rc = test_setup(dev);
	if (rc)
		return rc;

	/* Map a buffer */
	rma_mem.buffer = kzalloc(rma_mem.length, GFP_KERNEL);
	if (!rma_mem.buffer) {
		rc = -ENOMEM;
		goto free_lac;
	}

	snd_mem.buffer = kzalloc(snd_mem.length, GFP_KERNEL);
	if (!snd_mem.buffer) {
		rc = -ENOMEM;
		goto free_rma;
	}

	rc = test_map(&snd_mem, lac);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto free_snd;
	}

	rc = test_map(&rma_mem, lac);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto unmap_snd;
	}

	test_append_le(len, rma_mem.md, 0);

	WARN_ON(cq_target->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	for (i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

	memset(rma_mem.buffer, 0, len);

	rc = test_do_put(&snd_mem, len, 0, 0, index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto unmap_rma;
	}

	for (i = 0; i < len; i++) {
		if (snd_mem.buffer[i] != rma_mem.buffer[i]) {
			pr_info("Data mismatch: idx %2d - %02x != %02x\n",
				i, snd_mem.buffer[i], rma_mem.buffer[i]);
			errors++;
		}

		if (errors > 10)
			break;
	}

	if (errors) {
		pr_info("%s had errors before cp modify\n", __func__);
		rc = -EFAULT;
		goto unmap_rma;
	}

	rc = cxi_cp_modify(cp, VNI_INVALID);
	if (!rc) {
		pr_err("cxi_cp_modify was succeful with invalid vni. rc:%d\n", rc);
		rc = -EINVAL;
		goto unmap_rma;
	}

	rc = cxi_cp_modify(cp, VNI + VNI_IGNORE);
	if (rc) {
		pr_err("cxi_cp_modify failed. rc:%d\n", rc);
		goto unmap_rma;
	}

	memset(rma_mem.buffer, 0, len);

	rc = test_do_put(&snd_mem, len, 0, 0, index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto unmap_rma;
	}

	for (i = 0; i < len; i++) {
		if (snd_mem.buffer[i] != rma_mem.buffer[i]) {
			pr_info("Data mismatch: idx %2d - %02x != %02x\n",
				i, snd_mem.buffer[i], rma_mem.buffer[i]);
			errors++;
		}

		if (errors > 10)
			break;
	}

	if (errors) {
		pr_info("%s had errors after cp modify\n", __func__);
		rc = -EFAULT;
	}

unmap_rma:
	ret = test_unmap(&rma_mem);
	WARN(ret < 0, "cxi_unmap failed %d\n", ret);
unmap_snd:
	ret = test_unmap(&snd_mem);
	WARN(ret < 0, "cxi_unmap failed %d\n", ret);
free_snd:
	kfree(snd_mem.buffer);
free_rma:
	kfree(rma_mem.buffer);
free_lac:
	test_teardown();

	return rc;
}

static int run_tests(struct cxi_dev *dev)
{
	int rc;

	rc = test_rma(dev);
	if (rc) {
		test_pass = false;
		pr_err("test_rma failed: %d\n", rc);
		goto done;
	}

	pr_info("All tests pass\n");

done:
	return rc;
}

static int add_device(struct cxi_dev *cdev)
{
	dev = cdev;
	return 0;
}

static void remove_device(struct cxi_dev *cdev)
{
	dev = NULL;
}

static struct cxi_client cxiu_client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int rc;

	rc = cxi_register_client(&cxiu_client);
	if (rc) {
		pr_err("cxi_register_client failed: %d\n", rc);
		return rc;
	}

	rc = run_tests(dev);
	if (rc)
		pr_err("Failed:%d\n", rc);

	if (!test_pass)
		rc = -EIO;

	pr_info("unregistering client\n");
	cxi_unregister_client(&cxiu_client);

	return rc;
}

static void __exit cleanup(void)
{
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("CXI exclusive cp API test suite");
MODULE_AUTHOR("HPE");
