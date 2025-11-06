// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Cray Inc. All rights reserved */

/* Template client for CXI, validator for the CXI API. */

#include <linux/bvec.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/vmalloc.h>
#include <uapi/ethernet/cxi-abi.h>

#include "cxi_prov_hw.h"
#include "cxi_core.h"

#if KERNEL_VERSION(5, 3, 18) > LINUX_VERSION_CODE
#define MY_ITER_BVEC ITER_BVEC
#define MY_ITER_KVEC ITER_KVEC
#else
#define MY_ITER_BVEC (READ | WRITE)
#define MY_ITER_KVEC (READ | WRITE)
#endif

/* List of devices registered with this client */
static LIST_HEAD(dev_list);
static DEFINE_MUTEX(dev_list_mutex);

/* Keep track of known devices. Protected by dev_list_mutex. */
struct tdev {
	struct list_head dev_list;
	struct cxi_dev *dev;

	unsigned int vni;
	unsigned int pid;

	struct cxi_lni *lni;
	struct cxi_domain *domain;
	void *eq_buf;
	size_t eq_buf_len;
	struct cxi_md *eq_buf_md;
	u64 eq_flags;
	struct cxi_eq *eq;
	struct cxi_cp *cp;
	struct cxi_cq *cq_transmit;
	struct cxi_cq *cq_target;
	struct cxi_pte *pt;

	struct cxi_pt_alloc_opts pt_alloc_opts;
	struct cxi_cq_alloc_opts cq_alloc_opts;
	unsigned int pid_offset;
	unsigned int pt_index;

	/* Loopback address. */
	union c_fab_addr dfa;
	u8 index_ext;
};

struct mem_window {
	size_t length;
	u8 *buffer;
	struct cxi_md *md;
};

struct bvec_info {
	struct cxi_md *md;
	struct iov_iter iter;
};

#define MAP_NTA_RW (CXI_MAP_WRITE | CXI_MAP_READ)

static void wait_for_event(struct tdev *tdev, enum c_event_type type)
{
	int retry;
	const union c_event *event;

	for (retry = 100; retry > 0; retry--) {
		event = cxi_eq_get_event(tdev->eq);
		if (event && event->hdr.event_type == type)
			break;

		mdelay(10);
	}

	if (retry > 0)
		pr_debug("received event %d\n", type);

}

/*
 * Append a list entry to the priority list for the non-matching Portals table.
 * Enable the non-matching Portals table.
 */
static void test_append_le(struct tdev *tdev, u64 len, struct cxi_md *md,
			   size_t offset)
{
	union c_cmdu cq_cmd;

	/* Append a list entry to the priority list for the
	 * non-matching Portals table.
	 */
	memset(&cq_cmd, 0, sizeof(struct c_target_cmd));
	cq_cmd.command.opcode = C_CMD_TGT_APPEND;
	cq_cmd.target.ptl_list = C_PTL_LIST_PRIORITY;
	cq_cmd.target.ptlte_index = tdev->pt->id;
	cq_cmd.target.op_put = 1;
	cq_cmd.target.op_get = 1;
	cq_cmd.target.event_ct_comm = 1;
	cq_cmd.target.event_ct_overflow = 1;
	cq_cmd.target.start = md->iova + offset;
	cq_cmd.target.length = len;
	cq_cmd.target.lac = md->lac;

	cxi_cq_emit_target(tdev->cq_target, &cq_cmd);

	/* Enable the non-matching Portals table. */
	memset(&cq_cmd, 0, sizeof(struct c_target_cmd));
	cq_cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cq_cmd.set_state.ptlte_index = tdev->pt->id;
	cq_cmd.set_state.current_addr = md->iova;
	cq_cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	cxi_cq_emit_target(tdev->cq_target, &cq_cmd);

	WARN_ON(tdev->cq_target->status->rd_ptr != 0 + C_CQ_FIRST_WR_PTR);

	cxi_cq_ring(tdev->cq_target);

	wait_for_event(tdev, C_EVENT_LINK);
}

/* Add C_STATE command */
static void test_add_cstate(struct tdev *tdev)
{
	struct c_cstate_cmd c_state = {};

	c_state.eq   = tdev->eq->eqn;
	c_state.restricted = 1;
	c_state.index_ext = tdev->index_ext;

	cxi_cq_emit_c_state(tdev->cq_transmit, &c_state);

	WARN_ON(tdev->cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	cxi_cq_ring(tdev->cq_transmit);
}

static int test_idc_put(struct tdev *tdev, u64 *tgt_data)
{
	struct c_idc_put_cmd idc_put = {};
	u64 data[2];
	int ret = 0;

	idc_put.idc_header.dfa = tdev->dfa;

	/* Write a value to address 0 of ioData. */
	idc_put.idc_header.command.opcode = C_CMD_NOMATCH_PUT;
	data[0] = 1;
	cxi_cq_emit_idc_put(tdev->cq_transmit, &idc_put, data, 8);

	/* Write a value to address 8 of ioData. */
	idc_put.idc_header.remote_offset = 8;
	data[0] = 2;
	cxi_cq_emit_idc_put(tdev->cq_transmit, &idc_put, data, 8);

	/* Write a value to an address that crosses a page. */
	idc_put.idc_header.remote_offset = 0xff8;
	data[0] = 3;
	data[1] = 4;
	cxi_cq_emit_idc_put(tdev->cq_transmit, &idc_put, data, 16);

	cxi_cq_ring(tdev->cq_transmit);
	wait_for_event(tdev, C_EVENT_ACK);

	if ((tgt_data[0] != 1) ||
		(tgt_data[1] != 2) ||
		(tgt_data[511] != 3) ||
		(tgt_data[512] != 4)) {
		pr_err("Data not received\n");
		return -1;
	}

	pr_info("Data is %llx %llx %llx %llx\n",
		tgt_data[0], tgt_data[1], tgt_data[511], tgt_data[512]);

	pr_info("Test passed\n");

	return ret;
}

/* Do a DMA Put transaction. */
static int test_do_put(struct tdev *tdev, struct mem_window *mem_win,
		       size_t len, u64 r_off, u64 l_off, u32 index_ext)
{
	int rc;
	union c_cmdu cmd = {};

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_PUT;
	cmd.full_dma.index_ext = index_ext;
	cmd.full_dma.lac = mem_win->md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.dfa = tdev->dfa;
	cmd.full_dma.remote_offset = r_off;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(mem_win->md,
						 mem_win->buffer + l_off);
	cmd.full_dma.eq = tdev->eq->eqn;
	cmd.full_dma.user_ptr = 0;
	cmd.full_dma.request_len = len;
	cmd.full_dma.restricted = 1;
	cmd.full_dma.match_bits = 0;
	cmd.full_dma.initiator = 0;

	rc = cxi_cq_emit_dma(tdev->cq_transmit, &cmd.full_dma);
	if (rc)
		return rc;

	cxi_cq_ring(tdev->cq_transmit);

	wait_for_event(tdev, C_EVENT_ACK);

	return 0;
}

static int alloc_bvec_pages(int n, struct iov_iter *iter, struct bio_vec *bvec)
{
	int i;
	int rc = 0;
	size_t bvecs_to_free = 0;

	for (i = 0; i < n; i++) {
		bvec[i].bv_page = alloc_page(GFP_KERNEL);
		if (!bvec[i].bv_page) {
			rc = -ENOMEM;
			pr_err("page alloc failed\n");
			goto free_bvecs;
		}

		bvec[i].bv_len = PAGE_SIZE;
		bvec[i].bv_offset = 0;
		bvecs_to_free++;
	}

	iov_iter_bvec(iter, MY_ITER_BVEC, bvec, n, n * PAGE_SIZE);

	return 0;

free_bvecs:
	for (i = 0; i < bvecs_to_free; i++)
		__free_page(bvec[i].bv_page);

	return rc;
}

static int alloc_bvec(struct tdev *tdev, int n, struct iov_iter *iter,
		      struct cxi_md **md)
{
	int i;
	int rc;
	struct bio_vec *bvec;

	bvec = kmalloc_array(n, sizeof(*bvec), GFP_KERNEL);
	if (!bvec)
		return -ENOMEM;

	rc = alloc_bvec_pages(n, iter, bvec);
	if (rc)
		goto free_bvec;

	iov_iter_bvec(iter, MY_ITER_BVEC, bvec, n, n * PAGE_SIZE);

	*md = cxi_map_iov(tdev->lni, iter, MAP_NTA_RW);
	if (IS_ERR(*md)) {
		rc = PTR_ERR(*md);
		pr_err("map iov failed %d\n", rc);
		goto free_bvecs;
	}

	return 0;

free_bvecs:
	for (i = 0; i < n; i++)
		__free_page(bvec[i].bv_page);
free_bvec:
	kfree(bvec);

	return rc;
}

static void free_bvec(struct tdev *tdev, struct iov_iter *iter)
{
	int i;

	for (i = 0; i < iter->nr_segs; i++)
		__free_page(iter->bvec[i].bv_page);

	kfree(iter->bvec);
	iter->bvec = NULL;
}

static int test_map(struct tdev *tdev, struct mem_window *mem, bool phys,
		    int lac)
{
	if (phys) {
		struct device *device = &tdev->dev->pdev->dev;

		mem->md->va = (__u64)mem->buffer;
		mem->md->len = mem->length;
		mem->md->lac = lac;
		mem->md->iova = dma_map_single(device, mem->buffer, mem->length,
					       DMA_FROM_DEVICE);
		if (dma_mapping_error(device, mem->md->iova))
			return -ENOMEM;

		return 0;
	}

	mem->md = cxi_map(tdev->lni, (uintptr_t)mem->buffer, mem->length,
			  CXI_MAP_WRITE | CXI_MAP_READ, NULL);
	return PTR_ERR_OR_ZERO(mem->md);
}

static int test_unmap(struct tdev *tdev, struct mem_window *mem, bool phys)
{
	if (phys) {
		struct device *device = &tdev->dev->pdev->dev;

		dma_unmap_single(device, mem->md->iova, mem->md->len, DMA_FROM_DEVICE);

		return 0;
	}

	return cxi_unmap(mem->md);
}

static atomic_t eq_cb_called = ATOMIC_INIT(0);

static void eq_cb(void *context)
{
	pr_info("Got an EQ interrupt\n");
	atomic_inc(&eq_cb_called);
}

static int test_setup(struct tdev *tdev, int svc_id)
{
	int rc;
	int retry;
	struct cxi_eq_attr eq_attr = {};

	tdev->lni = cxi_lni_alloc(tdev->dev, svc_id);
	if (IS_ERR(tdev->lni)) {
		rc = PTR_ERR(tdev->lni);
		goto out;
	}

	/* Choose a random pid */
	tdev->vni = 1;
	tdev->pid = 30;
	tdev->pid_offset = 3;
	tdev->domain = cxi_domain_alloc(tdev->lni, tdev->vni, tdev->pid);
	if (IS_ERR(tdev->domain)) {
		rc = PTR_ERR(tdev->domain);
		goto free_ni;
	}

	tdev->cp = cxi_cp_alloc(tdev->lni, tdev->vni, CXI_TC_BEST_EFFORT,
				CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(tdev->cp)) {
		rc = PTR_ERR(tdev->cp);
		goto free_dom;
	}

	tdev->eq_buf_len = PAGE_SIZE * 4;
	tdev->eq_buf = kzalloc(tdev->eq_buf_len, GFP_KERNEL);
	if (!tdev->eq_buf) {
		rc = -ENOMEM;
		goto free_cp;
	}

	if (!(tdev->eq_flags & CXI_EQ_PASSTHROUGH)) {
		tdev->eq_buf_md = cxi_map(tdev->lni, (uintptr_t)tdev->eq_buf,
					 tdev->eq_buf_len,
					 CXI_MAP_WRITE | CXI_MAP_READ,
					 NULL);
		if (IS_ERR(tdev->eq_buf_md)) {
			rc = PTR_ERR(tdev->eq_buf_md);
			goto free_eq_buf;
		}
	}

	eq_attr.queue = tdev->eq_buf;
	eq_attr.queue_len = tdev->eq_buf_len;
	eq_attr.flags = tdev->eq_flags;

	tdev->eq = cxi_eq_alloc(tdev->lni, tdev->eq_buf_md, &eq_attr,
				eq_cb, NULL, NULL, NULL);
	if (IS_ERR(tdev->eq)) {
		rc = PTR_ERR(tdev->eq);
		goto unmap_eq_buf;
	}

	tdev->cq_alloc_opts.count = 50;
	tdev->cq_alloc_opts.flags = CXI_CQ_IS_TX;
	tdev->cq_alloc_opts.lcid = tdev->cp->lcid;
	tdev->cq_transmit = cxi_cq_alloc(tdev->lni, NULL, &tdev->cq_alloc_opts,
					 0);
	if (IS_ERR(tdev->cq_transmit)) {
		rc = PTR_ERR(tdev->cq_transmit);
		goto free_eq;
	}

	memset(&tdev->cq_alloc_opts, 0, sizeof(tdev->cq_alloc_opts));
	tdev->cq_alloc_opts.count = 50;
	tdev->cq_target = cxi_cq_alloc(tdev->lni, NULL, &tdev->cq_alloc_opts,
				       0);
	if (IS_ERR(tdev->cq_target)) {
		rc = PTR_ERR(tdev->cq_target);
		goto free_cq_transmit;
	}

	tdev->pt_alloc_opts = (struct cxi_pt_alloc_opts) {
		.is_matching = 0,
	};

	tdev->pt = cxi_pte_alloc(tdev->lni, tdev->eq, &tdev->pt_alloc_opts);
	if (IS_ERR(tdev->pt)) {
		rc = PTR_ERR(tdev->pt);
		goto free_cq_target;
	}

	rc = cxi_pte_map(tdev->pt, tdev->domain, tdev->pid_offset,
			 false, &tdev->pt_index);
	if (rc)
		goto free_pt;

	WARN_ON(tdev->cq_transmit->status->rd_ptr != 0 + C_CQ_FIRST_WR_PTR);

	/* Setup DFA. */
	cxi_build_dfa(tdev->dev->prop.nid, tdev->pid,
		      tdev->dev->prop.pid_bits, tdev->pid_offset,
		      &tdev->dfa, &tdev->index_ext);

	/* Set LCID to index 0 */
	rc = cxi_cq_emit_cq_lcid(tdev->cq_transmit, 0);
	if (rc)
		pr_err("Emit LCID failed: %d\n", rc);

	cxi_cq_ring(tdev->cq_transmit);

	/* LCID is 64 bytes plus 64 bytes NOP */
	retry = 100;
	while (retry-- &&
	       tdev->cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR)
		mdelay(10);
	WARN_ON(tdev->cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	return 0;

free_pt:
	cxi_pte_free(tdev->pt);
free_cq_target:
	cxi_cq_free(tdev->cq_target);
free_cq_transmit:
	cxi_cq_free(tdev->cq_transmit);
free_eq:
	cxi_eq_free(tdev->eq);
unmap_eq_buf:
	if (!(tdev->eq_flags & CXI_EQ_PASSTHROUGH))
		cxi_unmap(tdev->eq_buf_md);
free_eq_buf:
	kfree(tdev->eq_buf);
free_cp:
	cxi_cp_free(tdev->cp);
free_dom:
	cxi_domain_free(tdev->domain);
free_ni:
	cxi_lni_free(tdev->lni);
out:
	return rc;
}

static void test_teardown(struct tdev *tdev)
{
	cxi_pte_unmap(tdev->pt, tdev->domain, tdev->pt_index);
	cxi_pte_free(tdev->pt);
	cxi_cq_free(tdev->cq_target);
	cxi_cq_free(tdev->cq_transmit);
	cxi_eq_free(tdev->eq);
	if (!(tdev->eq_flags & CXI_EQ_PASSTHROUGH))
		cxi_unmap(tdev->eq_buf_md);
	kfree(tdev->eq_buf);
	cxi_cp_free(tdev->cp);
	cxi_domain_free(tdev->domain);
	cxi_lni_free(tdev->lni);
}

/* write either zeros or incrementing pattern to bvec array */
static void bvec_iter_init(struct iov_iter *iter, bool zero)
{
	int i;
	int j;

	for (j = 0, i = 0; i < iter->nr_segs; i++) {
		u32 *p = page_address(iter->bvec[i].bv_page);

		for (; j < (i + 1) * (PAGE_SIZE / sizeof(u32)); j++)
			*p++ = zero ? 0 : j;
	}
}

#define N_VECS 5
#define RES_VECS 3
#define VEC_OFFSET 256

/* map an invalid iter */
static int map_kvec_err(struct tdev *tdev)
{
	int i;
	int rc;
	void *addr;
	struct cxi_lni *lni;
	struct cxi_md *md;
	struct kvec kvec[RES_VECS];
	struct iov_iter iter = {};

	pr_info("%s\n", __func__);

	lni = cxi_lni_alloc(tdev->dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		goto out;
	}

	for (i = 0; i < RES_VECS; i++) {
		kvec[i].iov_base = &addr;
		kvec[i].iov_len = 64;
	}

	iov_iter_kvec(&iter, MY_ITER_KVEC, kvec, RES_VECS, 64 * RES_VECS);
	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (!IS_ERR(md)) {
		pr_err("map iov should have failed\n");
		rc = -1;
		goto free_lni;
	}

	kvec[1].iov_base = NULL;
	kvec[2].iov_base = NULL;
	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (!IS_ERR(md)) {
		pr_err("map iov should have failed\n");
		rc = -1;
		goto free_lni;
	}

	rc = 0;
free_lni:
	cxi_lni_free(lni);
out:
	return rc;
}

static int map_bvec_err(struct tdev *tdev)
{
	int i;
	int rc = 0;
	struct cxi_lni *lni;
	struct cxi_md *md;
	struct bio_vec bvec[RES_VECS];
	struct iov_iter iter = {};

	pr_info("%s\n", __func__);

	lni = cxi_lni_alloc(tdev->dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		goto out;
	}

	for (i = 0; i < RES_VECS; i++) {
		bvec[i].bv_page = alloc_page(GFP_KERNEL);
		bvec[i].bv_len = PAGE_SIZE;
		bvec[i].bv_offset = 0;
	}

	bvec[RES_VECS - 1].bv_len -= VEC_OFFSET;
	bvec[0].bv_len -= VEC_OFFSET;

	iov_iter_bvec(&iter, MY_ITER_BVEC, bvec, RES_VECS, 64 * RES_VECS);
	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (!IS_ERR(md)) {
		pr_err("map iov should have failed\n");
		rc = -1;
		goto free_lni;
	}

	bvec[0].bv_offset += VEC_OFFSET;
	bvec[1].bv_len -= VEC_OFFSET;

	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (!IS_ERR(md)) {
		pr_err("map iov should have failed\n");
		rc = -1;
		goto free_bvecs;
	}

	bvec[1].bv_offset -= VEC_OFFSET;
	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (!IS_ERR(md)) {
		pr_err("map iov should have failed\n");
		rc = -1;
		goto free_bvecs;
	}

	bvec[1].bv_offset = 0;
	bvec[1].bv_len = PAGE_SIZE;
	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (IS_ERR(md)) {
		rc = PTR_ERR(md);
		pr_err("map iov failed %d\n", rc);
		goto free_bvecs;
	}

	rc = cxi_unmap(md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);

free_bvecs:
	for (i = 0; i < RES_VECS; i++)
		__free_page(bvec[i].bv_page);
free_lni:
	cxi_lni_free(lni);
out:
	return rc;
}

static int map_kvec(struct tdev *tdev)
{
	int i;
	int rc;
	size_t count = 0;
	struct cxi_lni *lni;
	size_t iov_len = PAGE_SIZE - VEC_OFFSET;
	int kvecs_to_free = 0;
	struct cxi_md *md;
	struct iov_iter iter = {};
	struct kvec *kvec;

	pr_info("%s\n", __func__);

	lni = cxi_lni_alloc(tdev->dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		goto out;
	}

	kvec = kmalloc_array(N_VECS, sizeof(*kvec), GFP_KERNEL);
	if (!kvec) {
		rc = -ENOMEM;
		goto free_lni;
	}

	memset(kvec, 0, N_VECS * sizeof(*kvec));

	/* the first and last pages are less than a page */
	for (i = 0; i < N_VECS; i++) {
		if (i == (N_VECS - 1))
			iov_len = PAGE_SIZE - VEC_OFFSET;
		else if (i)
			iov_len = i * PAGE_SIZE;

		kvec[i].iov_base = kmalloc(iov_len, GFP_KERNEL);
		if (!kvec[i].iov_base) {
			rc = -ENOMEM;
			pr_err("alloc failed\n");
			goto free_kvecs;
		}

		pr_info("iov_base %p iov_len %ld len %ld\n", kvec[i].iov_base,
			iov_len, ksize(kvec[i].iov_base));
		kvec[i].iov_len = iov_len;
		count += iov_len;
		kvecs_to_free++;
	}

	pr_info("allocated %d kvecs %ld bytes\n", kvecs_to_free, count);

	iov_iter_kvec(&iter, MY_ITER_KVEC, kvec, N_VECS, count);

	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (!IS_ERR(md)) {
		rc = -ENOMEM;
		pr_err("map iov should have failed\n");
		goto free_kvecs;
	}

	kvec[0].iov_base += VEC_OFFSET;

	md = cxi_map_iov(lni, &iter, MAP_NTA_RW);
	if (IS_ERR(md)) {
		rc = PTR_ERR(md);
		pr_err("map iov failed %d\n", rc);
		goto fix_iov_base;
	}

	pr_info("map iov iova:0x%llx len:%ld\n", md->iova, md->len);

	pr_info("cxi_unmap addr:0x%llx len:%ld\n", md->va, md->len);
	rc = cxi_unmap(md);
	WARN(rc < 0, "%d cxi_unmap failed %d", __LINE__, rc);

fix_iov_base:
	kvec[0].iov_base -= VEC_OFFSET;
free_kvecs:
	for (i = 0; i < kvecs_to_free; i++)
		kfree(kvec[i].iov_base);
	kfree(kvec);
free_lni:
	cxi_lni_free(lni);
out:
	return rc;
}

static void free_sgtable(struct sg_table *sgt)
{
	int i;
	struct scatterlist *sg;

	for_each_sgtable_sg(sgt, sg, i)
		if (sg_page(sg))
			__free_pages(sg_page(sg), get_order(sg->length));

	sg_free_table(sgt);
}

static void unmap_free_sgtable(struct tdev *tdev, struct sg_table *sgt)
{
	dma_unmap_sgtable(&tdev->dev->pdev->dev, sgt, DMA_BIDIRECTIONAL, 0);
	free_sgtable(sgt);
}

static int alloc_sgtable(struct tdev *tdev, int npages, struct sg_table *sgt)
{
	int i;
	int rc;
	struct scatterlist *sg;
	struct page *page;
	struct device *device = &tdev->dev->pdev->dev;

	rc = sg_alloc_table(sgt, npages, GFP_KERNEL);
	if (rc)
		return rc;

	for_each_sgtable_sg(sgt, sg, i) {
		page = alloc_page(GFP_KERNEL);
		sg_set_page(sg, page, PAGE_SIZE, 0);
		memset(sg_virt(sg), 0, PAGE_SIZE);
	}

	rc = dma_map_sgtable(device, sgt, DMA_BIDIRECTIONAL, 0);
	if (rc)
		goto free_pages;

	return 0;

free_pages:
	free_sgtable(sgt);

	return rc;
}

static int check_sgtable_result(u8 *src, struct sg_table *sgt)
{
	int i;
	int j;
	int errors = 0;
	struct scatterlist *sg;

	for_each_sgtable_sg(sgt, sg, j) {
		u8 *sg_b = sg_virt(sg);
		long len = sg_dma_len(sg);

		for (i = 0; i < len; i++, src++) {
			if (*src != sg_b[i]) {
				pr_info("Data mismatch: src:%px idx %2d - %02x != %02x\n",
					src, i, *src, sg_b[i]);
				errors++;
			}

			if (errors > 10)
				break;
		}
	}

	return errors ? -1 : 0;
}

#define SGPAGES 10
static int test_sgtable1(struct tdev *tdev)
{
	int i;
	int rc;
	int ret;
	struct cxi_md *md;
	struct sg_table sgt = {};
	size_t len = SGPAGES * PAGE_SIZE;
	struct mem_window snd_mem = {.length = len};

	pr_info("%s\n", __func__);

	rc = test_setup(tdev, CXI_DEFAULT_SVC_ID);
	if (rc)
		return rc;

	snd_mem.buffer = kzalloc(snd_mem.length, GFP_KERNEL);
	if (!snd_mem.buffer) {
		rc = -ENOMEM;
		goto teardown;
	}

	rc = test_map(tdev, &snd_mem, false, 0);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto free_sndmem;
	}

	rc = alloc_sgtable(tdev, SGPAGES, &sgt);
	if (rc)
		goto unmap_snd;

	md = cxi_map(tdev->lni, 0, len, MAP_NTA_RW | CXI_MAP_ALLOC_MD, NULL);
	if (IS_ERR(md)) {
		rc = PTR_ERR(md);
		pr_err("cxi_map failed %d\n", rc);
		goto free_sgtable;
	}

	rc = cxi_update_sgtable(md, &sgt);
	if (rc)
		goto unmap_md;

	test_append_le(tdev, len, md, 0);

	for (i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto unmap_md;
	}

	rc = check_sgtable_result(snd_mem.buffer, &sgt);

unmap_md:
	ret = cxi_unmap(md);
	WARN(ret < 0, "cxi_unmap failed %d", ret);
free_sgtable:
	unmap_free_sgtable(tdev, &sgt);
unmap_snd:
	ret = test_unmap(tdev, &snd_mem, false);
	WARN(ret < 0, "cxi_unmap failed %d\n", ret);
free_sndmem:
	kfree(snd_mem.buffer);
teardown:
	test_teardown(tdev);

	return rc;
}

static int test_sgtable2(struct tdev *tdev)
{
	int i;
	int rc;
	struct cxi_md *sg_md;
	struct cxi_md snd_md = {};
	struct sg_table sgt1 = {};
	struct sg_table sgt2 = {};
	struct mem_window snd_mem;
	size_t len = SGPAGES * PAGE_SIZE;

	pr_info("%s\n", __func__);

	rc = test_setup(tdev, CXI_DEFAULT_SVC_ID);
	if (rc)
		return rc;

	snd_mem.length = len;
	snd_mem.md = &snd_md;

	snd_mem.buffer = kzalloc(snd_mem.length, GFP_KERNEL);
	if (!snd_mem.buffer) {
		rc = -ENOMEM;
		goto teardown;
	}

	rc = test_map(tdev, &snd_mem, false, 0);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto free_sndmem;
	}

	rc = alloc_sgtable(tdev, SGPAGES, &sgt1);
	if (rc)
		goto unmap_snd;

	rc = alloc_sgtable(tdev, SGPAGES, &sgt2);
	if (rc)
		goto free_sgtable1;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt1, CXI_MAP_WRITE | CXI_MAP_READ);
	if (IS_ERR(sg_md)) {
		rc = PTR_ERR(sg_md);
		pr_err("cxi_map_sgtable failed %d\n", rc);
		goto free_sgtable2;
	}

	test_append_le(tdev, len, sg_md, 0);

	for (i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto unmap_sg;
	}

	rc = check_sgtable_result(snd_mem.buffer, &sgt1);
	if (rc)
		goto unmap_sg;

	rc = cxi_clear_md(sg_md);
	if (rc)
		goto unmap_sg;

	rc = cxi_update_sgtable(sg_md, &sgt2);
	if (rc)
		goto unmap_sg;

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto unmap_sg;
	}

	rc = check_sgtable_result(snd_mem.buffer, &sgt2);

unmap_sg:
	rc = cxi_unmap(sg_md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);
free_sgtable2:
	unmap_free_sgtable(tdev, &sgt2);
free_sgtable1:
	unmap_free_sgtable(tdev, &sgt1);
unmap_snd:
	rc = test_unmap(tdev, &snd_mem, false);
	WARN(rc < 0, "cxi_unmap failed %d\n", rc);
free_sndmem:
	kfree(snd_mem.buffer);
teardown:
	test_teardown(tdev);

	return rc;
}

#define HPAGE (PAGE_SIZE >> 1)
#define QPAGE (PAGE_SIZE >> 2)

struct scatterdata {
	size_t offset;
	size_t length;
};

static int fill_sgtable(struct tdev *tdev, struct sg_table *sgt, int npages,
			struct scatterdata *sdata, size_t *length, bool init)
{
	int i;
	int rc;
	struct page *page;
	struct scatterlist *sg;
	struct device *device = &tdev->dev->pdev->dev;

	rc = sg_alloc_table(sgt, npages, GFP_KERNEL);
	if (rc)
		return rc;

	for_each_sgtable_sg(sgt, sg, i) {
		page = alloc_pages(GFP_KERNEL,
				   get_order(sdata[i].length + sdata[i].offset));
		if (!page) {
			pr_info("failed alloc_page\n");
			goto free_pages;
		}
		sg_set_page(sg, page, sdata[i].length, sdata[i].offset);

		if (init)
			memset(sg_virt(sg), 0, sdata[i].length);

		*length += sdata[i].length;
	}

	rc = dma_map_sgtable(device, sgt, DMA_BIDIRECTIONAL, 0);
	if (rc)
		goto free_pages;

	return 0;

free_pages:
	free_sgtable(sgt);

	return rc;
}

static bool check_md_length(struct cxi_md *md, int nentries,
			       struct scatterdata *sd)
{
	int i;
	size_t expected_len;

	for (i = 0, expected_len = 0; i < nentries; i++)
		expected_len += PAGE_ALIGN(sd->offset + sd->length);

	if (md->len != expected_len) {
		pr_err("MD length incorrect expected:%lx have:%lx\n",
		       expected_len, md->len);
		return false;
	}

	return true;
}

static int test_sgtable3(struct tdev *tdev)
{
	int i;
	u32 *b;
	int rc;
	int ret;
	size_t offset;
	size_t len = 0;
	struct cxi_md *sg_md;
	struct cxi_md snd_md = {};
	struct sg_table sgt = {};
	struct sg_table sgt_bad = {};
	struct mem_window snd_mem;
	struct scatterdata sdata[] = {
		{.offset = HPAGE, .length = HPAGE},
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 0, .length = QPAGE},
	};
	struct scatterdata bad_sdata0[] = {
		{.offset = HPAGE, .length = QPAGE}, // gap at end of page
		{.offset = 0, .length = PAGE_SIZE},
	};
	struct scatterdata bad_sdata1[] = {
		{.offset = HPAGE, .length = HPAGE},
		{.offset = 1, .length = PAGE_SIZE - 1}, // gap at beginning
		{.offset = 0, .length = QPAGE},
	};
	struct scatterdata bad_sdata2[] = {
		{.offset = HPAGE, .length = HPAGE},
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 1, .length = QPAGE}, // gap at beginning
	};
	struct scatterdata bad_sdata3[] = {
		{.offset = HPAGE, .length = PAGE_SIZE}, // runs into next page
		{.offset = 0, .length = QPAGE},
	};
	struct scatterdata bad_sdata4[] = {
		{.offset = PAGE_SIZE + 1, .length = QPAGE}, // large offset
	};
	struct scatterdata good_sdata0[] = {
		{.offset = 0, .length = QPAGE},
	};
	struct scatterdata good_sdata1[] = {
		{.offset = HPAGE, .length = QPAGE},
	};
	struct scatterdata good_sdata2[] = {
		{.offset = 0, .length = PAGE_SIZE * 2},
	};
	struct scatterdata good_sdata3[] = {
		{.offset = QPAGE * 3, .length = HPAGE},
	};
	struct scatterdata too_big[] = {
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 0, .length = PAGE_SIZE},
		{.offset = 0, .length = PAGE_SIZE},
	};

	pr_info("%s\n", __func__);

	rc = test_setup(tdev, CXI_DEFAULT_SVC_ID);
	if (rc)
		return rc;

	rc = fill_sgtable(tdev, &sgt_bad, ARRAY_SIZE(bad_sdata0), bad_sdata0,
			  &len, false);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt_bad, 0);
	rc = PTR_ERR(sg_md);
	if (rc != -EINVAL) {
		pr_err("cxi_map_sgtable should fail with -EINVAL %d\n", rc);
		goto free_bad_sgtable;
	}
	unmap_free_sgtable(tdev, &sgt_bad);

	rc = fill_sgtable(tdev, &sgt_bad, ARRAY_SIZE(bad_sdata1), bad_sdata1,
			  &len, false);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt_bad, 0);
	rc = PTR_ERR(sg_md);
	if (rc != -EINVAL) {
		pr_err("cxi_map_sgtable should fail with -EINVAL %d\n", rc);
		goto free_bad_sgtable;
	}
	unmap_free_sgtable(tdev, &sgt_bad);

	rc = fill_sgtable(tdev, &sgt_bad, ARRAY_SIZE(bad_sdata2), bad_sdata2,
			  &len, false);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt_bad, 0);
	rc = PTR_ERR(sg_md);
	if (rc != -EINVAL) {
		pr_err("cxi_map_sgtable should fail with -EINVAL %d\n", rc);
		goto free_bad_sgtable;
	}
	unmap_free_sgtable(tdev, &sgt_bad);

	rc = fill_sgtable(tdev, &sgt_bad, ARRAY_SIZE(bad_sdata3), bad_sdata3,
			  &len, false);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt_bad, 0);
	rc = PTR_ERR(sg_md);
	if (rc != -EINVAL) {
		pr_err("cxi_map_sgtable should fail with -EINVAL %d\n", rc);
		goto free_bad_sgtable;
	}
	unmap_free_sgtable(tdev, &sgt_bad);

	rc = fill_sgtable(tdev, &sgt_bad, ARRAY_SIZE(bad_sdata4), bad_sdata4,
			  &len, false);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt_bad, 0);
	rc = PTR_ERR(sg_md);
	if (rc != -EINVAL) {
		pr_err("cxi_map_sgtable should fail with -EINVAL %d\n", rc);
		goto free_bad_sgtable;
	}

	rc = fill_sgtable(tdev, &sgt, ARRAY_SIZE(good_sdata0), good_sdata0,
			  &len, true);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt, 0);
	if (IS_ERR(sg_md)) {
		rc = PTR_ERR(sg_md);
		pr_err("cxi_map_sgtable should succeed %d\n", rc);
		goto free_sgtable;
	}

	if (!check_md_length(sg_md, ARRAY_SIZE(good_sdata0), good_sdata0)) {
		ret = cxi_unmap(sg_md);
		WARN(ret < 0, "cxi_unmap failed %d", ret);
		goto free_sgtable;
	}

	ret = cxi_unmap(sg_md);
	WARN(ret < 0, "cxi_unmap failed %d", ret);
	unmap_free_sgtable(tdev, &sgt);

	rc = fill_sgtable(tdev, &sgt, ARRAY_SIZE(good_sdata1), good_sdata1,
			  &len, true);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt, 0);
	if (IS_ERR(sg_md)) {
		rc = PTR_ERR(sg_md);
		pr_err("cxi_map_sgtable should succeed %d\n", rc);
		goto free_sgtable;
	}

	if (!check_md_length(sg_md, ARRAY_SIZE(good_sdata1), good_sdata1)) {
		ret = cxi_unmap(sg_md);
		WARN(ret < 0, "cxi_unmap failed %d", ret);
		goto free_sgtable;
	}

	ret = cxi_unmap(sg_md);
	WARN(ret < 0, "cxi_unmap failed %d", ret);
	unmap_free_sgtable(tdev, &sgt);

	rc = fill_sgtable(tdev, &sgt, ARRAY_SIZE(good_sdata2), good_sdata2,
			  &len, true);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt, 0);
	if (IS_ERR(sg_md)) {
		rc = PTR_ERR(sg_md);
		pr_err("cxi_map_sgtable should succeed %d\n", rc);
		goto free_sgtable;
	}

	if (!check_md_length(sg_md, ARRAY_SIZE(good_sdata2), good_sdata2)) {
		ret = cxi_unmap(sg_md);
		WARN(ret < 0, "cxi_unmap failed %d", ret);
		goto free_sgtable;
	}

	ret = cxi_unmap(sg_md);
	WARN(ret < 0, "cxi_unmap failed %d", ret);
	unmap_free_sgtable(tdev, &sgt);

	rc = fill_sgtable(tdev, &sgt, ARRAY_SIZE(good_sdata3), good_sdata3,
			  &len, true);
	if (rc)
		goto teardown;

	sg_md = cxi_map_sgtable(tdev->lni, &sgt, 0);
	if (IS_ERR(sg_md)) {
		rc = PTR_ERR(sg_md);
		pr_err("cxi_map_sgtable should succeed %d\n", rc);
		goto free_sgtable;
	}

	if (!check_md_length(sg_md, ARRAY_SIZE(good_sdata3), good_sdata3)) {
		ret = cxi_unmap(sg_md);
		WARN(ret < 0, "cxi_unmap failed %d", ret);
		goto free_sgtable;
	}

	ret = cxi_unmap(sg_md);
	WARN(ret < 0, "cxi_unmap failed %d", ret);
	unmap_free_sgtable(tdev, &sgt);

	len = 0;
	rc = fill_sgtable(tdev, &sgt, ARRAY_SIZE(sdata), sdata, &len, true);
	if (rc)
		goto free_bad_sgtable;

	snd_mem.length = len;
	snd_mem.md = &snd_md;

	snd_mem.buffer = kzalloc(snd_mem.length, GFP_KERNEL);
	if (!snd_mem.buffer) {
		rc = -ENOMEM;
		goto free_sgtable;
	}

	rc = test_map(tdev, &snd_mem, false, 0);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto free_sndmem;
	}

	sg_md = cxi_map_sgtable(tdev->lni, &sgt, CXI_MAP_WRITE | CXI_MAP_READ);
	if (IS_ERR(sg_md)) {
		rc = PTR_ERR(sg_md);
		pr_err("cxi_map_sgtable failed %d\n", rc);
		goto unmap_snd;
	}

	rc = cxi_update_sgtable(sg_md, &sgt_bad);
	if (!rc) {
		pr_err("cxi_update_sgtable should fail with -EINVAL %d\n", rc);
		goto unmap_sg;
	}

	unmap_free_sgtable(tdev, &sgt_bad);

	rc = fill_sgtable(tdev, &sgt_bad, ARRAY_SIZE(too_big), too_big,
			  &len, false);
	if (rc)
		goto unmap_sg;

	rc = cxi_update_sgtable(sg_md, &sgt_bad);
	if (rc != -EINVAL) {
		pr_err("cxi_map_sgtable should fail with -EINVAL %d\n", rc);
		goto unmap_sg;
	}

	offset = (u64)sg_virt(sgt.sgl) - (u64)page_address(sg_page(sgt.sgl));
	test_append_le(tdev, snd_mem.length, sg_md, offset);

	for (i = 0, b = (u32 *)snd_mem.buffer; i < snd_mem.length / sizeof(u32); i++)
		b[i] = i;

	rc = test_do_put(tdev, &snd_mem, snd_mem.length, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto unmap_sg;
	}

	rc = check_sgtable_result(snd_mem.buffer, &sgt);
	if (rc)
		goto unmap_sg;

unmap_sg:
	ret = cxi_unmap(sg_md);
	WARN(ret < 0, "cxi_unmap failed %d", ret);
unmap_snd:
	ret = test_unmap(tdev, &snd_mem, false);
	WARN(ret < 0, "cxi_unmap failed %d\n", ret);
free_sndmem:
	kfree(snd_mem.buffer);
free_sgtable:
	unmap_free_sgtable(tdev, &sgt);
free_bad_sgtable:
	unmap_free_sgtable(tdev, &sgt_bad);
teardown:
	test_teardown(tdev);

	return rc;
}

static int map_bvec(struct tdev *tdev)
{
	int rc;
	struct cxi_md *md;
	struct iov_iter iter = {};

	pr_info("%s\n", __func__);

	tdev->lni = cxi_lni_alloc(tdev->dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(tdev->lni)) {
		rc = PTR_ERR(tdev->lni);
		goto out;
	}

	rc = alloc_bvec(tdev, N_VECS, &iter, &md);
	if (rc)
		goto free_lni;

	free_bvec(tdev, &iter);
	rc = cxi_unmap(md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);

free_lni:
	cxi_lni_free(tdev->lni);
out:
	return rc;
}

/* allocate and map 1M and 4M buffers */
static int map_test(struct tdev *tdev)
{
	int rc;
	u64 *buffer;
	u64 *vm_buffer;
	const size_t buf_size = 1 << 20;
	const size_t vm_buf_size = 1 << 22;
	struct cxi_md *md;
	struct cxi_md *vm_md;

	pr_info("%s\n", __func__);

	tdev->lni = cxi_lni_alloc(tdev->dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(tdev->lni)) {
		rc = PTR_ERR(tdev->lni);
		goto out;
	}

	buffer = kzalloc(buf_size, GFP_KERNEL);
	if (!buffer) {
		rc = -ENOMEM;
		goto free_lni;
	}

	vm_buffer = vmalloc(vm_buf_size);
	if (!vm_buffer) {
		rc = -ENOMEM;
		goto free_buffer;
	}

	md = cxi_map(tdev->lni, (uintptr_t)buffer, buf_size,
		     CXI_MAP_WRITE | CXI_MAP_READ, NULL);
	if (IS_ERR(md)) {
		rc = PTR_ERR(md);
		pr_err("cxi_map failed %d\n", rc);
		goto free_vmbuf;
	}

	pr_info("cxi_map va:%llx len:%ld iova:0x%llx\n", md->va, buf_size,
		(u64)md->iova);

	vm_md = cxi_map(tdev->lni, (uintptr_t)vm_buffer, vm_buf_size,
		     CXI_MAP_WRITE | CXI_MAP_READ, NULL);
	if (IS_ERR(vm_md)) {
		rc = PTR_ERR(vm_md);
		pr_err("cxi_map of vmalloc'd buffer failed %d\n", rc);
		goto unmap_buf;
	}

	pr_info("cxi_map va:%llx len:%ld iova:0x%llx\n", vm_md->va,
		vm_buf_size, (u64)vm_md->iova);

	pr_info("cxi_unmap addr:0x%llx len:%ld\n", vm_md->va, vm_md->len);
	rc = cxi_unmap(vm_md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);

unmap_buf:
	pr_info("cxi_unmap addr:0x%llx len:%ld\n", md->va, md->len);
	rc = cxi_unmap(md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);
free_vmbuf:
	vfree(vm_buffer);
free_buffer:
	kfree(buffer);
free_lni:
	cxi_lni_free(tdev->lni);
out:
	return rc;
}

static int test_rma(struct tdev *tdev, bool phys)
{
	int i;
	int rc;
	int lac = 0;
	int errors = 0;
	struct cxi_md snd_md = {};
	struct cxi_md rma_md = {};
	struct mem_window snd_mem;
	struct mem_window rma_mem;
	size_t len = 256 * 1024;

	pr_info("%s %s memory\n", __func__, phys ? "physical" : "virtual");

	rma_mem.length = len;
	snd_mem.length = len;
	rma_mem.md = &rma_md;
	snd_mem.md = &snd_md;

	rc = test_setup(tdev, CXI_DEFAULT_SVC_ID);
	if (rc)
		return rc;

	if (phys) {
		lac = cxi_phys_lac_alloc(tdev->lni);
		if (lac < 0) {
			pr_err("cxi_phys_lac_alloc failed %d\n", lac);
			goto pte_unmap;
		}
	}

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

	rc = test_map(tdev, &snd_mem, phys, lac);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto free_snd;
	}

	rc = test_map(tdev, &rma_mem, phys, lac);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto unmap_snd;
	}

	pr_info("cxi_map addr:%p len:%ld iova:0x%llx\n", rma_mem.buffer,
		rma_mem.length, (u64)snd_mem.md->iova);

	test_append_le(tdev, len, rma_mem.md, 0);

	WARN_ON(tdev->cq_target->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	for (i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

	memset(rma_mem.buffer, 0, len);

	pr_info("sending len:%ld\n", snd_mem.md->len);

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
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

	if (!errors)
		pr_info("%s success\n", __func__);

unmap_rma:
	rc = test_unmap(tdev, &rma_mem, phys);
	WARN(rc < 0, "cxi_unmap failed %d\n", rc);
unmap_snd:
	rc = test_unmap(tdev, &snd_mem, phys);
	WARN(rc < 0, "cxi_unmap failed %d\n", rc);
free_snd:
	kfree(snd_mem.buffer);
free_rma:
	kfree(rma_mem.buffer);
free_lac:
	if (phys)
		cxi_phys_lac_free(tdev->lni, lac);
pte_unmap:
	test_teardown(tdev);

	return rc;
}

static int init_bvec(struct tdev *tdev, struct bvec_info *bi, size_t nbv,
		     size_t len)
{
	int rc;

	rc = alloc_bvec(tdev, nbv, &bi->iter, &bi->md);
	if (rc)
		return rc;

	bvec_iter_init(&bi->iter, false);

	return rc;
}

static int check_bvec_result(struct bvec_info *src_bi, struct bvec_info *rma_bi,
			     size_t nbv)
{
	int i;
	int j;
	int errors = 0;

	for (i = 0; i < nbv; i++) {
		u32 *s = page_address(src_bi->iter.bvec[i].bv_page);
		u32 *d = page_address(rma_bi->iter.bvec[i].bv_page);

		for (j = 0; j < (PAGE_SIZE / sizeof(u32)) ; j += sizeof(u32)) {
			if (*s != *d) {
				pr_info("Data mismatch: %ld - 0x%x != 0x%x\n",
					(i * PAGE_SIZE) + j, *s, *d);
				errors++;
			}

			if (errors > 10)
				break;

			s++;
			d++;
		}

		if (errors)
			break;
	}

	return errors;
}

/*
 * Test RMA transaction with BVECs.
 * Allocate source and destination bvecs.
 * Initiate a put, check the data.
 * Invalidate a bvec and then update the MD with a new bvec.
 * Initiate a put, check the data.
 */
static int test_bvec_rma(struct tdev *tdev)
{
	int i;
	int rc;
	struct bio_vec *bvec;
	struct bio_vec *dbvec;
	struct bvec_info src_bi;
	struct bvec_info rma_bi;
	struct mem_window snd_mem;
	size_t nbv = 256;
	size_t len = nbv * PAGE_SIZE;

	pr_info("%s\n", __func__);

	rc = test_setup(tdev, CXI_DEFAULT_SVC_ID);

	rc = init_bvec(tdev, &src_bi, nbv, len);
	if (rc)
		goto tear_down;

	rc = init_bvec(tdev, &rma_bi, nbv, len);
	if (rc)
		goto clean_up_src_bi;

	dbvec = kmalloc_array(nbv, sizeof(*bvec), GFP_KERNEL);
	if (!dbvec)
		goto clean_up_rma_bi;

	test_append_le(tdev, len, rma_bi.md, 0);
	WARN_ON(tdev->cq_target->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	snd_mem.buffer = NULL;
	snd_mem.length = len;
	snd_mem.md = src_bi.md;

	pr_info("sending len:%ld\n", snd_mem.md->len);

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto free_dbvec;
	}

	rc = check_bvec_result(&src_bi, &rma_bi, nbv);
	if (rc)
		goto free_dbvec;

	/* Remove page table entries and invalidate cached entries
	 * associated with MD.
	 */
	rc = cxi_clear_md(src_bi.md);
	if (rc)
		goto free_dbvec;

	bvec = kmalloc_array(nbv, sizeof(*bvec), GFP_KERNEL);
	if (!bvec) {
		rc = -ENOMEM;
		goto free_dbvec;
	}

	/* Make sure we don't use the same pages as last time. Save for
	 * later removal.
	 */
	for (i = 0; i < nbv; i++)
		dbvec[i].bv_page = src_bi.iter.bvec[i].bv_page;

	rc = alloc_bvec_pages(nbv, &src_bi.iter, bvec);
	if (rc) {
		kfree(bvec);
		goto free_dbvec;
	}

	bvec_iter_init(&rma_bi.iter, true);
	bvec_iter_init(&src_bi.iter, false);

	/* Now update the MD with a new bvec. */
	rc = cxi_update_iov(src_bi.md, &src_bi.iter);
	if (rc)
		goto clean_up_bis;

	/* now test with new bvec */
	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto clean_up_bis;
	}

	rc = check_bvec_result(&src_bi, &rma_bi, nbv);
	if (!rc)
		pr_info("%s success\n", __func__);

clean_up_bis:
	for (i = 0; i < nbv; i++)
		__free_page(dbvec[i].bv_page);
free_dbvec:
	kfree(dbvec);
clean_up_rma_bi:
	free_bvec(tdev, &rma_bi.iter);
	rc = cxi_unmap(rma_bi.md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);
clean_up_src_bi:
	free_bvec(tdev, &src_bi.iter);
	rc = cxi_unmap(src_bi.md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);
tear_down:
	test_teardown(tdev);

	return rc;
}

/*
 * Test RMA transaction with BVECs using CXI_MAP_ALLOC_MD flag.
 * Allocate an MD wih the CXI_MAP_ALLOC_MD flag.
 * Test unmapping an allocated MD.
 * Allocate source and destination bvecs.
 * Allocate and update an MD.
 * Initiate a put, check the data.
 */
static int test_alloc_md(struct tdev *tdev)
{
	int rc;
	struct bio_vec *bvec;
	struct bvec_info src_bi;
	struct bvec_info rma_bi;
	struct mem_window snd_mem;
	size_t nbv = 256;
	size_t len = nbv * PAGE_SIZE;

	pr_info("%s\n", __func__);

	rc = test_setup(tdev, CXI_DEFAULT_SVC_ID);

	src_bi.md = cxi_map(tdev->lni, 0, nbv * PAGE_SIZE,
			    MAP_NTA_RW | CXI_MAP_ALLOC_MD, NULL);
	if (IS_ERR(src_bi.md)) {
		rc = PTR_ERR(src_bi.md);
		pr_err("cxi_map failed %d\n", rc);
		goto tear_down;
	}

	rc = cxi_unmap(src_bi.md);
	if (rc)
		goto tear_down;

	src_bi.md = cxi_map(tdev->lni, 0, nbv * PAGE_SIZE,
			    MAP_NTA_RW | CXI_MAP_ALLOC_MD, NULL);
	if (IS_ERR(src_bi.md)) {
		rc = PTR_ERR(src_bi.md);
		pr_err("map iov failed %d\n", rc);
		goto tear_down;
	}

	bvec = kmalloc_array(nbv, sizeof(*bvec), GFP_KERNEL);
	if (!bvec) {
		rc = -ENOMEM;
		goto unmap_src_md;
	}

	rc = alloc_bvec_pages(nbv, &src_bi.iter, bvec);
	if (rc) {
		kfree(bvec);
		goto free_bvec;
	}

	rc = cxi_update_iov(src_bi.md, &src_bi.iter);
	if (rc) {
		pr_err("map iov failed %d\n", rc);
		goto clean_up_src_bi;
	}

	rc = init_bvec(tdev, &rma_bi, nbv, len);
	if (rc)
		goto clean_up_src_bi;

	bvec_iter_init(&src_bi.iter, false);
	bvec_iter_init(&rma_bi.iter, true);

	test_append_le(tdev, len, rma_bi.md, 0);

	snd_mem.buffer = NULL;
	snd_mem.length = len;
	snd_mem.md = src_bi.md;

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
	if (rc < 0) {
		pr_err("test_do_put failed %d\n", rc);
		goto clean_up_rma_bi;
	}

	rc = check_bvec_result(&src_bi, &rma_bi, nbv);
	if (rc)
		pr_err("check_bvec_result failed %d\n", rc);

clean_up_rma_bi:
	free_bvec(tdev, &rma_bi.iter);
	rc = cxi_unmap(rma_bi.md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);
clean_up_src_bi:
	free_bvec(tdev, &src_bi.iter);
free_bvec:
	if (src_bi.iter.bvec)
		kfree(bvec);
unmap_src_md:
	rc = cxi_unmap(src_bi.md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);
tear_down:
	test_teardown(tdev);

	if (!rc)
		pr_info("%s success\n", __func__);

	return rc;
}

static int test_atu(struct tdev *tdev)
{
	int rc;
	int retry;
	u64 *tgt_data;
	const size_t tgt_data_size = 2 * PAGE_SIZE;
	struct cxi_md *md;
	struct cxi_eq_attr eq_attr = {};

	pr_info("%s\n", __func__);

	tdev->lni = cxi_lni_alloc(tdev->dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(tdev->lni)) {
		rc = PTR_ERR(tdev->lni);
		goto out;
	}

	/* Choose a random pid */
	tdev->vni = 1;
	tdev->pid = 30;
	tdev->pid_offset = 3;
	tdev->domain = cxi_domain_alloc(tdev->lni, tdev->vni, tdev->pid);
	if (IS_ERR(tdev->domain)) {
		rc = PTR_ERR(tdev->domain);
		goto free_ni;
	}

	tdev->cp = cxi_cp_alloc(tdev->lni, tdev->vni, CXI_TC_BEST_EFFORT,
				CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(tdev->cp)) {
		rc = PTR_ERR(tdev->cp);
		goto free_dom;
	}

	tdev->eq_buf_len = PAGE_SIZE * 4;
	tdev->eq_buf = kzalloc(tdev->eq_buf_len, GFP_KERNEL);
	if (!tdev->eq_buf) {
		rc = -ENOMEM;
		goto free_cp;
	}

	tdev->eq_buf_md = cxi_map(tdev->lni, (uintptr_t)tdev->eq_buf,
				 tdev->eq_buf_len,
				 CXI_MAP_WRITE | CXI_MAP_READ,
				 NULL);
	if (IS_ERR(tdev->eq_buf_md)) {
		rc = PTR_ERR(tdev->eq_buf_md);
		goto free_eq_buf;
	}

	eq_attr.queue = tdev->eq_buf;
	eq_attr.queue_len = tdev->eq_buf_len;
	eq_attr.flags = tdev->eq_flags;

	tdev->eq = cxi_eq_alloc(tdev->lni, tdev->eq_buf_md, &eq_attr,
				eq_cb, NULL, NULL, NULL);
	if (IS_ERR(tdev->eq)) {
		rc = PTR_ERR(tdev->eq);
		goto unmap_eq_buf;
	}

	memset(&tdev->cq_alloc_opts, 0, sizeof(tdev->cq_alloc_opts));
	tdev->cq_alloc_opts.count = 50;
	tdev->cq_alloc_opts.flags = CXI_CQ_IS_TX;
	tdev->cq_alloc_opts.lcid = tdev->cp->lcid;
	tdev->cq_transmit = cxi_cq_alloc(tdev->lni, tdev->eq,
					 &tdev->cq_alloc_opts, 0);
	if (IS_ERR(tdev->cq_transmit)) {
		rc = PTR_ERR(tdev->cq_transmit);
		goto free_eq;
	}

	memset(&tdev->cq_alloc_opts, 0, sizeof(tdev->cq_alloc_opts));
	tdev->cq_alloc_opts.count = 50;
	tdev->cq_target = cxi_cq_alloc(tdev->lni, tdev->eq,
				       &tdev->cq_alloc_opts, 0);
	if (IS_ERR(tdev->cq_target)) {
		rc = PTR_ERR(tdev->cq_target);
		goto free_cq_transmit;
	}

	tdev->pt_alloc_opts = (struct cxi_pt_alloc_opts) {
		.is_matching = 0,
		.en_flowctrl = 1,
	};
	tdev->pt = cxi_pte_alloc(tdev->lni, tdev->eq, &tdev->pt_alloc_opts);
	if (IS_ERR(tdev->pt)) {
		rc = PTR_ERR(tdev->pt);
		goto free_cq_target;
	}

	cxi_pte_free(tdev->pt);

	tdev->pt_alloc_opts = (struct cxi_pt_alloc_opts) {
		.is_matching = 0,
	};
	tdev->pt = cxi_pte_alloc(tdev->lni, tdev->eq, &tdev->pt_alloc_opts);
	if (IS_ERR(tdev->pt)) {
		rc = PTR_ERR(tdev->pt);
		goto free_cq_target;
	}

	rc = cxi_pte_map(tdev->pt, tdev->domain, tdev->pid_offset,
			 false, &tdev->pt_index);
	if (rc)
		goto free_pt;

	WARN_ON(tdev->cq_transmit->status->rd_ptr != 0 + C_CQ_FIRST_WR_PTR);

	/* Setup DFA. */
	cxi_build_dfa(tdev->dev->prop.nid, tdev->pid,
		      tdev->dev->prop.pid_bits, tdev->pid_offset, &tdev->dfa,
		      &tdev->index_ext);

	pr_info("ep_defined=%x index_ext=%x\n",
		tdev->dfa.unicast.endpoint_defined,
		tdev->index_ext);

	/* Set LCID to index 0 */
	rc = cxi_cq_emit_cq_lcid(tdev->cq_transmit, 0);
	if (rc)
		pr_err("Emit LCID failed: %d\n", rc);

	cxi_cq_ring(tdev->cq_transmit);

	/* LCID is 64 bytes plus 64 bytes NOP */
	retry = 100;
	while (retry-- &&
	       tdev->cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR)
		mdelay(10);
	WARN_ON(tdev->cq_transmit->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	/* Map a buffer */
	tgt_data = kzalloc(tgt_data_size, GFP_KERNEL);
	if (!tgt_data) {
		rc = -ENOMEM;
		goto pte_unmap;
	}

	md = cxi_map(tdev->lni, (uintptr_t)tgt_data, tgt_data_size,
		     CXI_MAP_WRITE | CXI_MAP_READ, NULL);
	if (IS_ERR(md)) {
		rc = PTR_ERR(md);
		pr_err("cxi_map failed %d\n", rc);
		goto free_buf;
	}

	pr_info("cxi_map addr:%p len:%ld iova:0x%llx\n", tgt_data,
		tgt_data_size, (u64)md->iova);

	test_append_le(tdev, tgt_data_size, md, 0);

	retry = 100;
	while (retry-- && tdev->cq_target->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR)
		mdelay(10);
	WARN_ON(tdev->cq_target->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	test_add_cstate(tdev);
	wait_for_event(tdev, C_EVENT_STATE_CHANGE);

	retry = 100;
	while (retry-- &&
	       tdev->cq_transmit->status->rd_ptr != 3 + C_CQ_FIRST_WR_PTR)
		mdelay(10);
	WARN_ON(tdev->cq_transmit->status->rd_ptr != 3 + C_CQ_FIRST_WR_PTR);

	test_idc_put(tdev, tgt_data);

	pr_info("cxi_unmap iova:0x%llx len:%ld\n", md->iova, md->len);
	rc = cxi_unmap(md);
	WARN(rc < 0, "cxi_unmap failed %d", rc);

	pr_info("interrupts seen: %d\n", atomic_read(&eq_cb_called));
	WARN(atomic_read(&eq_cb_called) == 0, "No interrupts were seen\n");

free_buf:
	kfree(tgt_data);
pte_unmap:
	cxi_pte_unmap(tdev->pt, tdev->domain, tdev->pt_index);
free_pt:
	cxi_pte_free(tdev->pt);
free_cq_target:
	cxi_cq_free(tdev->cq_target);
free_cq_transmit:
	cxi_cq_free(tdev->cq_transmit);
free_eq:
	cxi_eq_free(tdev->eq);
unmap_eq_buf:
	cxi_unmap(tdev->eq_buf_md);
free_eq_buf:
	kfree(tdev->eq_buf);
free_cp:
	cxi_cp_free(tdev->cp);
free_dom:
	cxi_domain_free(tdev->domain);
free_ni:
	cxi_lni_free(tdev->lni);
out:
	return rc;
}

#define LNIS_PER_RGID 2
static int build_service(struct cxi_dev *dev, struct cxi_svc_desc *desc,
			 int lpr)
{
	int rc;
	int llpr;

	desc->enable = 1,
	desc->is_system_svc = 1,
	desc->restricted_vnis = 1,
	desc->num_vld_vnis = 1,
	desc->vnis[0] = 8U,
	desc->resource_limits = false,
	desc->restricted_members = false,

	rc = cxi_svc_alloc(dev, desc, NULL, "test-atu");
	if (rc < 0) {
		pr_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}

	desc->svc_id = rc;

	rc = cxi_svc_update(dev, desc);
	if (rc < 0) {
		pr_err("cxi_svc_update_priv failed: %d\n", rc);
		goto update_err;
	}

	rc = cxi_svc_set_lpr(dev, desc->svc_id, lpr);
	if (rc < 0) {
		pr_err("cxi_svc_set_lpr failed: %d\n", rc);
		goto update_err;
	}

	llpr = cxi_svc_get_lpr(dev, desc->svc_id);
	if (llpr != lpr) {
		pr_err("cxi_svc_get_lpr failed llpr:%d expected:%d\n", llpr, lpr);
		goto update_err;
	}

	return 0;

update_err:
	cxi_svc_destroy(dev, desc->svc_id);
err:
	return rc;
}

static int test_lni_alloc(struct tdev *tdev)
{
	int rc;
	int i;
	int lpr = 1;
	struct cxi_lni *lni;
	struct cxi_lni **lni_l;
	struct cxi_svc_desc desc = {};

	pr_info("%s\n", __func__);

	rc = build_service(tdev->dev, &desc, lpr);
	if (rc)
		return rc;

	lni = cxi_lni_alloc(tdev->dev, desc.svc_id);
	if (IS_ERR(lni)) {
		pr_err("cxi_lni_alloc failed %ld\n", PTR_ERR(lni));
		rc = PTR_ERR(lni);
		goto dest_svc;
	}

	rc = cxi_svc_set_lpr(tdev->dev, desc.svc_id, lpr);
	if (!rc) {
		pr_err("cxi_svc_set_lpr should fail\n");
		cxi_lni_free(lni);
		goto dest_svc;
	}

	cxi_lni_free(lni);

	lni_l = kcalloc(C_NUM_RGIDS * lpr, sizeof(*lni_l),
			GFP_KERNEL);
	if (!lni_l) {
		rc = -ENOMEM;
		goto dest_svc;
	}

	for (i = 0; i < (C_NUM_RGIDS * lpr); i++) {
		lni_l[i] = cxi_lni_alloc(tdev->dev, desc.svc_id);
		if (IS_ERR(lni_l[i])) {
			/* should fail at (C_NUM_RGIDS - 1) * lpr */
			if (i == ((C_NUM_RGIDS - 1) * lpr)) {
				pr_info("%d LNIs\n", i);
				lni_l[i] = NULL;
				rc = 0;
			} else {
				pr_err("i:%d cxi_lni_alloc failed %ld\n", i,
				       PTR_ERR(lni_l[i]));
				rc = PTR_ERR(lni_l[i]);
			}

			break;
		}
	}

	/* pick random lni and free it to create a hole in the rgid list */
	i = 10;
	cxi_lni_free(lni_l[i]);

	/* try alloc another lni */
	lni_l[i] = cxi_lni_alloc(tdev->dev, desc.svc_id);
	if (IS_ERR(lni_l[i])) {
		pr_err("i:%d cxi_lni_alloc failed %ld\n", i,
		       PTR_ERR(lni_l[i]));
		rc = PTR_ERR(lni_l[i]);
		lni_l[i] = NULL;
	}

	for (i = 0; i < (C_NUM_RGIDS * lpr); i++) {
		if (lni_l[i])
			cxi_lni_free(lni_l[i]);
	}

	kfree(lni_l);
dest_svc:
	cxi_svc_destroy(tdev->dev, desc.svc_id);
	pr_err("%s failed\n", __func__);

	return rc;
}

static int test_share_lcid(struct tdev *tdev)
{
	int i;
	int rc;
	int lac = 0;
	int errors = 0;
	struct cxi_cp *cp;
	struct cxi_lni *lni;
	struct cxi_svc_desc desc = {};
	struct cxi_md snd_md = {};
	struct cxi_md rma_md = {};
	struct mem_window snd_mem;
	struct mem_window rma_mem;
	size_t len = 256 * 1024;

	pr_info("%s\n", __func__);

	rma_mem.length = len;
	snd_mem.length = len;
	rma_mem.md = &rma_md;
	snd_mem.md = &snd_md;

	rc = build_service(tdev->dev, &desc, LNIS_PER_RGID);
	if (rc)
		return rc;

	rc = test_setup(tdev, desc.svc_id);
	if (rc)
		goto svc_destroy;

	/*
	 * Check if CP/LCID cleanup is working correctly.
	 * Allocate an LNI and CP and then free them.
	 */
	lni = cxi_lni_alloc(tdev->dev, desc.svc_id);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		goto teardown;
	}

	cp = cxi_cp_alloc(lni, tdev->vni, CXI_TC_BEST_EFFORT,
			  CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(cp)) {
		rc = PTR_ERR(cp);
		cxi_lni_free(lni);
		goto teardown;
	}

	cxi_cp_free(cp);
	cxi_lni_free(lni);

	rma_mem.buffer = kzalloc(rma_mem.length, GFP_KERNEL);
	if (!rma_mem.buffer) {
		rc = -ENOMEM;
		goto teardown;
	}

	snd_mem.buffer = kzalloc(snd_mem.length, GFP_KERNEL);
	if (!snd_mem.buffer) {
		rc = -ENOMEM;
		goto free_rma;
	}

	rc = test_map(tdev, &snd_mem, false, lac);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto free_snd;
	}

	rc = test_map(tdev, &rma_mem, false, lac);
	if (rc < 0) {
		pr_err("cxi_map failed %d\n", rc);
		goto unmap_snd;
	}

	test_append_le(tdev, len, rma_mem.md, 0);

	WARN_ON(tdev->cq_target->status->rd_ptr != 2 + C_CQ_FIRST_WR_PTR);

	for (i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

	memset(rma_mem.buffer, 0, len);

	rc = test_do_put(tdev, &snd_mem, len, 0, 0, tdev->index_ext);
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

	if (!errors)
		pr_info("%s success\n", __func__);

unmap_rma:
	rc = test_unmap(tdev, &rma_mem, false);
	WARN(rc < 0, "cxi_unmap failed %d\n", rc);
unmap_snd:
	rc = test_unmap(tdev, &snd_mem, false);
	WARN(rc < 0, "cxi_unmap failed %d\n", rc);
free_snd:
	kfree(snd_mem.buffer);
free_rma:
	kfree(rma_mem.buffer);
teardown:
	test_teardown(tdev);
svc_destroy:
	cxi_svc_destroy(tdev->dev, desc.svc_id);

	if (errors)
		rc = -1;

	return rc;
}

static int test_rgid(struct tdev *tdev)
{
	int rc;

	rc = test_lni_alloc(tdev);
	if (rc)
		return rc;

	return test_share_lcid(tdev);
}

/* Core is adding a new device */
static int add_device(struct cxi_dev *dev)
{
	struct tdev *tdev;

	/* We could ignore that device here if it doesn't fill certain
	 * criteria.
	 */

	tdev = kzalloc(sizeof(*tdev), GFP_KERNEL);
	if (tdev == NULL)
		return -ENOMEM;

	tdev->dev = dev;

	/* Do something with that new device */
	/* TODO: Skip devices on VF for now, remove when VF devices
	 * are supported
	 */
	if (dev->is_physfn) {
		if (test_sgtable1(tdev) != 0)
			goto test_fail;

		if (test_sgtable2(tdev) != 0)
			goto test_fail;

		if (test_sgtable3(tdev) != 0)
			goto test_fail;

		if (test_rgid(tdev) != 0)
			goto test_fail;

		if (map_kvec_err(tdev) != 0)
			goto test_fail;

		if (map_bvec_err(tdev) != 0)
			goto test_fail;

		if (map_kvec(tdev) != 0)
			goto test_fail;

		if (map_bvec(tdev) != 0)
			goto test_fail;

		if (map_test(tdev) != 0)
			goto test_fail;

		if (test_rma(tdev, false) != 0)
			goto test_fail;

		if (test_rma(tdev, true) != 0)
			goto test_fail;

		/* Repeat RMA test using EQ passthrough */
		tdev->eq_flags = CXI_EQ_PASSTHROUGH;
		if (test_rma(tdev, false) != 0)
			goto test_fail;

		tdev->eq_flags = 0;
		if (test_bvec_rma(tdev) != 0)
			goto test_fail;

		if (test_alloc_md(tdev) != 0)
			goto test_fail;

		if (test_atu(tdev) != 0)
			goto test_fail;
	}

	pr_info("Adding template client for device %s\n", dev->name);

	mutex_lock(&dev_list_mutex);
	list_add_tail(&tdev->dev_list, &dev_list);
	mutex_unlock(&dev_list_mutex);

	return 0;

test_fail:
	kfree(tdev);
	return -ENODEV;
}

static void remove_device(struct cxi_dev *dev)
{
	struct tdev *tdev;
	bool found = false;

	/* Find the device in the list */
	mutex_lock(&dev_list_mutex);
	list_for_each_entry_reverse(tdev, &dev_list, dev_list) {
		if (tdev->dev == dev) {
			found = true;
			list_del(&tdev->dev_list);
			break;
		}
	}
	mutex_unlock(&dev_list_mutex);

	if (!found)
		return;

	kfree(tdev);
}

static struct cxi_client cxiu_client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int ret;

	ret = cxi_register_client(&cxiu_client);
	if (ret) {
		pr_err("Couldn't register client\n");
		goto out;
	}

	return 0;

out:
	return ret;
}

static void __exit cleanup(void)
{
	pr_info("Removing template client\n");
	cxi_unregister_client(&cxiu_client);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Cray eXascale Interconnect (CXI) template driver");
MODULE_AUTHOR("Cray Inc.");
