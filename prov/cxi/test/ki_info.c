/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

#include <stdlib.h>
#include <string.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"
#include "cxip_ki_test_common.h"
#include "fi_cxi_gpu_ext.h"

/* Test the metadata the KI ops vtable publishes about an enabled endpoint:
 * command-queue and counter/EQ handles, the peer-info blob round trip,
 * PTL indices across the MR key space, target/MR addressing, and
 * argument validation.
 */

TestSuite(ki_info, .init = cxit_setup_ki_full, .fini = cxit_teardown_ki_full,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

TestSuite(ki_info_client_key, .init = cxit_setup_ki_full_client_key,
	  .fini = cxit_teardown_ki_full, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Check PTL indices, match bits, offset adjustment and restriction against
 * the registered MR and its assigned key.
 */
static void ki_info_check_remote_mr(struct fid_mr *fid,
				    const struct fi_cxi_gpu_remote_mr *remote)
{
	struct cxip_mr *mr = container_of(fid, struct cxip_mr, mr_fid);
	struct cxip_mr_key key = { .raw = fi_mr_key(fid) };
	uint16_t write_ptl_idx = CXIP_PTL_IDX_WRITE_MR_STD;
	uint16_t read_ptl_idx = CXIP_PTL_IDX_READ_MR_STD;
	uint16_t idx;
	uint64_t offset_adjust = 0;
	bool events;

	cr_assert_eq(!!key.is_prov,
		     !!(cxit_fi->domain_attr->mr_mode & FI_MR_PROV_KEY),
		     "key=0x%lx does not match domain key mode", key.raw);

	/* Compute portal indices from the assigned key rather than the
	 * provider's mapping helper. The test uses mr->optimized to choose
	 * optimized vs. standard indices, so it checks the index calculation,
	 * not the provider's choice of mode.
	 */
	if (mr->optimized) {
		if (!key.is_prov) {
			idx = key.raw;
		} else if (key.cached) {
			idx = key.lac;
		} else {
			idx = CXIP_PTL_IDX_PROV_NUM_CACHE_IDX + key.id;
		}
		write_ptl_idx = CXIP_PTL_IDX_WRITE_MR_OPT_BASE + idx;
		read_ptl_idx = CXIP_PTL_IDX_READ_MR_OPT_BASE + idx;
	}

	if (key.is_prov && key.cached)
		offset_adjust = CXI_VA_TO_IOVA(mr->md->md, mr->buf);

	events = key.is_prov ? (!key.cached && key.events) :
		 !!(cxit_fi->caps & FI_RMA_EVENT);

	cr_assert_eq(remote->write_ptl_idx, write_ptl_idx,
		     "key=0x%lx write_ptl_idx is %u, expected %u", key.raw,
		     remote->write_ptl_idx, write_ptl_idx);
	cr_assert_eq(remote->read_ptl_idx, read_ptl_idx,
		     "key=0x%lx read_ptl_idx is %u, expected %u", key.raw,
		     remote->read_ptl_idx, read_ptl_idx);
	cr_assert_eq(remote->match_bits, CXIP_KEY_MATCH_BITS(key.raw),
		     "key=0x%lx match_bits is 0x%lx", key.raw,
		     remote->match_bits);
	cr_assert_eq(remote->offset_adjust, offset_adjust,
		     "key=0x%lx offset_adjust is 0x%lx, expected 0x%lx",
		     key.raw, remote->offset_adjust, offset_adjust);
	cr_assert_eq(remote->unrestricted, !mr->optimized || events,
		     "key=0x%lx unrestricted is %u", key.raw,
		     remote->unrestricted);
}

/* Test that get_gpu_ep_info() populates every field */
Test(ki_info, ep_info_fields)
{
	int ret;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	struct cxip_domain *dom;
	uint32_t send_ctn;

	dom = container_of(cxit_domain, struct cxip_domain,
			   util_domain.domain_fid);

	ret = cxit_ki_ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info returned %d", ret);

	/* Host pointers echo back the hint buffers passed at setup. */
	cr_assert_eq(gpu_ep.tx_cmdq.hptr_buff, cxit_ki_full_tx_buf,
		     "tx_cmdq.hptr_buff mismatch");
	cr_assert_not_null(gpu_ep.tx_cmdq.hptr_mmio_db);
	cr_assert_eq(gpu_ep.trig_cmdq.hptr_buff, cxit_ki_full_trig_buf,
		     "trig_cmdq.hptr_buff mismatch");
	cr_assert_not_null(gpu_ep.trig_cmdq.hptr_mmio_db);

	/* Each ring's slot count matches the buffer size requested. */
	cr_assert_eq(gpu_ep.tx_cmdq.cq.size32, s_page_size / 32,
		     "tx_cmdq.cq.size32 is %u", gpu_ep.tx_cmdq.cq.size32);
	cr_assert_eq(gpu_ep.trig_cmdq.cq.size32, s_page_size / 32,
		     "trig_cmdq.cq.size32 is %u", gpu_ep.trig_cmdq.cq.size32);

	/* Neither ring is touched by get_gpu_ep_info(), so both reset to
	 * the documented first write pointer -- but the wp32 invariant
	 * documented in fi_cxi_gpu_ext.h (always even) must still hold
	 * for both.
	 */
	cr_assert_eq(gpu_ep.tx_cmdq.cq.wp32, C_CQ_FIRST_WR_PTR_32,
		     "tx_cmdq.cq.wp32 is %lu", gpu_ep.tx_cmdq.cq.wp32);
	cr_assert_eq(gpu_ep.tx_cmdq.cq.wp32 % 2, 0, "tx_cmdq.cq.wp32 is odd");
	cr_assert_eq(gpu_ep.trig_cmdq.cq.wp32, C_CQ_FIRST_WR_PTR_32,
		     "trig_cmdq.cq.wp32 is %lu", gpu_ep.trig_cmdq.cq.wp32);
	cr_assert_eq(gpu_ep.trig_cmdq.cq.wp32 % 2, 0,
		     "trig_cmdq.cq.wp32 is odd");

	cr_assert_eq(gpu_ep.pid_bits, dom->iface->info->pid_bits,
		     "pid_bits is %u", gpu_ep.pid_bits);

	cr_assert_neq(gpu_ep.eqn, C_EQ_NONE,
		      "eqn is C_EQ_NONE with a TX CQ bound");

	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &cxit_send_cntr->fid,
					   &send_ctn);
	cr_assert_eq(ret, FI_SUCCESS, "get_cntr_number returned %d", ret);
	cr_assert_eq(gpu_ep.ctn, send_ctn,
		     "ctn (%u) != bound send counter (%u)", gpu_ep.ctn,
		     send_ctn);

	cr_assert_neq(gpu_ep.user_ptr, 0, "user_ptr is zero");
}

/* Test that repeated get_gpu_ep_info() calls return the same user_ptr.
 * KI operations share one error-reporting request per endpoint.
 */
Test(ki_info, user_ptr_stable)
{
	int ret;
	struct fi_cxi_gpu_ep_info first = {};
	struct fi_cxi_gpu_ep_info second = {};

	ret = cxit_ki_ops->get_gpu_ep_info(&cxit_ep->fid, &first);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info(1) returned %d", ret);

	ret = cxit_ki_ops->get_gpu_ep_info(&cxit_ep->fid, &second);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info(2) returned %d", ret);

	cr_assert_neq(first.user_ptr, 0, "user_ptr is zero");
	cr_assert_eq(first.user_ptr, second.user_ptr,
		     "user_ptr changed: 0x%lx -> 0x%lx", first.user_ptr,
		     second.user_ptr);
}

/* Test the host-side inline helpers in fi_cxi_gpu_ext.h on both command
 * queues: get_db returns the host doorbell, set_db stores the device
 * doorbell in dptr_mmio_db and cq.wp_addr, set_buf stores the device
 * buffer in dptr_buff and cq.cmds32.
 */
Test(ki_info, host_helpers)
{
	int ret;
	int i;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	struct fi_cxi_gpu_cmdq *cmdq;
	enum fi_cxi_ki_cmdq_type types[] = { FI_CXI_KI_TXC, FI_CXI_KI_TRIGC };
	uint64_t dev_db_storage[2];
	uint8_t dev_buf_storage[2];
	uint64_t *dev_db;
	void *dev_buf;
	uint64_t *db;

	ret = cxit_ki_ops->get_gpu_ep_info(&cxit_ep->fid, &gpu_ep);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info returned %d", ret);

	for (i = 0; i < 2; i++) {
		cmdq = (types[i] == FI_CXI_KI_TXC) ?
			&gpu_ep.tx_cmdq : &gpu_ep.trig_cmdq;
		dev_db = &dev_db_storage[i];
		dev_buf = &dev_buf_storage[i];
		db = NULL;

		fi_cxi_ki_cmdq_get_db(&gpu_ep, types[i], &db);
		cr_assert_eq(db, cmdq->hptr_mmio_db,
			     "type %d: get_db returned %p, hptr_mmio_db %p",
			     types[i], (void *)db, (void *)cmdq->hptr_mmio_db);

		fi_cxi_ki_cmdq_set_db(&gpu_ep, types[i], dev_db);
		cr_assert_eq(cmdq->dptr_mmio_db, dev_db,
			     "type %d: dptr_mmio_db not set", types[i]);
		cr_assert_eq(cmdq->cq.wp_addr, dev_db,
			     "type %d: cq.wp_addr not set", types[i]);

		fi_cxi_ki_cmdq_set_buf(&gpu_ep, types[i], dev_buf);
		cr_assert_eq(cmdq->dptr_buff, dev_buf,
			     "type %d: dptr_buff not set", types[i]);
		cr_assert_eq((void *)cmdq->cq.cmds32, dev_buf,
			     "type %d: cq.cmds32 not set", types[i]);

		/* Host pointers are untouched by the device setters. */
		cr_assert_eq(cmdq->hptr_mmio_db, db,
			     "type %d: hptr_mmio_db changed", types[i]);
	}
}

/* Test the peer-info blob round trip: get_peer_info() decoded via
 * get_gpu_remote_mrs() publishes the MR's IOVA, the assigned key's PTL
 * indices, and the key-derived match bits, offset adjustment and
 * restriction, and get_gpu_mr_info() reports the same IOVA for the
 * buffer start.
 */
Test(ki_info, peer_info_roundtrip)
{
	int ret;
	struct fi_cxi_gpu_remote_mr remote = {};
	struct fi_cxi_gpu_mr_info local_info = {};
	struct mem_region mem_window;
	struct cxi_md *md;
	uint64_t key = 0;

	cxit_ki_create_remote_mr(s_page_size, FI_REMOTE_READ | FI_REMOTE_WRITE,
				 &key, &mem_window, NULL, &remote);
	md = container_of(mem_window.mr, struct cxip_mr, mr_fid)->md->md;

	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mem_window.mr->fid,
					   mem_window.mem, &local_info);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_mr_info returned %d", ret);

	cr_assert_eq(remote.iova, CXI_VA_TO_IOVA(md, mem_window.mem),
		     "decoded iova (0x%lx) != MR iova (0x%lx)", remote.iova,
		     CXI_VA_TO_IOVA(md, mem_window.mem));
	cr_assert_eq(local_info.iova, remote.iova,
		     "local iova (0x%lx) != decoded iova (0x%lx)",
		     local_info.iova, remote.iova);

	ki_info_check_remote_mr(mem_window.mr, &remote);

	mr_destroy(&mem_window);
}

/* Test that get_gpu_remote_mrs() rejects a blob with a bad magic or
 * version byte, as from a peer running an incompatible provider.
 */
Test(ki_info, bad_peer_blob)
{
	int ret;
	struct fi_cxi_ki_peer_info peers[2] = {};
	struct fi_cxi_gpu_remote_mr remotes[2] = {};
	struct mem_region mem_window;
	uint64_t key = 0;

	cxit_ki_create_remote_mr(s_page_size, FI_REMOTE_READ | FI_REMOTE_WRITE,
				 &key, &mem_window, &peers[0], &remotes[0]);
	peers[1] = peers[0];

	/* Byte 0 is the magic, byte 1 the version */
	peers[1].opaque[0] ^= 0xff;
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, peers, 2, remotes);
	cr_assert_eq(ret, -FI_EINVAL, "bad magic returned %d", ret);

	peers[1] = peers[0];
	peers[1].opaque[1] += 1;
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, peers, 2, remotes);
	cr_assert_eq(ret, -FI_EINVAL, "bad version returned %d", ret);

	peers[1] = peers[0];
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, peers, 2, remotes);
	cr_assert_eq(ret, FI_SUCCESS, "intact blobs returned %d", ret);

	mr_destroy(&mem_window);
}

/* Test PTL indices and restriction across the client MR key space: optimized
 * keys map to BASE + key and allow restricted commands, keys at and
 * past CXIP_PTL_IDX_MR_OPT_CNT map to the fixed standard indices and
 * require unrestricted commands. Key 100's optimized value (17 + 100)
 * happens to equal CXIP_PTL_IDX_WRITE_MR_STD, so key 1000 distinguishes
 * the two paths. Expected values assume the default
 * FI_CXI_OPTIMIZED_MRS=1.
 */
Test(ki_info_client_key, ptl_idx_key_sweep)
{
	int i;
	struct fi_cxi_gpu_remote_mr remote;
	struct mem_region mem_window;
	uint64_t keys[] = { 0, 15, CXIP_PTL_IDX_MR_OPT_CNT - 1,
			    CXIP_PTL_IDX_MR_OPT_CNT, 1000 };
	uint64_t key;
	uint16_t write_ptl_idx;
	uint16_t read_ptl_idx;

	for (i = 0; i < ARRAY_SIZE(keys); i++) {
		key = keys[i];
		if (key < CXIP_PTL_IDX_MR_OPT_CNT) {
			write_ptl_idx = CXIP_PTL_IDX_WRITE_MR_OPT(key);
			read_ptl_idx = CXIP_PTL_IDX_READ_MR_OPT(key);
		} else {
			write_ptl_idx = CXIP_PTL_IDX_WRITE_MR_STD;
			read_ptl_idx = CXIP_PTL_IDX_READ_MR_STD;
		}

		cxit_ki_create_remote_mr(s_page_size,
					 FI_REMOTE_READ | FI_REMOTE_WRITE, &key,
					 &mem_window, NULL, &remote);

		cr_assert_eq(key, keys[i], "requested key %lu became 0x%lx",
			     keys[i], key);
		cr_assert_eq(remote.write_ptl_idx, write_ptl_idx,
			     "key=%lu write_ptl_idx is %u", key,
			     remote.write_ptl_idx);
		cr_assert_eq(remote.read_ptl_idx, read_ptl_idx,
			     "key=%lu read_ptl_idx is %u", key,
			     remote.read_ptl_idx);
		cr_assert_eq(remote.match_bits, CXIP_KEY_MATCH_BITS(key),
			     "key=%lu match_bits is 0x%lx", key,
			     remote.match_bits);
		cr_assert_eq(remote.unrestricted,
			     key >= CXIP_PTL_IDX_MR_OPT_CNT,
			     "key=%lu unrestricted is %u", key,
			     remote.unrestricted);

		mr_destroy(&mem_window);
	}
}

/* Test that a FI_REMOTE_READ-only MR still publishes both PTL indices;
 * index publication depends on the key, not the access flags.
 */
Test(ki_info, mr_access_indices)
{
	struct fi_cxi_gpu_remote_mr remote = {};
	struct mem_region mem_window;
	uint64_t key = 0;

	cxit_ki_create_remote_mr(s_page_size, FI_REMOTE_READ, &key,
				 &mem_window, NULL, &remote);

	ki_info_check_remote_mr(mem_window.mr, &remote);

	mr_destroy(&mem_window);
}

/* Test that get_peer_info() rejects an MR that is not enabled. With
 * provider keys the key, and so the portal indexes, are assigned at
 * enable; an MR without remote access is never enabled.
 */
Test(ki_info, peer_info_not_enabled)
{
	int ret;
	struct fi_cxi_ki_peer_info peer = {};
	struct fid_mr *mr;
	uint8_t *buf;

	buf = cxit_ki_page_alloc(s_page_size);

	/* Registered and bound but not enabled */
	ret = fi_mr_reg(cxit_domain, buf, s_page_size,
			FI_REMOTE_READ | FI_REMOTE_WRITE, 0, 0, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg returned %d", ret);

	ret = fi_mr_bind(mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind returned %d", ret);

	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &mr->fid, &peer);
	cr_assert_eq(ret, -FI_EOPBADSTATE,
		     "get_peer_info before enable returned %d", ret);

	ret = fi_mr_enable(mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable returned %d", ret);

	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &mr->fid, &peer);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_peer_info after enable returned %d", ret);

	fi_close(&mr->fid);

	/* Local-only access: enable is a no-op, so never enabled */
	ret = fi_mr_reg(cxit_domain, buf, s_page_size, FI_READ | FI_WRITE,
			0, 0, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg returned %d", ret);

	ret = fi_mr_bind(mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind returned %d", ret);

	ret = fi_mr_enable(mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable returned %d", ret);

	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &mr->fid, &peer);
	cr_assert_eq(ret, -FI_EOPBADSTATE,
		     "get_peer_info on local MR returned %d", ret);

	fi_close(&mr->fid);
	cxit_ki_page_free(buf);
}

/* Test that get_peer_info() rejects an MR that is not bound to the
 * given EP.
 */
Test(ki_info, peer_info_not_bound)
{
	int ret;
	struct fi_cxi_ki_peer_info peer = {};
	struct fid_mr *mr;
	uint8_t *buf;

	buf = cxit_ki_page_alloc(s_page_size);

	/* Registered but not bound to any EP */
	ret = fi_mr_reg(cxit_domain, buf, s_page_size,
			FI_REMOTE_READ | FI_REMOTE_WRITE, 0, 0, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg returned %d", ret);

	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &mr->fid, &peer);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_peer_info on unbound MR returned %d", ret);

	ret = fi_mr_bind(mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind returned %d", ret);

	ret = fi_mr_enable(mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable returned %d", ret);

	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &mr->fid, &peer);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_peer_info on bound MR returned %d", ret);

	fi_close(&mr->fid);
	cxit_ki_page_free(buf);
}

/* Test that get_gpu_mr_info() and get_cntr_number() accept an MR and
 * counter from the EP's domain that are not bound to the EP; only the
 * domain matters for an IOVA, LAC or counter number.
 */
Test(ki_info, domain_objects_unbound)
{
	int ret;
	struct fi_cxi_gpu_mr_info info = {};
	struct fid_mr *mr;
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = { .events = FI_CNTR_EVENTS_COMP };
	uint32_t ctn;
	uint8_t *buf;

	buf = cxit_ki_page_alloc(s_page_size);

	ret = fi_mr_reg(cxit_domain, buf, s_page_size,
			FI_REMOTE_READ | FI_REMOTE_WRITE, 0, 0, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg returned %d", ret);

	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, buf,
					   &info);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_gpu_mr_info on unbound MR returned %d", ret);

	fi_close(&mr->fid);

	ret = fi_cntr_open(cxit_domain, &cntr_attr, &cntr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_open returned %d", ret);

	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &cntr->fid, &ctn);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_cntr_number on unbound counter returned %d", ret);

	fi_close(&cntr->fid);
	cxit_ki_page_free(buf);
}

/* Test that get_gpu_mr_info() rejects an MR with no mapping. A
 * zero-length MR registers without mapping its buffer.
 */
Test(ki_info, mr_info_not_mapped)
{
	int ret;
	struct fi_cxi_gpu_mr_info info = {};
	struct fid_mr *mr;
	uint8_t *buf;

	buf = cxit_ki_page_alloc(s_page_size);

	ret = fi_mr_reg(cxit_domain, buf, 0, FI_REMOTE_READ | FI_REMOTE_WRITE,
			0, 0, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg returned %d", ret);

	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, buf,
					   &info);
	cr_assert_eq(ret, -FI_EOPBADSTATE,
		     "get_gpu_mr_info on unmapped MR returned %d", ret);

	fi_close(&mr->fid);
	cxit_ki_page_free(buf);
}

/* Test get_gpu_target_info() on the endpoint's own address, and that a
 * repeated address fills every requested entry.
 */
Test(ki_info, target_info_self)
{
	int ret;
	int i;
	struct fi_cxi_gpu_target_info targets[3] = {};
	fi_addr_t addrs[3];
	fi_addr_t addr;

	ret = fi_av_insert(cxit_av, &cxit_ep_addr, 1, &addr, 0, NULL);
	cr_assert_eq(ret, 1, "fi_av_insert returned %d", ret);

	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, &addr, 1,
					       &targets[0]);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_target_info returned %d", ret);

	cr_assert_eq(targets[0].nid, cxit_ep_addr.nic,
		     "nid is %u", targets[0].nid);
	cr_assert_eq(targets[0].pid, cxit_ep_addr.pid,
		     "pid is %u", targets[0].pid);

	/* Repeat the same address 3 times; every entry must be filled. */
	for (i = 0; i < 3; i++)
		addrs[i] = addr;
	memset(targets, 0, sizeof(targets));

	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, addrs, 3,
					       targets);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_target_info returned %d", ret);

	for (i = 0; i < 3; i++) {
		cr_assert_eq(targets[i].nid, cxit_ep_addr.nic,
			     "targets[%d].nid is %u", i, targets[i].nid);
		cr_assert_eq(targets[i].pid, cxit_ep_addr.pid,
			     "targets[%d].pid is %u", i, targets[i].pid);
	}
}

/* Test that an fi_addr_t never inserted into the AV is rejected */
Test(ki_info, target_info_bad_addr)
{
	int ret;
	struct fi_cxi_gpu_target_info target = {};
	fi_addr_t bad_addr = 1;

	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, &bad_addr, 1,
					       &target);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_target_info returned %d", ret);
}

/* Test that get_gpu_mr_info() echoes back the buf pointer passed and
 * reports the IOVA of that pointer, so iova + offset addresses buf +
 * offset, and that a pointer outside the MR is rejected.
 */
Test(ki_info, mr_info_offsets)
{
	int ret;
	struct fi_cxi_gpu_mr_info base = {};
	struct fi_cxi_gpu_mr_info info = {};
	struct mem_region mem_window;
	uint64_t key = 0;
	struct fid_mr *mr;
	uint8_t *buf;

	ret = mr_create(s_page_size, FI_REMOTE_READ | FI_REMOTE_WRITE, 0, &key,
			&mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create returned %d", ret);
	mr = mem_window.mr;
	buf = mem_window.mem;

	/* Exact base pointer. */
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, buf,
					   &base);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_mr_info(base) returned %d",
		     ret);
	cr_assert_eq(base.buf, buf, "get_gpu_mr_info(base) echoed %p",
		     base.buf);

	/* An interior pointer is echoed back and its iova advances by the
	 * same amount.
	 */
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, buf + 64,
					   &info);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_gpu_mr_info(interior) returned %d", ret);
	cr_assert_eq(info.buf, buf + 64,
		     "get_gpu_mr_info(interior) echoed %p", info.buf);
	cr_assert_eq(info.iova, base.iova + 64,
		     "get_gpu_mr_info(interior) iova 0x%lx != 0x%lx",
		     info.iova, base.iova + 64);
	cr_assert_eq(info.lac, base.lac,
		     "get_gpu_mr_info(interior) lac %u != %u", info.lac,
		     base.lac);

	/* The last byte of the MR is inside it; one past the end is not. */
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid,
					   buf + s_page_size - 1, &info);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_gpu_mr_info(last byte) returned %d", ret);
	cr_assert_eq(info.iova, base.iova + s_page_size - 1,
		     "get_gpu_mr_info(last byte) iova 0x%lx != 0x%lx",
		     info.iova, base.iova + s_page_size - 1);

	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid,
					   buf + s_page_size, &info);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_mr_info(past end) returned %d", ret);

	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, buf - 1,
					   &info);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_mr_info(before start) returned %d", ret);

	mr_destroy(&mem_window);
}

/* Test that distinct counters have distinct CT numbers */
Test(ki_info, cntr_number_distinct)
{
	int ret;
	struct fid_cntr *extra_cntr;
	uint32_t send_ctn;
	uint32_t extra_ctn;

	ret = fi_cntr_open(cxit_domain, NULL, &extra_cntr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_open returned %d", ret);

	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &cxit_send_cntr->fid,
					   &send_ctn);
	cr_assert_eq(ret, FI_SUCCESS, "get_cntr_number(send) returned %d",
		     ret);

	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &extra_cntr->fid,
					   &extra_ctn);
	cr_assert_eq(ret, FI_SUCCESS, "get_cntr_number(extra) returned %d",
		     ret);

	cr_assert_neq(send_ctn, extra_ctn,
		      "send_ctn (%u) == extra_ctn (%u)", send_ctn, extra_ctn);

	fi_close(&extra_cntr->fid);
}

/* Test that get_gpu_remote_mrs() decodes every entry of a multi-element
 * peer-info array.
 */
Test(ki_info, remote_mrs_array)
{
	int ret;
	int i;
	struct fi_cxi_ki_peer_info peers[3] = {};
	struct fi_cxi_gpu_remote_mr remotes[3] = {};
	struct fi_cxi_gpu_mr_info local[3] = {};
	struct mem_region mem_windows[3];
	struct cxi_md *md;
	uint64_t key;

	for (i = 0; i < 3; i++) {
		key = i;
		cxit_ki_create_remote_mr(s_page_size,
					 FI_REMOTE_READ | FI_REMOTE_WRITE, &key,
					 &mem_windows[i], &peers[i],
					 &remotes[i]);

		ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid,
						   &mem_windows[i].mr->fid,
						   mem_windows[i].mem,
						   &local[i]);
		cr_assert_eq(ret, FI_SUCCESS,
			     "get_gpu_mr_info(%d) returned %d", i, ret);
	}

	/* Decode all three blobs in one call */
	memset(remotes, 0, sizeof(remotes));
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, peers, 3,
					      remotes);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_remote_mrs returned %d", ret);

	for (i = 0; i < 3; i++) {
		md = container_of(mem_windows[i].mr, struct cxip_mr,
				  mr_fid)->md->md;
		cr_assert_eq(remotes[i].iova,
			     CXI_VA_TO_IOVA(md, mem_windows[i].mem),
			     "remotes[%d].iova 0x%lx != MR iova 0x%lx", i,
			     remotes[i].iova,
			     CXI_VA_TO_IOVA(md, mem_windows[i].mem));
		cr_assert_eq(local[i].iova, remotes[i].iova,
			     "local[%d].iova 0x%lx != decoded 0x%lx", i,
			     local[i].iova, remotes[i].iova);
		ki_info_check_remote_mr(mem_windows[i].mr, &remotes[i]);
	}

	for (i = 0; i < 3; i++)
		mr_destroy(&mem_windows[i]);
}

/* Test NULL pointer and non-positive count failure cases for every op */
Test(ki_info, null_args)
{
	int ret;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	struct fi_cxi_ki_peer_info peer = {};
	struct fi_cxi_gpu_target_info target = {};
	struct fi_cxi_gpu_remote_mr remote = {};
	struct fi_cxi_gpu_mr_info info = {};
	struct mem_region mem_window;
	uint64_t key = 0;
	struct fid_mr *mr;
	fi_addr_t addr;
	uint32_t ctn;
	void *buf;

	ret = mr_create(s_page_size, FI_REMOTE_READ | FI_REMOTE_WRITE, 0, &key,
			&mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create returned %d", ret);
	mr = mem_window.mr;
	buf = mem_window.mem;

	ret = fi_av_insert(cxit_av, &cxit_ep_addr, 1, &addr, 0, NULL);
	cr_assert_eq(ret, 1, "fi_av_insert returned %d", ret);

	/* get_gpu_ep_info */
	ret = cxit_ki_ops->get_gpu_ep_info(NULL, &gpu_ep);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_ep_info(NULL fid) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_ep_info(&cxit_ep->fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_ep_info(NULL out) returned %d",
		     ret);

	/* get_peer_info */
	ret = cxit_ki_ops->get_peer_info(NULL, &mr->fid, &peer);
	cr_assert_eq(ret, -FI_EINVAL, "get_peer_info(NULL ep) returned %d",
		     ret);
	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, NULL, &peer);
	cr_assert_eq(ret, -FI_EINVAL, "get_peer_info(NULL fid) returned %d",
		     ret);
	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &mr->fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "get_peer_info(NULL out) returned %d",
		     ret);

	/* get_gpu_target_info */
	ret = cxit_ki_ops->get_gpu_target_info(NULL, &addr, 1, &target);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_target_info(NULL fid) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, NULL, 1, &target);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_target_info(NULL addrs) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, &addr, 1, NULL);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_target_info(NULL targets) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, &addr, 0,
					       &target);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_target_info(npes=0) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, &addr, -1,
					       &target);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_target_info(npes=-1) returned %d", ret);

	/* get_gpu_remote_mrs */
	ret = cxit_ki_ops->get_gpu_remote_mrs(NULL, &peer, 1, &remote);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_remote_mrs(NULL fid) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, NULL, 1, &remote);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_remote_mrs(NULL peers) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, &peer, 1, NULL);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_remote_mrs(NULL out) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, &peer, 0, &remote);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_remote_mrs(npes=0) returned %d", ret);

	/* get_gpu_mr_info */
	ret = cxit_ki_ops->get_gpu_mr_info(NULL, &mr->fid, buf, &info);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_mr_info(NULL ep) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, NULL, buf, &info);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_mr_info(NULL fid) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, NULL,
					   &info);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_mr_info(NULL buf) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &mr->fid, buf, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_mr_info(NULL out) returned %d",
		     ret);

	/* get_cntr_number */
	ret = cxit_ki_ops->get_cntr_number(NULL, &cxit_send_cntr->fid, &ctn);
	cr_assert_eq(ret, -FI_EINVAL, "get_cntr_number(NULL ep) returned %d",
		     ret);
	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, NULL, &ctn);
	cr_assert_eq(ret, -FI_EINVAL, "get_cntr_number(NULL fid) returned %d",
		     ret);
	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &cxit_send_cntr->fid,
					   NULL);
	cr_assert_eq(ret, -FI_EINVAL, "get_cntr_number(NULL out) returned %d",
		     ret);

	mr_destroy(&mem_window);
}

/* Test wrong FID class failure cases for every op */
Test(ki_info, wrong_fclass)
{
	int ret;
	struct fi_cxi_gpu_ep_info gpu_ep = {};
	struct fi_cxi_ki_peer_info peer = {};
	struct fi_cxi_gpu_target_info target = {};
	struct fi_cxi_gpu_remote_mr remote = {};
	struct fi_cxi_gpu_mr_info info = {};
	struct mem_region mem_window;
	uint64_t key = 0;
	struct fid_mr *mr;
	fi_addr_t addr;
	uint32_t ctn;
	void *buf;

	ret = mr_create(s_page_size, FI_REMOTE_READ | FI_REMOTE_WRITE, 0, &key,
			&mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create returned %d", ret);
	mr = mem_window.mr;
	buf = mem_window.mem;

	ret = fi_av_insert(cxit_av, &cxit_ep_addr, 1, &addr, 0, NULL);
	cr_assert_eq(ret, 1, "fi_av_insert returned %d", ret);

	/* EP-class entries given an MR or counter FID. */
	ret = cxit_ki_ops->get_gpu_ep_info(&mr->fid, &gpu_ep);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_ep_info(MR fid) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_target_info(&mr->fid, &addr, 1, &target);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_target_info(MR fid) returned %d", ret);
	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_send_cntr->fid, &peer, 1,
					      &remote);
	cr_assert_eq(ret, -FI_EINVAL,
		     "get_gpu_remote_mrs(cntr fid) returned %d", ret);

	/* MR-class entries given an EP FID. */
	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &cxit_ep->fid, &peer);
	cr_assert_eq(ret, -FI_EINVAL, "get_peer_info(EP fid) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid, &cxit_ep->fid, buf,
					   &info);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_mr_info(EP fid) returned %d",
		     ret);

	/* Counter-class entry given an EP FID. */
	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &cxit_ep->fid, &ctn);
	cr_assert_eq(ret, -FI_EINVAL, "get_cntr_number(EP fid) returned %d",
		     ret);

	/* EP slot given an MR FID. */
	ret = cxit_ki_ops->get_peer_info(&mr->fid, &mr->fid, &peer);
	cr_assert_eq(ret, -FI_EINVAL, "get_peer_info(MR ep) returned %d",
		     ret);
	ret = cxit_ki_ops->get_gpu_mr_info(&mr->fid, &mr->fid, buf, &info);
	cr_assert_eq(ret, -FI_EINVAL, "get_gpu_mr_info(MR ep) returned %d",
		     ret);
	ret = cxit_ki_ops->get_cntr_number(&mr->fid, &cxit_send_cntr->fid,
					   &ctn);
	cr_assert_eq(ret, -FI_EINVAL, "get_cntr_number(MR ep) returned %d",
		     ret);

	mr_destroy(&mem_window);
}
