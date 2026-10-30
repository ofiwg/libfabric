/*
 * Copyright (c) 2023,2026 Hewlett Packard Enterprise Development LP
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 */
#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_DOMAIN, __VA_ARGS__)

enum cxi_traffic_class cxip_ofi_to_cxi_tc(uint32_t ofi_tclass)
{
	switch (ofi_tclass) {
	case FI_TC_BULK_DATA:
		return CXI_TC_BULK_DATA;
	case FI_TC_DEDICATED_ACCESS:
		return CXI_TC_DEDICATED_ACCESS;
	case FI_TC_LOW_LATENCY:
		return CXI_TC_LOW_LATENCY;
	case FI_TC_BEST_EFFORT:
	case FI_TC_NETWORK_CTRL:
	case FI_TC_SCAVENGER:
	default:
		return CXI_TC_BEST_EFFORT;
	}
}

/* Triggered CP separation (LCID 0 reserved for triggered CQ/cntr use) is only
 * in effect when cxil_alloc_trig_cp() is available in the runtime libcxi and
 * the kernel driver has not rejected it (see cxip_alloc_cp_hw()).
 */
static bool cxip_trig_cp_active(struct cxip_lni *lni)
{
	return cxip_libcxi_ops.alloc_trig_cp && !lni->trig_cp_unsupported;
}

static bool cxip_cp_lcid_match(struct cxip_lni *lni, unsigned int lcid,
			       bool trig_cp)
{
	if (cxip_trig_cp_active(lni))
		return !trig_cp ? lcid != 0 : lcid == 0;

	return true;
}

static int cxip_alloc_cp_hw(struct cxip_lni *lni, uint16_t vni,
			    enum cxi_traffic_class tc,
			    enum cxi_traffic_class_type tc_type,
			    bool trig_cp, struct cxi_cp **cp)
{
	int ret;

	if (cxip_trig_cp_active(lni)) {
		ret = cxip_libcxi_ops.alloc_trig_cp(lni->lni, vni, tc, tc_type,
						    trig_cp ? TRIG_LCID :
							      NON_TRIG_LCID,
						    cp);
		if (ret != -EINVAL)
			return ret;

		ret = cxil_alloc_cp(lni->lni, vni, tc, tc_type, cp);
		if (!ret) {
			CXIP_INFO("cxil_alloc_trig_cp unsupported by driver; "
				  "falling back to cxil_alloc_cp\n");
			lni->trig_cp_unsupported = true;
		}
		return ret;
	}

	return cxil_alloc_cp(lni->lni, vni, tc, tc_type, cp);
}

static int cxip_cp_find(struct cxip_lni *lni, uint16_t vni,
		       enum cxi_traffic_class tc,
		       enum cxi_traffic_class_type tc_type,
		       struct cxi_cp **cp, bool trig_cp)
{
	struct cxip_remap_cp *sw_cp;

	dlist_foreach_container(&lni->remap_cps, struct cxip_remap_cp, sw_cp,
				remap_entry) {
		/* LCID 0 is reserved for triggered CQ/cntr use when triggered
		 * CP allocation is available and supported.
		 */
		if (sw_cp->remap_cp.vni == vni &&
		    sw_cp->remap_cp.tc == tc &&
		    sw_cp->remap_cp.tc_type == tc_type &&
		    cxip_cp_lcid_match(lni, sw_cp->remap_cp.lcid, trig_cp)) {
			CXIP_INFO("Reusing SW lcid:%u VNI:%u TC:%s TYPE:%s trig_cp:%d\n",
				 sw_cp->remap_cp.lcid, sw_cp->remap_cp.vni,
				 cxi_tc_to_str(sw_cp->remap_cp.tc),
				 cxi_tc_type_to_str(sw_cp->remap_cp.tc_type),
				 trig_cp);
			*cp = &sw_cp->remap_cp;
			return FI_SUCCESS;
		}
	}

	return -FI_ENOENT;
}

static int cxip_cp_get(struct cxip_lni *lni, uint16_t vni,
		       enum cxi_traffic_class tc,
		       enum cxi_traffic_class_type tc_type,
		       struct cxi_cp **cp, bool trig_cp)
{
	int ret;
	int i;
	struct cxip_remap_cp *sw_cp;
	static const enum cxi_traffic_class remap_tc = CXI_TC_BEST_EFFORT;

	/* Always prefer SW remapped CPs over allocating HW CP. */
	pthread_rwlock_rdlock(&lni->cp_lock);
	ret = cxip_cp_find(lni, vni, tc, tc_type, cp, trig_cp);
	pthread_rwlock_unlock(&lni->cp_lock);

	if (ret == FI_SUCCESS)
		return FI_SUCCESS;

	/* Need to repeat search with write lock held to ensure no CPs have
	 * been added in threaded env.
	 */
	pthread_rwlock_wrlock(&lni->cp_lock);
	ret = cxip_cp_find(lni, vni, tc, tc_type, cp, trig_cp);

	if (ret == FI_SUCCESS)
		goto success_unlock;

	if (lni->n_cps >= MAX_HW_CPS) {
		CXIP_WARN("No room to allocate CP for VNI:%u\n", vni);
		ret = -FI_ENOSPC;
		goto err_unlock;
	}

	/* Allocate a new SW remapped CP entry and attempt to allocate the
	 * user requested HW CP.
	 */
	sw_cp = calloc(1, sizeof(*sw_cp));
	if (!sw_cp) {
		ret = -FI_ENOMEM;
		goto err_unlock;
	}

	ret = cxip_alloc_cp_hw(lni, vni, tc, tc_type, trig_cp,
			       &lni->hw_cps[lni->n_cps]);
	if (ret) {
		/* Attempt to fall back to remap traffic class with the same
		 * traffic class type and allocate HW CP if necessary.
		 */
		CXIP_WARN("Failed to allocate CP, ret: %d VNI: %u TC: %s TYPE: %s\n",
			  ret, vni, cxi_tc_to_str(tc),
			  cxi_tc_type_to_str(tc_type));
		CXIP_WARN("Remapping original TC from %s to %s\n",
			  cxi_tc_to_str(tc), cxi_tc_to_str(remap_tc));

		/* Check to see if a matching HW CP has already been allocated.
		 * If so, reuse the entry.
		 */
		for (i = 0; i < lni->n_cps; i++) {
			if (lni->hw_cps[i]->vni == vni &&
			    lni->hw_cps[i]->tc == remap_tc &&
			    lni->hw_cps[i]->tc_type == tc_type &&
			    cxip_cp_lcid_match(lni, lni->hw_cps[i]->lcid,
					       trig_cp)) {
				sw_cp->hw_cp = lni->hw_cps[i];
				goto found_hw_cp;
			}
		}

		/* Attempt to allocated a remapped HW CP. */
		ret = cxip_alloc_cp_hw(lni, vni, remap_tc, tc_type, trig_cp,
				       &lni->hw_cps[lni->n_cps]);
		if (ret) {
			CXIP_WARN("Failed to allocate CP, ret: %d VNI: %u TC: %s TYPE: %s\n",
				  ret, vni, cxi_tc_to_str(remap_tc),
				  cxi_tc_type_to_str(tc_type));
			ret = -FI_EINVAL;
			goto err_free_sw_cp;
		}
	}

	if (trig_cp && lni->hw_cps[lni->n_cps]->lcid)
		CXIP_WARN("Triggered CQ requires CP with LCID=0, LCID is %d",
			  lni->hw_cps[lni->n_cps]->lcid);

	CXIP_DBG("Allocated CP: %u VNI: %u TC: %s TYPE: %s\n",
		 lni->hw_cps[lni->n_cps]->lcid, vni,
		 cxi_tc_to_str(lni->hw_cps[lni->n_cps]->tc),
		 cxi_tc_type_to_str(lni->hw_cps[lni->n_cps]->tc_type));

	sw_cp->hw_cp = lni->hw_cps[lni->n_cps++];

found_hw_cp:
	sw_cp->remap_cp.vni = vni;
	sw_cp->remap_cp.tc = tc;
	sw_cp->remap_cp.tc_type = tc_type;
	sw_cp->remap_cp.lcid = sw_cp->hw_cp->lcid;
	dlist_insert_tail(&sw_cp->remap_entry, &lni->remap_cps);

	*cp = &sw_cp->remap_cp;

success_unlock:
	pthread_rwlock_unlock(&lni->cp_lock);

	return FI_SUCCESS;

err_free_sw_cp:
	free(sw_cp);
err_unlock:
	pthread_rwlock_unlock(&lni->cp_lock);

	return ret;
}

int cxip_cmdq_cp_set(struct cxip_cmdq *cmdq, uint16_t vni,
		     enum cxi_traffic_class tc,
		     enum cxi_traffic_class_type tc_type)
{
	struct cxi_cp *cp;
	int ret;

	if (cxip_cmdq_match(cmdq, vni, tc, tc_type))
		return FI_SUCCESS;

	if (cxip_cmdq_prev_match(cmdq, vni, tc, tc_type)) {
		cp = cmdq->prev_cp;
	} else {
		ret = cxip_cp_get(cmdq->lni, vni, tc, tc_type, &cp, false);
		if (ret != FI_SUCCESS) {
			CXIP_DBG("Failed to get CP: %d\n", ret);
			return -FI_EOTHER;
		}
	}

	ret = cxi_cq_emit_cq_lcid(cmdq->dev_cmdq, cp->lcid);
	if (ret) {
		CXIP_DBG("Failed to update CMDQ(%p) CP: %d\n", cmdq, ret);
		cxi_cq_ring(cmdq->dev_cmdq);
		ret = -FI_EAGAIN;
	} else {
		ret = FI_SUCCESS;
		cmdq->prev_cp = cmdq->cur_cp;
		cmdq->cur_cp = cp;

		CXIP_DBG("Updated CMDQ(%p) CP: %d VNI: %u TC: %s TYPE: %s\n",
			 cmdq, cp->lcid, cp->vni, cxi_tc_to_str(cp->tc),
			 cxi_tc_type_to_str(cp->tc_type));
	}

	return ret;
}

int cxip_cmdq_cp_modify(struct cxip_cmdq *cmdq, uint16_t vni,
			enum cxi_traffic_class tc)
{
	int ret;
	int i;
	struct cxi_cp *hw_cp = NULL;
	struct cxip_lni *lni = cmdq->lni;

	/* CXI_OP_CP_MODIFY was introduced by the same kernel ABI revision as
	 * CXI_OP_TRIG_CP_ALLOC and every GA driver ships the two together,
	 * and a transmit command queue cannot exist before its LNI's first HW
	 * CP allocation decided trig_cp_unsupported. Treat a driver that
	 * rejected triggered CP allocation as lacking CP modification and use
	 * the legacy CP set path there; a pre-GA build that split the two
	 * opcodes merely takes the legacy path as well.
	 */
	if (!cxip_libcxi_ops.modify_cp || !cxip_trig_cp_active(lni))
		goto cp_set;

	pthread_rwlock_wrlock(&lni->cp_lock);

	/* find the hw CP that matches the sw CP */
	for (i = 0; i < lni->n_cps; i++) {
		if (lni->hw_cps[i]->vni == cmdq->cur_cp->vni_pcp  &&
		    lni->hw_cps[i]->tc == cmdq->cur_cp->tc &&
		    lni->hw_cps[i]->tc_type == cmdq->cur_cp->tc_type &&
		    lni->hw_cps[i]->lcid == cmdq->cur_cp->lcid) {
			hw_cp = lni->hw_cps[i];
			break;
		}
	}

	if (hw_cp) {
		ret = cxip_libcxi_ops.modify_cp(cmdq->lni->lni, hw_cp, vni);
		if (!ret) {
			hw_cp->vni_pcp = vni;
			cmdq->cur_cp->vni_pcp = vni;
			pthread_rwlock_unlock(&lni->cp_lock);

			return FI_SUCCESS;
		}

		CXIP_WARN("cxil_modify_cp failed:%d\n", ret);
	}

	pthread_rwlock_unlock(&lni->cp_lock);

	return -FI_ENOENT;

cp_set:
	ret = cxip_cmdq_cp_set(cmdq, vni, tc, CXI_TC_TYPE_DEFAULT);
	if (ret)
		CXIP_WARN("Failed to change communication profile: %d\n", ret);

	return ret;
}

/*
 * cxip_alloc_dev_cmdq() - Allocate the libcxi command queue.
 *
 * With a user buffer, the ring is placed in that buffer; this requires
 * a runtime libcxi that provides cxil_alloc_buf_cmdq(), resolved into
 * cxip_libcxi_ops.alloc_buf_cmdq, and build headers that declare
 * struct cxi_cq_alloc_opts_buf.  cxip_cmdq_alloc() rejects a user
 * buffer before calling this when either is missing.
 */
static int cxip_alloc_dev_cmdq(struct cxip_lni *lni, struct cxi_eq *evtq,
			       struct cxi_cq_alloc_opts *cq_opts,
			       const struct fi_cxi_ki_cmdq_buf *ubuf,
			       struct cxi_cq **dev_cmdq)
{
#ifdef CXI_HAVE_ALLOC_BUF_CMDQ
	if (ubuf) {
		struct cxi_cq_alloc_opts_buf buf_opts = {
			.opts = *cq_opts,
			.buf = ubuf->buf,
			.is_host_mem = ubuf->is_host_mem,
			.dmabuf_fd = ubuf->dmabuf_fd,
			.dmabuf_offset = ubuf->dmabuf_offset,
		};

		return cxip_libcxi_ops.alloc_buf_cmdq(lni->lni, evtq,
						      &buf_opts, dev_cmdq);
	}
#endif

	return cxil_alloc_cmdq(lni->lni, evtq, cq_opts, dev_cmdq);
}

/*
 * cxip_cmdq_alloc() - Allocate a command queue.
 *
 * For a triggered CQ, the lcid must be 0.
 *
 * ubuf describes a user-provided buffer to hold the command queue, or
 * is NULL for a driver-allocated queue.
 */
int cxip_cmdq_alloc(struct cxip_lni *lni, struct cxi_eq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts, uint16_t vni,
		    enum cxi_traffic_class tc,
		    enum cxi_traffic_class_type tc_type,
		    const struct fi_cxi_ki_cmdq_buf *ubuf,
		    struct cxip_cmdq **cmdq)
{
	int ret;
	struct cxi_cq *dev_cmdq;
	struct cxip_cmdq *new_cmdq;
	struct cxi_cp *cp = NULL;

	/* Fail before allocating a communication profile for a request
	 * that cannot succeed.
	 */
	if (ubuf && !cxip_libcxi_ops.alloc_buf_cmdq) {
		CXIP_WARN("User cmdq buffer unsupported by this libcxi\n");
		return -FI_ENOSYS;
	}

	new_cmdq = calloc(1, sizeof(*new_cmdq));
	if (!new_cmdq) {
		CXIP_WARN("Unable to allocate CMDQ structure\n");
		return -FI_ENOMEM;
	}

	if (cq_opts->flags & CXI_CQ_IS_TX) {
		ret = cxip_cp_get(lni, vni, tc, tc_type, &cp,
				  !!(cq_opts->flags & CXI_CQ_TX_WITH_TRIG_CMDS));
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Failed to allocate CP: %d\n", ret);
			goto free_cmdq;
		}
		cq_opts->lcid = cp->lcid;

		new_cmdq->cur_cp = cp;
		new_cmdq->prev_cp = cp;

		/* Trig command queue can never use LL ring. */
		if (cq_opts->flags & CXI_CQ_TX_WITH_TRIG_CMDS ||
		    lni->iface->info->device_platform == CXI_PLATFORM_NETSIM)
			new_cmdq->llring_mode = CXIP_LLRING_NEVER;
		else
			new_cmdq->llring_mode = cxip_env.llring_mode;
	} else {
		new_cmdq->llring_mode = CXIP_LLRING_NEVER;
	}

	ret = cxip_alloc_dev_cmdq(lni, evtq, cq_opts, ubuf, &dev_cmdq);
	if (ret) {
		CXIP_WARN("Failed to allocate %s, ret: %d count: %u "
			  "policy: %d stat_cnt_pool: %u flags: %#x "
			  "lcid: %u lpe_cdt_thresh_id: %u vni: %u "
			  "tc: %d tc_type: %d buf: %p is_host: %d "
			  "dmabuf_fd: %d\n",
			  (cq_opts->flags & CXI_CQ_IS_TX) ? "TXQ" : "TGQ", ret,
			  cq_opts->count, (int)cq_opts->policy,
			  cq_opts->stat_cnt_pool, cq_opts->flags, cq_opts->lcid,
			  cq_opts->lpe_cdt_thresh_id, vni, (int)tc, (int)tc_type,
			  ubuf ? ubuf->buf : NULL,
			  ubuf ? ubuf->is_host_mem : 0,
			  ubuf ? ubuf->dmabuf_fd : -1);
		if (ubuf && ret == -EINVAL)
			CXIP_WARN("User cmdq buffer rejected by the driver; "
				  "it may lack CXI_OP_CQ_ALLOC_BUF\n");
		ret = -FI_ENOSPC;
		goto free_cmdq;
	}

	/* The kernel may round the ring up to more commands than requested.
	 * For a user-provided buffer, a larger ring would extend past the
	 * buffer, so refuse it.  The KI hints validation should prevent
	 * this; this is a backstop in case it and the kernel's rounding
	 * disagree.
	 */
	if (ubuf && (size_t)dev_cmdq->size * C_CQ_CMD_SIZE > ubuf->len) {
		CXIP_WARN("User-provided cmdq buffer %p: kernel sized ring "
			  "at %u commands (%zu bytes), buffer is %zu bytes\n",
			  ubuf->buf, dev_cmdq->size,
			  (size_t)dev_cmdq->size * C_CQ_CMD_SIZE, ubuf->len);
		cxil_destroy_cmdq(dev_cmdq);
		ret = -FI_EINVAL;
		goto free_cmdq;
	}

	new_cmdq->dev_cmdq = dev_cmdq;
	new_cmdq->lni = lni;
	*cmdq = new_cmdq;

	return FI_SUCCESS;

free_cmdq:
	free(new_cmdq);

	return ret;
}

/*
 * cxip_cmdq_free() - Free a command queue.
 */
void cxip_cmdq_free(struct cxip_cmdq *cmdq)
{
	int ret;

	ret = cxil_destroy_cmdq(cmdq->dev_cmdq);
	if (ret)
		CXIP_WARN("cxil_destroy_cmdq failed, ret: %d\n", ret);

	free(cmdq);
}

/* Must hold cmdq->lock. */
int cxip_cmdq_emit_c_state(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state)
{
	int ret;

	if (memcmp(&cmdq->c_state, c_state, sizeof(*c_state))) {
		ret = cxi_cq_emit_c_state(cmdq->dev_cmdq, c_state);
		if (ret) {
			CXIP_DBG("Failed to issue C_STATE command: %d\n", ret);
			cxi_cq_ring(cmdq->dev_cmdq);
			return -FI_EAGAIN;
		}

		cmdq->c_state = *c_state;
	}

	return FI_SUCCESS;
}

int cxip_cmdq_emit_idc_put(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state,
			   const struct c_idc_put_cmd *put, const void *buf,
			   size_t len, uint64_t flags)
{
	int ret;

	if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_WARN("Failed to issue fence command: %d:%s\n", ret,
				  fi_strerror(-ret));
			ret = -FI_EAGAIN;
			goto err;
		}
	}

	ret = cxip_cmdq_emit_c_state(cmdq, c_state);
	if (ret) {
		CXIP_WARN("Failed to emit c_state command: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err;
	}

	ret = cxi_cq_emit_idc_put(cmdq->dev_cmdq, put, buf, len);
	if (ret) {
		CXIP_WARN("Failed to emit idc_put command: %d:%s\n", ret,
			  fi_strerror(-ret));
		ret = -FI_EAGAIN;
		goto err;
	}

	return FI_SUCCESS;

err:
	/* On error (e.g. command queue full), always ring the CQ to prevent
	 * FI_MORE deadlock.
	 */
	cxi_cq_ring(cmdq->dev_cmdq);

	return ret;
}

int cxip_cmdq_emit_dma(struct cxip_cmdq *cmdq, struct c_full_dma_cmd *dma,
		       uint64_t flags)
{
	int ret;

	if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_WARN("Failed to issue fence command: %d:%s\n", ret,
				  fi_strerror(-ret));
			ret = -FI_EAGAIN;
			goto err;
		}
	}

	ret = cxi_cq_emit_dma(cmdq->dev_cmdq, dma);
	if (ret) {
		CXIP_WARN("Failed to emit dma command: %d:%s\n", ret,
			  fi_strerror(-ret));
		ret = -FI_EAGAIN;
		goto err;
	}

	return FI_SUCCESS;

err:
	/* On error (e.g. command queue full), always ring the CQ to prevent
	 * FI_MORE deadlock.
	 */
	cxi_cq_ring(cmdq->dev_cmdq);

	return ret;
}

int cxip_cmdq_emic_idc_amo(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state,
			   const struct c_idc_amo_cmd *amo, uint64_t flags,
			   bool fetching, bool flush)
{
	struct c_full_dma_cmd flush_cmd;
	bool fetching_flush = fetching && flush;
	int ret;

	if (fetching_flush) {
		memset(&flush_cmd, 0, sizeof(flush_cmd));
		flush_cmd.command.opcode = C_CMD_PUT;
		flush_cmd.index_ext = c_state->index_ext;
		flush_cmd.event_send_disable = 1;
		flush_cmd.dfa = amo->idc_header.dfa;
		flush_cmd.remote_offset = amo->idc_header.remote_offset;
		flush_cmd.eq = c_state->eq;
		flush_cmd.user_ptr = c_state->user_ptr;
		flush_cmd.flush = 1;
	}

	if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_WARN("Failed to issue fence command: %d:%s\n", ret,
				  fi_strerror(-ret));
			ret = -FI_EAGAIN;
			goto err;
		}
	}

	ret = cxip_cmdq_emit_c_state(cmdq, c_state);
	if (ret) {
		CXIP_WARN("Failed to emit c_state command: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err;
	}

	/* Fetching AMO with flush requires two commands. Ensure there is enough
	 * space. At worse at least 16x 32-byte slots are needed.
	 */
	if (fetching_flush && __cxi_cq_free_slots(cmdq->dev_cmdq) < 16) {
		CXIP_WARN("No space for FAMO with FI_DELIVERY_COMPLETE\n");
		ret = -FI_EAGAIN;
		goto err;
	}

	ret = cxi_cq_emit_idc_amo(cmdq->dev_cmdq, amo, fetching);
	if (ret) {
		CXIP_WARN("Failed to emit IDC amo\n");
		ret = -FI_EAGAIN;
		goto err;
	}

	if (fetching_flush) {
		/* CQ space check already occurred. Thus, return code can be
		 * ignored.
		 */
		ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &flush_cmd);
		assert(ret == 0);
	}

	return FI_SUCCESS;

err:
	/* On error (e.g. command queue full), always ring the CQ to prevent
	 * FI_MORE deadlock.
	 */
	cxi_cq_ring(cmdq->dev_cmdq);

	return ret;
}

int cxip_cmdq_emit_dma_amo(struct cxip_cmdq *cmdq, struct c_dma_amo_cmd *amo,
			   uint64_t flags, bool fetching, bool flush)
{
	struct c_full_dma_cmd flush_cmd;
	bool fetching_flush = fetching && flush;
	int ret;

	if (fetching_flush) {
		memset(&flush_cmd, 0, sizeof(flush_cmd));
		flush_cmd.command.opcode = C_CMD_PUT;
		flush_cmd.index_ext = amo->index_ext;
		flush_cmd.event_send_disable = 1;
		flush_cmd.dfa = amo->dfa;
		flush_cmd.remote_offset = amo->remote_offset;
		flush_cmd.eq = amo->eq;
		flush_cmd.user_ptr = amo->user_ptr;
		flush_cmd.flush = 1;
		flush_cmd.match_bits = amo->match_bits;
	}

	if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_WARN("Failed to issue fence command: %d:%s\n", ret,
				  fi_strerror(-ret));
			ret = -FI_EAGAIN;
			goto err;
		}
	}

	/* Fetching AMO with flush requires two commands. Ensure there is enough
	 * space. At worse at least 16x 32-byte slots are needed.
	 */
	if (fetching_flush && __cxi_cq_free_slots(cmdq->dev_cmdq) < 16) {
		CXIP_WARN("No space for FAMO with FI_DELIVERY_COMPLETE\n");
		ret = -FI_EAGAIN;
		goto err;
	}

	ret = cxi_cq_emit_dma_amo(cmdq->dev_cmdq, amo, fetching);
	if (ret) {
		CXIP_WARN("Failed to emit DMA amo\n");
		ret = -FI_EAGAIN;
		goto err;
	}

	if (fetching_flush) {
		/* CQ space check already occurred. Thus, return code can be
		 * ignored.
		 */
		ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &flush_cmd);
		assert(ret == 0);
	}

	return FI_SUCCESS;

err:
	/* On error (e.g. command queue full), always ring the CQ to prevent
	 * FI_MORE deadlock.
	 */
	cxi_cq_ring(cmdq->dev_cmdq);

	return ret;
}

int cxip_cmdq_emit_idc_msg(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state,
			   const struct c_idc_msg_hdr *msg, const void *buf,
			   size_t len, uint64_t flags)
{
	int ret;

	if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_WARN("Failed to issue fence command: %d:%s\n", ret,
				  fi_strerror(-ret));
			ret = -FI_EAGAIN;
			goto err;
		}
	}

	ret = cxip_cmdq_emit_c_state(cmdq, c_state);
	if (ret) {
		CXIP_WARN("Failed to emit c_state command: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err;
	}

	ret = cxi_cq_emit_idc_msg(cmdq->dev_cmdq, msg, buf, len);
	if (ret) {
		CXIP_WARN("Failed to emit idc_msg command: %d:%s\n", ret,
			  fi_strerror(-ret));
		ret = -FI_EAGAIN;
		goto err;
	}

	return FI_SUCCESS;

err:
	/* On error (e.g. command queue full), always ring the CQ to prevent
	 * FI_MORE deadlock.
	 */
	cxi_cq_ring(cmdq->dev_cmdq);

	return ret;
}
