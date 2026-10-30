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

void *cxit_ki_page_alloc(size_t len)
{
	void *buf;

	buf = aligned_alloc(s_page_size, len);
	cr_assert(buf);
	memset(buf, 0, len);

	return buf;
}

void cxit_ki_page_free(void *buf)
{
	free(buf);
}

void cxit_ki_init_hints(struct fi_cxi_ki_ep_hints *hints, void *tx_buf,
			size_t tx_len, void *trig_buf, size_t trig_len)
{
	memset(hints, 0, sizeof(*hints));

	hints->tx_cmdq.buf = tx_buf;
	hints->tx_cmdq.len = tx_len;
	hints->tx_cmdq.is_host_mem = true;
	hints->tx_cmdq.dmabuf_fd = -1;
	hints->tx_cmdq.dmabuf_offset = 0;

	hints->trig_cmdq.buf = trig_buf;
	hints->trig_cmdq.len = trig_len;
	hints->trig_cmdq.is_host_mem = true;
	hints->trig_cmdq.dmabuf_fd = -1;
	hints->trig_cmdq.dmabuf_offset = 0;
}

void cxit_ki_setup_ep(bool prov_key)
{
	cxit_setup_getinfo();

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_av_attr.type = FI_AV_TABLE;

	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	cxit_fi_hints->tx_attr->size = 512;
	cxit_fi_hints->caps = CXIT_KI_EP_CAPS;
	if (prov_key)
		cxit_fi_hints->domain_attr->mr_mode |= FI_MR_PROV_KEY;

	cxit_setup_ep();
	cxit_create_ep();
}

void cxit_ki_create_enabled_objects(bool with_cntrs)
{
	cxit_create_eq();
	cxit_bind_eq();
	cxit_create_cqs();
	cxit_bind_cqs();
	if (with_cntrs) {
		cxit_create_cntrs();
		cxit_bind_cntrs();
	}
	cxit_create_av();
	cxit_bind_av();
}

void cxit_ki_destroy_enabled_objects(void)
{
	if (cxit_av)
		cxit_destroy_av();

	cxit_destroy_cntrs();

	if (cxit_tx_cq)
		cxit_destroy_cqs();

	if (cxit_eq)
		cxit_destroy_eq();
}

struct fi_cxi_ki_ops *cxit_ki_ops;
void *cxit_ki_full_tx_buf;
void *cxit_ki_full_trig_buf;

static void cxit_setup_ki_full_common(bool prov_key)
{
	struct fi_cxi_ki_ep_hints hints;
	size_t addrlen = sizeof(cxit_ep_addr);
	int ret;

	cxit_ki_setup_ep(prov_key);
	cxit_ki_create_enabled_objects(true);

	cxit_ki_full_tx_buf = cxit_ki_page_alloc(s_page_size);
	cxit_ki_full_trig_buf = cxit_ki_page_alloc(s_page_size);
	cxit_ki_init_hints(&hints, cxit_ki_full_tx_buf, s_page_size,
			   cxit_ki_full_trig_buf, s_page_size);

	ret = fi_control(&cxit_ep->fid, FI_OPT_CXI_SET_KI_EP_HINTS_1, &hints);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control returned %d", ret);

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable returned %d", ret);

	/* The endpoint address is assigned during enable; tests insert it
	 * into the AV to resolve the endpoint's own target info.
	 */
	ret = fi_getname(&cxit_ep->fid, &cxit_ep_addr, &addrlen);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getname returned %d", ret);
	cr_assert_eq(addrlen, sizeof(cxit_ep_addr));

	ret = fi_open_ops(&cxit_ep->fid, FI_CXI_KI_OPS_1, 0,
			  (void **)&cxit_ki_ops, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_open_ops returned %d", ret);
}

void cxit_setup_ki_full(void)
{
	cxit_setup_ki_full_common(false);
}

void cxit_setup_ki_full_prov_key(void)
{
	cxit_setup_ki_full_common(true);
}

void cxit_teardown_ki_full(void)
{
	cxit_destroy_ep(); /* EP must be destroyed before bound objects */
	cxit_ki_destroy_enabled_objects();
	cxit_teardown_ep();

	cxit_ki_ops = NULL;
	cxit_ki_page_free(cxit_ki_full_tx_buf);
	cxit_ki_full_tx_buf = NULL;
	cxit_ki_page_free(cxit_ki_full_trig_buf);
	cxit_ki_full_trig_buf = NULL;
}

void cxit_ki_create_remote_mr(size_t len, uint64_t access, uint64_t *key,
			      struct mem_region *window,
			      struct fi_cxi_ki_peer_info *peer,
			      struct fi_cxi_gpu_remote_mr *remote)
{
	int ret;
	struct fi_cxi_ki_peer_info local_peer;

	if (!peer)
		peer = &local_peer;

	ret = mr_create(len, access, 0, key, window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create(key=%lu) returned %d", *key,
		     ret);

	ret = cxit_ki_ops->get_peer_info(&cxit_ep->fid, &window->mr->fid,
					 peer);
	cr_assert_eq(ret, FI_SUCCESS, "get_peer_info(key=%lu) returned %d",
		     *key, ret);

	ret = cxit_ki_ops->get_gpu_remote_mrs(&cxit_ep->fid, peer, 1, remote);
	cr_assert_eq(ret, FI_SUCCESS,
		     "get_gpu_remote_mrs(key=%lu) returned %d", *key, ret);
}
