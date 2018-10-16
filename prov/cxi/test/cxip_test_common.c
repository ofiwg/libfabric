/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"

struct fi_info *cxit_fi_hints;
struct fi_info *cxit_fi;
struct fid_fabric *cxit_fabric;
struct fid_domain *cxit_domain;
struct fid_ep *cxit_ep;
struct fid_ep *cxit_sep;
struct fi_cq_attr cxit_tx_cq_attr, cxit_rx_cq_attr;
struct fid_cq *cxit_tx_cq, *cxit_rx_cq;
struct fi_av_attr cxit_av_attr;
struct fid_av *cxit_av;
char *cxit_node, *cxit_service;
uint64_t cxit_flags;
int cxit_n_ifs;

#define	PRT(in, fmt, fld) \
	printf("%*s%-28s = " fmt "\n", in, "", #fld, fld)

#define	PRTHEX(in, fld, len) \
	cxit_dump_hex("%*s%-28s = ", in, #fld, fld, len)

static void cxit_dump_hex(const char *fmt, int in, const char *nam,
			  void *ptr, int len)
{
	uint8_t *b = (uint8_t *)ptr;

	printf(fmt, in, "", nam);
	if (b && len) {
		while (len-- > 0)
			printf("%02x", *b++);
	} else {
		printf("(nil)");
	}
	printf("\n");
}

void cxit_dump_tx_attr(struct fi_tx_attr *tx_attr)
{
	if (!tx_attr)
		tx_attr = cxit_fi->tx_attr;
	if (!tx_attr)
		return;
	PRT(2, "0x%016lx", tx_attr->caps);
	PRT(2, "0x%016lx", tx_attr->mode);
	PRT(2, "0x%016lx", tx_attr->op_flags);
	PRT(2, "0x%016lx", tx_attr->msg_order);
	PRT(2, "0x%016lx", tx_attr->comp_order);
	PRT(2, "%lu", tx_attr->inject_size);
	PRT(2, "%lu", tx_attr->size);
	PRT(2, "%lu", tx_attr->iov_limit);
	PRT(2, "%lu", tx_attr->rma_iov_limit);
}

void cxit_dump_rx_attr(struct fi_rx_attr *rx_attr)
{
	if (!rx_attr)
		rx_attr = cxit_fi->rx_attr;
	if (!rx_attr)
		return;
	PRT(2, "0x%016lx", rx_attr->caps);
	PRT(2, "0x%016lx", rx_attr->mode);
	PRT(2, "0x%016lx", rx_attr->op_flags);
	PRT(2, "0x%016lx", rx_attr->msg_order);
	PRT(2, "0x%016lx", rx_attr->comp_order);
	PRT(2, "%lu", rx_attr->total_buffered_recv);
	PRT(2, "%lu", rx_attr->size);
	PRT(2, "%lu", rx_attr->iov_limit);
}

void cxit_dump_ep_attr(struct fi_ep_attr *ep_attr)
{
	if (!ep_attr)
		ep_attr = cxit_fi->ep_attr;
	if (!ep_attr)
		return;
	PRT(2, "%d", ep_attr->type);
	PRT(2, "%u", ep_attr->protocol);
	PRT(2, "%u", ep_attr->protocol_version);
	PRT(2, "%lu", ep_attr->max_msg_size);
	PRT(2, "%lu", ep_attr->msg_prefix_size);
	PRT(2, "%lu", ep_attr->max_order_raw_size);
	PRT(2, "%lu", ep_attr->max_order_war_size);
	PRT(2, "%lu", ep_attr->max_order_waw_size);
	PRT(2, "0x%016lx", ep_attr->mem_tag_format);
	PRT(2, "%lu", ep_attr->tx_ctx_cnt);
	PRT(2, "%lu", ep_attr->rx_ctx_cnt);
	PRT(2, "%lu", ep_attr->auth_key_size);
	PRTHEX(2, ep_attr->auth_key, ep_attr->auth_key_size);
}

void cxit_dump_domain_attr(struct fi_domain_attr *dom_attr)
{
	if (!dom_attr)
		dom_attr = cxit_fi->domain_attr;
	if (!dom_attr)
		return;
	PRT(2, "%p", dom_attr->domain);
	PRT(2, "\"%s\"", dom_attr->name);
	PRT(2, "%d", dom_attr->threading);
	PRT(2, "%d", dom_attr->control_progress);
	PRT(2, "%d", dom_attr->data_progress);
	PRT(2, "%d", dom_attr->resource_mgmt);
	PRT(2, "%d", dom_attr->av_type);
	PRT(2, "%d", dom_attr->mr_mode);
	PRT(2, "%lu", dom_attr->mr_key_size);
	PRT(2, "%lu", dom_attr->cq_data_size);
	PRT(2, "%lu", dom_attr->cq_cnt);
	PRT(2, "%lu", dom_attr->ep_cnt);
	PRT(2, "%lu", dom_attr->tx_ctx_cnt);
	PRT(2, "%lu", dom_attr->rx_ctx_cnt);
	PRT(2, "%lu", dom_attr->max_ep_tx_ctx);
	PRT(2, "%lu", dom_attr->max_ep_rx_ctx);
	PRT(2, "%lu", dom_attr->max_ep_stx_ctx);
	PRT(2, "%lu", dom_attr->max_ep_srx_ctx);
	PRT(2, "%lu", dom_attr->cntr_cnt);
	PRT(2, "%lu", dom_attr->mr_iov_limit);
	PRT(2, "0x%016lx", dom_attr->caps);
	PRT(2, "0x%016lx", dom_attr->mode);
	PRT(2, "%lu", dom_attr->auth_key_size);
	PRTHEX(2, dom_attr->auth_key, dom_attr->auth_key_size);
	PRT(2, "%lu", dom_attr->max_err_data);
	PRT(2, "%lu", dom_attr->mr_cnt);
}

void cxit_dump_fabric_attr(struct fi_fabric_attr *fab_attr)
{
	if (!fab_attr)
		fab_attr = cxit_fi->fabric_attr;
	if (!fab_attr)
		return;
	PRT(2, "%p", fab_attr->fabric);
	PRT(2, "\"%s\"", fab_attr->name);
	PRT(2, "\"%s\"", fab_attr->prov_name);
	PRT(2, "0x%08x", fab_attr->prov_version);
	PRT(2, "0x%08x", fab_attr->api_version);
}

void cxit_dump_attr(struct fi_info *info)
{
	if (!info)
		info = cxit_fi;

	printf("\n========\n");
	PRT(0, "%p", info);
	if (!info)
		return;
	PRT(0, "0x%016lx", info->caps);
	PRT(0, "0x%016lx", info->mode);
	PRT(0, "0x%08x", info->addr_format);
	PRT(0, "%ld", info->src_addrlen);
	PRT(0, "%ld", info->dest_addrlen);
	PRTHEX(0, info->src_addr, info->src_addrlen);
	PRTHEX(0, info->dest_addr, info->dest_addrlen);
	PRT(0, "%p", info->handle);
	PRT(0, "%p", info->tx_attr);
	cxit_dump_tx_attr(info->tx_attr);
	PRT(0, "%p", info->rx_attr);
	cxit_dump_rx_attr(info->rx_attr);
	PRT(0, "%p", info->ep_attr);
	cxit_dump_ep_attr(info->ep_attr);
	PRT(0, "%p", info->domain_attr);
	cxit_dump_domain_attr(info->domain_attr);
	PRT(0, "%p", info->fabric_attr);
	cxit_dump_fabric_attr(info->fabric_attr);
	printf("========\n\n");
	fflush(stdout);
}
#undef	PRT

void cxit_create_fabric_info(void)
{
	int ret;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 cxit_node, cxit_service, cxit_flags, cxit_fi_hints,
			 &cxit_fi);
	cr_assert(ret == FI_SUCCESS, "fi_getinfo");
}

void cxit_destroy_fabric_info(void)
{
	fi_freeinfo(cxit_fi);
	cxit_fi = NULL;
}

void cxit_create_fabric(void)
{
	int ret;

	ret = fi_fabric(cxit_fi->fabric_attr, &cxit_fabric, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_fabric");
}

void cxit_destroy_fabric(void)
{
	int ret;

	ret = fi_close(&cxit_fabric->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close fabric");
	cxit_fabric = NULL;
}

void cxit_create_domain(void)
{
	int ret;

	ret = fi_domain(cxit_fabric, cxit_fi, &cxit_domain, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_domain");
}

void cxit_destroy_domain(void)
{
	int ret;

	ret = fi_close(&cxit_domain->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close domain. %d", ret);
	cxit_domain = NULL;
}

void cxit_create_ep(void)
{
	int ret;

	ret = fi_endpoint(cxit_domain, cxit_fi, &cxit_ep, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_domain");
	cr_assert_not_null(cxit_ep);
}

void cxit_destroy_ep(void)
{
	int ret;

	ret = fi_close(&cxit_ep->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close endpoint");
	cxit_ep = NULL;
}

void cxit_create_sep(void)
{
	int ret;

	ret = fi_scalable_ep(cxit_domain, cxit_fi, &cxit_sep, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_scalable_ep");
	cr_assert_not_null(cxit_sep);
}

void cxit_destroy_sep(void)
{
	int ret;

	ret = fi_close(&cxit_sep->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close scalable ep");
	cxit_sep = NULL;
}

void cxit_create_cqs(void)
{
	int ret;

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;

	ret = fi_cq_open(cxit_domain, &cxit_tx_cq_attr, &cxit_tx_cq, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cq_open (TX)");

	cxit_rx_cq_attr.format = FI_CQ_FORMAT_TAGGED;

	ret = fi_cq_open(cxit_domain, &cxit_rx_cq_attr, &cxit_rx_cq, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cq_open (RX)");
}

void cxit_destroy_cqs(void)
{
	int ret;

	ret = fi_close(&cxit_rx_cq->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close RX CQ");
	cxit_rx_cq = NULL;

	ret = fi_close(&cxit_tx_cq->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close TX CQ");
	cxit_tx_cq = NULL;
}

void cxit_bind_cqs(void)
{
	int ret;

	ret = fi_ep_bind(cxit_ep, &cxit_tx_cq->fid, FI_TRANSMIT);
	cr_assert(!ret, "fi_ep_bind TX CQ");

	ret = fi_ep_bind(cxit_ep, &cxit_rx_cq->fid, FI_RECV);
	cr_assert(!ret, "fi_ep_bind RX CQ");
}

void cxit_create_av(void)
{
	int ret;

	ret = fi_av_open(cxit_domain, &cxit_av_attr, &cxit_av, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_av_open");
}

void cxit_destroy_av(void)
{
	int ret;

	ret = fi_close(&cxit_av->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close AV. %d", ret);
	cxit_av = NULL;
}

void cxit_bind_av(void)
{
	int ret;

	ret = fi_ep_bind(cxit_ep, &cxit_av->fid, 0);
	cr_assert(!ret, "fi_ep_bind AV");
}

static void cxit_init(void)
{
	struct slist_entry *entry, *prev __attribute__ ((unused));
	int ret;

	/* Force provider init */
	ret = fi_getinfo(FI_VERSION(-1U, -1U),
			 NULL, NULL, 0, NULL,
			 NULL);
	cr_assert(ret == -FI_ENOSYS);

	slist_foreach(&cxip_if_list, entry, prev) {
		cxit_n_ifs++;
	}
}

void cxit_setup_getinfo(void)
{
	cxit_init();

	cxit_fi_hints = fi_allocinfo();
	cr_assert(cxit_fi_hints, "fi_allocinfo");
}

void cxit_teardown_getinfo(void)
{
	fi_freeinfo(cxit_fi_hints);
	cxit_fi_hints = NULL;
}

void cxit_setup_fabric(void)
{
	if (!cxit_fi_hints) {
		cxit_setup_getinfo();

		/* Always select CXI */
		cxit_fi_hints->fabric_attr->prov_name = strdup(cxip_prov_name);
	}

	cxit_create_fabric_info();
}

void cxit_teardown_fabric(void)
{
	cxit_destroy_fabric_info();
	cxit_teardown_getinfo();
}

void cxit_setup_domain(void)
{
	cxit_setup_fabric();
	cxit_create_fabric();
}

void cxit_teardown_domain(void)
{
	cxit_destroy_fabric();
	cxit_teardown_fabric();
}

void cxit_setup_ep(void)
{
	cxit_setup_domain();
	cxit_create_domain();
}

void cxit_teardown_ep(void)
{
	cxit_destroy_domain();
	cxit_teardown_domain();
}

void cxit_setup_rma(void)
{
	int ret;

	/* Request required capabilities for RMA */
	cxit_setup_getinfo();
	cxit_fi_hints->fabric_attr->prov_name = strdup(cxip_prov_name);
	cxit_fi_hints->caps = FI_WRITE | FI_READ;
	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_av_attr.type = FI_AV_TABLE;

	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

	cxit_setup_ep();

	/* Set up RMA objects */
	cxit_create_ep();
	cxit_create_cqs();
	cxit_bind_cqs();
	cxit_create_av();
	cxit_bind_av();

	/* Insert local address into AV to prepare to send to self */
	ret = fi_av_insert(cxit_av, cxit_fi->src_addr, 1, NULL, 0, NULL);
	cr_assert(ret == 1);

	ret = fi_enable(cxit_ep);
	cr_assert(ret == FI_SUCCESS);
}

void cxit_teardown_rma(void)
{
	/* Tear down RMA objects */
	cxit_destroy_ep(); /* EP must be destroyed before bound objects */

	cxit_destroy_av();
	cxit_destroy_cqs();
	cxit_teardown_ep();
}

