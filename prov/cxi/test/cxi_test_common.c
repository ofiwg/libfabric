/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"

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

	ret = fi_cq_open(cxit_domain, &cxit_tx_cq_attr, &cxit_tx_cq, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cq_open (TX)");

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
	struct slist_entry *entry, *prev;
	int ret;

	/* Force provider init */
	ret = fi_getinfo(FI_VERSION(-1U, -1U),
			 NULL, NULL, 0, NULL,
			 NULL);
	cr_assert(ret == -FI_ENOSYS);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxix_if_list, entry, prev) {
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
		cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);
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
	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);
	cxit_fi_hints->caps = FI_WRITE;
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

