/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018,2020 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <criterion/criterion.h>
#include <ltu_utils_pm.h>

#include "cxip_test_common.h"

struct fi_info *cxit_fi_hints;
struct fi_info *cxit_fi;
struct fid_fabric *cxit_fabric;
struct fid_domain *cxit_domain;
struct fi_cxi_dom_ops *dom_ops;
struct fid_ep *cxit_ep;
struct cxip_addr cxit_ep_addr;
fi_addr_t cxit_ep_fi_addr;
struct fid_ep *cxit_sep;
struct fi_eq_attr cxit_eq_attr = {};
struct fid_eq *cxit_eq;
struct fi_cq_attr cxit_tx_cq_attr = {
	.format = FI_CQ_FORMAT_TAGGED,
	.size = 16384
};
struct fi_cq_attr cxit_rx_cq_attr = { .format = FI_CQ_FORMAT_TAGGED };
uint64_t cxit_eq_bind_flags = 0;
uint64_t cxit_tx_cq_bind_flags = FI_TRANSMIT;
uint64_t cxit_rx_cq_bind_flags = FI_RECV;
struct fid_cq *cxit_tx_cq, *cxit_rx_cq;
struct fi_cntr_attr cxit_cntr_attr = {};
struct fid_cntr *cxit_send_cntr, *cxit_recv_cntr;
struct fid_cntr *cxit_read_cntr, *cxit_write_cntr;
struct fid_cntr *cxit_rem_cntr;
struct fi_av_attr cxit_av_attr;
struct fid_av *cxit_av;
struct cxit_coll_mc_list cxit_coll_mc_list = { .count = 5 };
char *cxit_node, *cxit_service;
uint64_t cxit_flags;
int cxit_n_ifs;
struct fid_av_set *cxit_av_set;
struct fid_mc *cxit_mc;

static ssize_t copy_from_hmem_iov(void *dest, size_t size,
				 enum fi_hmem_iface iface, uint64_t device,
				 const struct iovec *hmem_iov,
				 size_t hmem_iov_count,
				 uint64_t hmem_iov_offset)
{
	size_t cpy_size = MIN(size, hmem_iov->iov_len);

	assert(iface == FI_HMEM_SYSTEM);
	assert(hmem_iov_count == 1);
	assert(hmem_iov_offset == 0);

	memcpy(dest, hmem_iov->iov_base, cpy_size);

	return cpy_size;
}

static ssize_t copy_to_hmem_iov(enum fi_hmem_iface iface, uint64_t device,
				const struct iovec *hmem_iov,
				size_t hmem_iov_count,
				uint64_t hmem_iov_offset, const void *src,
				size_t size)
{
	size_t cpy_size = MIN(size, hmem_iov->iov_len);

	assert(iface == FI_HMEM_SYSTEM);
	assert(hmem_iov_count == 1);
	assert(hmem_iov_offset == 0);

	memcpy(hmem_iov->iov_base, src, cpy_size);

	return cpy_size;
}

struct fi_hmem_override_ops hmem_ops = {
	.copy_from_hmem_iov = copy_from_hmem_iov,
	.copy_to_hmem_iov = copy_to_hmem_iov,
};

void cxit_create_fabric_info(void)
{
	int ret;

	if (cxit_fi)
		return;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 cxit_node, cxit_service, cxit_flags, cxit_fi_hints,
			 &cxit_fi);
	cr_assert(ret == FI_SUCCESS, "fi_getinfo");
	cxit_fi->ep_attr->tx_ctx_cnt = cxit_fi->domain_attr->tx_ctx_cnt;
	cxit_fi->ep_attr->rx_ctx_cnt = cxit_fi->domain_attr->rx_ctx_cnt;
}

void cxit_destroy_fabric_info(void)
{
	fi_freeinfo(cxit_fi);
	cxit_fi = NULL;
}

void cxit_create_fabric(void)
{
	int ret;

	if (cxit_fabric)
		return;

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

	if (cxit_domain)
		return;

	ret = fi_domain(cxit_fabric, cxit_fi, &cxit_domain, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_domain");

	ret = fi_open_ops(&cxit_domain->fid, FI_CXI_DOM_OPS_1, 0,
			  (void **)&dom_ops, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_open_ops");

	ret = fi_set_ops(&cxit_domain->fid, FI_SET_OPS_HMEM_OVERRIDE, 0,
			 &hmem_ops, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_set_ops");
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
	cr_assert(ret == FI_SUCCESS, "fi_endpoint");
	cr_assert_not_null(cxit_ep);
}

void cxit_destroy_ep(void)
{
	int ret;

	if (cxit_ep != NULL) {
		ret = fi_close(&cxit_ep->fid);
		cr_assert(ret == FI_SUCCESS, "fi_close endpoint");
		cxit_ep = NULL;
	}
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

void cxit_create_eq(void)
{
	struct fi_eq_attr attr = {
		.size = 32,
		.flags = FI_WRITE,
		.wait_obj = FI_WAIT_NONE
	};
	int ret;

	ret = fi_eq_open(cxit_fabric, &attr, &cxit_eq, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_eq_open failed %d", ret);
	cr_assert_not_null(cxit_eq, "fi_eq_open returned NULL eq");
}

void cxit_destroy_eq(void)
{
	int ret;

	ret = fi_close(&cxit_eq->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close EQ failed %d", ret);
	cxit_eq = NULL;
}

void cxit_bind_eq(void)
{
	int ret;

	/* NOTE: ofi implementation does not allow any flags */
	ret = fi_ep_bind(cxit_ep, &cxit_eq->fid, cxit_eq_bind_flags);
	cr_assert(!ret, "fi_ep_bind EQ");
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

	ret = fi_ep_bind(cxit_ep, &cxit_tx_cq->fid, cxit_tx_cq_bind_flags);
	cr_assert(!ret, "fi_ep_bind TX CQ");

	ret = fi_ep_bind(cxit_ep, &cxit_rx_cq->fid, cxit_rx_cq_bind_flags);
	cr_assert(!ret, "fi_ep_bind RX CQ");
}

void cxit_create_cntrs(void)
{
	int ret;

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_send_cntr,
			   NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cntr_open (send)");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_recv_cntr,
			   NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cntr_open (recv)");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_read_cntr,
			   NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cntr_open (read)");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_write_cntr,
			   NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cntr_open (write)");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_rem_cntr, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_cntr_open (rem)");
}

void cxit_destroy_cntrs(void)
{
	int ret;

	ret = fi_close(&cxit_send_cntr->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close send_cntr");
	cxit_send_cntr = NULL;

	ret = fi_close(&cxit_recv_cntr->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close recv_cntr");
	cxit_recv_cntr = NULL;

	ret = fi_close(&cxit_read_cntr->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close read_cntr");
	cxit_read_cntr = NULL;

	ret = fi_close(&cxit_write_cntr->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close write_cntr");
	cxit_write_cntr = NULL;

	ret = fi_close(&cxit_rem_cntr->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close rem_cntr");
	cxit_rem_cntr = NULL;
}

void cxit_bind_cntrs(void)
{
	int ret;

	ret = fi_ep_bind(cxit_ep, &cxit_send_cntr->fid, FI_SEND);
	cr_assert(!ret, "fi_ep_bind send_cntr");

	ret = fi_ep_bind(cxit_ep, &cxit_recv_cntr->fid, FI_RECV);
	cr_assert(!ret, "fi_ep_bind recv_cntr");

	ret = fi_ep_bind(cxit_ep, &cxit_read_cntr->fid, FI_READ);
	cr_assert(!ret, "fi_ep_bind read_cntr");

	ret = fi_ep_bind(cxit_ep, &cxit_write_cntr->fid, FI_WRITE);
	cr_assert(!ret, "fi_ep_bind write_cntr");
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

/* expand AV and create av_sets for collectives */
static void _create_av_set(int count, int rank, struct fid_av_set **av_set_fid)
{
	struct cxip_ep *ep;
	struct cxip_comm_key comm_key = {
		.type = COMM_KEY_RANK,
		.rank.rank = rank,
		.rank.hwroot_idx = 0,
	};
	struct fi_av_set_attr attr = {
		.count = 0,
		.start_addr = FI_ADDR_NOTAVAIL,
		.end_addr = FI_ADDR_NOTAVAIL,
		.stride = 1,
		.comm_key_size = sizeof(comm_key),
		.comm_key = (void *)&comm_key,
		.flags = 0,
	};
	struct cxip_addr caddr;
	int i, ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);

	/* lookup initiator caddr */
	ret = _cxip_av_lookup(ep->ep_obj->av, cxit_ep_fi_addr, &caddr);
	cr_assert(ret == 0, "bad lookup on address %ld: %d\n",
		  cxit_ep_fi_addr, ret);

	/* create empty av_set */
	ret = fi_av_set(&ep->ep_obj->av->av_fid, &attr, av_set_fid, NULL);
	cr_assert(ret == 0, "av_set creation failed: %d\n", ret);

	/* add source address as multiple av entries */
	for (i = count - 1; i >= 0; i--) {
		fi_addr_t fi_addr;

		ret = fi_av_insert(&ep->ep_obj->av->av_fid, &caddr, 1,
				   &fi_addr, 0, NULL);
		cr_assert(ret == 1, "%d cxip_av_insert failed: %d\n", i, ret);
		ret = fi_av_set_insert(*av_set_fid, fi_addr);
		cr_assert(ret == 0, "%d fi_av_set_insert failed: %d\n", i, ret);
	}
}

void cxit_create_netsim_collective(int count)
{
	struct cxip_ep *ep;
	fi_addr_t mc_addr;
	fi_addr_t mc_ref;
	uint32_t event;
	int i, ret;

	cxit_coll_mc_list.count = count;
	cxit_coll_mc_list.av_set_fid = calloc(cxit_coll_mc_list.count,
					      sizeof(struct fid_av_set *));
	cxit_coll_mc_list.mc_fid = calloc(cxit_coll_mc_list.count,
					  sizeof(struct fid_mc *));

	for (i = 0; i < cxit_coll_mc_list.count; i++) {
		_create_av_set(cxit_coll_mc_list.count, i,
			       &cxit_coll_mc_list.av_set_fid[i]);

		ret = cxip_join_collective(cxit_ep, FI_ADDR_NOTAVAIL,
					   cxit_coll_mc_list.av_set_fid[i],
					   0, &cxit_coll_mc_list.mc_fid[i],
					   NULL);
		cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
		mc_addr = fi_mc_addr(cxit_coll_mc_list.mc_fid[i]);
		mc_ref = (fi_addr_t)(uintptr_t)cxit_coll_mc_list.mc_fid[i];
		cr_assert(mc_addr == mc_ref, "mc_addr, exp %ld, saw %ld\n",
			  mc_ref, mc_addr);

		ep = container_of(cxit_ep, struct cxip_ep, ep);
		do {
			sched_yield();
			ret = fi_eq_read(&ep->ep_obj->eq->util_eq.eq_fid,
					 &event, NULL, 0, 0);
		} while (ret == -FI_EAGAIN);
		cr_assert(event == FI_JOIN_COMPLETE, "join event = %d\n",
			  event);
	}
}

void cxit_destroy_netsim_collective(void)
{
	int i;

	for (i = cxit_coll_mc_list.count - 1; i >= 0; i--) {
		fi_close(&cxit_coll_mc_list.mc_fid[i]->fid);
		fi_close(&cxit_coll_mc_list.av_set_fid[i]->fid);
	}
	free(cxit_coll_mc_list.mc_fid);
	free(cxit_coll_mc_list.av_set_fid);
	cxit_coll_mc_list.mc_fid = NULL;
	cxit_coll_mc_list.av_set_fid = NULL;
}

static void cxit_init(void)
{
	struct slist_entry *entry, *prev __attribute__((unused));
	int ret;
	struct fi_info *hints = cxit_allocinfo();
	struct fi_info *info;

	setlinebuf(stdout);

	/* Force provider init */
	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 cxit_node, cxit_service, cxit_flags, hints,
			 &info);
	cr_assert(ret == FI_SUCCESS);

	slist_foreach(&cxip_if_list, entry, prev) {
		cxit_n_ifs++;
	}

	fi_freeinfo(info);
	fi_freeinfo(hints);
}

struct fi_info *cxit_allocinfo(void)
{
	struct fi_info *info;

	info = fi_allocinfo();
	cr_assert(info, "fi_allocinfo");

	/* Always select CXI */
	info->fabric_attr->prov_name = strdup(cxip_prov_name);
	info->domain_attr->mr_mode = FI_MR_ENDPOINT;

	return info;
}

void cxit_setup_getinfo(void)
{
	cxit_init();

	if (!cxit_fi_hints)
		cxit_fi_hints = cxit_allocinfo();
}

void cxit_teardown_getinfo(void)
{
	fi_freeinfo(cxit_fi_hints);
	cxit_fi_hints = NULL;
}

void cxit_setup_fabric(void)
{
	cxit_setup_getinfo();
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

void cxit_setup_enabled_ep(void)
{
	int ret;
	size_t addrlen = sizeof(cxit_ep_addr);

	cxit_setup_getinfo();

	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_av_attr.type = FI_AV_TABLE;
	cxit_av_attr.rx_ctx_bits = 4;

	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

	cxit_setup_ep();

	/* Set up RMA objects */
	cxit_create_ep();
	cxit_create_eq();
	cxit_bind_eq();
	cxit_create_cqs();
	cxit_bind_cqs();
	cxit_create_cntrs();
	cxit_bind_cntrs();
	cxit_create_av();
	cxit_bind_av();

	ret = fi_enable(cxit_ep);
	cr_assert(ret == FI_SUCCESS, "ret is: %d\n", ret);

	/* Find assigned Endpoint address. Address is assigned during enable. */
	ret = fi_getname(&cxit_ep->fid, &cxit_ep_addr, &addrlen);
	cr_assert(ret == FI_SUCCESS, "ret is %d\n", ret);
	cr_assert(addrlen == sizeof(cxit_ep_addr));
}

void cxit_setup_rma(void)
{
	int ret;
	struct cxip_addr fake_addr = {.nic = 0xad, .pid = 0xbc};

	cxit_setup_enabled_ep();

	/* Insert local address into AV to prepare to send to self */
	ret = fi_av_insert(cxit_av, (void *)&fake_addr, 1, NULL, 0, NULL);
	cr_assert(ret == 1);

	/* Insert local address into AV to prepare to send to self */
	ret = fi_av_insert(cxit_av, (void *)&cxit_ep_addr, 1, &cxit_ep_fi_addr,
			   0, NULL);
	cr_assert(ret == 1);
}

void cxit_teardown_rma(void)
{
	/* Tear down RMA objects */
	cxit_destroy_ep(); /* EP must be destroyed before bound objects */

	cxit_destroy_av();
	cxit_destroy_cntrs();
	cxit_destroy_cqs();
	cxit_destroy_eq();
	cxit_teardown_ep();
}

/* Everyone needs to wait sometime */
int cxit_await_completion(struct fid_cq *cq, struct fi_cq_tagged_entry *cqe)
{
	int ret;

	do {
		ret = fi_cq_read(cq, cqe, 1);
	} while (ret == -FI_EAGAIN);

	return ret;
}

void validate_tx_event(struct fi_cq_tagged_entry *cqe, uint64_t flags,
		       void *context)
{
	cr_assert(cqe->op_context == context, "TX CQE Context mismatch");
	cr_assert(cqe->flags == flags, "TX CQE flags mismatch");
	cr_assert(cqe->len == 0, "Invalid TX CQE length");
	cr_assert(cqe->buf == 0, "Invalid TX CQE address");
	cr_assert(cqe->data == 0, "Invalid TX CQE data");
	cr_assert(cqe->tag == 0, "Invalid TX CQE tag");
}

void validate_rx_event(struct fi_cq_tagged_entry *cqe, void *context,
		       size_t len, uint64_t flags, void *buf, uint64_t data,
		       uint64_t tag)
{
	cr_assert(cqe->op_context == context, "CQE Context mismatch");
	cr_assert(cqe->len == len, "Invalid CQE length");
	cr_assert(cqe->flags == flags, "CQE flags mismatch");
	cr_assert(cqe->buf == buf, "Invalid CQE address (%p %p)",
		  cqe->buf, buf);
	cr_assert(cqe->data == data, "Invalid CQE data");
	cr_assert(cqe->tag == tag, "Invalid CQE tag");
}

void validate_multi_recv_rx_event(struct fi_cq_tagged_entry *cqe, void
				  *context, size_t len, uint64_t flags,
				  uint64_t data, uint64_t tag)
{
	cr_assert(cqe->op_context == context, "CQE Context mismatch");
	cr_assert(cqe->len == len, "Invalid CQE length");
	cr_assert((cqe->flags & ~FI_MULTI_RECV) == flags,
		  "CQE flags mismatch (%#llx %#lx)",
		  (cqe->flags & ~FI_MULTI_RECV), flags);
	cr_assert(cqe->data == data, "Invalid CQE data");
	cr_assert(cqe->tag == tag, "Invalid CQE tag %#lx %#lx", cqe->tag, tag);
}

int mr_create_ext(size_t len, uint64_t access, uint8_t seed, uint64_t key,
		  struct fid_cntr *cntr, struct mem_region *mr)
{
	int ret;

	cr_assert_not_null(mr);

	if (len) {
		mr->mem = calloc(1, len);
		cr_assert_not_null(mr->mem, "Error allocating memory window");
	} else {
		mr->mem = 0;
	}

	for (size_t i = 0; i < len; i++)
		mr->mem[i] = i + seed;

	ret = fi_mr_reg(cxit_domain, mr->mem, len, access, 0, key, 0, &mr->mr,
			NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mr->mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind(ep) failed %d", ret);

	if (cntr) {
		ret = fi_mr_bind(mr->mr, &cntr->fid, FI_REMOTE_WRITE);
		cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind(cntr) failed %d",
			     ret);
	}

	return fi_mr_enable(mr->mr);
}

int mr_create(size_t len, uint64_t access, uint8_t seed, uint64_t key,
	      struct mem_region *mr)
{
	return mr_create_ext(len, access, seed, key, cxit_rem_cntr, mr);
}

void mr_destroy(struct mem_region *mr)
{
	fi_close(&mr->mr->fid);
	free(mr->mem);
}

