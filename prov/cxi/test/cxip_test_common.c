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
struct fi_cq_attr cxit_tx_cq_attr = { .format = FI_CQ_FORMAT_TAGGED };
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
int cxit_ranks = 0;
int cxit_rank = 0;

#define STD_MCAST_ROOT	0
#define STD_MCAST_TIMEOUT 20
uint32_t cxit_mcast_ref = 0;
uint32_t cxit_mcast_id = 0;
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

/**
 * Guarantee that LTU has been initialized only once. This should persist for
 * the entire job run, because LTU is not friendly about being re-initialized.
 */
static int ltu_init_count = 0;

/**
 * Search for a tag value in a json structure.
 *
 * This only works because all our JSON tags are unique, and the only ones we
 * care about are associated with integer values. Use the QUOTED() macro to
 * ensure supplied tags are properly delimited for string search.
 *
 * @param json : json structure
 * @param tag : tag to search for
 *
 * @return int : value associated with tag
 */
static int _get_tag_value(const char *json, const char *tag)
{
	char *ptr;

	ptr = strstr(json, tag);
	if (!ptr)
		return 0;
	while (*(++ptr) && *ptr != ':');
	while (*(++ptr) && *ptr == ' ');
	return atoi(ptr);
}
#define	QUOTED(str)	"\"" str "\""

/**
 * Fairly generic RESTful POST/DELETE handler. Uses popen(), which forks a shell
 * to execute the curl command, and then collects the stdout.
 *
 * If the REST server is down, this returns a NULL pointer. Certain errors can
 * result in a lot of data.
 *
 * Returned string must be freed.
 *
 * @param op : "POST" or "DELETE"
 * @param json : JSON data to send with operation
 * @param http : target of server
 *
 * @return char* : result from the curl command
 */
static char *_issue_curl(const char *op, const char *json, const char *http)
{
	const char *cmdfmt =
		"curl -s -H 'Content-Type: application/json'"
		" -X %s -d '%s' %s";
	FILE *fid;
	char cmd[1024];
	char *rsp = NULL;
	int siz = 0;
	int len;

	len = snprintf(cmd, sizeof(cmd), cmdfmt, op, json, http);
	cr_assert(len > 0 && len < sizeof(cmd), "Invalid CURL command\n");

	fid = popen(cmd, "r");
	if (!fid) {
		printf(">>> CURL command failed <<<\n");
		return NULL;
	}

	/* Expand array as necessary. Certain errors produce a lot of output. */
	rsp = NULL;
	siz = 0;
	len = 0;
	while (!(siz - len)) {
		if (!(siz - len)) {
			siz += 1024;
			rsp = realloc(rsp, siz);
		}
		len += fread(&rsp[len], 1, (siz - len), fid);
	}
	pclose(fid);
	/* Ensure string is terminated */
	rsp[len] = '\0';

	if (!len)
		printf(">>> Configuration server is not running <<<\n");
	return rsp;
}

/**
 * Acquire a multicast address using RESTful service.
 *
 * Only rank 0 is allowed to do this, since result must be broadcast to all the
 * nodes in the collective, and only rank 0 can broadcast using LTU.
 *
 * @param model : multicast model
 * @param hwroot : hw root rank
 * @param timeout : hw collective timeout
 * @param mcastref : returned multicast identifier
 * @param mcast_id : returned multicast address
 *
 * @return true if address acquired, false otherwise
 */
static void _get_mcast(const char *model, int hwroot, int timeout,
		       uint32_t *mcastref, uint32_t *mcast_id)
{
	const char *http = "http://10.1.1.1:5000/config";
	const char *jsonfmt = "{"
		"\"sysenv_cfg_path\":\"%s\","
		"\"params_ds\":[{"
		"\"mcast_id\":%d,"
		"\"root_port_idx\":%d,"
		"\"softportals\":0,"
		"\"timeout\":%d"
		"}]}";
	char json[1024];
	char *rsp;
	int ret;

	if (cxit_rank) {
		*mcastref = -1;
		*mcast_id = -1;
		return;
	}

	ret = snprintf(json, sizeof(json), jsonfmt,
		       model, *mcast_id, hwroot, timeout);
	cr_assert(ret < sizeof(json));

	rsp = _issue_curl("POST", json, http);
	if (rsp && strlen(rsp) > 0) {
		*mcastref = (uint32_t)_get_tag_value(rsp, QUOTED("id"));
		*mcast_id = (uint32_t)_get_tag_value(rsp, QUOTED("mcast_id"));
		free(rsp);
	}
}

/**
 * Delete a multicast address, using RESTful service.
 *
 * Only rank 0 is allowed to do this.
 *
 * @param mcastref : multicast ref value assocated with multicast address
 */
static void _del_mcast(uint32_t mcastref)
{
	const char *http = "http://10.1.1.1:5000/config";
	const char *jsonfmt = "{\"id\": %d}";
	char json[1024];
	char *rsp;
	int ret;

	if (cxit_rank)
		return;

	ret = snprintf(json, sizeof(json), jsonfmt, mcastref);
	cr_assert(ret < sizeof(json));

	rsp = _issue_curl("DELETE", json, http);
	free(rsp);
}

/**
 * Print elapsed time between two clock times.
 *
 * @param ts0 : initial clock time
 * @param ts1 : final clock time
 * @param func : function name calling this
 * @param tag : tag to identify the operation
 *
 * @return double delay in seconds
 */
double _print_delay(struct timespec *ts0, struct timespec *ts1,
		    const char *func, const char *tag)
{
	if (ts1->tv_nsec < ts0->tv_nsec) {
		ts1->tv_sec--;
		ts1->tv_nsec += 1000000000;
	}
	ts1->tv_nsec -= ts0->tv_nsec;
	ts1->tv_sec -= ts0->tv_sec;
	if (func && tag)
		printf("%s: %s %ld.%09ld\n", func, tag, ts1->tv_sec, ts1->tv_nsec);

	return (double)ts1->tv_sec + (double)ts1->tv_nsec / 1000000000.0;
}

/**
 * Remove all addresses from the test fi_av.
 */
void cxit_LTU_destroy_universe(void)
{
	fi_addr_t *all_fiaddrs;
	int i;

	cr_assert(ltu_init_count > 0, "Must cxit_LTU_create_universe()\n");

	all_fiaddrs = calloc(cxit_ranks, sizeof(fi_addr_t));
	for (i = 0; i < cxit_ranks; i++)
		all_fiaddrs[i] = (fi_addr_t)i;
	fi_av_remove(cxit_av, all_fiaddrs, cxit_ranks, 0);
	free(all_fiaddrs);
}

/**
 * Populate the test fi_av with NIC addresses of all devices in job, using
 * out-of-band (LTU) Allgather.
 *
 * This must be done once before performing any collective or distributed test,
 * and can only be done once. It populates a presumed-empty fi_av with all of
 * the NIC addresses in the job, in rank-order, i.e. fi_av_lookup(0) will
 * provide the NIC address of rank 0.
 *
 */
void cxit_LTU_create_universe(void)
{
	struct timespec ts0, ts1;
	struct cxip_addr *all_addrs, addr;
	size_t siz;
	int i, ret;

	setlinebuf(stdout);

	cr_assert(ltu_init_count == 0, "May gather universe only once\n");
	ltu_init_count = 1;

	ltu_pm_Job_size(&cxit_ranks);
	ltu_pm_Rank(&cxit_rank);
	if (cxit_ranks < 2) {
		cr_skip_test("%d nodes insufficient to test collectives\n",
			     cxit_ranks);
	}
	printf("Running as rank %d of %d\n", cxit_rank, cxit_ranks);


	/* Wait for all nodes to get to this point */
	clock_gettime(CLOCK_REALTIME, &ts0);
	ltu_pm_Barrier();
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "ltu_pm_Barrier");

	/* Share NIC addresses */
	all_addrs = calloc(cxit_ranks, sizeof(cxit_ep_addr));
	clock_gettime(CLOCK_REALTIME, &ts0);
	ltu_pm_Allgather(&cxit_ep_addr.raw, sizeof(cxit_ep_addr.raw), all_addrs);
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "ltu_pm_Allgather");

	cr_assert(all_addrs[cxit_rank].raw == cxit_ep_addr.raw);

	/* Clear space in addresses 0-(cxit_ranks-1) */
	cxit_LTU_destroy_universe();

	/* Build the fi_av list */
	ret = fi_av_insert(cxit_av, (void *)all_addrs, cxit_ranks, NULL, 0, NULL);
	cr_assert(ret == cxit_ranks);

	/* Validate mapping */
	for (i = 0; i < cxit_ranks; i++) {
		siz = sizeof(addr);
		ret = fi_av_lookup(cxit_av, (fi_addr_t)i, &addr, &siz);
		cr_assert(ret == 0);
		cr_assert(addr.raw != 0);
		if (i == cxit_rank)
			cr_assert(addr.raw == cxit_ep_addr.raw);
		else
			cr_assert(addr.raw != cxit_ep_addr.raw);
	}

	free(all_addrs);
}

/**
 * Create a multicast address using a RESTful interface, and distribute the
 * address to all nodes using out-of-band (LTU) Bcast.
 *
 * This can be called multiple times to create multiple mcast addresses.
 *
 * LIMITATIONS: the prototype REST interface cannot generate arbitrary multicast
 * trees, but multiple overlapping trees are possible, and the hwroot can be
 * moved around, and different timeouts can be applied.
 *
 * @param hwroot_rank : rank of the node serving as HW root for mcast tree
 * @param timeout : timeout of reduction engines in seconds
 * @param mcast_ref : return REST service multicast tree reference
 * @param mcast_id : return REST service multicast id (address)
 */
void cxit_LTU_create_coll_mcast(int hwroot_rank, int timeout,
				uint32_t *mcast_ref, uint32_t *mcast_id)
{
	struct timespec ts0, ts1;
	const char *model;

	cr_assert(ltu_init_count > 0, "Must cxit_LTU_create_universe()\n");

	switch (cxit_ranks) {
	case 2:
		model = "syscfgs/rosetta2-emu-build.sysconfig.yaml";
		break;
	case 4:
		model = "syscfgs/rosetta4-emu-build.sysconfig.yaml";
		break;
	default:
		cr_assert(true, "no model for %d node collective\n", cxit_ranks);
		return;
	}

	/* Create the mcast address */
	_get_mcast(model, hwroot_rank, timeout, mcast_ref, mcast_id);

	/* Rank zero broadcasts, other ranks receive */
	clock_gettime(CLOCK_REALTIME, &ts0);
	ltu_pm_Bcast(mcast_id, sizeof(*mcast_id));
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "ltu_pm_Bcast");
}

/**
 * Destroy the multicast address created with the RESTful interface.
 *
 * @param mcast_ref : returned REST service multicast tree reference
 */
void cxit_LTU_destroy_coll_mcast(uint32_t mcast_ref)
{
	cr_assert(ltu_init_count > 0, "Must cxit_LTU_create_universe()\n");
	_del_mcast(mcast_ref);
}

/**
 * Perform an out-of-band (LTU) Barrier.
 */
void cxit_LTU_barrier(void)
{
	struct timespec ts0, ts1;

	cr_assert(ltu_init_count > 0, "Must cxit_LTU_create_universe()\n");
	clock_gettime(CLOCK_REALTIME, &ts0);
	ltu_pm_Barrier();
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "ltu_pm_Barrier");
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

void cxit_setup_distributed(void)
{
	ltu_pm_Init(NULL, NULL);
	cxit_setup_enabled_ep();
	cxit_LTU_create_universe();
}

void cxit_teardown_distributed(void)
{
	cxit_LTU_destroy_universe();
	cxit_teardown_enabled_ep();
	ltu_pm_Finalize();
}

/* Note: set these during rapid development to reuse existing multicast addr  */
static int static_mcast_id = 0;
static int static_mcast_ref = 0;
static bool static_mcast_keep = false;

void cxit_setup_multicast(void)
{
	struct cxip_coll_comm_key comm_key;
	struct fi_av_set_attr av_set_attr = {};
	size_t size;
	int ret;

	cxit_setup_distributed();
	cr_skip_test("Environment not suitable for collectives\n");

	if (!static_mcast_id) {
		cxit_LTU_create_coll_mcast(STD_MCAST_ROOT, STD_MCAST_TIMEOUT,
					   &cxit_mcast_ref, &cxit_mcast_id);
	} else {
		cxit_mcast_id = static_mcast_id;
		cxit_mcast_ref = static_mcast_ref;
	}
	printf("MCAST_ID  = %d\n", cxit_mcast_id);
	printf("MCAST_REF = %d\n", cxit_mcast_ref);

	size = cxip_coll_init_mcast_comm_key(&comm_key, cxit_mcast_ref,
					     cxit_mcast_id, STD_MCAST_ROOT);
	av_set_attr.flags = FI_UNIVERSE;
	av_set_attr.comm_key_size = size;
	av_set_attr.comm_key = (uint8_t *)&comm_key;
	ret = fi_av_set(cxit_av, &av_set_attr, &cxit_av_set, NULL);
	cr_assert_not_null(cxit_av_set);

	ret = cxip_join_collective(cxit_ep, FI_ADDR_NOTAVAIL, cxit_av_set, 0,
				   &cxit_mc, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "cxip_join_collective() failed %d", ret);

	cxit_LTU_barrier();
}

void cxit_teardown_multicast(void)
{
	fi_close(&cxit_mc->fid);
	fi_close(&cxit_av_set->fid);
	if (!static_mcast_keep) {
		cxit_LTU_destroy_coll_mcast(cxit_mcast_ref);
	}
	cxit_LTU_barrier();
	cxit_teardown_distributed();
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

