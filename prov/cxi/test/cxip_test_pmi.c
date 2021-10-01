/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018,2020-2021 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <criterion/criterion.h>
#include <pmi_utils.h>

#include "cxip_test_common.h"
#include "cxip_test_pmi.h"

#define STD_MCAST_ROOT	0
#define STD_MCAST_TIMEOUT 20

int cxit_ranks = 0;
int cxit_rank = 0;

int cxit_hwroot_rank = 0;
uint32_t cxit_mcast_ref = 0;
uint32_t cxit_mcast_id = 0;

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

	pmi_GetNumRanks(&cxit_ranks);
	pmi_GetRank(&cxit_rank);

	if (cxit_ranks < 2) {
		cr_skip_test("%d nodes insufficient to test collectives\n",
			     cxit_ranks);
	}
	printf("Running as rank %d of %d\n", cxit_rank, cxit_ranks);

	/* Wait for all nodes to get to this point */
	clock_gettime(CLOCK_REALTIME, &ts0);
	pmi_Barrier();
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "pmi_Barrier");

	/* Share NIC addresses */
	all_addrs = calloc(cxit_ranks, sizeof(cxit_ep_addr));
	clock_gettime(CLOCK_REALTIME, &ts0);
	pmi_Allgather(&cxit_ep_addr.raw, sizeof(cxit_ep_addr.raw), all_addrs);
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "pmi_Allgather");

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
	cxit_hwroot_rank = hwroot_rank;
	if (cxit_rank == cxit_hwroot_rank) {
		clock_gettime(CLOCK_REALTIME, &ts0);
		_get_mcast(model, hwroot_rank, timeout, mcast_ref, mcast_id);
		clock_gettime(CLOCK_REALTIME, &ts1);
		_print_delay(&ts0, &ts1,  __func__, "_get_mcast");
		printf("created mcast address = %d\n", *mcast_id);
	}

	/* Rank zero broadcasts, other ranks receive */
	clock_gettime(CLOCK_REALTIME, &ts0);
	pmi_Bcast(0, mcast_id, sizeof(*mcast_id));
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "pmi_Bcast");
}

/**
 * Destroy the multicast address created with the RESTful interface.
 *
 * @param mcast_ref : returned REST service multicast tree reference
 */
void cxit_LTU_destroy_coll_mcast(uint32_t mcast_ref)
{
	struct timespec ts0, ts1;
	cr_assert(ltu_init_count > 0, "Must cxit_LTU_create_universe()\n");
	if (cxit_rank == cxit_hwroot_rank) {
		clock_gettime(CLOCK_REALTIME, &ts0);
		_del_mcast(mcast_ref);
		clock_gettime(CLOCK_REALTIME, &ts1);
		_print_delay(&ts0, &ts1,  __func__, "_del_mcast");
	}
}

/**
 * Perform an out-of-band (LTU) Barrier.
 */
void cxit_LTU_barrier(void)
{
	struct timespec ts0, ts1;

	cr_assert(ltu_init_count > 0, "Must cxit_LTU_create_universe()\n");
	clock_gettime(CLOCK_REALTIME, &ts0);
	pmi_Barrier();
	clock_gettime(CLOCK_REALTIME, &ts1);
	_print_delay(&ts0, &ts1,  __func__, "pmi_Barrier");
}

void cxit_setup_distributed(void)
{
	pmi_Init(NULL, NULL, NULL);
	cxit_setup_enabled_ep();
	cxit_LTU_create_universe();
}

void cxit_teardown_distributed(void)
{
	cxit_LTU_destroy_universe();
	cxit_teardown_enabled_ep();
	pmi_Finalize();
}

void cxit_setup_multicast(void)
{
	struct cxip_coll_comm_key comm_key;
	struct fi_av_set_attr av_set_attr = {};
	size_t size;
	int ret;

	cxit_setup_distributed();

	cxit_LTU_create_coll_mcast(STD_MCAST_ROOT, STD_MCAST_TIMEOUT,
				   &cxit_mcast_ref, &cxit_mcast_id);

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
	cxit_LTU_destroy_coll_mcast(cxit_mcast_ref);
	cxit_LTU_barrier();
	cxit_teardown_distributed();
}
