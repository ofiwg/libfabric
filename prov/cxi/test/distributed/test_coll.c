/*
 * (c) Copyright 2022 Hewlett Packard Enterprise Development LP
 */

/**
 * Test the coll functions in a real environment.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <assert.h>
#include <malloc.h>
#include <time.h>
#include <ofi.h>
#include <cxip.h>
#include "pmi_utils.h"
#include "pmi_frmwk.h"

/* see cxit_trace_enable() in each test framework */
#define	TRACE CXIP_TRACE

int verbose = 0;

/* Signaling NaN generation, for testing.
 * Linux feature requires GNU_SOURCE.
 * This generates a specific sNaN value.
 */
static inline double cxip_snan64(void)
{
	return _bits2dbl(0x7ff4000000000000);
}

/* initialize nsecs timer structure */
static inline void _init_nsecs(struct timespec *tsp)
{
	clock_gettime(CLOCK_MONOTONIC, tsp);
}

/* return elapsed nsecs since initialized tsp */
static inline long _measure_nsecs(struct timespec *tsp)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	ts.tv_nsec -= tsp->tv_nsec;
	ts.tv_sec -= tsp->tv_sec;
	if (ts.tv_nsec < 0) {
		ts.tv_nsec += 1000000000L;
		ts.tv_sec -= 1;
	}
	return 1000000000L*ts.tv_sec + ts.tv_nsec;
}

static int _wait_for_join(int count)
{
	struct cxip_ep *ep;
	struct fid_cq *txcq, *rxcq;
	struct fid_eq *eq;
	struct fi_cq_err_entry cqd = {};
	struct fi_eq_err_entry eqd = {};
	uint32_t event;
	int ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);
	rxcq = &ep->ep_obj->coll.rx_cq->util_cq.cq_fid;
	txcq = &ep->ep_obj->coll.tx_cq->util_cq.cq_fid;
	eq = &ep->ep_obj->coll.eq->util_eq.eq_fid;

	do {
		/* This drives join completion via cxip_coll_progress_join */
		ret = fi_eq_read(eq, &event, &eqd, sizeof(eqd), 0);
		if (ret == -FI_EAVAIL) {
			TRACE("=== error available!\n");
			ret = fi_eq_readerr(eq, &eqd, 0);
			if (ret >= 0) {
				TRACE("  event   = %d\n", event);
				TRACE("  fid     = %p\n", eqd.fid);
				TRACE("  context = %p\n", eqd.context);
				TRACE("  data    = %lx\n", eqd.data);
				TRACE("  err     = %s\n",
					fi_strerror(-eqd.err));
				TRACE("  prov_err= 0x%04x\n", eqd.prov_errno);
				TRACE("  err_data= %p\n", eqd.err_data);
				TRACE("  err_size= %ld\n", eqd.err_data_size);
				TRACE("  readerr = %d\n", ret);
			}
			TRACE("===\n");
			break;
		}
		if (ret >= 0) {
			if (event == FI_JOIN_COMPLETE) {
				TRACE("saw FI_JOIN_COMPLETE\n");
				count--;
			} else {
				// TODO make noise
			}
		} else if (ret != -FI_EAGAIN) {
			// TODO make noise
			return ret;
		}

		/* TODO : is this needed here? */
		ret = fi_cq_read(rxcq, &cqd, sizeof(cqd));
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(rxcq, &cqd, sizeof(cqd));
			// TODO make noise
			break;
		}

		/* TODO : is this needed here? */
		ret = fi_cq_read(txcq, &cqd, sizeof(cqd));
		if (ret == -FI_EAVAIL) {
			// TODO make noise
			ret = fi_cq_readerr(txcq, &cqd, sizeof(cqd));
			break;
		}
	} while (count > 0);

	return FI_SUCCESS;
}

int _test_fi_join_collective(struct cxip_ep *cxip_ep,
			     fi_addr_t *fiaddrs, size_t size,
			     bool multicast)
{
	struct cxip_comm_key comm_key = {
		.keytype = (multicast) ? COMM_KEY_NONE : COMM_KEY_UNICAST,
		.ucast.hwroot_idx = 0
	};
	struct fi_av_set_attr attr = {
		.flags=FI_UNIVERSE,
		.comm_key_size=sizeof(comm_key),
		.comm_key=(uint8_t *)&comm_key
	};
	struct fid_av_set *set = NULL;
	struct fid_mc *mc = NULL;
	struct {
		int placeholder;
	} context;
	int ret;

	ret = fi_av_set(cxit_av, &attr, &set, NULL);
	if (ret)
		goto done;

	ret = fi_join_collective(cxit_ep, FI_ADDR_NOTAVAIL,
				 set, 0L, &mc, &context);
	if (ret)
		goto done;

	TRACE("wait for join\n");
	ret = _wait_for_join(1);
	if (ret)
		goto done;
	TRACE("joined\n");

done:
	if (mc)
		fi_close(&mc->fid);
	if (set)
		fi_close(&set->fid);
	return ret;
}

int _test_fi_barrier(struct cxip_ep *cxip_ep,
		     fi_addr_t *fiaddrs, size_t size)
{
	struct cxip_comm_key comm_key = {
		.keytype = COMM_KEY_UNICAST,
		.ucast.hwroot_idx = 0
	};
	struct fi_av_set_attr attr = {
		.flags=FI_UNIVERSE,
		.comm_key_size=sizeof(comm_key),
		.comm_key=(uint8_t *)&comm_key
	};
	struct fid_av_set *set = NULL;
	struct fid_mc *mc = NULL;
	struct {
		int placeholder;
	} context;
	int ret;

	ret = fi_av_set(cxit_av, &attr, &set, NULL);
	if (ret)
		goto done;

	ret = fi_join_collective(cxit_ep, FI_ADDR_NOTAVAIL,
				 set, 0L, &mc, &context);
	if (ret)
		goto done;

	ret = _wait_for_join(1);
	if (ret)
		goto done;
	TRACE("joined\n");

	if (pmi_rank == 1) {
		TRACE("sleeping\n");
		sleep(1);
	}
	fi_barrier(cxit_ep, (fi_addr_t )mc, &context);
	//TRACE("barrier done in %ld nsec\n", _measure_nsecs(&ts));
	// TODO: need to spin on something...

done:
	if (mc)
		fi_close(&mc->fid);
	if (set)
		fi_close(&set->fid);
	return ret;
}

int _test_fi_broadcast(struct cxip_ep *cxip_ep,
		       fi_addr_t *fiaddrs, size_t size)
{
	struct cxip_comm_key comm_key = {
		.keytype = COMM_KEY_UNICAST,
		.ucast.hwroot_idx = 0
	};
	struct fi_av_set_attr attr = {
		.flags=FI_UNIVERSE,
		.comm_key_size=sizeof(comm_key),
		.comm_key=(uint8_t *)&comm_key
	};
	struct fid_av_set *set = NULL;
	struct fid_mc *mc = NULL;
	struct {
		int placeholder;
	} context;
	uint64_t data;
	int ret;

	ret = fi_av_set(cxit_av, &attr, &set, NULL);
	if (ret)
		goto done;

	ret = fi_join_collective(cxit_ep, FI_ADDR_NOTAVAIL,
				 set, 0L, &mc, &context);
	if (ret)
		goto done;

	ret = _wait_for_join(1);
	if (ret)
		goto done;
	TRACE("joined\n");

	if (pmi_rank == 1) {
		TRACE("sleeping\n");
		sleep(1);
	}
	fi_broadcast(cxit_ep, &data, sizeof(data), NULL, (fi_addr_t )mc,
			0, FI_UINT64, 0, &context);
	// TODO: need to spin on something...

done:
	if (mc)
		fi_close(&mc->fid);
	if (set)
		fi_close(&set->fid);
	return ret;
}

const char *testnames[] = {
	"test  0: no-op",
	"test  2: join collective (UNICAST)",
	"test  3: join collective (MULTICAST)",
	"test  4: barrier",
	NULL
};

int usage(int ret)
{
	int i;

	pmi_log0("Usage: test_coll [-hvV]\n"
		 "                 [-t testno[,testno...]]\n"
		 "\n");
	for (i = 0; testnames[i]; i++)
		pmi_log0("%s\n", testnames[i]);

	return ret;
}

/* scan for integers in -t option */
static inline char *scanint(char *ptr, int *val)
{
	char *p = ptr;
	while (*ptr >= '0' && *ptr <= '9')
		ptr++;
	*val = atoi(p);
	return ptr;
}

#define	TEST(n)		(1 << n)
#define	STDMSG(ret)	((ret > 0) ? "SKIP" : ((ret) ? "FAIL" : "ok"))

int main(int argc, char **argv)
{
	bool trace_enabled = false;
	struct cxip_ep *cxip_ep;
	fi_addr_t *fiaddrs = NULL;
	size_t size = 0;
	int errcnt = 0;
	int tstcnt = 0;
	struct timespec ts;

	uint64_t testmask;
	const char *testname;
	char opt;
	int ret;

	testmask = -1;	// run all tests

	while ((opt = getopt(argc, argv, "hvVt:")) != -1) {
		char *str, *s, *p;
		int i, j;

		switch (opt) {
		case 'h':
			return usage(0);
		case 'v':
			verbose = true;
			break;
		case 'V':
			trace_enabled = true;
			break;
		case 't':
			testmask = 0;
			str = optarg;
			i = j = 0;
			while (*str) {
				s = str;
				while (*str && *str != ',')
					str++;
				if (*str)
					*str++ = 0;
				p = s;
				while (*p && *p != '-')
					p++;
				i = atoi(s);
				j = (*p) ? atoi(++p) : i;
				while (i <= j)
					testmask |= 1 << i++;
			}
			break;
		default:
			return usage(1);
		}
	}

	setenv("PMI_MAX_KVS_ENTRIES", "5000", 1);
	pmi_Init();
	if (pmi_check_env(4))
		return -1;

	ret = pmi_init_libfabric();
	if (pmi_errmsg(ret, "pmi_init_libfabric()\n"))
		return ret;

	cxit_trace_enable(trace_enabled);
	TRACE("==== tracing enabled offset %d\n", pmi_rank + cxit_trace_offset);

	/* pmi_init_libfabric provides the basics for us */
	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);

	/* always start with FI_UNIVERSE */
	ret = pmi_populate_av(&fiaddrs, &size);
	errcnt += !!ret;
	if (pmi_errmsg(ret, "pmi_populate_av()\n"))
		goto done;

	if (testmask & TEST(0)) {
		testname = testnames[0];
		TRACE("======= %s\n", testname);
		ret = 0;
		tstcnt += 1;
		errcnt += !!ret;
		pmi_log0("%4s %s\n", STDMSG(ret), testname);
		pmi_Barrier();
	}

	if (testmask & TEST(1)) {
		testname = testnames[0];
		TRACE("======= %s\n", testname);
		ret = _test_fi_join_collective(
			cxip_ep, fiaddrs, size, false);
		tstcnt += 1;
		errcnt += !!ret;
		pmi_log0("%4s %s\n", STDMSG(ret), testname);
		pmi_Barrier();
	}

	if (testmask & TEST(2)) {
		testname = testnames[1];
		TRACE("======= %s\n", testname);
		TRACE("CURL addr='%s'\n", cxip_env.coll_fabric_mgr_url);
		ret = 0;
		if (cxip_env.coll_fabric_rest_api) {
			ret = _test_fi_join_collective(
				cxip_ep, fiaddrs, size, true);
		}
		tstcnt += 1;
		errcnt += !!ret;
		pmi_log0("%4s %s\n", STDMSG(ret), testname);
		pmi_Barrier();
	}

	if (testmask & TEST(3)) {
		testname = testnames[2];
		TRACE("======= %s\n", testname);

		_init_nsecs(&ts);
		ret = _test_fi_barrier(cxip_ep, fiaddrs, size);
		tstcnt += 1;
		errcnt += !!ret;
		pmi_log0("%4s %s\n", STDMSG(ret), testname);
		pmi_Barrier();
	}

	if (testmask & TEST(4)) {
		testname = testnames[3];
		TRACE("======= %s\n", testname);
		ret = _test_fi_broadcast(cxip_ep, fiaddrs, size);
		tstcnt += 1;
		errcnt += !!ret;
		pmi_log0("%4s %s\n", STDMSG(ret), testname);
		pmi_Barrier();
	}

done:
	pmi_log0("%2d tests run, %d failures\n", tstcnt, errcnt);
	free(fiaddrs);
	pmi_free_libfabric();
	pmi_Finalize();

	return !!errcnt;
}
