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
#include <pmi_utils.h>
#include <pmi_frmwk.h>

/* abbreviation */
#define	trc CXIP_TRACE

int verbose = 0;

int _test_fi_join_collective(struct cxip_ep *cxip_ep,
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
	struct fid_eq *eq;
	struct cxip_coll_mc *mc_obj;
	struct {
		int placeholder;
	} context;
	uint32_t event;
	int ret;

	trc("..fi_av_set\n");
	ret = fi_av_set(cxit_av, &attr, &set, NULL);
	if (ret)
		goto done;

	trc("..fi_join_collective\n");
	ret = fi_join_collective(cxit_ep, FI_ADDR_NOTAVAIL,
				 set, 0L, &mc, &context);
	if (ret)
		goto done;

	trc("..poll\n");
	eq = &cxip_ep->ep_obj->eq->util_eq.eq_fid;
	do {
		usleep(100);
		ret = fi_eq_read(eq, &event, NULL, 0, 0);
		trc("..ret=%s\n", fi_strerror(-ret));
	} while (ret == -FI_EAGAIN);
	trc("event = %d\n", event);

	mc_obj = container_of(mc, struct cxip_coll_mc, mc_fid.fid);

done:
	if (mc)
		fi_close(&mc->fid);
	if (set)
		fi_close(&set->fid);
	return ret;
}


const char *testnames[] = {
	"test  0: whatever",
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

#define	TEST(n)	(1 << n)

int main(int argc, char **argv)
{
	struct cxip_ep *cxip_ep;
	fi_addr_t *fiaddrs = NULL;
	size_t size = 0;
	int errcnt = 0;
	int tstcnt = 0;

	uint64_t testmask;
	const char *testname;
	char opt;
	int ret;

	setenv("PMI_MAX_KVS_ENTRIES", "5000", 1);
	ret = pmi_init_libfabric();
	if (pmi_errmsg(ret, "pmi_init_libfabric()\n"))
		return ret;

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
			pmi_trace_enable(true);
			trc("======= tracing enabled\n");
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

	/* tests require at least 4 ranks */
	if (pmi_errmsg(pmi_numranks < 4, "requires at least 4 nodes\n"))
		return -FI_EINVAL;

	/* pmi framework provids the basics for us */
	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);

	/* always start with FI_UNIVERSE */
	ret = pmi_populate_av(&fiaddrs, &size);
	if (pmi_errmsg(ret, "pmi_populate_av()\n"))
		return 1;

	if (testmask & TEST(0)) {
		testname = testnames[0];
		trc("======= %s\n", testname);
		ret = _test_fi_join_collective(cxip_ep, fiaddrs, size);
		tstcnt += 1;
		errcnt += !!ret;
		trc("rank %2d result = %s\n", pmi_rank, fi_strerror(-ret));
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	pmi_log0("%2d tests run, %d failures\n", tstcnt, errcnt);
	trc("Finished test run, cleaning up\n");
	free(fiaddrs);
	pmi_free_libfabric();
	return !!errcnt;
}