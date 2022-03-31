/*
 * (c) Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/**
 * Test the zbcoll functions in a real environment.
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

/* convert delays to nsecs */
#define	nUSEC(n)	(n * 1000L)
#define nMSEC(n)	(n * 1000000L)
#define	nSEC(n)		(n * 1000000000L)

int verbose = false;

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

/* introduce random jitter delay into operations per rank */
void _jitter(int usec)
{
	static unsigned int seed = 0;
	if (!seed)
		seed = rand() + pmi_rank + 1;
	if (usec) {
		usec = rand_r(&seed) % usec;
		trc("_jitter delay = %d usec\n", usec);
		usleep(usec);
	}
}

/* utility to do a primitive wait for send completion based on counters */
static int _send_wait(struct cxip_zbcoll_obj *zb, int sndcnt, int rcvcnt)
{
	struct cxip_ep_obj *ep_obj = zb->ep_obj;
	uint32_t dsc, err, ack, rcv;
	struct timespec ts;
	long nsecs = 0L;

	_init_nsecs(&ts);
	do {
		cxip_zbcoll_progress(ep_obj);
		cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
		if (err || dsc)
			break;
		if (ack >= sndcnt && rcv >= rcvcnt)
			break;
		nsecs = _measure_nsecs(&ts);
	} while (nsecs < nMSEC(100));
	trc("ns=%ld dsc=%d err=%d ack=%d rcv=%d rc=%d\n",
		   nsecs, dsc, err, ack, rcv, zb->error);
	if (nsecs >= nMSEC(100)) {
		trc("TIMEOUT\n");
		return 1;
	}
	if (err || dsc || ack < sndcnt || rcv < rcvcnt) {
		trc("TRANSPORT FAILURE\n");
		return 1;
	}
	if (zb->error) {
		trc("STATE FAILURE\n");
		return 1;
	}
	return 0;
}

/* send a single packet from node to node, and wait for completion */
int _test_send_to_dest(struct cxip_ep_obj *ep_obj,
		       size_t size, fi_addr_t *fiaddrs,
		       int src, int dst, uint64_t payload)
{
	struct cxip_zbcoll_obj *zb;
	int grp_rank;
	int sndcnt, rcvcnt;
	int i, ret;

	ret = cxip_zbcoll_alloc(ep_obj, size, fiaddrs, false, &zb);
	if (pmi_errmsg(ret, "%s: cxip_zbcoll_alloc()\n", __func__))
		return ret;

	grp_rank = zb->state[0].grp_rank;

	ep_obj->zbcoll.disable = true;
	zb->grpid = 0;
	cxip_zbcoll_reset_counters(ep_obj);
	if (src < 0 && dst < 0) {
		/* every source to every destination */
		sndcnt = size;
		rcvcnt = size;
		for (i = 0; i < size; i++)
			cxip_zbcoll_send(zb, grp_rank, i, payload);
	} else if (src < 0) {
		/* every source sends to one destination */
		sndcnt = 1;
		rcvcnt = (dst == grp_rank) ? size : 0;
		cxip_zbcoll_send(zb, grp_rank, dst, payload);
	} else if (dst < 0 && src == grp_rank) {
		/* this source sends to every destination */
		sndcnt = size;
		rcvcnt = 1;
		for (i = 0; i < size; i++)
			cxip_zbcoll_send(zb, grp_rank, i, payload);
	} else if (dst < 0) {
		/* some other src to every destination */
		sndcnt = 0;
		rcvcnt = 1;
	} else if (grp_rank == src) {
		/* this source to a destination */
		sndcnt = 1;
		rcvcnt = (grp_rank == dst) ? 1 : 0;
		cxip_zbcoll_send(zb, grp_rank, dst, payload);
	} else if (grp_rank == dst) {
		/* some other source to this destination */
		sndcnt = 0;
		rcvcnt = 1;
	} else {
		/* not participating */
		sndcnt = 0;
		rcvcnt = 0;
	}
	ret = _send_wait(zb, sndcnt, rcvcnt);
	ep_obj->zbcoll.disable = false;
	cxip_zbcoll_free(zb);

	return ret;
}

/* normal utility to wait for collective completion, returns coll error */
static int _coll_wait(struct cxip_zbcoll_obj *zb, long nsec_wait)
{
	uint32_t dsc, err, ack, rcv;
	struct timespec ts;
	long nsecs = 0L;

	if (!zb)
		return -FI_EINVAL;
	_init_nsecs(&ts);
	do {
		cxip_zbcoll_progress(zb->ep_obj);
		cxip_zbcoll_get_counters(zb->ep_obj, &dsc, &err, &ack, &rcv);
		/* this waits for a software completion */
		if (zb->error || !zb->busy)
			break;
		nsecs = _measure_nsecs(&ts);
	} while (nsecs < nsec_wait);
	trc("ns=%ld dsc=%d err=%d ack=%d rcv=%d\n",
		   nsecs, dsc, err, ack, rcv);
	if (nsecs >= nsec_wait) {
		trc("TIMEOUT\n");
		return -FI_ETIMEDOUT;
	}
	/* return the software error code -- may be -FI_EAGAIN */
	trc("return code = %d\n", zb->error);
	return zb->error;
}

/**
 * @brief Internal workhorse to create zb object and get group id.
 *
 * This will return FI_SUCCESS, delete the zb object (if any), and do nothing if
 * this endpoint is not in group.
 *
 * This creates a zb object as necessary.
 *
 * This destroys the zb object on any error.
 *
 * This call blocks for up to 100 msec waiting for completion.
 *
 * Once the cxip_zbcoll_getgroup() engages,
 *
 * @param ep_obj : endpoint
 * @param size   : number of NIDs in group
 * @param fiaddrs: fiaddrs in group
 * @param zbp    : return pointer to zb object (may be non-NULL)
 * @return int   : libfabric error code
 */
int _getgroup(struct cxip_ep_obj *ep_obj,
	      size_t size, fi_addr_t *fiaddrs,
	      struct cxip_zbcoll_obj **zbp)
{
	int ret;

	/* need a zbcoll object for this */
	if (!zbp)
		return -FI_EINVAL;
	if (!*zbp) {
		ret = cxip_zbcoll_alloc(ep_obj, size, fiaddrs, false, zbp);
		if (ret == -FI_EADDRNOTAVAIL) {
			trc("=== COMPLETED SKIP\n");
			return FI_SUCCESS;
		}
		if (pmi_errmsg(ret, "%s: cxip_zbcoll_alloc()\n", __func__))
			goto out;
	}

	/* getgroup collective */
	do {
		ret = cxip_zbcoll_getgroup(*zbp);
		if (pmi_errmsg(ret, "%s: cxip_zbcoll_getgroup()\n", __func__))
			goto out;
		/* Returns a collective completion error */
		ret = _coll_wait(*zbp, nMSEC(100));
	} while (ret == -FI_EAGAIN);

	/* clean up after error */
	if (ret)
		goto out;

	trc("=== COMPLETED ZBCOLL %d ret=%d\n", (*zbp)->grpid, ret);
	return FI_SUCCESS;

out:
	cxip_zbcoll_free(*zbp);
	*zbp = NULL;
	return ret;
}

/* detect overt getgroup errors */
int _check_getgroup_errs(struct cxip_zbcoll_obj *zb, int exp_grpid)
{
	return (pmi_errmsg(!zb, "zb == NULL") ||
		pmi_errmsg(zb->error, "zb->error == %d\n", zb->error) ||
		pmi_errmsg(zb->grpid != exp_grpid, "zb->grpid=%d exp=%d\n",
	    	           zb->grpid, exp_grpid));
}

/* rotate array[size] by rot positions */
void _rotate_array32(uint32_t *array, size_t size, int rot)
{
	uint32_t *copy;
	uint32_t i, j;

	copy = calloc(size, sizeof(uint32_t));
	memcpy(copy, array, size*sizeof(uint32_t));
	for (i = 0; i < size; i++) {
		j = (i + rot) % size;
		array[i] = copy[j];
	}
	free(copy);
}

/* shuffle array[size] randomly */
void _shuffle_array32(uint32_t *array, size_t size)
{
	uint32_t i, j, t;

	for (i = 0; i < size-1; i++) {
		j = i + (rand() / ((RAND_MAX / (size - i)) + 1));
		t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
}

/**
 * @brief Perform multiple concurrent getgroup operations.
 *
 * Parametrized test to thoroughly exercise getgroup edge conditions.
 *
 * This sets up to acquire 'nruns' group IDs.
 *
 * On each run it will only use 'naddrs' of the 'size' endpoints. If the default
 * value of -1 is used, each run will use a random number between 1 and 'size'.
 *
 * Prior to each run, the list of addresses is rotated. If 'rot' is -1, the list
 * is randomly shuffled. The purpose of rotation is to guarantee disjoint sets
 * of NIDs can be created. For instance, if you have 16 addresses (size=16), and
 * you set nruns=naddrs=rot=4, then all of the groups will be disjoint.
 *
 * This imposes a random jitter of up to 'usec' microseconds on each node, to
 * break up synchronous behavior among the nodes, and exaggerate race
 * conditions.
 *
 * This presumes a shared file system across all of the nodes under srun, and
 * writes results to files named using the rank number, overwriting old files
 * from prior runs. The rank 0 node will complete the test by reading back all
 * of the files and processing them to ensure correct behavior.
 *
 * @param ep_obj : endpoint object
 * @param size   : total number of NID addresses
 * @param fiaddrs: all NID addresses
 * @param nruns  : nruns of concurrency
 * @param naddrs : number of NIDs to use (-1 implies random)
 * @param rot    : nid rotations per run (-1 implies shuffle)
 * @param usec   : usec jitter to impose randomly
 * @return int   : 0 on success, or error code
 */
int _multigroup(struct cxip_ep_obj *ep_obj, size_t size, fi_addr_t *fiaddrs,
		int nruns, int naddrs, int rot, int usec)
{
	char fnam[256];
	FILE *fd;
	struct cxip_zbcoll_obj **zb;
	fi_addr_t *addrs;
	uint32_t *index;
	uint32_t **rows;
	uint32_t *length;
	int *grps;
	bool shuffle = false;
	uint32_t dsc, err, ack, rcv;
	int i, j, ret;

	cxip_zbcoll_reset_counters(ep_obj);

	ret = 0;
	if (nruns < 0)
		nruns = size;
	if (nruns > cxip_zbcoll_max_grps(false))
		nruns = cxip_zbcoll_max_grps(false);
	if (naddrs > size)
		naddrs = size;

	addrs = calloc(size, sizeof(fi_addr_t));// indices converted to addrs
	index = calloc(size, sizeof(uint32_t));	// nid indices (easier to read)
	for (j = 0; j < size; j++)
		index[j] = j;

	/* rows   : getgroup requests, list of nids involved
	 * length : number of addrs in each getgroup request, is <= size
	 * grps   : resulting group ID for each getgroup request
	 * zb     : zb_coll object for each getgroup request
	 */
	rows = calloc(nruns, sizeof(void *));
	length = calloc(nruns, sizeof(uint32_t));
	grps = calloc(nruns, sizeof(int));
	zb = calloc(nruns, sizeof(void *));
	for (i = 0; i < nruns; i++) {
		/* -1 means random sizes */
		if (naddrs < 0) {
			length[i] = 1 + (rand() % (size - 1));
		} else {
			length[i] = naddrs;
		}
		/* -1 means shuffle targets */
		if (rot < 0) {
			rot = 1;
			shuffle = true;
		}
		/* copy shuffled indices into row */
		rows[i] = calloc(length[i], sizeof(uint32_t));
		_rotate_array32(index, size, rot);
		if (shuffle)
			_shuffle_array32(index, size);
		memcpy(rows[i], index, length[i]*sizeof(uint32_t));
	}

	/* create zb with grpid, in same group order across nodes */
	for (i = 0; i < nruns; i++) {
		for (j = 0; j < length[i]; j++)
			addrs[j] = fiaddrs[rows[i][j]];
		_jitter(usec);
		ret = _getgroup(ep_obj, length[i], addrs, &zb[i]);
		if (pmi_errmsg(ret, "FAILURE getgroup %d\n", i)) {
			trc("FAILURE getgroup %d\n", i);
			goto done;
		}
		grps[i] = (zb[i]) ? zb[i]->grpid : -1;
	}

	/* need to compare each node result with other, write to file */
	sprintf(fnam, "grpid%d", pmi_rank);
	fd = fopen(fnam, "w");

	cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
	fprintf(fd, "%d %d %d %d\n", dsc, err, ack, rcv);
	for (i = 0; i < nruns; i++) {
		fprintf(fd, " %2d", grps[i]);
		for (j = 0; j < size; j++)
			fprintf(fd, " %2d", (j < length[i]) ? rows[i][j] : -1);
		fprintf(fd, "\n");
	}
	fclose(fd);


	/* clean up */
done:
	for (i = 0; i < nruns; i++) {
		cxip_zbcoll_free(zb[i]);
		free(rows[i]);
	}
	free(grps);
	free(length);
	free(rows);
	free(index);
	free(addrs);
	return ret;
}

/* display the accumulated data for the full test run */
void _printrun(size_t size, int irun, int ***data)
{
	int irank, inid;

	printf("Test run #%d\n", irun);
	for (irank = 0; irank < pmi_numranks; irank++) {
		printf("rank %2d: ", irank);
		if (data[irank][irun][0] < 0) {
			printf("SKIP\n");
			continue;
		}
		printf("GRP %2d:", data[irank][irun][0]);
		for (inid = 1; inid < size+1; inid++)
			printf(" %2d", data[irank][irun][inid]);
		printf("\n");
	}
}

/**
 * @brief Check _multigroup results across all nodes.
 *
 * This is run only on the rank 0 process, and verifies the prior test run.
 *
 * @param size  : total number of NID addresses
 * @param nruns : nruns of concurrency in test
 * @return int  : 0 on success, non-zero on failure
 */
int _multicheck(size_t size, int nruns)
{
	char fnam[256];
	FILE *fd;
	uint32_t *dsc, *err, *ack, *rcv;
	int ***data;
	uint64_t bitv, *mask;
	int grp, nid;
	int irank, irank2, irun, inid, ret;

	ret = 0;
	/* data[irank][irun][inid], inid==0 is grpid */
	data = calloc(pmi_numranks, sizeof(void *));
	for (irank = 0; irank < pmi_numranks; irank++) {
		data[irank] = calloc(nruns, sizeof(void *));
		for (irun = 0; irun < nruns; irun++) {
			data[irank][irun] = calloc(size + 1, sizeof(int));
		}
	}
	/* one bit for each nid, max is 64 */
	mask = calloc(size, sizeof(uint64_t));
	dsc = calloc(pmi_numranks, sizeof(uint32_t));
	err = calloc(pmi_numranks, sizeof(uint32_t));
	ack = calloc(pmi_numranks, sizeof(uint32_t));
	rcv = calloc(pmi_numranks, sizeof(uint32_t));

	/* read in the per-rank file data from the last test run */
	for (irank = 0; irank < pmi_numranks; irank++) {
		/* read file contents into data array */
		sprintf(fnam, "grpid%d", irank);
		fd = fopen(fnam, "r");
		if (! fd) {
			printf("Could not open %s\n", fnam);
			ret = 1;
			goto cleanup;
		}
		if (fscanf(fd, " %d %d %d %d",
			   &dsc[irank],
			   &err[irank],
			   &ack[irank],
			   &rcv[irank]) < 4) {
			printf("bad read (errs)\n");
			ret = 1;
			goto cleanup;
		}
		for (irun = 0; irun < nruns; irun++) {
			for (inid = 0; inid < size + 1; inid++) {
				int *ptr = &data[irank][irun][inid];
				if (fscanf(fd, " %d", ptr) < 1) {
					printf("bad read[%d,%d]\n", irun, inid);
					ret = 1;
					goto cleanup;
				}
			}
		}
		fclose(fd);
	}

	/* All ranks in any test run must use the same grpid, ranks */
	for (irun = 0; irun < nruns; irun++) {
		irank2 = -1;
		for (irank = 1; irank < pmi_numranks; irank++) {
			/* grpid < 0: rank not involved */
			if (data[irank][irun][0] < 0)
				continue;
			/* remember first involved rank */
			if (irank2 < 0)
				irank2 = irank;
			/* compare entire row with first involved */
			for (inid = 0; inid < size+1; inid++)
				if (data[irank][irun][inid] !=
				    data[irank2][irun][inid])
					break;
			/* miscompare is a failure */
			if (inid < size+1) {
				printf("ERROR in run #%d @ %d\n", irun, inid);
				printf("reductions do not match\n");
				_printrun(size, irun, data);
				ret = 1;
				goto cleanup;
			}
		}
	}
	/* validated that all ranks in each run are identical */

	/* No nid should reuse the same grpid, only check rank 0 */
	irank = 0;
	for (irun = 0; irun < nruns; irun++) {
		/* grpid < 0: rank not involved */
		if (data[irank][irun][0] < 0)
			continue;
		grp = data[irank][irun][0];
		for (inid = 1; inid < size+1; inid++) {
			/* ignore unused fiaddrs */
			if (data[irank][irun][inid] < 0)
				continue;
			nid = data[irank][irun][inid];
			bitv = 1L << grp;
			/* failure if grpid already used */
			if (mask[nid] & bitv) {
				printf("ERROR in run #%d @ %d\n",
					irun, inid);
				printf("reuse of grpid %d by %d\n",
					grp, nid);
				_printrun(size, irun, data);
				goto cleanup;
			}
			mask[nid] |= bitv;
		}
	}

	/* We don't expect discard or ack errors */
	for (irank = 0; irank < pmi_numranks; irank++)
		if (dsc[irank] || err[irank])
			break;
	if (irank < pmi_numranks) {
		printf("ERROR transmission errors\n");
		for (irank = 0; irank < pmi_numranks; irank++) {
			printf("rank %2d: dsc=%d err=%d ack=%d rcv=%d\n",
				irank, dsc[irank], err[irank],
				ack[irank], rcv[irank]);
		}
		goto cleanup;
	}

cleanup:
	if (verbose) {
		printf("==================\n");
		printf("Dump all test runs\n");
		for (irun = 0; irun < nruns; irun++)
			_printrun(size, irun, data);
		printf("getgroup test %s\n", !ret ? "passed" : "FAILED");
	}
	fflush(stdout);

	free(dsc);
	free(err);
	free(ack);
	free(rcv);
	free(mask);
	for (irank = 0; irank < pmi_numranks; irank++) {
		for (irun = 0; irun < nruns; irun++)
			free(data[irank][irun]);
		free(data[irank]);
	}
	free(data);
	return ret;
}

/* use up all group IDs, then free zb objects and add more */
int _exhaustgroup(struct cxip_ep_obj *ep_obj, size_t size, fi_addr_t *fiaddrs,
		 int nruns, int usec)
{
	struct cxip_zbcoll_obj **zb;
	int maxgrps;
	int i, n, ret = 0;

	maxgrps = cxip_zbcoll_max_grps(false);
	if (nruns < 0)
		nruns = maxgrps + 10;
	zb = calloc(nruns, sizeof(void *));
	n = 1;
	for (i = 0; i < nruns; i++) {
		_jitter(usec);
		ret = _getgroup(ep_obj, size, fiaddrs, &zb[i]);
		if (ret == -FI_EBUSY) {
			/* free an old zb, and try again */
			cxip_zbcoll_free(zb[n]);
			zb[n] = NULL;
			ret = _getgroup(ep_obj, size, fiaddrs, &zb[i]);
			if (pmi_errmsg(ret, "FAILURE\n")) {
				trc("FAILURE\n");
				break;
			}
			if (zb[i]->grpid != n) {
				trc("FAILURE\n");
				break;
			}
			n = (n + 3) % maxgrps;
		}
	}
	for (i = 0; i < nruns; i++)
		cxip_zbcoll_free(zb[i]);

	return 0;
}

/* callback test final callback */
void _callback_cleanup(struct cxip_zbcoll_obj *zb, void *data)
{
	int *running = (int *)data;

	trc("%2d cleanup\n", zb->grpid);
	(*running)--;
	cxip_zbcoll_free(zb);
}

/* callback test intermediate callback */
void _callback_notified(struct cxip_zbcoll_obj *zb, void *data)
{
	int *usec = (int *)data;
	int ret;

	trc("%2d delay %d usec\n", zb->grpid, *usec);
	_jitter(*usec);
	trc("%2d initiated barrier\n", zb->grpid);
	ret = cxip_zbcoll_barrier(zb);
	if (ret)
		cxip_zbcoll_pop_cb(zb);
}

/* test the callback system */
int _test_callback(struct cxip_ep_obj *ep_obj,
		   size_t size, fi_addr_t *fiaddrs,
		   int nruns, int usec)
{
	struct cxip_zbcoll_obj *zb;
	int running = 0;
	int i, ret = 0;

	if (nruns < 0)
		nruns = cxip_zbcoll_max_grps(false) + 10;
	for (i = 0; i < nruns; i++) {
		ret = cxip_zbcoll_alloc(ep_obj, size, fiaddrs, false, &zb);
		if (pmi_errmsg(ret, "%s: cxip_zbcoll_alloc()\n", __func__))
			return ret;
		cxip_zbcoll_push_cb(zb, _callback_cleanup, &running);
		cxip_zbcoll_push_cb(zb, _callback_notified, &usec);
		running++;
		do {
			ret = _getgroup(ep_obj, size, fiaddrs, &zb);
		} while (ret == -FI_EBUSY);
		/* once dispatched, zb object will self-delete */
	}
	do {
		cxip_zbcoll_progress(ep_obj);
	} while (running > 0);

	return 0;
}

/* barrier across all NIDs, return zb object */
int _test_barr(struct cxip_ep_obj *ep_obj,
	     size_t size, fi_addr_t *fiaddrs,
	     struct cxip_zbcoll_obj **zbp)
{
	struct cxip_zbcoll_obj *zb = NULL;
	int ret;

	/* need a zbcoll context for this */
	ret = _getgroup(ep_obj, size, fiaddrs, &zb);
	if (ret)
		goto out;

	/* reset counters */
	cxip_zbcoll_reset_counters(ep_obj);

	/* if this fails, do not continue */
	ret = cxip_zbcoll_barrier(zb);
	if (pmi_errmsg(ret, "barr0 return=%d, exp=%d\n", ret, 0))
		goto out;

	/* try this again, should fail with -FI_EAGAIN */
	ret = cxip_zbcoll_barrier(zb);
	if (pmi_errmsg((ret != -FI_EAGAIN), "barr1 return=%d, exp=%d\n",
		       ret, -FI_EAGAIN))
		goto out;

	*zbp = zb;
	return 0;
out:
	cxip_zbcoll_free(zb);
	return 1;
}

/* wait for a barrier to complete, and clean up */
int _test_barr_wait_free(struct cxip_zbcoll_obj *zb)
{
	int ret;

	/* wait for completion */
	ret = _coll_wait(zb, nMSEC(100));
	trc("barr ret=%d\n", ret);
	cxip_zbcoll_free(zb);

	return ret;
}

/* broadcast the payload from rank 0 to all other ranks, return zb object */
int _test_bcast(struct cxip_ep_obj *ep_obj,
	       size_t size, fi_addr_t *fiaddrs,
	       uint64_t *result, uint64_t payload,
	       struct cxip_zbcoll_obj **zbp)
{
	struct cxip_zbcoll_obj *zb = NULL;
	int ret;

	/* need a zbcoll context for this */
	ret = _getgroup(ep_obj, size, fiaddrs, &zb);
	if (!zb)
		goto out;

	/* set rank 0 to payload, all others to different values */
	*result = (!pmi_rank) ? payload : pmi_rank;

	/* reset counters */
	cxip_zbcoll_reset_counters(ep_obj);

	/* if this fails, do not continue */
	ret = cxip_zbcoll_broadcast(zb, result);
	trc("bcast payload=%08lx, ret=%d\n", *result, ret);
	if (pmi_errmsg(ret, "bcast0 return=%d, exp=%d\n", ret, 0))
		goto out;

	/* try this again, should fail with -FI_EAGAIN */
	ret = cxip_zbcoll_broadcast(zb, result);
	trc("bcast payload=%08lx, ret=%d\n", *result, ret);
	if (pmi_errmsg((ret != -FI_EAGAIN), "bcast1 return=%d, exp=%d\n",
		       ret, -FI_EAGAIN))
		goto out;

	*zbp = zb;
	return 0;
out:
	cxip_zbcoll_free(zb);
	return 1;
}

/* wait for a broadcast to complete, and clean up */
int _test_bcast_wait_free(struct cxip_zbcoll_obj *zb, uint64_t *result,
			 uint64_t payload)
{
	int ret;

	/* wait for completion */
	ret = _coll_wait(zb, nMSEC(100));
	pmi_errmsg(ret, "bcast wait failed\n");
	trc("bcast result=%08lx, ret=%d\n", *result, ret);
	if (!ret && *result != payload) {
		ret = 1;
		pmi_errmsg(ret, "bcast result=x%lx payload=x%lx\n",
			   *result, payload);
	}
	cxip_zbcoll_free(zb);

	return ret;
}

const char *testnames[] = {
	"test  0: send one packet 0 -> 0",
	"test  1: send one packet 0 -> 1",
	"test  2: send one packet 1 -> 0",
	"test  3: send one packet 0 -> N",
	"test  4: send one packet N -> 0",
	"test  5: send one packet N -> N",
	"test  6: single getgroup",
	"test  7: double getgroup full overlap",
	"test  8: double getgroup partial overlap",
	"test  9: getgroup randomized regression",
	"test 10: getgroup exahustion",
	"test 11: callbacks",
	"test 12: barrier",
	"test 13: broadcast (single)",
	"test 14: broadcast (concurrent)",
	"test 15: getgroup perf",
	"test 16: barrier perf",
	"test 17: broadcast perf",
	NULL
};

int usage(int ret)
{
	int i;

	pmi_log0("Usage: test_zbcoll [-hv] [-s seed] [-N nruns] [-M sublen]\n"
		"                   [-R rotate] [-D usec_delay]\n"
		"                   [-t testno[,testno...]]\n"
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
	fi_addr_t *fiaddrs = NULL;
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zb1 = NULL;
	struct cxip_zbcoll_obj *zb2 = NULL;
	size_t size = 0;
	unsigned int seed;
	uint64_t testmask;
	uint64_t result1, result2;
	int opt, nruns, naddrs, rot, usec, ret;
	int errcnt = 0;
	const char *testname;

	setenv("PMI_MAX_KVS_ENTRIES", "5000", 1);
	ret = pmi_init_libfabric();
	if (pmi_errmsg(ret, "pmi_init_libfabric()\n"))
		return ret;

	pmi_trace_enable(true);

	seed = 123;
	usec = 0;	// as fast as possible
	nruns = -1;	// run maximum number groups
	naddrs = -1;	// random selection of fiaddrs
	rot = -1;	// random shuffle of fiaddrs
	testmask = -1;	// run all tests

	while ((opt = getopt(argc, argv, "hvt:s:N:M:R:D:")) != -1) {
		char *str, *s, *p;
		int i, j;

		switch (opt) {
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
		case 's':
			seed = atoi(optarg);
			break;
		case 'N':
			nruns = atoi(optarg);
			break;
		case 'M':
			naddrs = atoi(optarg);
			break;
		case 'R':
			rot = atoi(optarg);
			break;
		case 'D':
			usec = atoi(optarg);
			break;
		case 'v':
			verbose = true;
			break;
		case 'h':
			return usage(0);
		default:
			return usage(1);
		}
	}

	if (pmi_errmsg(pmi_numranks < 4, "requires at least 4 nodes\n"))
		return -FI_EINVAL;

	srand(seed);
	if (naddrs < 0)
		naddrs = pmi_numranks;
	if (nruns < 0)
		nruns = pmi_numranks;
	if (nruns > cxip_zbcoll_max_grps(false))
		nruns = cxip_zbcoll_max_grps(false);

	pmi_log0("Using random seed = %d\n", seed);
	if (verbose) {
		pmi_log0("verbose = true\n");
		pmi_log0("nruns    = %d\n", nruns);
		pmi_log0("naddrs    = %d\n", naddrs);
		pmi_log0("rotate   = %d\n", rot);
		pmi_log0("delay    = %d usec\n", usec);
	}

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	/* always start with FI_UNIVERSE */
	ret = pmi_populate_av(&fiaddrs, &size);
	if (pmi_errmsg(ret, "pmi_populate_av()\n"))
		return 1;

	if (testmask & TEST(0)) {
		testname = testnames[0];
		trc("======= %s\n", testname);
		ret = _test_send_to_dest(ep_obj, size, fiaddrs, 0, 0, pmi_rank);
		errcnt += !!ret;
		trc("rank %2d result = %d\n", pmi_rank, ret);
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(1)) {
		testname = testnames[1];
		trc("======= %s\n", testname);
		ret = _test_send_to_dest(ep_obj, size, fiaddrs, 0, 1, pmi_rank);
		errcnt += !!ret;
		trc("rank %2d result = %d\n", pmi_rank, ret);
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(2)) {
		testname = testnames[2];
		trc("======= %s\n", testname);
		ret = _test_send_to_dest(ep_obj, size, fiaddrs, 1, 0, pmi_rank);
		errcnt += !!ret;
		trc("rank %2d result = %d\n", pmi_rank, ret);
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(3)) {
		testname = testnames[3];
		trc("======= %s\n", testname);
		ret = _test_send_to_dest(ep_obj, size, fiaddrs, 0, -1, pmi_rank);
		errcnt += !!ret;
		trc("rank %2d result = %d\n", pmi_rank, ret);
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(4)) {
		testname = testnames[4];
		trc("======= %s\n", testname);
		ret = _test_send_to_dest(ep_obj, size, fiaddrs, -1, 0, pmi_rank);
		errcnt += !!ret;
		trc("rank %2d result = %d\n", pmi_rank, ret);
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(5)) {
		testname = testnames[5];
		trc("======= %s\n", testname);
		ret = _test_send_to_dest(ep_obj, size, fiaddrs, -1, -1, pmi_rank);
		errcnt += !!ret;
		trc("rank %2d result = %d\n", pmi_rank, ret);
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(6)) {
		testname = testnames[6];
		trc("======= %s\n", testname);
		zb1 = NULL;
		ret = 0;
		ret += !!_getgroup(ep_obj, size, fiaddrs, &zb1);
		ret += !!_check_getgroup_errs(zb1, 0);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		cxip_zbcoll_free(zb1);
		pmi_Barrier();
	}

	if (testmask & TEST(7)) {
		testname = testnames[7];
		trc("======= %s\n", testname);
		zb1 = NULL;
		ret = 0;
		ret += !!_getgroup(ep_obj, size, fiaddrs, &zb1);
		ret += !!_getgroup(ep_obj, size, fiaddrs, &zb2);
		ret += !!_check_getgroup_errs(zb1, 0);
		ret += !!_check_getgroup_errs(zb2, 1);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		cxip_zbcoll_free(zb2);
		cxip_zbcoll_free(zb1);
		pmi_Barrier();
	}

	if (testmask & TEST(8)) {
		testname = testnames[8];
		trc("======= %s\n", testname);
		zb1 = zb2 = NULL;
		ret = 0;
		if (pmi_rank != pmi_numranks-1) {
			ret += !!_getgroup(ep_obj, size-1, &fiaddrs[0], &zb2);
			ret += !!_check_getgroup_errs(zb2, 0);
		} else {
			trc("SKIP\n");
		}
		if (pmi_rank != 0) {
			ret += !!_getgroup(ep_obj, size-1, &fiaddrs[1], &zb1);
			ret += !!_check_getgroup_errs(zb1, 1);
		} else {
			trc("SKIP\n");
		}
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		cxip_zbcoll_free(zb2);
		cxip_zbcoll_free(zb1);
		pmi_Barrier();
	}

	if (testmask & TEST(9)) {
		testname = testnames[9];
		trc("======= %s\n", testname);
		ret = 0;
		ret += !!_multigroup(ep_obj, size, fiaddrs, nruns, naddrs,
				     rot, usec);
		pmi_Barrier();
		if (!ret && pmi_rank == 0)
			ret += !!_multicheck(size, nruns);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(10)) {
		testname = testnames[10];
		trc("======= %s\n", testname);
		ret = 0;
		ret += !!_exhaustgroup(ep_obj, size, fiaddrs, nruns, usec);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(11)) {
		testname = testnames[11];
		trc("======= %s\n", testname);
		ret = 0;
		ret += !!_test_callback(ep_obj, size, fiaddrs, nruns, usec);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(12)) {
		testname = testnames[12];
		trc("======= %s\n", testname);
		zb1 = NULL;
		ret = 0;
		ret += !!_test_barr(ep_obj, size, fiaddrs, &zb1);
		ret += !!_test_barr_wait_free(zb1);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(13)) {
		testname = testnames[13];
		trc("======= %s\n", testname);
		zb1 = NULL;
		ret = 0;
		ret += !!_test_bcast(ep_obj, size, fiaddrs, &result1, 0x123,
				     &zb1);
		ret += !!_test_bcast_wait_free(zb1, &result1, 0x123);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(14)) {
		testname = testnames[14];
		trc("======= %s\n", testname);
		zb1 = zb2 = NULL;
		ret = 0;
		ret += !!_test_bcast(ep_obj, size, fiaddrs, &result1, 0x123,
				     &zb1);
		ret += !!_test_bcast(ep_obj, size, fiaddrs, &result2, 0x456,
				     &zb2);
		ret += !!_test_bcast_wait_free(zb1, &result1, 0x123);
		ret += !!_test_bcast_wait_free(zb2, &result2, 0x456);
		errcnt += !!ret;
		pmi_log0("%4s %s\n", ret ? "FAIL" : "ok", testname);
		pmi_Barrier();
	}

	if (testmask & TEST(15)) {
		struct timespec t0;
		long count = 0;
		double time;

		testname = testnames[15];
		trc("======= %s\n", testname);
		pmi_trace_enable(false);
		zb1 = NULL;
		ret = cxip_zbcoll_alloc(ep_obj, size, fiaddrs, false, &zb1);
		clock_gettime(CLOCK_MONOTONIC, &t0);
		while (!ret && count < 100000) {
			int ret2;
			do {
				ret += !!cxip_zbcoll_getgroup(zb1);
				ret2 = _coll_wait(zb1, nMSEC(100));
			} while (!ret && ret2 == -FI_EAGAIN);
			ret += !!ret2;
			cxip_zbcoll_rlsgroup(zb1);
			count++;
		}
		time = _measure_nsecs(&t0);
		time /= 1.0*count;
		time /= 1000.0;
		pmi_trace_enable(true);
		cxip_zbcoll_free(zb1);
		errcnt += !!ret;
		pmi_log0("%4s %s \tcount=%ld time=%1.2fus\n",
			 ret ? "FAIL" : "ok", testname, count, time);
		pmi_Barrier();
	}

	if (testmask & TEST(16)) {
		struct timespec t0;
		long count = 0;
		double time;

		testname = testnames[16];
		trc("======= %s\n", testname);
		pmi_trace_enable(false);
		zb1 = NULL;
		ret = _getgroup(ep_obj, size, fiaddrs, &zb1);
		clock_gettime(CLOCK_MONOTONIC, &t0);
		while (!ret && count < 100000) {
			ret += !!cxip_zbcoll_barrier(zb1);
			ret += !!_coll_wait(zb1, nMSEC(100));
			count++;
		}
		time = _measure_nsecs(&t0);
		time /= 1.0*count;
		time /= 1000.0;
		pmi_trace_enable(true);
		cxip_zbcoll_free(zb1);
		errcnt += !!ret;
		pmi_log0("%4s %s \tcount=%ld time=%1.2fus\n",
			 ret ? "FAIL" : "ok", testname, count, time);
		pmi_Barrier();
	}

	if (testmask & TEST(17)) {
		struct timespec t0;
		uint64_t result = 0x1234;
		long count = 0;
		double time;

		testname = testnames[17];
		trc("======= %s\n", testname);
		pmi_trace_enable(false);
		zb1 = NULL;
		ret = _getgroup(ep_obj, size, fiaddrs, &zb1);
		clock_gettime(CLOCK_MONOTONIC, &t0);
		while (!ret && count < 100000) {
			ret += !!cxip_zbcoll_broadcast(zb1, &result);
			ret += !!_coll_wait(zb1, nMSEC(100));
			count++;
		}
		time = _measure_nsecs(&t0);
		time /= 1.0*count;
		time /= 1000.0;
		pmi_trace_enable(true);
		cxip_zbcoll_free(zb1);
		errcnt += !!ret;
		pmi_log0("%4s %s \tcount=%ld time=%1.2fus\n",
			 ret ? "FAIL" : "ok", testname, count, time);
		pmi_Barrier();
	}

	trc("Finished test run, cleaning up\n");
	free(fiaddrs);
	pmi_free_libfabric();
	return !!errcnt;
}
