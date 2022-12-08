/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * (c) Copyright 2021-2023 Hewlett Packard Enterprise Development LP
 *
 * libfabric C test framework for multinode testing.
 *
 * This must be compiled with:
 *
 * - PLATFORM_CASSINI_HW=1 (or other hardware flag)
 *
 * Tests are run using srun: $ srun -Nn ./test_frmwk 'n' is the number of nodes
 * to use. Some tests may place requirements on 'n'.
 *
 * frmwk_init_libfabric() sets up
 * - generic fabric info for CXI driver
 * - one domain (fabric address)
 * - one endpoint
 * - one of each of the following
 *   - eq
 *   - tx cq
 *   - rx cq
 *   - send cntr
 *   - recv cntr
 *   - read cntr
 *   - write cntr
 *   - remote cntr
 *
 * frmwk_populate_av() uses a file-based Allgather operation to collect local
 * HSN addresses and distribute them over the entire set of nodes, and then
 * creates and binds the fi_av object for the endpoint. This 'populate' function
 * has been separated out from initialization, to allow the framework to use
 * other means of population (e.g. MPI). The following environment variables are
 * significant:
 * - PMI_SIZE		(WLM)  number of ranks in job (from WLM)
 * - PMI_RANK		(WLM)  rank of this process   (from WLM)
 * - PMI_SHARED_SECRET	(WLM)  unique job identifier  (from WLM)
 * - PMI_NUM_HSNS	(USER) optional, defaults to 1
 * - PMI_HOME		(USER) optional, preferred file system directory to use
 * - HOME		(USER) default file system directory to use
 *
 * frmwk_enable_libfabric() can be used after the fi_av object has been
 * initialized.
 *
 * frmwk_free_libfabric() terminates the libfabric instance and cleans up.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <time.h>

#include <ofi.h>
#include <cxip.h>

#include "multinode_frmwk.h"

/* see cxit_trace_enable() in each test framework */
#define	TRACE CXIP_TRACE

#define RETURN_ERROR(ret, txt) \
	if (ret != FI_SUCCESS) { \
		fprintf(stderr, "FAILED %s = %s\n", txt, fi_strerror(-ret)); \
		return ret; \
	}

#define	CLOSE_OBJ(obj)	do {if (obj) fi_close(&obj->fid); } while (0)

/* Taken from SLURM environment variables */
int frmwk_numranks;		/* PMI_SIZE */
int frmwk_rank;			/* PMI_RANK */
int frmwk_nics_per_rank;	/* PMI_NUM_HSNS (defaults to 1) */
const char *frmwk_unique;	/* PMI_SHARED_SECRET */
const char *frmwk_home;		/* PMI_HOME or HOME */
int frmwk_seq;			/* sequence number */
union nicaddr *frmwk_nics;	/* array of NIC addresses plus rank and hsn */

int _frmwk_init;

char *cxit_node;
char *cxit_service;
uint64_t cxit_flags;
struct fi_info *cxit_fi_hints;
struct fi_info *cxit_fi;

struct fid_fabric *cxit_fabric;
struct fid_domain *cxit_domain;
struct fi_cxi_dom_ops *cxit_dom_ops;

struct mem_region {
	uint8_t *mem;
	struct fid_mr *mr;
};

struct fid_ep *cxit_ep;
struct fi_eq_attr cxit_eq_attr = {
        .size = 32,
        .flags = FI_WRITE,
        .wait_obj = FI_WAIT_NONE
};
uint64_t cxit_eq_bind_flags = 0;

struct fid_eq *cxit_eq;

struct fi_cq_attr cxit_rx_cq_attr = {
        .format = FI_CQ_FORMAT_TAGGED

};
uint64_t cxit_rx_cq_bind_flags = FI_RECV;
struct fid_cq *cxit_rx_cq;

struct fi_cq_attr cxit_tx_cq_attr = {
	.format = FI_CQ_FORMAT_TAGGED,
	.size = 16384
};
uint64_t cxit_tx_cq_bind_flags = FI_TRANSMIT;
struct fid_cq *cxit_tx_cq;

fi_addr_t cxit_ep_fi_addr;

struct fi_cntr_attr cxit_cntr_attr = {};
struct fid_cntr *cxit_send_cntr;
struct fid_cntr *cxit_recv_cntr;
struct fid_cntr *cxit_read_cntr;
struct fid_cntr *cxit_write_cntr;
struct fid_cntr *cxit_rem_cntr;

struct fi_av_attr cxit_av_attr = {
	.type = FI_AV_TABLE,
	.rx_ctx_bits = 4
};
struct fid_av *cxit_av;

int cxit_n_ifs;
struct fid_av_set *cxit_av_set;
struct fid_mc *cxit_mc;
fi_addr_t cxit_mc_addr;

/* HMEM memory functional overlays */
int mr_create_ext(size_t len, uint64_t access, uint8_t seed, uint64_t key,
		  struct fid_cntr *cntr, struct mem_region *mr)
{
	int ret;

	if (len) {
		mr->mem = calloc(1, len);
		ret = (mr->mem != NULL) ? FI_SUCCESS : FI_ENOMEM;
 		RETURN_ERROR(ret, __func__);
	} else {
		mr->mem = 0;
	}

	for (size_t i = 0; i < len; i++)
		mr->mem[i] = i + seed;

	ret = fi_mr_reg(cxit_domain, mr->mem, len, access, 0, key, 0,
			&mr->mr, NULL);
	RETURN_ERROR(ret, "fi_mr_reg");

	ret = fi_mr_bind(mr->mr, &cxit_ep->fid, 0);
	RETURN_ERROR(ret, "fi_mr_bind ep");

	if (cntr) {
		ret = fi_mr_bind(mr->mr, &cntr->fid, FI_REMOTE_WRITE);
		RETURN_ERROR(ret, "fi_mr_bind cntr");
	}

	ret = fi_mr_enable(mr->mr);
	RETURN_ERROR(ret, "fi_mr_enable");

	return 0;
}

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

struct fi_hmem_override_ops cxit_hmem_ops = {
	.copy_from_hmem_iov = copy_from_hmem_iov,
	.copy_to_hmem_iov = copy_to_hmem_iov,
};

/* A minimal generic context for use with asynchronous operations */
struct mycontext {
	int rx_err;
	int rx_prov_err;
	int tx_err;
	int tx_prov_err;
};

/**
 * @brief Trace function.
 *
 * See the description in prov/cxi/test/cxip_test_common.c.
 *
 * This trace function is rank-aware. Enabling opens a file associated with the
 * rank, and disabling closes it. All rank trace output is delivered to the
 * file. Files are global across the network, allowing trace information to be
 * dynamically monitored.
 */

static FILE *frmwk_trace_fid;
int cxit_trace_offset;

void cxit_trace_flush(void)
{
	if (frmwk_trace_fid) {
		fflush(frmwk_trace_fid);
		fsync(fileno(frmwk_trace_fid));
	}
}

static int cxip_trace_attr frmwk_trace(const char *fmt, ...)
{
	va_list args;
	char *str;
	int len;

	va_start(args, fmt);
	len = vasprintf(&str, fmt, args);
	va_end(args);
	if (len >= 0) {
		len = fprintf(frmwk_trace_fid, "[%2d|%2d] %s",
			frmwk_rank, frmwk_numranks, str);
		cxit_trace_flush();
		free(str);
	}
	return len;
}

bool cxit_trace_enable(bool enable)
{
	static bool is_enabled = false;
	bool was_enabled = is_enabled;
	char fnam[256];

	is_enabled = !!frmwk_trace_fid;
	if (enable && !is_enabled) {
		sprintf(fnam, "./trace%d", frmwk_rank + cxit_trace_offset);
		frmwk_trace_fid = fopen(fnam, "w");
		if (!frmwk_trace_fid) {
			fprintf(stderr, "open(%s) failed: %s\n",
				fnam, strerror(errno));
		} else {
			cxip_trace_fn = frmwk_trace;
		}
	} else if (!enable) {
		if (frmwk_trace_fid) {
			fflush(frmwk_trace_fid);
			fclose(frmwk_trace_fid);
			frmwk_trace_fid = NULL;
		}
		cxip_trace_fn = NULL;
	}
	return was_enabled;
}

/* display message on stdout from rank 0 */
int frmwk_log0(const char *fmt, ...)
{
	va_list args;
	int len;

	if (frmwk_rank != 0)
		return 0;

	va_start(args, fmt);
	len = vfprintf(stdout, fmt, args);
	va_end(args);
	fflush(stdout);
	return len;
}

/**
 * @brief File-system-based Allgather.
 *
 * size indicates the size of the data block. If size is zero, both data and
 * rslt are ignored.
 *
 * data may be NULL, in which case size and rslt are ignored.
 *
 * rslt may be NULL, otherwise it must contain space for (size * numranks) bytes
 * of data.
 *
 * If size is non-zero, and data and rslt are non-NULL, and the result will be
 * an Allgather in which the data from each rank will be collected into the rslt
 * array, appearing in rank order.
 *
 * If size is non-zero, and data is non-NULL, but rslt is NULL on one or more
 * ranks, those ranks will not receive a result. If only one rank specifies a
 * non-NULL rslt pointer, this is equivalent to a Gather. If no ranks specify a
 * non-NULL rslt pointer, this is equivalant to a Barrier.
 *
 * If size is zero, data and rslt are ignored and can be NULL. The result is
 * equivalent to a Barrier function.
 *
 * Theory of operation:
 *
 * This uses two files for each rank, an empty file with a 'busy' suffix, and a
 * data file without the suffix.
 *
 * Each rank first creates its 'busy' file, then writes its allgather data to
 * the data file. These files are visible to all ranks through the common file
 * system.
 *
 * Each rank then attempts to collect the allgather data from each rank's data
 * file, and sorts it into the result array by rank. The data files will
 * generally appear in a random order, and may not appear for some time, since
 * the nodes are not running synchronously.
 *
 * Once a complete allgather result is collected on a rank, the rank deletes its
 * own busy file, and then waits for rank 0 to delete its data file.
 *
 * Rank 0 provides the collective synchronization by testing busy files. When
 * all busy files have been deleted by their respective rank, all ranks have
 * successsfully gathered the shared data from all ranks. Rank 0 can then safely
 * delete all of the data files in one sweep, and as the ranks see their own
 * data file deleted, they are free to proceed to the next operation.
 *
 * Note that if the job is aborted for any reason, this is a "dirty" operation
 * that may leave files in the common file system. These must be manually
 * deleted.
 *
 * There are many ways to implement this functionality. This method has been
 * chosen because other methods (e.g. PMI) undergo non-transparent changes from
 * time to time, and this breaks implementations and sometimes requires
 * refactoring of test code. Arbitrary configurations on experimental test
 * systems, such as nonstandard or on-the-fly use of socket numbers, can also
 * cause problems. A common FS is a baseline feature of multinode systems, and
 * the Linux FS API and code is extremely stable.
 *
 * @param size          size of data in bytes
 * @param data          data from this rank
 * @param rslt          result from all ranks
 * @return int          0 on success, or error code
 */
int frmwk_allgather(size_t size, void *data, void *rslt)
{
	const char *busyfmt = "%s/allg.%s.%d.%d.busy";
	const char *datafmt = "%s/allg.%s.%d.%d";
	char filename[256];
	char *mask = NULL;
	int i, count, len, ret, err;
	int read_stall = 0;
	int read_short = 0;
	int sync_stall = 0;
	FILE *fid;

	/* disambiguate */
	if (!data)
		size = 0L;
	if (!size) {
		data = NULL;
		rslt = NULL;
	}
	err = -1;

	if (!_frmwk_init) {
		fprintf(stderr, "Framework not initialized\n");
		goto done;
	}

	mask = calloc(frmwk_numranks, 1);
	if (!mask) {
		fprintf(stderr, "out of memory\n");
		goto done;
	}

	/* mark this rank as busy with an empty touch-file */
	sprintf(filename, busyfmt, frmwk_home, frmwk_unique,
		frmwk_seq, frmwk_rank);
	fid = fopen(filename, "w");
	if (!fid) {
		fprintf(stderr, "fopen(%s) failed %d\n", filename, errno);
		goto done;
	}
	fclose(fid);

	/* open a file to hold data for this rank */
	sprintf(filename, datafmt, frmwk_home, frmwk_unique,
		frmwk_seq, frmwk_rank);
	fid = fopen(filename, "w");
	if (!fid) {
		fprintf(stderr, "fopen(%s) failed %d\n", filename, errno);
		goto done;
	}
	ret = 0;
	if (size)
		ret = fwrite(data, 1, size, fid);
	fclose(fid);
	if (ret < size) {
		fprintf(stderr, "fwrite(%s) failed %d\n", filename, errno);
		goto done;
	}

	/* read each file of the data into rslt as it appears */
	count = frmwk_numranks;
	while (count) {
		/* do not flood FS */
		usleep(10000);
		for (i = 0; i < frmwk_numranks; i++) {
			/* avoid hitting the file system repeatedly */
			if (mask[i])
				continue;
			/* read contribution from a new rank */
			sprintf(filename, datafmt, frmwk_home, frmwk_unique,
				frmwk_seq, i);
			fid = fopen(filename, "r");
			/* if not yet written, move on */
			if (!fid) {
				read_stall++;
				continue;
			}
			len = fread((char *)rslt + size*i, 1, size, fid);
			fclose(fid);
			/* if count is short, try again later */
			if (len < size) {
				read_short++;
				continue;
			}
			/* looks good, count it and set mask */
			mask[i] = 1;
			count--;
		}
	}
	/* All ranks have been read into rslt */

	/* Remove our own touch file, we are no longer busy */
	sprintf(filename, busyfmt, frmwk_home, frmwk_unique,
		frmwk_seq, frmwk_rank);
	if (remove(filename)) {
		fprintf(stderr, "remove(%s) failed %d\n", filename, errno);
		goto done;
	}
	mask[frmwk_rank] = 0;
	err = 0;

	/* synchronize */
	while (1) {
		/* do not flood FS */
		usleep(10000);
		/* non-zero rank wait for rank zero to delete our rank data */	if (frmwk_rank != 0) {
			sprintf(filename, datafmt, frmwk_home, frmwk_unique,
				frmwk_seq, frmwk_rank);
			fid = fopen(filename, "r");
			if (fid) {
				fclose(fid);
				sync_stall++;
				continue;
			}
			/* rank is done waiting */
			break;
		}
		/* rank 0 does the actual sync cleanup */
		for (i = 0; i < frmwk_numranks; i++) {
			/* skip if already removed */
			if (!mask[i])
				continue;
			/* check for rank=i still busy */
			sprintf(filename, busyfmt, frmwk_home, frmwk_unique,
				frmwk_seq, i);
			/* if still busy, stop checking */
			fid = fopen(filename, "r");
			if (fid) {
				fclose(fid);
				sync_stall++;
				break;
			}
			/* rank=i is done, don't check again */
			mask[i] = 0;
		}
		/* at least one rank is busy, keep waiting */
		if (i < frmwk_numranks)
			continue;
		/* all processes not-busy, remove all data files */
		for (i = 0; i < frmwk_numranks; i++) {
			sprintf(filename, datafmt, frmwk_home, frmwk_unique,
				frmwk_seq, i);
			remove(filename);
			/* rank=i may start next allgather immediately */
		}
		/* rank 0 is also done */
		break;
	}
	frmwk_seq++;

done:
#if 0
	printf("%2d read_stall=%d\n", frmwk_rank, read_stall);
	printf("%2d read_short=%d\n", frmwk_rank, read_short);
	printf("%2d sync_stall=%d\n", frmwk_rank, sync_stall);
	fflush(stdout);
#endif
	free(mask);
	return err;
}

/**
 * @brief File-system-based Barrier.
 */
int frmwk_barrier(void)
{
	return frmwk_allgather(0L, NULL, NULL);
}

/**
 * @brief Check for minimum number of ranks
 *
 * @param minranks required minimum number of ranks
 * @return int error code, 0 on success
 */
int frmwk_check_env(int minranks)
{
	if (!_frmwk_init) {
		fprintf(stderr, "Framework not initialized\n");
		return -1;
	}
	if (frmwk_numranks < minranks) {
		/* only one rank makes noise */
		if (!frmwk_rank)
			fprintf(stderr, "Requires >= %d ranks\n", minranks);
		return -1;
	}
	return 0;
}

/**
 * @brief Shut down the libfabric test framework.
 */
void frmwk_free_libfabric(void)
{
	/* must close EP before closing anything bound to it */
	CLOSE_OBJ(cxit_ep);
	CLOSE_OBJ(cxit_av);
	CLOSE_OBJ(cxit_rem_cntr);
	CLOSE_OBJ(cxit_write_cntr);
	CLOSE_OBJ(cxit_read_cntr);
	CLOSE_OBJ(cxit_recv_cntr);
	CLOSE_OBJ(cxit_send_cntr);
	CLOSE_OBJ(cxit_eq);
	CLOSE_OBJ(cxit_tx_cq);
	CLOSE_OBJ(cxit_rx_cq);
	CLOSE_OBJ(cxit_domain);
	CLOSE_OBJ(cxit_fabric);
	fi_freeinfo(cxit_fi);
	fi_freeinfo(cxit_fi_hints);
}

/**
 * @brief Initialize the libfabric test framework.
 *
 * The ep_obj->src_addr has a PID value of 511 (PID_ANY) until the EP is
 * enabled, at which point the actual PID is assigned. Nothing works if the PIDs
 * are mismatched between ranks.
 *
 * @return int error code, 0 on success
 */
int frmwk_init_libfabric(void)
{
        int ret;

	if (!_frmwk_init) {
		fprintf(stderr, "Framework not initialized\n");
		return -1;
	}

	cxit_fi_hints = fi_allocinfo();
	ret = (cxit_fi_hints != NULL) ? FI_SUCCESS : FI_ENOMEM;

	cxit_fi_hints->fabric_attr->prov_name = strdup("cxi");
	cxit_fi_hints->domain_attr->mr_mode = FI_MR_ENDPOINT;
 	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

      	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 cxit_node, cxit_service, cxit_flags, cxit_fi_hints,
			 &cxit_fi);
	RETURN_ERROR(ret, "fi_getinfo");

	ret = fi_fabric(cxit_fi->fabric_attr, &cxit_fabric, NULL);
	RETURN_ERROR(ret, "fi_fabric");

	ret = fi_domain(cxit_fabric, cxit_fi, &cxit_domain, NULL);
	RETURN_ERROR(ret, "fi_domain");

	ret = fi_open_ops(&cxit_domain->fid, FI_CXI_DOM_OPS_1, 0,
			  (void **)&cxit_dom_ops, NULL);
	RETURN_ERROR(ret, "fi_open_ops 1");

	ret = fi_open_ops(&cxit_domain->fid, FI_CXI_DOM_OPS_2, 0,
			  (void **)&cxit_dom_ops, NULL);
	RETURN_ERROR(ret, "fi_open_ops 2");

	ret = fi_open_ops(&cxit_domain->fid, FI_CXI_DOM_OPS_3, 0,
			  (void **)&cxit_dom_ops, NULL);
	RETURN_ERROR(ret, "fi_open_ops 3");

	ret = fi_set_ops(&cxit_domain->fid, FI_SET_OPS_HMEM_OVERRIDE, 0,
			 &cxit_hmem_ops, NULL);
	RETURN_ERROR(ret, "fi_set_ops");

	ret = fi_endpoint(cxit_domain, cxit_fi, &cxit_ep, NULL);
	RETURN_ERROR(ret, "fi_endpoint");

	ret = fi_cq_open(cxit_domain, &cxit_rx_cq_attr, &cxit_rx_cq, NULL);
	RETURN_ERROR(ret, "fi_cq_open RX");

	ret = fi_ep_bind(cxit_ep, &cxit_rx_cq->fid, cxit_rx_cq_bind_flags);
	RETURN_ERROR(ret, "fi_ep_bind RX_CQ");

        ret = fi_cq_open(cxit_domain, &cxit_tx_cq_attr, &cxit_tx_cq, NULL);
	RETURN_ERROR(ret, "fi_cq_open TX");
	ret = fi_ep_bind(cxit_ep, &cxit_tx_cq->fid, cxit_tx_cq_bind_flags);
	RETURN_ERROR(ret, "fi_ep_bind TX_CQ");

	ret = fi_eq_open(cxit_fabric, &cxit_eq_attr, &cxit_eq, NULL);
	RETURN_ERROR(ret, "fi_eq_open");
	ret = fi_ep_bind(cxit_ep, &cxit_eq->fid, cxit_eq_bind_flags);
	RETURN_ERROR(ret, "fi_ep_bind EQ");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_send_cntr, NULL);
	RETURN_ERROR(ret, "fi_cntr_open SEND");
	ret = fi_ep_bind(cxit_ep, &cxit_send_cntr->fid, FI_SEND);
	RETURN_ERROR(ret, "fi_ep_bind SEND CNTR");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_recv_cntr, NULL);
	RETURN_ERROR(ret, "fi_cntr_open RECV");
	ret = fi_ep_bind(cxit_ep, &cxit_recv_cntr->fid, FI_RECV);
	RETURN_ERROR(ret, "fi_ep_bind RECV CNTR");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_read_cntr, NULL);
	RETURN_ERROR(ret, "fi_cntr_open READ");
	ret = fi_ep_bind(cxit_ep, &cxit_read_cntr->fid, FI_READ);
	RETURN_ERROR(ret, "fi_ep_bind READ CNTR");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_write_cntr, NULL);
	RETURN_ERROR(ret, "fi_cntr_open WRITE");
	ret = fi_ep_bind(cxit_ep, &cxit_write_cntr->fid, FI_WRITE);
	RETURN_ERROR(ret, "fi_ep_bind WRITE CNTR");

	ret = fi_cntr_open(cxit_domain, NULL, &cxit_rem_cntr, NULL);
	RETURN_ERROR(ret, "fi_cntr_open REMOTE");

	cxit_av_attr.count = 1024;
	ret = fi_av_open(cxit_domain, &cxit_av_attr, &cxit_av, NULL);
	RETURN_ERROR(ret, "fi_av_open");

	ret = fi_ep_bind(cxit_ep, &cxit_av->fid, 0);
	RETURN_ERROR(ret, "fi_ep_bind AV");

	ret = fi_enable(cxit_ep);
	RETURN_ERROR(ret, "fi_enable");

	return 0;
}

/**
 * @brief One way of populating the address vector.
 *
 * This uses frmwk_allgather() to perform the allgather of addresses across all
 * nodes in the job. To work properly, the libfabric endpoint must already be
 * enabled.
 *
 * This also serves as a barrier that ensures that all ranks have reached this
 * call, i.e. all ranks have enabled their respective endpoint. If an endpoint
 * is not enabled when another endpoint sends a packet, the sender will receive
 * an ACK, but the target will drop the packet.
 *
 * This routine can be replaced with anything that provides an accurate AV
 * across all nodes in the job, e.g. MPI, symmetric AVs distributed by some
 * other out-of-band means to all nodes, or logical (rank) addressing of the
 * Cassini chips.
 *
 * @param fiaddr : returns array of fi_addr_t in rank order
 * @param size   : returns size of fiaddr array
 * @return int error code, 0 on success.
 */
int frmwk_populate_av(fi_addr_t **fiaddrp, size_t *sizep)
{
	struct cxip_addr *alladdrs = NULL;
	fi_addr_t *fiaddrs = NULL;
	int i, ret;

	if (!fiaddrp || !sizep)
		return -FI_EINVAL;

	ret = -FI_EFAULT;
	ret = frmwk_gather_nics();
	if (ret < 0)
		goto fail;

	ret = -FI_ENOMEM;
	alladdrs = calloc(frmwk_numnics, sizeof(*alladdrs));
	fiaddrs = calloc(frmwk_numnics, sizeof(*fiaddrs));
	if (!fiaddrs || !alladdrs)
		goto fail;

	for (i = 0; i < frmwk_numnics; i++)
		alladdrs[i].nic = frmwk_nics[i].nic;
	ret = fi_av_insert(cxit_av, alladdrs, frmwk_numnics,
			   fiaddrs, 0, NULL);
	if (ret != frmwk_numnics)
		goto fail;

	*sizep = frmwk_numnics;
	*fiaddrp = fiaddrs;
	return FI_SUCCESS;

fail:
	free(fiaddrs);
	free(alladdrs);
	return ret;
}

/**
 * @brief Display an error message to stderr and return error code.
 *
 * This prints to stderr only if ret != 0. It includes rank of the failing
 * process and the size of the job. These values are meaningful only after
 * frmwk_populate_av() has successfully completed.
 *
 * @param ret : error code
 * @param fmt : printf format
 * @param ... : printf parameters
 * @return int value of ret
 */
int frmwk_errmsg(int ret, const char *fmt, ...)
{
	va_list args;
	char host[256];
	char *str;
	int len;

	if (!ret)
		return 0;

	if (gethostname(host, sizeof(host)))
		strcpy(host, "unknown");

	va_start(args, fmt);
	len = vasprintf(&str, fmt, args);
	va_end(args);
	if (len < 0)
		str = "(no errmsg)";
	fprintf(stderr, "%s rank %2d of %2d FAILED %d: %s",
		host, frmwk_rank, frmwk_numranks, ret, str);
	if (len >= 0)
		free(str);

	return ret;
}

/* Read /sys files to get the HSN nic addresses */
static void get_local_nic(int hsn, union nicaddr *nic)
{
	char fname[256];
	char text[256];
	char *ptr;
	FILE *fid;
	int i, n;

	/* default */
	strcpy(text, "FF:FF:FF:FF:FF:FF\n");
	/* read from file, if possible */
	snprintf(fname, sizeof(fname), "/sys/class/net/hsn%d/address", hsn);
	if ((fid = fopen(fname, "r"))) {
		n = fread(text, 1, sizeof(text), fid);
		fclose(fid);
		text[n] = 0;
	}
	/* parse "XX:XX:XX:XX:XX:XX\n" into 48-bit integer value */
	nic->value = 0L;
	ptr = text;
	for (i = 0; i < 6; i++) {
		nic->value <<= 8;
		nic->value |= strtol(ptr, &ptr, 16);
		ptr++;
	}
	nic->hsn = hsn;
	nic->rank = frmwk_rank;
}

/* Sort comparator */
static int _compare(const void *v1, const void *v2)
{
	uint64_t *a1 = (uint64_t *)v1;
	uint64_t *a2 = (uint64_t *)v2;

	if (*a1 < *a2)
		return -1;
	if (*a1 > *a2)
		return 1;
	return 0;
}

/* Allgather on NIC addresses across collective */
int frmwk_gather_nics(void)
{
	union nicaddr *mynics = NULL;
	int i, ret, localsize;

	if (frmwk_nics)
		return 0;

	localsize = frmwk_nics_per_rank * NICSIZE;
	mynics = calloc(1, localsize);
	frmwk_nics = calloc(frmwk_numranks, localsize);
	if (!mynics || !frmwk_nics)
		goto fail;

	for (i = 0; i < frmwk_nics_per_rank; i++)
		get_local_nic(i, &mynics[i]);

	ret = frmwk_allgather(localsize, mynics, frmwk_nics);
	if (ret)
		goto fail;

	frmwk_numnics = frmwk_numranks * frmwk_nics_per_rank;
	qsort(frmwk_nics, frmwk_numnics, NICSIZE, _compare);
	return 0;

fail:
	frmwk_numnics = 0;
	free(frmwk_nics);
	free(mynics);
	return -1;
}

/* User call for the address of rank, hsn */
int frmwk_nic_addr(int rank, int hsn)
{
	if (!frmwk_nics ||
	    rank < 0 || rank >= frmwk_numranks ||
	    hsn < 0 || hsn >= frmwk_nics_per_rank)
	    	return -1;
	return (long)frmwk_nics[rank*frmwk_nics_per_rank + hsn].nic;
}

/* Get environment variable as string representation of int */
static int getenv_int(const char *name)
{
	char *env;
	int value;

	value = -1;
	env = getenv(name);
	if (env)
		sscanf(env, "%d", &value);
	return value;
}

/* Initialize the framework */
void frmwk_init(void)
{
	int ret = -1;

	/* Values are provided by the WLM */
	frmwk_numranks = getenv_int("PMI_SIZE");
	frmwk_rank = getenv_int("PMI_RANK");
	frmwk_unique = getenv("PMI_SHARED_SECRET");
	if (frmwk_numranks < 1 || frmwk_rank < 0 || !frmwk_unique) {
		fprintf(stderr, "PMI_SIZE=%d invalid\n", frmwk_numranks);
		fprintf(stderr, "PMI_RANK=%d invalid\n", frmwk_rank);
		fprintf(stderr, "PMI_SHARED_SECRET=%s invalid\n", frmwk_unique);
		fprintf(stderr, "Must be run under compatible WLM\n");
		goto fail;
	}

	/* Give preference to PMI_HOME, fall back to HOME */
	frmwk_home = getenv("PMI_HOME");
	if (!frmwk_home)
		frmwk_home = getenv("HOME");
	if (!frmwk_home) {
		fprintf(stderr, "Neither PMI_HOME nor HOME set\n");
		fprintf(stderr, "Shared file system required\n");
		goto fail;
	}

	/* Optional for multiple HSNs, defaults to hsn0 */
	frmwk_nics_per_rank = getenv_int("PMI_NUM_HSNS");
	if (frmwk_nics_per_rank < 1)
		frmwk_nics_per_rank = 1;

	/* Re-export these as libfabric equivalents */
	setenv("FI_CXI_COLL_JOB_ID", frmwk_unique, 1);
	setenv("FI_CXI_COLL_STEP_ID", "0", 1);
	setenv("FI_CXI_COLL_MCAST_TOKEN", "what?", 1);
	setenv("FI_CXI_COLL_FABRIC_MGR_URL", "what?", 1);

	ret = 0;
fail:
	_frmwk_init = (!ret);
}

void frmwk_term(void)
{
	free(frmwk_nics);
	frmwk_nics = NULL;
	frmwk_unique = NULL;
	frmwk_home = NULL;
	frmwk_nics_per_rank = 0;
	frmwk_numranks = 0;
	frmwk_rank = 0;
	_frmwk_init = 0;
}
