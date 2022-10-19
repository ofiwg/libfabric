/*
 * (c) Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/**
 * libfabric C test framework for multinode testing.
 *
 * This must be compiled with:
 *
 * - PLATFORM_CASSINI_HW=1 (or other hardware flag)
 * - WITH_PMI=1
 *
 * Tests are run using srun:
 * $ srun -N8 prov/cxi/test/distributed/test_frmwk
 *
 * pmi_init_libfabric() sets up
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
 * pmi_populate_av() uses Cray pmi2 to gather the addresses of all of the
 * libfabric nodes running in this job, and then creates and binds the
 * fi_av object for the endpoint. This has been separated out from
 * initialization, since altetrnate means will need to be tested (e.g. MPI,
 * rank addressing, or others).
 *
 * pmi_enable_libfabric() can be used after the fi_av object has been
 * initialized.
 *
 * pmi_free_libfabric() terminates the libfabric instance and cleans up.
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

#include "pmi_utils.h"
#include "pmi_frmwk.h"

/* see cxit_trace_enable() in each test framework */
#define	TRACE CXIP_NOTRACE

#define RETURN_ERROR(ret, txt) \
	if (ret != FI_SUCCESS) { \
		fprintf(stderr, "FAILED %s = %s\n", txt, fi_strerror(ret)); \
		return ret; \
	}

#define	CLOSE_OBJ(obj)	do {if (obj) fi_close(&obj->fid); } while (0)

/* Exported information about the multi-node configuration */
int pmi_rank;			/* my rank within the configuration */
int pmi_numranks;		/* total number of ranks in the job */
int pmi_appnum;			/* application number from PMI */
const char *pmi_jobid;		/* application job ID under WLM */
char pmi_hostname[256];		/* my hostname */
struct cxip_addr *pmi_nids;	/* array of pmi_numrank nids */

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
 * @brief Shut down the libfabric test framework.
 *
 */
void pmi_free_libfabric(void)
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

#define	PRT(fmt, ...) do {if (!pmi_rank) printf(fmt, ##__VA_ARGS__);} while(0)

/**
 * @brief Initialize the libfabric test framework.
 *
 * The ep_obj->src_addr has a PID value of 511 (PID_ANY) until the EP is
 * enabled, at which point the actual PID is assigned. Nothing works if the PIDs
 * are mismatched between ranks.
 *
 * @return int error code, 0 on success
 */
int pmi_init_libfabric(void)
{
        int ret;

	pmi_numranks = pmi_GetNumRanks();
	pmi_rank = pmi_GetRank();
	pmi_appnum = pmi_GetAppNum();
	pmi_jobid = pmi_GetJobId();

	if (gethostname(pmi_hostname, sizeof(pmi_hostname)))
		snprintf(pmi_hostname, sizeof(pmi_hostname),
			 "unknown-host-%d", pmi_rank);

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
 * This uses PMI to perform the allgather of addresses across all nodes in the
 * job. To work properly, the libfabric endpoint must already be enabled.
 *
 * This also serves as a barrier that ensures that all ranks have reached this
 * call, i.e. all ranks have enabled their respective endpoint. If an endpoint
 * is not enabled when another endpoint sends a zbcoll packet, the sender will
 * receive an ACK, but the target will drop the packet, hanging the zbcoll
 * operation.
 *
 * PMI has some limitations. In particular, the Cray PMI2 version is quite
 * unfriendly with attempts to re-initialize PMI, and thus does not tolerate
 * child processes trying to initialize PMI, as under the Criterion system.
 *
 * This routine can be replaced with anything that provides an accurate AV
 * across all nodes in the job, e.g. MPI, symmetric AVs distributed by some
 * other out-of-band means to all nodes, or logical (rank) addressing of the
 * Cassini chips.
 *
 * If fiaddr is supplied, it will return an array of fi_addr_t values in rank
 * order. It must be freed by the caller.
 *
 * @param fiaddr : returns array of fi_addr_t in rank order
 * @param size   : returns size of fiaddr array
 * @return int error code, 0 on success.
 */
int pmi_populate_av(fi_addr_t **fiaddrp, size_t *sizep)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_addr mynid;
	fi_addr_t *fiaddr = NULL;
	size_t size = 0;
	int ret;

	if (!fiaddrp || !sizep)
		return -FI_EINVAL;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;
	mynid = ep_obj->src_addr;

	ret = -FI_ENOMEM;
	size = pmi_numranks;
	fiaddr = calloc(size, sizeof(*fiaddr));
	if (!fiaddr)
		goto free_fiaddr;
	pmi_nids = calloc(pmi_numranks, sizeof(struct cxip_addr));
	if (!pmi_nids)
		goto free_nids;

	ret = -FI_EINVAL;
	pmi_Allgather(&mynid, sizeof(struct cxip_addr), pmi_nids);
	if (mynid.raw != pmi_nids[pmi_rank].raw)
		goto free_nids;

	ret = fi_av_insert(cxit_av, pmi_nids, pmi_numranks,
			   fiaddr, 0, NULL);
	if (ret != pmi_numranks)
		goto free_nids;

	*sizep = size;
	*fiaddrp = fiaddr;
	return FI_SUCCESS;

free_nids:
	free(pmi_nids);
free_fiaddr:
	free(fiaddr);
	return ret;
}

/**
 * @brief Display an error message to stderr and return error code.
 *
 * This prints to stderr only if ret != 0. It includes rank of the failing
 * process and the size of the job. These values are meaningful only after
 * pmi_populate_av() has successfully completed.
 *
 * @param ret : error code
 * @param fmt : printf format
 * @param ... : printf parameters
 * @return int value of ret
 */
int pmi_errmsg(int ret, const char *fmt, ...)
{
	va_list args;
	char host[256];
	char *str;

	if (!ret)
		return 0;

	if (gethostname(host, sizeof(host)))
		strcpy(host, "unknown");

	va_start(args, fmt);
	vasprintf(&str, fmt, args);
	va_end(args);
	fprintf(stderr, "%s rank %2d of %2d FAILED %d: %s",
		host, pmi_rank, pmi_numranks, ret, str);
	free(str);

	return ret;
}

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

static FILE *pmi_trace_fid;
int cxit_trace_offset;

void cxit_trace_flush(void)
{
	if (pmi_trace_fid) {
		fflush(pmi_trace_fid);
		fsync(fileno(pmi_trace_fid));
	}
}

static int cxip_trace_attr pmi_trace(const char *fmt, ...)
{
	va_list args;
	char *str;
	int len;

	va_start(args, fmt);
	vasprintf(&str, fmt, args);
	va_end(args);
	len = fprintf(pmi_trace_fid, "[%2d|%2d] %s",
		      pmi_rank, pmi_numranks, str);
	cxit_trace_flush();
	free(str);
	return len;
}

bool cxit_trace_enable(bool enable)
{
	static bool is_enabled = false;
	bool was_enabled = is_enabled;
	char fnam[256];

	is_enabled = !!pmi_trace_fid;
	if (enable && !is_enabled) {
		sprintf(fnam, "./trace%d", pmi_rank + cxit_trace_offset);
		pmi_trace_fid = fopen(fnam, "w");
		if (!pmi_trace_fid) {
			fprintf(stderr, "open(%s) failed: %s\n",
				fnam, strerror(errno));
		} else {
			cxip_trace_fn = pmi_trace;
		}
	} else if (!enable) {
		if (pmi_trace_fid) {
			fflush(pmi_trace_fid);
			fclose(pmi_trace_fid);
			pmi_trace_fid = NULL;
		}
		cxip_trace_fn = NULL;
	}
	return was_enabled;
}

/* display message on stdout from rank 0 */
int pmi_log0(const char *fmt, ...)
{
	va_list args;
	int len;

	if (pmi_rank != 0)
		return 0;

	va_start(args, fmt);
	len = vfprintf(stdout, fmt, args);
	va_end(args);
	fflush(stdout);
	return len;
}
